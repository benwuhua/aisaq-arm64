// Copyright (c) KIOXIA Corporation. All rights reserved.
// AiSAQ best-first traversal and full-inline record interpretation are MIT licensed.
#include "compact_format.h"
#include "direct_reads.h"
#include "hot_nodes.h"
#include "l2_neon.h"
#include "diskann/aisaq.h"
#include "diskann/aisaq_pq_reader.h"
#include "diskann/pq_flash_aisaq_index.h"
#include <array>
#include <chrono>
#include <deque>
#include <limits>
#include <mutex>

namespace schedio {
constexpr size_t fields = 12, bins = 2002;
using Histogram = std::array<uint64_t, bins>;
struct QueryCounters {
    uint64_t values[fields]{};
    bool measure = false;
};
static thread_local QueryCounters current;
static thread_local Histogram* thread_histogram;
static std::mutex hist_lock;
static std::vector<std::unique_ptr<Histogram>> histograms;
static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
struct Options {
    HotNodes hot;
    bool packed = false, rolling = false, profile = false;
    unsigned width;
    compact::Geometry graph;
    diskann::FixedChunkPQTable table;
    std::array<uint8_t, 64> entry_code{};
    int fd = -1;
    Options(const char* prefix, unsigned window) : width(window) {
        const char* schedule = std::getenv("ANN_SCHEDULE");
        if (!schedule || (strcmp(schedule, "batch") && strcmp(schedule, "rolling")))
            throw std::runtime_error("ANN_SCHEDULE must be batch or rolling");
        rolling = !strcmp(schedule, "rolling");
        profile = std::getenv("ANN_SCHED_PROFILE") && !strcmp(std::getenv("ANN_SCHED_PROFILE"), "1");
        if (profile && (rolling || width != 2))
            throw std::runtime_error("pair-gap profiling requires batch W=2");
        const char* layout = std::getenv("ANN_LAYOUT");
        packed = layout && *layout;
        const char* file = std::getenv(packed ? "ANN_LAYOUT_FILE" : "ANN_GRAPH_FILE");
        if (!file || (fd = open(file, O_RDONLY | O_DIRECT | O_CLOEXEC)) < 0)
            throw std::runtime_error("cannot open O_DIRECT graph");
        try {
            alignas(4096) uint8_t header[4096];
            struct stat st{};
            if (fstat(fd, &st) || pread(fd, header, 4096, 0) != 4096)
                throw std::runtime_error("invalid graph file");
            if (packed) {
                graph = compact::Geometry::decode(header, st.st_size);
                unsigned k = !strcmp(layout, "sq64")   ? 64
                             : !strcmp(layout, "fp56") ? 56
                             : !strcmp(layout, "fp48") ? 48
                                                       : 0;
                if (k != graph.width) throw std::runtime_error("layout does not match header");
                std::memcpy(entry_code.data(), header + 64, k);
                auto pivots = std::getenv("ANN_LAYOUT_PIVOTS");
                if (!pivots && k != 64) throw std::runtime_error("missing compact PQ pivots");
                auto name = pivots ? std::string(pivots) : std::string(prefix) + "_pq_pivots.bin";
                table.load_pq_centroid_bin(name.c_str(), k);
                if (table.get_total_dims() != 1024 || table.get_num_chunks() != k)
                    throw std::runtime_error("PQ table shape differs from compact graph");
            } else {
                graph = {64, compact::load<uint64_t>(header + 8), 0, compact::load<uint64_t>(header + 24)};
                graph.validate();
                if (graph.rows > UINT32_MAX || compact::load<uint64_t>(header + 16) != 1024 ||
                    compact::load<uint64_t>(header + 32) != 6276 || compact::load<uint64_t>(header + 40) ||
                    uint64_t(st.st_size) != 4096 + 8192 * graph.rows)
                    throw std::runtime_error("unexpected native BGE geometry");
            }
            hot.load(fd, std::getenv("ANN_HOT_IDS"), packed ? &graph : nullptr);
        } catch (...) {
            close(fd);
            throw;
        }
    }
    ~Options() { close(fd); }
};
static std::unique_ptr<Options> options;

// Keep Knowhere's insertion order for equal PQ scores. visited has already removed duplicates.
struct Node {
    uint64_t id;
    float distance;
    bool expanded = false;
};
struct Frontier {
    std::vector<Node> nodes;
    size_t limit, cursor = 0;
    explicit Frontier(size_t l) : limit(l) { nodes.reserve(l + 1); }
    void insert(uint64_t id, float distance) {
        if (nodes.size() == limit && nodes.back().distance <= distance) return;
        auto it = std::upper_bound(nodes.begin(), nodes.end(), distance,
                                   [](float d, const Node& x) { return d < x.distance; });
        cursor = std::min(cursor, size_t(it - nodes.begin()));
        nodes.insert(it, {id, distance});
        if (nodes.size() > limit) nodes.pop_back();
    }
    bool next(uint64_t& id) {
        if (cursor == nodes.size()) return false;
        auto& n = nodes[cursor];
        n.expanded = true;
        id = n.id;
        do {
            ++cursor;
        } while (cursor < nodes.size() && nodes[cursor].expanded);
        return true;
    }
};
enum class State { free, pending, ready };
struct Slot {
    uint64_t id = 0;
    const uint8_t* hit = nullptr;
    State state = State::free;
};
struct Worker {
    DirectReads io;
    std::vector<Slot> slots;
    std::vector<unsigned> completions;
    std::deque<unsigned> ready;
    tsl::robin_set<uint64_t> visited;
    std::vector<Node> ranked;
    Worker(unsigned width, unsigned bytes) : io(width, bytes), slots(width) { completions.reserve(width); }
};
static thread_local std::unique_ptr<Worker> worker;

template <typename Score>
static void traverse(const float* lut, uint64_t entry, float entry_dist, size_t limit, size_t k,
                     int64_t* ids, float* distances, diskann::QueryStats& stats, Score score) {
    const auto& opt = *options;
    const unsigned bytes = opt.packed ? 4096 : 8192;
    if (!worker) worker = std::make_unique<Worker>(opt.width, bytes);
    auto& w = *worker;
    auto& metric = current.values;
    Frontier frontier(limit);
    frontier.insert(entry, entry_dist);
    w.ranked.clear();
    w.visited.clear();
    w.visited.insert(entry);
    unsigned pending = 0;
    bool pair = false;
    uint64_t first = 0;
    auto fill = [&]() {
        unsigned chosen = 0, reads = 0;
        for (unsigned s = 0; s < opt.width; ++s) {
            auto& slot = w.slots[s];
            if (slot.state != State::free) continue;
            if (!frontier.next(slot.id)) break;
            ++chosen;
            slot.hit = opt.hot.find(slot.id);
            if (slot.hit) {
                slot.state = State::ready;
                w.ready.push_back(s);
                ++stats.n_cache_hits;
            } else {
                uint64_t offset = opt.packed ? opt.graph.offset(slot.id) : 4096 + 8192 * slot.id;
                w.io.prepare(s, opt.fd, bytes, offset);
                slot.state = State::pending;
                ++reads;
            }
        }
        if (reads) {
            uint64_t start = now_ns();
            unsigned calls = w.io.submit();
            stats.io_us += (now_ns() - start) / 1000.;
            metric[3] += calls;
            stats.n_hops += calls;
            pending += reads;
            metric[0] += reads;
            metric[2] = std::max<uint64_t>(metric[2], pending);
            if (reads == 2) ++metric[5];
            if (reads == 1) ++metric[9];
            if (reads > 2) {
                metric[6] += reads;
                ++metric[7];
            }
            stats.n_ios += reads;
            (opt.packed ? stats.n_4k : stats.n_8k) += reads;
            if (opt.profile) {
                pair = reads == 2;
                first = 0;
            }
        }
        return chosen;
    };
    auto harvest = [&](unsigned min_events) {
        uint64_t start = now_ns();
        w.io.reap(min_events, w.completions);
        uint64_t observed = now_ns();
        stats.io_us += (observed - start) / 1000.;
        ++metric[4];
        metric[10] += w.completions.size();
        metric[11] += w.completions.size() == 2;
        for (unsigned s : w.completions) {
            if (w.slots[s].state != State::pending) throw std::logic_error("unexpected graph completion");
            if (pair) {
                if (!first)
                    first = observed;
                else {
                    uint64_t gap = observed - first;
                    if (current.measure && thread_histogram) {
                        size_t bin = gap ? std::min<uint64_t>(bins - 1, 1 + gap / 500) : 0;
                        ++(*thread_histogram)[bin];
                    }
                    first = 0;
                }
            }
            w.slots[s].state = State::ready;
            w.ready.push_back(s);
            --pending;
            ++metric[1];
        }
    };
    auto expand = [&](unsigned s) {
        uint64_t start = now_ns();
        auto& slot = w.slots[s];
        if (slot.state != State::ready) throw std::logic_error("graph buffer consumed before completion");
        metric[8] += pending != 0;
        const uint8_t* node = slot.hit ? slot.hit : w.io.buffer(s);
        unsigned count =
            opt.packed ? opt.graph.check_record(node, slot.id) : compact::load<uint32_t>(node + 4096);
        if (count > 32) throw std::runtime_error("invalid degree");
        w.ranked.push_back({slot.id, score(node)});
        ++stats.n_cmps;
        const auto* edges = node + (opt.packed ? 16 + opt.graph.vector_bytes() : 4100);
        const auto* codes = edges + 32 * (opt.packed ? 5 : 4);
        float pq[32];
        if (count) diskann::pq_dist_lookup(codes, count, opt.graph.width, lut, pq);
        for (unsigned j = 0; j < count; ++j) {
            uint64_t id =
                opt.packed ? compact::id40(edges + 5 * j) : compact::load<uint32_t>(edges + 4 * j);
            if (!opt.graph.contains(id)) throw std::runtime_error("edge outside graph");
            if (w.visited.insert(id).second) frontier.insert(id, pq[j]);
        }
        slot.state = State::free;
        stats.cpu_us += (now_ns() - start) / 1000.;
    };
    try {
        if (opt.rolling) {
            for (;;) {
                fill();
                if (pending) harvest(w.ready.empty() ? 1 : 0);
                if (w.ready.empty()) break;
                unsigned slot = w.ready.front();
                w.ready.pop_front();
                expand(slot);
            }
        } else {
            while (unsigned chosen = fill()) {
                while (pending) harvest(opt.profile ? 1 : pending);
                if (w.ready.size() != chosen) throw std::logic_error("incomplete graph batch");
                w.ready.clear();
                for (unsigned s = 0; s < chosen; ++s) expand(s);
            }
        }
        if (pending || metric[0] != metric[1]) throw std::logic_error("incomplete graph search");
    } catch (...) {
        worker.reset();
        throw;
    }
    std::sort(w.ranked.begin(), w.ranked.end(),
              [](const Node& a, const Node& b) { return a.distance < b.distance; });
    const size_t n = std::min(k, w.ranked.size());
    for (size_t i = 0; i < n; ++i) {
        ids[i] = w.ranked[i].id;
        if (distances) distances[i] = w.ranked[i].distance;
    }
    std::fill(ids + n, ids + k, -1);
    if (distances) std::fill(distances + n, distances + k, std::numeric_limits<float>::infinity());
}
}  // namespace schedio

unsigned schedule_setup(const char* prefix, unsigned width) {
    schedio::options = std::make_unique<schedio::Options>(prefix, width);
    return schedio::options->packed ? schedio::options->graph.width : 0;
}
void compact_search(const float* query, size_t k, size_t limit, int64_t* ids, float* distances,
                    diskann::QueryStats& stats) {
    const auto start = schedio::now_ns();
    auto& opt = *schedio::options;
    static thread_local std::vector<float> lut(256 * opt.graph.width);
    opt.table.populate_chunk_distances(query, lut.data());
    float best;
    diskann::pq_dist_lookup(opt.entry_code.data(), 1, opt.graph.width, lut.data(), &best);
    schedio::traverse(
        lut.data(), opt.graph.entry, best, limit, k, ids, distances, stats, [&](const uint8_t* p) {
            if (opt.graph.width != 64) return bge_l2::fp16(query, p + 16);
            return bge_l2::sq8(query, p + 16, compact::load<float>(p + 8), compact::load<float>(p + 12));
        });
    stats.total_us = (schedio::now_ns() - start) / 1000.;
}

namespace diskann {
template <>
void PQFlashAisaqIndex<float>::aisaq_cached_beam_search(
    const float* query, const uint64_t k, const uint64_t limit, int64_t* ids, float* distances,
    const uint64_t width, const bool reorder, QueryStats* stats,
    const knowhere::feder::diskann::FederResultUniq& feder, const knowhere::BitsetView bitset, const float,
    const struct aisaq_search_config* config) {
    auto& opt = *schedio::options;
    if (opt.packed || width != opt.width || reorder || !bitset.empty() || feder ||
        this->metric != Metric::L2 || this->max_degree != 32 || this->n_chunks != 64 ||
        this->max_node_len != 6276 || _aisaq_inline_pq_vectors != 32 || _aisaq_rearranged_vectors ||
        this->use_disk_index_pq || !this->nhood_cache.empty() || this->count_visited_nodes ||
        (config && config->pq_read_page_cache_size))
        throw std::runtime_error(
            "native benchmark requires unrearranged full-inline BGE with L2 and zero native PQ cache");
    auto start = schedio::now_ns();
    std::fill(ids, ids + k, -1);
    if (distances) std::fill(distances, distances + k, std::numeric_limits<float>::infinity());
    auto data = this->thread_data.pop();
    while (!data.scratch.sector_scratch) {
        this->thread_data.wait_for_push_notify();
        data = this->thread_data.pop();
    }
    auto release = [&]() {
        this->thread_data.push(data);
        this->thread_data.push_notify_all();
    };
    try {
        if (this->init_thread_data(data, query).has_value()) {
            auto& scratch = data.scratch;
            scratch.reset();
            auto lut = scratch.aligned_pqtable_dist_scratch;
            this->pq_table.populate_chunk_distances(scratch.aligned_query_float, lut);
            uint32_t entry;
            float best;
            get_entry_point_medoid(entry, best, lut, scratch.aligned_dist_scratch,
                                   scratch.aligned_query_float);
            QueryStats dummy{};
            schedio::traverse(
                lut, entry, best, limit, k, ids, distances, stats ? *stats : dummy, [&](const uint8_t* p) {
                    return this->dist_cmp(scratch.aligned_query_T, reinterpret_cast<const float*>(p),
                                          size_t(this->aligned_dim));
                });
        }
    } catch (...) {
        release();
        throw;
    }
    release();
    if (stats) stats->total_us = (schedio::now_ns() - start) / 1000.;
}
}  // namespace diskann

extern "C" void schedule_reset(int, int measure) {
    using namespace schedio;
    current = {};
    current.measure = measure != 0;
    if (measure && options->profile && !thread_histogram) {
        std::lock_guard<std::mutex> guard(hist_lock);
        histograms.push_back(std::make_unique<Histogram>());
        thread_histogram = histograms.back().get();
    }
}
extern "C" void schedule_collect(double* values, size_t n) {
    for (size_t i = 0; i < n; ++i) values[i] = i < schedio::fields ? schedio::current.values[i] : 0;
}
extern "C" void schedule_dump(const char* prefix) {
    using namespace schedio;
    if (!options->profile) return;
    auto path = std::string(prefix) + ".pairs.csv";
    std::ofstream out(path);
    out << "lo_ns,hi_ns,count\n";
    for (size_t i = 0; i < bins; ++i) {
        uint64_t count = 0;
        for (auto& hist : histograms) count += (*hist)[i];
        if (count)
            out << (i == 1 ? 1L
                    : i    ? long(i - 1) * 500
                           : 0L)
                << ','
                << (i == bins - 1 ? -1L
                    : i           ? long(i) * 500
                                  : 1L)
                << ',' << count << '\n';
    }
    out.close();
    if (!out) throw std::runtime_error("could not write pair-gap histogram");
}
