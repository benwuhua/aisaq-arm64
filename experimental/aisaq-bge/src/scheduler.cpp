// Copyright (c) KIOXIA Corporation. All rights reserved.
// Original AiSAQ search routine and helpers licensed under the MIT license.
#include <arm_neon.h>
#include <fcntl.h>
#include <libaio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "diskann/aio_context_pool.h"
#include "diskann/aisaq.h"
#include "diskann/aisaq_pq_reader.h"
#include "diskann/aisaq_utils.h"
#include "diskann/defaults.h"
#include "diskann/pq_flash_aisaq_index.h"
#include "diskann/utils.h"

namespace schedio {
constexpr size_t fields = 12;
// Histogram resolution is 0.5 us until 1 ms, with a separate overflow bin.
constexpr size_t bins = 2002;
using Histogram = std::array<uint64_t, bins>;
struct QueryCounters {
    uint64_t fields_[fields]{};
    bool measure = false;
};
static thread_local QueryCounters current;
static thread_local Histogram* thread_histogram;
static std::mutex hist_lock;
static std::vector<Histogram*> histograms;

[[noreturn]] static void fail(const char* message, int code = 0) {
    std::fprintf(stderr, "AiSAQ scheduler: %s (code=%d)\n", message, code);
    std::abort();  // A partial/failed query invalidates this benchmark process.
}
}  // namespace schedio
#include "hot_nodes.h"
namespace schedio {
struct Options {
    HotNodes hot;
    unsigned code_width = 0;
    uint32_t compact_rows = 0, compact_entry = 0;
    uint8_t medoid_code[64]{};
    mutable diskann::FixedChunkPQTable alt_table;
    bool rolling;
    bool profile;
    int fd;
    Options() {
        const char* mode = std::getenv("ANN_SCHEDULE");
        if (!mode || (std::strcmp(mode, "batch") && std::strcmp(mode, "rolling")))
            fail("invalid ANN_SCHEDULE");
        rolling = std::strcmp(mode, "rolling") == 0;
        const char* trace = std::getenv("ANN_SCHED_PROFILE");
        profile = trace && std::strcmp(trace, "1") == 0;
        const char* file = std::getenv("ANN_GRAPH_FILE");
        const char* fmt = std::getenv("ANN_LAYOUT");
        if (fmt && *fmt) {
            if (!strcmp(fmt, "sq64"))
                code_width = 64;
            else if (!strcmp(fmt, "fp48"))
                code_width = 48;
            else if (!strcmp(fmt, "fp56"))
                code_width = 56;
            else
                fail("unknown ANN_LAYOUT");
            if (std::getenv("ANN_HOT_IDS")) fail("compact experiment uses zero cache");
            file = std::getenv("ANN_LAYOUT_FILE");
        }
        if (!file) fail("ANN_GRAPH_FILE or ANN_LAYOUT_FILE missing");
        fd = ::open(file, O_DIRECT | O_RDONLY | O_CLOEXEC);
        int flags = fd < 0 ? -1 : ::fcntl(fd, F_GETFL);
        if (flags < 0 || !(flags & O_DIRECT)) fail("could not open O_DIRECT graph", errno);
        if (code_width) {
            alignas(4096) unsigned char header[4096];
            struct stat st{};
            if (fstat(fd, &st) || pread(fd, header, sizeof(header), 0) != 4096 ||
                memcmp(header, "BGELAY01", 8))
                fail("invalid compact file header");
            uint32_t file_width;
            memcpy(&file_width, header + 8, 4);
            memcpy(&compact_rows, header + 12, 4);
            memcpy(&compact_entry, header + 16, 4);
            if (file_width != code_width || !compact_rows || compact_entry >= compact_rows ||
                uint64_t(st.st_size) != 4096ULL * (1ULL + compact_rows))
                fail("invalid compact graph geometry");
            memcpy(medoid_code, header + 64, code_width);
            if (code_width != 64) {
                const char* p = std::getenv("ANN_LAYOUT_PIVOTS");
                if (!p) fail("missing full-dimensional codebook");
                alt_table.load_pq_centroid_bin(p, code_width);
            }
        } else
            hot.load(fd, std::getenv("ANN_HOT_IDS"));
    }
    ~Options() { ::close(fd); }
};
static const Options& options() {
    static const Options config;
    return config;
}
static uint64_t monotonic_ns() {
    timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return uint64_t(now.tv_sec) * 1000000000ULL + now.tv_nsec;
}
static void record_gap(uint64_t ns) {
    if (!current.measure || !thread_histogram) return;
    // Bin 0 contains exact ties. Bin 1 is [1,500), bin 2 [500,1000), etc.
    size_t bin = ns == 0 ? 0 : std::min<uint64_t>(bins - 1, 1 + ns / 500);
    ++(*thread_histogram)[bin];
}
}  // namespace schedio
extern "C" void schedule_reset(int, int measure) {
    schedio::current = {};
    schedio::current.measure = measure != 0;
    if (measure && schedio::options().profile && !schedio::thread_histogram) {
        auto* hist = new schedio::Histogram{};
        std::lock_guard<std::mutex> guard(schedio::hist_lock);
        schedio::histograms.push_back(hist);
        schedio::thread_histogram = hist;
    }
}
extern "C" void schedule_collect(double* values, size_t n) {
    for (size_t i = 0; i < n; ++i)
        values[i] = i < schedio::fields ? double(schedio::current.fields_[i]) : 0.;
}
extern "C" void schedule_dump(const char* prefix) {
    if (!schedio::options().profile || !prefix || !prefix[0]) return;
    std::string path = std::string(prefix) + ".pairs.csv";
    FILE* stream = std::fopen(path.c_str(), "w");
    if (!stream) schedio::fail("opening profile histogram", errno);
    std::fprintf(stream, "lo_ns,hi_ns,count\n");
    std::lock_guard<std::mutex> guard(schedio::hist_lock);
    for (size_t i = 0; i < schedio::bins; ++i) {
        uint64_t count = 0;
        for (auto* hist : schedio::histograms) count += (*hist)[i];
        if (!count) continue;
        long lo = i == 0 ? 0 : i == 1 ? 1 : long(i - 1) * 500;
        long hi = i == 0 ? 1 : i == schedio::bins - 1 ? -1 : long(i) * 500;
        std::fprintf(stream, "%ld,%ld,%llu\n", lo, hi, (unsigned long long)count);
    }
    if (std::fclose(stream) != 0) schedio::fail("writing profile histogram", errno);
}

namespace diskann {
using namespace diskann::defaults;

//

template <typename T>
inline uint64_t PQFlashAisaqIndex<T>::get_node_sector(uint64_t node_id) {
    return 1 + (this->nnodes_per_sector > 0
                    ? node_id / this->nnodes_per_sector
                    : node_id * DIV_ROUND_UP(this->max_node_len, diskann::defaults::SECTOR_LEN));
}

template <typename T>
inline char* PQFlashAisaqIndex<T>::offset_to_node(char* sector_buf, uint64_t node_id) {
    return sector_buf +
           (this->nnodes_per_sector == 0 ? 0 : (node_id % this->nnodes_per_sector) * this->max_node_len);
}

template <typename T>
inline char* PQFlashAisaqIndex<T>::aisaq_offset_to_node_aisaq_data(char* node_buf) {
    return node_buf + this->disk_bytes_per_point + ((this->max_degree + 1) * sizeof(uint32_t));
}

template <typename T>
inline uint32_t* PQFlashAisaqIndex<T>::offset_to_node_nhood(char* node_buf) {
    return (unsigned*)(node_buf + this->disk_bytes_per_point);
}

template <typename T>
inline T* PQFlashAisaqIndex<T>::offset_to_node_coords(char* node_buf) {
    return (T*)(node_buf);
}

template <>
void PQFlashAisaqIndex<float>::aisaq_cached_beam_search(
    const float* query1, const uint64_t k_search, const uint64_t l_search, int64_t* indices,
    float* distances, const uint64_t beam_width, const bool use_reorder_data, QueryStats* stats,
    const knowhere::feder::diskann::FederResultUniq& feder, const knowhere::BitsetView bitset,
    const float filter_ratio_in, const struct diskann::aisaq_search_config* aisaq_search_config) {
    using T = float;
    uint64_t num_sectors_per_node =
        this->nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(this->max_node_len, defaults::SECTOR_LEN);
    if (beam_width > num_sectors_per_node * defaults::MAX_N_SECTOR_READS)
        throw ANNException("Beamwidth can not be higher than defaults::MAX_N_SECTOR_READS", -1, __FUNCSIG__,
                           __FILE__, __LINE__);

    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
        this->thread_data.wait_for_push_notify();
        data = this->thread_data.pop();
    }
    auto query_norm_opt = this->init_thread_data(data, query1);
    if (!query_norm_opt.has_value()) {
        // return an empty answer when calcu a zero point
        this->thread_data.push(data);
        this->thread_data.push_notify_all();
        return;
    }
    float query_norm = query_norm_opt.value();
    AisaqThreadData aisaq_data = aisaq_thread_data.pop();
    auto ctx = this->reader->get_ctx();
    if (aisaq_data.aisaq_pq_reader_ctx) {
        _aisaq_pq_vectors_reader->set_io_ctx(*aisaq_data.aisaq_pq_reader_ctx, ctx);
    }
    auto release_data = [this, data, aisaq_data, ctx]() mutable {
        this->thread_data.push(data);
        this->thread_data.push_notify_all();
        this->aisaq_thread_data.push(aisaq_data);
        this->aisaq_thread_data.push_notify_all();
        if (aisaq_data.aisaq_pq_reader_ctx) {
            _aisaq_pq_vectors_reader->set_io_ctx(*aisaq_data.aisaq_pq_reader_ctx, nullptr);
        }
        this->reader->put_ctx(ctx);
    };
    if (beam_width < 2 || beam_width > 6 || schedio::options().profile || use_reorder_data ||
        !bitset.empty() || feder || _aisaq_inline_pq_vectors != this->max_degree ||
        _aisaq_rearranged_vectors || this->use_disk_index_pq || !this->nhood_cache.empty() ||
        this->count_visited_nodes ||
        (aisaq_search_config && aisaq_search_config->pq_read_page_cache_size != 0) ||
        num_sectors_per_node != 2) {
        schedio::fail(
            "experiment requires W=2..6, full inline PQ, raw vectors, two sectors, zero cache, no filter "
            "or reorder");
    }
    uint64_t local_l_search = l_search;
    auto query_scratch = &(data.scratch);
    // reset query scratch
    query_scratch->reset();
    aisaq_data.retset.clear();
    aisaq_data.full_retset.clear();
    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    T* aligned_query_T = query_scratch->aligned_query_T;
    float* query_float = query_scratch->aligned_query_float;

    // pointers to buffers for data
    T* data_buf = query_scratch->coord_scratch;
#if defined(__ARM_NEON) && defined(__aarch64__)
    __builtin_prefetch(data_buf, 1, 3);
#else
    _mm_prefetch((char*)data_buf, _MM_HINT_T1);
#endif

    // sector scratch
    char* sector_scratch = query_scratch->sector_scratch;
    // query <-> PQ chunk centers distances
    // we have a rotation matrix
    float* pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    const auto& config = schedio::options();
    if (config.code_width && config.code_width != 64)
        config.alt_table.populate_chunk_distances(query_float, pq_dists);
    else
        this->pq_table.populate_chunk_distances(query_float, pq_dists);

    float* dist_scratch = query_scratch->aligned_dist_scratch;
    uint64_t pq_read_page_cache_size = 0;
    Timer query_timer, io_timer, cpu_timer;
    if (aisaq_data.aisaq_pq_reader_ctx != nullptr)
        _aisaq_pq_vectors_reader->set_page_cache_size(*aisaq_data.aisaq_pq_reader_ctx,
                                                      pq_read_page_cache_size);

    tsl::robin_set<uint64_t>* visited = query_scratch->visited;
    NeighborPriorityQueue& retset = aisaq_data.retset;
    retset.reserve(local_l_search);
    std::vector<Neighbor>& full_retset = aisaq_data.full_retset;

    uint32_t best_medoid;
    float best_dist;
    if (config.code_width) {
        best_medoid = config.compact_entry;
        diskann::pq_dist_lookup(config.medoid_code, 1, config.code_width, pq_dists, &best_dist);
        num_sectors_per_node = 1;
    } else
        get_entry_point_medoid(best_medoid, best_dist, pq_dists, dist_scratch, query_float);

    retset.insert(Neighbor(best_medoid, best_dist));
    visited->insert(best_medoid);

    // Fixed-capacity direct-read slots per query. A slot may only be reused after
    // both harvesting its event and consuming its buffer.
    auto& metric = schedio::current.fields_;
    const auto bytes = num_sectors_per_node * defaults::SECTOR_LEN;
    enum class State { free, prepared, pending, ready };
    struct Slot {
        State state = State::free;
        uint32_t node = 0;
        char* cached_node = nullptr;
        iocb cb{};
        bool joint = false;
    } slots[6];
    int depth = 0;
    int ready[6];
    int nready = 0;
    uint64_t first_pair_ns = 0;
    size_t first_reap = 0;
    auto fill = [&]() {
        iocb* list[6];
        int indices_[6], n = 0, chosen = 0;
        for (int s = 0; s < int(beam_width) && retset.has_unexpanded_node(); ++s) {
            if (slots[s].state != State::free) continue;
            auto& slot = slots[s];
            size_t position;
            if (!retset.get_first_unexpanded_position(position, true))
                schedio::fail("queue cursor disagrees with has_unexpanded_node");
            slot.node = retset[position].id;
            slot.state = State::prepared;
            slot.joint = false;
            ++chosen;
            slot.cached_node = config.hot.find(slot.node);
            if (slot.cached_node != nullptr) {
                slot.state = State::ready;
                ready[nready++] = s;
                if (stats) ++stats->n_cache_hits;
                continue;
            }
            io_prep_pread(
                &slot.cb, config.fd, sector_scratch + s * bytes, bytes,
                (config.code_width ? 1ull + slot.node : get_node_sector(slot.node)) * defaults::SECTOR_LEN);
            slot.cb.data = (void*)(uintptr_t(s) + 1);
            indices_[n] = s;
            list[n++] = &slot.cb;
        }
        int sent = 0;
        while (sent < n) {
            io_timer.reset();
            int got = io_submit(ctx, n - sent, list + sent);
            if (got == -EINTR) continue;
            if (got <= 0 || got > n - sent)
                schedio::fail("io_submit failed or returned invalid count", got);
            if (stats) {
                stats->io_us += float(io_timer.elapsed());
                ++stats->n_hops;
                updata_io_stats(*stats, got, num_sectors_per_node);
            }
            depth += got;
            metric[0] += got;
            metric[2] = std::max(metric[2], uint64_t(depth));
            ++metric[3];
            if (got == 2) {
                ++metric[5];
                first_pair_ns = 0;
                first_reap = 0;
            } else if (got == 1) {
                ++metric[9];
            } else {
                metric[6] += got;
                ++metric[7];
            }
            for (int j = 0; j < got; ++j) {
                auto& slot = slots[indices_[sent + j]];
                slot.state = State::pending;
                slot.joint = got == 2;
            }
            sent += got;
            if (depth > int(beam_width)) schedio::fail("graph read cap exceeded");
        }
        return chosen;
    };
    auto harvest = [&](bool entire_batch) {
        if (depth < 1 || depth > int(beam_width)) schedio::fail("illegal harvest depth");
        io_event events[6];
        // In profile mode the baseline wakes after the first visible completion,
        // immediately harvests again, and still expands only after the barrier.
        int min_nr = entire_batch && !config.profile ? depth : 1;
        io_timer.reset();
        int got;
        do {
            got = io_getevents(ctx, min_nr, depth, events, nullptr);
        } while (got == -EINTR || got == 0);
        if (got < 0 || got > depth) schedio::fail("io_getevents failed", got);
        uint64_t observed = config.profile ? schedio::monotonic_ns() : 0;
        if (stats) stats->io_us += float(io_timer.elapsed());
        ++metric[4];
        metric[10] += got;
        metric[11] += got == 2;
        for (int j = 0; j < got; ++j) {
            int s = int(uintptr_t(events[j].data)) - 1;
            if (s < 0 || s >= int(beam_width) || slots[s].state != State::pending ||
                events[j].obj != &slots[s].cb || events[j].res != bytes || events[j].res2 != 0)
                schedio::fail("invalid or short AIO completion", int(events[j].res));
            if (slots[s].joint && config.profile) {
                if (!first_pair_ns) {
                    first_pair_ns = observed;
                    first_reap = metric[4];
                } else {
                    uint64_t gap = observed - first_pair_ns;
                    metric[6] += first_reap == metric[4];
                    metric[7] += gap;
                    schedio::record_gap(gap);
                    first_pair_ns = 0;
                }
            }
            slots[s].joint = false;
            slots[s].state = State::ready;
            ready[nready++] = s;
            --depth;
            ++metric[1];
        }
    };
    auto expand = [&](int s) {
        cpu_timer.reset();
        auto& slot = slots[s];
        if (slot.state != State::ready) schedio::fail("expand before completion");
        metric[8] += depth != 0;
        char* node =
            slot.cached_node ? slot.cached_node : offset_to_node(sector_scratch + s * bytes, slot.node);
        unsigned char* packed = reinterpret_cast<unsigned char*>(node);
        uint32_t count;
        T* coords;
        uint32_t* nhood = nullptr;
        unsigned char* packed_ids = nullptr;
        uint8_t* pq_codes;
        if (config.code_width) {
            uint32_t id;
            memcpy(&id, packed, 4);
            if (id != slot.node || packed[4] != 0x35 || packed[5] != 0x4b || packed[6] != 32 ||
                packed[7] != config.code_width)
                schedio::fail("missing or malformed physical compact record");
            count = 32;
            coords = data_buf;
            if (config.code_width == 64) {
                float l, scale;
                memcpy(&l, packed + 8, 4);
                memcpy(&scale, packed + 12, 4);
                for (int d = 0; d < 1024; d += 8) {
                    auto mid = vmovl_u8(vld1_u8(packed + 16 + d));
                    auto av = vcvtq_f32_u32(vmovl_u16(vget_low_u16(mid)));
                    auto bv = vcvtq_f32_u32(vmovl_u16(vget_high_u16(mid)));
                    vst1q_f32(coords + d, vmlaq_n_f32(vdupq_n_f32(l), av, scale));
                    vst1q_f32(coords + d + 4, vmlaq_n_f32(vdupq_n_f32(l), bv, scale));
                }
                packed_ids = packed + 16 + 1024;
            } else {
                for (int d = 0; d < 1024; d += 4)
                    vst1q_f32(coords + d,
                              vcvt_f32_f16(vld1_f16(reinterpret_cast<float16_t*>(packed + 16) + d)));
                packed_ids = packed + 16 + 2048;
            }
            pq_codes = packed_ids + 5 * 32;
        } else {
            coords = offset_to_node_coords(node);
            nhood = offset_to_node_nhood(node);
            count = *nhood;
            if (count > this->max_degree) schedio::fail("degree exceeds index header");
            if (uintptr_t(coords) & 31U) {
                std::memcpy(data_buf, coords, this->disk_bytes_per_point);
                coords = data_buf;
            }
            pq_codes = reinterpret_cast<uint8_t*>(aisaq_offset_to_node_aisaq_data(node));
        }
        const float exact_dist =
            this->dist_cmp_wrap(aligned_query_T, coords, size_t(this->aligned_dim), slot.node);
        full_retset.emplace_back(slot.node, exact_dist, uint32_t(0));
        if (stats) ++stats->n_cmps;
        diskann::pq_dist_lookup(pq_codes, count, config.code_width ? config.code_width : this->n_chunks,
                                pq_dists, dist_scratch);
        for (uint32_t j = 0; j < count; ++j) {
            uint32_t neighbor;
            if (packed_ids) {
                uint64_t logical = 0;
                memcpy(&logical, packed_ids + 5 * j, 5);
                if (logical >= config.compact_rows) schedio::fail("invalid 40-bit neighbor");
                neighbor = static_cast<uint32_t>(logical);
            } else
                neighbor = nhood[1 + j];
            if (!visited->insert(neighbor).second) continue;
            bool removed;
            uint32_t evicted;
            retset.insert_with_rem_info(Neighbor(neighbor, dist_scratch[j]), removed, evicted);
        }
        if (stats) stats->cpu_us += float(cpu_timer.elapsed());
        slot.state = State::free;
    };
    if (!config.rolling) {
        while (int n = fill()) {
            while (depth) harvest(true);
            if (nready != n) schedio::fail("barrier completed a mismatched batch");
            // fill always allocates slot 0 then 1. Original AiSAQ expands in
            // submission order, regardless of the order of its completions.
            for (int s = 0; s < n; ++s) expand(s);
            nready = 0;
        }
    } else {
        while (true) {
            fill();
            if (!nready) {
                if (!depth) break;
                harvest(false);
            }
            int s = ready[0];
            --nready;
            std::move(ready + 1, ready + 1 + nready, ready);
            expand(s);
            // Dispatch the nearest eligible node on the next pass immediately.
        }
    }
    if (depth || nready || metric[0] != metric[1]) schedio::fail("incomplete query drain");

    /* clear page cache between queries */
    // PqVectorsOnDisk::getInstance()->clear_page_cache(&data->pq_ctx);

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end());

    if (use_reorder_data) {
        if (!(this->_reorder_data_exists)) {
            release_data();
            throw ANNException(
                "Requested use of reordering data which does "
                "not exist in index "
                "file",
                -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        rerank_candidate_list(full_retset, k_search, sector_scratch, stats, io_timer, ctx, aligned_query_T);
    }

    prepare_search_results(full_retset, k_search, distances, indices, query_norm);

    if (stats != nullptr) {
        stats->total_us = (float)query_timer.elapsed();
    }
    if (aisaq_data.aisaq_pq_reader_ctx != nullptr)
        _aisaq_pq_vectors_reader->hibernate(*aisaq_data.aisaq_pq_reader_ctx);

    release_data();
}

}  // namespace diskann
