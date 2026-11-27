#include <array>
#include <cstddef>

#include "window.h"
extern "C" void schedule_reset(int query_id, int measure);
extern "C" void schedule_collect(double* values, size_t n);
extern "C" void schedule_dump(const char* prefix);

#include <omp.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "diskann/aisaq.h"
#include "diskann/aux_utils.h"
#include "diskann/linux_aligned_file_reader.h"
#include "diskann/pq_flash_aisaq_index.h"
#include "exact_rerank.h"

// The packaged Knowhere engine owns the implementation and worker pools.
namespace knowhere {
void InitBuildThreadPool(uint32_t num_threads);
}

namespace {
using Clock = std::chrono::steady_clock;
constexpr uint32_t top_k = 10;
const uint32_t search_threads = option_u64("ANN_THREADS", 4);
constexpr uint64_t gib = uint64_t{1} << 30;

struct Shape {
    uint32_t rows;
    uint32_t dim;
};

Shape read_shape(std::ifstream& file) {
    Shape shape{};
    file.read(reinterpret_cast<char*>(&shape.rows), sizeof(shape.rows));
    file.read(reinterpret_cast<char*>(&shape.dim), sizeof(shape.dim));
    if (!file || shape.rows == 0 || shape.dim == 0) {
        throw std::runtime_error("Invalid fbin header");
    }
    return shape;
}

uint64_t read_bytes() {
    std::ifstream file("/proc/self/io");
    std::string key;
    uint64_t value;
    while (file >> key >> value) {
        if (key == "read_bytes:") {
            return value;
        }
    }
    throw std::runtime_error("Could not read process storage counters");
}

uint64_t rss_kib() {
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            return std::stoull(line.substr(6));
        }
    }
    throw std::runtime_error("Could not read process RSS");
}

int build(int argc, char** argv) {
    std::ifstream input(argv[2], std::ios::binary);
    auto shape = read_shape(input);
    int inline_pq = std::stoi(argv[4]);
    if (inline_pq < -1 || inline_pq > 32 || shape.dim < 64) {
        throw std::runtime_error("Build expects inline_pq in [-1,32] and at least 64 dimensions");
    }
    uint32_t threads = argc > 5 ? std::stoul(argv[5]) : 32;
    double budget_gib = argc > 6 ? std::stod(argv[6]) : 24;
    bool rearrange = argc > 7 ? std::stoi(argv[7]) != 0 : true;
    omp_set_num_threads(threads);
    knowhere::InitBuildThreadPool(threads);

    diskann::BuildConfig config;
    config.data_file_path = argv[2];
    config.index_file_path = argv[3];
    config.compare_metric = diskann::Metric::L2;
    config.max_degree = 32;
    config.search_list_size = 100;
    config.pq_code_size_gb = double(uint64_t{shape.rows} * 64) / gib;
    config.index_mem_gb = budget_gib;
    config.aisaq_mode = true;
    config.inline_pq = static_cast<decltype(config.inline_pq)>(inline_pq);
    config.rearrange = rearrange;
    std::cout << "BUILD rows=" << shape.rows << " dim=" << shape.dim
              << " R=32 L=100 pq=64 inline_pq=" << inline_pq << " rearrange=" << rearrange
              << " threads=" << threads << '\n';
    return diskann::build_disk_index<float>(config);
}

int search(int argc, char** argv) {
    if (!std::getenv("ANN_WINDOW")) throw std::runtime_error("ANN_WINDOW path is required");
    std::ifstream input(argv[2], std::ios::binary);
    auto shape = read_shape(input);
    std::vector<float> queries(size_t{shape.rows} * shape.dim);
    input.read(reinterpret_cast<char*>(queries.data()), queries.size() * sizeof(float));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("Query file size does not match fbin header");
    }
    const auto depth = std::stoul(argv[4]);
    const std::string output = argv[5];
    if (depth < top_k) {
        throw std::runtime_error("Search L must be at least top-k");
    }
    diskann::aisaq_search_config config{
        1,
        diskann::aisaq_pq_io_engine_aio,
        argc > 7 ? std::stoull(argv[7]) : 0,
        argc > 6 ? std::stoull(argv[6]) : 0,
    };
    const auto rss_before_load = rss_kib();
    if (!AioContextPool::InitGlobalAioPool(search_threads, std::min<size_t>(default_max_events, 64))) {
        throw std::runtime_error("Could not configure the Linux AIO context pool");
    }
    auto reader = std::make_shared<LinuxAlignedFileReader>();
    auto index = std::make_unique<diskann::PQFlashAisaqIndex<float>>(reader, diskann::Metric::L2);
    if (index->aisaq_load(search_threads, argv[3]) != 0) {
        throw std::runtime_error("AiSAQ index load failed");
    }
    if (index->get_data_dim() != shape.dim || shape.dim != 1024) {
        throw std::runtime_error("BGE search requires matching 1024-dimensional queries and graph");
    }
    if (!index->get_rearranged_index()) {
        config.pq_read_page_cache_size = 0;
    }
    if (config.pq_cache_size > 0) {
        index->aisaq_load_pq_cache(std::string(argv[3]) + "_pq_compressed.bin", config.pq_cache_size,
                                   diskann::aisaq_pq_cache_policy_auto);
    }
    if (index->aisaq_init(config.pq_io_engine, argv[3]) != 0) {
        throw std::runtime_error("AiSAQ PQ reader initialization failed");
    }
    const auto rss_after_load = rss_kib();

    std::vector<int64_t> ids(size_t{shape.rows} * top_k, -1);
    std::vector<float> distances(ids.size());
    std::vector<diskann::QueryStats> stats(shape.rows);
    std::vector<double> wall_us(shape.rows);
    std::vector<std::array<double, 12>> counters(shape.rows);
    std::exception_ptr failure;
    const uint32_t cap = option_u64("RR_C", 0);
    const uint32_t width = option_u64("RR_WIDTH", 2);
    if (cap && (cap < top_k || cap > depth || shape.dim != 1024))
        throw std::runtime_error("Rerank requires 10<=RR_C<=L and 1024-dimensional BGE");
    std::vector<std::unique_ptr<ExactReranker>> workers;
    for (size_t t = 0; t < search_threads; ++t)
        workers.emplace_back(new ExactReranker(std::getenv("RR_POOL"), option_u64("RR_OFFSET", 0),
                                               option_u64("RR_STRIDE", 4096), width, cap));
    std::vector<RerankStats> rerank_stats(shape.rows);
    const auto window_width = option_u64("ANN_WIDTH", 2);
    auto run_row = [&](uint32_t row, diskann::QueryStats& graph_stats, RerankStats& rr) {
        ExactReranker& worker = *workers.at(omp_get_thread_num());
        const float* query = queries.data() + size_t{row} * shape.dim;
        int64_t* out_ids = ids.data() + size_t{row} * top_k;
        float* out_dist = distances.data() + size_t{row} * top_k;
        index->aisaq_cached_beam_search(query, cap ? cap : top_k, depth,
                                        cap ? worker.candidates.data() : out_ids,
                                        cap ? worker.estimates.data() : out_dist, window_width, false,
                                        &graph_stats, nullptr, nullptr, -1.0f, &config);
        if (cap) worker.run(query, out_ids, out_dist, rr);
    };
    const uint32_t warmup = std::min(shape.rows, std::max(32U, 2 * search_threads));
#pragma omp parallel for num_threads(search_threads) schedule(dynamic, 1)
    for (uint32_t row = 0; row < warmup; ++row) {
        try {
            diskann::QueryStats warm{};
            RerankStats rr{};
            schedule_reset(row, 0);
            run_row(row, warm, rr);
        } catch (...) {
#pragma omp critical(aisaq_failure)
            if (!failure) failure = std::current_exception();
        }
    }
    if (failure) std::rethrow_exception(failure);
    const auto before = read_bytes();
    capture_window();
    const auto begin = Clock::now();
#pragma omp parallel for num_threads(search_threads) schedule(dynamic, 1)
    for (uint32_t row = 0; row < shape.rows; ++row) {
        try {
            schedule_reset(row, 1);
            const auto started = Clock::now();
            run_row(row, stats[row], rerank_stats[row]);
            wall_us[row] = std::chrono::duration<double, std::micro>(Clock::now() - started).count();
            schedule_collect(counters[row].data(), counters[row].size());
        } catch (...) {
#pragma omp critical(aisaq_failure)
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    const auto elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
    capture_window();
    const auto after = read_bytes();
    if (failure) {
        std::rethrow_exception(failure);
    }

    schedule_dump(output.c_str());
    std::ofstream sample_file(output + ".schedule.csv");
    sample_file
        << "query_row,submitted,reaped,peak_depth,submit_calls,getevents_calls,joint_pairs,multi_reads,"
           "multi_calls,processed_with_pending,singleton_submits,returned_completions,double_harvests\n";
    sample_file << std::setprecision(17);
    for (size_t row = 0; row < counters.size(); ++row) {
        sample_file << row;
        for (double counter : counters[row]) sample_file << ',' << counter;
        sample_file << '\n';
    }
    sample_file.close();
    if (!sample_file) throw std::runtime_error("Could not write scheduler counters");
    std::ofstream csv(output + "_queries.csv");
    csv << "query_row,latency_us,total_us,io_us,cpu_us,io_requests,reads_4k,reads_8k,reads_12k,cache_hits,"
           "hops,rerank_reads,rerank_batches,rerank_us\n";
    uint64_t reported_bytes = 0, rerank_reads = 0, rerank_batches = 0;
    for (uint32_t row = 0; row < shape.rows; ++row) {
        const auto& s = stats[row];
        const auto& rr = rerank_stats[row];
        rerank_reads += rr.reads;
        rerank_batches += rr.batches;
        reported_bytes += uint64_t{4096} * rr.reads;
        reported_bytes += uint64_t{4096} * (s.n_4k + 2 * s.n_8k + 3 * s.n_12k);
        csv << row << ',' << wall_us[row] << ',' << s.total_us << ',' << s.io_us << ',' << s.cpu_us << ','
            << s.n_ios << ',' << s.n_4k << ',' << s.n_8k << ',' << s.n_12k << ',' << s.n_cache_hits << ','
            << s.n_hops << ',' << rr.reads << ',' << rr.batches << ',' << rr.microseconds << '\n';
    }
    std::ofstream id_file(output + "_ids.i64", std::ios::binary);
    id_file.write(reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(int64_t));
    std::ofstream distance_file(output + "_distances.f32", std::ios::binary);
    distance_file.write(reinterpret_cast<const char*>(distances.data()), distances.size() * sizeof(float));
    csv.close();
    id_file.close();
    distance_file.close();
    if (!csv || !id_file || !distance_file) {
        throw std::runtime_error("Could not write search results");
    }
    std::sort(wall_us.begin(), wall_us.end());
    const auto percentile = [&](double quantile) { return wall_us[std::ceil(shape.rows * quantile) - 1]; };
    std::cout << std::setprecision(12) << "BENCH_AISAQ L=" << depth << " queries=" << shape.rows
              << " qps=" << shape.rows / elapsed << " p50_ms=" << percentile(.50) / 1000
              << " p99_ms=" << percentile(.99) / 1000 << " RR_C=" << cap << " RR_WIDTH=" << width
              << " rerank_reads=" << rerank_reads << " rerank_batches=" << rerank_batches
              << " io_bytes=" << reported_bytes << " storage_read_bytes=" << after - before
              << " rss_before_load_kib=" << rss_before_load << " rss_after_load_kib=" << rss_after_load
              << " rss_after_search_kib=" << rss_kib() << " pq_chunks=" << index->get_n_chunks()
              << " max_node_len=" << index->get_max_node_len() << '\n';
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 5 && argc <= 8 && std::string(argv[1]) == "build") {
            return build(argc, argv);
        }
        if (argc >= 6 && argc <= 8 && std::string(argv[1]) == "search") {
            return search(argc, argv);
        }
        std::cerr
            << "Usage: engine build BASE.fbin PREFIX INLINE_PQ [THREADS] [DRAM_GIB] [REARRANGE]\n"
            << "       engine search QUERIES.fbin PREFIX L OUTPUT [PQ_PAGE_CACHE_BYTES] [PQ_CACHE_BYTES]\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
