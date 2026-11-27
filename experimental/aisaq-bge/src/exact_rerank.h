#pragma once
#include "compact_format.h"
#include "direct_reads.h"
#include "l2_neon.h"
#include <sys/stat.h>
#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

static uint64_t option_u64(const char* key, uint64_t fallback) {
    const char* value = std::getenv(key);
    if (!value) return fallback;
    std::string text(value);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(std::string("invalid unsigned value for ") + key);
    return std::stoull(text);
}
static unsigned bounded_width(const char* key, unsigned fallback) {
    auto n = option_u64(key, fallback);
    if (!n || n > DirectReads::max_width) throw std::runtime_error(std::string(key) + " must be 1..256");
    return unsigned(n);
}
struct RerankStats {
    uint32_t reads = 0, batches = 0, peak = 0;
    uint64_t bytes = 0;
    double microseconds = 0;
};
class ExactReranker {
    using Match = std::pair<float, int64_t>;
    std::vector<Match> ranked_;
    struct Slot {
        size_t candidate;
        unsigned skip;
    };
    std::vector<Slot> slots_;
    std::vector<unsigned> free_, done_;
    int fd_ = -1;
    std::unique_ptr<DirectReads> io_;
    uint64_t count_ = 0, base_ = 0, stride_ = 0, first_ = 0;
    unsigned width_ = 0, read_bytes_ = 4096;
    bool half_ = false, rolling_ = true;

 public:
    std::vector<int64_t> candidates;
    std::vector<float> estimates;
    ExactReranker(const char* vectors, uint64_t offset, uint64_t stride, unsigned width, size_t cap)
        : slots_(width), width_(width), candidates(cap, -1), estimates(cap) {
        if (!cap) return;
        if (!vectors || !width || width > DirectReads::max_width)
            throw std::runtime_error("invalid rerank configuration");
        const std::string mode = std::getenv("RR_SCHEDULE") ? std::getenv("RR_SCHEDULE") : "rolling";
        if (mode != "rolling" && mode != "batch") throw std::runtime_error("invalid RR_SCHEDULE");
        rolling_ = mode == "rolling";
        const std::string format = std::getenv("RR_FORMAT") ? std::getenv("RR_FORMAT") : "fp32";
        if (format != "fp16" && format != "fp32") throw std::runtime_error("invalid RR_FORMAT");
        half_ = format == "fp16";
        fd_ = open(vectors, O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("cannot open O_DIRECT rerank pool");
        try {
            struct stat st{};
            if (fstat(fd_, &st) || st.st_size < 4096) throw std::runtime_error("invalid rerank pool");
            first_ = option_u64("RR_FIRST_ID", 0);
            if (half_) {
                alignas(4096) uint8_t header[4096];
                if (pread(fd_, header, 4096, 0) != 4096 || std::memcmp(header, "BGEF16V1", 8) ||
                    (offset && offset != 4096) || (stride && stride != 2048))
                    throw std::runtime_error("FP16 pool requires its 4 KiB header and 2 KiB rows");
                count_ = compact::load<uint64_t>(header + 8);
                if (!count_ || first_ >= compact::id_limit || count_ > compact::id_limit - first_ ||
                    uint64_t(st.st_size) != 4096 + ((count_ + 1) / 2) * 4096)
                    throw std::runtime_error("incorrect FP16 pool length");
                base_ = 4096;
                stride_ = 2048;
                // Probe the second half of the first page: tests both length and offset.
                // Odd-sized pools are padded, so even a one-row pool has these bytes.
                ssize_t n;
                do {
                    n = pread(fd_, header, 2048, 6144);
                } while (n < 0 && errno == EINTR);
                if (n == 2048)
                    read_bytes_ = 2048;
                else if (n != -1 || (errno != EINVAL && errno != EOPNOTSUPP))
                    throw std::runtime_error("could not probe FP16 direct-I/O geometry");
            } else {
                base_ = offset;
                stride_ = stride ? stride : 4096;
                if (base_ % 4096 || stride_ % 4096 || stride_ < 4096 || base_ > uint64_t(st.st_size) - 4096)
                    throw std::runtime_error("unaligned FP32 vector pool");
                count_ = (uint64_t(st.st_size) - base_ - 4096) / stride_ + 1;
            }
            io_ = std::make_unique<DirectReads>(width_, 4096);
            ranked_.reserve(cap);
            free_.reserve(width_);
            done_.reserve(width_);
        } catch (...) {
            close(fd_);
            throw;
        }
    }
    ~ExactReranker() {
        io_.reset();
        if (fd_ >= 0) close(fd_);
    }
    void run(const float* query, int64_t* ids, float* distances, RerankStats& stats) {
        const auto start = std::chrono::steady_clock::now();
        stats = {};
        const auto end = std::find(candidates.begin(), candidates.end(), int64_t{-1});
        const size_t n = end - candidates.begin();
        if (!std::all_of(end, candidates.end(), [](int64_t id) { return id == -1; }))
            throw std::runtime_error("noncontiguous graph result");
        for (size_t i = 0; i < n; ++i) {
            int64_t id = candidates[i];
            if (id < 0 || uint64_t(id) < first_ || uint64_t(id) - first_ >= count_)
                throw std::runtime_error("graph candidate outside rerank pool");
        }
        ranked_.resize(n);
        free_.clear();
        for (unsigned i = width_; i; --i) free_.push_back(i - 1);
        size_t next = 0;
        unsigned pending = 0;
        // A failed query destroys its queue before a subsequent query can reuse any buffer.
        if (!io_) io_ = std::make_unique<DirectReads>(width_, 4096);
        try {
            while (next < n || pending) {
                if (rolling_ || !pending) {
                    while (next < n && !free_.empty()) {
                        unsigned s = free_.back();
                        free_.pop_back();
                        uint64_t offset = base_ + (uint64_t(candidates[next]) - first_) * stride_;
                        unsigned skip = offset % read_bytes_;
                        slots_[s] = {next++, skip};
                        io_->prepare(s, fd_, read_bytes_, offset - skip);
                        ++pending;
                        ++stats.reads;
                        stats.bytes += read_bytes_;
                    }
                    stats.batches += io_->submit();
                    stats.peak = std::max(stats.peak, pending);
                }
                io_->reap(rolling_ ? 1 : pending, done_);
                for (unsigned s : done_) {
                    auto slot = slots_[s];
                    const auto* v = io_->buffer(s) + slot.skip;
                    float d = half_ ? bge_l2::fp16(query, v)
                                    : bge_l2::fp32(query, reinterpret_cast<const float*>(v));
                    ranked_[slot.candidate] = {d, candidates[slot.candidate]};
                    free_.push_back(s);
                    --pending;
                }
            }
        } catch (...) {
            io_.reset();
            throw;
        }
        size_t k = std::min<size_t>(10, n);
        std::partial_sort(ranked_.begin(), ranked_.begin() + k, ranked_.end());
        for (size_t i = 0; i < k; ++i) {
            distances[i] = ranked_[i].first;
            ids[i] = ranked_[i].second;
        }
        std::fill(ids + k, ids + 10, -1);
        std::fill(distances + k, distances + 10, std::numeric_limits<float>::infinity());
        stats.microseconds =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    }
};
