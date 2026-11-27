#pragma once
#include <fcntl.h>
#include <libaio.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <limits>
#include <utility>

static uint64_t option_u64(const char* key, uint64_t default_value) {
    const char* value = std::getenv(key);
    return value ? std::stoull(value) : default_value;
}

struct RerankStats {
    uint32_t reads = 0;
    uint32_t batches = 0;
    double microseconds = 0;
};

class ExactReranker {
   public:
    ExactReranker(const char* vectors, uint64_t offset, uint64_t stride, uint32_t width, uint32_t cap)
        : base_(offset),
          stride_(stride),
          width_(width),
          candidates(cap),
          estimates(cap),
          ranked(cap),
          blocks_(width),
          requests_(width),
          completions_(width) {
        if (cap == 0) return;
        if (!vectors || stride < 4096 || stride % 4096 || offset % 4096 || !width || width > 256)
            throw std::runtime_error("Rerank needs a vector pool and aligned stride/base; width is 1..256");
        fd_ = open(vectors, O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error(std::string("Rerank open: ") + strerror(errno));
        struct stat st{};
        if (fstat(fd_, &st) || st.st_size < static_cast<off_t>(base_ + 4096))
            throw std::runtime_error("Vector pool is unreadable or too short");
        count_ = (uint64_t(st.st_size) - base_ - 4096) / stride_ + 1;
        if (posix_memalign(&buffers_, 4096, size_t(width_) * 4096) != 0) throw std::bad_alloc();
        memset(buffers_, 0, size_t(width_) * 4096);
        if (io_setup(width_, &context_) != 0) throw std::runtime_error("Rerank io_setup failed");
    }
    ~ExactReranker() {
        if (context_) io_destroy(context_);
        if (fd_ >= 0) close(fd_);
        free(buffers_);
    }
    void run(const float* query, int64_t* best_ids, float* best_dist, RerankStats& stats) {
        const auto begin = std::chrono::steady_clock::now();
        for (size_t start = 0; start < candidates.size(); start += width_) {
            const size_t n = std::min(size_t(width_), candidates.size() - start);
            for (size_t i = 0; i < n; ++i) {
                const auto id = candidates[start + i];
                if (id < 0 || uint64_t(id) >= count_)
                    throw std::runtime_error("Invalid graph candidate ID for vector pool");
                io_prep_pread(&blocks_[i], fd_, buffer(i), 4096, base_ + uint64_t(id) * stride_);
                requests_[i] = &blocks_[i];
            }
            size_t sent = 0;
            size_t done = 0;
            while (done < n) {
                while (sent < n) {
                    const int rc = io_submit(context_, n - sent, requests_.data() + sent);
                    if (rc == -EINTR) continue;
                    if (rc == -EAGAIN && sent > done) break;
                    if (rc <= 0)
                        throw std::runtime_error(std::string("Rerank io_submit: ") + strerror(-rc));
                    sent += rc;
                    ++stats.batches;
                }
                const int rc = io_getevents(context_, 1, sent - done, completions_.data(), nullptr);
                if (rc == -EINTR) continue;
                if (rc <= 0) throw std::runtime_error("Rerank io_getevents failed");
                for (int j = 0; j < rc; ++j) {
                    if (completions_[j].res != 4096 || completions_[j].res2)
                        throw std::runtime_error("Rerank O_DIRECT read failed or was short");
                }
                done += rc;
            }
            stats.reads += n;
            for (size_t i = 0; i < n; ++i) {
                const float* v = buffer(i);
                float squared_l2 = 0;
#pragma omp simd reduction(+ : squared_l2)
                for (size_t d = 0; d < 1024; ++d) {
                    float delta = v[d] - query[d];
                    squared_l2 += delta * delta;
                }
                ranked[start + i] = {squared_l2, candidates[start + i]};
            }
        }
        std::partial_sort(ranked.begin(), ranked.begin() + 10, ranked.end());
        for (size_t i = 0; i < 10; ++i) {
            best_ids[i] = ranked[i].second;
            best_dist[i] = ranked[i].first;
        }
        stats.microseconds =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
    }
    std::vector<int64_t> candidates;
    std::vector<float> estimates;

   private:
    std::vector<std::pair<float, int64_t>> ranked;
    int fd_ = -1;
    io_context_t context_{};
    uint64_t count_ = 0, base_ = 0, stride_ = 0;
    uint32_t width_ = 0;
    std::vector<iocb> blocks_;
    std::vector<iocb*> requests_;
    std::vector<io_event> completions_;
    void* buffers_ = nullptr;
    float* buffer(size_t i) { return reinterpret_cast<float*>(static_cast<char*>(buffers_) + i * 4096); }
};
