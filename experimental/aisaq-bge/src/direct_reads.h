#pragma once
#include <libaio.h>
#ifdef AISAQ_HAVE_URING
#include <liburing.h>
#endif
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Each slot owns its aligned buffer until the caller consumes its completion.
class DirectReads {
    struct Request {
        int fd;
        uint32_t bytes;
        uint64_t offset;
    };
    std::vector<Request> requests_;
    std::vector<unsigned> prepared_, saved_;
    std::vector<iocb> blocks_;
    std::vector<iocb*> pointers_;
    std::vector<io_event> events_;
    std::vector<uint8_t> busy_;
    std::unique_ptr<void, decltype(&std::free)> buffers_{nullptr, &std::free};
    size_t stride_;
    io_context_t aio_{};
    unsigned pending_ = 0;
#ifdef AISAQ_HAVE_URING
    io_uring ring_{};
    bool ring_open_ = false, sqpoll_ = false;
#endif
    [[noreturn]] static void error(const char* what, int rc) {
        throw std::runtime_error(std::string(what) + ": " + std::strerror(-rc));
    }
    void completed(unsigned slot, long result, std::vector<unsigned>& out) {
        if (!pending_) throw std::logic_error("unexpected direct read completion");
        --pending_;
        if (slot >= busy_.size() || busy_[slot] != 1) throw std::runtime_error("invalid completion slot");
        busy_[slot] = 2;
        if (result != requests_[slot].bytes)
            throw std::runtime_error("failed or short direct read (result " + std::to_string(result) + ")");
        out.push_back(slot);
    }
    void fetch(unsigned minimum, std::vector<unsigned>& out) {
        if (!pending_) return;
#ifdef AISAQ_HAVE_URING
        if (ring_open_) {
            const size_t start = out.size();
            do {
                if (out.size() - start < minimum && !io_uring_cq_ready(&ring_)) {
                    io_uring_cqe* first;
                    int rc;
                    do {
                        rc = io_uring_wait_cqe(&ring_, &first);
                    } while (rc == -EINTR);
                    if (rc < 0) error("io_uring_wait_cqe", rc);
                }
                unsigned head, seen = 0;
                io_uring_cqe* cqe;
                try {
                    io_uring_for_each_cqe(&ring_, head, cqe) {
                        ++seen;
                        completed(unsigned(cqe->user_data), cqe->res, out);
                    }
                } catch (...) {
                    io_uring_cq_advance(&ring_, seen);
                    throw;
                }
                if (seen) io_uring_cq_advance(&ring_, seen);
            } while (out.size() - start < minimum);
            return;
        }
#endif
        timespec zero{};
        int rc;
        do {
            rc = io_getevents(aio_, minimum, pending_, events_.data(), minimum ? nullptr : &zero);
        } while (rc == -EINTR);
        if (rc < 0) error("io_getevents", rc);
        for (int i = 0; i < rc; ++i) {
            auto& e = events_[i];
            unsigned slot = uintptr_t(e.data);
            if (slot >= blocks_.size() || e.obj != &blocks_[slot] || e.res2)
                throw std::runtime_error("invalid libaio completion");
            completed(slot, static_cast<long>(e.res), out);
        }
    }

 public:
    static constexpr unsigned max_width = 256;
    DirectReads(unsigned width, size_t stride)
        : requests_(width),
          blocks_(width),
          pointers_(width),
          events_(width),
          busy_(width),
          stride_(stride) {
        if (!width || width > max_width || !stride || stride % 4096)
            throw std::runtime_error("invalid I/O geometry");
        prepared_.reserve(width);
        saved_.reserve(width);
        void* p = nullptr;
        if (posix_memalign(&p, 4096, width * stride)) throw std::bad_alloc();
        buffers_.reset(p);
        const std::string backend =
            std::getenv("ANN_IO_BACKEND") ? std::getenv("ANN_IO_BACKEND") : "libaio";
        const bool sqpoll = std::getenv("ANN_SQPOLL") && std::string(std::getenv("ANN_SQPOLL")) == "1";
        if (backend == "uring") {
#ifdef AISAQ_HAVE_URING
            io_uring_params params{};
            if (sqpoll) {
                params.flags = IORING_SETUP_SQPOLL;
                params.sq_thread_idle = 1000;
            }
            int rc = io_uring_queue_init_params(width, &ring_, &params);
            if (rc < 0) error("io_uring_queue_init_params", rc);
            ring_open_ = true;
            sqpoll_ = sqpoll;
            return;
#else
            throw std::runtime_error("io_uring backend requires a build with liburing");
#endif
        }
        if (backend != "libaio" || sqpoll) throw std::runtime_error("invalid I/O backend or SQPOLL option");
        int rc = io_setup(width, &aio_);
        if (rc < 0) error("io_setup", rc);
    }
    ~DirectReads() {
#ifdef AISAQ_HAVE_URING
        if (ring_open_) {
            // On a failed search, retire previously submitted reads before freeing their buffers.
            while (pending_) {
                io_uring_cqe* cqe;
                int rc = io_uring_wait_cqe(&ring_, &cqe);
                if (rc == -EINTR) continue;
                if (rc < 0) break;
                io_uring_cqe_seen(&ring_, cqe);
                --pending_;
            }
            io_uring_queue_exit(&ring_);
        }
#endif
        if (aio_) io_destroy(aio_);
    }
    uint8_t* buffer(unsigned slot) { return static_cast<uint8_t*>(buffers_.get()) + slot * stride_; }
    void prepare(unsigned slot, int fd, uint32_t bytes, uint64_t offset) {
        if (slot >= busy_.size() || busy_[slot] || bytes > stride_)
            throw std::logic_error("busy or invalid read slot");
        busy_[slot] = true;
        requests_[slot] = {fd, bytes, offset};
        prepared_.push_back(slot);
    }
    unsigned submit() {
        unsigned total = prepared_.size();
        if (!total) return 0;
        for (unsigned i = 0; i < total; ++i) {
            unsigned slot = prepared_[i];
            auto& r = requests_[slot];
#ifdef AISAQ_HAVE_URING
            if (ring_open_) {
                auto sqe = io_uring_get_sqe(&ring_);
                if (!sqe) throw std::logic_error("unexpected full SQ");
                io_uring_prep_read(sqe, r.fd, buffer(slot), r.bytes, r.offset);
                io_uring_sqe_set_data64(sqe, slot);
                continue;
            }
#endif
            io_prep_pread(&blocks_[slot], r.fd, buffer(slot), r.bytes, r.offset);
            blocks_[slot].data = reinterpret_cast<void*>(uintptr_t(slot));
            pointers_[i] = &blocks_[slot];
        }
        unsigned sent = 0, calls = 0;
        while (sent < total) {
            int rc;
#ifdef AISAQ_HAVE_URING
            if (ring_open_)
                rc = io_uring_submit(&ring_);
            else
#endif
                rc = io_submit(aio_, total - sent, pointers_.data() + sent);
            ++calls;
            if (rc == -EINTR) continue;
            if (rc == -EAGAIN && pending_) {
                fetch(1, saved_);
                continue;
            }
            if (rc < 0) error("direct read submit", rc);
#ifdef AISAQ_HAVE_URING
            // SQPOLL publishes the whole batch. Its return count can also include older SQEs
            // that the polling thread has not consumed yet.
            if (ring_open_ && sqpoll_) rc = total - sent;
#endif
            if (rc == 0 || unsigned(rc) > total - sent)
                throw std::runtime_error("invalid submission count");
            sent += rc;
            pending_ += rc;
        }
        prepared_.clear();
        return calls;
    }
    void reap(unsigned minimum, std::vector<unsigned>& out) {
        out.clear();
        out.insert(out.end(), saved_.begin(), saved_.end());
        saved_.clear();
        const unsigned need = out.size() < minimum ? minimum - out.size() : 0;
        if (need > pending_) throw std::logic_error("completion request exceeds submitted reads");
        do {
            fetch(out.size() < minimum ? minimum - out.size() : 0, out);
        } while (out.size() < minimum);
        for (unsigned s : out) busy_[s] = false;
    }
};
