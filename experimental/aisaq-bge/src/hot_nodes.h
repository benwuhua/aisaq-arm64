#pragma once
#include "compact_format.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <memory>
#include <limits>
#include <cstdlib>
#include <cstdio>
#include <string>

namespace schedio {
class HotNodes {
    static constexpr size_t budget = 16U << 20;
    // Zero marks an empty bucket, so store a one-based slot in the low 16 bits.
    static constexpr unsigned slot_bits = std::numeric_limits<uint16_t>::digits;
    static constexpr uint64_t slot_mask = std::numeric_limits<uint16_t>::max();
    uint8_t* arena_ = nullptr;
    unsigned buckets_ = 0, stride_ = 0;
    uint64_t* keys() const { return reinterpret_cast<uint64_t*>(arena_); }
    uint8_t* record(unsigned slot) const {
        return arena_ + buckets_ * sizeof(uint64_t) + size_t(slot) * stride_;
    }
    unsigned bucket(uint64_t id) const {
        return unsigned((id * UINT64_C(11400714819323198485)) >> (buckets_ == 4096 ? 52 : 51));
    }

 public:
    void load(int graph_fd, const char* path, const compact::Geometry* layout) {
        if (!path || !*path) return;
        buckets_ = layout ? 8192 : 4096;
        stride_ = layout ? 4096 : 6304;
        const size_t byte_capacity = (budget - buckets_ * sizeof(uint64_t)) / stride_;
        // Linear probing needs an empty bucket even when every selected ID is present.
        const size_t capacity = std::min({byte_capacity, size_t(buckets_ - 1), size_t(slot_mask)});
        alignas(4096) uint8_t scratch[8192];
        uint64_t rows = 0;
        if (!layout) {
            if (pread(graph_fd, scratch, 4096, 0) != 4096 ||
                compact::load<uint64_t>(scratch + 16) != 1024 ||
                compact::load<uint64_t>(scratch + 32) != 6276 || compact::load<uint64_t>(scratch + 40) != 0)
                throw std::runtime_error("invalid native hot graph");
            rows = compact::load<uint64_t>(scratch + 8);
        }
        const char* width = std::getenv("ANN_HOT_ID_BYTES");
        const std::string format = width ? width : "4";
        if (format != "4" && format != "8") throw std::runtime_error("ANN_HOT_ID_BYTES must be 4 or 8");
        const unsigned id_bytes = format == "8" ? 8 : 4;
        const auto close_file = [](FILE* stream) { std::fclose(stream); };
        std::unique_ptr<FILE, decltype(close_file)> stream(fopen(path, "rb"), close_file);
        if (!stream) throw std::runtime_error("cannot open hot ID list");
        struct stat st{};
        if (fstat(fileno(stream.get()), &st) || st.st_size <= 0 || st.st_size % id_bytes)
            throw std::runtime_error("invalid hot ID file size");
        const uint64_t count = uint64_t(st.st_size) / id_bytes;
        if (count > capacity) throw std::runtime_error("hot ID list exceeds cache capacity");
        void* p = mmap(nullptr, budget, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) throw std::bad_alloc();
        arena_ = static_cast<uint8_t*>(p);
        std::memset(arena_, 0, budget);
        for (unsigned i = 0; i < count; ++i) {
            uint64_t id = 0;
            if (fread(&id, id_bytes, 1, stream.get()) != 1 || (layout ? !layout->contains(id) : id >= rows))
                throw std::runtime_error("invalid hot ID");
            unsigned pos = bucket(id);
            while (keys()[pos]) {
                if ((keys()[pos] >> slot_bits) == id) throw std::runtime_error("duplicate hot ID");
                pos = (pos + 1) & (buckets_ - 1);
            }
            unsigned bytes = layout ? 4096 : 8192;
            uint64_t offset = layout ? layout->offset(id) : 4096 + id * 8192;
            if (pread(graph_fd, scratch, bytes, offset) != bytes)
                throw std::runtime_error("short hot preload");
            if (layout) layout->check_record(scratch, id);
            std::memcpy(record(i), scratch, layout ? 4096 : 6276);
            keys()[pos] = (id << slot_bits) | (i + 1);
        }
        if (mprotect(arena_, budget, PROT_READ))
            throw std::runtime_error("cannot protect immutable hot cache");
        std::fprintf(stderr,
                     "HOT_CACHE nodes=%zu capacity=%zu allocated_bytes=%zu stride=%u table_bytes=%zu\n",
                     size_t(count), capacity, budget, stride_, buckets_ * sizeof(uint64_t));
    }
    const uint8_t* find(uint64_t id) const {
        if (!arena_) return nullptr;
        for (unsigned pos = bucket(id); keys()[pos]; pos = (pos + 1) & (buckets_ - 1)) {
            uint64_t key = keys()[pos];
            if ((key >> slot_bits) == id) return record(unsigned(key & slot_mask) - 1);
        }
        return nullptr;
    }
    ~HotNodes() {
        if (arena_) munmap(arena_, budget);
    }
};
}  // namespace schedio
