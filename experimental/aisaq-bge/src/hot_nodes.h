#include <sys/mman.h>

namespace schedio {
// Specific to the original BGE-M3-10M full-inline AiSAQ index used in this test.
class HotNodes {
    struct Key {
        uint32_t id_plus_one, record;
    };
    static_assert(sizeof(Key) == 8);
    static constexpr size_t bytes = 16U << 20;
    static constexpr uint32_t capacity = 4096;
    static constexpr size_t record_bytes = 6276;
    static constexpr size_t stride = 6304;  // 32-byte aligned original FP32 coordinates
    static constexpr uint32_t count = (bytes - capacity * sizeof(Key)) / stride;
    static_assert(count == 2656);
    char* arena = nullptr;

    static uint32_t bucket(uint32_t id) { return (id * 2654435761U) >> 20; }
    Key* keys() const { return reinterpret_cast<Key*>(arena); }
    char* record(uint32_t i) const { return arena + capacity * sizeof(Key) + i * stride; }

   public:
    void load(int graph_fd, const char* ids) {
        if (!ids || !*ids) return;
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) fail("hot arena allocation", errno);
        arena = static_cast<char*>(p);
        memset(arena, 0, bytes);  // Touch and charge the whole cache inside the process cgroup.
        void* scratch = nullptr;
        if (posix_memalign(&scratch, 4096, 8192)) fail("preload scratch allocation");
        if (pread(graph_fd, scratch, 4096, 0) != 4096) fail("read graph header", errno);
        uint64_t header[5];
        memcpy(header, static_cast<char*>(scratch) + 8, sizeof(header));
        if (header[0] < count || header[0] > UINT32_MAX || header[1] != 1024 || header[2] >= header[0] ||
            header[3] != record_bytes || header[4] != 0)
            fail("unexpected graph header");
        FILE* index = fopen(ids, "rb");
        if (!index) fail("open static hot IDs", errno);
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t id;
            if (fread(&id, sizeof(id), 1, index) != 1 || id >= header[0]) fail("bad static ID list");
            uint32_t pos = bucket(id);
            while (keys()[pos].id_plus_one) {
                if (keys()[pos].id_plus_one == id + 1) fail("duplicate hot ID");
                pos = (pos + 1) & (capacity - 1);
            }
            off_t offset = 4096 + off_t(id) * 8192;
            if (pread(graph_fd, scratch, 8192, offset) != 8192) fail("hot preload graph pread", errno);
            memcpy(record(i), scratch, record_bytes);
            keys()[pos] = {id + 1, i};
        }
        if (fgetc(index) != EOF || ferror(index)) fail("unexpected hot ID list length or read error");
        fclose(index);
        free(scratch);
        if (mprotect(arena, bytes, PROT_READ)) fail("protect immutable hot arena", errno);
        fprintf(stderr,
                "HOT_CACHE nodes=%u allocated_bytes=%zu record_bytes=%zu stride=%zu table_bytes=%zu "
                "graph_reads=%u\n",
                count, bytes, record_bytes, stride, capacity * sizeof(Key), count);
    }
    char* find(uint32_t id) const {
        if (!arena) return nullptr;
        uint32_t pos = bucket(id);
        while (keys()[pos].id_plus_one) {
            if (keys()[pos].id_plus_one == id + 1) return record(keys()[pos].record);
            pos = (pos + 1) & (capacity - 1);
        }
        return nullptr;
    }
    ~HotNodes() {
        if (arena) munmap(arena, bytes);
    }
};
}  // namespace schedio
