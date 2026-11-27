#include "compact_format.h"
#include <arm_neon.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using compact::load;
using compact::store;

struct View {
    const uint8_t* data;
    uint64_t size;
    explicit View(const char* path) {
        int fd = open(path, O_RDONLY | O_CLOEXEC);
        struct stat st{};
        if (fd < 0) throw std::runtime_error(path);
        if (fstat(fd, &st) || st.st_size <= 0) {
            close(fd);
            throw std::runtime_error(path);
        }
        size = st.st_size;
        data = static_cast<const uint8_t*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
        close(fd);
        if (data == MAP_FAILED) throw std::runtime_error("could not map input");
    }
    ~View() { munmap(const_cast<uint8_t*>(data), size); }
};

class Output {
    std::string path_, tmp_;
    int fd_;
    bool published_ = false;

 public:
    explicit Output(const char* path)
        : path_(path),
          tmp_(path_ + ".partial"),
          fd_(open(tmp_.c_str(), O_WRONLY | O_EXCL | O_CREAT | O_CLOEXEC, 0600)) {
        if (fd_ < 0) throw std::runtime_error("cannot create " + tmp_);
    }
    ~Output() {
        if (fd_ >= 0) close(fd_);
        if (!published_) unlink(tmp_.c_str());
    }
    void append(const void* buf, size_t size) {
        auto p = static_cast<const uint8_t*>(buf);
        while (size) {
            auto n = write(fd_, p, size);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) throw std::runtime_error("incomplete output write");
            p += n;
            size -= n;
        }
    }
    void finish() {
        if (fdatasync(fd_) || link(tmp_.c_str(), path_.c_str()))
            throw std::runtime_error("could not publish output (destination must be new)");
        if (unlink(tmp_.c_str())) throw std::runtime_error("could not remove temporary output name");
        published_ = true;
    }
};

static void half_row(const float* row, uint8_t* dst) {
    for (unsigned i = 0; i < 1024; i += 4)
        vst1_f16(reinterpret_cast<float16_t*>(dst) + i, vcvt_f16_f32(vld1q_f32(row + i)));
}
static void sq_row(const float* row, uint8_t* dst) {
    float32x4_t lo = vld1q_f32(row), hi = lo;
    for (unsigned i = 4; i < 1024; i += 4) {
        auto v = vld1q_f32(row + i);
        lo = vminq_f32(lo, v);
        hi = vmaxq_f32(hi, v);
    }
    const float minimum = vminvq_f32(lo), scale = (vmaxvq_f32(hi) - minimum) / 255.f;
    store(dst + 8, minimum);
    store(dst + 12, scale);
    auto quantize = [&](unsigned i) {
        auto diff = vsubq_f32(vld1q_f32(row + i), vdupq_n_f32(minimum));
        return vqmovn_u32(vcvtnq_u32_f32(vmulq_n_f32(diff, scale > 0 ? 1.f / scale : 0.f)));
    };
    for (unsigned i = 0; i < 1024; i += 8)
        vst1_u8(dst + 16 + i, vqmovn_u16(vcombine_u16(quantize(i), quantize(i + 4))));
}
static constexpr unsigned batch = 2048;
static void pack_half_pool(const View& base, const char* path, bool raw) {
    const unsigned offset = raw ? 0 : 8;
    if (base.size < offset) throw std::runtime_error("short FP32 pool");
    uint64_t rows = raw ? base.size / 4096 : load<uint32_t>(base.data);
    if (!rows || rows > compact::id_limit || (!raw && load<uint32_t>(base.data + 4) != 1024) ||
        base.size != offset + rows * 4096)
        throw std::runtime_error("invalid FP32 pool shape");
    Output out(path);
    std::array<uint8_t, 4096> header{};
    std::memcpy(header.data(), "BGEF16V1", 8);
    store(header.data() + 8, rows);
    out.append(header.data(), header.size());
    std::vector<uint8_t> buf(batch * 2048);
    for (uint64_t start = 0; start < rows; start += batch) {
        int n = std::min<uint64_t>(rows - start, batch);
#pragma omp parallel for schedule(static)
        for (int j = 0; j < n; ++j)
            half_row(reinterpret_cast<const float*>(base.data + offset + (start + j) * 4096),
                     buf.data() + j * 2048);
        out.append(buf.data(), n * 2048);
    }
    if (rows % 2) out.append(header.data() + 2048, 2048);
    out.finish();
}

static void pack(const compact::Geometry& g, const View& vectors, const View& codes, const View* adjacency,
                 const char* path) {
    g.validate();
    const bool wide = adjacency != nullptr;
    const uint8_t* pq = codes.data + (wide ? 0 : 8);
    if (codes.size != g.rows * g.width + (wide ? 0 : 8))
        throw std::runtime_error("PQ stream size mismatch");
    if (wide && (vectors.size != g.rows * 4096 || adjacency->size != g.rows * (4 + 32 * 8)))
        throw std::runtime_error("invalid wide source geometry");
    Output out(path);
    std::array<uint8_t, 4096> header{};
    g.encode(header.data());
    std::memcpy(header.data() + 64, pq + (g.entry - g.first) * g.width, g.width);
    out.append(header.data(), header.size());
    std::vector<uint8_t> buf(batch * 4096);
    for (uint64_t start = 0; start < g.rows; start += batch) {
        int n = std::min<uint64_t>(g.rows - start, batch);
        int invalid = n;
#pragma omp parallel for schedule(static) reduction(min : invalid)
        for (int j = 0; j < n; ++j) {
            uint64_t row = start + j;
            auto src = vectors.data + (wide ? row * 4096 : 4096 + row * 8192);
            auto edges = wide ? adjacency->data + row * (4 + 32 * 8) : src + 4096;
            auto degree = load<uint32_t>(edges);
            if (degree > 32) {
                invalid = std::min(invalid, j);
                continue;
            }
            auto* dst = buf.data() + 4096 * j;
            std::memset(dst, 0, 4096);
            compact::put_id40(dst, g.first + row);
            dst[5] = 0x4b;
            dst[6] = degree;
            dst[7] = g.width;
            auto vector = reinterpret_cast<const float*>(src);
            if (g.width == 64)
                sq_row(vector, dst);
            else
                half_row(vector, dst + 16);
            auto ids = dst + 16 + g.vector_bytes();
            auto neighbors = ids + 32 * 5;
            for (unsigned e = 0; e < degree; ++e) {
                uint64_t id = wide ? load<uint64_t>(edges + 4 + 8 * e) : load<uint32_t>(edges + 4 + 4 * e);
                if (!g.contains(id)) {
                    invalid = std::min(invalid, j);
                    break;
                }
                compact::put_id40(ids + 5 * e, id);
                const auto* code =
                    !wide && g.width == 64 ? src + 4228 + 64 * e : pq + (id - g.first) * g.width;
                std::memcpy(neighbors + g.width * e, code, g.width);
            }
        }
        if (invalid < n)
            throw std::runtime_error("invalid adjacency at node " +
                                     std::to_string(g.first + start + invalid));
        out.append(buf.data(), size_t(n) * 4096);
    }
    out.finish();
}
static unsigned parse_width(const char* arg) {
    const std::string value(arg);
    if (value == "48") return 48;
    if (value == "56") return 56;
    if (value == "64") return 64;
    throw std::runtime_error("PQ width must be 48, 56 or 64");
}

int main(int argc, char** argv) try {
    if (argc == 4 && (std::string(argv[1]) == "fp16-pool" || std::string(argv[1]) == "fp16-pool-raw")) {
        pack_half_pool(View(argv[2]), argv[3], std::string(argv[1]) == "fp16-pool-raw");
    } else if (argc == 9 && std::string(argv[1]) == "wide") {
        View vec(argv[5]), edges(argv[6]), codes(argv[7]);
        compact::Geometry g{parse_width(argv[2]), vec.size / 4096, std::stoull(argv[3]),
                            std::stoull(argv[4])};
        pack(g, vec, codes, &edges, argv[8]);
    } else if (argc == 5) {
        View native(argv[2]), codes(argv[3]);
        if (native.size < 4096 || codes.size < 8) throw std::runtime_error("short native header");
        compact::Geometry g{parse_width(argv[1]), load<uint64_t>(native.data + 8), 0,
                            load<uint64_t>(native.data + 24)};
        g.validate();
        if (g.rows > UINT32_MAX || load<uint64_t>(native.data + 16) != 1024 ||
            load<uint64_t>(native.data + 32) != 6276 || load<uint64_t>(native.data + 40) != 0 ||
            native.size != 4096 + 8192 * g.rows || load<uint32_t>(codes.data) != g.rows ||
            load<uint32_t>(codes.data + 4) != g.width)
            throw std::runtime_error("invalid native AiSAQ inputs");
        pack(g, native, codes, nullptr, argv[4]);
    } else
        throw std::runtime_error(
            "usage: pack K NATIVE CODES OUT | wide K FIRST ENTRY FP32_RAW ADJ_U64 PQ_RAW OUT | fp16-pool "
            "BASE.fbin OUT | fp16-pool-raw FP32_RAW OUT");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
