#include <arm_neon.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
struct View {
    size_t n;
    const unsigned char* p;
    View(const char* file) {
        int f = open(file, O_RDONLY);
        struct stat s{};
        if (f < 0 || fstat(f, &s)) throw std::runtime_error(file);
        n = s.st_size;
        p = (const unsigned char*)mmap(nullptr, n, PROT_READ, MAP_SHARED, f, 0);
        close(f);
        if (p == MAP_FAILED) throw std::runtime_error("map");
    }
    ~View() { munmap((void*)p, n); }
};
template <class T>
T get(const unsigned char* p) {
    T v;
    memcpy(&v, p, sizeof(v));
    return v;
}
template <class T>
void set(unsigned char* p, T v) {
    memcpy(p, &v, sizeof(v));
}

void writeall(int fd, const void* v, size_t n) {
    auto p = (const unsigned char*)v;
    while (n) {
        auto r = write(fd, p, n);
        if (r <= 0) throw std::runtime_error("write");
        p += r;
        n -= r;
    }
}
int main(int ac, char** av) {
    if (ac != 5) return 1;
    int k = std::stoi(av[1]);
    if (k != 48 && k != 56 && k != 64) return 2;
    View g(av[2]), pq(av[3]);
    if (g.n < 4096 || pq.n < 8) throw std::runtime_error("short input header");
    const uint64_t N = get<uint64_t>(g.p + 8), entry = get<uint64_t>(g.p + 24);
    if (N == 0 || N > UINT32_MAX || entry >= N || get<uint64_t>(g.p + 16) != 1024 ||
        get<uint64_t>(g.p + 32) != 6276 || get<uint64_t>(g.p + 40) != 0 || g.n != 4096 + 8192 * N ||
        get<uint32_t>(pq.p) != N || get<uint32_t>(pq.p + 4) != k || pq.n != 8 + uint64_t(k) * N)
        throw std::runtime_error("bad inputs");
    auto tmp = std::string(av[4]) + ".partial";
    int fd = open(tmp.c_str(), O_EXCL | O_WRONLY | O_CREAT, 0600);
    if (fd < 0) throw std::runtime_error(tmp);
    std::array<unsigned char, 4096> header{};
    memcpy(header.data(), "BGELAY01", 8);
    set(header.data() + 8, (uint32_t)k);
    set(header.data() + 12, (uint32_t)N);
    set(header.data() + 16, (uint32_t)entry);
    memcpy(header.data() + 64, pq.p + 8 + uint64_t(k) * entry, k);
    writeall(fd, header.data(), 4096);
    int vec = k == 64 ? 1024 : 2048;
    constexpr int B = 4096;
    std::vector<unsigned char> out(B * 4096);
    for (uint64_t start = 0; start < N; start += B) {
        int nr = std::min<uint64_t>(B, N - start);
#pragma omp parallel for schedule(static)
        for (int j = 0; j < nr; j++) {
            uint32_t id = start + j;
            auto src = g.p + 4096 + uint64_t(id) * 8192;
            auto dst = out.data() + 4096 * j;
            auto f = (const float*)src;
            auto degree = get<uint32_t>(src + 4096);
            if (degree != 32) abort();
            set(dst, id);
            set(dst + 4, uint16_t(0x4b35));
            dst[6] = degree;
            dst[7] = k;
            if (k == 64) {
                float32x4_t lo = vld1q_f32(f), hi = lo;
                for (int q = 4; q < 1024; q += 4) {
                    auto v = vld1q_f32(f + q);
                    lo = vminq_f32(lo, v);
                    hi = vmaxq_f32(hi, v);
                }
                float l = vminvq_f32(lo), h = vmaxvq_f32(hi), scale = (h - l) / 255.f;
                set(dst + 8, l);
                set(dst + 12, scale);
                auto bias = vdupq_n_f32(l), inv = vdupq_n_f32(scale > 0 ? 1.f / scale : 0.f);
                for (int q = 0; q < 1024; q += 8) {
                    auto a = vcvtnq_u32_f32(vmulq_f32(vsubq_f32(vld1q_f32(f + q), bias), inv));
                    auto b = vcvtnq_u32_f32(vmulq_f32(vsubq_f32(vld1q_f32(f + q + 4), bias), inv));
                    vst1_u8(dst + 16 + q, vqmovn_u16(vcombine_u16(vqmovn_u32(a), vqmovn_u32(b))));
                }
            } else {
                for (int q = 0; q < 1024; q += 4)
                    vst1_f16((float16_t*)(dst + 16) + q, vcvt_f16_f32(vld1q_f32(f + q)));
            }
            auto ids = dst + 16 + vec;
            auto codes = ids + 32 * 5;
            for (int e = 0; e < 32; e++) {
                auto nid = get<uint32_t>(src + 4100 + e * 4);
                if (nid >= N) abort();
                set(ids + 5 * e, nid);
                ids[5 * e + 4] = 0;
                if (k != 64) memcpy(codes + k * e, pq.p + 8 + uint64_t(nid) * k, k);
            }
            if (k == 64) memcpy(codes, src + 4228, 2048);
        }
        writeall(fd, out.data(), size_t(nr) * 4096);
        if (!(start % (B * 256))) fprintf(stderr, "packed %lu of %lu\n", start + nr, N);
    }
    if (fdatasync(fd) || close(fd) || rename(tmp.c_str(), av[4])) throw std::runtime_error("finalize");
}
