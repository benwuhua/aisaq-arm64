// Single-core comparison of distance kernels with identical FP32 accumulation.
#include "l2_neon.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

struct Row {
    std::array<uint8_t, 1024> sq;
    std::array<float16_t, 1024> fp;
};
static constexpr float lo = -.375f, scale = .003125f;
// Keep the intermediate buffer visible to the caller, as in the old expand path.
__attribute__((noinline)) static void materialize(const Row& row, bool half, float* dst) {
    if (half) {
        for (unsigned i = 0; i < 1024; i += 4)
            vst1q_f32(dst + i, vcvt_f32_f16(vld1_f16(row.fp.data() + i)));
    } else {
        for (unsigned i = 0; i < 1024; i += 8) {
            auto b = vmovl_u8(vld1_u8(row.sq.data() + i));
            auto a0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b)));
            auto a1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b)));
            vst1q_f32(dst + i, vfmaq_n_f32(vdupq_n_f32(lo), a0, scale));
            vst1q_f32(dst + i + 4, vfmaq_n_f32(vdupq_n_f32(lo), a1, scale));
        }
    }
}
int main() {
    std::mt19937 gen(117);
    std::uniform_real_distribution<float> sample(-1.f, 1.f);
    std::array<float, 1024> query;
    for (auto& x : query) x = sample(gen);
    std::vector<Row> rows(512);
    for (auto& r : rows) {
        for (auto& x : r.sq) x = uint8_t(gen());
        for (auto& x : r.fp) x = sample(gen);
    }
    std::puts("repetition,format,kernel,ns_per_vector,checksum");
    for (unsigned r = 0; r < 8; ++r) {
        for (unsigned h = 0; h < 2; ++h) {
            for (unsigned j = 0; j < 2; ++j) {
                const bool fuse = (j + r) % 2;
                double total = 0;
                auto start = std::chrono::steady_clock::now();
                for (unsigned pass = 0; pass < 64; ++pass) {
                    for (const auto& row : rows) {
                        if (fuse)
                            total += h ? bge_l2::fp16(query.data(), row.fp.data())
                                       : bge_l2::sq8(query.data(), row.sq.data(), lo, scale);
                        else {
                            alignas(64) float scratch[1024];
                            materialize(row, h, scratch);
                            total += bge_l2::fp32(query.data(), scratch);
                        }
                    }
                }
                auto ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
                              .count();
                std::printf("%u,%s,%s,%.3f,%.9f\n", r, h ? "FP16" : "SQ8", fuse ? "fused" : "buffer",
                            ns / (64 * rows.size()), total);
            }
        }
    }
}
