#pragma once
#include <arm_neon.h>
#include <cstdint>

namespace bge_l2 {
inline float32x4_t accumulate(float32x4_t sum, const float* query, float32x4_t vector) {
    const auto delta = vsubq_f32(vld1q_f32(query), vector);
    return vfmaq_f32(sum, delta, delta);
}
template <typename Load>
inline float l2(const float* query, Load load4) {
    auto a = vdupq_n_f32(0), b = a, c = a, d = a;
    for (unsigned i = 0; i < 1024; i += 16) {
        a = accumulate(a, query + i, load4(i));
        b = accumulate(b, query + i + 4, load4(i + 4));
        c = accumulate(c, query + i + 8, load4(i + 8));
        d = accumulate(d, query + i + 12, load4(i + 12));
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
}
inline float fp32(const float* query, const float* row) {
    return l2(query, [row](unsigned i) { return vld1q_f32(row + i); });
}
inline float fp16(const float* query, const void* vector) {
    auto row = static_cast<const float16_t*>(vector);
    return l2(query, [row](unsigned i) { return vcvt_f32_f16(vld1_f16(row + i)); });
}
inline float sq8(const float* query, const uint8_t* row, float minimum, float scale) {
    auto a = vdupq_n_f32(0), b = a, c = a, d = a;
    const auto decode = [=](uint16x4_t v) {
        return vfmaq_n_f32(vdupq_n_f32(minimum), vcvtq_f32_u32(vmovl_u16(v)), scale);
    };
    for (unsigned i = 0; i < 1024; i += 16) {
        auto v = vld1q_u8(row + i);
        auto x = vmovl_u8(vget_low_u8(v)), y = vmovl_u8(vget_high_u8(v));
        a = accumulate(a, query + i, decode(vget_low_u16(x)));
        b = accumulate(b, query + i + 4, decode(vget_high_u16(x)));
        c = accumulate(c, query + i + 8, decode(vget_low_u16(y)));
        d = accumulate(d, query + i + 12, decode(vget_high_u16(y)));
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
}
}  // namespace bge_l2
