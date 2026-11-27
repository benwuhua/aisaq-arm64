#include "exact_rerank.h"
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
int main(int argc, char** argv) try {
    if (argc != 2) return 2;
    alignas(64) std::array<float, 1024> query{}, row{};
    std::array<float16_t, 1024> half{};
    std::array<uint8_t, 1024> codes{};
    double expected = 0, expected_half = 0, expected_sq = 0;
    for (unsigned i = 0; i < 1024; ++i) {
        query[i] = std::sin(i * .013);
        row[i] = std::cos(i * .007);
        half[i] = row[i];
        codes[i] = uint8_t(i);
        expected += std::pow(double(query[i]) - row[i], 2);
        expected_half += std::pow(double(query[i]) - float(half[i]), 2);
        expected_sq += std::pow(double(query[i]) - std::fma(float(codes[i]), .013f, -1.2f), 2);
    }
    require(
        std::abs(bge_l2::sq8(query.data(), codes.data(), -1.2f, .013f) - expected_sq) < expected_sq * 3e-6,
        "SQ8 L2 differs from scalar");
    require(std::abs(bge_l2::fp16(query.data(), half.data()) - expected_half) < expected_half * 3e-6,
            "FP16 L2 differs from scalar");
    require(std::abs(bge_l2::fp32(query.data(), row.data()) - expected) < expected * 3e-6,
            "FP32 L2 differs from scalar");
    std::fill(query.begin(), query.end(), 0);
    for (size_t cap : {1, 3, 16}) {
        ExactReranker rr(argv[1], 0, 0, 4, cap);
        for (size_t n :
             {std::min(cap, size_t(13)), std::min(cap, size_t(2)), size_t(0), cap > 1 ? size_t(1) : cap}) {
            std::fill(rr.candidates.begin(), rr.candidates.end(), -1);
            for (size_t i = 0; i < n; ++i) rr.candidates[i] = 12 - i;
            int64_t ids[10];
            float d[10];
            RerankStats stats;
            rr.run(query.data(), ids, d, stats);
            require(stats.reads == n && stats.peak <= 4, "rerank submitted invalid reads");
            for (size_t i = 0; i < 10; ++i) {
                if (i < n) {
                    int64_t id = 13 - n + i;
                    require(ids[i] == id && d[i] == 1024.f * id * id, "incorrect valid candidate");
                } else
                    require(ids[i] == -1 && std::isinf(d[i]), "stale candidate leaked");
            }
        }
    }
    std::cout << "distance kernels and successive short, empty and full candidate lists passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
