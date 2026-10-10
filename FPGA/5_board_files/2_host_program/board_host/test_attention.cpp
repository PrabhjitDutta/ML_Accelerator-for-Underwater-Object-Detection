// test_attention.cpp - y26_attn_qk / y26_attn_vp must equal the plain ascending sums bit for bit.
//   g++ -O2 -ffp-contract=off [-msse4.1 | -mcpu=cortex-a53] -std=c++17 -I<vitis_include> test_attention.cpp
// Sizes cover the vector passes and both scalar tails (odd N, VD not a multiple of 4).
#include <cstdio>
#include <cstring>
#include <random>
#include "../reference_model/layer_ops.h"

int main() {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> U(-4.f, 4.f);
    long bad = 0, tot = 0;
    for (int t = 0; t < 200; ++t) {
        const int N = 1 + (int)(rng() % 420), KD = 1 + (int)(rng() % 40), VD = 1 + (int)(rng() % 70);
        std::vector<float> qb((size_t)KD * 8), kt((size_t)N * KD), v((size_t)VD * N), P((size_t)N * 8), o((size_t)VD * 8);
        for (auto* a : {&qb, &kt, &v}) for (float& x : *a) x = rng() % 7 ? U(rng) : 0.f;
        y26_attn_qk(P.data(), qb.data(), kt.data(), N, KD);
        for (int n2 = 0; n2 < N; ++n2)
            for (int j = 0; j < 8; ++j) {
                float a = 0.f;
                for (int kd = 0; kd < KD; ++kd) a += qb[(size_t)kd * 8 + j] * kt[(size_t)n2 * KD + kd];
                bad += std::memcmp(&a, &P[(size_t)n2 * 8 + j], 4) != 0;
            }
        y26_attn_vp(o.data(), v.data(), P.data(), N, VD);
        for (int vd = 0; vd < VD; ++vd)
            for (int j = 0; j < 8; ++j) {
                float a = 0.f;
                for (int n2 = 0; n2 < N; ++n2) a += v[(size_t)vd * N + n2] * P[(size_t)n2 * 8 + j];
                bad += std::memcmp(&a, &o[(size_t)vd * 8 + j], 4) != 0;
            }
        tot += (long)N * 8 + (long)VD * 8;
    }
    std::printf("test_attention: %ld sums, %ld differ from the plain loops -> %s\n", tot, bad, bad ? "FAIL" : "PASS");
    return bad != 0;
}
