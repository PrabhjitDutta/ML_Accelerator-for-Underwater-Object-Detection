// Integer attention on 8-bit codes: the arithmetic the PL attention block implements, as plain C++.
//   q, k, v codes are signed (FakeQuantize code - 128), so value = code * step.
//   score[n][n2] = sum_kd q[n][kd] * k[n2][kd]                  (times sq*sk in real units)
//   e[n2]        = 2^24 * exp(-(max - score[n][n2]) * sq*sk)    (table, y26_attn_exp)
//   p[n2]        = min(255, round(e[n2] / (sum(e) * s_sm)))     (softmax as its 8-bit code)
//   J[n][vd]     = sum_n2 p[n2] * v[n2][vd]                     (times s_sm*sv in real units)
#pragma once
#include <cstdint>
#include <vector>
#include "attention_exp_lut.h"

// m = round(2^28 * sq*sk / ln 2), g = round(2^16 / s_sm).
struct AttnInt { uint32_t m, g; };

inline uint32_t y26_attn_exp(uint32_t d, uint32_t m) {
    const uint64_t y = ((uint64_t)d * m + (1u << 15)) >> 16;    // d * sq*sk in octaves, Y26_ATTN_EXP_FB fractional bits
    const uint64_t n = y >> Y26_ATTN_EXP_FB;
    return n > 24 ? 0u : Y26_ATTN_EXP[y & ((1u << Y26_ATTN_EXP_FB) - 1)] >> n;
}

// sum >= 2^24: the largest score has e = 2^24.
inline uint32_t y26_attn_recip(uint64_t sum, uint32_t g) { return (uint32_t)(((uint64_t)g << 28) / sum); }

inline uint32_t y26_attn_prob(uint32_t e, uint32_t r) {
    const uint64_t p = ((uint64_t)e * r + (1ull << 43)) >> 44;
    return p > 255 ? 255u : (uint32_t)p;
}

// One head. q, k: [N][KD]; v: [N][VD]; J: [N][VD].
inline void y26_attn_int_head(const int8_t* q, const int8_t* k, const int8_t* v, int32_t* J, int N, int KD, int VD,
                              AttnInt c) {
    std::vector<int32_t> s(N);
    std::vector<uint32_t> e(N);
    for (int n = 0; n < N; ++n) {
        int32_t mx = INT32_MIN;
        for (int n2 = 0; n2 < N; ++n2) {
            int32_t a = 0;
            for (int kd = 0; kd < KD; ++kd) a += (int32_t)q[(size_t)n * KD + kd] * k[(size_t)n2 * KD + kd];
            s[n2] = a;
            if (a > mx) mx = a;
        }
        uint64_t sum = 0;
        for (int n2 = 0; n2 < N; ++n2) { e[n2] = y26_attn_exp((uint32_t)(mx - s[n2]), c.m); sum += e[n2]; }
        const uint32_t r = y26_attn_recip(sum, c.g);
        int32_t* o = J + (size_t)n * VD;
        for (int vd = 0; vd < VD; ++vd) o[vd] = 0;
        for (int n2 = 0; n2 < N; ++n2) {
            const int32_t p = (int32_t)y26_attn_prob(e[n2], r);
            if (p) for (int vd = 0; vd < VD; ++vd) o[vd] += p * v[(size_t)n2 * VD + vd];
        }
    }
}
