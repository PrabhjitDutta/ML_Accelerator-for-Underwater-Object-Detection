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

// The whole Attention module between its two convs (block v2), on their code planes:
//   X: qkv's output as codes, [NH * (2*KD + VD)][N] unsigned; per head KD planes of q, KD of k, VD of v; code ^ 0x80 is
//      the signed code above.
//   PE[n][c]   = sum over the 3x3 neighbours t of n inside the map of W[c][t] * v[t][c]        (pe, depthwise)
//   code[n][c] = clamp((sat18(J[n][c]) * m1[c] + PE[n][c] * 2^(OF - WF) + B[c] + 2^(OF - 1)) >> OF, 0, 255)
//   D: the reader conv's input codes, [NH * VD][N].
// W holds pe's taps (BN folded) and m1 the J scale, both in the reader's code units; B its bias and zero point.
#define Y26_ATTN_WF 12           // fractional bits of W;  |W| < 2^17
#define Y26_ATTN_OF 28           // fractional bits of m1 and B;  m1 < 2^26, |B| < 2^39
struct AttnPar { int32_t w[9]; uint32_t m1; int64_t b; };

inline uint8_t y26_attn2_code(int32_t J, int64_t pe, const AttnPar& p) {
    const int64_t j = J < -(1 << 17) ? -(1 << 17) : J > (1 << 17) - 1 ? (1 << 17) - 1 : J;
    const int64_t r = (j * (int64_t)p.m1 + pe * ((int64_t)1 << (Y26_ATTN_OF - Y26_ATTN_WF)) + p.b + ((int64_t)1 << (Y26_ATTN_OF - 1))) >> Y26_ATTN_OF;
    return (uint8_t)(r < 0 ? 0 : r > 255 ? 255 : r);
}

// Wd: the map's width (N = H * Wd).
inline void y26_attn2_ref(const uint8_t* X, uint8_t* D, const AttnPar* P, int NH, int N, int Wd, int KD, int VD, AttnInt c) {
    const int PH = 2 * KD + VD, H = N / Wd;
    std::vector<int8_t> q((size_t)N * KD), k((size_t)N * KD), v((size_t)N * VD);
    std::vector<int32_t> J((size_t)N * VD);
    for (int h = 0; h < NH; ++h) {
        const uint8_t* xh = X + (size_t)h * PH * N;
        for (int n = 0; n < N; ++n) {
            for (int d = 0; d < KD; ++d) {
                q[(size_t)n * KD + d] = (int8_t)(xh[(size_t)d * N + n] ^ 0x80);
                k[(size_t)n * KD + d] = (int8_t)(xh[(size_t)(KD + d) * N + n] ^ 0x80);
            }
            for (int d = 0; d < VD; ++d) v[(size_t)n * VD + d] = (int8_t)(xh[(size_t)(2 * KD + d) * N + n] ^ 0x80);
        }
        y26_attn_int_head(q.data(), k.data(), v.data(), J.data(), N, KD, VD, c);
        for (int n = 0; n < N; ++n)
            for (int d = 0; d < VD; ++d) {
                const AttnPar& p = P[h * VD + d];
                int64_t pe = 0;
                for (int t = 0; t < 9; ++t) {
                    const int y = n / Wd + t / 3 - 1, x = n % Wd + t % 3 - 1;
                    if (y >= 0 && y < H && x >= 0 && x < Wd) pe += (int64_t)p.w[t] * v[(size_t)(y * Wd + x) * VD + d];
                }
                D[(size_t)(h * VD + d) * N + n] = y26_attn2_code(J[(size_t)n * VD + d], pe, p);
            }
    }
}

// P as the block reads it: per head 9 taps of VD int32 W, then VD uint32 m1, then VD int64 B. o: NH * 12 * VD words.
inline void y26_attn2_pack(const AttnPar* P, int NH, int VD, uint32_t* o) {
    for (int h = 0; h < NH; ++h)
        for (int c = 0; c < VD; ++c) {
            const AttnPar& p = P[h * VD + c];
            uint32_t* b = o + (size_t)h * 12 * VD;
            for (int t = 0; t < 9; ++t) b[t * VD + c] = (uint32_t)p.w[t];
            b[9 * VD + c] = p.m1;
            b[10 * VD + 2 * c] = (uint32_t)(uint64_t)p.b;
            b[10 * VD + 2 * c + 1] = (uint32_t)((uint64_t)p.b >> 32);
        }
}
