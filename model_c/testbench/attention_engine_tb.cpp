// y26_attn_top against y26_attn_int_head on random codes, both attention blocks' constants.
//   attention_engine_tb [nh=4] [n=400] [seed=1]     exit 1 on any differing J
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "attention_engine.h"
#include "attention_int.h"

int main(int argc, char** argv) {
    const int NH = argc > 1 ? std::atoi(argv[1]) : Y26_ATTN_NH, N = argc > 2 ? std::atoi(argv[2]) : Y26_ATTN_NMAX;
    const int KD = Y26_ATTN_KD, VD = Y26_ATTN_VD;
    std::mt19937 rng(argc > 3 ? std::atoi(argv[3]) : 1);
    // attn_fq.txt steps: q, k, sm
    const double S[2][3] = {{(9.323260784e-01 + 9.396672249e-01) / 255, (7.652013302 + 7.712265491) / 255, 4.648685157e-01 / 255},
                            {(1.720385790 + 1.733932137) / 255, (13.01305199 + 13.11551666) / 255, 9.996662140e-01 / 255}};
    long bad = 0, tot = 0;
    for (int b = 0; b < 2; ++b) {
        const AttnInt ai = {(uint32_t)std::llround(std::ldexp(S[b][0] * S[b][1] / std::log(2.0), 28)), (uint32_t)std::llround(65536.0 / S[b][2])};
        std::normal_distribution<double> G(0, b ? 40.0 : 14.0);
        auto c8 = [&] { const double x = std::round(G(rng)); return (int8_t)(x < -128 ? -128 : x > 127 ? 127 : x); };
        std::vector<int8_t> q((size_t)NH * N * KD), k(q.size()), v((size_t)NH * N * VD);
        for (auto& x : q) x = c8();
        for (auto& x : k) x = c8();
        for (auto& x : v) x = c8();
        // extremes: one saturated query row, one all-equal row
        for (int kd = 0; kd < KD; ++kd) { q[kd] = 127; k[kd] = -128; q[(size_t)KD + kd] = 0; }

        std::vector<ap_uint<128> > X(Y26_ATTN_DEPTH_X), Y(Y26_ATTN_DEPTH_Y);   // full depth: co-sim copies the whole port
        for (int i = 0; i < NH * N; ++i) {
            uint8_t rec[128];
            std::memcpy(rec, &q[(size_t)i * KD], KD);
            std::memcpy(rec + 32, &k[(size_t)i * KD], KD);
            std::memcpy(rec + 64, &v[(size_t)i * VD], VD);
            for (int w = 0; w < Y26_ATTN_XW; ++w) {
                ap_uint<128> x = 0;
                for (int t = 15; t >= 0; --t) x = (x << 8) | rec[w * 16 + t];
                X[(size_t)i * Y26_ATTN_XW + w] = x;
            }
        }
        y26_attn_top(X.data(), Y.data(), NH, N, ai.m, ai.g);

        std::vector<int32_t> J((size_t)NH * N * VD);
        for (int h = 0; h < NH; ++h)
            y26_attn_int_head(&q[(size_t)h * N * KD], &k[(size_t)h * N * KD], &v[(size_t)h * N * VD], &J[(size_t)h * N * VD], N, KD, VD, ai);
        long nz = 0;
        for (size_t i = 0; i < J.size(); ++i) {
            const int32_t got = (int32_t)(uint32_t)Y[i / 4].range((i % 4) * 32 + 31, (i % 4) * 32).to_uint();
            bad += got != J[i]; ++tot; nz += J[i] != 0;
        }
        std::printf("[attn-tb] block %d: m=%u g=%u, %zu J, %ld nonzero\n", b, ai.m, ai.g, J.size(), nz);
    }
    std::printf("[attn-tb] nh=%d n=%d: %ld J compared, %ld differ -> %s\n", NH, N, tot, bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
