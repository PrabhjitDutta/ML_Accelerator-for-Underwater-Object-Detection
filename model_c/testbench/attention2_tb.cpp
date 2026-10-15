// y26_attn2_top against y26_attn2_ref on random codes and parameters, both attention blocks' constants.
//   attention2_tb [nh n w [seed=1]]     no arguments: 4 x 400 (20 wide), 2 x 64 (8 wide), 1 x 16 (4 wide). Exit 1 on any
//   differing code.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "attention2_engine.h"
#include "attention_int.h"

int main(int argc, char** argv) {
    const int KD = Y26_ATTN_KD, VD = Y26_ATTN_VD, XC = Y26_ATTN2_XC;
    std::vector<std::vector<int> > cases;
    if (argc > 3) cases.push_back({std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])});
    else cases = {{Y26_ATTN_NH, Y26_ATTN_NMAX, 20}, {2, 64, 8}, {1, 16, 4}};
    std::mt19937 rng(argc > 4 ? std::atoi(argv[4]) : 1);
    // attn_fq.txt steps: q, k, sm
    const double S[2][3] = {{(9.323260784e-01 + 9.396672249e-01) / 255, (7.652013302 + 7.712265491) / 255, 4.648685157e-01 / 255},
                            {(1.720385790 + 1.733932137) / 255, (13.01305199 + 13.11551666) / 255, 9.996662140e-01 / 255}};
    long bad = 0, tot = 0;
    for (const auto& cs : cases)
        for (int b = 0; b < 2; ++b) {
            const int NH = cs[0], N = cs[1], W = cs[2];
            const AttnInt ai = {(uint32_t)std::llround(std::ldexp(S[b][0] * S[b][1] / std::log(2.0), 28)), (uint32_t)std::llround(65536.0 / S[b][2])};
            std::normal_distribution<double> G(0, b ? 40.0 : 14.0);
            std::vector<ap_uint<128> > X(Y26_ATTN2_DEPTH_X), D(Y26_ATTN2_DEPTH_D), P(Y26_ATTN2_DEPTH_P);   // full depth: co-sim copies the whole port
            std::vector<uint8_t> x((size_t)NH * XC * N), ref((size_t)NH * VD * N);
            for (auto& e : x) { const double r = std::round(G(rng)); e = (uint8_t)((int8_t)(r < -128 ? -128 : r > 127 ? 127 : r)) ^ 0x80; }
            // extremes: position 0 of head 0 a saturated query against saturated keys, position 1 an all-zero query
            for (int d = 0; d < KD; ++d) { x[(size_t)d * N] = 0xff;  x[(size_t)(KD + d) * N] = 0x00;  x[(size_t)d * N + 1] = 0x80; }
            // parameters: three channels in four at the model's sizes, one in four at the format's limits
            std::vector<AttnPar> par((size_t)NH * VD);
            auto U = [&](double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng); };
            for (size_t c = 0; c < par.size(); ++c) {
                const bool wide = c % 4 == 3;
                for (int t = 0; t < 9; ++t) par[c].w[t] = (int32_t)std::llround(wide ? U(-131071, 131071) : U(-8000, 8000));
                par[c].m1 = (uint32_t)std::llround(wide ? U(0, 67108862) : U(262144, 8388608));
                par[c].b = std::llround(wide ? U(-5.4e11, 5.4e11) : U(88, 168) * 268435456.0);
            }
            std::vector<uint32_t> pk((size_t)NH * Y26_ATTN2_PW * 4);
            y26_attn2_pack(par.data(), NH, VD, pk.data());
            std::memcpy((void*)P.data(), pk.data(), pk.size() * 4);
            // planes are N/16 words each, back to back
            for (size_t i = 0; i < x.size(); ++i) X[i / 16].range((i % 16) * 8 + 7, (i % 16) * 8) = x[i];
            y26_attn2_top(X.data(), D.data(), P.data(), NH, N, W, ai.m, ai.g);
            y26_attn2_ref(x.data(), ref.data(), par.data(), NH, N, W, KD, VD, ai);
            long lo = 0, hi = 0, bd = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const uint8_t got = (uint8_t)D[i / 16].range((i % 16) * 8 + 7, (i % 16) * 8).to_uint();
                bd += got != ref[i];  lo += ref[i] == 0;  hi += ref[i] == 255;
            }
            bad += bd;  tot += (long)ref.size();
            std::printf("[attn2-tb] nh=%d n=%d w=%d block %d: %zu codes, %ld at 0, %ld at 255, %ld differ\n", NH, N, W, b, ref.size(), lo, hi, bd);
        }
    std::printf("[attn2-tb] %ld codes compared, %ld differ -> %s\n", tot, bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
