// test_yq8_codes.cpp - the kernel's code-mode slot (y26_q8, conv_engine.h) decoded by y26_yq_code must equal the
// host's q_u8 code for code, including exact .5 ties and the 0/255 clamps; an escaped slot must carry v's bits + 256.
//   g++ -O2 -std=c++14 -ffp-contract=off -I<vitis_include> <hls_build_flags.txt> -DY26_YQ8 test_yq8_codes.cpp
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include "../hls_kernel/conv_engine.h"
#include "../reference_model/layer_ops.h"

int main() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    long bad = 0, tot = 0, ties = 0, esc[2] = {0, 0}, n[2] = {0, 0};
    for (int t = 0; t < 20000; ++t) {
        const float s = std::ldexp(0.5f + U(rng), (int)(rng() % 9) - 4), st = std::ldexp(0.5f + U(rng), -(int)(rng() % 12));
        const float lo = (U(rng) - 0.5f) * 300.f * st;
        for (int j = 0; j < 400; ++j) {
            float v;
            if (j % 2) v = ((U(rng) - 0.2f) * 330.f * st + lo) / s;                        // spread, both clamps
            else {                                                                            // on / near a tie
                float x = ((float)(rng() % 260) - 2.5f) * st + lo;
                for (int u = (int)(rng() % 5) - 2; u; u += u > 0 ? -1 : 1) x = std::nextafter(x, u > 0 ? 1e30f : -1e30f);
                v = x / s;
            }
            const uint32_t slot = (uint32_t)y26_q8(v, s, lo, 1.f / st);
            uint32_t vb;
            std::memcpy(&vb, &v, 4);
            const bool badesc = slot > 255 && slot - 256 != vb;
            const int want = (int)q_u8(v, s, lo, st), got = y26_yq_code(slot, s, lo, st);
            const float d = (v * s - lo) / st;
            ties += d == std::floor(d) + 0.5f;
            esc[j % 2] += slot > 255; ++n[j % 2];
            bad += want != got || badesc; ++tot;
            if ((want != got || badesc) && bad <= 5)
                std::printf("  v=%a s=%a lo=%a st=%a d=%a: q_u8 %d decoded %d slot %u\n", v, s, lo, st, d, want, got, slot);
        }
    }
    std::printf("test_yq8_codes: %ld values (%ld exact .5 ties), %ld differ -> %s; escapes %.3f%% of the near-tie half, %.3f%% of the spread half\n",
                tot, ties, bad, bad ? "FAIL" : "PASS", 100.0 * esc[0] / n[0], 100.0 * esc[1] / n[1]);
    return bad != 0;
}
