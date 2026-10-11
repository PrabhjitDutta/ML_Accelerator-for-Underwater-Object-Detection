// y26_qc (the kernel's direct-code quantizer) against q_u8 over the constants of every quantized conv input:
// a code may differ only by 1, and only where the exact value sits next to a rounding tie.
//   g++ -O2 -std=c++14 -I<vitis include> <shipping flags> -DY26_YC board_host/test_direct_codes.cpp -o test_direct_codes
//   ./test_direct_codes <weights_dir>
#include "../hls_kernel/conv_engine.h"
#include "../reference_model/layer_ops.h"
#include <cstdio>
#include <random>

int main(int argc, char** argv) {
    Weights W;
    W.load(argc > 1 ? argv[1] : "../weights");
    std::mt19937 rng(1);
    long n = 0, diff = 0, far = 0, offtie = 0;
    for (const auto& kv : W.conv) {
        const ConvW& c = kv.second;
        if (!(c.quant && c.asym)) continue;
        for (int ic = 0; ic < c.ic; ++ic) {
            const float s = c.ssc[ic], lo = c.lo_of(ic), st = c.step_of(ic);
            const float k = s / st, b = lo / st;
            if (!(k >= 1.f && k < 65536.f && b >= -256.f && b <= 0.f)) { std::printf("FAIL: %s ch %d: k %g b %g out of range\n", c.name.c_str(), ic, k, b); return 1; }
            // values across the code range and a little beyond both clamps, plus far outliers
            std::uniform_real_distribution<float> u((lo - 8 * st) / s, (lo + 263 * st) / s);
            for (int i = 0; i < 64; ++i) {
                const float vf = i < 60 ? u(rng) : (i & 1 ? 1.f : -1.f) * (float)(1 << (i - 56)) * 100.f;
                const y26_fx_t v = vf;
                const int got = (int)y26_qc(v, k, b), want = (int)q_u8(v.to_float(), s, lo, st);
                const double m = ((double)v.to_double() * s - lo) / st, fr = m - std::floor(m);
                ++n;  diff += got != want;  far += got > want + 1 || want > got + 1;
                offtie += got != want && std::fabs(fr - 0.5) > 1e-3;
            }
        }
    }
    std::printf("y26_qc vs q_u8: %ld values, %ld differ, %ld by more than 1, %ld away from a tie\n", n, diff, far, offtie);
    if (far || offtie) { std::printf("FAIL\n"); return 1; }
    std::printf("PASS\n");
    return 0;
}
