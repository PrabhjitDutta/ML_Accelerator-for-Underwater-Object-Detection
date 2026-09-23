// test_quantizer.cpp - quant_plane and FQ::apply (multiply + tie fallback) must equal their divide forms exactly.
//   g++ -O2 -ffp-contract=off [-msse4.1 | -mcpu=cortex-a53] -std=c++17 -I<vitis_include> test_quantizer.cpp
// Random activations/scales, plus values planted on and 1-4 ulps around every .5 tie of the code range.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include "../reference_model/layer_ops.h"

int main() {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    long bad = 0, tot = 0;
    for (int t = 0; t < 4000; ++t) {
        const float s = std::ldexp(0.5f + U(rng), (int)(rng() % 9) - 4), st = std::ldexp(0.5f + U(rng), -(int)(rng() % 10));
        const float lo = (U(rng) - 0.7f) * 300.f * st;
        const size_t n = 64 * 40 + rng() % 64;               // blocks + a scalar tail
        std::vector<float> x(n);
        for (size_t j = 0; j < n; ++j) {
            if (j % 3) { x[j] = (U(rng) - 0.3f) * 400.f * st / s; continue; }
            float v = ((float)(rng() % 258) - 1.5f) * st + lo; // on a tie ...
            for (int u = (int)(rng() % 9) - 4; u; u += u > 0 ? -1 : 1) v = std::nextafter(v, u > 0 ? 1e30f : -1e30f);  // ... +-ulps
            x[j] = v / s;
        }
        std::vector<uint8_t> a(n);
        quant_plane(a.data(), x.data(), n, s, lo, st);
        for (size_t j = 0; j < n; ++j) bad += a[j] != (uint8_t)(int)q_u8(x[j], s, lo, st);
        tot += n;
    }
    std::printf("test_quant: %ld values, %ld differ from q_u8 -> %s\n", tot, bad, bad ? "FAIL" : "PASS");
    // FQ::apply (attention fake-quant, divide-free) == FQ::operator()(x * pre), bit for bit: stride 1 and QB=8,
    // in place, pre = 1 (k, v) and a random pre (q's scale, softmax's 1/sum); lo < 0 < hi and lo = 0 (softmax).
    long fbad = 0, ftot = 0;
    for (int t = 0; t < 4000; ++t) {
        FQ f;
        const float h = std::ldexp(0.5f + U(rng), (int)(rng() % 8) - 4);
        f.set(t % 4 == 0 ? 0.f : -h * (0.9f + 0.2f * U(rng)), h);
        const float pre = t % 2 ? 1.f : std::ldexp(0.5f + U(rng), (int)(rng() % 5) - 2);
        const size_t stride = t % 3 ? 1 : 8, n = 64 * 7 + rng() % 64;
        std::vector<float> x(n * stride), y;
        for (size_t j = 0; j < n * stride; ++j) {
            if (j % 3) { x[j] = (U(rng) - 0.5f) * 3.f * h / pre; continue; }
            float v = ((float)(rng() % 258) - 1.5f) * f.step + f.lo;   // on a tie (of the pre-scaled value) ...
            for (int u = (int)(rng() % 9) - 4; u; u += u > 0 ? -1 : 1) v = std::nextafter(v, u > 0 ? 1e30f : -1e30f);
            x[j] = v / pre;
        }
        y = x;
        f.apply(y.data(), y.data(), n, stride, pre);
        for (size_t j = 0; j < n * stride; ++j) {
            const float want = j % stride ? x[j] : f(x[j] * pre);
            fbad += std::memcmp(&want, &y[j], 4) != 0;
        }
        ftot += n;
    }
    std::printf("test_quant: FQ::apply %ld values, %ld differ from FQ() -> %s\n", ftot, fbad, fbad ? "FAIL" : "PASS");
    return bad != 0 || fbad != 0;
}
