// y26_ops_top against y26_ops_ref on random planes, every layout. Also the HLS testbench (C simulation and co-simulation).
//   g++ -std=c++14 -I<vitis include> -Ihls_kernel testbench/ops_tb.cpp hls_kernel/ops_engine.cpp -o ops_tb && ./ops_tb
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "ops_engine.h"

int main() {
    static ap_uint<128> Sa[Y26_OPS_DEPTH], Sb[Y26_OPS_DEPTH], Dst[Y26_OPS_DEPTH], Par[Y26_OPS_CMAX];
    std::mt19937 rng(26);
    auto bytes = [&](ap_uint<128>* m, std::vector<uint8_t>& v) {
        for (size_t i = 0; i < v.size(); ++i) m[i / 16](8 * (i % 16) + 7, 8 * (i % 16)) = v[i] = (uint8_t)rng();
    };
    // {C, units, amode, bmode, in place}
    const int cases[][5] = {{7, 20, 0, 0, 0}, {2, 320, 0, 1, 1}, {9, 5, 1, 0, 0}, {9, 5, 0, 2, 0}, {6, 5, 1, 2, 0},
                            {4, 20, 2, 0, 0}, {4, 20, 3, 0, 0}, {3, 80, 4, 0, 0}, {3, 80, 4, 1, 0}, {1, 1, 0, 1, 1},
                            {800, 1, 0, 0, 0}, {11, 20, 3, 1, 1}};
    int bad = 0, n = 0;
    for (const auto& k : cases) {
        const int C = k[0], units = k[1], amode = k[2], bmode = k[3];
        const size_t nd = (size_t)C * units * Y26_OPS_UNIT;
        const size_t na = (size_t)C * y26_ops_src_bytes((size_t)units * Y26_OPS_UNIT, amode);
        const size_t nb = bmode ? (size_t)C * y26_ops_src_bytes((size_t)units * Y26_OPS_UNIT, bmode == 2) : 0;
        if (!y26_ops_ok(units, amode, bmode) || nd > 16u * Y26_OPS_DEPTH || na > 16u * Y26_OPS_DEPTH || nb > 16u * Y26_OPS_DEPTH) {
            std::printf("case %d: does not fit\n", n);
            return 2;
        }
        std::vector<uint8_t> a(na), b(nb), want(nd);
        std::vector<OpsPar> p(C);
        bytes(Sa, a);
        ap_uint<128>* sb = k[4] ? Dst : Sb;            // in place: the planes to add are the destination's
        bytes(sb, b);
        for (int c = 0; c < C; ++c) {
            const int kind = (c + n) % 4;               // identity, near 1, small, anything
            p[c].a = kind == 0 ? 65536u : kind == 1 ? 60000u + rng() % 12000 : kind == 2 ? rng() % 4096 : rng() % Y26_OPS_A_MAX;
            p[c].b = kind == 0 ? 0 : kind == 3 ? (int32_t)rng() : (int32_t)(rng() % (1u << 25)) - (1 << 24);
            p[c].pad[0] = p[c].pad[1] = rng();          // the block must ignore these
            Par[c] = (ap_uint<32>(p[c].pad[1]), ap_uint<32>(p[c].pad[0]), ap_uint<32>((uint32_t)p[c].b), ap_uint<32>(p[c].a));
        }
        y26_ops_ref(a.data(), b.data(), want.data(), p.data(), C, units, amode, bmode);
        y26_ops_top(Sa, sb, Dst, Par, C, units, amode, bmode);
        size_t diff = 0;
        for (size_t i = 0; i < nd; ++i) diff += (uint8_t)Dst[i / 16](8 * (i % 16) + 7, 8 * (i % 16)) != want[i];
        std::printf("case %d: C=%d units=%d amode=%d bmode=%d%s: %zu of %zu codes differ\n", n, C, units, amode, bmode,
                    k[4] ? " in place" : "", diff, nd);
        bad += diff != 0;
        ++n;
    }
    // The arithmetic itself: saturation both ways and the rounding point.
    const OpsPar one = {65536u, 0, {0, 0}}, half = {32768u, 0, {0, 0}}, neg = {65536u, -(300 << 16), {0, 0}};
    bad += y26_ops_code(200, 100, one) != 255 || y26_ops_code(3, 0, half) != 2 || y26_ops_code(1, 0, half) != 1 ||
           y26_ops_code(250, 20, neg) != 0 || y26_ops_code(255, 0, {Y26_OPS_A_MAX - 1, 0, {0, 0}}) != 255;
    std::printf("%s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}
