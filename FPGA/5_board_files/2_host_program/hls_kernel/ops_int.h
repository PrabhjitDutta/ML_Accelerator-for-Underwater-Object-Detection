// Byte ops on code planes: the arithmetic the PL ops block implements, as plain C++.
//   d = clamp(floor(a * A + B + 1/2) + b, 0, 255)       per channel: A >= 0 and B, 16 fractional bits each
// a is read through a layout, so one op also copies rows of 20 codes out of rows padded to 24 and upsamples 2x:
//   mode bit 0     : the source rows are 20 codes padded to 24 (the kernel's code rows of a 20-wide map)
//   mode bits 2..1 : 2x nearest upsample of a 20-wide (1) or a 40-wide (2) map
// b (optional) is added as it is; it may be the destination itself.
#pragma once
#include <cstddef>
#include <cstdint>

#define Y26_OPS_UNIT 80          // bytes; every plane is a whole number of units (4 with an upsample)
#define Y26_OPS_A_MAX (1u << 24) // A < 256

struct OpsPar { uint32_t a; int32_t b; uint32_t pad[2]; };   // one port word per channel

inline uint8_t y26_ops_code(uint8_t a, uint8_t b, const OpsPar& p) {
    const int64_t r = (((int64_t)a * p.a + p.b + 32768) >> 16) + b;
    return (uint8_t)(r < 0 ? 0 : r > 255 ? 255 : r);
}

// The source byte of destination byte j of a plane.
inline size_t y26_ops_src(size_t j, int mode) {
    const int up = mode >> 1;
    if (!up) return (mode & 1) ? j / 20 * 24 + j % 20 : j;
    const size_t w = up == 1 ? 40 : 80, y = j / w / 2, x = j % w / 2;
    return (mode & 1) ? y * 24 + x : y * (w / 2) + x;
}
// Bytes of a source plane whose destination plane has n bytes.
inline size_t y26_ops_src_bytes(size_t n, int mode) {
    if (mode >> 1) n /= 4;
    return (mode & 1) ? n / 20 * 24 : n;
}
// amode 5 does not exist: only a 20-wide map has padded rows.
inline bool y26_ops_ok(int units, int amode, int bmode) {
    return units > 0 && amode >= 0 && amode < 5 && bmode >= 0 && bmode < 3 && (!(amode >> 1) || units % 4 == 0);
}

// a: [C][source plane]; b: [C][plane] (bmode 1) or [C][rows of 24] (bmode 2), unused for bmode 0; d: [C][units * 80].
inline void y26_ops_ref(const uint8_t* a, const uint8_t* b, uint8_t* d, const OpsPar* p, int C, int units, int amode, int bmode) {
    const size_t n = (size_t)units * Y26_OPS_UNIT, na = y26_ops_src_bytes(n, amode), nb = y26_ops_src_bytes(n, bmode == 2);
    for (int c = 0; c < C; ++c)
        for (size_t j = 0; j < n; ++j)
            d[c * n + j] = y26_ops_code(a[c * na + y26_ops_src(j, amode)], bmode ? b[c * nb + y26_ops_src(j, bmode == 2)] : 0, p[c]);
}
