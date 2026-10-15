// PL ops block: y26_ops_ref (ops_int.h), 16 codes per cycle.
//   read Sa -> layout -> \
//                         calc -> write Dst
//   read Sb -> layout -> /
// A layout stage passes whole planes through, or works on units: 5 port words (80 codes) in, or 6 (four rows of 20
// padded to 24), and 5 out, or 20 with the 2x upsample (80 source codes are four rows of 20 or two rows of 40).
#include "ops_engine.h"
#include <hls_stream.h>

namespace {
typedef ap_uint<128> w_t;
typedef ap_uint<56> par_t;       // B (32) above A (24)

void ops_read(const w_t* M, hls::stream<w_t>& s, int n) {
READ:
    for (int i = 0; i < n; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1600 max=1600
#pragma HLS PIPELINE II=1
        s.write(M[i]);
    }
}

// nsrc source units. p24: the source rows are padded; up: 0, or 1 / 2 = upsample a 20- / 40-wide map.
void ops_layout(hls::stream<w_t>& in, hls::stream<w_t>& out, int nsrc, bool p24, ap_uint<2> up) {
    if (!p24 && up == 0) {
    PASS:
        for (int i = 0; i < nsrc * Y26_OPS_WU; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1600 max=1600
#pragma HLS PIPELINE II=1
            out.write(in.read());
        }
        return;
    }
    const int nin = p24 ? 6 : 5, nout = up != 0 ? 20 : 5;
UNIT:
    for (int u = 0; u < nsrc; ++u) {
#pragma HLS LOOP_TRIPCOUNT min=320 max=320
        ap_uint<768> r = 0;
    GATHER:   // newest word on top: after 6 words r = {w5 .. w0}, after 5 the low word is empty
        for (int i = 0; i < nin; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=6
#pragma HLS PIPELINE II=1
            r = (in.read(), r(767, 128));
        }
        ap_uint<640> unit = r(767, 128);
        if (p24)
            for (int j = 0; j < 80; ++j) {
#pragma HLS UNROLL
                const int s = j / 20 * 24 + j % 20;
                unit(8 * j + 7, 8 * j) = r(8 * s + 7, 8 * s);
            }
        w_t g[20];
#pragma HLS ARRAY_PARTITION variable=g complete
        for (int i = 0; i < 20; ++i) {
#pragma HLS UNROLL
            for (int t = 0; t < 16; ++t) {
#pragma HLS UNROLL
                const int j = i * 16 + t;
                const int s0 = j % 80, s1 = j / 40 / 2 * 20 + j % 40 / 2, s2 = j / 80 / 2 * 40 + j % 80 / 2;
                const ap_uint<8> b0 = unit(8 * s0 + 7, 8 * s0), b1 = unit(8 * s1 + 7, 8 * s1), b2 = unit(8 * s2 + 7, 8 * s2);
                g[i](8 * t + 7, 8 * t) = up == 1 ? b1 : up == 2 ? b2 : b0;
            }
        }
    EMIT:
        for (int i = 0; i < nout; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=20
#pragma HLS PIPELINE II=1
            out.write(g[i]);
        }
    }
}

// wpp port words per destination plane.
void ops_calc(hls::stream<w_t>& sa, hls::stream<w_t>& sb, hls::stream<w_t>& so, const par_t pmem[Y26_OPS_CMAX],
              int C, int wpp, bool hasb) {
    int c = 0, w = 0;
CALC:
    for (int i = 0; i < C * wpp; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1600 max=1600
#pragma HLS PIPELINE II=1
        const par_t p = pmem[c];
        const ap_uint<24> A = p(23, 0);
        const ap_int<32> B = p(55, 24);
        const w_t a = sa.read();
        w_t b = 0;
        if (hasb) b = sb.read();
        w_t o;
        for (int t = 0; t < 16; ++t) {
#pragma HLS UNROLL
            const ap_uint<8> av = a(8 * t + 7, 8 * t), bv = b(8 * t + 7, 8 * t);
            const ap_int<35> m = ap_int<35>(av * A) + B + 32768;
            const ap_int<20> r = ap_int<19>(m >> 16) + bv;
            o(8 * t + 7, 8 * t) = r < 0 ? ap_uint<8>(0) : r > 255 ? ap_uint<8>(255) : ap_uint<8>(r);
        }
        so.write(o);
        if (++w == wpp) { w = 0; ++c; }
    }
}

void ops_write(hls::stream<w_t>& s, w_t* M, int n) {
WRITE:
    for (int i = 0; i < n; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1600 max=1600
#pragma HLS PIPELINE II=1
        M[i] = s.read();
    }
}

void ops_run(const w_t* Sa, const w_t* Sb, w_t* Dst, const par_t pmem[Y26_OPS_CMAX], int C, int wpp,
             int asrc, int aw, bool ap24, ap_uint<2> up, int bsrc, int bw, bool bp24, bool hasb, int dw) {
#pragma HLS DATAFLOW
    hls::stream<w_t> a0("a0"), a1("a1"), b0("b0"), b1("b1"), o("o");
#pragma HLS STREAM variable=a0 depth=64
#pragma HLS STREAM variable=a1 depth=64
#pragma HLS STREAM variable=b0 depth=64
#pragma HLS STREAM variable=b1 depth=64
#pragma HLS STREAM variable=o depth=64
    ops_read(Sa, a0, aw);
    ops_layout(a0, a1, asrc, ap24, up);
    ops_read(Sb, b0, bw);
    ops_layout(b0, b1, bsrc, bp24, 0);
    ops_calc(a1, b1, o, pmem, C, wpp, hasb);
    ops_write(o, Dst, dw);
}
}  // namespace

void y26_ops_top(const ap_uint<128>* Sa, const ap_uint<128>* Sb, ap_uint<128>* Dst, const ap_uint<128>* Par,
                 int C, int units, int amode, int bmode) {
#pragma HLS INTERFACE m_axi port=Sa  offset=slave bundle=gmem_opa depth=Y26_OPS_DEPTH num_read_outstanding=16 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=Par offset=slave bundle=gmem_opa depth=Y26_OPS_CMAX num_read_outstanding=16 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=Sb  offset=slave bundle=gmem_opd depth=Y26_OPS_DEPTH num_read_outstanding=16 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=Dst offset=slave bundle=gmem_opd depth=Y26_OPS_DEPTH num_write_outstanding=16 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=Sa bundle=control
#pragma HLS INTERFACE s_axilite port=Sb bundle=control
#pragma HLS INTERFACE s_axilite port=Dst bundle=control
#pragma HLS INTERFACE s_axilite port=Par bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=units bundle=control
#pragma HLS INTERFACE s_axilite port=amode bundle=control
#pragma HLS INTERFACE s_axilite port=bmode bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static par_t pmem[Y26_OPS_CMAX];
PAR:
    for (int c = 0; c < C; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128
#pragma HLS PIPELINE II=1
        const ap_uint<128> w = Par[c];
        pmem[c] = (w(63, 32), w(23, 0));
    }
    const bool ap24 = amode & 1, bp24 = bmode == 2, hasb = bmode != 0;
    const ap_uint<2> up = amode >> 1;
    const int asrc = C * (up != 0 ? units >> 2 : units), bsrc = hasb ? C * units : 0;
    ops_run(Sa, Sb, Dst, pmem, C, units * Y26_OPS_WU, asrc, asrc * (ap24 ? 6 : 5), ap24, up,
            bsrc, bsrc * (bp24 ? 6 : 5), bp24, hasb, C * units * Y26_OPS_WU);
}
