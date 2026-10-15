// PL attention block v2: y26_attn2_ref (attention_int.h) for nh heads, Y26_ATTN_QB queries in parallel.
//   X: per head 2*KD + VD planes of n unsigned codes (q, k, v), n/16 port words a plane.
//   D: per head VD planes of n codes: the reader conv's input.
//   P: per head Y26_ATTN2_PW words: 9 taps of VD int32 W, VD uint32 m1, VD int64 B (y26_attn2_pack).
#include "attention2_engine.h"
#include "attention_exp_lut.h"

namespace {
typedef ap_int<21> score_t;     // |sum of 32 products of int8| <= 2^19
typedef ap_uint<25> exp_t;      // 2^24 * e, e <= 1
typedef ap_int<26> acc_t;       // |sum of 400 (p <= 255) * int8| < 2^24
typedef ap_int<30> pe_t;        // |sum of 9 (|W| < 2^17) * int8| < 2^28

exp_t attn_exp(ap_uint<21> d, ap_uint<24> m) {
#pragma HLS INLINE
    const ap_uint<45> y = (d * m + ap_uint<45>(1 << 15)) >> 16;
    const ap_uint<33> n = y >> Y26_ATTN_EXP_FB;
    const ap_uint<Y26_ATTN_EXP_FB> f = y;
    const exp_t t = Y26_ATTN_EXP[f];
    return n > 24 ? exp_t(0) : exp_t(t >> n(4, 0));
}

ap_uint<8> attn_prob(exp_t e, ap_uint<31> r) {
#pragma HLS INLINE
    const ap_uint<57> p = (e * r + (ap_uint<57>(1) << 43)) >> 44;
    return p > 255 ? ap_uint<8>(255) : ap_uint<8>(p);
}
}  // namespace

void y26_attn2_top(const ap_uint<128>* X, ap_uint<128>* D, const ap_uint<128>* P, int nh, int n, int w, ap_uint<24> m, ap_uint<26> g) {
#pragma HLS INTERFACE m_axi port=X offset=slave bundle=gmem_attn depth=Y26_ATTN2_DEPTH_X num_read_outstanding=16 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem_attn depth=Y26_ATTN2_DEPTH_D num_write_outstanding=16 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=P offset=slave bundle=gmem_attn depth=Y26_ATTN2_DEPTH_P num_read_outstanding=16 max_read_burst_length=64
#pragma HLS INTERFACE s_axilite port=X bundle=control
#pragma HLS INTERFACE s_axilite port=D bundle=control
#pragma HLS INTERFACE s_axilite port=P bundle=control
#pragma HLS INTERFACE s_axilite port=nh bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port=w bundle=control
#pragma HLS INTERFACE s_axilite port=m bundle=control
#pragma HLS INTERFACE s_axilite port=g bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static ap_uint<256> qmem[Y26_ATTN_NMAX], kmem[Y26_ATTN_NMAX];
    static ap_uint<512> vmem[Y26_ATTN_NMAX];
    static score_t smem[Y26_ATTN_QB][Y26_ATTN_NMAX];
    static exp_t emem[Y26_ATTN_QB][Y26_ATTN_NMAX];
    static ap_int<18> wt[9][Y26_ATTN_VD];          // pe's taps, [tap][channel]
#pragma HLS ARRAY_PARTITION variable=smem dim=1 complete
#pragma HLS ARRAY_PARTITION variable=emem dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=2 complete
    ap_uint<26> m1r[Y26_ATTN_VD];
    ap_int<40> br[Y26_ATTN_VD];
    ap_uint<128> pb[16];                           // 16 parameter words: one value per channel
    ap_uint<128> col[Y26_ATTN2_XC];                // one port word of every input plane: 16 positions
    ap_uint<128> ob[Y26_ATTN_VD];                  // 16 positions of every output plane
#pragma HLS ARRAY_PARTITION variable=m1r complete
#pragma HLS ARRAY_PARTITION variable=br complete
#pragma HLS ARRAY_PARTITION variable=pb complete
#pragma HLS ARRAY_PARTITION variable=col complete
#pragma HLS ARRAY_PARTITION variable=ob complete

    const int nw = n >> 4;                         // port words a plane

HEAD:
    for (int h = 0; h < nh; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
    PLOAD:
        for (int i = 0; i < Y26_ATTN2_PW; ++i) {
#pragma HLS PIPELINE II=1
            const ap_uint<128> pw = P[h * Y26_ATTN2_PW + i];
            const ap_uint<4> k = i;
            const ap_uint<4> r = i >> 4;           // 0..8: a tap, 9: m1, 10 and 11: B
            pb[k] = pw;
            if (k == 15) {
                for (int c = 0; c < Y26_ATTN_VD; ++c) {
#pragma HLS UNROLL
                    const ap_uint<128> s = c / 4 == 15 ? pw : pb[c / 4];
                    const ap_uint<32> f = s((c % 4) * 32 + 31, (c % 4) * 32);
                    if (r < 9) wt[r][c] = ap_int<18>(f(17, 0));
                    if (r == 9) m1r[c] = f(25, 0);
                }
                for (int c = 0; c < Y26_ATTN_VD / 2; ++c) {
#pragma HLS UNROLL
                    const ap_uint<128> s = c / 2 == 15 ? pw : pb[c / 2];
                    const ap_uint<64> f = s((c % 2) * 64 + 63, (c % 2) * 64);
                    if (r == 10) br[c] = ap_int<40>(f(39, 0));
                    if (r == 11) br[Y26_ATTN_VD / 2 + c] = ap_int<40>(f(39, 0));
                }
            }
        }

    LOADW:
        for (int wd = 0; wd < nw; ++wd) {
#pragma HLS LOOP_TRIPCOUNT min=25 max=25
            int xa = h * Y26_ATTN2_XC * nw + wd;
        LOAD:   // one port word of every plane, then its 16 positions
            for (int i = 0; i < Y26_ATTN2_XC + 16; ++i) {
#pragma HLS PIPELINE II=1
                if (i < Y26_ATTN2_XC) {
                    col[i] = X[xa];
                    xa += nw;
                } else {
                    ap_uint<256> q, k;
                    ap_uint<512> v;
                    for (int d = 0; d < Y26_ATTN_KD; ++d) {
#pragma HLS UNROLL
                        q(d * 8 + 7, d * 8) = ap_uint<8>(col[d](7, 0)) ^ ap_uint<8>(0x80);
                        k(d * 8 + 7, d * 8) = ap_uint<8>(col[Y26_ATTN_KD + d](7, 0)) ^ ap_uint<8>(0x80);
                    }
                    for (int d = 0; d < Y26_ATTN_VD; ++d) {
#pragma HLS UNROLL
                        v(d * 8 + 7, d * 8) = ap_uint<8>(col[2 * Y26_ATTN_KD + d](7, 0)) ^ ap_uint<8>(0x80);
                    }
                    for (int c = 0; c < Y26_ATTN2_XC; ++c) {   // the next position's codes come down
#pragma HLS UNROLL
                        col[c] >>= 8;
                    }
                    const int p = wd * 16 + i - Y26_ATTN2_XC;
                    qmem[p] = q;
                    kmem[p] = k;
                    vmem[p] = v;
                }
            }
        }

        int x = 0;                                 // column of the next position written out
    GROUP:
        for (int n0 = 0; n0 < n; n0 += Y26_ATTN_QB) {
#pragma HLS LOOP_TRIPCOUNT min=100 max=100
            ap_uint<256> q[Y26_ATTN_QB];
            score_t mx[Y26_ATTN_QB];
            ap_uint<34> sum[Y26_ATTN_QB];
            ap_uint<31> r[Y26_ATTN_QB];
            acc_t acc[Y26_ATTN_QB][Y26_ATTN_VD];
            pe_t pe[Y26_ATTN_VD];
#pragma HLS ARRAY_PARTITION variable=q complete
#pragma HLS ARRAY_PARTITION variable=mx complete
#pragma HLS ARRAY_PARTITION variable=sum complete
#pragma HLS ARRAY_PARTITION variable=r complete
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
#pragma HLS ARRAY_PARTITION variable=pe complete

        QLOAD:
            for (int j = 0; j < Y26_ATTN_QB; ++j) {
#pragma HLS PIPELINE II=1
                q[j] = qmem[n0 + j];
                mx[j] = score_t(-(1 << 20));
                sum[j] = 0;
            }

        SCORE:
            for (int n2 = 0; n2 < n; ++n2) {
#pragma HLS LOOP_TRIPCOUNT min=400 max=400
#pragma HLS PIPELINE II=1
                const ap_uint<256> kk = kmem[n2];
                for (int j = 0; j < Y26_ATTN_QB; ++j) {
#pragma HLS UNROLL
                    score_t s = 0;
                    for (int kd = 0; kd < Y26_ATTN_KD; ++kd) {
#pragma HLS UNROLL
                        const ap_int<8> a = q[j](kd * 8 + 7, kd * 8), b = kk(kd * 8 + 7, kd * 8);
                        s += a * b;
                    }
                    smem[j][n2] = s;
                    if (s > mx[j]) mx[j] = s;
                }
            }

        EXP:
            for (int n2 = 0; n2 < n; ++n2) {
#pragma HLS LOOP_TRIPCOUNT min=400 max=400
#pragma HLS PIPELINE II=1
                for (int j = 0; j < Y26_ATTN_QB; ++j) {
#pragma HLS UNROLL
                    const exp_t e = attn_exp(ap_uint<21>(mx[j] - smem[j][n2]), m);
                    emem[j][n2] = e;
                    sum[j] += e;
                }
            }

        RECIP:
            for (int j = 0; j < Y26_ATTN_QB; ++j) {
#pragma HLS PIPELINE II=1
                r[j] = (ap_uint<54>(g) << 28) / sum[j];
            }

        VP:
            for (int n2 = 0; n2 < n; ++n2) {
#pragma HLS LOOP_TRIPCOUNT min=400 max=400
#pragma HLS PIPELINE II=1
                const ap_uint<512> vv = vmem[n2];
                for (int j = 0; j < Y26_ATTN_QB; ++j) {
#pragma HLS UNROLL
                    const ap_int<9> p = attn_prob(emem[j][n2], r[j]);
                    for (int vd = 0; vd < Y26_ATTN_VD; ++vd) {
#pragma HLS UNROLL
                        const ap_int<8> v = vv(vd * 8 + 7, vd * 8);
                        const acc_t t = p * v;
                        acc[j][vd] = n2 == 0 ? t : acc_t(acc[j][vd] + t);
                    }
                }
            }

            // pe on the v codes, then the reader's code of each position: nine taps a position, all channels at once
            ap_uint<4> t = 0;
            ap_uint<3> j = 0;
        CODE:
            for (int i = 0; i < Y26_ATTN_QB * 9; ++i) {
#pragma HLS PIPELINE II=1
                const int p = n0 + j;
                const ap_uint<2> kh = t < 3 ? 0 : t < 6 ? 1 : 2;
                const ap_uint<2> kw = t - 3 * kh;
                const bool ok = !(kh == 0 && p < w) && !(kh == 2 && p + w >= n) && !(kw == 0 && x == 0) && !(kw == 2 && x + 1 == w);
                const int a = p + (kh == 0 ? -w : kh == 2 ? w : 0) + int(kw) - 1;
                const ap_uint<512> vv = vmem[ok ? a : 0];
                const bool last = t == 8;
                const ap_uint<2> jj = j;
                for (int c = 0; c < Y26_ATTN_VD; ++c) {
#pragma HLS UNROLL
                    const ap_int<8> v = ok ? ap_int<8>(vv(c * 8 + 7, c * 8)) : ap_int<8>(0);
                    const ap_int<26> pr = wt[t][c] * v;
                    const pe_t s = (t == 0 ? pe_t(0) : pe[c]) + pr;
                    pe[c] = s;
                    if (last) {
                        const acc_t J = acc[jj][c];
                        const ap_int<18> js = J < -(1 << 17) ? ap_int<18>(-(1 << 17)) : J > (1 << 17) - 1 ? ap_int<18>((1 << 17) - 1) : ap_int<18>(J);
                        const ap_int<48> y = js * m1r[c] + (ap_int<48>(s) << 16) + br[c] + ap_int<48>(1 << 27);
                        const ap_int<20> z = y >> 28;
                        const ap_uint<8> code = z < 0 ? ap_uint<8>(0) : z > 255 ? ap_uint<8>(255) : ap_uint<8>(z);
                        ob[c] = (code, ob[c](127, 8));
                    }
                }
                if (last) { t = 0;  ++j;  x = x + 1 == w ? 0 : x + 1; }
                else ++t;
            }

            if ((n0 & 15) == 16 - Y26_ATTN_QB) {
                int da = h * Y26_ATTN_VD * nw + (n0 >> 4);
            FLUSH:
                for (int c = 0; c < Y26_ATTN_VD; ++c) {
#pragma HLS PIPELINE II=1
                    D[da] = ob[c];
                    da += nw;
                }
            }
        }
    }
}
