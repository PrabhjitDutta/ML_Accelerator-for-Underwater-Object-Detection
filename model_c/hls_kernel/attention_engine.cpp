// PL attention block: y26_attn_int_head (attention_int.h) for nh heads, Y26_ATTN_QB queries in parallel.
//   X: per head, per position n, one 128-byte record {q[32], k[32], v[64]} of signed codes (8 words of 128 bits).
//   Y: per head, per position n, J[64] as int32 (16 words of 128 bits).
#include "attention_engine.h"
#include "attention_exp_lut.h"

namespace {
typedef ap_int<21> score_t;     // |sum of 32 products of int8| <= 2^19
typedef ap_uint<25> exp_t;      // 2^24 * e, e <= 1
typedef ap_int<26> acc_t;       // |sum of 400 (p <= 255) * int8| < 2^24

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

void y26_attn_top(const ap_uint<128>* X, ap_uint<128>* Y, int nh, int n, ap_uint<24> m, ap_uint<26> g) {
#pragma HLS INTERFACE m_axi port=X offset=slave bundle=gmem_attn depth=Y26_ATTN_DEPTH_X num_read_outstanding=16 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=Y offset=slave bundle=gmem_attn depth=Y26_ATTN_DEPTH_Y num_write_outstanding=16 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=X bundle=control
#pragma HLS INTERFACE s_axilite port=Y bundle=control
#pragma HLS INTERFACE s_axilite port=nh bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port=m bundle=control
#pragma HLS INTERFACE s_axilite port=g bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static ap_uint<256> qmem[Y26_ATTN_NMAX], kmem[Y26_ATTN_NMAX];
    static ap_uint<512> vmem[Y26_ATTN_NMAX];
    static score_t smem[Y26_ATTN_QB][Y26_ATTN_NMAX];
    static exp_t emem[Y26_ATTN_QB][Y26_ATTN_NMAX];
#pragma HLS ARRAY_PARTITION variable=smem dim=1 complete
#pragma HLS ARRAY_PARTITION variable=emem dim=1 complete

HEAD:
    for (int h = 0; h < nh; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
        const int base = h * n;
        ap_uint<512> sh = 0;
    LOAD:
        for (int i = 0; i < n * 8; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=3200 max=3200
#pragma HLS PIPELINE II=1
            const ap_uint<128> w = X[base * 8 + i];
            const int p = i >> 3;
            const ap_uint<3> k = i;
            sh = (ap_uint<512>(w), sh(511, 128));     // newest word on top: after 4 words sh = {w3, w2, w1, w0}
            if (k == 1) qmem[p] = sh(511, 256);
            if (k == 3) kmem[p] = sh(511, 256);
            if (k == 7) vmem[p] = sh;
        }

    GROUP:
        for (int n0 = 0; n0 < n; n0 += Y26_ATTN_QB) {
#pragma HLS LOOP_TRIPCOUNT min=100 max=100
            ap_uint<256> q[Y26_ATTN_QB];
            score_t mx[Y26_ATTN_QB];
            ap_uint<34> sum[Y26_ATTN_QB];
            ap_uint<31> r[Y26_ATTN_QB];
            acc_t acc[Y26_ATTN_QB][Y26_ATTN_VD];
#pragma HLS ARRAY_PARTITION variable=q complete
#pragma HLS ARRAY_PARTITION variable=mx complete
#pragma HLS ARRAY_PARTITION variable=sum complete
#pragma HLS ARRAY_PARTITION variable=r complete
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

        QLOAD:   // lanes past n read row 0: computed, never written out
            for (int j = 0; j < Y26_ATTN_QB; ++j) {
#pragma HLS PIPELINE II=1
                q[j] = qmem[n0 + j < n ? n0 + j : 0];
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

            const int nb = n - n0 < Y26_ATTN_QB ? n - n0 : Y26_ATTN_QB;
        OUT:
            for (int i = 0; i < nb * (Y26_ATTN_VD / 4); ++i) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
#pragma HLS PIPELINE II=1
                const int j = i / (Y26_ATTN_VD / 4), w = i % (Y26_ATTN_VD / 4);
                ap_uint<128> o;
                for (int t = 0; t < 4; ++t) {
#pragma HLS UNROLL
                    o(t * 32 + 31, t * 32) = ap_int<32>(acc[j][w * 4 + t]);
                }
                Y[(base + n0) * (Y26_ATTN_VD / 4) + i] = o;
            }
        }
    }
}
