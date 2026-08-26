// yolo26_utils.h - FP32 primitives for the YOLO26s trunk C-simulation.
//
// Adapted from htdet's fpga_utils.h conv/activation kernels, generalized to YOLO26's needs
// (arbitrary stride/pad/groups incl. depthwise, k5 maxpool for SPPF, nearest 2x upsample, and the
// C2PSA/PSABlock attention primitive). BN is pre-folded at export time, so conv applies
// out = act(conv(x) * scale + bias) with scale/bias read per output channel.
#pragma once
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include "hls_synth.h"
#include "yolo26_types.h"
#include "weights_loader.h"

// SiLU in double then rounded to float -- closest portable match to torch's float32 SiLU (the
// residual ULP gap vs torch's vectorized exp is the main non-integer term left in the INT8 gate).
static inline float silu(float v) { double d = v; return (float)(d / (1.0 + std::exp(-d))); }

// Symmetric signed-INT8 quantize to an integer code (as float): clamp(round(v/s), -127, 127).
// nearbyint uses round-half-to-even (FE_TONEAREST), matching torch.round in the Python oracle.
static inline float q_i8(float v, float s) {
    float q = std::nearbyint(v / s);
    if (q > 127.f) q = 127.f; else if (q < -127.f) q = -127.f;
    return q;
}

// SmoothQuant asymmetric-uint8 quantize: clamp(round((v*ssc - lo)/step), 0, 255).
// v*ssc applies the per-input-channel smooth pre-scale; (·-lo)/step is the per-tensor FakeQuantize.
static inline float q_u8(float v, float ssc, float lo, float step) {
    float q = std::nearbyint((v * ssc - lo) / step);
    if (q > 255.f) q = 255.f; else if (q < 0.f) q = 0.f;
    return q;
}

// Output-column range [o0,o1) over which tap column `base = kw - pad_w` lands inside the image,
// i.e. 0 <= ow*sw + base <= W-1. Computing this once per tap replaces the per-element bounds test:
// the zero padding is expressed as a shorter loop range rather than as a branch taken OW times.
static inline void ow_range(int base, int sw, int W, int OW, int& o0, int& o1) {
    o0 = base >= 0 ? 0 : (-base + sw - 1) / sw;
    o1 = W - 1 - base;
    o1 = o1 < 0 ? 0 : o1 / sw + 1;
    if (o1 > OW) o1 = OW;
}

// Generic 2D convolution + folded BN/scale + activation. Handles groups (g==C => depthwise).
// INT8 mode (c.quant): the exact integer W8A8 dataflow -- the input is quantized to int8 codes
// (c.sa), c.w already holds the per-channel int8 weight codes, and each output accumulates the
// integer products in an exact int32 (order-independent) before dequant+bias:
//     real[o] = acc_int[o] * (c.sa * c.wsc[o]) + bias[o]     then activate.
// This integer accumulation is what makes the C-sim bit-stable against the PyTorch oracle (float
// conv reduction-order noise, which the hard rounding would otherwise amplify, cannot arise).
inline Tensor conv2d(const Tensor& x, const ConvW& c) {
    CSIM_HOST_ASSERT(x.C == c.ic, "conv " + c.name + ": input has C=" + std::to_string(x.C) +
                                  " but manifest ic=" + std::to_string(c.ic));
    const int OH = (x.H + 2 * c.ph - c.kh) / c.sh + 1;
    const int OW = (x.W + 2 * c.pw - c.kw) / c.sw + 1;
    Tensor xq;                                  // quantized activation codes (INT8/SmoothQuant only)
    if (c.quant) {
        xq = Tensor(x.C, x.H, x.W);
        if (c.asym) {                           // SmoothQuant: asymmetric uint8 + per-IC smooth scale
            const size_t HW = (size_t)x.H * x.W;
            CSIM_OMP_STATIC
            for (int ic = 0; ic < x.C; ++ic) {
                const float s = c.ssc[ic];
                const float lo = c.lo_of(ic), st = c.step_of(ic);   // per-tensor, or per-IC if perch
                const float* xp = &x.d[(size_t)ic * HW];
                float* qp = &xq.d[(size_t)ic * HW];
                for (size_t j = 0; j < HW; ++j) qp[j] = q_u8(xp[j], s, lo, st);
            }
        } else {                                // Phase-2: symmetric signed int8, per-tensor sa
            CSIM_OMP_STATIC
            for (int i = 0; i < (int)x.d.size(); ++i) xq.d[i] = q_i8(x.d[i], c.sa);
        }
    }
    const Tensor& X = c.quant ? xq : x;         // read activations through X below
    Tensor y(c.oc, OH, OW);
    const int icpg = c.ic / c.groups;      // input channels per group
    const int ocpg = c.oc / c.groups;      // output channels per group
    const int ktap = c.kh * c.kw;
    // SmoothQuant zero-point: accw is Sum(w_int) over the IN-IMAGE taps. The icl axis contributes the
    // same weights at every output position, so pre-sum it once per conv here; the per-row pass below
    // then costs kh*kw range-adds instead of icpg*kh*kw (one add per tap). Bit-safe to reorder: these
    // are int8 CODES, so the sums are exact integers in a double (|sum| <= 127*4608 << 2^53).
    std::vector<double> kwsum;
    if (c.asym) {
        kwsum.assign((size_t)c.oc * ktap, 0.0);
        for (int o = 0; o < c.oc; ++o)
            for (int icl = 0; icl < icpg; ++icl) {
                const float* wp = &c.w[((size_t)o * icpg + icl) * ktap];
                for (int t = 0; t < ktap; ++t) kwsum[(size_t)o * ktap + t] += (double)wp[t];
            }
    }
    // Loop order: the reduction axes (icl, kh, kw) are OUTSIDE and the output column `ow` is innermost,
    // accumulating into a per-row buffer. The obvious order (ow outermost, icl innermost) reads
    // X(ic, ih, iw) with a stride of H*W floats, so a 1x1 conv touches a fresh cache line on EVERY MAC
    // -- measured 0.31 GMAC/s vs 0.76 for 3x3, and 65% of total C-sim runtime sat in the 1x1 nest.
    // Walking `ow` innermost makes the input read contiguous (unit stride for sw=1) and vectorizable.
    // Bit-exact: for a fixed output element the additions still occur in icl -> kh -> kw order, exactly
    // as before, and (double)float * (double)float is exact so FMA contraction cannot change it.
    // This is also the right shape for synthesis: the innermost loop writes distinct accrow[ow], so it
    // carries no dependency and admits #pragma HLS PIPELINE II=1, whereas accumulating into one scalar
    // `acc` bounds II by the adder latency.
    CSIM_OMP_STATIC
    for (int oc = 0; oc < c.oc; ++oc) {
        const int g   = oc / ocpg;
        const int ic0 = g * icpg;
        // dequant/BN scale for this output channel: INT8 -> sa*sw[oc]; FP32 -> folded BN scale (or 1)
        const float sc = c.quant ? (c.sa * c.wsc[oc]) : (c.has_bn ? c.s[oc] : 1.f);
        const float bs = c.b.empty() ? 0.f : c.b[oc];
        // perch is depthwise-only (loader-enforced), so this conv's single accumulated input channel
        // is ic0 == oc and one step/lo pair governs the whole dequant for this output channel.
        const float step_o = c.step_of(ic0);
        const float lo_o   = c.lo_of(ic0);
        std::vector<double> accrow(OW), accwrow(c.asym ? (size_t)OW : 0);
        for (int oh = 0; oh < OH; ++oh) {
            const int ih0 = oh * c.sh - c.ph;
            std::fill(accrow.begin(), accrow.end(), 0.0);
            if (c.asym) {
                std::fill(accwrow.begin(), accwrow.end(), 0.0);
                for (int kh = 0; kh < c.kh; ++kh) {          // zero-point: once per tap, not per MAC
                    const int ih = ih0 + kh;
                    if (ih < 0 || ih >= X.H) continue;        // whole row is padding: contributes none
                    for (int kw = 0; kw < c.kw; ++kw) {
                        const double ws = kwsum[(size_t)oc * ktap + kh * c.kw + kw];
                        int o0, o1;
                        ow_range(kw - c.pw, c.sw, X.W, OW, o0, o1);
                        for (int ow = o0; ow < o1; ++ow) accwrow[ow] += ws;
                    }
                }
            }
            for (int icl = 0; icl < icpg; ++icl) {
                const int ic = ic0 + icl;
                const float* wp = &c.w[((size_t)oc * icpg + icl) * ktap];
                for (int kh = 0; kh < c.kh; ++kh) {
                    const int ih = ih0 + kh;
                    if (ih < 0 || ih >= X.H) continue;
                    const float* xr = &X.d[((size_t)ic * X.H + ih) * X.W];
                    for (int kw = 0; kw < c.kw; ++kw) {
                        const double w = (double)wp[kh * c.kw + kw];
                        const int base = kw - c.pw;
                        int o0, o1;
                        ow_range(base, c.sw, X.W, OW, o0, o1);
                        for (int ow = o0; ow < o1; ++ow)
                            accrow[ow] += (double)xr[ow * c.sw + base] * w;
                    }
                }
            }
            // SmoothQuant asymmetric: real = sw*(step*Sum(w_int*q) + lo*Sum_valid(w_int)) + bias.
            // The valid-tap weight sum makes padded borders contribute real 0.0 (not `lo`), matching
            // OV exactly; folding lo into a constant bias errs at borders (and propagates).
            float* yr = &y.d[((size_t)oc * OH + oh) * OW];
            for (int ow = 0; ow < OW; ++ow) {
                float v = c.asym ? (float)(c.wsc[oc] * (step_o * accrow[ow] + lo_o * accwrow[ow])) + bs
                                 : (float)accrow[ow] * sc + bs;
                if (c.act == ACT_SILU) v = silu(v);
                yr[ow] = v;
            }
        }
    }
    return y;
}

// MaxPool2d, square kernel k, stride s, pad p (pad value -inf). Used by SPPF (k5 s1 p2).
inline Tensor maxpool(const Tensor& x, int k, int s, int p) {
    const int OH = (x.H + 2 * p - k) / s + 1;
    const int OW = (x.W + 2 * p - k) / s + 1;
    Tensor y(x.C, OH, OW);
    CSIM_OMP_STATIC
    for (int c = 0; c < x.C; ++c)
        for (int oh = 0; oh < OH; ++oh)
            for (int ow = 0; ow < OW; ++ow) {
                float m = -1e30f;
                for (int kh = 0; kh < k; ++kh) {
                    const int ih = oh * s - p + kh;
                    if (ih < 0 || ih >= x.H) continue;
                    for (int kw = 0; kw < k; ++kw) {
                        const int iw = ow * s - p + kw;
                        if (iw < 0 || iw >= x.W) continue;
                        const float v = x(c, ih, iw);
                        if (v > m) m = v;
                    }
                }
                y(c, oh, ow) = m;
            }
    return y;
}

// Nearest-neighbour 2x upsample (matches nn.Upsample(scale_factor=2, mode='nearest')).
inline Tensor upsample2x(const Tensor& x) {
    Tensor y(x.C, x.H * 2, x.W * 2);
    for (int c = 0; c < x.C; ++c)
        for (int h = 0; h < x.H; ++h)
            for (int w = 0; w < x.W; ++w) {
                const float v = x(c, h, w);
                y(c, 2 * h, 2 * w) = v;   y(c, 2 * h, 2 * w + 1) = v;
                y(c, 2 * h + 1, 2 * w) = v; y(c, 2 * h + 1, 2 * w + 1) = v;
            }
    return y;
}

// Channel-dim concatenation (all inputs share H,W).
inline Tensor concat(const std::vector<const Tensor*>& xs) {
    int C = 0; const int H = xs[0]->H, W = xs[0]->W;
    for (auto* t : xs) {
        CSIM_HOST_ASSERT(t->H == H && t->W == W, "concat: spatial mismatch, got " +
                         std::to_string(t->H) + "x" + std::to_string(t->W) + " vs " +
                         std::to_string(H) + "x" + std::to_string(W));
        C += t->C;
    }
    Tensor y(C, H, W);
    int off = 0;
    for (auto* t : xs) {
        std::copy(t->d.begin(), t->d.end(), y.d.begin() + (size_t)off * H * W);
        off += t->C;
    }
    return y;
}

// Elementwise add (residual). a,b same shape -- checked, because the loop is bounded by a's size and
// a narrower b would be read past its end (silently wrong output, exit 0), the same failure mode the
// SPPF residual and the truncated-weight-file checks already guard against.
inline Tensor add(const Tensor& a, const Tensor& b) {
    CSIM_HOST_ASSERT(a.C == b.C && a.H == b.H && a.W == b.W,
                     "add: shape mismatch " + std::to_string(a.C) + "x" + std::to_string(a.H) + "x" +
                     std::to_string(a.W) + " vs " + std::to_string(b.C) + "x" + std::to_string(b.H) +
                     "x" + std::to_string(b.W));
    Tensor y(a.C, a.H, a.W);
    for (size_t i = 0; i < a.d.size(); ++i) y.d[i] = a.d[i] + b.d[i];
    return y;
}

// Channel slice [c0, c0+n) -> new tensor.
inline Tensor slice_ch(const Tensor& x, int c0, int n) {
    CSIM_HOST_ASSERT(c0 >= 0 && n >= 0 && c0 + n <= x.C, "slice_ch: [" + std::to_string(c0) + "," +
                     std::to_string(c0 + n) + ") out of range for C=" + std::to_string(x.C));
    Tensor y(n, x.H, x.W);
    std::copy(x.d.begin() + (size_t)c0 * x.H * x.W,
              x.d.begin() + (size_t)(c0 + n) * x.H * x.W, y.d.begin());
    return y;
}

// C2PSA/PSABlock Attention (ultralytics block.Attention). dim = x.C.
//   num_heads = max(dim/64, 1); head_dim = dim/num_heads; key_dim = head_dim/2; scale = 1/sqrt(key_dim)
//   qkv(x) -> view[nh, 2*key_dim+head_dim, N] split q,k,v
//   attn = softmax_{n2}( (q*scale)^T @ k );  out = (v @ attn^T) + pe(v);  return proj(out)
// `fq` carries the five interior quantizers OV applies (q*scale, k, softmax, v->matmul, v->pe).
// A default-constructed AttnFQ is all-off and every FQ is the identity, which reproduces the Phase-3
// FP32-interior behaviour exactly -- so weight dirs without an attn_fq.txt still run unchanged.
// The q@k product and the softmax itself stay FP32 on purpose: OV puts no FakeQuantize on the MatMul
// output and its SoftMax is a float op, so a fully-integer attention would NOT match the reference.
inline Tensor attention(const Tensor& x, const ConvW& qkv, const ConvW& proj, const ConvW& pe,
                        const AttnFQ& fq = AttnFQ()) {
    const int C = x.C, H = x.H, W = x.W, N = H * W;
    const int num_heads = std::max(C / 64, 1);
    const int head_dim  = C / num_heads;
    const int key_dim   = head_dim / 2;            // attn_ratio 0.5
    const float scale   = 1.f / std::sqrt((float)key_dim);
    const int per_head  = 2 * key_dim + head_dim;  // channels per head in qkv output
    // num_heads/head_dim/key_dim are derived from the INPUT width, but every q/k/v index below assumes
    // qkv's OUTPUT is exactly nh*per_head wide. That holds because compaction protects the attention
    // convs (verified: ic=256 -> oc=512 in both the dense and the compacted+folded manifests), but it
    // is load-bearing and silent if broken -- a compacted qkv would mis-slice every head.
    CSIM_HOST_ASSERT(qkv.oc == num_heads * per_head, "attention: qkv oc=" + std::to_string(qkv.oc) +
                     " but num_heads*per_head=" + std::to_string(num_heads * per_head));

    Tensor q = conv2d(x, qkv);                     // [nh*per_head, H, W]
    // Pre-quantize the matmul operands. OV scales q BEFORE quantizing it (the FakeQuantize sits on
    // the output of the *scale Multiply), so the scale must be folded in here rather than applied to
    // the q@k product afterwards -- same math, but a different quantization boundary.
    std::vector<float> qs((size_t)num_heads * key_dim * N), ks(qs.size()),
                       vm((size_t)num_heads * head_dim * N);
    for (int hh = 0; hh < num_heads; ++hh)
        for (int d = 0; d < key_dim; ++d) {
            const float* sq = &q.d[(size_t)(hh * per_head + d) * N];
            const float* sk = &q.d[(size_t)(hh * per_head + key_dim + d) * N];
            float* dq = &qs[(size_t)(hh * key_dim + d) * N];
            float* dk = &ks[(size_t)(hh * key_dim + d) * N];
            for (int n = 0; n < N; ++n) { dq[n] = fq.q(sq[n] * scale); dk[n] = fq.k(sk[n]); }
        }
    // v reshaped back to [C,H,W] for pe(): channel = head*head_dim + vd. v is quantized TWICE with
    // separately calibrated ranges -- v_mm for the v@attn^T matmul, v_pe for pe(v).
    Tensor vt(C, H, W);
    for (int hh = 0; hh < num_heads; ++hh)
        for (int vd = 0; vd < head_dim; ++vd) {
            const float* sv = &q.d[(size_t)(hh * per_head + 2 * key_dim + vd) * N];
            float* dpe = &vt.d[(size_t)(hh * head_dim + vd) * N];
            float* dmm = &vm[(size_t)(hh * head_dim + vd) * N];
            for (int n = 0; n < N; ++n) { dpe[n] = fq.v_pe(sv[n]); dmm[n] = fq.v_mm(sv[n]); }
        }
    Tensor pe_out = conv2d(vt, pe);                // dw3x3 positional encoding on v

    Tensor out(C, H, W);                            // v @ attn^T, laid out as [C,H,W]
    CSIM_OMP_DYNAMIC
    for (int hh = 0; hh < num_heads; ++hh) {
        const float* qh = &qs[(size_t)(hh * key_dim) * N];
        const float* kh = &ks[(size_t)(hh * key_dim) * N];
        const float* vh = &vm[(size_t)(hh * head_dim) * N];
        std::vector<float> attn((size_t)N);                              // one query row at a time
        for (int n1 = 0; n1 < N; ++n1) {
            float mx = -1e30f;
            for (int n2 = 0; n2 < N; ++n2) {
                float s = 0.f;
                for (int kd = 0; kd < key_dim; ++kd)
                    s += qh[(size_t)kd * N + n1] * kh[(size_t)kd * N + n2];
                attn[n2] = s;
                if (s > mx) mx = s;
            }
            float sum = 0.f;
            for (int n2 = 0; n2 < N; ++n2) { attn[n2] = std::exp(attn[n2] - mx); sum += attn[n2]; }
            const float inv = 1.f / sum;
            for (int n2 = 0; n2 < N; ++n2) attn[n2] = fq.sm(attn[n2] * inv);   // quantized probs
            for (int vd = 0; vd < head_dim; ++vd) {
                float acc = 0.f;
                for (int n2 = 0; n2 < N; ++n2) acc += vh[(size_t)vd * N + n2] * attn[n2];
                out.d[(size_t)(hh * head_dim + vd) * N + n1] = acc;
            }
        }
    }
    Tensor xattn = add(out, pe_out);
    return conv2d(xattn, proj);
}
