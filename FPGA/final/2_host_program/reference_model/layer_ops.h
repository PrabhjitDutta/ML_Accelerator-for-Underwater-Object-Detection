// FP32/INT8 primitives for the YOLO26s reference model. BN is folded at export: out = act(conv(x)*scale + bias).
#pragma once
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <cstdint>
#include <memory>
#include "synthesis_guards.h"
#ifdef Y26_BOARD
#include <chrono>
#include <map>
#include <mutex>
#endif
#include "tensor_types.h"
#include "weights_loader.h"

// SiLU in double, rounded to float: the closest portable match to torch's float32 SiLU.
static inline float silu(float v) { double d = v; return (float)(d / (1.0 + std::exp(-d))); }

// Symmetric INT8 code: clamp(round(v/s), -127, 127). nearbyint rounds half-to-even, like torch.round.
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

// q_u8 over n values without a divide, bit-identical to q_u8: x*(1/step) rounds like x/step except near a .5
// tie, and such a 64-block is redone with the divide. Written so GCC vectorizes it (NEON).
// Checked by board_host/test_quantizer.cpp.
__attribute__((optimize("vect-cost-model=dynamic")))
inline void quant_plane(uint8_t* __restrict xo, const float* __restrict xp, size_t n, float s, float lo, float st) {
    const float inv = 1.f / st;
    size_t j = 0;
    for (; j + 64 <= n; j += 64) {
        int tie = 0;
        for (int k = 0; k < 64; ++k) {
            const float q = (xp[j + k] * s - lo) * inv;
            float r = std::nearbyint(q);
            tie |= std::fabs(q - r) > 0.5f - 0.000244140625f;
            r = r > 255.f ? 255.f : (r < 0.f ? 0.f : r);
            xo[j + k] = (uint8_t)(int)r;
        }
        if (tie) for (int k = 0; k < 64; ++k) xo[j + k] = (uint8_t)(int)q_u8(xp[j + k], s, lo, st);
    }
    for (; j < n; ++j) xo[j] = (uint8_t)(int)q_u8(xp[j], s, lo, st);
}

// Output-column range [o0,o1) where tap column base = kw - pad_w is inside the image: padding becomes a shorter loop.
static inline void ow_range(int base, int sw, int W, int OW, int& o0, int& o1) {
    o0 = base >= 0 ? 0 : (-base + sw - 1) / sw;
    o1 = W - 1 - base;
    o1 = o1 < 0 ? 0 : o1 / sw + 1;
    if (o1 > OW) o1 = OW;
}

// One source of a conv's channel-concatenated input. PLAIN is a tensor. ADD (a+b), UP2X (nearest 2x of a) and
// POOL (k chained maxpool(5,1,2) of a) are read only by that conv: the board fuses them into its quantizer, the
// CPU path materializes them (y26_materialize). Exact: q is elementwise and monotone for ssc, step > 0.
struct Src {
    enum Op { PLAIN, ADD, UP2X, POOL };
    Op op; const Tensor* a; const Tensor* b; int k;
    Src(const Tensor* t) : op(PLAIN), a(t), b(nullptr), k(0) {}
    static Src sum(const Tensor* x, const Tensor* y) { Src s(x); s.op = ADD; s.b = y; return s; }
    static Src up2x(const Tensor* x) { Src s(x); s.op = UP2X; return s; }
    static Src pool(const Tensor* x, int n) { Src s(x); s.op = POOL; s.k = n; return s; }
    int C() const { return a->C; }
    int H() const { return op == UP2X ? 2 * a->H : a->H; }
    int W() const { return op == UP2X ? 2 * a->W : a->W; }
};
// Board: one PLAIN source's X codes, packed ahead of its conv (on its own thread when async). Null off the board.
struct Y26Pre;
using Pre = std::shared_ptr<Y26Pre>;
#ifdef Y26_BOARD
// board_host.cpp: runs the conv on the kernel. xs = the channel-concatenated input sources; pre = one already packed.
Tensor y26_board_conv(const std::vector<Src>& xs, const ConvW& c, const Pre& pre = nullptr);
// t = c's FIRST (last = false) or LAST input source. t's storage must outlive the conv; t itself may be moved.
Pre y26_board_prepack(const Tensor* t, const ConvW& c, bool last);
// The frame loop's input as uint8 k (float k/255.f): 0.conv packs it through a 256-entry table. u8 = null forgets d.
void y26_input_u8(const float* d, std::shared_ptr<const std::vector<uint8_t>> u8);
// Board-only PS profile: self time per op (nested timed scopes excluded), per thread.
struct Y26Prof {
    struct E { double ms = 0; int n = 0; };
    static std::map<std::string, E>& tab() { static auto* t = new std::map<std::string, E>; return *t; }  // leaked: read at exit
    static std::mutex& mu() { static auto* m = new std::mutex; return *m; }
    static double& inner() { thread_local double d = 0; return d; }
    const char* name; double saved; std::chrono::steady_clock::time_point t0;
    explicit Y26Prof(const char* n) : name(n), saved(inner()), t0(std::chrono::steady_clock::now()) { inner() = 0; }
    ~Y26Prof() {
        const double e = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        { std::lock_guard<std::mutex> g(mu()); E& r = tab()[name]; r.ms += e - inner(); ++r.n; }
        inner() = saved + e;
    }
};
#define Y26_PROF(n) Y26Prof y26_prof_(n)
bool y26_async();   // board_host.cpp: run independent trunk branches on their own threads (Y26_ASYNC=0: off)
void y26_frame_end(int frame);   // board_host.cpp: end of frame, print its timing split
void y26_board_weights(const Weights& W);   // board_host.cpp: consumer params for -DY26_YQ8
#else
inline Pre y26_board_prepack(const Tensor*, const ConvW&, bool) { return nullptr; }
inline void y26_input_u8(const float*, std::shared_ptr<const std::vector<uint8_t>>) {}
inline void y26_board_weights(const Weights&) {}
#define Y26_PROF(n)
inline bool y26_async() { return false; }
inline void y26_frame_end(int) {}
#endif

// 2D conv + folded BN/scale + activation; groups == C is depthwise. INT8 mode quantizes the input and accumulates
// the integer products exactly (order-independent), then real = acc * (sa * wsc[o]) + bias[o].
inline Tensor conv2d(const Tensor& x, const ConvW& c) {
#ifdef Y26_BOARD
    if (c.quant && c.asym) return y26_board_conv({&x}, c);   // same eligibility as the kernel testbench
#endif
    Y26_PROF("cpu conv (pe dw, float)");
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
        } else {                                // symmetric signed int8, per-tensor sa
            CSIM_OMP_STATIC
            for (int i = 0; i < (int)x.d.size(); ++i) xq.d[i] = q_i8(x.d[i], c.sa);
        }
    }
    const Tensor& X = c.quant ? xq : x;
    Tensor y(c.oc, OH, OW);
    const int icpg = c.ic / c.groups;      // input channels per group
    const int ocpg = c.oc / c.groups;      // output channels per group
    const int ktap = c.kh * c.kw;
    // SmoothQuant zero-point needs Sum(w_int) over in-image taps. Pre-sum the icl axis once per conv so the row pass
    // costs one add per tap. Exact: int8 codes sum exactly in a double.
    std::vector<double> kwsum;
    if (c.asym) {
        kwsum.assign((size_t)c.oc * ktap, 0.0);
        for (int o = 0; o < c.oc; ++o)
            for (int icl = 0; icl < icpg; ++icl) {
                const float* wp = &c.w[((size_t)o * icpg + icl) * ktap];
                for (int t = 0; t < ktap; ++t) kwsum[(size_t)o * ktap + t] += (double)wp[t];
            }
    }
    // Reduction axes outside, output column ow innermost: unit-stride, vectorizable input reads.
    // Bit-exact: each output still sums in icl -> kh -> kw order, and float*float is exact in double.
    CSIM_OMP_STATIC
    for (int oc = 0; oc < c.oc; ++oc) {
        const int g   = oc / ocpg;
        const int ic0 = g * icpg;
        // dequant/BN scale for this output channel: INT8 -> sa*sw[oc]; FP32 -> folded BN scale (or 1)
        const float sc = c.quant ? (c.sa * c.wsc[oc]) : (c.has_bn ? c.s[oc] : 1.f);
        const float bs = c.b.empty() ? 0.f : c.b[oc];
        // perch is depthwise-only, so one step/lo pair (input channel ic0 == oc) governs this output's dequant.
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
            // SmoothQuant: real = sw*(step*Sum(w_int*q) + lo*Sum_valid(w_int)) + bias.
            // Summing only valid taps makes padding contribute 0.0, not lo.
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
    Y26_PROF("maxpool");
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
    Y26_PROF("upsample");
    Tensor y(x.C, x.H * 2, x.W * 2, NoInit());
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
    Y26_PROF("concat");
    int C = 0; const int H = xs[0]->H, W = xs[0]->W;
    for (auto* t : xs) {
        CSIM_HOST_ASSERT(t->H == H && t->W == W, "concat: spatial mismatch, got " +
                         std::to_string(t->H) + "x" + std::to_string(t->W) + " vs " +
                         std::to_string(H) + "x" + std::to_string(W));
        C += t->C;
    }
    Tensor y(C, H, W, NoInit());
    int off = 0;
    for (auto* t : xs) {
        std::copy(t->d.begin(), t->d.end(), y.d.begin() + (size_t)off * H * W);
        off += t->C;
    }
    return y;
}

inline Tensor add(const Tensor& a, const Tensor& b);
// A fused source as the tensor it stands for (the CPU path, and the board's Y26_FUSE_CHECK).
inline Tensor y26_materialize(const Src& s) {
    if (s.op == Src::ADD) return add(*s.a, *s.b);
    if (s.op == Src::UP2X) return upsample2x(*s.a);
    CSIM_HOST_ASSERT(s.op == Src::POOL && s.k >= 1, "y26_materialize: not a fused source");
    Tensor t = maxpool(*s.a, 5, 1, 2);
    for (int i = 1; i < s.k; ++i) t = maxpool(t, 5, 1, 2);   // ponytail: SPPF's 3 sources redo 1+2+3 pools on the CPU path
    return t;
}

// conv2d over the channel concat of xs. The board packs X straight from the sources (X is channel-major, so
// the bytes equal packing the concat) and fuses ADD/UP2X/POOL into the quantizer; the CPU path materializes.
inline Tensor conv2d_cat(const std::vector<Src>& xs, const ConvW& c, const Pre& pre = nullptr) {
#ifdef Y26_BOARD
    if (c.quant && c.asym) return y26_board_conv(xs, c, pre);
#endif
    std::vector<Tensor> m;
    m.reserve(xs.size());                          // no reallocation: p keeps pointers into m
    std::vector<const Tensor*> p;
    for (const Src& s : xs) {
        if (s.op == Src::PLAIN) { p.push_back(s.a); continue; }
        m.push_back(y26_materialize(s));
        p.push_back(&m.back());
    }
    return p.size() == 1 ? conv2d(*p[0], c) : conv2d(concat(p), c);
}

// Elementwise add (residual). Shapes checked: a narrower b would be read past its end.
inline Tensor add(const Tensor& a, const Tensor& b) {
    Y26_PROF("add");
    CSIM_HOST_ASSERT(a.C == b.C && a.H == b.H && a.W == b.W,
                     "add: shape mismatch " + std::to_string(a.C) + "x" + std::to_string(a.H) + "x" +
                     std::to_string(a.W) + " vs " + std::to_string(b.C) + "x" + std::to_string(b.H) +
                     "x" + std::to_string(b.W));
    Tensor y(a.C, a.H, a.W, NoInit());
    for (size_t i = 0; i < a.d.size(); ++i) y.d[i] = a.d[i] + b.d[i];
    return y;
}

// Channel slice [c0, c0+n) -> new tensor.
inline Tensor slice_ch(const Tensor& x, int c0, int n) {
    Y26_PROF("slice");
    CSIM_HOST_ASSERT(c0 >= 0 && n >= 0 && c0 + n <= x.C, "slice_ch: [" + std::to_string(c0) + "," +
                     std::to_string(c0 + n) + ") out of range for C=" + std::to_string(x.C));
    Tensor y(n, x.H, x.W, NoInit());
    std::copy(x.d.begin() + (size_t)c0 * x.H * x.W,
              x.d.begin() + (size_t)(c0 + n) * x.H * x.W, y.d.begin());
    return y;
}

// C2PSA/PSABlock attention (ultralytics block.Attention), dim = x.C.
//   num_heads = max(dim/64, 1); head_dim = dim/num_heads; key_dim = head_dim/2; scale = 1/sqrt(key_dim)
//   attn = softmax((q*scale)^T @ k);  out = (v @ attn^T) + pe(v);  return proj(out)
// fq: OpenVINO's five interior quantizers (default all-off = FP32). q@k and softmax stay FP32, as in OpenVINO.
inline Tensor attention(const Tensor& x, const ConvW& qkv, const ConvW& proj, const ConvW& pe,
                        const AttnFQ& fq = AttnFQ()) {
    Y26_PROF("attention (matmul+softmax)");
    const int C = x.C, H = x.H, W = x.W, N = H * W;
    const int num_heads = std::max(C / 64, 1);
    const int head_dim  = C / num_heads;
    const int key_dim   = head_dim / 2;            // attn_ratio 0.5
    const float scale   = 1.f / std::sqrt((float)key_dim);
    const int per_head  = 2 * key_dim + head_dim;  // channels per head in qkv output
    // Head sizes come from the input width; every q/k/v index assumes qkv's output is exactly nh*per_head wide.
    CSIM_HOST_ASSERT(qkv.oc == num_heads * per_head, "attention: qkv oc=" + std::to_string(qkv.oc) +
                     " but num_heads*per_head=" + std::to_string(num_heads * per_head));

    Tensor q = conv2d(x, qkv);                     // [nh*per_head, H, W]
    // q is scaled BEFORE it is quantized (OpenVINO's FakeQuantize follows the scale Multiply).
    std::vector<float> qs((size_t)num_heads * key_dim * N), ks(qs.size()),
                       vm((size_t)num_heads * head_dim * N);
    // Divide-free FQ::apply (bit-identical), one channel per iteration.
    CSIM_OMP_STATIC
    for (int hd = 0; hd < num_heads * key_dim; ++hd) {
        const int hh = hd / key_dim, d = hd % key_dim;
        fq.q.apply(&qs[(size_t)hd * N], &q.d[(size_t)(hh * per_head + d) * N], N, 1, scale);
        fq.k.apply(&ks[(size_t)hd * N], &q.d[(size_t)(hh * per_head + key_dim + d) * N], N);
    }
    // v as [C,H,W] for pe(): channel = head*head_dim + vd. v has two quantizers: v_mm (matmul) and v_pe (pe).
    Tensor vt(C, H, W);
    CSIM_OMP_STATIC
    for (int hv = 0; hv < num_heads * head_dim; ++hv) {
        const int hh = hv / head_dim, vd = hv % head_dim;
        const float* sv = &q.d[(size_t)(hh * per_head + 2 * key_dim + vd) * N];
        fq.v_pe.apply(&vt.d[(size_t)hv * N], sv, N);
        fq.v_mm.apply(&vm[(size_t)hv * N], sv, N);
    }
    Tensor pe_out = conv2d(vt, pe);                // dw3x3 positional encoding on v

    Tensor out(C, H, W);                            // v @ attn^T, laid out as [C,H,W]
    CSIM_OMP_DYNAMIC
    for (int hh = 0; hh < num_heads; ++hh) {
        const float* qh = &qs[(size_t)(hh * key_dim) * N];
        const float* kh = &ks[(size_t)(hh * key_dim) * N];
        const float* vh = &vm[(size_t)(hh * head_dim) * N];
        // QB queries at a time, P laid out [n2][QB]: unit-stride inner loops vectorize without reordering any sum.
        constexpr int QB = 8;
        std::vector<float> P((size_t)N * QB);
        for (int n0 = 0; n0 < N; n0 += QB) {
            const int nb = std::min(QB, N - n0);           // tail lanes compute on q=0, never stored
            std::fill(P.begin(), P.end(), 0.f);
            for (int kd = 0; kd < key_dim; ++kd) {
                float qv[QB] = {};
                for (int j = 0; j < nb; ++j) qv[j] = qh[(size_t)kd * N + n0 + j];
                const float* kr = &kh[(size_t)kd * N];
                for (int n2 = 0; n2 < N; ++n2)
                    for (int j = 0; j < QB; ++j) P[(size_t)n2 * QB + j] += qv[j] * kr[n2];
            }
            for (int j = 0; j < nb; ++j) {
                float mx = -1e30f;
                for (int n2 = 0; n2 < N; ++n2) { const float s = P[(size_t)n2 * QB + j]; if (s > mx) mx = s; }
                float sum = 0.f;
                for (int n2 = 0; n2 < N; ++n2) {
                    const float e = std::exp(P[(size_t)n2 * QB + j] - mx);
                    P[(size_t)n2 * QB + j] = e; sum += e;
                }
                const float inv = 1.f / sum;
                fq.sm.apply(&P[j], &P[j], N, QB, inv);   // quantized probs: fq.sm(P * inv), divide-free
            }
            for (int vd = 0; vd < head_dim; ++vd) {
                float acc[QB] = {};
                const float* vr = &vh[(size_t)vd * N];
                for (int n2 = 0; n2 < N; ++n2)
                    for (int j = 0; j < QB; ++j) acc[j] += vr[n2] * P[(size_t)n2 * QB + j];
                for (int j = 0; j < nb; ++j) out.d[(size_t)(hh * head_dim + vd) * N + n0 + j] = acc[j];
            }
        }
    }
    Tensor xattn = add(out, pe_out);
    return conv2d(xattn, proj);
}
