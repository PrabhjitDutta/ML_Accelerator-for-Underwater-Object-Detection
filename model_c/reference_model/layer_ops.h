// layer_ops.h - FP32 primitives for the YOLO26s trunk C-simulation.
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

// q_u8 over n values -> uint8 codes, bit-identical to q_u8 per element but with no divide. x*(1/step) is within
// |q|*2^-22.4 of x/step, so both round alike unless a .5 tie lies between them - and then x*(1/step) is within
// 2^-12 of it (|q| < 1024; beyond that both clamp alike). Such a 64-block is redone with the divide (~1% of
// blocks). Hoisted __restrict pointers + the dynamic cost model let GCC vectorize the block (NEON on the A53,
// whose divider is unpipelined). board_host/test_quantizer.cpp checks it against q_u8.
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
// One source of a conv's input channel-concat (ps-notes §10). PLAIN is a tensor. The rest are an op whose ONLY
// reader is that conv, so the board fuses it into its quantizer instead of building the tensor (the CPU path
// materializes it, y26_materialize): ADD = a+b (a residual), UP2X = nearest 2x upsample of a, POOL = k chained
// maxpool(5,1,2) of a (SPPF). Exact: ADD is the same float add; q is elementwise (commutes with the upsample) and
// monotone for ssc > 0, step > 0 (commutes with max - asserted where it is used).
struct Src {
    enum Op { PLAIN, ADD, UP2X, POOL };
    Op op; const Tensor* a; const Tensor* b; int k;
    Src(const Tensor* t) : op(PLAIN), a(t), b(nullptr), k(0) {}   // implicit: {&x, &y} call sites are unchanged
    static Src sum(const Tensor* x, const Tensor* y) { Src s(x); s.op = ADD; s.b = y; return s; }
    static Src up2x(const Tensor* x) { Src s(x); s.op = UP2X; return s; }
    static Src pool(const Tensor* x, int n) { Src s(x); s.op = POOL; s.k = n; return s; }
    int C() const { return a->C; }
    int H() const { return op == UP2X ? 2 * a->H : a->H; }
    int W() const { return op == UP2X ? 2 * a->W : a->W; }
};
// Board: one PLAIN source's X codes, packed ahead of its conv (ps-notes §13) - on its own thread when async, so it
// overlaps the convs before it; deferred to the conv otherwise. Null (a no-op) off the board or for a CPU conv.
struct Y26Pre;
using Pre = std::shared_ptr<Y26Pre>;
#ifdef Y26_BOARD
// board_host/board_host.cpp: the ZCU102 kernel. xs = the conv's input as channel-concatenated sources; pre = one of
// them already packed (y26_board_prepack).
Tensor y26_board_conv(const std::vector<Src>& xs, const ConvW& c, const Pre& pre = nullptr);
// t = c's FIRST (last = false) or LAST input source. t's storage must outlive the conv; t itself may be moved.
Pre y26_board_prepack(const Tensor* t, const ConvW& c, bool last);
// The frame loop's input as uint8 k (float k/255.f): 0.conv packs it through a 256-entry table. u8 = null forgets d.
void y26_input_u8(const float* d, std::shared_ptr<const std::vector<uint8_t>> u8);
// PS profile (board builds only): SELF time per op - a scope's time excludes the timed scopes inside it,
// so the table sums to the trunk's CPU time. Per thread: with async branches the rows sum past the wall clock.
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
bool y26_async();   // board_host/board_host.cpp: independent trunk branches on their own threads (Y26_ASYNC=0: off)
void y26_frame_end(int frame);   // board_host/board_host.cpp: frame loop - print this frame's split, start the next
void y26_board_weights(const Weights& W);   // frame loop: HW lever B (-DY26_YQ8, Y26_YQ=1) reads consumer params
#else
inline Pre y26_board_prepack(const Tensor*, const ConvW&, bool) { return nullptr; }
inline void y26_input_u8(const float*, std::shared_ptr<const std::vector<uint8_t>>) {}
inline void y26_board_weights(const Weights&) {}
#define Y26_PROF(n)
inline bool y26_async() { return false; }
inline void y26_frame_end(int) {}
#endif
inline Tensor conv2d(const Tensor& x, const ConvW& c) {
#ifdef Y26_BOARD
    if (c.quant && c.asym) return y26_board_conv({&x}, c);   // same eligibility as the tb gates
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

// Elementwise add (residual). a,b same shape -- checked, because the loop is bounded by a's size and
// a narrower b would be read past its end (silently wrong output, exit 0), the same failure mode the
// SPPF residual and the truncated-weight-file checks already guard against.
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
    Y26_PROF("attention (matmul+softmax)");
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
    // Divide-free FQ::apply (bit-identical, ps-notes §12), one channel per iteration across the cores.
    CSIM_OMP_STATIC
    for (int hd = 0; hd < num_heads * key_dim; ++hd) {
        const int hh = hd / key_dim, d = hd % key_dim;
        fq.q.apply(&qs[(size_t)hd * N], &q.d[(size_t)(hh * per_head + d) * N], N, 1, scale);
        fq.k.apply(&ks[(size_t)hd * N], &q.d[(size_t)(hh * per_head + key_dim + d) * N], N);
    }
    // v reshaped back to [C,H,W] for pe(): channel = head*head_dim + vd. v is quantized TWICE with
    // separately calibrated ranges -- v_mm for the v@attn^T matmul, v_pe for pe(v).
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
        // QB queries at a time, P laid out [n2][QB]: every inner loop is unit-stride over the block, so
        // it vectorizes WITHOUT reassociating - each score/acc still sums kd (resp. n2) in ascending order,
        // bit-identical to the one-query-at-a-time loop (3.7x on the PC, 1 thread, 256ch @ 20x20).
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
