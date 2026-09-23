// Synthesizable W8A8 conv datapath: no STL, no heap, static bounds, ap_int accumulation. It shares the arithmetic
// of conv2d() in layer_ops.h, which stays the golden model. Integer accumulation is exact, so with the float
// dequant the kernel is bit-exact against conv2d(); -DY26_FX_DEQUANT switches to fixed point (not bit-exact).
// Build: g++ -O2 -std=c++14 -I<Vitis>/include ...
#pragma once
#include <ap_int.h>
#include <ap_fixed.h>

// Bounds from the shipping manifest. Macros so they stay one-line parameter changes.
#ifndef Y26_MAX_OW
#define Y26_MAX_OW    320    // layer 0 output row (640 stride-2); ties the 160x160 activations
#endif
#ifndef Y26_MAX_K
#define Y26_MAX_K     3      // no 5x5 conv in the model - SPPF's k5 is a maxpool, not a conv
#endif
#define Y26_MAX_DEPTH 2304   // max (ic/groups)*kh*kw, at 22.m.0.0.cv1.conv
#define Y26_MAX_OC    512
#define Y26_MAX_IC    663

// Y26_LANES: MAC lanes, run over the INPUT CHANNEL axis (the lane index is also the activation bank index);
// the column-parallel form is the fallback for icpg < Y26_LANES (depthwise). A layer uses at most icpg lanes.
// Must be a power of two: channel c -> bank c % Y26_LANES must lower to a mask and a shift.
#ifndef Y26_LANES
#define Y26_LANES     16
#endif

// Y26_OCPACK: run Y26_OCPACK_P output channels at once through the dense MAC pass when icpg <= Y26_ICGRP
// (= Y26_LANES / Y26_OCPACK_P), on lanes that would otherwise idle. Only the weight differs per member; the
// activation read is shared. P is compile-time and both p and ic_local are unrolled, so the bank index
// p*Y26_ICGRP + ic_local is a constant per lane (no crossbar). Convs with icpg > Y26_ICGRP run unpacked, so
// npass stays 1 and xbuf is unchanged.
#ifdef Y26_OCPACK
  #ifndef Y26_OCPACK_P
  #define Y26_OCPACK_P 4
  #endif
  #if (Y26_LANES % Y26_OCPACK_P) != 0
    #error "Y26_OCPACK_P must divide Y26_LANES evenly, or the group width is not an integer."
  #endif
  #define Y26_ICGRP (Y26_LANES / Y26_OCPACK_P)
#endif

// Y26_TAPLANE: put the kh row taps on idle lanes. Lane l = t*Y26_TPCG + cc carries channel cc at row tap t
// (both unroll constants), so one iteration retires all kh row taps and the loop walks only kw.
// The activation is replicated into the idle banks (one bank cannot serve two rows per cycle); the weights are
// only permuted. The row shift is applied at read time (+t*W), keeping the staging address word-aligned.
#ifdef Y26_TAPLANE
  #ifndef Y26_TPCG
  #define Y26_TPCG 16
  #endif
  #define Y26_TP (Y26_ICGRP / Y26_TPCG)
  #if !defined(Y26_OCPACK) || !defined(Y26_A1) || !defined(Y26_XB64)
    #error "Y26_TAPLANE needs the packed A1/XB64 dense path - it maps the ICGRP lane group."
  #endif
  #if (Y26_ICGRP % Y26_TPCG)
    #error "Y26_TPCG must divide Y26_ICGRP evenly, or the lane sub-group is not an integer."
  #endif
  // Eligible iff the channels fit one lane sub-group and the row taps fit Y26_TP. The check that Y26_TPCG is a
  // multiple of Y26_WPE sits next to Y26_WPE below (an #if on an undefined macro reads as 0).
  #define Y26_TL_OK(icpg_, kh_) ((icpg_) <= Y26_TPCG && (kh_) > 1 && (kh_) <= Y26_TP)
#else
  #define Y26_TL_OK(icpg_, kh_) false
#endif

// Y26_DWP: depthwise output-channel packing, Y26_DWP members per pass. Activations are staged at modulus
// Y26_DWP so member p's channel ic0+p sits at bank p (an unroll constant); a runtime bank index builds a crossbar.
// A smaller Y26_DWP deepens banks and y26_Rblk shrinks R to compensate (down to 0): check it before lowering.
// accrow grows with Y26_ACCP.
#ifndef Y26_DWP
#define Y26_DWP 1                 // 1 = off (every use is gated)
#endif
#if (Y26_DWP > 1)
  #if !defined(Y26_OCPACK)
    #error "Y26_DWP > 1 requires -DY26_OCPACK: the packed epilogue and kwsum shapes are OCPACK-only."
  #endif
  #if (Y26_LANES % Y26_DWP) != 0
    #error "Y26_DWP must divide Y26_LANES evenly - member p reads bank p."
  #endif
#endif
#ifdef Y26_OCPACK
  // Every per-member array is sized by the WIDER of the two packing widths, because one conv packs
  // dense members and another packs depthwise members through the SAME arrays.
  #define Y26_ACCP ((Y26_DWP > Y26_OCPACK_P) ? Y26_DWP : Y26_OCPACK_P)
#endif

// Y26_ROWS: output rows per pipeline fill, amortizing fill/drain across rows. 1 = one row per fill.
// Costs FF and accrow/accwrow BRAM. Check the fused loop still reaches II=1.
#ifndef Y26_ROWS
#define Y26_ROWS      1
#endif

// Y26_EPI_WIDE: output elements per cycle in the zeroing pass and the epilogue (both share the accrow partition).
// accrow is cyclic-partitioned by Y26_EPI_WIDE, so each row is padded to OWP = ceil(OW/N)*N: a group crossing a
// row boundary would hit one bank twice and drop to II=2. The MAC's accrow store at a runtime ow gets a
// 1-of-N demux, affordable only for small N.
#ifndef Y26_EPI_WIDE
#define Y26_EPI_WIDE  1
#endif

// Y26_WT_WORD: gmem_wt port width in bits (HLS takes the port width from the pointer type).
// Y26_WT_TAPMAJOR: order each oc slice (tap, icl) instead of (icl, tap).
// Defined above Y26_WBANK because the padding below sets the bank depth.
#if defined(Y26_WT_WORD) && (Y26_WT_WORD > 8)
#define Y26_WPE (Y26_WT_WORD / 8)    // weight elements carried per port word
// DWP: member p's channel is staged at bank p*Y26_WPE, so Y26_LANES/Y26_WPE caps the pack width.
#if defined(Y26_DWP) && (Y26_DWP > 1) && ((Y26_DWP) * (Y26_WPE) > Y26_LANES)
  #error "Y26_DWP * Y26_WPE exceeds Y26_LANES: member p's weights would be staged past wbuf."
#endif
#else
#define Y26_WPE 1
#endif

// TAPLANE: the permuted weight destination steps by Y26_TPCG/Y26_WPE port words, so Y26_TPCG must be a multiple.
#if defined(Y26_TAPLANE) && (Y26_TPCG % Y26_WPE)
  #error "Y26_TPCG must be a multiple of Y26_WPE: the tap sub-group step is counted in port words."
#endif

// Padding (needed once Y26_WPE > 1): each oc slice (under TAPMAJOR, each (oc, tap) channel run) is rounded up
// to a multiple of Y26_WPE, so slices start on a port word and one word's elements land in banks that differ only
// in their low bits (provably conflict-free). Pad elements are written but never read.
#define Y26_WALIGN(n) ((((n) + Y26_WPE - 1) / Y26_WPE) * Y26_WPE)
#if defined(Y26_WT_TAPMAJOR)
  #define Y26_WICPG(icpg)               Y26_WALIGN(icpg)             // padded channels per tap
  #define Y26_WSTRIDE(icpg, ktap)       (Y26_WICPG(icpg) * (ktap))
  #define Y26_WIDX(icpg, ktap, icl, t)  ((t) * Y26_WICPG(icpg) + (icl))
#else
  #define Y26_WICPG(icpg)               (icpg)
  #define Y26_WSTRIDE(icpg, ktap)       Y26_WALIGN((icpg) * (ktap))
  #define Y26_WIDX(icpg, ktap, icl, t)  ((icl) * (ktap) + (t))
#endif

// Depth of one lane bank of wbuf (staged per-output-channel weights): ceil(icpg/Y26_LANES) rows of ktap, plus
// the partial last tile and one row for padding. Too small = a silent out-of-bounds write.
#if Y26_WPE > 1
#if defined(Y26_A1) && defined(Y26_OCPACK)
// Packed path: weights bank by Y26_ICGRP, so a bank holds ceil(icpg/Y26_ICGRP) rows of ktap.
#define Y26_WBANK ((Y26_MAX_DEPTH + Y26_ICGRP - 1) / Y26_ICGRP + 2 * Y26_MAX_K * Y26_MAX_K)
#else
#define Y26_WBANK ((Y26_MAX_DEPTH + Y26_LANES - 1) / Y26_LANES + 2 * Y26_MAX_K * Y26_MAX_K)
#endif
#else
#if defined(Y26_A1) && defined(Y26_OCPACK)
#define Y26_WBANK ((Y26_MAX_DEPTH + Y26_ICGRP - 1) / Y26_ICGRP + Y26_MAX_K * Y26_MAX_K)
#else
#define Y26_WBANK ((Y26_MAX_DEPTH + Y26_LANES - 1) / Y26_LANES + Y26_MAX_K * Y26_MAX_K)
#endif
#endif

// Activations are held whole by default; Y26_ACT_STRIPS = 2 would tile them into H-strips.
// Largest activation: 160x160x64 == 320x320x16 elements.
#ifndef Y26_ACT_STRIPS
#define Y26_ACT_STRIPS 1     // 1 = untiled
#endif
#ifndef Y26_ACT_MAX_ELEMS   // overridable
#define Y26_ACT_MAX_ELEMS  1638400UL                             // 160*160*64 == 320*320*16
#endif
#define Y26_ACT_BUF_ELEMS  (Y26_ACT_MAX_ELEMS / Y26_ACT_STRIPS)  // per-strip resident buffer

// Max INPUT width, needed because layer 0 consumes the 640x640 image (every later layer is <=320).
#define Y26_MAX_IW    640

// Activation banking: xbuf[l][off] with l = channel % Y26_LANES, partitioned completely on l, so the bank index
// is the unrolled loop variable and HLS can prove lanes never conflict (a runtime cyclic bank index builds a
// crossbar). All lanes read the same in-bank offset.
//   bank c % Y26_LANES, off (c / Y26_LANES)*H*W + y*W + x
// A conv with ic < Y26_LANES at large H*W (e.g. a 640x640x3 input) exceeds the per-bank budget. That is only
// asserted in simulation; no shipping conv reaches it.
#define Y26_ACT_LANE_ELEMS (Y26_ACT_BUF_ELEMS / Y26_LANES)

// -DY26_FX_DEQUANT: fixed-point dequant instead of float (not bit-exact; gate on accuracy).
// Two types: the raw accumulator stays ap_int and is consumed by a multiply, never narrowed into y26_fx_t.
//   y26_scale_t - calibration constants (step, lo, wsc): small, mostly fractional bits.
//   y26_fx_t    - values; must hold step*acc (~1e4-1e5) before wsc scales it back down.
#ifndef Y26_FX_SCALE_W
#define Y26_FX_SCALE_W 32
#endif
#ifndef Y26_FX_SCALE_I
#define Y26_FX_SCALE_I 8     // range +-128
#endif
#ifndef Y26_FX_W
#define Y26_FX_W 48          // total bits
#endif
#ifndef Y26_FX_I
#define Y26_FX_I 24          // integer bits - must cover step*acc, not just the output
#endif
// AP_RND / AP_SAT: round to nearest and saturate, so an overflow clips instead of wrapping.
typedef ap_fixed<Y26_FX_SCALE_W, Y26_FX_SCALE_I, AP_RND, AP_SAT> y26_scale_t;
typedef ap_fixed<Y26_FX_W, Y26_FX_I, AP_RND, AP_SAT>             y26_fx_t;

// -DY26_FUSE (with Y26_FX_DEQUANT): v = (wsc*step)*acc + (wsc*lo)*zpw + b. Both products are per-oc constants
// formed once in weight staging, removing a wide multiply from every epilogue lane. y26_kscale_t is all-fractional
// (range +-0.5); AP_SAT clips if a calibration exceeds it.
#if defined(Y26_FUSE) && defined(Y26_FX_DEQUANT)
#define Y26_FUSE_ON 1
typedef ap_fixed<32, 0, AP_RND, AP_SAT> y26_kscale_t;
// Accumulator operands narrowed to the DSP48E2's 27-bit port (32x27 = 2 DSPs, 32x32 = 4). The analytical worst
// case slightly exceeds 27 bits; a csim assert at the fold site checks the real values.
#define Y26_KACC_W 27
#endif

// Types.
// SmoothQuant activations are asymmetric uint8 codes: clamp(round((v*ssc - lo)/step), 0, 255).
typedef ap_uint<8>  y26_act_t;
// -DY26_ACT_WORD: gmem_act port width in bits. HLS takes the width from the pointer type, so X is declared as a
// wide word and unpacked on-chip.
#if defined(Y26_ACT_WORD) && (Y26_ACT_WORD > 8)
typedef ap_uint<Y26_ACT_WORD> y26_xw_t;
#define Y26_XPE (Y26_ACT_WORD / 8)   // activation elements carried per port word
#else
typedef y26_act_t             y26_xw_t;
#define Y26_XPE 1
#endif
// Weights are symmetric int8 codes, per-output-channel scaled.
typedef ap_int<8>   y26_wt_t;
// Weight port word. Activations are unsigned codes, weights signed: unpack through ap_int<8>, since a plain
// (y26_wt_t)range() sign-extends wrongly.
#if defined(Y26_WT_WORD) && (Y26_WT_WORD > 8)
typedef ap_uint<Y26_WT_WORD> y26_ww_t;
#else
typedef y26_wt_t             y26_ww_t;
#endif
// -DY26_OUT_WORD: gmem_out port width. With float* Y, N epilogue stores per cycle share one 32-bit write port
// (II = N); packing them into one wide word keeps II=1.
#if defined(Y26_OUT_WORD) && (Y26_OUT_WORD > 32)
typedef ap_uint<Y26_OUT_WORD> y26_yw_t;
#define Y26_YPE (Y26_OUT_WORD / 32)  // output floats carried per port word
#else
typedef float                 y26_yw_t;
#define Y26_YPE 1
#endif
// Bit-cast container. Union punning works in both g++ and Vitis HLS; a reinterpret_cast breaks strict aliasing.
union y26_fp32 { float f; uint32_t u; };
// Wide stores need (oc*OH + ohb)*OW to be a multiple of Y26_YPE and no partial last word. True for this network
// (square maps, OW in {20,40,80,160,320}), so there is no tail path. A beat must also sit in one ybuf group,
// which needs OW % Y26_YPE == 0: Y26_OUT_WORD=128 is supported, 256 needs a padded host row stride.
#define Y26_YBANK Y26_EPI_WIDE
// The store loop needs Y26_YPE <= Y26_YBANK, and the epilogue's group counter assumes the bank count equals the
// epilogue width. Neither would show up as a compile error.
#if Y26_YPE > Y26_YBANK
#error "Y26_OUT_WORD/32 must not exceed Y26_EPI_WIDE - widen the epilogue first"
#endif
// ybuf is sized in PADDED units: Y26_ROWS rows of OWP = ceil(OW/Y26_YBANK)*Y26_YBANK columns, so the
// worst case adds one whole group per row over the unpadded bound.
#define Y26_YBUF_GRP ((Y26_ROWS * (Y26_MAX_OW + Y26_YBANK)) / Y26_YBANK)

// Y26_YDIRECT: with YS == OWP and Y26_YPE == Y26_YBANK the store loop reads ybuf in write order, so the epilogue
// packs the m_axi word itself and ybuf is dropped.
#if defined(Y26_YDIRECT) && (Y26_YPE > 1) && (Y26_YPE == Y26_YBANK) && defined(Y26_YSTRIDE_PAD)
#define Y26_YDIRECT_ON 1
#endif
// Y26_EFLAT: fuses the epilogue's member loop (p < pack) into the store loop. Needs Y26_YDIRECT_ON and Y26_OCPACK.
#if defined(Y26_EFLAT) && defined(Y26_YDIRECT_ON) && defined(Y26_OCPACK)
#define Y26_EFLAT_ON 1
#endif
// Worst case |acc| = 255 * 127 * 2304 = 74,615,040 -> 27 bits signed; 32 is the natural width.
typedef ap_int<32>  y26_acc_t;

// Index types. Offsets computed in `long` make HLS build 64-bit multipliers (several DSPs each) for addresses;
// the largest offsets need 21-22 bits. ap_uint products widen and assignment truncates silently, so the casts
// at each multiply site are deliberate.
typedef ap_uint<24> y26_idx_t;   // flat element index into any activation / weight / output tensor
typedef ap_uint<22> y26_out_t;   // outer-product intermediate before the final stride multiply

// Per-conv descriptor (POD, synthesizable): the ConvW fields the SmoothQuant path uses.
struct Y26ConvCfg {
    int oc, ic, kh, kw, sh, sw, ph, pw, groups;
    int act;              // 0 identity, 1 silu
    // Dequant: real = wsc[o] * (step * acc + lo * accw) + bias, then activate.
    // `perch` (depthwise-only) swaps the per-tensor step/lo for per-input-channel vectors.
    int perch;            // 0 -> use step/lo scalars; 1 -> use step_v/lo_v indexed by channel
    float step, lo;
    const float* step_v;  // [ic], perch only (may be null)
    const float* lo_v;    // [ic], perch only (may be null)
    const float* wsc;     // [oc] per-output-channel weight scale
    const float* bias;    // [oc]
#ifdef Y26_YQ8
    // yq != 0: each 32-bit Y slot carries the consumer's uint8 code q_u8(v, q_s[o], q_lo[o], q_st[o]) instead of
    // the float v, so the host narrows instead of re-quantizing. Params are per output channel.
    int yq;
    const float* q_s;
    const float* q_lo;
    const float* q_st;
#endif
};

#ifdef Y26_YQ8
// q_u8 (layer_ops.h) on the kernel side, bit for bit: the same three IEEE single ops in the same order, then
// round-half-even and the [0,255] clamp done on an EXACT fixed-point copy (0.5 < d < 254.5 has its lowest
// significant bit at >= 2^-24, so ap_ufixed<32,8> holds it exactly) instead of nearbyint.
#if !defined(Y26_PREFETCH) || !defined(Y26_OCPACK)
#error "Y26_YQ8 stages its params through the PREFETCH arrays of the OCPACK epilogue"
#endif
inline ap_uint<8> y26_q8(float v, float s, float lo, float st) {
    const float t = v * s;
    const float u = t - lo;
    const float d = u / st;
    if (!(d > 0.5f)) return 0;                     // nearbyint(d) <= 0 (0.5 -> 0, even)
    if (d >= 254.5f) return d > 254.5f ? 255 : 254;
    const ap_ufixed<32, 8> f = d;
    const ap_uint<8>  k  = f.range(31, 24);
    const ap_uint<24> fr = f.range(23, 0);
    const ap_uint<24> half = (ap_uint<24>)1 << 23;
    return (ap_uint<8>)(k + ((fr > half || (fr == half && k[0])) ? 1 : 0));
}
#endif

// One conv over pre-quantized activation codes.
//   X   [ic][H][W]                uint8 activation codes
//   Wt  [oc][ic/groups][kh][kw]   int8 weight codes
//   Y   [oc][OH][OW]              float output (see yq)
// OH/OW as in conv2d: (H + 2*p - k)/s + 1.
void y26_conv2d_hls(const y26_xw_t* X, int H, int W,
                    const y26_ww_t*  Wt,
                    const Y26ConvCfg& c,
                    y26_yw_t* Y);        // `float*` unless -DY26_OUT_WORD widens gmem_out

// Output dims, shared by kernel and host so they cannot drift apart.
inline int y26_oh(int H, const Y26ConvCfg& c) { return (H + 2 * c.ph - c.kh) / c.sh + 1; }
inline int y26_ow(int W, const Y26ConvCfg& c) { return (W + 2 * c.pw - c.kw) / c.sw + 1; }

// AXI top-level function (synthesis entry point). The signature is flattened because a struct of pointers
// cannot cross the AXI boundary. step_v/lo_v are read only when perch != 0 (may be null otherwise).
// Y26_DEPTH_*: m_axi depths, needed by co-simulation only (no effect on hardware or on reported cycles).
// Each is the design maximum; the testbench sizes its buffers to them and skips larger convs. depth= takes
// a single constant, so they are literals. All are overridable.
#ifndef Y26_DEPTH_X
#define Y26_DEPTH_X    1638400   // Y26_ACT_MAX_ELEMS
#endif
#ifndef Y26_DEPTH_Y
#define Y26_DEPTH_Y    1638400   // same bound: the largest output is also 320*320*16
#endif
// depth= counts port words.
#define Y26_DEPTH_XW   (Y26_DEPTH_X / Y26_XPE)
// Same for the output port.
#define Y26_DEPTH_YW   (Y26_DEPTH_Y / Y26_YPE)
#ifndef Y26_DEPTH_WT
#if Y26_WPE > 1
// Padded weight blob: worst case adds (Y26_WPE-1) elements per (oc, tap).
#define Y26_DEPTH_WT (Y26_MAX_OC * (Y26_MAX_DEPTH + Y26_MAX_K * Y26_MAX_K * (Y26_WPE - 1)))
#else
#define Y26_DEPTH_WT   1179648   // Y26_MAX_OC * Y26_MAX_DEPTH == 512 * 2304
#endif
#endif
// Port words.
#define Y26_DEPTH_WTW  (Y26_DEPTH_WT / Y26_WPE)
#ifndef Y26_DEPTH_OC
#define Y26_DEPTH_OC   512       // Y26_MAX_OC   - wsc[oc], bias[oc]
#endif
#ifndef Y26_DEPTH_IC
#define Y26_DEPTH_IC   663       // Y26_MAX_IC   - step_v[ic], lo_v[ic]
#endif
void y26_conv_top(const y26_xw_t*  X,        // activations (m_axi, Y26_ACT_WORD bits)
                  const y26_ww_t*  Wt,       // int8 weight codes (m_axi)
                  const float*     wsc,      // [oc] per-output-channel weight scale (m_axi)
                  const float*     bias,     // [oc] (m_axi)
                  const float*     step_v,   // [ic] perch only (m_axi)
                  const float*     lo_v,     // [ic] perch only (m_axi)
                  y26_yw_t*        Y,        // output (m_axi, Y26_OUT_WORD bits)
                  int H, int W,
                  int oc, int ic, int kh, int kw,
                  int sh, int sw, int ph, int pw,
                  int groups, int act, int perch,
                  float step, float lo
#ifdef Y26_YQ8
                  , int yq, const float* q_s, const float* q_lo, const float* q_st
#endif
                  );
