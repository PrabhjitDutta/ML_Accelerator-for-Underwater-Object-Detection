// Synthesizable conv datapath. Mirrors conv2d()'s SmoothQuant path in layer_ops.h (loop order, tap ranges,
// zero-point weight sum, dequant promotions) with ap_int<32> accumulators and no STL/heap/throw, so with the
// float dequant it is bit-exact against conv2d(). Integer sums of int8/uint8 codes are exact, so the MAC
// reduction may be reordered freely; the dequant may not.
#include "conv_engine.h"
#include <cmath>
#include <cassert>

// Local copies of two conv2d helpers: layer_ops.h pulls in STL headers that cannot be synthesized.

// Output-column range [o0,o1) where tap column `base` lands inside the image: padding becomes a loop bound.
static inline void y26_ow_range(int base, int sw, int W, int OW, int& o0, int& o1) {
    // ponytail: keep this pragma. If the function grows past the HLS inline heuristic it is silently outlined into
    // a non-pipelined submodule inside the MAC nest; check `y26_ow_range` is absent from csynth.rpt.
#pragma HLS INLINE
    // `/ sw` is a shift (no divider core). Precondition sw in {1,2}, asserted once at conv entry.
    // The base >= 0 arm differs (trunc vs floor) but the ternary discards it.
    const int c8b_sh = sw >> 1;            // sw==1 -> 0, sw==2 -> 1
    o0 = base >= 0 ? 0 : (-base + sw - 1) >> c8b_sh;
    o1 = W - 1 - base;
    o1 = o1 < 0 ? 0 : (o1 >> c8b_sh) + 1;
    if (o1 > OW) o1 = OW;
}

// SiLU in double, rounded to float: identical to layer_ops.h's silu().
static inline float y26_silu(float v) { double d = v; return (float)(d / (1.0 + std::exp(-d))); }

#ifdef Y26_SILU_LUT
#include "silu_lut.h"
// Fixed-point SiLU: table lookup + linear interpolation, replacing the double exp/div chain. Used only with
// Y26_FX_DEQUANT (the float build stays bit-exact). Entries are raw fixed-point ints that drop into `.V`.
static inline y26_fx_t y26_silu_fx(y26_fx_t v) {
    // Outside +/-16, SiLU(x) - x and SiLU(-x) are < 2e-6, so clamping is exact enough; it also bounds the table index.
    if (v >= (y26_fx_t)Y26_SILU_HI) return v;
    if (v <= (y26_fx_t)Y26_SILU_LO) return (y26_fx_t)0;

    // Position in table units: (v + 16) * 32, a shift.
    const ap_fixed<40,12> t = (ap_fixed<40,12>)((v - (y26_fx_t)Y26_SILU_LO) * (y26_fx_t)32);
    const int             i = t.to_int();                       // t > 0, so truncation == floor
    // Defensive clamp (already guaranteed by the early returns): a bad range edit gives a wrong answer, not a bad ROM read.
    const int ii = (i < 0) ? 0 : ((i > Y26_SILU_N - 1) ? (Y26_SILU_N - 1) : i);

    const ap_ufixed<16,0> f = (ap_ufixed<16,0>)(t - (ap_fixed<40,12>)ii);
    const ap_int<32>      a = Y26_SILU_TAB[ii];
    const ap_int<32>      b = Y26_SILU_TAB[ii + 1];
    const ap_int<33>      d = (ap_int<33>)b - (ap_int<33>)a;

    // Bit-reinterpret rather than rescale: `.V` assignment is ambiguous and a 2^-24 multiply would infer a
    // 48x48 multiplier. range(hi,lo) is free.
    const ap_int<49> raw = (ap_int<49>)a + (ap_int<49>)(d * f);
    y26_fx_t out = (y26_fx_t)0;
    out.range(Y26_FX_W - 1, 0) = (ap_uint<Y26_FX_W>)raw;
    return out;
}
#endif

#ifdef Y26_YQ8
// The code as the 32 bits of a float slot (the port word packs slots by bit pattern, never by value).
static inline float y26_q8_slot(float v, float s, float lo, float st) {
    y26_fp32 cvt;
    cvt.u = (uint32_t)(unsigned)y26_q8(v, s, lo, st);
    return cvt.f;
}
#endif
void y26_conv2d_hls(const y26_xw_t* X, int H, int W,
                    const y26_ww_t*  Wt,
                    const Y26ConvCfg& c,
                    y26_yw_t* Y) {
    const int OH   = y26_oh(H, c);
    const int OW   = y26_ow(W, c);
    // Checked once per conv. Compiled out of synthesis.
    assert((c.sw == 1 || c.sw == 2) && "C8b: y26_ow_range substitutes a shift for / sw");
    const int icpg = c.ic / c.groups;      // input channels per group
    const int ocpg = c.oc / c.groups;      // output channels per group
    const int ktap = c.kh * c.kw;

    // Activation staging: the input is copied once per conv into a banked on-chip buffer (the MAC lanes cannot be
    // fed through one m_axi port), amortized over every output channel.
    // Layout: channel c -> bank c % Y26_LANES, offset (c/Y26_LANES)*H*W + y*W + x. `complete dim=1` makes each
    // lane its own memory with a constant index after UNROLL, which is what gives II=1.
#ifdef Y26_XB64
    // XB64: with A1 + OCPACK + DWP, dense convs bank at ICGRP and depthwise at DWP, so banks [ICGRP, LANES) are
    // unused by this model.
  #if !(defined(Y26_A1) && defined(Y26_OCPACK) && Y26_DWP > 1 && Y26_DWP <= Y26_ICGRP)
    #error "Y26_XB64 needs Y26_A1, Y26_OCPACK and 1 < Y26_DWP <= Y26_ICGRP"
  #endif
    static y26_act_t xbuf[Y26_ICGRP][Y26_ACT_LANE_ELEMS];
#else
    static y26_act_t xbuf[Y26_LANES][Y26_ACT_LANE_ELEMS];
#endif
    #pragma HLS ARRAY_PARTITION variable=xbuf complete dim=1
    // Y26_XBUF_WIDE: reshape the bank so Y26_XBUF_WIDE consecutive elements form one wide word, matching a wide
    // gmem_act port. Only useful when Y26_ACT_WORD > 8.
#if defined(Y26_XBUF_WIDE) && (Y26_XBUF_WIDE > 1)
    #pragma HLS ARRAY_RESHAPE variable=xbuf cyclic factor=Y26_XBUF_WIDE dim=2
#endif

    const ap_uint<19> hw = (ap_uint<19>)((ap_uint<11>)H * (ap_uint<11>)W);
    // groups == 1 implies ic0 == 0, which makes bank(lane l) == l a compile-time fact in the MAC pass.
    const bool g1 = (c.groups == 1);

    // A2 row banding: a whole map needs ceil(ic/modulus)*H*W <= Y26_ACT_LANE_ELEMS per bank, which large convs
    // exceed. So only the input rows one output row-block reads, [y26_y0, +y26_bh), are staged.
    // A1: on the packed path the banking modulus is Y26_ICGRP (channel c -> bank c % Y26_ICGRP), so the packed
    // MAC finds channel icbi*Y26_ICGRP + l at bank l. Banks become Y26_OCPACK_P times deeper; banding absorbs it
    // by shrinking R.
#if defined(Y26_A1) && defined(Y26_OCPACK)
  #if Y26_DWP > 1
    // DWP: a packed depthwise conv is banked at the pack width, so member p's bank is the unroll constant p.
  #ifdef Y26_XB64
    // ponytail: XB64 has no grouped icpg > 1 datapath; such a conv is unsupported (asserted in sim only).
    assert(g1 || icpg == 1 || !"XB64: grouped icpg > 1 conv has no datapath");
    const int  y26_xmod   = (g1 && Y26_OCPACK_P > 1) ? Y26_ICGRP : Y26_DWP;
  #else
    const int  y26_xmod   = (g1 && Y26_OCPACK_P > 1) ? Y26_ICGRP
                          : ((icpg == 1) ? Y26_DWP : Y26_LANES);
  #endif
  #else
    const int  y26_xmod   = (g1 && Y26_OCPACK_P > 1) ? Y26_ICGRP : Y26_LANES;
  #endif
#else
    const int  y26_xmod   = Y26_LANES;
#endif
    const int  y26_icgn   = (c.ic + y26_xmod - 1) / y26_xmod;
    const bool y26_banded = ((long)y26_icgn * (long)H * (long)W) > (long)Y26_ACT_LANE_ELEMS;
    // Rblk = largest block height whose band fits. A block of R output rows reads
    // (R-1)*sh + kh input rows, so rows_cap input rows admit R = (rows_cap - kh)/sh + 1.
    int y26_Rblk = Y26_ROWS;
    // y26_p = smallest row count whose y26_p*W is a whole Y26_XPE word (1 when W % XPE == 0). The band is widened
    // to multiples of y26_p; reserving the extra rows here keeps the widened band inside the lane.
    int y26_p = 1;
    while (y26_p < Y26_XPE && ((y26_p * W) & (Y26_XPE - 1)) != 0) y26_p <<= 1;   // masks, not %: p and XPE are powers of 2
    if (y26_banded) {
        const int rows_cap = Y26_ACT_LANE_ELEMS / (y26_icgn * W);
        int r = ((rows_cap & ~(y26_p - 1)) - (y26_p - 1) - c.kh) / c.sh + 1;
        if (r > Y26_ROWS) r = Y26_ROWS;          // accrow is only Y26_ROWS deep
        // ponytail: FAIL-OPEN. When rows_cap < kh no block height fits and clamping to 1 reads unstaged rows.
        // Unreachable for the shipping manifest; the assert at the staging site is sim-only. Upgrade path: a status
        // output that fails the conv.
        y26_Rblk = (r < 1) ? 1 : r;
    }
#ifdef Y26_ACCFOLD
    // ACCFOLD: accrow's column axis is Y26_MAX_OW/2, so a conv wider than that stores each output row as two
    // plane rows and may use at most Y26_ROWS/2 rows per block.
    const bool y26_af = OW > Y26_MAX_OW / 2;
    if (y26_af && y26_Rblk > Y26_ROWS / 2) y26_Rblk = Y26_ROWS / 2;
#endif

    // On-chip weight staging: one output channel's weights (<= Y26_MAX_DEPTH), read from DRAM once per conv by a
    // stride-1 burst instead of once per output row. Banked by lane like xbuf (channel c -> bank c % Y26_LANES),
    // so lane l always reads bank l: no arbitration, one parallel read per tap. Banks are small (LUTRAM).
    static y26_wt_t wbuf[Y26_LANES][Y26_WBANK];
    #pragma HLS ARRAY_PARTITION variable=wbuf complete dim=1

    // Zero-point weight sums for the current output channel: Sum over icl of the int8 weight codes, per tap,
    // computed from the staged wbuf. Exact in int32: |sum| <= 127 * 2304.
#ifdef Y26_OCPACK
    // One kwsum row per packed member, filled by an unrolled p-loop (constant indices, no runtime-indexed array).
    ap_int<32> kwsum_o[Y26_ACCP][Y26_MAX_K * Y26_MAX_K];   // DWP: widest pack, not the dense one
    #define Y26_KWSUM(p_, t_) kwsum_o[p_][t_]
#else
    ap_int<32> kwsum_o[Y26_MAX_K * Y26_MAX_K];
    #define Y26_KWSUM(p_, t_) kwsum_o[t_]
#endif

    // Per-output-row accumulators, one per output column. The lanes run over input channels, so the MAC touches
    // one accrow[ow] per cycle. One plane per fused row (Y26_ROWS).
#ifdef Y26_OCPACK
    // One accrow plane per packed member. Slots p >= pack hold don't-care values that are never read.
    // Present whenever Y26_OCPACK is compiled in.
  #ifdef Y26_ACCFOLD
    #if (Y26_MAX_OW / 2) % Y26_EPI_WIDE != 0 || Y26_ROWS % 2 != 0
      #error "Y26_ACCFOLD needs Y26_EPI_WIDE | Y26_MAX_OW/2 (groups stay in one plane row) and even Y26_ROWS"
    #endif
    ap_int<32> accrow [Y26_ACCP][Y26_ROWS][Y26_MAX_OW / 2];  // ACCFOLD: see y26_af at the Rblk clamp
  #else
    ap_int<32> accrow [Y26_ACCP][Y26_ROWS][Y26_MAX_OW];     // DWP: widest pack, not the dense one
  #endif
  #if Y26_EPI_WIDE > 1
    // Cyclic by epilogue width on the OW axis (dim 3 behind the pack dimension).
    #pragma HLS ARRAY_PARTITION variable=accrow  cyclic factor=Y26_EPI_WIDE dim=3
  #endif
    // Y26_ACCROW(p, rr, ow) addresses accrow in every configuration; without Y26_OCPACK it is accrow[rr][ow].
  #ifdef Y26_ACCFOLD
    // Folded conv: output (r, o) lives at plane row 2r + (o >= 160), column o mod 160. 160 is a multiple of
    // Y26_EPI_WIDE, so every epilogue group stays row-local.
    #define Y26_AFB(o_)             (y26_af && (o_) >= Y26_MAX_OW / 2)
    #define Y26_ACCROW(p_, r_, o_)  accrow[p_][y26_af ? (((r_) << 1) | (Y26_AFB(o_) ? 1 : 0)) : (r_)] \
                                          [Y26_AFB(o_) ? (o_) - Y26_MAX_OW / 2 : (o_)]
  #else
    #define Y26_ACCROW(p_, r_, o_)  accrow[p_][r_][o_]
  #endif
#else
  #ifdef Y26_ACCFOLD
    #error "Y26_ACCFOLD needs Y26_OCPACK"
  #endif
    ap_int<32> accrow [Y26_ROWS][Y26_MAX_OW];
#if Y26_EPI_WIDE > 1
    // Cyclic by the epilogue width on the column axis, so a row-local group of Y26_EPI_WIDE columns is one bank each.
    #pragma HLS ARRAY_PARTITION variable=accrow  cyclic factor=Y26_EPI_WIDE dim=2
#endif
    #define Y26_ACCROW(p_, r_, o_)  accrow[r_][o_]
#endif

    // The zero-point plane depends only on geometry and kwsum_o, so it is reconstructed in the epilogue instead
    // of stored: accw(p,er,owl) = SUM of kwsum over taps with kh live for er and owl in [o0(kw),o1(kw)).
    // At most Y26_MAX_K column intervals exist; they are hoisted here.
    int zpo0[Y26_MAX_K], zpo1[Y26_MAX_K];
    #pragma HLS ARRAY_PARTITION variable=zpo0 complete
    #pragma HLS ARRAY_PARTITION variable=zpo1 complete
    for (int kwi = 0; kwi < Y26_MAX_K; ++kwi) {
        #pragma HLS UNROLL
        // Taps past c.kw get an empty range.
        int a = 0, b = 0;
        if (kwi < c.kw) y26_ow_range(kwi - c.pw, c.sw, W, OW, a, b);
        zpo0[kwi] = a;
        zpo1[kwi] = b;
    }

#ifdef Y26_OCPACK
    // pack_cap: output channels packed per MAC pass for this conv. Dense convs (g1) and depthwise (icpg == 1)
    // pack; a general grouped conv gets 1 and takes the unpacked fallback (none in this model).
#ifdef Y26_A1
    // dense convs pack at any icpg (the re-banking above keeps channel l at bank l).
#if Y26_DWP > 1
    // DWP: depthwise packs Y26_DWP wide. The loop width, activation modulus, weight group width, kwsum shape and
    // per-member array extents all change together.
    const int pack_cap = g1 ? Y26_OCPACK_P : ((icpg == 1) ? Y26_DWP : 1);
#else
    const int pack_cap = g1 ? Y26_OCPACK_P : 1;   // dense at any icpg; depthwise stays unpacked
#endif
#else
    const int pack_cap = (g1 && icpg <= Y26_ICGRP) ? Y26_OCPACK_P : 1;   // dwA: dense only
#endif
#endif

    // TAPLANE: one conv-level decision read by both staging loops and the MAC, so they cannot disagree.
    const bool y26_tl = (pack_cap > 1) && g1 && Y26_TL_OK(icpg, c.kh);

    // Row-block loop, outside the oc loop so each band is staged once. The original inner block loop is made
    // single-trip at the block chosen here.
    int y26_staged_y0 = -1, y26_staged_bh = -1;      // band currently resident in xbuf
#ifdef Y26_PREFETCH
    // Per-conv vectors (wsc, bias, step, lo) are burst into BRAM once here, one loop per array (one loop over the
    // bundle would not burst).
    assert(!c.perch || (icpg == 1 && ocpg == 1));   // perch index below is ocp, which needs both
    static float pf_wsc[Y26_MAX_OC], pf_bias[Y26_MAX_OC], pf_step[Y26_MAX_OC], pf_lo[Y26_MAX_OC];
    #pragma HLS BIND_STORAGE variable=pf_wsc  type=ram_1p impl=bram
    #pragma HLS BIND_STORAGE variable=pf_bias type=ram_1p impl=bram
    #pragma HLS BIND_STORAGE variable=pf_step type=ram_1p impl=bram
    #pragma HLS BIND_STORAGE variable=pf_lo   type=ram_1p impl=bram
    for (int o = 0; o < c.oc; ++o) {
        #pragma HLS PIPELINE II=1
        const float w = c.wsc[o];
        pf_wsc[o] = w;
#ifdef Y26_FUSE_ON
        // L-FUSE: k1 = wsc*step and k2 = wsc*lo are formed in this per-conv loop, not per member per block.
        if (!c.perch) { pf_step[o] = (float)(w * c.step); pf_lo[o] = (float)(w * c.lo); }
#endif
    }
    if (c.bias) {
        for (int o = 0; o < c.oc; ++o) {
            #pragma HLS PIPELINE II=1
            pf_bias[o] = c.bias[o];
        }
    } else {
        for (int o = 0; o < c.oc; ++o) {
            #pragma HLS PIPELINE II=1
            pf_bias[o] = 0.f;
        }
    }
#ifdef Y26_YQ8
  #if !defined(Y26_OCPACK) || defined(Y26_EFLAT_ON) || !defined(Y26_PREFETCH)
    #error "Y26_YQ8 is implemented for the shipping OCPACK + PREFETCH, non-EFLAT epilogue only"
  #endif
    static float pf_qs[Y26_MAX_OC], pf_qlo[Y26_MAX_OC], pf_qst[Y26_MAX_OC];
    #pragma HLS BIND_STORAGE variable=pf_qs  type=ram_1p impl=bram
    #pragma HLS BIND_STORAGE variable=pf_qlo type=ram_1p impl=bram
    #pragma HLS BIND_STORAGE variable=pf_qst type=ram_1p impl=bram
    if (c.yq) {
        for (int o = 0; o < c.oc; ++o) {
            #pragma HLS PIPELINE II=1
            pf_qs[o] = c.q_s[o];
        }
        for (int o = 0; o < c.oc; ++o) {
            #pragma HLS PIPELINE II=1
            pf_qlo[o] = c.q_lo[o];
        }
        for (int o = 0; o < c.oc; ++o) {
            #pragma HLS PIPELINE II=1
            pf_qst[o] = c.q_st[o];
        }
    }
#endif
    if (c.perch) {
        for (int o = 0; o < c.oc; ++o) {
            #pragma HLS PIPELINE II=1
#ifdef Y26_FUSE_ON
            pf_step[o] = (float)(pf_wsc[o] * c.step_v[o]);   // L-FUSE: k1
#else
            pf_step[o] = c.step_v[o];
#endif
        }
        for (int o = 0; o < c.oc; ++o) {
            #pragma HLS PIPELINE II=1
#ifdef Y26_FUSE_ON
            pf_lo[o] = (float)(pf_wsc[o] * c.lo_v[o]);       // L-FUSE: k2
#else
            pf_lo[o] = c.lo_v[o];
#endif
        }
    }
#endif
    for (int ohbo = 0; ohbo < OH; ohbo += y26_Rblk) {
        // Input rows this block reads, clamped to the image, so band-relative 0 <= ih < y26_bh equals the absolute
        // test 0 <= ih < H.
        int y26_y0 = 0, y26_bh = H;
        if (y26_banded) {
            const int rlast = ((ohbo + y26_Rblk) < OH ? (ohbo + y26_Rblk) : OH) - 1;
            int t = ohbo * c.sh - c.ph;              // first input row touched
            int b = rlast * c.sh - c.ph + c.kh;      // one past the last
            if (t < 0) t = 0;
            if (b > H) b = H;
            // Widen to the word-aligned row period (no-op at p == 1). The extra rows are never read by a tap, and the
            // clamp keeps y0+bh <= H.
            y26_y0 = t & ~(y26_p - 1);
            y26_bh = (b - y26_y0 + y26_p - 1) & ~(y26_p - 1);
            if (y26_y0 + y26_bh > H) y26_bh = H - y26_y0;
        }
        const ap_uint<19> xb_hw = (ap_uint<19>)((ap_uint<11>)y26_bh * (ap_uint<11>)W);
        // Stage only when the band changes; unbanded convs stage once per conv. Both keys are needed: blocks can
        // share y0 but differ in bh.
        if (y26_staged_y0 != y26_y0 || y26_staged_bh != y26_bh) {
            y26_staged_y0 = y26_y0; y26_staged_bh = y26_bh;
#ifndef __SYNTHESIS__
            // The wide staging path reads X in Y26_XPE-element words, so the band's offset and length must be word
            // multiples (guaranteed by the y26_p widening unless H is not a multiple of y26_p).
            assert(((long)y26_y0 * W) % Y26_XPE == 0 && ((long)y26_bh * W) % Y26_XPE == 0);
            assert((long)y26_icgn * (long)y26_bh * (long)W <= (long)Y26_ACT_LANE_ELEMS);
#endif
        for (int ic = 0; ic < c.ic; ++ic) {
            const int         l    = ic & (y26_xmod - 1);          // bank  (both moduli are powers of 2)
            const ap_uint<8>  cg   = (ap_uint<8>)(ic / y26_xmod);   // channel-group within the bank
            const y26_idx_t   dst0 = (y26_idx_t)(cg * xb_hw);
            const y26_idx_t   src0 = (y26_idx_t)((ap_uint<11>)ic * hw)
                                   + (y26_idx_t)((ap_uint<11>)y26_y0 * (ap_uint<11>)W);
    #if Y26_XPE > 1
            // X is a Y26_ACT_WORD-bit port: one word carries Y26_XPE consecutive elements of one channel, word index
            // (src0 + j)/Y26_XPE, byte lane j % Y26_XPE. hw is a multiple of 8 for every conv, so src0 is word-aligned.
            // The inner UNROLL coalesces the Y26_XPE byte writes into one wide bank word.
            const y26_idx_t src0w = (y26_idx_t)(src0 / Y26_XPE);
            const y26_idx_t hww   = (y26_idx_t)(xb_hw / Y26_XPE);
            // dst0w makes the destination's word alignment syntactic: dst0 == cg*hw is a multiple of Y26_XPE, but hw is a
            // runtime value HLS cannot reason about, and it would build a barrel shifter (staging II 1 -> 8). Written in
            // word units, the Y26_XPE writes provably hit one word. Numerically identical.
            const y26_idx_t dst0w = (y26_idx_t)(cg * hww);
            for (y26_idx_t jw = 0; jw < hww; ++jw) {
                #pragma HLS PIPELINE II=1
                const y26_xw_t  wrd  = X[src0w + jw];
                const y26_idx_t dstb = (y26_idx_t)((dst0w + jw) * Y26_XPE);
                for (int b = 0; b < Y26_XPE; ++b) {
                    #pragma HLS UNROLL
                    xbuf[l][dstb + b] = (y26_act_t)wrd.range(b * 8 + 7, b * 8);
                }
#ifdef Y26_TAPLANE
                // TAPLANE: the same word into the Y26_TP-1 idle sub-groups. Each bank has its own write port, so this adds
                // writes, not cycles. `tp` is an unroll constant.
                if (y26_tl) {
                    for (int tp = 1; tp < Y26_TP; ++tp) {
                        #pragma HLS UNROLL
                        for (int b = 0; b < Y26_XPE; ++b) {
                            #pragma HLS UNROLL
                            xbuf[tp * Y26_TPCG + l][dstb + b] = (y26_act_t)wrd.range(b * 8 + 7, b * 8);
                        }
                    }
                }
#endif
            }
    #else
            for (y26_idx_t j = 0; j < xb_hw; ++j) {
                #pragma HLS PIPELINE II=1
                // Unrolled by the same factor so consecutive j coalesce into one word write (hw is a multiple of 8).
    #if defined(Y26_XBUF_WIDE) && (Y26_XBUF_WIDE > 1)
                #pragma HLS UNROLL factor=Y26_XBUF_WIDE
    #endif
                xbuf[l][dst0 + j] = X[src0 + j];
            }
    #endif
        }
        }

    for (int oc = 0; oc < c.oc; ) {
#ifdef Y26_OCPACK
        const int remain = c.oc - oc;
        const int pack   = (pack_cap < remain) ? pack_cap : remain;
#else
        const int pack = 1;
#endif
        const int g   = oc / ocpg;
        const int ic0 = g * icpg;
#ifdef Y26_OCPACK
        // Per-member step/lo: depthwise perch indexes by ic0, and each depthwise member has its own ic0 == oc + p.
        float step_o[Y26_ACCP];
        float lo_o[Y26_ACCP];
        #pragma HLS ARRAY_PARTITION variable=step_o complete
        #pragma HLS ARRAY_PARTITION variable=lo_o   complete
        // Per-member dequant scale/bias, filled by the staging loop below.
        float wsc_o[Y26_ACCP];
        float bs[Y26_ACCP];
#ifdef Y26_YQ8
        float qs_o[Y26_ACCP], qlo_o[Y26_ACCP], qst_o[Y26_ACCP];
        #pragma HLS ARRAY_PARTITION variable=qs_o  complete
        #pragma HLS ARRAY_PARTITION variable=qlo_o complete
        #pragma HLS ARRAY_PARTITION variable=qst_o complete
#endif
#else
        const float wsc_o  = c.wsc[oc];
#ifdef Y26_FUSE_ON
        // L-FUSE: step_o / lo_o carry k1 = wsc*step and k2 = wsc*lo from here on.
        const float step_o = (float)(wsc_o * (c.perch ? c.step_v[ic0] : c.step));
        const float lo_o   = (float)(wsc_o * (c.perch ? c.lo_v[ic0]   : c.lo));
#else
        const float step_o = c.perch ? c.step_v[ic0] : c.step;
        const float lo_o   = c.perch ? c.lo_v[ic0]   : c.lo;
#endif
        const float bs     = c.bias ? c.bias[oc] : 0.f;
#endif

        // Stage this group's weights. Output channel oc owns the contiguous slice [oc*Y26_WSTRIDE, +Y26_WSTRIDE) of Wt,
        // so this is a stride-1 scan that bursts. Y26_WSTRIDE == icpg*ktap unless Y26_WT_WORD pads it to a port word.
        const int       icpgp = Y26_WICPG(icpg);                 // padded channels per tap
        const int       wlen  = Y26_WSTRIDE(icpg, ktap);         // elements per oc slice
#ifdef Y26_OCPACK
  #if Y26_WPE > 1 && !defined(Y26_WT_TAPMAJOR)
    #error "Y26_OCPACK + wide weight port requires -DY26_WT_TAPMAJOR, for the same reason the unpacked path does: without tap-major order one port word straddles taps as well as channels, the destination bank becomes a runtime value, and the write needs exactly the Y26_LANES:1 crossbar W9 exists to avoid."
  #endif
  #if Y26_WPE == 1 && defined(Y26_WT_TAPMAJOR)
    #error "Y26_OCPACK + -DY26_WT_TAPMAJOR at the 8-bit port is NOT implemented. The packed 8-bit staging loop below walks `i` assuming CHANNEL-major order (Y26_WIDX = icl*ktap + t); under tap-major the layout is t*WICPG + icl and that loop would silently stage every weight to the wrong place. This guard converts a wrong answer into a build failure."
  #endif
        // Sequential over p: it reads the one gmem_wt port, so unrolling would not run in parallel.
        // wgrp is the group width the staging targets: Y26_ICGRP for packed dense, Y26_WPE for packed depthwise (one
        // channel per member; the word-counting loop cannot express 1), Y26_LANES when unpacked. Member p lands at bank
        // p*wgrp, which is what the MAC and kwsum read.
        const int wgrp = (pack_cap > 1) ? (g1 ? Y26_ICGRP : Y26_WPE) : Y26_LANES;
        for (int p = 0; p < pack; ++p) {
            const int       ocp   = oc + p;
            const y26_idx_t wbase = (y26_idx_t)((y26_out_t)ocp * (ap_uint<13>)wlen);
#ifdef Y26_PREFETCH
            // perch => icpg == ocpg == 1 (asserted at the prefetch), so ic0p == ocp: no divide.
            wsc_o[p]  = pf_wsc[ocp];
            bs[p]     = pf_bias[ocp];
#ifdef Y26_YQ8
            qs_o[p]  = pf_qs[ocp];                    // stale when yq == 0: never used then
            qlo_o[p] = pf_qlo[ocp];
            qst_o[p] = pf_qst[ocp];
#endif
#ifdef Y26_FUSE_ON
            step_o[p] = pf_step[ocp];   // L-FUSE: already wsc*step, filled on both arms above
            lo_o[p]   = pf_lo[ocp];     // L-FUSE: already wsc*lo
#else
            step_o[p] = c.perch ? pf_step[ocp] : c.step;
            lo_o[p]   = c.perch ? pf_lo[ocp]   : c.lo;
#endif
#else
            wsc_o[p] = c.wsc[ocp];
            bs[p]    = c.bias ? c.bias[ocp] : 0.f;
            // Per-member step/lo; without perch they are the same conv-wide scalar for every p.
            // The divide stays even under a `g1 ?` ternary (both arms are synthesized); removing it needs #if.
            const int ic0p = (ocp / ocpg) * icpg;
            step_o[p] = c.perch ? c.step_v[ic0p] : c.step;
            lo_o[p]   = c.perch ? c.lo_v[ic0p]   : c.lo;
#ifdef Y26_FUSE_ON
            // No prefetch arrays on this arm, so it folds per block (slower; not a shipping configuration).
            step_o[p] = (float)(wsc_o[p] * step_o[p]);
            lo_o[p]   = (float)(wsc_o[p] * lo_o[p]);
#endif
#endif
            const int bank_off = p * wgrp;
#if Y26_WPE > 1
            // Wide tap-major packed staging: one port word carries Y26_WPE consecutive channels of one tap. Counters are in
            // port words so the destination bank stays `(...) * Y26_WPE + b` with `b` an unroll constant (conflict-free).
            const int       wgrpw  = wgrp / Y26_WPE;        // bank ring, in words
            const int       icpgw  = icpgp / Y26_WPE;       // words per tap (icpgp is WALIGNed)
            const int       bank_offw = p * wgrpw;          // packed base, in words
            const y26_idx_t wbasew = (y26_idx_t)(wbase / Y26_WPE);
            const int       nwords = wlen / Y26_WPE;
            int wt = 0, wiclw = 0, wblw = 0, wbr = 0;
#ifdef Y26_TAPLANE
            // TAPLANE: tap wt = kh*c.kw + kw of channel cc goes to lane kh*Y26_TPCG + cc at offset kw. Tracked with
            // counters, not `wt / c.kw` (a runtime divide builds a divider core).
            int tl_kh = 0, tl_kw = 0;
            const int tl_khw = Y26_TPCG / Y26_WPE;
#endif
            for (int iw = 0; iw < nwords; ++iw) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=4 max=((Y26_MAX_DEPTH + Y26_MAX_K * Y26_MAX_K * (Y26_WPE - 1)) / Y26_WPE) avg=144
                const y26_ww_t wrd = Wt[wbasew + (y26_idx_t)iw];
                for (int b = 0; b < Y26_WPE; ++b) {
                    #pragma HLS UNROLL
                    // Signed int8 codes in an unsigned byte container: the cast through ap_int<8> is required.
#ifdef Y26_TAPLANE
                    if (y26_tl)
                        wbuf[(bank_offw + tl_kh * tl_khw + wblw) * Y26_WPE + b][tl_kw] =
                            (y26_wt_t)(ap_int<8>)wrd.range(b * 8 + 7, b * 8);
                    else
#endif
                    wbuf[(bank_offw + wblw) * Y26_WPE + b][wbr + wt] =
                        (y26_wt_t)(ap_int<8>)wrd.range(b * 8 + 7, b * 8);
                }
                ++wblw;
                ++wiclw;
                // bank ring wraps within the group; wgrp % Y26_WPE == 0 so the wrap is word-aligned
                if (wblw == wgrpw) { wblw = 0; wbr += ktap; }
                // end of this tap's channel run -> next tap, back to the base of the group
                if (wiclw == icpgw) {
                    wiclw = 0; wblw = 0; wbr = 0; ++wt;
#ifdef Y26_TAPLANE
                    if (++tl_kw == c.kw) { tl_kw = 0; ++tl_kh; }
#endif
                }
            }
#else
            int wicl = 0, wt = 0, wbl = 0, wbr = 0;
            for (int i = 0; i < wlen; ++i) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=32 max=Y26_MAX_DEPTH avg=1152
                wbuf[bank_off + wbl][wbr + wt] = Wt[wbase + (y26_idx_t)i];
                if (++wt == ktap) {
                    wt = 0;
                    ++wicl;
                    wbl = wicl % wgrp;              // bank (within group) = channel % wgrp
                    wbr = (wicl / wgrp) * ktap;     // row  = channel / wgrp
                }
            }
#endif
        }
#else
        const y26_idx_t wbase = (y26_idx_t)((y26_out_t)oc * (ap_uint<13>)wlen);
        // One flat loop: a nested `for icl { for t }` would refill the ktap-deep pipeline icpg times. Walking `i` linearly
        // keeps the Wt read stride-1 (so it bursts) while counters scatter into the lane banks at II=1.
        // The counters invert Y26_WIDX(), which the testbench uses to place each weight; they must change together.
        // wbuf's own layout ([bank][group*ktap + tap]) is unchanged; only the DRAM order moves.
#if Y26_WPE > 1
  #if !defined(Y26_WT_TAPMAJOR)
    #error "Y26_WT_WORD>8 requires -DY26_WT_TAPMAJOR. Without tap-major ordering the Y26_WPE elements of one port word straddle taps as well as channels, so the destination bank becomes a runtime value and the write needs exactly the Y26_LANES:1 crossbar W9 exists to avoid. Buildable pairs: (8-bit, either layout) or (wide, tapmajor)."
  #endif
  #if (Y26_LANES % Y26_WPE) != 0
    #error "Y26_LANES must be a multiple of Y26_WPE, or one port word's elements do not land in distinct banks."
  #endif
        // One port word carries Y26_WPE consecutive channels of one tap. `wicl` advances by Y26_WPE, so the bank index
        // is a multiple of Y26_WPE and the byte lane `b` supplies its low bits: `wbl0*Y26_WPE + b` with `b` an unroll
        // constant lets HLS prove the writes hit distinct banks (`wbl+b` would not).
        const y26_idx_t wbasew = (y26_idx_t)(wbase / Y26_WPE);
        const int       nwords = wlen / Y26_WPE;
        int wt = 0, wicl = 0, wbl0 = 0, grpk = 0;
        for (int iw = 0; iw < nwords; ++iw) {
            #pragma HLS PIPELINE II=1
            // Counts port words, so the bounds are the element bounds / Y26_WPE. csynth reports latency from max=.
            #pragma HLS LOOP_TRIPCOUNT min=1 max=((Y26_MAX_DEPTH + Y26_MAX_K * Y26_MAX_K * (Y26_WPE - 1)) / Y26_WPE) avg=144
            const y26_ww_t wrd = Wt[wbasew + (y26_idx_t)iw];
            for (int b = 0; b < Y26_WPE; ++b) {
                #pragma HLS UNROLL
                // Signed int8 codes in an unsigned byte container: the cast through ap_int<8> is required.
                wbuf[wbl0 * Y26_WPE + b][grpk + wt] =
                    (y26_wt_t)(ap_int<8>)wrd.range(b * 8 + 7, b * 8);
            }
            wicl += Y26_WPE;
            ++wbl0;
            if (wbl0 == Y26_LANES / Y26_WPE) { wbl0 = 0; grpk += ktap; }
            if (wicl >= icpgp) { wicl = 0; wbl0 = 0; grpk = 0; ++wt; }
        }
#elif defined(Y26_WT_TAPMAJOR)
        // Tap-major at the 8-bit port: channels walk fastest, so the bank advances every iteration (an increment).
        int wicl = 0, wt = 0, wbl = 0, grpk = 0;
        for (int i = 0; i < wlen; ++i) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=32 max=Y26_MAX_DEPTH avg=1152
            wbuf[wbl][grpk + wt] = Wt[wbase + (y26_idx_t)i];
            ++wicl;
            if (++wbl == Y26_LANES) { wbl = 0; grpk += ktap; }
            if (wicl == icpgp)      { wicl = 0; wbl = 0; grpk = 0; ++wt; }
        }
#else
        int wicl = 0, wt = 0, wbl = 0, wbr = 0;
        for (int i = 0; i < wlen; ++i) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=32 max=Y26_MAX_DEPTH avg=1152
            wbuf[wbl][wbr + wt] = Wt[wbase + (y26_idx_t)i];
            if (++wt == ktap) {
                wt = 0;
                ++wicl;
                wbl = wicl % Y26_LANES;              // bank  = channel % lanes  (mask, power of two)
                wbr = (wicl / Y26_LANES) * ktap;     // row   = channel / lanes  (shift)
            }
        }
#endif
#endif

        // Per-tap zero-point sums, one row per live member, from the staged banks. Lane-parallel with an adder tree:
        // a scalar walk would read a runtime bank and build a crossbar. Reassociating integer sums is exact.
#ifdef Y26_OCPACK
        // Two fixed-shape reductions selected at runtime by pack_cap, not one with a runtime width, so p*Y26_ICGRP
        // (p unrolled) stays provably inside wbuf.
#if Y26_DWP > 1
        if (icpg == 1 && pack_cap > 1) {
            // DWP: a depthwise member sums exactly one input channel, at bank p*wgrp, so the adder tree collapses to one read.
            for (int t = 0; t < ktap; ++t) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=Y26_MAX_K*Y26_MAX_K avg=9
                for (int p = 0; p < Y26_DWP; ++p) {
                    #pragma HLS UNROLL
                    kwsum_o[p][t] = (ap_int<32>)wbuf[p * Y26_WPE][t];
                }
            }
        } else
#endif
        if (pack_cap > 1) {
            // p is unrolled (matching the MAC's bank arithmetic). p >= pack computes dead values that are never read.
            const int nrows = (icpg + Y26_ICGRP - 1) / Y26_ICGRP;
#ifndef Y26_A1
            (void)nrows;
#endif
            // One flat pipeline region per tap: the p UNROLL sits inside the body, not around an outlined loop.
            // nrows is 1 here (pack_cap > 1 => icpg <= Y26_ICGRP), so there is no r loop.
#ifdef Y26_A1
            // A1: nrows can exceed 1, so the r axis is back. PIPELINE on r keeps one region entry per tap when nrows == 1.
#ifdef Y26_TAPLANE
            // Tap odometer: counters instead of t/c.kw and t%c.kw, which would build a divider.
            int tl_kh = 0, tl_kw = 0;
#endif
            for (int t = 0; t < ktap; ++t) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=Y26_MAX_K*Y26_MAX_K avg=1
                ap_int<32> sacc[Y26_ACCP];
                #pragma HLS ARRAY_PARTITION variable=sacc complete
                for (int p = 0; p < Y26_OCPACK_P; ++p) {
                    #pragma HLS UNROLL
                    sacc[p] = 0;
                }
                for (int r = 0; r < nrows; ++r) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=(Y26_MAX_IC / Y26_ICGRP + 1) avg=1
                    // Dead lanes of a partial channel block contribute 0, matching the MAC pass.
                    const int lrem = icpg - r * Y26_ICGRP;
                    const int lmax = (lrem < Y26_ICGRP) ? lrem : Y26_ICGRP;
                    for (int p = 0; p < Y26_OCPACK_P; ++p) {
                        #pragma HLS UNROLL
                        ap_int<32> s = 0;
                        for (int l = 0; l < Y26_ICGRP; ++l) {
                            #pragma HLS UNROLL
#ifdef Y26_TAPLANE
                            // TAPLANE: tap t == tl_kh*c.kw + tl_kw of channel cc lives at bank tl_kh*Y26_TPCG + cc, offset tl_kw.
                            // Every wbuf reader must use the permuted layout, not only the MAC.
                            if (y26_tl) {
                                s += (l < lmax)
                                   ? (ap_int<32>)wbuf[p * Y26_ICGRP + tl_kh * Y26_TPCG + l][tl_kw]
                                   : (ap_int<32>)0;
                            } else
#endif
                            s += (l < lmax) ? (ap_int<32>)wbuf[p * Y26_ICGRP + l][r * ktap + t]
                                            : (ap_int<32>)0;
                        }
                        sacc[p] += s;
                    }
                }
                for (int p = 0; p < Y26_OCPACK_P; ++p) {
                    #pragma HLS UNROLL
                    kwsum_o[p][t] = sacc[p];
                }
#ifdef Y26_TAPLANE
                if (++tl_kw == c.kw) { tl_kw = 0; ++tl_kh; }
#endif
            }
#else
            for (int t = 0; t < ktap; ++t) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=Y26_MAX_K*Y26_MAX_K avg=1
                for (int p = 0; p < Y26_OCPACK_P; ++p) {
                    #pragma HLS UNROLL
                    ap_int<32> s = 0;
                    for (int l = 0; l < Y26_ICGRP; ++l) {
                        #pragma HLS UNROLL
                        s += (l < icpg) ? (ap_int<32>)wbuf[p * Y26_ICGRP + l][t]
                                        : (ap_int<32>)0;
                    }
                    kwsum_o[p][t] = s;
                }
            }
#endif
#ifdef Y26_XB64
        }   // pack_cap > 1 for every supported conv
#else
        } else {
            // Not packable (icpg > Y26_ICGRP, or grouped): the single Y26_LANES-wide reduction into kwsum_o[0].
            const int nrows = (icpg + Y26_LANES - 1) / Y26_LANES;
            for (int t = 0; t < ktap; ++t) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=Y26_MAX_K*Y26_MAX_K avg=1
                ap_int<32> s = 0;
                for (int r = 0; r < nrows; ++r) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=Y26_MAX_IC avg=4
                    ap_int<32> ptmp = 0;
                    for (int l = 0; l < Y26_LANES; ++l) {
                        #pragma HLS UNROLL
                        ptmp += (r * Y26_LANES + l < icpg) ? (ap_int<32>)wbuf[l][r * ktap + t]
                                                            : (ap_int<32>)0;
                    }
                    s += ptmp;
                }
                kwsum_o[0][t] = s;
            }
        }
#endif
#else
        const int nrows = (icpg + Y26_LANES - 1) / Y26_LANES;
        for (int t = 0; t < ktap; ++t) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=Y26_MAX_K*Y26_MAX_K avg=1
            ap_int<32> s = 0;
            for (int r = 0; r < nrows; ++r) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=Y26_MAX_IC avg=4
                ap_int<32> p = 0;
                for (int l = 0; l < Y26_LANES; ++l) {
                    #pragma HLS UNROLL
                    // Dead lanes of the partial final tile contribute 0, not a stale bank value.
                    p += (r * Y26_LANES + l < icpg) ? (ap_int<32>)wbuf[l][r * ktap + t]
                                                    : (ap_int<32>)0;
                }
                s += p;
            }
            kwsum_o[t] = s;
        }
#endif

        // Row fusion: rows step in blocks, so every loop inside fills its pipeline once per block. R = live rows in
        // this block. Single-trip: the outer ohbo loop already chose the block.
        for (int ohb = ohbo; ohb < ohbo + y26_Rblk && ohb < OH; ohb += y26_Rblk) {
            const int Rrem = OH - ohb;
            const int R    = (Rrem < y26_Rblk) ? Rrem : y26_Rblk;
            // Zero accrow sequentially in one flat R*OW pass. OWP pads each row to whole Y26_EPI_WIDE groups so no group
            // straddles a row boundary.
            const int OWP = ((OW + Y26_EPI_WIDE - 1) / Y26_EPI_WIDE) * Y26_EPI_WIDE;
            // Y26_YSTRIDE_PAD: Y's DRAM row stride becomes OWP instead of OW, so every row starts Y26_YPE-aligned.
            // The host must then read Y at stride OWP.
#ifdef Y26_YSTRIDE_PAD
            const int YS = OWP;
#else
            const int YS = OW;
#endif
#ifdef Y26_OCPACK
            const bool need_zero = !g1 && icpg > 1;
#else
            const bool need_zero = !g1;
#endif
            if (need_zero) {
                int zr = 0, zw = 0;
                for (int i = 0; i < R * OWP; i += Y26_EPI_WIDE) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=(Y26_ROWS*Y26_MAX_OW) avg=64
                    for (int n = 0; n < Y26_EPI_WIDE; ++n) {
                        #pragma HLS UNROLL
                        // zw is a multiple of Y26_EPI_WIDE, so lane n is bank n: a compile-time bank, no mux.
                        if (zw + n < OW) {
#ifdef Y26_OCPACK
                            // Zero every live plane; p is unrolled and predicated on the runtime `pack`.
                            for (int p = 0; p < Y26_ACCP; ++p) {
                                #pragma HLS UNROLL
                                if (p < pack) Y26_ACCROW(p, zr, zw + n) = 0;
                            }
#else
                            Y26_ACCROW(0, zr, zw + n) = 0;
#endif
                        }
                    }
                    zw += Y26_EPI_WIDE;
                    if (zw >= OWP) { zw = 0; ++zr; }
                }
            }

            // MAC pass. Lanes run along the input channel axis: each cycle the lanes read different input channels at the
            // same position, multiply by their own weights, and an adder tree folds the products. Lane l always reads
            // bank l (a constant after UNROLL), so there is no crossbar and all lanes share one address.
            // Flattened: one pipeline entry per output row, not per tap. Out-of-image rows and columns are predicated
            // (`tok`) instead of skipped. ow is outermost and the tap innermost, so the sum is a register recurrence and
            // accrow gets one store per ow (a load-add-store on accrow would force II=2).
            // Reassociating the integer sum is exact; the dequant below is order-sensitive.
#ifdef Y26_OCPACK
#ifdef Y26_A1
            // A1: xbuf and wbuf are ICGRP-banked for every g1 conv, so every g1 conv must take this pass, even a final
            // oc group with pack == 1 (the unpacked branch reads with the Y26_LANES modulus).
            if (g1) {
#else
            if (g1 && pack > 1) {
#endif
                // OC-packed dense MAC: `pack` output channels run through the same tap walk at once. p and l are unrolled, so
                // `p*Y26_ICGRP + l` is a compile-time wbuf bank. The activation read xbuf[l][xo] does not depend on p and is
                // broadcast to all Y26_OCPACK_P multiplies.
#ifdef Y26_A1
                // A1: channel block icbi covers channels [icbi*Y26_ICGRP, +Y26_ICGRP), at xbuf offset icbi*xb_hw and wbuf row
                // icbi*ktap.
                const int npass = (icpg + Y26_ICGRP - 1) / Y26_ICGRP;
                // TAPLANE: the kh axis is on the lanes, so the walk is kw only (npass is 1 here).
                const int ntap  = y26_tl ? c.kw : (npass * ktap);
                const int total = R * OW * ntap;
#else
                const int total = R * OW * ktap;
#endif

                int        ow = 0, t = 0, rr = 0;
                int        ih0 = ohb * c.sh - c.ph - y26_y0;
                int        kh = 0, kw = 0;
#ifdef Y26_A1
                int        icbi = 0;
#endif
                ap_int<32> acc[Y26_OCPACK_P];
                #pragma HLS ARRAY_PARTITION variable=acc complete
                for (int p = 0; p < Y26_OCPACK_P; ++p) {
                    #pragma HLS UNROLL
                    acc[p] = 0;
                }

                for (int i = 0; i < total; ++i) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=16 max=(Y26_ROWS*2880) avg=1024
                    const int  ih  = ih0 + kh;
                    const int  iw  = ow * c.sw + kw - c.pw;
                    const bool tok = (ih >= 0 && ih < y26_bh && iw >= 0 && iw < W);
#ifdef Y26_A1
                    const int  woff = icbi * ktap + kh * c.kw + kw;
                    const int  lrem = icpg - icbi * Y26_ICGRP;
                    const int  lmax = (lrem < Y26_ICGRP) ? lrem : Y26_ICGRP;
                    // All lanes share one within-bank offset.
                    const y26_idx_t xo = (y26_idx_t)((ap_uint<8>)icbi * xb_hw)
                                       + (y26_idx_t)((ap_uint<11>)ih * (ap_uint<11>)W)
                                       + (y26_idx_t)iw;
#else
                    const int  woff = kh * c.kw + kw;
                    const int  lmax = icpg;
                    const y26_idx_t xo = (y26_idx_t)((ap_uint<11>)ih * (ap_uint<11>)W) + (y26_idx_t)iw;
#endif

#ifdef Y26_LPAIR
                    // L-PAIR: members p and p+1 share lane l's activation x, so one DSP48E2 computes ((w_p << 18) + w_p1) * x.
                    // |w_p1 * x| <= 32,640 < 2^17, so lo = sext(P[17:0]) = w_p1*x and hi = P[35:18] + P[17] = w_p*x, exactly.
#if Y26_OCPACK_P % 2
#error "Y26_LPAIR needs an even Y26_OCPACK_P"
#endif
                    ap_int<32> s[Y26_OCPACK_P];
                    #pragma HLS ARRAY_PARTITION variable=s complete
                    for (int p = 0; p < Y26_OCPACK_P; ++p) {
                        #pragma HLS UNROLL
                        s[p] = 0;
                    }
                    if (tok) {
                        for (int l = 0; l < Y26_ICGRP; ++l) {
                            #pragma HLS UNROLL
                            const ap_int<9> x = (ap_uint<8>)xbuf[l][xo];
                            for (int p = 0; p < Y26_OCPACK_P; p += 2) {
                                #pragma HLS UNROLL
                                const y26_wt_t wh = (l < lmax) ? wbuf[p * Y26_ICGRP + l][woff]     : (y26_wt_t)0;
                                const y26_wt_t wl = (l < lmax) ? wbuf[(p + 1) * Y26_ICGRP + l][woff] : (y26_wt_t)0;
                                const ap_int<27> a = ((ap_int<27>)wh << 18) + (ap_int<27>)wl;
                                ap_int<36> pr = a * x;
                                #pragma HLS BIND_OP variable=pr op=mul impl=dsp
                                const ap_int<18> lo = pr.range(17, 0);
                                const ap_int<18> hi = (ap_int<18>)pr.range(35, 18) + (ap_int<18>)(ap_uint<1>)pr[17];
#ifdef Y26_LPAIR_NEG
                                s[p] += lo; s[p + 1] += hi;   // negative control: split swapped
#else
                                s[p] += hi; s[p + 1] += lo;
#endif
                            }
                        }
                    }
                    for (int p = 0; p < Y26_OCPACK_P; ++p) {
                        #pragma HLS UNROLL
                        acc[p] += s[p];
                    }
#else
#ifdef Y26_TAPLANE
                    if (y26_tl) {
                        // Column validity is shared by every lane; row validity is per sub-group (lane t carries row tap t).
                        const bool colok = (iw >= 0 && iw < W);
                        y26_idx_t xot[Y26_TP];
                        bool      rowok[Y26_TP];
                        #pragma HLS ARRAY_PARTITION variable=xot complete
                        #pragma HLS ARRAY_PARTITION variable=rowok complete
                        for (int tp = 0; tp < Y26_TP; ++tp) {
                            #pragma HLS UNROLL
#ifdef Y26_TAPLANE_NEG
                            const int iht = ih0 - tp;   // negative control: row shift reversed
#else
                            const int iht = ih0 + tp;
#endif
                            rowok[tp] = colok && tp < c.kh && iht >= 0 && iht < y26_bh;
                            // The address forms only for a row that is used, so an out-of-band tap never indexes xbuf out of range.
                            const y26_idx_t xa = (y26_idx_t)((ap_uint<11>)iht * (ap_uint<11>)W)
                                               + (y26_idx_t)iw;
                            xot[tp]   = rowok[tp] ? xa : (y26_idx_t)0;
                        }
                        for (int p = 0; p < Y26_OCPACK_P; ++p) {
                            #pragma HLS UNROLL
                            ap_int<32> s = 0;
                            for (int tp = 0; tp < Y26_TP; ++tp) {
                                #pragma HLS UNROLL
                                for (int cc = 0; cc < Y26_TPCG; ++cc) {
                                    #pragma HLS UNROLL
                                    const int l = tp * Y26_TPCG + cc;
                                    // The weight offset is the column tap alone; staging already put row tap tp in this bank.
                                    const y26_wt_t w = (rowok[tp] && cc < icpg)
                                                     ? wbuf[p * Y26_ICGRP + l][kw] : (y26_wt_t)0;
                                    s += (ap_int<32>)(xbuf[l][xot[tp]] * w);
                                }
                            }
                            acc[p] += s;
                        }
                    } else
#endif
                    for (int p = 0; p < Y26_OCPACK_P; ++p) {
                        #pragma HLS UNROLL
                        ap_int<32> s = 0;
                        if (tok) {
                            for (int l = 0; l < Y26_ICGRP; ++l) {
                                #pragma HLS UNROLL
                                // Dead lanes of a partial block contribute 0, not a stale bank value.
                                const y26_wt_t w = (l < lmax) ? wbuf[p * Y26_ICGRP + l][woff]
                                                               : (y26_wt_t)0;
                                s += (ap_int<32>)(xbuf[l][xo] * w);
                            }
                        }
                        acc[p] += s;
                    }
#endif

#ifdef Y26_A1
                    // Odometer: kw fastest, then kh, then the channel block. ow advances when a pixel's taps are spent.
                    if (++t == ntap) {
                        for (int p = 0; p < Y26_OCPACK_P; ++p) {
                            #pragma HLS UNROLL
                            Y26_ACCROW(p, rr, ow) = acc[p];
                            acc[p] = 0;
                        }
                        t = 0; icbi = 0; kh = 0; kw = 0;
                        if (++ow == OW) { ow = 0; ++rr; ih0 += c.sh; }
                    } else if (++kw == c.kw) {
                        // TAPLANE never gets here: ntap == c.kw, so kw wraps exactly when t does.
                        kw = 0;
                        if (++kh == c.kh) { kh = 0; ++icbi; }
                    }
#else
                    if (++t == ktap) {
                        for (int p = 0; p < Y26_OCPACK_P; ++p) {
                            #pragma HLS UNROLL
                            Y26_ACCROW(p, rr, ow) = acc[p];
                            acc[p] = 0;
                        }
                        t = 0; kh = 0; kw = 0;
                        if (++ow == OW) { ow = 0; ++rr; ih0 += c.sh; }
                    } else if (++kw == c.kw) {
                        kw = 0;
                        ++kh;
                    }
#endif
                }
            } else
#endif
#ifndef Y26_XB64   // XB64: every g1 conv takes the packed MAC above, so this arm is unused
            if (g1) {
                // groups == 1  =>  ic0 == 0  =>  bank(lane l) == l, statically provable.
                const int npass = (icpg + Y26_LANES - 1) / Y26_LANES;
                const int ntap  = npass * ktap;          // taps accumulated per output pixel
                // R rows in one flat loop: the pipeline fills once per block.
                const int total = R * OW * ntap;

                int        ow = 0, t = 0, rr = 0;
                int        ih0 = ohb * c.sh - c.ph - y26_y0;       // advances by sh as rr advances
                int        icbi = 0, kh = 0, kw = 0;
                ap_int<32> acc = 0;

                for (int i = 0; i < total; ++i) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=16 max=(Y26_ROWS*20736) avg=1024
                    const int  ih  = ih0 + kh;
                    // Offset stays SIGNED until known in range: kw - pw can be negative, so the
                    // bounds test must run on the signed value before any cast to y26_idx_t.
                    const int  iw  = ow * c.sw + kw - c.pw;
                    const bool tok = (ih >= 0 && ih < y26_bh && iw >= 0 && iw < W);

                    ap_int<32> s = 0;
                    if (tok) {
                        const int woff = icbi * ktap + kh * c.kw + kw;
                        const int lrem = icpg - icbi * Y26_LANES;
                        const int lmax = (lrem < Y26_LANES) ? lrem : Y26_LANES;
                        // All lanes share ONE within-bank offset - they differ only by bank - so
                        // this is a single address computation, not Y26_LANES of them.
                        const y26_idx_t xo = (y26_idx_t)((ap_uint<8>)icbi * xb_hw)
                                           + (y26_idx_t)((ap_uint<11>)ih * (ap_uint<11>)W)
                                           + (y26_idx_t)iw;
                        for (int l = 0; l < Y26_LANES; ++l) {
                            #pragma HLS UNROLL
                            // Native-width multiply: ap_uint<8> * ap_int<8> is an exact ap_int<16> and maps to a DSP. Only the accumulation
                            // is 32-bit; widening the operands would build a 32x32 fabric multiplier.
                            const y26_wt_t w = (l < lmax) ? wbuf[l][woff] : (y26_wt_t)0;
                            s += (ap_int<32>)(xbuf[l][xo] * w);
                        }
                    }
                    acc += s;

                    // Odometer: kw fastest, then kh, then the channel block. ow advances when a pixel's taps are spent.
                    if (++t == ntap) {
                        Y26_ACCROW(0, rr, ow) = acc;  // pure store - this is the row's ONLY write to [ow]
                        acc  = 0;
                        t    = 0;
                        icbi = 0; kh = 0; kw = 0;
                        // ow wrapping advances the row; ih0 steps by sh so the recurrence has no multiply.
                        if (++ow == OW) { ow = 0; ++rr; ih0 += c.sh; }
                    } else if (++kw == c.kw) {
                        kw = 0;
                        if (++kh == c.kh) { kh = 0; ++icbi; }
                    }
                }
            } else
#endif
#ifdef Y26_OCPACK
            if (icpg == 1) {
                // Depthwise, fused and packed (pack_cap > 1 on a non-g1 conv means icpg == 1). Same odometer as the dense
                // packed pass, but each member has exactly one input channel, so the lane reduction is a single multiply.
                // One pipeline entry per (oc-group, row-block) instead of R*ktap.
                const int total = R * OW * ktap;
                int ow = 0, t = 0, rr = 0, kh = 0, kw = 0;
                int ih0 = ohb * c.sh - c.ph - y26_y0;
                ap_int<32> acc[Y26_DWP];
                #pragma HLS ARRAY_PARTITION variable=acc complete
                for (int p = 0; p < Y26_DWP; ++p) {
                    #pragma HLS UNROLL
                    acc[p] = 0;
                }
                for (int i = 0; i < total; ++i) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=16 max=(Y26_ROWS*2880) avg=1024
                    const int  ih   = ih0 + kh;
                    const int  iw   = ow * c.sw + kw - c.pw;
                    const bool tok  = (ih >= 0 && ih < y26_bh && iw >= 0 && iw < W);
                    const int  woff = kh * c.kw + kw;
                    const y26_idx_t rowo = (y26_idx_t)((ap_uint<11>)ih * (ap_uint<11>)W)
                                         + (y26_idx_t)iw;
#if Y26_DWP > 1
                    // DWP: activations are banked at modulus Y26_DWP and ic0 is a multiple of it, so member p's channel is at bank
                    // p (a compile-time constant) and all members share one address. A dead member (p >= pack) accumulates
                    // values the epilogue never reads.
                    const y26_idx_t xo = (y26_idx_t)((ap_uint<8>)(ic0 / Y26_DWP) * xb_hw) + rowo;
                    for (int p = 0; p < Y26_DWP; ++p) {
                        #pragma HLS UNROLL
                        // Member p's taps are at bank p*Y26_WPE (the staging loop's bank_off).
                        const y26_wt_t w = wbuf[p * Y26_WPE][woff];
                        if (tok) acc[p] += (ap_int<32>)(xbuf[p][xo] * w);
                    }
#else
                    for (int p = 0; p < 1; ++p) {
                        #pragma HLS UNROLL
                        // ocpg == 1, so member p's channel is ic0 + p. The clamp keeps a dead member of a short final group inside
                        // the staged rows; its accumulator is never read.
                        const int       ic  = ic0 + ((p < pack) ? p : 0);
                        const int       bnk = ic & (Y26_LANES - 1);
                        const y26_idx_t xo  = (y26_idx_t)((ap_uint<8>)(ic / Y26_LANES) * xb_hw) + rowo;
                        // icpg == 1: this member's only lane is at the base of its bank group.
                        const y26_wt_t  w   = wbuf[p * Y26_ICGRP][woff];
                        if (tok) acc[p] += (ap_int<32>)(xbuf[bnk][xo] * w);
                    }
#endif
                    // Odometer: kw fastest, then kh. The row slot is written when the pixel's taps are spent.
                    if (++t == ktap) {
                        for (int p = 0; p < Y26_DWP; ++p) {
                            #pragma HLS UNROLL
                            Y26_ACCROW(p, rr, ow) = acc[p];   // pure store, so no zero pass needed
                            acc[p] = 0;
                        }
                        t = 0; kh = 0; kw = 0;
                        if (++ow == OW) { ow = 0; ++rr; ih0 += c.sh; }
                    } else if (++kw == c.kw) {
                        kw = 0;
                        ++kh;
                    }
                }
            } else
#endif
#ifdef Y26_XB64
            { }   // XB64: grouped icpg > 1 is unsupported
#else
            // Grouped fallback, not fused: a plain per-row loop. Correct for any grouped conv, not fast.
            for (int rr = 0; rr < R; ++rr) {
            const int ih0 = (ohb + rr) * c.sh - c.ph - y26_y0;
            for (int icb = 0; icb < icpg; icb += Y26_LANES) {
                for (int kh = 0; kh < c.kh; ++kh) {
                    const int ih = ih0 + kh;
                    if (ih < 0 || ih >= y26_bh) continue;
                    const y26_idx_t rowoff = (y26_idx_t)((ap_uint<11>)ih * (ap_uint<11>)W);
                    for (int kw = 0; kw < c.kw; ++kw) {
                        const int base = kw - c.pw;
                        int o0, o1;
                        y26_ow_range(base, c.sw, W, OW, o0, o1);

                        // Per-lane weights for this (channel group, tap), hoisted into registers from wbuf so the MAC reads only xbuf.
                        // Native int8 width (a 32-bit operand would build fabric multipliers). Lane l reads bank l at a shared offset,
                        // so the whole tile loads in one cycle; do not reintroduce a loop here.
                        y26_wt_t wl[Y26_LANES];
                        #pragma HLS ARRAY_PARTITION variable=wl complete

                        // Dead lanes read 0, not the previous tile's weights, so the MAC can run unpredicated over all Y26_LANES.
                        const int lrem = icpg - icb;
                        const int lmax = (lrem < Y26_LANES) ? lrem : Y26_LANES;
                        const int woff = (icb / Y26_LANES) * ktap + kh * c.kw + kw;
                        for (int l = 0; l < Y26_LANES; ++l) {
                            #pragma HLS UNROLL
                            wl[l] = (l < lmax) ? wbuf[l][woff] : (y26_wt_t)0;
                        }

                        {
                            // Grouped. Every grouped conv in this model is depthwise (icpg == 1), so this runs once. Bounded by lmax, not
                            // Y26_LANES: a guarded dead iteration still costs cycles in the generated FSM.
                            for (int l = 0; l < lmax; ++l) {
                                #pragma HLS LOOP_TRIPCOUNT min=1 max=Y26_LANES avg=1
                                const int       icl  = icb + l;
                                const int       ic   = ic0 + icl;
                                const int       bnk  = ic & (Y26_LANES - 1);
                                const y26_idx_t xrow =
                                    (y26_idx_t)((ap_uint<8>)(ic / Y26_LANES) * xb_hw) + rowoff;
#ifdef Y26_C5DW
                                // Depthwise: Y26_EPI_WIDE output columns per cycle. accrow is cyclic-partitioned by Y26_EPI_WIDE on the column
                                // axis, so consecutive ow hit distinct banks, and each ow is its own accumulator. Strip-mined by hand rather
                                // than UNROLL on the pipelined loop, whose precedence would be left to an HLS heuristic. `owu < o1` handles
                                // the tail; the index set is exactly [o0, o1).
                                for (int ow = o0; ow < o1; ow += Y26_EPI_WIDE) {
                                    #pragma HLS PIPELINE II=1
                                    for (int u = 0; u < Y26_EPI_WIDE; ++u) {
                                        #pragma HLS UNROLL
                                        const int owu = ow + u;
                                        if (owu < o1) {
                                            const y26_idx_t xo =
                                                xrow + (y26_idx_t)(owu * c.sw + base);
                                            Y26_ACCROW(0, rr, owu) +=
                                                (ap_int<32>)(xbuf[bnk][xo] * wl[l]);
                                        }
                                    }
                                }
#else
                                for (int ow = o0; ow < o1; ++ow) {
                                    #pragma HLS PIPELINE II=1
                                    const y26_idx_t xo = xrow + (y26_idx_t)(ow * c.sw + base);
                                    // Same native-width multiply as the g1 branch.
                                    Y26_ACCROW(0, rr, ow) += (ap_int<32>)(xbuf[bnk][xo] * wl[l]);
                                }
#endif
                            }
                        }
                    }
                }
            }
            }   // rr - grouped fallback
#endif

            // Dequant + activation: real = wsc * (step * Sum(w_int*q) + lo * Sum_valid(w_int)) + bias.
            // Summing only valid taps makes padding contribute 0.0, not lo. The promotions match conv2d() exactly
            // (float*double -> double, one cast to float, then += bias); reordering changes the result.
            // Fused over the whole block; a channel's output rows are contiguous in Y. With Y26_YPE > 1 the epilogue writes
            // a block-local buffer that a separate loop packs out. Under packing, this block runs once per real member.
#if defined(Y26_EFLAT_ON)
            // L-EFLAT: one pipeline over (member, row, group), so the per-member prologue and drain are paid once per group.
            // The m_axi_gmem_out write must still infer a burst (its address jumps by OH*YS/Y26_YPE per member); check the
            // csynth burst table. ywbase is stepped by an add, never recomputed with a multiply.
            const int y26_NI = (R * OWP) / Y26_EPI_WIDE;   // store iterations per member
            int p = 0, y26_ii = 0;
            int er = 0, ow = 0, gi = 0;
            y26_idx_t ybase = 0;
            y26_idx_t ywbase = (y26_idx_t)(((y26_out_t)(oc * OH + ohb) * (ap_uint<11>)YS) / Y26_YPE);
            const y26_idx_t y26_YWSTEP = (y26_idx_t)(((y26_out_t)OH * (ap_uint<11>)YS) / Y26_YPE);
            for (int q = 0; q < pack * y26_NI; ++q) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=(Y26_OCPACK_P*Y26_ROWS*Y26_MAX_OW) avg=1024
                const int   ocp   = oc + p;
                const float wsc_op = wsc_o[p];
                const float bs_p   = bs[p];
                const float step_op = step_o[p];
                const float lo_op   = lo_o[p];
#elif defined(Y26_OCPACK)
            for (int p = 0; p < pack; ++p) {
                const int   ocp   = oc + p;
                const float wsc_op = wsc_o[p];
                const float bs_p   = bs[p];
                const float step_op = step_o[p];      // per member
                const float lo_op   = lo_o[p];
#ifdef Y26_YQ8
                const float qs_op  = qs_o[p];
                const float qlo_op = qlo_o[p];
                const float qst_op = qst_o[p];
#endif
#else
            {
                const int   p     = 0;
                const int   ocp   = oc;
                const float wsc_op = wsc_o;
                const float bs_p   = bs;
                const float step_op = step_o;
                const float lo_op   = lo_o;
#endif
#ifndef Y26_EFLAT_ON
            const y26_idx_t yblk = (y26_idx_t)((y26_out_t)(ocp * OH + ohb) * (ap_uint<11>)YS);
#endif
#if Y26_YPE > 1
            // ybuf[group][lane] with dim 2 complete-partitioned makes lane n bank n syntactically. A flat cyclic-partitioned
            // array indexed by runtime ybase + ow + n cannot be proven conflict-free and drops to II=4.
            // The group index is the epilogue's iteration counter, valid while Y26_YBANK == Y26_EPI_WIDE.
#ifndef Y26_YDIRECT_ON
            static float ybuf[Y26_YBUF_GRP][Y26_YBANK];
            #pragma HLS ARRAY_PARTITION variable=ybuf complete dim=2
#endif
#else
            float* const yr = &Y[yblk];
#endif
            // Same OWP group structure as the zeroing pass. `ybase` tracks er*OW without a multiply. Y26_YOUT names the
            // destination element for all dequant variants; `gi` is the group counter.
#ifdef Y26_YQ8
            // yq: the slot carries the consumer's uint8 code instead of the float.
            #define Y26_YQ(v_) (c.yq ? y26_q8_slot((v_), qs_op, qlo_op, qst_op) : (v_))
#else
            #define Y26_YQ(v_) (v_)
#endif
#if defined(Y26_YDIRECT_ON)
            #define Y26_YOUT(n_, owl_) wlane[(n_)]
#elif Y26_YPE > 1
            #define Y26_YOUT(n_, owl_) ybuf[gi][(n_)]
#else
            #define Y26_YOUT(n_, owl_) yr[ybase + (owl_)]
#endif
#ifndef Y26_EFLAT_ON
            int er = 0, ow = 0, gi = 0;
            y26_idx_t ybase = 0;
#ifdef Y26_YDIRECT_ON
            const y26_idx_t ywbase = (y26_idx_t)(yblk / Y26_YPE);
#endif
            for (int i = 0; i < R * OWP; i += Y26_EPI_WIDE) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=(Y26_ROWS*Y26_MAX_OW) avg=64
#endif
#ifdef Y26_YDIRECT_ON
                float wlane[Y26_YPE];
                #pragma HLS ARRAY_PARTITION variable=wlane complete
                for (int n = 0; n < Y26_YPE; ++n) {
                    #pragma HLS UNROLL
                    wlane[n] = 0.0f;
                }
#endif
                // Per-column-tap sum over the kh's live for this output row. All lanes share `er`, so this is computed once per
                // iteration and unrolls into <= 9 predicated adds.
                ap_int<32> zpcs[Y26_MAX_K];
                #pragma HLS ARRAY_PARTITION variable=zpcs complete
                {
                    const int ih0e = (ohb + er) * c.sh - c.ph;
                    for (int kwi = 0; kwi < Y26_MAX_K; ++kwi) {
                        #pragma HLS UNROLL
                        ap_int<32> s = 0;
                        for (int khi = 0; khi < Y26_MAX_K; ++khi) {
                            #pragma HLS UNROLL
                            const int ihe = ih0e + khi;
                            if (khi < c.kh && kwi < c.kw && ihe >= 0 && ihe < H)
                                s += Y26_KWSUM(p, khi * c.kw + kwi);
                        }
                        zpcs[kwi] = s;
                    }
                }
              for (int n = 0; n < Y26_EPI_WIDE; ++n) {
                #pragma HLS UNROLL
                if (ow + n >= OW) continue;   // tail lanes of the padded group
                const int owl = ow + n;      // bank n, statically
                // This lane's zero-point sum: <= 3 predicated adds of the per-column-tap sums.
                ap_int<32> zpw = 0;
                for (int kwi = 0; kwi < Y26_MAX_K; ++kwi) {
                    #pragma HLS UNROLL
                    if (owl >= zpo0[kwi] && owl < zpo1[kwi]) zpw += zpcs[kwi];
                }
#ifdef Y26_FX_DEQUANT
                // Fixed-point dequant (not bit-exact). The accumulators stay integer and are consumed by a multiply, narrowed
                // once at the assignment.
#ifdef Y26_FUSE_ON
                // L-FUSE: step_op / lo_op are already wsc*step and wsc*lo, so the wsc multiply is gone. Accumulators are
                // narrowed to the 27-bit DSP port; the assert checks the bound in simulation.
                const y26_kscale_t s_k1 = step_op;
                const y26_kscale_t s_k2 = lo_op;
                const ap_int<Y26_KACC_W> a_k = (ap_int<Y26_KACC_W>)Y26_ACCROW(p, er, owl);
                const ap_int<Y26_KACC_W> z_k = (ap_int<Y26_KACC_W>)zpw;
                assert(a_k == Y26_ACCROW(p, er, owl) && z_k == zpw);   // 27-bit port bound
                y26_fx_t v = (y26_fx_t)(s_k1 * a_k) + (y26_fx_t)(s_k2 * z_k) + (y26_fx_t)bs_p;
#else
                const y26_scale_t s_step = step_op;
                const y26_scale_t s_lo   = lo_op;
                const y26_scale_t s_wsc  = wsc_op;
                const y26_fx_t t = (y26_fx_t)(s_step * Y26_ACCROW(p, er, owl))
                                 + (y26_fx_t)(s_lo   * zpw);
                y26_fx_t v = (y26_fx_t)(s_wsc * t) + (y26_fx_t)bs_p;
#endif
#ifdef Y26_SILU_LUT
                // Fully fixed-point: no float/double core for the activation.
                if (c.act == 1) v = y26_silu_fx(v);
                Y26_YOUT(n, owl) = Y26_YQ(v.to_float());
#else
                float vf = v.to_float();
                if (c.act == 1) vf = y26_silu(vf);   // still double std::exp - see y26_silu_fx
                Y26_YOUT(n, owl) = Y26_YQ(vf);
#endif
#else
                // Float dequant (default): bit-exact vs conv2d(). Promotion order matches conv2d() exactly.
                const double a  = (double)Y26_ACCROW(p, er, owl).to_int64();   // exact: accumulator is integral
                const double aw = (double)zpw.to_int64();
                float v = (float)(wsc_op * (step_op * a + lo_op * aw)) + bs_p;
                if (c.act == 1) v = y26_silu(v);
                Y26_YOUT(n, owl) = Y26_YQ(v);
#endif
              }
#ifdef Y26_YDIRECT_ON
                {
                    y26_yw_t word = 0;
                    for (int n = 0; n < Y26_YPE; ++n) {
                        #pragma HLS UNROLL
                        y26_fp32 cvt;
                        cvt.f = wlane[n];
                        word.range(n * 32 + 31, n * 32) = cvt.u;
                    }
                    Y[ywbase + gi] = word;   // gi == the store loop's `w`
                }
#endif
                ow += Y26_EPI_WIDE;
                ++gi;
                if (ow >= OWP) { ow = 0; ++er; ybase += (y26_idx_t)OW; }
#ifdef Y26_EFLAT_ON
                // Retire a member: reset the row/group digits and step the store base by one output map.
                if (++y26_ii == y26_NI) {
                    y26_ii = 0; ++p;
                    er = 0; ow = 0; gi = 0; ybase = 0;
                    ywbase += y26_YWSTEP;
                }
#endif
            }
            #undef Y26_YOUT
            #undef Y26_YQ
#if Y26_YPE > 1 && !defined(Y26_YDIRECT_ON)
            // Store: one m_axi beat per Y26_YPE outputs; this loop sets store throughput. A beat must be Y26_YPE-aligned in
            // Y, whose rows are OW apart (OW = 20 is not a multiple of 8, so Y26_YPE is 4 unless Y26_YSTRIDE_PAD).
            // The odometer walks output elements (stride OW) while indexing ybuf in padded units (stride OWP), so `pb` is
            // carried separately. All lanes of a beat live in one ybuf group; padded tail columns are never read.
            {
                const y26_idx_t wbase = (y26_idx_t)(yblk / Y26_YPE);
                const int       nw    = (R * YS) / Y26_YPE;
                int sc = 0, pb = 0;
                for (int w = 0; w < nw; ++w) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=((Y26_ROWS*Y26_MAX_OW)/Y26_YPE) avg=64
                    const int p   = pb + sc;
                    const int grp = p / Y26_YBANK;
                    const int lan = p % Y26_YBANK;
                    y26_yw_t word = 0;
                    for (int n = 0; n < Y26_YPE; ++n) {
                        #pragma HLS UNROLL
                        y26_fp32 cvt;
                        cvt.f = ybuf[grp][lan + n];
                        word.range(n * 32 + 31, n * 32) = cvt.u;
                    }
                    Y[wbase + w] = word;
                    sc += Y26_YPE;
                    if (sc >= YS) { sc = 0; pb += OWP; }
                }
            }
#endif
#ifndef Y26_EFLAT_ON
            }   // p: per-member epilogue + store
#endif
        }
        oc += pack;
    }
    }   // ohbo
#undef Y26_ACCROW
#undef Y26_AFB
}
