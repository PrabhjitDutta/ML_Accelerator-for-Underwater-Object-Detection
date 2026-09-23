// conv_engine.cpp - stage 1 of the synthesizable trunk datapath: integer accumulator.
//
// Mirrors layer_ops.h's conv2d() SmoothQuant (asymmetric uint8) path exactly, with two changes:
//   * the two accumulators become ap_int<32> instead of double
//   * no STL / heap / throw; static-bounded local arrays only
//
// Everything else - loop order, tap ranges, the zero-point weight-sum, the dequant expression and
// its exact float/double promotions - is preserved verbatim, because the validation gate is EXACT
// float equality against conv2d(). Any deviation shows up as a mismatch rather than as "close
// enough", which is the entire point of staging the ap_fixed dequant separately.
//
// Why integer accumulation is safe to reorder (and float accumulation is not): these are int8/uint8
// CODES, so every partial sum is an exact integer and the reduction is associative. That is what
// licenses the MAC-lane parallelism in design target 1. Do NOT extend the same reasoning to the
// dequant path.
#include "conv_engine.h"
#include <cmath>
#include <cassert>

// ---------------------------------------------------------------------------------------------
// Local copies of the two conv2d helpers. Deliberately duplicated rather than shared: layer_ops.h
// drags in <vector>/<string>/weights_loader.h, none of which may cross into a synthesizable TU.
// These are byte-for-byte the same arithmetic.
// ---------------------------------------------------------------------------------------------

// Output-column range [o0,o1) over which tap column `base` lands inside the image. Expressing the
// zero padding as a loop bound instead of a per-element predicate keeps the branch out of the
// datapath - it is also what makes the innermost loop a clean II=1 candidate.
static inline void y26_ow_range(int base, int sw, int W, int OW, int& o0, int& o1) {
    // ponytail: this pragma is load-bearing and the ceiling is the HLS inline heuristic. Measured
    // 2026-09-05 (runs A and A', pwr/SCORE_RUNA.txt): adding two compares and two muxes here was
    // enough to push this function OVER that threshold, and HLS silently OUTLINED it into a
    // 71-cycle, II=71, NON-PIPELINED submodule called from :1206 -- inside the `rr / icb / kh / kw`
    // MAC nest. Nothing in the source said so; only the module hierarchy did. Keep the pragma, and
    // if this function ever grows again, check `+ y26_ow_range` is absent from csynth.rpt.
#pragma HLS INLINE
    // C8a was tried here and REVERTED, run A' (pwr/SCORE_RUNA2.txt). Writing the sw == 1 cases as
    // `sw == 1 ? -base : (-base + sw - 1) / sw` looked like it removed 6 of 17 divider cores. It
    // removed NONE: run A had OUTLINED this function, so its four inlined copies collapsed to one
    // shared copy, and 8 dividers became 2 for that reason alone. With the inline restored the
    // count is 17 again -- sdiv_ln40 x4 + sdiv_ln42 x4, structurally identical to the baseline --
    // and the muxes cost +494 LUT for nothing. A runtime select does not remove a static schedule's
    // divide. Measure the count with the CALL SITES HELD FIXED, or you are measuring inlining.
    // C8b (2026-09-05): the `/ sw` pair becomes a SHIFT. Unlike C8a there is no `/` left in the
    // expression, so HLS cannot emit a divider core -- a runtime select would not have removed
    // one (run A', pwr/SCORE_RUNA2.txt). PRECONDITION sw in {1,2}, verified over all 126 convs
    // and asserted ONCE at conv entry, not here: this function's inline is load-bearing.
    // Equivalence brute-forced by notes/frame-thirds-scripts/test_c8b.py (the dead base >= 0
    // arm disagrees -- trunc vs floor -- but the ternary discards it; see plan 0.8ap).
    const int c8b_sh = sw >> 1;            // sw==1 -> 0, sw==2 -> 1
    o0 = base >= 0 ? 0 : (-base + sw - 1) >> c8b_sh;
    o1 = W - 1 - base;
    o1 = o1 < 0 ? 0 : (o1 >> c8b_sh) + 1;
    if (o1 > OW) o1 = OW;
}

// SiLU in double, rounded to float - identical to layer_ops.h's silu(), which is the closest
// portable match to torch's float32 SiLU.
// NOTE for synthesis: std::exp on double is legal in HLS but expensive. Replacing this with a
// fixed-point LUT is a stage-2/3 decision and MUST be gated on mAP delta, not on the bit-exact gate.
static inline float y26_silu(float v) { double d = v; return (float)(d / (1.0 + std::exp(-d))); }

#ifdef Y26_SILU_LUT
#include "silu_lut.h"
// Fixed-point SiLU by table lookup + linear interpolation. Added 2026-08-17.
//
// What it replaces: `std::exp` on double, which csynth turned into a dexp core (39 cycles) feeding a
// ddiv core (59 cycles), plus the dadd/dmul/sitodp/fptrunc conversion chain around them. That is the
// bulk of the dequant loop's depth-181 pipeline, in a loop that executes once per OUTPUT PIXEL.
//
// Only reachable from the Y26_FX_DEQUANT branch, and deliberately so: the stage-1 build keeps the
// float SiLU because its gate is EXACT equality against conv2d(). Letting the LUT into that path
// would convert a bit-exact regression guard into a tolerance check, which is the one thing the
// two-stage split exists to prevent.
//
// The table stores RAW fixed-point integers with Y26_SILU_FB == Y26_FX_I fractional bits, so an entry
// drops straight into `.V` - no scaling multiply, no rounding, no conversion core.
static inline y26_fx_t y26_silu_fx(y26_fx_t v) {
    // Outside +/-16 the table is unnecessary: SiLU(x) - x and SiLU(-x) are both < 2e-6 there, which is
    // ~40x smaller than the interpolation error inside the range. Clamping is exact enough AND it is
    // what bounds the table index, so these two tests are load-bearing, not just an optimization.
    if (v >= (y26_fx_t)Y26_SILU_HI) return v;
    if (v <= (y26_fx_t)Y26_SILU_LO) return (y26_fx_t)0;

    // Position in table units: (v - LO) * (N / (HI - LO)) = (v + 16) * 32, so the scale is a shift.
    const ap_fixed<40,12> t = (ap_fixed<40,12>)((v - (y26_fx_t)Y26_SILU_LO) * (y26_fx_t)32);
    const int             i = t.to_int();                       // t > 0, so truncation == floor
    // Defensive clamp. The early returns above already guarantee 0 <= i <= N-1; this costs two
    // comparators and turns any future edit to the range constants into a wrong ANSWER rather than an
    // out-of-range ROM read.
    const int ii = (i < 0) ? 0 : ((i > Y26_SILU_N - 1) ? (Y26_SILU_N - 1) : i);

    const ap_ufixed<16,0> f = (ap_ufixed<16,0>)(t - (ap_fixed<40,12>)ii);
    const ap_int<32>      a = Y26_SILU_TAB[ii];
    const ap_int<32>      b = Y26_SILU_TAB[ii + 1];
    const ap_int<33>      d = (ap_int<33>)b - (ap_int<33>)a;

    // Bit-reinterpret rather than rescale. Assigning through `.V` is ambiguous (ap_private has ~15
    // assignment overloads and ap_int<48> matches none of them uniquely); multiplying by 2^-24 would
    // be unambiguous but would infer a 48x48 multiplier for what is a pure relabelling of bits.
    // range(hi,lo) returns an assignable proxy and costs nothing.
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
    // C8b precondition, checked once per conv rather than per tap. Compiled out of synthesis.
    assert((c.sw == 1 || c.sw == 2) && "C8b: y26_ow_range substitutes a shift for / sw");
    const int icpg = c.ic / c.groups;      // input channels per group
    const int ocpg = c.oc / c.groups;      // output channels per group
    const int ktap = c.kh * c.kw;

    // -----------------------------------------------------------------------------------------
    // ACTIVATION STAGING - the fix for the 2026-08-06 II=16 finding.
    //
    // The MAC lanes cannot be fed through the single m_axi port on X; csynth serialized them to one
    // read per cycle. So the input is copied ONCE per conv into a banked on-chip buffer, and the MAC
    // loop reads that instead. This is design target 3's "hold activations whole in BRAM" made real.
    //
    // Staging whole rather than per-row is what makes the copy cheap. The copy is O(ic*H*W); the MACs
    // are O(oc*OH*OW*icpg*kh*kw). Staging per (icl,kh) instead would have been catastrophic for 1x1
    // convs - there the copy is O(W) against O(W/lanes) of MAC work, i.e. the copy would dominate by
    // the lane count. Hoisting it out of the oc loop amortizes it over every output channel.
    // -----------------------------------------------------------------------------------------
    // Layout (see conv_engine.h): channel c -> bank c % Y26_LANES, offset (c/Y26_LANES)*H*W + y*W + x.
    // `complete dim=1` makes the LANE axis a set of independent memories whose index is the literal
    // loop variable after UNROLL - that, and not the mapping arithmetic, is what buys II=1.
#ifdef Y26_XB64
    // XB64 (ledger ng/nh): with A1 + OCPACK + DWP, dense banks at ICGRP and depthwise at DWP, so banks
    // [ICGRP, LANES) were reachable only by grouped icpg > 1 convs - none in this model. Depth unchanged.
  #if !(defined(Y26_A1) && defined(Y26_OCPACK) && Y26_DWP > 1 && Y26_DWP <= Y26_ICGRP)
    #error "Y26_XB64 needs Y26_A1, Y26_OCPACK and 1 < Y26_DWP <= Y26_ICGRP"
  #endif
    static y26_act_t xbuf[Y26_ICGRP][Y26_ACT_LANE_ELEMS];
#else
    static y26_act_t xbuf[Y26_LANES][Y26_ACT_LANE_ELEMS];
#endif
    #pragma HLS ARRAY_PARTITION variable=xbuf complete dim=1
    // GATE G0 (plan 4.4): 1t proved a wide gmem_act port ELIMINATES the staging DRAM exposure
    // (256 -> ~1), but 1v measured the bundle split paying ~0.898 cycles per staging element in
    // INTERCEPT, which put the crossover at D~96 and killed it. The suspected mechanism is a wide
    // port feeding an 8-bit-wide bank one byte per iteration. RESHAPE packs 8 consecutive elements
    // of the offset axis into one 64-bit word, so the bank is physically wide while every
    // xbuf[l][i] in this file - including both MAC read sites - stays source-identical.
    // MEASURED 2026-08-19 (notes 1x): this knob delivers NOTHING as things stand, and the reason
    // matters. m_axi_gmem_act is an 8-BIT port - "Data Width 8 -> 8", Widen Fail 214-307, "could not
    // widen since type i8 size is >= alignment 1(bytes)" - so the staging loop is fed one byte per
    // beat no matter how wide this bank is. RESHAPE alone: bit-identical, +8,427 LUT. RESHAPE plus
    // the UNROLL below: staging loop II 1 -> 8, exactly cancelling the unroll, +58,000 LUT.
    // KEEP THE KNOB: it becomes correct the moment X is declared as a wide word type (a 64-bit port
    // wants a 64-bit bank). It is the consequence half of W5, not the mechanism. Do not enable it
    // before the interface table shows gmem_act Data Width > 8.
#if defined(Y26_XBUF_WIDE) && (Y26_XBUF_WIDE > 1)
    #pragma HLS ARRAY_RESHAPE variable=xbuf cyclic factor=Y26_XBUF_WIDE dim=2
#endif

    const ap_uint<19> hw = (ap_uint<19>)((ap_uint<11>)H * (ap_uint<11>)W);
    // groups == 1 implies ic0 == 0 for every output channel, which is precisely the condition that
    // makes bank(lane l) == l a COMPILE-TIME fact in the MAC pass below. It covers 88 of the model's
    // 102 convs and essentially all of its MACs; the 14 grouped convs are depthwise (icpg == 1) and
    // have no channel parallelism to extract regardless.
    const bool g1 = (c.groups == 1);

    // ---- A2: xbuf ROW BANDING (CORRECTNESS. NOT a power step - pwr/FINDING_A2_BRAM.txt) -------
    // xbuf holds ceil(ic/LANES) channel-groups of H*W each, so staging a whole map needs
    //     ceil(ic/LANES) * H * W  <=  Y26_ACT_LANE_ELEMS   (3,200)
    // and 38 of the model's 126 convs violate it. The staging loop has NO bound check, so those
    // convs silently write past their bank. Every gate this design ever passed ran at substituted
    // spatial 16-40 - precisely the regime where the bound happens to hold. The header's claim of
    // a runtime guard is false, and its "naturally balanced" arithmetic divides by 16 rather than
    // Y26_LANES: written for a 16-lane design and never re-derived when lanes moved to 512.
    // FIX: stage only the input rows one output row-block actually reads, [y26_y0, +y26_bh).
    // NOTE ON BRAM: this saves NONE. A bank is 64 b x 400 and a BRAM18 is <=36 b x 512, so its
    // 2 blocks are WIDTH-driven; cutting depth crosses no ceil boundary. Measured twice (trunk
    // 1,133 at 64 b, ACT128 2,165 at 128 b). Expect BRAM to be UNCHANGED.
    // ---- A1: the xbuf banking MODULUS is Y26_ICGRP on the packed path -----------------------
    // The packed MAC reads xbuf[l] for l in [0, Y26_ICGRP) with l an UNROLL constant, and it must
    // find channel (icbi*Y26_ICGRP + l) at bank l. That is a re-banking: c -> bank c % Y26_ICGRP,
    // offset (c / Y26_ICGRP) * band. Both moduli are compile-time powers of two, so this is a
    // 2:1 select feeding a shift/mask - NOT the runtime bank index the crossbar law forbids.
    // COST, and it is the whole price of A1: per-bank capacity becomes ceil(ic/Y26_ICGRP)*band
    // instead of ceil(ic/Y26_LANES)*band, i.e. Y26_OCPACK_P times deeper. The banding below
    // (y26_banded / y26_Rblk) absorbs it by shrinking R, and the staging assert is the
    // FAIL-OPEN below, not fail-closed - see the note at the y26_Rblk clamp. Unreachable on
    // the shipping manifest. See notes 1y.65.C - the filed capacity table for this was wrong.
#if defined(Y26_A1) && defined(Y26_OCPACK)
  #if Y26_DWP > 1
    // DWP: a packed DEPTHWISE conv is banked at the PACK width, which is what turns member p's
    // bank into the unroll constant p (see the Y26_DWP note in the header). Dense is unchanged.
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
    // RBLK: y26_p = smallest row count whose y26_p*W is a whole Y26_XPE word (1 when W % XPE == 0,
    // 2 at W=20). The staging path needs the band's y0*W and bh*W in whole words, so the band below
    // is widened to multiples of y26_p. Widening adds up to 2*(p-1) rows; reserving them here keeps
    // the widened band inside the lane: raw span <= floor(rows_cap/p)*p - (p-1), and rounding that
    // out lands <= floor(rows_cap/p)*p <= rows_cap. At p == 1 this is the old formula exactly.
    // Pack 16 made W=20 depthwise convs banded (xmod 512 -> 16); before this they mis-staged.
    int y26_p = 1;
    while (y26_p < Y26_XPE && ((y26_p * W) & (Y26_XPE - 1)) != 0) y26_p <<= 1;   // masks, not %: p and XPE are powers of 2, and % synthesised 36-cycle srem cores (ledger nc)
    if (y26_banded) {
        const int rows_cap = Y26_ACT_LANE_ELEMS / (y26_icgn * W);
        int r = ((rows_cap & ~(y26_p - 1)) - (y26_p - 1) - c.kh) / c.sh + 1;
        if (r > Y26_ROWS) r = Y26_ROWS;          // accrow is only Y26_ROWS deep
        // ponytail: FAIL-OPEN, deliberately, and this is the ceiling.  When rows_cap < kh no block
        // height stages a band that fits, and clamping to 1 emits a conv whose taps read rows that
        // were never staged.  There is no correct value here without banding the CHANNEL axis too,
        // and y26_conv_top is `void` -- no status port to report it on.  Measured 2026-09-05
        // (notes/frame-thirds-scripts/guard_v41.py): 26 of 94 convs band, and rows_cap >= kh on
        // EVERY one, so this branch is unreachable for the shipping manifest.  The assert below at
        // the staging site is the only detector and it is sim-only.  UPGRADE PATH, if a future
        // geometry reaches this: add a status output and fail the conv, do not clamp.
        y26_Rblk = (r < 1) ? 1 : r;
    }
#ifdef Y26_ACCFOLD
    // ACCFOLD (ledger nm): accrow's column axis is Y26_MAX_OW/2, so each bank is ROWS*160/8 = 320 deep
    // (1 BRAM18) instead of 640 (2). A conv wider than that stores each output row as two plane rows,
    // so it may use at most Y26_ROWS/2 per block. Only 0.conv (OW 320) qualifies and it bands to Rblk 2,
    // so this clamp changes no conv in the manifest; it is here so a wider geometry degrades, not corrupts.
    const bool y26_af = OW > Y26_MAX_OW / 2;
    if (y26_af && y26_Rblk > Y26_ROWS / 2) y26_Rblk = Y26_ROWS / 2;
#endif

    // ---- ON-CHIP WEIGHT STAGING (design target 2, landed 2026-08-18) ----
    //
    // One OUTPUT CHANNEL's weights, staged in BRAM. Sized by Y26_MAX_DEPTH, which is already defined
    // as max (ic/groups)*kh*kw = 2304 - i.e. exactly the per-oc weight count. 2.25 KB, ~1 BRAM36,
    // against the ~387 tiles measured free in notes/yolo26s-zu9eg-measurements.md 3c.
    //
    // WHY PER-OUTPUT-CHANNEL and not per-conv: the `oc` loop is the OUTERMOST loop, and output
    // channel oc's weights occupy the CONTIGUOUS slice [oc*icpg*ktap, (oc+1)*icpg*ktap). Staging a
    // whole conv would need up to oc*icpg*ktap bytes (megabytes, and unbounded by any existing
    // constant); staging one output channel needs 2304 and removes exactly the same redundancy,
    // because the re-fetching was across `oh`, which is INSIDE `oc`.
    //
    // WHAT IT FIXES (the staging copy below is the only DRAM read of Wt in the MAC path now):
    //   1. OH-FOLD REDUNDANT TRAFFIC. The hoist used to sit inside the `oh` loop, so every weight was
    //      re-fetched once per output row: oc*OH*icpg*ktap reads of an oc*icpg*ktap tensor. Now each
    //      weight is read EXACTLY ONCE per conv. At production spatial that is an 80-160x cut.
    //   2. BURST INFERENCE. csynth reported `214-230 Stride is incompatible` on both Wt reads,
    //      because consecutive lanes step the address by `ktap`, not 1 - so HLS refused to burst and
    //      every access was a single-beat 8-bit transaction on a 128-bit-capable port. The copy
    //      below is a STRIDE-1 linear scan, which is the pattern that does burst.
    //   3. DRAM LATENCY IMMUNITY in the inner loop, which now reads BRAM, not m_axi.
    //
    // HONEST SCOPE - what it does NOT fix: the per-(icb,tap) hoist still runs once per output row,
    // so the total number of weight fetches into `wl` is unchanged; staging changes where they come
    // FROM, not how many there are. Removing them needs the `oh` loop moved INSIDE the tap loops
    // (weight-stationary), which needs an OH x OW accumulator plane instead of accrow[OW]. That is a
    // separate change - see the note. Do not attribute the OH-fold cycle win to this commit.
    //
    // ---- BANKED BY LANE, and that is NOT cosmetic - MEASURED 2026-08-18 ----
    //
    // The first version of this was a FLAT `wbuf[Y26_MAX_DEPTH]` read by a sequential hoist loop.
    // It cost 857,272 cycles against the 457,004 baseline: +87.6%, a REGRESSION. csynth did not
    // predict it - with trip counts pinned to the cosim conv, the same static model that reproduces
    // the flat-Wt baseline to 0.4% (459k predicted / 457,004 measured) predicted 404k for the flat
    // staged version. The missing ~453,000 cycles were runtime stall on a single-ported BRAM shared
    // between the staging writer and the hoist reader: ~3.5 cycles per read, invisible to a schedule
    // that reports II=1. Do not trust an II=1 report to mean a memory is actually keeping up.
    //
    // Banking by lane is the same trick `xbuf` already uses, and for the same reason: channel c goes
    // to bank c % Y26_LANES, the `icb` tile step is exactly Y26_LANES, so lane l ALWAYS reads bank l
    // - a literal index once UNROLL has run, provably conflict-free. Two consequences:
    //   * no arbitration, so the stall above disappears;
    //   * the hoist collapses from Y26_LANES SEQUENTIAL reads to ONE parallel read, which is what
    //     finally makes lanes close to free on the weight side rather than ~1.6 cycles each.
    //
    // BANK DEPTH: a bank holds ceil(icpg/Y26_LANES) rows of ktap taps. Bounded by
    // Y26_MAX_DEPTH/Y26_LANES rounded up, plus one extra row (Y26_MAX_K^2) for the partial final
    // tile. Small enough that each bank is LUTRAM/registers rather than a BRAM, which is exactly
    // what we want - a BRAM per lane would not fit at high lane counts.
    static y26_wt_t wbuf[Y26_LANES][Y26_WBANK];
    #pragma HLS ARRAY_PARTITION variable=wbuf complete dim=1

    // Zero-point weight sums for the CURRENT output channel. Sum over the icl axis of the int8
    // weight codes, per tap. The icl axis contributes identically at every output position, so it is
    // pre-summed once per (oc, tap); the per-row pass below then costs kh*kw range-adds instead of
    // icpg*kh*kw. Exact in int32: |sum| <= 127 * 2304 = 292,608.
    //
    // MOVED INSIDE THE `oc` LOOP 2026-08-18. It used to be a whole-tensor precompute into
    // kwsum[Y26_MAX_OC * K * K] (4,608 int32 = 18 KB) that made its own redundant pass over Wt from
    // DRAM - the second `Stride is incompatible` site. Computing it per-oc from the staged wbuf
    // removes that pass entirely and shrinks the array to ktap entries. The summation ORDER over
    // icl is unchanged (ascending), so the values are identical and the bit-exact gate still holds.
#ifdef Y26_OCPACK
    // W3: one kwsum row per PACKED GROUP MEMBER. Filled by an UNROLLED p-loop (see the staging
    // code below), so this is Y26_OCPACK_P independent small arrays, not a runtime-indexed one -
    // matching the "p must be compile-time to keep the bank arithmetic free" rule in the header.
    ap_int<32> kwsum_o[Y26_ACCP][Y26_MAX_K * Y26_MAX_K];   // DWP: widest pack, not the dense one
    #define Y26_KWSUM(p_, t_) kwsum_o[p_][t_]
#else
    ap_int<32> kwsum_o[Y26_MAX_K * Y26_MAX_K];
    #define Y26_KWSUM(p_, t_) kwsum_o[t_]
#endif

    // Per-output-row accumulators, one per OUTPUT COLUMN. The loop interchange that produced them is
    // still load-bearing (it is what keeps the reduction out of a loop-carried dependency), but the
    // PARALLELISM no longer comes from this axis - see the banking note in conv_engine.h. As of
    // 2026-08-17 the lanes run along the INPUT CHANNEL axis, so the MAC loop touches exactly one
    // accrow[ow] per cycle.
    //
    // Consequently accrow is NOT partitioned any more. Partitioning it would now cost a 16-way
    // mux/demux on a runtime `ow` index while delivering nothing, since only one access per cycle is
    // issued.
    //
    // accwrow IS NO LONGER PARTITIONED EITHER - corrected 2026-08-17. This comment used to read
    // "accwrow IS still partitioned, because the zero-point pass below stays lane-tiled over output
    // columns (it reads no activations, so it was always at II=1 and needs no restructuring)". The
    // II claim was true and the conclusion was WRONG: `cyclic factor=Y26_LANES` on an index whose
    // tile base `o0` is a RUNTIME value is the exact same runtime-rotation crossbar that gave the MAC
    // loop II=8 - it just paid in AREA instead of initiation interval, so it stayed invisible.
    //
    // Measured before the fix (csynth, XCZU9EG): the zero-point pass Outline_VITIS_LOOP_193_11 cost
    // 8,293 LUT at 16 lanes and 26,581 at 32 - 3.2x LUT and 4.5x FF for 2x lanes, i.e. O(LANES^2),
    // growing from 43% to 67% of kernel LUT. The MAC pass next door scaled 1.3x. It also owned the
    // 32-lane critical path (Pipeline_VITIS_LOOP_200_13 -> accwrow_6_U). See
    // notes/yolo26s-zu9eg-measurements.md.
    //
    // The pass adds a CONSTANT `ws` across a contiguous [o0,o1), so it does not need lane
    // parallelism at all - sequential at II=1 needs no partition and builds no crossbar. Do not
    // "restore" the tiling to make this loop look faster; the area it costs is superlinear and the
    // cycles it saves are trivial (see the cost note at the loop itself).
    //
    // The old comment here warned against "simplifying back to a scalar accumulator". That warning
    // still applies to a SCALAR: it would serialize the reduction into a dependency chain. It does
    // NOT apply to the per-lane psum + single adder tree used below, which has no carried dependency
    // because the tree is combinational within one pipeline stage. Do not conflate the two.
    // W8: one plane per fused row. At Y26_ROWS == 1 these are the historical [Y26_MAX_OW] arrays
    // with an extra dimension of extent 1, which HLS folds away - the control build is unchanged.
#ifdef Y26_OCPACK
    // W3: one accrow/accwrow PLANE per packed group member. Slot p is only meaningful while
    // p < pack for the group currently being processed; unused slots (pack < Y26_OCPACK_P, or any
    // conv that doesn't pack at all) hold don't-care values that are never read - see Y26_ACCROW.
    // This is EXTRA HARDWARE, permanently present once Y26_OCPACK is compiled in, independent of
    // whether a given conv call actually packs - see the header note's COVERAGE/COST table for
    // the BRAM this costs.
  #ifdef Y26_ACCFOLD
    #if (Y26_MAX_OW / 2) % Y26_EPI_WIDE != 0 || Y26_ROWS % 2 != 0
      #error "Y26_ACCFOLD needs Y26_EPI_WIDE | Y26_MAX_OW/2 (groups stay in one plane row) and even Y26_ROWS"
    #endif
    ap_int<32> accrow [Y26_ACCP][Y26_ROWS][Y26_MAX_OW / 2];  // ACCFOLD: see y26_af at the Rblk clamp
  #else
    ap_int<32> accrow [Y26_ACCP][Y26_ROWS][Y26_MAX_OW];     // DWP: widest pack, not the dense one
  #endif
  #if Y26_EPI_WIDE > 1
    // Same cyclic-by-epilogue-width partition as the non-packed case, now on dim=3 (the OW axis
    // moved down one slot because of the new leading PACK dimension).
    #pragma HLS ARRAY_PARTITION variable=accrow  cyclic factor=Y26_EPI_WIDE dim=3
  #endif
    // One reference macro for both shapes, so every existing accrow/accwrow call site below reads
    // Y26_ACCROW(p, rr, ow) regardless of which struct is live. Under !Y26_OCPACK this collapses to
    // the historical accrow[rr][ow] with p silently dropped (always called with p==0 there) - a
    // pure rename, not a behavior change. Mirrors the Y26_YOUT macro used at the epilogue below.
  #ifdef Y26_ACCFOLD
    // Folded conv: output (r, o) lives at plane row 2r + (o >= 160), column o mod 160. 160 is a multiple
    // of Y26_EPI_WIDE, so col & 7 == o & 7 and every epilogue group is still row-local.
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
    // W4: cyclic by the epilogue width on the COLUMN axis, so a row-local group of Y26_EPI_WIDE
    // consecutive columns occupies one bank each. See the long note at Y26_EPI_WIDE in the header
    // for why the groups must be row-local (straddling collides) and for the MAC-side demux this
    // buys. Y26_MAX_OW is 320 and every deployed OW is a multiple of 4, so bank 0 always coincides
    // with a row start for the factors this is swept at.
    #pragma HLS ARRAY_PARTITION variable=accrow  cyclic factor=Y26_EPI_WIDE dim=2
#endif
    #define Y26_ACCROW(p_, r_, o_)  accrow[r_][o_]
#endif

    // ---- W10b: the zero-point plane is GEOMETRY, so it is reconstructed, not stored -------------
    // `accwrow` used to hold, per output pixel, the sum of kwsum over every (kh,kw) tap whose input
    // pixel was in range. That value has no data dependence at all - it is a pure function of the
    // conv's geometry and the per-tap constants kwsum_o[p][tap]:
    //     accwrow(p,er,owl) = SUM over kh live for er, kw with owl in [o0(kw),o1(kw)) of kwsum
    // The column condition is an INTERVAL per kw and Y26_MAX_K is 3, so at most 3 intervals exist.
    // Both sums therefore collapse into a handful of unrolled conditional adds that fit inside the
    // epilogue's existing II=1 pipeline - see the reconstruction at the dequant below.
    // This deletes the whole zero-point PASS (its cycles AND its ~46.7-cyc-per-entry region cost,
    // together 48% of the measured frame - notes 1y.45/47) and the accwrow array with it.
    // The column intervals depend only on the conv, so they are hoisted out of every loop here.
    int zpo0[Y26_MAX_K], zpo1[Y26_MAX_K];
    #pragma HLS ARRAY_PARTITION variable=zpo0 complete
    #pragma HLS ARRAY_PARTITION variable=zpo1 complete
    for (int kwi = 0; kwi < Y26_MAX_K; ++kwi) {
        #pragma HLS UNROLL
        // y26_ow_range is the SAME call the deleted pass used (:651), so stride/pad handling is
        // carried over verbatim rather than re-derived. Taps past c.kw get an empty range.
        int a = 0, b = 0;
        if (kwi < c.kw) y26_ow_range(kwi - c.pw, c.sw, W, OW, a, b);
        zpo0[kwi] = a;
        zpo1[kwi] = b;
    }

#ifdef Y26_OCPACK
    // W3: pack_cap is a per-CONV constant, computed once. True (>1) exactly when this conv is dense
    // (g1) and its icpg fits inside one packed group's channel width - see the header note for why
    // that guard is what keeps npass at 1 and leaves xbuf untouched.
    // W12: pure DEPTHWISE (icpg == 1) now packs too. The old gate was `g1` alone because the packed
    // MAC needs every member's input channels inside one Y26_ICGRP-wide weight slice; icpg == 1
    // satisfies that trivially - one channel per member. A general grouped conv (2 <= icpg) still
    // gets pack_cap == 1 and takes the unfused fallback, which stays CORRECT, just slow. None exist
    // in this model, so that path is insurance rather than a cost.
#ifdef Y26_A1
    // A1: the icpg <= Y26_ICGRP restriction is LIFTED for dense convs. It existed only to keep
    // npass == 1 so xbuf's Y26_LANES banking still placed channel l at bank l; the re-banking
    // above supplies that property at any icpg, and the MAC/kwsum below carry the icbi axis.
    // This is the change that retires the unpacked dense datapath.
#if Y26_DWP > 1
    // DWP: depthwise packs Y26_DWP wide. The A1 comment below is why this was NOT simply switched
    // on: arming the W12 loop alone leaves it `p < 1` while the oc loop steps by `pack`, dropping
    // channels. Every one of those pieces - the loop width, the activation modulus, the weight
    // group width, the kwsum shape and the per-member array extents - moves together in this build.
    const int pack_cap = g1 ? Y26_OCPACK_P : ((icpg == 1) ? Y26_DWP : 1);
#else
    const int pack_cap = g1 ? Y26_OCPACK_P : 1;   // A1: dense at ANY icpg. NOT icpg==1: the
                                                 // W12 depthwise packed loop at :1092 is the dwA
                                                 // counterfactual (p < 1) while the oc loop still
                                                 // steps by pack, so arming it drops 3 of every 4
                                                 // depthwise channels. Depthwise is out of A1 scope.
#endif
#else
    const int pack_cap = (g1 && icpg <= Y26_ICGRP) ? Y26_OCPACK_P : 1;   // dwA: dense only
#endif
#endif

    // TAPLANE: one conv-level decision, read by BOTH staging loops and the MAC. Keeping it in
    // one place is the whole safety argument - a replication the MAC does not expect, or an
    // expectation the staging did not serve, is a silent wrong answer, not a compile error.
    const bool y26_tl = (pack_cap > 1) && g1 && Y26_TL_OK(icpg, c.kh);

    // ---- A2: ROW-BLOCK LOOP, HOISTED OUTSIDE THE oc LOOP -----------------------------------
    // The block loop used to sit INSIDE the oc loop. Banding needs staging per block, so leaving
    // it there would stage oc*nblocks times (the :106 comment warns exactly that). Hoisted, the
    // cost is weights re-staged per block - at most 7.7 ms network-wide, and the K-probe found no
    // per-entry blow-up (pwr/PREDICT_A2K.txt). The body does not move: the outer loop picks the
    // block and the original inner loop is made SINGLE-TRIP at it, which is what interchanging
    // them means. src_H is the control leg; the negative control is a deliberately wrong band.
    int y26_staged_y0 = -1, y26_staged_bh = -1;      // band currently resident in xbuf
#ifdef Y26_PREFETCH
    // L-PREFETCH (ledger nt, plan 0.8df; C7 steps 2-3). The per-member staging loop below read four
    // scalars from DRAM per member per ROW BLOCK - single-beat, burst-Fail - plus a sequential sdiv for
    // the step/lo index: ~97 cycles per visit, 36,937 visits a frame. The vectors are conv-invariant,
    // so burst them into BRAM once here, one loop per array (a single loop over the bundle is what
    // csynth refuses: "multiple potential reads to the same bundle in the same region").
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
        // L-FUSE: the conv-wide arm of k1 = wsc*step / k2 = wsc*lo rides on this burst, so it costs
        // no loop and no iteration. MEASURED (ledger ot, first build): doing the fold in the
        // per-member staging loop inside the block loop instead cost +153 cycles of that loop's
        // INTERVAL per visit and made all 8 cosim convs ~0.43% SLOWER - far more than the 5 pipeline
        // stages the epilogue gave back. The fold has to be conv-invariant work, and it is.
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
        // Input rows this block reads, clamped to the image. Because y26_y0 >= 0 and
        // y26_y0 + y26_bh <= H, band-relative 0 <= ih < y26_bh is EXACTLY the old absolute
        // test 0 <= ih < H for every tap this block issues - which is why the only edit at the
        // four read sites is to ih0 and the channel-group stride.
        int y26_y0 = 0, y26_bh = H;
        if (y26_banded) {
            const int rlast = ((ohbo + y26_Rblk) < OH ? (ohbo + y26_Rblk) : OH) - 1;
            int t = ohbo * c.sh - c.ph;              // first input row touched
            int b = rlast * c.sh - c.ph + c.kh;      // one past the last
            if (t < 0) t = 0;
            if (b > H) b = H;
            // RBLK: widen to the word-aligned row period (no-op at p == 1). Extra rows are real
            // input rows no tap of this block reads ([y0,t) and [b,y0+bh) lie outside [t,b)), and
            // the clamp keeps y0+bh <= H so band-relative ih < bh still equals absolute ih < H.
            y26_y0 = t & ~(y26_p - 1);
            y26_bh = (b - y26_y0 + y26_p - 1) & ~(y26_p - 1);
            if (y26_y0 + y26_bh > H) y26_bh = H - y26_y0;
        }
        const ap_uint<19> xb_hw = (ap_uint<19>)((ap_uint<11>)y26_bh * (ap_uint<11>)W);
        // Stage only when the band actually changed. Unbanded convs keep (0, H) for every block,
        // so they stage ONCE per conv exactly as before. Both keys are needed: at Rblk == 1 with
        // sh == 1, ph == 1 the first two blocks share y0 == 0 but differ in bh.
        if (y26_staged_y0 != y26_y0 || y26_staged_bh != y26_bh) {
            y26_staged_y0 = y26_y0; y26_staged_bh = y26_bh;
#ifndef __SYNTHESIS__
            // The wide staging path indexes X in Y26_XPE-element words, so the band's byte offset
            // and length must both be word multiples. ic*hw already is; y0*W and bh*W are because
            // the band is widened to multiples of y26_p above (RBLK). The old reason - "every banded
            // conv has W >= 80" - was broken by Y26_DWP=16, which bands W=20. This assert still fails
            // loudly if H is not a multiple of y26_p (the H clamp above), rather than mis-staging.
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
            // ---- G0b: X is a Y26_ACT_WORD-bit port ----------------------------------------------
            // One port word carries Y26_XPE consecutive spatial elements of ONE channel, so the word
            // index is (src0 + j)/Y26_XPE and the byte lane is j % Y26_XPE. src0 == ic*hw and hw is a
            // multiple of 8 for every conv in the deployed set (min HW is 8x8 = 64, and every spatial
            // size in the pyramid is a multiple of 8), so src0 is word-aligned and there is no
            // epilogue. The inner UNROLL is what makes the Y26_XPE byte-writes coalesce into the one
            // wide bank word that G0's RESHAPE created - G0 leg 2 had the reshape but was still fed
            // one byte per iteration FROM DRAM, which is the whole finding of 1x.
            const y26_idx_t src0w = (y26_idx_t)(src0 / Y26_XPE);
            const y26_idx_t hww   = (y26_idx_t)(xb_hw / Y26_XPE);
            // dst0w exists ONLY to make the destination's Y26_XPE-alignment STRUCTURAL rather than a
            // runtime fact. `dst0 == cg*hw` is arithmetically a multiple of Y26_XPE (hw is, for every
            // conv in the deployed set - same guarantee src0w above already leans on), but hw is a
            // RUNTIME value, so HLS cannot prove it. MEASURED CONSEQUENCE at 512 lanes with
            // Y26_XBUF_WIDE=8 (notes 1y.35): HLS assumed the Y26_XPE byte-writes could straddle two
            // reshaped words and emitted a full barrel shifter plus a serialising dependency -
            // **staging loop II 1 -> 8, "Memory Dependency", 576,098 LUT / 300,631 FF in that module
            // alone**, which cancels the whole 8.1x that the 64-bit port (W5/G0b) bought. Writing the
            // base in WORD units makes `(dst0w + jw) * Y26_XPE` a syntactic multiple of Y26_XPE, so the
            // Y26_XPE writes provably hit ONE word and coalesce. Numerically identical - dst0w*XPE ==
            // cg*hww*XPE == cg*hw == dst0 - so the gates are unchanged with or without the reshape.
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
                // TAPLANE: the SAME word into the Y26_TP-1 dead sub-groups. Same address, same
                // iteration, same DRAM read - only the bank differs, and each bank is its own BRAM
                // with its own write port, so this adds writes, not cycles. `tp` is an unroll
                // constant, so `tp*Y26_TPCG + l` is the compile-time-plus-runtime-base index form
                // the packed MAC already uses for wbuf.
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
                // G0 leg 2: RESHAPE alone was a null (measH) - it widened the bank to 64 bits but this
                // loop still issues ONE 8-bit write per iteration, so HLS spent 6,208 LUTs on byte-lane
                // muxing (staging loop 323 -> 6,531 LUT) and the cycle count did not move at all.
                // Unrolling by the same factor puts 8 consecutive j in one iteration, which is what lets
                // the 8 byte-writes coalesce into a single word-write. hw is always a multiple of 8 for
                // every conv in the deployed set (min HW is 8x8=64), so no epilogue is needed.
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
        // W12: step/lo ARE PER MEMBER NOW. They used to be one conv-wide pair, and the note here
        // justified that with "packing only ever fires under g1, where perch is always false".
        // Depthwise packing is exactly the case that breaks it: `perch` is depthwise-only and
        // indexes by ic0, and with ocpg == 1 every member has its OWN ic0 == oc + p. Sharing member
        // zero's pair would dequant all P channels with channel zero's scale - a bug that passes any
        // dense-only check and corrupts precisely the convs this change exists to speed up.
        float step_o[Y26_ACCP];
        float lo_o[Y26_ACCP];
        #pragma HLS ARRAY_PARTITION variable=step_o complete
        #pragma HLS ARRAY_PARTITION variable=lo_o   complete
        // Per-member dequant scale/bias. Filled by the staging loop below, one member at a time.
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

        // --- stage this group's weights into BRAM (see the wbuf note above) ---
        // STRIDE-1 by construction: output channel oc owns the contiguous slice
        // [oc*icpg*ktap, +icpg*ktap) of Wt, so this is the linear scan HLS can turn into AXI bursts.
        // The MAC hoist's `(icb+l)*ktap + tap` pattern is the SAME data in a strided order - staging
        // it here is what converts that strided DRAM access into a strided BRAM access.
        // W9 (2026-08-19): the slice stride is Y26_WSTRIDE, not icpg*ktap. They are equal in the
        // control build; under -DY26_WT_WORD it is rounded up so every oc slice starts on a PORT
        // WORD boundary, and under -DY26_WT_TAPMAJOR the rounding lands on the channel run.
        const int       icpgp = Y26_WICPG(icpg);                 // padded channels per tap
        const int       wlen  = Y26_WSTRIDE(icpg, ktap);         // elements per oc slice
#ifdef Y26_OCPACK
  #if Y26_WPE > 1 && !defined(Y26_WT_TAPMAJOR)
    #error "Y26_OCPACK + wide weight port requires -DY26_WT_TAPMAJOR, for the same reason the unpacked path does: without tap-major order one port word straddles taps as well as channels, the destination bank becomes a runtime value, and the write needs exactly the Y26_LANES:1 crossbar W9 exists to avoid."
  #endif
  #if Y26_WPE == 1 && defined(Y26_WT_TAPMAJOR)
    #error "Y26_OCPACK + -DY26_WT_TAPMAJOR at the 8-bit port is NOT implemented. The packed 8-bit staging loop below walks `i` assuming CHANNEL-major order (Y26_WIDX = icl*ktap + t); under tap-major the layout is t*WICPG + icl and that loop would silently stage every weight to the wrong place. This guard converts a wrong answer into a build failure."
  #endif
        // W3: SEQUENTIAL over p - this loop touches m_axi (gmem_wt), and per the W4 finding
        // elsewhere in this file, "if a loop touches m_axi, the port IS the item": Y26_OCPACK_P
        // simultaneous burst state machines would still serialize through the one real port, so
        // unrolling here would only pay for hardware that can never actually run in parallel.
        //
        // wgrp is the RUNTIME (but per-conv-constant) group width this staging targets: when this
        // conv is not packable (pack_cap == 1, e.g. icpg > Y26_ICGRP or grouped/depthwise), it must
        // stay Y26_LANES - the unpacked MAC/kwsum read paths below still expect the ORIGINAL
        // channel%Y26_LANES layout, and packing this conv with a narrower Y26_ICGRP modulus would
        // silently scatter channels >= Y26_ICGRP into the wrong bank/row (MEASURED: this was
        // tried and every conv with icpg > Y26_ICGRP failed the bit-exact gate). p==0 with
        // wgrp==Y26_ICGRP lands in banks [0, Y26_ICGRP) IDENTICALLY to the pack_cap==1 case with
        // wgrp==Y26_LANES, because pack_cap > 1 guarantees icpg <= Y26_ICGRP (the header note's
        // guard), so wicl never reaches Y26_ICGRP and the two moduli agree bit for bit there. Only
        // p >= 1 needs the extra p*wgrp offset - that offset is the entire mechanism this feature
        // spends area on. This loop is SEQUENTIAL (not unrolled - see the note above), so a runtime
        // wgrp costs nothing extra: it is one more scalar in an address computation that already
        // runs once per element, not a per-lane term HLS has to size hardware for at every width.
        // DWP: a depthwise member owns exactly ONE channel, so its nine taps are a single bank
        // and the group width is the narrowest the staging loop can EXPRESS - Y26_WPE, not 1: the
        // wide tap-major path below counts in PORT WORDS (wgrpw = wgrp / Y26_WPE), so a width of 1
        // divides to ZERO and stages every member on top of member 0. MEASURED as a bit-exact
        // FAILURE on exactly the 6 depthwise convs with the dense 94 still green, 2026-09-07 -
        // which is what tb_s1 is for, and what neither earlier depthwise probe ever ran.
        // Member p lands at bank p*Y26_WPE, which is what the MAC and kwsum read. At
        // Y26_DWP == 1 depthwise has pack_cap == 1 and takes the Y26_LANES arm exactly as before.
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
            // W12: per-member step/lo. Under !perch these collapse to the same conv-wide scalar for
            // every p, so the dense packed path is bit-identical to before this change.
            // C7 was tried here and REVERTED, run A, 2026-09-05 (pwr/SCORE_RUNA.txt). Writing this
            // as `g1 ? 0 : (ocp / ocpg) * icpg` changed the divider count by exactly ZERO: `g1` is a
            // runtime value, so HLS synthesises both arms and the sdiv core stays. A ternary is not
            // a compile-time exclusion. If this divide is ever worth removing it needs a `#if` or a
            // template parameter, not a select -- and measure the divider count, not the log text.
            const int ic0p = (ocp / ocpg) * icpg;
            step_o[p] = c.perch ? c.step_v[ic0p] : c.step;
            lo_o[p]   = c.perch ? c.lo_v[ic0p]   : c.lo;
#ifdef Y26_FUSE_ON
            // No prefetch arrays to fold into on this arm, so it folds per block and pays the +153
            // the PREFETCH arm above exists to avoid. Not a shipping configuration.
            step_o[p] = (float)(wsc_o[p] * step_o[p]);
            lo_o[p]   = (float)(wsc_o[p] * lo_o[p]);
#endif
#endif
            const int bank_off = p * wgrp;
#if Y26_WPE > 1
            // ---- W9 x OCPACK: WIDE, TAP-MAJOR packed staging ---------------------------------
            // One port word carries Y26_WPE consecutive CHANNELS of ONE tap. Every counter below
            // is kept in PORT-WORD units so the destination bank stays syntactically
            // `(something) * Y26_WPE + b` with `b` an unroll constant - see the header of this
            // patch for the distinctness proof and why element units would break it.
            const int       wgrpw  = wgrp / Y26_WPE;        // bank ring, in words
            const int       icpgw  = icpgp / Y26_WPE;       // words per tap (icpgp is WALIGNed)
            const int       bank_offw = p * wgrpw;          // packed base, in words
            const y26_idx_t wbasew = (y26_idx_t)(wbase / Y26_WPE);
            const int       nwords = wlen / Y26_WPE;
            int wt = 0, wiclw = 0, wblw = 0, wbr = 0;
#ifdef Y26_TAPLANE
            // TAPLANE: tap wt = kh*c.kw + kw of channel cc belongs to lane kh*Y26_TPCG + cc at
            // within-bank offset kw. Tracked with counters, not `wt / c.kw`: a runtime divide here
            // would buy back the 35-cycle sdiv core C7 spent two runs removing. Y26_TPCG is a
            // multiple of Y26_WPE, so the sub-group step stays WORD-aligned and the destination
            // keeps the `(...) * Y26_WPE + b` shape whose distinctness the W9 proof rests on.
            int tl_kh = 0, tl_kw = 0;
            const int tl_khw = Y26_TPCG / Y26_WPE;
#endif
            for (int iw = 0; iw < nwords; ++iw) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=4 max=((Y26_MAX_DEPTH + Y26_MAX_K * Y26_MAX_K * (Y26_WPE - 1)) / Y26_WPE) avg=144
                const y26_ww_t wrd = Wt[wbasew + (y26_idx_t)iw];
                for (int b = 0; b < Y26_WPE; ++b) {
                    #pragma HLS UNROLL
                    // Weights are SIGNED int8 in an unsigned byte container - the cast through
                    // ap_int<8> is load-bearing, see the y26_ww_t note in the header.
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
        // ONE FLAT loop, and it has to stay flat. The natural nesting (`for icl { for t }`) re-enters
        // a ktap-deep pipeline icpg times, and with latency=30 on gmem_wt each re-entry pays the
        // fill: at ktap=1 that is icpg*(1+30) per output channel instead of icpg. Walking `i`
        // linearly and carrying icl/t/bank as counters keeps the Wt read STRIDE-1 (so it bursts -
        // confirmed by `[HLS 214-115] burst read ... inferred on bundle 'gmem_wt'`, where the flat
        // pre-staging code reported `214-230 Stride is incompatible`) while still scattering into
        // the lane banks. The counter updates are increments and power-of-two ops, so II=1 holds.
        //
        // The counters below are the INVERSE of Y26_WIDX(): they recover (icl, t) from the linear
        // walk, where the testbench PLACES each weight at Y26_WIDX(icpg, ktap, icl, t). The two are
        // a matched pair - change one and the g++ bit-exact gate fails, which is the intent.
        //
        // NOTE what does NOT change here: `wbuf`'s own layout is still [bank][group*ktap + tap], so
        // the kwsum tree and the MAC read pattern below are untouched by W9. Only the DRAM-side
        // ORDER moves. That keeps the blast radius to this loop and the testbench's packer.
#if Y26_WPE > 1
  #if !defined(Y26_WT_TAPMAJOR)
    #error "Y26_WT_WORD>8 requires -DY26_WT_TAPMAJOR. Without tap-major ordering the Y26_WPE elements of one port word straddle taps as well as channels, so the destination bank becomes a runtime value and the write needs exactly the Y26_LANES:1 crossbar W9 exists to avoid. Buildable pairs: (8-bit, either layout) or (wide, tapmajor)."
  #endif
  #if (Y26_LANES % Y26_WPE) != 0
    #error "Y26_LANES must be a multiple of Y26_WPE, or one port word's elements do not land in distinct banks."
  #endif
        // One port word carries Y26_WPE consecutive CHANNELS of ONE tap. `wicl` advances by Y26_WPE
        // and Y26_LANES is a multiple of Y26_WPE, so the bank index is always a multiple of Y26_WPE
        // and the byte lane `b` supplies its low bits: the Y26_WPE writes provably hit Y26_WPE
        // DISTINCT banks. Indexing as `wbl0*Y26_WPE + b` with `b` an unroll constant is what makes
        // that visible to HLS's bit-level analysis - `wbl+b` would leave it an unprovable runtime
        // add and rebuild the crossbar. **This is the measurement §1y.12 has never had.**
        const y26_idx_t wbasew = (y26_idx_t)(wbase / Y26_WPE);
        const int       nwords = wlen / Y26_WPE;
        int wt = 0, wicl = 0, wbl0 = 0, grpk = 0;
        for (int iw = 0; iw < nwords; ++iw) {
            #pragma HLS PIPELINE II=1
            // This loop counts PORT WORDS, so its bounds are the element bounds divided by Y26_WPE.
            // CORRECTED after the first G10 run, where `max=Y26_MAX_DEPTH` was left over from the
            // byte loop: csynth reports the MAX trip count for a variable-trip loop, so both legs
            // reported ~2304 cycles and the staging loop looked UNCHANGED by W9. The pragma was
            // 8x too high; it measured my annotation, not the design. A LOOP_TRIPCOUNT is not a
            // hint here - it is the number the report prints.
            #pragma HLS LOOP_TRIPCOUNT min=1 max=((Y26_MAX_DEPTH + Y26_MAX_K * Y26_MAX_K * (Y26_WPE - 1)) / Y26_WPE) avg=144
            const y26_ww_t wrd = Wt[wbasew + (y26_idx_t)iw];
            for (int b = 0; b < Y26_WPE; ++b) {
                #pragma HLS UNROLL
                // Weights are SIGNED int8 in an unsigned byte container - the cast through
                // ap_int<8> is load-bearing, see the y26_ww_t note in the header.
                wbuf[wbl0 * Y26_WPE + b][grpk + wt] =
                    (y26_wt_t)(ap_int<8>)wrd.range(b * 8 + 7, b * 8);
            }
            wicl += Y26_WPE;
            ++wbl0;
            if (wbl0 == Y26_LANES / Y26_WPE) { wbl0 = 0; grpk += ktap; }
            if (wicl >= icpgp) { wicl = 0; wbl0 = 0; grpk = 0; ++wt; }
        }
#elif defined(Y26_WT_TAPMAJOR)
        // Tap-major at the 8-bit port: the relayout on its own, so its latency effect can be read
        // apart from the port width. `i` now walks channels fastest, so the rollover is on the
        // CHANNEL axis and the bank advances every iteration (an increment, not a modulo).
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

        // Per-tap zero-point sums, one row per LIVE group member, from the staged banks.
        //
        // Lane-parallel with a combinational adder tree, mirroring the MAC below - NOT a scalar walk
        // over `icl`. A scalar walk would index wbuf[icl % Y26_LANES][...] with a RUNTIME bank, which
        // builds the Y26_LANES:1 read crossbar that the zero-point pass was restructured to retire
        // (see the accwrow note further down). This form reads bank l at lane l, statically.
        //
        // BIT-EXACTNESS: the tree reassociates the sum over input channels. Exact for integers -
        // same argument as the MAC's adder tree, and the reason these accumulators are ap_int and
        // not float. The stage-1 gate is an EXACT-equality gate and covers this.
#ifdef Y26_OCPACK
        // W3: TWO compile-time-fixed-shape reductions, selected at RUNTIME by pack_cap - not one
        // reduction parameterized by a runtime width. p*Y26_ICGRP with p UNROLLED needs Y26_ICGRP
        // itself to be the array-index unit at EVERY unrolled instance for the address to stay
        // providably in [0, Y26_LANES); a conv with pack_cap == 1 (icpg > Y26_ICGRP) has no use for
        // the p axis at all, and reusing this shape for it with a differently-scaled "group width"
        // would make p*width a value HLS has to size for the WORST case across both branches - i.e.
        // up to 3*Y26_LANES, off the end of wbuf. Keeping the two shapes separate, each with its own
        // compile-time-constant reduction width, is what avoids that.
#if Y26_DWP > 1
        if (icpg == 1 && pack_cap > 1) {
            // DWP: a depthwise member sums over exactly ONE input channel, so the Y26_ICGRP-wide
            // adder tree below collapses to a single read - and it MUST, because wgrp is 1 on this
            // path: member p's taps are at bank p, not at bank p*Y26_ICGRP. Reusing the dense shape
            // here would read the wrong banks and produce a wrong answer that still synthesises.
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
            // p is UNROLLED (0..Y26_OCPACK_P-1, compile-time), matching the MAC pass's bank
            // arithmetic - see the header note for why p must be unrolled here, unlike the staging
            // loop above. nrows collapses to 1 whenever pack_cap > 1 (icpg <= Y26_ICGRP by
            // construction), so this is the same total 512-wide reduction work as the unpacked form
            // below, just split into Y26_OCPACK_P independent Y26_ICGRP-wide trees instead of one
            // Y26_LANES-wide tree - no more adds than today, not fewer. p >= pack reads whatever the
            // previous group left in that slice of wbuf and writes it to kwsum_o[p]; that value is
            // never consumed (the zero-point and epilogue passes only touch p < pack), so it is dead
            // computation, not a correctness gap.
            // W11.1: nrows is provably 1 in this branch (pack_cap > 1 => icpg <=
            // Y26_ICGRP), so the reduction below no longer loops over it. Kept as a
            // named assertion of that fact rather than deleted.
            const int nrows = (icpg + Y26_ICGRP - 1) / Y26_ICGRP;
#ifndef Y26_A1
            (void)nrows;
#endif
            // W11.1: ONE flat pipeline region instead of Y26_OCPACK_P of them.
            //
            // WHAT WAS WRONG: the `p` UNROLL sat OUTSIDE the pipelined `r` loop, so HLS outlined P
            // separate modules and entered each one `ktap` times per oc-group. At the measured
            // K = 60.45 cycles per region ENTRY that is P*ktap entries where one would do -
            // MEASURED SHAPE: 68,044 entries across the packed convs, against 3,411 flattened,
            // i.e. 64,633 wasted entries = ~15.6 ms. Same adders, same work, pure entry arithmetic.
            // This is the W8/W10 pattern (flatten the nest, unroll INSIDE the body) applied a third
            // time; it was the single biggest win both previous times.
            //
            // `nrows` IS PROVABLY 1 HERE and the r loop is therefore gone: this branch runs only
            // when pack_cap > 1, which by the pack_cap rule requires icpg <= Y26_ICGRP, so
            // ceil(icpg / Y26_ICGRP) == 1. The old `r * ktap + t` collapses to `t` and
            // `r * Y26_ICGRP + l` to `l`. Dropping a loop HLS could not prove was single-trip is
            // what lets the whole thing sit in one II=1 region.
            //
            // Integer adds are associative, so removing the ptmp partial keeps this BIT-EXACT
            // against the old form - which the g++ gate checks on all 100 convs.
#ifdef Y26_A1
            // A1: nrows is NO LONGER provably 1 - that proof rested on the pack gate this build
            // lifts. The r axis is restored. The PIPELINE sits on the r loop so a conv with
            // nrows == 1 still enters exactly one region per tap, i.e. W11.1's entry count is
            // preserved for every conv that packed before A1 and only the newly-packed convs pay
            // more. Integer adds are associative, so this stays bit-exact against the old form.
#ifdef Y26_TAPLANE
            // Tap odometer for the permuted read below - counters, not t/c.kw and t%c.kw, for the
            // same reason C7 spent two runs removing one sdiv: a runtime divide here is a core.
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
                    // Dead lanes of a partial channel-block contribute 0, matching the MAC pass.
                    const int lrem = icpg - r * Y26_ICGRP;
                    const int lmax = (lrem < Y26_ICGRP) ? lrem : Y26_ICGRP;
                    for (int p = 0; p < Y26_OCPACK_P; ++p) {
                        #pragma HLS UNROLL
                        ap_int<32> s = 0;
                        for (int l = 0; l < Y26_ICGRP; ++l) {
                            #pragma HLS UNROLL
#ifdef Y26_TAPLANE
                            // TAPLANE permuted wbuf: tap t == tl_kh*c.kw + tl_kw of channel cc lives
                            // at bank tl_kh*Y26_TPCG + cc, offset tl_kw. This sum is the ACTIVATION
                            // ZERO-POINT correction, and it is the reason the first cut of this
                            // lever failed 5 of 6 eligible convs while `0.conv` PASSED: 0.conv's
                            // input is the image, zero point 0, so its wrong kwsum was multiplied
                            // by zero. Any wbuf reader outside the MAC has to be permuted too.
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
        }   // XB64: pack_cap > 1 for every supported conv, so the 512-lane reduction below is gone
#else
        } else {
            // Not packable this conv (icpg > Y26_ICGRP, or grouped/depthwise) - the ORIGINAL
            // Y26_LANES-wide single-group reduction, unchanged, just addressed at kwsum_o[0].
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

        // W8 ROW FUSION: the row loop now steps in blocks of Y26_ROWS. Every loop inside pays its
        // pipeline fill once per BLOCK rather than once per row. `R` is the live rows in this block
        // - the last block is short whenever Y26_ROWS does not divide OH, and every loop below is
        // bounded by R rather than Y26_ROWS so no dead row is ever computed or stored.
        // single-trip: the outer ohbo loop already selected this block
        for (int ohb = ohbo; ohb < ohbo + y26_Rblk && ohb < OH; ohb += y26_Rblk) {
            const int Rrem = OH - ohb;
            const int R    = (Rrem < y26_Rblk) ? Rrem : y26_Rblk;
            // accrow is unpartitioned now, so zero it sequentially - O(OW) once per output row is
            // negligible against the MAC pass's O(icpg*kh*kw*OW) and keeps the array single-ported.
            // FUSED: one flat R*OW pass instead of R separate OW passes, so the fill is paid once.
            // W4: OWP pads the row to a whole number of Y26_EPI_WIDE-groups so no group straddles
            // a row boundary and every group is bank-perfect (header note at Y26_EPI_WIDE). At
            // Y26_EPI_WIDE == 1 this is OWP == OW and the loop is the historical one.
            const int OWP = ((OW + Y26_EPI_WIDE - 1) / Y26_EPI_WIDE) * Y26_EPI_WIDE;
            // C3: Y's DRAM ROW STRIDE. The store comment at the bottom of this function calls the
            // padded stride "a HOST-SIDE LAYOUT CONTRACT and not a kernel decision" and leaves it
            // "as a knob". This is the knob. Off, YS == OW and every address below is byte-for-byte
            // what shipped. On, rows are OWP apart -- the SAME padding ybuf already uses -- which
            // makes every row start Y26_YPE-aligned for any Y26_YPE <= Y26_EPI_WIDE, and that is
            // the whole of what caps Y26_YPE at 4 today (OW=20 is not a multiple of 8; OWP=24 is).
            // THE HOST MUST READ Y AT STRIDE OWP. The tb does not, so a bit-exact gate CANNOT be
            // run against this build without the matching host change -- see pwr/PREDICT_C3.txt.
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
                        // zw is a multiple of Y26_EPI_WIDE, so lane n is bank n - a COMPILE-TIME
                        // bank, which is the whole reason this widening is mux-free.
                        if (zw + n < OW) {
#ifdef Y26_OCPACK
                            // W3: zero every LIVE plane of this group. p is compile-time unrolled
                            // (small, Y26_OCPACK_P) and predicated on the runtime `pack` for this
                            // group, same masking idiom as the MAC pass's dead-lane guard.
                            for (int p = 0; p < Y26_ACCP; ++p) {
                                #pragma HLS UNROLL
                                if (p < pack) Y26_ACCROW(p, zr, zw + n) = 0;
                            }
#else
                            Y26_ACCROW(0, zr, zw + n) = 0;   // W10b: accwrow is gone, one plane left
#endif
                        }
                    }
                    zw += Y26_EPI_WIDE;
                    if (zw >= OWP) { zw = 0; ++zr; }
                }
            }

            // --- zero-point pass: DELETED by W10b ---
            // It used to run here, R*ktap*OW adds writing accwrow, entered as a separate
            // pipeline region once per live (rr,kh,kw). Measured at 48% of the whole frame
            // (86.9 ms of adds + 70.1 ms of region-entry overhead, notes 1y.45/47). Its
            // result is pure geometry and is now reconstructed inside the epilogue's
            // existing II=1 loop from kwsum_o and the zpo0/zpo1 intervals - see the
            // W10b note at the accrow declaration.

            // --- MAC pass: lanes run along the INPUT CHANNEL axis (restructured 2026-08-17) ---
            //
            // Order is (icb, kh, kw) outside, output column `ow` innermost and SEQUENTIAL. Each cycle
            // the Y26_LANES lanes read Y26_LANES different input channels at the SAME spatial
            // position, multiply by their own weights, and a combinational adder tree folds the
            // products into one accrow[ow].
            //
            // Why this schedules at II=1 where the column-parallel form stalled at 8: channel c lives
            // in bank c % Y26_LANES, the tile step is exactly Y26_LANES, and ic0 == 0 whenever
            // groups == 1 - so lane l always reads bank l, a literal index once UNROLL has run. HLS
            // can prove non-conflict, which retires the 17-bank crossbar and its 34 address muxes.
            // A second benefit falls out: all lanes share ONE within-bank offset, because they differ
            // only by bank, so there is a single address computation instead of Y26_LANES of them.
            //
            // BIT-EXACTNESS: the adder tree reassociates the sum over input channels. That is exact
            // for integers - the reason the accumulator is ap_int and not float, per the header note -
            // so the stage-1 gate stays an EXACT-equality gate and applies here unchanged. Do NOT
            // extend this reasoning to the dequant below, which is order-sensitive.
            // ================= FLATTENED DENSE MAC PASS (lever A, 2026-08-18) =================
            // One pipeline entry per OUTPUT ROW instead of one per (icb, kh, kw) tap.
            //
            // WHY. Measured 2026-08-18 (notes 1k): this design pays one UNHIDDEN DRAM round-trip
            // per LOOP ENTRY, and dense-path entries are rows*npass*kh*kw = 2,006,420 per frame.
            // At a realistic 100-cycle DDR latency that is pure stall, and cosim cannot see it
            // (its AXI slave answers with delay 0). Flattening cuts dense entries to rows =
            // 373,860 and also retires the ~55-cycle fill/drain the old nest paid per tap.
            //
            // WHAT CHANGED. The old nest could not be flattened by HLS because of two things, and
            // both are now handled by predication instead:
            //   * `if (ih < 0 || ih >= H) continue;`  -> folded into `tok`
            //   * the variable inner bound [o0,o1)    -> folded into `tok`
            // Skipped taps now cost a predicated no-op iteration each. That is a few iterations
            // per BORDER tap only (interior rows have every kh valid), and it is exactly what
            // buys the flattening.
            //
            // WALK ORDER: ow OUTERMOST, tap innermost. The first cut of this lever had it the
            // other way (tap outer, ow inner, to mirror the nest it replaced) and csynth scheduled
            // it at II=2: `accrow[ow] += s` was loop-carried, OW is a runtime value so HLS could
            // not prove the reuse distance, and the load->add->store path measured 3.205 ns
            // against a 2.920 ns budget. Putting ow outermost turns the accumulation into a
            // REGISTER recurrence and leaves accrow with exactly one pure STORE per ow - no load,
            // no dependence. II=1, and the timing violation goes with it.
            //
            // BIT-EXACTNESS. The set of (tap, ow) pairs reaching the accumulator is identical to
            // the old nest's; the ORDER is not - this walk is ow-major where the nest was
            // tap-major. That is a reassociation of an INTEGER sum, which is exact, and it is the
            // same argument the lane adder tree above already relies on. The stage-1 EXACT-equality
            // gate therefore still applies unchanged. Do NOT extend it to the dequant below, which
            // is order-sensitive.
#ifdef Y26_OCPACK
#ifdef Y26_A1
            // A1: xbuf AND wbuf are ICGRP-banked for EVERY g1 conv (see y26_xmod at the top and
            // wgrp at :518, both keyed on pack_cap, not pack). So every g1 conv MUST come through
            // this pass. Gating on `pack` instead sent a short FINAL oc group (pack < P) to the
            // unpacked branch below, which reads both buffers with the Y26_LANES modulus - wrong
            // bank and wrong row. MEASURED: exactly the convs with oc % P != 0 and icpg > ICGRP.
            // acc[p]/ACCROW for p >= pack are dead (the epilogue stops at p < pack), so running
            // this pass at pack == 1 is dead work, not a wrong answer.
            if (g1) {
#else
            if (g1 && pack > 1) {
#endif
                // ================= W3: OC-PACKED DENSE MAC PASS (2026-08-20) =================
                // `pack` real output channels oc..oc+pack-1 run through this SAME tap-walk at once.
                // pack > 1 is only ever set when icpg <= Y26_ICGRP (see the oc-loop's pack_cap
                // computation), so npass is guaranteed 1 here - unlike the unpacked branch below,
                // there is no icbi/channel-block odometer axis at all.
                //
                // BANK ARITHMETIC: p is UNROLLED (0..Y26_OCPACK_P-1, a compile-time constant loop),
                // and l is UNROLLED (0..Y26_ICGRP-1). `p*Y26_ICGRP + l` is therefore a per-unrolled-
                // instance COMPILE-TIME index into wbuf - the same "index is the loop variable after
                // UNROLL" property the base design relies on for Y26_LANES banking, just with an
                // extra compile-time addend. See the header note for why this requires p to be
                // unrolled rather than a runtime loop variable.
                //
                // ACTIVATION IS BROADCAST, NOT REPLICATED: `xo` does not depend on p, so xbuf[l][xo]
                // is read ONCE per (l, cycle) and fans out combinationally to all Y26_OCPACK_P
                // multiplies that use it. xbuf's staging loop and layout are UNTOUCHED by this
                // branch - see the header note for why icpg <= Y26_ICGRP guarantees the existing
                // staging already placed channel l at bank l for every l in [0, icpg).
#ifdef Y26_A1
                // A1: npass is no longer 1. Channel block icbi covers channels
                // [icbi*Y26_ICGRP, +Y26_ICGRP), lives at xbuf offset icbi*xb_hw by the re-banking
                // at the top of this function, and at wbuf row icbi*ktap by the staging loop's
                // existing `row = channel / wgrp` arithmetic (which was already general - only
                // this MAC pass and the kwsum tree ever assumed npass == 1).
                const int npass = (icpg + Y26_ICGRP - 1) / Y26_ICGRP;
                // TAPLANE: the kh axis moved onto the lanes, so the walk is kw only. npass is 1
                // here by construction (icpg <= Y26_TPCG <= Y26_ICGRP), so ntap loses the whole
                // kh factor and nothing else.
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
                    // All Y26_ICGRP lanes share ONE within-bank offset - they differ only by bank -
                    // so this stays a single address computation, as in the unpacked branch.
                    const y26_idx_t xo = (y26_idx_t)((ap_uint<8>)icbi * xb_hw)
                                       + (y26_idx_t)((ap_uint<11>)ih * (ap_uint<11>)W)
                                       + (y26_idx_t)iw;
#else
                    const int  woff = kh * c.kw + kw;
                    const int  lmax = icpg;
                    const y26_idx_t xo = (y26_idx_t)((ap_uint<11>)ih * (ap_uint<11>)W) + (y26_idx_t)iw;
#endif

#ifdef Y26_LPAIR
                    // L-PAIR (ledger nx, plan 0.8dj; AMD WP486). Members p and p+1 share lane l's
                    // activation x, so ONE DSP48E2 computes ((w_p << 18) + w_p1) * x. |w_p1 * x| <=
                    // 32,640 < 2^17, so lo = sext(P[17:0]) = w_p1*x and hi = P[35:18] + P[17] = w_p*x,
                    // exactly: same integers, same reassociable sum.
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
                        // Column validity is shared by every lane (all taps of this iteration sit in
                        // the same column); ROW validity is per sub-group, because lane t carries
                        // row tap kh == t. `t` and `cc` below are unroll constants.
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
                            // Address only forms for a row we will actually use, so an out-of-band
                            // tap never indexes xbuf out of range - the mask is not doing that job.
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
                                    // The weight offset is the COLUMN tap alone - the staging loop
                                    // above already put row tap tp in this bank.
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
                                // Dead lanes of a partial block contribute 0, not a stale bank
                                // value - same mask idiom as the unpacked branch below.
                                const y26_wt_t w = (l < lmax) ? wbuf[p * Y26_ICGRP + l][woff]
                                                               : (y26_wt_t)0;
                                s += (ap_int<32>)(xbuf[l][xo] * w);
                            }
                        }
                        acc[p] += s;
                    }
#endif

#ifdef Y26_A1
                    // Same odometer as the unpacked branch: kw fastest, then kh, then the channel
                    // block. ow advances only when a pixel's whole tap set is spent, which is
                    // exactly when acc[] is complete.
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
#ifndef Y26_XB64   // XB64: every g1 conv takes the packed MAC above (A1), so this arm is dead
            if (g1) {
                // groups == 1  =>  ic0 == 0  =>  bank(lane l) == l, statically provable.
                const int npass = (icpg + Y26_LANES - 1) / Y26_LANES;
                const int ntap  = npass * ktap;          // taps accumulated per output pixel
                // W8: R rows in ONE flat loop. This is the 91.0 ms half of §1y.18's fill/drain -
                // the pipeline is filled once per BLOCK, not once per row.
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
                            // Native-width multiply: ap_uint<8> * ap_int<8> resolves to an exact
                            // ap_int<16>. Only the ACCUMULATION is 32-bit. Casting the operands to
                            // ap_int<32> first asks for a 32x32 multiplier and gets fabric instead
                            // of a DSP, with an identical product.
                            const y26_wt_t w = (l < lmax) ? wbuf[l][woff] : (y26_wt_t)0;
                            s += (ap_int<32>)(xbuf[l][xo] * w);
                        }
                    }
                    acc += s;

                    // Tap odometer: kw fastest, then kh, then the channel block. ow advances only
                    // when a pixel's whole tap set is spent, which is exactly when acc is complete.
                    if (++t == ntap) {
                        Y26_ACCROW(0, rr, ow) = acc;  // pure store - this is the row's ONLY write to [ow]
                        acc  = 0;
                        t    = 0;
                        icbi = 0; kh = 0; kw = 0;
                        // W8: ow wrapping advances the ROW within the block. ih0 steps by sh rather
                        // than being recomputed, so the odometer stays adds-and-compares and II=1
                        // has no multiply in its recurrence.
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
                // ---- W12: DEPTHWISE, FUSED + PACKED --------------------------------------------
                // Reached only when pack_cap > 1 on a non-g1 conv, which by the pack_cap rule above
                // means icpg == 1: pure depthwise.
                //
                // WHY, MEASURED (1y.55-1y.57, three cosim points on THIS path, held-out at +0.94%):
                //   K = 60.45 cycles per pipeline-region ENTRY, G = 230.0 per oc-group. Unfused and
                //   unpacked, the 8 deployed depthwise convs cost 174.3 ms - MORE than all 94 dense
                //   convs combined - and 78% of that is entries, not arithmetic. This loop collapses
                //   R*ktap entries per (oc-group, row-block) to ONE, and does P output channels per
                //   pass, so the entry count and the group count both divide by P.
                //   The W8 note below, that this path "carries no measured latency", predates every
                //   one of those measurements. Retiring it is the point of this change.
                //
                // Shape is the g1 packed odometer above, with one difference: a dense member sums
                // Y26_ICGRP input channels, a depthwise member has exactly ONE, so the inner lane
                // reduction collapses to a single multiply and the bank becomes per member.
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
                    // DWP: the activation buffer is banked at modulus Y26_DWP for this conv and
                    // ic0 is a multiple of the pack width, so member p's channel ic0+p sits at
                    // bank p - a COMPILE-TIME CONSTANT, the same statically-provable form the dense
                    // packed MAC gets from xbuf[l]. The channel-group offset is the same for every
                    // member, so all Y26_DWP reads share ONE address computation. No clamp is
                    // needed on the bank: p is always a real bank. A dead member (p >= pack, only
                    // possible if oc is not a multiple of Y26_DWP) folds a real channel into an
                    // accumulator the epilogue never reads - dead work, not a wrong answer.
                    const y26_idx_t xo = (y26_idx_t)((ap_uint<8>)(ic0 / Y26_DWP) * xb_hw) + rowo;
                    for (int p = 0; p < Y26_DWP; ++p) {
                        #pragma HLS UNROLL
                        // wgrp == 1 on this path, so the staging loop's `bank_off = p * wgrp` put
                        // member p's nine taps at bank p.
                        const y26_wt_t w = wbuf[p * Y26_WPE][woff];
                        if (tok) acc[p] += (ap_int<32>)(xbuf[p][xo] * w);
                    }
#else
                    for (int p = 0; p < 1; ++p) {
                        #pragma HLS UNROLL
                        // ocpg == 1 here, so member p's single input channel is ic0 + p. The clamp
                        // matters only for a short final group (oc not a multiple of Y26_OCPACK_P):
                        // without it a dead member would address past the staged channel rows of
                        // xbuf. Its accumulator is never read - the epilogue stops at p < pack - so
                        // folding member zero's data into it is dead work, not a wrong answer.
                        const int       ic  = ic0 + ((p < pack) ? p : 0);
                        const int       bnk = ic & (Y26_LANES - 1);
                        const y26_idx_t xo  = (y26_idx_t)((ap_uint<8>)(ic / Y26_LANES) * xb_hw) + rowo;
                        // icpg == 1 => this member's only lane sits at the base of its bank group,
                        // which is where the staging loop's `bank_off = p * wgrp` put it.
                        const y26_wt_t  w   = wbuf[p * Y26_ICGRP][woff];
                        if (tok) acc[p] += (ap_int<32>)(xbuf[bnk][xo] * w);
                    }
#endif
                    // Tap odometer: kw fastest, then kh. A pixel's taps are spent exactly when its
                    // accumulator is complete, which is when the row slot is written - same shape as
                    // the g1 loop, so the same reasoning about padding and stride carries over.
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
            { }   // XB64: grouped icpg > 1 is unsupported; the fallback below read banks >= ICGRP
#else
            // W8: the grouped/depthwise fallback is NOT fused - it keeps a plain per-row loop.
            // Every grouped conv in this model is depthwise (icpg == 1) and all 8 are OUTSIDE the 94
            // deployed-dense convs, so fusing here would add odometer complexity to a path that
            // carries no measured latency. Correctness is what matters on this branch; speed is not.
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

                        // Per-lane weights for this (channel group, tap), hoisted into registers so
                        // the inner loop reads no memory except xbuf. Lanes past icpg are zero-filled,
                        // so a partial final tile contributes nothing without needing a predicate
                        // inside the MAC itself.
                        //
                        // NOTE: this reads BRAM (wbuf), not m_axi - staging landed 2026-08-18, see the
                        // wbuf note at the top of the function. It used to be Y26_LANES unbursted
                        // scalar AXI reads at ~6 cycles each (MEASURED ~98 cycles per (icb, tap) tile
                        // at 16 lanes), which is what made the kernel DRAM-latency sensitive: +75.6%
                        // cycles at 20-cycle read latency before `latency=30` closed it.
                        //
                        // STILL TRUE AFTER STAGING: this hoist runs once per OUTPUT ROW, because it
                        // sits inside the `oh` loop. Staging removed the redundant DRAM TRAFFIC (each
                        // weight is now fetched once per conv, not OH times) but not these ITERATIONS
                        // - roughly 1.6 cycles per lane per (icb, tap) tile, which is the dominant
                        // term in the lane-invariant floor A of the L(n) = A + B/n fit. Removing them
                        // needs the `oh` loop moved INSIDE the tap loops so each staged weight is used
                        // for every output row before being replaced (weight-stationary). That costs
                        // an OH x OW accumulator plane in place of accrow[OW] (~102 KB, ~23 BRAM36 at
                        // 160x160, affordable given BRAM sits flat at 1,049/1,824 through 512 lanes).
                        // Do not attribute that cycle win to staging - they are separate changes.
                        // WIDTH: keep these at the NATIVE int8 weight width (y26_wt_t == ap_int<8>).
                        // They were ap_int<32> until 2026-08-17, which made the MAC below a 32x32
                        // multiply for what is arithmetically int8 x uint8 - so the multipliers were
                        // built in FABRIC instead of DSP48E2, and LUT scaled at ~1,300 per lane
                        // (21,145 at 16 lanes -> 42,148 at 32) while 2,418 of 2,520 DSPs sat idle.
                        // The values provably fit: Wt is int8 by construction. Do not widen these
                        // back "for safety" - widen the ACCUMULATOR (`s`), which is already 32-bit.
                        //
                        // NO LOOP. Since wbuf is banked by lane and the tile step is exactly
                        // Y26_LANES, lane l reads bank l at a single shared offset - so the whole
                        // tile's weights land in ONE cycle, fully unrolled, instead of Y26_LANES
                        // sequential reads. This is the payoff of the banking, and it is why the
                        // hoist no longer appears in the cost model at all.
                        //
                        // HISTORY, because two earlier shapes here were both wrong in ways that
                        // measured plausibly:
                        //   * `for l < Y26_LANES` with a PREDICATED LOAD suppressed the read but not
                        //     the ITERATION, so a partial final tile cost one cycle per dead lane.
                        //     MEASURED (icpg=128, 8x8): 128 lanes = 387,244 cycles, 256 lanes =
                        //     518,316 - WORSE than 64 lanes (463,020). The gap is 131,072 = exactly
                        //     1 cyc x 128 dead lanes x 1024 rows.
                        //   * `for l < lmax` reading a FLAT wbuf fixed that, but serialised 64 reads
                        //     through one BRAM port and stalled ~3.5 cycles each at runtime while
                        //     still reporting II=1 (857,272 cycles vs a 457,004 baseline).
                        // Both are retired by the form below. Do not reintroduce a loop here.
                        //
                        // Why over-provisioning is now free: icpg ranges 32..512 across this model,
                        // and dead lanes are masked to 0 combinationally rather than costing a
                        // cycle each, so one large Y26_LANES can serve every layer.
                        y26_wt_t wl[Y26_LANES];
                        #pragma HLS ARRAY_PARTITION variable=wl complete

                        // Dead lanes read 0, not a stale bank value. The mask is what lets the MAC
                        // below stay unpredicated over all Y26_LANES (0*x = 0). Do NOT drop it and
                        // write only [0,lmax) - wl would keep the previous tile's weights in the
                        // dead lanes and the stage-1 bit-exact gate would fail.
                        const int lrem = icpg - icb;
                        const int lmax = (lrem < Y26_LANES) ? lrem : Y26_LANES;
                        const int woff = (icb / Y26_LANES) * ktap + kh * c.kw + kw;
                        for (int l = 0; l < Y26_LANES; ++l) {
                            #pragma HLS UNROLL
                            wl[l] = (l < lmax) ? wbuf[l][woff] : (y26_wt_t)0;
                        }

                        {
                            // Grouped. Every grouped conv in this model is depthwise (icpg == 1), so
                            // this loop runs once and there is no channel parallelism to extract - a
                            // property of the layer, not a scheduling failure. Written for general
                            // icpg anyway so a future non-depthwise grouped conv stays CORRECT (it
                            // would simply be slow). The bank is runtime here, but each iteration
                            // issues a single read, so it costs one mux and no initiation interval.
                            //
                            // BOUND BY lmax, NOT Y26_LANES (2026-08-22, notes 1y.55). The old form
                            // ran `l < Y26_LANES` with an `icl < icpg` guard inside. That guard IS
                            // `l < lmax`: icb+l < icpg <=> l < icpg-icb, and l < Y26_LANES already
                            // holds, so this is the same predicate moved into the bound - not a
                            // behaviour change. But the guarded form still SPENT the iteration.
                            // READ OUT OF THE GENERATED FSM (not modelled, not simulated): a dead
                            // lane traversed state458 -> state462 -> state458 = 2 cycles, because
                            // ap_block_state462_on_subcall_done is ANDed with the guard and ap_start
                            // is asserted only in state461, which a dead lane never enters.
                            // At icpg == 1 that is 511*2 = 1,022 wasted cycles per (oc, row, tap):
                            // 392,055,552 cycles = 1,553.7 ms over the 8 deployed grouped convs,
                            // more than 9x the entire 94-conv dense frame.
                            // NOT the retired `l < lmax` of the history note above - THAT one was
                            // the WEIGHT-LOAD loop serialising a flat wbuf through a single BRAM
                            // port. wl[] here is complete-partitioned registers, a 512:1 mux either
                            // way, so no memory port is involved and that failure cannot recur.
                            for (int l = 0; l < lmax; ++l) {
                                #pragma HLS LOOP_TRIPCOUNT min=1 max=Y26_LANES avg=1
                                const int       icl  = icb + l;
                                const int       ic   = ic0 + icl;
                                const int       bnk  = ic & (Y26_LANES - 1);
                                const y26_idx_t xrow =
                                    (y26_idx_t)((ap_uint<8>)(ic / Y26_LANES) * xb_hw) + rowoff;
#ifdef Y26_C5DW
                                // C5-DW (section 2.6): the depthwise MAC is 31.5% of the measured
                                // frame and runs ONE multiply per cycle at icpg == 1 -- 0.2% lane
                                // occupancy. The starved axis is CHANNEL; the spatial axis is not
                                // starved, so take f = Y26_EPI_WIDE columns per cycle instead of
                                // one. accrow is already ARRAY_PARTITION cyclic factor=Y26_EPI_WIDE
                                // on the column axis (:319/:334), so 8 consecutive ow land in 8
                                // distinct banks REGARDLESS of alignment (bank = ow % 8, and eight
                                // consecutive indices hit all eight banks exactly once). Each ow is
                                // its own accumulator -- no reduction across columns, no carried
                                // dependence. And xbuf is already RESHAPE cyclic factor=8 on dim 2
                                // with sw == 1 on every depthwise conv, so the eight reads are one
                                // or two word accesses, not eight.
                                //
                                // STRIP-MINED DELIBERATELY, not `#pragma HLS UNROLL factor=8` on
                                // the pipelined loop. UNROLL and PIPELINE on the SAME loop leaves
                                // the precedence to an HLS heuristic, and runs A/A' (0.8af) were
                                // both no-ops that LOOKED like wins because a heuristic moved
                                // underneath them. This form means what it says whichever way the
                                // heuristic goes. The `owu < o1` guard is the remainder tail:
                                // OW % 8 == 4 for the two 20x20 convs (2.0.0, 2.1.0). The index set
                                // is identical to the original by construction -- the union over
                                // ow = o0, o0+8, ... of {ow+u : u < 8, ow+u < o1} is exactly
                                // [o0, o1), each element once. That is an identity, not a heuristic.
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
                                    // Same native-width multiply as the g1 branch above.
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

            // --- dequant + activation (stage 1: still float, on purpose) ---
            // real = wsc * (step * Sum(w_int*q) + lo * Sum_valid(w_int)) + bias.
            // The VALID-tap weight sum is what makes padded borders contribute real 0.0 rather than
            // `lo`; folding lo into a constant bias errs at borders and propagates.
            // Promotions below are ordered to match conv2d() exactly: float*double -> double, the
            // sum stays double, the wsc multiply stays double, then ONE cast to float, then += bias
            // in float. Reordering these changes the result.
            // W8: FUSED over the whole block. This is the 42.6 ms half of §1y.18's fill/drain -
            // ~195k epilogue row passes, each paying ~55 cycles to refill a pipeline that then runs
            // for OW iterations. Output rows of one output channel are CONTIGUOUS in Y, so the
            // destination index just increments monotonically across all R*OW elements and no
            // per-row base address has to be recomputed.
            // W4b: the epilogue's destination. At Y26_YPE == 1 this is the historical direct m_axi
            // pointer and nothing below changes. Above 1, `gmem_out` is a wide port that cannot take
            // a single float, so the epilogue lands in a BLOCK-LOCAL buffer and a separate loop
            // packs it out Y26_YPE floats at a time. `ybase + owl` is a BLOCK-RELATIVE flat index in
            // both cases, which is exactly why one index expression serves both.
            // W3: the epilogue+store below is written per REAL output channel, exactly as before
            // packing existed. Rather than thread a pack dimension through ybuf/Y26_YOUT/the store
            // odometer (all delicately tuned for a single-oc invocation - see the W4b bank-index
            // note just below), this runs the WHOLE unmodified block `pack` times, once per real
            // member. Every state item here (ybuf, the odometer counters, wbase) is either static
            // scratch or declared fresh each pass, so repeating the block is behaviorally identical
            // to the oc-loop calling it once per real oc, which is what it already did.
#if defined(Y26_EFLAT_ON)
            // ---- L-EFLAT (ledger pa) -------------------------------------------------------------
            // WHAT THIS COSTS TODAY, read off the shipping csynth and not fitted to anything:
            //   VITIS_LOOP_1764_37  iteration latency 7728, trip 17, Pipelined: NO
            //   VITIS_LOOP_1827_38  iteration latency   27, II 1, trip 7680, total 7705
            // 1827 is 1764's ONLY child, so 7728 - 7705 = 23 cycles per member are spent in this
            // prologue with the loop un-pipelined, and the depth-27 inner pipeline drains 26 more.
            // 49 cycles per (block, MEMBER). Fused, the prologue folds into the II=1 pipeline and the
            // drain is paid once per (block, GROUP): 49 * nblk * (oc - ocgn) = 889,007 cyc = 3.60 ms
            // gross on the 19,020,564-cycle YDIR frame.
            //
            // THE KNOWN RISK IS THE BURST, NOT THE CYCLE COUNT (ji / 0.8at). cpp:1827 writes
            // m_axi_gmem_out with burst Inferred because `ywbase + gi` is strictly monotonic. Fused,
            // the address jumps by OH*YS/Y26_YPE every y26_NI iterations - piecewise linear, which is
            // EXACTLY the shape that cost C2 +98 cyc/(grp,blk) when it lost the burst. Netted at C2's
            // own rate this is still +766,115 cyc = 3.10 ms, and break-even needs 784 per grp-blk,
            // 8x worse than C2 ever paid. READ THE csynth BURST ROW FOR m_axi_gmem_out BEFORE READING
            // THE CYCLE COUNT: if it says Fail/"Could not analyze pattern", that is the verdict.
            //
            // `ywbase` is stepped by an ADD and never recomputed. The product (oc+p)*OH + ohb) * YS
            // only ever advances by OH*YS per member, and a per-iteration multiply inside an II=1
            // pipeline is a new hazard for no reason. Same reasoning as the existing `ybase`, whose
            // comment says it tracks er*OW "without a multiply".
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
                const float step_op = step_o[p];      // W12: per member now, see the declaration
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
            // ---- THE BANK INDEX IS A DIMENSION, NOT AN ARITHMETIC FACT (measured, §1y.39) --------
            // The first W4b attempt declared this FLAT and cyclic-partitioned by Y26_YBANK, reasoning
            // that any 8 consecutive indices occupy 8 distinct banks. That is TRUE and it is USELESS:
            // the index is `ybase + ow + n` with ybase and ow both runtime, so HLS cannot PROVE which
            // bank lane n hits and wires all 8 writers to all 8 banks -
            //
            //   WARNING: [HLS 200-448] Lower bound of II is 4 due to multiple operations accessing
            //                          core:RAM:...ap_uint_1 {8 stores at conv_engine.cpp:778}
            //
            // - i.e. exactly the §1y.35 failure (`dst0 = cg*hw`) in a new place, and for the third
            // time the fix is the same: MAKE THE DIVISOR PART OF THE TYPE. `ybuf[group][lane]` with
            // dim 2 complete-partitioned means lane n IS bank n, syntactically, with nothing left to
            // prove. Do not "simplify" this back to a flat array with a cyclic pragma.
            //
            // The group index is just the epilogue's iteration counter: the loop steps `i` by
            // Y26_EPI_WIDE over the OWP-PADDED block, so group == i / Y26_EPI_WIDE == a counter that
            // increments by one. That is only true while Y26_YBANK == Y26_EPI_WIDE, hence the guard.
#ifndef Y26_YDIRECT_ON
            static float ybuf[Y26_YBUF_GRP][Y26_YBANK];
            #pragma HLS ARRAY_PARTITION variable=ybuf complete dim=2
#endif
#else
            float* const yr = &Y[yblk];
#endif
            // Not lane-tiled: this loop is the DEQUANT, and it is the stage-2 target (-> ap_fixed).
            // Parallelising it now would bake in a shape before its arithmetic type is settled, and its
            // cost is O(OW) against the MAC loop's O(icpg*kh*kw*OW) - it is not the bottleneck.
            // (W4 is what widens THIS loop; W8 only removes its per-row refill. They are separate
            // halves of the same term and the plan's 32.4 ms needs BOTH - §1y.18.)
            // W4: same OWP group structure as the zeroing pass above. `ybase` tracks er*OW without
            // a multiply (the historical code got this for free from a monotonic `i`; the padded
            // odometer no longer coincides with the output index, so it is carried explicitly).
            // W4b: one name for the destination element, so the three dequant variants below stay
            // single-sourced. `gi` is the group counter described at the ybuf declaration.
#ifdef Y26_YQ8
            // HW lever B: the slot's payload - the float, or (yq) the consumer's code as a 32-bit integer.
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
                // W10b: per-COLUMN-TAP sum over the kh's that are live for this output row. Every
                // lane below shares `er` (Y26_EPI_WIDE divides OWP, so a group never straddles a
                // row), so this is computed ONCE per iteration, not once per lane. `p` is fixed by
                // the enclosing per-real-channel loop. Both bounds are Y26_MAX_K (3), so the whole
                // thing unrolls into <=9 predicated adds - no array, no loop, no new region.
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
                            // exactly the deleted pass's guard: `if (ih < 0 || ih >= H) continue;`
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
                // W10b: this lane's zero-point accumulator, in place of the deleted accwrow read.
                // Each kw contributes over a contiguous column interval, so this is <=3 predicated
                // adds of values already computed above.
                ap_int<32> zpw = 0;
                for (int kwi = 0; kwi < Y26_MAX_K; ++kwi) {
                    #pragma HLS UNROLL
                    if (owl >= zpo0[kwi] && owl < zpo1[kwi]) zpw += zpcs[kwi];
                }
#ifdef Y26_FX_DEQUANT
                // STAGE 2: fixed-point scale arithmetic - the real hardware rounding.
                // Deliberately NOT bit-exact; this is the step whose cost must be MEASURED (cosine +
                // deviation now, mAP delta once the trunk exists), never assumed to be free.
                // The accumulators stay INTEGER and are consumed by a multiply. Casting them into a
                // value-width ap_fixed first is the bug that collapsed cosine to 0.064 - see the
                // two-type rationale in conv_engine.h. `scale * ap_int` promotes to a wide fixed
                // result, which is then narrowed once, at the assignment.
#ifdef Y26_FUSE_ON
                // L-FUSE: step_op / lo_op are already wsc*step and wsc*lo (folded at staging), so
                // the wsc multiply and the 80-bit saturate that followed it are simply gone. The
                // accumulators are narrowed to the 27-bit DSP port; the assert is the bound check
                // and costs nothing in synthesis.
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
                                 + (y26_fx_t)(s_lo   * zpw);                    // W10b: reconstructed, was accwrow
                y26_fx_t v = (y26_fx_t)(s_wsc * t) + (y26_fx_t)bs_p;
#endif
#ifdef Y26_SILU_LUT
                // Fully fixed-point tail: no float/double core is instantiated for the activation.
                if (c.act == 1) v = y26_silu_fx(v);
                Y26_YOUT(n, owl) = Y26_YQ(v.to_float());
#else
                float vf = v.to_float();
                if (c.act == 1) vf = y26_silu(vf);   // still double std::exp - see y26_silu_fx
                Y26_YOUT(n, owl) = Y26_YQ(vf);
#endif
#else
                // STAGE 1 (default): float dequant, so the kernel stays BIT-EXACT vs conv2d() and the
                // exact-equality gate keeps guarding the integer accumulator. Promotion order below
                // matches conv2d() exactly - float*double -> double, one cast to float, then += bias.
                const double a  = (double)Y26_ACCROW(p, er, owl).to_int64();   // exact: accumulator is integral
                const double aw = (double)zpw.to_int64();      // W10b: reconstructed, was accwrow
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
                    Y[ywbase + gi] = word;   // gi == the store loop's `w`, proven at the decl above
                }
#endif
                ow += Y26_EPI_WIDE;
                ++gi;
                if (ow >= OWP) { ow = 0; ++er; ybase += (y26_idx_t)OW; }
#ifdef Y26_EFLAT_ON
                // the third digit: retire a member, re-zero the row/group digits, step the store
                // base by one output map. This is the ONLY place `p` advances under EFLAT.
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
            // ---- W4b: THE STORE ------------------------------------------------------------------
            // One m_axi beat per Y26_YPE outputs. This loop, not the epilogue above it, is what
            // actually sets the store-side throughput: the epilogue can be as wide as we like and
            // still cannot retire faster than the port drains.
            //
            // WHY Y26_YPE IS CAPPED AT 4 AND NOT 8, WHICH COSTS ~3.5 ms - the honest reason. A beat
            // of Y26_YPE floats must be Y26_YPE-aligned in Y, and Y's rows are OW apart. This network
            // has OW in {20,40,80,160,320}: every one is a multiple of 4, but **20 is not a multiple
            // of 8**. At Y26_YPE=8 the 20x20 convs would need a beat that straddles a row boundary,
            // and a straddling group reads two different `ybuf` rows - which is the bank collision
            // that OWP padding exists to prevent. 8 is reachable only by padding Y's DRAM row stride,
            // which is a HOST-SIDE LAYOUT CONTRACT and not a kernel decision. Left as a knob.
            //
            // The odometer walks OUTPUT elements (stride OW) while indexing ybuf in PADDED units
            // (stride OWP), which is why `pb` is carried separately from `wbase + w`. Since OWP is a
            // multiple of Y26_YBANK and OW is a multiple of Y26_YPE, the lane offset `p % Y26_YBANK`
            // is always 0 or 4 and all Y26_YPE lanes of a beat live in ONE group - so this is a
            // single array read with a small lane mux, not a gather. The padded tail columns are
            // never read: the odometer only ever visits real output columns.
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
            }   // p - W3 per-real-output-channel epilogue+store repeat
#endif
        }
        oc += pack;
    }
    }   // ohbo
#undef Y26_ACCROW
#undef Y26_AFB
}
