// yolo26_hls.cpp - stage 1 of the synthesizable trunk datapath: integer accumulator.
//
// Mirrors yolo26_utils.h's conv2d() SmoothQuant (asymmetric uint8) path exactly, with two changes:
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
#include "yolo26_hls.h"
#include <cmath>

// ---------------------------------------------------------------------------------------------
// Local copies of the two conv2d helpers. Deliberately duplicated rather than shared: yolo26_utils.h
// drags in <vector>/<string>/weights_loader.h, none of which may cross into a synthesizable TU.
// These are byte-for-byte the same arithmetic.
// ---------------------------------------------------------------------------------------------

// Output-column range [o0,o1) over which tap column `base` lands inside the image. Expressing the
// zero padding as a loop bound instead of a per-element predicate keeps the branch out of the
// datapath - it is also what makes the innermost loop a clean II=1 candidate.
static inline void y26_ow_range(int base, int sw, int W, int OW, int& o0, int& o1) {
    o0 = base >= 0 ? 0 : (-base + sw - 1) / sw;
    o1 = W - 1 - base;
    o1 = o1 < 0 ? 0 : o1 / sw + 1;
    if (o1 > OW) o1 = OW;
}

// SiLU in double, rounded to float - identical to yolo26_utils.h's silu(), which is the closest
// portable match to torch's float32 SiLU.
// NOTE for synthesis: std::exp on double is legal in HLS but expensive. Replacing this with a
// fixed-point LUT is a stage-2/3 decision and MUST be gated on mAP delta, not on the bit-exact gate.
static inline float y26_silu(float v) { double d = v; return (float)(d / (1.0 + std::exp(-d))); }

#ifdef Y26_SILU_LUT
#include "yolo26_silu_lut.h"
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

void y26_conv2d_hls(const y26_xw_t* X, int H, int W,
                    const y26_ww_t*  Wt,
                    const Y26ConvCfg& c,
                    y26_yw_t* Y) {
    const int OH   = y26_oh(H, c);
    const int OW   = y26_ow(W, c);
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
    // Layout (see yolo26_hls.h): channel c -> bank c % Y26_LANES, offset (c/Y26_LANES)*H*W + y*W + x.
    // `complete dim=1` makes the LANE axis a set of independent memories whose index is the literal
    // loop variable after UNROLL - that, and not the mapping arithmetic, is what buys II=1.
    static y26_act_t xbuf[Y26_LANES][Y26_ACT_LANE_ELEMS];
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
    for (int ic = 0; ic < c.ic; ++ic) {
        const int         l    = ic & (Y26_LANES - 1);          // bank  (Y26_LANES is a power of 2)
        const ap_uint<8>  cg   = (ap_uint<8>)(ic / Y26_LANES);  // channel-group within the bank
        const y26_idx_t   dst0 = (y26_idx_t)(cg * hw);
        const y26_idx_t   src0 = (y26_idx_t)((ap_uint<11>)ic * hw);
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
        const y26_idx_t hww   = (y26_idx_t)(hw   / Y26_XPE);
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
        }
#else
        for (y26_idx_t j = 0; j < hw; ++j) {
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
    ap_int<32> kwsum_o[Y26_OCPACK_P][Y26_MAX_K * Y26_MAX_K];
    #define Y26_KWSUM(p_, t_) kwsum_o[p_][t_]
#else
    ap_int<32> kwsum_o[Y26_MAX_K * Y26_MAX_K];
    #define Y26_KWSUM(p_, t_) kwsum_o[t_]
#endif

    // Per-output-row accumulators, one per OUTPUT COLUMN. The loop interchange that produced them is
    // still load-bearing (it is what keeps the reduction out of a loop-carried dependency), but the
    // PARALLELISM no longer comes from this axis - see the banking note in yolo26_hls.h. As of
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
    ap_int<32> accrow [Y26_OCPACK_P][Y26_ROWS][Y26_MAX_OW];
  #if Y26_EPI_WIDE > 1
    // Same cyclic-by-epilogue-width partition as the non-packed case, now on dim=3 (the OW axis
    // moved down one slot because of the new leading PACK dimension).
    #pragma HLS ARRAY_PARTITION variable=accrow  cyclic factor=Y26_EPI_WIDE dim=3
  #endif
    // One reference macro for both shapes, so every existing accrow/accwrow call site below reads
    // Y26_ACCROW(p, rr, ow) regardless of which struct is live. Under !Y26_OCPACK this collapses to
    // the historical accrow[rr][ow] with p silently dropped (always called with p==0 there) - a
    // pure rename, not a behavior change. Mirrors the Y26_YOUT macro used at the epilogue below.
    #define Y26_ACCROW(p_, r_, o_)  accrow[p_][r_][o_]
#else
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
    const int pack_cap = ((g1 || icpg == 1) && icpg <= Y26_ICGRP) ? Y26_OCPACK_P : 1;
#endif

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
        float step_o[Y26_OCPACK_P];
        float lo_o[Y26_OCPACK_P];
        #pragma HLS ARRAY_PARTITION variable=step_o complete
        #pragma HLS ARRAY_PARTITION variable=lo_o   complete
        // Per-member dequant scale/bias. Filled by the staging loop below, one member at a time.
        float wsc_o[Y26_OCPACK_P];
        float bs[Y26_OCPACK_P];
#else
        const float step_o = c.perch ? c.step_v[ic0] : c.step;
        const float lo_o   = c.perch ? c.lo_v[ic0]   : c.lo;
        const float wsc_o  = c.wsc[oc];
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
  #if Y26_WPE > 1
    #error "Y26_OCPACK does not yet compose with the wide weight port (W9, Y26_WPE>1) - the multi-write proof for wbl0*Y26_WPE+b has not been checked with a runtime p*Y26_ICGRP base added in. See the header note. Build without -DY26_WT_WORD/-DY26_WT_TAPMAJOR while Y26_OCPACK is on."
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
        const int wgrp = (pack_cap > 1) ? Y26_ICGRP : Y26_LANES;
        for (int p = 0; p < pack; ++p) {
            const int       ocp   = oc + p;
            const y26_idx_t wbase = (y26_idx_t)((y26_out_t)ocp * (ap_uint<13>)wlen);
            wsc_o[p] = c.wsc[ocp];
            bs[p]    = c.bias ? c.bias[ocp] : 0.f;
            // W12: per-member step/lo. Under !perch these collapse to the same conv-wide scalar for
            // every p, so the dense packed path is bit-identical to before this change.
            const int ic0p = (ocp / ocpg) * icpg;
            step_o[p] = c.perch ? c.step_v[ic0p] : c.step;
            lo_o[p]   = c.perch ? c.lo_v[ic0p]   : c.lo;
            const int bank_off = p * wgrp;
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
            const int nrows = (icpg + Y26_ICGRP - 1) / Y26_ICGRP;
            for (int p = 0; p < Y26_OCPACK_P; ++p) {
                #pragma HLS UNROLL
                for (int t = 0; t < ktap; ++t) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=Y26_MAX_K*Y26_MAX_K avg=1
                    ap_int<32> s = 0;
                    for (int r = 0; r < nrows; ++r) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=Y26_MAX_IC avg=4
                        ap_int<32> ptmp = 0;
                        for (int l = 0; l < Y26_ICGRP; ++l) {
                            #pragma HLS UNROLL
                            ptmp += (r * Y26_ICGRP + l < icpg) ? (ap_int<32>)wbuf[p * Y26_ICGRP + l][r * ktap + t]
                                                                : (ap_int<32>)0;
                        }
                        s += ptmp;
                    }
                    kwsum_o[p][t] = s;
                }
            }
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
        for (int ohb = 0; ohb < OH; ohb += Y26_ROWS) {
            const int Rrem = OH - ohb;
            const int R    = (Rrem < Y26_ROWS) ? Rrem : Y26_ROWS;
            // accrow is unpartitioned now, so zero it sequentially - O(OW) once per output row is
            // negligible against the MAC pass's O(icpg*kh*kw*OW) and keeps the array single-ported.
            // FUSED: one flat R*OW pass instead of R separate OW passes, so the fill is paid once.
            // W4: OWP pads the row to a whole number of Y26_EPI_WIDE-groups so no group straddles
            // a row boundary and every group is bank-perfect (header note at Y26_EPI_WIDE). At
            // Y26_EPI_WIDE == 1 this is OWP == OW and the loop is the historical one.
            const int OWP = ((OW + Y26_EPI_WIDE - 1) / Y26_EPI_WIDE) * Y26_EPI_WIDE;
            {
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
                            for (int p = 0; p < Y26_OCPACK_P; ++p) {
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
            if (g1 && pack > 1) {
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
                const int total = R * OW * ktap;

                int        ow = 0, t = 0, rr = 0;
                int        ih0 = ohb * c.sh - c.ph;
                int        kh = 0, kw = 0;
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
                    const bool tok = (ih >= 0 && ih < H && iw >= 0 && iw < W);
                    const int  woff = kh * c.kw + kw;
                    const y26_idx_t xo = (y26_idx_t)((ap_uint<11>)ih * (ap_uint<11>)W) + (y26_idx_t)iw;

                    for (int p = 0; p < Y26_OCPACK_P; ++p) {
                        #pragma HLS UNROLL
                        ap_int<32> s = 0;
                        if (tok) {
                            for (int l = 0; l < Y26_ICGRP; ++l) {
                                #pragma HLS UNROLL
                                // Dead lanes of a partial group (icpg < Y26_ICGRP) contribute 0, not
                                // a stale bank value - same mask idiom as the unpacked branch below.
                                const y26_wt_t w = (l < icpg) ? wbuf[p * Y26_ICGRP + l][woff]
                                                               : (y26_wt_t)0;
                                s += (ap_int<32>)(xbuf[l][xo] * w);
                            }
                        }
                        acc[p] += s;
                    }

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
                }
            } else
#endif
            if (g1) {
                // groups == 1  =>  ic0 == 0  =>  bank(lane l) == l, statically provable.
                const int npass = (icpg + Y26_LANES - 1) / Y26_LANES;
                const int ntap  = npass * ktap;          // taps accumulated per output pixel
                // W8: R rows in ONE flat loop. This is the 91.0 ms half of §1y.18's fill/drain -
                // the pipeline is filled once per BLOCK, not once per row.
                const int total = R * OW * ntap;

                int        ow = 0, t = 0, rr = 0;
                int        ih0 = ohb * c.sh - c.ph;       // advances by sh as rr advances
                int        icbi = 0, kh = 0, kw = 0;
                ap_int<32> acc = 0;

                for (int i = 0; i < total; ++i) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=16 max=(Y26_ROWS*20736) avg=1024
                    const int  ih  = ih0 + kh;
                    // Offset stays SIGNED until known in range: kw - pw can be negative, so the
                    // bounds test must run on the signed value before any cast to y26_idx_t.
                    const int  iw  = ow * c.sw + kw - c.pw;
                    const bool tok = (ih >= 0 && ih < H && iw >= 0 && iw < W);

                    ap_int<32> s = 0;
                    if (tok) {
                        const int woff = icbi * ktap + kh * c.kw + kw;
                        const int lrem = icpg - icbi * Y26_LANES;
                        const int lmax = (lrem < Y26_LANES) ? lrem : Y26_LANES;
                        // All lanes share ONE within-bank offset - they differ only by bank - so
                        // this is a single address computation, not Y26_LANES of them.
                        const y26_idx_t xo = (y26_idx_t)((ap_uint<8>)icbi * hw)
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
#ifdef Y26_OCPACK
            if (pack_cap > 1) {
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
                int ih0 = ohb * c.sh - c.ph;
                ap_int<32> acc[Y26_OCPACK_P];
                #pragma HLS ARRAY_PARTITION variable=acc complete
                for (int p = 0; p < Y26_OCPACK_P; ++p) {
                    #pragma HLS UNROLL
                    acc[p] = 0;
                }
                for (int i = 0; i < total; ++i) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=16 max=(Y26_ROWS*2880) avg=1024
                    const int  ih   = ih0 + kh;
                    const int  iw   = ow * c.sw + kw - c.pw;
                    const bool tok  = (ih >= 0 && ih < H && iw >= 0 && iw < W);
                    const int  woff = kh * c.kw + kw;
                    const y26_idx_t rowo = (y26_idx_t)((ap_uint<11>)ih * (ap_uint<11>)W)
                                         + (y26_idx_t)iw;
                    for (int p = 0; p < Y26_OCPACK_P; ++p) {
                        #pragma HLS UNROLL
                        // ocpg == 1 here, so member p's single input channel is ic0 + p. The clamp
                        // matters only for a short final group (oc not a multiple of Y26_OCPACK_P):
                        // without it a dead member would address past the staged channel rows of
                        // xbuf. Its accumulator is never read - the epilogue stops at p < pack - so
                        // folding member zero's data into it is dead work, not a wrong answer.
                        const int       ic  = ic0 + ((p < pack) ? p : 0);
                        const int       bnk = ic & (Y26_LANES - 1);
                        const y26_idx_t xo  = (y26_idx_t)((ap_uint<8>)(ic / Y26_LANES) * hw) + rowo;
                        // icpg == 1 => this member's only lane sits at the base of its bank group,
                        // which is where the staging loop's `bank_off = p * wgrp` put it.
                        const y26_wt_t  w   = wbuf[p * Y26_ICGRP][woff];
                        if (tok) acc[p] += (ap_int<32>)(xbuf[bnk][xo] * w);
                    }
                    // Tap odometer: kw fastest, then kh. A pixel's taps are spent exactly when its
                    // accumulator is complete, which is when the row slot is written - same shape as
                    // the g1 loop, so the same reasoning about padding and stride carries over.
                    if (++t == ktap) {
                        for (int p = 0; p < Y26_OCPACK_P; ++p) {
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
            // W8: the grouped/depthwise fallback is NOT fused - it keeps a plain per-row loop.
            // Every grouped conv in this model is depthwise (icpg == 1) and all 8 are OUTSIDE the 94
            // deployed-dense convs, so fusing here would add odometer complexity to a path that
            // carries no measured latency. Correctness is what matters on this branch; speed is not.
            for (int rr = 0; rr < R; ++rr) {
            const int ih0 = (ohb + rr) * c.sh - c.ph;
            for (int icb = 0; icb < icpg; icb += Y26_LANES) {
                for (int kh = 0; kh < c.kh; ++kh) {
                    const int ih = ih0 + kh;
                    if (ih < 0 || ih >= H) continue;
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
                                    (y26_idx_t)((ap_uint<8>)(ic / Y26_LANES) * hw) + rowoff;
                                for (int ow = o0; ow < o1; ++ow) {
                                    #pragma HLS PIPELINE II=1
                                    const y26_idx_t xo = xrow + (y26_idx_t)(ow * c.sw + base);
                                    // Same native-width multiply as the g1 branch above.
                                    Y26_ACCROW(0, rr, ow) += (ap_int<32>)(xbuf[bnk][xo] * wl[l]);
                                }
                            }
                        }
                    }
                }
            }
            }   // rr - grouped fallback

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
#ifdef Y26_OCPACK
            for (int p = 0; p < pack; ++p) {
                const int   ocp   = oc + p;
                const float wsc_op = wsc_o[p];
                const float bs_p   = bs[p];
                const float step_op = step_o[p];      // W12: per member now, see the declaration
                const float lo_op   = lo_o[p];
#else
            {
                const int   p     = 0;
                const int   ocp   = oc;
                const float wsc_op = wsc_o;
                const float bs_p   = bs;
                const float step_op = step_o;
                const float lo_op   = lo_o;
#endif
            const y26_idx_t yblk = (y26_idx_t)((y26_out_t)(ocp * OH + ohb) * (ap_uint<11>)OW);
#if Y26_YPE > 1
            // ---- THE BANK INDEX IS A DIMENSION, NOT AN ARITHMETIC FACT (measured, §1y.39) --------
            // The first W4b attempt declared this FLAT and cyclic-partitioned by Y26_YBANK, reasoning
            // that any 8 consecutive indices occupy 8 distinct banks. That is TRUE and it is USELESS:
            // the index is `ybase + ow + n` with ybase and ow both runtime, so HLS cannot PROVE which
            // bank lane n hits and wires all 8 writers to all 8 banks -
            //
            //   WARNING: [HLS 200-448] Lower bound of II is 4 due to multiple operations accessing
            //                          core:RAM:...ap_uint_1 {8 stores at yolo26_hls.cpp:778}
            //
            // - i.e. exactly the §1y.35 failure (`dst0 = cg*hw`) in a new place, and for the third
            // time the fix is the same: MAKE THE DIVISOR PART OF THE TYPE. `ybuf[group][lane]` with
            // dim 2 complete-partitioned means lane n IS bank n, syntactically, with nothing left to
            // prove. Do not "simplify" this back to a flat array with a cyclic pragma.
            //
            // The group index is just the epilogue's iteration counter: the loop steps `i` by
            // Y26_EPI_WIDE over the OWP-PADDED block, so group == i / Y26_EPI_WIDE == a counter that
            // increments by one. That is only true while Y26_YBANK == Y26_EPI_WIDE, hence the guard.
            static float ybuf[Y26_YBUF_GRP][Y26_YBANK];
            #pragma HLS ARRAY_PARTITION variable=ybuf complete dim=2
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
#if Y26_YPE > 1
            #define Y26_YOUT(n_, owl_) ybuf[gi][(n_)]
#else
            #define Y26_YOUT(n_, owl_) yr[ybase + (owl_)]
#endif
            int er = 0, ow = 0, gi = 0;
            y26_idx_t ybase = 0;
            for (int i = 0; i < R * OWP; i += Y26_EPI_WIDE) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=(Y26_ROWS*Y26_MAX_OW) avg=64
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
                // two-type rationale in yolo26_hls.h. `scale * ap_int` promotes to a wide fixed
                // result, which is then narrowed once, at the assignment.
                const y26_scale_t s_step = step_op;
                const y26_scale_t s_lo   = lo_op;
                const y26_scale_t s_wsc  = wsc_op;
                const y26_fx_t t = (y26_fx_t)(s_step * Y26_ACCROW(p, er, owl))
                                 + (y26_fx_t)(s_lo   * zpw);                    // W10b: reconstructed, was accwrow
                y26_fx_t v = (y26_fx_t)(s_wsc * t) + (y26_fx_t)bs_p;
#ifdef Y26_SILU_LUT
                // Fully fixed-point tail: no float/double core is instantiated for the activation.
                if (c.act == 1) v = y26_silu_fx(v);
                Y26_YOUT(n, owl) = v.to_float();
#else
                float vf = v.to_float();
                if (c.act == 1) vf = y26_silu(vf);   // still double std::exp - see y26_silu_fx
                Y26_YOUT(n, owl) = vf;
#endif
#else
                // STAGE 1 (default): float dequant, so the kernel stays BIT-EXACT vs conv2d() and the
                // exact-equality gate keeps guarding the integer accumulator. Promotion order below
                // matches conv2d() exactly - float*double -> double, one cast to float, then += bias.
                const double a  = (double)Y26_ACCROW(p, er, owl).to_int64();   // exact: accumulator is integral
                const double aw = (double)zpw.to_int64();      // W10b: reconstructed, was accwrow
                float v = (float)(wsc_op * (step_op * a + lo_op * aw)) + bs_p;
                if (c.act == 1) v = y26_silu(v);
                Y26_YOUT(n, owl) = v;
#endif
              }
                ow += Y26_EPI_WIDE;
                ++gi;
                if (ow >= OWP) { ow = 0; ++er; ybase += (y26_idx_t)OW; }
            }
            #undef Y26_YOUT
#if Y26_YPE > 1
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
                const int       nw    = (R * OW) / Y26_YPE;
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
                    if (sc >= OW) { sc = 0; pb += OWP; }
                }
            }
#endif
            }   // p - W3 per-real-output-channel epilogue+store repeat
        }
        oc += pack;
    }
#undef Y26_ACCROW
}
