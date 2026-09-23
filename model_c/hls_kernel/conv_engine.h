// conv_engine.h - synthesizable W8A8 conv datapath for the YOLO26s pruned50 trunk.
//
// This is NOT a retype of the Tensor C-sim. layer_ops.h stays the golden model: it is
// std::vector<float>-backed, the dumps are float32, and five Python gates read them. Mutating its
// element type would cascade destructively (see Step 1 item 2 in notes/yolo26s-hls-next-steps.md).
// Instead this header declares a SEPARATE kernel that shares the *arithmetic* but is synthesizable:
// no STL, no heap, no throw, static bounds, ap_int accumulation.
//
// Staging (deliberate, and the gates differ):
//   stage 1 (this file) - integer accumulator -> ap_int<32>. Integer accumulation is exact and
//                         reduction-order-safe, so this is BIT-EXACT against conv2d(): the gate is
//                         exact float equality on the output, and it is compiler-invariant (so it
//                         holds under MSYS2 g++ here as well as the server's toolchain).
//   stage 2 (later)     - the dequant/scale path -> ap_fixed<>. NOT bit-exact (it is the real
//                         hardware rounding of the scale arithmetic); gate on cosine + mAP delta
//                         with ~0.001 mAP50 as the noise floor, never md5.
//
// Build (real Vitis headers, not htdet's stubs):
//   g++ -O2 -std=c++14 -I"C:/AMDDesignTools/2026.1/Vitis/include" ...
#pragma once
#include <ap_int.h>
#include <ap_fixed.h>

// ---------------------------------------------------------------------------------------------
// Bounds - measured from weights_sq_compact_fold/manifest_sq.txt (2026-08-06), not estimated.
// Kept as macros so the contingency in design target 3 (tile layers 0/1/2 into 2 H-strips) and the
// MAC-lane sweep on the ZU6EG proxy stay one-line parameter changes rather than a redesign.
// ---------------------------------------------------------------------------------------------
#ifndef Y26_MAX_OW
#define Y26_MAX_OW    320    // layer 0 output row (640 stride-2); ties the 160x160 activations
#endif
#ifndef Y26_MAX_K
#define Y26_MAX_K     3      // no 5x5 conv in the model - SPPF's k5 is a maxpool, not a conv
#endif
#define Y26_MAX_DEPTH 2304   // max (ic/groups)*kh*kw, at 22.m.0.0.cv1.conv
#define Y26_MAX_OC    512
#define Y26_MAX_IC    663

// ---------------------------------------------------------------------------------------------
// Step 1 item 3 - MAC lanes.
//
// Y26_LANES concurrent MAC chains. This is the knob that spends the DSP budget (design target 1:
// size for >=2000 lanes, not 512).
//
// WHICH AXIS IS PARALLEL - corrected 2026-08-17. This used to read "parallelism is taken on the `ow`
// axis: Y26_LANES columns are computed concurrently". That was true of the original loop interchange
// and is NO LONGER the main path. The lanes now run over the INPUT CHANNEL axis, because that is the
// axis whose index can also be the activation bank index - which is what took II from 8 to 1. The
// column-parallel form survives only as the fallback for icpg < Y26_LANES (i.e. depthwise). See the
// banking note below before changing either.
//
// It is a PARAMETER on purpose. Sweep it - synthesize, confirm II=1 and timing, step it, and read how
// resources scale.
//
// SWEPT TO COMPLETION 2026-08-18, POST-ROUTE on XCZU9EG. The "BRAM is the wall" caution that used to
// sit here was CSYNTH-derived and is WRONG: post-route BRAM18K is 1,048 / 1,049 / 1,049 / 1,049 at
// 64 / 128 / 256 / 512 lanes - FLAT. It is entirely the fixed activation buffer and does not respond
// to lane count at all. At 512 lanes the part is 7.03% LUT, 22.0% DSP, 57.5% BRAM, timing MET with
// the best margin of any run. **NOTHING BINDS. The constraint is the dataflow, not the device.**
//
// USEFUL LANES ARE CAPPED BY icpg, NOT BY THE PART. Lanes run over input channels, so a layer uses at
// most icpg of them; across this model `ic` runs 3..663 with the mass at 32/64/128/256 (plus 14
// depthwise convs at icpg == 1). Above ~256 most layers idle. Design target 1's ">=2000 lanes" is
// therefore UNREACHABLE on this axis and needs a second parallel axis (output channels or columns)
// or INT8 DSP packing - not a bigger part.
//
// Since 2026-08-18 the weight-hoist loop is bounded by the LIVE channel count, so over-provisioning
// is free rather than linear (it used to cost 1 cycle per dead lane per row: 256 lanes measured
// WORSE than 64 before the fix). 128 is the measured knee - lowest power of any config (1.542 W).
// Do not raise it further to chase latency: at 128 the lane-scalable term is only 19% of the total,
// while the lane-INVARIANT floor is 81%, and that floor is what design target 2 attacks.
//
// Still true: raising lanes shrinks Y26_ACT_LANE_ELEMS (= Y26_ACT_BUF_ELEMS / Y26_LANES), tightening
// the per-bank budget that the stem already exceeds.
//
// MUST BE A POWER OF TWO. Since 2026-08-17 the lane index is also the activation bank index
// (channel c -> bank c % Y26_LANES), and the staging loop relies on `& (Y26_LANES-1)` / `/ Y26_LANES`
// lowering to a mask and a shift. A non-power-of-two would silently reintroduce a runtime divider on
// the staging path. (The old requirement - "divide evenly into the ARRAY_PARTITION factor" - referred
// to the cyclic-prime scheme that has since been removed; see the banking note below.)
#ifndef Y26_LANES
#define Y26_LANES     16
#endif

// ---------------------------------------------------------------------------------------------
// W3 (2026-08-20) - OC-PACKING. Run Y26_OCPACK_P output channels concurrently through the dense
// MAC pass instead of one, for convs whose icpg is small enough that most lanes go idle today.
//
// WHY THIS EXISTS. 93 of 94 deployed-dense convs have npass==1 (icpg <= Y26_LANES), which means the
// MAC loop's trip count is INDEPENDENT of lane count - lanes past icpg multiply a zero-masked
// weight and do no useful work. §1y.42 found the plan that assumed this was already exploited
// ("W3") was never built; the honest routed-kernel throughput without it is ~2.0 FPS, not the
// 12.3 FPS figure that got retracted the same day.
//
// WHY OC-PACKING AND NOT OW-PACKING. The natural read is "pack idle lanes with extra OUTPUT
// COLUMNS" (more `ow` per cycle). That needs the ACTIVATION to vary per packed lane - a big,
// per-channel plane - replicated into a runtime-selected destination bank, which is exactly the
// O(LANES^2) crossbar this file has scar tissue about (see the accwrow note ~30 lines below this
// one, and the G0b/dst0w note in the .cpp). OC-packing inverts which operand varies: packed lane p
// wants output channel oc+p, so only the WEIGHT differs per p (small, already BRAM-staged data);
// the ACTIVATION READ IS IDENTICAL for every p in the group and is issued ONCE per cycle, fanned
// out combinationally to all P weight-multiplies. Replicating weights is cheap. Replicating
// activations is not. This is the whole reason W3 is buildable at all.
//
// WHY A FIXED, COMPILE-TIME P AND NOT AN ADAPTIVE PACK FACTOR. An adaptive `pack = LANES/icpg`
// (varying per conv) was the first design and IS NOT SAFE: the bank a packed lane's weight lives
// in is `p*group_width + ic_local`, and if `group_width` itself varies per conv (a RUNTIME value),
// that product is a runtime multiply feeding a `complete`-partitioned array's write/read index -
// the same class of failure as the dst0/hw finding two sections down, just rediscovered here.
// Fixing Y26_OCPACK_P (and therefore Y26_ICGRP = Y26_LANES/Y26_OCPACK_P) at COMPILE TIME, and then
// fully UNROLLING both the p axis (0..P-1) and the ic_local axis (0..Y26_ICGRP-1), makes
// `p*Y26_ICGRP + ic_local` a per-unrolled-instance COMPILE-TIME CONSTANT - literally the same
// "index is the loop variable after UNROLL" property Y26_LANES banking already relies on, just
// with an extra compile-time-constant addend. That is what makes it free.
//
// WHY PACKING ONLY FIRES WHEN icpg <= Y26_ICGRP (never a wider net). Splitting the 512 lanes into
// P groups of Y26_ICGRP each necessarily narrows the per-group channel-reduction width from
// Y26_LANES to Y26_ICGRP. For a conv with icpg > Y26_ICGRP that would INCREASE npass beyond 1,
// which increases the required xbuf per-bank depth by the same factor - and xbuf is already the
// dominant BRAM consumer (~1,024 of the ~1,101-1,150 routed blocks), so that growth is not
// affordable at any P worth having. Restricting packing to icpg <= Y26_ICGRP keeps npass at
// EXACTLY 1 for every conv that packs, so xbuf's required depth never changes and xbuf itself is
// UNTOUCHED by this feature - see the .cpp MAC pass note for why the existing staging layout
// already puts channel ic_local at bank ic_local for these convs, with no staging-loop edit needed.
//
// COVERAGE / COST, per hls/model/_w3fixed.py (2026-08-20), fixed-P + fallback-to-unpacked when
// icpg > Y26_ICGRP:
//   P=2 (grp=256): 86/94 packed,  3.7 FPS,  +36  accrow/accwrow BRAM blocks @ ROWS=16
//   P=4 (grp=128): 66/94 packed,  6.1 FPS,  +71  accrow/accwrow BRAM blocks @ ROWS=16  <- shipped
//   P=8 (grp=64):  41/94 packed,  6.2 FPS,  +142 accrow/accwrow BRAM blocks @ ROWS=16
//   P=16/32 regress (BRAM cost outruns coverage). P=4 is the chosen point: best FPS per block, and
//   the cheapest of the two similar-FPS options. This is NOT the 10-11 FPS an earlier (WRONG)
//   adaptive-pack sketch predicted - that sketch is exactly the unsafe design two paragraphs up.
//
// WHAT DOES NOT COMPOSE YET. Y26_WPE>1 (W9, the wide weight port) is NOT verified safe under
// Y26_OCPACK: the multi-write proof for W9's `wbl0*Y26_WPE+b` pattern has not been checked with a
// runtime `p*Y26_ICGRP` base added in. Build with the 8-bit weight port until that is measured -
// the #error guard near the weight-staging code enforces this. Y26_XPE (activation wide port) and
// Y26_EPI_WIDE/Y26_YPE (epilogue widening, output wide port) are UNAFFECTED and compose freely -
// neither touches xbuf's layout nor the per-real-output-channel epilogue, which W3 still runs once
// per real oc (just called `pack` times per group instead of once per group).
#ifdef Y26_OCPACK
  #ifndef Y26_OCPACK_P
  #define Y26_OCPACK_P 4
  #endif
  #if (Y26_LANES % Y26_OCPACK_P) != 0
    #error "Y26_OCPACK_P must divide Y26_LANES evenly, or the group width is not an integer."
  #endif
  #define Y26_ICGRP (Y26_LANES / Y26_OCPACK_P)
#endif

// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// TAPLANE (2026-09-16, plan 0.8dn/0.8do) - PUT THE kh TAPS ON THE IDLE LANES.
//
// WHY.  The dense group is indexed by INPUT CHANNEL only, so a conv with icpg < Y26_ICGRP leaves
// Y26_ICGRP-icpg lanes dead AND still walks its k^2 taps one per cycle.  Measured on the 99-conv
// frame (0.8dn): dense MACs fill 58.6% of issue slots and 2.50 BILLION slots are idle lanes.
// `0.conv` - the single biggest conv in the network, 10.4% of the frame - runs 95.3% idle-lane.
//
// THE MAPPING.  Split the Y26_ICGRP lanes into Y26_TP groups of Y26_TPCG channels.  Lane
// l = t*Y26_TPCG + cc carries CHANNEL cc at ROW TAP kh == t.  Both t and cc are UNROLL CONSTANTS
// (Y26_TPCG is compile-time), which is the same property the whole engine's banking rests on.
// One iteration then retires all c.kh row taps at once and the loop walks only kw: the tap axis
// shrinks from kh*kw to kw.
//
// WHY THE ACTIVATION IS REPLICATED AND THE WEIGHT IS PERMUTED.  Lane (t,cc) needs x[cc] at row
// ih0+t: same channel, different row, so a single bank cannot serve two t in one cycle - the copy
// is forced by PORT COUNT, not by capacity.  It is free: icpg <= Y26_TPCG means banks
// [icpg, Y26_ICGRP) are already instantiated by XB64 and sitting dead, and every copy uses the
// SAME depth.  Zero extra BRAM, zero extra DRAM traffic (staging writes Y26_TP banks from the one
// word it already read).  The weights are not replicated at all, only PERMUTED: tap (t,kw) of
// channel cc moves from bank cc offset t*kw+kw to bank t*Y26_TPCG+cc offset kw.
//
// WHY THE ROW SHIFT IS AT READ TIME, NOT IN STORAGE.  Storing copy t pre-shifted by -t rows would
// be the obvious form and it is the wrong one: the destination address stops being a multiple of
// Y26_XPE and HLS answers with the barrel shifter + serialising dependency measured at 1y.35
// (staging II 1 -> 8, 576,098 LUT).  Keeping storage identical and adding t*W to the READ address
// costs Y26_TP distinct addresses across Y26_ICGRP banks - one adder per lane, LUT not BRAM - and
// the code comment at the MAC's `xo` about "ONE within-bank offset" becomes "Y26_TP of them".
// That is the price this lever pays and the reason csynth must be read before it is believed.
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
  // The other half of this check - Y26_TPCG being a multiple of the weight PORT WORD, which is what
  // keeps the permuted weight destination word-aligned - lives next to Y26_WPE below, because
  // Y26_WPE is defined further down this header and an #if on an undefined macro reads as 0.
  // A conv is eligible iff its channels fit ONE lane sub-group and its row taps fit Y26_TP.
  #define Y26_TL_OK(icpg_, kh_) ((icpg_) <= Y26_TPCG && (kh_) > 1 && (kh_) <= Y26_TP)
#else
  #define Y26_TL_OK(icpg_, kh_) false
#endif

// DWP (2026-09-07) - DEPTHWISE OUTPUT-CHANNEL PACKING.  Y26_DWP members per pass.
//
// WHY.  Depthwise is 39.0% of the frame and 89.9% of that is MAC ISSUE AT 1 MAC/CYCLE (16,128,000
// of 17,939,912 measured cycles, pwr/FRAME_A1.csv).  W12's fusion already took the entry share from
// 78% to 10%, so the only lever left on this block is width.  The remaining 10% is NOT a fixed
// floor: it is PER CHANNEL (23.one2one_cv3.0.1.0 at oc=128 is 2.0000x 23.one2one_cv3.0.0.0 at
// oc=64), so packing members divides BOTH terms.  See notes 0.8ce.
//
// WHY THE BANK INDEX IS A COMPILE-TIME CONSTANT, WHICH IS THE WHOLE DESIGN.  A depthwise member
// needs its OWN input channel (unlike a dense member, which shares all Y26_ICGRP of them), so the
// naive form indexes xbuf at a RUNTIME bank ic0+p and builds the Y26_LANES:1 crossbar the whole
// engine is written to avoid.  MEASURED, on the DWLIT probe's filed negative control
// (pwr/PREDICT_DWLIT_RT.txt): making that index runtime cost +352 BRAM18 and +40,734 LUT.
// The fix is to stage the ACTIVATIONS at modulus Y26_DWP.  Then channel c sits at bank
// c % Y26_DWP, member p's channel is ic0+p, and ic0 is a multiple of the pack width, so
//     bank(member p) == p        - an UNROLL CONSTANT, exactly like the dense MAC's xbuf[l] -
// and the channel-group offset (ic0 / Y26_DWP) is SHARED by every member, one address for all.
// That is the dense packed MAC's structure with Y26_ICGRP replaced by Y26_DWP.
//
// WHY Y26_DWP AND Y26_ICGRP ARE NOT INTERCHANGEABLE.  The modulus above is the PACK WIDTH, not the
// dense group width: the two are equal only at Y26_DWP == Y26_ICGRP.  A narrower Y26_DWP makes
// per-bank capacity ceil(ic/Y26_DWP)*band deeper and the row-block banding (y26_Rblk) absorbs it by
// shrinking R - at Y26_DWP == 16 five of the six depthwise convs fall to R == 3 and
// 23.one2one_cv3.0.1.0 hits the FAIL-OPEN R == 0 clamp.  At Y26_DWP == 64 all six keep R == 16.
// The width is therefore NOT freely tunable downward; check y26_Rblk before moving it.
//
// COST TO WATCH, and it is the reason this is a knob and not a default: accrow is
// [Y26_ACCP][Y26_ROWS][Y26_MAX_OW] and Y26_ACCP scales with the pack width.  Desk model
// (cyclic-by-Y26_EPI_WIDE on the OW axis, mapped 512x36): 80 BRAM18 at 8, 640 at 64.  The DWLIT
// probe reported this FLAT (1,134 -> 1,136 across DWP 1 -> 64) but its weight staging was never
// adapted, so its accumulators were dead and eliminated; that row is not evidence.  Read the real
// number off this build's csynth Storage Report, not off the probe.
#ifndef Y26_DWP
#define Y26_DWP 1                 // 1 == the shipping behaviour, byte-identical (every use is gated)
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

// ---------------------------------------------------------------------------------------------
// W8 (2026-08-19) - ROW FUSION. Output rows processed per pipeline fill.
//
// §1y.18 decomposed the 258.9 ms per-row term and found **133.6 ms of it - 52% - is pipeline
// fill/drain, not work**: ~195k epilogue row passes and ~414k MAC row-block passes, each paying
// ~55 cycles of pure latency to refill a pipeline that then runs for OW iterations. Widening the
// datapath cannot touch any of it (that is W4's half); only amortising the fill across `r` rows can.
//
// Y26_ROWS == 1 is the historical structure EXACTLY - same loop trip counts, same order, same
// arithmetic - so the control build stays bit-identical and every prior measurement reproduces.
//
// Cost is FF (more in-flight state) and BRAM (`accrow`/`accwrow` x Y26_ROWS, ~1.25 KB per row per
// array at Y26_MAX_OW). **Zero LUT**: what changes is a loop bound and an odometer, not a memory
// port count - which is why this is the cheapest FPS per unit area in the plan.
//
// GATE G9 is the whole risk: fusing lengthens the loop-carried dependence, and an II=2 fused loop
// is WORSE than an II=1 unfused one. Read the csynth loop table, never the total.
#ifndef Y26_ROWS
#define Y26_ROWS      1
#endif

// ---------------------------------------------------------------------------------------------
// W4 (2026-08-20) - EPILOGUE WIDTH. Process Y26_EPI_WIDE output elements per cycle in BOTH the
// zeroing pass and the dequant/SiLU/store epilogue.
//
// WHY BOTH LOOPS. G11 (1y.30) measured the per-element slope at 4.950 cyc and decomposed it as
// `ntap + 2` = MAC ntap + zeroing 1.0 + epilogue 1.0. The widely-quoted 2.295 was the epilogue
// PLUS the zeroing pass, never the epilogue alone. Widening only the epilogue buys half the item
// (24.4 ms of 48.7) for nearly all of its cost, because both loops need the SAME accrow/accwrow
// partition. Doing one is the worst trade available.
//
// WHY THE ROW IS PADDED TO A MULTIPLE OF Y26_EPI_WIDE (`OWP` at the loops). accrow/accwrow are
// cyclic-partitioned by Y26_EPI_WIDE on dim 2, so element `ow` lives in bank `ow % Y26_EPI_WIDE`.
// A group of N consecutive elements hits N DISTINCT banks - one access each, NO mux - only if the
// group starts at a multiple of N. If a group is allowed to STRADDLE a row boundary it does not:
// at OW=20, N=8, the last group of a row is {16,17,18,19, 0,1,2,3} of the next, which maps to
// banks {0,1,2,3,0,1,2,3} - a two-way collision on every bank, which HLS resolves by serialising
// to II=2. Padding each row to OWP = ceil(OW/N)*N and predicating the tail lanes keeps every group
// row-local and bank-perfect. Cost is the padding waste: OW=20 at N=8 runs 24 slots for 20
// elements (17%), and OW in {40,80,160,320} wastes nothing. It does NOT reintroduce a per-row
// pipeline fill - the loop stays flat across the whole W8 block, which is the point of W8.
//
// THE THING TO WATCH IS THE MAC's WRITE, NOT THE EPILOGUE's READ. conv_engine.cpp:262 argues
// accrow must NOT be partitioned because it would "cost a 16-way mux/demux on a runtime `ow` index
// while delivering nothing". The second half stops being true here - the epilogue now issues N
// accesses per cycle - but the FIRST half still applies: the MAC's `accrow[rr][ow] = acc` store is
// at a runtime `ow`, so it acquires a 1-of-N demux. That is the same class of construct as the
// O(LANES^2) accwrow crossbar (1j), and it is affordable ONLY because N is 4-16 rather than 256.
// **Read the per-module LUT across two N values before believing any N.** II=1 does not mean a
// memory structure is healthy - that is exactly what 1j got wrong.
//
// Y26_EPI_WIDE == 1 leaves OWP == OW and the odometer degenerate, so the control build is
// unchanged; the bit-exact gate is what proves it.
#ifndef Y26_EPI_WIDE
#define Y26_EPI_WIDE  1
#endif

// ---------------------------------------------------------------------------------------------
// W9 (2026-08-19) - the WEIGHT PORT WORD, and the tap-major weight layout.
//
// This is G0b's finding applied to the OTHER m_axi port. The cost model's "MAC" term is not
// arithmetic at all: sum(oc*icpg*ktap) = 3,568,635 is exactly the weight ELEMENT count, so while
// `Wt` is declared `const y26_wt_t*` the staging loop moves ONE BYTE PER CYCLE and the term is
// lane-invariant - which is precisely why §1y.20 found it unmoved by lane count. HLS derives the
// port width from the POINTER TYPE, so widening the pointer is the only lever.
//
// The two halves are SEPARATE KNOBS so they can be measured apart - the plan predicts 13.9 FPS for
// the port alone, 14.6 for the relayout alone, 15.0 together, and those are three different builds:
//   -DY26_WT_WORD=64     widen the gmem_wt port to 64 bits
//   -DY26_WT_TAPMAJOR    re-order each oc slice from (icl, tap) to (tap, icl)
// Both undefined reproduces every measurement recorded before this date byte-identically.
//
// The macros live HERE, above Y26_WBANK, because the padding below feeds the bank-depth bound.
// The `y26_ww_t` typedef itself sits with the other typedefs further down.
#if defined(Y26_WT_WORD) && (Y26_WT_WORD > 8)
#define Y26_WPE (Y26_WT_WORD / 8)    // weight elements carried per port word
// DWP: member p's single channel is staged at bank p*Y26_WPE (the wide tap-major staging loop
// counts in PORT WORDS, so Y26_WPE is the narrowest group width it can express - a width of 1
// divides to zero and stacks every member on bank 0). That makes Y26_LANES/Y26_WPE the hard
// ceiling on the pack width: at Y26_WT_WORD=64 it is 64, and Y26_DWP=128 wrote past wbuf and
// crashed the tb silently before this guard existed (2026-09-07).
#if defined(Y26_DWP) && (Y26_DWP > 1) && ((Y26_DWP) * (Y26_WPE) > Y26_LANES)
  #error "Y26_DWP * Y26_WPE exceeds Y26_LANES: member p's weights would be staged past wbuf."
#endif
#else
#define Y26_WPE 1
#endif

// TAPLANE: the permuted weight destination steps by Y26_TPCG/Y26_WPE PORT WORDS, so a Y26_TPCG
// that is not a multiple of Y26_WPE divides to a wrong (or zero) sub-group step and stages every
// row tap on top of tap 0 - the same failure mode Y26_DWP=1 hit on 2026-09-07, silent in csynth.
#if defined(Y26_TAPLANE) && (Y26_TPCG % Y26_WPE)
  #error "Y26_TPCG must be a multiple of Y26_WPE: the tap sub-group step is counted in port words."
#endif

// PADDING - not optional once Y26_WPE > 1.
//
// Output channel oc owns the slice [oc*stride, +stride) of Wt. Unpadded, stride = icpg*ktap, which
// is NOT generally a multiple of Y26_WPE (22.cv2.conv: 663*9 = 5967, seven short), so every
// following oc would start MID-WORD and the staging loop would need head/tail epilogues - the exact
// LUT this item exists to avoid spending. Rounding up costs at most Y26_WPE-1 elements per slice
// (per (oc, tap) under TAPMAJOR): under 0.2% of the blob.
//
// Under TAPMAJOR the padding goes on the CHANNEL RUN rather than the whole slice, and that is
// load-bearing for a second reason. It makes `icl` word-aligned at every word boundary, so the
// Y26_WPE elements of one port word land in Y26_WPE banks whose indices differ ONLY in their low
// bits - which is what lets HLS prove the writes conflict-free without building a crossbar.
// **That proof is the whole content of §1y.12's untested claim, and W9 is its cheap test.**
//
// The pad elements are written but never read: both the kwsum tree and the MAC guard on
// `r*Y26_LANES + l < icpg`, and a padded icl lands at a (bank, row) no live channel occupies.
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

// Depth of ONE lane-bank of the staged per-output-channel weight buffer (`wbuf` in the kernel).
//
// A bank holds ceil(icpg / Y26_LANES) rows of ktap taps, and icpg*ktap is bounded by Y26_MAX_DEPTH,
// so Y26_MAX_DEPTH/Y26_LANES rounded up covers the full tiles; the extra Y26_MAX_K^2 covers the
// PARTIAL final tile, whose row exists in every bank even when only some lanes are live. Getting
// this wrong is a silent out-of-bounds write in the staging copy, not a compile error - the g++
// gates would catch it as a mismatch, which is one more reason never to skip them.
//
// Kept small on purpose: at 64 lanes this is 45 bytes per bank, so HLS builds each bank from LUTRAM
// or registers. A BRAM per lane would not fit past a few hundred lanes, and banking is only worth
// having because it is cheap - see the wbuf note in conv_engine.cpp for what the flat version cost.
// W9: padding can push the last channel of a tap into one MORE channel-group than the unpadded
// bound allows (icl reaches icpg + Y26_WPE - 2), which costs one extra row of ktap taps. Collapses
// to the historical expression at Y26_WPE == 1, so the control build is bit-identical.
#if Y26_WPE > 1
#if defined(Y26_A1) && defined(Y26_OCPACK)
// A1: the packed path banks weights by Y26_ICGRP, not Y26_LANES, so a bank holds
// ceil(icpg / Y26_ICGRP) rows of ktap - Y26_OCPACK_P times as many rows as the unpacked form.
// This is wbuf, NOT xbuf: A1's filed capacity analysis only ever checked xbuf and missed this
// array entirely (notes 1y.65/1y.66).
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

// ---------------------------------------------------------------------------------------------
// Step 1 item 5 - activation buffer sizing, from a parameter rather than a literal.
//
// Design target 3 says hold activations WHOLE (no tiling): the untiled high-water mark is 715/912
// BRAM36 = 78.4%, and tiling buys little while risking the weight-refetch trap. So the default is
// Y26_ACT_STRIPS = 1, i.e. no tiling at all.
//
// The contingency, if _csynth.rpt disagrees, is to tile layers 0/1/2 into 2 H-strips (peak -> 597).
// Keeping the divisor here as a macro is what makes that a ONE-LINE change instead of a redesign -
// that is the entire point of this item; it is not an endorsement of tiling.
//
// Largest single activation in the trunk is 1 600 KB: 160x160x64 at layers 1/2/3, tied by 320x320x16
// at layer 0 and 80x80x256 at 16.cv1. Sized by element count so all three shapes are covered.
#ifndef Y26_ACT_STRIPS
#define Y26_ACT_STRIPS 1     // 1 = untiled (the design). 2 = the layers-0/1/2 H-strip contingency.
#endif
#ifndef Y26_ACT_MAX_ELEMS   // ledger oj L2: overridable so LANE_ELEMS can sit AT the xbuf 512-word BRAM18 step (4096)
#define Y26_ACT_MAX_ELEMS  1638400UL                             // 160*160*64 == 320*320*16
#endif
#define Y26_ACT_BUF_ELEMS  (Y26_ACT_MAX_ELEMS / Y26_ACT_STRIPS)  // per-strip resident buffer

// Max INPUT width, needed because layer 0 consumes the 640x640 image (every later layer is <=320).
#define Y26_MAX_IW    640

// ---------------------------------------------------------------------------------------------
// Activation banking - REWRITTEN 2026-08-17. The previous scheme (cyclic factor=17, prime) is gone.
//
// HISTORY, because the discarded idea is seductive and will otherwise be reinvented:
// the old comment argued for a PRIME cyclic factor so that Y26_LANES consecutive output columns,
// read at stride `sw`, always land in distinct banks. That argument is ARITHMETICALLY CORRECT and it
// still failed. csynth reported `HLS 200-448` once per bank, each listing all 16 loads, and held
// II at 8 (= 16 loads / 2 BRAM ports).
//
// The reason is not the mapping, it is what the COMPILER can prove. Cyclic partitioning becomes
// parallel hardware only when the bank index is resolvable at compile time. The address was
// `xrow + o*sw + base` with xrow, sw and base all runtime, so HLS could not prove the lanes hit
// distinct banks and built an arbitrated crossbar instead - 34 eight-way address muxes, ~138k LUT,
// and no parallelism to show for it.
//
// Note especially that PER-LAYER SPECIALIZATION DOES NOT FIX THIS. Making sw/base compile-time
// constants still leaves `addr(l) = R + l*sw` with R runtime, so bank(l) = (R + l*sw) mod 17 is a
// runtime ROTATION, and Vitis will not infer a barrel shifter from a cyclic directive.
//
// THE FIX IS DIMENSIONAL: give the lane axis its own array dimension and partition it completely,
// so the bank index is the literal loop variable after UNROLL.
//
//   xbuf[l][off]   with   l = channel % Y26_LANES   ->   bank index is `l`, a constant per lane
//
// The parallel axis therefore moves from OUTPUT COLUMN to INPUT CHANNEL. That is what makes the
// address provably conflict-free, and it has a second benefit: all Y26_LANES lanes read the SAME
// within-bank offset (they differ only in channel, which selects the bank), so one address is
// broadcast to all banks instead of 16 independent address computations.
//
// Layout: activation (c, y, x)  ->  bank  c % Y26_LANES
//                                   off   (c / Y26_LANES) * H*W + y*W + x
//
// Sizing: because c/Y26_LANES rounds up, a conv needs ceil(ic/LANES)*H*W per bank. For every layer
// with ic >= LANES this lands at or below Y26_ACT_MAX_ELEMS/LANES - the interleave is naturally
// balanced (160x160x64 -> 4*25600; 80x80x256 -> 16*6400; 40x40x512 -> 32*1600; all <= 102,400).
//
// KNOWN LIMITATION - the stem. Layer 0 is 640x640x3: ic=3 < LANES, so only 3 banks are populated and
// the requirement is ceil(3/16)*409,600 = 409,600 per bank, 4x the budget. Sizing for it would cost
// 16*409,600 = 6.5 MB against a 4 MB device. It is NOT guarded at runtime: the Y26_ACT_LANE_ELEMS
// assert is inside `#ifndef __SYNTHESIS__` and is compiled OUT of the RTL, and the y26_Rblk clamp
// fails OPEN. No conv in the shipping manifest reaches it (guard_v41.py). Guarded in SIM only, rather
// than silently mis-addressed. The right answer for a 3-channel stem is a line-buffer + window
// REGISTER file, where taps come out of registers (unlimited read ports) instead of RAM - a separate
// dataflow, and properly the trunk graph's problem. At gate spatial sizes (16-40) layer 0 fits here
// and is exercised normally.
#define Y26_ACT_LANE_ELEMS (Y26_ACT_BUF_ELEMS / Y26_LANES)

// ---------------------------------------------------------------------------------------------
// Stage 2 - the ap_fixed dequant path.
//
// Enabled by -DY26_FX_DEQUANT. OFF by default ON PURPOSE: with it off the kernel keeps the float
// dequant and therefore stays BIT-EXACT against conv2d(), which preserves the stage-1 gate as a
// permanent regression guard on the accumulator. Turning it on trades that exact gate for a
// TOLERANCE gate (cosine + deviation, ultimately mAP delta) - which is the real hardware rounding of
// the scale arithmetic, and is expected to move the numbers slightly.
//
// Width/integer-bit split is parameterized because the correct values come from the per-layer ActMax
// in the ptq/manifest data, and are a tuning exercise against mAP - NOT something to hard-code from a
// first guess. The defaults below are a starting point for measurement, not a validated choice.
// TWO types, not one - learned by getting it wrong on 2026-08-06. A single ap_fixed<32,16> used for
// both the accumulator and the value saturated instantly (16 integer bits = +-32768, against a
// measured max|acc| of 924,719) and collapsed cosine to 0.064. The raw integer accumulator must NEVER
// be narrowed into the value type; it stays ap_int and is consumed by a MULTIPLY, whose product is
// what has activation-like magnitude.
//
//   y26_scale_t - the calibration constants (step, lo, wsc). Tiny magnitudes (step ~2e-2,
//                 lo ~-6.5e-2, wsc smaller still), so almost all the bits should be fractional.
//   y26_fx_t    - the value/intermediate type. Must hold step*acc, which is ~1e4-1e5 before the
//                 wsc multiply brings it back to activation scale - hence far more integer bits than
//                 the final output alone would need.
#ifndef Y26_FX_SCALE_W
#define Y26_FX_SCALE_W 32
#endif
#ifndef Y26_FX_SCALE_I
#define Y26_FX_SCALE_I 8     // range +-128; every calibration constant is well inside this
#endif
#ifndef Y26_FX_W
#define Y26_FX_W 48          // total bits
#endif
#ifndef Y26_FX_I
#define Y26_FX_I 24          // integer bits - must cover step*acc, not just the output
#endif
// AP_RND / AP_SAT: round-to-nearest and saturate rather than wrap. Saturation matters - a wrapped
// overflow in the dequant would be catastrophic and silent, where saturation merely clips. It is also
// what made the width bug above visible as a graceful accuracy collapse rather than as garbage.
typedef ap_fixed<Y26_FX_SCALE_W, Y26_FX_SCALE_I, AP_RND, AP_SAT> y26_scale_t;
typedef ap_fixed<Y26_FX_W, Y26_FX_I, AP_RND, AP_SAT>             y26_fx_t;

// ---- L-FUSE (plan 0.8da, flag -DY26_FUSE) ----------------------------------------------------
// v = wsc*(step*acc + lo*zpw) + b = (wsc*step)*acc + (wsc*lo)*zpw + b. Both products are
// per-output-channel constants, so they are formed ONCE in the sequential weight-staging loop and
// the epilogue loses a whole 48x32 multiply and its 80-bit round-and-saturate chain in every one of
// the Y26_EPI_WIDE lanes.
//
// It is a NO-OP without Y26_FX_DEQUANT, on purpose: the float arm below exists to keep the
// bit-exact gate honest and folding there would change its promotion order. Hence Y26_FUSE_ON.
//
// y26_kscale_t is all-fractional (I = 0 -> range +-0.5, 32 fractional bits). MEASURED over the
// shipping weights (100 quant convs, 21,429 channel/constant pairs): |wsc*step| <= 1.112e-2,
// |wsc*lo| <= 4.539e-2, so the headroom is 11x and AP_SAT clips rather than wraps if a future
// calibration exceeds it. The fold is MORE accurate than the pair it replaces, not less: the
// smallest wsc are ~5e-5 and keep only 9-10 bits in y26_scale_t's 24 fractional bits, where the
// product keeps 32. Relative error of the scale constant, mean 1.468e-05 -> 3.785e-06, worst
// 5.241e-04 -> 1.125e-04. That is why the gate for this lever is mAP (0.8cr), not bit-exactness.
#if defined(Y26_FUSE) && defined(Y26_FX_DEQUANT)
#define Y26_FUSE_ON 1
typedef ap_fixed<32, 0, AP_RND, AP_SAT> y26_kscale_t;
// The accumulator operands are narrowed to the DSP48E2's 27-bit port, which is the whole point:
// 32x27 binds to 2 DSP where 32x32 binds to 4. Worst-case |acc| for u8 x s8 x 2,304 taps is 75.2M,
// which is 0.56 bits OVER ap_int<27> (+-67.1M); the measured network maximum is 924,719 (21 bits),
// 6.6 bits of headroom. The csim assert at the fold site is the check, and it runs on all 800
// netgate images before any of this reaches silicon.
#define Y26_KACC_W 27
#endif

// ---------------------------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------------------------
// SmoothQuant activations are ASYMMETRIC uint8 codes: clamp(round((v*ssc - lo)/step), 0, 255).
typedef ap_uint<8>  y26_act_t;
// ---- G0b (2026-08-19): the ACTIVATION PORT WORD type ----------------------------------------
// Gate G0 proved that widening the on-chip BANK is useless while `m_axi_gmem_act` is 8 bits wide
// (csynth interface table: "Data Width 8 -> 8", with Widen Fail 214-307 "Could not widen since
// type i8 size is greater than or equal to alignment 1(bytes)"). HLS derives the port width from
// the POINTER TYPE, so the only way to widen the port is to declare X as a wide word and unpack
// it on-chip. -DY26_ACT_WORD=64 does exactly that. Undefined keeps the 8-bit port, and every
// measurement recorded before this date reproduces byte-identically.
#if defined(Y26_ACT_WORD) && (Y26_ACT_WORD > 8)
typedef ap_uint<Y26_ACT_WORD> y26_xw_t;
#define Y26_XPE (Y26_ACT_WORD / 8)   // activation elements carried per port word
#else
typedef y26_act_t             y26_xw_t;
#define Y26_XPE 1
#endif
// ---- ACT128 IS *SHIPPING*. THE 2026-08-25 CLOSURE BELOW EXPIRED WHEN ALE MOVED. --------------
// 2026-09-19 (ledger pf/pi, 0.8ek + 0.8en). READ THIS BEFORE THE BLOCK THAT FOLLOWS.
// The design now ships -DY26_ACT_WORD=128 -DY26_XBUF_WIDE=16 (sol_YA128): 72.42 ms / 13.81 FPS /
// 269.5 mJ, WNS +0.022 with 0 failing, output bit-exact. The note below forbade exactly that.
//
// THE NOTE WAS NOT WRONG - IT WENT STALE. Its arithmetic is correct and is retained verbatim.
// It was measured at Y26_ACT_LANE_ELEMS = 3200. ALE is now 8192, and bank depth is ALE/WIDE:
//     ALE=3200  WIDE=8   400 x  64b -> 2 wide x 1 deep = 2 blocks/bank, depth  78%
//     ALE=3200  WIDE=16  200 x 128b -> 4 wide x 1 deep = 4 blocks/bank, depth  39%  <- the 2x
//     ALE=8192  WIDE=8  1024 x  64b -> 2 wide x 2 deep = 4 blocks/bank, depth 100%
//     ALE=8192  WIDE=16  512 x 128b -> 4 wide x 1 deep = 4 blocks/bank, depth 100%  <- NO 2x
// At ALE=8192 both land on 4 blocks/bank because 8192/16 = 512 is exactly one full block depth,
// so the doubling simply disappears. MEASURED, not argued: csynth reports 20 xbu arrays at 80
// BRAM18 = 4 blocks each, and routed BRAM moved 208.5 -> 212 tiles (+1.7%) with LUT FALLING
// 61,271 -> 60,543. The note predicted +91% BRAM.
//
// A CLOSURE IS A STATEMENT ABOUT A DESIGN POINT, NOT ABOUT A LEVER. This one read as a permanent
// law ("Do not retry at any Y26_XBUF_WIDE") and outlived its design point by three weeks. It
// survived because it lived HERE, in the source, rather than in the ledger that re-checks
// closures - the ROWS/ALE family was re-derived at A8K and stayed dead, but nothing re-read this.
// DO NOT WRITE A PERMANENT-SOUNDING CLOSURE IN THIS HEADER. Put it in the ledger where it will
// be re-checked, and if you must note it here, name the design point it was measured at.
//
// WHAT IS STILL TRUE: the axis is a PLATEAU and shipping sits on its TOP EDGE (0.8en).
// blocks/bank = ceil(8*WIDE/36) * ceil((ALE/WIDE)/512); at ALE=8192 that is 4 for WIDE in
// {4,8,16} and 8 for WIDE=32. So Y26_XBUF_WIDE=32 (ACT256) costs 2x xbuf BRAM, and sweeping ALE
// does not rescue it: ALE=16384 is the only value where 32 joins the plateau and it gets there by
// doubling the FLOOR. ACT256 is OPEN, not free, and see the pj warning below before pricing it.
//
// PRICING WARNING (pj, 2026-09-19): do NOT assume BRAM power tracks BRAM COUNT. The L6 run moved
// BRAM tiles 212 -> 220 (+3.8%) while BRAM POWER went 0.566 -> 0.863 W (+52%) - that is ACTIVITY,
// not capacity, and it broke a projection built on count. Price any widening on a measured L1B.
// THE SUPERSEDED VERDICT, preserved verbatim because it IS the record (guard R170 requires this
// line to survive: a closure deleted from the source is a closure nobody can audit, and it was a
// closure without its stated precondition that caused the stall in the first place):
//
//     "ACT128 IS CLOSED. MEASURED TWICE, 2026-08-25. Do not retry at any Y26_XBUF_WIDE."
//
// That sentence is NO LONGER OPERATIVE. It is quoted here as history, not as an instruction.
// ---- the ORIGINAL 2026-08-25 note follows, unaltered, for its arithmetic ---------------------
// Widening this port past 64 bits costs a 2x xbuf block count, and it does so for TWO DIFFERENT
// reasons that happen to land on the same total (1,134 -> 2,165 BRAM18K both times):
//   ACT=64,  WIDE=8   trunk    ram_1p   2 blocks/bank   64 x 400   <- the sweet spot
//   ACT=128, WIDE=8            ram_t2p  4 blocks/bank   64 x 400   <- HLS promotes to TRUE DUAL
//                                                                     PORT to absorb the wider
//                                                                     write; the second port is
//                                                                     what doubles the blocks.
//   ACT=128, WIDE=16           ram_1p   4 blocks/bank  128 x 200   <- the t2p promotion is GONE
//                                                                     (the reshape fix worked on
//                                                                     its own target) but a
//                                                                     128-bit bank needs
//                                                                     ceil(128/36)=4 blocks of
//                                                                     WIDTH, while depth 200 uses
//                                                                     39% of the 512 a block
//                                                                     holds. Pure aspect-ratio
//                                                                     waste.
// This is law 2 (BRAM is allocated in BLOCKS, not bits) acting on the WIDTH axis. The trunk's
// 64x400 is efficient precisely because 64 bits = 2 full 36-bit block widths AND 400/512 = 78% of
// a block depth. Any widening halves an already-short depth and buys nothing back, so there is no
// third setting that works: feeding a 128-bit port needs either a second write port or a 128-bit
// bank, and both cost exactly 2x. It gets WORSE under A2 row-banding, which shortens depth further.
// NOTE what this retracts: an #error guard forbidding Y26_XPE > Y26_XBUF_WIDE briefly lived here.
// It was removed the same day - it was built on the t2p case alone, and its advice ("widen the
// reshape") is measurably wrong, since widening only swaps one 2x for another.
// Weights are symmetric int8 codes, per-output-channel scaled.
typedef ap_int<8>   y26_wt_t;
// ---- W9 (2026-08-19): the WEIGHT PORT WORD type ---------------------------------------------
// See the Y26_WPE / Y26_WIDX block near Y26_WBANK for why this exists and what it costs.
// NOTE the asymmetry with y26_xw_t: activations are UNSIGNED uint8 codes, weights are SIGNED int8.
// The port word is unsigned either way (it is a byte container, not a number), so every unpack site
// must cast back through ap_int<8> - a plain (y26_wt_t)range() would sign-extend a negative code
// wrongly. Same class of trap as the y26_idx_t truncation caution below.
#if defined(Y26_WT_WORD) && (Y26_WT_WORD > 8)
typedef ap_uint<Y26_WT_WORD> y26_ww_t;
#else
typedef y26_wt_t             y26_ww_t;
#endif
// ---- W4b (2026-08-20): the OUTPUT PORT WORD type --------------------------------------------
// MEASURED CAUSE (notes 1y.37). W4 widened the epilogue to Y26_EPI_WIDE elements per iteration and
// csynth answered:
//
//   WARNING: [HLS 200-448] Lower bound of II is 8 due to multiple operations accessing 'gmem_out'
//                          m_axi write: {gmem_out_addr_write_ln760 .. _addr_7_write_ln760}
//
// i.e. `yr[...]` is NOT a local buffer - it is a direct m_axi store, and `float* Y` makes gmem_out a
// 32-bit port. Eight scalar writes on one write port is eight cycles no matter how wide the compute
// in front of them is, so W4 at N=8 ran the epilogue at II=8 = 1.0 cyc/element: EXACTLY the width it
// replaced. The zeroing half of W4 hit II=1 and is real; the store half collected nothing.
//
// This is the same lesson as W5 (gmem_act) and W9 (gmem_wt) for the third time: **HLS derives the
// port width from the POINTER TYPE, and a loop can never beat its narrowest port.** The item is not
// "widen the epilogue", it is "widen the epilogue AND its port".
//
// Undefined leaves Y a `float*` and every measurement recorded before this date reproduces
// byte-identically; the store loop below collapses to the historical direct write.
#if defined(Y26_OUT_WORD) && (Y26_OUT_WORD > 32)
typedef ap_uint<Y26_OUT_WORD> y26_yw_t;
#define Y26_YPE (Y26_OUT_WORD / 32)  // output floats carried per port word
#else
typedef float                 y26_yw_t;
#define Y26_YPE 1
#endif
// Bit-cast container. The epilogue produces a `float`; the port word is a bag of bits. Union
// punning is the one form both g++ and Vitis HLS agree on - a reinterpret_cast through float* is
// UB under strict aliasing and HLS is entitled to (and does) reorder around it.
union y26_fp32 { float f; uint32_t u; };
// ALIGNMENT IS STRUCTURAL, NOT LUCKY - CHECKED, 2026-08-20. A wide store needs the block base
// (oc*OH + ohb)*OW to be a multiple of Y26_YPE and needs R*OW to have no partial last word. Swept
// over all 94 deployed-dense convs x every oc x every row block: **0 misaligned bases and 0 partial
// words at Y26_YPE = 4/8/16 crossed with Y26_ROWS = 8/16/32.** The reason is that every output map
// in this network is square with OH = OW in {20,40,80,160,320} - all multiples of 4 - and `ohb`
// steps by Y26_ROWS, so ohb*OW carries the alignment. There is therefore NO tail path below, and
// that absence is a measured fact about THIS network, not a general property. Re-run the sweep
// before trusting it on a different graph or a non-square head.
//
// THAT SWEEP IS NECESSARY BUT NOT SUFFICIENT, and the difference cost one csynth run. It says the
// DRAM side divides; it says nothing about whether a beat's Y26_YPE elements sit in one `ybuf`
// group. That needs OW itself to be a multiple of Y26_YPE - true at 4 for every shape here, FALSE
// at 8 because OW=20. See the store loop in conv_engine.cpp for what that costs and what would buy
// it back. **Y26_OUT_WORD=128 is the supported design point; 256 needs a padded host row stride.**
#define Y26_YBANK Y26_EPI_WIDE
// The epilogue's group counter is its own iteration index, which is only the right ybuf group while
// the bank count equals the epilogue width - and the store loop's "one beat lives in one group"
// property additionally needs Y26_YPE <= Y26_YBANK. Both are checked here rather than discovered as
// a wrong answer, because neither shows up as a compile error.
#if Y26_YPE > Y26_YBANK
#error "Y26_OUT_WORD/32 must not exceed Y26_EPI_WIDE - widen the epilogue first"
#endif
// ybuf is sized in PADDED units: Y26_ROWS rows of OWP = ceil(OW/Y26_YBANK)*Y26_YBANK columns, so the
// worst case adds one whole group per row over the unpadded bound.
#define Y26_YBUF_GRP ((Y26_ROWS * (Y26_MAX_OW + Y26_YBANK)) / Y26_YBANK)

// ---- L-YDIRECT ----------------------------------------------------------------------------------
// When YS == OWP (Y26_YSTRIDE_PAD) and Y26_YPE == Y26_YBANK, the W4b store loop reads ybuf back in
// EXACTLY the order the epilogue wrote it: its odometer collapses to grp == gi, lan == 0, one beat
// per epilogue iteration. ybuf is then a pure pass-through and the second pass is dead weight - so
// the epilogue packs the m_axi word itself and ybuf is not instantiated at all. Off, or in any other
// configuration, the two-pass path below is unchanged byte for byte.
#if defined(Y26_YDIRECT) && (Y26_YPE > 1) && (Y26_YPE == Y26_YBANK) && defined(Y26_YSTRIDE_PAD)
#define Y26_YDIRECT_ON 1
#endif
// ---- L-EFLAT (ledger pa) ------------------------------------------------------------------------
// Fuses the epilogue's `for (p < pack)` member loop into the store loop as a third odometer digit.
// It maintains `ywbase` itself and fuses the OCPACK member loop, so it is meaningless without both
// Y26_YDIRECT_ON and Y26_OCPACK - defined here only when both hold, rather than failing obscurely
// 1200 lines later. Undefined, every line below is byte-identical to the shipping build.
#if defined(Y26_EFLAT) && defined(Y26_YDIRECT_ON) && defined(Y26_OCPACK)
#define Y26_EFLAT_ON 1
#endif
// Accumulator width: worst case |acc| = 255 * 127 * 2304 = 74,615,040 -> 27 bits signed.
// 32 is the natural fabric/DSP width and leaves headroom; verified against real ap_int.h under g++.
typedef ap_int<32>  y26_acc_t;

// ---------------------------------------------------------------------------------------------
// INDEX types - added 2026-08-17 after the first csynth showed DSPs going to ADDRESS arithmetic.
//
// The kernel originally computed every tensor offset in `long`. HLS took that literally and emitted
// 63/64-bit multiplier cores (`mul_32s_32s_64`, `mul_32s_33s_64`, `mul_33s_32s_63`,
// `mul_32ns_32ns_63`) purely to compute array indices - each several DSP48E2s. That is a large part
// of why the MAC loop reported 100 DSPs while containing only 16 int8 multipliers.
//
// Nothing here needs 64 bits, or even 32:
//   * largest tensor  = 160*160*64 = 1,638,400 elements  -> 21 bits
//   * largest weight  = 512 * 2304 = 1,179,648 elements  -> 21 bits
//   * largest "outer" intermediate (oc*icpg, ic*H, oc*OH) bounded by the same products -> 22 bits
//
// Widths below carry margin over the DECLARED Y26_MAX_* bounds, not over the values the testbench
// happens to produce - the same discipline as the 27-bit accumulator bound (design to the analytical
// worst case, never to the measured one).
//
// CAUTION when editing: ap_uint arithmetic widens (`ap_uint<A> * ap_uint<B>` -> `ap_uint<A+B>`) and
// assignment back to a narrower type TRUNCATES SILENTLY. The casts at each multiply site are
// deliberate and load-bearing; do not "simplify" them away.
typedef ap_uint<24> y26_idx_t;   // flat element index into any activation / weight / output tensor
typedef ap_uint<22> y26_out_t;   // outer-product intermediate before the final stride multiply

// Per-conv descriptor. Plain-old-data, no STL - safe to pass through a synthesizable boundary.
// Mirrors the fields of ConvW that the asymmetric (qmode==1) path actually consumes.
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
    // HW lever B (2026-09-22): yq != 0 -> each 32-bit Y slot carries the CONSUMER's uint8 code of the output,
    // q_u8(v, q_s[o], q_lo[o], q_st[o]), instead of the float v. Same slots, same layout, same port: only the
    // payload changes, so the host narrows u32 -> u8 instead of re-quantizing. Params are per OUTPUT channel
    // (= the consumer's input channel), read only when yq != 0.
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

// ---------------------------------------------------------------------------------------------
// The kernel. Computes one conv over pre-quantized activation codes.
//
//   X   [ic][H][W]                    uint8 activation codes (quantization happens upstream; in
//                                     hardware the producing layer emits codes directly)
//   Wt  [oc][ic/groups][kh][kw]       int8 weight codes
//   Y   [oc][OH][OW]                  float output (stage 1 keeps the dequant in float on purpose,
//                                     so any mismatch vs conv2d() is attributable to the ACCUMULATOR
//                                     alone - that is what makes this a clean bit-exact gate)
//
// OH/OW are derived exactly as conv2d does: (H + 2*p - k)/s + 1.
// ---------------------------------------------------------------------------------------------
void y26_conv2d_hls(const y26_xw_t* X, int H, int W,
                    const y26_ww_t*  Wt,
                    const Y26ConvCfg& c,
                    y26_yw_t* Y);        // W4b: `float*` unless -DY26_OUT_WORD widens gmem_out

// Output dims, shared by kernel and host so they cannot drift apart.
inline int y26_oh(int H, const Y26ConvCfg& c) { return (H + 2 * c.ph - c.kh) / c.sh + 1; }
inline int y26_ow(int W, const Y26ConvCfg& c) { return (W + 2 * c.pw - c.kw) / c.sw + 1; }

// ---------------------------------------------------------------------------------------------
// Step 1 item 4 - the AXI top-level function: the PL<->PS handoff.
//
// This is the synthesis entry point. Everything above is the datapath; this is the boundary. The
// signature is FLATTENED (scalars, not the Y26ConvCfg struct) because s_axilite carries scalars and
// m_axi carries the buffers - a struct holding pointers cannot cross that boundary.
//
// Division of labour, unchanged from the plan: the PL runs the trunk and DMAs the raw one2one head
// maps out to DDR; the ARM PS reads them and runs detection_decode.h in software. Decode is never synthesized.
//
// `perch` selects between the scalar (step, lo) and the per-input-channel vectors (step_v, lo_v);
// the vectors are only read when perch != 0, so a host may pass null for them otherwise.
//
// INTERFACE DEPTHS - added 2026-08-17 for co-simulation.
//
// These are a COSIM-ONLY requirement and do not change the generated hardware. An m_axi port is a
// raw pointer, so HLS cannot infer how many elements exist behind it; without a depth, cosim refuses
// to start with:
//
//   ERROR: A depth specification is required for interface port 'X' for cosimulation.
//
// Each is the DESIGN maximum, not the maximum any one conv touches - a depth smaller than the
// largest access would corrupt cosim, and these are the same bounds the buffers are already sized
// against, so they cannot drift independently. They must remain single integer constants: `depth=`
// takes a constant, not an arithmetic expression, so do not inline the products at the pragma site.
//
// Cost note: depth sizes the AXI memory model and the C-side buffers; the RTL only issues the reads
// it actually needs, so a small conv does not pay for the full depth in simulated cycles.
// All five are OVERRIDABLE (-DY26_DEPTH_X=... etc).
//
// CORRECTED 2026-08-17. This block used to claim that cosim's reported latency "SCALES WITH THESE",
// on the theory that the testbench marshals `depth` elements per transaction. THAT WAS TESTED AND IS
// WRONG: cutting Y26_DEPTH_X by 200x moved the reported latency by 1.9x, and that 1.9x was fully
// accounted for by the run testing a DIFFERENT conv (icpg 256 vs 128). A single per-tile cost model
// fits both runs to 0.2%. See "Co-simulation: what the first real cycle counts say" in
// notes/yolo26s-hls-next-steps.md for the derivation.
//
// What the depths DO affect: cosim wall-clock and memory, which is reason enough to shrink them
// (run B simulated in ~3 min). What they do NOT affect: the cycle count you report. Do not attribute
// a latency number to marshalling without subtracting two runs that differ only in depth.
//
// Shrinking them has a sharp edge: the testbench allocates its port buffers to exactly these sizes,
// so a conv larger than the declared depth overruns the heap and csim dies with the useless
// "child killed: unknown signal". The tb guards against this and SKIPs such convs by name.
#ifndef Y26_DEPTH_X
#define Y26_DEPTH_X    1638400   // Y26_ACT_MAX_ELEMS - 160*160*64 == 320*320*16
#endif
#ifndef Y26_DEPTH_Y
#define Y26_DEPTH_Y    1638400   // same bound: the largest output is also 320*320*16
#endif
// G0b: the m_axi depth= counts PORT WORDS, not activation elements. 1638400 is a multiple of 8,
// so this divides exactly at Y26_ACT_WORD=64 and collapses to Y26_DEPTH_X in the control build.
#define Y26_DEPTH_XW   (Y26_DEPTH_X / Y26_XPE)
// W4b: same rule on the output side - depth= counts PORT WORDS. 1638400 is a multiple of 16, so
// this divides exactly at Y26_OUT_WORD up to 512 and collapses to Y26_DEPTH_Y in the control build.
#define Y26_DEPTH_YW   (Y26_DEPTH_Y / Y26_YPE)
#ifndef Y26_DEPTH_WT
#if Y26_WPE > 1
// W9: the padded blob is larger. Worst case adds (Y26_WPE-1) elements per (oc, tap).
#define Y26_DEPTH_WT (Y26_MAX_OC * (Y26_MAX_DEPTH + Y26_MAX_K * Y26_MAX_K * (Y26_WPE - 1)))
#else
#define Y26_DEPTH_WT   1179648   // Y26_MAX_OC * Y26_MAX_DEPTH == 512 * 2304
#endif
#endif
// W9: like Y26_DEPTH_XW, the m_axi depth= on gmem_wt counts PORT WORDS, not weight elements.
// Y26_DEPTH_WT is a multiple of Y26_WPE by construction above, so this divides exactly.
#define Y26_DEPTH_WTW  (Y26_DEPTH_WT / Y26_WPE)
#ifndef Y26_DEPTH_OC
#define Y26_DEPTH_OC   512       // Y26_MAX_OC   - wsc[oc], bias[oc]
#endif
#ifndef Y26_DEPTH_IC
#define Y26_DEPTH_IC   663       // Y26_MAX_IC   - step_v[ic], lo_v[ic]
#endif
void y26_conv_top(const y26_xw_t*  X,        // activations in  (m_axi, Y26_ACT_WORD bits - G0b)
                  const y26_ww_t*  Wt,       // int8 weight codes (m_axi)
                  const float*     wsc,      // [oc] per-output-channel weight scale (m_axi)
                  const float*     bias,     // [oc] (m_axi)
                  const float*     step_v,   // [ic] perch only (m_axi)
                  const float*     lo_v,     // [ic] perch only (m_axi)
                  y26_yw_t*        Y,        // output (m_axi, Y26_OUT_WORD bits - W4b)
                  int H, int W,
                  int oc, int ic, int kh, int kw,
                  int sh, int sw, int ph, int pw,
                  int groups, int act, int perch,
                  float step, float lo
#ifdef Y26_YQ8
                  , int yq, const float* q_s, const float* q_lo, const float* q_st
#endif
                  );
