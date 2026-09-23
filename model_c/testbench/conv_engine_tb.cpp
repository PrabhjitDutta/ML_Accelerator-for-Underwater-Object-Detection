// conv_engine_tb.cpp - the stage-1 bit-exact gate for the synthesizable kernel.
//
// Asserts that y26_conv2d_hls() (ap_int<32> accumulator) reproduces conv2d() (double accumulator)
// EXACTLY - not "within tolerance". Exact equality is the right bar here because integer
// accumulation is order-safe and the dequant expression is deliberately still float in stage 1, so
// there is no legitimate source of divergence. A single ULP of difference means a real bug.
//
// This gate is COMPILER-INVARIANT (unlike the C-sim's own md5, which -march=native FMA contraction
// perturbs), so it is meaningful to run it here under MSYS2 g++ even though the golden dumps were
// produced on the server.
//
// Build and run ALL THREE gates (stage-1 here, stage-2 via -DY26_FX_DEQUANT, stage-2+LUT by adding
// -DY26_SILU_LUT) with:
//   bash scripts/run_kernel_gates.sh [weights_dir] [spatial]
// Use the script, not an ad hoc g++ line: these binaries are easy to leave stale, and a stale gate
// certifies whatever the code used to do. Outputs land in build/tb_s{1,2,2lut}.exe.
//
// This file is ALSO the Vitis HLS csim and cosim testbench (hls_csim_synth_cosim.tcl adds it with
// `add_files -tb`), deliberately - one testbench means csim cannot certify something the g++ gates
// never checked. Two consequences to preserve when editing:
//   * it calls y26_conv_top(), not y26_conv2d_hls(). Cosim traces arguments at the synthesis
//     boundary, so a tb that bypasses the top can be C-simulated but never RTL-co-simulated.
//   * the m_axi port buffers are sized to the declared Y26_DEPTH_* constants, not to each conv's
//     exact dims. See the allocation block in main() for why cosim requires this.
#include "../hls_kernel/conv_engine.h"
#include "../reference_model/layer_ops.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

// Deterministic input generator - same sequence on every machine, so a mismatch is always the
// kernel and never the stimulus. Spans a range wide enough to drive the uint8 quantizer into both
// clamps (0 and 255) as well as the linear region.
static float stim(int i) {
    unsigned x = (unsigned)i * 1664525u + 1013904223u;
    x ^= x >> 16;
    return ((float)(x % 20001) / 10000.f - 1.0f) * 3.0f;   // ~[-3, 3]
}


// ---- C3: THE HOST SIDE OF THE PADDED ROW STRIDE ---------------------------------------------
// -DY26_YSTRIDE_PAD makes the kernel write Y at OWP = ceil(OW/Y26_EPI_WIDE)*Y26_EPI_WIDE instead
// of OW, which is the whole reason Y26_YPE=8 is reachable (conv_engine.cpp's store note). It is a
// HOST-SIDE LAYOUT CONTRACT: the buffer is larger than the tensor and the pad columns hold
// don't-care values the kernel never writes. Without this the comparison below reads the pad as
// output and a bit-exact gate fails MEANINGLESSLY -- which is exactly what PREDICT_C3.txt filed
// before the csynth, and why sol_C3 was reported as UNGATED.
// With the flag OFF this is `(ow)` and every expression using it collapses to the historical
// flat form, so the shipping gate is byte-identical.
#ifdef Y26_YSTRIDE_PAD
#define Y26_TB_YS(ow) ((((ow) + Y26_EPI_WIDE - 1) / Y26_EPI_WIDE) * Y26_EPI_WIDE)
#else
#define Y26_TB_YS(ow) (ow)
#endif

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "../weights_sq_compact_fold";
    const int SP = argc > 2 ? atoi(argv[2]) : 24;          // test spatial size (H == W)
    // Event-cost probe (ledger nf): "WxH" sets H separately, e.g. "16x17". Plain "16" is H == W as before.
    const char* SPX = argc > 2 ? strchr(argv[2], 'x') : nullptr;
    const int SPH = SPX ? atoi(SPX + 1) : SP;
    // argv[3] - stop after N convs (0 / absent = all). Exists ONLY for RTL co-simulation: cosim
    // runs the full Verilog through a simulator, so the ~100-conv sweep that takes seconds natively
    // would take days. The g++ gates never pass this and keep testing everything - do not let a
    // reduced cosim scope become the project's coverage claim.
    const int MAXC = argc > 3 ? atoi(argv[3]) : 0;
    // argv[4] - SKIP the first N eligible convs before testing (0 / absent = start at the first).
    // Added 2026-08-18 for the same reason as MAXC, and it is the other half of that knob: cosim at
    // spatial 16 exhausts xsim's ~2.5 GB heap after ONE transaction, so a multi-conv cosim is not
    // possible at all and coverage has to come from several SINGLE-conv runs. Without an offset
    // every one of those runs would test the same first conv. SKIP+MAXC together select one conv by
    // index, which is what lets a cosim sweep span the icpg range instead of resampling icpg=32.
    // Like MAXC, the g++ gates never pass this - do not let a sampled cosim become a coverage claim.
    // Accepts EITHER an integer (skip the first N eligible convs) OR an EXACT conv NAME.
    // PREFER THE NAME. An index is not a stable identifier here for two independent reasons, both
    // observed 2026-08-18 in one run:
    //   * `Wg.conv` is not lexicographically ordered (the depth guard below says the same), and its
    //     iteration order DIFFERS BETWEEN BUILDS - index 0 was 23.one2one_cv3.2.1.1.conv under g++
    //     and 23.one2one_cv3.2.0.1.conv under the HLS csim build of the same source. Indices
    //     harvested from one binary therefore select a different conv in the other.
    //   * The depth guard rejects convs that do not fit the Y26_DEPTH_* bounds AFTER this point, so
    //     selection slides to whichever later conv happens to fit. With depths sized for a
    //     128-channel conv it lands on a 128-channel conv, which looks plausible and is wrong.
    // The net effect was two runs silently measuring the SAME conv and returning identical cycle
    // counts. Matching on name removes both failure modes.
    const std::string SEL = argc > 4 ? argv[4] : "";
    const bool SEL_IS_IDX = !SEL.empty() &&
                            SEL.find_first_not_of("0123456789") == std::string::npos;
    const int SKIPC = SEL_IS_IDX ? atoi(SEL.c_str()) : 0;
    int eligible = 0;

    Weights Wg;
    try { Wg.load(dir); }
    catch (const std::exception& e) { printf("FATAL: %s\n", e.what()); return 2; }

    // ---- port buffers, sized to the DECLARED m_axi depths (conv_engine.h: Y26_DEPTH_*) ----
    //
    // Co-simulation marshals `depth` elements from each m_axi pointer into its memory model, so a
    // pointer to an exactly-sized vector is read out of bounds whenever the conv is smaller than the
    // design maximum - which is almost always. Sizing the buffers to the declared depths removes the
    // whole class of problem, and it also lets the three optional ports (bias, step_v, lo_v) be
    // passed as valid zero-filled memory instead of nullptr, which cosim cannot marshal at all.
    //
    // Hoisted out of the conv loop and allocated once (~9.4 MB total), so the g++ gates do not pay
    // 100 allocations. Only the used prefix of each is written per conv; the tail is never read by
    // the kernel, which stays inside the real dims.
    std::vector<y26_act_t> pX  (Y26_DEPTH_X);
#if Y26_XPE > 1
    // G0b: the kernel now takes a Y26_ACT_WORD-bit port. pX stays the byte-quantized reference
    // buffer (the fill code below is unchanged); pXw is the packed view actually handed to the top.
    std::vector<y26_xw_t>  pXw (Y26_DEPTH_XW);
#endif
    std::vector<y26_wt_t>  pW  (Y26_DEPTH_WT);
#if Y26_WPE > 1
    // W9: same shape as pXw above. pW stays the element-wise reference buffer in the kernel's
    // DRAM layout; pWw is the packed view actually handed to the top.
    std::vector<y26_ww_t>  pWw (Y26_DEPTH_WTW);
#endif
    std::vector<float>     pWsc(Y26_DEPTH_OC), pBias(Y26_DEPTH_OC);
    std::vector<float>     pStp(Y26_DEPTH_IC), pLo  (Y26_DEPTH_IC);
    std::vector<float>     pY  (Y26_DEPTH_Y);
#ifdef Y26_YQ8
    // HW lever B: code mode. After the float run, the SAME conv runs again with yq=1 and every real slot must
    // hold exactly q_u8(float slot, per-oc params) - the host's own quantizer applied to the kernel's own
    // float. Internal consistency, so it is an EXACT bar under FX_DEQUANT too. Y26_TB_YQ=0 skips it (the
    // frame-cycle cosim, where a second transaction would blur the per-conv latency).
    std::vector<float>     pQs(Y26_DEPTH_OC), pQlo(Y26_DEPTH_OC), pQst(Y26_DEPTH_OC), pY2(Y26_DEPTH_Y);
    const char* yqe = getenv("Y26_TB_YQ");
    const bool do_yq = !(yqe && strcmp(yqe, "0") == 0);
    int yq_tested = 0, yq_failed = 0;
    long yq_clamp0 = 0, yq_clamp255 = 0, yq_elems = 0;
#endif

    int tested = 0, failed = 0, skipped = 0;
    double worst_ulp = 0.0;
    std::string worst_name;
    double worst_cos = 1.0, worst_rel_any = 0.0, worst_abs = 0.0, worst_abs_ref = 0.0;
    std::string worst_cos_name;

    for (const auto& kv : Wg.conv) {
        const std::string& name = kv.first;
        const ConvW& c = kv.second;

        // Stage 1 covers the SmoothQuant integer path only. The 2 surviving FP32-fallback convs
        // (10.m.0.attn.pe, 22.m.0.1.attn.pe) are not integer convs and are out of scope here.
        if (!c.quant || !c.asym) { ++skipped; continue; }

        const int H = SPH, W = SP;
        if (H + 2 * c.ph < c.kh || W + 2 * c.pw < c.kw) { ++skipped; continue; }
        // Counted AFTER the eligibility filters above, so an index means "the Nth conv this
        // testbench would actually test", not the Nth line of the manifest.
        if (!SEL.empty() && !SEL_IS_IDX) {
            // EXACT match, not substring: "9.cv2.conv" is a substring of "19.cv2.conv" and
            // silently selected the wrong layer (icpg=192 instead of 512) when this was find().
            if (name != SEL) { ++skipped; ++eligible; continue; }
        } else if (eligible < SKIPC) { ++skipped; ++eligible; continue; }
        ++eligible;
        if (MAXC && tested >= MAXC) { ++skipped; continue; }
        // Identify the conv under test. Needed because SKIPC selects convs by INDEX, so a cosim
        // cycle count is meaningless without knowing which conv produced it - and icpg is what caps
        // lane scaling (see notes 1g), so it is printed rather than left to be derived.
        printf("[conv %3d] %-34s oc=%-4d ic=%-4d k=%dx%d s=%d g=%-4d icpg=%d\n",
               eligible - 1, name.c_str(), c.oc, c.ic, c.kh, c.kw, c.sh, c.groups,
               c.groups ? c.ic / c.groups : c.ic);
        fflush(stdout);

        // ---- golden: the Tensor C-sim ----
        // Computed BEFORE the depth guard below, deliberately: `golden` owns its own storage (it is a
        // Tensor, not one of the depth-sized port buffers), so it is always safe to build, and its
        // size is the authoritative output element count. Deriving that count from a hand-written
        // (H + 2*ph - kh)/sh + 1 here would duplicate conv2d()'s formula and could silently disagree
        // with it - which is precisely the kind of drift the guard exists to catch.
        Tensor x(c.ic, H, W);
        for (size_t i = 0; i < x.d.size(); ++i) x.d[i] = stim((int)i);
        Tensor golden = conv2d(x, c);

        // Depth guard. The port buffers are sized to the Y26_DEPTH_* constants, which are overridable
        // so a cosim run can shrink them (see conv_engine.h). Under-sizing them silently overruns the
        // heap - and the observed symptom names nothing useful: csim dies with
        //     @E Simulation failed with unknown error: child killed: unknown signal
        // which cost a full debug cycle on 2026-08-17 (depths were cut to 4096 on the assumption the
        // first conv tested would be the small stem; Wg.conv is not lexicographically ordered, and it
        // is actually 23.one2one_cv3.2.1.1.conv, needing X=8192/Wt=16384/Y=8192). Refuse the conv
        // instead, and name the bound that was hit.
        {
            const long need_x = (long)c.ic * H * W;
            // W9: the guard must count the PADDED blob, not c.w.size() - under -DY26_WT_WORD the
            // layout is larger than the PyTorch tensor and it is pW that gets overrun, not c.w.
            const long need_w = (long)c.oc * Y26_WSTRIDE(c.ic / c.groups, c.kh * c.kw);
            // C3: the guard counts the PADDED blob, not the tensor -- same reason W9 made
            // need_w count the padded weight layout rather than c.w.size().
            const long need_y = (long)golden.C * golden.H * Y26_TB_YS(golden.W);
            if (need_x > Y26_DEPTH_X || need_w > Y26_DEPTH_WT || need_y > Y26_DEPTH_Y ||
                c.oc > Y26_DEPTH_OC || c.ic > Y26_DEPTH_IC) {
                printf("  SKIP %-28s exceeds declared depths "
                       "(X %ld/%d, Wt %ld/%d, Y %ld/%d, oc %d/%d, ic %d/%d)\n",
                       name.c_str(), need_x, Y26_DEPTH_X, need_w, Y26_DEPTH_WT, need_y, Y26_DEPTH_Y,
                       c.oc, Y26_DEPTH_OC, c.ic, Y26_DEPTH_IC);
                ++skipped; continue;
            }
        }

        // ---- the same input, pre-quantized to uint8 codes (conv2d does this internally) ----
        const size_t HW = (size_t)H * W;
        for (int ic = 0; ic < c.ic; ++ic) {
            const float s = c.ssc[ic], lo = c.lo_of(ic), st = c.step_of(ic);
            for (size_t j = 0; j < HW; ++j)
                pX[(size_t)ic * HW + j] = (y26_act_t)(int)q_u8(x.d[(size_t)ic * HW + j], s, lo, st);
        }
#if Y26_XPE > 1
        // Repack only the USED prefix into port words. Packing all of Y26_DEPTH_X would cost
        // 1.6 MB of pointless work per conv. ic*HW <= Y26_DEPTH_X and Y26_DEPTH_X is a multiple of
        // Y26_XPE, so the round-up below can never run off the end of pX.
        {
            const size_t nw = ((size_t)c.ic * HW + Y26_XPE - 1) / Y26_XPE;
            for (size_t w = 0; w < nw; ++w) {
                y26_xw_t v = 0;
                for (int b = 0; b < Y26_XPE; ++b)
                    v.range(b * 8 + 7, b * 8) = (ap_uint<8>)pX[w * Y26_XPE + b];
                pXw[w] = v;
            }
        }
#endif

        // ---- int8 weight codes ----
        // W9 (2026-08-19): c.w arrives in PyTorch order - (oc, icl, kh, kw), i.e. channel-major
        // within each output channel. The kernel's DRAM layout is Y26_WIDX(), which is that same
        // order in the control build and TAP-MAJOR under -DY26_WT_TAPMAJOR, with each oc slice
        // padded to a whole number of port words. So this is no longer a flat copy: it is a
        // PERMUTATION, and it is the only place the host-side layout is defined.
        //
        // Y26_WIDX() is shared with the kernel's staging counters precisely so these two cannot
        // drift apart silently - a mismatch here is a bit-exact gate failure, not a wrong number.
        {
            const int ktap  = c.kh * c.kw;
            const int icpg  = c.ic / c.groups;
            const int wstr  = Y26_WSTRIDE(icpg, ktap);
            // Pad elements are never read (kwsum and the MAC both guard on icl < icpg), but zero
            // them anyway: a stale value that IS read would otherwise be invisible until it moved.
            std::fill(pW.begin(), pW.end(), (y26_wt_t)0);
            for (int o = 0; o < c.oc; ++o)
                for (int il = 0; il < icpg; ++il)
                    for (int t = 0; t < ktap; ++t)
                        pW[(size_t)o * wstr + Y26_WIDX(icpg, ktap, il, t)] =
                            (y26_wt_t)(int)c.w[((size_t)o * icpg + il) * ktap + t];
#if Y26_WPE > 1
            // Pack to port words. Y26_WSTRIDE is a multiple of Y26_WPE by construction, so slice
            // boundaries coincide with word boundaries and there is no straddling epilogue.
            const size_t nw = ((size_t)c.oc * wstr + Y26_WPE - 1) / Y26_WPE;
            for (size_t w = 0; w < nw; ++w) {
                y26_ww_t v = 0;
                for (int b = 0; b < Y26_WPE; ++b)
                    v.range(b * 8 + 7, b * 8) = (ap_uint<8>)(ap_int<8>)pW[w * Y26_WPE + b];
                pWw[w] = v;
            }
#endif
        }

        // ---- scale vectors, into depth-sized buffers ----
        // The optional ports are zero-filled rather than passed as nullptr. The kernel still gates
        // them on `perch`/`bias != 0` semantics via cfg, so this changes no arithmetic; it only
        // gives cosim something valid to marshal.
        std::fill(pWsc.begin(),  pWsc.end(),  0.f);
        std::fill(pBias.begin(), pBias.end(), 0.f);
        std::fill(pStp.begin(),  pStp.end(),  0.f);
        std::fill(pLo.begin(),   pLo.end(),   0.f);
        std::copy(c.wsc.begin(), c.wsc.end(), pWsc.begin());
        if (!c.b.empty())  std::copy(c.b.begin(),    c.b.end(),    pBias.begin());
        if (c.perch)       std::copy(c.sa_v.begin(), c.sa_v.end(), pStp.begin());
        if (c.perch)       std::copy(c.lo_v.begin(), c.lo_v.end(), pLo.begin());

        // ---- the synthesizable kernel ----
        Y26ConvCfg cfg;
        cfg.oc = c.oc; cfg.ic = c.ic; cfg.kh = c.kh; cfg.kw = c.kw;
        cfg.sh = c.sh; cfg.sw = c.sw; cfg.ph = c.ph; cfg.pw = c.pw;
        cfg.groups = c.groups; cfg.act = c.act;
        cfg.perch  = c.perch ? 1 : 0;
        cfg.step   = c.sa;  cfg.lo = c.lo;
        // All four point at the depth-sized buffers, never at nullptr. A conv with no bias gets a
        // zeroed pBias, which is arithmetically identical to the old `bias ? bias[oc] : 0.f`; the
        // per-channel vectors are still gated on cfg.perch inside the kernel.
        cfg.step_v = pStp.data();
        cfg.lo_v   = pLo.data();
        cfg.wsc    = pWsc.data();
        cfg.bias   = pBias.data();

        const size_t n = (size_t)golden.size();
        float* out = pY.data();
        // Call through the SYNTHESIS TOP, not y26_conv2d_hls() directly. The top is a pure
        // forwarder (it only rebuilds cfg from flattened AXI scalars), so this changes nothing
        // numerically and the exact-equality bar still holds. It matters for cosim: Vitis captures
        // argument traces at the top boundary, so a tb that bypasses the top can be C-simulated but
        // never RTL-co-simulated. Routing through it here means ONE testbench certifies both.
#if Y26_XPE > 1
        const y26_xw_t* xptr = pXw.data();
#else
        const y26_xw_t* xptr = pX.data();
#endif
#if Y26_WPE > 1
        const y26_ww_t* wptr = pWw.data();
#else
        const y26_ww_t* wptr = pW.data();
#endif
        // W4b: gmem_out is a Y26_OUT_WORD-bit port, so the top takes a wide-word pointer. Unlike X
        // and Wt there is NO repacked shadow buffer here: the wide word is a bit-packed VIEW of the
        // same contiguous float array, in the same order, so `out` below still reads back as floats
        // and the comparison loop is untouched. That is only true because the pack is little-endian
        // lane 0 = element 0, which is what the store loop in conv_engine.cpp writes.
#if Y26_YPE > 1
        y26_yw_t* yptr = reinterpret_cast<y26_yw_t*>(out);
#else
        y26_yw_t* yptr = out;
#endif
        y26_conv_top(xptr, wptr, cfg.wsc, cfg.bias, cfg.step_v, cfg.lo_v, yptr,
                     H, W, cfg.oc, cfg.ic, cfg.kh, cfg.kw, cfg.sh, cfg.sw, cfg.ph, cfg.pw,
                     cfg.groups, cfg.act, cfg.perch, cfg.step, cfg.lo
#ifdef Y26_YQ8
                     , 0, pQs.data(), pQlo.data(), pQst.data()
#endif
                     );
#ifdef Y26_YQ8
        if (do_yq) {
            // Per-oc params from this channel's own float distribution: its 5th percentile maps to code 0 and
            // its 95th to 255, so every channel exercises both clamps and a well-populated linear region.
            const int OHt = golden.H, OWt = golden.W, YSq = Y26_TB_YS(OWt);
            std::vector<float> vs((size_t)OHt * OWt);
            for (int o = 0; o < c.oc; ++o) {
                for (int r_ = 0; r_ < OHt; ++r_)
                    for (int w_ = 0; w_ < OWt; ++w_) vs[(size_t)r_ * OWt + w_] = out[((size_t)o * OHt + r_) * YSq + w_];
                std::sort(vs.begin(), vs.end());
                const float p5 = vs[vs.size() / 20], p95 = vs[vs.size() - 1 - vs.size() / 20];
                const float sc = 0.75f + 0.125f * (float)(o % 5);
                pQs[o] = sc; pQlo[o] = p5 * sc; pQst[o] = std::max((p95 - p5) * sc, 1e-4f) / 255.f;
            }
            y26_yw_t* yptr2 = reinterpret_cast<y26_yw_t*>(pY2.data());
            y26_conv_top(xptr, wptr, cfg.wsc, cfg.bias, cfg.step_v, cfg.lo_v, yptr2,
                         H, W, cfg.oc, cfg.ic, cfg.kh, cfg.kw, cfg.sh, cfg.sw, cfg.ph, cfg.pw,
                         cfg.groups, cfg.act, cfg.perch, cfg.step, cfg.lo,
                         1, pQs.data(), pQlo.data(), pQst.data());
            size_t qbad = 0;
            for (int o = 0; o < c.oc; ++o)
                for (int r_ = 0; r_ < OHt; ++r_)
                    for (int w_ = 0; w_ < OWt; ++w_) {
                        const size_t i = ((size_t)o * OHt + r_) * YSq + w_;
                        const uint32_t want = (uint32_t)(int)q_u8(out[i], pQs[o], pQlo[o], pQst[o]);
                        uint32_t got; memcpy(&got, &pY2[i], 4);
                        qbad += got != want;
                        yq_clamp0 += want == 0; yq_clamp255 += want == 255; ++yq_elems;
                    }
            ++yq_tested;
            if (qbad) {
                ++yq_failed;
                printf("  YQ8 MISMATCH %-28s %zu codes differ\n", name.c_str(), qbad);
            }
        }
#endif

        // ---- comparison ----
        // Stage 1 (float dequant): exact equality is the bar. Stage 2 (-DY26_FX_DEQUANT): the
        // dequant is genuinely re-rounded in fixed point, so the bar is a tolerance + cosine, and
        // the per-conv statistics below are the measurement that must be reported, not hidden.
        size_t bad = 0;
        double wu = 0.0;
        double dot = 0.0, na = 0.0, nb = 0.0;
        const int YSt = Y26_TB_YS(golden.W);
        for (size_t i_ = 0; i_ < n; ++i_) {
            // C3: walk the TENSOR and index the BUFFER at the padded stride. At YSt == golden.W
            // io == ig == i_ and this is the historical flat loop, element for element.
            const int ww_ = (int)(i_ % (size_t)golden.W);
            const int rr_ = (int)((i_ / (size_t)golden.W) % (size_t)golden.H);
            const int cc_ = (int)(i_ / ((size_t)golden.W * golden.H));
            const size_t i  = ((size_t)cc_ * golden.H + rr_) * (size_t)YSt + ww_;
            const size_t ig = i_;
            const double o = (double)out[i], gd = (double)golden.d[ig];
            dot += o * gd; na += o * o; nb += gd * gd;
            if (memcmp(&out[i], &golden.d[ig], sizeof(float)) != 0) {
                ++bad;
                double d = std::fabs(o - gd);
                double m = std::fabs(gd);
                if (d > worst_abs) { worst_abs = d; worst_abs_ref = m; }
                // Relative error is meaningless when the reference is ~0 (a tiny absolute error over
                // a tiny denominator reports as a huge ratio). Gate the ratio on a reference floor so
                // the reported number describes real error rather than a division artifact.
                if (m > 1e-3) { double rel = d / m; if (rel > wu) wu = rel; }
            }
        }
        const double cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 1.0;
        if (cos < worst_cos) { worst_cos = cos; worst_cos_name = name; }
        if (wu > worst_rel_any) { worst_rel_any = wu; }
        ++tested;
        if (bad) {
            ++failed;
            if (wu > worst_ulp) { worst_ulp = wu; worst_name = name; }
#ifndef Y26_FX_DEQUANT
            if (failed <= 8)
                printf("  MISMATCH %-28s %6zu/%-8zu elems  worst_rel=%.3e  "
                       "(oc=%d ic=%d k=%dx%d s=%d g=%d act=%d perch=%d)\n",
                       name.c_str(), bad, n, wu,
                       c.oc, c.ic, c.kh, c.kw, c.sh, c.groups, c.act, (int)c.perch);
#endif
        }
    }

#ifdef Y26_YQ8
    if (do_yq) {
        printf("\n=== YQ8 code-mode gate: %d convs, %d mismatched (%ld codes: %.1f%% at 0, %.1f%% at 255) ===\n",
               yq_tested, yq_failed, yq_elems, yq_elems ? 100.0 * yq_clamp0 / yq_elems : 0.0,
               yq_elems ? 100.0 * yq_clamp255 / yq_elems : 0.0);
        if (yq_failed) { printf("\nFAIL - YQ8 codes differ from q_u8(kernel float).\n"); return 1; }
    }
#endif
#ifdef Y26_FX_DEQUANT
    printf("\n=== stage-2 TOLERANCE gate: ap_fixed<%d,%d> dequant (%s, spatial %dx%d) ===\n",
           Y26_FX_W, Y26_FX_I, dir.c_str(), SP, SP);
    printf("  integer convs tested   : %d\n", tested);
    printf("  convs differing at all : %d  (expected: most - the dequant is re-rounded)\n", failed);
    printf("  worst cosine           : %.9f   (at %s)\n", worst_cos, worst_cos_name.c_str());
    printf("  worst relative deviation: %.6e   (references > 1e-3 only)\n", worst_rel_any);
    printf("  worst ABSOLUTE deviation: %.6e   (at reference magnitude %.6e)\n",
           worst_abs, worst_abs_ref);
    // The bar: cosine must stay essentially 1. The c-sim notes put task-level noise at ~0.001 mAP50
    // and the neck cosine at 0.9988-0.9995, so anything at or above that band is in family. This is
    // an indicative bar only - the REAL gate is mAP over the full trunk, which does not exist yet.
    if (worst_cos < 0.999) {
        printf("\nFAIL - cosine %.9f is below the 0.999 indicative floor; widen ap_fixed.\n",
               worst_cos);
        return 1;
    }
    printf("\nPASS (indicative) - cosine >= 0.999 on every integer conv.\n"
           "  NOT a substitute for an mAP measurement: per-conv cosine does not compose into a\n"
           "  task-level number, and the trunk graph needed for mAP is not written yet.\n");
    return 0;
#else
    printf("\n=== stage-1 bit-exact gate (%s, spatial %dx%d) ===\n", dir.c_str(), SP, SPH);
    printf("  integer convs tested : %d\n", tested);
    printf("  skipped (non-integer): %d\n", skipped);
    printf("  MISMATCHED           : %d\n", failed);
    if (failed) {
        printf("  worst relative diff  : %.6e  (at %s)\n", worst_ulp, worst_name.c_str());
        printf("\nFAIL - the ap_int<32> accumulator is NOT bit-exact vs conv2d().\n");
        return 1;
    }
    printf("\nPASS - ap_int<32> accumulator is bit-exact vs the double accumulator on every "
           "integer conv.\n");
    return 0;
#endif
}
