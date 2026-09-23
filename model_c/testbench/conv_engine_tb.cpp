// Kernel testbench: runs every SmoothQuant conv through y26_conv_top() and through conv2d() and compares.
// Float dequant: exact equality (integer accumulation leaves no legitimate divergence).
// -DY26_FX_DEQUANT: tolerance + cosine. Run all three gates with scripts/run_kernel_gates.sh.
// Also the Vitis HLS csim/cosim testbench, so it calls the synthesis top (cosim traces arguments there) and sizes
// the m_axi buffers to the Y26_DEPTH_* constants.
#include "../hls_kernel/conv_engine.h"
#include "../reference_model/layer_ops.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

// Deterministic stimulus, wide enough to hit both uint8 clamps (0 and 255) and the linear region.
static float stim(int i) {
    unsigned x = (unsigned)i * 1664525u + 1013904223u;
    x ^= x >> 16;
    return ((float)(x % 20001) / 10000.f - 1.0f) * 3.0f;   // ~[-3, 3]
}

// -DY26_YSTRIDE_PAD: the kernel writes Y at row stride OWP instead of OW; the pad columns are don't-care, so
// the comparison must index the buffer at the padded stride. Off, this is `(ow)`.
#ifdef Y26_YSTRIDE_PAD
#define Y26_TB_YS(ow) ((((ow) + Y26_EPI_WIDE - 1) / Y26_EPI_WIDE) * Y26_EPI_WIDE)
#else
#define Y26_TB_YS(ow) (ow)
#endif

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "../weights_sq_compact_fold";
    const int SP = argc > 2 ? atoi(argv[2]) : 24;          // test spatial size (H == W)
    // "WxH" sets H separately, e.g. "16x17". Plain "16" is H == W.
    const char* SPX = argc > 2 ? strchr(argv[2], 'x') : nullptr;
    const int SPH = SPX ? atoi(SPX + 1) : SP;
    // argv[3]: stop after N convs (0 / absent = all). For RTL co-simulation only, which is far slower.
    const int MAXC = argc > 3 ? atoi(argv[3]) : 0;
    // argv[4]: select one conv, by exact NAME (preferred) or by index among eligible convs (skip the first N).
    // Indices are not stable: Wg.conv's iteration order differs between builds, and the depth guard below can
    // reject convs after this point. For co-simulation only.
    const std::string SEL = argc > 4 ? argv[4] : "";
    const bool SEL_IS_IDX = !SEL.empty() &&
                            SEL.find_first_not_of("0123456789") == std::string::npos;
    const int SKIPC = SEL_IS_IDX ? atoi(SEL.c_str()) : 0;
    int eligible = 0;

    Weights Wg;
    try { Wg.load(dir); }
    catch (const std::exception& e) { printf("FATAL: %s\n", e.what()); return 2; }

    // Port buffers sized to the declared m_axi depths: cosim reads `depth` elements from each pointer, and the
    // optional ports (bias, step_v, lo_v) get valid zero-filled memory instead of nullptr. Allocated once;
    // only the used prefix is written per conv.
    std::vector<y26_act_t> pX  (Y26_DEPTH_X);
#if Y26_XPE > 1
    // pX: byte-quantized reference buffer; pXw: the packed Y26_ACT_WORD view handed to the top.
    std::vector<y26_xw_t>  pXw (Y26_DEPTH_XW);
#endif
    std::vector<y26_wt_t>  pW  (Y26_DEPTH_WT);
#if Y26_WPE > 1
    // pW: element-wise reference buffer in the kernel's DRAM layout; pWw: the packed view handed to the top.
    std::vector<y26_ww_t>  pWw (Y26_DEPTH_WTW);
#endif
    std::vector<float>     pWsc(Y26_DEPTH_OC), pBias(Y26_DEPTH_OC);
    std::vector<float>     pStp(Y26_DEPTH_IC), pLo  (Y26_DEPTH_IC);
    std::vector<float>     pY  (Y26_DEPTH_Y);
#ifdef Y26_YQ8
    // yq check: the same conv runs again with yq=1 and every slot must decode (y26_yq_code) to exactly
    // q_u8(float slot, per-oc params). Exact even under FX_DEQUANT. Y26_TB_YQ=0 skips it.
    std::vector<float>     pQs(Y26_DEPTH_OC), pQlo(Y26_DEPTH_OC), pQst(Y26_DEPTH_OC), pQinv(Y26_DEPTH_OC), pY2(Y26_DEPTH_Y);
    const char* yqe = getenv("Y26_TB_YQ");
    const bool do_yq = !(yqe && strcmp(yqe, "0") == 0);
    int yq_tested = 0, yq_failed = 0;
    long yq_clamp0 = 0, yq_clamp255 = 0, yq_elems = 0, yq_ties = 0;
#endif

    int tested = 0, failed = 0, skipped = 0;
    double worst_ulp = 0.0;
    std::string worst_name;
    double worst_cos = 1.0, worst_rel_any = 0.0, worst_abs = 0.0, worst_abs_ref = 0.0;
    std::string worst_cos_name;

    for (const auto& kv : Wg.conv) {
        const std::string& name = kv.first;
        // Y26_TB_IM2COL=1: a dense kxk conv with ic*kh*kw <= Y26_ICGRP and input zero point 0 runs as the host's
        // Img2Col 1x1 over ic*kh*kw planes (board_host.cpp im2col_conv); pass SP = its output size.
        static const bool tb_i2c = getenv("Y26_TB_IM2COL") && strcmp(getenv("Y26_TB_IM2COL"), "1") == 0;
        ConvW i2c_w;
        const ConvW& k0 = kv.second;
        const bool i2c = tb_i2c && k0.groups == 1 && !k0.perch && k0.lo == 0.f && k0.kh * k0.kw > 1 &&
                         k0.ic * k0.kh * k0.kw <= Y26_ICGRP;
        if (i2c) {
            i2c_w = k0;
            i2c_w.ic = k0.ic * k0.kh * k0.kw;  i2c_w.kh = i2c_w.kw = i2c_w.sh = i2c_w.sw = 1;  i2c_w.ph = i2c_w.pw = 0;
            i2c_w.ssc.clear();
            for (int i = 0; i < i2c_w.ic; ++i) i2c_w.ssc.push_back(k0.ssc[i / (k0.kh * k0.kw)]);
        }
        const ConvW& c = i2c ? i2c_w : k0;

        // SmoothQuant integer convs only; the 2 FP32-fallback convs (attn.pe) are out of scope.
        if (!c.quant || !c.asym) { ++skipped; continue; }

        const int H = SPH, W = SP;
        if (H + 2 * c.ph < c.kh || W + 2 * c.pw < c.kw) { ++skipped; continue; }
        // Counted after the eligibility filters: an index means the Nth conv this testbench would test.
        if (!SEL.empty() && !SEL_IS_IDX) {
            // Exact match: "9.cv2.conv" is a substring of "19.cv2.conv".
            if (name != SEL) { ++skipped; ++eligible; continue; }
        } else if (eligible < SKIPC) { ++skipped; ++eligible; continue; }
        ++eligible;
        if (MAXC && tested >= MAXC) { ++skipped; continue; }
        // Identify the conv under test (icpg caps lane use).
        printf("[conv %3d] %-34s oc=%-4d ic=%-4d k=%dx%d s=%d g=%-4d icpg=%d\n",
               eligible - 1, name.c_str(), c.oc, c.ic, c.kh, c.kw, c.sh, c.groups,
               c.groups ? c.ic / c.groups : c.ic);
        fflush(stdout);

        // Golden: the reference model. Computed before the depth guard; its size is the authoritative output count.
        Tensor x(c.ic, H, W);
        for (size_t i = 0; i < x.d.size(); ++i) x.d[i] = stim((int)i);
        Tensor golden = conv2d(x, c);

        // Depth guard: the port buffers are sized to the overridable Y26_DEPTH_* constants, and overrunning them
        // crashes csim with no useful message. Refuse the conv and name the bound.
        {
            const long need_x = (long)c.ic * H * W;
            // Count the padded weight blob (larger than c.w under -DY26_WT_WORD).
            const long need_w = (long)c.oc * Y26_WSTRIDE(c.ic / c.groups, c.kh * c.kw);
            // Count the padded output (Y26_YSTRIDE_PAD).
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

        // The same input, pre-quantized to uint8 codes (conv2d does this internally).
        const size_t HW = (size_t)H * W;
        for (int ic = 0; ic < c.ic; ++ic) {
            const float s = c.ssc[ic], lo = c.lo_of(ic), st = c.step_of(ic);
            for (size_t j = 0; j < HW; ++j)
                pX[(size_t)ic * HW + j] = (y26_act_t)(int)q_u8(x.d[(size_t)ic * HW + j], s, lo, st);
        }
#if Y26_XPE > 1
        // Pack only the used prefix into port words. Y26_DEPTH_X is a multiple of Y26_XPE, so the round-up stays in pX.
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

        // Int8 weight codes. c.w is in PyTorch order (oc, icl, kh, kw); the kernel's DRAM layout is Y26_WIDX() (the
        // same order, or tap-major under -DY26_WT_TAPMAJOR, each oc slice padded to whole port words). Y26_WIDX() is
        // shared with the kernel's staging counters, so they cannot drift apart.
        {
            const int ktap  = c.kh * c.kw;
            const int icpg  = c.ic / c.groups;
            const int wstr  = Y26_WSTRIDE(icpg, ktap);
            // Pad elements are never read; zeroed anyway.
            std::fill(pW.begin(), pW.end(), (y26_wt_t)0);
            for (int o = 0; o < c.oc; ++o)
                for (int il = 0; il < icpg; ++il)
                    for (int t = 0; t < ktap; ++t)
                        pW[(size_t)o * wstr + Y26_WIDX(icpg, ktap, il, t)] =
                            (y26_wt_t)(int)c.w[((size_t)o * icpg + il) * ktap + t];
#if Y26_WPE > 1
            // Pack to port words. Y26_WSTRIDE is a multiple of Y26_WPE, so slices never straddle words.
            const size_t nw = ((size_t)c.oc * wstr + Y26_WPE - 1) / Y26_WPE;
            for (size_t w = 0; w < nw; ++w) {
                y26_ww_t v = 0;
                for (int b = 0; b < Y26_WPE; ++b)
                    v.range(b * 8 + 7, b * 8) = (ap_uint<8>)(ap_int<8>)pW[w * Y26_WPE + b];
                pWw[w] = v;
            }
#endif
        }

        // Scale vectors, into depth-sized buffers. Optional ports are zero-filled rather than nullptr (cosim cannot
        // marshal nullptr); the kernel still gates them on cfg.
        std::fill(pWsc.begin(),  pWsc.end(),  0.f);
        std::fill(pBias.begin(), pBias.end(), 0.f);
        std::fill(pStp.begin(),  pStp.end(),  0.f);
        std::fill(pLo.begin(),   pLo.end(),   0.f);
        std::copy(c.wsc.begin(), c.wsc.end(), pWsc.begin());
        if (!c.b.empty())  std::copy(c.b.begin(),    c.b.end(),    pBias.begin());
        if (c.perch)       std::copy(c.sa_v.begin(), c.sa_v.end(), pStp.begin());
        if (c.perch)       std::copy(c.lo_v.begin(), c.lo_v.end(), pLo.begin());

        // The synthesizable kernel.
        Y26ConvCfg cfg;
        cfg.oc = c.oc; cfg.ic = c.ic; cfg.kh = c.kh; cfg.kw = c.kw;
        cfg.sh = c.sh; cfg.sw = c.sw; cfg.ph = c.ph; cfg.pw = c.pw;
        cfg.groups = c.groups; cfg.act = c.act;
        cfg.perch  = c.perch ? 1 : 0;
        cfg.step   = c.sa;  cfg.lo = c.lo;
        // All point at depth-sized buffers, never nullptr; a zeroed pBias equals no bias.
        cfg.step_v = pStp.data();
        cfg.lo_v   = pLo.data();
        cfg.wsc    = pWsc.data();
        cfg.bias   = pBias.data();

        const size_t n = (size_t)golden.size();
        float* out = pY.data();
        // Call through the synthesis top: it only rebuilds cfg, and cosim captures argument traces there.
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
        // Y26_OUT_WORD: the top takes a wide-word pointer that is a little-endian view of the same float array
        // (lane 0 = element 0), so `out` still reads back as floats.
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
            // Per-oc params from the channel's own distribution: 5th percentile -> code 0, 95th -> 255.
            const int OHt = golden.H, OWt = golden.W, YSq = Y26_TB_YS(OWt);
            std::vector<float> vs((size_t)OHt * OWt);
            for (int o = 0; o < c.oc; ++o) {
                for (int r_ = 0; r_ < OHt; ++r_)
                    for (int w_ = 0; w_ < OWt; ++w_) vs[(size_t)r_ * OWt + w_] = out[((size_t)o * OHt + r_) * YSq + w_];
                std::sort(vs.begin(), vs.end());
                const float p5 = vs[vs.size() / 20], p95 = vs[vs.size() - 1 - vs.size() / 20];
                const float sc = 0.75f + 0.125f * (float)(o % 5);
                pQs[o] = sc; pQlo[o] = p5 * sc; pQst[o] = std::max((p95 - p5) * sc, 1e-4f) / 255.f;
                pQinv[o] = 1.f / pQst[o];
            }
            y26_yw_t* yptr2 = reinterpret_cast<y26_yw_t*>(pY2.data());
            y26_conv_top(xptr, wptr, cfg.wsc, cfg.bias, cfg.step_v, cfg.lo_v, yptr2,
                         H, W, cfg.oc, cfg.ic, cfg.kh, cfg.kw, cfg.sh, cfg.sw, cfg.ph, cfg.pw,
                         cfg.groups, cfg.act, cfg.perch, cfg.step, cfg.lo,
                         1, pQs.data(), pQlo.data(), pQinv.data());
            size_t qbad = 0;
            for (int o = 0; o < c.oc; ++o)
                for (int r_ = 0; r_ < OHt; ++r_)
                    for (int w_ = 0; w_ < OWt; ++w_) {
                        const size_t i = ((size_t)o * OHt + r_) * YSq + w_;
                        const uint32_t want = (uint32_t)(int)q_u8(out[i], pQs[o], pQlo[o], pQst[o]);
                        uint32_t slot; memcpy(&slot, &pY2[i], 4);
                        const uint32_t got = y26_yq_code(slot, pQs[o], pQlo[o], pQst[o]);
                        if (slot > 255) {                // a near-tie escape must carry the float slot itself
                            ++yq_ties;
                            uint32_t fb; memcpy(&fb, &out[i], 4);
                            qbad += slot - 256 != fb;
                        }
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

        // Comparison. Float dequant: exact equality. -DY26_FX_DEQUANT: tolerance + cosine, with per-conv statistics.
        size_t bad = 0;
        double wu = 0.0;
        double dot = 0.0, na = 0.0, nb = 0.0;
        const int YSt = Y26_TB_YS(golden.W);
        for (size_t i_ = 0; i_ < n; ++i_) {
            // Walk the tensor and index the buffer at the padded stride (the same index when unpadded).
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
                // Relative error only where the reference is not ~0.
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
        printf("\n=== YQ8 code-mode gate: %d convs, %d mismatched (%ld codes: %.1f%% at 0, %.1f%% at 255, %ld near-tie escapes) ===\n",
               yq_tested, yq_failed, yq_elems, yq_elems ? 100.0 * yq_clamp0 / yq_elems : 0.0,
               yq_elems ? 100.0 * yq_clamp255 / yq_elems : 0.0, yq_ties);
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
    // Indicative bar: cosine must stay essentially 1. The real gate for fixed point is accuracy (mAP).
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
