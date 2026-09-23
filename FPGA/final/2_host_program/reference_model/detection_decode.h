// Decodes the YOLO26 one2one head on the host CPU. Dependency-free, so it also builds on the board.
// The model is DFL-free (box channels are ltrb distances) and NMS-free (one2one head), so decoding is:
// dist2bbox (xyxy) * stride, sigmoid on class logits, then a two-stage top-k, as in ultralytics' end2end path.
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>

namespace yolo26 {

struct Det { float x1, y1, x2, y2, score; int cls; };

// Maps are [4+nc, H, W]: channels 0..3 = ltrb distances (grid units), then class logits.
// Scales: 80x80 stride 8, 40x40 stride 16, 20x20 stride 32. Returns up to max_det boxes, score-descending.
inline std::vector<Det> decode_o2o(const float* m0, const float* m1, const float* m2,
                                   int nc = 4, int max_det = 300) {
    struct Scale { const float* m; int H, W, stride; };
    const Scale sc[3] = {{m0, 80, 80, 8}, {m1, 40, 40, 16}, {m2, 20, 20, 32}};

    const int nA = 80 * 80 + 40 * 40 + 20 * 20;   // 8400 anchors total
    // Per-anchor max class logit; sigmoid is monotone, so only the survivors are sigmoided.
    std::vector<float> amax(nA);
    std::vector<int>   idx(nA);
    int g = 0;
    for (int s = 0; s < 3; ++s) {
        const int HW = sc[s].H * sc[s].W;
        for (int p = 0; p < HW; ++p) {
            float mx = -1e30f;
            for (int c = 0; c < nc; ++c) {
                float v = sc[s].m[(size_t)(4 + c) * HW + p];
                if (v > mx) mx = v;
            }
            amax[g] = mx; idx[g] = g; ++g;
        }
    }

    // Stage 1: the top-k anchors by max-class score (order irrelevant, stage 2 re-ranks).
    const int k = std::min(max_det, nA);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return amax[a] > amax[b]; });

    auto decode_box = [&](int gi, float& x1, float& y1, float& x2, float& y2) {
        int s = 0, base = 0;
        for (; s < 3; ++s) { int HW = sc[s].H * sc[s].W; if (gi < base + HW) break; base += HW; }
        const int HW = sc[s].H * sc[s].W;
        const int p  = gi - base;
        const int px = p % sc[s].W, py = p / sc[s].W;
        const float ax = px + 0.5f, ay = py + 0.5f;     // grid-cell centre (offset 0.5)
        const float l = sc[s].m[(size_t)0 * HW + p], t = sc[s].m[(size_t)1 * HW + p];
        const float r = sc[s].m[(size_t)2 * HW + p], b = sc[s].m[(size_t)3 * HW + p];
        const float st = (float)sc[s].stride;
        x1 = (ax - l) * st; y1 = (ay - t) * st;         // dist2bbox xyxy, then * stride
        x2 = (ax + r) * st; y2 = (ay + b) * st;
    };

    // Stage 2: top-k (anchor, class) pairs by score.
    struct Cand { float score; int gi; int cls; };
    std::vector<Cand> cand;
    cand.reserve((size_t)k * nc);
    for (int i = 0; i < k; ++i) {
        const int gi = idx[i];
        int s = 0, base = 0;
        for (; s < 3; ++s) { int HW = sc[s].H * sc[s].W; if (gi < base + HW) break; base += HW; }
        const int HW = sc[s].H * sc[s].W;
        const int p  = gi - base;
        for (int c = 0; c < nc; ++c) {
            const float logit = sc[s].m[(size_t)(4 + c) * HW + p];
            cand.push_back({1.f / (1.f + std::exp(-logit)), gi, c});
        }
    }
    const int kk = std::min(max_det, (int)cand.size());
    std::partial_sort(cand.begin(), cand.begin() + kk, cand.end(),
                      [](const Cand& a, const Cand& b) { return a.score > b.score; });

    std::vector<Det> out;
    out.reserve(kk);
    for (int i = 0; i < kk; ++i) {
        Det d; d.score = cand[i].score; d.cls = cand[i].cls;
        decode_box(cand[i].gi, d.x1, d.y1, d.x2, d.y2);
        out.push_back(d);
    }
    return out;
}

}
