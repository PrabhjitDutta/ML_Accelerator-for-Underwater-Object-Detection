// detection_decode.h -- host-side (ARM PS) decode of the YOLO26 one2one head.
//
// This is the off-fabric tail of the deployment: the trunk (backbone + neck + head prediction convs)
// runs on the PL fabric / C-sim and emits three raw one2one prediction maps; this file turns those into
// final detections. It is deliberately dependency-free portable C++ (no PyTorch, no ultralytics, no
// third-party headers) so the SAME source compiles with g++ for the C-sim here AND cross-compiles for
// the ZCU102 ARM Cortex-A53 PS unchanged.
//
// The model is DFL-free (reg_max=1 -> the DFL is Identity, box channels ARE the ltrb distances) and
// NMS-free (one2one head, one-to-one label assignment -> no duplicate boxes to suppress). So the whole
// decode is: dist2bbox (xyxy) * stride  +  sigmoid on class logits  +  a two-stage top-k selection.
//
// Bit-faithful to ultralytics' end2end Detect path (verified against Yolo26Trunk.decode in
// training/yolo26s/yolo26_trunk.py):
//   _inference:  dbox = dist2bbox(boxes, anchors, xywh=False /*forced by end2end*/) * strides
//                out  = cat(dbox, sigmoid(scores))
//   postprocess: get_topk_index(scores, max_det) then gather -> [x1,y1,x2,y2, score, class]
// The xywh flag matters: end2end forces dist2bbox to return xyxy, NOT the xywh default.
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>

namespace yolo26 {

struct Det { float x1, y1, x2, y2, score; int cls; };

// Three one2one maps, each [C=4+nc, H, W] row-major (channel outer, then row, then col).
//   scale 0: 80x80 stride 8   scale 1: 40x40 stride 16   scale 2: 20x20 stride 32   (640 input)
//   channels [0..3] = ltrb distances (grid units), channels [4..4+nc-1] = per-class logits.
// Returns up to max_det detections in xyxy pixel coords (640-input space), sorted by score descending.
inline std::vector<Det> decode_o2o(const float* m0, const float* m1, const float* m2,
                                   int nc = 4, int max_det = 300) {
    struct Scale { const float* m; int H, W, stride; };
    const Scale sc[3] = {{m0, 80, 80, 8}, {m1, 40, 40, 16}, {m2, 20, 20, 32}};

    const int nA = 80 * 80 + 40 * 40 + 20 * 20;   // 8400 anchors total
    // Per-anchor max-class logit (sigmoid is monotone, so ranking by logit == ranking by sigmoid;
    // we only sigmoid the survivors). gidx enumerates scale0 row-major, then scale1, then scale2.
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

    // Stage 1: top-k anchors by max-class score (k = min(max_det, nA)). torch.topk selects the k
    // largest as a set; stage-1 ORDER does not matter (stage 2 re-ranks), only membership does.
    const int k = std::min(max_det, nA);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return amax[a] > amax[b]; });

    // Decode the box for each stage-1 anchor (dist2bbox xyxy * stride). Map gidx -> (scale, p).
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

    // Stage 2: among the k stage-1 anchors x nc classes, take the top-k (anchor,class) pairs by score.
    struct Cand { float score; int gi; int cls; };
    std::vector<Cand> cand;
    cand.reserve((size_t)k * nc);
    for (int i = 0; i < k; ++i) {
        const int gi = idx[i];
        // recover (scale,p) once to read class logits
        int s = 0, base = 0;
        for (; s < 3; ++s) { int HW = sc[s].H * sc[s].W; if (gi < base + HW) break; base += HW; }
        const int HW = sc[s].H * sc[s].W;
        const int p  = gi - base;
        for (int c = 0; c < nc; ++c) {
            const float logit = sc[s].m[(size_t)(4 + c) * HW + p];
            cand.push_back({1.f / (1.f + std::exp(-logit)), gi, c});   // sigmoid
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
    return out;   // sorted by score descending, xyxy in 640-input pixel space
}

}  // namespace yolo26
