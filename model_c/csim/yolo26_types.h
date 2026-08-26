// yolo26_types.h - FP32 C-simulation types for the pruned50 YOLO26s trunk.
//
// This is the float reference model: weight_t == acc_t == float, so the C-sim reproduces the PyTorch
// graph to numerical precision. The INT8 fake-quant pass (later phase) swaps weight_t/act_t for
// fixed-point and adds per-tensor scales; the graph code stays identical.
#pragma once
#include <vector>
#include <cstddef>

// A batch-1 CHW tensor, contiguous as d[(c*H + h)*W + w].
struct Tensor {
    int C = 0, H = 0, W = 0;
    std::vector<float> d;
    Tensor() {}
    Tensor(int c, int h, int w) : C(c), H(h), W(w), d((size_t)c * h * w, 0.f) {}
    inline float&       operator()(int c, int h, int w)       { return d[((size_t)c * H + h) * W + w]; }
    inline const float& operator()(int c, int h, int w) const { return d[((size_t)c * H + h) * W + w]; }
    size_t size() const { return d.size(); }
};

enum Act { ACT_IDENTITY = 0, ACT_SILU = 1 };
