// tensor_types.h - FP32 C-simulation types for the pruned50 YOLO26s trunk.
//
// This is the float reference model: weight_t == acc_t == float, so the C-sim reproduces the PyTorch
// graph to numerical precision. The INT8 fake-quant pass (later phase) swaps weight_t/act_t for
// fixed-point and adds per-tensor scales; the graph code stays identical.
#pragma once
#include <vector>
#include <cstddef>
#include <memory>
#include <new>
#include <utility>

// Allocator whose value-initialization is a no-op, so Tensor(c,h,w,NoInit) skips the zero fill that an op
// would overwrite anyway. Explicit values (Tensor(c,h,w) passes 0.f) are constructed as usual.
#ifdef Y26_BOARD
void* y26_tensor_alloc(size_t bytes);             // board_host/board_host.cpp: tensors live in the DMA buffer
void  y26_tensor_free(void* p, size_t bytes);
#endif
template <class T> struct NoInitAlloc : std::allocator<T> {
    template <class U> struct rebind { using other = NoInitAlloc<U>; };
    NoInitAlloc() = default;
    template <class U> NoInitAlloc(const NoInitAlloc<U>&) noexcept {}
    template <class U> void construct(U*) noexcept {}
    template <class U, class... A> void construct(U* p, A&&... a) { ::new ((void*)p) U(std::forward<A>(a)...); }
#ifdef Y26_BOARD
    T* allocate(size_t n) { return static_cast<T*>(y26_tensor_alloc(n * sizeof(T))); }
    void deallocate(T* p, size_t n) { y26_tensor_free(p, n * sizeof(T)); }
#endif
};
struct NoInit {};

// A batch-1 CHW tensor, contiguous as d[(c*H + h)*W + w].
struct Tensor {
    int C = 0, H = 0, W = 0;
    std::vector<float, NoInitAlloc<float>> d;
    Tensor() {}
    Tensor(int c, int h, int w) : C(c), H(h), W(w), d((size_t)c * h * w, 0.f) {}
    Tensor(int c, int h, int w, NoInit) : C(c), H(h), W(w), d((size_t)c * h * w) {}   // caller writes every element
    inline float&       operator()(int c, int h, int w)       { return d[((size_t)c * H + h) * W + w]; }
    inline const float& operator()(int c, int h, int w) const { return d[((size_t)c * H + h) * W + w]; }
    size_t size() const { return d.size(); }
};

enum Act { ACT_IDENTITY = 0, ACT_SILU = 1 };
