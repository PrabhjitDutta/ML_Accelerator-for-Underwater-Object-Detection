// PL attention block (attention_engine.cpp). Sizes are fixed by the model: C = 256, 4 heads, at most 20x20 positions.
#pragma once
#include <ap_int.h>

#define Y26_ATTN_KD   32     // key_dim
#define Y26_ATTN_VD   64     // head_dim
#define Y26_ATTN_NMAX 400    // positions per head
#define Y26_ATTN_NH   4
#ifndef Y26_ATTN_QB
#define Y26_ATTN_QB   4      // queries in parallel
#endif
// Port words (128 bits) per position: X record 128 bytes, Y record 4*VD bytes.
#define Y26_ATTN_XW 8
#define Y26_ATTN_YW (Y26_ATTN_VD / 4)
#define Y26_ATTN_DEPTH_X (Y26_ATTN_NH * Y26_ATTN_NMAX * Y26_ATTN_XW)
#define Y26_ATTN_DEPTH_Y (Y26_ATTN_NH * Y26_ATTN_NMAX * Y26_ATTN_YW)

// m, g: AttnInt (attention_int.h). nh <= Y26_ATTN_NH, n <= Y26_ATTN_NMAX.
void y26_attn_top(const ap_uint<128>* X, ap_uint<128>* Y, int nh, int n, ap_uint<24> m, ap_uint<26> g);
