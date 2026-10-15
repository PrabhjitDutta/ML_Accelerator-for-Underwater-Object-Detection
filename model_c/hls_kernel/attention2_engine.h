// PL attention block v2 (attention2_engine.cpp): the Attention module between its two convs, on 8-bit code planes.
#pragma once
#include "attention_engine.h"

#define Y26_ATTN2_XC (2 * Y26_ATTN_KD + Y26_ATTN_VD)     // code planes per head in X
#define Y26_ATTN2_PW (12 * Y26_ATTN_VD / 4)              // port words of parameters per head (y26_attn2_pack)
#define Y26_ATTN2_DEPTH_X (Y26_ATTN_NH * Y26_ATTN2_XC * Y26_ATTN_NMAX / 16)
#define Y26_ATTN2_DEPTH_D (Y26_ATTN_NH * Y26_ATTN_VD * Y26_ATTN_NMAX / 16)
#define Y26_ATTN2_DEPTH_P (Y26_ATTN_NH * Y26_ATTN2_PW)

// m, g: AttnInt (attention_int.h). nh <= Y26_ATTN_NH; n <= Y26_ATTN_NMAX, a multiple of 16; w: the map's width.
void y26_attn2_top(const ap_uint<128>* X, ap_uint<128>* D, const ap_uint<128>* P, int nh, int n, int w, ap_uint<24> m, ap_uint<26> g);
