// PL ops block (ops_engine.cpp): y26_ops_ref (ops_int.h) on 128-bit port words.
#pragma once
#include <ap_int.h>
#include "ops_int.h"

#define Y26_OPS_CMAX 1024        // channels per call
#define Y26_OPS_WU   5           // port words per unit
// Co-simulation depths (port words): the testbench's arrays are exactly this long.
#ifndef Y26_OPS_DEPTH
#define Y26_OPS_DEPTH 4096
#endif

// Sa: the source planes; Sb: the planes to add (bmode != 0); Dst: the result; Par: one OpsPar word per channel.
// C <= Y26_OPS_CMAX; units = 80-byte units per destination plane; amode, bmode: ops_int.h (y26_ops_ok).
void y26_ops_top(const ap_uint<128>* Sa, const ap_uint<128>* Sb, ap_uint<128>* Dst, const ap_uint<128>* Par,
                 int C, int units, int amode, int bmode);
