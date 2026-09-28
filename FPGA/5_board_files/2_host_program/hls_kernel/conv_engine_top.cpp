// AXI top level (the synthesis top). No arithmetic: rebuilds Y26ConvCfg from the flattened AXI arguments and
// calls the datapath. m_axi: every buffer, in DDR. s_axilite (bundle control): scalars, buffer base addresses,
// ap_ctrl. Y26ConvCfg holds pointers, which neither interface can carry, so it is rebuilt here.
#include "conv_engine.h"

void y26_conv_top(const y26_xw_t*  X,
                  const y26_ww_t*  Wt,
                  const float*     wsc,
                  const float*     bias,
                  const float*     step_v,
                  const float*     lo_v,
                  y26_yw_t*        Y,
                  int H, int W,
                  int oc, int ic, int kh, int kw,
                  int sh, int sw, int ph, int pw,
                  int groups, int act, int perch,
                  float step, float lo
#ifdef Y26_YQ8
                  , int yq, const float* q_s, const float* q_lo, const float* q_st
#endif
                  ) {
    // Buffers: separate bundles so weight and activation traffic do not serialize.
    // depth= is for co-simulation only (see Y26_DEPTH_* in conv_engine.h).
    // latency= tells HLS the DRAM latency so it pipelines deep enough to keep reads in flight.
    // Both ports of a bundle get identical options.
#ifndef Y26_ACT_LATENCY
#define Y26_ACT_LATENCY 30
#endif
// Reads in flight = min(pipeline depth, num_read_outstanding): raise it together with Y26_ACT_LATENCY.
// Costs BRAM in the AXI read adapter.
#ifndef Y26_ACT_NRO
#define Y26_ACT_NRO 64
#endif
// gmem_out is write-only, so only num_write_outstanding matters here.
#ifndef Y26_OUT_NWO
#define Y26_OUT_NWO 16
#endif
#ifndef Y26_OUT_LATENCY
#define Y26_OUT_LATENCY Y26_ACT_LATENCY
#endif
#ifndef Y26_PRAGMA_STR2
#define Y26_PRAGMA_STR2(x) #x
#define Y26_PRAGMA_STR(x)  Y26_PRAGMA_STR2(x)
#endif
    // Y26_SPLIT_OUT: Y on its own bundle. A shared bundle is as wide as its widest type (float Y, 32 bits),
    // which blocks bursting the 8-bit X reads.
#ifdef Y26_SPLIT_OUT
    _Pragma(Y26_PRAGMA_STR(HLS INTERFACE m_axi port=X offset=slave bundle=gmem_act
                           depth=Y26_DEPTH_XW num_read_outstanding=Y26_ACT_NRO latency=Y26_ACT_LATENCY))
    _Pragma(Y26_PRAGMA_STR(HLS INTERFACE m_axi port=Y offset=slave bundle=gmem_out
                           depth=Y26_DEPTH_YW num_write_outstanding=Y26_OUT_NWO
                           latency=Y26_OUT_LATENCY))
#else
    _Pragma(Y26_PRAGMA_STR(HLS INTERFACE m_axi port=X offset=slave bundle=gmem_act
                           depth=Y26_DEPTH_XW num_read_outstanding=Y26_ACT_NRO latency=Y26_ACT_LATENCY))
    _Pragma(Y26_PRAGMA_STR(HLS INTERFACE m_axi port=Y offset=slave bundle=gmem_act
                           depth=Y26_DEPTH_YW num_read_outstanding=Y26_ACT_NRO latency=Y26_ACT_LATENCY))
#endif
    // latency= makes HLS pipeline deep enough to cover DRAM latency; num_read_outstanding lets those reads be in
    // flight. Default 30 cycles ~ ZynqMP HP-port DDR4 read latency at 250 MHz.
    // _Pragma because HLS does not macro-expand inside #pragma.
    #ifndef Y26_WT_LATENCY
    #define Y26_WT_LATENCY 30
    #endif
    #define Y26_PRAGMA_STR2(x) #x
    #define Y26_PRAGMA_STR(x)  Y26_PRAGMA_STR2(x)
    _Pragma(Y26_PRAGMA_STR(HLS INTERFACE m_axi port=Wt offset=slave bundle=gmem_wt
                           depth=Y26_DEPTH_WTW num_read_outstanding=64 latency=Y26_WT_LATENCY))
    #pragma HLS INTERFACE m_axi port=wsc    offset=slave bundle=gmem_scale depth=Y26_DEPTH_OC
    #pragma HLS INTERFACE m_axi port=bias   offset=slave bundle=gmem_scale depth=Y26_DEPTH_OC
    #pragma HLS INTERFACE m_axi port=step_v offset=slave bundle=gmem_scale depth=Y26_DEPTH_IC
    #pragma HLS INTERFACE m_axi port=lo_v   offset=slave bundle=gmem_scale depth=Y26_DEPTH_IC
#ifdef Y26_YQ8
    #pragma HLS INTERFACE m_axi port=q_s    offset=slave bundle=gmem_scale depth=Y26_DEPTH_OC
    #pragma HLS INTERFACE m_axi port=q_lo   offset=slave bundle=gmem_scale depth=Y26_DEPTH_OC
    #pragma HLS INTERFACE m_axi port=q_st   offset=slave bundle=gmem_scale depth=Y26_DEPTH_OC
    #pragma HLS INTERFACE s_axilite port=q_s  bundle=control
    #pragma HLS INTERFACE s_axilite port=q_lo bundle=control
    #pragma HLS INTERFACE s_axilite port=q_st bundle=control
    #pragma HLS INTERFACE s_axilite port=yq   bundle=control
#endif

    // Base addresses, scalars, control.
    #pragma HLS INTERFACE s_axilite port=X      bundle=control
    #pragma HLS INTERFACE s_axilite port=Y      bundle=control
    #pragma HLS INTERFACE s_axilite port=Wt     bundle=control
    #pragma HLS INTERFACE s_axilite port=wsc    bundle=control
    #pragma HLS INTERFACE s_axilite port=bias   bundle=control
    #pragma HLS INTERFACE s_axilite port=step_v bundle=control
    #pragma HLS INTERFACE s_axilite port=lo_v   bundle=control
    #pragma HLS INTERFACE s_axilite port=H      bundle=control
    #pragma HLS INTERFACE s_axilite port=W      bundle=control
    #pragma HLS INTERFACE s_axilite port=oc     bundle=control
    #pragma HLS INTERFACE s_axilite port=ic     bundle=control
    #pragma HLS INTERFACE s_axilite port=kh     bundle=control
    #pragma HLS INTERFACE s_axilite port=kw     bundle=control
    #pragma HLS INTERFACE s_axilite port=sh     bundle=control
    #pragma HLS INTERFACE s_axilite port=sw     bundle=control
    #pragma HLS INTERFACE s_axilite port=ph     bundle=control
    #pragma HLS INTERFACE s_axilite port=pw     bundle=control
    #pragma HLS INTERFACE s_axilite port=groups bundle=control
    #pragma HLS INTERFACE s_axilite port=act    bundle=control
    #pragma HLS INTERFACE s_axilite port=perch  bundle=control
    #pragma HLS INTERFACE s_axilite port=step   bundle=control
    #pragma HLS INTERFACE s_axilite port=lo     bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    Y26ConvCfg c;
    c.oc = oc; c.ic = ic; c.kh = kh; c.kw = kw;
    c.sh = sh; c.sw = sw; c.ph = ph; c.pw = pw;
    c.groups = groups; c.act = act; c.perch = perch;
    c.step = step; c.lo = lo;
    c.step_v = step_v; c.lo_v = lo_v;
    c.wsc = wsc; c.bias = bias;
#ifdef Y26_YQ8
    c.yq = yq; c.q_s = q_s; c.q_lo = q_lo; c.q_st = q_st;
#endif

    y26_conv2d_hls(X, H, W, Wt, c, Y);
}
