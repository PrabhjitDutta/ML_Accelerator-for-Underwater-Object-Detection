// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2026.1 (64-bit)
// Tool Version Limit: 2026.06
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2026 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
`timescale 1ns/1ps
(* DowngradeIPIdentifiedWarnings="yes" *) module y26_conv_top_control_s_axi
#(parameter
    C_S_AXI_ADDR_WIDTH = 9,
    C_S_AXI_DATA_WIDTH = 32
)(
    input  wire                          ACLK,
    input  wire                          ARESET,
    input  wire                          ACLK_EN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] AWADDR,
    input  wire                          AWVALID,
    output wire                          AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1:0] WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0] WSTRB,
    input  wire                          WVALID,
    output wire                          WREADY,
    output wire [1:0]                    BRESP,
    output wire                          BVALID,
    input  wire                          BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] ARADDR,
    input  wire                          ARVALID,
    output wire                          ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0] RDATA,
    output wire [1:0]                    RRESP,
    output wire                          RVALID,
    input  wire                          RREADY,
    output wire                          interrupt,
    output wire [63:0]                   X,
    output wire [63:0]                   Wt,
    output wire [63:0]                   wsc,
    output wire [63:0]                   bias,
    output wire [63:0]                   step_v,
    output wire [63:0]                   lo_v,
    output wire [63:0]                   Y,
    output wire [31:0]                   H,
    output wire [31:0]                   W,
    output wire [31:0]                   oc,
    output wire [31:0]                   ic,
    output wire [31:0]                   kh,
    output wire [31:0]                   kw,
    output wire [31:0]                   sh,
    output wire [31:0]                   sw,
    output wire [31:0]                   ph,
    output wire [31:0]                   pw,
    output wire [31:0]                   groups,
    output wire [31:0]                   act,
    output wire [31:0]                   perch,
    output wire [31:0]                   step,
    output wire [31:0]                   lo,
    output wire [63:0]                   Yc,
    output wire [31:0]                   yq,
    output wire [63:0]                   q_k,
    output wire [63:0]                   q_b,
    output wire                          ap_start,
    input  wire                          ap_done,
    input  wire                          ap_ready,
    input  wire                          ap_idle
);
//------------------------Address Info-------------------
// Protocol Used: ap_ctrl_hs
//
// 0x000 : Control signals
//         bit 0  - ap_start (Read/Write/COH)
//         bit 1  - ap_done (Read/COR)
//         bit 2  - ap_idle (Read)
//         bit 3  - ap_ready (Read/COR)
//         bit 7  - auto_restart (Read/Write)
//         bit 9  - interrupt (Read)
//         others - reserved
// 0x004 : Global Interrupt Enable Register
//         bit 0  - Global Interrupt Enable (Read/Write)
//         others - reserved
// 0x008 : IP Interrupt Enable Register (Read/Write)
//         bit 0 - enable ap_done interrupt (Read/Write)
//         bit 1 - enable ap_ready interrupt (Read/Write)
//         others - reserved
// 0x00c : IP Interrupt Status Register (Read/TOW)
//         bit 0 - ap_done (Read/TOW)
//         bit 1 - ap_ready (Read/TOW)
//         others - reserved
// 0x010 : Data signal of X
//         bit 31~0 - X[31:0] (Read/Write)
// 0x014 : Data signal of X
//         bit 31~0 - X[63:32] (Read/Write)
// 0x018 : reserved
// 0x01c : Data signal of Wt
//         bit 31~0 - Wt[31:0] (Read/Write)
// 0x020 : Data signal of Wt
//         bit 31~0 - Wt[63:32] (Read/Write)
// 0x024 : reserved
// 0x028 : Data signal of wsc
//         bit 31~0 - wsc[31:0] (Read/Write)
// 0x02c : Data signal of wsc
//         bit 31~0 - wsc[63:32] (Read/Write)
// 0x030 : reserved
// 0x034 : Data signal of bias
//         bit 31~0 - bias[31:0] (Read/Write)
// 0x038 : Data signal of bias
//         bit 31~0 - bias[63:32] (Read/Write)
// 0x03c : reserved
// 0x040 : Data signal of step_v
//         bit 31~0 - step_v[31:0] (Read/Write)
// 0x044 : Data signal of step_v
//         bit 31~0 - step_v[63:32] (Read/Write)
// 0x048 : reserved
// 0x04c : Data signal of lo_v
//         bit 31~0 - lo_v[31:0] (Read/Write)
// 0x050 : Data signal of lo_v
//         bit 31~0 - lo_v[63:32] (Read/Write)
// 0x054 : reserved
// 0x058 : Data signal of Y
//         bit 31~0 - Y[31:0] (Read/Write)
// 0x05c : Data signal of Y
//         bit 31~0 - Y[63:32] (Read/Write)
// 0x060 : reserved
// 0x064 : Data signal of H
//         bit 31~0 - H[31:0] (Read/Write)
// 0x068 : reserved
// 0x06c : Data signal of W
//         bit 31~0 - W[31:0] (Read/Write)
// 0x070 : reserved
// 0x074 : Data signal of oc
//         bit 31~0 - oc[31:0] (Read/Write)
// 0x078 : reserved
// 0x07c : Data signal of ic
//         bit 31~0 - ic[31:0] (Read/Write)
// 0x080 : reserved
// 0x084 : Data signal of kh
//         bit 31~0 - kh[31:0] (Read/Write)
// 0x088 : reserved
// 0x08c : Data signal of kw
//         bit 31~0 - kw[31:0] (Read/Write)
// 0x090 : reserved
// 0x094 : Data signal of sh
//         bit 31~0 - sh[31:0] (Read/Write)
// 0x098 : reserved
// 0x09c : Data signal of sw
//         bit 31~0 - sw[31:0] (Read/Write)
// 0x0a0 : reserved
// 0x0a4 : Data signal of ph
//         bit 31~0 - ph[31:0] (Read/Write)
// 0x0a8 : reserved
// 0x0ac : Data signal of pw
//         bit 31~0 - pw[31:0] (Read/Write)
// 0x0b0 : reserved
// 0x0b4 : Data signal of groups
//         bit 31~0 - groups[31:0] (Read/Write)
// 0x0b8 : reserved
// 0x0bc : Data signal of act
//         bit 31~0 - act[31:0] (Read/Write)
// 0x0c0 : reserved
// 0x0c4 : Data signal of perch
//         bit 31~0 - perch[31:0] (Read/Write)
// 0x0c8 : reserved
// 0x0cc : Data signal of step
//         bit 31~0 - step[31:0] (Read/Write)
// 0x0d0 : reserved
// 0x0d4 : Data signal of lo
//         bit 31~0 - lo[31:0] (Read/Write)
// 0x0d8 : reserved
// 0x0dc : Data signal of Yc
//         bit 31~0 - Yc[31:0] (Read/Write)
// 0x0e0 : Data signal of Yc
//         bit 31~0 - Yc[63:32] (Read/Write)
// 0x0e4 : reserved
// 0x0e8 : Data signal of yq
//         bit 31~0 - yq[31:0] (Read/Write)
// 0x0ec : reserved
// 0x0f0 : Data signal of q_k
//         bit 31~0 - q_k[31:0] (Read/Write)
// 0x0f4 : Data signal of q_k
//         bit 31~0 - q_k[63:32] (Read/Write)
// 0x0f8 : reserved
// 0x0fc : Data signal of q_b
//         bit 31~0 - q_b[31:0] (Read/Write)
// 0x100 : Data signal of q_b
//         bit 31~0 - q_b[63:32] (Read/Write)
// 0x104 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

//------------------------Parameter----------------------
localparam
    ADDR_AP_CTRL       = 9'h000,
    ADDR_GIE           = 9'h004,
    ADDR_IER           = 9'h008,
    ADDR_ISR           = 9'h00c,
    ADDR_X_DATA_0      = 9'h010,
    ADDR_X_DATA_1      = 9'h014,
    ADDR_X_CTRL        = 9'h018,
    ADDR_WT_DATA_0     = 9'h01c,
    ADDR_WT_DATA_1     = 9'h020,
    ADDR_WT_CTRL       = 9'h024,
    ADDR_WSC_DATA_0    = 9'h028,
    ADDR_WSC_DATA_1    = 9'h02c,
    ADDR_WSC_CTRL      = 9'h030,
    ADDR_BIAS_DATA_0   = 9'h034,
    ADDR_BIAS_DATA_1   = 9'h038,
    ADDR_BIAS_CTRL     = 9'h03c,
    ADDR_STEP_V_DATA_0 = 9'h040,
    ADDR_STEP_V_DATA_1 = 9'h044,
    ADDR_STEP_V_CTRL   = 9'h048,
    ADDR_LO_V_DATA_0   = 9'h04c,
    ADDR_LO_V_DATA_1   = 9'h050,
    ADDR_LO_V_CTRL     = 9'h054,
    ADDR_Y_DATA_0      = 9'h058,
    ADDR_Y_DATA_1      = 9'h05c,
    ADDR_Y_CTRL        = 9'h060,
    ADDR_H_DATA_0      = 9'h064,
    ADDR_H_CTRL        = 9'h068,
    ADDR_W_DATA_0      = 9'h06c,
    ADDR_W_CTRL        = 9'h070,
    ADDR_OC_DATA_0     = 9'h074,
    ADDR_OC_CTRL       = 9'h078,
    ADDR_IC_DATA_0     = 9'h07c,
    ADDR_IC_CTRL       = 9'h080,
    ADDR_KH_DATA_0     = 9'h084,
    ADDR_KH_CTRL       = 9'h088,
    ADDR_KW_DATA_0     = 9'h08c,
    ADDR_KW_CTRL       = 9'h090,
    ADDR_SH_DATA_0     = 9'h094,
    ADDR_SH_CTRL       = 9'h098,
    ADDR_SW_DATA_0     = 9'h09c,
    ADDR_SW_CTRL       = 9'h0a0,
    ADDR_PH_DATA_0     = 9'h0a4,
    ADDR_PH_CTRL       = 9'h0a8,
    ADDR_PW_DATA_0     = 9'h0ac,
    ADDR_PW_CTRL       = 9'h0b0,
    ADDR_GROUPS_DATA_0 = 9'h0b4,
    ADDR_GROUPS_CTRL   = 9'h0b8,
    ADDR_ACT_DATA_0    = 9'h0bc,
    ADDR_ACT_CTRL      = 9'h0c0,
    ADDR_PERCH_DATA_0  = 9'h0c4,
    ADDR_PERCH_CTRL    = 9'h0c8,
    ADDR_STEP_DATA_0   = 9'h0cc,
    ADDR_STEP_CTRL     = 9'h0d0,
    ADDR_LO_DATA_0     = 9'h0d4,
    ADDR_LO_CTRL       = 9'h0d8,
    ADDR_YC_DATA_0     = 9'h0dc,
    ADDR_YC_DATA_1     = 9'h0e0,
    ADDR_YC_CTRL       = 9'h0e4,
    ADDR_YQ_DATA_0     = 9'h0e8,
    ADDR_YQ_CTRL       = 9'h0ec,
    ADDR_Q_K_DATA_0    = 9'h0f0,
    ADDR_Q_K_DATA_1    = 9'h0f4,
    ADDR_Q_K_CTRL      = 9'h0f8,
    ADDR_Q_B_DATA_0    = 9'h0fc,
    ADDR_Q_B_DATA_1    = 9'h100,
    ADDR_Q_B_CTRL      = 9'h104,
    WRIDLE             = 2'd0,
    WRDATA             = 2'd1,
    WRRESP             = 2'd2,
    WRRESET            = 2'd3,
    RDIDLE             = 2'd0,
    RDDATA             = 2'd1,
    RDRESET            = 2'd2,
    ADDR_BITS                = 9;

//------------------------Local signal-------------------
    reg  [1:0]                    wstate = WRRESET;
    reg  [1:0]                    wnext;
    reg  [ADDR_BITS-1:0]          waddr;
    wire [C_S_AXI_DATA_WIDTH-1:0] wmask;
    wire                          aw_hs;
    wire                          w_hs;
    reg  [1:0]                    rstate = RDRESET;
    reg  [1:0]                    rnext;
    reg  [C_S_AXI_DATA_WIDTH-1:0] rdata;
    wire                          ar_hs;
    wire [ADDR_BITS-1:0]          raddr;
    // internal registers
    reg                           int_ap_idle = 1'b0;
    reg                           int_ap_ready = 1'b0;
    wire                          task_ap_ready;
    reg                           int_ap_done = 1'b0;
    wire                          task_ap_done;
    reg                           int_task_ap_done = 1'b0;
    reg                           int_ap_start = 1'b0;
    reg                           int_interrupt = 1'b0;
    reg                           int_auto_restart = 1'b0;
    reg                           auto_restart_status = 1'b0;
    wire                          auto_restart_done;
    reg                           int_gie = 1'b0;
    reg  [1:0]                    int_ier = 2'b0;
    reg  [1:0]                    int_isr = 2'b0;
    reg  [63:0]                   int_X = 'b0;
    reg  [63:0]                   int_Wt = 'b0;
    reg  [63:0]                   int_wsc = 'b0;
    reg  [63:0]                   int_bias = 'b0;
    reg  [63:0]                   int_step_v = 'b0;
    reg  [63:0]                   int_lo_v = 'b0;
    reg  [63:0]                   int_Y = 'b0;
    reg  [31:0]                   int_H = 'b0;
    reg  [31:0]                   int_W = 'b0;
    reg  [31:0]                   int_oc = 'b0;
    reg  [31:0]                   int_ic = 'b0;
    reg  [31:0]                   int_kh = 'b0;
    reg  [31:0]                   int_kw = 'b0;
    reg  [31:0]                   int_sh = 'b0;
    reg  [31:0]                   int_sw = 'b0;
    reg  [31:0]                   int_ph = 'b0;
    reg  [31:0]                   int_pw = 'b0;
    reg  [31:0]                   int_groups = 'b0;
    reg  [31:0]                   int_act = 'b0;
    reg  [31:0]                   int_perch = 'b0;
    reg  [31:0]                   int_step = 'b0;
    reg  [31:0]                   int_lo = 'b0;
    reg  [63:0]                   int_Yc = 'b0;
    reg  [31:0]                   int_yq = 'b0;
    reg  [63:0]                   int_q_k = 'b0;
    reg  [63:0]                   int_q_b = 'b0;

//------------------------Instantiation------------------


//------------------------AXI write fsm------------------
assign AWREADY = (wstate == WRIDLE);
assign WREADY  = (wstate == WRDATA);
assign BVALID  = (wstate == WRRESP);
assign BRESP   = 2'b00;  // OKAY
assign wmask   = { {8{WSTRB[3]}}, {8{WSTRB[2]}}, {8{WSTRB[1]}}, {8{WSTRB[0]}} };
assign aw_hs   = AWVALID & AWREADY;
assign w_hs    = WVALID & WREADY;

// wstate
always @(posedge ACLK) begin
    if (ARESET)
        wstate <= WRRESET;
    else if (ACLK_EN)
        wstate <= wnext;
end

// wnext
always @(*) begin
    case (wstate)
        WRIDLE:
            if (AWVALID)
                wnext = WRDATA;
            else
                wnext = WRIDLE;
        WRDATA:
            if (WVALID)
                wnext = WRRESP;
            else
                wnext = WRDATA;
        WRRESP:
            if (BREADY & BVALID)
                wnext = WRIDLE;
            else
                wnext = WRRESP;
        default:
            wnext = WRIDLE;
    endcase
end

// waddr
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (aw_hs)
            waddr <= {AWADDR[ADDR_BITS-1:2], {2{1'b0}}};
    end
end

//------------------------AXI read fsm-------------------
assign ARREADY = (rstate == RDIDLE);
assign RDATA   = rdata;
assign RRESP   = 2'b00;  // OKAY
assign RVALID  = (rstate == RDDATA);
assign ar_hs   = ARVALID & ARREADY;
assign raddr   = ARADDR[ADDR_BITS-1:0];

// rstate
always @(posedge ACLK) begin
    if (ARESET)
        rstate <= RDRESET;
    else if (ACLK_EN)
        rstate <= rnext;
end

// rnext
always @(*) begin
    case (rstate)
        RDIDLE:
            if (ARVALID)
                rnext = RDDATA;
            else
                rnext = RDIDLE;
        RDDATA:
            if (RREADY & RVALID)
                rnext = RDIDLE;
            else
                rnext = RDDATA;
        default:
            rnext = RDIDLE;
    endcase
end

// rdata
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (ar_hs) begin
            rdata <= 'b0;
            case (raddr)
                ADDR_AP_CTRL: begin
                    rdata[0] <= int_ap_start;
                    rdata[1] <= int_task_ap_done;
                    rdata[2] <= int_ap_idle;
                    rdata[3] <= int_ap_ready;
                    rdata[7] <= int_auto_restart;
                    rdata[9] <= int_interrupt;
                end
                ADDR_GIE: begin
                    rdata <= int_gie;
                end
                ADDR_IER: begin
                    rdata <= int_ier;
                end
                ADDR_ISR: begin
                    rdata <= int_isr;
                end
                ADDR_X_DATA_0: begin
                    rdata <= int_X[31:0];
                end
                ADDR_X_DATA_1: begin
                    rdata <= int_X[63:32];
                end
                ADDR_WT_DATA_0: begin
                    rdata <= int_Wt[31:0];
                end
                ADDR_WT_DATA_1: begin
                    rdata <= int_Wt[63:32];
                end
                ADDR_WSC_DATA_0: begin
                    rdata <= int_wsc[31:0];
                end
                ADDR_WSC_DATA_1: begin
                    rdata <= int_wsc[63:32];
                end
                ADDR_BIAS_DATA_0: begin
                    rdata <= int_bias[31:0];
                end
                ADDR_BIAS_DATA_1: begin
                    rdata <= int_bias[63:32];
                end
                ADDR_STEP_V_DATA_0: begin
                    rdata <= int_step_v[31:0];
                end
                ADDR_STEP_V_DATA_1: begin
                    rdata <= int_step_v[63:32];
                end
                ADDR_LO_V_DATA_0: begin
                    rdata <= int_lo_v[31:0];
                end
                ADDR_LO_V_DATA_1: begin
                    rdata <= int_lo_v[63:32];
                end
                ADDR_Y_DATA_0: begin
                    rdata <= int_Y[31:0];
                end
                ADDR_Y_DATA_1: begin
                    rdata <= int_Y[63:32];
                end
                ADDR_H_DATA_0: begin
                    rdata <= int_H[31:0];
                end
                ADDR_W_DATA_0: begin
                    rdata <= int_W[31:0];
                end
                ADDR_OC_DATA_0: begin
                    rdata <= int_oc[31:0];
                end
                ADDR_IC_DATA_0: begin
                    rdata <= int_ic[31:0];
                end
                ADDR_KH_DATA_0: begin
                    rdata <= int_kh[31:0];
                end
                ADDR_KW_DATA_0: begin
                    rdata <= int_kw[31:0];
                end
                ADDR_SH_DATA_0: begin
                    rdata <= int_sh[31:0];
                end
                ADDR_SW_DATA_0: begin
                    rdata <= int_sw[31:0];
                end
                ADDR_PH_DATA_0: begin
                    rdata <= int_ph[31:0];
                end
                ADDR_PW_DATA_0: begin
                    rdata <= int_pw[31:0];
                end
                ADDR_GROUPS_DATA_0: begin
                    rdata <= int_groups[31:0];
                end
                ADDR_ACT_DATA_0: begin
                    rdata <= int_act[31:0];
                end
                ADDR_PERCH_DATA_0: begin
                    rdata <= int_perch[31:0];
                end
                ADDR_STEP_DATA_0: begin
                    rdata <= int_step[31:0];
                end
                ADDR_LO_DATA_0: begin
                    rdata <= int_lo[31:0];
                end
                ADDR_YC_DATA_0: begin
                    rdata <= int_Yc[31:0];
                end
                ADDR_YC_DATA_1: begin
                    rdata <= int_Yc[63:32];
                end
                ADDR_YQ_DATA_0: begin
                    rdata <= int_yq[31:0];
                end
                ADDR_Q_K_DATA_0: begin
                    rdata <= int_q_k[31:0];
                end
                ADDR_Q_K_DATA_1: begin
                    rdata <= int_q_k[63:32];
                end
                ADDR_Q_B_DATA_0: begin
                    rdata <= int_q_b[31:0];
                end
                ADDR_Q_B_DATA_1: begin
                    rdata <= int_q_b[63:32];
                end
            endcase
        end
    end
end


//------------------------Register logic-----------------
assign interrupt         = int_interrupt;
assign ap_start          = int_ap_start;
assign task_ap_done      = (ap_done && !auto_restart_status) || auto_restart_done;
assign task_ap_ready     = ap_ready && !int_auto_restart;
assign auto_restart_done = auto_restart_status && (ap_idle && !int_ap_idle);
assign X                 = int_X;
assign Wt                = int_Wt;
assign wsc               = int_wsc;
assign bias              = int_bias;
assign step_v            = int_step_v;
assign lo_v              = int_lo_v;
assign Y                 = int_Y;
assign H                 = int_H;
assign W                 = int_W;
assign oc                = int_oc;
assign ic                = int_ic;
assign kh                = int_kh;
assign kw                = int_kw;
assign sh                = int_sh;
assign sw                = int_sw;
assign ph                = int_ph;
assign pw                = int_pw;
assign groups            = int_groups;
assign act               = int_act;
assign perch             = int_perch;
assign step              = int_step;
assign lo                = int_lo;
assign Yc                = int_Yc;
assign yq                = int_yq;
assign q_k               = int_q_k;
assign q_b               = int_q_b;
// int_interrupt
always @(posedge ACLK) begin
    if (ARESET)
        int_interrupt <= 1'b0;
    else if (ACLK_EN) begin
        if (int_gie && (|int_isr))
            int_interrupt <= 1'b1;
        else
            int_interrupt <= 1'b0;
    end
end

// int_ap_start
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_start <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AP_CTRL && WSTRB[0] && WDATA[0])
            int_ap_start <= 1'b1;
        else if (ap_ready)
            int_ap_start <= int_auto_restart; // clear on handshake/auto restart
    end
end

// int_ap_done
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_done <= 1'b0;
    else if (ACLK_EN) begin
            int_ap_done <= ap_done;
    end
end

// int_task_ap_done
always @(posedge ACLK) begin
    if (ARESET)
        int_task_ap_done <= 1'b0;
    else if (ACLK_EN) begin
        if (task_ap_done)
            int_task_ap_done <= 1'b1;
        else if (ar_hs && raddr == ADDR_AP_CTRL)
            int_task_ap_done <= 1'b0; // clear on read
    end
end

// int_ap_idle
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_idle <= 1'b0;
    else if (ACLK_EN) begin
            int_ap_idle <= ap_idle;
    end
end

// int_ap_ready
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_ready <= 1'b0;
    else if (ACLK_EN) begin
        if (task_ap_ready)
            int_ap_ready <= 1'b1;
        else if (ar_hs && raddr == ADDR_AP_CTRL)
            int_ap_ready <= 1'b0;
    end
end

// int_auto_restart
always @(posedge ACLK) begin
    if (ARESET)
        int_auto_restart <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AP_CTRL && WSTRB[0])
            int_auto_restart <= WDATA[7];
    end
end

// auto_restart_status
always @(posedge ACLK) begin
    if (ARESET)
        auto_restart_status <= 1'b0;
    else if (ACLK_EN) begin
        if (int_auto_restart)
            auto_restart_status <= 1'b1;
        else if (ap_idle)
            auto_restart_status <= 1'b0;
    end
end

// int_gie
always @(posedge ACLK) begin
    if (ARESET)
        int_gie <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_GIE && WSTRB[0])
            int_gie <= WDATA[0];
    end
end

// int_ier
always @(posedge ACLK) begin
    if (ARESET)
        int_ier <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IER && WSTRB[0])
            int_ier <= WDATA[1:0];
    end
end

// int_isr[0]
always @(posedge ACLK) begin
    if (ARESET)
        int_isr[0] <= 1'b0;
    else if (ACLK_EN) begin
        if (int_ier[0] & ap_done)
            int_isr[0] <= 1'b1;
        else if (w_hs && waddr == ADDR_ISR && WSTRB[0])
            int_isr[0] <= int_isr[0] ^ WDATA[0]; // toggle on write
    end
end

// int_isr[1]
always @(posedge ACLK) begin
    if (ARESET)
        int_isr[1] <= 1'b0;
    else if (ACLK_EN) begin
        if (int_ier[1] & ap_ready)
            int_isr[1] <= 1'b1;
        else if (w_hs && waddr == ADDR_ISR && WSTRB[0])
            int_isr[1] <= int_isr[1] ^ WDATA[1]; // toggle on write
    end
end

// int_X[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_X[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_X_DATA_0)
            int_X[31:0] <= (WDATA[31:0] & wmask) | (int_X[31:0] & ~wmask);
    end
end

// int_X[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_X[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_X_DATA_1)
            int_X[63:32] <= (WDATA[31:0] & wmask) | (int_X[63:32] & ~wmask);
    end
end

// int_Wt[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_Wt[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_WT_DATA_0)
            int_Wt[31:0] <= (WDATA[31:0] & wmask) | (int_Wt[31:0] & ~wmask);
    end
end

// int_Wt[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_Wt[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_WT_DATA_1)
            int_Wt[63:32] <= (WDATA[31:0] & wmask) | (int_Wt[63:32] & ~wmask);
    end
end

// int_wsc[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_wsc[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_WSC_DATA_0)
            int_wsc[31:0] <= (WDATA[31:0] & wmask) | (int_wsc[31:0] & ~wmask);
    end
end

// int_wsc[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_wsc[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_WSC_DATA_1)
            int_wsc[63:32] <= (WDATA[31:0] & wmask) | (int_wsc[63:32] & ~wmask);
    end
end

// int_bias[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_bias[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_BIAS_DATA_0)
            int_bias[31:0] <= (WDATA[31:0] & wmask) | (int_bias[31:0] & ~wmask);
    end
end

// int_bias[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_bias[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_BIAS_DATA_1)
            int_bias[63:32] <= (WDATA[31:0] & wmask) | (int_bias[63:32] & ~wmask);
    end
end

// int_step_v[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_step_v[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_STEP_V_DATA_0)
            int_step_v[31:0] <= (WDATA[31:0] & wmask) | (int_step_v[31:0] & ~wmask);
    end
end

// int_step_v[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_step_v[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_STEP_V_DATA_1)
            int_step_v[63:32] <= (WDATA[31:0] & wmask) | (int_step_v[63:32] & ~wmask);
    end
end

// int_lo_v[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_lo_v[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LO_V_DATA_0)
            int_lo_v[31:0] <= (WDATA[31:0] & wmask) | (int_lo_v[31:0] & ~wmask);
    end
end

// int_lo_v[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_lo_v[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LO_V_DATA_1)
            int_lo_v[63:32] <= (WDATA[31:0] & wmask) | (int_lo_v[63:32] & ~wmask);
    end
end

// int_Y[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_Y[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_Y_DATA_0)
            int_Y[31:0] <= (WDATA[31:0] & wmask) | (int_Y[31:0] & ~wmask);
    end
end

// int_Y[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_Y[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_Y_DATA_1)
            int_Y[63:32] <= (WDATA[31:0] & wmask) | (int_Y[63:32] & ~wmask);
    end
end

// int_H[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_H[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_H_DATA_0)
            int_H[31:0] <= (WDATA[31:0] & wmask) | (int_H[31:0] & ~wmask);
    end
end

// int_W[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_W[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_W_DATA_0)
            int_W[31:0] <= (WDATA[31:0] & wmask) | (int_W[31:0] & ~wmask);
    end
end

// int_oc[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_oc[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_OC_DATA_0)
            int_oc[31:0] <= (WDATA[31:0] & wmask) | (int_oc[31:0] & ~wmask);
    end
end

// int_ic[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_ic[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IC_DATA_0)
            int_ic[31:0] <= (WDATA[31:0] & wmask) | (int_ic[31:0] & ~wmask);
    end
end

// int_kh[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_kh[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_KH_DATA_0)
            int_kh[31:0] <= (WDATA[31:0] & wmask) | (int_kh[31:0] & ~wmask);
    end
end

// int_kw[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_kw[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_KW_DATA_0)
            int_kw[31:0] <= (WDATA[31:0] & wmask) | (int_kw[31:0] & ~wmask);
    end
end

// int_sh[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_sh[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_SH_DATA_0)
            int_sh[31:0] <= (WDATA[31:0] & wmask) | (int_sh[31:0] & ~wmask);
    end
end

// int_sw[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_sw[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_SW_DATA_0)
            int_sw[31:0] <= (WDATA[31:0] & wmask) | (int_sw[31:0] & ~wmask);
    end
end

// int_ph[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_ph[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_PH_DATA_0)
            int_ph[31:0] <= (WDATA[31:0] & wmask) | (int_ph[31:0] & ~wmask);
    end
end

// int_pw[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_pw[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_PW_DATA_0)
            int_pw[31:0] <= (WDATA[31:0] & wmask) | (int_pw[31:0] & ~wmask);
    end
end

// int_groups[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_groups[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_GROUPS_DATA_0)
            int_groups[31:0] <= (WDATA[31:0] & wmask) | (int_groups[31:0] & ~wmask);
    end
end

// int_act[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_act[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_ACT_DATA_0)
            int_act[31:0] <= (WDATA[31:0] & wmask) | (int_act[31:0] & ~wmask);
    end
end

// int_perch[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_perch[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_PERCH_DATA_0)
            int_perch[31:0] <= (WDATA[31:0] & wmask) | (int_perch[31:0] & ~wmask);
    end
end

// int_step[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_step[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_STEP_DATA_0)
            int_step[31:0] <= (WDATA[31:0] & wmask) | (int_step[31:0] & ~wmask);
    end
end

// int_lo[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_lo[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LO_DATA_0)
            int_lo[31:0] <= (WDATA[31:0] & wmask) | (int_lo[31:0] & ~wmask);
    end
end

// int_Yc[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_Yc[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_YC_DATA_0)
            int_Yc[31:0] <= (WDATA[31:0] & wmask) | (int_Yc[31:0] & ~wmask);
    end
end

// int_Yc[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_Yc[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_YC_DATA_1)
            int_Yc[63:32] <= (WDATA[31:0] & wmask) | (int_Yc[63:32] & ~wmask);
    end
end

// int_yq[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_yq[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_YQ_DATA_0)
            int_yq[31:0] <= (WDATA[31:0] & wmask) | (int_yq[31:0] & ~wmask);
    end
end

// int_q_k[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_q_k[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_Q_K_DATA_0)
            int_q_k[31:0] <= (WDATA[31:0] & wmask) | (int_q_k[31:0] & ~wmask);
    end
end

// int_q_k[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_q_k[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_Q_K_DATA_1)
            int_q_k[63:32] <= (WDATA[31:0] & wmask) | (int_q_k[63:32] & ~wmask);
    end
end

// int_q_b[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_q_b[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_Q_B_DATA_0)
            int_q_b[31:0] <= (WDATA[31:0] & wmask) | (int_q_b[31:0] & ~wmask);
    end
end

// int_q_b[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_q_b[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_Q_B_DATA_1)
            int_q_b[63:32] <= (WDATA[31:0] & wmask) | (int_q_b[63:32] & ~wmask);
    end
end

//synthesis translate_off
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (int_gie & ~int_isr[0] & int_ier[0] & ap_done)
            $display ("// Interrupt Monitor : interrupt for ap_done detected @ \"%0t\"", $time);
        if (int_gie & ~int_isr[1] & int_ier[1] & ap_ready)
            $display ("// Interrupt Monitor : interrupt for ap_ready detected @ \"%0t\"", $time);
    end
end
//synthesis translate_on

//------------------------Memory logic-------------------

endmodule
