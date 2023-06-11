// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
`timescale 1ns/1ps
module color_convert_control_s_axi
#(parameter
    C_S_AXI_ADDR_WIDTH = 7,
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
    output wire [9:0]                    c1_0,
    output wire [9:0]                    c1_1,
    output wire [9:0]                    c1_2,
    output wire [9:0]                    c2_0,
    output wire [9:0]                    c2_1,
    output wire [9:0]                    c2_2,
    output wire [9:0]                    c3_0,
    output wire [9:0]                    c3_1,
    output wire [9:0]                    c3_2,
    output wire [9:0]                    bias_0,
    output wire [9:0]                    bias_1,
    output wire [9:0]                    bias_2
);
//------------------------Address Info-------------------
// 0x00 : reserved
// 0x04 : reserved
// 0x08 : reserved
// 0x0c : reserved
// 0x10 : Data signal of c1_0
//        bit 9~0 - c1_0[9:0] (Read/Write)
//        others  - reserved
// 0x14 : reserved
// 0x18 : Data signal of c1_1
//        bit 9~0 - c1_1[9:0] (Read/Write)
//        others  - reserved
// 0x1c : reserved
// 0x20 : Data signal of c1_2
//        bit 9~0 - c1_2[9:0] (Read/Write)
//        others  - reserved
// 0x24 : reserved
// 0x28 : Data signal of c2_0
//        bit 9~0 - c2_0[9:0] (Read/Write)
//        others  - reserved
// 0x2c : reserved
// 0x30 : Data signal of c2_1
//        bit 9~0 - c2_1[9:0] (Read/Write)
//        others  - reserved
// 0x34 : reserved
// 0x38 : Data signal of c2_2
//        bit 9~0 - c2_2[9:0] (Read/Write)
//        others  - reserved
// 0x3c : reserved
// 0x40 : Data signal of c3_0
//        bit 9~0 - c3_0[9:0] (Read/Write)
//        others  - reserved
// 0x44 : reserved
// 0x48 : Data signal of c3_1
//        bit 9~0 - c3_1[9:0] (Read/Write)
//        others  - reserved
// 0x4c : reserved
// 0x50 : Data signal of c3_2
//        bit 9~0 - c3_2[9:0] (Read/Write)
//        others  - reserved
// 0x54 : reserved
// 0x58 : Data signal of bias_0
//        bit 9~0 - bias_0[9:0] (Read/Write)
//        others  - reserved
// 0x5c : reserved
// 0x60 : Data signal of bias_1
//        bit 9~0 - bias_1[9:0] (Read/Write)
//        others  - reserved
// 0x64 : reserved
// 0x68 : Data signal of bias_2
//        bit 9~0 - bias_2[9:0] (Read/Write)
//        others  - reserved
// 0x6c : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

//------------------------Parameter----------------------
localparam
    ADDR_C1_0_DATA_0   = 7'h10,
    ADDR_C1_0_CTRL     = 7'h14,
    ADDR_C1_1_DATA_0   = 7'h18,
    ADDR_C1_1_CTRL     = 7'h1c,
    ADDR_C1_2_DATA_0   = 7'h20,
    ADDR_C1_2_CTRL     = 7'h24,
    ADDR_C2_0_DATA_0   = 7'h28,
    ADDR_C2_0_CTRL     = 7'h2c,
    ADDR_C2_1_DATA_0   = 7'h30,
    ADDR_C2_1_CTRL     = 7'h34,
    ADDR_C2_2_DATA_0   = 7'h38,
    ADDR_C2_2_CTRL     = 7'h3c,
    ADDR_C3_0_DATA_0   = 7'h40,
    ADDR_C3_0_CTRL     = 7'h44,
    ADDR_C3_1_DATA_0   = 7'h48,
    ADDR_C3_1_CTRL     = 7'h4c,
    ADDR_C3_2_DATA_0   = 7'h50,
    ADDR_C3_2_CTRL     = 7'h54,
    ADDR_BIAS_0_DATA_0 = 7'h58,
    ADDR_BIAS_0_CTRL   = 7'h5c,
    ADDR_BIAS_1_DATA_0 = 7'h60,
    ADDR_BIAS_1_CTRL   = 7'h64,
    ADDR_BIAS_2_DATA_0 = 7'h68,
    ADDR_BIAS_2_CTRL   = 7'h6c,
    WRIDLE             = 2'd0,
    WRDATA             = 2'd1,
    WRRESP             = 2'd2,
    WRRESET            = 2'd3,
    RDIDLE             = 2'd0,
    RDDATA             = 2'd1,
    RDRESET            = 2'd2,
    ADDR_BITS                = 7;

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
    reg  [9:0]                    int_c1_0 = 'b0;
    reg  [9:0]                    int_c1_1 = 'b0;
    reg  [9:0]                    int_c1_2 = 'b0;
    reg  [9:0]                    int_c2_0 = 'b0;
    reg  [9:0]                    int_c2_1 = 'b0;
    reg  [9:0]                    int_c2_2 = 'b0;
    reg  [9:0]                    int_c3_0 = 'b0;
    reg  [9:0]                    int_c3_1 = 'b0;
    reg  [9:0]                    int_c3_2 = 'b0;
    reg  [9:0]                    int_bias_0 = 'b0;
    reg  [9:0]                    int_bias_1 = 'b0;
    reg  [9:0]                    int_bias_2 = 'b0;

//------------------------Instantiation------------------


//------------------------AXI write fsm------------------
assign AWREADY = (wstate == WRIDLE);
assign WREADY  = (wstate == WRDATA);
assign BRESP   = 2'b00;  // OKAY
assign BVALID  = (wstate == WRRESP);
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
            if (BREADY)
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
            waddr <= AWADDR[ADDR_BITS-1:0];
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
                ADDR_C1_0_DATA_0: begin
                    rdata <= int_c1_0[9:0];
                end
                ADDR_C1_1_DATA_0: begin
                    rdata <= int_c1_1[9:0];
                end
                ADDR_C1_2_DATA_0: begin
                    rdata <= int_c1_2[9:0];
                end
                ADDR_C2_0_DATA_0: begin
                    rdata <= int_c2_0[9:0];
                end
                ADDR_C2_1_DATA_0: begin
                    rdata <= int_c2_1[9:0];
                end
                ADDR_C2_2_DATA_0: begin
                    rdata <= int_c2_2[9:0];
                end
                ADDR_C3_0_DATA_0: begin
                    rdata <= int_c3_0[9:0];
                end
                ADDR_C3_1_DATA_0: begin
                    rdata <= int_c3_1[9:0];
                end
                ADDR_C3_2_DATA_0: begin
                    rdata <= int_c3_2[9:0];
                end
                ADDR_BIAS_0_DATA_0: begin
                    rdata <= int_bias_0[9:0];
                end
                ADDR_BIAS_1_DATA_0: begin
                    rdata <= int_bias_1[9:0];
                end
                ADDR_BIAS_2_DATA_0: begin
                    rdata <= int_bias_2[9:0];
                end
            endcase
        end
    end
end


//------------------------Register logic-----------------
assign c1_0   = int_c1_0;
assign c1_1   = int_c1_1;
assign c1_2   = int_c1_2;
assign c2_0   = int_c2_0;
assign c2_1   = int_c2_1;
assign c2_2   = int_c2_2;
assign c3_0   = int_c3_0;
assign c3_1   = int_c3_1;
assign c3_2   = int_c3_2;
assign bias_0 = int_bias_0;
assign bias_1 = int_bias_1;
assign bias_2 = int_bias_2;
// int_c1_0[9:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_c1_0[9:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_C1_0_DATA_0)
            int_c1_0[9:0] <= (WDATA[31:0] & wmask) | (int_c1_0[9:0] & ~wmask);
    end
end

// int_c1_1[9:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_c1_1[9:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_C1_1_DATA_0)
            int_c1_1[9:0] <= (WDATA[31:0] & wmask) | (int_c1_1[9:0] & ~wmask);
    end
end

// int_c1_2[9:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_c1_2[9:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_C1_2_DATA_0)
            int_c1_2[9:0] <= (WDATA[31:0] & wmask) | (int_c1_2[9:0] & ~wmask);
    end
end

// int_c2_0[9:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_c2_0[9:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_C2_0_DATA_0)
            int_c2_0[9:0] <= (WDATA[31:0] & wmask) | (int_c2_0[9:0] & ~wmask);
    end
end

// int_c2_1[9:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_c2_1[9:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_C2_1_DATA_0)
            int_c2_1[9:0] <= (WDATA[31:0] & wmask) | (int_c2_1[9:0] & ~wmask);
    end
end

// int_c2_2[9:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_c2_2[9:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_C2_2_DATA_0)
            int_c2_2[9:0] <= (WDATA[31:0] & wmask) | (int_c2_2[9:0] & ~wmask);
    end
end

// int_c3_0[9:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_c3_0[9:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_C3_0_DATA_0)
            int_c3_0[9:0] <= (WDATA[31:0] & wmask) | (int_c3_0[9:0] & ~wmask);
    end
end

// int_c3_1[9:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_c3_1[9:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_C3_1_DATA_0)
            int_c3_1[9:0] <= (WDATA[31:0] & wmask) | (int_c3_1[9:0] & ~wmask);
    end
end

// int_c3_2[9:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_c3_2[9:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_C3_2_DATA_0)
            int_c3_2[9:0] <= (WDATA[31:0] & wmask) | (int_c3_2[9:0] & ~wmask);
    end
end

// int_bias_0[9:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_bias_0[9:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_BIAS_0_DATA_0)
            int_bias_0[9:0] <= (WDATA[31:0] & wmask) | (int_bias_0[9:0] & ~wmask);
    end
end

// int_bias_1[9:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_bias_1[9:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_BIAS_1_DATA_0)
            int_bias_1[9:0] <= (WDATA[31:0] & wmask) | (int_bias_1[9:0] & ~wmask);
    end
end

// int_bias_2[9:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_bias_2[9:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_BIAS_2_DATA_0)
            int_bias_2[9:0] <= (WDATA[31:0] & wmask) | (int_bias_2[9:0] & ~wmask);
    end
end


//------------------------Memory logic-------------------

endmodule
