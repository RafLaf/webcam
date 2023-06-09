// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// control
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

#define XCOLOR_CONVERT_CONTROL_ADDR_C1_0_DATA   0x10
#define XCOLOR_CONVERT_CONTROL_BITS_C1_0_DATA   10
#define XCOLOR_CONVERT_CONTROL_ADDR_C1_1_DATA   0x18
#define XCOLOR_CONVERT_CONTROL_BITS_C1_1_DATA   10
#define XCOLOR_CONVERT_CONTROL_ADDR_C1_2_DATA   0x20
#define XCOLOR_CONVERT_CONTROL_BITS_C1_2_DATA   10
#define XCOLOR_CONVERT_CONTROL_ADDR_C2_0_DATA   0x28
#define XCOLOR_CONVERT_CONTROL_BITS_C2_0_DATA   10
#define XCOLOR_CONVERT_CONTROL_ADDR_C2_1_DATA   0x30
#define XCOLOR_CONVERT_CONTROL_BITS_C2_1_DATA   10
#define XCOLOR_CONVERT_CONTROL_ADDR_C2_2_DATA   0x38
#define XCOLOR_CONVERT_CONTROL_BITS_C2_2_DATA   10
#define XCOLOR_CONVERT_CONTROL_ADDR_C3_0_DATA   0x40
#define XCOLOR_CONVERT_CONTROL_BITS_C3_0_DATA   10
#define XCOLOR_CONVERT_CONTROL_ADDR_C3_1_DATA   0x48
#define XCOLOR_CONVERT_CONTROL_BITS_C3_1_DATA   10
#define XCOLOR_CONVERT_CONTROL_ADDR_C3_2_DATA   0x50
#define XCOLOR_CONVERT_CONTROL_BITS_C3_2_DATA   10
#define XCOLOR_CONVERT_CONTROL_ADDR_BIAS_0_DATA 0x58
#define XCOLOR_CONVERT_CONTROL_BITS_BIAS_0_DATA 10
#define XCOLOR_CONVERT_CONTROL_ADDR_BIAS_1_DATA 0x60
#define XCOLOR_CONVERT_CONTROL_BITS_BIAS_1_DATA 10
#define XCOLOR_CONVERT_CONTROL_ADDR_BIAS_2_DATA 0x68
#define XCOLOR_CONVERT_CONTROL_BITS_BIAS_2_DATA 10

