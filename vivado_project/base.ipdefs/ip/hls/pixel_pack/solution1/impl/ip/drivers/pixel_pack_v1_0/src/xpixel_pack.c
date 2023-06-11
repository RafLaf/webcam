// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xpixel_pack.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XPixel_pack_CfgInitialize(XPixel_pack *InstancePtr, XPixel_pack_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XPixel_pack_Set_mode(XPixel_pack *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPixel_pack_WriteReg(InstancePtr->Control_BaseAddress, XPIXEL_PACK_CONTROL_ADDR_MODE_DATA, Data);
}

u32 XPixel_pack_Get_mode(XPixel_pack *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPixel_pack_ReadReg(InstancePtr->Control_BaseAddress, XPIXEL_PACK_CONTROL_ADDR_MODE_DATA);
    return Data;
}

void XPixel_pack_Set_alpha(XPixel_pack *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPixel_pack_WriteReg(InstancePtr->Control_BaseAddress, XPIXEL_PACK_CONTROL_ADDR_ALPHA_DATA, Data);
}

u32 XPixel_pack_Get_alpha(XPixel_pack *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPixel_pack_ReadReg(InstancePtr->Control_BaseAddress, XPIXEL_PACK_CONTROL_ADDR_ALPHA_DATA);
    return Data;
}

