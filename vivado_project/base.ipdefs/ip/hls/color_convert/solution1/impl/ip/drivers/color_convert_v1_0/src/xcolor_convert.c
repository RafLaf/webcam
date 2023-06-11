// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xcolor_convert.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XColor_convert_CfgInitialize(XColor_convert *InstancePtr, XColor_convert_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XColor_convert_Set_c1_0(XColor_convert *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XColor_convert_WriteReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C1_0_DATA, Data);
}

u32 XColor_convert_Get_c1_0(XColor_convert *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XColor_convert_ReadReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C1_0_DATA);
    return Data;
}

void XColor_convert_Set_c1_1(XColor_convert *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XColor_convert_WriteReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C1_1_DATA, Data);
}

u32 XColor_convert_Get_c1_1(XColor_convert *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XColor_convert_ReadReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C1_1_DATA);
    return Data;
}

void XColor_convert_Set_c1_2(XColor_convert *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XColor_convert_WriteReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C1_2_DATA, Data);
}

u32 XColor_convert_Get_c1_2(XColor_convert *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XColor_convert_ReadReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C1_2_DATA);
    return Data;
}

void XColor_convert_Set_c2_0(XColor_convert *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XColor_convert_WriteReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C2_0_DATA, Data);
}

u32 XColor_convert_Get_c2_0(XColor_convert *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XColor_convert_ReadReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C2_0_DATA);
    return Data;
}

void XColor_convert_Set_c2_1(XColor_convert *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XColor_convert_WriteReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C2_1_DATA, Data);
}

u32 XColor_convert_Get_c2_1(XColor_convert *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XColor_convert_ReadReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C2_1_DATA);
    return Data;
}

void XColor_convert_Set_c2_2(XColor_convert *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XColor_convert_WriteReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C2_2_DATA, Data);
}

u32 XColor_convert_Get_c2_2(XColor_convert *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XColor_convert_ReadReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C2_2_DATA);
    return Data;
}

void XColor_convert_Set_c3_0(XColor_convert *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XColor_convert_WriteReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C3_0_DATA, Data);
}

u32 XColor_convert_Get_c3_0(XColor_convert *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XColor_convert_ReadReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C3_0_DATA);
    return Data;
}

void XColor_convert_Set_c3_1(XColor_convert *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XColor_convert_WriteReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C3_1_DATA, Data);
}

u32 XColor_convert_Get_c3_1(XColor_convert *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XColor_convert_ReadReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C3_1_DATA);
    return Data;
}

void XColor_convert_Set_c3_2(XColor_convert *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XColor_convert_WriteReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C3_2_DATA, Data);
}

u32 XColor_convert_Get_c3_2(XColor_convert *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XColor_convert_ReadReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_C3_2_DATA);
    return Data;
}

void XColor_convert_Set_bias_0(XColor_convert *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XColor_convert_WriteReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_BIAS_0_DATA, Data);
}

u32 XColor_convert_Get_bias_0(XColor_convert *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XColor_convert_ReadReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_BIAS_0_DATA);
    return Data;
}

void XColor_convert_Set_bias_1(XColor_convert *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XColor_convert_WriteReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_BIAS_1_DATA, Data);
}

u32 XColor_convert_Get_bias_1(XColor_convert *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XColor_convert_ReadReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_BIAS_1_DATA);
    return Data;
}

void XColor_convert_Set_bias_2(XColor_convert *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XColor_convert_WriteReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_BIAS_2_DATA, Data);
}

u32 XColor_convert_Get_bias_2(XColor_convert *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XColor_convert_ReadReg(InstancePtr->Control_BaseAddress, XCOLOR_CONVERT_CONTROL_ADDR_BIAS_2_DATA);
    return Data;
}

