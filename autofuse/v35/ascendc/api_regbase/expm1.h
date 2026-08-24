/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_V35_ASCENDC_API_REGBASE_EXPM1_H_
#define AUTOFUSE_V35_ASCENDC_API_REGBASE_EXPM1_H_

using namespace AscendC;

constexpr uint32_t EXPM1_THREAD_NUM = 1024;
constexpr float EXPM1_INV_LN2_APPROX = 1.4426950216293335f;
constexpr float EXPM1_LN2_HALF_APPROX = 0.4099999964237213f;
constexpr float EXPM1_LN2_APPROX = 0.693145751953125f;
constexpr float EXPM1_ONE_MINUS_LN2_APPROX = 0.000001428606765330187f;
constexpr float EXPM1_FLOAT_128 = 128.000000f;
constexpr float EXPM1_FLOAT_NEG_ONE = -1.00000000f;
constexpr float EXPM1_C5 = 0.008382412604987621f;
constexpr float EXPM1_C4 = 0.0013879507314413786f;
constexpr float EXPM1_C3 = 0.04166783019900322f;
constexpr float EXPM1_C2 = 0.1666639745235443f;
constexpr float EXPM1_C1 = 0.4999999403953552f;
constexpr float EXPM1_FLOAT_INF = __builtin_inff();
constexpr float EXPM1_FLOAT_NEG_25 = -25.0000000f;
constexpr float EXPM1_FLOAT_2 = 2.0000000f;

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(EXPM1_THREAD_NUM) inline void Expm1SimtCompute(__ubuf__ T *x, __ubuf__ T *y,
                                                                                   const int64_t total_num) {
  for (int64_t i = threadIdx.x; i < total_num; i += blockDim.x) {
    float f1 = x[i];
    float f0 = expm1f(f1);
    float f2 = f1 * EXPM1_INV_LN2_APPROX;
    float f3 = roundf(f2);
    float f4 = fabsf(f1);
    bool p1 = f4 < EXPM1_LN2_HALF_APPROX;
    float f5 = p1 ? 0.0f : f3;
    float f6 = -f5;
    float f7 = EXPM1_LN2_APPROX;
    float f8 = fmaf(f6, f7, f1);
    float f9 = EXPM1_ONE_MINUS_LN2_APPROX;
    float f10 = fmaf(f6, f9, f8);
    bool p2 = f5 == EXPM1_FLOAT_128;
    float f11 = f5 + EXPM1_FLOAT_NEG_ONE;
    float f12 = p2 ? f11 : f5;
    float f13 = EXPM1_C5;
    float f14 = EXPM1_C4;
    float f15 = fmaf(f14, f10, f13);
    float f16 = EXPM1_C3;
    float f17 = fmaf(f15, f10, f16);
    float f18 = EXPM1_C2;
    float f19 = fmaf(f17, f10, f18);
    float f20 = EXPM1_C1;
    float f21 = fmaf(f19, f10, f20);
    float f22 = f10 * f21;
    float f23 = fmaf(f22, f10, f10);
    float f24 = exp2f(f12);
    float f25 = f24 + EXPM1_FLOAT_NEG_ONE;
    float f26 = fmaf(f23, f24, f25);
    float f27 = f26 + f26;
    float f28 = p2 ? f27 : f26;
    bool p3 = f12 > EXPM1_FLOAT_128;
    float f29 = p3 ? EXPM1_FLOAT_INF : f28;
    bool p4 = f12 < EXPM1_FLOAT_NEG_25;
    float f30 = p4 ? EXPM1_FLOAT_NEG_ONE : f29;
    bool p5 = f1 == 0.0f;
    float f31 = f1 + f1;
    float f32 = p5 ? f31 : f30;
    y[i] = fabsf(x[i]) > EXPM1_FLOAT_2 ? f0 : f32;
  }
}

template <typename T>
__aicore__ inline void Expm1Extend(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t calc_cnt) {
  AscendC::Simt::VF_CALL<Expm1SimtCompute<T>>(AscendC::Simt::Dim3(EXPM1_THREAD_NUM), (__ubuf__ T *)src.GetPhyAddr(),
                                              (__ubuf__ T *)dst.GetPhyAddr(), calc_cnt);
}
#endif  // AUTOFUSE_V35_ASCENDC_API_REGBASE_EXPM1_H_
