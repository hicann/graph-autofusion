/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_ATANH_H__
#define __ASCENDC_API_REGBASE_ATANH_H__

constexpr uint32_t ATANH_THREAD_NUM = 1024;
constexpr float ATANH_SMALL_INPUT_BOUND = 0.125f;

// 对齐 ops 仓 arch35 atanh 单算子（OpAtanhSimtKernel 的 AtanhFloat）：
// atanhf 的对数公式 ln((1+x)/(1-x)) 在零附近丢有效位，
// |x| < 0.125 时改用泰勒展开 x + x^3/3 + x^5/5 + x^7/7，省略项低于 FP32 精度。
__simt_callee__ __aicore__ inline float AtanhFloat(float value) {
  if (value > -ATANH_SMALL_INPUT_BOUND && value < ATANH_SMALL_INPUT_BOUND) {
    float square = value * value;
    return value + value * square * (1.0f / 3.0f + square * (1.0f / 5.0f + square * (1.0f / 7.0f)));
  }
  return atanhf(value);
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(ATANH_THREAD_NUM) inline void AtanhSimtCompute(__ubuf__ T *x, __ubuf__ T *y,
                                                                                   const int64_t total_num) {
  for (int64_t i = threadIdx.x; i < total_num; i += blockDim.x) {
    y[i] = AtanhFloat(x[i]);
  }
}

template <typename T>
__aicore__ inline void AtanhExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                   const LocalTensor<uint8_t> &tmp_buf, const uint32_t calc_cnt) {
  AscendC::Simt::VF_CALL<AtanhSimtCompute<T>>(AscendC::Simt::Dim3(ATANH_THREAD_NUM), (__ubuf__ T *)src.GetPhyAddr(),
                                              (__ubuf__ T *)dst.GetPhyAddr(), calc_cnt);
}
#endif  // __ASCENDC_API_REGBASE_ATANH_H__
