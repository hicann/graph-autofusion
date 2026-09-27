/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_REMAINDER_H__
#define __ASCENDC_API_REGBASE_REMAINDER_H__

// Remainder(x1, x2) = x1 - x2 * floor(x1/x2)

// int32 走 % 运算（参考 ops-math FloorModInt_1）
constexpr uint32_t REMAINDER_THREAD_NUM = 1024;

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(REMAINDER_THREAD_NUM) inline void RemainderIntSimtCompute(__ubuf__ T *dst,
                                                                                              __ubuf__ T *src1,
                                                                                              __ubuf__ T *src2,
                                                                                              const int64_t total_num) {
  for (int64_t i = threadIdx.x; i < total_num; i += blockDim.x) {
    const auto rem = src1[i] % src2[i];
    const bool signs_differ = (rem < 0) != (src2[i] < 0);
    dst[i] = (signs_differ && rem != 0) ? rem + src2[i] : rem;
  }
}

// 浮点走 fmodf 精确算法 + floor 语义修正：
// - fmodf 为位分解精确算法（大商场景如 x1/x2 溢出 float 时仍精确，对齐单算子参考实现），
//   且 |x2|=Inf 时结果为 x1、|x1|=Inf 时结果为 NaN 天然正确
// - fmodf 为截断语义，与 Remainder 的 floor 语义不一致：截断余数与 x2 异号时加 x2 修正
// - 修正条件需排除 |x2|=Inf（此时结果应为 x1，若修正会错误地加上 Inf）；
//   有限值判断 (x2f - x2f) == 0 仅在 x2 为有限值时成立（Inf-Inf 与 NaN-NaN 均为 NaN），
//   NaN 输入时 fmodf 结果为 NaN，无需修正
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(REMAINDER_THREAD_NUM) inline void RemainderSimtCompute(__ubuf__ T *dst,
                                                                                           __ubuf__ T *src1,
                                                                                           __ubuf__ T *src2,
                                                                                           const int64_t total_num) {
  for (int64_t i = threadIdx.x; i < total_num; i += blockDim.x) {
    const float x1f = static_cast<float>(src1[i]);
    const float x2f = static_cast<float>(src2[i]);
    const float truncRem = fmodf(x1f, x2f);
    const bool x2Finite = (x2f - x2f) == 0.0f;
    const float rem = (x2Finite && truncRem * x2f < 0.0f) ? truncRem + x2f : truncRem;
    dst[i] = static_cast<T>(rem);
  }
}

template <typename T>
__aicore__ inline void RemainderExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src1,
                                       const AscendC::LocalTensor<T> &src2, const uint32_t size) {
  AscendC::Simt::VF_CALL<RemainderSimtCompute<T>>(AscendC::Simt::Dim3(REMAINDER_THREAD_NUM),
                                                  (__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src1.GetPhyAddr(),
                                                  (__ubuf__ T *)src2.GetPhyAddr(), size);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void RemainderExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src1,
                                       const AscendC::LocalTensor<T> &src2, const LocalTensor<uint8_t> &sharedTmpBuffer,
                                       const uint32_t size) {
  static_assert(SupportType<T, int32_t, half, float>(),
                "RemainderExtend only support int32_t/half/float data type on current device!");

  if constexpr (IsSameType<T, int32_t>::value) {
    AscendC::Simt::VF_CALL<RemainderIntSimtCompute<T>>(AscendC::Simt::Dim3(REMAINDER_THREAD_NUM),
                                                       (__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src1.GetPhyAddr(),
                                                       (__ubuf__ T *)src2.GetPhyAddr(), size);
  } else {
    RemainderExtend(dst, src1, src2, size);
  }
}

#endif  // __ASCENDC_API_REGBASE_REMAINDER_H__
