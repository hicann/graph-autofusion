/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_REMAINDER_H
#define __ASCENDC_API_REGBASE_REMAINDER_H

// Remainder(x1, x2) = x1 - x2 * floor(x1/x2)

// int32 走 SIMT 实现（参考 ops-math FloorModInt_1）
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

template <typename T>
__aicore__ inline void RemainderImplVF(__ubuf__ T *dst, __ubuf__ T *src1, __ubuf__ T *src2, uint32_t count,
                                       uint16_t repeat_time) {
  constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  AscendC::MicroAPI::RegTensor<T> srcVreg1;
  AscendC::MicroAPI::RegTensor<T> srcVreg2;
  AscendC::MicroAPI::MaskReg mask;
  AscendC::MicroAPI::RegTensor<T> quotient;
  AscendC::MicroAPI::RegTensor<T> truncatedQuotient;

  for (uint16_t i = 0; i < repeat_time; i++) {
    mask = AscendC::MicroAPI::UpdateMask<T>(count);
    AscendC::MicroAPI::DataCopy(srcVreg1, src1 + i * oneRepElm);
    AscendC::MicroAPI::DataCopy(srcVreg2, src2 + i * oneRepElm);

    // Calculate x1 / x2
    AscendC::MicroAPI::Div(quotient, srcVreg1, srcVreg2, mask);

    // Truncate to floor
    AscendC::MicroAPI::Truncate<T, AscendC::RoundMode::CAST_FLOOR>(truncatedQuotient, quotient, mask);

    // Calculate remainder: x1 - x2 * floor(x1/x2)
    // srcVreg1 = x1 + x2 * (-floorQ)
    AscendC::MicroAPI::Muls(truncatedQuotient, truncatedQuotient, T(-1), mask);
    AscendC::MicroAPI::MulAddDst(srcVreg1, srcVreg2, truncatedQuotient, mask);

    AscendC::MicroAPI::DataCopy(dst + i * oneRepElm, srcVreg1, mask);
  }
}

template <typename T>
__aicore__ inline void RemainderExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src1,
                                       const AscendC::LocalTensor<T> &src2, const uint32_t size) {
  constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(size, oneRepElm));
  VF_CALL<RemainderImplVF<T>>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src1.GetPhyAddr(),
                              (__ubuf__ T *)src2.GetPhyAddr(), size, repeat_time);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void RemainderExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src1,
                                       const AscendC::LocalTensor<T> &src2, const LocalTensor<uint8_t> &sharedTmpBuffer,
                                       const uint32_t size) {
  static_assert(SupportType<T, int32_t, float>(),
                "RemainderExtend only support int32_t/float data type on current device!");

  if constexpr (IsSameType<T, int32_t>::value) {
    AscendC::Simt::VF_CALL<RemainderIntSimtCompute<T>>(AscendC::Simt::Dim3(REMAINDER_THREAD_NUM),
                                                       (__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src1.GetPhyAddr(),
                                                       (__ubuf__ T *)src2.GetPhyAddr(), size);
  } else {
    RemainderExtend(dst, src1, src2, size);
  }
}

#endif  // __ASCENDC_API_REGBASE_REMAINDER_H
