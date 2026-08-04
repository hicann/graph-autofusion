/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_CHEBYSHEV_POLYNOMIAL_W_H__
#define __ASCENDC_API_REGBASE_CHEBYSHEV_POLYNOMIAL_W_H__

#include "chebyshev_polynomial_utils.h"

namespace AscendC {

template <typename T, int64_t N>
__aicore__ inline void ChebyshevPolynomialWCal(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount) {
  uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, vlSize));

  Reg::RegTensor<T> srcReg, coefReg, dstReg, temp1Reg, temp2Reg;
  Reg::MaskReg mask;
  mask = Reg::UpdateMask<T>(calCount);

  for (uint16_t i = 0U; i < repeatTime; ++i) {
    Reg::LoadAlign(srcReg, src + i * vlSize);

    Reg::Muls(dstReg, srcReg, (T)2.0, mask);
    Reg::Adds(dstReg, dstReg, (T)1.0, mask);

    Reg::Muls(coefReg, srcReg, (T)2.0, mask);

    Reg::Duplicate(temp1Reg, (T)1.0, mask);

    ChebyshevPolynomialCal<T, N>(dstReg, coefReg, temp1Reg, temp2Reg, mask);

    Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
  }
}

template <typename T, int64_t N>
__aicore__ inline void ChebyshevPolynomialWExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                                  const LocalTensor<uint8_t> &sharedTmpBuffer,
                                                  const uint32_t calCount) {
  static_assert(SupportType<T, float>(), "Current data type is not supported on current device!");

  if constexpr (N < 0) {
    Duplicate(dst, (T)0.0, calCount);
  } else if constexpr (N == 0) {
    Duplicate(dst, (T)1.0, calCount);
  } else if constexpr (N == 1) {
    Muls(dst, src, (T)2.0, calCount);
    Adds(dst, dst, (T)1.0, calCount);
  } else {
    VF_CALL<ChebyshevPolynomialWCal<T, N>>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), calCount);
  }
}

}  // namespace AscendC

#endif
