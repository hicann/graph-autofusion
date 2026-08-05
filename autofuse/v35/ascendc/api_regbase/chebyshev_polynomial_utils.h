/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_CHEBYSHEV_POLYNOMIAL_UTILS_H__
#define __ASCENDC_API_REGBASE_CHEBYSHEV_POLYNOMIAL_UTILS_H__

#include "kernel_operator.h"

namespace AscendC {

template <typename T, int64_t N, int64_t curN = 2>
__simd_callee__ inline void ChebyshevPolynomialCalByTemplateExpansion(Reg::RegTensor<T> &res, Reg::RegTensor<T> &coef,
                                                                      Reg::RegTensor<T> &temp1,
                                                                      Reg::RegTensor<T> &temp2, Reg::MaskReg &mask) {
  Reg::Move(temp2, res);
  Reg::Mul(res, coef, res, mask);
  Reg::Sub(res, res, temp1, mask);
  Reg::Move(temp1, temp2);

  if constexpr (curN < N) {
    ChebyshevPolynomialCalByTemplateExpansion<T, N, curN + 1>(res, coef, temp1, temp2, mask);
  }
}

template <typename T, int64_t N>
__simd_callee__ inline void ChebyshevPolynomialCalByLoop(Reg::RegTensor<T> &res, Reg::RegTensor<T> &coef,
                                                         Reg::RegTensor<T> &temp1, Reg::RegTensor<T> &temp2,
                                                         Reg::MaskReg &mask) {
  for (int k = 2; k <= N; k++) {
    Reg::Move(temp2, res);
    Reg::Mul(res, coef, res, mask);
    Reg::Sub(res, res, temp1, mask);
    Reg::Move(temp1, temp2);
  }
}

template <typename T, int64_t N>
__simd_callee__ inline void ChebyshevPolynomialCal(Reg::RegTensor<T> &res, Reg::RegTensor<T> &coef,
                                                   Reg::RegTensor<T> &temp1, Reg::RegTensor<T> &temp2,
                                                   Reg::MaskReg &mask) {
  if constexpr (N <= 1000) {
    ChebyshevPolynomialCalByTemplateExpansion<T, N>(res, coef, temp1, temp2, mask);
  } else {
    ChebyshevPolynomialCalByLoop<T, N>(res, coef, temp1, temp2, mask);
  }
}

}  // namespace AscendC

#endif
