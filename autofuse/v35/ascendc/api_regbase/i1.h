/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_I1_H__
#define __ASCENDC_API_REGBASE_I1_H__

namespace AscendC {

constexpr float I1_THRESHOLD = 9.0f;

// Small-branch coefficients: P(x^2) degree-10 polynomial for |x| < 9
// I1(x) = (x/2) * P(x^2),  P(z) = q0 + q1*z + ... + q10*z^10
// Array indexed by degree, stored low-to-high: q0..q10
// Horner evaluation order: q10..q0 (reverse index)
constexpr float I1_POLY_SMALL[11] = {
    1.0000000084596425f,      // q0
    0.12499997371748357f,     // q1
    0.005208346765793173f,    // q2
    0.00010850427922121898f,  // q3
    1.3566067390896953e-06f,  // q4
    1.1286940958075224e-08f,  // q5
    6.785629676912552e-11f,   // q6
    2.8695877452994646e-13f,  // q7
    1.2396464592287248e-15f,  // q8
    1.1468478141470368e-18f,  // q9
    1.4840922967013815e-20f   // q10
};

// Large-branch coefficients: Q(1/x) degree-5 polynomial for |x| >= 9
// I1(x) = sign(x) * Q(1/|x|) / sqrt(|x|) * exp(|x|)
// Q(t) = d0 + d1*t + d2*t^2 + ... + d5*t^5
// Array indexed by degree, stored low-to-high: d0..d5
// Horner evaluation order: d5..d0 (reverse index)
constexpr float I1_POLY_LARGE[6] = {
    0.39894228274632065f,    // d0 = 1/sqrt(2*pi)
    -0.1496039980193778f,    // d1
    -0.04669567796467902f,   // d2
    -0.04287756634503034f,   // d3
    -0.026092109318235582f,  // d4
    -0.31734317225223657f    // d5
};

// Small branch: Horner evaluation of P(z) where z = x^2
// P(z) = q0 + q1*z + q2*z^2 + ... + q10*z^10
// Evaluated as: y = ((((q10*z + q9)*z + q8)*z + ...)*z + q0)
// Then: y = (x/2) * y  to give  I1(x) = (x/2) * P(x^2)
template <typename T>
__simd_callee__ inline void I1SmallCompute(Reg::RegTensor<T> &yReg, Reg::RegTensor<T> &srcReg,
                                           Reg::RegTensor<T> &xSqReg, Reg::MaskReg &branchMask) {
  Reg::Duplicate(yReg, (T)I1_POLY_SMALL[10], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I1_POLY_SMALL[9], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I1_POLY_SMALL[8], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I1_POLY_SMALL[7], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I1_POLY_SMALL[6], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I1_POLY_SMALL[5], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I1_POLY_SMALL[4], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I1_POLY_SMALL[3], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I1_POLY_SMALL[2], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I1_POLY_SMALL[1], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I1_POLY_SMALL[0], branchMask);

  // y = (x/2) * P(x^2)
  Reg::Mul(yReg, yReg, srcReg, branchMask);
  Reg::Muls(yReg, yReg, (T)0.5, branchMask);
}

// Large branch: asymptotic expansion
// y = sign(x) * Q(1/a) / sqrt(a) * exp(a)   where a = |x|
// Q(t) = d0 + d1*t + d2*t^2 + ... + d5*t^5
// Evaluated as: y = ((((d5*t + d4)*t + d3)*t + ...)*t + d0)
// Then: y = y / sqrt(a) * exp(a) * sign(x)
template <typename T>
__simd_callee__ inline void I1LargeCompute(Reg::RegTensor<T> &yReg, Reg::RegTensor<T> &srcReg,
                                           Reg::RegTensor<T> &clampedXReg, Reg::MaskReg &branchMask) {
  Reg::RegTensor<T> tReg, sqrtReg, expReg, signReg, negSignReg;
  Reg::MaskReg signMask;

  Reg::Duplicate(tReg, (T)1.0f, branchMask);
  Reg::Div(tReg, tReg, clampedXReg, branchMask);

  Reg::Duplicate(yReg, (T)I1_POLY_LARGE[5], branchMask);

  Reg::Mul(yReg, yReg, tReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I1_POLY_LARGE[4], branchMask);

  Reg::Mul(yReg, yReg, tReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I1_POLY_LARGE[3], branchMask);

  Reg::Mul(yReg, yReg, tReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I1_POLY_LARGE[2], branchMask);

  Reg::Mul(yReg, yReg, tReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I1_POLY_LARGE[1], branchMask);

  Reg::Mul(yReg, yReg, tReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I1_POLY_LARGE[0], branchMask);

  Reg::Sqrt(sqrtReg, clampedXReg, branchMask);
  Reg::Div(yReg, yReg, sqrtReg, branchMask);

  Reg::Exp(expReg, clampedXReg, branchMask);
  Reg::Mul(yReg, yReg, expReg, branchMask);

  // Apply sign(x): I1(-x) = -I1(x)
  Reg::Duplicate(signReg, (T)1.0f, branchMask);
  Reg::Duplicate(negSignReg, (T)(-1.0f), branchMask);
  Reg::Compares<T, CMPMODE::LT>(signMask, srcReg, (T)0.0f, branchMask);
  Reg::Select(signReg, negSignReg, signReg, signMask);
  Reg::Mul(yReg, yReg, signReg, branchMask);
}

template <typename T>
__simd_vf__ inline void I1CoreImpl(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount) {
  uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, vlSize));

  Reg::RegTensor<T> srcReg, absXReg, xSqReg;
  Reg::RegTensor<T> smallYReg, largeYReg, dstReg;
  Reg::MaskReg mask, cmpMask;

  for (uint16_t i = 0U; i < repeatTime; ++i) {
    mask = Reg::UpdateMask<T>(calCount);
    Reg::LoadAlign(srcReg, src + i * vlSize);

    Reg::Abs(absXReg, srcReg, mask);

    Reg::Mul(xSqReg, absXReg, absXReg, mask);

    I1SmallCompute<T>(smallYReg, srcReg, xSqReg, mask);
    I1LargeCompute<T>(largeYReg, srcReg, absXReg, mask);

    Reg::Compares<T, CMPMODE::LT>(cmpMask, absXReg, (T)I1_THRESHOLD, mask);
    Reg::Select(dstReg, smallYReg, largeYReg, cmpMask);

    // Handle NaN: I1(NaN) = NaN
    Reg::RegTensor<T> nanReg;
    Reg::Compare<T, CMPMODE::NE>(cmpMask, srcReg, srcReg, mask);
    Reg::Duplicate(nanReg, (float &)F32_NAN, mask);
    Reg::Select(dstReg, nanReg, dstReg, cmpMask);

    // Handle Inf: I1(+/-Inf) = +/-Inf
    Reg::RegTensor<T> signReg, negSignReg, infReg;
    Reg::MaskReg signMask;
    Reg::Duplicate(signReg, (T)1.0f, mask);
    Reg::Duplicate(negSignReg, (T)(-1.0f), mask);
    Reg::Compares<T, CMPMODE::LT>(signMask, srcReg, (T)0.0f, mask);
    Reg::Select(signReg, negSignReg, signReg, signMask);
    Reg::Duplicate(infReg, (float &)F32_INF, mask);
    Reg::Mul(infReg, infReg, signReg, mask);
    Reg::Compares<T, CMPMODE::GT>(cmpMask, absXReg, NumericLimits<T>::Max(), mask);
    Reg::Select(dstReg, infReg, dstReg, cmpMask);

    Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
  }
}

template <typename T>
__aicore__ inline void I1Extend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount) {
  static_assert((std::is_same_v<T, float>), "I1 only supports float on current device!");
  // Only for AI Vector Core.
  if ASCEND_IS_AIC {
    return;
  }
  __ubuf__ T *dstUb = (__ubuf__ T *)dst.GetPhyAddr();
  __ubuf__ T *srcUb = (__ubuf__ T *)src.GetPhyAddr();
  I1CoreImpl<T>(dstUb, srcUb, calCount);
}

}  // namespace AscendC

#endif  // __ASCENDC_API_REGBASE_I1_H__
