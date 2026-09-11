/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_I0_H__
#define __ASCENDC_API_REGBASE_I0_H__

namespace AscendC {

constexpr float I0_THRESHOLD = 9.0f;

// Small-branch coefficients: P(x^2) degree-10 polynomial for |x| < 9
// Array indexed by degree, stored low-to-high: c0..c10
// Horner evaluation order: c10..c0 (reverse index)
constexpr float I0_POLY_SMALL[11] = {
    1.0f,              // c0
    0.25f,             // c1
    0.015625f,         // c2
    0.00043402583f,    // c3
    0.0000067820783f,  // c4
    6.7778003e-8f,     // c5
    4.7306625e-10f,    // c6
    2.3349575e-12f,    // c7
    1.0687647e-14f,    // c8
    1.4492505e-17f,    // c9
    1.551427e-19f      // c10
};

// Large-branch coefficients: Q(1/x) degree-5 polynomial for |x| >= 9
// Array indexed by degree, stored low-to-high: d0..d5
// Horner evaluation order: d5..d0 (reverse index)
constexpr float I0_POLY_LARGE[6] = {
    0.39894226f,     // d0 = 1/sqrt(2*pi)
    0.04987063f,     // d1
    0.027889195f,    // d2
    0.033347155f,    // d3
    -0.0054563344f,  // d4
    0.34872168f      // d5
};

// Small branch: Horner evaluation of P(z) where z = x^2
// P(z) = c0 + c1*z + c2*z^2 + ... + c10*z^10
// Evaluated as: y = ((((c10*z + c9)*z + c8)*z + ...)*z + c0)
template <typename T>
__simd_callee__ inline void I0SmallCompute(Reg::RegTensor<T> &yReg, Reg::RegTensor<T> &xSqReg,
                                           Reg::MaskReg &branchMask) {
  Reg::Duplicate(yReg, (T)I0_POLY_SMALL[10], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I0_POLY_SMALL[9], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I0_POLY_SMALL[8], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I0_POLY_SMALL[7], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I0_POLY_SMALL[6], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I0_POLY_SMALL[5], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I0_POLY_SMALL[4], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I0_POLY_SMALL[3], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I0_POLY_SMALL[2], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I0_POLY_SMALL[1], branchMask);

  Reg::Mul(yReg, yReg, xSqReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I0_POLY_SMALL[0], branchMask);
}

// Large branch: asymptotic expansion
// y = Q(1/a) / sqrt(a) * exp(a)   where a = |x|
// Q(t) = d0 + d1*t + d2*t^2 + ... + d5*t^5
// Evaluated as: y = ((((d5*t + d4)*t + d3)*t + ...)*t + d0)
// Then: y = y / sqrt(a) * exp(a)
template <typename T>
__simd_callee__ inline void I0LargeCompute(Reg::RegTensor<T> &yReg, Reg::RegTensor<T> &clampedXReg,
                                           Reg::MaskReg &branchMask) {
  Reg::RegTensor<T> tReg, sqrtReg, expReg;

  Reg::Duplicate(tReg, (T)1.0f, branchMask);
  Reg::Div(tReg, tReg, clampedXReg, branchMask);

  Reg::Duplicate(yReg, (T)I0_POLY_LARGE[5], branchMask);

  Reg::Mul(yReg, yReg, tReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I0_POLY_LARGE[4], branchMask);

  Reg::Mul(yReg, yReg, tReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I0_POLY_LARGE[3], branchMask);

  Reg::Mul(yReg, yReg, tReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I0_POLY_LARGE[2], branchMask);

  Reg::Mul(yReg, yReg, tReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I0_POLY_LARGE[1], branchMask);

  Reg::Mul(yReg, yReg, tReg, branchMask);
  Reg::Adds(yReg, yReg, (T)I0_POLY_LARGE[0], branchMask);

  Reg::Sqrt(sqrtReg, clampedXReg, branchMask);
  Reg::Div(yReg, yReg, sqrtReg, branchMask);

  Reg::Exp(expReg, clampedXReg, branchMask);
  Reg::Mul(yReg, yReg, expReg, branchMask);
}

template <typename T>
__simd_vf__ inline void I0CoreImpl(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount) {
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

    I0SmallCompute<T>(smallYReg, xSqReg, mask);
    I0LargeCompute<T>(largeYReg, absXReg, mask);

    Reg::Compares<T, CMPMODE::LT>(cmpMask, absXReg, (T)I0_THRESHOLD, mask);
    Reg::Select(dstReg, smallYReg, largeYReg, cmpMask);

    Reg::RegTensor<T> nanReg;
    Reg::Compare<T, CMPMODE::NE>(cmpMask, srcReg, srcReg, mask);
    Reg::Duplicate(nanReg, (float &)F32_NAN, mask);
    Reg::Select(dstReg, nanReg, dstReg, cmpMask);

    Reg::RegTensor<T> infReg;
    Reg::Compares<T, CMPMODE::GT>(cmpMask, absXReg, NumericLimits<T>::Max(), mask);
    Reg::Duplicate(infReg, (float &)F32_INF, mask);
    Reg::Select(dstReg, infReg, dstReg, cmpMask);

    Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
  }
}

template <typename T>
__aicore__ inline void I0Extend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount) {
  static_assert((std::is_same_v<T, float>), "I0 only supports float on current device!");
  // Only for AI Vector Core.
  if ASCEND_IS_AIC {
    return;
  }
  __ubuf__ T *dstUb = (__ubuf__ T *)dst.GetPhyAddr();
  __ubuf__ T *srcUb = (__ubuf__ T *)src.GetPhyAddr();
  I0CoreImpl<T>(dstUb, srcUb, calCount);
}

}  // namespace AscendC

#endif  // __ASCENDC_API_REGBASE_I0_H__
