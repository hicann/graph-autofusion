/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_SINH_H__
#define __ASCENDC_API_REGBASE_SINH_H__

namespace AscendC {
namespace SinhAPI {

constexpr float TAYLOR_C2 = 0.16666'667f;
constexpr float TAYLOR_C3 = 0.00833'3347f;
constexpr float TAYLOR_C4 = 0.00019'841270f;
constexpr float TAYLOR_C5 = 0.00000'27557319f;
constexpr float ONE = 1.0f;
constexpr float ZERO = 0.0f;
constexpr float QUARTER = 0.25f;
constexpr float OVERFLOW_THRESHOLD = 89.41599'27368164f;
constexpr float NEG_LN2_HI = -0.69314'718246'45996f;
constexpr float LN2_LO = 1.90465'42e-9f;

constexpr Reg::CastTrait CAST_F16_TO_F32 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};
constexpr Reg::CastTrait CAST_F32_TO_F16 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};
constexpr Reg::DivSpecificMode HIGH_PRECISION_DIV = {Reg::MaskMergeMode::ZEROING, true};

template <typename T>
__simd_vf__ inline void SinhCompute(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount, uint16_t repeatTimes) {
  constexpr uint32_t oneRepSize = static_cast<uint32_t>(GetVecLen() / sizeof(float));
  const float inf = __builtin_huge_valf();

  Reg::RegTensor<T> srcReg, dstReg;
  Reg::RegTensor<float> xReg, absReg, x2Reg, polyReg, tmpReg, expReg, resultReg;
  Reg::MaskReg mask, negativeMask, selectMask;

  for (uint16_t i = 0; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<float>(calCount);
    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, src + i * oneRepSize);
      Reg::Cast<float, half, CAST_F16_TO_F32>(xReg, srcReg, mask);
    } else {
      Reg::LoadAlign(xReg, src + i * oneRepSize);
    }

    Reg::Abs(absReg, xReg, mask);

    // |x| < 1: x + x^3 * P(x^2).
    Reg::Mul(x2Reg, xReg, xReg, mask);
    Reg::Duplicate(polyReg, TAYLOR_C5);
    Reg::Mul(polyReg, polyReg, x2Reg, mask);
    Reg::Adds(polyReg, polyReg, TAYLOR_C4, mask);
    Reg::Mul(polyReg, polyReg, x2Reg, mask);
    Reg::Adds(polyReg, polyReg, TAYLOR_C3, mask);
    Reg::Mul(polyReg, polyReg, x2Reg, mask);
    Reg::Adds(polyReg, polyReg, TAYLOR_C2, mask);
    Reg::Mul(tmpReg, x2Reg, xReg, mask);
    Reg::Mul(polyReg, polyReg, tmpReg, mask);
    Reg::Add(polyReg, xReg, polyReg, mask);

    // |x| >= 1: exp(|x| - ln(2)) - 0.25 / exp(|x| - ln(2)).
    Reg::Adds(tmpReg, absReg, NEG_LN2_HI, mask);
    Reg::Sub(expReg, tmpReg, absReg, mask);
    Reg::Sub(resultReg, tmpReg, expReg, mask);
    Reg::Sub(resultReg, absReg, resultReg, mask);
    Reg::Neg(x2Reg, expReg, mask);
    Reg::Adds(x2Reg, x2Reg, NEG_LN2_HI, mask);
    Reg::Add(resultReg, resultReg, x2Reg, mask);
    Reg::Adds(resultReg, resultReg, LN2_LO, mask);

    Reg::Exp(expReg, tmpReg, mask);
    Reg::CompareScalar<float, CMPMODE::LT>(selectMask, absReg, OVERFLOW_THRESHOLD, mask);
    Reg::Mul(x2Reg, expReg, resultReg, selectMask);
    Reg::Add(x2Reg, expReg, x2Reg, selectMask);
    Reg::Select(expReg, x2Reg, expReg, selectMask);
    Reg::Duplicate(tmpReg, QUARTER);
    Reg::Div<float, &HIGH_PRECISION_DIV>(tmpReg, tmpReg, expReg, mask);
    Reg::Sub(expReg, expReg, tmpReg, mask);

    Reg::CompareScalar<float, CMPMODE::LT>(negativeMask, xReg, ZERO, mask);
    Reg::Neg(tmpReg, expReg, mask);
    Reg::Select(expReg, tmpReg, expReg, negativeMask);
    Reg::CompareScalar<float, CMPMODE::LT>(selectMask, absReg, ONE, mask);
    Reg::Select(resultReg, polyReg, expReg, selectMask);

    Reg::CompareScalar<float, CMPMODE::GE>(selectMask, absReg, OVERFLOW_THRESHOLD, mask);
    Reg::Duplicate(tmpReg, inf);
    Reg::Neg(expReg, tmpReg, mask);
    Reg::Select(tmpReg, expReg, tmpReg, negativeMask);
    Reg::Select(resultReg, tmpReg, resultReg, selectMask);

    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::Cast<half, float, CAST_F32_TO_F16>(dstReg, resultReg, mask);
      Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(dst + i * oneRepSize, dstReg, mask);
    } else {
      Reg::StoreAlign(dst + i * oneRepSize, resultReg, mask);
    }
  }
}

}  // namespace SinhAPI
}  // namespace AscendC

template <typename T>
__aicore__ inline void SinhExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                  AscendC::LocalTensor<uint8_t> &tmpBuf, const uint32_t calCount) {
  static_assert(AscendC::SupportType<T, half, float>(), "SinhExtend only supports half and float on current device!");
  (void)tmpBuf;

  constexpr uint32_t oneRepSize = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
  const uint16_t repeatTimes = AscendC::CeilDivision(calCount, oneRepSize);
  AscendC::SinhAPI::SinhCompute((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), calCount, repeatTimes);
}

#endif  // __ASCENDC_API_REGBASE_SINH_H__
