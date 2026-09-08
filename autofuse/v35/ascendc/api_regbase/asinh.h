/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_ASINH_H__
#define __ASCENDC_API_REGBASE_ASINH_H__

namespace AscendC {
namespace AsinhAPI {

constexpr float ONE = 1.0f;
constexpr float NEG_ONE = -1.0f;
constexpr float ZERO = 0.0f;
constexpr float S_MIN = 1.0e-45f;
constexpr float S_MAX = 3.40282'35e34f;
constexpr float LN2 = 0.69314'72f;
constexpr float SMALL_THRESHOLD = 0.00024'414063f;
constexpr float DIRECT_LOG_THRESHOLD = 10.0f;
constexpr float ASYMPTOTIC_THRESHOLD = 268'435'456.0f;

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

template <typename T>
__simd_vf__ inline void AsinhCompute(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount, uint16_t repeatTimes) {
  constexpr uint32_t oneRepSize = static_cast<uint32_t>(GetVecLen() / sizeof(float));

  Reg::RegTensor<T> srcReg, dstReg;
  Reg::RegTensor<float> xReg, absReg, bReg, rReg, sReg, tmpReg;
  Reg::MaskReg mask, selectMask, signMask;

  for (uint16_t i = 0; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<float>(calCount);
    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, src + i * oneRepSize);
      Reg::Cast<float, half, CAST_F16_TO_F32>(xReg, srcReg, mask);
    } else {
      Reg::LoadAlign(xReg, src + i * oneRepSize);
    }

    Reg::Abs(absReg, xReg, mask);

    // r = |x| + |x| / (sqrt(1 / |x|^2 + 1) + 1 / |x|).
    Reg::Duplicate(bReg, ONE);
    Reg::Div(bReg, bReg, absReg, mask);
    Reg::Mul(rReg, bReg, bReg, mask);
    Reg::Adds(rReg, rReg, ONE, mask);
    Reg::Sqrt(rReg, rReg, mask);
    Reg::Add(rReg, rReg, bReg, mask);
    Reg::Div(rReg, absReg, rReg, mask);
    Reg::Add(rReg, absReg, rReg, mask);

    // log1p-style compensation for |x| < 10.
    Reg::Adds(sReg, rReg, ONE, mask);
    Reg::Adds(bReg, sReg, NEG_ONE, mask);
    Reg::Maxs(bReg, bReg, S_MIN, mask);
    Reg::Mins(bReg, bReg, S_MAX, mask);
    Reg::Ln(sReg, sReg, mask);
    Reg::Duplicate(tmpReg, ONE);
    Reg::CompareScalar<float, CMPMODE::LT>(selectMask, absReg, DIRECT_LOG_THRESHOLD, mask);
    Reg::Select(rReg, rReg, tmpReg, selectMask);
    Reg::Select(bReg, bReg, tmpReg, selectMask);
    Reg::Mul(sReg, sReg, rReg, mask);
    Reg::Div(sReg, sReg, bReg, mask);

    // Use log(|x|) + ln(2) + 1 / |x|^2 when |x| is very large.
    Reg::Ln(rReg, absReg, mask);
    Reg::Adds(rReg, rReg, LN2, mask);
    Reg::Duplicate(bReg, ONE);
    Reg::Div(bReg, bReg, absReg, mask);
    Reg::Mul(bReg, bReg, bReg, mask);
    Reg::Add(rReg, rReg, bReg, mask);
    Reg::Adds(bReg, rReg, ZERO, mask);
    Reg::Duplicate(tmpReg, S_MAX);
    Reg::CompareScalar<float, CMPMODE::LT>(selectMask, absReg, DIRECT_LOG_THRESHOLD, mask);
    Reg::Select(rReg, rReg, tmpReg, selectMask);
    Reg::CompareScalar<float, CMPMODE::GE>(selectMask, absReg, ASYMPTOTIC_THRESHOLD, mask);
    Reg::Select(rReg, bReg, rReg, selectMask);

    // Comparisons with NaN are false; keep the correction path for NaN lanes.
    Reg::Compare<float, CMPMODE::EQ>(selectMask, absReg, absReg, mask);
    Reg::Select(rReg, rReg, bReg, selectMask);
    Reg::Min(sReg, sReg, rReg, mask);

    Reg::CompareScalar<float, CMPMODE::LT>(selectMask, absReg, SMALL_THRESHOLD, mask);
    Reg::Select(sReg, absReg, sReg, selectMask);
    Reg::Neg(tmpReg, sReg, mask);
    Reg::CompareScalar<float, CMPMODE::GE>(signMask, xReg, ZERO, mask);
    Reg::Select(rReg, sReg, tmpReg, signMask);

    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::Cast<half, float, CAST_F32_TO_F16>(dstReg, rReg, mask);
      Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(dst + i * oneRepSize, dstReg, mask);
    } else {
      Reg::StoreAlign(dst + i * oneRepSize, rReg, mask);
    }
  }
}

}  // namespace AsinhAPI
}  // namespace AscendC

template <typename T>
__aicore__ inline void AsinhExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                   AscendC::LocalTensor<uint8_t> &tmpBuf, const uint32_t calCount) {
  static_assert(AscendC::SupportType<T, half, float>(), "AsinhExtend only supports half and float on current device!");
  (void)tmpBuf;

  constexpr uint32_t oneRepSize = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
  const uint16_t repeatTimes = AscendC::CeilDivision(calCount, oneRepSize);
  AscendC::AsinhAPI::AsinhCompute((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), calCount,
                                  repeatTimes);
}

#endif  // __ASCENDC_API_REGBASE_ASINH_H__
