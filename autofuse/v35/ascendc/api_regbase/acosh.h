/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_ACOSH_H__
#define __ASCENDC_API_REGBASE_ACOSH_H__

namespace AscendC {
namespace AcoshAPI {

constexpr float ONE = 1.0f;
constexpr float NEG_ONE = -1.0f;
constexpr float S_MIN = 1.0e-45f;
constexpr float S_MAX = 3.4028235e34f;
constexpr float LN2 = 0.693147180559945286227f;
// 分段阈值用 acosh 值比较，避免 alias 路径额外保存原输入 x：
//   acosh(10) = 2.993222846...，direct < 阈值（即 x < 10）时保留 log1p 补偿；
//   acosh(sqrt(FLT_MAX)) = 45.054566706...，direct ≥ 阈值（x² 即将溢出）时切换渐近式。
constexpr float NEAR_ACOSH_THRESHOLD = 2.9932229518890380859375f;
constexpr float LARGE_ACOSH_THRESHOLD = 45.054'5654296875f;

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
__simd_vf__ inline void AcoshCompute(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount, uint16_t repeatTimes) {
  constexpr uint32_t oneRepSize = static_cast<uint32_t>(GetVecLen() / sizeof(float));

  Reg::RegTensor<T> srcReg, dstReg;
  Reg::RegTensor<float> xReg, tReg, sReg, uReg, asymReg;
  Reg::MaskReg mask, selectMask;

  for (uint16_t i = 0; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<float>(calCount);
    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, src + i * oneRepSize);
      Reg::Cast<float, half, CAST_F16_TO_F32>(xReg, srcReg, mask);
    } else {
      Reg::LoadAlign(xReg, src + i * oneRepSize);
    }

    // 渐近式分支：asym = ln(x) + ln(2)，超大 x（x² 溢出）时替代 direct。
    Reg::Ln(asymReg, xReg, mask);
    Reg::Adds(asymReg, asymReg, LN2, mask);

    // t = x - 1。
    Reg::Adds(tReg, xReg, NEG_ONE, mask);

    // u = 2 * (x - 1)。
    Reg::Add(uReg, tReg, tReg, mask);

    // s = (x-1)² + 2(x-1) = x² - 1。
    Reg::Mul(sReg, tReg, tReg, mask);
    Reg::Add(sReg, sReg, uReg, mask);

    // s = sqrt(x² - 1)。
    Reg::Sqrt(sReg, sReg, mask);

    // s = (x - 1) + sqrt(x² - 1)。
    Reg::Add(sReg, tReg, sReg, mask);

    // u = s + 1 = x + sqrt(x² - 1)。
    Reg::Adds(uReg, sReg, ONE, mask);

    // t = ln(u)，直接公式结果 direct（acosh 单调，可继续用于阈值比较）。
    Reg::Ln(tReg, uReg, mask);

    // s = ln(u) * s，补偿公式分子。
    Reg::Mul(sReg, tReg, sReg, mask);

    // u = clip(u - 1)：下界防 Div 除零，上界防溢出。
    Reg::Adds(uReg, uReg, NEG_ONE, mask);
    Reg::Maxs(uReg, uReg, S_MIN, mask);
    Reg::Mins(uReg, uReg, S_MAX, mask);

    // s = ln(u) * s / clip(u - 1)，近 1 区间的 log1p 补偿结果。
    Reg::Div(sReg, sReg, uReg, mask);

    // 分段 1：direct < acosh(10) 使用补偿公式，否则 direct。
    // NaN 与阈值比较为 false，仍选 direct = NaN。
    Reg::CompareScalar<float, CMPMODE::LT>(selectMask, tReg, NEAR_ACOSH_THRESHOLD, mask);
    Reg::Select(sReg, sReg, tReg, selectMask);

    // 分段 2：direct ≥ acosh(sqrt(FLT_MAX)) 时使用 ln(x) + ln(2) 渐近式，
    // 避免 x² 溢出后的中间 inf 参与最终结果。
    Reg::CompareScalar<float, CMPMODE::GE>(selectMask, tReg, LARGE_ACOSH_THRESHOLD, mask);
    Reg::Select(sReg, asymReg, sReg, selectMask);

    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::Cast<half, float, CAST_F32_TO_F16>(dstReg, sReg, mask);
      Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(dst + i * oneRepSize, dstReg, mask);
    } else {
      Reg::StoreAlign(dst + i * oneRepSize, sReg, mask);
    }
  }
}

}  // namespace AcoshAPI
}  // namespace AscendC

template <typename T>
__aicore__ inline void AcoshExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                   AscendC::LocalTensor<uint8_t> &tmpBuf, const uint32_t calCount) {
  static_assert(AscendC::SupportType<T, half, float>(), "AcoshExtend only supports half and float on current device!");
  (void)tmpBuf;

  constexpr uint32_t oneRepSize = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
  const uint16_t repeatTimes = AscendC::CeilDivision(calCount, oneRepSize);
  AscendC::AcoshAPI::AcoshCompute((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), calCount,
                                  repeatTimes);
}

#endif  // __ASCENDC_API_REGBASE_ACOSH_H__
