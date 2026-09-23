/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_RSQRT_H__
#define __ASCENDC_API_REGBASE_RSQRT_H__

namespace AscendC {
namespace RsqrtAPI {

constexpr Reg::DivSpecificMode HIGH_PRECISION_DIV = {Reg::MaskMergeMode::ZEROING, true};
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

/*
 * rsqrt(x) = 1 / sqrt(x)
 *
 * Implemented at register level with the high-precision Div mode
 * ({MaskMergeMode::ZEROING, true}) instead of the built-in Rsqrt/Reciprocal
 * wrapper, so the final 1/sqrt(x) division benefits from the enhanced
 * Newton-Raphson refinement used by the high-precision Div path.
 *
 * Subnormal handling (float only):
 *   The dav-3510 FTZ mode flushes subnormal Sqrt inputs/outputs to 0, causing
 *   Div(1, 0) = +Inf where the correct result is a large finite number.
 *   Fix: detect subnormals via CompareScalar(x < tiny), scale x by 2^126 to
 *   normal range before Sqrt, then scale the rsqrt result by 2^63 back:
 *     rsqrt(x) = rsqrt(x * 2^126) * 2^63
 *   For non-subnormal elements the scaled values are computed but discarded
 *   by Select, so correctness is unaffected.
 *   half: Cast(half→float) converts half subnormals to float normals, so no
 *         subnormal handling is needed for the half path.
 */
template <typename T>
__simd_vf__ inline void RsqrtCompute(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount, uint16_t repeatTimes) {
  constexpr uint32_t oneRepSize = static_cast<uint32_t>(GetVecLen() / sizeof(float));
  constexpr float F32_TINY = 1.1754943508222875e-38f;  // min normal float32
  constexpr float SCALE_UP = 8.507059173023674e37f;    // 2^126
  constexpr float SCALE_DOWN = 9.223372036854776e18f;  // 2^63

  Reg::RegTensor<T> srcReg, dstReg;
  Reg::RegTensor<float> xReg, sqrtReg, onesReg, scaledReg;
  Reg::MaskReg mask, subMask;

  for (uint16_t i = 0; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<float>(calCount);
    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, src + i * oneRepSize);
      Reg::Cast<float, half, CAST_F16_TO_F32>(xReg, srcReg, mask);
    } else {
      Reg::LoadAlign(xReg, src + i * oneRepSize);
      // Subnormal detection: x < tiny (positive subnormals, also catches
      // 0, -0.0 and negatives where the scaled path gives correct Inf/NaN)
      Reg::CompareScalar<float, CMPMODE::LT>(subMask, xReg, F32_TINY, mask);
      Reg::Muls(scaledReg, xReg, SCALE_UP, mask);
      Reg::Select(xReg, scaledReg, xReg, subMask);
    }

    // sqrt(x_eff)
    Reg::Sqrt(sqrtReg, xReg, mask);

    // 1 / sqrt(x_eff) via high-precision Div
    Reg::Duplicate(onesReg, 1.0f, mask);
    Reg::Div<float, &HIGH_PRECISION_DIV>(onesReg, onesReg, sqrtReg, mask);

    if constexpr (sizeof(T) == sizeof(float)) {
      // Scale back: rsqrt(x) = rsqrt(x*2^126) * 2^63
      Reg::Muls(scaledReg, onesReg, SCALE_DOWN, mask);
      Reg::Select(onesReg, scaledReg, onesReg, subMask);
    }

    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::Cast<half, float, CAST_F32_TO_F16>(dstReg, onesReg, mask);
      Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(dst + i * oneRepSize, dstReg, mask);
    } else {
      Reg::StoreAlign(dst + i * oneRepSize, onesReg, mask);
    }
  }
}

}  // namespace RsqrtAPI
}  // namespace AscendC

/**
 * @brief RsqrtExtend - compute the reciprocal of the square root: y = 1 / sqrt(x)
 *
 * Uses register-level Sqrt followed by a high-precision Div (ones / sqrt(x))
 * instead of the built-in Rsqrt/Reciprocal wrapper.
 * - float: Sqrt + high-precision Div, all in float
 * - half:  cast to float, Sqrt + high-precision Div in float, cast back to half
 *
 * @tparam T data type, supports float, half
 * @param dst output tensor
 * @param src input tensor
 * @param tmpBuf temporary buffer (unused; kept for API compatibility)
 * @param calCount number of elements to compute
 */
template <typename T>
__aicore__ inline void RsqrtExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                   AscendC::LocalTensor<uint8_t> &tmpBuf, const uint32_t calCount) {
  static_assert(AscendC::SupportType<T, half, float>(), "RsqrtExtend only supports half and float on current device!");
  (void)tmpBuf;

  constexpr uint32_t oneRepSize = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
  const uint16_t repeatTimes = AscendC::CeilDivision(calCount, oneRepSize);
  AscendC::RsqrtAPI::RsqrtCompute((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), calCount,
                                  repeatTimes);
}

#endif  // __ASCENDC_API_REGBASE_RSQRT_H__
