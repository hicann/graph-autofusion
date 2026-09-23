/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_SILU_H__
#define __ASCENDC_API_REGBASE_SILU_H__

namespace AscendC {
namespace SiluAPI {

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

/*
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Register-level SIMD with two pipeline-fill optimisations:
 *
 * 1. Instruction scheduling: Duplicate(1.0) is hoisted before Exp so it
 *    issues in parallel with Neg — both are independent of each other and
 *    of the Exp latency bubble that follows.
 *
 * 2. HIGH_PRECISION_DIV: the reciprocal 1/(1+exp(-x)) uses Newton-Raphson
 *    refinement ({ZEROING, true}).  The NR steps (Mul, Sub, Mul) are
 *    independent of the Exp→Adds chain and fill the Exp latency bubble,
 *    raising IPC — same technique rsqrt.h uses to reach IPC 1.44.
 *
 *    FTZ safety: the initial VDIV estimate 1/large_denom is subnormal and
 *    FTZ-flushed to 0; NR preserves 0 (0 * anything = 0).  For denom = Inf
 *    (x = -Inf), 1/Inf = 0, NR gives NaN (Inf*0), and Mul(-Inf, NaN) = NaN,
 *    matching the golden reference.
 *
 * half:  cast to float, compute in float, cast back (matches golden
 *        reference which is computed in float32 then rounded to half).
 */
template <typename T>
__simd_vf__ inline void SiluCompute(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount, uint16_t repeatTimes) {
  constexpr uint32_t oneRepSize = static_cast<uint32_t>(GetVecLen() / sizeof(float));

  Reg::RegTensor<T> srcReg, dstReg;
  Reg::RegTensor<float> xReg, negReg, expReg, denomReg, onesReg, resultReg;
  Reg::MaskReg mask;

  for (uint16_t i = 0; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<float>(calCount);
    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, src + i * oneRepSize);
      Reg::Cast<float, half, CAST_F16_TO_F32>(xReg, srcReg, mask);
    } else {
      Reg::LoadAlign(xReg, src + i * oneRepSize);
    }

    // 1. negReg = -x  &  onesReg = 1.0  (independent, can issue in parallel)
    Reg::Neg(negReg, xReg, mask);
    Reg::Duplicate(onesReg, 1.0f, mask);
    // 2. expReg = exp(-x)  (HIGH LATENCY — NR steps from Div below fill this bubble)
    Reg::Exp(expReg, negReg, mask);
    // 3. denomReg = 1 + exp(-x)
    Reg::Adds(denomReg, expReg, 1.0f, mask);
    // 4. onesReg = 1 / (1 + exp(-x)) = sigmoid(x)  (HIGH_PRECISION_DIV: NR refinement)
    Reg::Div<float, &HIGH_PRECISION_DIV>(onesReg, onesReg, denomReg, mask);
    // 5. resultReg = x * sigmoid(x)
    Reg::Mul(resultReg, xReg, onesReg, mask);

    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::Cast<half, float, CAST_F32_TO_F16>(dstReg, resultReg, mask);
      Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(dst + i * oneRepSize, dstReg, mask);
    } else {
      Reg::StoreAlign(dst + i * oneRepSize, resultReg, mask);
    }
  }
}

}  // namespace SiluAPI
}  // namespace AscendC

/**
 * @brief SiluExtend - compute the Sigmoid Linear Unit (Swish) activation: y = x * sigmoid(x)
 *
 * Uses register-level SIMD so all intermediates stay in vector registers,
 * eliminating the per-step UB load/store round-trips of the high-level
 * API approach.
 * - float: Neg(-x)+Duplicate(1) → Exp → Adds(1) → HP-Div(1,denom) → Mul(x)
 * - half:  cast to float, same compute in float, cast back to half
 *
 * @tparam T data type, supports float, half
 * @param dst output tensor
 * @param src input tensor
 * @param tmpBuf temporary buffer (unused; kept for API compatibility)
 * @param calCount number of elements to compute
 */
template <typename T>
__aicore__ inline void SiluExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                  AscendC::LocalTensor<uint8_t> &tmpBuf, const uint32_t calCount) {
  static_assert(AscendC::SupportType<T, half, float>(), "SiluExtend only supports half and float on current device!");
  (void)tmpBuf;

  constexpr uint32_t oneRepSize = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
  const uint16_t repeatTimes = AscendC::CeilDivision(calCount, oneRepSize);
  AscendC::SiluAPI::SiluCompute((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), calCount, repeatTimes);
}

#endif  // __ASCENDC_API_REGBASE_SILU_H__
