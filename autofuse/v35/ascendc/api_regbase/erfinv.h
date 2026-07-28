/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_ERFINV_H
#define CANN_GRAPH_ENGINE_ERFINV_H

namespace AscendC {
namespace ErfinvAPI {

constexpr Reg::CastTrait castTraitF162F32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::UNKNOWN};
constexpr Reg::CastTrait castTraitF322F16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::CAST_RINT};

// Threshold for selecting approximation method
constexpr float ERFINV_LOG2_THRESHOLD = -8.2f;
// nan
constexpr uint32_t FLOAT_NAN = 0x7fc00000;

#ifndef INFINITY
#define INFINITY (1.0f / 0.0f)
#endif

__simd_callee__ inline void ErfinvPrepareMiddleInput(Reg::RegTensor<float> &srcReg, Reg::RegTensor<float> &log2Reg,
                                                     Reg::RegTensor<float> &negLog2Reg, Reg::MaskReg &mask) {
  Reg::RegTensor<float> tempReg;
  // tempReg <== -x
  Reg::Neg(tempReg, srcReg, mask);
  // tempReg <== 1 + x * (-x) = 1 - x^2
  Reg::Mul(tempReg, srcReg, tempReg, mask);
  Reg::Adds(tempReg, tempReg, 1.0f, mask);
  // log2Reg = log2(tempReg)
  Reg::Log2(log2Reg, tempReg, mask);
  // negLog2Reg = -log2Reg
  Reg::Neg(negLog2Reg, log2Reg, mask);
}

// Polynomial coefficients for near-±1 approximation (using rsqrtNegLog)
// poly = ((((- 0.5899144 * rsqrtNegLog - 0.6630042) * rsqrtNegLog + 1.5970111) * rsqrtNegLog - 0.67521554) *
// rsqrtNegLog - 0.09522479) * rsqrtNegLog + 0.83535343
__simd_callee__ inline void ErfinvComputeNearOne(Reg::RegTensor<float> &dstReg, Reg::RegTensor<float> &srcReg,
                                                 Reg::RegTensor<float> &negLog2Reg, Reg::MaskReg &mask) {
  Reg::RegTensor<float> rsqrtNegLogReg, absFinalReg, tempReg, constTempReg;
  Reg::MaskReg signBitReg;

  // tempReg <== sqrt(negLog2Reg)
  Reg::Sqrt(tempReg, negLog2Reg, mask);
  // rsqrtNegLogReg = 1 / sqrt(negLog2Reg)
  Reg::Duplicate(constTempReg, 1.0f, mask);
  Reg::Div(rsqrtNegLogReg, constTempReg, tempReg, mask);

  constexpr float COEF0 = -0.5899144f;
  constexpr float COEF1 = -0.6630042f;
  constexpr float COEF2 = 1.5970111f;
  constexpr float COEF3 = -0.67521554f;
  constexpr float COEF4 = -0.09522479f;
  constexpr float COEF5 = 0.83535343f;

  Reg::Muls(tempReg, rsqrtNegLogReg, COEF0, mask);
  Reg::Adds(tempReg, tempReg, COEF1, mask);
  Reg::Mul(tempReg, tempReg, rsqrtNegLogReg, mask);
  Reg::Adds(tempReg, tempReg, COEF2, mask);
  Reg::Mul(tempReg, tempReg, rsqrtNegLogReg, mask);
  Reg::Adds(tempReg, tempReg, COEF3, mask);
  Reg::Mul(tempReg, tempReg, rsqrtNegLogReg, mask);
  Reg::Adds(tempReg, tempReg, COEF4, mask);
  Reg::Mul(tempReg, tempReg, rsqrtNegLogReg, mask);
  Reg::Adds(tempReg, tempReg, COEF5, mask);

  // rsqrtNegLogReg = 1.0f / rsqrtNegLogReg
  Reg::Div(rsqrtNegLogReg, constTempReg, rsqrtNegLogReg, mask);
  // tempReg <== tempReg / rsqrtNegLog
  Reg::Mul(tempReg, tempReg, rsqrtNegLogReg, mask);

  // Extract sign bit from original x
  Reg::Duplicate(constTempReg, 0.0f, mask);
  Reg::Compare<float, CMPMODE::LT>(signBitReg, srcReg, constTempReg, mask);
  Reg::Abs(absFinalReg, tempReg, mask);
  // Negate result if sign bit was set
  Reg::Neg(tempReg, absFinalReg, signBitReg);
  // Apply sign bit to near-±1 result
  Reg::Select(dstReg, tempReg, absFinalReg, signBitReg);
}

// Polynomial coefficients for middle range approximation (using negLog2)
// poly = (((((((-2.5172708e-10 * negLog2 + 9.427429e-9) * negLog2 - 1.2054752e-7) * negLog2 + 2.1697005e-7) * negLog2
// + 8.0621484e-6) * negLog2 - 3.1675492e-5) * negLog2 - 0.0007743631) * negLog2 + 0.005546588) * negLog2 + 0.16082023)
// * negLog2 + 0.8862269
__simd_callee__ inline void ErfinvComputeMiddle1(Reg::RegTensor<float> &dstReg, Reg::RegTensor<float> &negLog2Reg,
                                                 Reg::MaskReg &mask) {
  Reg::RegTensor<float> tempReg;

  constexpr float COEF0 = -2.5172708e-10f;
  constexpr float COEF1 = 9.427429e-9f;
  constexpr float COEF2 = -1.2054752e-7f;
  constexpr float COEF3 = 2.1697005e-7f;
  constexpr float COEF4 = 0.0000080621484f;

  Reg::Muls(tempReg, negLog2Reg, COEF0, mask);
  Reg::Adds(tempReg, tempReg, COEF1, mask);
  Reg::Mul(tempReg, tempReg, negLog2Reg, mask);
  Reg::Adds(tempReg, tempReg, COEF2, mask);
  Reg::Mul(tempReg, tempReg, negLog2Reg, mask);
  Reg::Adds(tempReg, tempReg, COEF3, mask);
  Reg::Mul(tempReg, tempReg, negLog2Reg, mask);
  Reg::Adds(tempReg, tempReg, COEF4, mask);
  Reg::Mul(dstReg, tempReg, negLog2Reg, mask);
}

__simd_callee__ inline void ErfinvComputeMiddle2(Reg::RegTensor<float> &dstReg, Reg::RegTensor<float> &srcReg,
                                                 Reg::RegTensor<float> &negLog2Reg, Reg::MaskReg &mask) {
  Reg::RegTensor<float> tempReg;

  constexpr float COEF5 = -0.000031675492f;
  constexpr float COEF6 = -0.0007743631f;
  constexpr float COEF7 = 0.005546588f;
  constexpr float COEF8 = 0.16082023f;
  constexpr float COEF9 = 0.8862269f;

  Reg::Adds(tempReg, dstReg, COEF5, mask);
  Reg::Mul(tempReg, tempReg, negLog2Reg, mask);
  Reg::Adds(tempReg, tempReg, COEF6, mask);
  Reg::Mul(tempReg, tempReg, negLog2Reg, mask);
  Reg::Adds(tempReg, tempReg, COEF7, mask);
  Reg::Mul(tempReg, tempReg, negLog2Reg, mask);
  Reg::Adds(tempReg, tempReg, COEF8, mask);
  Reg::Mul(tempReg, tempReg, negLog2Reg, mask);
  Reg::Adds(tempReg, tempReg, COEF9, mask);

  Reg::Mul(dstReg, tempReg, srcReg, mask);
}

// Handle special cases: NaN, Inf, |x| > 1
__simd_callee__ inline void ErfinvHandleNanCases(Reg::RegTensor<float> &dstReg, Reg::RegTensor<float> &srcReg,
                                                 Reg::MaskReg &mask) {
  Reg::RegTensor<float> tempReg, constTempReg;
  Reg::MaskReg notNanMask;

  // not equal nan
  Reg::Compare<float, CMPMODE::EQ>(notNanMask, srcReg, srcReg, mask);
  // not equal +INFINITY
  Reg::Duplicate(constTempReg, INFINITY, mask);
  Reg::Compare<float, CMPMODE::NE>(notNanMask, srcReg, constTempReg, notNanMask);
  // not equal -INFINITY
  Reg::Neg(constTempReg, constTempReg, mask);
  Reg::Compare<float, CMPMODE::NE>(notNanMask, srcReg, constTempReg, notNanMask);
  // less than 1.0f
  Reg::Duplicate(constTempReg, 1.0f, mask);
  Reg::Abs(tempReg, srcReg, mask);
  Reg::Compare<float, CMPMODE::LE>(notNanMask, tempReg, constTempReg, notNanMask);
  // notNanMask: 合法输入的mask为1,不合法输入为0
  Reg::Duplicate(tempReg, (float &)FLOAT_NAN, mask);
  Reg::Select(dstReg, dstReg, tempReg, notNanMask);
}

// Handle special cases: |x| == 1
__simd_callee__ inline void ErfinvHandleOneCases(Reg::RegTensor<float> &dstReg, Reg::RegTensor<float> &srcReg,
                                                 Reg::MaskReg &mask) {
  Reg::RegTensor<float> tempReg, constTempReg;
  Reg::MaskReg notOneMask;
  // not equal 1.0f
  Reg::Duplicate(constTempReg, 1.0f, mask);
  Reg::Compare<float, CMPMODE::NE>(notOneMask, srcReg, constTempReg, mask);
  Reg::Duplicate(tempReg, INFINITY, mask);
  Reg::Select(dstReg, dstReg, tempReg, notOneMask);
  // not equal -1.0f
  Reg::Neg(constTempReg, constTempReg, mask);
  Reg::Compare<float, CMPMODE::NE>(notOneMask, srcReg, constTempReg, mask);
  Reg::Neg(tempReg, tempReg, mask);
  Reg::Select(dstReg, dstReg, tempReg, notOneMask);
}

template <typename T, bool isReuseSource = false>
__simd_vf__ inline void ErfinvCoreImpl(__ubuf__ T *dstUb, __ubuf__ T *srcUb, uint32_t calCount, uint16_t repeatTimes) {
  Reg::MaskReg mask, thresholdMask;
  Reg::RegTensor<T> srcReg;
  Reg::RegTensor<float> dstReg, castReg, constTempReg, log2Reg, negLog2Reg;
  uint32_t calCount1 = calCount;
  uint32_t calCount2 = calCount;

  for (uint16_t i = 0; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<float>(calCount1);
    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcUb + i * B32_DATA_NUM_PER_REPEAT);
      Reg::Cast<float, T, castTraitF162F32>(castReg, srcReg, mask);
    } else {
      Reg::LoadAlign(castReg, srcUb + i * B32_DATA_NUM_PER_REPEAT);
    }

    // compute partial values of middle case
    ErfinvPrepareMiddleInput(castReg, log2Reg, negLog2Reg, mask);
    Reg::Duplicate(constTempReg, ERFINV_LOG2_THRESHOLD, mask);
    Reg::Compare<float, CMPMODE::GE>(thresholdMask, log2Reg, constTempReg, mask);
    ErfinvComputeMiddle1(dstReg, negLog2Reg, thresholdMask);

    Reg::StoreAlign(dstUb + i * B32_DATA_NUM_PER_REPEAT, dstReg, mask);
  }

  Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

  for (uint16_t i = 0; i < repeatTimes; ++i) {
    Reg::RegTensor<float> tempResultReg;
    mask = Reg::UpdateMask<float>(calCount2);
    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcUb + i * B32_DATA_NUM_PER_REPEAT);
      Reg::Cast<float, T, castTraitF162F32>(castReg, srcReg, mask);
    } else {
      Reg::LoadAlign(castReg, srcUb + i * B32_DATA_NUM_PER_REPEAT);
    }
    // compute remainder values of middle case
    ErfinvPrepareMiddleInput(castReg, log2Reg, negLog2Reg, mask);
    Reg::Duplicate(constTempReg, ERFINV_LOG2_THRESHOLD, mask);
    Reg::Compare<float, CMPMODE::GE>(thresholdMask, log2Reg, constTempReg, mask);
    Reg::LoadAlign(dstReg, dstUb + i * B32_DATA_NUM_PER_REPEAT);
    ErfinvComputeMiddle2(dstReg, castReg, negLog2Reg, thresholdMask);
    // compute near one case
    Reg::Compare<float, CMPMODE::LT>(thresholdMask, log2Reg, constTempReg, mask);
    ErfinvComputeNearOne(tempResultReg, castReg, negLog2Reg, thresholdMask);
    Reg::Select(dstReg, tempResultReg, dstReg, thresholdMask);
    // solve special cases
    ErfinvHandleNanCases(dstReg, castReg, mask);
    ErfinvHandleOneCases(dstReg, castReg, mask);

    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::Cast<T, float, castTraitF322F16>(srcReg, dstReg, mask);
      Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dstUb + i * B32_DATA_NUM_PER_REPEAT, srcReg, mask);
    } else {
      Reg::StoreAlign(dstUb + i * B32_DATA_NUM_PER_REPEAT, dstReg, mask);
    }
  }
}

}  // namespace ErfinvAPI

template <typename T, bool isReuseSource = false>
__aicore__ inline void Erfinv(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                              const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount) {
  // Only for AI Vector Core.
  if ASCEND_IS_AIC {
    return;
  }

  __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
  __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
  uint16_t repeatTimes = CeilDivision(calCount, B32_DATA_NUM_PER_REPEAT);
  ErfinvAPI::ErfinvCoreImpl<T, isReuseSource>(dstUb, srcUb, calCount, repeatTimes);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void Erfinv(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                              const uint32_t calCount) {
  // Only for AI Vector Core.
  if ASCEND_IS_AIC {
    return;
  }

  // Using Stack Space to Allocate tmpBuffer
  LocalTensor<uint8_t> sharedTmpBuffer;
  bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
  ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
  Erfinv<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}

}  // namespace AscendC

#endif  // CANN_GRAPH_ENGINE_ERFINV_H
