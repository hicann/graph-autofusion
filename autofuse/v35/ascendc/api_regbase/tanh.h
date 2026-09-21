/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_TANH_H__
#define __ASCENDC_API_REGBASE_TANH_H__

namespace AscendC {
namespace TanhAPI {

constexpr float FP32_ZERO_015 = 0.0'157'396'831f;
constexpr float FP32_ZERO_NEG_052 = -0.0'523'039'624f;
constexpr float FP32_ZERO_133 = 0.133'152'977f;
constexpr float FP32_ZERO_NEG_333 = -0.333'327'681f;
constexpr float FP32_TWO = 2.0f;
constexpr float FP32_ONE = 1.0f;
constexpr float FP32_ZERO_NEG_TWO = -2.0f;
constexpr float FP32_ZERO_6 = 0.60'000'002'384'185'791'016f;
constexpr float FP32_SAT_BOUND = 9.010'913'848'876'953'125f;
constexpr uint32_t TANH_FP32_SIGN_MASK = 0x80'000'000;

constexpr Reg::CastTrait CAST_B16_TO_F32 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};
constexpr Reg::CastTrait CAST_F32_TO_B16 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

__simd_callee__ inline void TanhCoreCompute(Reg::RegTensor<float> &value1, Reg::RegTensor<float> &value2,
                                            Reg::RegTensor<float> &one, Reg::RegTensor<uint32_t> &signMask,
                                            Reg::RegTensor<float> &input, Reg::RegTensor<float> &output,
                                            Reg::MaskReg &mask) {
  Reg::RegTensor<float> inputSqr;
  Reg::RegTensor<float> inputAbs;
  Reg::RegTensor<float> inputMid;
  Reg::RegTensor<uint32_t> sign;
  Reg::MaskReg cmpMask;
  Reg::MaskReg satMask;

  Reg::Mul(inputSqr, input, input, mask);
  Reg::Muls(output, inputSqr, FP32_ZERO_015, mask);
  Reg::Adds(output, output, FP32_ZERO_NEG_052, mask);
  Reg::MulDstAdd(output, inputSqr, value1, mask);
  Reg::MulDstAdd(output, inputSqr, value2, mask);
  Reg::Mul(output, output, inputSqr, mask);
  Reg::MulDstAdd(output, input, input, mask);
  Reg::Abs(inputAbs, input, mask);
  Reg::Muls(inputMid, inputAbs, FP32_TWO, mask);
  Reg::Exp(inputMid, inputMid, mask);
  Reg::Adds(inputMid, inputMid, FP32_ONE, mask);
  Reg::Div(inputMid, one, inputMid, mask);
  Reg::Muls(inputMid, inputMid, FP32_ZERO_NEG_TWO, mask);
  Reg::Adds(inputMid, inputMid, FP32_ONE, mask);
  Reg::Compares<float, CMPMODE::GE>(satMask, inputAbs, FP32_SAT_BOUND, mask);
  Reg::Select(inputMid, one, inputMid, satMask);
  Reg::And(sign, signMask, (Reg::RegTensor<uint32_t> &)input, mask);
  Reg::Compares<float, CMPMODE::GE>(cmpMask, inputAbs, FP32_ZERO_6, mask);
  Reg::Select(output, inputMid, output, cmpMask);
  Reg::Or((Reg::RegTensor<uint32_t> &)output, (Reg::RegTensor<uint32_t> &)output, sign, mask);
}

template <typename T>
__simd_vf__ inline void TanhCompute(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount, uint16_t repeatTimes) {
  constexpr uint32_t oneRepSize = static_cast<uint32_t>(GetVecLen() / sizeof(float));

  Reg::RegTensor<float> input;
  Reg::RegTensor<float> output;
  Reg::RegTensor<float> value1;
  Reg::RegTensor<float> value2;
  Reg::RegTensor<float> one;
  Reg::RegTensor<uint32_t> signMask;
  Reg::RegTensor<T> inputB16;
  Reg::RegTensor<T> outputB16;
  Reg::MaskReg mask;

  Reg::Duplicate(value1, FP32_ZERO_133);
  Reg::Duplicate(value2, FP32_ZERO_NEG_333);
  Reg::Duplicate(one, FP32_ONE);
  Reg::Duplicate(signMask, TANH_FP32_SIGN_MASK);
  for (uint16_t i = 0; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<float>(calCount);
    if constexpr (SupportType<T, float>()) {
      Reg::LoadAlign(input, src + i * oneRepSize);
    } else {
      Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(inputB16, src + i * oneRepSize);
      Reg::Cast<float, T, CAST_B16_TO_F32>(input, inputB16, mask);
    }

    TanhCoreCompute(value1, value2, one, signMask, input, output, mask);

    if constexpr (SupportType<T, float>()) {
      Reg::StoreAlign(dst + i * oneRepSize, output, mask);
    } else {
      Reg::Cast<T, float, CAST_F32_TO_B16>(outputB16, output, mask);
      Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dst + i * oneRepSize, outputB16, mask);
    }
  }
}

}  // namespace TanhAPI
}  // namespace AscendC

template <typename T>
__aicore__ inline void TanhExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                  AscendC::LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount) {
  static_assert(AscendC::SupportType<T, half, float, bfloat16_t>(),
                "TanhExtend only supports half, float and bfloat16_t on current device!");
  (void)sharedTmpBuffer;

  constexpr uint32_t oneRepSize = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
  const uint16_t repeatTimes = AscendC::CeilDivision(calCount, oneRepSize);
  AscendC::TanhAPI::TanhCompute((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), calCount, repeatTimes);
}

#endif  // __ASCENDC_API_REGBASE_TANH_H__
