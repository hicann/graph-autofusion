/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_RECIPROCAL_H__
#define __ASCENDC_API_REGBASE_RECIPROCAL_H__

namespace AscendC {
namespace ReciprocalAPI {

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
constexpr Reg::DivSpecificMode HIGH_PRECISION_DIV = {Reg::MaskMergeMode::ZEROING, true};

template <typename T>
__simd_vf__ inline void ReciprocalCompute(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount, uint16_t repeatTimes) {
  constexpr uint32_t oneRepSize = static_cast<uint32_t>(GetVecLen() / sizeof(float));

  Reg::RegTensor<T> inputB16;
  Reg::RegTensor<T> outputB16;
  Reg::RegTensor<float> input;
  Reg::RegTensor<float> one;
  Reg::RegTensor<float> output;
  Reg::MaskReg mask;

  for (uint16_t i = 0; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<float>(calCount);
    if constexpr (SupportType<T, float>()) {
      Reg::LoadAlign(input, src + i * oneRepSize);
    } else {
      Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(inputB16, src + i * oneRepSize);
      Reg::Cast<float, T, CAST_B16_TO_F32>(input, inputB16, mask);
    }

    Reg::Duplicate(one, 1.0f, mask);
    Reg::Div<float, &HIGH_PRECISION_DIV>(output, one, input, mask);

    if constexpr (SupportType<T, float>()) {
      Reg::StoreAlign(dst + i * oneRepSize, output, mask);
    } else {
      Reg::Cast<T, float, CAST_F32_TO_B16>(outputB16, output, mask);
      if constexpr (SupportType<T, bfloat16_t>()) {
        Reg::RegTensor<uint32_t> specialValue;
        Reg::RegTensor<uint32_t> exponent;
        Reg::MaskReg specialMask;

        Reg::Duplicate(exponent, 0x7f80U, mask);
        Reg::Or(specialValue, (Reg::RegTensor<uint32_t> &)inputB16, exponent, mask);
        Reg::Compares<float, CMPMODE::EQ>(specialMask, input, 0.0f, mask);
        Reg::Select((Reg::RegTensor<uint32_t> &)outputB16, specialValue, (Reg::RegTensor<uint32_t> &)outputB16,
                    specialMask);

        Reg::Compare<float, CMPMODE::NE>(specialMask, input, input, mask);
        Reg::Select((Reg::RegTensor<uint32_t> &)outputB16, (Reg::RegTensor<uint32_t> &)inputB16,
                    (Reg::RegTensor<uint32_t> &)outputB16, specialMask);
      }
      Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dst + i * oneRepSize, outputB16, mask);
    }
  }
}

}  // namespace ReciprocalAPI
}  // namespace AscendC

template <typename T>
__aicore__ inline void ReciprocalExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount) {
  static_assert(AscendC::SupportType<T, half, float, bfloat16_t>(),
                "ReciprocalExtend only supports half, float and bfloat16_t on current device!");
  (void)sharedTmpBuffer;

  constexpr uint32_t oneRepSize = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
  const uint16_t repeatTimes = AscendC::CeilDivision(calCount, oneRepSize);
  AscendC::ReciprocalAPI::ReciprocalCompute((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), calCount,
                                            repeatTimes);
}

#endif  // __ASCENDC_API_REGBASE_RECIPROCAL_H__
