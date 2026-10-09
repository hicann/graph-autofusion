/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. This file is provided on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied.
 */
#ifndef __ASCENDC_API_REGBASE_ROUND_H__
#define __ASCENDC_API_REGBASE_ROUND_H__

namespace AscendC {
namespace RoundAPI {

constexpr uint32_t ROUND_FLOAT_SIGN_MASK = 0x80000000u;
constexpr uint16_t ROUND_B16_SIGN_MASK = 0x8000u;
constexpr uint16_t ROUND_BF16_EXPONENT_MASK = 0x7f80u;
constexpr uint16_t ROUND_BF16_MANTISSA_MASK = 0x007fu;
// torch.ops.aten.round.default canonicalizes BF16 NaNs to this bit pattern.
constexpr uint16_t ROUND_BF16_CANONICAL_NAN = 0xffffu;

__simd_callee__ inline void RestoreBf16SpecialBits(Reg::RegTensor<uint16_t> &dstBitsReg,
                                                   Reg::RegTensor<uint16_t> &srcBitsReg,
                                                   Reg::RegTensor<uint16_t> &exponentMaskReg,
                                                   Reg::RegTensor<uint16_t> &mantissaMaskReg,
                                                   Reg::RegTensor<uint16_t> &canonicalNanReg,
                                                   Reg::MaskReg &activeMask) {
  Reg::RegTensor<uint16_t> exponentReg;
  Reg::RegTensor<uint16_t> mantissaReg;
  Reg::MaskReg exponentNanMask;
  Reg::MaskReg mantissaNanMask;
  Reg::MaskReg mantissaInfMask;
  Reg::MaskReg nanMask;
  Reg::MaskReg infMask;

  Reg::And(exponentReg, srcBitsReg, exponentMaskReg, activeMask);
  Reg::And(mantissaReg, srcBitsReg, mantissaMaskReg, activeMask);
  Reg::CompareScalar<uint16_t, CMPMODE::EQ>(exponentNanMask, exponentReg, ROUND_BF16_EXPONENT_MASK, activeMask);
  Reg::CompareScalar<uint16_t, CMPMODE::NE>(mantissaNanMask, mantissaReg, 0u, activeMask);
  Reg::And(nanMask, exponentNanMask, mantissaNanMask, activeMask);
  Reg::Select(dstBitsReg, canonicalNanReg, dstBitsReg, nanMask);
  Reg::CompareScalar<uint16_t, CMPMODE::EQ>(mantissaInfMask, mantissaReg, 0u, activeMask);
  Reg::And(infMask, exponentNanMask, mantissaInfMask, activeMask);
  Reg::Select(dstBitsReg, srcBitsReg, dstBitsReg, infMask);
}

// Round to nearest integer with ties-to-even, matching torch.round numerically.
template <typename T, typename BitT>
__simd_callee__ inline void RoundComputeImpl(__ubuf__ T *dstUb, __ubuf__ T *srcUb, uint32_t calCount,
                                             const uint16_t repeatTimes, const BitT signBitMask) {
  uint32_t remainingCount = calCount;
  Reg::MaskReg activeMask;
  Reg::RegTensor<T> srcReg, dstReg;
  Reg::RegTensor<BitT> signMaskReg;
  Reg::RegTensor<BitT> signBitsReg;
  Reg::RegTensor<uint16_t> bf16ExponentMaskReg;
  Reg::RegTensor<uint16_t> bf16MantissaMaskReg;
  Reg::RegTensor<uint16_t> bf16CanonicalNanReg;
  constexpr uint32_t elementsPerRepeat = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  Reg::Duplicate(signMaskReg, signBitMask);
  if constexpr (std::is_same_v<T, bfloat16_t>) {
    Reg::Duplicate(bf16ExponentMaskReg, ROUND_BF16_EXPONENT_MASK);
    Reg::Duplicate(bf16MantissaMaskReg, ROUND_BF16_MANTISSA_MASK);
    Reg::Duplicate(bf16CanonicalNanReg, ROUND_BF16_CANONICAL_NAN);
  }

  for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
    activeMask = Reg::UpdateMask<T>(remainingCount);
    Reg::LoadAlign<T>(srcReg, srcUb + repeat * elementsPerRepeat);
    Reg::Truncate<T, RoundMode::CAST_RINT, Reg::MaskMergeMode::ZEROING>(dstReg, srcReg, activeMask);
    Reg::And(signBitsReg, signMaskReg, (Reg::RegTensor<BitT> &)srcReg, activeMask);
    Reg::Or((Reg::RegTensor<BitT> &)dstReg, (Reg::RegTensor<BitT> &)dstReg, signBitsReg, activeMask);
    if constexpr (std::is_same_v<T, bfloat16_t>) {
      // Truncate may alter BF16 NaN and infinity bits. Restore the expected special-value encodings.
      RestoreBf16SpecialBits((Reg::RegTensor<uint16_t> &)dstReg, (Reg::RegTensor<uint16_t> &)srcReg,
                             bf16ExponentMaskReg, bf16MantissaMaskReg, bf16CanonicalNanReg, activeMask);
    }
    Reg::StoreAlign<T>(dstUb + repeat * elementsPerRepeat, dstReg, activeMask);
  }
}

template <typename T>
__simd_vf__ inline void RoundCompute(__ubuf__ T *dstUb, __ubuf__ T *srcUb, uint32_t calCount,
                                     const uint16_t repeatTimes) {
  if constexpr (std::is_same_v<T, float>) {
    RoundComputeImpl<T, uint32_t>(dstUb, srcUb, calCount, repeatTimes, ROUND_FLOAT_SIGN_MASK);
  } else {
    RoundComputeImpl<T, uint16_t>(dstUb, srcUb, calCount, repeatTimes, ROUND_B16_SIGN_MASK);
  }
}

}  // namespace RoundAPI
}  // namespace AscendC

/**
 * @brief RoundExtend - round to nearest integer with ties-to-even.
 *
 * Supports float, half, and bfloat16_t. The temporary buffer is retained for
 * compatibility with the other register-based APIs and is not used here.
 */
template <typename T>
__aicore__ inline void RoundExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                   AscendC::LocalTensor<uint8_t> &tmpBuf, const uint32_t calCount) {
  static_assert(AscendC::SupportType<T, half, float, bfloat16_t>(),
                "RoundExtend only supports half, float and bfloat16_t on current device!");
  (void)tmpBuf;

  constexpr uint32_t elementsPerRepeat = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  const uint16_t repeatTimes = AscendC::CeilDivision(calCount, elementsPerRepeat);
  AscendC::RoundAPI::RoundCompute((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), calCount,
                                  repeatTimes);
}

#endif  // __ASCENDC_API_REGBASE_ROUND_H__
