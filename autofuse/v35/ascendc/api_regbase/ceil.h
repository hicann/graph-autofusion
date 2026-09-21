/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_CEIL_H__
#define __ASCENDC_API_REGBASE_CEIL_H__

namespace AscendC {
namespace CeilAPI {

// 符号位掩码：Truncate<CAST_CEIL> 的取整结果不携带符号位，需从原输入回填。
constexpr uint32_t UINT32_SIGN = 0x80000000u;  // fp32
constexpr uint16_t UINT16_SIGN = 0x8000u;      // fp16

template <typename T>
__simd_vf__ inline void CeilCompute(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount, uint16_t repeatTimes) {
  constexpr uint32_t oneRepSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));

  Reg::RegTensor<T> srcReg, dstReg;
  Reg::MaskReg mask;

  if constexpr (sizeof(T) == sizeof(float)) {
    Reg::RegTensor<uint32_t> signReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
      mask = Reg::UpdateMask<T>(calCount);
      Reg::LoadAlign(srcReg, src + i * oneRepSize);

      // 向 +∞ 取整：ceil(2.5)=3、ceil(-2.5)=-2；结果不带符号位。
      Reg::Truncate<T, RoundMode::CAST_CEIL, Reg::MaskMergeMode::ZEROING>(dstReg, srcReg, mask);
      // 回填符号位：sign = sign(x) | |ceil(x)|。NaN/±Inf 位模式经 OR 天然透传。
      Reg::Duplicate(signReg, UINT32_SIGN, mask);
      Reg::And(signReg, signReg, (Reg::RegTensor<uint32_t> &)srcReg, mask);
      Reg::Or(signReg, signReg, (Reg::RegTensor<uint32_t> &)dstReg, mask);

      Reg::StoreAlign(dst + i * oneRepSize, (Reg::RegTensor<T> &)signReg, mask);
    }
  } else {
    Reg::RegTensor<uint16_t> signReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
      mask = Reg::UpdateMask<T>(calCount);
      Reg::LoadAlign(srcReg, src + i * oneRepSize);

      Reg::Truncate<T, RoundMode::CAST_CEIL, Reg::MaskMergeMode::ZEROING>(dstReg, srcReg, mask);
      Reg::Duplicate(signReg, UINT16_SIGN, mask);
      Reg::And(signReg, signReg, (Reg::RegTensor<uint16_t> &)srcReg, mask);
      Reg::Or(signReg, signReg, (Reg::RegTensor<uint16_t> &)dstReg, mask);

      Reg::StoreAlign(dst + i * oneRepSize, (Reg::RegTensor<T> &)signReg, mask);
    }
  }
}

}  // namespace CeilAPI
}  // namespace AscendC

template <typename T>
__aicore__ inline void CeilExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                  AscendC::LocalTensor<uint8_t> &tmpBuf, const uint32_t calCount) {
  static_assert(AscendC::SupportType<T, half, float>(), "CeilExtend only supports half and float on current device!");
  (void)tmpBuf;

  constexpr uint32_t oneRepSize = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  const uint16_t repeatTimes = AscendC::CeilDivision(calCount, oneRepSize);
  AscendC::CeilAPI::CeilCompute((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), calCount, repeatTimes);
}

#endif  // __ASCENDC_API_REGBASE_CEIL_H__
