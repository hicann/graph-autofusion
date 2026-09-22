/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_COSH_H__
#define __ASCENDC_API_REGBASE_COSH_H__

namespace AscendC {
namespace CoshAPI {

// 换底常量：log2(e)。
constexpr float LOG2E = 1.442695041f;
// Cody-Waite 双常量分解（非 FMA 版本）：
//   LN2_HI = 0x3F317200 = 0.693145751953125，尾数低 9 位为 0，n ≤ 511 内 n·LN2_HI 精确；
//   LN2_LO = ln(2) - LN2_HI 的 fp32 近似。
//   乘积精确 + Sterbenz 减法精确，与双 FMA 单次舍入的精度等价。
constexpr float NEG_LN2_HI = -0.693145751953125f;
constexpr float LN2_LO = 1.4286068e-6f;
// n 钳位上限：125 + n ≤ 251 < 256，位构造 2^(n-2) 不越 IEEE754 指数字段。
constexpr float N_CLAMP = 126.0f;
// bit 0x4B40007D = 1.5·2^23 + 125（含 +125 偏置）；n + magic 精确（< 2^24），
// 位重解释后 <<23，低 9 位 125+n 落入指数字段 ⇒ 2^(n-2)。
constexpr float EXP_MAGIC = 12583037.0f;
constexpr int16_t FP32_MANTISSA_BITS = 23;
constexpr float TWO = 2.0f;
constexpr float ONE_EIGHTH = 0.125f;
// |x| ≥ 90 ⇒ +inf；[≈89.42, 90) 由 2·(e^x/4) 自然溢出兜底。
constexpr float OVF_THRESHOLD = 90.0f;

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
__simd_vf__ inline void CoshCompute(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount, uint16_t repeatTimes) {
  constexpr uint32_t oneRepSize = static_cast<uint32_t>(GetVecLen() / sizeof(float));
  const float inf = __builtin_huge_valf();

  Reg::RegTensor<T> srcReg, dstReg;
  Reg::RegTensor<float> xReg, absReg, nReg, cwReg, rReg, p2Reg, erReg, workReg, tmpReg;
  Reg::MaskReg mask, selectMask;

  for (uint16_t i = 0; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<float>(calCount);
    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, src + i * oneRepSize);
      Reg::Cast<float, half, CAST_F16_TO_F32>(xReg, srcReg, mask);
    } else {
      Reg::LoadAlign(xReg, src + i * oneRepSize);
    }

    // ① ax = |x|（cosh 偶函数）。
    Reg::Abs(absReg, xReg, mask);

    // ② n = trunc(ax · log2e)；ax ≥ 0，floor 即 trunc。
    Reg::Muls(nReg, absReg, LOG2E, mask);
    Reg::Truncate<float, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(nReg, nReg, mask);

    // ③ n = min(n, 126)：钳位在范围缩减之前，使 r 与 2^(n-2) 用同一 n 合成；
    //    被钳元素残差 r 变大由 Exp 吸收，e^r · 2^(n-2) = e^x/4 仍精确。
    Reg::Mins(nReg, nReg, N_CLAMP, mask);

    // ④⑤ Cody-Waite 范围缩减 r = ax − n·ln2_hi − n·ln2_lo。
    //    cwReg = n·(−ln2_hi) 已带负号，第一项用 Add 完成 ax − n·ln2_hi
    //    本实现 LN2_HI = 0x3F317200 欠估 ln2，残差 LN2_LO = ln2 − LN2_HI > 0，
    Reg::Muls(cwReg, nReg, NEG_LN2_HI, mask);
    Reg::Add(rReg, absReg, cwReg, mask);
    Reg::Muls(cwReg, nReg, LN2_LO, mask);
    Reg::Sub(rReg, rReg, cwReg, mask);

    // ⑥⑨ 位操作构造 p2 = 2^(n-2)：(n + 0x4B40007D) 位重解释后 <<23，
    //    32 位逐位左移高位丢弃、低 9 位 (125+n) ≤ 251 落入 IEEE754 指数字段。
    //    内联引用强转，使 ShiftLefts 直接写回 p2Reg 同一寄存器。
    Reg::Adds(p2Reg, nReg, EXP_MAGIC, mask);
    Reg::ShiftLefts((Reg::RegTensor<uint32_t> &)p2Reg, (Reg::RegTensor<uint32_t> &)p2Reg, FP32_MANTISSA_BITS, mask);

    // ⑩⑪ 合成 e^x/4 = exp(r) · 2^(n-2)。
    Reg::Exp(erReg, rReg, mask);
    Reg::Mul(erReg, erReg, p2Reg, mask);

    // ⑫⑮ cosh 合成：y = 2·(e^x/4) + 0.125/(e^x/4) = e^x/2 + e^(-x)/2。
    Reg::Duplicate(tmpReg, ONE_EIGHTH);
    Reg::Div<float, &HIGH_PRECISION_DIV>(workReg, tmpReg, erReg, mask);
    Reg::Muls(p2Reg, erReg, TWO, mask);
    Reg::Add(workReg, p2Reg, workReg, mask);

    // ⑯⑰ 溢出兜底：|x| ≥ 90 ⇒ +inf。
    //    NaN 与阈值比较恒为 false ⇒ 选 work，work 对 NaN 天然为 NaN。
    Reg::CompareScalar<float, CMPMODE::GE>(selectMask, absReg, OVF_THRESHOLD, mask);
    Reg::Duplicate(tmpReg, inf);
    Reg::Select(workReg, tmpReg, workReg, selectMask);

    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::Cast<half, float, CAST_F32_TO_F16>(dstReg, workReg, mask);
      Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(dst + i * oneRepSize, dstReg, mask);
    } else {
      Reg::StoreAlign(dst + i * oneRepSize, workReg, mask);
    }
  }
}

}  // namespace CoshAPI
}  // namespace AscendC

template <typename T>
__aicore__ inline void CoshExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                  AscendC::LocalTensor<uint8_t> &tmpBuf, const uint32_t calCount) {
  static_assert(AscendC::SupportType<T, half, float>(), "CoshExtend only supports half and float on current device!");
  (void)tmpBuf;

  constexpr uint32_t oneRepSize = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
  const uint16_t repeatTimes = AscendC::CeilDivision(calCount, oneRepSize);
  AscendC::CoshAPI::CoshCompute((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), calCount, repeatTimes);
}

#endif  // __ASCENDC_API_REGBASE_COSH_H__
