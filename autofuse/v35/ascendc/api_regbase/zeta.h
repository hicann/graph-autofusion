/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_ZETA_H__
#define __ASCENDC_API_REGBASE_ZETA_H__

namespace AscendC {
template <typename T>
__simd_callee__ inline void ZetaSeriesTerm(Reg::RegTensor<T> &bTensor, Reg::RegTensor<T> &aTensor,
                                           Reg::RegTensor<T> &mulTmp, Reg::MaskReg &maskSignFlip, Reg::MaskReg &mask) {
  constexpr T ZERO = static_cast<T>(0.0);
  constexpr T ONE = static_cast<T>(1.0);
  constexpr T NEG_ONE = static_cast<T>(-1.0);
  Reg::RegTensor<T> lnTmp, tTensor, wTensor;
  Reg::MaskReg mask2, mask3;
  Reg::Abs(lnTmp, aTensor, mask);
  Reg::Log(tTensor, lnTmp, mask);
  Reg::Mul(tTensor, mulTmp, tTensor, mask);
  Reg::Exp(bTensor, tTensor, mask);
  Reg::Compares<T, CMPMODE::LT>(mask2, aTensor, ZERO, mask);
  Reg::Duplicate(lnTmp, ZERO, mask);
  Reg::Duplicate(tTensor, ZERO, mask);
  Reg::Adds(tTensor, tTensor, ONE, mask);
  Reg::Select(lnTmp, tTensor, lnTmp, mask2);
  Reg::Duplicate(wTensor, ZERO, mask);
  Reg::Select(lnTmp, lnTmp, wTensor, maskSignFlip);
  Reg::Compares<T, CMPMODE::GT>(mask3, lnTmp, ZERO, mask);
  Reg::Muls(wTensor, bTensor, NEG_ONE, mask);
  Reg::Select(bTensor, wTensor, bTensor, mask3);
}

template <typename T>
__simd_vf__ inline void ZetaImplVF(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint32_t calCount) {
  constexpr T ZERO = static_cast<T>(0.0);
  constexpr T HALF = static_cast<T>(0.5);
  constexpr T ONE = static_cast<T>(1.0);
  constexpr T NEG_ONE = static_cast<T>(-1.0);
  constexpr T TEN = static_cast<T>(10.0);

  uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, vlSize));

  Reg::MaskReg mask;

  Reg::RegTensor<T> src0Reg, src1Reg, dstReg;
  Reg::RegTensor<T> aTensor, bTensor, wTensor, tTensor, lnTmp, mulTmp, xMinusOne, kPlusX, nanTensor;
  Reg::MaskReg mask0, mask1, mask2, mask3;
  Reg::MaskReg maskSignFlip;
  Reg::MaskReg maskLoop;
  Reg::RegTensor<T> bLast, wLast;
  Reg::RegTensor<T> aStartReg;

  for (uint16_t i = 0U; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<T>(calCount);

    Reg::LoadAlign(src0Reg, src0 + i * vlSize);
    Reg::LoadAlign(src1Reg, src1 + i * vlSize);

    Reg::Duplicate(nanTensor, (float &)F32_NAN, mask);

    Reg::Compares<T, CMPMODE::LT>(mask0, src0Reg, ONE, mask);
    Reg::Compares<T, CMPMODE::EQ>(mask1, src0Reg, ONE, mask);
    Reg::Compares<T, CMPMODE::LE>(mask2, src1Reg, ZERO, mask);

    Reg::Truncate<T, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(aTensor, src1Reg, mask);
    Reg::Compare<T, CMPMODE::EQ>(mask3, aTensor, src1Reg, mask);

    Reg::Truncate<T, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(tTensor, src0Reg, mask);
    Reg::Compare<T, CMPMODE::EQ>(mask2, tTensor, src0Reg, mask);
    Reg::Muls(lnTmp, src0Reg, HALF, mask);
    Reg::Truncate<T, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(wTensor, lnTmp, mask);
    Reg::Compare<T, CMPMODE::NE>(mask3, wTensor, lnTmp, mask);
    Reg::Duplicate(lnTmp, ZERO, mask);
    Reg::Duplicate(tTensor, ZERO, mask);
    Reg::Adds(tTensor, tTensor, ONE, mask);
    Reg::Select(lnTmp, tTensor, lnTmp, mask2);
    Reg::Duplicate(wTensor, ZERO, mask);
    Reg::Select(lnTmp, lnTmp, wTensor, mask3);
    Reg::Compares<T, CMPMODE::GT>(maskSignFlip, lnTmp, ZERO, mask);

    // Shift series starting point to skip negligible terms for large negative q.
    // N_skip = max(0, -floor(q) - 10), a_start = q + N_skip, so |a_start| <= ~10
    // and the dominant terms near a≈0 are within 9+12 = 21 iterations.
    // Special case: when x is odd AND frac(q)==0.5, the dominant terms at
    // a=-0.5 and a=0.5 cancel exactly (for odd x).
    // x <  128: terms are finite in f32, cancellation works, N_skip=0.
    // x >= 128 AND |q| <  1: first term pow(q,-x) overflows f32 → ±inf;
    // keep N_skip=0 and let overflow detection return ±inf.
    // x >= 128 AND |q| >= 1: first term is finite but ±2^x terms overflow
    // f32 to ±inf → NaN.  Skip to a_start=1.5 (N_skip=-floor(q)+1) and
    // zero the Euler-Maclaurin correction (the skipped ±2^x cancel to 0,
    // but the integral approximation cannot capture this).
    // aTensor still holds floor(q) from the Truncate above.
    Reg::Muls(tTensor, aTensor, NEG_ONE, mask);
    Reg::Adds(tTensor, tTensor, static_cast<T>(-10.0), mask);
    Reg::Compares<T, CMPMODE::LT>(mask0, tTensor, ZERO, mask);
    Reg::Select(tTensor, wTensor, tTensor, mask0);
    // Check: frac(q)==0.5 AND x odd
    Reg::Sub(lnTmp, src1Reg, aTensor, mask);
    Reg::Compares<T, CMPMODE::EQ>(mask3, lnTmp, HALF, mask);
    Reg::Duplicate(bTensor, ONE, mask);
    Reg::Select(lnTmp, bTensor, wTensor, mask3);
    Reg::Select(kPlusX, bTensor, wTensor, maskSignFlip);
    Reg::Mul(lnTmp, lnTmp, kPlusX, mask);
    Reg::Compares<T, CMPMODE::GT>(mask3, lnTmp, ZERO, mask);
    // For frac==0.5 AND x odd: force N_skip=0 (default for |q|<1 / x<128)
    Reg::Select(tTensor, wTensor, tTensor, mask3);
    // For frac==0.5 AND x odd AND |q|>=1 AND x>=128: N_skip=-floor(q)+1
    Reg::Abs(lnTmp, src1Reg, mask);
    Reg::Compares<T, CMPMODE::GE>(mask1, lnTmp, ONE, mask);
    Reg::Compares<T, CMPMODE::GE>(mask2, src0Reg, static_cast<T>(128.0), mask);
    Reg::Select(kPlusX, bTensor, wTensor, mask1);
    Reg::Select(lnTmp, bTensor, wTensor, mask2);
    Reg::Mul(kPlusX, kPlusX, lnTmp, mask);
    Reg::Select(lnTmp, bTensor, wTensor, mask3);
    Reg::Mul(lnTmp, lnTmp, kPlusX, mask);
    Reg::Compares<T, CMPMODE::GT>(mask1, lnTmp, ZERO, mask);
    Reg::Muls(lnTmp, aTensor, NEG_ONE, mask);
    Reg::Adds(lnTmp, lnTmp, ONE, mask);
    Reg::Select(tTensor, lnTmp, tTensor, mask1);
    // mask0: true where N_skip > 0 (for skipped-sum guard later)
    Reg::Compares<T, CMPMODE::GT>(mask0, tTensor, ZERO, mask);
    Reg::Add(aStartReg, src1Reg, tTensor, mask);

    Reg::Muls(mulTmp, src0Reg, NEG_ONE, mask);

    // First term: pow(a_start, -x) = exp(-x * ln|a_start|), with sign correction
    Reg::Abs(lnTmp, aStartReg, mask);
    Reg::Compares<T, CMPMODE::EQ>(mask3, lnTmp, ONE, mask);
    Reg::Log(tTensor, lnTmp, mask);
    Reg::Mul(tTensor, mulTmp, tTensor, mask);
    Reg::Exp(dstReg, tTensor, mask);
    // Fix: when |a_start|==1, ln|a_start|==0, -inf*0=NaN. pow(1,-x)=1 for any x.
    Reg::Duplicate(wTensor, ONE, mask);
    Reg::Select(dstReg, wTensor, dstReg, mask3);
    Reg::Compares<T, CMPMODE::LT>(mask2, aStartReg, ZERO, mask);
    Reg::Duplicate(lnTmp, ZERO, mask);
    Reg::Duplicate(tTensor, ZERO, mask);
    Reg::Adds(tTensor, tTensor, ONE, mask);
    Reg::Select(lnTmp, tTensor, lnTmp, mask2);
    Reg::Duplicate(wTensor, ZERO, mask);
    Reg::Select(lnTmp, lnTmp, wTensor, maskSignFlip);
    Reg::Compares<T, CMPMODE::GT>(mask3, lnTmp, ZERO, mask);
    Reg::Muls(wTensor, dstReg, NEG_ONE, mask);
    Reg::Select(dstReg, wTensor, dstReg, mask3);

    // Compute pow(q, -x) for the Euler-Maclaurin correction of skipped terms.
    // When N_skip > 0, the series starts at a_start = q + N_skip, skipping
    // terms k=0..N_skip-1.  Approximate that skipped sum with the
    // Euler-Maclaurin integral + 1/2 endpoint correction:
    // integral = (q^(1-x) - a_start^(1-x)) / (x-1)
    // Since q^(1-x) = q * pow(q,-x) and a_start^(1-x) = a_start * pow(a_start,-x),
    // we can reuse the sign-corrected first terms.
    Reg::Abs(lnTmp, src1Reg, mask);
    Reg::Compares<T, CMPMODE::EQ>(mask3, lnTmp, ONE, mask);
    Reg::Log(tTensor, lnTmp, mask);
    Reg::Mul(tTensor, mulTmp, tTensor, mask);
    Reg::Exp(bLast, tTensor, mask);
    Reg::Duplicate(wTensor, ONE, mask);
    Reg::Select(bLast, wTensor, bLast, mask3);
    Reg::Compares<T, CMPMODE::LT>(mask2, src1Reg, ZERO, mask);
    Reg::Duplicate(lnTmp, ZERO, mask);
    Reg::Duplicate(tTensor, ZERO, mask);
    Reg::Adds(tTensor, tTensor, ONE, mask);
    Reg::Select(lnTmp, tTensor, lnTmp, mask2);
    Reg::Duplicate(wTensor, ZERO, mask);
    Reg::Select(lnTmp, lnTmp, wTensor, maskSignFlip);
    Reg::Compares<T, CMPMODE::GT>(mask3, lnTmp, ZERO, mask);
    Reg::Muls(wTensor, bLast, NEG_ONE, mask);
    Reg::Select(bLast, wTensor, bLast, mask3);
    // bLast = pow(q, -x) with sign correction

    // integral = (q * pow(q,-x) - a_start * pow(a_start,-x)) / (x - 1)
    Reg::Mul(lnTmp, src1Reg, bLast, mask);
    Reg::Mul(tTensor, aStartReg, dstReg, mask);
    Reg::Sub(lnTmp, lnTmp, tTensor, mask);
    Reg::Adds(tTensor, src0Reg, NEG_ONE, mask);
    Reg::Div(lnTmp, lnTmp, tTensor, mask);
    // half-corr = (pow(q,-x) - pow(a_start,-x)) / 2
    Reg::Sub(tTensor, bLast, dstReg, mask);
    Reg::Muls(tTensor, tTensor, HALF, mask);
    // Zero out skipped sum where N_skip = 0 (mask0 = N_skip > 0)
    // to avoid inf-inf=NaN when the first term is inf.
    // Also zero where the special ±2^x cancellation skip was used (mask1),
    // since the integral approximation cannot capture the exact cancellation.
    Reg::Duplicate(wTensor, ZERO, mask);
    Reg::Select(lnTmp, lnTmp, wTensor, mask0);
    Reg::Select(tTensor, tTensor, wTensor, mask0);
    Reg::Select(lnTmp, wTensor, lnTmp, mask1);
    Reg::Select(tTensor, wTensor, tTensor, mask1);
    // Add skipped-sum approximation to the result
    Reg::Add(dstReg, dstReg, lnTmp, mask);
    Reg::Add(dstReg, dstReg, tTensor, mask);

    Reg::Muls(aTensor, aStartReg, ONE, mask);

    // Series loop part 1: 9 unconditional iterations (q+1 through q+9)
    for (uint16_t iter = 0; iter < 9; ++iter) {
      Reg::Adds(aTensor, aTensor, ONE, mask);
      ZetaSeriesTerm<T>(bTensor, aTensor, mulTmp, maskSignFlip, mask);
      Reg::Add(dstReg, dstReg, bTensor, mask);
    }

    // Save last b and a from unconditional loop
    Reg::Muls(bLast, bTensor, ONE, mask);
    Reg::Muls(wLast, aTensor, ONE, mask);

    for (uint16_t iter = 0; iter < 12; ++iter) {
      Reg::Adds(aTensor, aTensor, ONE, mask);
      Reg::Compares<T, CMPMODE::LE>(maskLoop, aTensor, TEN, mask);
      ZetaSeriesTerm<T>(bTensor, aTensor, mulTmp, maskSignFlip, mask);
      Reg::Select(bLast, bTensor, bLast, maskLoop);
      Reg::Select(wLast, aTensor, wLast, maskLoop);
      Reg::Duplicate(wTensor, ZERO, mask);
      Reg::Select(bTensor, bTensor, wTensor, maskLoop);
      Reg::Add(dstReg, dstReg, bTensor, mask);
    }

    // Second block: s += b * w / (x - 1), using last valid b and w
    Reg::Adds(xMinusOne, src0Reg, NEG_ONE, mask);
    Reg::Mul(mulTmp, bLast, wLast, mask);
    Reg::Div(mulTmp, mulTmp, xMinusOne, mask);
    Reg::Add(dstReg, dstReg, mulTmp, mask);

    // Third block: s -= half * b
    Reg::Muls(lnTmp, bLast, HALF, mask);
    Reg::Sub(dstReg, dstReg, lnTmp, mask);

    // Save result for NaN check later
    Reg::Muls(xMinusOne, dstReg, ONE, mask);

    // Prepare for Bernoulli: w = wLast, b = bLast, a=1, lnTmp = 1/w
    Reg::Muls(wTensor, wLast, ONE, mask);
    Reg::Muls(bTensor, bLast, ONE, mask);
    Reg::Duplicate(lnTmp, ZERO, mask);
    Reg::Adds(lnTmp, lnTmp, ONE, mask);
    Reg::Div(lnTmp, lnTmp, wTensor, mask);
    Reg::Duplicate(aTensor, ONE, mask);

    // Bernoulli series — 12 terms with compile-time constants
    // Handle float32 overflow/underflow: zero out NaN terms (a*b = inf*0 = NaN)
    // Term  0: B_2  / 2!  =  1/12
    {
      T kVal = ZERO;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Compare<T, CMPMODE::NE>(mask3, mulTmp, mulTmp, mask);
      Reg::Duplicate(wTensor, ZERO, mask);
      Reg::Select(mulTmp, wTensor, mulTmp, mask3);
      Reg::Muls(tTensor, mulTmp, 1.0f / 12.0f, mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  1: B_4  / 4!  =  -1/720
    {
      T kVal = 2 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Compare<T, CMPMODE::NE>(mask3, mulTmp, mulTmp, mask);
      Reg::Duplicate(wTensor, ZERO, mask);
      Reg::Select(mulTmp, wTensor, mulTmp, mask3);
      Reg::Muls(tTensor, mulTmp, -1.0f / 720.0f, mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  2: B_6  / 6!  =  1/30240
    {
      T kVal = 4 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Compare<T, CMPMODE::NE>(mask3, mulTmp, mulTmp, mask);
      Reg::Duplicate(wTensor, ZERO, mask);
      Reg::Select(mulTmp, wTensor, mulTmp, mask3);
      Reg::Muls(tTensor, mulTmp, (1.0f / 30240.0f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  3: B_8  / 8!  =  -1/1209600
    {
      T kVal = 6 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Compare<T, CMPMODE::NE>(mask3, mulTmp, mulTmp, mask);
      Reg::Duplicate(wTensor, ZERO, mask);
      Reg::Select(mulTmp, wTensor, mulTmp, mask3);
      Reg::Muls(tTensor, mulTmp, (-1.0f / 1209600.0f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  4: B_10 / 10! =  1/47900160
    {
      T kVal = 8 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Compare<T, CMPMODE::NE>(mask3, mulTmp, mulTmp, mask);
      Reg::Duplicate(wTensor, ZERO, mask);
      Reg::Select(mulTmp, wTensor, mulTmp, mask3);
      Reg::Muls(tTensor, mulTmp, (1.0f / 47900160.0f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  5
    {
      T kVal = 10 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Compare<T, CMPMODE::NE>(mask3, mulTmp, mulTmp, mask);
      Reg::Duplicate(wTensor, ZERO, mask);
      Reg::Select(mulTmp, wTensor, mulTmp, mask3);
      Reg::Muls(tTensor, mulTmp, (-1.0f / 1.8924375803183791606e9f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  6
    {
      T kVal = 12 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Compare<T, CMPMODE::NE>(mask3, mulTmp, mulTmp, mask);
      Reg::Duplicate(wTensor, ZERO, mask);
      Reg::Select(mulTmp, wTensor, mulTmp, mask3);
      Reg::Muls(tTensor, mulTmp, (1.0f / 7.47242496e10f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  7
    {
      T kVal = 14 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Compare<T, CMPMODE::NE>(mask3, mulTmp, mulTmp, mask);
      Reg::Duplicate(wTensor, ZERO, mask);
      Reg::Select(mulTmp, wTensor, mulTmp, mask3);
      Reg::Muls(tTensor, mulTmp, (-1.0f / 2.950130727918164224e12f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  8
    {
      T kVal = 16 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Compare<T, CMPMODE::NE>(mask3, mulTmp, mulTmp, mask);
      Reg::Duplicate(wTensor, ZERO, mask);
      Reg::Select(mulTmp, wTensor, mulTmp, mask3);
      Reg::Muls(tTensor, mulTmp, (1.0f / 1.1646782814350067249e14f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term  9
    {
      T kVal = 18 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Compare<T, CMPMODE::NE>(mask3, mulTmp, mulTmp, mask);
      Reg::Duplicate(wTensor, ZERO, mask);
      Reg::Select(mulTmp, wTensor, mulTmp, mask3);
      Reg::Muls(tTensor, mulTmp, (-1.0f / 4.5979787224074726105e15f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term 10
    {
      T kVal = 20 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Compare<T, CMPMODE::NE>(mask3, mulTmp, mulTmp, mask);
      Reg::Duplicate(wTensor, ZERO, mask);
      Reg::Select(mulTmp, wTensor, mulTmp, mask3);
      Reg::Muls(tTensor, mulTmp, (1.0f / 1.8152105401943546773e17f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
      kVal += ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
    }
    // Term 11
    {
      T kVal = 22 * ONE;
      Reg::Adds(kPlusX, src0Reg, kVal, mask);
      Reg::Mul(aTensor, aTensor, kPlusX, mask);
      Reg::Mul(bTensor, bTensor, lnTmp, mask);
      Reg::Mul(mulTmp, aTensor, bTensor, mask);
      Reg::Compare<T, CMPMODE::NE>(mask3, mulTmp, mulTmp, mask);
      Reg::Duplicate(wTensor, ZERO, mask);
      Reg::Select(mulTmp, wTensor, mulTmp, mask3);
      Reg::Muls(tTensor, mulTmp, (1.0f / -7.1661652561756670113e18f), mask);
      Reg::Add(dstReg, dstReg, tTensor, mask);
    }

    Reg::Compare<T, CMPMODE::NE>(mask2, dstReg, dstReg, mask);
    Reg::Select(dstReg, xMinusOne, dstReg, mask2);

    Reg::Duplicate(tTensor, ZERO, mask);
    Reg::Duplicate(mulTmp, ZERO, mask);
    Reg::Adds(tTensor, tTensor, ONE, mask);
    Reg::Div(mulTmp, tTensor, mulTmp, mask);

    Reg::Duplicate(nanTensor, (float &)F32_NAN, mask);

    // Recompute mask0/mask1 for boundary conditions (may have been overwritten)
    Reg::Compares<T, CMPMODE::LT>(mask0, src0Reg, ONE, mask);
    Reg::Compares<T, CMPMODE::EQ>(mask1, src0Reg, ONE, mask);
    // x < 1 -> NaN; x == 1 -> inf
    Reg::Select(dstReg, nanTensor, dstReg, mask0);
    Reg::Select(dstReg, mulTmp, dstReg, mask1);

    Reg::Compares<T, CMPMODE::LE>(mask2, src1Reg, ZERO, mask);
    Reg::Truncate<T, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(tTensor, src1Reg, mask);
    Reg::Compare<T, CMPMODE::EQ>(mask3, tTensor, src1Reg, mask);

    Reg::Duplicate(bTensor, ZERO, mask);
    Reg::Adds(bTensor, bTensor, ONE, mask);
    Reg::Duplicate(wTensor, ZERO, mask);

    Reg::Compares<T, CMPMODE::GT>(mask0, src0Reg, ONE, mask);
    Reg::Select(kPlusX, bTensor, wTensor, mask0);

    Reg::Truncate<T, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(tTensor, src0Reg, mask);
    Reg::Compare<T, CMPMODE::NE>(mask1, tTensor, src0Reg, mask);
    Reg::Select(aTensor, bTensor, wTensor, mask1);

    Reg::Select(lnTmp, bTensor, wTensor, mask2);
    Reg::Select(xMinusOne, bTensor, wTensor, mask3);

    Reg::Mul(tTensor, lnTmp, xMinusOne, mask);
    Reg::Mul(tTensor, tTensor, kPlusX, mask);
    Reg::Compares<T, CMPMODE::GT>(mask3, tTensor, ZERO, mask);
    Reg::Mul(tTensor, lnTmp, xMinusOne, mask);

    Reg::Select(dstReg, mulTmp, dstReg, mask3);

    Reg::Muls(tTensor, xMinusOne, NEG_ONE, mask);
    Reg::Adds(tTensor, tTensor, ONE, mask);
    Reg::Mul(tTensor, tTensor, lnTmp, mask);
    Reg::Mul(tTensor, tTensor, aTensor, mask);
    Reg::Mul(tTensor, tTensor, kPlusX, mask);
    Reg::Compares<T, CMPMODE::GT>(mask2, tTensor, ZERO, mask);
    Reg::Select(dstReg, nanTensor, dstReg, mask2);

    // Overflow detection: q < 0, q non-integer, x integer > 1
    // The series terms near k ≈ -q overflow float32, so the Euler-Maclaurin
    // result is wrong. The dominant term is 1/min(frac(q), 1-frac(q))^x.
    // If it overflows, the result is +inf (frac < 0.5 or x even) or -inf.
    // Compute floor(q) and frac(q)
    Reg::Truncate<T, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(tTensor, src1Reg, mask);
    Reg::Sub(aTensor, src1Reg, tTensor, mask);
    // aTensor = frac(q) ∈ [0, 1)

    // 1 - frac(q)
    Reg::Duplicate(bTensor, ONE, mask);
    Reg::Sub(bTensor, bTensor, aTensor, mask);
    // bTensor = 1 - frac(q)

    // min(frac, 1-frac) = |base| of dominant term
    Reg::Compare<T, CMPMODE::LT>(mask0, aTensor, bTensor, mask);
    Reg::Select(lnTmp, aTensor, bTensor, mask0);
    // lnTmp = min(frac, 1-frac)

    // Dominant term magnitude = exp(-x * ln|base|) = 1/|base|^x
    Reg::Log(lnTmp, lnTmp, mask);
    Reg::Mul(lnTmp, src0Reg, lnTmp, mask);
    Reg::Muls(lnTmp, lnTmp, NEG_ONE, mask);
    Reg::Exp(wLast, lnTmp, mask);
    // wLast = 1/|base|^x (possibly +inf)

    // Check overflow: |wLast| == inf (mulTmp = +inf from 1/0 above)
    Reg::Abs(wLast, wLast, mask);
    Reg::Compare<T, CMPMODE::EQ>(mask3, wLast, mulTmp, mask);
    // mask3: dominant term overflows

    // Build combined condition: q <= 0 AND q non-integer AND x integer AND x > 1 AND overflow
    Reg::Duplicate(bTensor, ZERO, mask);
    Reg::Adds(bTensor, bTensor, ONE, mask);
    Reg::Duplicate(wTensor, ZERO, mask);

    Reg::Compares<T, CMPMODE::LE>(mask0, src1Reg, ZERO, mask);
    Reg::Select(aTensor, bTensor, wTensor, mask0);
    // aTensor = 1 where q <= 0

    Reg::Truncate<T, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(tTensor, src1Reg, mask);
    Reg::Compare<T, CMPMODE::NE>(mask1, tTensor, src1Reg, mask);
    Reg::Select(kPlusX, bTensor, wTensor, mask1);
    // kPlusX = 1 where q non-integer

    Reg::Truncate<T, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(tTensor, src0Reg, mask);
    Reg::Compare<T, CMPMODE::EQ>(mask2, tTensor, src0Reg, mask);
    Reg::Select(xMinusOne, bTensor, wTensor, mask2);
    // xMinusOne = 1 where x integer

    Reg::Compares<T, CMPMODE::GT>(mask0, src0Reg, ONE, mask);
    Reg::Select(bLast, bTensor, wTensor, mask0);
    // bLast = 1 where x > 1

    Reg::Select(wLast, bTensor, wTensor, mask3);
    // wLast = 1 where overflow

    Reg::Mul(aTensor, aTensor, kPlusX, mask);
    Reg::Mul(aTensor, aTensor, xMinusOne, mask);
    Reg::Mul(aTensor, aTensor, bLast, mask);
    Reg::Mul(aTensor, aTensor, wLast, mask);
    // aTensor = 1 where ALL conditions met
    // Don't apply overflow detection where:
    // The series already produced a finite result (correct value from
    // skipping the ±2^x cancellation pair), OR
    // |q| < 1 AND result is NaN AND NOT (frac==0.5 AND x odd).
    // For |q| < 1 with frac==0.5 AND x odd, the first term pow(q,-x)
    // overflows to ±inf, so the overflow detection should fire (±inf).
    // Check if result is finite (not NaN, not inf)
    Reg::Compare<T, CMPMODE::NE>(mask2, dstReg, dstReg, mask);
    Reg::Abs(tTensor, dstReg, mask);
    Reg::Compare<T, CMPMODE::EQ>(mask3, tTensor, mulTmp, mask);
    Reg::Select(kPlusX, wTensor, bTensor, mask2);
    Reg::Select(lnTmp, wTensor, kPlusX, mask3);
    Reg::Compares<T, CMPMODE::GT>(mask3, lnTmp, ZERO, mask);
    Reg::Select(aTensor, wTensor, aTensor, mask3);
    // Check |q| < 1 AND NaN AND NOT (frac==0.5 AND x odd)
    Reg::Abs(tTensor, src1Reg, mask);
    Reg::Compares<T, CMPMODE::LT>(mask0, tTensor, ONE, mask);
    Reg::Truncate<T, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(tTensor, src1Reg, mask);
    Reg::Sub(tTensor, src1Reg, tTensor, mask);
    Reg::Compares<T, CMPMODE::EQ>(mask1, tTensor, HALF, mask);
    Reg::Select(lnTmp, bTensor, wTensor, mask1);
    Reg::Select(kPlusX, bTensor, wTensor, maskSignFlip);
    Reg::Mul(lnTmp, lnTmp, kPlusX, mask);
    Reg::Compares<T, CMPMODE::GT>(mask1, lnTmp, ZERO, mask);
    Reg::Select(lnTmp, bTensor, wTensor, mask0);
    Reg::Select(kPlusX, bTensor, wTensor, mask2);
    Reg::Mul(lnTmp, lnTmp, kPlusX, mask);
    Reg::Select(kPlusX, wTensor, bTensor, mask1);
    Reg::Mul(lnTmp, lnTmp, kPlusX, mask);
    Reg::Compares<T, CMPMODE::GT>(mask3, lnTmp, ZERO, mask);
    Reg::Select(aTensor, wTensor, aTensor, mask3);

    Reg::Compares<T, CMPMODE::GT>(mask0, aTensor, ZERO, mask);
    // mask0: overflow case

    // Step 1: set +inf where overflow
    Reg::Select(dstReg, mulTmp, dstReg, mask0);

    // Step 2: determine negative sign (frac >= 0.5 AND x odd)
    Reg::Truncate<T, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(tTensor, src1Reg, mask);
    Reg::Sub(aTensor, src1Reg, tTensor, mask);
    // aTensor = frac(q)

    Reg::Duplicate(bTensor, HALF, mask);
    Reg::Compare<T, CMPMODE::LT>(mask1, aTensor, bTensor, mask);
    // mask1: frac < 0.5

    // neg_cond = (frac >= 0.5) AND overflow_case AND (x is odd)
    // maskSignFlip is true when x is an odd integer (sign needs flipping)
    Reg::Duplicate(bTensor, ZERO, mask);
    Reg::Adds(bTensor, bTensor, ONE, mask);
    Reg::Duplicate(wTensor, ZERO, mask);
    Reg::Select(aTensor, wTensor, bTensor, mask1);
    // aTensor = 0 where frac < 0.5, 1 where frac >= 0.5
    Reg::Select(aTensor, aTensor, wTensor, mask0);
    // aTensor = 1 where frac >= 0.5 AND overflow case
    Reg::Select(kPlusX, bTensor, wTensor, maskSignFlip);
    // kPlusX = 1 where maskSignFlip (x odd), 0 where not (x even or non-integer)
    Reg::Mul(aTensor, aTensor, kPlusX, mask);
    // aTensor = 1 where neg_cond met

    Reg::Compares<T, CMPMODE::GT>(mask1, aTensor, ZERO, mask);

    // Create -inf and override where negative
    Reg::Muls(xMinusOne, mulTmp, NEG_ONE, mask);
    Reg::Select(dstReg, xMinusOne, dstReg, mask1);

    Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
  }
}

template <typename T>
__aicore__ inline void ZetaExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src0, const LocalTensor<T> &src1,
                                  const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert((std::is_same_v<T, float>), "Zeta only supports float on current device!");
  ZetaImplVF<T>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src0.GetPhyAddr(), (__ubuf__ T *)src1.GetPhyAddr(),
                calCount);
}
}  // namespace AscendC

#endif  // __ASCENDC_API_REGBASE_ZETA_H__
