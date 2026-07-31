/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_I1E_H__
#define __ASCENDC_API_REGBASE_I1E_H__

#include "modified_bessel_i1.h"

// Scaled small-branch factor: i1e removes exp(|x|), keeps x factor (odd function)
template <typename T>
__simd_callee__ inline void I1eFactorSmallCompute(AscendC::Reg::RegTensor<T> &absXReg,
                                                  AscendC::Reg::RegTensor<T> &srcReg,
                                                  AscendC::Reg::RegTensor<T> &smallDstReg,
                                                  AscendC::Reg::MaskReg &branchMask) {
  // i1e small: result = 0.5 * (a - p) * x, no exp(|x|)
  AscendC::Reg::Mul(smallDstReg, smallDstReg, srcReg, branchMask);
}

// Scaled big-branch factor: i1e keeps 1/sqrt(|x|) * sign(x), removes exp(|x|)
template <typename T>
__simd_callee__ inline void I1eFactorBigCompute(AscendC::Reg::RegTensor<T> &absXReg, AscendC::Reg::RegTensor<T> &srcReg,
                                                AscendC::Reg::RegTensor<T> &bigDstReg,
                                                AscendC::Reg::MaskReg &branchMask) {
  AscendC::Reg::RegTensor<T> factorReg;
  AscendC::Reg::Sqrt(factorReg, absXReg, branchMask);
  AscendC::Reg::Div(bigDstReg, bigDstReg, factorReg, branchMask);

  // sign(x): odd function, x < 0 → -1
  AscendC::Reg::RegTensor<T> signReg, negSignReg;
  AscendC::Reg::MaskReg signMask;
  AscendC::Reg::Duplicate(signReg, (T)1.0, branchMask);
  AscendC::Reg::Duplicate(negSignReg, (T)(-1.0), branchMask);
  AscendC::Reg::Compares<T, CMPMODE::LT>(signMask, srcReg, (T)0.0, branchMask);
  AscendC::Reg::Select(signReg, negSignReg, signReg, signMask);
  AscendC::Reg::Mul(bigDstReg, bigDstReg, signReg, branchMask);
}

template <typename T, uint32_t currentIteration, uint32_t endIteration, uint32_t sliceNum>
__simd_callee__ inline void I1eSmallSliceCompute(AscendC::Reg::RegTensor<T> &absXReg,
                                                 AscendC::Reg::RegTensor<T> &srcReg,
                                                 AscendC::Reg::RegTensor<T> &smallDstReg,
                                                 AscendC::Reg::MaskReg &branchMask, __ubuf__ T *dst, __ubuf__ T *tmpBuf,
                                                 uint32_t offSet, uint32_t tensorLen) {
  AscendC::Reg::RegTensor<T> pReg, qReg, xFactorReg, constReg, iterReg;
  ModifiedBesselImportData<T, sliceNum, I1_A>(pReg, qReg, constReg, branchMask, dst, tmpBuf, offSet, tensorLen);

  // x_factor = |x|/2 - 2
  AscendC::Reg::Muls(xFactorReg, absXReg, (T)0.5, branchMask);
  AscendC::Reg::Adds(xFactorReg, xFactorReg, (T)(-2.0), branchMask);

  mainIter<T, currentIteration, endIteration, I1_A>(pReg, qReg, constReg, xFactorReg, iterReg, branchMask);
  if constexpr (sliceNum == 1) {
    // result_small = 0.5 * (a - p) * x, no exp(|x|)
    AscendC::Reg::Sub(smallDstReg, constReg, pReg, branchMask);
    AscendC::Reg::Muls(smallDstReg, smallDstReg, (T)0.5, branchMask);
    I1eFactorSmallCompute<T>(absXReg, srcReg, smallDstReg, branchMask);
  } else {
    ModifiedBesselExportData<T>(pReg, qReg, constReg, branchMask, dst, tmpBuf, offSet, tensorLen);
  }
}

template <typename T, uint32_t currentIteration, uint32_t endIteration, uint32_t sliceNum>
__simd_callee__ inline void I1eBigSliceCompute(AscendC::Reg::RegTensor<T> &absXReg, AscendC::Reg::RegTensor<T> &srcReg,
                                               AscendC::Reg::RegTensor<T> &bigDstReg, AscendC::Reg::MaskReg &branchMask,
                                               __ubuf__ T *dst, __ubuf__ T *tmpBuf, uint32_t offSet,
                                               uint32_t tensorLen) {
  AscendC::Reg::RegTensor<T> pReg, qReg, xFactorReg, constReg, iterReg;
  ModifiedBesselImportData<T, sliceNum, I1_B>(pReg, qReg, constReg, branchMask, dst, tmpBuf, offSet, tensorLen);

  // x_factor = 32/|x| - 2
  AscendC::Reg::Duplicate(xFactorReg, (T)32.0, branchMask);
  AscendC::Reg::Div(xFactorReg, xFactorReg, absXReg, branchMask);
  AscendC::Reg::Adds(xFactorReg, xFactorReg, (T)(-2.0), branchMask);

  mainIter<T, currentIteration, endIteration, I1_B>(pReg, qReg, constReg, xFactorReg, iterReg, branchMask);
  if constexpr (sliceNum == 1) {
    // result_big = 0.5 * (b - p) / sqrt(|x|) * sign(x), no exp(|x|)
    AscendC::Reg::Sub(bigDstReg, constReg, pReg, branchMask);
    AscendC::Reg::Muls(bigDstReg, bigDstReg, (T)0.5, branchMask);
    I1eFactorBigCompute<T>(absXReg, srcReg, bigDstReg, branchMask);
  } else {
    ModifiedBesselExportData<T>(pReg, qReg, constReg, branchMask, dst, tmpBuf, offSet, tensorLen);
  }
}

template <typename T>
__simd_vf__ inline void I1eImplVF(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *tmpBuf, uint32_t calCount) {
  uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeatTime = static_cast<uint16_t>(AscendC::CeilDivision(calCount, vlSize));
  uint32_t tensorLen = repeatTime * vlSize;
  uint32_t calCount2 = calCount;

  AscendC::Reg::RegTensor<T> srcReg, absXReg, smallDstReg, bigDstReg, dstReg, nanReg;
  AscendC::Reg::MaskReg mask, branchMask;

  for (uint16_t i = 0U; i < repeatTime; ++i) {
    mask = AscendC::Reg::UpdateMask<T>(calCount);
    AscendC::Reg::LoadAlign(srcReg, src + i * vlSize);
    AscendC::Reg::Abs(absXReg, srcReg, mask);

    // ===== Small branch: |x| <= 8.0 =====
    AscendC::Reg::Compares<T, CMPMODE::LE>(branchMask, absXReg, (T)8.0, mask);
    I1eSmallSliceCompute<T, 1, 14, 0>(absXReg, srcReg, smallDstReg, branchMask, dst, tmpBuf, i * vlSize, tensorLen);

    // ===== Large branch: |x| > 8.0 =====
    AscendC::Reg::Compares<T, CMPMODE::GT>(branchMask, absXReg, (T)8.0, mask);
    I1eBigSliceCompute<T, 1, 12, 0>(absXReg, srcReg, bigDstReg, branchMask, dst, tmpBuf, i * vlSize, tensorLen);
  }

  Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

  for (uint16_t i = 0U; i < repeatTime; ++i) {
    mask = AscendC::Reg::UpdateMask<T>(calCount2);
    AscendC::Reg::LoadAlign(srcReg, src + i * vlSize);
    AscendC::Reg::Abs(absXReg, srcReg, mask);

    // ===== Small branch: |x| <= 8.0 =====
    AscendC::Reg::Compares<T, CMPMODE::LE>(branchMask, absXReg, (T)8.0, mask);
    I1eSmallSliceCompute<T, 15, 28, 1>(absXReg, srcReg, smallDstReg, branchMask, dst, tmpBuf, i * vlSize, tensorLen);

    // ===== Large branch: |x| > 8.0 =====
    AscendC::Reg::Compares<T, CMPMODE::GT>(branchMask, absXReg, (T)8.0, mask);
    I1eBigSliceCompute<T, 13, 24, 1>(absXReg, srcReg, bigDstReg, branchMask, dst, tmpBuf, i * vlSize, tensorLen);

    AscendC::Reg::Select(dstReg, bigDstReg, smallDstReg, branchMask);

    // handle nan input
    AscendC::Reg::Compare<T, CMPMODE::NE>(branchMask, srcReg, srcReg, mask);
    AscendC::Reg::Duplicate(nanReg, (float &)MODIFIED_BESSEL_FLOAT_NAN, mask);
    AscendC::Reg::Select(dstReg, nanReg, dstReg, branchMask);

    // Store output
    AscendC::Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
  }
}

template <typename T>
__aicore__ inline void I1eExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                 const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount) {
  static_assert(SupportType<T, float>(), "Current data type is  not supported on current device!");
  // Only for AI Vector Core.
  if ASCEND_IS_AIC {
    return;
  }
  I1eImplVF<T>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(),
               (__ubuf__ T *)sharedTmpBuffer.ReinterpretCast<T>().GetPhyAddr(), calCount);
}

#endif  // __ASCENDC_API_REGBASE_I1E_H__
