/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_MODIFIED_BESSEL_I1_H__
#define __ASCENDC_API_REGBASE_MODIFIED_BESSEL_I1_H__

constexpr float I1_A[29] = {
    +2.77791411276104639959e-18, -2.11142121435816608115e-17, +1.55363195773620046921e-16, -1.10559694773538630805e-15,
    +7.60068429473540693410e-15, -5.04218550472791168711e-14, +3.22379336594557470981e-13, -1.98397439776494371520e-12,
    +1.17361862988909016308e-11, -6.66348972350202774223e-11, +3.62559028155211703701e-10, -1.88724975172282928790e-09,
    +9.38153738649577178388e-09, -4.44505912879632808065e-08, +2.00329475355213526229e-07, -8.56872026469545474066e-07,
    +3.47025130813767847674e-06, -1.32731636560394358279e-05, +4.78156510755005422638e-05, -1.61760815825896745588e-04,
    +5.12285956168575772895e-04, -1.51357245063125314899e-03, +4.15642294431288815669e-03, -1.05640848946261981558e-02,
    +2.47264490306265168283e-02, -5.29459812080949914269e-02, +1.02643658689847095384e-01, -1.76416518357834055153e-01,
    +2.52587186443633654823e-01,
};

constexpr float I1_B[25] = {
    +7.51729631084210481353e-18, +4.41434832307170791151e-18, -4.65030536848935832153e-17, -3.20952592199342395980e-17,
    +2.96262899764595013876e-16, +3.30820231092092828324e-16, -1.88035477551078244854e-15, -3.81440307243700780478e-15,
    +1.04202769841288027642e-14, +4.27244001671195135429e-14, -2.10154184277266431302e-14, -4.08355111109219731823e-13,
    -7.19855177624590851209e-13, +2.03562854414708950722e-12, +1.41258074366137813316e-11, +3.25260358301548823856e-11,
    -1.89749581235054123450e-11, -5.58974346219658380687e-10, -3.83538038596423702205e-09, -2.63146884688951950684e-08,
    -2.51223623787020892529e-07, -3.88256480887769039346e-06, -1.10588938762623716291e-04, -9.76109749136146840777e-03,
    +7.78576235018280120474e-01,
};

template <typename T>
__simd_callee__ inline void ModifiedBesselI1FactorSmallCompute(AscendC::Reg::RegTensor<T> &absXReg,
                                                               AscendC::Reg::RegTensor<T> &srcReg,
                                                               AscendC::Reg::RegTensor<T> &smallDstReg,
                                                               AscendC::Reg::MaskReg &branchMask) {
  AscendC::Reg::RegTensor<T> factorReg;
  AscendC::Reg::Exp(factorReg, absXReg, branchMask);
  AscendC::Reg::Mul(smallDstReg, smallDstReg, srcReg, branchMask);
  AscendC::Reg::Mul(smallDstReg, smallDstReg, factorReg, branchMask);
}

template <typename T>
__simd_callee__ inline void ModifiedBesselI1FactorBigCompute(AscendC::Reg::RegTensor<T> &absXReg,
                                                             AscendC::Reg::RegTensor<T> &srcReg,
                                                             AscendC::Reg::RegTensor<T> &bigDstReg,
                                                             AscendC::Reg::MaskReg &branchMask) {
  AscendC::Reg::RegTensor<T> factorReg, signReg, negSignReg;
  AscendC::Reg::MaskReg signMask;
  AscendC::Reg::Exp(factorReg, absXReg, branchMask);
  AscendC::Reg::Mul(bigDstReg, bigDstReg, factorReg, branchMask);
  AscendC::Reg::Sqrt(factorReg, absXReg, branchMask);
  AscendC::Reg::Div(bigDstReg, bigDstReg, factorReg, branchMask);

  AscendC::Reg::Duplicate(signReg, (T)1.0, branchMask);
  AscendC::Reg::Duplicate(negSignReg, (T)(-1.0), branchMask);
  AscendC::Reg::Compares<T, CMPMODE::LT>(signMask, srcReg, (T)0.0, branchMask);
  AscendC::Reg::Select(signReg, negSignReg, signReg, signMask);

  AscendC::Reg::Mul(bigDstReg, bigDstReg, signReg, branchMask);
}

template <typename T, uint32_t currentIteration, uint32_t endIteration, uint32_t sliceNum>
__simd_callee__ inline void ModifiedBesselI1SmallSliceCompute(AscendC::Reg::RegTensor<T> &absXReg,
                                                              AscendC::Reg::RegTensor<T> &srcReg,
                                                              AscendC::Reg::RegTensor<T> &smallDstReg,
                                                              AscendC::Reg::MaskReg &branchMask, __ubuf__ T *dst,
                                                              __ubuf__ T *tmpBuf, uint32_t offSet, uint32_t tensorLen) {
  AscendC::Reg::RegTensor<T> pReg, qReg, xFactorReg, constReg, iterReg;
  ModifiedBesselImportData<T, sliceNum, I1_A>(pReg, qReg, constReg, branchMask, dst, tmpBuf, offSet, tensorLen);

  // x_factor = |x|/2 - 2
  AscendC::Reg::Muls(xFactorReg, absXReg, (T)0.5, branchMask);
  AscendC::Reg::Adds(xFactorReg, xFactorReg, (T)(-2.0), branchMask);

  mainIter<T, currentIteration, endIteration, I1_A>(pReg, qReg, constReg, xFactorReg, iterReg, branchMask);
  if constexpr (sliceNum == 1) {
    // result_small = 0.5 * (a - p) * x * exp(|x|)
    AscendC::Reg::Sub(smallDstReg, constReg, pReg, branchMask);
    AscendC::Reg::Muls(smallDstReg, smallDstReg, (T)0.5, branchMask);
    ModifiedBesselI1FactorSmallCompute<T>(absXReg, srcReg, smallDstReg, branchMask);
  } else {
    ModifiedBesselExportData<T>(pReg, qReg, constReg, branchMask, dst, tmpBuf, offSet, tensorLen);
  }
}

template <typename T, uint32_t currentIteration, uint32_t endIteration, uint32_t sliceNum>
__simd_callee__ inline void ModifiedBesselI1BigSliceCompute(AscendC::Reg::RegTensor<T> &absXReg,
                                                            AscendC::Reg::RegTensor<T> &srcReg,
                                                            AscendC::Reg::RegTensor<T> &bigDstReg,
                                                            AscendC::Reg::MaskReg &branchMask, __ubuf__ T *dst,
                                                            __ubuf__ T *tmpBuf, uint32_t offSet, uint32_t tensorLen) {
  AscendC::Reg::RegTensor<T> pReg, qReg, xFactorReg, constReg, iterReg;
  ModifiedBesselImportData<T, sliceNum, I1_B>(pReg, qReg, constReg, branchMask, dst, tmpBuf, offSet, tensorLen);

  // x_factor = 32/|x| - 2
  AscendC::Reg::Duplicate(xFactorReg, (T)32.0, branchMask);
  AscendC::Reg::Div(xFactorReg, xFactorReg, absXReg, branchMask);
  AscendC::Reg::Adds(xFactorReg, xFactorReg, (T)(-2.0), branchMask);

  mainIter<T, currentIteration, endIteration, I1_B>(pReg, qReg, constReg, xFactorReg, iterReg, branchMask);
  if constexpr (sliceNum == 1) {
    // result_big = exp(|x|) * 0.5 * (b - p) / sqrt(|x|)
    AscendC::Reg::Sub(bigDstReg, constReg, pReg, branchMask);
    AscendC::Reg::Muls(bigDstReg, bigDstReg, (T)0.5, branchMask);
    ModifiedBesselI1FactorBigCompute<T>(absXReg, srcReg, bigDstReg, branchMask);
  } else {
    ModifiedBesselExportData<T>(pReg, qReg, constReg, branchMask, dst, tmpBuf, offSet, tensorLen);
  }
}

template <typename T>
__simd_vf__ inline void ModifiedBesselI1ImplVF(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *tmpBuf,
                                               uint32_t calCount) {
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
    ModifiedBesselI1SmallSliceCompute<T, 1, 14, 0>(absXReg, srcReg, smallDstReg, branchMask, dst, tmpBuf, i * vlSize,
                                                   tensorLen);

    // ===== Large branch: |x| > 8.0 =====
    AscendC::Reg::Compares<T, CMPMODE::GT>(branchMask, absXReg, (T)8.0, mask);
    ModifiedBesselI1BigSliceCompute<T, 1, 12, 0>(absXReg, srcReg, bigDstReg, branchMask, dst, tmpBuf, i * vlSize,
                                                 tensorLen);
  }

  Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

  for (uint16_t i = 0U; i < repeatTime; ++i) {
    mask = AscendC::Reg::UpdateMask<T>(calCount2);
    AscendC::Reg::LoadAlign(srcReg, src + i * vlSize);
    AscendC::Reg::Abs(absXReg, srcReg, mask);

    // ===== Small branch: |x| <= 8.0 =====
    AscendC::Reg::Compares<T, CMPMODE::LE>(branchMask, absXReg, (T)8.0, mask);
    ModifiedBesselI1SmallSliceCompute<T, 15, 28, 1>(absXReg, srcReg, smallDstReg, branchMask, dst, tmpBuf, i * vlSize,
                                                    tensorLen);

    // ===== Large branch: |x| > 8.0 =====
    AscendC::Reg::Compares<T, CMPMODE::GT>(branchMask, absXReg, (T)8.0, mask);
    ModifiedBesselI1BigSliceCompute<T, 13, 24, 1>(absXReg, srcReg, bigDstReg, branchMask, dst, tmpBuf, i * vlSize,
                                                  tensorLen);

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
__aicore__ inline void ModifiedBesselI1Extend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                              const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount) {
  static_assert(SupportType<T, float>(), "Current data type is  not supported on current device!");
  auto tmpUB = sharedTmpBuffer.ReinterpretCast<T>();
  ModifiedBesselI1ImplVF<T>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(),
                            (__ubuf__ T *)tmpUB.GetPhyAddr(), calCount);
}

#endif  // __ASCENDC_API_REGBASE_MODIFIED_BESSEL_I1_H__
