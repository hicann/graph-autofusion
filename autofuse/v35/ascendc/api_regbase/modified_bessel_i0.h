/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_MODIFIED_BESSEL_I0_H__
#define __ASCENDC_API_REGBASE_MODIFIED_BESSEL_I0_H__

constexpr float I0_A[30] = {
    -4.41534164647933937950e-18, +3.33079451882223809783e-17, -2.43127984654795469359e-16, +1.71539128555513303061e-15,
    -1.16853328779934516808e-14, +7.67618549860493561688e-14, -4.85644678311192946090e-13, +2.95505266312963983461e-12,
    -1.72682629144155570723e-11, +9.67580903537323691224e-11, -5.18979560163526290666e-10, +2.65982372468238665035e-09,
    -1.30002500998624804212e-08, +6.04699502254191894932e-08, -2.67079385394061173391e-07, +1.11738753912010371815e-06,
    -4.41673835845875056359e-06, +1.64484480707288970893e-05, -5.75419501008210370398e-05, +1.88502885095841655729e-04,
    -5.76375574538582365885e-04, +1.63947561694133579842e-03, -4.32430999505057594430e-03, +1.05464603945949983183e-02,
    -2.37374148058994688156e-02, +4.93052842396707084878e-02, -9.49010970480476444210e-02, +1.71620901522208775349e-01,
    -3.04682672343198398683e-01, +6.76795274409476084995e-01,
};

constexpr float I0_B[25] = {
    -7.23318048787475395456e-18, -4.83050448594418207126e-18, +4.46562142029675999901e-17, +3.46122286769746109310e-17,
    -2.82762398051658348494e-16, -3.42548561967721913462e-16, +1.77256013305652638360e-15, +3.81168066935262242075e-15,
    -9.55484669882830764870e-15, -4.15056934728722208663e-14, +1.54008621752140982691e-14, +3.85277838274214270114e-13,
    +7.18012445138366623367e-13, -1.79417853150680611778e-12, -1.32158118404477131188e-11, -3.14991652796324136454e-11,
    +1.18891471078464383424e-11, +4.94060238822496958910e-10, +3.39623202570838634515e-09, +2.26666899049817806459e-08,
    +2.04891858946906374183e-07, +2.89137052083475648297e-06, +6.88975834691682398426e-05, +3.36911647825569408990e-03,
    +8.04490411014108831608e-01,
};

template <typename T>
__simd_callee__ inline void ModifiedBesselI0FactorSmallCompute(AscendC::Reg::RegTensor<T> &absXReg,
                                                               AscendC::Reg::RegTensor<T> &smallDstReg,
                                                               AscendC::Reg::MaskReg &branchMask) {
  AscendC::Reg::RegTensor<T> factorReg;
  AscendC::Reg::Exp(factorReg, absXReg, branchMask);
  AscendC::Reg::Mul(smallDstReg, smallDstReg, factorReg, branchMask);
}

template <typename T>
__simd_callee__ inline void ModifiedBesselI0FactorBigCompute(AscendC::Reg::RegTensor<T> &absXReg,
                                                             AscendC::Reg::RegTensor<T> &bigDstReg,
                                                             AscendC::Reg::MaskReg &branchMask) {
  AscendC::Reg::RegTensor<T> factorReg;
  AscendC::Reg::Exp(factorReg, absXReg, branchMask);
  AscendC::Reg::Mul(bigDstReg, bigDstReg, factorReg, branchMask);
  AscendC::Reg::Sqrt(factorReg, absXReg, branchMask);
  AscendC::Reg::Div(bigDstReg, bigDstReg, factorReg, branchMask);
}

template <typename T, uint32_t currentIteration, uint32_t endIteration, uint32_t sliceNum>
__simd_callee__ inline void ModifiedBesselI0SmallSliceCompute(AscendC::Reg::RegTensor<T> &absXReg,
                                                              AscendC::Reg::RegTensor<T> &smallDstReg,
                                                              AscendC::Reg::MaskReg &branchMask, __ubuf__ T *dst,
                                                              __ubuf__ T *tmpBuf, uint32_t offSet, uint32_t tensorLen) {
  AscendC::Reg::RegTensor<T> pReg, qReg, xFactorReg, constReg, iterReg;
  ModifiedBesselImportData<T, sliceNum, I0_A>(pReg, qReg, constReg, branchMask, dst, tmpBuf, offSet, tensorLen);

  // x_factor = |x|/2 - 2
  AscendC::Reg::Muls(xFactorReg, absXReg, (T)0.5, branchMask);
  AscendC::Reg::Adds(xFactorReg, xFactorReg, (T)(-2.0), branchMask);

  mainIter<T, currentIteration, endIteration, I0_A>(pReg, qReg, constReg, xFactorReg, iterReg, branchMask);
  if constexpr (sliceNum == 1) {
    // result_small *= exp(|x|)
    AscendC::Reg::Sub(smallDstReg, constReg, pReg, branchMask);
    AscendC::Reg::Muls(smallDstReg, smallDstReg, (T)0.5, branchMask);
    ModifiedBesselI0FactorSmallCompute<T>(absXReg, smallDstReg, branchMask);
  } else {
    ModifiedBesselExportData<T>(pReg, qReg, constReg, branchMask, dst, tmpBuf, offSet, tensorLen);
  }
}

template <typename T, uint32_t currentIteration, uint32_t endIteration, uint32_t sliceNum>
__simd_callee__ inline void ModifiedBesselI0BigSliceCompute(AscendC::Reg::RegTensor<T> &absXReg,
                                                            AscendC::Reg::RegTensor<T> &bigDstReg,
                                                            AscendC::Reg::MaskReg &branchMask, __ubuf__ T *dst,
                                                            __ubuf__ T *tmpBuf, uint32_t offSet, uint32_t tensorLen) {
  AscendC::Reg::RegTensor<T> pReg, qReg, xFactorReg, constReg, iterReg;
  ModifiedBesselImportData<T, sliceNum, I0_B>(pReg, qReg, constReg, branchMask, dst, tmpBuf, offSet, tensorLen);

  // x_factor = 32/|x| - 2
  AscendC::Reg::Duplicate(xFactorReg, (T)32.0, branchMask);
  AscendC::Reg::Div(xFactorReg, xFactorReg, absXReg, branchMask);
  AscendC::Reg::Adds(xFactorReg, xFactorReg, (T)(-2.0), branchMask);

  mainIter<T, currentIteration, endIteration, I0_B>(pReg, qReg, constReg, xFactorReg, iterReg, branchMask);
  if constexpr (sliceNum == 1) {
    // result_small *= exp(|x|)
    AscendC::Reg::Sub(bigDstReg, constReg, pReg, branchMask);
    AscendC::Reg::Muls(bigDstReg, bigDstReg, (T)0.5, branchMask);
    ModifiedBesselI0FactorBigCompute<T>(absXReg, bigDstReg, branchMask);
  } else {
    ModifiedBesselExportData<T>(pReg, qReg, constReg, branchMask, dst, tmpBuf, offSet, tensorLen);
  }
}

template <typename T>
__simd_vf__ inline void ModifiedBesselI0ImplVF(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *tmpBuf,
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
    ModifiedBesselI0SmallSliceCompute<T, 1, 14, 0>(absXReg, smallDstReg, branchMask, dst, tmpBuf, i * vlSize,
                                                   tensorLen);

    // ===== Large branch: |x| > 8.0 =====
    AscendC::Reg::Compares<T, CMPMODE::GT>(branchMask, absXReg, (T)8.0, mask);
    ModifiedBesselI0BigSliceCompute<T, 1, 12, 0>(absXReg, bigDstReg, branchMask, dst, tmpBuf, i * vlSize, tensorLen);
  }

  Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

  for (uint16_t i = 0U; i < repeatTime; ++i) {
    mask = AscendC::Reg::UpdateMask<T>(calCount2);
    AscendC::Reg::LoadAlign(srcReg, src + i * vlSize);
    AscendC::Reg::Abs(absXReg, srcReg, mask);

    // ===== Small branch: |x| <= 8.0 =====
    AscendC::Reg::Compares<T, CMPMODE::LE>(branchMask, absXReg, (T)8.0, mask);
    ModifiedBesselI0SmallSliceCompute<T, 15, 29, 1>(absXReg, smallDstReg, branchMask, dst, tmpBuf, i * vlSize,
                                                    tensorLen);

    // ===== Large branch: |x| > 8.0 =====
    AscendC::Reg::Compares<T, CMPMODE::GT>(branchMask, absXReg, (T)8.0, mask);
    ModifiedBesselI0BigSliceCompute<T, 13, 24, 1>(absXReg, bigDstReg, branchMask, dst, tmpBuf, i * vlSize, tensorLen);

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
__aicore__ inline void ModifiedBesselI0Extend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                              const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount) {
  static_assert(SupportType<T, float>(), "Current data type is  not supported on current device!");
  auto tmpUB = sharedTmpBuffer.ReinterpretCast<T>();
  ModifiedBesselI0ImplVF<T>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(),
                            (__ubuf__ T *)tmpUB.GetPhyAddr(), calCount);
}

#endif  // __ASCENDC_API_REGBASE_MODIFIED_BESSEL_I0_H__
