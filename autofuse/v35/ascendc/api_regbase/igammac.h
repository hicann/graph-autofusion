/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_IGAMMAC_H__
#define __ASCENDC_API_REGBASE_IGAMMAC_H__

namespace AscendC {
namespace IGammaCInternal {

template <typename T, bool isIgam>
__simd_vf__ inline void IgammaCImplPreJudge(__ubuf__ T *dstUb, __ubuf__ T *src0Ub, __ubuf__ T *src1Ub,
                                            __ubuf__ float *workUb, const uint32_t calCount) {
  Reg::MaskReg aMask, xMask, mask1, mask2, fullMask;
  Reg::RegTensor<T> src0Reg, src1Reg, dstReg;

  uint32_t sreg = calCount;
  fullMask = Reg::UpdateMask<float>(sreg);

  Reg::LoadAlign(src0Reg, src0Ub);
  Reg::LoadAlign(src1Reg, src1Ub);
  Reg::LoadAlign(dstReg, dstUb);

  // x < 0 || a < 0
  Reg::Compares<T, CMPMODE::LT>(aMask, src0Reg, 0.0f, fullMask);
  Reg::Compares<T, CMPMODE::LT>(xMask, src1Reg, 0.0f, fullMask);
  Reg::Or(mask1, aMask, xMask, fullMask);
  Reg::Duplicate<T, Reg::MaskMergeMode::MERGING>(dstReg, (float &)F32_NAN, mask1);

  // a == 0
  Reg::Not(mask1, mask1, fullMask);
  Reg::Compares<T, CMPMODE::EQ>(aMask, src0Reg, 0.0f, fullMask);
  Reg::Compares<T, CMPMODE::GT>(xMask, src1Reg, 0.0f, fullMask);
  //      x > 0
  Reg::And(mask2, mask1, aMask, fullMask);
  Reg::And(mask2, mask2, xMask, fullMask);
  Reg::Duplicate<T, Reg::MaskMergeMode::MERGING>(dstReg, isIgam ? 1.0f : 0.0f, mask2);
  //      x <= 0
  Reg::And(mask2, mask1, aMask, fullMask);
  Reg::Not(xMask, xMask, fullMask);
  Reg::And(mask2, mask2, xMask, fullMask);
  Reg::Duplicate<T, Reg::MaskMergeMode::MERGING>(dstReg, (float &)F32_NAN, mask2);

  // x == 0
  Reg::Not(aMask, aMask, fullMask);
  Reg::And(mask1, mask1, aMask, fullMask);
  Reg::Compares<T, CMPMODE::EQ>(xMask, src1Reg, 0.0f, fullMask);
  Reg::And(mask2, mask1, xMask, fullMask);
  Reg::Duplicate<T, Reg::MaskMergeMode::MERGING>(dstReg, isIgam ? 0.0f : 1.0f, mask2);

  // isinf(a)
  Reg::Not(xMask, xMask, fullMask);
  Reg::And(mask1, mask1, xMask, fullMask);
  Reg::Compares<T, CMPMODE::EQ>(aMask, src0Reg, (float &)F32_INF, fullMask);
  Reg::Compares<T, CMPMODE::EQ>(xMask, src1Reg, (float &)F32_INF, fullMask);
  //      isInf(x)
  Reg::And(mask2, mask1, aMask, fullMask);
  Reg::And(mask2, mask2, xMask, fullMask);
  Reg::Duplicate<T, Reg::MaskMergeMode::MERGING>(dstReg, (float &)F32_NAN, mask2);
  //      ! isinf(x)
  Reg::And(mask2, mask1, aMask, fullMask);
  Reg::Not(xMask, xMask, fullMask);
  Reg::And(mask2, mask2, xMask, fullMask);
  Reg::Duplicate<T, Reg::MaskMergeMode::MERGING>(dstReg, isIgam ? 0.0f : 1.0f, mask2);

  // isinf(x)
  Reg::Not(aMask, aMask, fullMask);
  Reg::And(mask1, mask1, aMask, fullMask);
  Reg::Compares<T, CMPMODE::EQ>(xMask, src1Reg, (float &)F32_INF, fullMask);
  Reg::And(mask2, mask1, xMask, fullMask);
  Reg::Duplicate<T, Reg::MaskMergeMode::MERGING>(dstReg, isIgam ? 1.0f : 0.0f, mask2);

  // store dst
  Reg::StoreAlign(dstUb, dstReg, fullMask);

  // store mask
  Reg::Not(xMask, xMask, fullMask);
  Reg::And(mask1, mask1, xMask, fullMask);
  Reg::StoreAlign(workUb + 256, mask1);
}

template <typename T>
__simd_vf__ inline void IgammaCImplAsymptoticMask(__ubuf__ T *src0Ub, __ubuf__ T *src1Ub, __ubuf__ float *workUb) {
  constexpr float small = 20.0f;
  constexpr float large = 200.0f;
  constexpr float smallratio = 0.3f;
  constexpr float largeratio = 4.5f;

  Reg::RegTensor<T> src0Reg, src1Reg, absxmaReg, tmpReg1, tmpReg2;
  Reg::MaskReg mask, mask1, mask2;

  Reg::LoadAlign(src0Reg, src0Ub);
  Reg::LoadAlign(src1Reg, src1Ub);
  Reg::LoadAlign(mask, workUb + 256);

  Reg::Sub(absxmaReg, src0Reg, src1Reg, mask);
  Reg::Abs(absxmaReg, absxmaReg, mask);
  Reg::Div(absxmaReg, absxmaReg, src0Reg, mask);

  // a > small && a < large && absxma < smallratio
  Reg::Compares<T, CMPMODE::GT>(mask1, src0Reg, small, mask);
  Reg::Compares<T, CMPMODE::LT>(mask1, src0Reg, large, mask1);
  Reg::Compares<T, CMPMODE::LT>(mask1, absxmaReg, smallratio, mask1);

  // a > large &&  absxma < largeratio / sqrt(a)
  Reg::Compares<T, CMPMODE::GT>(mask2, src0Reg, large, mask);
  Reg::Duplicate(tmpReg1, largeratio, mask);
  Reg::Sqrt(tmpReg2, src0Reg, mask);
  Reg::Div(tmpReg1, tmpReg1, tmpReg2, mask);
  Reg::Compare<T, CMPMODE::LT>(mask2, absxmaReg, tmpReg1, mask2);

  // store mask
  Reg::Or(mask1, mask1, mask2, mask);
  Reg::StoreAlign(workUb + 264, mask1);  // asymptotic 中使用的mask
  Reg::Not(mask1, mask1, mask);
  Reg::StoreAlign(workUb + 256, mask1);  // asymptotic 之后使用的起始mask
}

template <typename T, bool isIgam>
__aicore__ inline void IgammaCImplAsymptotic(__ubuf__ T *dstUb, __ubuf__ T *src0Ub, __ubuf__ T *src1Ub,
                                             __ubuf__ float *workUb) {
  /*
   mask  8 * sizeof(float)   workUB + 264
   eta   64 * sizeof(float)  workUB + 272
   correctionTerms 64 * sizeof(float)  workUB + 336
   afrac 64 * sizeof(float) workUB + 400
  */

  // 计算主项
  IgammacAsymptoticMainTerms<T, isIgam>(dstUb, src0Ub, src1Ub, workUb);
  AscendC::PipeBarrier<PIPE_V>();
  // 计算修正项
  IgammacAsymptoticCorrectionTerms<T, 0, 5>(src0Ub, workUb);
  AscendC::PipeBarrier<PIPE_V>();
  IgammacAsymptoticCorrectionTerms<T, 6, 11>(src0Ub, workUb);
  AscendC::PipeBarrier<PIPE_V>();
  IgammacAsymptoticCorrectionTerms<T, 12, 17>(src0Ub, workUb);
  AscendC::PipeBarrier<PIPE_V>();
  IgammacAsymptoticCorrectionTerms<T, 17, 22>(src0Ub, workUb);
  AscendC::PipeBarrier<PIPE_V>();
  IgammacAsymptoticCorrectionTerms<T, 22, 24>(src0Ub, workUb);
  AscendC::PipeBarrier<PIPE_V>();
  // 最终整合
  IgammacAsymptoticFinalProcess<T, isIgam>(dstUb, src0Ub, workUb);
}

template <typename T>
__simd_vf__ inline void IgammaCImplXgt1p1(__ubuf__ T *dstUb, __ubuf__ T *src0Ub, __ubuf__ T *src1Ub,
                                          __ubuf__ float *workUb) {
  Reg::RegTensor<T> src0Reg, src1Reg, dstReg, lgammaReg, powReg, tmpReg;
  Reg::MaskReg mask, mask1, mask2;

  Reg::LoadAlign(src0Reg, src0Ub);
  Reg::LoadAlign(src1Reg, src1Ub);
  Reg::LoadAlign(dstReg, dstUb);
  Reg::LoadAlign(lgammaReg, workUb);
  Reg::LoadAlign(powReg, workUb + 128);
  Reg::LoadAlign(mask, workUb + 256);

  // x > 1.1
  Reg::Compares<T, CMPMODE::GT>(mask1, src1Reg, 1.1f, mask);
  //    a > x
  Reg::Compare<T, CMPMODE::GT>(mask2, src0Reg, src1Reg, mask1);
  Igammac_helper_series_float(tmpReg, src0Reg, src1Reg, lgammaReg, powReg, mask2);  // dst用tmp防止merging问题
  Reg::Muls(tmpReg, tmpReg, -1.0f, mask2);
  Reg::Adds(tmpReg, tmpReg, 1.0f, mask2);
  Reg::Copy(dstReg, tmpReg, mask2);
  //    a <= x
  Reg::Not(mask2, mask2, mask1);
  Igammac_helper_continued_fraction_float(tmpReg, src0Reg, src1Reg, lgammaReg, powReg,
                                          mask2);  // dst用tmp防止merging问题
  Reg::Copy(dstReg, tmpReg, mask2);

  // store dst
  Reg::StoreAlign(dstUb, dstReg, mask1);

  // store mask
  Reg::Not(mask1, mask1, mask);
  Reg::StoreAlign(workUb + 256, mask1);
}

template <typename T>
__simd_vf__ inline void IgammaCImplXle0p5(__ubuf__ T *dstUb, __ubuf__ T *src0Ub, __ubuf__ T *src1Ub,
                                          __ubuf__ float *workUb) {
  Reg::RegTensor<T> src0Reg, src1Reg, dstReg, lgammaReg, powReg, tmpReg1, tmpReg2;
  Reg::MaskReg mask, mask1, mask2;

  Reg::LoadAlign(src0Reg, src0Ub);
  Reg::LoadAlign(src1Reg, src1Ub);
  Reg::LoadAlign(dstReg, dstUb);
  Reg::LoadAlign(lgammaReg, workUb);
  Reg::LoadAlign(powReg, workUb + 128);
  Reg::LoadAlign(mask, workUb + 256);

  // x <= 0.5f
  Reg::Compares<T, CMPMODE::LE>(mask1, src1Reg, 0.5f, mask);
  //      a > -0.4 / log(x)
  Reg::Duplicate(tmpReg1, -0.4f, mask1);
  Reg::Log(tmpReg2, src1Reg, mask1);
  Reg::Div(tmpReg1, tmpReg1, tmpReg2, mask1);
  Reg::Compare<T, CMPMODE::LT>(mask2, tmpReg1, src0Reg, mask1);
  Reg::Duplicate(tmpReg1, 1.0f, mask2);
  Igammac_helper_series_float(tmpReg2, src0Reg, src1Reg, lgammaReg, powReg, mask2);
  Reg::Sub(tmpReg1, tmpReg1, tmpReg2, mask2);
  Reg::Copy(dstReg, tmpReg1, mask2);
  //      a <= -0.4 / log(x)
  Reg::Not(mask2, mask2, mask1);
  Igammac_helper_series_complement_float(tmpReg1, src0Reg, src1Reg, lgammaReg, mask2);
  Reg::Copy(dstReg, tmpReg1, mask2);

  // store dst
  Reg::StoreAlign(dstUb, dstReg, mask1);

  // store mask
  Reg::Not(mask1, mask1, mask);
  Reg::StoreAlign(workUb + 256, mask1);
}

template <typename T>
__simd_vf__ inline void IgammaCImplXother(__ubuf__ T *dstUb, __ubuf__ T *src0Ub, __ubuf__ T *src1Ub,
                                          __ubuf__ float *workUb) {
  Reg::MaskReg mask, mask2;
  Reg::RegTensor<T> src0Reg, src1Reg, dstReg, lgammaReg, powReg, tmpReg1, tmpReg2;

  Reg::LoadAlign(src0Reg, src0Ub);
  Reg::LoadAlign(src1Reg, src1Ub);
  Reg::LoadAlign(dstReg, dstUb);
  Reg::LoadAlign(lgammaReg, workUb);
  Reg::LoadAlign(powReg, workUb + 128);
  Reg::LoadAlign(mask, workUb + 256);

  // a > 1.1f * x
  Reg::Muls(tmpReg1, src1Reg, 1.1f, mask);
  Reg::Compare<T, CMPMODE::GT>(mask2, src0Reg, tmpReg1, mask);
  Reg::Duplicate(tmpReg1, 1.0f, mask2);
  Igammac_helper_series_float(tmpReg2, src0Reg, src1Reg, lgammaReg, powReg, mask2);
  Reg::Sub(tmpReg1, tmpReg1, tmpReg2, mask2);
  Reg::Copy(dstReg, tmpReg1, mask2);

  // a <= 1.1 * x
  Reg::Not(mask2, mask2, mask);
  Igammac_helper_series_complement_float(tmpReg1, src0Reg, src1Reg, lgammaReg, mask2);
  Reg::Copy(dstReg, tmpReg1, mask2);

  // store dst
  Reg::StoreAlign(dstUb, dstReg, mask);
}

}  // namespace IGammaCInternal

template <typename T>
__aicore__ inline void IgammacExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src0, const LocalTensor<T> &src1,
                                     const LocalTensor<uint8_t> &tmp, const uint32_t calCount) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert(SupportType<T, float>(), "current data type is not supported on current device!");
  constexpr float lanczos_g = 6.024680040776729583740234375;

  constexpr uint32_t stride = GetVecLen() / sizeof(T);
  uint16_t repeatTime = (calCount + stride - 1) / stride;

  uint16_t tailNum = calCount % stride;
  bool hasTail = tailNum > 0;

  __ubuf__ T *workUb = (__ubuf__ T *)tmp.GetPhyAddr();
  __ubuf__ T *dstUb = (__ubuf__ T *)dst.GetPhyAddr();
  __ubuf__ T *src0Ub = (__ubuf__ T *)src0.GetPhyAddr();
  __ubuf__ T *src1Ub = (__ubuf__ T *)src1.GetPhyAddr();

  LocalTensor<T> LgammaLocal = tmp.ReinterpretCast<T>();
  LocalTensor<T> DivLocal = tmp[256].ReinterpretCast<T>();
  LocalTensor<T> PowLocal = tmp[512].ReinterpretCast<T>();
  LocalTensor<T> tmpLocal = tmp[768].ReinterpretCast<T>();

  for (uint16_t i = 0; i < repeatTime; ++i) {
    uint32_t sliceCnt = stride;
    if (i == repeatTime - 1 && hasTail) {
      sliceCnt = tailNum;
    }
    __ubuf__ T *dstChunkUb = dstUb + i * stride;
    __ubuf__ T *src0ChunkUb = src0Ub + i * stride;
    __ubuf__ T *src1ChunkUb = src1Ub + i * stride;

    // compute lgamma on UB
    Lgamma(LgammaLocal, src0[stride * i], sliceCnt);
    AscendC::PipeBarrier<PIPE_V>();

    // compute powf(x / fac, a) on UB
    // fac = a + lanczos_g - 0.5f
    AscendC::Adds(tmpLocal, src0[stride * i], lanczos_g - 0.5f, sliceCnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(DivLocal, src1[stride * i], tmpLocal, sliceCnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Power(PowLocal, DivLocal, src0[stride * i], sliceCnt);
    AscendC::PipeBarrier<PIPE_V>();

    IGammaCInternal::IgammaCImplPreJudge<T, false>(dstChunkUb, src0ChunkUb, src1ChunkUb, workUb, sliceCnt);
    AscendC::PipeBarrier<PIPE_V>();
    IGammaCInternal::IgammaCImplAsymptoticMask<T>(src0ChunkUb, src1ChunkUb, workUb);
    AscendC::PipeBarrier<PIPE_V>();
    IGammaCInternal::IgammaCImplAsymptotic<T, false>(dstChunkUb, src0ChunkUb, src1ChunkUb, workUb);
    AscendC::PipeBarrier<PIPE_V>();
    IGammaCInternal::IgammaCImplXgt1p1<T>(dstChunkUb, src0ChunkUb, src1ChunkUb, workUb);
    AscendC::PipeBarrier<PIPE_V>();
    IGammaCInternal::IgammaCImplXle0p5<T>(dstChunkUb, src0ChunkUb, src1ChunkUb, workUb);
    AscendC::PipeBarrier<PIPE_V>();
    IGammaCInternal::IgammaCImplXother<T>(dstChunkUb, src0ChunkUb, src1ChunkUb, workUb);
    AscendC::PipeBarrier<PIPE_V>();
  }
}
}  // namespace AscendC

#endif  // __ASCENDC_API_REGBASE_IGAMMA_H__
