/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_REDUCE_EXTEND_REDUCE_SUM_REDUCE_SUM_C310_IMPL_H
#define AUTOFUSE_REDUCE_EXTEND_REDUCE_SUM_REDUCE_SUM_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "reduce/reduce_common_util_3510_impl.h"
#include "reduce/reduce_common_ar_ra_reuse_unalign_3510_impl.h"
#include "reduce/reduce_common_ar_reuse_align_3510_impl.h"
#include "reduce/reduce_common_ra_reuse_align_3510_impl.h"

namespace AscendC {
namespace ReduceExtendInternal {
namespace Internal {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
template <const Reg::RegTrait &Trait>
__simd_callee__ inline void ReduceSumInt16(Reg::RegTensor<int16_t, Trait> &dst, Reg::RegTensor<int16_t, Trait> src,
                                           Reg::MaskReg mask) {
  Reg::RegTensor<int32_t, Trait> sum;
  Reg::ReduceSum<int32_t, int16_t, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<int32_t, Trait>,
                 Reg::RegTensor<int16_t, Trait>>(sum, src, mask);
  Reg::Pack<uint16_t, int32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t, Trait> &)dst, sum);
}
#endif

template <class T, bool isReuseSource = false>
__aicore__ inline void ReduceSumARB64ReuseSourceCompute(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
                                                        const uint32_t srcShape[]) {
  if ((srcShape[1] * sizeof(T)) % 32 == 0) {
    ReduceARImpl<T, Reg::RegTraitNumTwo,
                 Reg::Add<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>>,
                 Reg::ReduceSum<T, T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>,
                                Reg::RegTensor<T, Reg::RegTraitNumTwo>>,
                 isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape[0], srcShape[1]);
  } else {
    ReduceARReuseSourceUnAligned<
        T, Reg::RegTraitNumTwo, Reg::Add<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>>,
        Reg::ReduceSum<T, T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>,
                       Reg::RegTensor<T, Reg::RegTraitNumTwo>>>(dstAddr, srcAddr, srcShape[0], srcShape[1]);
  }
}

template <class T, bool isReuseSource = false>
__aicore__ inline void ReduceSumARReuseSourceCompute(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
                                                     const uint32_t srcShape[]) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
  if constexpr (std::is_same_v<T, int16_t>) {
    if ((srcShape[1] * sizeof(T)) % 32 == 0 || srcShape[1] == 1) {
      ReduceARImpl<T, Reg::RegTraitNumOne, Reg::Add<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>,
                   ReduceSumInt16<Reg::RegTraitNumOne>, isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape[0],
                                                                       srcShape[1]);
    } else {
      ReduceARReuseSourceUnAligned<T, Reg::RegTraitNumOne, Reg::Add<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>,
                                   ReduceSumInt16<Reg::RegTraitNumOne>>(dstAddr, srcAddr, srcShape[0], srcShape[1]);
    }
  } else {
    if ((srcShape[1] * sizeof(T)) % 32 == 0 || srcShape[1] == 1) {
      ReduceARImpl<T, Reg::RegTraitNumOne, Reg::Add<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>,
                   Reg::ReduceSum<T, T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>, Reg::RegTensor<T>>,
                   isReuseSource, ReduceType::SUM>(dstAddr, srcAddr, tmpAddr, srcShape[0], srcShape[1]);
    } else {
      ReduceARReuseSourceUnAligned<
          T, Reg::RegTraitNumOne, Reg::Add<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>,
          Reg::ReduceSum<T, T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>, Reg::RegTensor<T>>>(
          dstAddr, srcAddr, srcShape[0], srcShape[1]);
    }
  }
#else
  if ((srcShape[1] * sizeof(T)) % 32 == 0 || srcShape[1] == 1) {
    ReduceARImpl<T, Reg::RegTraitNumOne, Reg::Add<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>,
                 Reg::ReduceSum<T, T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>, Reg::RegTensor<T>>, isReuseSource,
                 ReduceType::SUM>(dstAddr, srcAddr, tmpAddr, srcShape[0], srcShape[1]);
  } else {
    ReduceARReuseSourceUnAligned<
        T, Reg::RegTraitNumOne, Reg::Add<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>,
        Reg::ReduceSum<T, T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>, Reg::RegTensor<T>>>(
        dstAddr, srcAddr, srcShape[0], srcShape[1]);
  }
#endif
}

template <class T, bool isReuseSource = false>
__aicore__ inline void ReduceSumRAB64ReuseSourceCompute(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
                                                        const uint32_t srcShape[]) {
  if ((srcShape[1] * sizeof(T)) % 32 == 0) {
    ReduceRAB64ReuseSource<T, Reg::RegTraitNumTwo,
                           Reg::Add<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>>,
                           isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape[1], srcShape[0]);
  } else {
    ReduceRAReuseSourceUnAlignedB64<T, Reg::RegTraitNumTwo,
                                    Reg::Add<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>>>(
        dstAddr, srcAddr, srcShape[1], srcShape[0]);
  }
}

template <class T, bool isReuseSource = false>
__aicore__ inline void ReduceSumRAReuseSourceCompute(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
                                                     const uint32_t srcShape[]) {
  if ((srcShape[1] * sizeof(T)) % 32 == 0) {
    ReduceRAImpl<T, Reg::RegTraitNumOne, Reg::Add<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>, isReuseSource>(
        dstAddr, srcAddr, tmpAddr, srcShape[1], srcShape[0]);
  } else {
    ReduceRAReuseSourceUnAligned<T, Reg::RegTraitNumOne, Reg::Add<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>>(
        dstAddr, srcAddr, srcShape[1], srcShape[0]);
  }
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
__simd_vf__ inline void NarrowReduceSumOutput(__ubuf__ int8_t *dstAddr, __ubuf__ int16_t *srcAddr, uint32_t count) {
  constexpr uint32_t elementsPerLoop = GetVecLen() / sizeof(int16_t);
  const uint32_t loops = CeilDivision(count, elementsPerLoop);
  uint32_t remaining = count;
  Reg::RegTensor<int16_t> srcReg;
  Reg::RegTensor<int8_t> dstReg;
  Reg::MaskReg mask;
  for (uint32_t i = 0; i < loops; ++i) {
    mask = Reg::UpdateMask<int16_t>(remaining);
    Reg::LoadAlign(srcReg, srcAddr + i * elementsPerLoop);
    Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint8_t> &)dstReg,
                                                           (Reg::RegTensor<uint16_t> &)srcReg);
    Reg::MaskPack(mask, mask);
    Reg::StoreAlign(dstAddr + i * elementsPerLoop, dstReg, mask);
  }
}

template <class pattern>
__aicore__ inline void ReduceSumInt8Compute(const LocalTensor<int8_t> &dst, const LocalTensor<int8_t> &src,
                                            const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t srcShape[]) {
  constexpr uint32_t blockBytes = 32;
  const uint32_t inputCount = srcShape[0] * srcShape[1];
  const uint32_t outputCount = std::is_same_v<pattern, Pattern::Reduce::AR> ? srcShape[0] : srcShape[1];
  const uint32_t widenedInputBytes = CeilDivision(inputCount * sizeof(int16_t), blockBytes) * blockBytes;
  LocalTensor<int16_t> widenedSrcTensor = sharedTmpBuffer.ReinterpretCast<int16_t>();
  LocalTensor<int16_t> widenedDstTensor = widenedSrcTensor[widenedInputBytes / sizeof(int16_t)];
  Cast(widenedSrcTensor, src, RoundMode::CAST_NONE, inputCount);
  PipeBarrier<PIPE_V>();
  __ubuf__ int16_t *widenedSrc = (__ubuf__ int16_t *)widenedSrcTensor.GetPhyAddr();
  __ubuf__ int16_t *widenedDst = (__ubuf__ int16_t *)widenedDstTensor.GetPhyAddr();
  if constexpr (std::is_same_v<pattern, Pattern::Reduce::AR>) {
    ReduceSumARReuseSourceCompute<int16_t, true>(widenedDst, widenedSrc, widenedDst, srcShape);
  } else {
    ReduceSumRAReuseSourceCompute<int16_t, true>(widenedDst, widenedSrc, widenedDst, srcShape);
  }
  PipeBarrier<PIPE_V>();
  NarrowReduceSumOutput((__ubuf__ int8_t *)dst.GetPhyAddr(), widenedDst, outputCount);
}
#endif

template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceSumImpl(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                     const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t srcShape[],
                                     bool srcInnerPad) {
  CheckTensorPos<T>(dst, Hardware::UB, "dstTensor", "VECIN/VECCALC/VECOUT", "ReduceSum");
  CheckTensorPos<T>(src, Hardware::UB, "srcTensor", "VECIN/VECCALC/VECOUT", "ReduceSum");
  CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN/VECCALC/VECOUT", "ReduceSum");
  static_assert(std::is_same_v<pattern, Pattern::Reduce::AR> || std::is_same_v<pattern, Pattern::Reduce::RA>,
                "ReduceSum only support AR and RA pattern on current device!");
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
  static_assert(SupportType<T, int8_t, int16_t, int32_t, uint32_t, float, int64_t, uint64_t>(),
                "ReduceSum only support int8_t/int16_t/int32_t/uint32_t/float/int64_t/uint64_t data type on current "
                "device!");
#else
  static_assert(SupportType<T, int32_t, uint32_t, float, int64_t, uint64_t>(),
                "ReduceSum only support int32_t/uint32_t/float/int64_t/uint64_t data type on current device!");
#endif
  __ubuf__ T *dstAddr = (__ubuf__ T *)dst.GetPhyAddr();
  __ubuf__ T *srcAddr = (__ubuf__ T *)src.GetPhyAddr();
  LocalTensor<T> tmpBuf = sharedTmpBuffer.ReinterpretCast<T>();
  __ubuf__ T *tmpAddr = (__ubuf__ T *)tmpBuf.GetPhyAddr();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
  if constexpr (std::is_same_v<T, int8_t>) {
    ReduceSumInt8Compute<pattern>(dst, src, sharedTmpBuffer, srcShape);
  } else
#endif
  {
    if constexpr (std::is_same_v<pattern, Pattern::Reduce::AR>) {
      if constexpr (SupportBytes<T, 8>()) {
        ReduceSumARB64ReuseSourceCompute<T, isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape);
      } else {
        ReduceSumARReuseSourceCompute<T, isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape);
      }
    } else {
      if constexpr (SupportBytes<T, 8>()) {
        ReduceSumRAB64ReuseSourceCompute<T, isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape);
      } else {
        ReduceSumRAReuseSourceCompute<T, isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape);
      }
    }
  }
}
}  // namespace Internal
}  // namespace ReduceExtendInternal
}  // namespace AscendC
#endif  // AUTOFUSE_REDUCE_EXTEND_REDUCE_SUM_REDUCE_SUM_C310_IMPL_H
