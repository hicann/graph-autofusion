/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_ARGMAX_H__
#define __ASCENDC_API_REGBASE_ARGMAX_H__

#include <cstdint>
#include <limits>
#include <type_traits>

namespace AscendC {
constexpr Reg::CastTrait kArgMaxCastTrait = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::UNKNOWN};

template <typename T>
__simd_callee__ inline void ArgMaxUpdate(Reg::RegTensor<T> &bestValue, Reg::RegTensor<int32_t> &bestIndex,
                                         Reg::RegTensor<T> &currentValue, Reg::RegTensor<int32_t> &currentIndex,
                                         Reg::MaskReg &activeMask) {
  Reg::MaskReg updateMask, valueMask, bestMask;
  if constexpr (std::is_same_v<T, float>) {
    Reg::RegTensor<T> zeroValue;
    Reg::Duplicate(zeroValue, static_cast<T>(0), activeMask);
    Reg::Compare<T, CMPMODE::EQ>(valueMask, currentValue, zeroValue, activeMask);
    Reg::Select(currentValue, zeroValue, currentValue, valueMask);
  }
  Reg::Compare<T, CMPMODE::GT>(updateMask, currentValue, bestValue, activeMask);
  if constexpr (std::is_same_v<T, float>) {
    Reg::Compare<T, CMPMODE::NE>(valueMask, currentValue, currentValue, activeMask);
    Reg::Compare<T, CMPMODE::NE>(bestMask, bestValue, bestValue, activeMask);
    Reg::Not(bestMask, bestMask, activeMask);
    Reg::And(valueMask, valueMask, bestMask, activeMask);
    Reg::Or(updateMask, updateMask, valueMask, activeMask);
  }
  Reg::Select(bestValue, currentValue, bestValue, updateMask);
  Reg::Select(bestIndex, currentIndex, bestIndex, updateMask);
}

template <typename T, typename U>
__simd_callee__ inline void ArgMaxStore(__ubuf__ T *dst, Reg::RegTensor<U> &bestValue,
                                        Reg::RegTensor<int32_t> &bestIndex, uint32_t count) {
  Reg::MaskReg activeMask = Reg::UpdateMask<U>(count);
  Reg::MaskReg valueMask, bestMask;
  Reg::RegTensor<U> reducedValue, broadcastValue;
  Reg::RegTensor<int32_t> reducedIndex;
  Reg::Reduce<Reg::ReduceType::MAX>(reducedValue, bestValue, activeMask);
  Reg::Duplicate(broadcastValue, reducedValue, activeMask);
  Reg::Compare<U, CMPMODE::EQ>(valueMask, bestValue, broadcastValue, activeMask);
  if constexpr (std::is_same_v<U, float>) {
    Reg::Compare<U, CMPMODE::NE>(bestMask, bestValue, bestValue, activeMask);
    Reg::Or(valueMask, valueMask, bestMask, activeMask);
  }
  Reg::Reduce<Reg::ReduceType::MIN>(reducedIndex, bestIndex, valueMask);
  if constexpr (std::is_same_v<T, int64_t>) {
    // Reduce leaves one index in lane 0; VL1 converts only that index.
    Reg::RegTensor<int64_t> result;
    activeMask = Reg::CreateMask<int32_t, Reg::MaskPattern::VL1>();
    Reg::Cast<int64_t, int32_t, kArgMaxCastTrait>(result, reducedIndex, activeMask);
    Reg::Store(dst, result, 1);
  } else {
    Reg::Store(dst, reducedIndex, 1);
  }
}

template <typename T>
__simd_callee__ inline void ArgMaxMerge(Reg::RegTensor<T> &bestValue, Reg::RegTensor<int32_t> &bestIndex,
                                        Reg::RegTensor<T> &nextBestValue, Reg::RegTensor<int32_t> &nextBestIndex) {
  constexpr bool isFloat = std::is_same_v<T, float>;
  constexpr uint32_t vlSize = GetVecLen() / sizeof(T);
  Reg::MaskReg activeMask, updateMask, valueMask, nanMask;
  uint32_t maskCount = vlSize;
  activeMask = Reg::UpdateMask<T>(maskCount);
  Reg::Compare<int32_t, CMPMODE::LT>(updateMask, nextBestIndex, bestIndex, activeMask);
  Reg::Compare<T, CMPMODE::EQ>(valueMask, nextBestValue, bestValue, activeMask);
  Reg::And(valueMask, valueMask, updateMask, activeMask);
  if constexpr (isFloat) {
    // A new NaN wins if the old value is not NaN, or its index is earlier.
    Reg::Compare<T, CMPMODE::NE>(nanMask, bestValue, bestValue, activeMask);
    Reg::Not(nanMask, nanMask, activeMask);
    Reg::Or(updateMask, updateMask, nanMask, activeMask);
    Reg::Compare<T, CMPMODE::NE>(nanMask, nextBestValue, nextBestValue, activeMask);
    Reg::And(updateMask, updateMask, nanMask, activeMask);
    Reg::Or(valueMask, valueMask, updateMask, activeMask);
  }
  Reg::Compare<T, CMPMODE::GT>(updateMask, nextBestValue, bestValue, activeMask);
  Reg::Or(updateMask, updateMask, valueMask, activeMask);
  Reg::Select(bestValue, nextBestValue, bestValue, updateMask);
  Reg::Select(bestIndex, nextBestIndex, bestIndex, updateMask);
}

template <typename U, typename T, bool isRa, bool isAligned = false>
__simd_vf__ inline void ArgMaxImpl(__ubuf__ U *indexDst, __ubuf__ T *src, uint32_t dimA, uint32_t dimR,
                                   uint32_t mainCount, uint32_t repeatTimes) {
  constexpr uint32_t vlSize = GetVecLen() / sizeof(T);
  const T lowest = std::is_same_v<T, float> ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();

  Reg::MaskReg activeMask;
  Reg::RegTensor<T> currentValue, nextValue, bestValue, nextBestValue;
  Reg::RegTensor<int32_t> currentIndex, nextIndex, gatherIndex, bestIndex, nextBestIndex;

  Reg::UnalignRegForLoad loadReg;
  for (uint32_t i = 0; i < dimA; ++i) {
    Reg::Duplicate(bestValue, lowest);
    Reg::Arange(bestIndex, static_cast<int32_t>(0));
    uint32_t remaining = dimR;
    uint32_t offset = 0;
    if constexpr (isAligned) {
      Reg::Duplicate(nextBestValue, lowest);
      Reg::Arange(nextBestIndex, static_cast<int32_t>(vlSize));
      for (uint32_t j = 0; j < mainCount; j += 2) {
        activeMask = Reg::UpdateMask<T>(remaining);
        Reg::LoadAlign(currentValue, src + i * dimR + offset);
        Reg::LoadAlign(nextValue, src + i * dimR + offset + vlSize);
        Reg::Arange(currentIndex, static_cast<int32_t>(offset));
        Reg::Arange(nextIndex, static_cast<int32_t>(offset + vlSize));
        ArgMaxUpdate(bestValue, bestIndex, currentValue, currentIndex, activeMask);
        activeMask = Reg::UpdateMask<T>(remaining);
        ArgMaxUpdate(nextBestValue, nextBestIndex, nextValue, nextIndex, activeMask);
        offset += 2 * vlSize;
      }
    }
    for (uint32_t j = mainCount; j < repeatTimes; ++j) {
      activeMask = Reg::UpdateMask<T>(remaining);
      if constexpr (isRa) {
        Reg::Arange(gatherIndex, static_cast<int32_t>(offset));
        Reg::Muls(gatherIndex, gatherIndex, static_cast<int32_t>(dimA), activeMask);
        Reg::Adds(gatherIndex, gatherIndex, static_cast<int32_t>(i), activeMask);
        Reg::Gather(currentValue, src, (Reg::RegTensor<uint32_t> &)gatherIndex, activeMask);
        Reg::Arange(currentIndex, static_cast<int32_t>(offset));
      } else if constexpr (isAligned) {
        Reg::Arange(currentIndex, static_cast<int32_t>(offset));
        Reg::Gather(currentValue, src + i * dimR, (Reg::RegTensor<uint32_t> &)currentIndex, activeMask);
      } else {
        Reg::LoadUnAlignPre(loadReg, src + i * dimR + offset);
        Reg::LoadUnAlign(currentValue, loadReg, src + i * dimR + offset);
        Reg::Arange(currentIndex, static_cast<int32_t>(offset));
      }
      ArgMaxUpdate(bestValue, bestIndex, currentValue, currentIndex, activeMask);
      offset += vlSize;
    }

    if constexpr (isAligned) {
      ArgMaxMerge(bestValue, bestIndex, nextBestValue, nextBestIndex);
    }

    ArgMaxStore(indexDst + i, bestValue, bestIndex, dimR);
  }
}

template <typename T, typename U, class pattern>
__aicore__ inline void ArgMaxExtendImpl(const LocalTensor<T> &indexDst, const LocalTensor<U> &src,
                                        const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t *srcShape) {
  const uint32_t dimA = SupportType<pattern, Pattern::Reduce::AR>() ? srcShape[0] : srcShape[1];
  const uint32_t dimR = SupportType<pattern, Pattern::Reduce::AR>() ? srcShape[1] : srcShape[0];
  constexpr uint32_t vlSize = GetVecLen() / sizeof(U);
  const uint32_t repeatTimes = CeilDivision(dimR, vlSize);
  __ubuf__ T *localIndexDst = (__ubuf__ T *)indexDst.GetPhyAddr();
  if constexpr (SupportType<pattern, Pattern::Reduce::AR>()) {
    if (dimR % (32U / sizeof(U)) == 0) {
      uint32_t mainCount = dimR / (2 * vlSize) * 2;
      asc_vf_call<ArgMaxImpl<T, U, false, true>>(localIndexDst, (__ubuf__ U *)src.GetPhyAddr(), dimA, dimR, mainCount,
                                                 repeatTimes);
    } else {
      asc_vf_call<ArgMaxImpl<T, U, false>>(localIndexDst, (__ubuf__ U *)src.GetPhyAddr(), dimA, dimR, 0, repeatTimes);
    }
  } else {
    asc_vf_call<ArgMaxImpl<T, U, true>>(localIndexDst, (__ubuf__ U *)src.GetPhyAddr(), dimA, dimR, 0, repeatTimes);
  }
}

/**
 * Returns the first maximum index for AR or RA data.
 * sharedTmpBuffer is reserved; this implementation needs no temporary buffer.
 * srcInnerPad is reserved for compatibility; pass false for compact input.
 */
template <typename T, typename U, class pattern>
__aicore__ inline void ArgMaxExtend(const LocalTensor<T> &dst, const LocalTensor<U> &src,
                                    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t *srcShape,
                                    bool srcInnerPad) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert(SupportType<U, float, int32_t>(), "ArgMaxExtend: input type must be float or int32_t");
  static_assert(SupportType<T, int32_t, int64_t>(), "ArgMaxExtend: index type must be int32_t or int64_t");
  static_assert(SupportType<pattern, Pattern::Reduce::AR, Pattern::Reduce::RA>(),
                "ArgMaxExtend: only AR and RA patterns are supported");
  ASCENDC_ASSERT(srcShape != nullptr, { KERNEL_LOG(KERNEL_ERROR, "ArgMaxExtend: srcShape must not be null"); });
  ASCENDC_ASSERT(srcShape[0] > 0 && srcShape[1] > 0,
                 { KERNEL_LOG(KERNEL_ERROR, "ArgMaxExtend: both dimensions must be positive"); });
  const uint64_t count = static_cast<uint64_t>(srcShape[0]) * srcShape[1];
  const uint32_t dimA = SupportType<pattern, Pattern::Reduce::AR>() ? srcShape[0] : srcShape[1];
  ASCENDC_ASSERT(count <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
                 { KERNEL_LOG(KERNEL_ERROR, "ArgMaxExtend: input count must not exceed INT32_MAX"); });
  ASCENDC_ASSERT(src.GetSize() >= count,
                 { KERNEL_LOG(KERNEL_ERROR, "ArgMaxExtend: source tensor is smaller than srcShape"); });
  ASCENDC_ASSERT(dst.GetSize() >= dimA,
                 { KERNEL_LOG(KERNEL_ERROR, "ArgMaxExtend: destination tensor is smaller than A"); });
  ASCENDC_ASSERT(!srcInnerPad,
                 { KERNEL_LOG(KERNEL_ERROR, "ArgMaxExtend: only compact input (srcInnerPad=false) is supported"); });
  ArgMaxExtendImpl<T, U, pattern>(dst, src, sharedTmpBuffer, srcShape);
}

}  // namespace AscendC

#endif  // __ASCENDC_API_REGBASE_ARGMAX_H__
