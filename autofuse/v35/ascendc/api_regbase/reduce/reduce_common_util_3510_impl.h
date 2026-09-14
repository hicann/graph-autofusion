/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message( \
    "impl/adv_api/detail/reduce/reduce_common_util_3510_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"adv_api/reduce/reduce.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_REDUCE_REDUCE_COMMON_UTIL_C310_IMPL_H__
#endif

#ifndef AUTOFUSE_REDUCE_EXTEND_REDUCE_COMMON_UTIL_C310_IMPL_H
#define AUTOFUSE_REDUCE_EXTEND_REDUCE_COMMON_UTIL_C310_IMPL_H

#include "kernel_basic_intf.h"
#include "kernel_tensor.h"

namespace AscendC {
namespace ReduceExtendInternal {
namespace ReduceOpInternal {
// Calculate the sum of two points based on count. The main block size is ex.2000->1024 900->512.
__aicore__ inline uint32_t CalculateRMainBlock(uint32_t count) {
  constexpr int32_t TWO_BITS = 2;
  constexpr int32_t FOUR_BITS = 4;
  constexpr int32_t EIGHT_BITS = 8;
  constexpr int32_t SIXTEEN_BITS = 16;
  count |= count >> 1;
  count |= count >> TWO_BITS;
  count |= count >> FOUR_BITS;
  count |= count >> EIGHT_BITS;
  count |= count >> SIXTEEN_BITS;
  return (count + 1) >> 1;
}

constexpr uint16_t BASE_FOLD = 4;
constexpr uint16_t BASE_FOLD_B64 = 3;
constexpr uint16_t FOLD_THREE = 3;
constexpr uint16_t FOLD_TWO = 2;
constexpr uint16_t FOLD_ONE = 1;
constexpr uint32_t U16_STRIDE = 65535;
constexpr uint16_t REGULAR_FOLD_NUM = 2;
constexpr uint16_t NO_REUSE_FOLD_NUM = 2;

static constexpr Reg::CastTrait CastTraitBF16F32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                    Reg::MaskMergeMode::ZEROING};
static constexpr Reg::CastTrait CastTraitF32BF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                    Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
static constexpr Reg::CastTrait CastTraitB8F16 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
static constexpr Reg::CastTrait CastTraitF16B8 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

__aicore__ inline uint32_t CalculateMainR(int64_t dimR, bool isAR, uint16_t vlSize) {
  constexpr uint16_t MAX_OFFSET = 16;
  int64_t mainR = 1;
  for (uint16_t i = 1; i < MAX_OFFSET; i++) {
    if ((dimR >> i) == 0) {
      break;
    }
    mainR = 1 << i;
  }

  if (isAR && dimR < vlSize) {
    mainR = dimR;
  }
  return static_cast<uint32_t>(mainR);
}

__aicore__ inline uint16_t CalculateFolds(const uint16_t count) {
  constexpr uint16_t MAX_OFFSET = 16;
  uint16_t folds = 0;
  uint16_t base = count;
  for (uint16_t i = 1; i < MAX_OFFSET; i++) {
    if ((base >> i) == 0) {
      break;
    }
    folds++;
  }
  return folds;
}

template <class T, class U, const Reg::RegTrait &Trait, const Reg::CastTrait &CastTraitUppper,
          const Reg::CastTrait &CastTraitLower, auto Binaryfunc, auto Reducefunc>
__simd_callee__ inline void ReduceARCastfoldOneToThree(Reg::RegTensor<T, Trait> &vreg0, Reg::RegTensor<T, Trait> &vreg1,
                                                       Reg::MaskReg &fullMask) {
  Reg::RegTensor<U, Trait> vreg0CastB32;
  Reg::RegTensor<U, Trait> vreg1CastB32;
  Reg::RegTensor<T, Trait> vreg2;
  Reg::RegTensor<T, Trait> vreg3;
  Reg::RegTensor<T, Trait> vreg4;
  Duplicate(vreg4, static_cast<T>(0), fullMask);
  Interleave(vreg2, vreg3, vreg0, vreg4);
  Binaryfunc(vreg0, vreg2, vreg3, fullMask);
  Reg::Cast<U, T, CastTraitUppper>(vreg0CastB32, vreg0, fullMask);
  Reducefunc(vreg1CastB32, vreg0CastB32, fullMask);
  Reg::Cast<T, U, CastTraitLower>(vreg1, vreg1CastB32, fullMask);
}

template <class T>
__simd_vf__ inline void ReduceCopyOutImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t calCount) {
  uint16_t repeatElm = GetVecLen();
  uint32_t sreg = calCount * sizeof(T);
  uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(sreg, repeatElm));
  Reg::MaskReg preg;
  Reg::RegTensor<uint8_t> srcReg;
  for (uint16_t i = 0; i < repeatTime; ++i) {
    preg = Reg::UpdateMask<uint8_t>(sreg);
    Reg::LoadAlign(srcReg, (__ubuf__ uint8_t *)src + i * repeatElm);
    Reg::StoreAlign((__ubuf__ uint8_t *)dst + i * repeatElm, srcReg, preg);
  }
}

template <typename T>
struct ExtractReduceCastType {
  using CastT = T;
};

template <>
struct ExtractReduceCastType<bfloat16_t> {
  using CastT = float;
};

template <>
struct ExtractReduceCastType<uint8_t> {
  using CastT = half;
};

template <>
struct ExtractReduceCastType<int8_t> {
  using CastT = half;
};

template <uint32_t size = sizeof(uint8_t)>
struct ExtractUnsignedTypeBySize {
  using T = uint8_t;
};

template <>
struct ExtractUnsignedTypeBySize<sizeof(uint16_t)> {
  using T = uint16_t;
};

template <>
struct ExtractUnsignedTypeBySize<sizeof(uint32_t)> {
  using T = uint32_t;
};
}  // namespace ReduceOpInternal
}  // namespace ReduceExtendInternal
}  // namespace AscendC
#endif  // AUTOFUSE_REDUCE_EXTEND_REDUCE_COMMON_UTIL_C310_IMPL_H

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_REDUCE_REDUCE_COMMON_UTIL_C310_IMPL_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_REDUCE_REDUCE_COMMON_UTIL_C310_IMPL_H__
#endif
