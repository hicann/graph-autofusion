/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_CHECK_BOUNDS_H__
#define __ASCENDC_API_REGBASE_CHECK_BOUNDS_H__

template <typename T, bool kCheckLower, bool kCheckUpper, const AscendC::Reg::RegTrait &regTrait, typename ErrorReg>
__simd_callee__ inline void CheckBoundsSetError(ErrorReg &errorReg, ErrorReg &oneReg, ErrorReg &zeroReg,
                                                AscendC::Reg::RegTensor<T, regTrait> &indicesReg,
                                                AscendC::Reg::RegTensor<T, regTrait> &sizesReg,
                                                AscendC::Reg::MaskReg &lowerMask, AscendC::Reg::MaskReg &upperMask,
                                                AscendC::Reg::MaskReg &mask) {
  if constexpr (kCheckLower && kCheckUpper) {
    AscendC::Reg::Compares<T, CMPMODE::LT>(lowerMask, indicesReg, static_cast<T>(0), mask);
    AscendC::Reg::Select(errorReg, oneReg, zeroReg, lowerMask);
    AscendC::Reg::Compare<T, CMPMODE::GE>(upperMask, indicesReg, sizesReg, mask);
    AscendC::Reg::Select(errorReg, oneReg, errorReg, upperMask);
  } else if constexpr (kCheckLower) {
    AscendC::Reg::Compares<T, CMPMODE::LT>(lowerMask, indicesReg, static_cast<T>(0), mask);
    AscendC::Reg::Select(errorReg, oneReg, zeroReg, lowerMask);
  } else {
    AscendC::Reg::Compare<T, CMPMODE::GE>(upperMask, indicesReg, sizesReg, mask);
    AscendC::Reg::Select(errorReg, oneReg, zeroReg, upperMask);
  }
}

template <typename T, const AscendC::Reg::RegTrait &regTrait>
__simd_callee__ inline void CheckBoundsPrepareUpper(AscendC::Reg::RegTensor<T, regTrait> &sizesMinusOneReg,
                                                    AscendC::Reg::RegTensor<T, regTrait> &sizesReg,
                                                    AscendC::Reg::RegTensor<T, regTrait> &oneReg,
                                                    AscendC::Reg::RegTensor<T, regTrait> &zeroReg,
                                                    AscendC::Reg::MaskReg &lowerMask, AscendC::Reg::MaskReg &mask) {
  AscendC::Reg::Sub(sizesMinusOneReg, sizesReg, oneReg, mask);
  AscendC::Reg::Compare<T, CMPMODE::EQ>(lowerMask, sizesReg, zeroReg, mask);
  AscendC::Reg::Select(sizesMinusOneReg, zeroReg, sizesMinusOneReg, lowerMask);
}

template <typename T, bool kCheckLower, bool kCheckUpper, const AscendC::Reg::RegTrait &regTrait>
__simd_callee__ inline void CheckBoundsSanitize(AscendC::Reg::RegTensor<T, regTrait> &sanitizedReg,
                                                AscendC::Reg::RegTensor<T, regTrait> &tempReg,
                                                AscendC::Reg::RegTensor<T, regTrait> &indicesReg,
                                                AscendC::Reg::RegTensor<T, regTrait> &sizesMinusOneReg,
                                                AscendC::Reg::MaskReg &upperMask, AscendC::Reg::MaskReg &mask) {
  if constexpr (kCheckLower && kCheckUpper) {
    AscendC::Reg::Maxs(tempReg, indicesReg, static_cast<T>(0), mask);
    AscendC::Reg::Compare<T, CMPMODE::GT>(upperMask, tempReg, sizesMinusOneReg, mask);
    AscendC::Reg::Select(sanitizedReg, sizesMinusOneReg, tempReg, upperMask);
  } else if constexpr (kCheckLower) {
    AscendC::Reg::Maxs(sanitizedReg, indicesReg, static_cast<T>(0), mask);
  } else {
    AscendC::Reg::Compare<T, CMPMODE::GT>(upperMask, indicesReg, sizesMinusOneReg, mask);
    AscendC::Reg::Select(sanitizedReg, sizesMinusOneReg, indicesReg, upperMask);
  }
}

// Compile-time mode and register-width specialization keeps one readable loop.
template <typename T, bool kCheckLower, bool kCheckUpper,
          const AscendC::Reg::RegTrait &regTrait = AscendC::Reg::RegTraitNumOne>
__simd_callee__ inline void CheckBoundsImplVFLoop(__ubuf__ T *sanitizedUb, __ubuf__ uint32_t *errorUb,
                                                  __ubuf__ T *indicesUb, __ubuf__ T *sizesUb, uint32_t calCount) {
  constexpr bool kNeedPackStore = sizeof(T) > sizeof(uint32_t);
  constexpr uint32_t vlSize = static_cast<uint32_t>(regTrait.REG_NUM * GetVecLen() / sizeof(T));
  uint16_t repeatTime = static_cast<uint16_t>(AscendC::CeilDivision(calCount, vlSize));
  AscendC::Reg::RegTensor<T, regTrait> indicesReg, sizesReg, sizesMinusOneReg, sanitizedReg, tempReg;
  AscendC::Reg::RegTensor<T, regTrait> errTReg, oneTReg, zeroTReg;
  AscendC::Reg::RegTensor<uint32_t> errReg, oneU32Reg, zeroU32Reg;
  AscendC::Reg::MaskReg upperMask, lowerMask;
  AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL, regTrait>();
  AscendC::Reg::Duplicate(oneTReg, static_cast<T>(1), mask);
  AscendC::Reg::Duplicate(zeroTReg, static_cast<T>(0), mask);
  if constexpr (!kNeedPackStore) {
    AscendC::Reg::Duplicate(oneU32Reg, static_cast<uint32_t>(1), mask);
    AscendC::Reg::Duplicate(zeroU32Reg, static_cast<uint32_t>(0), mask);
  }
  for (uint16_t i = 0U; i < repeatTime; ++i) {
    uint32_t errStoreCount = calCount > vlSize ? vlSize : calCount;
    mask = AscendC::Reg::UpdateMask<T, regTrait>(calCount);
    AscendC::Reg::LoadAlign(indicesReg, indicesUb + i * vlSize);
    if constexpr (kCheckUpper) {
      AscendC::Reg::LoadAlign(sizesReg, sizesUb + i * vlSize);
      CheckBoundsPrepareUpper<T, regTrait>(sizesMinusOneReg, sizesReg, oneTReg, zeroTReg, lowerMask, mask);
    }
    CheckBoundsSanitize<T, kCheckLower, kCheckUpper, regTrait>(sanitizedReg, tempReg, indicesReg, sizesMinusOneReg,
                                                               upperMask, mask);
    AscendC::Reg::StoreAlign(sanitizedUb + i * vlSize, sanitizedReg, mask);
    if constexpr (kNeedPackStore) {
      CheckBoundsSetError<T, kCheckLower, kCheckUpper, regTrait>(errTReg, oneTReg, zeroTReg, indicesReg, sizesReg,
                                                                 lowerMask, upperMask, mask);
      AscendC::Reg::Pack<uint32_t, T>(errReg, errTReg);
    } else {
      CheckBoundsSetError<T, kCheckLower, kCheckUpper, regTrait>(errReg, oneU32Reg, zeroU32Reg, indicesReg, sizesReg,
                                                                 lowerMask, upperMask, mask);
    }
    mask = AscendC::Reg::UpdateMask<uint32_t>(errStoreCount);
    AscendC::Reg::StoreAlign(errorUb + i * vlSize, errReg, mask);
  }
}

template <typename T, bool kCheckLower, bool kCheckUpper>
__simd_callee__ inline void CheckBoundsSpecializedVFLoop(__ubuf__ T *sanitizedUb, __ubuf__ uint32_t *errorUb,
                                                         __ubuf__ T *indicesUb, __ubuf__ T *sizesUb,
                                                         uint32_t calCount) {
  if constexpr (sizeof(T) > sizeof(uint32_t)) {
    CheckBoundsImplVFLoop<T, kCheckLower, kCheckUpper, AscendC::Reg::RegTraitNumTwo>(sanitizedUb, errorUb, indicesUb,
                                                                                     sizesUb, calCount);
  } else {
    CheckBoundsImplVFLoop<T, kCheckLower, kCheckUpper>(sanitizedUb, errorUb, indicesUb, sizesUb, calCount);
  }
}

template <typename T>
__simd_vf__ inline void CheckBoundsImplVF(__ubuf__ T *sanitizedUb, __ubuf__ uint32_t *errorUb, __ubuf__ T *indicesUb,
                                          __ubuf__ T *sizesUb, uint32_t calCount, bool checkLower, bool checkUpper) {
  if (checkLower) {
    if (checkUpper) {
      CheckBoundsSpecializedVFLoop<T, true, true>(sanitizedUb, errorUb, indicesUb, sizesUb, calCount);
    } else {
      CheckBoundsSpecializedVFLoop<T, true, false>(sanitizedUb, errorUb, indicesUb, sizesUb, calCount);
    }
  } else {
    CheckBoundsSpecializedVFLoop<T, false, true>(sanitizedUb, errorUb, indicesUb, sizesUb, calCount);
  }
}

/**
 * @brief 动态索引越界检查与安全规约高阶 API (对标 LoopIR check_bounds)
 *
 * 对给定的动态索引进行越界检测和安全钳位。钳位后的合法索引写入 sanitizedIndices，
 * 逐元素的越界错误标志写入 errorFlag (0 = 正常, 1 = 越界)。
 *
 * @tparam T 索引数据类型，支持 int32_t, int64_t, uint32_t, uint64_t
 * @param sanitizedIndices 目的操作数 (经过安全钳位后的合法索引)，LocalTensor<T> 类型
 * @param errorFlag 目的操作数 (全局越界错误标志寄存器，0为正常，1为越界)，LocalTensor<uint32_t> 类型
 * @param indices 源操作数 (待校验的动态索引，对标 LoopIR expr)，LocalTensor<T> 类型
 * @param sizes 源操作数 (边界限制大小，对标 LoopIR size)，LocalTensor<T> 类型
 * @param sharedTmpBuffer 分配的临时计算空间，LocalTensor<uint8_t> 类型
 * @param calCount 当前处理的有效索引元素个数
 * @param checkLower 是否启用下界校验 (indices >= 0)
 * @param checkUpper 是否启用上界校验 (indices < sizes)
 */
template <typename T>
__aicore__ inline void CheckBoundsExtend(const LocalTensor<T> &sanitizedIndices, const LocalTensor<uint32_t> &errorFlag,
                                         const LocalTensor<T> &indices, const LocalTensor<T> &sizes,
                                         const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount,
                                         bool checkLower = true, bool checkUpper = true) {
  static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> || std::is_same_v<T, uint32_t> ||
                    std::is_same_v<T, uint64_t>,
                "CheckBounds only supports int32_t, int64_t, uint32_t, uint64_t");

  if ASCEND_IS_AIC {
    return;
  }
  if (calCount == 0) {
    return;
  }

  if (!checkLower && !checkUpper) {
    // Align calCount up to the vector width so that DataCopy receives a
    // 32 B-aligned byte count.  The caller has already allocated the output
    // buffers with aligned capacity, so the extra lanes are safe to touch.
    uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    uint32_t alignedCount = AscendC::AlignUp(calCount, vlSize);
    AscendC::DataCopy(sanitizedIndices, indices, alignedCount);
    AscendC::Duplicate(errorFlag, static_cast<uint32_t>(0), alignedCount);
    return;
  }

  // Dispatch to compile-time specialized VF loop so that checkLower/checkUpper
  // are resolved via if constexpr, eliminating runtime branches inside the loop.
  __ubuf__ T *sanitizedUb = (__ubuf__ T *)sanitizedIndices.GetPhyAddr();
  __ubuf__ uint32_t *errorUb = (__ubuf__ uint32_t *)errorFlag.GetPhyAddr();
  __ubuf__ T *indicesUb = (__ubuf__ T *)indices.GetPhyAddr();
  __ubuf__ T *sizesUb = (__ubuf__ T *)sizes.GetPhyAddr();

  CheckBoundsImplVF<T>(sanitizedUb, errorUb, indicesUb, sizesUb, calCount, checkLower, checkUpper);
}

#endif  // __ASCENDC_API_REGBASE_CHECK_BOUNDS_H__
