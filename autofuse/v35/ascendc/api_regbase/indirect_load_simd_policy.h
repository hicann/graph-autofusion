/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMD_POLICY_H_
#define AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMD_POLICY_H_

namespace AscendC {
namespace Internal {
template <typename Index>
struct IndirectLoadSimdIndexPolicy {
  static constexpr bool kSupported = false;
};

template <>
struct IndirectLoadSimdIndexPolicy<int32_t> {
  static constexpr bool kSupported = true;
  static constexpr uint32_t kElementsPerLoad = VECTOR_REG_WIDTH / sizeof(uint32_t);
  static constexpr uint32_t kElementsPerRepeat = kElementsPerLoad;
  struct LoadState {
    MicroAPI::UnalignReg unalign;
    __ubuf__ uint32_t *address;
  };

  __simd_callee__ inline static void Init(LoadState &state, __ubuf__ int32_t *address) {
    state.address = (__ubuf__ uint32_t *)address;
    MicroAPI::DataCopyUnAlignPre(state.unalign, state.address);
  }

  __simd_callee__ inline static void Load(MicroAPI::RegTensor<uint32_t> &index, MicroAPI::MaskReg &valid_mask,
                                          LoadState &state, uint32_t element_count, MicroAPI::MaskReg lane_mask) {
    MicroAPI::DataCopyUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(index, state.unalign, state.address,
                                                                                 element_count);
    valid_mask = lane_mask;
  }

  __simd_callee__ inline static void LoadPair(MicroAPI::RegTensor<uint32_t> &index0,
                                              MicroAPI::RegTensor<uint32_t> &index1, MicroAPI::MaskReg &valid_mask0,
                                              MicroAPI::MaskReg &valid_mask1, LoadState &state,
                                              uint32_t element_count) {
    uint32_t count0 = element_count > kElementsPerLoad ? kElementsPerLoad : element_count;
    uint32_t count1 = element_count - count0;
    valid_mask0 = MicroAPI::UpdateMask<uint32_t>(count0);
    valid_mask1 = MicroAPI::UpdateMask<uint32_t>(count1);
    MicroAPI::DataCopyUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(index0, state.unalign, state.address,
                                                                                 count0);
    if (count1 != 0U) {
      MicroAPI::DataCopyUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(index1, state.unalign, state.address,
                                                                                   count1);
    }
  }
};

template <>
struct IndirectLoadSimdIndexPolicy<int64_t> {
  static constexpr bool kSupported = true;
  static constexpr uint32_t kElementsPerLoad = VECTOR_REG_WIDTH / sizeof(uint64_t);
  static constexpr uint32_t kElementsPerRepeat = VECTOR_REG_WIDTH / sizeof(uint32_t);
  struct LoadState {
    MicroAPI::UnalignReg unalign;
    __ubuf__ uint32_t *address;
  };

  __simd_callee__ inline static void Init(LoadState &state, __ubuf__ int64_t *address) {
    state.address = (__ubuf__ uint32_t *)address;
    MicroAPI::DataCopyUnAlignPre(state.unalign, state.address);
  }

  __simd_callee__ inline static void Load(MicroAPI::RegTensor<uint32_t> &index, MicroAPI::MaskReg &valid_mask,
                                          LoadState &state, uint32_t element_count, MicroAPI::MaskReg lane_mask) {
    LoadHalf(index, state, element_count);
    valid_mask = lane_mask;
  }

  __simd_callee__ inline static void LoadPair(MicroAPI::RegTensor<uint32_t> &index0,
                                              MicroAPI::RegTensor<uint32_t> &index1, MicroAPI::MaskReg &valid_mask0,
                                              MicroAPI::MaskReg &valid_mask1, LoadState &state,
                                              uint32_t element_count) {
    constexpr uint32_t kPairElements = 2U * kElementsPerLoad;
    const uint32_t count0 = element_count > kPairElements ? kPairElements : element_count;
    const uint32_t count1 = element_count - count0;
    LoadHalf(index0, state, count0);
    LoadHalf(index1, state, count1);
    uint32_t mask_count0 = count0;
    uint32_t mask_count1 = count1;
    valid_mask0 = MicroAPI::UpdateMask<uint32_t>(mask_count0);
    valid_mask1 = MicroAPI::UpdateMask<uint32_t>(mask_count1);
  }

 private:
  __simd_callee__ inline static void LoadHalf(MicroAPI::RegTensor<uint32_t> &index, LoadState &state,
                                              uint32_t element_count) {
    MicroAPI::RegTensor<uint32_t> raw_index0;
    MicroAPI::RegTensor<uint32_t> raw_index1;
    MicroAPI::RegTensor<uint32_t> high;
    const uint32_t count0 = element_count > kElementsPerLoad ? kElementsPerLoad : element_count;
    const uint32_t count1 = element_count - count0;
    if (count0 != kElementsPerLoad || count1 != kElementsPerLoad) {
      uint32_t duplicate_count = kElementsPerLoad * 2U;
      MicroAPI::MaskReg raw_mask = MicroAPI::UpdateMask<uint32_t>(duplicate_count);
      if (count0 != kElementsPerLoad) {
        MicroAPI::Duplicate(raw_index0, 0U, raw_mask);
      }
      if (count1 != kElementsPerLoad) {
        MicroAPI::Duplicate(raw_index1, 0U, raw_mask);
      }
    }
    if (count0 != 0U) {
      MicroAPI::DataCopyUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(raw_index0, state.unalign,
                                                                                   state.address, count0 * 2U);
    }
    if (count1 != 0U) {
      MicroAPI::DataCopyUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(raw_index1, state.unalign,
                                                                                   state.address, count1 * 2U);
    }
    MicroAPI::DeInterleave<uint32_t>(index, high, raw_index0, raw_index1);
  }
};

template <typename X>
struct IndirectLoadSimdValueSupported {
  static constexpr bool kSupported = false;
};

#define INDIRECT_LOAD_SIMD_SUPPORTED_VALUE_TYPE(type) \
  template <>                                         \
  struct IndirectLoadSimdValueSupported<type> {       \
    static constexpr bool kSupported = true;          \
  }
INDIRECT_LOAD_SIMD_SUPPORTED_VALUE_TYPE(int16_t);
INDIRECT_LOAD_SIMD_SUPPORTED_VALUE_TYPE(uint16_t);
INDIRECT_LOAD_SIMD_SUPPORTED_VALUE_TYPE(half);
INDIRECT_LOAD_SIMD_SUPPORTED_VALUE_TYPE(bfloat16_t);
INDIRECT_LOAD_SIMD_SUPPORTED_VALUE_TYPE(int32_t);
INDIRECT_LOAD_SIMD_SUPPORTED_VALUE_TYPE(uint32_t);
INDIRECT_LOAD_SIMD_SUPPORTED_VALUE_TYPE(float);
#undef INDIRECT_LOAD_SIMD_SUPPORTED_VALUE_TYPE

template <typename X, size_t Size = sizeof(X)>
struct IndirectLoadSimdValuePolicy {
  static constexpr bool kSupported = false;
};

template <typename X>
struct IndirectLoadSimdValuePolicy<X, sizeof(uint32_t)> {
  static constexpr bool kSupported = IndirectLoadSimdValueSupported<X>::kSupported;

  __simd_callee__ inline static void GatherAndStore(__ubuf__ X *x, __ubuf__ X *y,
                                                    MicroAPI::RegTensor<uint32_t> &source_index,
                                                    MicroAPI::MaskReg lane_mask, MicroAPI::MaskReg valid_mask,
                                                    uint32_t element_count, uint32_t input_actual_size) {
    (void)element_count;
    (void)input_actual_size;
    MicroAPI::RegTensor<X> value;
    MicroAPI::DataCopyGather(value, x, source_index, valid_mask);
    MicroAPI::DataCopy(y, value, lane_mask);
  }
};

template <typename X>
struct IndirectLoadSimdValuePolicy<X, sizeof(uint16_t)> {
  static constexpr bool kSupported = IndirectLoadSimdValueSupported<X>::kSupported;

  __simd_callee__ inline static void GatherAndStore(__ubuf__ X *x, __ubuf__ X *y,
                                                    MicroAPI::RegTensor<uint32_t> &source_index,
                                                    MicroAPI::MaskReg lane_mask, MicroAPI::MaskReg valid_mask,
                                                    uint32_t element_count, uint32_t input_actual_size) {
    MicroAPI::RegTensor<X> value;
    MicroAPI::RegTensor<uint32_t> window_base_reg;
    MicroAPI::RegTensor<uint32_t> local_index;
    MicroAPI::RegTensor<uint32_t> zero_index;
    MicroAPI::RegTensor<uint16_t> gather_index;
    MicroAPI::RegTensor<uint16_t> high_index;
    uint32_t value_count = element_count;
    MicroAPI::MaskReg value_mask = MicroAPI::UpdateMask<X>(value_count);
    MicroAPI::MaskReg empty_mask;
    MicroAPI::MaskReg gather_mask;
    MicroAPI::MaskReg high_mask;
    MicroAPI::Duplicate(zero_index, 0U, lane_mask);
    MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(empty_mask, source_index, 0U, lane_mask);
    if (input_actual_size <= 65536U) {
      MicroAPI::DeInterleave<uint16_t>(gather_index, high_index, (MicroAPI::RegTensor<uint16_t> &)source_index,
                                       (MicroAPI::RegTensor<uint16_t> &)zero_index);
      MicroAPI::MaskDeInterleave<uint16_t>(gather_mask, high_mask, valid_mask, empty_mask);
      MicroAPI::DataCopyGather(value, x, gather_index, gather_mask);
      MicroAPI::DataCopy(y, value, value_mask);
      return;
    }
    for (uint32_t window_base = 0U; window_base < input_actual_size;) {
      const uint32_t window_size = input_actual_size - window_base > 65536U ? 65536U : input_actual_size - window_base;
      const uint32_t window_end = window_base + window_size;
      MicroAPI::MaskReg lower_mask;
      MicroAPI::MaskReg upper_mask;
      MicroAPI::MaskReg window_mask;
      MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(lower_mask, source_index, window_base, lane_mask);
      MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(upper_mask, source_index, window_end, lane_mask);
      MicroAPI::MaskAnd(window_mask, lower_mask, upper_mask, lane_mask);
      MicroAPI::Duplicate(window_base_reg, window_base, lane_mask);
      MicroAPI::Sub(local_index, source_index, window_base_reg, lane_mask);
      MicroAPI::DeInterleave<uint16_t>(gather_index, high_index, (MicroAPI::RegTensor<uint16_t> &)local_index,
                                       (MicroAPI::RegTensor<uint16_t> &)zero_index);
      MicroAPI::MaskDeInterleave<uint16_t>(gather_mask, high_mask, window_mask, empty_mask);
      MicroAPI::DataCopyGather(value, x + window_base, gather_index, gather_mask);
      window_base = window_end;
    }
    MicroAPI::DataCopy(y, value, value_mask);
  }

  __simd_callee__ inline static void GatherAndStorePair(__ubuf__ X *x, __ubuf__ X *y,
                                                        MicroAPI::RegTensor<uint32_t> &source_index0,
                                                        MicroAPI::RegTensor<uint32_t> &source_index1,
                                                        MicroAPI::MaskReg lane_mask0, MicroAPI::MaskReg lane_mask1,
                                                        uint32_t element_count, uint32_t input_actual_size) {
    MicroAPI::RegTensor<X> value;
    MicroAPI::RegTensor<uint16_t> gather_index;
    MicroAPI::RegTensor<uint16_t> high_index;
    MicroAPI::MaskReg gather_mask;
    MicroAPI::MaskReg high_mask;
    uint32_t value_count = element_count;
    MicroAPI::MaskReg value_mask = MicroAPI::UpdateMask<X>(value_count);
    if (input_actual_size <= 65536U) {
      MicroAPI::DeInterleave<uint16_t>(gather_index, high_index, (MicroAPI::RegTensor<uint16_t> &)source_index0,
                                       (MicroAPI::RegTensor<uint16_t> &)source_index1);
      MicroAPI::MaskDeInterleave<uint16_t>(gather_mask, high_mask, lane_mask0, lane_mask1);
      MicroAPI::DataCopyGather(value, x, gather_index, gather_mask);
      MicroAPI::DataCopy(y, value, value_mask);
      return;
    }
    MicroAPI::RegTensor<uint32_t> window_base_reg;
    MicroAPI::RegTensor<uint32_t> local_index0;
    MicroAPI::RegTensor<uint32_t> local_index1;
    for (uint32_t window_base = 0U; window_base < input_actual_size;) {
      const uint32_t window_size = input_actual_size - window_base > 65536U ? 65536U : input_actual_size - window_base;
      const uint32_t window_end = window_base + window_size;
      MicroAPI::MaskReg window_mask0;
      MicroAPI::MaskReg window_mask1;
      WindowMask(window_mask0, source_index0, lane_mask0, window_base, window_end);
      WindowMask(window_mask1, source_index1, lane_mask1, window_base, window_end);
      MicroAPI::Duplicate(window_base_reg, window_base, lane_mask0);
      MicroAPI::Sub(local_index0, source_index0, window_base_reg, lane_mask0);
      MicroAPI::Duplicate(window_base_reg, window_base, lane_mask1);
      MicroAPI::Sub(local_index1, source_index1, window_base_reg, lane_mask1);
      MicroAPI::DeInterleave<uint16_t>(gather_index, high_index, (MicroAPI::RegTensor<uint16_t> &)local_index0,
                                       (MicroAPI::RegTensor<uint16_t> &)local_index1);
      MicroAPI::MaskDeInterleave<uint16_t>(gather_mask, high_mask, window_mask0, window_mask1);
      MicroAPI::DataCopyGather(value, x + window_base, gather_index, gather_mask);
      window_base = window_end;
    }
    MicroAPI::DataCopy(y, value, value_mask);
  }

 private:
  __simd_callee__ inline static void WindowMask(MicroAPI::MaskReg &window_mask,
                                                MicroAPI::RegTensor<uint32_t> &source_index,
                                                MicroAPI::MaskReg lane_mask, uint32_t window_base,
                                                uint32_t window_end) {
    MicroAPI::MaskReg lower_mask;
    MicroAPI::MaskReg upper_mask;
    MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(lower_mask, source_index, window_base, lane_mask);
    MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(upper_mask, source_index, window_end, lane_mask);
    MicroAPI::MaskAnd(window_mask, lower_mask, upper_mask, lane_mask);
  }
};
}  // namespace Internal
}  // namespace AscendC

#endif  // AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMD_POLICY_H_
