/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMD_H_
#define AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMD_H_

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

struct IndirectLoadSimdAddressContext {
  uint32_t output_position;
  uint32_t input_actual_size;
  uint32_t input_inner;
  uint32_t index_inner;
  uint32_t input_slice_count;
  uint32_t output_slice_count;
  bool inner_layout_matches;
};

__simd_callee__ inline void IndirectLoadSimdDivMod(MicroAPI::RegTensor<uint32_t> &quotient,
                                                   MicroAPI::RegTensor<uint32_t> &remainder,
                                                   MicroAPI::RegTensor<uint32_t> &value, uint32_t divisor,
                                                   MicroAPI::MaskReg mask) {
  MicroAPI::RegTensor<uint32_t> divisor_reg;
  MicroAPI::RegTensor<uint32_t> product;
  MicroAPI::Duplicate(divisor_reg, divisor, mask);
  MicroAPI::Div(quotient, value, divisor_reg, mask);
  MicroAPI::Mul(product, quotient, divisor_reg, mask);
  MicroAPI::Sub(remainder, value, product, mask);
}

template <size_t Index, typename First, typename... Rest>
__simd_callee__ inline uint32_t IndirectLoadSimdShapeValue(First first, Rest... rest) {
  static_assert(Index < sizeof...(Rest) + 1UL, "IndirectLoad SIMD shape index is invalid.");
  if constexpr (Index == 0UL) {
    return static_cast<uint32_t>(first);
  } else {
    return IndirectLoadSimdShapeValue<Index - 1UL>(rest...);
  }
}

template <int32_t Dim, int32_t Axis, int32_t Rank, typename... ShapeArgs>
__simd_callee__ inline void IndirectLoadSimdAddInnerOffset(MicroAPI::RegTensor<uint32_t> &source_index,
                                                           MicroAPI::RegTensor<uint32_t> &position,
                                                           MicroAPI::MaskReg mask, ShapeArgs... shape_args) {
  MicroAPI::RegTensor<uint32_t> quotient;
  MicroAPI::RegTensor<uint32_t> remainder;
  MicroAPI::RegTensor<uint32_t> input_offset;
  IndirectLoadSimdDivMod(quotient, remainder, position,
                         IndirectLoadSimdShapeValue<static_cast<size_t>(Dim)>(shape_args...), mask);
  MicroAPI::Muls(input_offset, remainder, IndirectLoadSimdShapeValue<static_cast<size_t>(Rank + Dim)>(shape_args...),
                 mask);
  MicroAPI::Add(source_index, source_index, input_offset, mask);
  if constexpr (Dim > Axis + 1) {
    IndirectLoadSimdAddInnerOffset<Dim - 1, Axis, Rank>(source_index, quotient, mask, shape_args...);
  }
}

template <int32_t Rank, int32_t Axis, typename... ShapeArgs>
__simd_callee__ inline void IndirectLoadSimdApplyAddress(MicroAPI::RegTensor<uint32_t> &source_index,
                                                         uint32_t repeat_base, uint32_t output_position,
                                                         uint32_t input_inner, uint32_t index_inner,
                                                         uint32_t input_slice_count, uint32_t output_slice_count,
                                                         bool inner_layout_matches, MicroAPI::MaskReg lane_mask,
                                                         ShapeArgs... shape_args) {
  if constexpr (Axis + 1 == Rank) {
    (void)source_index;
    (void)repeat_base;
    (void)output_position;
    (void)input_inner;
    (void)index_inner;
    (void)input_slice_count;
    (void)output_slice_count;
    (void)inner_layout_matches;
    (void)lane_mask;
    return;
  }
  MicroAPI::RegTensor<int32_t> signed_position;
  auto &position = (MicroAPI::RegTensor<uint32_t> &)signed_position;
  MicroAPI::RegTensor<uint32_t> outer;
  MicroAPI::RegTensor<uint32_t> divisor;
  MicroAPI::RegTensor<uint32_t> outer_offset;
  MicroAPI::Arange(signed_position, 0);
  MicroAPI::Adds(position, position, output_position + repeat_base, lane_mask);
  MicroAPI::Duplicate(divisor, output_slice_count, lane_mask);
  MicroAPI::Div(outer, position, divisor, lane_mask);
  MicroAPI::Muls(source_index, source_index, input_inner, lane_mask);
  MicroAPI::Muls(outer_offset, outer, input_slice_count, lane_mask);
  MicroAPI::Add(source_index, source_index, outer_offset, lane_mask);
  if (index_inner != 1U) {
    if (inner_layout_matches) {
      MicroAPI::RegTensor<uint32_t> quotient;
      MicroAPI::RegTensor<uint32_t> inner_offset;
      IndirectLoadSimdDivMod(quotient, inner_offset, position, index_inner, lane_mask);
      MicroAPI::Add(source_index, source_index, inner_offset, lane_mask);
    } else {
      IndirectLoadSimdAddInnerOffset<Rank - 1, Axis, Rank>(source_index, position, lane_mask, shape_args...);
    }
  }
}

template <typename X, typename Index, int32_t Rank, int32_t Axis>
struct IndirectLoadSimdRegTraits {
  using IndexPolicy = IndirectLoadSimdIndexPolicy<Index>;
  using ValuePolicy = IndirectLoadSimdValuePolicy<X>;
  static constexpr bool kSupported =
      IndexPolicy::kSupported && ValuePolicy::kSupported && Rank > 0 && Axis >= 0 && Axis < Rank;
  static constexpr uint32_t kElementsPerRepeat =
      sizeof(X) == sizeof(uint16_t) ? VECTOR_REG_WIDTH / sizeof(uint16_t) : IndexPolicy::kElementsPerRepeat;
};

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimdRegGather(__ubuf__ X *x, __ubuf__ Index *index, __ubuf__ X *y,
                                                 uint32_t actual_size,
                                                 const IndirectLoadSimdAddressContext &address_context,
                                                 ShapeArgs... shape_args) {
  using Traits = IndirectLoadSimdRegTraits<X, Index, Rank, Axis>;
  static_assert(Traits::kSupported, "IndirectLoad SIMD register Gather specialization is not implemented.");
  using IndexPolicy = typename Traits::IndexPolicy;
  using ValuePolicy = typename Traits::ValuePolicy;
  constexpr uint32_t elements_per_repeat = Traits::kElementsPerRepeat;
  const uint16_t repeat_count = static_cast<uint16_t>((actual_size + elements_per_repeat - 1U) / elements_per_repeat);
  uint32_t remaining = actual_size;
  __VEC_SCOPE__ {
    typename IndexPolicy::LoadState index_load_state;
    IndexPolicy::Init(index_load_state, index);
    for (uint16_t repeat = 0U; repeat < repeat_count; ++repeat) {
      const uint32_t element_count = remaining > elements_per_repeat ? elements_per_repeat : remaining;
      if constexpr (sizeof(X) == sizeof(uint16_t)) {
        MicroAPI::RegTensor<uint32_t> source_index0;
        MicroAPI::RegTensor<uint32_t> source_index1;
        MicroAPI::MaskReg lane_mask0;
        MicroAPI::MaskReg lane_mask1;
        IndexPolicy::LoadPair(source_index0, source_index1, lane_mask0, lane_mask1, index_load_state, element_count);
        IndirectLoadSimdApplyAddress<Rank, Axis>(
            source_index0, repeat * elements_per_repeat, address_context.output_position, address_context.input_inner,
            address_context.index_inner, address_context.input_slice_count, address_context.output_slice_count,
            address_context.inner_layout_matches, lane_mask0, shape_args...);
        IndirectLoadSimdApplyAddress<Rank, Axis>(
            source_index1, repeat * elements_per_repeat + VECTOR_REG_WIDTH / sizeof(uint32_t),
            address_context.output_position, address_context.input_inner, address_context.index_inner,
            address_context.input_slice_count, address_context.output_slice_count, address_context.inner_layout_matches,
            lane_mask1, shape_args...);
        ValuePolicy::GatherAndStorePair(x, y + repeat * elements_per_repeat, source_index0, source_index1, lane_mask0,
                                        lane_mask1, element_count, address_context.input_actual_size);
      } else {
        MicroAPI::RegTensor<uint32_t> source_index;
        uint32_t mask_count = element_count;
        MicroAPI::MaskReg lane_mask = MicroAPI::UpdateMask<uint32_t>(mask_count);
        MicroAPI::MaskReg valid_mask;
        IndexPolicy::Load(source_index, valid_mask, index_load_state, element_count, lane_mask);
        IndirectLoadSimdApplyAddress<Rank, Axis>(
            source_index, repeat * elements_per_repeat, address_context.output_position, address_context.input_inner,
            address_context.index_inner, address_context.input_slice_count, address_context.output_slice_count,
            address_context.inner_layout_matches, lane_mask, shape_args...);
        ValuePolicy::GatherAndStore(x, y + repeat * elements_per_repeat, source_index, lane_mask, valid_mask,
                                    element_count, address_context.input_actual_size);
      }
      remaining -= element_count;
    }
  }
}

template <typename X>
__simd_callee__ inline void IndirectLoadSimdStoreByteOffsets(__ubuf__ uint32_t *offsets,
                                                             MicroAPI::RegTensor<uint32_t> &source_index,
                                                             MicroAPI::MaskReg mask) {
  MicroAPI::Muls(source_index, source_index, static_cast<uint32_t>(sizeof(X)), mask);
  MicroAPI::DataCopy(offsets, source_index, mask);
}

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimdBuildOffsets(__ubuf__ Index *index, __ubuf__ uint32_t *offsets,
                                                    uint32_t actual_size,
                                                    const IndirectLoadSimdAddressContext &address_context,
                                                    ShapeArgs... shape_args) {
  using Traits = IndirectLoadSimdRegTraits<X, Index, Rank, Axis>;
  using IndexPolicy = typename Traits::IndexPolicy;
  constexpr uint32_t elements_per_repeat = Traits::kElementsPerRepeat;
  const uint16_t repeat_count = static_cast<uint16_t>((actual_size + elements_per_repeat - 1U) / elements_per_repeat);
  uint32_t remaining = actual_size;
  __VEC_SCOPE__ {
    typename IndexPolicy::LoadState state;
    if constexpr (sizeof(Index) != sizeof(uint32_t)) {
      IndexPolicy::Init(state, index);
    }
    for (uint16_t repeat = 0U; repeat < repeat_count; ++repeat) {
      const uint32_t count = remaining > elements_per_repeat ? elements_per_repeat : remaining;
      const uint32_t base = repeat * elements_per_repeat;
      if constexpr (sizeof(Index) == sizeof(uint32_t)) {
        IndexPolicy::Init(state, index + base);
      }
      MicroAPI::RegTensor<uint32_t> index0, index1;
      MicroAPI::MaskReg mask0, mask1;
      IndexPolicy::LoadPair(index0, index1, mask0, mask1, state, count);
      IndirectLoadSimdApplyAddress<Rank, Axis>(index0, base, address_context.output_position,
                                               address_context.input_inner, address_context.index_inner,
                                               address_context.input_slice_count, address_context.output_slice_count,
                                               address_context.inner_layout_matches, mask0, shape_args...);
      IndirectLoadSimdStoreByteOffsets<X>(offsets + base, index0, mask0);
      constexpr uint32_t elements_per_offset_reg = VECTOR_REG_WIDTH / sizeof(uint32_t);
      if (count > elements_per_offset_reg) {
        const uint32_t second_base = base + elements_per_offset_reg;
        IndirectLoadSimdApplyAddress<Rank, Axis>(index1, second_base, address_context.output_position,
                                                 address_context.input_inner, address_context.index_inner,
                                                 address_context.input_slice_count, address_context.output_slice_count,
                                                 address_context.inner_layout_matches, mask1, shape_args...);
        IndirectLoadSimdStoreByteOffsets<X>(offsets + second_base, index1, mask1);
      }
      remaining -= count;
    }
  }
}

template <int32_t Rank, int32_t Axis>
__aicore__ inline IndirectLoadSimdAddressContext InitIndirectLoadSimdAddressContext(int64_t output_offset,
                                                                                    uint32_t input_actual_size,
                                                                                    int64_t input_axis,
                                                                                    const int64_t *shape) {
  uint32_t index_inner = 1U;
  uint32_t expected_input_stride = 1U;
  bool inner_layout_matches = true;
  for (int32_t dim = Rank - 1; dim > Axis; --dim) {
    inner_layout_matches &= static_cast<uint32_t>(shape[Rank + dim]) == expected_input_stride;
    expected_input_stride *= static_cast<uint32_t>(shape[dim]);
    index_inner *= static_cast<uint32_t>(shape[dim]);
  }
  const uint32_t input_inner = static_cast<uint32_t>(shape[Rank + Axis]);
  const uint32_t input_slice = static_cast<uint32_t>(input_axis) * input_inner;
  const uint32_t output_slice = static_cast<uint32_t>(shape[Axis]) * index_inner;
  return {static_cast<uint32_t>(output_offset % static_cast<int64_t>(output_slice)),
          input_actual_size,
          input_inner,
          index_inner,
          input_slice,
          output_slice,
          inner_layout_matches};
}
}  // namespace Internal

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimd(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                        const LocalTensor<X> &y, uint32_t actual_size, int64_t output_offset,
                                        uint32_t input_actual_size, int64_t input_axis, ShapeArgs... shape_args) {
  static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMD rank or axis is invalid.");
  static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(2 * Rank), "IndirectLoad SIMD shape is invalid.");
  const int64_t shape[] = {static_cast<int64_t>(shape_args)...};
  __ubuf__ X *x_address = (__ubuf__ X *)x.GetPhyAddr();
  __ubuf__ Index *index_address = (__ubuf__ Index *)index.GetPhyAddr();
  __ubuf__ X *y_address = (__ubuf__ X *)y.GetPhyAddr();
  const Internal::IndirectLoadSimdAddressContext context =
      Internal::InitIndirectLoadSimdAddressContext<Rank, Axis>(output_offset, input_actual_size, input_axis, shape);
  Internal::IndirectLoadSimdRegGather<X, Index, Rank, Axis>(x_address, index_address, y_address, actual_size, context,
                                                            shape_args...);
}

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimdGatherApi(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                                 const LocalTensor<X> &y, uint32_t actual_size, int64_t output_offset,
                                                 uint32_t input_actual_size, int64_t input_axis,
                                                 ShapeArgs... shape_args) {
  static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMD rank or axis is invalid.");
  static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(2 * Rank), "IndirectLoad SIMD shape is invalid.");
  const int64_t shape[] = {static_cast<int64_t>(shape_args)...};
  // Gather consumes uint32 byte offsets, so reuse the dead index UB instead of allocating another buffer.
  __ubuf__ Index *index_address = (__ubuf__ Index *)index.GetPhyAddr();
  __ubuf__ uint32_t *offset_address = (__ubuf__ uint32_t *)index.GetPhyAddr();
  const Internal::IndirectLoadSimdAddressContext context =
      Internal::InitIndirectLoadSimdAddressContext<Rank, Axis>(output_offset, input_actual_size, input_axis, shape);
  Internal::IndirectLoadSimdBuildOffsets<X, Index, Rank, Axis>(index_address, offset_address, actual_size, context,
                                                               shape_args...);
  PipeBarrier<PIPE_V>();
  const LocalTensor<uint32_t> offsets = index.template ReinterpretCast<uint32_t>();
  Gather(y, x, offsets, 0U, actual_size);
}

}  // namespace AscendC

#endif  // AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMD_H_
