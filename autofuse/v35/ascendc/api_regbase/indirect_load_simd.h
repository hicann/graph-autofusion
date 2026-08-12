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

#include <type_traits>

#ifndef AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMD_POLICY_H_
#include "indirect_load_simd_policy.h"
#endif

namespace AscendC {
namespace Internal {
template <int32_t Dim, int32_t Axis, int32_t StrideBase>
struct IndirectLoadSimdInnerOffset {
  __aicore__ inline static int64_t Call(int64_t inner, const int64_t *shape) {
    int64_t offset = inner % shape[Dim] * shape[StrideBase + Dim];
    if constexpr (Dim > Axis + 1) {
      offset += IndirectLoadSimdInnerOffset<Dim - 1, Axis, StrideBase>::Call(inner / shape[Dim], shape);
    }
    return offset;
  }
};

template <int32_t Dim, int32_t StrideBase>
struct IndirectLoadSimdOuterOffset {
  __aicore__ inline static int64_t Call(int64_t outer, const int64_t *shape) {
    int64_t offset = outer % shape[Dim] * shape[StrideBase + Dim];
    if constexpr (Dim > 0) {
      offset += IndirectLoadSimdOuterOffset<Dim - 1, StrideBase>::Call(outer / shape[Dim], shape);
    }
    return offset;
  }
};

template <int32_t Dim, int32_t Axis>
struct IndirectLoadSimdInnerSize {
  __aicore__ inline static int64_t Call(const int64_t *shape) {
    if constexpr (Dim == Axis) {
      return 1;
    } else {
      return shape[Dim] * IndirectLoadSimdInnerSize<Dim - 1, Axis>::Call(shape);
    }
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

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimdDenseImpl(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                                 const LocalTensor<X> &y, uint32_t actual_size, int64_t output_offset,
                                                 uint32_t input_actual_size, int64_t input_axis,
                                                 ShapeArgs... shape_args) {
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
__aicore__ inline void IndirectLoadSimdStridedImpl(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                                   const LocalTensor<X> &y, const LocalTensor<uint8_t> &tmp,
                                                   uint32_t actual_size, int64_t output_offset,
                                                   ShapeArgs... shape_args) {
  static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMD rank or axis is invalid.");
  static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(3 * Rank), "IndirectLoad SIMD shape is invalid.");
  const int64_t shape[] = {static_cast<int64_t>(shape_args)...};
  const int64_t index_inner = Internal::IndirectLoadSimdInnerSize<Rank - 1, Axis>::Call(shape);
  const int64_t output_slice_count = shape[Axis] * index_inner;
  const int64_t outer_begin = output_offset / output_slice_count;
  int64_t input_window_base = 0;
  int64_t index_window_base = 0;
  if constexpr (Axis > 0) {
    input_window_base = Internal::IndirectLoadSimdOuterOffset<Axis - 1, Rank>::Call(outer_begin, shape);
    index_window_base = Internal::IndirectLoadSimdOuterOffset<Axis - 1, 2 * Rank>::Call(outer_begin, shape);
  }
  LocalTensor<uint32_t> offsets = tmp.template ReinterpretCast<uint32_t>();
  for (int64_t i = 0; i < actual_size; ++i) {
    const int64_t global_idx = output_offset + i;
    const int64_t outer_global = global_idx / output_slice_count;
    const int64_t tail = global_idx % output_slice_count;
    const int64_t axis_coord = tail / index_inner;
    const int64_t inner = tail % index_inner;
    int64_t index_offset = axis_coord * shape[2 * Rank + Axis];
    if constexpr (Axis > 0) {
      index_offset +=
          Internal::IndirectLoadSimdOuterOffset<Axis - 1, 2 * Rank>::Call(outer_global, shape) - index_window_base;
    }
    if constexpr (Axis + 1 < Rank) {
      index_offset += Internal::IndirectLoadSimdInnerOffset<Rank - 1, Axis, 2 * Rank>::Call(inner, shape);
    }
    const int64_t index_value = static_cast<int64_t>(index.GetValue(index_offset));
    int64_t input_inner_offset = 0;
    if constexpr (Axis + 1 < Rank) {
      input_inner_offset = Internal::IndirectLoadSimdInnerOffset<Rank - 1, Axis, Rank>::Call(inner, shape);
    }
    int64_t input_outer_offset = 0;
    if constexpr (Axis > 0) {
      input_outer_offset =
          Internal::IndirectLoadSimdOuterOffset<Axis - 1, Rank>::Call(outer_global, shape) - input_window_base;
    }
    const int64_t src_idx = input_outer_offset + index_value * shape[Rank + Axis] + input_inner_offset;
    offsets.SetValue(i, static_cast<uint32_t>(src_idx * sizeof(X)));
  }
  int32_t offset_event_id = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
  AscendC::SetFlag<AscendC::HardEvent::S_V>(offset_event_id);
  AscendC::WaitFlag<AscendC::HardEvent::S_V>(offset_event_id);
  Gather(y, x, offsets, static_cast<uint32_t>(0), actual_size);
}
}  // namespace Internal

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename FirstArg, typename... Args>
__aicore__ inline void IndirectLoadSimd(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                        const LocalTensor<X> &y, FirstArg first_arg, Args... args) {
  static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMD rank or axis is invalid.");
  constexpr bool has_tmp = std::is_same_v<std::decay_t<FirstArg>, LocalTensor<uint8_t>>;
  if constexpr (has_tmp) {
    static_assert(sizeof...(Args) == static_cast<size_t>(2 + 3 * Rank),
                  "IndirectLoad SIMD strided arguments are invalid.");
    Internal::IndirectLoadSimdStridedImpl<X, Index, Rank, Axis>(x, index, y, first_arg, args...);
  } else {
    static_assert(sizeof...(Args) == static_cast<size_t>(3 + 2 * Rank),
                  "IndirectLoad SIMD dense arguments are invalid.");
    Internal::IndirectLoadSimdDenseImpl<X, Index, Rank, Axis>(x, index, y, first_arg, args...);
  }
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
