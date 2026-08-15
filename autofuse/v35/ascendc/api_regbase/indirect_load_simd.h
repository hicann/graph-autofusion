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

template <typename X>
struct IndirectLoadSimdGatherAction {
  template <typename ValuePolicy>
  __simd_callee__ inline static void Commit(__ubuf__ X *x, __ubuf__ X *y, __ubuf__ uint32_t *, uint32_t input_size,
                                            uint32_t base, MicroAPI::RegTensor<uint32_t> &source_index,
                                            MicroAPI::MaskReg mask, uint32_t count) {
    ValuePolicy::GatherAndStore(x, y + base, source_index, mask, mask, count, input_size);
  }

  template <typename ValuePolicy>
  __simd_callee__ inline static void CommitPair(__ubuf__ X *x, __ubuf__ X *y, __ubuf__ uint32_t *, uint32_t input_size,
                                                uint32_t base, MicroAPI::RegTensor<uint32_t> &index0,
                                                MicroAPI::RegTensor<uint32_t> &index1, MicroAPI::MaskReg mask0,
                                                MicroAPI::MaskReg mask1, uint32_t count) {
    ValuePolicy::GatherAndStorePair(x, y + base, index0, index1, mask0, mask1, count, input_size);
  }
};

template <typename X>
struct IndirectLoadSimdOffsetAction {
  template <typename ValuePolicy>
  __simd_callee__ inline static void Commit(__ubuf__ X *, __ubuf__ X *, __ubuf__ uint32_t *offsets, uint32_t,
                                            uint32_t base, MicroAPI::RegTensor<uint32_t> &source_index,
                                            MicroAPI::MaskReg mask, uint32_t) {
    IndirectLoadSimdStoreByteOffsets<X>(offsets + base, source_index, mask);
  }

  template <typename ValuePolicy>
  __simd_callee__ inline static void CommitPair(__ubuf__ X *, __ubuf__ X *, __ubuf__ uint32_t *offsets, uint32_t,
                                                uint32_t base, MicroAPI::RegTensor<uint32_t> &index0,
                                                MicroAPI::RegTensor<uint32_t> &index1, MicroAPI::MaskReg mask0,
                                                MicroAPI::MaskReg mask1, uint32_t count) {
    constexpr uint32_t elements_per_reg = VECTOR_REG_WIDTH / sizeof(uint32_t);
    IndirectLoadSimdStoreByteOffsets<X>(offsets + base, index0, mask0);
    if (count > elements_per_reg) {
      IndirectLoadSimdStoreByteOffsets<X>(offsets + base + elements_per_reg, index1, mask1);
    }
  }
};

template <typename X, typename Index>
struct IndirectLoadSimdArgs {
  __ubuf__ X *x;
  __ubuf__ Index *index;
  __ubuf__ X *y;
  __ubuf__ uint32_t *offsets;
  uint32_t actual_size;
};

template <IndirectLoadSimdAddressMode Mode, typename X, typename Index, int32_t Rank, int32_t Axis, typename Action,
          typename... ShapeArgs>
__simd_callee__ inline void IndirectLoadSimdRunRepeat(IndirectLoadSimdArgs<X, Index> &args,
                                                      typename IndirectLoadSimdIndexPolicy<Index>::LoadState &state,
                                                      uint32_t base, uint32_t count, MicroAPI::MaskReg mask,
                                                      MicroAPI::RegTensor<uint32_t> &input_inner,
                                                      MicroAPI::RegTensor<uint32_t> &address_invariant,
                                                      const IndirectLoadSimdAddressContext &context,
                                                      ShapeArgs... shape_args) {
  using Traits = IndirectLoadSimdModeTraits<Mode, X, Index, Rank, Axis>;
  using IndexPolicy = typename Traits::IndexPolicy;
  using ValuePolicy = typename Traits::ValuePolicy;
  MicroAPI::RegTensor<uint32_t> source_index;
  IndexPolicy::Load(source_index, state, count);
  IndirectLoadSimdApplyAddress<Mode, Rank, Axis>(source_index, base, context, input_inner, address_invariant, mask,
                                                 shape_args...);
  Action::template Commit<ValuePolicy>(args.x, args.y, args.offsets, context.input_actual_size, base, source_index,
                                       mask, count);
}

template <typename X, typename Index, typename Action>
__simd_callee__ inline void IndirectLoadSimdRunPair(IndirectLoadSimdArgs<X, Index> &args,
                                                    typename IndirectLoadSimdIndexPolicy<Index>::LoadState &state,
                                                    uint32_t base, uint32_t count,
                                                    const IndirectLoadSimdAddressContext &context) {
  using Traits = IndirectLoadSimdRegTraits<X, Index, 1, 0>;
  using IndexPolicy = typename Traits::IndexPolicy;
  using ValuePolicy = typename Traits::ValuePolicy;
  MicroAPI::RegTensor<uint32_t> index0, index1;
  MicroAPI::MaskReg mask0, mask1;
  IndexPolicy::LoadPair(index0, index1, mask0, mask1, state, count);
  Action::template CommitPair<ValuePolicy>(args.x, args.y, args.offsets, context.input_actual_size, base, index0,
                                           index1, mask0, mask1, count);
}

template <IndirectLoadSimdAddressMode Mode, typename X, typename Index, int32_t Rank, int32_t Axis, typename Action,
          typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimdRunMode(IndirectLoadSimdArgs<X, Index> &args,
                                               const IndirectLoadSimdAddressContext &context, ShapeArgs... shape_args) {
  using Traits = IndirectLoadSimdModeTraits<Mode, X, Index, Rank, Axis>;
  using IndexPolicy = typename Traits::IndexPolicy;
  constexpr uint32_t elements_per_repeat = Traits::kElementsPerRepeat;
  const uint16_t full_repeats = static_cast<uint16_t>(args.actual_size / elements_per_repeat);
  const uint32_t tail_count = args.actual_size % elements_per_repeat;
  __VEC_SCOPE__ {
    typename IndexPolicy::LoadState state;
    IndexPolicy::Init(state, args.index);
    if constexpr (sizeof(X) == sizeof(uint16_t) && Mode == IndirectLoadSimdAddressMode::kDirect) {
      for (uint16_t repeat = 0U; repeat < full_repeats; ++repeat) {
        const uint32_t base = static_cast<uint32_t>(repeat) * elements_per_repeat;
        IndirectLoadSimdRunPair<X, Index, Action>(args, state, base, elements_per_repeat, context);
      }
      if (tail_count != 0U) {
        IndirectLoadSimdRunPair<X, Index, Action>(args, state, full_repeats * elements_per_repeat, tail_count, context);
      }
    } else {
      MicroAPI::MaskReg full_mask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
      MicroAPI::RegTensor<uint32_t> input_inner;
      MicroAPI::RegTensor<uint32_t> address_invariant;
      IndirectLoadSimdInitInvariants<Mode>(input_inner, address_invariant, context, full_mask);
      for (uint16_t repeat = 0U; repeat < full_repeats; ++repeat) {
        const uint32_t base = static_cast<uint32_t>(repeat) * elements_per_repeat;
        IndirectLoadSimdRunRepeat<Mode, X, Index, Rank, Axis, Action>(
            args, state, base, elements_per_repeat, full_mask, input_inner, address_invariant, context, shape_args...);
      }
      if (tail_count != 0U) {
        uint32_t mask_count = tail_count;
        MicroAPI::MaskReg tail_mask = MicroAPI::UpdateMask<uint32_t>(mask_count);
        IndirectLoadSimdRunRepeat<Mode, X, Index, Rank, Axis, Action>(args, state, full_repeats * elements_per_repeat,
                                                                      tail_count, tail_mask, input_inner,
                                                                      address_invariant, context, shape_args...);
      }
    }
  }
}

template <typename X, typename Index, typename Action>
__aicore__ inline void IndirectLoadSimdRunReuse(IndirectLoadSimdArgs<X, Index> &args,
                                                const IndirectLoadSimdAddressContext &context) {
  using Traits = IndirectLoadSimdRegTraits<X, Index, 1, 0>;
  using IndexPolicy = typename Traits::IndexPolicy;
  using ValuePolicy = typename Traits::ValuePolicy;
  constexpr uint32_t elements_per_repeat = VECTOR_REG_WIDTH / sizeof(uint32_t);
  const uint16_t full_repeats = static_cast<uint16_t>(args.actual_size / elements_per_repeat);
  const uint32_t tail_count = args.actual_size % elements_per_repeat;
  __VEC_SCOPE__ {
    typename IndexPolicy::LoadState state;
    IndexPolicy::Init(state, args.index);
    MicroAPI::RegTensor<uint32_t> inner_offset;
    MicroAPI::RegTensor<uint32_t> input_inner;
    MicroAPI::MaskReg full_mask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    IndirectLoadSimdInitInnerOffset(inner_offset, context, full_mask);
    MicroAPI::Duplicate(input_inner, context.input_inner, full_mask);
    for (uint16_t repeat = 0U; repeat < full_repeats; ++repeat) {
      const uint32_t base = repeat * elements_per_repeat;
      MicroAPI::RegTensor<uint32_t> source_index;
      IndexPolicy::Load(source_index, state, elements_per_repeat);
      MicroAPI::Mul(source_index, source_index, input_inner, full_mask);
      MicroAPI::Add(source_index, source_index, inner_offset, full_mask);
      Action::template Commit<ValuePolicy>(args.x, args.y, args.offsets, context.input_actual_size, base, source_index,
                                           full_mask, elements_per_repeat);
    }
    if (tail_count != 0U) {
      const uint32_t base = full_repeats * elements_per_repeat;
      uint32_t mask_count = tail_count;
      MicroAPI::MaskReg tail_mask = MicroAPI::UpdateMask<uint32_t>(mask_count);
      MicroAPI::RegTensor<uint32_t> source_index;
      IndexPolicy::Load(source_index, state, tail_count);
      MicroAPI::Mul(source_index, source_index, input_inner, tail_mask);
      MicroAPI::Add(source_index, source_index, inner_offset, tail_mask);
      Action::template Commit<ValuePolicy>(args.x, args.y, args.offsets, context.input_actual_size, base, source_index,
                                           tail_mask, tail_count);
    }
  }
}

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename Action>
struct IndirectLoadSimdDispatchPolicy {
  using Args = IndirectLoadSimdArgs<X, Index>;

  __aicore__ inline static void RunReuse(Args &args, const IndirectLoadSimdAddressContext &context) {
    IndirectLoadSimdRunReuse<X, Index, Action>(args, context);
  }

  template <IndirectLoadSimdAddressMode Mode, typename... ShapeArgs>
  __aicore__ inline static void RunMode(Args &args, const IndirectLoadSimdAddressContext &context,
                                        ShapeArgs... shape_args) {
    IndirectLoadSimdRunMode<Mode, X, Index, Rank, Axis, Action>(args, context, shape_args...);
  }
};

template <int32_t Rank, int32_t Axis, typename DispatchPolicy, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimdDispatch(typename DispatchPolicy::Args &args,
                                                const IndirectLoadSimdAddressContext &context,
                                                ShapeArgs... shape_args) {
  constexpr uint32_t reuse_elements = VECTOR_REG_WIDTH / sizeof(uint32_t);
  if constexpr (Axis + 1 == Rank) {
    DispatchPolicy::template RunMode<IndirectLoadSimdAddressMode::kDirect>(args, context, shape_args...);
    return;
  }
  if (!context.inner_layout_matches) {
    DispatchPolicy::template RunMode<IndirectLoadSimdAddressMode::kStrided>(args, context, shape_args...);
    return;
  }
  if (IndirectLoadSimdIsPowerOfTwo(context.index_inner)) {
    if (args.actual_size > reuse_elements && (reuse_elements & (context.index_inner - 1U)) == 0U) {
      DispatchPolicy::RunReuse(args, context);
    } else {
      DispatchPolicy::template RunMode<IndirectLoadSimdAddressMode::kDensePow2>(args, context, shape_args...);
    }
    return;
  }
  DispatchPolicy::template RunMode<IndirectLoadSimdAddressMode::kDenseGeneric>(args, context, shape_args...);
}

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimdRegGather(__ubuf__ X *x, __ubuf__ Index *index, __ubuf__ X *y,
                                                 uint32_t actual_size, const IndirectLoadSimdAddressContext &context,
                                                 ShapeArgs... shape_args) {
  static_assert(IndirectLoadSimdRegTraits<X, Index, Rank, Axis>::kSupported,
                "IndirectLoad SIMD register Gather specialization is not implemented.");
  IndirectLoadSimdArgs<X, Index> args{x, index, y, nullptr, actual_size};
  using Policy = IndirectLoadSimdDispatchPolicy<X, Index, Rank, Axis, IndirectLoadSimdGatherAction<X>>;
  IndirectLoadSimdDispatch<Rank, Axis, Policy>(args, context, shape_args...);
}

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimdBuildOffsets(__ubuf__ Index *index, __ubuf__ uint32_t *offsets,
                                                    uint32_t actual_size, const IndirectLoadSimdAddressContext &context,
                                                    ShapeArgs... shape_args) {
  IndirectLoadSimdArgs<X, Index> args{nullptr, index, nullptr, offsets, actual_size};
  using Policy = IndirectLoadSimdDispatchPolicy<X, Index, Rank, Axis, IndirectLoadSimdOffsetAction<X>>;
  IndirectLoadSimdDispatch<Rank, Axis, Policy>(args, context, shape_args...);
}

template <int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline IndirectLoadSimdAddressContext InitIndirectLoadSimdAddressContext(int64_t output_offset,
                                                                                    uint32_t input_actual_size,
                                                                                    int64_t input_axis,
                                                                                    ShapeArgs... shape_args) {
  (void)input_axis;
  if constexpr (Axis + 1 == Rank) {
    return {0U, input_actual_size, 1U, 1U, true};
  }
  const int64_t shape[] = {static_cast<int64_t>(shape_args)...};
  uint32_t index_inner = 1U;
  uint32_t expected_input_stride = 1U;
  bool inner_layout_matches = true;
  for (int32_t dim = Rank - 1; dim > Axis; --dim) {
    inner_layout_matches &= static_cast<uint32_t>(shape[Rank + dim]) == expected_input_stride;
    expected_input_stride *= static_cast<uint32_t>(shape[dim]);
    index_inner *= static_cast<uint32_t>(shape[dim]);
  }
  const uint32_t input_inner = static_cast<uint32_t>(shape[Rank + Axis]);
  const uint32_t output_slice = static_cast<uint32_t>(shape[Axis]) * index_inner;
  return {static_cast<uint32_t>(output_offset % static_cast<int64_t>(output_slice)), input_actual_size, input_inner,
          index_inner, inner_layout_matches};
}

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimdDenseImpl(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                                 const LocalTensor<X> &y, uint32_t actual_size, int64_t output_offset,
                                                 uint32_t input_actual_size, int64_t input_axis,
                                                 ShapeArgs... shape_args) {
  static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMD rank or axis is invalid.");
  static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(2 * Rank), "IndirectLoad SIMD shape is invalid.");
  __ubuf__ X *x_address = (__ubuf__ X *)x.GetPhyAddr();
  __ubuf__ Index *index_address = (__ubuf__ Index *)index.GetPhyAddr();
  __ubuf__ X *y_address = (__ubuf__ X *)y.GetPhyAddr();
  const Internal::IndirectLoadSimdAddressContext context = Internal::InitIndirectLoadSimdAddressContext<Rank, Axis>(
      output_offset, input_actual_size, input_axis, shape_args...);
  Internal::IndirectLoadSimdRegGather<X, Index, Rank, Axis>(x_address, index_address, y_address, actual_size, context,
                                                            shape_args...);
}

template <typename X, typename Index, int32_t Rank, int32_t Axis>
__aicore__ inline bool TryIndirectLoadSimdEmbedding(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                                    const LocalTensor<X> &y, uint32_t actual_size,
                                                    int64_t output_offset, const int64_t (&shape)[3 * Rank]) {
  if constexpr (Rank == 2 && Axis == 0 && (std::is_same_v<Index, int32_t> || std::is_same_v<Index, int64_t>) &&
                sizeof(X) <= AscendC::ONE_BLK_SIZE) {
    const int64_t embedding_size = shape[1];
    const int64_t block_elements = static_cast<int64_t>(AscendC::ONE_BLK_SIZE / sizeof(X));
    const bool full_rows = embedding_size > 0 && embedding_size % block_elements == 0 && shape[2] == embedding_size &&
                           shape[3] == 1 && shape[4] == 1 && shape[5] == 0 && output_offset % embedding_size == 0 &&
                           actual_size % embedding_size == 0;
    if (full_rows) {
      const int64_t first_row = output_offset / embedding_size;
      const uint32_t row_count = actual_size / static_cast<uint32_t>(embedding_size);
      for (uint32_t row = 0; row < row_count; ++row) {
        const int64_t index_value = static_cast<int64_t>(index.GetValue(first_row + row));
        const int64_t source_offset = index_value * embedding_size;
        AscendC::DataCopy(y[row * embedding_size], x[source_offset], static_cast<uint32_t>(embedding_size));
      }
      return true;
    }
  }
  return false;
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
  if (TryIndirectLoadSimdEmbedding<X, Index, Rank, Axis>(x, index, y, actual_size, output_offset, shape)) {
    return;
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
  // Gather consumes uint32 byte offsets, so reuse the dead index UB instead of allocating another buffer.
  __ubuf__ Index *index_address = (__ubuf__ Index *)index.GetPhyAddr();
  __ubuf__ uint32_t *offset_address = (__ubuf__ uint32_t *)index.GetPhyAddr();
  const Internal::IndirectLoadSimdAddressContext context = Internal::InitIndirectLoadSimdAddressContext<Rank, Axis>(
      output_offset, input_actual_size, input_axis, shape_args...);
  Internal::IndirectLoadSimdBuildOffsets<X, Index, Rank, Axis>(index_address, offset_address, actual_size, context,
                                                               shape_args...);
  PipeBarrier<PIPE_V>();
  const LocalTensor<uint32_t> offsets = index.template ReinterpretCast<uint32_t>();
  Gather(y, x, offsets, 0U, actual_size);
}

}  // namespace AscendC

#endif  // AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMD_H_
