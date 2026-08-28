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
template <int32_t Rank>
struct IndirectLoadSimdStridedParams {
  uint32_t logical_size;
  uint32_t physical_size;
  int64_t output_offset;
  int64_t index_sizes[Rank];
  int64_t input_strides[Rank];
  int64_t index_strides[Rank];
  int64_t output_strides[Rank];
};

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
    const int64_t input_row_stride = shape[2];
    const int64_t index_row_stride = shape[4];
    const bool full_rows = embedding_size > 0 && embedding_size % block_elements == 0 && shape[2] == embedding_size &&
                           shape[3] == 1 && index_row_stride > 0 && shape[5] == 0 &&
                           output_offset % embedding_size == 0 && actual_size % embedding_size == 0;
    if (full_rows) {
      const int64_t first_row = output_offset / embedding_size;
      const uint32_t row_count = actual_size / static_cast<uint32_t>(embedding_size);
      for (uint32_t row = 0; row < row_count; ++row) {
        const int64_t index_value = static_cast<int64_t>(index.GetValue((first_row + row) * index_row_stride));
        const int64_t source_offset = index_value * input_row_stride;
        AscendC::DataCopy(y[row * embedding_size], x[source_offset], static_cast<uint32_t>(embedding_size));
      }
      return true;
    }
  }
  return false;
}

template <int32_t Rank, int32_t Axis>
struct IndirectLoadSimdStridedContext {
  const int64_t *shape;
  const int64_t *output_strides;
  uint32_t logical_size;
  uint32_t physical_size;
  int64_t output_offset;
  int64_t index_inner;
  int64_t output_slice_count;
  int64_t input_window_base;
  int64_t index_window_base;
};

template <int32_t Rank>
__aicore__ inline void BuildIndirectLoadSimdStridedShape(int64_t (&shape)[3 * Rank],
                                                         const IndirectLoadSimdStridedParams<Rank> &params) {
  for (int32_t dim = 0; dim < Rank; ++dim) {
    shape[dim] = params.index_sizes[dim];
    shape[Rank + dim] = params.input_strides[dim];
    shape[2 * Rank + dim] = params.index_strides[dim];
  }
}

template <int32_t Rank, int32_t Axis>
__aicore__ inline IndirectLoadSimdStridedContext<Rank, Axis> MakeIndirectLoadSimdStridedContext(
    const int64_t *shape, const IndirectLoadSimdStridedParams<Rank> &params) {
  const int64_t index_inner = Internal::IndirectLoadSimdInnerSize<Rank - 1, Axis>::Call(shape);
  const int64_t output_slice_count = shape[Axis] * index_inner;
  const int64_t outer_begin = params.output_offset / output_slice_count;
  int64_t input_window_base = 0;
  int64_t index_window_base = 0;
  if constexpr (Axis > 0) {
    input_window_base = Internal::IndirectLoadSimdOuterOffset<Axis - 1, Rank>::Call(outer_begin, shape);
    index_window_base = Internal::IndirectLoadSimdOuterOffset<Axis - 1, 2 * Rank>::Call(outer_begin, shape);
  }
  return {shape,       params.output_strides, params.logical_size, params.physical_size, params.output_offset,
          index_inner, output_slice_count,    input_window_base,   index_window_base};
}

template <int32_t Rank, int32_t Axis>
__aicore__ inline int64_t GetIndirectLoadSimdStridedIndexOffset(int64_t global_idx,
                                                                const IndirectLoadSimdStridedContext<Rank, Axis> &ctx) {
  const int64_t outer_global = global_idx / ctx.output_slice_count;
  const int64_t tail = global_idx % ctx.output_slice_count;
  const int64_t axis_coord = tail / ctx.index_inner;
  const int64_t inner = tail % ctx.index_inner;
  int64_t index_offset = axis_coord * ctx.shape[2 * Rank + Axis];
  if constexpr (Axis > 0) {
    index_offset += Internal::IndirectLoadSimdOuterOffset<Axis - 1, 2 * Rank>::Call(outer_global, ctx.shape) -
                    ctx.index_window_base;
  }
  if constexpr (Axis + 1 < Rank) {
    index_offset += Internal::IndirectLoadSimdInnerOffset<Rank - 1, Axis, 2 * Rank>::Call(inner, ctx.shape);
  }
  return index_offset;
}

template <int32_t Rank, int32_t Axis>
__aicore__ inline int64_t GetIndirectLoadSimdStridedInputOffset(int64_t global_idx, int64_t index_value,
                                                                const IndirectLoadSimdStridedContext<Rank, Axis> &ctx) {
  const int64_t outer_global = global_idx / ctx.output_slice_count;
  const int64_t tail = global_idx % ctx.output_slice_count;
  const int64_t inner = tail % ctx.index_inner;
  int64_t input_inner_offset = 0;
  int64_t input_outer_offset = 0;
  if constexpr (Axis + 1 < Rank) {
    input_inner_offset = Internal::IndirectLoadSimdInnerOffset<Rank - 1, Axis, Rank>::Call(inner, ctx.shape);
  }
  if constexpr (Axis > 0) {
    input_outer_offset =
        Internal::IndirectLoadSimdOuterOffset<Axis - 1, Rank>::Call(outer_global, ctx.shape) - ctx.input_window_base;
  }
  return input_outer_offset + index_value * ctx.shape[Rank + Axis] + input_inner_offset;
}

template <typename X, typename Index, int32_t Rank, int32_t Axis>
__aicore__ inline void BuildIndirectLoadSimdStridedOffsets(const LocalTensor<Index> &index,
                                                           const LocalTensor<uint32_t> &offsets,
                                                           const IndirectLoadSimdStridedContext<Rank, Axis> &ctx) {
  for (uint32_t i = 0; i < ctx.logical_size; ++i) {
    const int64_t global_idx = ctx.output_offset + static_cast<int64_t>(i);
    const int64_t index_offset = GetIndirectLoadSimdStridedIndexOffset<Rank, Axis>(global_idx, ctx);
    const int64_t index_value = static_cast<int64_t>(index.GetValue(index_offset));
    const int64_t source_offset = GetIndirectLoadSimdStridedInputOffset<Rank, Axis>(global_idx, index_value, ctx);
    offsets.SetValue(i, static_cast<uint32_t>(source_offset * sizeof(X)));
  }
}

template <int32_t Rank, int32_t Axis>
__aicore__ inline int64_t GetIndirectLoadSimdStridedOutputOffset(
    int64_t global_idx, const IndirectLoadSimdStridedContext<Rank, Axis> &ctx) {
  int64_t current = global_idx;
  int64_t base = ctx.output_offset;
  int64_t output_offset = 0;
  for (int32_t dim = Rank - 1; dim >= 0; --dim) {
    const int64_t coord = current % ctx.shape[dim];
    const int64_t base_coord = base % ctx.shape[dim];
    current /= ctx.shape[dim];
    base /= ctx.shape[dim];
    output_offset += (coord - base_coord) * ctx.output_strides[dim];
  }
  return output_offset;
}

template <typename X, int32_t Rank, int32_t Axis>
__aicore__ inline void ScatterIndirectLoadSimdStridedOutput(const LocalTensor<X> &y,
                                                            const IndirectLoadSimdStridedContext<Rank, Axis> &ctx) {
  // Gather writes packed logical values first. Repack from the end so padding lanes do not
  // overwrite a source value that is still needed by the supported compact output layouts.
  for (int64_t i = static_cast<int64_t>(ctx.logical_size) - 1; i >= 0; --i) {
    const int64_t output_offset = GetIndirectLoadSimdStridedOutputOffset<Rank, Axis>(ctx.output_offset + i, ctx);
    y.SetValue(output_offset, y.GetValue(i));
  }
}

// Strided SIMD output may contain alignment holes (for example, a logical 23-element row
// is stored with a physical stride of 24). Keep the gather packed and expand it into the
// padded destination layout afterwards.
template <typename X, typename Index, int32_t Rank, int32_t Axis>
__aicore__ inline void IndirectLoadSimdStridedImpl(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                                   const LocalTensor<X> &y, const LocalTensor<uint8_t> &tmp,
                                                   const IndirectLoadSimdStridedParams<Rank> &params) {
  static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMD rank or axis is invalid.");
  int64_t shape[3 * Rank];
  BuildIndirectLoadSimdStridedShape(shape, params);
  const auto context = MakeIndirectLoadSimdStridedContext<Rank, Axis>(shape, params);
  LocalTensor<uint32_t> offsets = tmp.template ReinterpretCast<uint32_t>();
  BuildIndirectLoadSimdStridedOffsets<X, Index, Rank, Axis>(index, offsets, context);
  int32_t offset_event_id = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
  AscendC::SetFlag<AscendC::HardEvent::S_V>(offset_event_id);
  AscendC::WaitFlag<AscendC::HardEvent::S_V>(offset_event_id);
  Gather(y, x, offsets, static_cast<uint32_t>(0), context.logical_size);
  ScatterIndirectLoadSimdStridedOutput<X, Rank, Axis>(y, context);
}
}  // namespace Internal

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimd(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                        const LocalTensor<X> &y, uint32_t actual_size, int64_t output_offset,
                                        uint32_t input_actual_size, int64_t input_axis, ShapeArgs... shape_args) {
  static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMD rank or axis is invalid.");
  static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(2 * Rank), "IndirectLoad SIMD shape is invalid.");
  Internal::IndirectLoadSimdDenseImpl<X, Index, Rank, Axis>(x, index, y, actual_size, output_offset, input_actual_size,
                                                            input_axis, shape_args...);
}

template <typename X, typename Index, int32_t Rank, int32_t Axis>
__aicore__ inline void IndirectLoadSimdStrided(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                               const LocalTensor<X> &y, const LocalTensor<uint8_t> &tmp,
                                               const IndirectLoadSimdStridedParams<Rank> &params) {
  static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMD rank or axis is invalid.");
  Internal::IndirectLoadSimdStridedImpl<X, Index, Rank, Axis>(x, index, y, tmp, params);
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
