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
template <int32_t Dim, int32_t Axis, int32_t Rank>
struct IndirectLoadSimdInnerOffset {
  __aicore__ inline static int64_t Call(int64_t inner, const int64_t *shape) {
    int64_t offset = inner % shape[Dim] * shape[Rank + Dim];
    if constexpr (Dim > Axis + 1) {
      offset += IndirectLoadSimdInnerOffset<Dim - 1, Axis, Rank>::Call(inner / shape[Dim], shape);
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
}  // namespace Internal

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimd(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                        const LocalTensor<X> &y, const LocalTensor<uint8_t> &tmp, uint32_t actual_size,
                                        int64_t output_offset, int64_t input_axis, ShapeArgs... shape_args) {
  static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMD rank or axis is invalid.");
  static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(2 * Rank), "IndirectLoad SIMD shape is invalid.");
  const int64_t shape[] = {static_cast<int64_t>(shape_args)...};
  const int64_t index_inner = Internal::IndirectLoadSimdInnerSize<Rank - 1, Axis>::Call(shape);
  const int64_t input_inner = shape[Rank + Axis];
  const int64_t input_slice_count = input_axis * input_inner;
  const int64_t output_slice_count = shape[Axis] * index_inner;
  const int64_t outer_begin = output_offset / output_slice_count;
  LocalTensor<uint32_t> offsets = tmp.template ReinterpretCast<uint32_t>();
  for (int64_t i = 0; i < actual_size; ++i) {
    const int64_t global_idx = output_offset + i;
    const int64_t outer_local = global_idx / output_slice_count - outer_begin;
    const int64_t index_value = static_cast<int64_t>(index.GetValue(i));
    int64_t input_inner_offset = 0;
    if constexpr (Axis + 1 < Rank) {
      input_inner_offset =
          Internal::IndirectLoadSimdInnerOffset<Rank - 1, Axis, Rank>::Call(global_idx % index_inner, shape);
    }
    const int64_t src_idx = outer_local * input_slice_count + index_value * input_inner + input_inner_offset;
    offsets.SetValue(i, static_cast<uint32_t>(src_idx * sizeof(X)));
  }
  Gather(y, x, offsets, static_cast<uint32_t>(0), actual_size);
}
}  // namespace AscendC

#endif  // AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMD_H_
