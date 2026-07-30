/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMT_H_
#define AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMT_H_

namespace AscendC {
namespace Internal {
constexpr uint32_t kIndirectLoadSimtThreadNum = 1024U;

template <int32_t Dim, int32_t Axis, int32_t Rank>
struct IndirectLoadSimtOffset {
  __simt_callee__ __aicore__ inline static int64_t Call(int64_t linear_index, int64_t indirect_index,
                                                        const int64_t *shape) {
    int64_t input_offset = 0;
    if constexpr (Dim == Axis) {
      input_offset = indirect_index * shape[Rank + Dim];
    } else {
      input_offset = linear_index % shape[Dim] * shape[Rank + Dim];
    }
    if constexpr (Dim > 0) {
      input_offset +=
          IndirectLoadSimtOffset<Dim - 1, Axis, Rank>::Call(linear_index / shape[Dim], indirect_index, shape);
    }
    return input_offset;
  }
};

template <typename X, typename Index, typename Y, int32_t Rank, int32_t Axis, typename IndexTransform,
          typename OutputTransform, typename... ShapeArgs>
__simt_vf__ __aicore__ LAUNCH_BOUND(kIndirectLoadSimtThreadNum) inline void IndirectLoadSimtKernel(
    __gm__ X *x, __gm__ Index *index, __gm__ Y *y, uint32_t actual_size, int64_t output_offset, int64_t x_axis_size,
    ShapeArgs... shape_args) {
  static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMT rank or axis is invalid.");
  static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(2 * Rank), "IndirectLoad SIMT shape is invalid.");
  const int64_t shape[] = {static_cast<int64_t>(shape_args)...};
  for (uint32_t i = threadIdx.x; i < actual_size; i += blockDim.x) {
    const int64_t output_index = output_offset + i;
    const int64_t indirect_index = static_cast<int64_t>(IndexTransform::Call(index[output_index]));
    if (unlikely(indirect_index < 0 || indirect_index >= x_axis_size)) {
      y[output_index] = static_cast<Y>(0);
      continue;
    }
    const int64_t input_offset =
        IndirectLoadSimtOffset<Rank - 1, Axis, Rank>::Call(output_index, indirect_index, shape);
    y[output_index] = OutputTransform::Call(x[input_offset]);
  }
}
}  // namespace Internal

template <typename X, typename Index, typename Y, int32_t Rank, int32_t Axis, typename IndexTransform,
          typename OutputTransform, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimt(__gm__ X *x, __gm__ Index *index, __gm__ Y *y, uint32_t actual_size,
                                        int64_t output_offset, int64_t x_axis_size, ShapeArgs... shape_args) {
  Simt::VF_CALL<
      Internal::IndirectLoadSimtKernel<X, Index, Y, Rank, Axis, IndexTransform, OutputTransform, ShapeArgs...>>(
      Simt::Dim3(Internal::kIndirectLoadSimtThreadNum), x, index, y, actual_size, output_offset, x_axis_size,
      shape_args...);
  const int32_t event_id = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
  SetFlag<HardEvent::V_MTE3>(event_id);
  WaitFlag<HardEvent::V_MTE3>(event_id);
}
}  // namespace AscendC

#endif  // AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMT_H_
