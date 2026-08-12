/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SK_H_
#define AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SK_H_

namespace AscendC {
namespace Internal {
template <int32_t Dim, int32_t Axis, int32_t StrideBase>
struct IndirectLoadSkInnerOffset {
  __aicore__ inline static int64_t Call(int64_t inner, const int64_t *shape) {
    int64_t offset = inner % shape[Dim] * shape[StrideBase + Dim];
    if constexpr (Dim > Axis + 1) {
      offset += IndirectLoadSkInnerOffset<Dim - 1, Axis, StrideBase>::Call(inner / shape[Dim], shape);
    }
    return offset;
  }
};

template <int32_t Dim, int32_t StrideBase>
struct IndirectLoadSkOuterOffset {
  __aicore__ inline static int64_t Call(int64_t outer, const int64_t *shape) {
    int64_t offset = outer % shape[Dim] * shape[StrideBase + Dim];
    if constexpr (Dim > 0) {
      offset += IndirectLoadSkOuterOffset<Dim - 1, StrideBase>::Call(outer / shape[Dim], shape);
    }
    return offset;
  }
};

template <int32_t Dim, int32_t Axis>
struct IndirectLoadSkInnerSize {
  __aicore__ inline static int64_t Call(const int64_t *shape) {
    if constexpr (Dim == Axis) {
      return 1;
    } else {
      return shape[Dim] * IndirectLoadSkInnerSize<Dim - 1, Axis>::Call(shape);
    }
  }
};

struct IndirectLoadSkParams {
  const int64_t *shape;
  int64_t index_inner;
  int64_t input_axis_stride;
  int64_t input_slice_count;
  int64_t output_slice_count;
  int64_t outer_begin;
  int64_t outer_count;
};

template <int32_t Rank, int32_t Axis>
__aicore__ inline IndirectLoadSkParams MakeIndirectLoadSkParams(const int64_t *shape, uint32_t actual_size,
                                                                int64_t output_offset, int64_t input_axis) {
  const int64_t index_inner = IndirectLoadSkInnerSize<Rank - 1, Axis>::Call(shape);
  int64_t input_inner_span = 1;
  for (int32_t dim = Axis + 1; dim < Rank; ++dim) {
    input_inner_span += (shape[dim] - 1) * shape[Rank + dim];
  }
  const int64_t input_axis_stride = shape[Rank + Axis];
  const int64_t input_slice_count = input_inner_span + (input_axis - 1) * input_axis_stride;
  const int64_t output_slice_count = shape[Axis] * index_inner;
  const int64_t outer_begin = output_offset / output_slice_count;
  const int64_t outer_end = (output_offset + actual_size + output_slice_count - 1) / output_slice_count;
  return {shape,
          index_inner,
          input_axis_stride,
          input_slice_count,
          output_slice_count,
          outer_begin,
          outer_end - outer_begin};
}

template <typename X, int32_t Rank, int32_t Axis>
__aicore__ inline void CopyIndirectLoadSkInput(const GlobalTensor<X> &x, const LocalTensor<X> &input,
                                               const IndirectLoadSkParams &params) {
  for (int64_t outer_local = 0; outer_local < params.outer_count; ++outer_local) {
    int64_t outer_linear = params.outer_begin + outer_local;
    int64_t input_offset = 0;
    for (int32_t dim = Axis - 1; dim >= 0; --dim) {
      const int64_t coord = outer_linear % params.shape[dim];
      outer_linear /= params.shape[dim];
      input_offset += coord * params.shape[Rank + dim];
    }
    DataCopyPadExtend<X, AscendC::PaddingMode::Normal>(input[outer_local * params.input_slice_count], x[input_offset],
                                                       1, params.input_slice_count, 0, 0);
  }
}

template <typename X, typename Index, int32_t Rank, int32_t Axis>
__aicore__ inline void BuildIndirectLoadSkOffsets(const LocalTensor<Index> &index, const LocalTensor<uint32_t> &offsets,
                                                  const IndirectLoadSkParams &params, uint32_t actual_size,
                                                  int64_t output_offset) {
  int64_t index_window_base = 0;
  if constexpr (Axis > 0) {
    index_window_base = IndirectLoadSkOuterOffset<Axis - 1, 2 * Rank>::Call(params.outer_begin, params.shape);
  }
  for (int64_t i = 0; i < actual_size; ++i) {
    const int64_t global_idx = output_offset + i;
    const int64_t outer_global = global_idx / params.output_slice_count;
    const int64_t outer_local = outer_global - params.outer_begin;
    const int64_t tail = global_idx % params.output_slice_count;
    const int64_t axis_coord = tail / params.index_inner;
    const int64_t inner = tail % params.index_inner;
    int64_t index_offset = axis_coord * params.shape[2 * Rank + Axis];
    if constexpr (Axis > 0) {
      index_offset +=
          IndirectLoadSkOuterOffset<Axis - 1, 2 * Rank>::Call(outer_global, params.shape) - index_window_base;
    }
    if constexpr (Axis + 1 < Rank) {
      index_offset += IndirectLoadSkInnerOffset<Rank - 1, Axis, 2 * Rank>::Call(inner, params.shape);
    }
    const int64_t index_value = static_cast<int64_t>(index.GetValue(index_offset));
    int64_t input_inner_offset = 0;
    if constexpr (Axis + 1 < Rank) {
      input_inner_offset = IndirectLoadSkInnerOffset<Rank - 1, Axis, Rank>::Call(inner, params.shape);
    }
    const int64_t src_idx =
        outer_local * params.input_slice_count + index_value * params.input_axis_stride + input_inner_offset;
    offsets.SetValue(i, static_cast<uint32_t>(src_idx * sizeof(X)));
  }
}
}  // namespace Internal

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSk(const GlobalTensor<X> &x, const LocalTensor<Index> &index,
                                      const LocalTensor<X> &y, const LocalTensor<uint8_t> &tmp, uint32_t actual_size,
                                      int64_t output_offset, int64_t input_axis, ShapeArgs... shape_args) {
  static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SK rank or axis is invalid.");
  static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(3 * Rank), "IndirectLoad SK shape is invalid.");
  const int64_t shape[] = {static_cast<int64_t>(shape_args)...};
  const auto params = Internal::MakeIndirectLoadSkParams<Rank, Axis>(shape, actual_size, output_offset, input_axis);
  constexpr int64_t kBlockBytes = 32;
  const int64_t input_bytes =
      (params.outer_count * params.input_slice_count * sizeof(X) + kBlockBytes - 1) / kBlockBytes * kBlockBytes;
  LocalTensor<X> input = tmp.template ReinterpretCast<X>();
  Internal::CopyIndirectLoadSkInput<X, Rank, Axis>(x, input, params);
  int32_t event_id = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
  LocalTensor<uint32_t> offsets = tmp[input_bytes].template ReinterpretCast<uint32_t>();
  Internal::BuildIndirectLoadSkOffsets<X, Index, Rank, Axis>(index, offsets, params, actual_size, output_offset);
  int32_t offset_event_id = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
  AscendC::SetFlag<AscendC::HardEvent::S_V>(offset_event_id);
  AscendC::WaitFlag<AscendC::HardEvent::S_V>(offset_event_id);
  Gather(y, input, offsets, static_cast<uint32_t>(0), actual_size);
  int32_t gather_event_id = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(gather_event_id);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(gather_event_id);
}
}  // namespace AscendC

#endif  // AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SK_H_
