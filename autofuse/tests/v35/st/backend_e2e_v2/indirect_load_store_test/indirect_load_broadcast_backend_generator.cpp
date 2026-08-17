/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <sstream>

#include "common_utils.h"
#include "indirect_load_backend_generator_common.h"

#ifndef IL_CLEAR_BROADCAST_SOURCE_VIEW
#define IL_CLEAR_BROADCAST_SOURCE_VIEW 0
#endif
#ifndef IL_INPUT_BROADCAST
#define IL_INPUT_BROADCAST 0
#endif
#ifndef IL_INDEX_BROADCAST
#define IL_INDEX_BROADCAST 0
#endif
#ifndef IL_AIC_REPRO
#define IL_AIC_REPRO 0
#endif
#ifndef IL_COMPLEX_BROADCAST
#define IL_COMPLEX_BROADCAST 0
#endif
#ifndef IL_COMPLEX_SIMT
#define IL_COMPLEX_SIMT 0
#endif
#ifndef IL_RETAIN_BROADCAST
#define IL_RETAIN_BROADCAST 0
#endif
#ifndef IL_DEGENERATE_BROADCAST
#define IL_DEGENERATE_BROADCAST 0
#endif
#ifndef IL_CONTINUOUS_BROADCAST
#define IL_CONTINUOUS_BROADCAST 0
#endif
#ifndef IL_CONTINUOUS_INDEX_BROADCAST
#define IL_CONTINUOUS_INDEX_BROADCAST 0
#endif
#ifndef IL_BROADCAST_POST_REDUCE
#define IL_BROADCAST_POST_REDUCE 0
#endif
#ifndef IL_OUTPUT_S0
#define IL_OUTPUT_S0 4
#endif
#ifndef IL_OUTPUT_S1
#define IL_OUTPUT_S1 5
#endif
#ifndef IL_OUTPUT_S2
#define IL_OUTPUT_S2 4
#endif
#ifndef IL_OUTPUT_S3
#define IL_OUTPUT_S3 16
#endif
namespace {
constexpr int64_t kOutputS0 = IL_OUTPUT_S0;
constexpr int64_t kOutputS1 = IL_OUTPUT_S1;
constexpr int64_t kOutputS2 = IL_OUTPUT_S2;
constexpr int64_t kOutputS3 = IL_OUTPUT_S3;
constexpr std::array<int64_t, 4> kOutputShape = {kOutputS0, kOutputS1, kOutputS2, kOutputS3};
constexpr bool kComplexBroadcast = IL_COMPLEX_BROADCAST;
constexpr bool kComplexSimt = IL_COMPLEX_SIMT;
constexpr bool kRetainBroadcast = IL_RETAIN_BROADCAST;
constexpr bool kDegenerateBroadcast = IL_DEGENERATE_BROADCAST;
constexpr bool kContinuousBroadcast = IL_CONTINUOUS_BROADCAST;
constexpr bool kContinuousIndexBroadcast = IL_CONTINUOUS_INDEX_BROADCAST;
constexpr int32_t kInputElementCount = IL_HAS_INPUT_ELEMENT;
constexpr int32_t kIndexElementCount = IL_HAS_INDEX_ELEMENT;
constexpr bool kHasOutputRelu = IL_HAS_OUTPUT_RELU;
constexpr bool kInputBroadcast = IL_INPUT_BROADCAST && !kComplexBroadcast;
constexpr bool kIndexBroadcast = IL_INDEX_BROADCAST;
constexpr uint32_t kBroadcastAxesMask = IL_BROADCAST_AXES_MASK;
constexpr bool kClearBroadcastSourceView = IL_CLEAR_BROADCAST_SOURCE_VIEW;
constexpr bool kExpectSimt = IL_EXPECT_SIMT;
constexpr bool kExpectSk = IL_EXPECT_SK;
constexpr bool kAicRepro = IL_AIC_REPRO;

using indirect_load_test::SetView;

struct TensorView {
  std::vector<af::AxisId> axes;
  std::vector<af::Expression> repeats;
  std::vector<af::Expression> strides;
  af::DataType dtype;
};

struct BroadcastGraphView {
  std::shared_ptr<af::AscGraph> graph;
  TensorView input;
  TensorView output;
  TensorView input_source;
  TensorView index_source;
  TensorView input_broadcast;
  TensorView input_intermediate_broadcast;
  TensorView index_broadcast;
  TensorView index_intermediate_broadcast;
};

template <typename Op>
void SetView(Op &op, const TensorView &view) {
  SetView(op, view.axes, view.repeats, view.strides, view.dtype);
}

template <typename Op>
void ClearView(Op &op) {
  op.y.axis->clear();
  op.y.repeats->clear();
  op.y.strides->clear();
}

std::vector<af::Expression> MakeDenseStrides(const std::vector<af::Expression> &repeats) {
  std::vector<af::Expression> strides(repeats.size(), af::ops::One);
  af::Expression stride = af::ops::One;
  for (size_t index = repeats.size(); index > 0UL; --index) {
    const size_t dim = index - 1UL;
    strides[dim] = stride;
    stride = stride * repeats[dim];
  }
  return strides;
}

TensorView MakeIntermediateBroadcastView(const TensorView &broadcast_view, const TensorView &logical_view) {
  constexpr size_t kFirstBroadcastAxis = 3UL;
  TensorView view = broadcast_view;
  view.repeats[kFirstBroadcastAxis] = logical_view.repeats[kFirstBroadcastAxis];
  view.strides = MakeDenseStrides(view.repeats);
  for (size_t dim = 0UL; dim < view.strides.size(); ++dim) {
    if ((kBroadcastAxesMask & (1U << dim)) != 0U &&
        af::SymbolicUtils::StaticCheckEq(view.repeats[dim], af::ops::One) == af::TriBool::kTrue) {
      view.strides[dim] = af::ops::Zero;
    }
  }
  return view;
}

std::array<int64_t, 4> MakeLogicalStrides(bool broadcast) {
  std::array<int64_t, 4> strides{};
  int64_t stride = 1;
  for (size_t index = kOutputShape.size(); index > 0UL; --index) {
    const size_t dim = index - 1UL;
    const bool is_broadcast_axis = broadcast && (kBroadcastAxesMask & (1U << dim)) != 0U;
    strides[dim] = is_broadcast_axis ? 0 : stride;
    if (!is_broadcast_axis) {
      stride *= kOutputShape[dim];
    }
  }
  return strides;
}

std::string MakeShapeArgs(bool is_sk) {
  const auto input_strides = MakeLogicalStrides(kInputBroadcast && !kDegenerateBroadcast);
  const auto index_strides = MakeLogicalStrides(kIndexBroadcast && !kDegenerateBroadcast);
  std::ostringstream stream;
  if (is_sk) {
    stream << ", 4";
  }
  for (const int64_t size : kOutputShape) {
    stream << ", " << size;
  }
  for (const int64_t stride : input_strides) {
    stream << ", " << stride;
  }
  for (const int64_t stride : index_strides) {
    stream << ", " << stride;
  }
  stream << ");";
  return stream.str();
}

BroadcastGraphView CreateGraphView() {
  BroadcastGraphView view;
  view.graph = std::make_shared<af::AscGraph>("indirect_load_broadcast_test");
  std::vector<af::AxisId> input_source_axis_candidates;
  std::vector<af::AxisId> index_source_axis_candidates;
  for (size_t dim = 0UL; dim < kOutputShape.size(); ++dim) {
    const auto input_size = view.graph->CreateSizeVar(kOutputShape[dim]);
    const auto output_size = view.graph->CreateSizeVar(kOutputShape[dim]);
    view.input.repeats.emplace_back(input_size);
    view.output.repeats.emplace_back(output_size);
    view.input.axes.emplace_back(view.graph->CreateAxis(("z" + std::to_string(dim)).c_str(), input_size).id);
    view.output.axes.emplace_back(view.graph->CreateAxis(("z" + std::to_string(dim + 4UL)).c_str(), output_size).id);
    input_source_axis_candidates.emplace_back(
        view.graph->CreateAxis(("z" + std::to_string(dim) + "_input").c_str(), af::ops::One).id);
    index_source_axis_candidates.emplace_back(
        view.graph->CreateAxis(("z" + std::to_string(dim + 4UL) + "_index").c_str(), af::ops::One).id);
  }
  view.input.dtype = af::DT_FLOAT16;
  view.output.dtype = af::DT_INT64;
  view.input_source = view.input;
  view.index_source = view.output;
  for (size_t dim = 0UL; dim < kOutputShape.size(); ++dim) {
    if ((kBroadcastAxesMask & (1U << dim)) != 0U) {
      view.input_source.axes[dim] = input_source_axis_candidates[dim];
      view.index_source.axes[dim] = index_source_axis_candidates[dim];
      view.input_source.repeats[dim] = af::ops::One;
      view.index_source.repeats[dim] = af::ops::One;
    }
  }
  view.input.strides = MakeDenseStrides(view.input.repeats);
  view.output.strides = MakeDenseStrides(view.output.repeats);
  view.input_source.strides = MakeDenseStrides(view.input_source.repeats);
  view.index_source.strides = MakeDenseStrides(view.index_source.repeats);
  view.input_broadcast = view.input;
  view.index_broadcast = view.output;
  for (size_t dim = 0UL; dim < kOutputShape.size(); ++dim) {
    view.input_broadcast.strides[dim] =
        (kBroadcastAxesMask & (1U << dim)) == 0U ? view.input_source.strides[dim] : af::ops::Zero;
    view.index_broadcast.strides[dim] =
        (kBroadcastAxesMask & (1U << dim)) == 0U ? view.index_source.strides[dim] : af::ops::Zero;
  }
  view.input_intermediate_broadcast = view.input_broadcast;
  if constexpr (kContinuousBroadcast) {
    view.input_intermediate_broadcast = MakeIntermediateBroadcastView(view.input_broadcast, view.input);
  }
  view.index_intermediate_broadcast = view.index_broadcast;
  if constexpr (kContinuousIndexBroadcast) {
    view.index_intermediate_broadcast = MakeIntermediateBroadcastView(view.index_broadcast, view.output);
  }
  return view;
}

template <typename Destination>
void ConnectAbsChain(const std::shared_ptr<af::AscGraph> &graph, const char *prefix, int32_t count,
                     const af::AscOpOutput &source, const TensorView &view, Destination &destination) {
  std::vector<std::unique_ptr<af::ascir_op::Abs>> elements;
  for (int32_t i = 0; i < count; ++i) {
    const auto name = std::string(prefix) + std::to_string(i);
    auto element = std::make_unique<af::ascir_op::Abs>(name.c_str());
    graph->AddNode(*element);
    element->attr.api.compute_type = af::ComputeType::kComputeElewise;
    element->x = elements.empty() ? source : elements.back()->y;
    SetView(*element, view);
    elements.emplace_back(std::move(element));
  }
  destination = elements.empty() ? source : elements.back()->y;
}

void BuildInputPath(const BroadcastGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  if constexpr (kComplexSimt) {
    af::ascir_op::Data x("x");
    af::ascir_op::Load input_load("input_load");
    view.graph->AddNode(x);
    view.graph->AddNode(input_load);
    x.ir_attr.SetIndex(0);
    input_load.x = x.y;
    SetView(x, view.input);
    SetView(input_load, view.input);
    indirect_load.x1 = input_load.y;
    return;
  }
  constexpr bool use_broadcast = kInputBroadcast || kComplexBroadcast;
  af::ascir_op::Data x("x");
  view.graph->AddNode(x);
  x.ir_attr.SetIndex(0);
  SetView(x, use_broadcast ? view.input_source : view.input);
  if (kClearBroadcastSourceView && kInputBroadcast) {
    ClearView(x);
  }
  af::ascir_op::Load input_load("input_load");
  view.graph->AddNode(input_load);
  input_load.x = x.y;
  SetView(input_load, use_broadcast ? view.input_source : view.input);
  if (use_broadcast) {
    if constexpr (kContinuousBroadcast) {
      af::ascir_op::Broadcast first_broadcast("input_first_broadcast");
      af::ascir_op::Broadcast second_broadcast("input_second_broadcast");
      view.graph->AddNode(first_broadcast);
      view.graph->AddNode(second_broadcast);
      first_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
      second_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
      first_broadcast.x = input_load.y;
      second_broadcast.x = first_broadcast.y;
      SetView(first_broadcast, view.input_intermediate_broadcast);
      SetView(second_broadcast, view.input_broadcast);
      ConnectAbsChain(view.graph, "input_abs_", kInputElementCount, second_broadcast.y, view.input_broadcast,
                      indirect_load.x1);
      return;
    }
    af::ascir_op::Broadcast broadcast("input_broadcast");
    view.graph->AddNode(broadcast);
    broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
    broadcast.x = input_load.y;
    SetView(broadcast, view.input_broadcast);
    if constexpr (kComplexBroadcast) {
      af::ascir_op::Add input_add("input_add");
      view.graph->AddNode(input_add);
      input_add.x1 = broadcast.y;
      input_add.x2 = input_load.y;
      SetView(input_add, view.input_broadcast);
      indirect_load.x1 = input_add.y;
    } else {
      ConnectAbsChain(view.graph, "input_abs_", kInputElementCount, broadcast.y, view.input_broadcast,
                      indirect_load.x1);
    }
  } else {
    indirect_load.x1 = input_load.y;
  }
}

void BuildIndexPath(const BroadcastGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Data index("index");
  view.graph->AddNode(index);
  index.ir_attr.SetIndex(1);
  SetView(index, kIndexBroadcast ? view.index_source : view.output);
  if (kClearBroadcastSourceView && kIndexBroadcast) {
    ClearView(index);
  }
  af::ascir_op::Load index_load("index_load");
  view.graph->AddNode(index_load);
  index_load.x = index.y;
  SetView(index_load, kIndexBroadcast ? view.index_source : view.output);
  if (kIndexBroadcast) {
    if constexpr (kContinuousIndexBroadcast) {
      af::ascir_op::Broadcast first_broadcast("index_first_broadcast");
      af::ascir_op::Broadcast second_broadcast("index_second_broadcast");
      view.graph->AddNode(first_broadcast);
      view.graph->AddNode(second_broadcast);
      first_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
      second_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
      first_broadcast.x = index_load.y;
      second_broadcast.x = first_broadcast.y;
      SetView(first_broadcast, view.index_intermediate_broadcast);
      SetView(second_broadcast, view.index_broadcast);
      ConnectAbsChain(view.graph, "index_abs_", kIndexElementCount, second_broadcast.y, view.index_broadcast,
                      indirect_load.x2);
      return;
    }
    af::ascir_op::Broadcast broadcast("index_broadcast");
    view.graph->AddNode(broadcast);
    broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
    broadcast.x = index_load.y;
    SetView(broadcast, view.index_broadcast);
    ConnectAbsChain(view.graph, "index_abs_", kIndexElementCount, broadcast.y, view.index_broadcast, indirect_load.x2);
  } else {
    ConnectAbsChain(view.graph, "index_abs_", kIndexElementCount, index_load.y, view.output, indirect_load.x2);
  }
}

void BuildOutputPath(const BroadcastGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
#if IL_BROADCAST_POST_REDUCE
  af::ascir_op::Sum sum("output_sum");
  af::ascir_op::Store store("store");
  af::ascir_op::Output output("y");
  view.graph->AddNode(sum);
  view.graph->AddNode(store);
  view.graph->AddNode(output);
  indirect_load.ir_attr.SetAxis(2);
  SetView(indirect_load, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.attr.sched.axis = view.output.axes;
  sum.x = indirect_load.y;
  auto reduce_repeats = view.output.repeats;
  auto reduce_strides = view.output.strides;
  reduce_repeats[2] = af::ops::One;
  reduce_repeats[3] = af::ops::One;
  reduce_strides[0] = view.output.repeats[1];
  reduce_strides[1] = af::ops::One;
  reduce_strides[2] = af::ops::Zero;
  reduce_strides[3] = af::ops::Zero;
  SetView(sum, view.output.axes, reduce_repeats, reduce_strides, af::DT_FLOAT16);
  store.x = sum.y;
  SetView(store, view.output.axes, reduce_repeats, reduce_strides, af::DT_FLOAT16);
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetView(output, view.output.axes, reduce_repeats, reduce_strides, af::DT_FLOAT16);
  return;
#endif
  if constexpr (kRetainBroadcast) {
    TensorView source_view = view.output;
    source_view.dtype = af::DT_FLOAT16;
    for (size_t dim = 0UL; dim < source_view.repeats.size(); ++dim) {
      if ((kBroadcastAxesMask & (1U << dim)) != 0U) {
        source_view.repeats[dim] = af::ops::One;
      }
    }
    source_view.strides = MakeDenseStrides(source_view.repeats);
    af::ascir_op::Scalar source("output_source", *view.graph);
    source.ir_attr.SetValue("1.5");
    source.y.dtype = af::DT_FLOAT16;
    af::ascir_op::Abs source_abs("output_source_abs");
    af::ascir_op::Broadcast broadcast("output_retained_broadcast");
    af::ascir_op::Add output_add("output_add");
    af::ascir_op::Store store("store");
    af::ascir_op::Output output("y");
    view.graph->AddNode(source_abs);
    view.graph->AddNode(broadcast);
    view.graph->AddNode(output_add);
    view.graph->AddNode(store);
    view.graph->AddNode(output);
    source_abs.x = source.y;
    broadcast.x = source_abs.y;
    broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
    output_add.x1 = indirect_load.y;
    output_add.x2 = broadcast.y;
    indirect_load.ir_attr.SetAxis(2);
    SetView(indirect_load, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    SetView(source_abs, source_view);
    SetView(broadcast, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    SetView(output_add, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    store.x = output_add.y;
    SetView(store, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    output.x = store.y;
    output.ir_attr.SetIndex(0);
    SetView(output, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    return;
  }
  if constexpr (kComplexBroadcast) {
    af::ascir_op::Scalar scalar0("output_scalar0", *view.graph);
    af::ascir_op::Scalar scalar1("output_scalar1", *view.graph);
    scalar0.ir_attr.SetValue("0.5");
    scalar1.ir_attr.SetValue("1.0");
    scalar0.y.dtype = af::DT_FLOAT16;
    scalar1.y.dtype = af::DT_FLOAT16;
    af::ascir_op::Broadcast broadcast0("output_broadcast0");
    af::ascir_op::Broadcast broadcast1("output_broadcast1");
    af::ascir_op::Add scalar_add("output_scalar_add");
    af::ascir_op::Add output_add("output_add");
    af::ascir_op::Store store("store");
    af::ascir_op::Output output("y");
    for (af::ascir_op::Broadcast *broadcast : {&broadcast0, &broadcast1}) {
      view.graph->AddNode(*broadcast);
      broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
      SetView(*broadcast, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    }
    view.graph->AddNode(scalar_add);
    view.graph->AddNode(output_add);
    view.graph->AddNode(store);
    view.graph->AddNode(output);
    broadcast0.x = scalar0.y;
    broadcast1.x = scalar1.y;
    scalar_add.x1 = broadcast0.y;
    scalar_add.x2 = broadcast1.y;
    output_add.x1 = indirect_load.y;
    output_add.x2 = scalar_add.y;
    indirect_load.ir_attr.SetAxis(2);
    SetView(indirect_load, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    SetView(scalar_add, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    SetView(output_add, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    store.x = output_add.y;
    SetView(store, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    output.x = store.y;
    output.ir_attr.SetIndex(0);
    SetView(output, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    return;
  }
  indirect_load_test::BuildOutputPath(view.graph, indirect_load, view.output.axes, view.output.repeats,
                                      view.output.strides, kHasOutputRelu);
}

void BuildComplexIndexPath(const BroadcastGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Data index("index");
  af::ascir_op::Load index_load("index_load");
  view.graph->AddNode(index);
  view.graph->AddNode(index_load);
  index.ir_attr.SetIndex(1);
  index_load.x = index.y;
  SetView(index, view.index_source);
  SetView(index_load, view.index_source);

  af::ascir_op::Scalar scalar0("index_scalar0", *view.graph);
  af::ascir_op::Scalar scalar1("index_scalar1", *view.graph);
  scalar0.ir_attr.SetValue("0");
  scalar1.ir_attr.SetValue("0");
  scalar0.y.dtype = af::DT_INT64;
  scalar1.y.dtype = af::DT_INT64;
  af::ascir_op::Broadcast broadcast0("index_broadcast0");
  af::ascir_op::Broadcast broadcast1("index_broadcast1");
  af::ascir_op::Add scalar_add("index_scalar_add");
  af::ascir_op::Add index_add("index_add");
  af::ascir_op::Broadcast final_broadcast("index_final_broadcast");
  for (af::ascir_op::Broadcast *broadcast : {&broadcast0, &broadcast1}) {
    view.graph->AddNode(*broadcast);
    broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
    SetView(*broadcast, view.index_source);
  }
  view.graph->AddNode(scalar_add);
  view.graph->AddNode(index_add);
  view.graph->AddNode(final_broadcast);
  final_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast0.x = scalar0.y;
  broadcast1.x = scalar1.y;
  scalar_add.x1 = broadcast0.y;
  scalar_add.x2 = broadcast1.y;
  index_add.x1 = index_load.y;
  index_add.x2 = scalar_add.y;
  final_broadcast.x = index_add.y;
  SetView(scalar_add, view.index_source);
  SetView(index_add, view.index_source);
  SetView(final_broadcast, view.index_broadcast);
  indirect_load.x2 = final_broadcast.y;
}

af::ComputeGraphPtr CreateGraph() {
  indirect_load_test::BackendGraph backend("indirect_load_broadcast_test", "data0", "data1", af::DT_FLOAT16);
  const auto build_index = kComplexBroadcast ? BuildComplexIndexPath : BuildIndexPath;
  return backend.Finalize(
      indirect_load_test::CreateSubGraph(CreateGraphView(), BuildInputPath, build_index, BuildOutputPath), "output");
}

af::ComputeGraphPtr CreateAicReproGraph() {
  auto graph = std::make_shared<af::AscGraph>("indirect_load_aic_repro");
  const af::Expression input_s0 = graph->CreateSizeVar(100000);
  const af::Expression output_s0 = graph->CreateSizeVar(1024);
  const af::Expression s1 = graph->CreateSizeVar(1024);
  const af::Expression one = af::ops::One;
  const af::AxisId input_axis0 = graph->CreateAxis("x0", input_s0).id;
  const af::AxisId input_axis1 = graph->CreateAxis("x1", s1).id;
  const af::AxisId output_axis0 = graph->CreateAxis("y0", output_s0).id;
  const af::AxisId output_axis1 = graph->CreateAxis("y1", s1).id;

  af::ascir_op::Data input("input");
  af::ascir_op::Load input_load("input_load");
  graph->AddNode(input);
  graph->AddNode(input_load);
  input.ir_attr.SetIndex(0);
  input_load.x = input.y;
  SetView(input, {input_axis0, input_axis1}, {input_s0, s1}, {s1, one}, af::DT_FLOAT);
  SetView(input_load, {input_axis0, input_axis1}, {input_s0, s1}, {s1, one}, af::DT_FLOAT);

  af::ascir_op::Data index("index");
  af::ascir_op::Load index_load("index_load");
  af::ascir_op::Broadcast index_broadcast("index_broadcast");
  graph->AddNode(index);
  graph->AddNode(index_load);
  graph->AddNode(index_broadcast);
  index.ir_attr.SetIndex(1);
  index_load.x = index.y;
  index_broadcast.x = index_load.y;
  index_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  SetView(index, {output_axis0, output_axis1}, {output_s0, one}, {one, af::ops::Zero}, af::DT_INT64);
  SetView(index_load, {output_axis0, output_axis1}, {output_s0, one}, {one, af::ops::Zero}, af::DT_INT64);
  SetView(index_broadcast, {output_axis0, output_axis1}, {output_s0, s1}, {s1, one}, af::DT_INT64);

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  af::ascir_op::Store store("store");
  af::ascir_op::Output output("output");
  graph->AddNode(indirect_load);
  graph->AddNode(store);
  graph->AddNode(output);
  indirect_load.x1 = input_load.y;
  indirect_load.x2 = index_broadcast.y;
  indirect_load.ir_attr.SetAxis(0);
  SetView(indirect_load, {output_axis0, output_axis1}, {output_s0, s1}, {s1, one}, af::DT_FLOAT);
  store.x = indirect_load.y;
  SetView(store, {output_axis0, output_axis1}, {output_s0, s1}, {s1, one}, af::DT_FLOAT);
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetView(output, {output_axis0, output_axis1}, {output_s0, s1}, {s1, one}, af::DT_FLOAT);

  indirect_load_test::BackendGraph backend("indirect_load_aic_repro", "data0", "data1", af::DT_FLOAT);
  return backend.Finalize(graph, "output");
}

void CheckSkKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find("// IndirectLoad SK"), std::string::npos);
  EXPECT_EQ(kernel.find("// IndirectLoad SIMD"), std::string::npos);
  EXPECT_EQ(kernel.find("// IndirectLoad SIMT"), std::string::npos);
  EXPECT_EQ(kernel.find("BroadcastExtend<"), std::string::npos);
  EXPECT_NE(kernel.find(MakeShapeArgs(true)), std::string::npos);
}

void CheckSimtKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find("// IndirectLoad SIMT"), std::string::npos);
  EXPECT_NE(kernel.find("IndirectLoadSimt<"), std::string::npos);
  EXPECT_EQ(kernel.find("// IndirectLoad SIMD"), std::string::npos);
  if constexpr (kOutputS0 == 4 && kOutputS1 == 5 && kOutputS2 == 4 && kOutputS3 == 16) {
    EXPECT_NE(kernel.find(MakeShapeArgs(false)), std::string::npos);
  }
  EXPECT_EQ(kernel.find("AscendC::BroadcastExtend<"), std::string::npos);
}

void CheckSimdElements(const std::string &kernel) {
  const auto input_abs_pos = kernel.find("Abs(");
  const auto indirect_load_pos = kernel.find("// IndirectLoad SIMD");
  ASSERT_NE(indirect_load_pos, std::string::npos);
  if (kInputElementCount > 0 || kIndexElementCount > 0) {
    ASSERT_NE(input_abs_pos, std::string::npos);
    EXPECT_LT(input_abs_pos, indirect_load_pos);
  } else {
    EXPECT_EQ(input_abs_pos, std::string::npos);
  }
  const auto output_relu_pos = kernel.find("Relu(");
  EXPECT_EQ(output_relu_pos == std::string::npos, !kHasOutputRelu);
}

void CheckSimdKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find("// IndirectLoad SIMD"), std::string::npos);
  EXPECT_NE(kernel.find("IndirectLoadSimd<"), std::string::npos);
  if constexpr (kDegenerateBroadcast) {
    EXPECT_NE(kernel.find(", 20, 10, 10, 20, 20, 4000, 400, 20, 1);"), std::string::npos);
    EXPECT_NE(kernel.find("Duplicate(local_7[0], local_6.GetValue(0), local_7_actual_size);"), std::string::npos);
    EXPECT_NE(kernel.find("Duplicate(local_9[0], local_8.GetValue(0), local_9_actual_size);"), std::string::npos);
    EXPECT_NE(kernel.find("const uint32_t local_7_actual_size = (400 - 1) + 1;"), std::string::npos);
    EXPECT_NE(kernel.find("const uint32_t local_9_actual_size = (400 - 1) + 1;"), std::string::npos);
  } else {
#if IL_BROADCAST_POST_REDUCE
    EXPECT_NE(kernel.find("ReduceSum"), std::string::npos);
#else
    EXPECT_NE(kernel.find(MakeShapeArgs(false)), std::string::npos);
#endif
  }
  if constexpr (!kDegenerateBroadcast && !IL_BROADCAST_POST_REDUCE) {
    EXPECT_NE(kernel.find("const int64_t indirect_load_outert_axis_size = 1;"), std::string::npos);
    EXPECT_NE(kernel.find("block_dim_offset = indirect_load_outerTB * t->indirect_load_outerTb_size"),
              std::string::npos);
  }
  if (!kIndexBroadcast && kBroadcastAxesMask == 2U) {
    EXPECT_NE(kernel.find("global_0[(int64_t)z4 * (int64_t)64 + 0 + 0]"), std::string::npos);
    EXPECT_EQ(kernel.find("global_0[(int64_t)z4 * (int64_t)64 + (int64_t)z5 * (int64_t)64"), std::string::npos);
  }
  EXPECT_EQ(kernel.find("AscendC::BroadcastExtend<"), std::string::npos);
  CheckSimdElements(kernel);
}

void CheckGeneratedKernel(const std::string &kernel, ascir::TemplateId template_id) {
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    CheckSkKernel(kernel);
  } else if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    CheckSimtKernel(kernel);
  } else {
    CheckSimdKernel(kernel);
  }
}

const af::Axis *FindDerivedAxis(const af::AscGraph &graph, af::Axis::Type type, af::AxisId from) {
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis->type == type && axis->from == std::vector<af::AxisId>{from}) {
      return axis.get();
    }
  }
  return nullptr;
}

void ExpectComplexNodeSchedule(const af::AscNodePtr &node, const std::vector<af::AxisId> &axes,
                               af::AxisId vectorized_axis) {
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->attr.sched.axis, axes) << node->GetName();
  ASSERT_GE(axes.size(), 2UL);
  EXPECT_EQ(node->attr.sched.loop_axis, axes[axes.size() - 2UL]) << node->GetName();
  ASSERT_FALSE(node->outputs().empty());
  EXPECT_EQ(node->outputs()[0]->attr.vectorized_axis, std::vector<af::AxisId>{vectorized_axis}) << node->GetName();
}

void ExpectComplexOuterAxes(af::AscGraph &graph, const ascgen_utils::indirect_load::TemplateAxes &axes,
                            const ascgen_utils::indirect_load::TemplateLogicalView &view,
                            std::vector<af::AxisId> &outer_loops) {
  const af::Axis *outer = graph.FindAxis(axes.outer_axis);
  const af::Axis *inner = graph.FindAxis(axes.inner_axis);
  const af::Axis *input_inner = graph.FindAxis(axes.input_inner_axis);
  const af::Axis *index_inner = graph.FindAxis(axes.index_inner_axis);
  ASSERT_NE(outer, nullptr);
  ASSERT_NE(inner, nullptr);
  ASSERT_NE(input_inner, nullptr);
  ASSERT_NE(index_inner, nullptr);
  EXPECT_EQ(outer->from, std::vector<af::AxisId>({view.output.axis_ids[0], view.output.axis_ids[1]}));
  EXPECT_EQ(inner->from, std::vector<af::AxisId>({view.output.axis_ids[2], view.output.axis_ids[3]}));
  EXPECT_EQ(input_inner->from, std::vector<af::AxisId>({view.input.axis_ids[2], view.input.axis_ids[3]}));
  EXPECT_EQ(index_inner->from, std::vector<af::AxisId>({view.output.axis_ids[2], view.output.axis_ids[3]}));
  const af::Axis *tile_outer = FindDerivedAxis(graph, af::Axis::Type::kAxisTypeTileOuter, outer->id);
  const af::Axis *tile_inner = FindDerivedAxis(graph, af::Axis::Type::kAxisTypeTileInner, outer->id);
  ASSERT_NE(tile_outer, nullptr);
  ASSERT_NE(tile_inner, nullptr);
  const af::Axis *block_outer = FindDerivedAxis(graph, af::Axis::Type::kAxisTypeBlockOuter, tile_outer->id);
  const af::Axis *block_inner = FindDerivedAxis(graph, af::Axis::Type::kAxisTypeBlockInner, tile_outer->id);
  ASSERT_NE(block_outer, nullptr);
  ASSERT_NE(block_inner, nullptr);
  outer_loops = {block_outer->id, block_inner->id, tile_inner->id};
}

void ExpectComplexLogicalView(const ascgen_utils::indirect_load::TemplateLogicalView &view) {
  ASSERT_EQ(view.output.axis_ids.size(), 4UL);
  ASSERT_EQ(view.input.axis_ids.size(), 4UL);
  EXPECT_EQ(view.input.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kZeroStrideCompact);
  EXPECT_EQ(view.index.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kZeroStrideCompact);
  ASSERT_EQ(view.input.physical_repeats.size(), 4UL);
  ASSERT_EQ(view.index.physical_repeats.size(), 4UL);
  EXPECT_TRUE(ascgen_utils::ExpressEq(view.input.physical_repeats[0], af::ops::One));
  EXPECT_TRUE(ascgen_utils::ExpressEq(view.input.physical_repeats[1], af::ops::One));
  EXPECT_TRUE(ascgen_utils::ExpressEq(view.index.physical_repeats[0], af::ops::One));
  EXPECT_TRUE(ascgen_utils::ExpressEq(view.index.physical_repeats[1], af::ops::One));
}

void ExpectComplexBrcRewrite(af::AscGraph &graph, const af::AscNodePtr &input_add, const af::AscNodePtr &index_vf,
                             const af::AscNodePtr &post_vf) {
  ASSERT_TRUE(af::ops::IsOps<af::ascir_op::Add>(input_add));
  ASSERT_TRUE(af::ops::IsOps<af::ascir_op::VectorFunc>(index_vf));
  ASSERT_TRUE(af::ops::IsOps<af::ascir_op::VectorFunc>(post_vf));
  EXPECT_EQ(index_vf->inputs.Size(), 3UL);
  EXPECT_EQ(post_vf->inputs.Size(), 3UL);
  EXPECT_EQ(graph.FindNode("input_broadcast"), nullptr);
  ASSERT_EQ(input_add->inputs.Size(), 2UL);
  for (size_t i = 0UL; i < input_add->inputs.Size(); ++i) {
    const af::AscNodePtr producer = ascgen_utils::indirect_load::GetInputProducer(input_add, i);
    ASSERT_NE(producer, nullptr);
    EXPECT_EQ(producer->GetName(), "input_load") << "input index=" << i;
  }
  for (const char *name : {"index_final_broadcast", "index_broadcast0", "index_broadcast1", "index_scalar_add",
                           "output_broadcast0", "output_broadcast1", "output_scalar_add"}) {
    EXPECT_EQ(graph.FindNode(name), nullptr) << name;
  }
}

void ExpectComplexSimdSchedule(af::AscGraph &graph, const af::AscNodePtr &indirect_load) {
  ascgen_utils::indirect_load::TemplateAxes axes;
  ascgen_utils::indirect_load::TemplateLogicalView view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, view), af::SUCCESS);
  ExpectComplexLogicalView(view);
  std::vector<af::AxisId> outer_loops;
  ExpectComplexOuterAxes(graph, axes, view, outer_loops);
  ASSERT_EQ(outer_loops.size(), 3UL);
  const std::vector<af::AxisId> index_axes = {outer_loops[0], outer_loops[1], outer_loops[2], axes.index_inner_axis};
  const std::vector<af::AxisId> input_axes = {outer_loops[0], outer_loops[1], outer_loops[2], axes.input_inner_axis};
  const std::vector<af::AxisId> output_axes = {outer_loops[0], outer_loops[1], outer_loops[2], axes.inner_axis};
  const af::AscNodePtr input_add = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  const af::AscNodePtr index_vf = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 1UL);
  const af::AscNodePtr post_vf = ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load);
  ExpectComplexBrcRewrite(graph, input_add, index_vf, post_vf);
  ExpectComplexNodeSchedule(graph.FindNode("index_load"), index_axes, axes.index_inner_axis);
  ExpectComplexNodeSchedule(index_vf, index_axes, axes.index_inner_axis);
  ExpectComplexNodeSchedule(graph.FindNode("input_load"), input_axes, axes.input_inner_axis);
  ExpectComplexNodeSchedule(input_add, input_axes, axes.input_inner_axis);
  for (const af::AscNodePtr &node : {indirect_load, post_vf, graph.FindNode("store")}) {
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->attr.sched.axis, output_axes) << node->GetName();
    EXPECT_EQ(node->attr.sched.loop_axis, outer_loops.back()) << node->GetName();
    ASSERT_FALSE(node->outputs().empty());
    EXPECT_EQ(node->outputs()[0]->attr.vectorized_axis, axes.vectorized_axes) << node->GetName();
  }
  std::vector<af::AscNodePtr> nodes;
  for (const af::AscNodePtr &node : graph.GetAllNodes()) {
    nodes.emplace_back(node);
  }
  const auto pos = [&](const af::AscNodePtr &node) { return std::find(nodes.begin(), nodes.end(), node); };
  EXPECT_LT(pos(graph.FindNode("index_load")), pos(index_vf));
  EXPECT_LT(pos(graph.FindNode("input_load")), pos(input_add));
  EXPECT_LT(pos(input_add), pos(indirect_load));
  EXPECT_LT(pos(indirect_load), pos(post_vf));
  EXPECT_LT(pos(post_vf), pos(graph.FindNode("store")));
}

}  // namespace

using TestBackendIndirectLoadBroadcastE2e = indirect_load_test::PrecisionBackendE2e;

TEST_F(TestBackendIndirectLoadBroadcastE2e, IndirectLoadBroadcastCodegen) {
  const auto graph = kAicRepro ? CreateAicReproGraph() : CreateGraph();
  ASSERT_NE(graph, nullptr);
  if (kAicRepro) {
    ascir::FusedScheduledResult scheduled_result;
    optimize::Optimizer optimizer(optimize::OptimizerOptions{.graph_type = optimize::GraphType::kFusedAscBackend});
    ASSERT_EQ(optimizer.Optimize(graph, scheduled_result), af::SUCCESS);
    ASSERT_TRUE(indirect_load_test::HasTemplate(scheduled_result, ascir::TemplateId::kIndirectLoadSimt));
    codegen::Codegen codegen(codegen::CodegenOptions{});
    codegen::CodegenResult result;
    ASSERT_EQ(codegen.Generate({}, scheduled_result, result), af::SUCCESS);
    EXPECT_NE(result.kernel.find("// IndirectLoad SIMT"), std::string::npos);
    EXPECT_NE(result.kernel.find("IndirectLoadSimtStridedPolicy<uint32_t, 2, 0, 3ULL, 1ULL>"), std::string::npos);
    EXPECT_NE(result.kernel.find(", 1024, 1024, 1024, 1, 1, 0);"), std::string::npos);
    indirect_load_test::WriteGeneratedFiles(result);
    return;
  }
  const std::map<std::string, std::string> shape_info = {{"s0", "stub_s0"}, {"s1", "stub_s1"}, {"s2", "stub_s2"},
                                                         {"s3", "stub_s3"}, {"s4", "stub_s4"}, {"s5", "stub_s5"},
                                                         {"s6", "stub_s6"}, {"s7", "stub_s7"}};
  const auto expected_template = indirect_load_test::GetExpectedTemplate(kExpectSimt, kExpectSk);
  if constexpr (kRetainBroadcast) {
    ascir::FusedScheduledResult scheduled_result;
    ASSERT_TRUE(indirect_load_test::SelectTemplate(graph, expected_template, scheduled_result));
    codegen::Codegen codegen(codegen::CodegenOptions{});
    codegen::CodegenResult result;
    ASSERT_EQ(codegen.Generate(shape_info, scheduled_result, result), af::SUCCESS);
    EXPECT_NE(result.kernel.find(kExpectSimt ? "IndirectLoadSimt<" : "IndirectLoadSimd<"), std::string::npos);
    if constexpr (kExpectSimt) {
      EXPECT_EQ(result.kernel.find("BroadcastExtend<"), std::string::npos);
    } else {
      EXPECT_NE(result.kernel.find("BroadcastExtend<"), std::string::npos);
    }
    indirect_load_test::WriteGeneratedFiles(result);
    return;
  }
  if constexpr (kComplexBroadcast) {
    ascir::FusedScheduledResult scheduled_result;
    ASSERT_TRUE(indirect_load_test::SelectTemplate(graph, expected_template, scheduled_result));
    size_t template_graph_count = 0UL;
    for (auto &candidates : scheduled_result.node_idx_to_scheduled_results) {
      for (auto &candidate : candidates) {
        for (auto &group : candidate.schedule_groups) {
          for (auto &impl_graph : group.impl_graphs) {
            const af::AscNodePtr indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(impl_graph);
            if (indirect_load != nullptr && ascir::GetTemplateIdOrDefault(*indirect_load) == expected_template) {
              ++template_graph_count;
              if (expected_template == ascir::TemplateId::kIndirectLoadSimd) {
                ExpectComplexSimdSchedule(impl_graph, indirect_load);
              }
            }
          }
        }
      }
    }
    EXPECT_GT(template_graph_count, 0UL);
    codegen::Codegen codegen(codegen::CodegenOptions{});
    codegen::CodegenResult result;
    ASSERT_EQ(codegen.Generate(shape_info, scheduled_result, result), af::SUCCESS);
    EXPECT_NE(result.kernel.find(expected_template == ascir::TemplateId::kIndirectLoadSimd ? "IndirectLoadSimd<"
                                                                                           : "IndirectLoadSimt<"),
              std::string::npos);
    EXPECT_NE(result.kernel.find("Add("), std::string::npos);
    EXPECT_EQ(result.kernel.find("BroadcastExtend<"), std::string::npos);
    indirect_load_test::WriteGeneratedFiles(result);
    return;
  }
  codegen::CodegenResult result;
  indirect_load_test::GenerateForTemplate(graph, shape_info, expected_template, result);
  CheckGeneratedKernel(result.kernel, expected_template);
  indirect_load_test::WriteGeneratedFiles(result);
}
