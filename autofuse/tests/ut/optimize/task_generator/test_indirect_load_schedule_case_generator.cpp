/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "ascir_ops.h"
#include "common/platform_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/utils/graph_utils.h"
#include "indirect_load_utils.h"
#include "schedule_result.h"
#include "task_generator/indirect_load_schedule_case_generator.h"

namespace {
constexpr int64_t kSimtDcacheSize = 32 * 1024;

class PlatformContextReset {
 public:
  ~PlatformContextReset() {
    ge::PlatformContext::GetInstance().Reset();
  }
};

std::vector<std::string> AxisNames(af::AscGraph &graph, const std::vector<af::AxisId> &axis_ids) {
  std::vector<std::string> names;
  names.reserve(axis_ids.size());
  for (af::AxisId axis_id : axis_ids) {
    const auto *axis = graph.FindAxis(axis_id);
    names.push_back(axis == nullptr ? "" : axis->name);
  }
  return names;
}

af::AxisId FindAxisByName(af::AscGraph &graph, const std::string &name) {
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis != nullptr && axis->name == name) {
      return axis->id;
    }
  }
  return af::kIdNone;
}

void ExpectAxisNames(af::AscGraph &graph, const std::vector<af::AxisId> &axis_ids,
                     const std::vector<std::string> &expected) {
  EXPECT_EQ(AxisNames(graph, axis_ids), expected);
}

void ExpectMergedFrom(af::AscGraph &graph, const std::string &merged_name,
                      const std::vector<std::string> &expected_from) {
  const af::AxisId merged_axis = FindAxisByName(graph, merged_name);
  ASSERT_NE(merged_axis, af::kIdNone);
  const auto *axis = graph.FindAxis(merged_axis);
  ASSERT_NE(axis, nullptr);
  EXPECT_EQ(axis->type, af::Axis::Type::kAxisTypeMerged);
  ExpectAxisNames(graph, axis->from, expected_from);
}

void ExpectFixedTileSplit(af::AscGraph &graph, af::AxisId outer_axis_id) {
  const auto *source_axis = graph.FindAxis(outer_axis_id);
  ASSERT_NE(source_axis, nullptr);
  const auto all_axes = graph.GetAllAxis();
  const auto tile_outer_axis = std::find_if(all_axes.begin(), all_axes.end(), [outer_axis_id](const auto &axis) {
    return axis != nullptr && axis->type == af::Axis::Type::kAxisTypeTileOuter &&
           axis->from == std::vector<af::AxisId>{outer_axis_id};
  });
  const auto tile_inner_axis = std::find_if(all_axes.begin(), all_axes.end(), [outer_axis_id](const auto &axis) {
    return axis != nullptr && axis->type == af::Axis::Type::kAxisTypeTileInner &&
           axis->from == std::vector<af::AxisId>{outer_axis_id};
  });
  ASSERT_NE(tile_outer_axis, all_axes.end());
  ASSERT_NE(tile_inner_axis, all_axes.end());
  EXPECT_EQ((*tile_outer_axis)->type, af::Axis::Type::kAxisTypeTileOuter);
  EXPECT_EQ((*tile_inner_axis)->type, af::Axis::Type::kAxisTypeTileInner);
  EXPECT_EQ((*tile_outer_axis)->size, source_axis->size);
  EXPECT_EQ(af::SymbolicUtils::StaticCheckEq((*tile_inner_axis)->size, af::ops::One), af::TriBool::kTrue);
  EXPECT_EQ((*tile_outer_axis)->split_pair_other_id, (*tile_inner_axis)->id);
  EXPECT_EQ((*tile_inner_axis)->split_pair_other_id, (*tile_outer_axis)->id);
}

void CollectAxisOrigins(af::AscGraph &graph, af::AxisId axis_id, std::vector<af::AxisId> &origins) {
  const auto *axis = graph.FindAxis(axis_id);
  ASSERT_NE(axis, nullptr);
  if (axis->from.empty()) {
    origins.emplace_back(axis_id);
    return;
  }
  for (const af::AxisId from : axis->from) {
    CollectAxisOrigins(graph, from, origins);
  }
}

void ExpectAxisOrigins(af::AscGraph &graph, af::AxisId axis_id, const std::vector<af::AxisId> &expected) {
  std::vector<af::AxisId> origins;
  CollectAxisOrigins(graph, axis_id, origins);
  EXPECT_EQ(origins, expected);
}

std::vector<af::Expression> DenseStrides(af::AscGraph &graph, const std::vector<af::AxisId> &axis_ids) {
  std::vector<af::Expression> strides(axis_ids.size(), af::sym::kSymbolOne);
  for (size_t i = 0UL; i + 1UL < axis_ids.size(); ++i) {
    strides[i] = graph.FindAxis(axis_ids[i + 1UL])->size;
    for (size_t j = i + 2UL; j < axis_ids.size(); ++j) {
      strides[i] = strides[i] * graph.FindAxis(axis_ids[j])->size;
    }
  }
  return strides;
}

template <typename Op>
void SetNodeView(Op &op, af::DataType dtype, const std::vector<af::AxisId> &axes,
                 const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  op.y.dtype = dtype;
  op.attr.sched.axis = axes;
  *op.y.axis = axes;
  *op.y.repeats = repeats;
  *op.y.strides = strides;
}

template <typename Op>
void SetVectorApi(Op &op) {
  op.attr.api.compute_type = af::ComputeType::kComputeElewise;
  op.attr.api.type = af::ApiType::kAPITypeCompute;
  op.attr.api.unit = af::ComputeUnit::kUnitVector;
}

void BuildInputPreChain(const af::AscOpOutput &input_load_output, const std::vector<af::AxisId> &axes,
                        const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides,
                        bool enabled, bool full_prefix, af::ascir_op::IndirectLoad &indirect_load) {
  if (!enabled) {
    indirect_load.x1 = input_load_output;
    return;
  }
  if (!full_prefix) {
    af::ascir_op::Abs pre_abs("pre_abs");
    pre_abs.x = input_load_output;
    SetNodeView(pre_abs, af::DT_FLOAT16, axes, repeats, strides);
    indirect_load.x1 = pre_abs.y;
    return;
  }
  af::ascir_op::Relu input_relu("input_relu");
  input_relu.x = input_load_output;
  SetNodeView(input_relu, af::DT_FLOAT16, axes, repeats, strides);
  af::ascir_op::Exp2 input_exp2("input_exp2");
  input_exp2.x = input_relu.y;
  SetNodeView(input_exp2, af::DT_FLOAT16, axes, repeats, strides);
  indirect_load.x1 = input_exp2.y;
}

void BuildIndexPreChain(const af::AscOpOutput &index_load_output, const std::vector<af::AxisId> &axes,
                        const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides,
                        bool full_prefix, af::ascir_op::IndirectLoad &indirect_load) {
  if (!full_prefix) {
    indirect_load.x2 = index_load_output;
    return;
  }
  af::ascir_op::Abs index_abs("index_abs");
  index_abs.x = index_load_output;
  SetNodeView(index_abs, af::DT_INT32, axes, repeats, strides);
  af::ascir_op::Abs index_abs2("index_abs2");
  index_abs2.x = index_abs.y;
  SetNodeView(index_abs2, af::DT_INT32, axes, repeats, strides);
  af::ascir_op::Cast index_cast("index_cast_float");
  index_cast.x = index_abs2.y;
  SetNodeView(index_cast, af::DT_FLOAT, axes, repeats, strides);
  af::ascir_op::Exp2 index_exp2("index_exp2");
  index_exp2.x = index_cast.y;
  SetNodeView(index_exp2, af::DT_FLOAT, axes, repeats, strides);
  af::ascir_op::Log2 index_log2("index_log2");
  index_log2.x = index_exp2.y;
  SetNodeView(index_log2, af::DT_FLOAT, axes, repeats, strides);
  af::ascir_op::FloorToInt index_floor_to_int("index_floor_to_int");
  index_floor_to_int.x = index_log2.y;
  SetNodeView(index_floor_to_int, af::DT_INT32, axes, repeats, strides);
  indirect_load.x2 = index_floor_to_int.y;
}

af::AscGraph BuildIndirectLoadGraph(int64_t axis, bool has_input_pre_node = false, bool full_prefix = false) {
  af::AscGraph graph("indirect_load_ut_graph");
  const af::Expression s0 = graph.CreateSizeVar("s0");
  const af::Expression s1 = graph.CreateSizeVar("s1");
  const af::Expression s2 = graph.CreateSizeVar("s2");
  const af::Expression s3 = graph.CreateSizeVar("s3");
  const af::Expression s4 = graph.CreateSizeVar("s4");
  const af::Expression s5 = graph.CreateSizeVar("s5");
  const af::Expression s6 = graph.CreateSizeVar("s6");
  const af::Expression s7 = graph.CreateSizeVar("s7");
  const auto z0 = graph.CreateAxis("z0", s0);
  const auto z1 = graph.CreateAxis("z1", s1);
  const auto z2 = graph.CreateAxis("z2", s2);
  const auto z3 = graph.CreateAxis("z3", s3);
  const auto z4 = graph.CreateAxis("z4", s4);
  const auto z5 = graph.CreateAxis("z5", s5);
  const auto z6 = graph.CreateAxis("z6", s6);
  const auto z7 = graph.CreateAxis("z7", s7);
  const std::vector<af::AxisId> input_axes = {z0.id, z1.id, z2.id, z3.id};
  const std::vector<af::AxisId> output_axes = {z4.id, z5.id, z6.id, z7.id};
  const std::vector<af::Expression> input_repeats = {s0, s1, s2, s3};
  const std::vector<af::Expression> output_repeats = {s4, s5, s6, s7};
  const std::vector<af::Expression> input_strides = {s1 * s2 * s3, s2 * s3, s3, af::sym::kSymbolOne};
  const std::vector<af::Expression> output_strides = {s5 * s6 * s7, s6 * s7, s7, af::sym::kSymbolOne};

  af::ascir_op::Data x("x", graph);
  x.y.dtype = ge::DT_FLOAT16;
  x.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x.attr.api.type = af::ApiType::kAPITypeBuffer;
  x.ir_attr.SetIndex(0);
  x.attr.sched.axis = input_axes;
  *x.y.axis = input_axes;
  *x.y.repeats = input_repeats;
  *x.y.strides = input_strides;
  af::ascir_op::Load input_load("input_load");
  input_load.x = x.y;
  SetNodeView(input_load, af::DT_FLOAT16, input_axes, input_repeats, input_strides);

  af::ascir_op::Data index("index", graph);
  index.y.dtype = ge::DT_INT32;
  index.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  index.attr.api.type = af::ApiType::kAPITypeBuffer;
  index.ir_attr.SetIndex(1);
  index.attr.sched.axis = output_axes;
  *index.y.axis = output_axes;
  *index.y.repeats = output_repeats;
  *index.y.strides = output_strides;
  af::ascir_op::Load index_load("index_load");
  index_load.x = index.y;
  SetNodeView(index_load, af::DT_INT32, output_axes, output_repeats, output_strides);

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  BuildInputPreChain(input_load.y, input_axes, input_repeats, input_strides, has_input_pre_node, full_prefix,
                     indirect_load);
  BuildIndexPreChain(index_load.y, output_axes, output_repeats, output_strides, full_prefix, indirect_load);
  indirect_load.y.dtype = ge::DT_FLOAT16;
  indirect_load.attr.sched.axis = output_axes;
  indirect_load.ir_attr.SetAxis(axis);
  *indirect_load.y.axis = output_axes;
  *indirect_load.y.repeats = output_repeats;
  *indirect_load.y.strides = output_strides;

  af::ascir_op::Store store("store");
  store.x = indirect_load.y;
  SetNodeView(store, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Output y("y");
  y.x = store.y;
  y.y.dtype = ge::DT_FLOAT16;
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.ir_attr.SetIndex(0);
  y.attr.sched.axis = output_axes;
  *y.y.axis = output_axes;
  *y.y.repeats = output_repeats;
  *y.y.strides = output_strides;

  return graph;
}

struct BroadcastGraphView {
  std::vector<af::AxisId> source_axes;
  std::vector<af::AxisId> input_axes;
  std::vector<af::AxisId> output_axes;
  std::vector<af::Expression> source_repeats;
  std::vector<af::Expression> input_repeats;
  std::vector<af::Expression> output_repeats;
};

BroadcastGraphView CreateBroadcastGraphView(af::AscGraph &graph) {
  BroadcastGraphView view;
  const std::vector<int64_t> shape = {2L, 3L, 4L};
  for (size_t dim = 0UL; dim < shape.size(); ++dim) {
    const af::Expression input_size = graph.CreateSizeVar(shape[dim]);
    const af::Expression output_size = graph.CreateSizeVar(shape[dim]);
    view.input_repeats.emplace_back(input_size);
    view.output_repeats.emplace_back(output_size);
    view.input_axes.emplace_back(graph.CreateAxis(("broadcast_i" + std::to_string(dim)).c_str(), input_size).id);
    view.output_axes.emplace_back(graph.CreateAxis(("broadcast_o" + std::to_string(dim)).c_str(), output_size).id);
  }
  view.source_axes = view.input_axes;
  view.source_repeats = view.input_repeats;
  view.source_axes[1] = graph.CreateAxis("broadcast_source", af::ops::One).id;
  view.source_repeats[1] = af::ops::One;
  return view;
}

void BuildBroadcastInputPreChain(const af::AscOpOutput &broadcast_output, const BroadcastGraphView &view,
                                 const std::vector<af::Expression> &input_strides, bool with_input_element,
                                 af::ascir_op::IndirectLoad &indirect_load) {
  if (!with_input_element) {
    indirect_load.x1 = broadcast_output;
    return;
  }
  af::ascir_op::Abs input_abs("broadcast_input_abs");
  input_abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  input_abs.x = broadcast_output;
  SetNodeView(input_abs, af::DT_FLOAT16, view.input_axes, view.input_repeats, input_strides);
  indirect_load.x1 = input_abs.y;
}

af::AscGraph BuildIndirectLoadBroadcastGraph(bool with_input_element = true) {
  af::AscGraph graph("indirect_load_broadcast_ut_graph");
  const BroadcastGraphView view = CreateBroadcastGraphView(graph);
  const std::vector<af::Expression> source_strides = {view.source_repeats[2], view.source_repeats[2], af::ops::One};
  const std::vector<af::Expression> input_strides = {view.input_repeats[2], af::ops::Zero, af::ops::One};
  const std::vector<af::Expression> output_strides = {view.output_repeats[1] * view.output_repeats[2],
                                                      view.output_repeats[2], af::ops::One};
  const af::Expression index_axis_stride = view.output_repeats[2] * (af::ops::One + af::ops::One);
  const std::vector<af::Expression> index_strides = {view.output_repeats[1] * index_axis_stride, index_axis_stride,
                                                     af::ops::One};
  af::ascir_op::Data x("broadcast_x", graph);
  x.ir_attr.SetIndex(0);
  SetNodeView(x, af::DT_FLOAT16, view.source_axes, view.source_repeats, source_strides);
  af::ascir_op::Load input_load("broadcast_input_load");
  input_load.x = x.y;
  SetNodeView(input_load, af::DT_FLOAT16, view.source_axes, view.source_repeats, source_strides);
  af::ascir_op::Broadcast broadcast("input_broadcast");
  broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast.x = input_load.y;
  SetNodeView(broadcast, af::DT_FLOAT16, view.input_axes, view.input_repeats, input_strides);
  af::ascir_op::Data index("broadcast_index", graph);
  index.ir_attr.SetIndex(1);
  SetNodeView(index, af::DT_INT32, view.output_axes, view.output_repeats, index_strides);
  af::ascir_op::Load index_load("broadcast_index_load");
  index_load.x = index.y;
  SetNodeView(index_load, af::DT_INT32, view.output_axes, view.output_repeats, index_strides);
  af::ascir_op::Abs index_abs("broadcast_index_abs");
  index_abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  index_abs.x = index_load.y;
  SetNodeView(index_abs, af::DT_INT32, view.output_axes, view.output_repeats, index_strides);
  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  BuildBroadcastInputPreChain(broadcast.y, view, input_strides, with_input_element, indirect_load);
  indirect_load.x2 = index_abs.y;
  indirect_load.ir_attr.SetAxis(1L);
  SetNodeView(indirect_load, af::DT_FLOAT16, view.output_axes, view.output_repeats, output_strides);
  af::ascir_op::Store store("broadcast_store");
  store.x = indirect_load.y;
  SetNodeView(store, af::DT_FLOAT16, view.output_axes, view.output_repeats, output_strides);
  af::ascir_op::Output output("broadcast_output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetNodeView(output, af::DT_FLOAT16, view.output_axes, view.output_repeats, output_strides);
  return graph;
}

af::AscGraph BuildIndirectLoadIndexBroadcastGraph() {
  af::AscGraph graph("indirect_load_index_broadcast_ut_graph");
  const af::Expression input_rows = graph.CreateSizeVar(100000);
  const af::Expression rows = graph.CreateSizeVar(1024);
  const af::Expression columns = graph.CreateSizeVar(1024);
  const af::AxisId input_axis0 = graph.CreateAxis("index_broadcast_x0", input_rows).id;
  const af::AxisId input_axis1 = graph.CreateAxis("index_broadcast_x1", columns).id;
  const af::AxisId output_axis0 = graph.CreateAxis("index_broadcast_y0", rows).id;
  const af::AxisId output_axis1 = graph.CreateAxis("index_broadcast_y1", columns).id;
  const std::vector<af::AxisId> input_axes = {input_axis0, input_axis1};
  const std::vector<af::AxisId> output_axes = {output_axis0, output_axis1};
  const std::vector<af::Expression> input_sizes = {input_rows, columns};
  const std::vector<af::Expression> output_sizes = {rows, columns};
  const std::vector<af::Expression> dense_strides = {columns, af::ops::One};
  af::ascir_op::Data input("index_broadcast_input", graph);
  input.ir_attr.SetIndex(0);
  SetNodeView(input, af::DT_FLOAT, input_axes, input_sizes, dense_strides);
  af::ascir_op::Load input_load("index_broadcast_input_load");
  input_load.x = input.y;
  SetNodeView(input_load, af::DT_FLOAT, input_axes, input_sizes, dense_strides);
  af::ascir_op::Data index("index_broadcast_index", graph);
  index.ir_attr.SetIndex(1);
  SetNodeView(index, af::DT_INT64, output_axes, {rows, af::ops::One}, {af::ops::One, af::ops::Zero});
  af::ascir_op::Load index_load("index_broadcast_index_load");
  index_load.x = index.y;
  SetNodeView(index_load, af::DT_INT64, output_axes, {rows, af::ops::One}, {af::ops::One, af::ops::Zero});
  af::ascir_op::Broadcast broadcast("index_broadcast");
  broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast.x = index_load.y;
  SetNodeView(broadcast, af::DT_INT64, output_axes, output_sizes, dense_strides);
  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  indirect_load.x1 = input_load.y;
  indirect_load.x2 = broadcast.y;
  indirect_load.ir_attr.SetAxis(0L);
  SetNodeView(indirect_load, af::DT_FLOAT, output_axes, output_sizes, dense_strides);
  af::ascir_op::Store store("index_broadcast_store");
  store.x = indirect_load.y;
  SetNodeView(store, af::DT_FLOAT, output_axes, output_sizes, dense_strides);
  af::ascir_op::Output output("index_broadcast_output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetNodeView(output, af::DT_FLOAT, output_axes, output_sizes, dense_strides);
  return graph;
}

bool AddSideConsumer(af::AscGraph &graph, const char *producer_name) {
  af::ascir_op::Abs side_consumer("coverage_side_consumer");
  graph.AddNode(side_consumer);
  const auto producer = graph.FindNode(producer_name);
  const auto consumer = graph.FindNode("coverage_side_consumer");
  return producer != nullptr && consumer != nullptr &&
         af::GraphUtils::AddEdge(producer->GetOutDataAnchor(0), consumer->GetInDataAnchor(0)) == ge::GRAPH_SUCCESS;
}

void SetStridedInputPath(af::AscGraph &graph) {
  const std::vector<af::Expression> strides = {af::Symbol(64), af::Symbol(2)};
  for (const char *name : {"data0", "load0", "data1", "load1", "x_copy_sign"}) {
    const auto node = graph.FindNode(name);
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->outputs().size(), 1UL);
    node->outputs()[0]->attr.strides = strides;
  }
}

af::AscGraph BuildIndirectLoadPrecisionCastGraph() {
  af::AscGraph graph("indirect_load_precision_cast_ut_graph");
  const af::Expression s0 = graph.CreateSizeVar("s0");
  const af::Expression s1 = graph.CreateSizeVar("s1");
  const af::Expression s2 = graph.CreateSizeVar("s2");
  const af::Expression s3 = graph.CreateSizeVar("s3");
  const auto z0 = graph.CreateAxis("z0", s0);
  const auto z1 = graph.CreateAxis("z1", s1);
  const auto z2 = graph.CreateAxis("z2", s2);
  const auto z3 = graph.CreateAxis("z3", s3);
  const std::vector<af::AxisId> input_axes = {z0.id, z1.id};
  const std::vector<af::AxisId> output_axes = {z2.id, z3.id};
  const std::vector<af::Expression> input_repeats = {s0, s1};
  const std::vector<af::Expression> output_repeats = {s2, s3};
  const std::vector<af::Expression> input_strides = {s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> output_strides = {s3, af::sym::kSymbolOne};

  af::ascir_op::Data x("x", graph);
  x.ir_attr.SetIndex(0);
  SetNodeView(x, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Load input_load("input_load");
  input_load.x = x.y;
  SetNodeView(input_load, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Cast input_cast("input_cast");
  input_cast.x = input_load.y;
  SetNodeView(input_cast, af::DT_FLOAT, input_axes, input_repeats, input_strides);

  af::ascir_op::Data index("index", graph);
  index.ir_attr.SetIndex(1);
  SetNodeView(index, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::Load index_load("index_load");
  index_load.x = index.y;
  SetNodeView(index_load, af::DT_INT32, output_axes, output_repeats, output_strides);

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  indirect_load.x1 = input_cast.y;
  indirect_load.x2 = index_load.y;
  indirect_load.ir_attr.SetAxis(1);
  SetNodeView(indirect_load, af::DT_FLOAT, output_axes, output_repeats, output_strides);
  af::ascir_op::Exp output_exp("output_exp");
  output_exp.x = indirect_load.y;
  SetNodeView(output_exp, af::DT_FLOAT, output_axes, output_repeats, output_strides);
  af::ascir_op::Cast output_cast("output_cast");
  output_cast.x = output_exp.y;
  SetNodeView(output_cast, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Store store("store");
  store.x = output_cast.y;
  SetNodeView(store, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Output y("y");
  y.x = store.y;
  y.ir_attr.SetIndex(0);
  SetNodeView(y, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  return graph;
}

af::AscGraph BuildExpandedMultiInputGraph(bool add_unary = false, int64_t output_dim0 = 2L, int64_t output_dim1 = 17L) {
  af::AscGraph graph("indirect_load_expanded_multi_input_ut_graph");
  const af::Expression x_s0 = graph.CreateSizeVar(2);
  const af::Expression x_s1 = graph.CreateSizeVar(16);
  const af::Expression y_s0 = graph.CreateSizeVar(output_dim0);
  const af::Expression y_s1 = graph.CreateSizeVar(output_dim1);
  const auto x0 = graph.CreateAxis("x0", x_s0);
  const auto x1 = graph.CreateAxis("x1", x_s1);
  const auto y0 = graph.CreateAxis("y0", y_s0);
  const auto y1 = graph.CreateAxis("y1", y_s1);
  const std::vector<af::AxisId> input_axes = {x0.id, x1.id};
  const std::vector<af::AxisId> output_axes = {y0.id, y1.id};
  const std::vector<af::Expression> input_repeats = {x_s0, x_s1};
  const std::vector<af::Expression> output_repeats = {y_s0, y_s1};
  const std::vector<af::Expression> input_strides = {x_s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> output_strides = {y_s1, af::sym::kSymbolOne};

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  SetNodeView(data0, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  SetNodeView(load0, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  SetNodeView(data1, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  SetNodeView(load1, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::CopySign copy_sign("x_copy_sign");
  copy_sign.x1 = load0.y;
  copy_sign.x2 = load1.y;
  SetNodeView(copy_sign, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Data index("index", graph);
  index.ir_attr.SetIndex(2);
  SetNodeView(index, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::Load index_load("index_load");
  index_load.x = index.y;
  SetNodeView(index_load, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  indirect_load.x1 = copy_sign.y;
  indirect_load.x2 = index_load.y;
  indirect_load.ir_attr.SetAxis(1);
  SetNodeView(indirect_load, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Store store("store");
  if (add_unary) {
    af::ascir_op::Abs pre_abs("pre_abs");
    pre_abs.x = indirect_load.y;
    SetNodeView(pre_abs, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
    store.x = pre_abs.y;
  } else {
    store.x = indirect_load.y;
  }
  SetNodeView(store, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetNodeView(output, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  return graph;
}

enum class OutputPostTopology { kSum, kExp2Sum, kAbsExp2Sum, kCastSum, kAddSum, kBesselK0Sum, kBesselK0 };

struct PostTopologyCase {
  OutputPostTopology topology;
  std::vector<const char *> chain;
  bool expect_simt;
  af::DataType post_dtype;
  bool direct_output;
};

void ExpectNodeView(const af::AscNodePtr &node, af::DataType dtype, const std::vector<af::AxisId> &axes,
                    const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides);

void ExpectPostTerminalGraph(af::AscGraph &graph, const PostTopologyCase &test_case, const af::AscNodePtr &producer) {
  const auto indirect_load = graph.FindNode("indirect_load");
  const auto store = graph.FindNode("store");
  const auto output = graph.FindNode("output");
  ASSERT_NE(indirect_load, nullptr);
  ASSERT_NE(store, nullptr);
  ASSERT_NE(output, nullptr);
  const auto &post_view = indirect_load->outputs()[0]->attr;
  const auto &terminal_view = store->outputs()[0]->attr;
  const auto sum = graph.FindNode("sum");
  if (test_case.topology == OutputPostTopology::kBesselK0) {
    ASSERT_TRUE(test_case.direct_output);
    ASSERT_EQ(sum, nullptr);
    const auto store_input = ascgen_utils::indirect_load::GetInputProducer(store, 0UL);
    ASSERT_NE(store_input, nullptr);
    EXPECT_EQ(store_input->GetName(), producer->GetName());
  } else {
    ASSERT_FALSE(test_case.direct_output);
    ASSERT_NE(sum, nullptr);
    ExpectNodeView(sum, test_case.post_dtype, terminal_view.axis, terminal_view.repeats, terminal_view.strides);
    const auto sum_input = ascgen_utils::indirect_load::GetInputProducer(sum, 0UL);
    const auto store_input = ascgen_utils::indirect_load::GetInputProducer(store, 0UL);
    ASSERT_NE(sum_input, nullptr);
    ASSERT_NE(store_input, nullptr);
    EXPECT_EQ(sum_input->GetName(), producer->GetName());
    EXPECT_EQ(store_input->GetName(), sum->GetName());
  }
  if (test_case.topology == OutputPostTopology::kAddSum) {
    const auto add = graph.FindNode("output_add");
    const auto addend_load = graph.FindNode("output_addend_load");
    ASSERT_NE(add, nullptr);
    ASSERT_NE(addend_load, nullptr);
    const auto addend_input = ascgen_utils::indirect_load::GetInputProducer(add, 1UL);
    ASSERT_NE(addend_input, nullptr);
    EXPECT_EQ(addend_input->GetName(), addend_load->GetName());
    ExpectNodeView(addend_load, af::DT_FLOAT16, post_view.axis, post_view.repeats, post_view.strides);
  }
  ExpectNodeView(store, test_case.post_dtype, terminal_view.axis, terminal_view.repeats, terminal_view.strides);
  ExpectNodeView(output, test_case.post_dtype, terminal_view.axis, terminal_view.repeats, terminal_view.strides);
  const auto output_input = ascgen_utils::indirect_load::GetInputProducer(output, 0UL);
  ASSERT_NE(output_input, nullptr);
  EXPECT_EQ(output_input->GetName(), store->GetName());
}

void ExpectNodeView(const af::AscNodePtr &node, af::DataType dtype, const std::vector<af::AxisId> &axes,
                    const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  ASSERT_NE(node, nullptr);
  ASSERT_FALSE(node->outputs().empty());
  const auto &view = node->outputs()[0]->attr;
  EXPECT_EQ(view.dtype, dtype) << node->GetName();
  EXPECT_EQ(view.axis, axes) << node->GetName();
  EXPECT_EQ(view.repeats, repeats) << node->GetName();
  EXPECT_EQ(view.strides, strides) << node->GetName();
}

void ExpectPostTopologyGraph(af::AscGraph &graph, const PostTopologyCase &test_case) {
  const auto indirect_load = graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  const auto &post_view = indirect_load->outputs()[0]->attr;
  ExpectNodeView(indirect_load, af::DT_FLOAT16, post_view.axis, post_view.repeats, post_view.strides);
  af::AscNodePtr producer = indirect_load;
  for (const char *name : test_case.chain) {
    const auto node = graph.FindNode(name);
    ASSERT_NE(node, nullptr) << name;
    ExpectNodeView(node, test_case.post_dtype, post_view.axis, post_view.repeats, post_view.strides);
    const auto actual_producer = ascgen_utils::indirect_load::GetInputProducer(node, 0UL);
    ASSERT_NE(actual_producer, nullptr) << name;
    EXPECT_EQ(actual_producer->GetName(), producer->GetName()) << name;
    producer = node;
  }
  ExpectPostTerminalGraph(graph, test_case, producer);
}

void BuildAbsExp2Chain(const af::AscOpOutput &input, const std::vector<af::AxisId> &axes,
                       const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides,
                       af::ascir_op::Sum &sum) {
  af::ascir_op::Abs abs("output_abs");
  abs.x = input;
  SetNodeView(abs, af::DT_FLOAT16, axes, repeats, strides);
  SetVectorApi(abs);
  af::ascir_op::Exp2 exp2("output_exp2");
  exp2.x = abs.y;
  SetNodeView(exp2, af::DT_FLOAT16, axes, repeats, strides);
  SetVectorApi(exp2);
  sum.x = exp2.y;
}

void BuildAddChain(af::AscGraph &graph, const af::AscOpOutput &input, const std::vector<af::AxisId> &axes,
                   const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides,
                   af::ascir_op::Sum &sum) {
  af::ascir_op::Data addend("output_addend", graph);
  addend.ir_attr.SetIndex(2);
  SetNodeView(addend, af::DT_FLOAT16, axes, repeats, strides);
  af::ascir_op::Load addend_load("output_addend_load");
  addend_load.x = addend.y;
  SetNodeView(addend_load, af::DT_FLOAT16, axes, repeats, strides);
  addend_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  addend_load.attr.api.type = af::ApiType::kAPITypeCompute;
  addend_load.attr.api.unit = af::ComputeUnit::kUnitMTE2;
  af::ascir_op::Add add("output_add");
  add.x1 = input;
  add.x2 = addend_load.y;
  SetNodeView(add, af::DT_FLOAT16, axes, repeats, strides);
  SetVectorApi(add);
  sum.x = add.y;
}

void BuildOutputPostChain(af::AscGraph &graph, OutputPostTopology topology, const af::AscOpOutput &input,
                          const std::vector<af::AxisId> &axes, const std::vector<af::Expression> &repeats,
                          const std::vector<af::Expression> &strides, af::ascir_op::Sum &sum,
                          af::ascir_op::Store &store, af::DataType &post_dtype, bool &direct_output) {
  if (topology == OutputPostTopology::kAbsExp2Sum) {
    BuildAbsExp2Chain(input, axes, repeats, strides, sum);
    return;
  }
  if (topology == OutputPostTopology::kAddSum) {
    BuildAddChain(graph, input, axes, repeats, strides, sum);
    return;
  }
  if (topology == OutputPostTopology::kExp2Sum) {
    af::ascir_op::Exp2 exp2("output_exp2");
    exp2.x = input;
    SetNodeView(exp2, af::DT_FLOAT16, axes, repeats, strides);
    SetVectorApi(exp2);
    sum.x = exp2.y;
    return;
  }
  if (topology == OutputPostTopology::kCastSum) {
    af::ascir_op::Cast cast("output_cast");
    cast.x = input;
    SetNodeView(cast, af::DT_FLOAT, axes, repeats, strides);
    SetVectorApi(cast);
    post_dtype = af::DT_FLOAT;
    sum.x = cast.y;
    return;
  }
  if (topology == OutputPostTopology::kBesselK0Sum || topology == OutputPostTopology::kBesselK0) {
    af::ascir_op::ModifiedBesselK0 bessel("output_modified_bessel_k0");
    bessel.x = input;
    SetNodeView(bessel, af::DT_FLOAT16, axes, repeats, strides);
    SetVectorApi(bessel);
    if (topology == OutputPostTopology::kBesselK0) {
      store.x = bessel.y;
      direct_output = true;
    } else {
      sum.x = bessel.y;
    }
    return;
  }
  sum.x = input;
}

af::AscGraph BuildPostReduceGraph(const std::string &suffix, bool reduce_outer = false,
                                  OutputPostTopology topology = OutputPostTopology::kSum) {
  af::AscGraph graph("indirect_load_post_reduce_ut_graph");
  const af::Expression s0 = graph.CreateSizeVar(2);
  const af::Expression s1 = suffix[0] == 'B' ? af::Expression(af::sym::kSymbolOne) : graph.CreateSizeVar(3);
  const af::Expression s2 = suffix[1] == 'B' ? af::Expression(af::sym::kSymbolOne) : graph.CreateSizeVar(4);
  const af::Expression s3 = suffix[2] == 'B' ? af::Expression(af::sym::kSymbolOne) : graph.CreateSizeVar(5);
  const auto x0 = graph.CreateAxis("x0", s0);
  const auto x1 = graph.CreateAxis("x1", s1);
  const auto x2 = graph.CreateAxis("x2", s2);
  const auto x3 = graph.CreateAxis("x3", s3);
  const auto y0 = graph.CreateAxis("y0", s0);
  const auto y1 = graph.CreateAxis("y1", s1);
  const auto y2 = graph.CreateAxis("y2", s2);
  const auto y3 = graph.CreateAxis("y3", s3);
  const std::vector<af::AxisId> input_axes = {x0.id, x1.id, x2.id, x3.id};
  const std::vector<af::AxisId> output_axes = {y0.id, y1.id, y2.id, y3.id};
  const std::vector<af::Expression> input_repeats = {s0, s1, s2, s3};
  const std::vector<af::Expression> output_repeats = {s0, s1, s2, s3};
  std::vector<af::Expression> reduce_repeats = output_repeats;
  const std::vector<af::Expression> input_strides = {s1 * s2 * s3, s2 * s3, s3, af::sym::kSymbolOne};
  const std::vector<af::Expression> output_strides = {s1 * s2 * s3, s2 * s3, s3, af::sym::kSymbolOne};
  std::vector<af::Expression> reduce_input_strides = output_strides;
  std::vector<af::Expression> reduce_strides = output_strides;
  if (reduce_outer) {
    reduce_repeats[0] = af::sym::kSymbolOne;
    reduce_strides[0] = af::sym::kSymbolZero;
  }
  for (size_t i = 1UL; i < reduce_strides.size(); ++i) {
    if (suffix[i - 1UL] == 'R') {
      reduce_repeats[i] = af::sym::kSymbolOne;
      reduce_strides[i] = af::sym::kSymbolZero;
    } else if (suffix[i - 1UL] == 'B') {
      reduce_input_strides[i] = af::sym::kSymbolZero;
      reduce_strides[i] = af::sym::kSymbolZero;
    }
  }

  af::ascir_op::Data x("x", graph);
  x.ir_attr.SetIndex(0);
  SetNodeView(x, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Load input_load("input_load");
  input_load.x = x.y;
  SetNodeView(input_load, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Data index("index", graph);
  index.ir_attr.SetIndex(1);
  SetNodeView(index, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::Load index_load("index_load");
  index_load.x = index.y;
  SetNodeView(index_load, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  indirect_load.x1 = input_load.y;
  indirect_load.x2 = index_load.y;
  indirect_load.ir_attr.SetAxis(1);
  SetNodeView(indirect_load, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Sum sum("sum");
  af::ascir_op::Store store("store");
  af::DataType post_dtype = af::DT_FLOAT16;
  bool direct_output = false;
  if (suffix.find('B') == std::string::npos) {
    BuildOutputPostChain(graph, topology, indirect_load.y, output_axes, output_repeats, reduce_input_strides, sum,
                         store, post_dtype, direct_output);
  } else {
    af::ascir_op::Abs common_zero_view("common_zero_view");
    common_zero_view.x = indirect_load.y;
    SetNodeView(common_zero_view, af::DT_FLOAT16, output_axes, output_repeats, reduce_input_strides);
    BuildOutputPostChain(graph, topology, common_zero_view.y, output_axes, output_repeats, reduce_input_strides, sum,
                         store, post_dtype, direct_output);
  }
  if (!direct_output) {
    sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
    sum.attr.sched.axis = output_axes;
    SetNodeView(sum, post_dtype, output_axes, reduce_repeats, reduce_strides);
    store.x = sum.y;
  }
  const auto &store_repeats = direct_output ? output_repeats : reduce_repeats;
  const auto &store_strides = direct_output ? output_strides : reduce_strides;
  SetNodeView(store, post_dtype, output_axes, store_repeats, store_strides);
  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetNodeView(output, post_dtype, output_axes, store_repeats, store_strides);
  return graph;
}

af::AscGraph BuildShapeEnlargingInputPreGraph() {
  af::AscGraph graph("indirect_load_shape_enlarging_input_pre_ut_graph");
  const af::Expression x_s0 = graph.CreateSizeVar(2);
  const af::Expression x_s1 = graph.CreateSizeVar(16);
  const af::Expression y_s0 = graph.CreateSizeVar(2);
  const af::Expression y_s1 = graph.CreateSizeVar(17);
  const auto x0 = graph.CreateAxis("x0", x_s0);
  const auto x1 = graph.CreateAxis("x1", x_s1);
  const auto y0 = graph.CreateAxis("y0", y_s0);
  const auto y1 = graph.CreateAxis("y1", y_s1);
  const std::vector<af::AxisId> input_axes = {x0.id, x1.id};
  const std::vector<af::AxisId> output_axes = {y0.id, y1.id};
  const std::vector<af::Expression> input_repeats = {x_s0, x_s1};
  const std::vector<af::Expression> output_repeats = {y_s0, y_s1};
  const std::vector<af::Expression> input_strides = {x_s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> output_strides = {y_s1, af::sym::kSymbolOne};

  af::ascir_op::Data data("data", graph);
  data.ir_attr.SetIndex(0);
  SetNodeView(data, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Load load("input_load");
  load.x = data.y;
  SetNodeView(load, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Cast pre_cast("pre_cast");
  pre_cast.x = load.y;
  SetNodeView(pre_cast, af::DT_FLOAT, input_axes, input_repeats, input_strides);
  af::ascir_op::Data index("index", graph);
  index.ir_attr.SetIndex(1);
  SetNodeView(index, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::Load index_load("index_load");
  index_load.x = index.y;
  SetNodeView(index_load, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  indirect_load.x1 = pre_cast.y;
  indirect_load.x2 = index_load.y;
  indirect_load.ir_attr.SetAxis(1);
  SetNodeView(indirect_load, af::DT_FLOAT, output_axes, output_repeats, output_strides);
  af::ascir_op::Store store("store");
  store.x = indirect_load.y;
  SetNodeView(store, af::DT_FLOAT, output_axes, output_repeats, output_strides);
  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetNodeView(output, af::DT_FLOAT, output_axes, output_repeats, output_strides);
  return graph;
}

std::vector<af::AscGraph> GenerateIndirectLoadCases(int64_t axis) {
  auto graph = BuildIndirectLoadGraph(axis);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_EQ(graphs.size(), 4UL);
  EXPECT_EQ(score_functions.size(), 4UL);
  EXPECT_EQ(score_functions[0], score_functions[1]);
  EXPECT_EQ(score_functions[1], score_functions[2]);
  EXPECT_EQ(score_functions[2], score_functions[3]);
  return graphs;
}

std::vector<af::AscGraph>::iterator FindGeneratedGraphByTemplate(std::vector<af::AscGraph> &graphs,
                                                                 ascir::TemplateId template_id) {
  return std::find_if(graphs.begin(), graphs.end(), [template_id](const af::AscGraph &graph) {
    const auto node = graph.FindNode("indirect_load");
    return node != nullptr && ascir::GetTemplateIdOrDefault(*node) == template_id;
  });
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimtSetsDcacheAndUsesUnifiedVectorizedAxisWithoutReduce) {
  auto graphs = GenerateIndirectLoadCases(2);
  const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_NE(simt_iter, graphs.end());
  auto &simt_graph = *simt_iter;
  const auto indirect_load = simt_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);

  EXPECT_EQ(ascir::GetDcacheSize(*indirect_load), kSimtDcacheSize);
  ExpectMergedFrom(simt_graph, "indirect_load_outer", {"z4", "z5", "z6", "z7"});
  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ExpectAxisNames(simt_graph, indirect_load->attr.sched.axis, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simt_graph, {axes.outer_axis}, {"indirect_load_outer"});
  EXPECT_TRUE(indirect_load->outputs()[0]->attr.vectorized_axis.empty());
  EXPECT_TRUE(indirect_load->outputs()[0]->attr.vectorized_strides.empty());

  const auto behavior = ascgen_utils::indirect_load::GetTemplateBehavior(indirect_load);
  EXPECT_FALSE(behavior.skips_main_schedule_tiling);
  EXPECT_FALSE(behavior.skips_api_emit);
  EXPECT_TRUE(behavior.uses_direct_gm_pipeline);
  EXPECT_TRUE(behavior.skips_ub_lifecycle);
  EXPECT_TRUE(behavior.preserves_vectorized_axis);
  EXPECT_FALSE(ascgen_utils::indirect_load::ShouldApplyInputInnerVectorization(indirect_load));
  EXPECT_FALSE(ascgen_utils::indirect_load::ShouldSkipMainScheduleTiling(indirect_load));
  EXPECT_TRUE(ascgen_utils::indirect_load::ShouldPreserveVectorizedAxis(indirect_load));
  EXPECT_TRUE(ascgen_utils::indirect_load::ShouldDisableRegularVectorFunc(indirect_load));
}

TEST(IndirectLoadScheduleCaseGeneratorTest, ClassifiesDenseLayout) {
  const ascgen_utils::indirect_load::LogicalTensorView view = {
      {0, 1, 2}, {af::Symbol(2), af::Symbol(3), af::Symbol(4)}, {af::Symbol(12), af::Symbol(4), af::Symbol(1)}};
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  ASSERT_EQ(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(view, layout), af::SUCCESS);
  EXPECT_EQ(layout.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense);
  EXPECT_EQ(layout.physical_repeats, view.sizes);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, ClassifiesZeroStrideCompactLayout) {
  const ascgen_utils::indirect_load::LogicalTensorView view = {
      {0, 1, 2}, {af::Symbol(2), af::Symbol(3), af::Symbol(4)}, {af::Symbol(4), af::Symbol(0), af::Symbol(1)}};
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  ASSERT_EQ(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(view, layout), af::SUCCESS);
  EXPECT_EQ(layout.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kZeroStrideCompact);
  EXPECT_EQ(layout.physical_repeats, (std::vector<af::Expression>{af::Symbol(2), af::Symbol(1), af::Symbol(4)}));
}

TEST(IndirectLoadScheduleCaseGeneratorTest, ClassifiesNonOverlappingStridedLayout) {
  const ascgen_utils::indirect_load::LogicalTensorView view = {
      {0, 1, 2}, {af::Symbol(2), af::Symbol(3), af::Symbol(4)}, {af::Symbol(20), af::Symbol(4), af::Symbol(1)}};
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  ASSERT_EQ(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(view, layout), af::SUCCESS);
  EXPECT_EQ(layout.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kStrided);
  EXPECT_EQ(layout.physical_repeats, view.sizes);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, RejectsOverlappingStridedLayout) {
  const ascgen_utils::indirect_load::LogicalTensorView view = {
      {0, 1, 2}, {af::Symbol(2), af::Symbol(3), af::Symbol(4)}, {af::Symbol(10), af::Symbol(4), af::Symbol(1)}};
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  ASSERT_EQ(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(view, layout), af::SUCCESS);
  EXPECT_EQ(layout.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, RejectsMixedZeroStrideAndPhysicalGapLayout) {
  const ascgen_utils::indirect_load::LogicalTensorView view = {
      {0, 1, 2}, {af::Symbol(2), af::Symbol(3), af::Symbol(4)}, {af::Symbol(8), af::Symbol(0), af::Symbol(1)}};
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  ASSERT_EQ(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(view, layout), af::SUCCESS);
  EXPECT_EQ(layout.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, RejectsInvalidLogicalLayout) {
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  const ascgen_utils::indirect_load::LogicalTensorView rank_mismatch = {
      {0, 1}, {af::Symbol(2)}, {af::Symbol(1), af::Symbol(1)}};
  EXPECT_NE(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(rank_mismatch, layout), af::SUCCESS);

  const ascgen_utils::indirect_load::LogicalTensorView negative_stride = {
      {0, 1}, {af::Symbol(2), af::Symbol(3)}, {af::Symbol(-3), af::Symbol(1)}};
  ASSERT_EQ(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(negative_stride, layout), af::SUCCESS);
  EXPECT_EQ(layout.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, BroadcastElementPathUsesPhysicalViewForAllTemplates) {
  PlatformContextReset platform_reset;
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "3510";
  platform_info.ub_size = 256 * 1024;
  platform_info.aiv_num = 48;
  ge::PlatformContext::GetInstance().SetPlatformInfo(platform_info);
  auto graph = BuildIndirectLoadBroadcastGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_EQ(graphs.size(), 4UL);
  ASSERT_EQ(score_functions.size(), graphs.size());
  EXPECT_NE(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd), graphs.end());
  EXPECT_NE(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt), graphs.end());
  EXPECT_NE(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSK), graphs.end());

  for (auto &candidate : graphs) {
    EXPECT_EQ(candidate.FindNode("input_broadcast"), nullptr);
    const auto indirect_load = candidate.FindNode("indirect_load");
    const auto input_abs = candidate.FindNode("broadcast_input_abs");
    const auto index_abs = candidate.FindNode("broadcast_index_abs");
    ASSERT_NE(indirect_load, nullptr);
    ASSERT_NE(input_abs, nullptr);
    ASSERT_NE(index_abs, nullptr);
    ascgen_utils::indirect_load::TemplateLogicalView logical_view;
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
    EXPECT_EQ(logical_view.input.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kZeroStrideCompact);
    ASSERT_EQ(logical_view.input.physical_repeats.size(), 3UL);
    EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(logical_view.input.physical_repeats[1], af::ops::One),
              af::TriBool::kTrue);
    if (ascir::GetTemplateIdOrDefault(*indirect_load) != ascir::TemplateId::kIndirectLoadSimt) {
      EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(index_abs),
                ascgen_utils::indirect_load::TemplateRole::kStridedUbPath);
    }
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, BroadcastDirectPathUsesPhysicalViewForAllTemplates) {
  PlatformContextReset platform_reset;
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "3510";
  platform_info.ub_size = 256 * 1024;
  platform_info.aiv_num = 48;
  ge::PlatformContext::GetInstance().SetPlatformInfo(platform_info);
  auto graph = BuildIndirectLoadBroadcastGraph(false);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_EQ(graphs.size(), 4UL);
  ASSERT_EQ(score_functions.size(), graphs.size());
  for (auto &candidate : graphs) {
    EXPECT_EQ(candidate.FindNode("input_broadcast"), nullptr);
    EXPECT_EQ(candidate.FindNode("broadcast_input_abs"), nullptr);
    const auto indirect_load = candidate.FindNode("indirect_load");
    ASSERT_NE(indirect_load, nullptr);
    ascgen_utils::indirect_load::TemplateLogicalView logical_view;
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
    EXPECT_EQ(logical_view.input.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kZeroStrideCompact);
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, IndexBroadcastUsesFinalPhysicalView) {
  auto graph = BuildIndirectLoadIndexBroadcastGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simt = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_NE(simt, graphs.end());
  EXPECT_EQ(simt->FindNode("index_broadcast"), nullptr);
  const auto indirect_load = simt->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  EXPECT_EQ(logical_view.index.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kZeroStrideCompact);
  EXPECT_EQ(logical_view.index.sizes, (std::vector<af::Expression>{af::Symbol(1024), af::Symbol(1024)}));
  EXPECT_EQ(logical_view.index.strides, (std::vector<af::Expression>{af::ops::One, af::ops::Zero}));
  EXPECT_EQ(logical_view.index.physical_repeats, (std::vector<af::Expression>{af::Symbol(1024), af::ops::One}));
}

TEST(IndirectLoadScheduleCaseGeneratorTest, CompletesMissingDataViewAfterBroadcastRewrite) {
  auto graph = BuildIndirectLoadBroadcastGraph();
  const auto input = graph.FindNode("broadcast_x");
  ASSERT_NE(input, nullptr);
  ASSERT_EQ(input->outputs().size(), 1UL);
  input->outputs()[0]->attr.axis.clear();
  input->outputs()[0]->attr.repeats.clear();
  input->outputs()[0]->attr.strides.clear();

  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simd = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd, graphs.end());
  const auto completed_input = simd->FindNode("broadcast_x");
  const auto input_load = simd->FindNode("broadcast_input_load");
  ASSERT_NE(completed_input, nullptr);
  ASSERT_NE(input_load, nullptr);
  EXPECT_EQ(completed_input->outputs()[0]->attr.axis.size(), 3UL);
  EXPECT_EQ(completed_input->outputs()[0]->attr.repeats, input_load->outputs()[0]->attr.repeats);
  EXPECT_EQ(completed_input->outputs()[0]->attr.strides, input_load->outputs()[0]->attr.strides);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, RejectsInvalidBroadcastAndPostElementPaths) {
  for (const char *producer_name : {"input_broadcast", "broadcast_input_abs"}) {
    auto graph = BuildIndirectLoadBroadcastGraph();
    ASSERT_TRUE(AddSideConsumer(graph, producer_name));
    optimize::IndirectLoadScheduleCaseGenerator generator;
    std::vector<af::AscGraph> graphs;
    std::vector<std::string> score_functions;
    ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
    EXPECT_TRUE(graphs.empty()) << producer_name;
    EXPECT_TRUE(score_functions.empty()) << producer_name;
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimdAcceptsStridedInputPath) {
  auto graph = BuildExpandedMultiInputGraph(false, 2L, 16L);
  SetStridedInputPath(graph);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simd = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd, graphs.end());
  const auto indirect_load = simd->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  EXPECT_EQ(logical_view.input.strides, (std::vector<af::Expression>{af::Symbol(64), af::Symbol(2)}));
  for (const char *name : {"load0", "load1", "x_copy_sign"}) {
    const auto node = simd->FindNode(name);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(node),
              ascgen_utils::indirect_load::TemplateRole::kSimdInputPreStridedUbPath);
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, StoresTemplateAxesWithoutOverwritingSchedAxis) {
  auto graphs = GenerateIndirectLoadCases(2);
  const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd_iter, graphs.end());
  auto &simd_graph = *simd_iter;
  const auto indirect_load = simd_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);

  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ExpectAxisNames(simd_graph, indirect_load->attr.sched.axis, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simd_graph, {axes.outer_axis}, {"indirect_load_outer"});
  ExpectAxisNames(simd_graph, {axes.inner_axis}, {"indirect_load_inner"});
  ExpectFixedTileSplit(simd_graph, axes.outer_axis);

  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  ExpectAxisNames(simd_graph, logical_view.input.axis_ids, {"z0", "z1", "z2", "z3"});
  ExpectAxisNames(simd_graph, logical_view.index.axis_ids, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simd_graph, logical_view.output.axis_ids, {"z4", "z5", "z6", "z7"});
  EXPECT_EQ(logical_view.input.sizes.size(), 4UL);
  EXPECT_EQ(logical_view.input.strides.size(), 4UL);
  EXPECT_EQ(logical_view.index.sizes.size(), 4UL);
  EXPECT_EQ(logical_view.index.strides.size(), 4UL);
  EXPECT_EQ(logical_view.output.sizes.size(), 4UL);
  EXPECT_EQ(logical_view.output.strides.size(), 4UL);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, StoresFixedTileAxesAsTemplateMetadata) {
  auto graphs = GenerateIndirectLoadCases(2);
  const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd_iter, graphs.end());
  auto &simd_graph = *simd_iter;
  const auto indirect_load = simd_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  const auto op_desc = indirect_load->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  EXPECT_EQ(op_desc->TryGetExtAttr("af.internal.indirect_load.tile_outer_axis", static_cast<int64_t>(af::kIdNone)),
            axes.tile_outer_axis);
  EXPECT_EQ(op_desc->TryGetExtAttr("af.internal.indirect_load.tile_inner_axis", static_cast<int64_t>(af::kIdNone)),
            axes.tile_inner_axis);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SkBuildsInputInnerAxisFromInputBoundary) {
  auto graph = BuildIndirectLoadGraph(2, true);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto sk_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSK);
  ASSERT_NE(sk_iter, graphs.end());
  auto &sk_graph = *sk_iter;
  const auto indirect_load = sk_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  const auto input_boundary = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  ASSERT_NE(input_boundary, nullptr);

  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  const auto *input_inner_axis = sk_graph.FindAxis(axes.input_inner_axis);
  ASSERT_NE(input_inner_axis, nullptr);
  EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(input_boundary),
            ascgen_utils::indirect_load::TemplateRole::kSkInputBoundary);
  ExpectAxisNames(sk_graph, input_boundary->outputs()[0]->attr.axis, {"z0", "z1", "z2", "z3"});
  ExpectAxisNames(sk_graph, input_inner_axis->from, {"z2", "z3"});
}

TEST(IndirectLoadScheduleCaseGeneratorTest, GeneratorTaskKeepsSkPartitionOrder) {
  auto graph = BuildIndirectLoadGraph(2);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<optimize::ScheduleTask> tasks;
  ASSERT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  const auto sk_task = std::find_if(tasks.begin(), tasks.end(), [](const optimize::ScheduleTask &task) {
    const auto node = task.optimize_graph.FindNode("indirect_load");
    return node != nullptr && ascir::GetTemplateIdOrDefault(*node) == ascir::TemplateId::kIndirectLoadSK;
  });
  ASSERT_NE(sk_task, tasks.end());

  auto &grouped_graphs = sk_task->grouped_graphs;
  ASSERT_EQ(grouped_graphs.size(), 4UL);
  EXPECT_NE(grouped_graphs[0].FindNode("x"), nullptr);
  EXPECT_NE(grouped_graphs[1].FindNode("index"), nullptr);
  const auto indirect_load = grouped_graphs[2].FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  EXPECT_NE(grouped_graphs[3].FindNode("y"), nullptr);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, GeneratorTaskRestoresSkLogicalViewAfterPartition) {
  auto graph = BuildIndirectLoadGraph(2);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<optimize::ScheduleTask> tasks;
  ASSERT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  const auto sk_task = std::find_if(tasks.begin(), tasks.end(), [](const optimize::ScheduleTask &task) {
    const auto node = task.optimize_graph.FindNode("indirect_load");
    return node != nullptr && ascir::GetTemplateIdOrDefault(*node) == ascir::TemplateId::kIndirectLoadSK;
  });
  ASSERT_NE(sk_task, tasks.end());

  const auto grouped_graph =
      std::find_if(sk_task->grouped_graphs.begin(), sk_task->grouped_graphs.end(),
                   [](const af::AscGraph &candidate) { return candidate.FindNode("indirect_load") != nullptr; });
  ASSERT_NE(grouped_graph, sk_task->grouped_graphs.end());
  const auto indirect_load = grouped_graph->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);

  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  EXPECT_EQ(logical_view.input.axis_ids, indirect_load->inputs()[0]->attr.axis);
  EXPECT_EQ(logical_view.input.sizes, indirect_load->inputs()[0]->attr.repeats);
  EXPECT_EQ(logical_view.input.strides, indirect_load->inputs()[0]->attr.strides);
  EXPECT_EQ(logical_view.index.axis_ids, indirect_load->inputs()[1]->attr.axis);
  EXPECT_EQ(logical_view.index.sizes, indirect_load->inputs()[1]->attr.repeats);
  EXPECT_EQ(logical_view.index.strides, indirect_load->inputs()[1]->attr.strides);
  EXPECT_EQ(logical_view.output.axis_ids, indirect_load->outputs()[0]->attr.axis);
  EXPECT_EQ(logical_view.output.sizes, indirect_load->outputs()[0]->attr.repeats);
  EXPECT_EQ(logical_view.output.strides, indirect_load->outputs()[0]->attr.strides);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimdNormalizesNonExpandedGraph) {
  auto graph = BuildIndirectLoadGraph(2, true);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd_iter, graphs.end());
  auto &simd_graph = *simd_iter;
  const auto indirect_load = simd_graph.FindNode("indirect_load");
  const auto input_load = simd_graph.FindNode("input_load");
  const auto pre_abs = simd_graph.FindNode("pre_abs");
  ASSERT_NE(indirect_load, nullptr);
  ASSERT_NE(input_load, nullptr);
  ASSERT_NE(pre_abs, nullptr);

  EXPECT_EQ(
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex),
      input_load);
  EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(input_load),
            ascgen_utils::indirect_load::TemplateRole::kSimdInputPre);
  ExpectAxisNames(simd_graph, input_load->attr.sched.axis, {"z4", "z5", "z2", "z3"});
  ExpectAxisNames(simd_graph, input_load->outputs()[0]->attr.axis, {"z4", "z5", "z2", "z3"});
  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(pre_abs, 0UL), indirect_load);
  EXPECT_EQ(ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load), pre_abs);
  ExpectAxisNames(simd_graph, pre_abs->attr.sched.axis, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simd_graph, pre_abs->outputs()[0]->attr.axis, {"z4", "z5", "z6", "z7"});

  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ExpectAxisNames(simd_graph, indirect_load->attr.sched.axis, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simd_graph, {axes.outer_axis}, {"indirect_load_outer"});
  ExpectAxisNames(simd_graph, {axes.inner_axis}, {"indirect_load_inner"});
  ExpectAxisNames(simd_graph, {axes.input_inner_axis}, {"indirect_load_input_inner"});
  ExpectMergedFrom(simd_graph, "indirect_load_input_inner", {"z2", "z3"});
  ExpectFixedTileSplit(simd_graph, axes.outer_axis);

  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  ExpectAxisNames(simd_graph, logical_view.input.axis_ids, {"z0", "z1", "z2", "z3"});
  ExpectAxisNames(simd_graph, logical_view.index.axis_ids, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simd_graph, logical_view.output.axis_ids, {"z4", "z5", "z6", "z7"});
  EXPECT_EQ(logical_view.input.strides.size(), 4UL);
  EXPECT_EQ(logical_view.index.strides.size(), 4UL);
  EXPECT_EQ(logical_view.output.strides.size(), 4UL);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimdNormalizesNegativeAxis) {
  auto graph = BuildIndirectLoadGraph(-2);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd_iter, graphs.end());
  const auto indirect_load = simd_iter->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ExpectAxisNames(*simd_iter, {axes.outer_axis}, {"indirect_load_outer"});
  ExpectAxisNames(*simd_iter, {axes.inner_axis}, {"indirect_load_inner"});
  ExpectAxisNames(*simd_iter, {axes.input_inner_axis}, {"indirect_load_input_inner"});
  ExpectMergedFrom(*simd_iter, "indirect_load_input_inner", {"z2", "z3"});
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  ExpectAxisNames(*simd_iter, logical_view.input.axis_ids, {"z0", "z1", "z2", "z3"});
  ExpectAxisNames(*simd_iter, logical_view.index.axis_ids, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(*simd_iter, logical_view.output.axis_ids, {"z4", "z5", "z6", "z7"});
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimdRegionMetadataAndSimtRejectsMultiInputRegion) {
  auto graph = BuildExpandedMultiInputGraph(true, 2L, 16L);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  ASSERT_EQ(graphs.size(), 3UL);
  const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd_iter, graphs.end());
  const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_EQ(simt_iter, graphs.end());
  auto &simd_graph = *simd_iter;
  const auto indirect_load = simd_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);

  const auto copy_sign = simd_graph.FindNode("x_copy_sign");
  ASSERT_NE(copy_sign, nullptr);
  EXPECT_EQ(
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex),
      copy_sign);
  for (const char *name : {"load0", "load1", "x_copy_sign"}) {
    const auto node = simd_graph.FindNode(name);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(node),
              ascgen_utils::indirect_load::TemplateRole::kSimdInputPre);
    ExpectAxisNames(simd_graph, node->attr.sched.axis, {"y0", "x1"});
    ExpectAxisNames(simd_graph, node->outputs()[0]->attr.axis, {"y0", "x1"});
  }
  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ExpectAxisNames(simd_graph, {axes.outer_axis, axes.inner_axis, axes.input_inner_axis}, {"y0", "y1", "x1"});
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  ExpectAxisNames(simd_graph, logical_view.input.axis_ids, {"x0", "x1"});
  ExpectAxisNames(simd_graph, logical_view.index.axis_ids, {"y0", "y1"});
  ExpectAxisNames(simd_graph, logical_view.output.axis_ids, {"y0", "y1"});
}

TEST(IndirectLoadScheduleCaseGeneratorTest, PostReduceMetadataCoversReduceAxisLayouts) {
  const std::vector<std::pair<std::string, std::string>> layouts = {
      {"R", "RRR"}, {"AR", "ARR"}, {"RA", "RRA"}, {"ARA", "ARA"}};
  for (const auto &layout : layouts) {
    const auto &name = layout.first;
    const auto &suffix = layout.second;
    auto graph = BuildPostReduceGraph(suffix);
    const auto input_axes = graph.FindNode("x")->outputs()[0]->attr.axis;
    const auto output_axes = graph.FindNode("indirect_load")->outputs()[0]->attr.axis;
    const auto expected_input_strides = DenseStrides(graph, input_axes);
    const auto expected_output_strides = DenseStrides(graph, output_axes);
    optimize::IndirectLoadScheduleCaseGenerator generator;
    std::vector<af::AscGraph> graphs;
    std::vector<std::string> score_functions;
    ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS) << "layout=" << name;
    const size_t expected_candidates = name == "ARA" ? 2UL : 4UL;
    ASSERT_EQ(graphs.size(), expected_candidates) << "layout=" << name;

    const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
    const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
    ASSERT_NE(simt_iter, graphs.end()) << "layout=" << name;
    auto &simt_graph = *simt_iter;
    const auto simt_indirect_load = simt_graph.FindNode("indirect_load");
    ASSERT_NE(simt_indirect_load, nullptr);
    ascgen_utils::indirect_load::TemplateAxes simt_axes;
    ascgen_utils::indirect_load::TemplateLogicalView simt_view;
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(simt_indirect_load, simt_axes), af::SUCCESS);
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(simt_indirect_load, simt_view), af::SUCCESS);
    const size_t first_reduce = suffix.find('R') + 1UL;
    const std::vector<af::AxisId> simt_outer(output_axes.begin(),
                                             output_axes.begin() + static_cast<int64_t>(first_reduce));
    const std::vector<af::AxisId> simt_inner(output_axes.begin() + static_cast<int64_t>(first_reduce),
                                             output_axes.end());
    ExpectAxisOrigins(simt_graph, simt_axes.outer_axis, simt_outer);
    ExpectAxisOrigins(simt_graph, simt_axes.inner_axis, simt_inner);
    EXPECT_EQ(simt_axes.input_inner_axis, af::kIdNone);
    ExpectFixedTileSplit(simt_graph, simt_axes.outer_axis);
    EXPECT_EQ(simt_view.input.axis_ids, input_axes);
    EXPECT_EQ(simt_view.index.axis_ids, output_axes);
    EXPECT_EQ(simt_view.output.axis_ids, output_axes);
    EXPECT_EQ(simt_view.input.strides, expected_input_strides);
    EXPECT_EQ(simt_view.index.strides, expected_output_strides);
    EXPECT_EQ(simt_view.output.strides, expected_output_strides);

    if (name == "ARA") {
      EXPECT_EQ(simd_iter, graphs.end());
      continue;
    }
    ASSERT_NE(simd_iter, graphs.end()) << "layout=" << name;
    auto &simd_graph = *simd_iter;
    const auto simd_indirect_load = simd_graph.FindNode("indirect_load");
    ASSERT_NE(simd_indirect_load, nullptr);
    ascgen_utils::indirect_load::TemplateAxes simd_axes;
    ascgen_utils::indirect_load::TemplateLogicalView simd_view;
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(simd_indirect_load, simd_axes), af::SUCCESS);
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(simd_indirect_load, simd_view), af::SUCCESS);
    const std::vector<af::AxisId> simd_outer(output_axes.begin(), output_axes.begin() + 1L);
    const std::vector<af::AxisId> simd_inner(output_axes.begin() + 1L, output_axes.end());
    const std::vector<af::AxisId> simd_input_inner(input_axes.begin() + 1L, input_axes.end());
    ExpectAxisOrigins(simd_graph, simd_axes.outer_axis, simd_outer);
    ExpectAxisOrigins(simd_graph, simd_axes.inner_axis, simd_inner);
    ExpectAxisOrigins(simd_graph, simd_axes.input_inner_axis, simd_input_inner);
    ExpectFixedTileSplit(simd_graph, simd_axes.outer_axis);
    EXPECT_EQ(simd_view.input.axis_ids, input_axes);
    EXPECT_EQ(simd_view.index.axis_ids, output_axes);
    EXPECT_EQ(simd_view.output.axis_ids, output_axes);
    EXPECT_EQ(simd_view.input.strides, expected_input_strides);
    EXPECT_EQ(simd_view.index.strides, expected_output_strides);
    EXPECT_EQ(simd_view.output.strides, expected_output_strides);
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, PostReduceRejectsSimdWhenOuterAxisIsReduced) {
  auto graph = BuildPostReduceGraph("RRR", true);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);

  EXPECT_EQ(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd), graphs.end());
  EXPECT_NE(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt), graphs.end());
}

TEST(IndirectLoadScheduleCaseGeneratorTest, PostReduceRejectsMultipleReduceSegmentsForBothTemplates) {
  auto graph = BuildPostReduceGraph("RAR");
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);

  EXPECT_EQ(graphs.size(), 1UL);
  EXPECT_EQ(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd), graphs.end());
  EXPECT_EQ(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt), graphs.end());
  EXPECT_NE(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSK), graphs.end());
}

TEST(IndirectLoadScheduleCaseGeneratorTest, PostReduceSkipsCommonZeroStrideAxes) {
  for (const std::string suffix : {"RBR", "BRA"}) {
    auto graph = BuildPostReduceGraph(suffix);
    const auto indirect_load = graph.FindNode("indirect_load");
    const auto reduce = graph.FindNode("sum");
    const auto common_zero_view = graph.FindNode("common_zero_view");
    ASSERT_NE(indirect_load, nullptr);
    ASSERT_NE(reduce, nullptr);
    ASSERT_NE(common_zero_view, nullptr);
    EXPECT_EQ(common_zero_view->inputs()[0]->attr.dtype, af::DT_FLOAT16);
    EXPECT_EQ(common_zero_view->outputs()[0]->attr.dtype, af::DT_FLOAT16);
    const auto output_axes = indirect_load->outputs()[0]->attr.axis;
    const size_t common_zero = suffix.find('B') + 1UL;
    EXPECT_EQ(indirect_load->outputs()[0]->attr.strides, DenseStrides(graph, output_axes));
    EXPECT_EQ(reduce->inputs()[0]->attr.strides[common_zero], af::sym::kSymbolZero);
    EXPECT_EQ(reduce->outputs()[0]->attr.strides[common_zero], af::sym::kSymbolZero);
    optimize::IndirectLoadScheduleCaseGenerator generator;
    std::vector<af::AscGraph> graphs;
    std::vector<std::string> score_functions;
    ge::PlatformContext::GetInstance().SetPlatform("3510");
    const auto status = generator.Generate(graph, graphs, score_functions);
    ge::PlatformContext::GetInstance().Reset();
    ASSERT_EQ(status, af::SUCCESS) << "suffix=" << suffix;
    EXPECT_NE(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd), graphs.end())
        << "suffix=" << suffix;
    EXPECT_NE(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt), graphs.end())
        << "suffix=" << suffix;
    ASSERT_EQ(graphs.size(), 4UL) << "suffix=" << suffix;

    const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
    ASSERT_NE(simt_iter, graphs.end()) << "suffix=" << suffix;
    const auto simt_indirect_load = simt_iter->FindNode("indirect_load");
    ASSERT_NE(simt_indirect_load, nullptr);
    ascgen_utils::indirect_load::TemplateAxes axes;
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(simt_indirect_load, axes), af::SUCCESS);
    const size_t first_reduce = suffix.find('R') + 1UL;
    ExpectAxisOrigins(*simt_iter, axes.outer_axis,
                      std::vector<af::AxisId>(output_axes.begin(), output_axes.begin() + first_reduce));
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, PostReduceRejectsBothTemplatesForUnknownOrIllegalInputStride) {
  struct StrideCase {
    const char *name;
    const char *suffix;
    size_t stride_index;
    bool use_unknown;
    bool mutate_output;
  };
  const std::vector<StrideCase> cases = {
      {"outer_input_unknown", "RRR", 0UL, true, false},        {"outer_output_unknown", "RRR", 0UL, true, true},
      {"outer_zero_output_nonzero", "RRR", 0UL, false, false}, {"inner_input_unknown", "RRR", 1UL, true, false},
      {"inner_output_unknown", "RRR", 1UL, true, true},        {"inner_zero_output_nonzero", "ARR", 1UL, false, false}};
  for (const auto &test_case : cases) {
    auto graph = BuildPostReduceGraph(test_case.suffix);
    const auto reduce = graph.FindNode("sum");
    ASSERT_NE(reduce, nullptr) << "case=" << test_case.name;
    ASSERT_FALSE(reduce->inputs().empty()) << "case=" << test_case.name;
    ASSERT_FALSE(reduce->outputs().empty()) << "case=" << test_case.name;
    const auto reduce_input = reduce->inputs()[0];
    const auto reduce_output = reduce->outputs()[0];
    ASSERT_NE(reduce_input, nullptr) << "case=" << test_case.name;
    ASSERT_NE(reduce_output, nullptr) << "case=" << test_case.name;
    const auto output_strides = reduce_output->attr.strides;
    af::Expression input_stride = af::ops::Zero;
    if (test_case.use_unknown) {
      input_stride = graph.CreateSizeVar("unknown_stride");
    }
    auto &strides = test_case.mutate_output ? reduce_output->attr.strides : reduce_input->attr.strides;
    strides[test_case.stride_index] = input_stride;

    EXPECT_EQ(strides[test_case.stride_index], input_stride) << "case=" << test_case.name;
    if (!test_case.mutate_output) {
      EXPECT_EQ(reduce_output->attr.strides, output_strides) << "case=" << test_case.name;
    }
    if (test_case.use_unknown) {
      EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(input_stride, af::ops::Zero), af::TriBool::kUnknown)
          << "case=" << test_case.name;
    }

    optimize::IndirectLoadScheduleCaseGenerator generator;
    std::vector<af::AscGraph> graphs;
    std::vector<std::string> score_functions;
    ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS) << "case=" << test_case.name;
    EXPECT_EQ(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd), graphs.end())
        << "case=" << test_case.name;
    EXPECT_EQ(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt), graphs.end())
        << "case=" << test_case.name;
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, OutputPostTopologySelectsEligibleTemplatesAndPreservesChain) {
  const std::vector<PostTopologyCase> cases = {
      {OutputPostTopology::kSum, {}, true, af::DT_FLOAT16, false},
      {OutputPostTopology::kExp2Sum, {"output_exp2"}, false, af::DT_FLOAT16, false},
      {OutputPostTopology::kAbsExp2Sum, {"output_abs", "output_exp2"}, false, af::DT_FLOAT16, false},
      {OutputPostTopology::kCastSum, {"output_cast"}, false, af::DT_FLOAT, false},
      {OutputPostTopology::kAddSum, {"output_add"}, false, af::DT_FLOAT16, false},
      {OutputPostTopology::kBesselK0Sum, {"output_modified_bessel_k0"}, false, af::DT_FLOAT16, false},
      {OutputPostTopology::kBesselK0, {"output_modified_bessel_k0"}, false, af::DT_FLOAT16, true},
  };
  for (const auto &test_case : cases) {
    auto graph = BuildPostReduceGraph("R", false, test_case.topology);
    ExpectPostTopologyGraph(graph, test_case);

    optimize::IndirectLoadScheduleCaseGenerator generator;
    std::vector<af::AscGraph> graphs;
    std::vector<std::string> score_functions;
    ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
    const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
    const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
    ASSERT_NE(simd_iter, graphs.end());
    EXPECT_EQ(simt_iter != graphs.end(), test_case.expect_simt);
    ExpectPostTopologyGraph(*simd_iter, test_case);
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, ShapeEnlargingIndirectLoadStopsSimdInputPreButNotSimt) {
  auto graph = BuildShapeEnlargingInputPreGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  const auto status = generator.Generate(graph, graphs, score_functions);
  ge::PlatformContext::GetInstance().Reset();
  ASSERT_EQ(status, af::SUCCESS);
  ASSERT_EQ(graphs.size(), 4UL);

  const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd_iter, graphs.end());
  auto &simd_graph = *simd_iter;
  const auto simd_indirect_load = simd_graph.FindNode("indirect_load");
  const auto simd_load = simd_graph.FindNode("input_load");
  const auto simd_pre_cast = simd_graph.FindNode("pre_cast");
  ASSERT_NE(simd_indirect_load, nullptr);
  ASSERT_NE(simd_load, nullptr);
  ASSERT_NE(simd_pre_cast, nullptr);
  EXPECT_EQ(
      ascgen_utils::indirect_load::GetInputProducer(simd_indirect_load, ascgen_utils::indirect_load::kInputTensorIndex),
      simd_pre_cast);
  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(simd_pre_cast, 0UL), simd_load);

  const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_NE(simt_iter, graphs.end());
  auto &simt_graph = *simt_iter;
  const auto simt_indirect_load = simt_graph.FindNode("indirect_load");
  const auto simt_load = simt_graph.FindNode("input_load");
  const auto simt_pre_cast = simt_graph.FindNode("pre_cast");
  ASSERT_NE(simt_indirect_load, nullptr);
  ASSERT_NE(simt_load, nullptr);
  ASSERT_NE(simt_pre_cast, nullptr);
  EXPECT_EQ(
      ascgen_utils::indirect_load::GetInputProducer(simt_indirect_load, ascgen_utils::indirect_load::kInputTensorIndex),
      simt_load);
  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(simt_pre_cast, 0UL), simt_indirect_load);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, GenerateFailsWhenNonAxisInputDimensionIsSmallerThanIndex) {
  auto graph = BuildExpandedMultiInputGraph(false, 3L, 17L);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  EXPECT_NE(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_TRUE(graphs.empty());
}

TEST(IndirectLoadScheduleCaseGeneratorTest, GeneratedSkCandidateUsesSkBehavior) {
  auto graphs = GenerateIndirectLoadCases(2);
  const auto sk_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSK);
  ASSERT_NE(sk_iter, graphs.end());
  auto &sk_graph = *sk_iter;
  const auto indirect_load = sk_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);

  const auto behavior = ascgen_utils::indirect_load::GetTemplateBehavior(indirect_load);
  EXPECT_FALSE(behavior.skips_main_schedule_tiling);
  EXPECT_FALSE(behavior.skips_api_emit);
  EXPECT_FALSE(behavior.uses_direct_gm_pipeline);
  EXPECT_FALSE(behavior.skips_ub_lifecycle);
  EXPECT_FALSE(behavior.preserves_vectorized_axis);
  EXPECT_FALSE(ascgen_utils::indirect_load::ShouldDisableRegularVectorFunc(indirect_load));

  const auto input_boundary = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  ASSERT_NE(input_boundary, nullptr);
  const auto input_behavior = ascgen_utils::indirect_load::GetTemplateBehavior(input_boundary);
  EXPECT_TRUE(input_behavior.skips_main_schedule_tiling);
  EXPECT_TRUE(input_behavior.skips_api_emit);
  EXPECT_FALSE(input_behavior.uses_direct_gm_pipeline);
  EXPECT_FALSE(input_behavior.skips_ub_lifecycle);
  EXPECT_TRUE(input_behavior.preserves_vectorized_axis);
  EXPECT_TRUE(ascgen_utils::indirect_load::ShouldApplyInputInnerVectorization(input_boundary));
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimdMovesInputPreAfterIndirectLoad) {
  auto graph = BuildIndirectLoadGraph(2, true);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd_iter, graphs.end());
  auto &simd_graph = *simd_iter;
  const auto indirect_load = simd_graph.FindNode("indirect_load");
  const auto pre_abs = simd_graph.FindNode("pre_abs");
  ASSERT_NE(indirect_load, nullptr);
  ASSERT_NE(pre_abs, nullptr);

  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL)->GetName(), "input_load");
  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(pre_abs, 0UL), indirect_load);
  EXPECT_EQ(ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load), pre_abs);
  ExpectAxisNames(simd_graph, pre_abs->attr.sched.axis, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simd_graph, pre_abs->outputs()[0]->attr.axis, {"z4", "z5", "z6", "z7"});
}

struct FullPrefixNodeCase {
  const char *name;
  af::DataType dtype;
};

void ExpectFullPrefixChain(af::AscGraph &graph, const std::vector<std::pair<const char *, const char *>> &edges,
                           const std::vector<FullPrefixNodeCase> &nodes, const std::vector<af::AxisId> &axes,
                           const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  for (const auto &edge : edges) {
    const auto producer = graph.FindNode(edge.first);
    const auto consumer = graph.FindNode(edge.second);
    ASSERT_NE(producer, nullptr) << edge.first;
    ASSERT_NE(consumer, nullptr) << edge.second;
    const size_t input_index =
        std::strcmp(edge.second, "indirect_load") == 0 && std::strcmp(edge.first, "index_floor_to_int") == 0 ? 1UL
                                                                                                             : 0UL;
    const auto actual_producer = ascgen_utils::indirect_load::GetInputProducer(consumer, input_index);
    ASSERT_NE(actual_producer, nullptr) << edge.second;
    EXPECT_EQ(actual_producer->GetName(), producer->GetName()) << edge.second;
  }
  for (const auto &test_case : nodes) {
    const auto node = graph.FindNode(test_case.name);
    ASSERT_NE(node, nullptr) << test_case.name;
    ASSERT_FALSE(node->outputs().empty());
    const auto &output = node->outputs()[0]->attr;
    EXPECT_EQ(output.dtype, test_case.dtype) << test_case.name;
    EXPECT_EQ(output.axis, axes) << test_case.name;
    EXPECT_EQ(output.repeats, repeats) << test_case.name;
    EXPECT_EQ(output.strides, strides) << test_case.name;
  }
}

void ExpectFullPrefixOriginalGraph(af::AscGraph &graph,
                                   const std::vector<std::pair<const char *, const char *>> &input_edges,
                                   const std::vector<std::pair<const char *, const char *>> &index_edges) {
  const auto x_output = graph.FindNode("x")->outputs()[0]->attr;
  const auto index_output = graph.FindNode("index")->outputs()[0]->attr;
  ExpectFullPrefixChain(graph, input_edges, {{"input_relu", af::DT_FLOAT16}, {"input_exp2", af::DT_FLOAT16}},
                        x_output.axis, x_output.repeats, x_output.strides);
  ExpectFullPrefixChain(graph, index_edges,
                        {{"index_abs", af::DT_INT32},
                         {"index_abs2", af::DT_INT32},
                         {"index_cast_float", af::DT_FLOAT},
                         {"index_exp2", af::DT_FLOAT},
                         {"index_log2", af::DT_FLOAT},
                         {"index_floor_to_int", af::DT_INT32}},
                        index_output.axis, index_output.repeats, index_output.strides);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimdFullPrefixPreservesInputAndIndexPreMetadata) {
  auto graph = BuildIndirectLoadGraph(2, true, true);
  const std::vector<std::pair<const char *, const char *>> input_edges = {
      {"input_load", "input_relu"}, {"input_relu", "input_exp2"}, {"input_exp2", "indirect_load"}};
  const std::vector<std::pair<const char *, const char *>> index_edges = {{"index_load", "index_abs"},
                                                                          {"index_abs", "index_abs2"},
                                                                          {"index_abs2", "index_cast_float"},
                                                                          {"index_cast_float", "index_exp2"},
                                                                          {"index_exp2", "index_log2"},
                                                                          {"index_log2", "index_floor_to_int"},
                                                                          {"index_floor_to_int", "indirect_load"}};
  ExpectFullPrefixOriginalGraph(graph, input_edges, index_edges);

  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd_iter, graphs.end());
  const auto indirect_load = simd_iter->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  const auto input_producer = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  const auto index_producer = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 1UL);
  ASSERT_NE(input_producer, nullptr);
  ASSERT_NE(index_producer, nullptr);
  EXPECT_EQ(input_producer->GetName(), "input_load");
  EXPECT_EQ(index_producer->GetName(), "index_floor_to_int");
  const std::vector<af::AxisId> moved_axes = indirect_load->outputs()[0]->attr.axis;
  const std::vector<af::Expression> moved_repeats = indirect_load->outputs()[0]->attr.repeats;
  const std::vector<af::Expression> moved_strides = indirect_load->outputs()[0]->attr.strides;
  ExpectFullPrefixChain(
      *simd_iter, {{"input_load", "indirect_load"}, {"indirect_load", "input_relu"}, {"input_relu", "input_exp2"}},
      {{"input_relu", af::DT_FLOAT16}, {"input_exp2", af::DT_FLOAT16}}, moved_axes, moved_repeats, moved_strides);
  const auto index_output = graph.FindNode("index")->outputs()[0]->attr;
  ExpectFullPrefixChain(*simd_iter, index_edges,
                        {{"index_abs", af::DT_INT32},
                         {"index_abs2", af::DT_INT32},
                         {"index_cast_float", af::DT_FLOAT},
                         {"index_exp2", af::DT_FLOAT},
                         {"index_log2", af::DT_FLOAT},
                         {"index_floor_to_int", af::DT_INT32}},
                        index_output.axis, index_output.repeats, index_output.strides);
  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  EXPECT_NE(axes.input_inner_axis, af::kIdNone);
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  EXPECT_EQ(logical_view.input.axis_ids, graph.FindNode("x")->outputs()[0]->attr.axis);
  EXPECT_EQ(logical_view.index.axis_ids, graph.FindNode("index")->outputs()[0]->attr.axis);
  EXPECT_EQ(logical_view.output.axis_ids, graph.FindNode("indirect_load")->outputs()[0]->attr.axis);
  EXPECT_EQ(logical_view.input.strides, graph.FindNode("x")->outputs()[0]->attr.strides);
  EXPECT_EQ(logical_view.index.strides, graph.FindNode("index")->outputs()[0]->attr.strides);
  EXPECT_EQ(logical_view.output.strides, graph.FindNode("indirect_load")->outputs()[0]->attr.strides);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimtHandlesInputPrecisionCast) {
  auto graph = BuildIndirectLoadPrecisionCastGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  const auto status = generator.Generate(graph, graphs, score_functions);
  ge::PlatformContext::GetInstance().Reset();
  ASSERT_EQ(status, af::SUCCESS);
  ASSERT_EQ(graphs.size(), 4UL);

  const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_NE(simt_iter, graphs.end());
  auto &simt_graph = *simt_iter;
  const auto input_cast = simt_graph.FindNode("input_cast");
  EXPECT_NE(input_cast, nullptr);
  EXPECT_NE(simt_graph.FindNode("output_cast"), nullptr);
  const auto simt_indirect_load = simt_graph.FindNode("indirect_load");
  const auto output_exp = simt_graph.FindNode("output_exp");
  ASSERT_NE(simt_indirect_load, nullptr);
  ASSERT_NE(output_exp, nullptr);
  EXPECT_EQ(simt_indirect_load->outputs()[0]->attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(output_exp->outputs()[0]->attr.dtype, af::DT_FLOAT);
  const auto input_producer =
      ascgen_utils::indirect_load::GetInputProducer(simt_indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
  const auto cast_producer = ascgen_utils::indirect_load::GetInputProducer(input_cast, 0UL);
  const auto cast_consumer = ascgen_utils::indirect_load::GetOnlyOutputConsumer(input_cast);
  ASSERT_NE(input_producer, nullptr);
  EXPECT_EQ(input_producer->GetName(), "input_load");
  EXPECT_EQ(cast_producer, simt_indirect_load);
  EXPECT_EQ(cast_consumer, output_exp);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, GenerateFailsWhenAxisOutOfRange) {
  for (int64_t axis : {-5L, 8L}) {
    auto graph = BuildIndirectLoadGraph(axis);
    optimize::IndirectLoadScheduleCaseGenerator generator;
    std::vector<af::AscGraph> graphs;
    std::vector<std::string> score_functions;
    EXPECT_NE(generator.Generate(graph, graphs, score_functions), af::SUCCESS) << "axis=" << axis;
    EXPECT_TRUE(graphs.empty());
  }
}

}  // namespace
