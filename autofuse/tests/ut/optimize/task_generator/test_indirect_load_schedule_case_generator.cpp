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
#include <string>
#include <utility>
#include <vector>

#include "ascir_ops.h"
#include "common/platform_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "indirect_load_utils.h"
#include "schedule_result.h"
#include "task_generator/indirect_load_schedule_case_generator.h"

namespace {
constexpr int64_t kSimtDcacheSize = 32 * 1024;

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

af::AscGraph BuildIndirectLoadGraph(int64_t axis, bool has_input_pre_node = false) {
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
  if (has_input_pre_node) {
    af::ascir_op::Abs pre_abs("pre_abs");
    pre_abs.x = input_load.y;
    pre_abs.y.dtype = ge::DT_FLOAT16;
    pre_abs.attr.sched.axis = input_axes;
    *pre_abs.y.axis = input_axes;
    *pre_abs.y.repeats = input_repeats;
    *pre_abs.y.strides = input_strides;
    indirect_load.x1 = pre_abs.y;
  } else {
    indirect_load.x1 = input_load.y;
  }

  indirect_load.x2 = index_load.y;
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

af::AscGraph BuildPostReduceGraph(const std::string &suffix, bool reduce_outer = false) {
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
  if (suffix.find('B') == std::string::npos) {
    sum.x = indirect_load.y;
  } else {
    af::ascir_op::Abs common_zero_view("common_zero_view");
    common_zero_view.x = indirect_load.y;
    SetNodeView(common_zero_view, af::DT_FLOAT16, output_axes, output_repeats, reduce_input_strides);
    sum.x = common_zero_view.y;
  }
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.attr.sched.axis = output_axes;
  SetNodeView(sum, af::DT_FLOAT16, output_axes, reduce_repeats, reduce_strides);
  af::ascir_op::Store store("store");
  store.x = sum.y;
  SetNodeView(store, af::DT_FLOAT16, output_axes, reduce_repeats, reduce_strides);
  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetNodeView(output, af::DT_FLOAT16, output_axes, reduce_repeats, reduce_strides);
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
  EXPECT_EQ(graphs.size(), 3UL);
  EXPECT_EQ(score_functions.size(), 3UL);
  EXPECT_EQ(score_functions[0], score_functions[1]);
  EXPECT_EQ(score_functions[1], score_functions[2]);
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
  EXPECT_EQ(logical_view.input.strides.size(), 4UL);
  EXPECT_EQ(logical_view.index.strides.size(), 4UL);
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
  EXPECT_EQ(logical_view.input.strides, indirect_load->inputs()[0]->attr.strides);
  EXPECT_EQ(logical_view.index.axis_ids, indirect_load->inputs()[1]->attr.axis);
  EXPECT_EQ(logical_view.index.strides, indirect_load->inputs()[1]->attr.strides);
  EXPECT_EQ(logical_view.output.axis_ids, indirect_load->outputs()[0]->attr.axis);
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

TEST(IndirectLoadScheduleCaseGeneratorTest, SimdRegionMetadataAndSimtRejectsMultiInputRegion) {
  auto graph = BuildExpandedMultiInputGraph(true, 2L, 16L);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  ASSERT_EQ(graphs.size(), 2UL);
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
    const size_t expected_candidates = name == "ARA" ? 2UL : 3UL;
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
    ASSERT_EQ(graphs.size(), 3UL) << "suffix=" << suffix;

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

TEST(IndirectLoadScheduleCaseGeneratorTest, ShapeEnlargingIndirectLoadStopsSimdInputPreButNotSimt) {
  auto graph = BuildShapeEnlargingInputPreGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  const auto status = generator.Generate(graph, graphs, score_functions);
  ge::PlatformContext::GetInstance().Reset();
  ASSERT_EQ(status, af::SUCCESS);
  ASSERT_EQ(graphs.size(), 3UL);

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

TEST(IndirectLoadScheduleCaseGeneratorTest, SimtHandlesInputPrecisionCast) {
  auto graph = BuildIndirectLoadPrecisionCastGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  const auto status = generator.Generate(graph, graphs, score_functions);
  ge::PlatformContext::GetInstance().Reset();
  ASSERT_EQ(status, af::SUCCESS);
  ASSERT_EQ(graphs.size(), 3UL);

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
