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

void ExpectFixedTileSplit(af::AscGraph &graph, const std::string &axis_name) {
  const af::AxisId outer_axis_id = FindAxisByName(graph, axis_name);
  const af::AxisId tile_outer_axis_id = FindAxisByName(graph, axis_name + "T");
  const af::AxisId tile_inner_axis_id = FindAxisByName(graph, axis_name + "t");
  ASSERT_NE(outer_axis_id, af::kIdNone);
  ASSERT_NE(tile_outer_axis_id, af::kIdNone);
  ASSERT_NE(tile_inner_axis_id, af::kIdNone);

  const auto *tile_outer_axis = graph.FindAxis(tile_outer_axis_id);
  const auto *tile_inner_axis = graph.FindAxis(tile_inner_axis_id);
  ASSERT_NE(tile_outer_axis, nullptr);
  ASSERT_NE(tile_inner_axis, nullptr);
  EXPECT_EQ(tile_outer_axis->type, af::Axis::Type::kAxisTypeTileOuter);
  EXPECT_EQ(tile_inner_axis->type, af::Axis::Type::kAxisTypeTileInner);
  ExpectAxisNames(graph, tile_outer_axis->from, {axis_name});
  ExpectAxisNames(graph, tile_inner_axis->from, {axis_name});
  EXPECT_EQ(tile_outer_axis->split_pair_other_id, tile_inner_axis_id);
  EXPECT_EQ(tile_inner_axis->split_pair_other_id, tile_outer_axis_id);
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

  af::ascir_op::Data index("index", graph);
  index.y.dtype = ge::DT_INT32;
  index.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  index.attr.api.type = af::ApiType::kAPITypeBuffer;
  index.ir_attr.SetIndex(1);
  index.attr.sched.axis = output_axes;
  *index.y.axis = output_axes;
  *index.y.repeats = output_repeats;
  *index.y.strides = output_strides;

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  if (has_input_pre_node) {
    af::ascir_op::Abs pre_abs("pre_abs");
    pre_abs.x = x.y;
    pre_abs.y.dtype = ge::DT_FLOAT16;
    pre_abs.attr.sched.axis = input_axes;
    *pre_abs.y.axis = input_axes;
    *pre_abs.y.repeats = input_repeats;
    *pre_abs.y.strides = input_strides;
    indirect_load.x1 = pre_abs.y;
  } else {
    indirect_load.x1 = x.y;
  }

  indirect_load.x2 = index.y;
  indirect_load.y.dtype = ge::DT_FLOAT16;
  indirect_load.attr.sched.axis = output_axes;
  indirect_load.ir_attr.SetAxis(axis);
  *indirect_load.y.axis = output_axes;
  *indirect_load.y.repeats = output_repeats;
  *indirect_load.y.strides = output_strides;

  af::ascir_op::Output y("y");
  y.x = indirect_load.y;
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

template <typename Op>
void SetNodeView(Op &op, af::DataType dtype, const std::vector<af::AxisId> &axes,
                 const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  op.y.dtype = dtype;
  op.attr.sched.axis = axes;
  *op.y.axis = axes;
  *op.y.repeats = repeats;
  *op.y.strides = strides;
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

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  indirect_load.x1 = input_cast.y;
  indirect_load.x2 = index.y;
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

std::vector<af::AscGraph> GenerateIndirectLoadCases(int64_t axis) {
  auto graph = BuildIndirectLoadGraph(axis);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_EQ(graphs.size(), 3UL);
  EXPECT_EQ(score_functions.size(), 3UL);
  return graphs;
}

af::AscGraph &FindGeneratedGraphByTemplate(std::vector<af::AscGraph> &graphs, ascir::TemplateId template_id) {
  const auto iter = std::find_if(graphs.begin(), graphs.end(), [template_id](const af::AscGraph &graph) {
    const auto node = graph.FindNode("indirect_load");
    return node != nullptr && ascir::GetTemplateIdOrDefault(*node) == template_id;
  });
  EXPECT_NE(iter, graphs.end());
  return *iter;
}

class IndirectLoadScheduleCaseGeneratorTest : public ::testing::TestWithParam<int64_t> {};

TEST(IndirectLoadScheduleCaseGeneratorTest, SimtSetsDcacheAndUsesUnifiedVectorizedAxisWithoutReduce) {
  auto graphs = GenerateIndirectLoadCases(2);
  auto &simt_graph = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
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
}

TEST(IndirectLoadScheduleCaseGeneratorTest, StoresTemplateAxesWithoutOverwritingSchedAxis) {
  auto graphs = GenerateIndirectLoadCases(2);
  auto &simd_graph = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  const auto indirect_load = simd_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);

  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ExpectAxisNames(simd_graph, indirect_load->attr.sched.axis, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simd_graph, {axes.outer_axis}, {"indirect_load_outer"});
  ExpectAxisNames(simd_graph, {axes.inner_axis}, {"indirect_load_inner"});
  ExpectFixedTileSplit(simd_graph, "indirect_load_outer");

  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  ExpectAxisNames(simd_graph, logical_view.data.axis_ids, {"z0", "z1", "z2", "z3"});
  ExpectAxisNames(simd_graph, logical_view.index.axis_ids, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simd_graph, logical_view.output.axis_ids, {"z4", "z5", "z6", "z7"});
  EXPECT_EQ(logical_view.data.strides.size(), 4UL);
  EXPECT_EQ(logical_view.index.strides.size(), 4UL);
  EXPECT_EQ(logical_view.output.strides.size(), 4UL);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, DoesNotStoreFixedTileAxesAsTemplateMetadata) {
  auto graphs = GenerateIndirectLoadCases(2);
  auto &simd_graph = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  const auto indirect_load = simd_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  const auto op_desc = indirect_load->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  EXPECT_EQ(op_desc->TryGetExtAttr("af.internal.indirect_load.tile_outer_axis", static_cast<int64_t>(af::kIdNone)),
            af::kIdNone);
  EXPECT_EQ(op_desc->TryGetExtAttr("af.internal.indirect_load.tile_inner_axis", static_cast<int64_t>(af::kIdNone)),
            af::kIdNone);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SkBuildsInputInnerAxisFromInputBoundary) {
  auto graph = BuildIndirectLoadGraph(2, true);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  auto &sk_graph = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSK);
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
  EXPECT_EQ(logical_view.data.axis_ids, indirect_load->inputs()[0]->attr.axis);
  EXPECT_EQ(logical_view.data.strides, indirect_load->inputs()[0]->attr.strides);
  EXPECT_EQ(logical_view.index.axis_ids, indirect_load->inputs()[1]->attr.axis);
  EXPECT_EQ(logical_view.index.strides, indirect_load->inputs()[1]->attr.strides);
  EXPECT_EQ(logical_view.output.axis_ids, indirect_load->outputs()[0]->attr.axis);
  EXPECT_EQ(logical_view.output.strides, indirect_load->outputs()[0]->attr.strides);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, GeneratedSimdCandidateKeepsPublicBehavior) {
  auto graph = BuildIndirectLoadGraph(2, true);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  auto &simd_graph = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  const auto indirect_load = simd_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);

  const auto il_behavior = ascgen_utils::indirect_load::GetTemplateBehavior(indirect_load);
  EXPECT_FALSE(il_behavior.skips_main_schedule_tiling);
  EXPECT_FALSE(il_behavior.skips_api_emit);
  EXPECT_FALSE(il_behavior.uses_direct_gm_pipeline);
  EXPECT_FALSE(il_behavior.skips_ub_lifecycle);
  EXPECT_FALSE(il_behavior.preserves_vectorized_axis);
  EXPECT_FALSE(ascgen_utils::indirect_load::ShouldDisableRegularVectorFunc(indirect_load));
}

TEST(IndirectLoadScheduleCaseGeneratorTest, GeneratedSimtCandidateKeepsPublicBehavior) {
  auto graphs = GenerateIndirectLoadCases(2);
  auto &simt_graph = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  const auto indirect_load = simt_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);

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

TEST(IndirectLoadScheduleCaseGeneratorTest, GeneratedSkCandidateUsesSkBehavior) {
  auto graphs = GenerateIndirectLoadCases(2);
  auto &sk_graph = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSK);
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
  auto &simd_graph = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  const auto indirect_load = simd_graph.FindNode("indirect_load");
  const auto pre_abs = simd_graph.FindNode("pre_abs");
  ASSERT_NE(indirect_load, nullptr);
  ASSERT_NE(pre_abs, nullptr);

  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL)->GetName(), "x");
  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(pre_abs, 0UL), indirect_load);
  EXPECT_EQ(ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load), pre_abs);
  ExpectAxisNames(simd_graph, pre_abs->attr.sched.axis, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simd_graph, pre_abs->outputs()[0]->attr.axis, {"z4", "z5", "z6", "z7"});
}

TEST(IndirectLoadScheduleCaseGeneratorTest, MovesInputPrecisionCastAfterIndirectLoad) {
  auto graph = BuildIndirectLoadPrecisionCastGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  const auto status = generator.Generate(graph, graphs, score_functions);
  ge::PlatformContext::GetInstance().Reset();
  ASSERT_EQ(status, af::SUCCESS);
  ASSERT_EQ(graphs.size(), 3UL);

  auto &simd_graph = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  EXPECT_NE(simd_graph.FindNode("input_cast"), nullptr);
  EXPECT_NE(simd_graph.FindNode("output_cast"), nullptr);
  ASSERT_NE(simd_graph.FindNode("indirect_load"), nullptr);
  ASSERT_NE(simd_graph.FindNode("output_exp"), nullptr);
  EXPECT_EQ(simd_graph.FindNode("indirect_load")->outputs()[0]->attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(simd_graph.FindNode("output_exp")->outputs()[0]->attr.dtype, af::DT_FLOAT);
  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(simd_graph.FindNode("indirect_load"), 0UL)->GetName(),
            "input_load");
  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(simd_graph.FindNode("input_cast"), 0UL),
            simd_graph.FindNode("indirect_load"));
  EXPECT_EQ(ascgen_utils::indirect_load::GetOnlyOutputConsumer(simd_graph.FindNode("input_cast"))->GetName(),
            "output_exp");

  auto &simt_graph = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  const auto input_cast = simt_graph.FindNode("input_cast");
  EXPECT_NE(input_cast, nullptr);
  EXPECT_NE(simt_graph.FindNode("output_cast"), nullptr);
  const auto simt_indirect_load = simt_graph.FindNode("indirect_load");
  const auto output_exp = simt_graph.FindNode("output_exp");
  ASSERT_NE(simt_indirect_load, nullptr);
  ASSERT_NE(output_exp, nullptr);
  EXPECT_EQ(simt_indirect_load->outputs()[0]->attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(output_exp->outputs()[0]->attr.dtype, af::DT_FLOAT);
  const auto input_producer = ascgen_utils::indirect_load::GetInputProducer(simt_indirect_load, 0UL);
  const auto cast_producer = ascgen_utils::indirect_load::GetInputProducer(input_cast, 0UL);
  const auto cast_consumer = ascgen_utils::indirect_load::GetOnlyOutputConsumer(input_cast);
  ASSERT_NE(input_producer, nullptr);
  EXPECT_EQ(input_producer->GetName(), "input_load");
  EXPECT_EQ(cast_producer, simt_indirect_load);
  EXPECT_EQ(cast_consumer, output_exp);
}

TEST_P(IndirectLoadScheduleCaseGeneratorTest, SimdSplitsOuterAndInnerAxesByNormalizedAxis) {
  const int64_t axis = GetParam();
  auto graphs = GenerateIndirectLoadCases(axis);
  auto &simd_graph = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  const auto indirect_load = simd_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);

  const int64_t output_rank = 4L;
  const size_t expected_axis_index = static_cast<size_t>(axis < 0L ? axis + output_rank : axis);
  const std::vector<std::string> output_axis_names = {"z4", "z5", "z6", "z7"};
  std::vector<std::string> expected_outer;
  std::vector<std::string> expected_inner;
  expected_outer.assign(output_axis_names.begin(),
                        output_axis_names.begin() + static_cast<int64_t>(expected_axis_index));
  expected_inner.assign(output_axis_names.begin() + static_cast<int64_t>(expected_axis_index), output_axis_names.end());

  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ExpectAxisNames(simd_graph, indirect_load->attr.sched.axis, output_axis_names);
  if (expected_outer.empty()) {
    EXPECT_NE(FindAxisByName(simd_graph, "indirect_load_single_outer"), af::kIdNone);
    ExpectAxisNames(simd_graph, {axes.outer_axis}, {"indirect_load_single_outer"});
  } else {
    ExpectMergedFrom(simd_graph, "indirect_load_outer", expected_outer);
    ExpectAxisNames(simd_graph, {axes.outer_axis}, {"indirect_load_outer"});
  }

  if (expected_inner.size() > 1UL) {
    ExpectMergedFrom(simd_graph, "indirect_load_inner", expected_inner);
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, GenerateFailsWhenAxisOutOfRange) {
  for (int64_t axis : {-5L, 8L}) {
    auto graph = BuildIndirectLoadGraph(axis);
    optimize::IndirectLoadScheduleCaseGenerator generator;
    std::vector<af::AscGraph> graphs;
    std::vector<std::string> score_functions;
    EXPECT_NE(generator.Generate(graph, graphs, score_functions), af::SUCCESS) << "axis=" << axis;
    EXPECT_TRUE(graphs.empty());
    EXPECT_TRUE(score_functions.empty());
  }
}

INSTANTIATE_TEST_SUITE_P(AxisBoundary, IndirectLoadScheduleCaseGeneratorTest, ::testing::Values(-1, 0, 2, 3));
}  // namespace
