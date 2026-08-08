/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <functional>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "graph/utils/graph_utils.h"
#include "optimize/graph_pass/same_source_broadcast_cse_pass.h"
#include "optimize/schedule_utils.h"

namespace optimize {
namespace {
using af::ascir_op::Add;
using af::ascir_op::Broadcast;
using af::ascir_op::Data;
using af::ascir_op::Load;
using af::ascir_op::Mean;
using af::ascir_op::Output;
using af::ascir_op::Store;

struct BroadcastCseGraph {
  af::AscGraph graph{"same_source_broadcast_cse"};
  af::AscNodePtr reduce;
  af::AscNodePtr broadcast0;
  af::AscNodePtr broadcast1;
  af::AscNodePtr add;
};

void SetTensorAttr(af::AscOpOutput &tensor, const std::vector<af::AxisId> &axes,
                   const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  tensor.dtype = ge::DT_FLOAT;
  *tensor.axis = axes;
  *tensor.repeats = repeats;
  *tensor.strides = strides;
  *tensor.vectorized_axis = axes;
  *tensor.vectorized_strides = strides;
}

template <typename Op>
void SetComputeAttr(Op &op, const std::vector<af::AxisId> &axes, af::ComputeType compute_type) {
  op.attr.sched.axis = axes;
  op.attr.sched.loop_axis = axes.front();
  op.attr.api.compute_type = compute_type;
  op.attr.api.type = af::ApiType::kAPITypeCompute;
}

BroadcastCseGraph BuildBroadcastCseGraph(const std::function<void(af::AscNodePtr)> &mutate_broadcast1 = {}) {
  BroadcastCseGraph result;
  const auto rows = result.graph.CreateSizeVar(16);
  const auto columns = result.graph.CreateSizeVar(128);
  const auto row_axis = result.graph.CreateAxis("row", rows);
  const auto reduce_axis = result.graph.CreateAxis("reduce", columns);
  const std::vector<af::AxisId> axes{row_axis.id, reduce_axis.id};

  Data data("data", result.graph);
  data.ir_attr.SetIndex(0);
  data.attr.api.type = af::ApiType::kAPITypeBuffer;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  SetTensorAttr(data.y, axes, {rows, columns}, {columns, af::sym::kSymbolOne});

  Load load("load");
  load.x = data.y;
  SetComputeAttr(load, axes, af::ComputeType::kComputeLoad);
  SetTensorAttr(load.y, axes, {rows, columns}, {columns, af::sym::kSymbolOne});

  Mean mean("mean");
  mean.x = load.y;
  SetComputeAttr(mean, axes, af::ComputeType::kComputeReduce);
  SetTensorAttr(mean.y, axes, {rows, af::sym::kSymbolOne}, {af::sym::kSymbolOne, af::sym::kSymbolZero});

  Broadcast broadcast0("broadcast0");
  broadcast0.x = mean.y;
  SetComputeAttr(broadcast0, axes, af::ComputeType::kComputeBroadcast);
  SetTensorAttr(broadcast0.y, axes, {rows, columns}, {columns, af::sym::kSymbolOne});

  Broadcast broadcast1("broadcast1");
  broadcast1.x = mean.y;
  SetComputeAttr(broadcast1, axes, af::ComputeType::kComputeBroadcast);
  SetTensorAttr(broadcast1.y, axes, {rows, columns}, {columns, af::sym::kSymbolOne});

  Add add("add");
  add.x1 = broadcast0.y;
  add.x2 = broadcast1.y;
  SetComputeAttr(add, axes, af::ComputeType::kComputeElewise);
  SetTensorAttr(add.y, axes, {rows, columns}, {columns, af::sym::kSymbolOne});

  Store store("store");
  store.x = add.y;
  SetComputeAttr(store, axes, af::ComputeType::kComputeStore);
  SetTensorAttr(store.y, axes, {rows, columns}, {columns, af::sym::kSymbolOne});

  Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  output.attr.api.type = af::ApiType::kAPITypeBuffer;
  output.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  SetTensorAttr(output.y, axes, {rows, columns}, {columns, af::sym::kSymbolOne});

  result.reduce = result.graph.FindNode("mean");
  result.broadcast0 = result.graph.FindNode("broadcast0");
  result.broadcast1 = result.graph.FindNode("broadcast1");
  result.add = result.graph.FindNode("add");
  EXPECT_NE(result.reduce, nullptr);
  EXPECT_NE(result.broadcast0, nullptr);
  EXPECT_NE(result.broadcast1, nullptr);
  EXPECT_NE(result.add, nullptr);
  if (mutate_broadcast1 && result.broadcast1 != nullptr) {
    mutate_broadcast1(result.broadcast1);
  }
  EXPECT_EQ(ScheduleUtils::TopologicalSorting(result.graph), af::SUCCESS);
  return result;
}

void ExpectSeparateBroadcastInputs(const BroadcastCseGraph &test_graph) {
  const auto broadcast0 = test_graph.graph.FindNode("broadcast0");
  const auto broadcast1 = test_graph.graph.FindNode("broadcast1");
  const auto add = test_graph.graph.FindNode("add");
  ASSERT_NE(broadcast0, nullptr);
  ASSERT_NE(broadcast1, nullptr);
  ASSERT_NE(add, nullptr);
  EXPECT_EQ(add->GetInDataAnchor(0)->GetPeerOutAnchor(), broadcast0->GetOutDataAnchor(0));
  EXPECT_EQ(add->GetInDataAnchor(1)->GetPeerOutAnchor(), broadcast1->GetOutDataAnchor(0));
}
}  // namespace

TEST(SameSourceBroadcastCsePassTest, MergesEquivalentBroadcastsInNormGraph) {
  auto test_graph = BuildBroadcastCseGraph();
  ASSERT_TRUE(ScheduleUtils::IsNormStruct(test_graph.graph));

  SameSourceBroadcastCsePass pass;
  ASSERT_EQ(pass.RunPass(test_graph.graph), af::SUCCESS);

  const auto canonical = test_graph.graph.FindNode("broadcast0");
  const auto add = test_graph.graph.FindNode("add");
  ASSERT_NE(canonical, nullptr);
  ASSERT_NE(add, nullptr);
  EXPECT_EQ(test_graph.graph.FindNode("broadcast1"), nullptr);
  EXPECT_EQ(add->GetInDataAnchor(0)->GetPeerOutAnchor(), canonical->GetOutDataAnchor(0));
  EXPECT_EQ(add->GetInDataAnchor(1)->GetPeerOutAnchor(), canonical->GetOutDataAnchor(0));
  EXPECT_EQ(test_graph.reduce->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1UL);
}

TEST(SameSourceBroadcastCsePassTest, KeepsBroadcastsWhenTensorViewDiffers) {
  auto test_graph = BuildBroadcastCseGraph(
      [](const af::AscNodePtr &broadcast) { broadcast->outputs[0].attr.vectorized_strides.back() = af::Symbol(2); });
  ASSERT_TRUE(ScheduleUtils::IsNormStruct(test_graph.graph));

  SameSourceBroadcastCsePass pass;
  ASSERT_EQ(pass.RunPass(test_graph.graph), af::SUCCESS);
  ExpectSeparateBroadcastInputs(test_graph);
}

TEST(SameSourceBroadcastCsePassTest, KeepsBroadcastsWhenScheduleDiffers) {
  auto test_graph = BuildBroadcastCseGraph([](const af::AscNodePtr &broadcast) {
    broadcast->attr.sched.exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  });
  ASSERT_TRUE(ScheduleUtils::IsNormStruct(test_graph.graph));

  SameSourceBroadcastCsePass pass;
  ASSERT_EQ(pass.RunPass(test_graph.graph), af::SUCCESS);
  ExpectSeparateBroadcastInputs(test_graph);
}

TEST(SameSourceBroadcastCsePassTest, KeepsBroadcastsWhenControlEdgeExists) {
  auto test_graph = BuildBroadcastCseGraph();
  ASSERT_TRUE(ScheduleUtils::IsNormStruct(test_graph.graph));
  af::GraphUtils::AddEdge(test_graph.broadcast1->GetOutControlAnchor(), test_graph.add->GetInControlAnchor());

  SameSourceBroadcastCsePass pass;
  ASSERT_EQ(pass.RunPass(test_graph.graph), af::SUCCESS);
  ExpectSeparateBroadcastInputs(test_graph);
  EXPECT_EQ(test_graph.broadcast1->GetOutControlNodesSize(), 1U);
  EXPECT_EQ(test_graph.add->GetInControlNodesSize(), 1U);
}

TEST(SameSourceBroadcastCsePassTest, KeepsBroadcastsWhenReduceDoesNotChangeShape) {
  auto test_graph = BuildBroadcastCseGraph([](const af::AscNodePtr &) {});
  test_graph.reduce->outputs[0].attr.repeats = test_graph.reduce->inputs[0].attr.repeats;
  ASSERT_TRUE(ScheduleUtils::IsNormStruct(test_graph.graph));

  SameSourceBroadcastCsePass pass;
  ASSERT_EQ(pass.RunPass(test_graph.graph), af::SUCCESS);
  ExpectSeparateBroadcastInputs(test_graph);
}
}  // namespace optimize
