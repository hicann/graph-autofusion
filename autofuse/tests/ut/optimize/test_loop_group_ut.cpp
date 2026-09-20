/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "schedule_utils.h"

namespace optimize {
namespace {
// 水平融合双reduce（真实业务场景）：load0同输入分叉两路，两路reduce规约轴相同、各自输出。
// loop_axis赋值模拟真实调度分布：reduce及其前置节点同轴(B)，reduce后继节点异轴(A)。
// data -> load -> {abs0(B) -> max1(B,reduce) -> relu0(A) -> store0 -> output0,
//                  abs1(B) -> max2(B,reduce) -> relu1(A) -> store1 -> output1}
af::AscGraph CreateHorizontalFusionTwoReduce(const char *name) {
  af::AscGraph graph(name);
  af::ascir_op::Data data("data", graph);
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load");
  load.x = data.y;
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Abs abs0("abs0");
  abs0.x = load.y;
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Abs abs1("abs1");
  abs1.x = load.y;
  abs1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs1.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Max max1("max1");
  max1.x = abs0.y;
  max1.attr.api.compute_type = af::ComputeType::kComputeReduce;
  max1.attr.api.type = af::ApiType::kAPITypeCompute;
  max1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Max max2("max2");
  max2.x = abs1.y;
  max2.attr.api.compute_type = af::ComputeType::kComputeReduce;
  max2.attr.api.type = af::ApiType::kAPITypeCompute;
  max2.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Relu relu0("relu0");
  relu0.x = max1.y;
  relu0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  relu0.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Relu relu1("relu1");
  relu1.x = max2.y;
  relu1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  relu1.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Store store0("store0");
  store0.x = relu0.y;
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.attr.api.type = af::ApiType::kAPITypeCompute;
  store0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output0("output0");
  output0.x = store0.y;
  output0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output0.attr.api.type = af::ApiType::kAPITypeBuffer;
  output0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store1("store1");
  store1.x = relu1.y;
  store1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store1.attr.api.type = af::ApiType::kAPITypeCompute;
  store1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output1("output1");
  output1.x = store1.y;
  output1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output1.attr.api.type = af::ApiType::kAPITypeBuffer;
  output1.y.dtype = ge::DT_FLOAT;
  return graph;
}

// 同轴区域被异轴分支隔断：max1(A)的输出分叉，relu0支与max1同轴连通，
// exp0(B)支下游回到A轴形成第二个reduce区域（第2轮发现，与组0同轴但不连通）：
// data -> load -> abs0(A) -> max1(A,reduce) -> {relu0(A) -> store0 -> output0,
//                                               exp0(B) -> abs2(A) -> max2(A,reduce) -> store1 -> output1}
af::AscGraph CreateSameAxisDisconnectedBranch(const char *name) {
  af::AscGraph graph(name);
  af::ascir_op::Data data("data", graph);
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load");
  load.x = data.y;
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Abs abs0("abs0");
  abs0.x = load.y;
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Max max1("max1");
  max1.x = abs0.y;
  max1.attr.api.compute_type = af::ComputeType::kComputeReduce;
  max1.attr.api.type = af::ApiType::kAPITypeCompute;
  max1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Relu relu0("relu0");
  relu0.x = max1.y;
  relu0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  relu0.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Store store0("store0");
  store0.x = relu0.y;
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.attr.api.type = af::ApiType::kAPITypeCompute;
  store0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output0("output0");
  output0.x = store0.y;
  output0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output0.attr.api.type = af::ApiType::kAPITypeBuffer;
  output0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Exp exp0("exp0");
  exp0.x = max1.y;
  exp0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  exp0.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Abs abs2("abs2");
  abs2.x = exp0.y;
  abs2.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs2.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Max max2("max2");
  max2.x = abs2.y;
  max2.attr.api.compute_type = af::ComputeType::kComputeReduce;
  max2.attr.api.type = af::ApiType::kAPITypeCompute;
  max2.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store1("store1");
  store1.x = max2.y;
  store1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store1.attr.api.type = af::ApiType::kAPITypeCompute;
  store1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output1("output1");
  output1.x = store1.y;
  output1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output1.attr.api.type = af::ApiType::kAPITypeBuffer;
  output1.y.dtype = ge::DT_FLOAT;
  return graph;
}

// 独立双reduce链：两路各自独立读输入，无多引用结构，IsNeedFixTopo不触发：
// {data0 -> load0 -> abs0(B) -> max1(B,reduce) -> relu0(A) -> store0 -> output0,
//  data1 -> load1 -> abs1(B) -> max2(B,reduce) -> relu1(A) -> store1 -> output1}
af::AscGraph CreateIndependentTwoReduceChains(const char *name) {
  af::AscGraph graph(name);
  af::ascir_op::Data data0("data0", graph);
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Max max1("max1");
  max1.x = abs0.y;
  max1.attr.api.compute_type = af::ComputeType::kComputeReduce;
  max1.attr.api.type = af::ApiType::kAPITypeCompute;
  max1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Relu relu0("relu0");
  relu0.x = max1.y;
  relu0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  relu0.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Store store0("store0");
  store0.x = relu0.y;
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.attr.api.type = af::ApiType::kAPITypeCompute;
  store0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output0("output0");
  output0.x = store0.y;
  output0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output0.attr.api.type = af::ApiType::kAPITypeBuffer;
  output0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Data data1("data1", graph);
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Abs abs1("abs1");
  abs1.x = load1.y;
  abs1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs1.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Max max2("max2");
  max2.x = abs1.y;
  max2.attr.api.compute_type = af::ComputeType::kComputeReduce;
  max2.attr.api.type = af::ApiType::kAPITypeCompute;
  max2.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Relu relu1("relu1");
  relu1.x = max2.y;
  relu1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  relu1.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Store store1("store1");
  store1.x = relu1.y;
  store1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store1.attr.api.type = af::ApiType::kAPITypeCompute;
  store1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output1("output1");
  output1.x = store1.y;
  output1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output1.attr.api.type = af::ApiType::kAPITypeBuffer;
  output1.y.dtype = ge::DT_FLOAT;
  return graph;
}

// 同轮同轴双种子：max1(A)的一轮扩散同时发现exp0(B)/abs1(C)/exp1(B)三个不连通分支种子：
// data -> load -> abs0(A) -> max1(A,reduce) -> {exp0(B) -> store0 -> output0,
//                                               abs1(C) -> store1 -> output1,
//                                               exp1(B) -> store2 -> output2}
af::AscGraph CreateSameRoundSameAxisSeeds(const char *name) {
  af::AscGraph graph(name);
  af::ascir_op::Data data("data", graph);
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load");
  load.x = data.y;
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Abs abs0("abs0");
  abs0.x = load.y;
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Max max1("max1");
  max1.x = abs0.y;
  max1.attr.api.compute_type = af::ComputeType::kComputeReduce;
  max1.attr.api.type = af::ApiType::kAPITypeCompute;
  max1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Exp exp0("exp0");
  exp0.x = max1.y;
  exp0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  exp0.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Abs abs1("abs1");
  abs1.x = max1.y;
  abs1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs1.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Exp exp1("exp1");
  exp1.x = max1.y;
  exp1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  exp1.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Store store0("store0");
  store0.x = exp0.y;
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.attr.api.type = af::ApiType::kAPITypeCompute;
  store0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output0("output0");
  output0.x = store0.y;
  output0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output0.attr.api.type = af::ApiType::kAPITypeBuffer;
  output0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store1("store1");
  store1.x = abs1.y;
  store1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store1.attr.api.type = af::ApiType::kAPITypeCompute;
  store1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output1("output1");
  output1.x = store1.y;
  output1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output1.attr.api.type = af::ApiType::kAPITypeBuffer;
  output1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store2("store2");
  store2.x = exp1.y;
  store2.attr.api.compute_type = af::ComputeType::kComputeStore;
  store2.attr.api.type = af::ApiType::kAPITypeCompute;
  store2.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output2("output2");
  output2.x = store2.y;
  output2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output2.attr.api.type = af::ApiType::kAPITypeBuffer;
  output2.y.dtype = ge::DT_FLOAT;
  return graph;
}

struct LoopAxisIds {
  int64_t axis_a;
  int64_t axis_b;
};

LoopAxisIds CreateLoopAxisIds(af::AscGraph &graph) {
  return {graph.CreateAxis("z_a", af::Symbol(4)).id, graph.CreateAxis("z_b", af::Symbol(8)).id};
}

void SetNodeLoopAxis(af::AscGraph &graph, const char *node_name, const int64_t loop_axis) {
  auto node = graph.FindNode(node_name);
  ASSERT_NE(node, nullptr);
  node->attr.sched.loop_axis = loop_axis;
}

int64_t GetNodeId(af::AscGraph &graph, const char *node_name) {
  auto node = graph.FindNode(node_name);
  EXPECT_NE(node, nullptr);
  return (node == nullptr) ? -1 : node->GetOpDescBarePtr()->GetId();
}

af::Node *GetNodePtr(af::AscGraph &graph, const char *node_name) {
  auto node = graph.FindNode(node_name);
  EXPECT_NE(node, nullptr);
  return (node == nullptr) ? nullptr : node.get();
}

// 为水平融合图赋loop_axis：reduce及其前置节点为axis_b，reduce后继节点为axis_a
void SetHorizontalFusionLoopAxis(af::AscGraph &graph, const LoopAxisIds &axes) {
  SetNodeLoopAxis(graph, "load", axes.axis_b);
  SetNodeLoopAxis(graph, "abs0", axes.axis_b);
  SetNodeLoopAxis(graph, "abs1", axes.axis_b);
  SetNodeLoopAxis(graph, "max1", axes.axis_b);
  SetNodeLoopAxis(graph, "max2", axes.axis_b);
  SetNodeLoopAxis(graph, "relu0", axes.axis_a);
  SetNodeLoopAxis(graph, "relu1", axes.axis_a);
}

TEST(LoopGroupTest, IsNeedLoopGroupingNotNeedSingleReduce) {
  af::AscGraph graph("single_reduce");
  af::ascir_op::Data data("data", graph);
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load");
  load.x = data.y;
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Abs abs0("abs0");
  abs0.x = load.y;
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Max max1("max1");
  max1.x = abs0.y;
  max1.attr.api.compute_type = af::ComputeType::kComputeReduce;
  max1.attr.api.type = af::ApiType::kAPITypeCompute;
  max1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store("store");
  store.x = max1.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.api.type = af::ApiType::kAPITypeCompute;
  store.y.dtype = ge::DT_FLOAT;

  const auto axes = CreateLoopAxisIds(graph);
  SetNodeLoopAxis(graph, "max1", axes.axis_a);
  EXPECT_FALSE(ScheduleUtils::IsNeedLoopGrouping(graph));
}

TEST(LoopGroupTest, IsNeedLoopGroupingNotNeedUnassignedLoopAxis) {
  auto graph = CreateHorizontalFusionTwoReduce("two_reduce_one_unassigned");
  const auto axes = CreateLoopAxisIds(graph);
  // max2保持-1（未经过AutoScheduler），存在未赋值reduce节点，不应进入分组
  SetNodeLoopAxis(graph, "load", axes.axis_b);
  SetNodeLoopAxis(graph, "abs0", axes.axis_b);
  SetNodeLoopAxis(graph, "abs1", axes.axis_b);
  SetNodeLoopAxis(graph, "max1", axes.axis_b);
  SetNodeLoopAxis(graph, "relu0", axes.axis_a);
  SetNodeLoopAxis(graph, "relu1", axes.axis_a);
  EXPECT_FALSE(ScheduleUtils::IsNeedLoopGrouping(graph));
}

TEST(LoopGroupTest, IsNeedLoopGroupingNeedMultipleUniqueAxis) {
  auto graph = CreateHorizontalFusionTwoReduce("two_reduce_valid");
  const auto axes = CreateLoopAxisIds(graph);
  // 双reduce均已赋值，且全图有效loop_axis种类数>1（reduce链B轴+后继A轴）
  SetHorizontalFusionLoopAxis(graph, axes);
  EXPECT_TRUE(ScheduleUtils::IsNeedLoopGrouping(graph));
}

TEST(LoopGroupTest, IsNeedLoopGroupingNotNeedSingleUniqueAxis) {
  auto graph = CreateHorizontalFusionTwoReduce("two_reduce_single_axis");
  const auto axes = CreateLoopAxisIds(graph);
  // 全图仅一种有效loop_axis，无需分组
  SetHorizontalFusionLoopAxis(graph, axes);
  SetNodeLoopAxis(graph, "relu0", axes.axis_b);
  SetNodeLoopAxis(graph, "relu1", axes.axis_b);
  EXPECT_FALSE(ScheduleUtils::IsNeedLoopGrouping(graph));
}

TEST(LoopGroupTest, BuildLoopGroupsHorizontalFusionReduceSameGroup) {
  auto graph = CreateHorizontalFusionTwoReduce("horizontal_fusion");
  const auto axes = CreateLoopAxisIds(graph);
  SetHorizontalFusionLoopAxis(graph, axes);

  std::vector<LoopGroup> loop_groups;
  std::unordered_map<af::Node *, size_t> node_to_group;
  ASSERT_EQ(ScheduleUtils::BuildLoopGroups(graph, loop_groups, node_to_group), af::SUCCESS);

  // 组0（B轴）：两路reduce经共同输入load连通吸收进同一分组，data/load等-1节点随组扩散并入
  ASSERT_EQ(loop_groups.size(), 2UL);
  EXPECT_EQ(loop_groups[0].loop_axis, axes.axis_b);
  EXPECT_EQ(loop_groups[0].nodes.size(), 6UL);
  EXPECT_EQ(loop_groups[0].nodes[0], GetNodePtr(graph, "max1"));  // 首个reduce为种子
  const size_t reduce_group = node_to_group[GetNodePtr(graph, "max1")];
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "max2")], reduce_group);  // 水平融合：双reduce同组
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "abs0")], reduce_group);
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "abs1")], reduce_group);
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "load")], reduce_group);
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "data")], reduce_group);

  // 组1（A轴）：两路reduce后继同轮同轴种子合并为一组，不被拆成两个循环
  EXPECT_EQ(loop_groups[1].loop_axis, axes.axis_a);
  EXPECT_EQ(loop_groups[1].nodes.size(), 6UL);
  const size_t post_group = node_to_group[GetNodePtr(graph, "relu0")];
  EXPECT_NE(post_group, reduce_group);
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "relu1")], post_group);
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "store0")], post_group);
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "store1")], post_group);
}

TEST(LoopGroupTest, BuildLoopGroupsSameAxisDisconnectedSplitGroups) {
  auto graph = CreateSameAxisDisconnectedBranch("same_axis_disconnected");
  const auto axes = CreateLoopAxisIds(graph);
  SetNodeLoopAxis(graph, "abs0", axes.axis_a);
  SetNodeLoopAxis(graph, "max1", axes.axis_a);
  SetNodeLoopAxis(graph, "relu0", axes.axis_a);
  SetNodeLoopAxis(graph, "exp0", axes.axis_b);
  SetNodeLoopAxis(graph, "abs2", axes.axis_a);
  SetNodeLoopAxis(graph, "max2", axes.axis_a);

  std::vector<LoopGroup> loop_groups;
  std::unordered_map<af::Node *, size_t> node_to_group;
  ASSERT_EQ(ScheduleUtils::BuildLoopGroups(graph, loop_groups, node_to_group), af::SUCCESS);

  // 同轴但不连通的区域拆分为不同编号：组0和组2同为axis_a，组1为axis_b
  // （abs2/max2在exp0分支下游第2轮发现，与组0同轴但不连通，不并入组0）
  ASSERT_EQ(loop_groups.size(), 3UL);
  EXPECT_EQ(loop_groups[0].loop_axis, axes.axis_a);
  EXPECT_EQ(loop_groups[1].loop_axis, axes.axis_b);
  EXPECT_EQ(loop_groups[2].loop_axis, axes.axis_a);
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "max1")], 0UL);
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "relu0")], 0UL);
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "exp0")], 1UL);
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "abs2")], 2UL);
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "max2")], 2UL);
}

TEST(LoopGroupTest, BuildLoopGroupsSameRoundSameAxisSeedsMerged) {
  auto graph = CreateSameRoundSameAxisSeeds("same_round_same_axis_seeds");
  const auto axes = CreateLoopAxisIds(graph);
  const int64_t axis_c = graph.CreateAxis("z_c", af::Symbol(2)).id;
  SetNodeLoopAxis(graph, "abs0", axes.axis_a);
  SetNodeLoopAxis(graph, "max1", axes.axis_a);
  SetNodeLoopAxis(graph, "exp0", axes.axis_b);
  SetNodeLoopAxis(graph, "abs1", axis_c);
  SetNodeLoopAxis(graph, "exp1", axes.axis_b);

  std::vector<LoopGroup> loop_groups;
  std::unordered_map<af::Node *, size_t> node_to_group;
  ASSERT_EQ(ScheduleUtils::BuildLoopGroups(graph, loop_groups, node_to_group), af::SUCCESS);

  // 组0的一轮扩散发现exp0/abs1/exp1三个种子：同轮同轴的exp0与exp1归入同一分组，
  // 不因abs1(C)分组创建在中间而被拆成两个B分组
  ASSERT_EQ(loop_groups.size(), 3UL);
  const size_t b_group = node_to_group[GetNodePtr(graph, "exp0")];
  EXPECT_EQ(loop_groups[b_group].loop_axis, axes.axis_b);
  EXPECT_EQ(loop_groups[b_group].nodes.size(), 6UL);
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "exp1")], b_group);
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "store0")], b_group);
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "store2")], b_group);
  EXPECT_NE(node_to_group[GetNodePtr(graph, "abs1")], b_group);
  EXPECT_EQ(node_to_group[GetNodePtr(graph, "max1")], 0UL);
}

TEST(LoopGroupTest, TopologicalSortingLoopGroupRuleApplied) {
  auto graph = CreateHorizontalFusionTwoReduce("loop_group_sort");
  const auto axes = CreateLoopAxisIds(graph);
  SetHorizontalFusionLoopAxis(graph, axes);
  // 前置条件：load多引用触发IsNeedFixTopo，双reduce有效且多轴触发IsNeedLoopGrouping
  ASSERT_TRUE(ScheduleUtils::IsNeedLoopGrouping(graph));

  ASSERT_EQ(ScheduleUtils::TopologicalSorting(graph, true), af::SUCCESS);

  // 组内保持依赖topo序
  EXPECT_TRUE(GetNodeId(graph, "load") < GetNodeId(graph, "abs0"));
  EXPECT_TRUE(GetNodeId(graph, "abs0") < GetNodeId(graph, "max1"));
  EXPECT_TRUE(GetNodeId(graph, "relu0") < GetNodeId(graph, "store0"));
  // 两路reduce同组且整体先于reduce后继组：同级for循环不互相穿插
  EXPECT_TRUE(GetNodeId(graph, "max1") < GetNodeId(graph, "relu0"));
  EXPECT_TRUE(GetNodeId(graph, "max2") < GetNodeId(graph, "relu0"));
  EXPECT_TRUE(GetNodeId(graph, "max2") < GetNodeId(graph, "relu1"));
}

TEST(LoopGroupTest, TopologicalSortingLoopGroupRuleSkippedWithoutMulConsumer) {
  auto graph = CreateIndependentTwoReduceChains("loop_group_sort_skipped");
  const auto axes = CreateLoopAxisIds(graph);
  SetNodeLoopAxis(graph, "load0", axes.axis_b);
  SetNodeLoopAxis(graph, "abs0", axes.axis_b);
  SetNodeLoopAxis(graph, "max1", axes.axis_b);
  SetNodeLoopAxis(graph, "relu0", axes.axis_a);
  SetNodeLoopAxis(graph, "load1", axes.axis_b);
  SetNodeLoopAxis(graph, "abs1", axes.axis_b);
  SetNodeLoopAxis(graph, "max2", axes.axis_b);
  SetNodeLoopAxis(graph, "relu1", axes.axis_a);
  ASSERT_TRUE(ScheduleUtils::IsNeedLoopGrouping(graph));

  // 两路独立无多引用：IsNeedFixTopo不触发，分组排序分支不生效，保持依赖拓扑序
  ASSERT_EQ(ScheduleUtils::TopologicalSorting(graph, true), af::SUCCESS);
  EXPECT_TRUE(GetNodeId(graph, "load0") < GetNodeId(graph, "abs0"));
  EXPECT_TRUE(GetNodeId(graph, "abs0") < GetNodeId(graph, "max1"));
  EXPECT_TRUE(GetNodeId(graph, "max1") < GetNodeId(graph, "relu0"));
  EXPECT_TRUE(GetNodeId(graph, "load1") < GetNodeId(graph, "abs1"));
  EXPECT_TRUE(GetNodeId(graph, "abs1") < GetNodeId(graph, "max2"));
  EXPECT_TRUE(GetNodeId(graph, "relu1") < GetNodeId(graph, "store1"));
}
}  // namespace
}  // namespace optimize
