/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

#include <gtest/gtest.h>

#include "ascir_ops.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "schedule_utils.h"

namespace optimize {
namespace {

TEST(GraphSchedTest, ApplyGraphSchedAxisToNodes) {
  af::AscGraph graph("graph_sched");
  auto &axis0 = graph.CreateAxis("z0", af::Symbol(4));
  auto &axis1 = graph.CreateAxis("z1", af::Symbol(8));
  af::ascir_op::Data data("data", graph);
  af::ascir_op::Data buffer("buffer", graph);
  const auto node = graph.FindNode("data");
  const auto buffer_node = graph.FindNode("buffer");
  ASSERT_NE(node, nullptr);
  ASSERT_NE(buffer_node, nullptr);
  node->attr.sched.axis = {axis1.id};
  node->attr.sched.loop_axis = axis0.id;
  node->attr.sched.exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  buffer_node->attr.api.type = af::ApiType::kAPITypeBuffer;
  buffer_node->attr.sched.axis = {axis1.id};

  const auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  ASSERT_NE(compute_graph, nullptr);
  const auto graph_attr = compute_graph->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  ASSERT_NE(graph_attr, nullptr);
  graph_attr->sched.axis = {axis0.id, axis1.id};
  graph_attr->sched.loop_axis = axis1.id;

  ASSERT_EQ(ScheduleUtils::ApplyGraphSchedAxisToNodes(graph), af::SUCCESS);
  EXPECT_EQ(node->attr.sched.axis, (std::vector<int64_t>{axis0.id, axis1.id}));
  EXPECT_EQ(node->attr.sched.loop_axis, axis0.id);
  EXPECT_EQ(node->attr.sched.exec_condition, af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  EXPECT_EQ(buffer_node->attr.sched.axis, (std::vector<int64_t>{axis1.id}));
  EXPECT_EQ(graph_attr->sched.loop_axis, af::kIdNone);
}

TEST(GraphSchedTest, ApplyGraphSchedLoopAxisIsResetWithoutGraphAxis) {
  af::AscGraph graph("graph_sched_loop_axis");
  const auto loop_axis = graph.CreateAxis("z0", af::Symbol(4)).id;

  const auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  ASSERT_NE(compute_graph, nullptr);
  const auto graph_attr = compute_graph->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  ASSERT_NE(graph_attr, nullptr);
  graph_attr->sched.loop_axis = loop_axis;

  ASSERT_EQ(ScheduleUtils::ApplyGraphSchedAxisToNodes(graph), af::SUCCESS);
  EXPECT_EQ(graph_attr->sched.loop_axis, af::kIdNone);
}

TEST(GraphSchedTest, NormalizeGraphSchedAxisIds) {
  af::AscGraph graph("graph_sched_normalize");
  auto &axis0 = graph.CreateAxis("z0", af::Symbol(4));
  auto &axis1 = graph.CreateAxis("z1", af::Symbol(8));
  af::ascir_op::Data data("data", graph);
  const auto node = graph.FindNode("data");
  ASSERT_NE(node, nullptr);

  axis0.id = 5;
  axis1.id = 9;
  node->attr.sched.axis = {axis0.id, axis1.id};
  const auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  ASSERT_NE(compute_graph, nullptr);
  const auto graph_attr = compute_graph->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  ASSERT_NE(graph_attr, nullptr);
  graph_attr->sched.axis = {axis0.id, axis1.id};
  graph_attr->sched.loop_axis = axis1.id;

  ScheduleUtils::NormalizeAxisIds(graph);
  EXPECT_EQ(graph_attr->sched.axis, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(graph_attr->sched.loop_axis, 1);
  EXPECT_EQ(node->attr.sched.axis, (std::vector<int64_t>{0, 1}));
}

}  // namespace
}  // namespace optimize
