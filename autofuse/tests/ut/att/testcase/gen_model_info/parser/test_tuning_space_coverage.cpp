/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "expr_gen/arg_list_reorder.h"
#include "parser/tuning_space.h"

namespace att {
namespace {
TEST(TuningSpaceCoverageTest, AxisDiagnosticIncludesRelationsAndOptionalExpression) {
  SubAxis parent;
  parent.name = "parent";
  SubAxis axis;
  axis.name = "tile";
  axis.axis_type = AxisPosition::INNER;
  axis.orig_axis_name = {"original"};
  axis.parent_axis = {&parent};
  axis.repeat = CreateExpr(16);
  axis.is_bind_multi_core = true;
  axis.enable_pad = true;
  const auto description = axis.ToString();
  EXPECT_NE(description.find("name: tile"), std::string::npos);
  EXPECT_NE(description.find("axis_type: INNER"), std::string::npos);
  EXPECT_NE(description.find("repeat: 16"), std::string::npos);
  EXPECT_NE(description.find("orig_axis_name: original,"), std::string::npos);
  EXPECT_NE(description.find("parent_axis_name: parent,"), std::string::npos);
  axis.repeat = Expr();
  EXPECT_NE(axis.ToString().find("repeat: ,"), std::string::npos);
}

TEST(TuningSpaceCoverageTest, TensorDiagnosticIncludesOriginalAndGlobalLayout) {
  SubAxis axis;
  axis.name = "tile";
  Tensor tensor;
  tensor.name = "input";
  tensor.data_type_size = 2;
  tensor.resource_id = 7;
  tensor.dim_info = {&axis};
  tensor.repeat = {CreateExpr(8), Expr()};
  tensor.ori_repeat = {CreateExpr(32), Expr()};
  tensor.stride = {CreateExpr(1), Expr()};
  tensor.ori_stride = {CreateExpr(4), Expr()};
  tensor.gm_stride = {CreateExpr(16), Expr()};
  const auto description = tensor.ToString();
  EXPECT_NE(description.find("name: input, datasize: 2, resource_id: 7"), std::string::npos);
  EXPECT_NE(description.find("axis {tile, }"), std::string::npos);
  EXPECT_NE(description.find("repeat: {8, , }"), std::string::npos);
  EXPECT_NE(description.find("ori_repeat: {32, , }"), std::string::npos);
  EXPECT_NE(description.find("stride: {1, , }"), std::string::npos);
  EXPECT_NE(description.find("ori_stride: {4, , }"), std::string::npos);
  EXPECT_NE(description.find("gm_stride: {16, , }"), std::string::npos);
}

TEST(TuningSpaceCoverageTest, NodeDiagnosticReportsDataDependenciesAndExecutionCondition) {
  NodeInfo node;
  node.name = "add";
  node.node_type = "Add";
  node.node_unit = "UnitVector";
  node.inputs = {std::make_shared<Tensor>()};
  node.outputs = {std::make_shared<Tensor>()};
  node.from_data = {"input0", "input1"};
  node.sub_nodes_infos.emplace_back();
  const auto description = node.DebugString();
  EXPECT_NE(description.find("NodeInfo {add, Add, UnitVector"), std::string::npos);
  EXPECT_NE(description.find("input size=1, output size=1"), std::string::npos);
  EXPECT_NE(description.find("from_data=input0, input1, "), std::string::npos);
  EXPECT_NE(description.find("sub_nodes_infos size=1"), std::string::npos);
  EXPECT_NE(description.find("exec_condition = " + std::to_string(static_cast<int32_t>(node.exec_condition))),
            std::string::npos);
}

TEST(TuningSpaceCoverageTest, GlobalCacheExposesConfiguredBufferMultiplicity) {
  GlobalCache cache("global");
  EXPECT_EQ(cache.GetBufferNum(), 1);
  cache.buffer_num = 3;
  const Container &container = cache;
  EXPECT_EQ(container.GetBufferNum(), 3);
  auto tensor = std::make_shared<Tensor>();
  cache.coexist_tensors = {{tensor}};
  EXPECT_EQ(cache.GetCoTensors(), (std::vector<std::vector<TensorPtr>>{{tensor}}));
}

TEST(ArgPriorityGraphCoverageTest, DfsAndBfsRespectDependenciesAndRejectInvalidVertices) {
  ArgPriorityGraph graph(4);
  EXPECT_FALSE(graph.AddEdge(0, 1));
  EXPECT_FALSE(graph.AddEdge(1, 5));
  ASSERT_TRUE(graph.AddEdge(1, 2));
  ASSERT_TRUE(graph.AddEdge(1, 3));
  ASSERT_TRUE(graph.AddEdge(2, 4));
  ASSERT_TRUE(graph.AddEdge(3, 4));
  ASSERT_TRUE(graph.AddEdge(1, 2));
  EXPECT_TRUE(graph.WillCreateCycle(4, 1));
  ASSERT_TRUE(graph.AddEdge(4, 1));  // Conflicting precedence is ignored, preserving the DAG.
  EXPECT_EQ(graph.TopologicalSort(), (std::vector<size_t>{1, 2, 3, 4}));
  EXPECT_EQ(graph.TopologicalSort(false), (std::vector<size_t>{1, 3, 2, 4}));
  EXPECT_EQ(graph.TopologicalSort(false), (std::vector<size_t>{1, 3, 2, 4}));
}

TEST(ArgPriorityGraphCoverageTest, EmptyGraphHasNoOrderingAndSelfCycleIsDetected) {
  ArgPriorityGraph empty(0);
  EXPECT_TRUE(empty.TopologicalSort().empty());
  EXPECT_TRUE(empty.TopologicalSort(false).empty());
  ArgPriorityGraph graph(1);
  ASSERT_TRUE(graph.AddEdge(1, 1));
  EXPECT_TRUE(graph.TopologicalSort().empty());
  EXPECT_TRUE(graph.TopologicalSort(false).empty());
}
}  // namespace
}  // namespace att
