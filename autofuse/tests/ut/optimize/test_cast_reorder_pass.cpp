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

#include "asc_graph_builder.h"
#include "graph_pass/cast_reorder_pass.h"

using namespace ge;
using namespace af::testing;

namespace af {
namespace {
void ExpectSingleDataEdge(AscGraph &graph, const std::string &src_name, const std::string &dst_name) {
  const auto src_node = graph.FindNode(src_name.c_str());
  ASSERT_NE(src_node, nullptr);
  const auto out_anchor = src_node->GetOutDataAnchor(0);
  ASSERT_NE(out_anchor, nullptr);
  const auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  ASSERT_EQ(peer_in_anchors.size(), 1U);
  ASSERT_NE(peer_in_anchors.at(0), nullptr);
  ASSERT_NE(peer_in_anchors.at(0)->GetOwnerNode(), nullptr);
  EXPECT_EQ(peer_in_anchors.at(0)->GetOwnerNode()->GetName(), dst_name);
}

DataType GetOutputDtype(AscGraph &graph, const std::string &node_name) {
  const auto node = graph.FindNode(node_name.c_str());
  EXPECT_NE(node, nullptr);
  if (node == nullptr) {
    return DT_UNDEFINED;
  }
  return node->outputs[0].attr.dtype;
}
}  // namespace

class TestTransposeCastReorderPass : public ::testing::Test {};

TEST_F(TestTransposeCastReorderPass, LoadUpcastTransposeMovesCastAfterTranspose) {
  auto graph = AscGraphBuilder("load_upcast_transpose")
                   .Loops({Sym(2), Sym(3)})
                   .Data("data0", 0, DT_FLOAT16)
                   .Load("load0", "data0")
                   .Cast("cast0", "load0", DT_FLOAT)
                   .Transpose("transpose0", "cast0", {1, 0})
                   .Abs("abs0", "transpose0")
                   .Store("store0", "abs0")
                   .Output("output0", "store0", 0, DT_FLOAT)
                   .Build();
  const auto expected_transpose_axis = graph.FindNode("transpose0")->outputs[0].attr.axis;
  const auto expected_transpose_repeats = graph.FindNode("transpose0")->outputs[0].attr.repeats;
  const auto expected_transpose_strides = graph.FindNode("transpose0")->outputs[0].attr.strides;

  ASSERT_EQ(optimize::SwapCastWithPrecisionAgnosticOpsPass::Run(graph), SUCCESS);

  ExpectSingleDataEdge(graph, "load0", "transpose0");
  ExpectSingleDataEdge(graph, "transpose0", "cast0");
  ExpectSingleDataEdge(graph, "cast0", "abs0");
  EXPECT_EQ(GetOutputDtype(graph, "transpose0"), DT_FLOAT16);
  EXPECT_EQ(GetOutputDtype(graph, "cast0"), DT_FLOAT);
  const auto transpose = graph.FindNode("transpose0");
  const auto cast = graph.FindNode("cast0");
  ASSERT_NE(transpose, nullptr);
  ASSERT_NE(cast, nullptr);
  EXPECT_EQ(transpose->outputs[0].attr.axis, expected_transpose_axis);
  EXPECT_EQ(transpose->outputs[0].attr.repeats, expected_transpose_repeats);
  EXPECT_EQ(transpose->outputs[0].attr.strides, expected_transpose_strides);
  EXPECT_EQ(cast->outputs[0].attr.axis, expected_transpose_axis);
  EXPECT_EQ(cast->outputs[0].attr.repeats, expected_transpose_repeats);
  EXPECT_EQ(cast->outputs[0].attr.strides, expected_transpose_strides);
  EXPECT_EQ(transpose->GetOpDesc()->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetDataType(), DT_FLOAT16);
}

TEST_F(TestTransposeCastReorderPass, TransposeDowncastStoreMovesCastBeforeTranspose) {
  auto graph = AscGraphBuilder("transpose_downcast_store")
                   .Loops({Sym(2), Sym(3)})
                   .Data("data0", 0, DT_FLOAT)
                   .Load("load0", "data0")
                   .Transpose("transpose0", "load0", {1, 0})
                   .Cast("cast0", "transpose0", DT_FLOAT16)
                   .Store("store0", "cast0")
                   .Output("output0", "store0", 0, DT_FLOAT16)
                   .Build();
  const auto expected_cast_axis = graph.FindNode("load0")->outputs[0].attr.axis;
  const auto expected_cast_repeats = graph.FindNode("load0")->outputs[0].attr.repeats;
  const auto expected_cast_strides = graph.FindNode("load0")->outputs[0].attr.strides;
  const auto expected_transpose_axis = graph.FindNode("transpose0")->outputs[0].attr.axis;
  const auto expected_transpose_repeats = graph.FindNode("transpose0")->outputs[0].attr.repeats;
  const auto expected_transpose_strides = graph.FindNode("transpose0")->outputs[0].attr.strides;

  ASSERT_EQ(optimize::SwapCastWithPrecisionAgnosticOpsPass::Run(graph), SUCCESS);

  ExpectSingleDataEdge(graph, "load0", "cast0");
  ExpectSingleDataEdge(graph, "cast0", "transpose0");
  ExpectSingleDataEdge(graph, "transpose0", "store0");
  EXPECT_EQ(GetOutputDtype(graph, "cast0"), DT_FLOAT16);
  EXPECT_EQ(GetOutputDtype(graph, "transpose0"), DT_FLOAT16);
  const auto load = graph.FindNode("load0");
  const auto cast = graph.FindNode("cast0");
  const auto transpose = graph.FindNode("transpose0");
  ASSERT_NE(load, nullptr);
  ASSERT_NE(cast, nullptr);
  ASSERT_NE(transpose, nullptr);
  EXPECT_EQ(cast->outputs[0].attr.axis, expected_cast_axis);
  EXPECT_EQ(cast->outputs[0].attr.repeats, expected_cast_repeats);
  EXPECT_EQ(cast->outputs[0].attr.strides, expected_cast_strides);
  EXPECT_EQ(transpose->outputs[0].attr.axis, expected_transpose_axis);
  EXPECT_EQ(transpose->outputs[0].attr.repeats, expected_transpose_repeats);
  EXPECT_EQ(transpose->outputs[0].attr.strides, expected_transpose_strides);
  EXPECT_EQ(cast->GetOpDesc()->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(transpose->GetOpDesc()->GetInputDesc(0).GetDataType(), DT_FLOAT16);
}

TEST_F(TestTransposeCastReorderPass, CastWithMultipleOutputReferencesDoesNotReorder) {
  auto graph = AscGraphBuilder("shared_cast")
                   .Loops({Sym(2), Sym(3)})
                   .Data("data0", 0, DT_FLOAT16)
                   .Load("load0", "data0")
                   .Cast("cast0", "load0", DT_FLOAT)
                   .Transpose("transpose0", "cast0", {1, 0})
                   .Abs("abs0", "cast0")
                   .Store("store0", "transpose0")
                   .Store("store1", "abs0")
                   .Output("output0", "store0", 0, DT_FLOAT)
                   .Output("output1", "store1", 1, DT_FLOAT)
                   .Build();

  ASSERT_EQ(optimize::SwapCastWithPrecisionAgnosticOpsPass::Run(graph), SUCCESS);

  const auto cast = graph.FindNode("cast0");
  ASSERT_NE(cast, nullptr);
  EXPECT_EQ(cast->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 2U);
  EXPECT_EQ(graph.FindNode("transpose0")->GetInDataNodes().at(0)->GetName(), "cast0");
  EXPECT_EQ(GetOutputDtype(graph, "transpose0"), DT_FLOAT);
}

TEST_F(TestTransposeCastReorderPass, TransposeWithMultipleOutputReferencesDoesNotReorder) {
  auto graph = AscGraphBuilder("shared_transpose")
                   .Loops({Sym(2), Sym(3)})
                   .Data("data0", 0, DT_FLOAT)
                   .Load("load0", "data0")
                   .Transpose("transpose0", "load0", {1, 0})
                   .Cast("cast0", "transpose0", DT_FLOAT16)
                   .Abs("abs0", "transpose0")
                   .Store("store0", "cast0")
                   .Store("store1", "abs0")
                   .Output("output0", "store0", 0, DT_FLOAT16)
                   .Output("output1", "store1", 1, DT_FLOAT)
                   .Build();

  ASSERT_EQ(optimize::SwapCastWithPrecisionAgnosticOpsPass::Run(graph), SUCCESS);

  const auto transpose = graph.FindNode("transpose0");
  ASSERT_NE(transpose, nullptr);
  EXPECT_EQ(transpose->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 2U);
  EXPECT_EQ(graph.FindNode("cast0")->GetInDataNodes().at(0)->GetName(), "transpose0");
  EXPECT_EQ(GetOutputDtype(graph, "transpose0"), DT_FLOAT);
}
}  // namespace af
