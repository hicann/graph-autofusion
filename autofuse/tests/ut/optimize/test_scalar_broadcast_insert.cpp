/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "asc_graph_builder.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "pre_process/scalar_broadcast_insert.h"
#include "autoschedule/autoschedule.h"
#include "ascgraph_info_complete.h"
#include "platform_context.h"
#include "runtime_stub.h"

using namespace af;
using namespace af::testing;
using namespace af::pre_process;
using namespace optimize::autoschedule;

namespace {

class TestScalarBroadcastInsert : public ::testing::Test {
 protected:
  void SetUp() override {
    auto stub_v1 = std::make_shared<ge::RuntimeStub>();
    ge::RuntimeStub::SetInstance(stub_v1);
  }

  void TearDown() override {}
};

size_t CountNodesByType(AscGraph &graph, const std::string &type) {
  size_t count = 0U;
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetType() == type) {
      ++count;
    }
  }
  return count;
}

bool HasNodeWithName(AscGraph &graph, const std::string &name) {
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetName() == name) {
      return true;
    }
  }
  return false;
}

bool IsConnected(AscGraph &graph, const std::string &src_name, const std::string &dst_name) {
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetName() == src_name) {
      auto out_anchor = node->GetOutDataAnchor(0);
      if (out_anchor == nullptr) return false;
      for (const auto &peer : out_anchor->GetPeerInDataAnchors()) {
        if (peer != nullptr && peer->GetOwnerNode()->GetName() == dst_name) {
          return true;
        }
      }
    }
  }
  return false;
}

size_t CountOutDataEdges(AscGraph &graph, const std::string &node_name) {
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetName() == node_name) {
      auto out_anchor = node->GetOutDataAnchor(0);
      if (out_anchor == nullptr) return 0U;
      return out_anchor->GetPeerInDataAnchors().size();
    }
  }
  return 0U;
}

}  // namespace

// ==================== Scalar 直连计算节点 → 插入 Broadcast ====================

TEST_F(TestScalarBroadcastInsert, ScalarDirectToCompute_InsertsBroadcast) {
  auto graph = AscGraphBuilder("test_scalar_to_add")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .Scalar("scalar0", "1.0")
                   .Add("add0", "load0", "scalar0")
                   .Store("store0", "add0")
                   .Output("output0", "store0")
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  size_t brc_before = CountNodesByType(graph, ascir_op::Broadcast::Type);
  ASSERT_EQ(brc_before, 0U);

  auto ret = InsertBroadcastAfterScalarForAscGraph(graph);
  ASSERT_EQ(ret, af::SUCCESS);

  size_t brc_after = CountNodesByType(graph, ascir_op::Broadcast::Type);
  EXPECT_GT(brc_after, 0U);
  // scalar0 和 add0 之间应有 broadcast
  EXPECT_FALSE(IsConnected(graph, "scalar0", "add0"));
}

// ==================== Scalar 已有 Broadcast → 不再插入 ====================

TEST_F(TestScalarBroadcastInsert, ScalarAlreadyHasBroadcast_NoInsert) {
  auto graph = AscGraphBuilder("test_scalar_with_brc")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .Scalar("scalar0", "1.0")
                   .Broadcast("brc0", "scalar0", {Sym("s0"), Sym("s1")})
                   .Add("add0", "load0", "brc0")
                   .Store("store0", "add0")
                   .Output("output0", "store0")
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  size_t brc_before = CountNodesByType(graph, ascir_op::Broadcast::Type);

  auto ret = InsertBroadcastAfterScalarForAscGraph(graph);
  ASSERT_EQ(ret, af::SUCCESS);

  size_t brc_after = CountNodesByType(graph, ascir_op::Broadcast::Type);
  EXPECT_EQ(brc_after, brc_before);
}

// ==================== Scalar 直连 Store → 插入 Broadcast ====================

TEST_F(TestScalarBroadcastInsert, ScalarDirectToStore_InsertsBroadcast) {
  auto graph = AscGraphBuilder("test_scalar_to_store")
                   .Loops({Sym("s0")})
                   .Scalar("scalar0", "2.0")
                   .Store("store0", "scalar0")
                   .Output("output0", "store0")
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto ret = InsertBroadcastAfterScalarForAscGraph(graph);
  ASSERT_EQ(ret, af::SUCCESS);

  size_t brc_after = CountNodesByType(graph, ascir_op::Broadcast::Type);
  EXPECT_GT(brc_after, 0U);
  EXPECT_FALSE(IsConnected(graph, "scalar0", "store0"));
}

// ==================== Scalar 无下游 → 不插入 ====================

TEST_F(TestScalarBroadcastInsert, ScalarNoDownstream_NoInsert) {
  auto graph = AscGraphBuilder("test_scalar_no_downstream")
                   .Loops({Sym("s0")})
                   .Scalar("scalar0", "1.0")
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .Store("store0", "load0")
                   .Output("output0", "store0")
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto ret = InsertBroadcastAfterScalarForAscGraph(graph);
  ASSERT_EQ(ret, af::SUCCESS);

  size_t brc_after = CountNodesByType(graph, ascir_op::Broadcast::Type);
  EXPECT_EQ(brc_after, 0U);
}

// ==================== Scalar 多下游计算节点 → 插入一个共享 Broadcast ====================

TEST_F(TestScalarBroadcastInsert, ScalarMultipleDownstreams_InsertsOneSharedBroadcast) {
  auto graph = AscGraphBuilder("test_scalar_multi_downstream")
                   .Loops({Sym("s0")})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0")
                   .Load("load1", "data1")
                   .Scalar("scalar0", "1.0")
                   .Add("add0", "load0", "scalar0")
                   .Mul("mul0", "load1", "scalar0")
                   .Store("store0", "add0")
                   .Store("store1", "mul0")
                   .Output("output0", "store0")
                   .Output("output1", "store1", 1)
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto ret = InsertBroadcastAfterScalarForAscGraph(graph);
  ASSERT_EQ(ret, af::SUCCESS);

  // 应只插入一个 broadcast（scalar0 的所有下游共享）
  size_t brc_after = CountNodesByType(graph, ascir_op::Broadcast::Type);
  EXPECT_EQ(brc_after, 1U);

  // broadcast 应同时连到 add0 和 mul0
  // 找到 broadcast 节点名
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetType() == ascir_op::Broadcast::Type) {
      auto out_anchor = node->GetOutDataAnchor(0);
      ASSERT_TRUE(out_anchor != nullptr);
      EXPECT_EQ(out_anchor->GetPeerInDataAnchors().size(), 2U);
      break;
    }
  }
}

TEST_F(TestScalarBroadcastInsert, ScalarMultipleDownstreamsWithDifferentView_InsertsSeparateBroadcasts) {
  auto graph = AscGraphBuilder("test_scalar_multi_downstream_different_view")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0")
                   .Load("load1", "data1")
                   .Scalar("scalar0", "1.0")
                   .Add("add0", "load0", "scalar0")
                   .Mul("mul0", "load1", "scalar0")
                   .Store("store0", "add0")
                   .Store("store1", "mul0")
                   .Output("output0", "store0")
                   .Output("output1", "store1", 1)
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  auto mul_node = graph.FindNode("mul0");
  ASSERT_NE(mul_node, nullptr);
  ASSERT_GE(mul_node->outputs[0].attr.repeats.size(), 1U);
  mul_node->outputs[0].attr.repeats[0] = af::sym::kSymbolOne;

  auto ret = InsertBroadcastAfterScalarForAscGraph(graph);
  ASSERT_EQ(ret, af::SUCCESS);

  EXPECT_EQ(CountNodesByType(graph, ascir_op::Broadcast::Type), 2U);
  EXPECT_FALSE(IsConnected(graph, "scalar0", "add0"));
  EXPECT_FALSE(IsConnected(graph, "scalar0", "mul0"));
  EXPECT_EQ(CountOutDataEdges(graph, "scalar0"), 2U);
}

TEST_F(TestScalarBroadcastInsert, ScalarMultipleDownstreamsWithDifferentShapes_GroupsCompatibleViews) {
  auto graph = AscGraphBuilder("test_scalar_multi_downstream_different_shapes")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .Scalar("scalar0", "2.0")
                   .Add("add0", "load0", "scalar0")
                   .Mul("mul0", "load0", "scalar0")
                   .Add("add1", "load0", "scalar0")
                   .Sub("sub0", "load0", "scalar0")
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  auto mul_node = std::dynamic_pointer_cast<AscNode>(graph.FindNode("mul0"));
  auto add1_node = std::dynamic_pointer_cast<AscNode>(graph.FindNode("add1"));
  ASSERT_NE(mul_node, nullptr);
  ASSERT_NE(add1_node, nullptr);
  ASSERT_FALSE(mul_node->outputs[0].attr.repeats.empty());
  ASSERT_FALSE(add1_node->outputs[0].attr.repeats.empty());
  mul_node->outputs[0].attr.repeats[0] = Sym(14);
  mul_node->outputs[0].attr.strides = {Sym(322)};
  add1_node->outputs[0].attr.repeats[0] = Sym(32);
  add1_node->outputs[0].attr.strides = {Sym(736)};

  ASSERT_EQ(InsertBroadcastAfterScalarForAscGraph(graph), af::SUCCESS);
  EXPECT_EQ(CountNodesByType(graph, ascir_op::Broadcast::Type), 3U);

  const auto add_brc = graph.FindNode("add0")->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
  const auto mul_brc = graph.FindNode("mul0")->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
  const auto add1_brc = graph.FindNode("add1")->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
  const auto sub_brc = graph.FindNode("sub0")->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
  ASSERT_NE(add_brc, nullptr);
  ASSERT_NE(mul_brc, nullptr);
  ASSERT_NE(add1_brc, nullptr);
  ASSERT_NE(sub_brc, nullptr);
  EXPECT_EQ(add_brc, sub_brc);
  EXPECT_NE(add_brc, mul_brc);
  EXPECT_NE(add_brc, add1_brc);
  EXPECT_NE(mul_brc, add1_brc);
  EXPECT_EQ(CountOutDataEdges(graph, add_brc->GetName()), 2U);
}

TEST_F(TestScalarBroadcastInsert, ScalarDownstreamWithoutTensorAttr_DoesNotSynthesizeBroadcastAttr) {
  auto graph = AscGraphBuilder("test_scalar_downstream_without_tensor_attr")
                   .Loops({Sym("s0")})
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .Scalar("scalar0", "1.0")
                   .Add("add0", "load0", "scalar0")
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  auto add_node = std::dynamic_pointer_cast<AscNode>(graph.FindNode("add0"));
  ASSERT_NE(add_node, nullptr);
  ASSERT_FALSE(add_node->attr.sched.axis.empty());
  add_node->outputs[0].attr.axis.clear();
  add_node->outputs[0].attr.repeats.clear();
  add_node->outputs[0].attr.strides.clear();

  ASSERT_EQ(InsertBroadcastAfterScalarForAscGraph(graph), af::SUCCESS);

  auto brc_node = std::dynamic_pointer_cast<AscNode>(
      graph.FindNode("add0")->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode());
  ASSERT_NE(brc_node, nullptr);
  EXPECT_TRUE(brc_node->outputs[0].attr.axis.empty());
  EXPECT_TRUE(brc_node->outputs[0].attr.repeats.empty());
  EXPECT_TRUE(brc_node->outputs[0].attr.strides.empty());
}

// ==================== Scalar 直连 Output → 不插入 ====================

TEST_F(TestScalarBroadcastInsert, ScalarDirectToOutput_NoInsert) {
  auto graph = AscGraphBuilder("test_scalar_to_output")
                   .Loops({Sym("s0")})
                   .Scalar("scalar0", "1.0")
                   .Output("output0", "scalar0")
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto ret = InsertBroadcastAfterScalarForAscGraph(graph);
  ASSERT_EQ(ret, af::SUCCESS);

  size_t brc_after = CountNodesByType(graph, ascir_op::Broadcast::Type);
  EXPECT_EQ(brc_after, 0U);
}

// ==================== 无 Scalar 节点 → 无变化 ====================

TEST_F(TestScalarBroadcastInsert, NoScalarInGraph_NoChange) {
  auto graph = AscGraphBuilder("test_no_scalar")
                   .Loops({Sym("s0")})
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Store("store0", "abs0")
                   .Output("output0", "store0")
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto ret = InsertBroadcastAfterScalarForAscGraph(graph);
  ASSERT_EQ(ret, af::SUCCESS);

  size_t brc_after = CountNodesByType(graph, ascir_op::Broadcast::Type);
  EXPECT_EQ(brc_after, 0U);
}

// ==================== 部分 Scalar 已有 Broadcast，部分没有 ====================

TEST_F(TestScalarBroadcastInsert, MixedScalarOnlyInsertsForDirectOnes) {
  auto graph = AscGraphBuilder("test_mixed_scalar")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0")
                   .Load("load1", "data1")
                   // scalar0 直连 add0（需要插 broadcast）
                   .Scalar("scalar0", "1.0")
                   .Add("add0", "load0", "scalar0")
                   // scalar1 已有 broadcast（不需要插）
                   .Scalar("scalar1", "2.0")
                   .Broadcast("brc0", "scalar1", {Sym("s0"), Sym("s1")})
                   .Mul("mul0", "load1", "brc0")
                   .Store("store0", "add0")
                   .Store("store1", "mul0")
                   .Output("output0", "store0")
                   .Output("output1", "store1", 1)
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  size_t brc_before = CountNodesByType(graph, ascir_op::Broadcast::Type);
  ASSERT_EQ(brc_before, 1U);  // 只有 brc0

  auto ret = InsertBroadcastAfterScalarForAscGraph(graph);
  ASSERT_EQ(ret, af::SUCCESS);

  size_t brc_after = CountNodesByType(graph, ascir_op::Broadcast::Type);
  EXPECT_EQ(brc_after, 2U);  // brc0 + 新插入的 1 个

  // scalar0 不再直连 add0
  EXPECT_FALSE(IsConnected(graph, "scalar0", "add0"));
  // scalar1 仍然连 brc0
  EXPECT_TRUE(IsConnected(graph, "scalar1", "brc0"));
}

// ==================== Broadcast dtype 应与 Scalar 一致 ====================

TEST_F(TestScalarBroadcastInsert, BroadcastDtypeMatchesScalar) {
  auto graph = AscGraphBuilder("test_brc_dtype")
                   .Loops({Sym("s0")})
                   .Scalar("scalar0", "1.0", ge::DT_FLOAT16)
                   .Abs("abs0", "scalar0")
                   .Store("store0", "abs0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto ret = InsertBroadcastAfterScalarForAscGraph(graph);
  ASSERT_EQ(ret, af::SUCCESS);

  // 找到插入的 broadcast 节点，检查 dtype
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetType() == ascir_op::Broadcast::Type) {
      auto desc = node->GetOpDesc();
      ASSERT_TRUE(desc != nullptr);
      auto output_desc = desc->MutableOutputDesc(0);
      ASSERT_TRUE(output_desc != nullptr);
      EXPECT_EQ(output_desc->GetDataType(), ge::DT_FLOAT16);
      break;
    }
  }
}

// ==================== 多 Scalar 各自独立处理 ====================

TEST_F(TestScalarBroadcastInsert, MultipleScalarsAllGetBroadcast) {
  auto graph = AscGraphBuilder("test_multi_scalars")
                   .Loops({Sym("s0")})
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .Scalar("scalar0", "1.0")
                   .Scalar("scalar1", "2.0")
                   .Add("add0", "load0", "scalar0")
                   .Mul("mul0", "load0", "scalar1")
                   .Store("store0", "add0")
                   .Store("store1", "mul0")
                   .Output("output0", "store0")
                   .Output("output1", "store1", 1)
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto ret = InsertBroadcastAfterScalarForAscGraph(graph);
  ASSERT_EQ(ret, af::SUCCESS);

  // 两个 scalar 各插一个 broadcast
  size_t brc_after = CountNodesByType(graph, ascir_op::Broadcast::Type);
  EXPECT_EQ(brc_after, 2U);

  EXPECT_FALSE(IsConnected(graph, "scalar0", "add0"));
  EXPECT_FALSE(IsConnected(graph, "scalar1", "mul0"));
}

// ==================== Scalar 直连 Data → 不插入 ====================

TEST_F(TestScalarBroadcastInsert, ScalarDirectToData_NoInsert) {
  auto graph = AscGraphBuilder("test_scalar_to_data")
                   .Loops({Sym("s0")})
                   .Scalar("scalar0", "1.0")
                   .Data("data0", 0)
                   .Output("output0", "data0")
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto ret = InsertBroadcastAfterScalarForAscGraph(graph);
  ASSERT_EQ(ret, af::SUCCESS);

  size_t brc_after = CountNodesByType(graph, ascir_op::Broadcast::Type);
  EXPECT_EQ(brc_after, 0U);
}

TEST_F(TestScalarBroadcastInsert, ScalarToBothComputeAndOutput_OnlyInterceptsCompute) {
  auto graph = AscGraphBuilder("test_scalar_compute_output")
                   .Loops({Sym("s0")})
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .Scalar("scalar0", "1.0")
                   .Add("add0", "load0", "scalar0")
                   .Store("store0", "add0")
                   .Output("output0", "store0")
                   .Output("output1", "scalar0", 1)
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto ret = InsertBroadcastAfterScalarForAscGraph(graph);
  ASSERT_EQ(ret, af::SUCCESS);

  // 插入 1 个 broadcast
  EXPECT_EQ(CountNodesByType(graph, ascir_op::Broadcast::Type), 1U);
  // scalar 不再直连 add0（经过 broadcast）
  EXPECT_FALSE(IsConnected(graph, "scalar0", "add0"));
  EXPECT_TRUE(IsConnected(graph, "scalar0", "output1"));
}

// ==================== Broadcast Reorder Tests ====================

TEST_F(TestScalarBroadcastInsert, BroadcastReorderBasic) {
  auto s0 = Sym(4);
  auto s1 = Sym(512 * 1024);  // > 256*1024
  auto graph =
      AscGraphBuilder("broadcast_reorder_basic")
          .Loops({s0, s1})
          .Data("data0", 0, {af::sym::kSymbolOne, s1}, {af::sym::kSymbolZero, af::sym::kSymbolOne}, ge::DT_FLOAT16)
          .Load("load0", "data0", {af::sym::kSymbolOne, s1}, {af::sym::kSymbolZero, af::sym::kSymbolOne})
          .Abs("abs0", "load0")
          .Store("store0", "abs0")
          .Output("y", "store0", 0, ge::DT_FLOAT16)
          .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto load_node = graph.FindNode("load0");
  ASSERT_NE(load_node, nullptr);
  auto original_axis = load_node->attr.sched.axis;
  ASSERT_EQ(original_axis.size(), 2UL);

  std::vector<AutoScheduleOutput> results;
  AutoSchedule schedule(graph, results);
  EXPECT_EQ(schedule.DoAutoSchedule(), af::SUCCESS);

  // broadcast axis (z0, stride=0) moved to inner: expected [z1, z0]
  std::vector<int64_t> expected_order = {original_axis[1], original_axis[0]};
  EXPECT_EQ(load_node->attr.sched.axis, expected_order);

  auto abs_node = graph.FindNode("abs0");
  ASSERT_NE(abs_node, nullptr);
  EXPECT_EQ(abs_node->attr.sched.axis, expected_order);
}

TEST_F(TestScalarBroadcastInsert, BroadcastReorderSkipBrcTooLarge) {
  auto s0 = Sym(32);
  auto s1 = Sym(512 * 1024);
  auto graph =
      AscGraphBuilder("broadcast_reorder_large_brc")
          .Loops({s0, s1})
          .Data("data0", 0, {af::sym::kSymbolOne, s1}, {af::sym::kSymbolZero, af::sym::kSymbolOne}, ge::DT_FLOAT16)
          .Load("load0", "data0", {af::sym::kSymbolOne, s1}, {af::sym::kSymbolZero, af::sym::kSymbolOne})
          .Abs("abs0", "load0")
          .Store("store0", "abs0")
          .Output("y", "store0", 0, ge::DT_FLOAT16)
          .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto load_node = graph.FindNode("load0");
  ASSERT_NE(load_node, nullptr);
  auto original_axis = load_node->attr.sched.axis;

  std::vector<AutoScheduleOutput> results;
  AutoSchedule schedule(graph, results);
  EXPECT_EQ(schedule.DoAutoSchedule(), af::SUCCESS);

  // brc_size=32 > 16, reorder skipped
  EXPECT_EQ(load_node->attr.sched.axis, original_axis);
}

TEST_F(TestScalarBroadcastInsert, BroadcastReorderSkipDataTooSmall) {
  auto s0 = Sym(4);
  auto s1 = Sym(8);
  auto graph =
      AscGraphBuilder("broadcast_reorder_small_data")
          .Loops({s0, s1})
          .Data("data0", 0, {af::sym::kSymbolOne, s1}, {af::sym::kSymbolZero, af::sym::kSymbolOne}, ge::DT_FLOAT16)
          .Load("load0", "data0", {af::sym::kSymbolOne, s1}, {af::sym::kSymbolZero, af::sym::kSymbolOne})
          .Abs("abs0", "load0")
          .Store("store0", "abs0")
          .Output("y", "store0", 0, ge::DT_FLOAT16)
          .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto load_node = graph.FindNode("load0");
  ASSERT_NE(load_node, nullptr);
  auto original_axis = load_node->attr.sched.axis;

  std::vector<AutoScheduleOutput> results;
  AutoSchedule schedule(graph, results);
  EXPECT_EQ(schedule.DoAutoSchedule(), af::SUCCESS);

  // non_brc_size=8 < 256*1024, reorder skipped
  EXPECT_EQ(load_node->attr.sched.axis, original_axis);
}
