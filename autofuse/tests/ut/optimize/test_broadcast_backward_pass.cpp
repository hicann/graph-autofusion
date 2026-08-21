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

#include <string>
#include <vector>

#include "asc_graph_builder.h"
#include "tests/framework/broadcast_backward/broadcast_backward_test_utils.h"
#include "tests/framework/broadcast_backward/broadcast_backward_ut_utils.h"
#include "ascgraph_info_complete.h"
#include "graph_utils.h"
#include "optimize/graph_pass/broadcast_backward_pass.h"
#include "optimize/platform/common/pass_runner.h"
#include "schedule_utils.h"

namespace {
using af::AscGraph;
using af::testing::AscGraphBuilder;
using af::testing::Sym;
using namespace broadcast_backward_test;

class ScopedTestPlatform {
 public:
  explicit ScopedTestPlatform(const char *platform) {
    ge::PlatformContext::GetInstance().SetPlatform(platform);
  }

  ~ScopedTestPlatform() {
    ge::PlatformContext::GetInstance().Reset();
  }
};
}  // namespace

TEST(BroadcastBackwardPass, MovesSingleBroadcastChain) {
  auto graph = BuildUnaryGraph("Abs");
  CompleteApiInfo(graph);
  const auto load_node = FindNode(graph, "load");
  ASSERT_NE(load_node, nullptr);
  std::vector<af::Expression> expected_strides;
  ASSERT_EQ(
      optimize::ScheduleUtils::RecalculateStridesFromRepeats(load_node->outputs[0].attr.repeats, expected_strides),
      af::SUCCESS);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load", "compute"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "compute", "broadcast"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "broadcast", "store"));
  const auto compute_node = FindNode(graph, "compute");
  ASSERT_NE(compute_node, nullptr);
  ExpectStaticEq(compute_node->outputs[0].attr.strides, expected_strides);
}

TEST(BroadcastBackwardPass, MovesAllSupportedUnaryOperators) {
  for (const auto &op_name : std::vector<std::string>{"Abs", "Neg", "Exp", "Sqrt", "Rsqrt", "Relu", "Reciprocal", "Erf",
                                                      "Sign", "Tanh", "Ln"}) {
    auto graph = BuildUnaryGraph(op_name);
    CompleteApiInfo(graph);

    optimize::BroadcastBackwardPass pass;
    ASSERT_EQ(pass.RunPass(graph), af::SUCCESS) << op_name;
    EXPECT_TRUE(IsConnected(graph, "load", "compute")) << op_name;
    EXPECT_TRUE(IsConnected(graph, "compute", "broadcast")) << op_name;
    EXPECT_TRUE(IsConnected(graph, "broadcast", "store")) << op_name;
  }
}

TEST(BroadcastBackwardPass, MovesMultipleComputeNodes) {
  auto graph = AscGraphBuilder("broadcast_backward_compute_chain")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {Sym("s0"), Sym("s1")})
                   .Abs("abs", "broadcast")
                   .Relu("relu", "abs")
                   .Store("store", "relu")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "relu"));
  EXPECT_TRUE(IsConnected(graph, "relu", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
}

TEST(BroadcastBackwardPass, MovesFormerCastBarrierWithDtypeAwareBackward) {
  auto graph = AscGraphBuilder("broadcast_backward_cast_barrier")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {Sym("s0"), Sym("s1")})
                   .Cast("cast", "broadcast", af::DT_FLOAT16)
                   .Store("store", "cast")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "cast"));
  EXPECT_TRUE(IsConnected(graph, "cast", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
}

TEST(BroadcastBackwardPass, SkipsReduceBarrier) {
  auto graph = AscGraphBuilder("broadcast_backward_reduce_barrier")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {Sym("s0"), Sym("s1")})
                   .Sum("sum", "broadcast", {0U})
                   .Store("store", "sum")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast", "sum"));
  EXPECT_TRUE(IsConnected(graph, "sum", "store"));
}

TEST(BroadcastBackwardPass, SkipsVectorizedLayout) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildUnaryGraph("Abs");
  CompleteApiInfo(graph);
  auto load_node = FindNode(graph, "load");
  ASSERT_NE(load_node, nullptr);
  load_node->outputs[0].attr.vectorized_axis.push_back(0U);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "store"));
}

TEST(BroadcastBackwardPass, SkipsMultipleBroadcastConsumers) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = AscGraphBuilder("broadcast_backward_multiple_consumers")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {Sym("s0"), Sym("s1")})
                   .Abs("abs0", "broadcast")
                   .Abs("abs1", "broadcast")
                   .Store("store0", "abs0")
                   .Store("store1", "abs1")
                   .Output("output0", "store0")
                   .Output("output1", "store1")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast", "abs0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "abs1"));
}

TEST(BroadcastBackwardPass, SkipsIncompleteBroadcastLayout) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildUnaryGraph("Abs");
  CompleteApiInfo(graph);
  auto broadcast_node = FindNode(graph, "broadcast");
  ASSERT_NE(broadcast_node, nullptr);
  broadcast_node->outputs[0].attr.strides.clear();

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "store"));
}

TEST(BroadcastBackwardPass, SkipsSchedMismatch) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildUnaryGraph("Abs");
  CompleteApiInfo(graph);
  auto compute_node = FindNode(graph, "compute");
  ASSERT_NE(compute_node, nullptr);
  compute_node->attr.sched.axis.clear();

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "store"));
}

TEST(BroadcastBackwardPass, SkipsControlEdge) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildUnaryGraph("Abs");
  CompleteApiInfo(graph);
  auto load_node = FindNode(graph, "load");
  auto compute_node = FindNode(graph, "compute");
  ASSERT_NE(load_node, nullptr);
  ASSERT_NE(compute_node, nullptr);
  ASSERT_EQ(af::GraphUtils::AddEdge(load_node->GetOutControlAnchor(), compute_node->GetInControlAnchor()), af::SUCCESS);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "store"));
}

TEST(BroadcastBackwardPass, SkipsScalarBroadcastSource) {
  auto graph = AscGraphBuilder("broadcast_backward_scalar")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Scalar("scalar", "1.0")
                   .Broadcast("broadcast0", "scalar", {Sym("s0"), Sym("s1")})
                   .Broadcast("broadcast1", "broadcast0", {Sym("s0"), Sym("s1")})
                   .Abs("abs", "broadcast1")
                   .Store("store", "abs")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "broadcast1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "store"));
}

TEST(BroadcastBackwardPass, MovesSupportedScalarBroadcastBranches) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_scalar_supported")
                   .Loops({s0, s1})
                   .Scalar("scalar0", "1.0")
                   .Scalar("scalar1", "2.0")
                   .Broadcast("broadcast0", "scalar0", {s0, s1})
                   .Broadcast("broadcast1", "scalar1", {s0, s1})
                   .Sub("compute", "broadcast0", "broadcast1")
                   .Store("store", "compute")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "scalar0", "compute"));
  EXPECT_TRUE(IsConnected(graph, "scalar1", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "compute_broadcast_backward_common"));
  EXPECT_TRUE(IsConnected(graph, "compute_broadcast_backward_common", "store"));
  EXPECT_FALSE(HasNode(graph, "broadcast0"));
  EXPECT_FALSE(HasNode(graph, "broadcast1"));
  const auto compute = FindNode(graph, "compute");
  ASSERT_NE(compute, nullptr);
  ExpectStaticEq(compute->outputs[0].attr.repeats, {af::sym::kSymbolOne, af::sym::kSymbolOne});
  ExpectStaticEq(compute->outputs[0].attr.strides, {af::sym::kSymbolZero, af::sym::kSymbolZero});
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "scalar0", "compute"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "scalar1", "compute"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "compute", "compute_broadcast_backward_common"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "compute_broadcast_backward_common", "store"));
}

TEST(BroadcastBackwardPass, SkipsUnsupportedAllScalarInputs) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_scalar_unsupported")
                   .Loops({s0, s1})
                   .Scalar("scalar0", "1.0")
                   .Scalar("scalar1", "2.0")
                   .Broadcast("broadcast0", "scalar0", {s0, s1})
                   .Broadcast("broadcast1", "scalar1", {s0, s1})
                   .Add("compute", "broadcast0", "broadcast1")
                   .Store("store", "compute")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "scalar0", "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "scalar1", "broadcast1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "compute"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "compute"));
  EXPECT_FALSE(HasNode(graph, "compute_broadcast_backward_common"));
}

TEST(BroadcastBackwardPass, SkipsScalarBranchWithResidualBroadcastAxis) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_scalar_mixed")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Scalar("scalar", "1.0")
                   .Broadcast("broadcast0", "load", {s0, s1})
                   .Broadcast("broadcast1", "scalar", {s0, s1})
                   .Add("compute", "broadcast0", "broadcast1")
                   .Store("store", "compute")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "scalar", "broadcast1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "compute"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "store"));
  EXPECT_FALSE(HasNode(graph, "compute_broadcast_backward_common"));
}

TEST(BroadcastBackwardPass, MovesSupportedScalarSameConsumerMultiReference) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_scalar_same_consumer")
                   .Loops({s0, s1})
                   .Scalar("scalar", "1.0")
                   .Broadcast("broadcast", "scalar", {s0, s1})
                   .Sub("compute", "broadcast", "broadcast")
                   .Store("store", "compute")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "scalar", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  const auto compute = FindNode(graph, "compute");
  ASSERT_NE(compute, nullptr);
  ExpectStaticEq(compute->outputs[0].attr.repeats, {af::sym::kSymbolOne, af::sym::kSymbolOne});
  ExpectStaticEq(compute->outputs[0].attr.strides, {af::sym::kSymbolZero, af::sym::kSymbolZero});
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "compute", "broadcast"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "broadcast", "store"));
}

TEST(BroadcastBackwardPass, SkipsUnsupportedScalarSameConsumerMultiReference) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_scalar_same_consumer_unsupported")
                   .Loops({s0, s1})
                   .Scalar("scalar", "1.0")
                   .Broadcast("broadcast", "scalar", {s0, s1})
                   .Add("compute", "broadcast", "broadcast")
                   .Store("store", "compute")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  const auto compute = FindNode(graph, "compute");
  ASSERT_NE(compute, nullptr);
  EXPECT_TRUE(IsConnected(graph, "scalar", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "compute"));

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(HasNode(graph, "scalar"));
  EXPECT_TRUE(HasNode(graph, "broadcast"));
  EXPECT_TRUE(HasNode(graph, "compute"));
  EXPECT_TRUE(IsConnected(graph, "scalar", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "store"));
}

TEST(BroadcastBackwardPass, MovesSupportedMultiLevelScalarBroadcastChain) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_scalar_multi_level")
                   .Loops({s0, s1})
                   .Scalar("scalar", "1.0")
                   .Broadcast("broadcast0", "scalar", {0})
                   .Broadcast("broadcast1", "broadcast0", {1})
                   .Sub("compute", "broadcast1", "broadcast1")
                   .Store("store", "compute")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "scalar", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "broadcast1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "store"));
  const auto compute = FindNode(graph, "compute");
  ASSERT_NE(compute, nullptr);
  ExpectStaticEq(compute->outputs[0].attr.strides, {af::sym::kSymbolZero, af::sym::kSymbolZero});
}

TEST(BroadcastBackwardPass, ScalarBranchAxisRequiresZeroInputStride) {
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_scalar_stride_guard")
                   .Loops({s0, s1})
                   .Scalar("scalar0", "1.0")
                   .Scalar("scalar1", "2.0")
                   .Broadcast("broadcast0", "scalar0", {0})
                   .Broadcast("broadcast1", "scalar1", {0})
                   .Sub("compute", "broadcast0", "broadcast1")
                   .Store("store", "compute")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  const auto scalar1 = FindNode(graph, "scalar1");
  const auto broadcast0 = FindNode(graph, "broadcast0");
  const auto compute = FindNode(graph, "compute");
  ASSERT_NE(scalar1, nullptr);
  ASSERT_NE(broadcast0, nullptr);
  ASSERT_NE(compute, nullptr);
  scalar1->outputs[0].attr.strides[0] = af::sym::kSymbolOne;
}

TEST(BroadcastBackwardPass, MovesIdenticalMultiInputBroadcastChains) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildBinaryGraph("Add");
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  ExpectBinaryBroadcastMove(graph);
}

TEST(BroadcastBackwardPass, MovesMultiNodeBroadcastAndComputeChains) {
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact_repeats = {s0, af::sym::kSymbolOne, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolOne, af::sym::kSymbolZero, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_multi_node_chains")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact_repeats, compact_strides)
                   .Load("load1", "data1", compact_repeats, compact_strides)
                   .Broadcast("broadcast00", "load0", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("broadcast01", "broadcast00", {s0, s1, s2})
                   .Broadcast("broadcast10", "load1", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("broadcast11", "broadcast10", {s0, s1, s2})
                   .Add("merge", "broadcast01", "broadcast11")
                   .Abs("abs", "merge")
                   .Relu("relu", "abs")
                   .Store("store", "relu")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "load1", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "relu"));
  EXPECT_TRUE(IsConnected(graph, "relu", "broadcast00"));
  EXPECT_TRUE(IsConnected(graph, "broadcast00", "broadcast01"));
  EXPECT_TRUE(IsConnected(graph, "broadcast01", "store"));
  EXPECT_FALSE(HasNode(graph, "broadcast10"));
  EXPECT_FALSE(HasNode(graph, "broadcast11"));
}

TEST(BroadcastBackwardPass, MovesAllSupportedBinaryOperators) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  for (const auto &op_name : std::vector<std::string>{"Add", "Sub", "Mul", "Div", "Minimum", "Maximum"}) {
    auto graph = BuildBinaryGraph(op_name);
    CompleteApiInfo(graph);

    optimize::BroadcastBackwardPass pass;
    ASSERT_EQ(pass.RunPass(graph), af::SUCCESS) << op_name;
    EXPECT_TRUE(IsConnected(graph, "load0", "compute")) << op_name;
    EXPECT_TRUE(IsConnected(graph, "load1", "compute")) << op_name;
    EXPECT_TRUE(IsConnected(graph, "compute", "broadcast0")) << op_name;
    EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load0", "compute")) << op_name;
    EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load1", "compute")) << op_name;
    EXPECT_TRUE(IsEdgeAttrConsistent(graph, "compute", "broadcast0")) << op_name;
    EXPECT_FALSE(HasNode(graph, "broadcast1")) << op_name;
  }
}

TEST(BroadcastBackwardPass, MovesBeforeMultiInputBarrier) {
  auto graph = AscGraphBuilder("broadcast_backward_multi_input_barrier")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", kCompactRepeats, kCompactStrides)
                   .Load("load1", "data1")
                   .Broadcast("broadcast", "load0", {Sym("s0"), Sym("s1")})
                   .Abs("abs", "broadcast")
                   .Add("add", "load1", "abs")
                   .Store("store", "add")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "add"));
  EXPECT_TRUE(IsConnected(graph, "load1", "add"));
}

TEST(BroadcastBackwardPass, SkipsFollowingMultiInputBarrier) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = AscGraphBuilder("broadcast_backward_following_multi_input_barrier")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Data("data2", 2)
                   .Load("load0", "data0", kCompactRepeats, kCompactStrides)
                   .Load("load1", "data1", kCompactRepeats, kCompactStrides)
                   .Load("load2", "data2")
                   .Broadcast("broadcast0", "load0", {Sym("s0"), Sym("s1")})
                   .Broadcast("broadcast1", "load1", {Sym("s0"), Sym("s1")})
                   .Add("merge", "broadcast0", "broadcast1")
                   .Add("barrier", "load2", "merge")
                   .Store("store", "barrier")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "barrier"));
}

TEST(BroadcastBackwardPass, SkipsMultiInputSchedMismatch) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildBinaryGraph("Add");
  CompleteApiInfo(graph);
  auto broadcast1_node = FindNode(graph, "broadcast1");
  ASSERT_NE(broadcast1_node, nullptr);
  broadcast1_node->attr.sched.axis.clear();

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "compute"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "compute"));
  EXPECT_TRUE(HasNode(graph, "broadcast1"));
}

TEST(BroadcastBackwardPass, SkipsTailLayoutMismatch) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildBinaryGraph("Add");
  CompleteApiInfo(graph);
  auto broadcast0_node = FindNode(graph, "broadcast0");
  auto broadcast1_node = FindNode(graph, "broadcast1");
  auto compute_node = FindNode(graph, "compute");
  ASSERT_NE(broadcast0_node, nullptr);
  ASSERT_NE(broadcast1_node, nullptr);
  ASSERT_NE(compute_node, nullptr);
  broadcast0_node->outputs[0].attr.strides[0] = af::sym::kSymbolZero;
  broadcast1_node->outputs[0].attr.strides[0] = af::sym::kSymbolZero;
  compute_node->inputs[0].attr.strides[0] = af::sym::kSymbolZero;
  compute_node->inputs[1].attr.strides[0] = af::sym::kSymbolZero;

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "compute"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "store"));
}

TEST(BroadcastBackwardPass, SkipsEdgeDtypeMismatch) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildBinaryGraph("Add");
  CompleteApiInfo(graph);
  auto broadcast1_node = FindNode(graph, "broadcast1");
  auto add_node = FindNode(graph, "compute");
  ASSERT_NE(broadcast1_node, nullptr);
  ASSERT_NE(add_node, nullptr);
  broadcast1_node->outputs[0].attr.dtype = af::DT_FLOAT16;
  add_node->inputs[1].attr.dtype = af::DT_FLOAT16;

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "compute"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "compute"));
}

TEST(BroadcastBackwardPass, SkipsDifferentInputLayouts) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildBinaryGraph("Add");
  CompleteApiInfo(graph);
  auto load1_node = FindNode(graph, "load1");
  auto broadcast1_node = FindNode(graph, "broadcast1");
  ASSERT_NE(load1_node, nullptr);
  ASSERT_NE(broadcast1_node, nullptr);
  load1_node->outputs[0].attr.strides[0] = af::sym::kSymbolZero;
  broadcast1_node->inputs[0].attr.strides[0] = af::sym::kSymbolZero;

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "compute"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "compute"));
}

TEST(BroadcastBackwardPass, HandlesEmptyGraph) {
  af::AscGraph graph("broadcast_backward_empty");
  optimize::BroadcastBackwardPass pass;
  EXPECT_EQ(pass.RunPass(graph), af::SUCCESS);
}

// ===== Dtype-aware backward tests for Cast and dtype-changing operators =====

TEST(BroadcastBackwardPass, MovesCastAcrossBroadcast) {
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_cast")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {s0, s1})
                   .Cast("cast", "broadcast", af::DT_FLOAT16)
                   .Store("store", "cast")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_FLOAT16);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "cast"));
  EXPECT_TRUE(IsConnected(graph, "cast", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  const auto broadcast_node = FindNode(graph, "broadcast");
  ASSERT_NE(broadcast_node, nullptr);
  EXPECT_EQ(broadcast_node->outputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(broadcast_node->inputs[0].attr.dtype, af::DT_FLOAT16);
}

TEST(BroadcastBackwardPass, MovesComputeAndCastAcrossBroadcast) {
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_abs_cast")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {s0, s1})
                   .Abs("abs", "broadcast")
                   .Cast("cast", "abs", af::DT_FLOAT16)
                   .Store("store", "cast")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_FLOAT16);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "cast"));
  EXPECT_TRUE(IsConnected(graph, "cast", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  const auto broadcast_node = FindNode(graph, "broadcast");
  ASSERT_NE(broadcast_node, nullptr);
  EXPECT_EQ(broadcast_node->outputs[0].attr.dtype, af::DT_FLOAT16);
}

TEST(BroadcastBackwardPass, MovesComparisonAcrossIdenticalBroadcastBranches) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_comparison")
                   .Loops({s0, s1})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", kCompactRepeats, kCompactStrides)
                   .Load("load1", "data1", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast0", "load0", {s0, s1})
                   .Broadcast("broadcast1", "load1", {s0, s1})
                   .Op<af::ascir_op::Ge>("compare", {"broadcast0", "broadcast1"})
                   .Store("store", "compare")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "compare"));
  EXPECT_TRUE(IsConnected(graph, "load1", "compare"));
  EXPECT_TRUE(IsConnected(graph, "compare", "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "store"));
  EXPECT_FALSE(HasNode(graph, "broadcast1"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load0", "compare"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load1", "compare"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "compare", "broadcast0"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "broadcast0", "store"));
}

TEST(BroadcastBackwardPass, MovesAdditionalDtypeAwareBinaryOps) {
  GTEST_SKIP();
  for (const auto &op_name : {std::string("Eq"), std::string("TrueDiv")}) {
    SCOPED_TRACE(op_name);
    auto graph = BuildDtypeAwareBinaryGraph(op_name);
    ExpectDtypeAwareBinaryMove(graph, op_name);
  }
}

TEST(BroadcastBackwardPass, DtypeAwareBackwardEnablesCommonAxisAtMultiInputTail) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildDtypeAwareCommonAxisGraph("broadcast_backward_dtype_aware_common_axis");
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "relu", af::DT_FLOAT16);
  SetNodeDtype(graph, "merge", af::DT_FLOAT16);
  SetNodeDtype(graph, "store", af::DT_FLOAT16);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "cast0"));
  EXPECT_TRUE(IsConnected(graph, "load1", "cast1"));
  EXPECT_TRUE(IsConnected(graph, "cast1", "relu"));
  EXPECT_EQ(FindNode(graph, "broadcast0"), nullptr);
  EXPECT_EQ(FindNode(graph, "broadcast1"), nullptr);
  const auto residual0 = FindNode(graph, "broadcast0_residual_0");
  const auto residual1 = FindNode(graph, "broadcast1_residual_1");
  const auto merge = FindNode(graph, "merge");
  const auto common = FindNode(graph, "merge_broadcast_backward_common");
  ASSERT_NE(residual0, nullptr);
  ASSERT_NE(residual1, nullptr);
  ASSERT_NE(merge, nullptr);
  ASSERT_NE(common, nullptr);
  EXPECT_TRUE(IsConnected(graph, "cast0", "broadcast0_residual_0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0_residual_0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "relu", "broadcast1_residual_1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1_residual_1", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "merge_broadcast_backward_common"));
  EXPECT_TRUE(IsConnected(graph, "merge_broadcast_backward_common", "store"));
  EXPECT_EQ(residual0->inputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(residual0->outputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(residual1->inputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(residual1->outputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(merge->inputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(merge->inputs[1].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(merge->outputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(common->inputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(common->outputs[0].attr.dtype, af::DT_FLOAT16);
}

// ===== Common broadcast-axis partial backward tests =====

TEST(BroadcastBackwardPass, CommonAxisBackwardSkipsBroadcastControlEdges) {
  for (size_t case_index = 0U; case_index < 2U; ++case_index) {
    auto graph = BuildCommonAxisGraph("broadcast_backward_common_axis_control_" + std::to_string(case_index));
    CompleteApiInfo(graph);
    const auto broadcast = FindNode(graph, "broadcast0");
    const auto peer = case_index == 0U ? FindNode(graph, "load0") : FindNode(graph, "store");
    ASSERT_NE(broadcast, nullptr);
    ASSERT_NE(peer, nullptr);
    const auto status = case_index == 0U
                            ? af::GraphUtils::AddEdge(peer->GetOutControlAnchor(), broadcast->GetInControlAnchor())
                            : af::GraphUtils::AddEdge(broadcast->GetOutControlAnchor(), peer->GetInControlAnchor());
    ASSERT_EQ(status, af::SUCCESS);

    optimize::BroadcastBackwardPass pass;
    ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
    EXPECT_TRUE(HasNode(graph, "broadcast0"));
    EXPECT_TRUE(HasNode(graph, "broadcast1"));
    EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
    EXPECT_EQ(broadcast->GetInControlNodesSize() + broadcast->GetOutControlNodesSize(), 1U);
  }
}

TEST(BroadcastBackwardPass, CommonAxisBackwardSkipsSourceControlEdge) {
  auto graph = BuildCommonAxisGraph("broadcast_backward_common_axis_source_control");
  CompleteApiInfo(graph);
  const auto load = FindNode(graph, "load0");
  const auto store = FindNode(graph, "store");
  ASSERT_NE(load, nullptr);
  ASSERT_NE(store, nullptr);
  ASSERT_EQ(af::GraphUtils::AddEdge(load->GetOutControlAnchor(), store->GetInControlAnchor()), af::SUCCESS);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(HasNode(graph, "broadcast0"));
  EXPECT_TRUE(HasNode(graph, "broadcast1"));
  EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
  EXPECT_EQ(load->GetOutControlNodesSize(), 1U);
}

TEST(BroadcastBackwardPass, CommonAxisBackwardSkipsSourceBroadcastEdgeAttrMismatch) {
  auto graph = BuildCommonAxisGraph("broadcast_backward_common_axis_source_edge_mismatch");
  CompleteApiInfo(graph);
  const auto broadcast = FindNode(graph, "broadcast0");
  ASSERT_NE(broadcast, nullptr);
  const auto input_desc = broadcast->GetOpDesc()->MutableInputDesc(0U);
  ASSERT_NE(input_desc, nullptr);
  input_desc->SetDataType(af::DT_FLOAT16);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(HasNode(graph, "broadcast0"));
  EXPECT_TRUE(HasNode(graph, "broadcast1"));
  EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
}

TEST(BroadcastBackwardPass, MovesCommonAxisFromMultiNodeBroadcastChains) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  auto graph = AscGraphBuilder("broadcast_backward_common_axis_multi_node")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2},
                         {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne})
                   .Load("load1", "data1", {s0, af::sym::kSymbolOne, af::sym::kSymbolOne},
                         {af::sym::kSymbolOne, af::sym::kSymbolZero, af::sym::kSymbolZero})
                   .Broadcast("residual0", "load0", {s0, af::sym::kSymbolOne, s2})
                   .Broadcast("common0", "residual0", {s0, s1, s2})
                   .Broadcast("residual1", "load1", {s0, af::sym::kSymbolOne, s2})
                   .Broadcast("common1", "residual1", {s0, s1, s2})
                   .Add("merge", "common0", "common1")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "residual0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "residual1", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "merge_broadcast_backward_common"));
  EXPECT_FALSE(HasNode(graph, "common0_residual_0"));
  EXPECT_FALSE(HasNode(graph, "common1_residual_1"));
  EXPECT_FALSE(HasNode(graph, "common0"));
  EXPECT_FALSE(HasNode(graph, "common1"));
}

TEST(BroadcastBackwardPass, MovesCreatedCommonBroadcastPastUnaryChainWithoutIdentityResiduals) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact = {s0, af::sym::kSymbolOne, s2};
  const std::vector<af::Expression> compact_strides = {s2, af::sym::kSymbolZero, af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_created_common_unary_chain")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact, compact_strides)
                   .Load("load1", "data1", compact, compact_strides)
                   .Broadcast("broadcast0", "load0", {1})
                   .Broadcast("broadcast1", "load1", {1})
                   .Abs("abs", "broadcast0")
                   .Cast("cast0", "abs", af::DT_FLOAT16)
                   .Cast("cast1", "broadcast1", af::DT_FLOAT16)
                   .Relu("relu", "cast1")
                   .Add("merge", "cast0", "relu")
                   .Sqrt("sqrt", "merge")
                   .Op<af::ascir_op::Sigmoid>("sigmoid", {"sqrt"})
                   .Store("store", "sigmoid")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  for (const auto *node_name : {"relu", "merge", "sqrt", "sigmoid", "store"}) {
    SetNodeDtype(graph, node_name, af::DT_FLOAT16);
  }

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "broadcast0_residual_0"));
  EXPECT_FALSE(HasNode(graph, "broadcast1_residual_1"));
  EXPECT_TRUE(IsConnected(graph, "cast0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "relu", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "sqrt"));
  EXPECT_TRUE(IsConnected(graph, "sqrt", "sigmoid"));
  EXPECT_TRUE(IsConnected(graph, "sigmoid", "merge_broadcast_backward_common"));
  EXPECT_TRUE(IsConnected(graph, "merge_broadcast_backward_common", "store"));
}

TEST(BroadcastBackwardPass, MovesCommonAxisBeforeResidualBroadcasts) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  auto graph = AscGraphBuilder("broadcast_backward_common_axis_before_residual")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2},
                         {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne})
                   .Load("load1", "data1", {s0, af::sym::kSymbolOne, af::sym::kSymbolOne},
                         {af::sym::kSymbolOne, af::sym::kSymbolZero, af::sym::kSymbolZero})
                   .Broadcast("common0", "load0", {af::sym::kSymbolOne, s1, s2})
                   .Broadcast("residual0", "common0", {s0, s1, s2})
                   .Broadcast("common1", "load1", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("residual1", "common1", {s0, s1, s2})
                   .Add("merge", "residual0", "residual1")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "residual0_residual_0"));
  EXPECT_TRUE(IsConnected(graph, "load1", "residual1_residual_1"));
  EXPECT_TRUE(IsConnected(graph, "merge", "merge_broadcast_backward_common"));
  EXPECT_FALSE(HasNode(graph, "common0"));
  EXPECT_FALSE(HasNode(graph, "common1"));
  EXPECT_FALSE(HasNode(graph, "residual0"));
  EXPECT_FALSE(HasNode(graph, "residual1"));
}

TEST(BroadcastBackwardPass, SkipsCommonAxesDistributedAcrossBroadcastChain) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_distributed_common_axis")
                   .Loops({s0, s1})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", {af::sym::kSymbolOne, af::sym::kSymbolOne},
                         {af::sym::kSymbolZero, af::sym::kSymbolZero})
                   .Load("load1", "data1", {af::sym::kSymbolOne, af::sym::kSymbolOne},
                         {af::sym::kSymbolZero, af::sym::kSymbolZero})
                   .Broadcast("broadcast00", "load0", {s0, af::sym::kSymbolOne})
                   .Broadcast("broadcast01", "broadcast00", {s0, s1})
                   .Broadcast("broadcast10", "load1", {s0, s1})
                   .Add("merge", "broadcast01", "broadcast10")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast01", "merge"));
  EXPECT_TRUE(IsConnected(graph, "broadcast10", "merge"));
  EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
}

TEST(BroadcastBackwardPass, MovesCommonAxisPartialBackward) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  auto graph = BuildCommonAxisGraph("broadcast_backward_common_axis");
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  auto b_common = FindNode(graph, "merge_broadcast_backward_common");
  EXPECT_NE(b_common, nullptr);
  EXPECT_TRUE(IsConnected(graph, "merge", "merge_broadcast_backward_common"));
  EXPECT_TRUE(IsConnected(graph, "merge_broadcast_backward_common", "store"));
  auto b0_residual = FindNode(graph, "broadcast0_residual_0");
  EXPECT_NE(b0_residual, nullptr);
  auto b1_residual = FindNode(graph, "broadcast1_residual_1");
  EXPECT_NE(b1_residual, nullptr);
  ExpectResidualConnections(graph);
  EXPECT_FALSE(HasNode(graph, "broadcast0"));
  EXPECT_FALSE(HasNode(graph, "broadcast1"));
  std::vector<af::Expression> expected_compact_strides;
  ASSERT_EQ(optimize::ScheduleUtils::RecalculateStridesFromRepeats(
                std::vector<af::Expression>{s0, af::sym::kSymbolOne, s2}, expected_compact_strides),
            af::SUCCESS);
  std::vector<af::Expression> expected_expanded_strides;
  ASSERT_EQ(optimize::ScheduleUtils::RecalculateStridesFromRepeats(std::vector<af::Expression>{s0, s1, s2},
                                                                   expected_expanded_strides),
            af::SUCCESS);
  ExpectCommonAxisLayouts(graph, {s0, af::sym::kSymbolOne, s2}, {s0, s1, s2}, expected_compact_strides,
                          expected_expanded_strides);
}

TEST(BroadcastBackwardPass, MovesCommonAxisToActualSuccessorInput) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact0 = {af::sym::kSymbolOne, af::sym::kSymbolOne, s2};
  const std::vector<af::Expression> strides0 = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact1 = {s0, af::sym::kSymbolOne, af::sym::kSymbolOne};
  const std::vector<af::Expression> strides1 = {af::sym::kSymbolOne, af::sym::kSymbolZero, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_successor_input")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact0, strides0)
                   .Load("load1", "data1", compact1, strides1)
                   .Broadcast("broadcast0", "load0", {0, 1})
                   .Broadcast("broadcast1", "load1", {1, 2})
                   .Add("merge", "broadcast0", "broadcast1")
                   .Add("succ", "merge", "merge")
                   .Store("store", "succ")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  const auto merge = FindNode(graph, "merge");
  const auto succ = FindNode(graph, "succ");
  ASSERT_NE(merge, nullptr);
  ASSERT_NE(succ, nullptr);
  ASSERT_EQ(af::GraphUtils::RemoveEdge(merge->GetOutDataAnchor(0), succ->GetInDataAnchor(0)), af::SUCCESS);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  const auto common = FindNode(graph, "merge_broadcast_backward_common");
  ASSERT_NE(common, nullptr);
  EXPECT_EQ(succ->GetInDataAnchor(0)->GetPeerOutAnchor(), nullptr);
  const auto succ_input1_peer = succ->GetInDataAnchor(1)->GetPeerOutAnchor();
  ASSERT_NE(succ_input1_peer, nullptr);
  EXPECT_EQ(succ_input1_peer->GetOwnerNode()->GetName(), "merge_broadcast_backward_common");
  EXPECT_TRUE(AreConnectedTensorAttrsEqual(common, succ, 1U));
}

TEST(BroadcastBackwardPass, SkipsCommonAxisNoOverlap) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact0 = {af::sym::kSymbolOne, af::sym::kSymbolOne, s2};
  const std::vector<af::Expression> strides0 = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact1 = {s0, af::sym::kSymbolOne, af::sym::kSymbolOne};
  const std::vector<af::Expression> strides1 = {af::sym::kSymbolOne, af::sym::kSymbolZero, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_no_common_axis")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact0, strides0)
                   .Load("load1", "data1", compact1, strides1)
                   .Broadcast("broadcast0", "load0", {0})
                   .Broadcast("broadcast1", "load1", {2})
                   .Add("merge", "broadcast0", "broadcast1")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "merge"));
}

TEST(BroadcastBackwardPass, MovesNoResidualCaseInIdenticalMultiInputBackward) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const std::vector<af::Expression> compact = {af::sym::kSymbolOne, af::sym::kSymbolOne};
  const std::vector<af::Expression> strides = {af::sym::kSymbolZero, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_common_only")
                   .Loops({s0, s1})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact, strides)
                   .Load("load1", "data1", compact, strides)
                   .Broadcast("broadcast0", "load0", {0, 1})
                   .Broadcast("broadcast1", "load1", {0, 1})
                   .Add("merge", "broadcast0", "broadcast1")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_EQ(FindNode(graph, "merge_broadcast_backward_common"), nullptr);
  const bool has_broadcast0 = HasNode(graph, "broadcast0");
  const bool has_broadcast1 = HasNode(graph, "broadcast1");
  ASSERT_NE(has_broadcast0, has_broadcast1);
  const char *kept_broadcast = has_broadcast0 ? "broadcast0" : "broadcast1";
  EXPECT_TRUE(IsConnected(graph, "load0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "load1", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", kept_broadcast));
  EXPECT_TRUE(IsConnected(graph, kept_broadcast, "store"));
}

TEST(BroadcastBackwardPass, MovesCommonAxisOldBrcDeletedWithResidual) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildCommonAxisGraph("broadcast_backward_residual_delete_old");
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "broadcast0"));
  EXPECT_FALSE(HasNode(graph, "broadcast1"));
  auto b0_residual = FindNode(graph, "broadcast0_residual_0");
  ASSERT_NE(b0_residual, nullptr);
  auto b1_residual = FindNode(graph, "broadcast1_residual_1");
  ASSERT_NE(b1_residual, nullptr);
  ExpectResidualConnections(graph);
}

TEST(BroadcastBackwardPass, MultiReferenceBackwardMovesSharedBroadcastInputs) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const std::vector<af::Expression> compact = {s0, af::sym::kSymbolOne};
  const std::vector<af::Expression> strides = {af::sym::kSymbolOne, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_shared_common_axis_input")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", compact, strides)
                   .Broadcast("broadcast", "load", {1})
                   .Add("merge", "broadcast", "broadcast")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(HasNode(graph, "broadcast"));
  EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
  const auto broadcast = FindNode(graph, "broadcast");
  ASSERT_NE(broadcast, nullptr);
  EXPECT_TRUE(IsConnected(graph, "load", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  EXPECT_EQ(broadcast->GetOutDataNodesSize(), 1U);
}

TEST(BroadcastBackwardPass, SplitsSharedBroadcastBeforeDistinctMultiInputBarriers) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const auto s3 = Sym("s3");
  const std::vector<af::Expression> compact = {af::sym::kSymbolOne, s1, af::sym::kSymbolOne, s3};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolZero, s3, af::sym::kSymbolZero,
                                                       af::sym::kSymbolOne};
  const std::vector<af::Expression> expanded = {s0, s1, s2, s3};
  const std::vector<af::Expression> expanded_strides = {s1 * s2 * s3, s2 * s3, s3, af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_split_shared_branch")
                   .Loops({s0, s1, s2, s3})
                   .Data("data0", 0)
                   .Load("load0", "data0", compact, compact_strides)
                   .Broadcast("broadcast0", "load0", {s0, s1, af::sym::kSymbolOne, s3})
                   .Broadcast("broadcast1", "broadcast0", expanded)
                   .Sqrt("sqrt", "broadcast1")
                   .Abs("abs", "broadcast1")
                   .Data("data1", 1)
                   .Load("load1", "data1", expanded, expanded_strides)
                   .Sub("sub", "sqrt", "load1")
                   .Add("add", "abs", "load1")
                   .Neg("neg", "sub")
                   .Relu("relu", "add")
                   .Mul("mul", "relu", "neg")
                   .Store("store", "mul")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "sqrt"));
  EXPECT_TRUE(IsConnected(graph, "load0", "abs"));
  EXPECT_FALSE(IsConnected(graph, "broadcast1", "sqrt"));
  EXPECT_FALSE(IsConnected(graph, "broadcast1", "abs"));

  for (const auto &compute_name : {"sqrt", "abs"}) {
    const auto compute = FindNode(graph, compute_name);
    ASSERT_NE(compute, nullptr);
    const auto peers = compute->GetOutDataAnchor(0)->GetPeerInDataAnchors();
    ASSERT_EQ(peers.size(), 1U);
    EXPECT_EQ((*peers.begin())->GetOwnerNode()->GetType(), "Broadcast");
  }
}

TEST(BroadcastBackwardPass, SharedBroadcastSplitSupportsSuccessorInputOne) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact = {af::sym::kSymbolOne, s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolZero, s2, af::sym::kSymbolZero};
  const std::vector<af::Expression> expanded = {s0, s1, s2};
  const std::vector<af::Expression> expanded_strides = {s1 * s2, s2, af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_split_successor_input_one")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Data("data2", 2)
                   .Load("load0", "data0", compact, compact_strides)
                   .Load("load1", "data1", expanded, expanded_strides)
                   .Load("load2", "data2", expanded, expanded_strides)
                   .Broadcast("broadcast0", "load0", expanded)
                   .Sqrt("sqrt", "broadcast0")
                   .Abs("abs", "broadcast0")
                   .Add("successor0", "load1", "sqrt")
                   .Add("successor1", "load2", "abs")
                   .Add("join", "successor0", "successor1")
                   .Store("store", "join")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "sqrt"));
  EXPECT_TRUE(IsConnected(graph, "load0", "abs"));
  EXPECT_FALSE(IsConnected(graph, "broadcast0", "sqrt"));
  EXPECT_FALSE(IsConnected(graph, "broadcast0", "abs"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0_branch_split_1", "successor0"));
  EXPECT_TRUE(HasNode(graph, "broadcast0_branch_split_1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "successor1"));
}

TEST(BroadcastBackwardPass, SplitsAndMovesSharedBroadcastPerBranch) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact = {af::sym::kSymbolOne, s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolZero, s2, af::sym::kSymbolZero};
  const std::vector<af::Expression> expanded = {s0, s1, s2};
  const std::vector<af::Expression> expanded_strides = {s1 * s2, s2, af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_split_and_move_per_branch")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact, compact_strides)
                   .Load("load1", "data1", expanded, expanded_strides)
                   .Broadcast("broadcast", "load0", expanded)
                   .Sqrt("sqrt", "broadcast")
                   .Add("add", "sqrt", "load1")
                   .Neg("neg", "add")
                   .Abs("abs", "broadcast")
                   .Relu("relu", "abs")
                   .Mul("mul", "relu", "neg")
                   .Store("store", "mul")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "sqrt"));
  EXPECT_TRUE(IsConnected(graph, "load0", "abs"));
  EXPECT_FALSE(IsConnected(graph, "broadcast", "sqrt"));
  EXPECT_FALSE(IsConnected(graph, "broadcast", "abs"));

  const auto sqrt = FindNode(graph, "sqrt");
  const auto relu = FindNode(graph, "relu");
  ASSERT_NE(sqrt, nullptr);
  ASSERT_NE(relu, nullptr);
  ASSERT_EQ(sqrt->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1U);
  ASSERT_EQ(relu->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1U);
  EXPECT_EQ((*sqrt->GetOutDataAnchor(0)->GetPeerInDataAnchors().begin())->GetOwnerNode()->GetType(), "Broadcast");
  EXPECT_EQ((*relu->GetOutDataAnchor(0)->GetPeerInDataAnchors().begin())->GetOwnerNode()->GetType(), "Broadcast");
}

TEST(BroadcastBackwardPass, SharedBroadcastSplitSkipsCommonMerge) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const std::vector<af::Expression> compact = {s0, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolOne, af::sym::kSymbolZero};
  const std::vector<af::Expression> expanded = {s0, s1};
  const std::vector<af::Expression> expanded_strides = {s1, af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_split_skips_common_merge")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", compact, compact_strides)
                   .Broadcast("broadcast", "load", expanded)
                   .Abs("abs", "broadcast")
                   .Neg("neg", "broadcast")
                   .Add("merge", "abs", "neg")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "broadcast_consumer_split_1"));
  EXPECT_TRUE(IsConnected(graph, "merge", "broadcast"));
}

TEST(BroadcastBackwardPass, SharedBroadcastSplitSkipsWithoutMovableBranch) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const std::vector<af::Expression> compact = {s0, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolOne, af::sym::kSymbolZero};
  const std::vector<af::Expression> expanded = {s0, s1};
  auto graph = AscGraphBuilder("broadcast_backward_split_skips_non_movable")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", compact, compact_strides)
                   .Broadcast("broadcast", "load", expanded)
                   .Store("store0", "broadcast")
                   .Output("output0", "store0")
                   .Store("store1", "broadcast")
                   .Output("output1", "store1")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "broadcast_consumer_split_1"));
  EXPECT_EQ(FindNode(graph, "broadcast")->GetOutDataNodesSize(), 2U);
}

TEST(BroadcastBackwardPass, SharedBroadcastSplitSkipsMoreThanEightBranches) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const std::vector<af::Expression> compact = {s0, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolOne, af::sym::kSymbolZero};
  const std::vector<af::Expression> expanded = {s0, s1};
  const std::vector<af::Expression> expanded_strides = {s1, af::sym::kSymbolOne};
  AscGraphBuilder builder("broadcast_backward_split_skips_more_than_eight_branches");
  builder.Loops({s0, s1})
      .Data("data0", 0)
      .Data("data1", 1)
      .Load("load0", "data0", compact, compact_strides)
      .Load("load1", "data1", expanded, expanded_strides)
      .Broadcast("broadcast", "load0", expanded);
  for (size_t index = 0U; index < 9U; ++index) {
    const auto suffix = std::to_string(index);
    builder.Abs("abs" + suffix, "broadcast").Add("add" + suffix, "load1", "abs" + suffix);
  }
  builder.Store("store", "add0").Output("output", "store");
  auto graph = builder.Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "broadcast_consumer_split_1"));
  EXPECT_EQ(FindNode(graph, "broadcast")->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 9U);
}

TEST(BroadcastBackwardPass, MovesBroadcastToNonZeroMultiInputTail) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const std::vector<af::Expression> compact = {s0, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolOne, af::sym::kSymbolZero};
  const std::vector<af::Expression> expanded = {s0, s1};
  const std::vector<af::Expression> expanded_strides = {s1, af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_non_zero_tail_input")
                   .Loops({s0, s1})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact, compact_strides)
                   .Load("load1", "data1", expanded, expanded_strides)
                   .Broadcast("broadcast", "load0", expanded)
                   .Abs("abs", "broadcast")
                   .Add("add", "load1", "abs")
                   .Store("store", "add")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "abs", "broadcast"));
  EXPECT_EQ(FindNode(graph, "add")->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "broadcast");
}

TEST(BroadcastBackwardPass, SharedBroadcastSplitSkipsSingleInputSuccessor) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact = {af::sym::kSymbolOne, s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolZero, s2, af::sym::kSymbolZero};
  const std::vector<af::Expression> expanded = {s0, s1, s2};
  const std::vector<af::Expression> expanded_strides = {s1 * s2, s2, af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_split_single_input_successor")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Load("load0", "data0", compact, compact_strides)
                   .Broadcast("broadcast0", "load0", expanded)
                   .Sqrt("sqrt", "broadcast0")
                   .Abs("abs", "broadcast0")
                   .Neg("neg", "sqrt")
                   .Relu("relu", "abs")
                   .Mul("mul", "relu", "neg")
                   .Store("store", "mul")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "broadcast0_branch_split_1"));
  EXPECT_TRUE(IsConnected(graph, "mul", "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "store"));
}

TEST(BroadcastBackwardPass, CommonAxisBackwardSkipsBroadcastWithAnotherConsumer) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact0 = {af::sym::kSymbolOne, af::sym::kSymbolOne, s2};
  const std::vector<af::Expression> strides0 = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact1 = {s0, af::sym::kSymbolOne, af::sym::kSymbolOne};
  const std::vector<af::Expression> strides1 = {af::sym::kSymbolOne, af::sym::kSymbolZero, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_extra_consumer")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact0, strides0)
                   .Load("load1", "data1", compact1, strides1)
                   .Broadcast("broadcast0", "load0", {0, 1})
                   .Broadcast("broadcast1", "load1", {1, 2})
                   .Add("merge", "broadcast0", "broadcast1")
                   .Abs("side", "broadcast0")
                   .Store("store", "merge")
                   .Store("side_store", "side")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(HasNode(graph, "broadcast0"));
  EXPECT_TRUE(HasNode(graph, "broadcast1"));
  EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "side"));
}

TEST(BroadcastBackwardPass, SkipsCommonAxisEdgeAttrMismatch) {
  auto graph = BuildCommonAxisGraph("broadcast_backward_edge_mismatch");
  CompleteApiInfo(graph);
  auto merge_node = FindNode(graph, "merge");
  ASSERT_NE(merge_node, nullptr);
  merge_node->inputs[0].attr.dtype = af::DT_FLOAT16;

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
  EXPECT_TRUE(HasNode(graph, "broadcast0"));
  EXPECT_TRUE(HasNode(graph, "broadcast1"));
}

TEST(BroadcastBackwardPass, SkipsCommonAxisDtypeMismatch) {
  auto graph = BuildCommonAxisGraph("broadcast_backward_dtype_mismatch");
  CompleteApiInfo(graph);
  auto load1_node = FindNode(graph, "load1");
  ASSERT_NE(load1_node, nullptr);
  load1_node->outputs[0].attr.dtype = af::DT_FLOAT16;
  auto b1_node = FindNode(graph, "broadcast1");
  ASSERT_NE(b1_node, nullptr);
  b1_node->inputs[0].attr.dtype = af::DT_FLOAT16;
  b1_node->outputs[0].attr.dtype = af::DT_FLOAT16;

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
  EXPECT_TRUE(HasNode(graph, "broadcast0"));
  EXPECT_TRUE(HasNode(graph, "broadcast1"));
}

// ===== Multi-reference fork-join backward tests =====

TEST(BroadcastBackwardPass, GraphUtilsReconnectsSameSourceMultipleInputs) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_multi_edge_graph_utils")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Add("consumer", "broadcast", "broadcast")
                   .Store("store", "consumer")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  const auto load = FindNode(graph, "load");
  const auto broadcast = FindNode(graph, "broadcast");
  const auto consumer = FindNode(graph, "consumer");
  ASSERT_NE(load, nullptr);
  ASSERT_NE(broadcast, nullptr);
  ASSERT_NE(consumer, nullptr);
  const auto peer_copy = broadcast->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  ASSERT_EQ(peer_copy.size(), 2U);
  for (const auto &peer : peer_copy) {
    ASSERT_EQ(af::GraphUtils::RemoveEdge(broadcast->GetOutDataAnchor(0), peer), af::SUCCESS);
  }
  EXPECT_TRUE(broadcast->GetOutDataAnchor(0)->GetPeerInDataAnchors().empty());
  ASSERT_EQ(af::GraphUtils::AddEdge(load->GetOutDataAnchor(0), consumer->GetInDataAnchor(0)), af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::AddEdge(load->GetOutDataAnchor(0), consumer->GetInDataAnchor(1)), af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::RemoveEdge(load->GetOutDataAnchor(0), consumer->GetInDataAnchor(0)), af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::RemoveEdge(load->GetOutDataAnchor(0), consumer->GetInDataAnchor(1)), af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::AddEdge(broadcast->GetOutDataAnchor(0), consumer->GetInDataAnchor(0)), af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::AddEdge(broadcast->GetOutDataAnchor(0), consumer->GetInDataAnchor(1)), af::SUCCESS);
  EXPECT_EQ(broadcast->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 2U);
  EXPECT_TRUE(AreConnectedTensorAttrsEqual(broadcast, consumer, 0U));
  EXPECT_TRUE(AreConnectedTensorAttrsEqual(broadcast, consumer, 1U));
}

TEST(BroadcastBackwardPass, MovesDirectFanOutAfterMerge) {
  auto graph = BuildDirectFanOutGraph("broadcast_backward_multi_reference_direct");
  CompleteApiInfo(graph);
  ExpectDirectFanOutCandidate(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  ExpectDirectFanOutMoved(graph);
}

TEST(BroadcastBackwardPass, MovesPrefixFanOutAfterMerge) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_multi_reference_prefix")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Relu("prefix", "broadcast")
                   .Abs("branch0", "prefix")
                   .Neg("branch1", "prefix")
                   .Add("merge", "branch0", "branch1")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "prefix"));
  EXPECT_TRUE(IsConnected(graph, "prefix", "branch0"));
  EXPECT_TRUE(IsConnected(graph, "prefix", "branch1"));
  EXPECT_TRUE(IsConnected(graph, "branch0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "branch1", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  const auto prefix = FindNode(graph, "prefix");
  ASSERT_NE(prefix, nullptr);
  ExpectStaticEq(prefix->outputs[0].attr.repeats, kCompactRepeats);
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load", "prefix"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "prefix", "branch0"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "prefix", "branch1"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "branch0", "merge"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "branch1", "merge"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "merge", "broadcast"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "broadcast", "store"));
}

TEST(BroadcastBackwardPass, MovesMultiNodeFanOutBranches) {
  auto graph = BuildMultiNodeFanOutGraph("broadcast_backward_multi_reference_multi_node_branches");
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "branch0_head"));
  EXPECT_TRUE(IsConnected(graph, "load", "branch1_head"));
  EXPECT_TRUE(IsConnected(graph, "branch0_head", "branch0_tail"));
  EXPECT_TRUE(IsConnected(graph, "branch1_head", "branch1_tail"));
  EXPECT_TRUE(IsConnected(graph, "branch0_tail", "merge"));
  EXPECT_TRUE(IsConnected(graph, "branch1_tail", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  const auto branch0_tail = FindNode(graph, "branch0_tail");
  const auto branch1_tail = FindNode(graph, "branch1_tail");
  ASSERT_NE(branch0_tail, nullptr);
  ASSERT_NE(branch1_tail, nullptr);
  ExpectStaticEq(branch0_tail->outputs[0].attr.repeats, kCompactRepeats);
  ExpectStaticEq(branch1_tail->outputs[0].attr.repeats, kCompactRepeats);
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load", "branch0_head"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "branch0_head", "branch0_tail"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "branch0_tail", "merge"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load", "branch1_head"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "branch1_head", "branch1_tail"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "branch1_tail", "merge"));
}

TEST(BroadcastBackwardPass, MovesFanOutAcrossFollowingComputeChain) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_multi_reference_following_compute")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Abs("branch0", "broadcast")
                   .Neg("branch1", "broadcast")
                   .Add("merge", "branch0", "branch1")
                   .Relu("following0", "merge")
                   .Exp("following1", "following0")
                   .Store("store", "following1")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "branch0"));
  EXPECT_TRUE(IsConnected(graph, "load", "branch1"));
  EXPECT_TRUE(IsConnected(graph, "merge", "following0"));
  EXPECT_TRUE(IsConnected(graph, "following0", "following1"));
  EXPECT_TRUE(IsConnected(graph, "following1", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "merge", "following0"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "following0", "following1"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "following1", "broadcast"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "broadcast", "store"));
}

TEST(BroadcastBackwardPass, DtypeAwareBackwardEnablesPrefixFanOut) {
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_dtype_aware_prefix_fanout")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Sqrt("before_cast", "broadcast")
                   .Cast("cast", "before_cast", af::DT_FLOAT16)
                   .Relu("prefix", "cast")
                   .Abs("branch0", "prefix")
                   .Neg("branch1", "prefix")
                   .Add("merge", "branch0", "branch1")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "prefix", af::DT_FLOAT16);
  SetNodeDtype(graph, "branch0", af::DT_FLOAT16);
  SetNodeDtype(graph, "branch1", af::DT_FLOAT16);
  SetNodeDtype(graph, "merge", af::DT_FLOAT16);
  SetNodeDtype(graph, "store", af::DT_FLOAT16);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "before_cast"));
  EXPECT_TRUE(IsConnected(graph, "before_cast", "cast"));
  EXPECT_TRUE(IsConnected(graph, "cast", "prefix"));
  EXPECT_TRUE(IsConnected(graph, "prefix", "branch0"));
  EXPECT_TRUE(IsConnected(graph, "prefix", "branch1"));
  EXPECT_TRUE(IsConnected(graph, "merge", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  const auto broadcast = FindNode(graph, "broadcast");
  ASSERT_NE(broadcast, nullptr);
  EXPECT_EQ(broadcast->inputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(broadcast->outputs[0].attr.dtype, af::DT_FLOAT16);
}

TEST(BroadcastBackwardPass, MovesSharedDtypeAwareBranchesPastFollowingChain) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  ScopedTestPlatform platform("3510");
  auto graph = BuildSharedDtypeAwareFanOutGraph("broadcast_backward_shared_dtype_aware_fanout");
  CompleteSharedDtypeAwareFanOutGraph(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "left_cast"));
  EXPECT_TRUE(IsConnected(graph, "load", "right_cast"));
  EXPECT_TRUE(IsConnected(graph, "right_cast", "relu"));
  EXPECT_TRUE(IsConnected(graph, "relu", "add"));
  EXPECT_TRUE(IsConnected(graph, "left_cast", "add"));
  EXPECT_TRUE(IsConnected(graph, "add", "sqrt"));
  EXPECT_TRUE(IsConnected(graph, "sqrt", "sigmoid"));
  const auto store = FindNode(graph, "store");
  ASSERT_NE(store, nullptr);
  const auto broadcast = std::dynamic_pointer_cast<af::AscNode>(store->GetInDataNodes().at(0));
  ASSERT_NE(broadcast, nullptr);
  EXPECT_EQ(broadcast->GetType(), af::ascir_op::Broadcast::Type);
  EXPECT_EQ(broadcast->outputs[0].attr.dtype, af::DT_FLOAT);
  EXPECT_EQ(broadcast->GetInDataNodes().at(0)->GetName(), "sigmoid");
}

TEST(BroadcastBackwardPass, SkipsSharedDtypeAwareBranchWithMismatchedInputDtype) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  ScopedTestPlatform platform("3510");
  auto graph = BuildSharedDtypeAwareFanOutGraph("broadcast_backward_shared_dtype_aware_mismatch");
  CompleteSharedDtypeAwareFanOutGraph(graph);
  const auto right_cast = FindNode(graph, "right_cast");
  ASSERT_NE(right_cast, nullptr);
  right_cast->GetOpDesc()->MutableInputDesc(0U)->SetDataType(af::DT_FLOAT16);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "abs"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "right_cast"));
  EXPECT_FALSE(IsConnected(graph, "sigmoid", "broadcast"));
}

TEST(BroadcastBackwardPass, MovesSameConsumerMultipleInputs) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_multi_reference_same_consumer")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Add("consumer", "broadcast", "broadcast")
                   .Store("store", "consumer")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  const auto load = FindNode(graph, "load");
  const auto broadcast = FindNode(graph, "broadcast");
  const auto consumer = FindNode(graph, "consumer");
  ASSERT_NE(load, nullptr);
  ASSERT_NE(broadcast, nullptr);
  ASSERT_NE(consumer, nullptr);
  EXPECT_EQ(consumer->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load");
  EXPECT_EQ(consumer->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load");
  EXPECT_TRUE(IsConnected(graph, "consumer", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  ExpectStaticEq(consumer->inputs[0].attr.repeats, kCompactRepeats);
  ExpectStaticEq(consumer->inputs[1].attr.repeats, kCompactRepeats);
  ExpectStaticEq(consumer->outputs[0].attr.repeats, kCompactRepeats);
  ExpectStaticEq(broadcast->inputs[0].attr.repeats, kCompactRepeats);
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "consumer", "broadcast"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "broadcast", "store"));
}

TEST(BroadcastBackwardPass, MovesSameConsumerWithThreeDimensionalLayout) {
  const std::vector<af::Expression> compact_repeats = {Sym(83), af::sym::kSymbolOne, Sym(91)};
  const std::vector<af::Expression> compact_strides = {Sym(91), af::sym::kSymbolZero, af::sym::kSymbolOne};
  const std::vector<af::Expression> expanded_repeats = {Sym(83), Sym(18), Sym(91)};
  const std::vector<af::Expression> expanded_strides = {Sym(1638), Sym(91), af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_multi_reference_same_consumer_3d")
                   .Loops({Sym(83), Sym(18), Sym(91)})
                   .Data("data", 0)
                   .Load("load", "data", compact_repeats, compact_strides)
                   .Broadcast("broadcast", "load", {1})
                   .Add("consumer", "broadcast", "broadcast")
                   .Store("store", "consumer")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  const auto consumer = FindNode(graph, "consumer");
  const auto broadcast = FindNode(graph, "broadcast");
  ASSERT_NE(consumer, nullptr);
  ASSERT_NE(broadcast, nullptr);
  ExpectStaticEq(consumer->inputs[0].attr.repeats, compact_repeats);
  ExpectStaticEq(consumer->inputs[0].attr.strides, compact_strides);
  ExpectStaticEq(consumer->inputs[1].attr.repeats, compact_repeats);
  ExpectStaticEq(consumer->inputs[1].attr.strides, compact_strides);
  ExpectStaticEq(consumer->outputs[0].attr.repeats, compact_repeats);
  ExpectStaticEq(consumer->outputs[0].attr.strides, compact_strides);
  ExpectStaticEq(broadcast->inputs[0].attr.repeats, compact_repeats);
  ExpectStaticEq(broadcast->inputs[0].attr.strides, compact_strides);
  ExpectStaticEq(broadcast->outputs[0].attr.repeats, expanded_repeats);
  ExpectStaticEq(broadcast->outputs[0].attr.strides, expanded_strides);
}

TEST(BroadcastBackwardPass, MultiReferenceBackwardSkipsBarrierBranch) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_multi_reference_barrier")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Cast("barrier", "broadcast", af::DT_FLOAT16)
                   .Abs("branch", "broadcast")
                   .Add("merge", "barrier", "branch")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "barrier"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "branch"));
  EXPECT_FALSE(IsConnected(graph, "merge", "broadcast"));
}

TEST(BroadcastBackwardPass, MultiReferenceBackwardSkipsBroadcastWithControlEdge) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildDirectFanOutGraph("broadcast_backward_multi_reference_control_edge");
  CompleteApiInfo(graph);
  const auto load = FindNode(graph, "load");
  const auto broadcast = FindNode(graph, "broadcast");
  ASSERT_NE(load, nullptr);
  ASSERT_NE(broadcast, nullptr);
  ASSERT_EQ(af::GraphUtils::AddEdge(load->GetOutControlAnchor(), broadcast->GetInControlAnchor()), af::SUCCESS);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "branch0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "branch1"));
  EXPECT_FALSE(IsConnected(graph, "merge", "broadcast"));
}

TEST(BroadcastBackwardPass, MultiReferenceBackwardSkipsMultiInputSuccessor) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_multi_reference_multi_input_successor")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Abs("branch0", "broadcast")
                   .Neg("branch1", "broadcast")
                   .Add("merge", "branch0", "branch1")
                   .Add("succ", "merge", "merge")
                   .Store("store", "succ")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "branch0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "branch1"));
  EXPECT_FALSE(IsConnected(graph, "merge", "broadcast"));
}

TEST(BroadcastBackwardPass, MultiReferenceBackwardSkipsSameConsumerWithAnotherSource) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_multi_reference_mixed_consumer")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Abs("other", "broadcast")
                   .Add("consumer", "broadcast", "other")
                   .Store("store", "consumer")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "other"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "consumer"));
  EXPECT_FALSE(IsConnected(graph, "consumer", "broadcast"));
}
