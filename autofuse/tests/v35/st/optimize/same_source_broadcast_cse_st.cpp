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
#include "ascgraph_info_complete.h"
#define private public
#include "optimize.h"
#undef private
#include "platform_context.h"
#include "runtime_stub.h"
#include "schedule_utils.h"

namespace optimize {
namespace {
using af::testing::AscGraphBuilder;

class SameSourceBroadcastCseStTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    ge::RuntimeStub::SetInstance(std::make_shared<af::RuntimeStubV2>());
  }

  void TearDown() override {
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }

  Optimizer optimizer{OptimizerOptions{}};
};

af::AscGraph BuildNormGraphWithEquivalentBroadcasts() {
  const auto rows = af::Symbol(16);
  const auto columns = af::Symbol(128);
  return AscGraphBuilder("same_source_broadcast_cse_st")
      .Loops({rows, columns})
      .Data("data", 0, {rows, columns}, {columns, af::sym::kSymbolOne}, af::DT_FLOAT)
      .Load("load", "data")
      .Sum("reduce", "load", {1})
      .Broadcast("broadcast0", "reduce", {1})
      .Broadcast("broadcast1", "reduce", {1})
      .Add("add", "broadcast0", "broadcast1")
      .Store("store", "add")
      .Output("output", "store", 0, af::DT_FLOAT)
      .Build();
}

TEST_F(SameSourceBroadcastCseStTest, MergesEquivalentBroadcastsThroughGraphPassRunner) {
  auto graph = BuildNormGraphWithEquivalentBroadcasts();
  ASSERT_EQ(AscGraphInfoComplete::CompleteApiInfo(graph), af::SUCCESS);
  ASSERT_TRUE(ScheduleUtils::IsNormStruct(graph));
  const auto reduce = graph.FindNode("reduce");
  ASSERT_NE(reduce, nullptr);
  ASSERT_EQ(reduce->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 2UL);

  ASSERT_EQ(optimizer.GraphPass(graph), af::SUCCESS);

  const auto add = graph.FindNode("add");
  ASSERT_NE(add, nullptr);
  EXPECT_EQ(graph.FindNode("broadcast1"), nullptr);

  const auto input0_peer = add->GetInDataAnchor(0)->GetPeerOutAnchor();
  const auto input1_peer = add->GetInDataAnchor(1)->GetPeerOutAnchor();
  ASSERT_NE(input0_peer, nullptr);
  ASSERT_NE(input1_peer, nullptr);
  EXPECT_EQ(input0_peer, input1_peer);
  ASSERT_NE(input0_peer->GetOwnerNode(), nullptr);
  EXPECT_EQ(input0_peer->GetOwnerNode()->GetName(), "reduce");
  EXPECT_EQ(input1_peer->GetOwnerNode()->GetName(), "reduce");
  EXPECT_EQ(reduce->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 2UL);
}

TEST_F(SameSourceBroadcastCseStTest, SkipsGraphWithoutNormStructureThroughGraphPassRunner) {
  auto graph = AscGraphBuilder("non_norm_graph")
                   .Loops({16, 128})
                   .Data("data", 0, af::DT_FLOAT)
                   .Load("load", "data")
                   .Abs("abs", "load")
                   .Store("store", "abs")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  ASSERT_EQ(AscGraphInfoComplete::CompleteApiInfo(graph), af::SUCCESS);
  ASSERT_FALSE(ScheduleUtils::IsNormStruct(graph));

  ASSERT_EQ(optimizer.GraphPass(graph), af::SUCCESS);
  EXPECT_NE(graph.FindNode("abs"), nullptr);
}
}  // namespace
}  // namespace optimize
