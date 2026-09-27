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
#include "optimize/graph_pass/redundant_broadcast_eliminate_pass.h"
#include "tests/framework/broadcast_backward/broadcast_backward_ut_utils.h"

namespace {
using af::AscGraph;
using af::testing::AscGraphBuilder;
using af::testing::Sym;
using broadcast_backward_test::FindNode;
using broadcast_backward_test::HasNode;
using broadcast_backward_test::IsConnected;

// builder 构造的 Broadcast 未显式设置输出 strides（默认为空），而前驱 Load 有 strides，
// 完整视图比较会因此判不等。这里把 broadcast 输出 strides 同步为前驱输出 strides，
// 构造"各轴 repeat 一致且完整视图一致"的真实冗余 Broadcast。
void MakeBroadcastOutputViewSameAsPredecessor(af::AscGraph &graph, const char *bro_name, const char *pre_name) {
  const auto bro_node = FindNode(graph, bro_name);
  const auto pre_node = FindNode(graph, pre_name);
  ASSERT_NE(bro_node, nullptr);
  ASSERT_NE(pre_node, nullptr);
  bro_node->outputs[0].attr.strides = pre_node->outputs[0].attr.strides;
}
}  // namespace

TEST(RedundantBroadcastEliminatePass, RemovesBroadcastWithSameOutputRepeats) {
  auto graph = AscGraphBuilder("redundant_broadcast_removal")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data", 0)
                   .Load("load", "data", {Sym("s0"), Sym("s1")}, {af::sym::kSymbolOne, af::sym::kSymbolOne})
                   .Broadcast("broadcast", "load", {Sym("s0"), Sym("s1")})
                   .Store("store", "broadcast")
                   .Output("output", "store")
                   .Build();

  MakeBroadcastOutputViewSameAsPredecessor(graph, "broadcast", "load");

  optimize::RedundantBroadcastEliminatePass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "load", "store"));
}

TEST(RedundantBroadcastEliminatePass, KeepsScalarLikePredecessorBroadcast) {
  // scalar -> broadcast -> sqrt：broadcast 输出 repeats 全 1。scalar 是 scalar-like 生产者，
  // 其 Broadcast 承担"标量接上有效输入视图"的职责，即使 repeats 全 1 也不应被剔除。
  auto graph = AscGraphBuilder("scalar_broadcast_sqrt")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Scalar("scalar", "1", af::DT_FLOAT)
                   .Broadcast("broadcast", "scalar", {Sym(1), Sym(1)})
                   .Sqrt("sqrt", "broadcast")
                   .Store("store", "sqrt")
                   .Output("output", "store")
                   .Build();

  optimize::RedundantBroadcastEliminatePass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(HasNode(graph, "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "scalar", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "sqrt"));
}

TEST(RedundantBroadcastEliminatePass, KeepsBroadcastWithExpandedRepeats) {
  auto graph = AscGraphBuilder("non_redundant_broadcast")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data", 0)
                   .Load("load", "data", {Sym("s0"), Sym("s1")}, {af::sym::kSymbolOne, af::sym::kSymbolOne})
                   .Broadcast("broadcast", "load", {Sym("s0"), Sym("s2")})
                   .Store("store", "broadcast")
                   .Output("output", "store")
                   .Build();

  MakeBroadcastOutputViewSameAsPredecessor(graph, "broadcast", "load");

  optimize::RedundantBroadcastEliminatePass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(HasNode(graph, "broadcast"));
  EXPECT_FALSE(IsConnected(graph, "load", "store"));
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
}

TEST(RedundantBroadcastEliminatePass, RelinksAllConsumersToPredecessor) {
  auto graph = AscGraphBuilder("redundant_broadcast_multi_consumer")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data", 0)
                   .Load("load", "data", {Sym("s0"), Sym("s1")}, {af::sym::kSymbolOne, af::sym::kSymbolOne})
                   .Broadcast("broadcast", "load", {Sym("s0"), Sym("s1")})
                   .Store("store0", "broadcast")
                   .Store("store1", "broadcast")
                   .Output("output0", "store0")
                   .Output("output1", "store1")
                   .Build();

  MakeBroadcastOutputViewSameAsPredecessor(graph, "broadcast", "load");

  optimize::RedundantBroadcastEliminatePass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "load", "store0"));
  EXPECT_TRUE(IsConnected(graph, "load", "store1"));
}
