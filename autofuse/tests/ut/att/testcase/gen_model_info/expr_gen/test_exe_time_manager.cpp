/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <limits>
#include "gtest/gtest.h"
#include "test_brc_buf_graph.h"
#include "parser/ascend_graph_parser.h"
#include "expr_gen/arg_list_manager.h"
#include "expr_gen/exe_time_pass.h"
#include "expr_gen/cache_guard_count.h"
#include "graph_construct_utils.h"

namespace att {
namespace {
Expr Symbol(const char *name) {
  return af::Expression::Parse(name);
}

CacheGuardInfo MakeOriginGuard(const Expr &guard_axis, const Expr &loop_extent) {
  return {CacheGuardKind::kOriginBroadcast, guard_axis, Expr(), Expr(), loop_extent, true, true};
}

CacheGuardInfo MakeFusedGuard(const Expr &guard_axis, const Expr &period, const Expr &block_inner_extent,
                              const Expr &loop_extent) {
  return {CacheGuardKind::kFusedBroadcast, guard_axis, period, block_inner_extent, loop_extent, true, true};
}

uint64_t EvaluateConst(const Expr &expr) {
  uint64_t value = 0;
  EXPECT_TRUE(expr.GetConstValue(value));
  return value;
}

TuningSpacePtr MakeGuardTuningSpace(const Expr &period, const Expr &inner_extent, const Expr &block_dim) {
  auto tuning_space = std::make_shared<TuningSpace>();
  auto origin = std::make_unique<SubAxis>();
  origin->id = 1;
  origin->name = "guarded";
  origin->repeat = inner_extent;
  auto outer = std::make_unique<SubAxis>();
  outer->id = 2;
  outer->name = "outer";
  outer->axis_type = AxisPosition::OUTER;
  outer->repeat = period;
  auto inner = std::make_unique<SubAxis>();
  inner->id = 3;
  inner->name = "block_inner";
  inner->axis_type = AxisPosition::INNER;
  inner->is_bind_multi_core = true;
  inner->repeat = inner_extent;
  inner->orig_axis = {origin.get()};
  // Parser split outputs reference their common source axis through parent_axis.
  inner->parent_axis = {origin.get()};
  outer->parent_axis = {origin.get()};
  auto block = std::make_unique<SubAxis>();
  block->id = 4;
  block->name = "block";
  block->axis_type = AxisPosition::OUTER;
  block->repeat = block_dim;
  tuning_space->sub_axes.emplace_back(std::move(origin));
  tuning_space->sub_axes.emplace_back(std::move(outer));
  tuning_space->sub_axes.emplace_back(std::move(inner));
  tuning_space->sub_axes.emplace_back(std::move(block));
  tuning_space->block_dims = {{tuning_space->sub_axes[3].get()}};
  NodeInfo node;
  node.name = "cast0";
  node.exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  node.loop_axes = {tuning_space->sub_axes[2].get()};
  tuning_space->node_infos.emplace_back(node);
  return tuning_space;
}
}  // namespace

TEST(CacheGuardInfoTest, OriginGuardCountsOneHit) {
  CacheGuardInfo info = MakeOriginGuard(Symbol("i"), Symbol("19"));
  EXPECT_EQ(EvaluateConst(CountGuardHits(info, Symbol("0"))), 1);
}

TEST(CacheGuardInfoTest, FusedGuardCountsPeriodHitsPerBlock) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("4"), Symbol("2"), Symbol("6"));
  EXPECT_EQ(EvaluateConst(CountGuardHits(info, Symbol("0"))), 1);
  EXPECT_EQ(EvaluateConst(CountGuardHits(info, Symbol("1"))), 1);
}

TEST(CacheGuardInfoTest, FusedGuardReturnsZeroBeyondPeriod) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("4"), Symbol("2"), Symbol("2"));
  EXPECT_EQ(EvaluateConst(CountGuardHits(info, Symbol("3"))), 0);
}

TEST(CacheGuardInfoTest, InvalidPeriodDoesNotBuildModulo) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("0"), Symbol("2"), Symbol("6"));
  EXPECT_FALSE(ValidateCacheGuardInfo(info));
}

TEST(CacheGuardInfoTest, BlockDependentCountFallsBackUnlessUniform) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("5"), Symbol("2"), Symbol("6"));
  EXPECT_FALSE(IsBlockCountUniform(info, Symbol("54")));
}

TEST(CacheGuardInfoTest, SymbolicCanonicalBlockSplitProvesUniformity) {
  const auto period = Symbol("period");
  const auto inner = Symbol("inner");
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), period, inner, inner);
  const auto block_dim = af::sym::Ceiling(af::sym::Div(period, inner));
  EXPECT_TRUE(IsBlockCountUniform(info, block_dim));
}

TEST(CacheGuardInfoTest, ArbitrarySymbolicBlockDimFallsBack) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("period"), Symbol("inner"), Symbol("inner"), Symbol("inner"));
  EXPECT_FALSE(IsBlockCountUniform(info, Symbol("block_dim")));
}

TEST(CacheGuardInfoTest, UnitExtentTailMakesBlockCountNonUniform) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("5"), Symbol("2"), Symbol("1"));
  EXPECT_FALSE(IsBlockCountUniform(info, Symbol("4")));
}

TEST(CacheGuardInfoTest, UnitPeriodTailMakesBlockCountNonUniform) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("1"), Symbol("2"), Symbol("2"));
  EXPECT_FALSE(IsBlockCountUniform(info, Symbol("2")));
}

TEST(CacheGuardInfoTest, UnitPeriodUnitExtentCountsSingleHit) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("1"), Symbol("2"), Symbol("1"));
  EXPECT_EQ(EvaluateConst(CountGuardHits(info, Symbol("0"))), 1);
}

TEST(CacheGuardInfoTest, RejectsNonPositiveLoopExtent) {
  EXPECT_FALSE(ValidateCacheGuardInfo(MakeOriginGuard(Symbol("i"), Symbol("0"))));
  EXPECT_FALSE(ValidateCacheGuardInfo(MakeOriginGuard(Symbol("i"), Symbol("-1"))));
}

TEST(CacheGuardInfoTest, AllowsMixedSymbolicAndConstantMetadata) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("period"), Symbol("2"), Symbol("extent"));
  EXPECT_TRUE(ValidateCacheGuardInfo(info));
}

TEST(CacheGuardInfoTest, RejectsUnprovenSymbolicMetadata) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("period"), Symbol("2"), Symbol("extent"));
  info.positive_range_proven = false;
  EXPECT_FALSE(ValidateCacheGuardInfo(info));
}

TEST(CacheGuardInfoTest, RejectsNonIntegerConstantMetadata) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("2.5"), Symbol("2"), Symbol("6"));
  EXPECT_FALSE(ValidateCacheGuardInfo(info));
}

TEST(CacheGuardInfoTest, NoneGuardSafelyReturnsZero) {
  CacheGuardInfo info;
  EXPECT_EQ(EvaluateConst(CountGuardHits(info, Symbol("0"))), 0);
}

TEST(CacheGuardInfoTest, OverflowingBlockRangeIsNotUniform) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), CreateExpr(1U),
                                       CreateExpr(std::numeric_limits<uint64_t>::max() / 2U), CreateExpr(2U));
  EXPECT_FALSE(IsBlockCountUniform(info, CreateExpr(3U)));
}

TEST(CacheGuardInfoTest, LargeLoopExtentUsesConstantTimeCounting) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), CreateExpr(4U), CreateExpr(2U), CreateExpr(1000000000000UL));
  EXPECT_EQ(EvaluateConst(CountGuardHits(info, CreateExpr(0U))), 1);
}

TEST(CacheGuardInfoTest, RejectsRealExtentBeyondUint64) {
  EXPECT_FALSE(ValidateCacheGuardInfo(MakeOriginGuard(Symbol("i"), Symbol("18446744073709551616.0"))));
}

TEST(CacheGuardInfoTest, ReconstructsFusedGuardMetadataFromBlockAxes) {
  auto tuning_space = std::make_shared<TuningSpace>();
  auto origin = std::make_unique<SubAxis>();
  origin->id = 1;
  origin->name = "guarded";
  origin->repeat = Symbol("19");
  auto outer = std::make_unique<SubAxis>();
  outer->id = 2;
  outer->name = "outer";
  outer->axis_type = AxisPosition::OUTER;
  outer->repeat = Symbol("19");
  auto inner = std::make_unique<SubAxis>();
  inner->id = 3;
  inner->name = "block_inner";
  inner->axis_type = AxisPosition::INNER;
  inner->is_bind_multi_core = true;
  inner->repeat = Symbol("19");
  inner->orig_axis = {origin.get()};
  inner->parent_axis = {origin.get()};
  outer->parent_axis = {origin.get()};
  tuning_space->sub_axes.emplace_back(std::move(origin));
  tuning_space->sub_axes.emplace_back(std::move(outer));
  tuning_space->sub_axes.emplace_back(std::move(inner));

  NodeInfo node;
  node.name = "fused";
  node.exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  node.loop_axes = {tuning_space->sub_axes[2].get()};
  ExeTimePassManager manager(tuning_space);
  CacheGuardInfo info;
  ASSERT_TRUE(manager.TryBuildCacheGuardInfo(node, info));
  EXPECT_EQ(info.kind, CacheGuardKind::kFusedBroadcast);
  EXPECT_EQ(Str(info.period), "19");
  EXPECT_EQ(Str(info.block_inner_extent), "19");
  EXPECT_EQ(Str(info.loop_extent), "19");
  EXPECT_TRUE(info.positive_range_proven);

  // The block-outer sibling shares the source parent but is not the guard period.
  auto block_outer = std::make_unique<SubAxis>();
  block_outer->id = 4;
  block_outer->name = "block_outer";
  block_outer->axis_type = AxisPosition::OUTER;
  block_outer->is_bind_multi_core = true;
  block_outer->repeat = Symbol("54");
  block_outer->parent_axis = {tuning_space->sub_axes[0].get()};
  tuning_space->sub_axes.emplace_back(std::move(block_outer));
  ASSERT_TRUE(manager.TryBuildCacheGuardInfo(node, info));
  EXPECT_EQ(Str(info.period), "19");

  auto second_outer = std::make_unique<SubAxis>();
  second_outer->id = 5;
  second_outer->name = "outer_2";
  second_outer->axis_type = AxisPosition::OUTER;
  second_outer->repeat = Symbol("19");
  second_outer->parent_axis = {tuning_space->sub_axes[0].get()};
  tuning_space->sub_axes.emplace_back(std::move(second_outer));
  EXPECT_FALSE(manager.TryBuildCacheGuardInfo(node, info));
}

TEST(CacheGuardInfoTest, ReconstructsOriginGuardAndRejectsUnsafeAxes) {
  auto tuning_space = std::make_shared<TuningSpace>();
  auto inner = std::make_unique<SubAxis>();
  inner->id = 1;
  inner->name = "block_inner";
  inner->axis_type = AxisPosition::INNER;
  inner->is_bind_multi_core = true;
  inner->repeat = Symbol("19");
  tuning_space->sub_axes.emplace_back(std::move(inner));
  NodeInfo node;
  node.name = "origin";
  node.exec_condition = af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
  node.loop_axes = {tuning_space->sub_axes[0].get()};
  ExeTimePassManager manager(tuning_space);
  CacheGuardInfo info;
  ASSERT_TRUE(manager.TryBuildCacheGuardInfo(node, info));
  EXPECT_EQ(info.kind, CacheGuardKind::kOriginBroadcast);
  EXPECT_EQ(Str(info.loop_extent), "19");

  node.exec_condition = af::ExecuteCondition::kNoCache;
  EXPECT_FALSE(manager.TryBuildCacheGuardInfo(node, info));
  node.exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  EXPECT_FALSE(manager.TryBuildCacheGuardInfo(node, info));
  tuning_space->sub_axes[0]->is_bind_multi_core = false;
  EXPECT_FALSE(manager.TryBuildCacheGuardInfo(node, info));
}

TEST(CacheGuardInfoTest, RejectsGuardWhenBlockInnerIsNotLoopLeaf) {
  auto tuning_space = MakeGuardTuningSpace(Symbol("4"), Symbol("2"), Symbol("2"));
  auto trailing_axis = std::make_unique<SubAxis>();
  trailing_axis->id = 5;
  trailing_axis->name = "trailing_inner";
  trailing_axis->axis_type = AxisPosition::INNER;
  trailing_axis->repeat = Symbol("2");
  tuning_space->sub_axes.emplace_back(std::move(trailing_axis));
  auto &node = tuning_space->node_infos.front();
  node.loop_axes.emplace_back(tuning_space->sub_axes.back().get());
  ExeTimePassManager manager(tuning_space);
  CacheGuardInfo info;
  EXPECT_FALSE(manager.TryBuildCacheGuardInfo(node, info));
}

TEST(CacheGuardInfoTest, NddmaGuardAllowsTrailingVectorAxis) {
  auto tuning_space = MakeGuardTuningSpace(Symbol("4"), Symbol("2"), Symbol("2"));
  auto trailing_axis = std::make_unique<SubAxis>();
  trailing_axis->id = 5;
  trailing_axis->name = "trailing_vector";
  trailing_axis->axis_type = AxisPosition::INNER;
  trailing_axis->repeat = Symbol("2");
  tuning_space->sub_axes.emplace_back(std::move(trailing_axis));
  auto &node = tuning_space->node_infos.front();
  node.node_type = "Nddma";
  node.loop_axes.emplace_back(tuning_space->sub_axes.back().get());
  ExeTimePassManager manager(tuning_space);
  CacheGuardInfo info;
  EXPECT_TRUE(manager.TryBuildCacheGuardInfo(node, info));
  EXPECT_EQ(info.kind, CacheGuardKind::kFusedBroadcast);
}

class TestExeTimePass : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TestExeTimePass, FusedGuardUsesHitCount) {
  auto tuning_space = MakeGuardTuningSpace(Symbol("4"), Symbol("2"), Symbol("2"));
  const auto &node = tuning_space->node_infos.front();
  ExeTimePassManager exe_mgr(tuning_space);
  auto exe = exe_mgr.UpdateNodeExeTime(node, Symbol("19"));
  EXPECT_EQ(exe.GetTernaryOpStr(), "1");
}

TEST_F(TestExeTimePass, SymbolicCanonicalBlockSplitUsesHitCount) {
  const auto period = Symbol("period");
  const auto inner = Symbol("inner");
  auto tuning_space = MakeGuardTuningSpace(period, inner, af::sym::Ceiling(af::sym::Div(period, inner)));
  const auto &node = tuning_space->node_infos.front();
  ExeTimePassManager exe_mgr(tuning_space);
  const auto exe = exe_mgr.UpdateNodeExeTime(node, Symbol("19"));
  EXPECT_NE(exe.GetTernaryOpStr(), "19");
}

TEST_F(TestExeTimePass, OriginGuardUsesSingleHit) {
  auto tuning_space = MakeGuardTuningSpace(Symbol("4"), Symbol("2"), Symbol("2"));
  auto &node = tuning_space->node_infos.front();
  node.exec_condition = af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
  ExeTimePassManager exe_mgr(tuning_space);
  auto exe = exe_mgr.UpdateNodeExeTime(node, Symbol("19"));
  EXPECT_EQ(exe.GetTernaryOpStr(), "1");
}

TEST_F(TestExeTimePass, TailBlockUsesSingleGuardHit) {
  auto tuning_space = MakeGuardTuningSpace(Symbol("1024"), Symbol("19"), Symbol("54"));
  const auto &node = tuning_space->node_infos.front();
  ExeTimePassManager exe_mgr(tuning_space);
  auto exe = exe_mgr.UpdateNodeExeTime(node, Symbol("19"));
  EXPECT_EQ(exe.GetTernaryOpStr(), "1");
}

TEST_F(TestExeTimePass, NonUniformGuardFallsBackToLegacyRepeat) {
  auto tuning_space = MakeGuardTuningSpace(Symbol("5"), Symbol("6"), Symbol("54"));
  const auto &node = tuning_space->node_infos.front();
  ExeTimePassManager exe_mgr(tuning_space);
  auto exe = exe_mgr.UpdateNodeExeTime(node, Symbol("19"));
  EXPECT_EQ(exe.GetTernaryOpStr(), "19");
}

TEST_F(TestExeTimePass, MissingMetadataFallsBackToLegacyRepeat) {
  auto tuning_space = MakeGuardTuningSpace(Symbol("4"), Symbol("2"), Symbol("54"));
  auto &node = tuning_space->node_infos.front();
  node.exec_condition = af::ExecuteCondition::kConditionInvalid;
  ExeTimePassManager exe_mgr(tuning_space);
  auto exe = exe_mgr.UpdateNodeExeTime(node, Symbol("19"));
  EXPECT_EQ(exe.GetTernaryOpStr(), "19");
}

void GetExeTime(const TuningSpacePtr &tuning_space, TernaryOp &exe_cast0, TernaryOp &exe_store) {
  ExeTimePassManager exe_mgr(tuning_space);
  Expr exe_time;
  for (const auto &node : tuning_space->node_infos) {
    exe_time = CreateExpr(1U);
    for (auto &loop_axis : node.loop_axes) {
      exe_time = af::sym::Mul(exe_time, loop_axis->repeat);
    }
    if (node.name == "cast0") {
      exe_cast0 = exe_mgr.UpdateNodeExeTime(node, exe_time);
    } else if (node.name == "store") {
      exe_store = exe_mgr.UpdateNodeExeTime(node, exe_time);
    }
  }
}

TEST_F(TestExeTimePass, case1) {
  af::AscGraph graph("graph");
  att::BrcBufBeforeAutoFuse1(graph);
  att::BrcBufAfterScheduler1(graph);
  att::BrcBufAfterQueBufAlloc1(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  auto ret = ascend_graph_parser.GraphParser(graph);
  TernaryOp exe_time_cast0;
  TernaryOp exe_time_store;
  GetExeTime(tuning_space, exe_time_cast0, exe_time_store);
  EXPECT_EQ(exe_time_cast0.GetTernaryOpStr(), "z0z2Tb_size");
  EXPECT_EQ(exe_time_store.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
}

TEST_F(TestExeTimePass, case2) {
  af::AscGraph graph("graph");
  att::BrcBufBeforeAutoFuse2(graph);
  att::BrcBufAfterScheduler1(graph);
  att::BrcBufAfterQueBufAlloc1(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  auto ret = ascend_graph_parser.GraphParser(graph);
  TernaryOp exe_time_cast0;
  TernaryOp exe_time_store;
  GetExeTime(tuning_space, exe_time_cast0, exe_time_store);
  EXPECT_EQ(exe_time_cast0.GetTernaryOpStr(),
            "TernaryOp(IsEqual(z0z2Tb_size, 1.0), 1, (Ceiling((Z1 / (z1t_size))) * z0z2Tb_size))");
  EXPECT_EQ(exe_time_store.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
}

TEST_F(TestExeTimePass, case3) {
  af::AscGraph graph("graph");
  att::BrcBufBeforeAutoFuse3(graph);
  att::BrcBufAfterScheduler3(graph);
  att::BrcBufAfterQueBufAlloc3(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  auto ret = ascend_graph_parser.GraphParser(graph);
  TernaryOp exe_time_cast0;
  TernaryOp exe_time_store;
  GetExeTime(tuning_space, exe_time_cast0, exe_time_store);
  EXPECT_EQ(exe_time_cast0.GetTernaryOpStr(), "z0z2Tb_size");
  EXPECT_EQ(exe_time_store.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
}

TEST_F(TestExeTimePass, case4) {
  af::AscGraph graph("graph");
  att::BrcBufBeforeAutoFuse4(graph);
  att::BrcBufAfterScheduler4(graph);
  att::BrcBufAfterQueBufAlloc4(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  auto ret = ascend_graph_parser.GraphParser(graph);
  TernaryOp exe_time_cast0;
  TernaryOp exe_time_store;
  GetExeTime(tuning_space, exe_time_cast0, exe_time_store);
  EXPECT_EQ(exe_time_cast0.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
  EXPECT_EQ(exe_time_store.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
}

TEST_F(TestExeTimePass, exec_condition) {
  af::AscGraph graph("graph");
  att::BrcBufBeforeAutoFuse3(graph);
  att::BrcBufAfterScheduler3(graph);
  att::BrcBufAfterQueBufAlloc3(graph);
  auto load = graph.FindNode("cast0");
  load->attr.sched.exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  auto ret = ascend_graph_parser.GraphParser(graph);
  TernaryOp exe_time_cast0;
  TernaryOp exe_time_store;
  GetExeTime(tuning_space, exe_time_cast0, exe_time_store);
  EXPECT_EQ(exe_time_cast0.GetTernaryOpStr(),
            "Max(1, (Ceiling((Z1 / (z1t_size))) * z0z2Tb_size / (Ceiling((Z2 / (z2t_size))))))");
  EXPECT_EQ(exe_time_store.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
}
}  // namespace att
