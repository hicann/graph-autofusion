/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "base/base_types.h"
#include "base/model_info.h"

namespace att {
namespace {

class DefaultTilingScheduleConfigTableForTest final : public TilingScheduleConfigTable {
 public:
  bool IsEnableBlockLoopAutoTune() const override {
    return false;
  }
  bool IsEnableCacheLineCheck() const override {
    return false;
  }
  TradeOffConfig GetTradeOffConfig() const override {
    return {};
  }
  double GetUbThresholdPerfValEffect() const override {
    return 0.0;
  }
  TilingScheduleConfig GetModelTilingScheduleConfig() const override {
    return {};
  }
  uint32_t GetCacheLineSize() const override {
    return arch_param::kDefaultCacheLineSize;
  }
  bool IsCoreNumThresholdPenaltyEnable() const override {
    return false;
  }
};

TEST(BaseTypesCoverageTest, TensorShapeFormatsEmptySingleAndMultipleDimensions) {
  TensorShapeInfo shape{};
  EXPECT_EQ(shape.GetDimExpr(), "");

  shape.dims = {af::Symbol("n")};
  EXPECT_EQ(shape.GetDimExpr(), "n");

  shape.dims = {af::Symbol("n"), af::Symbol(1), af::Symbol(64)};
  EXPECT_EQ(shape.GetDimExpr(), "n,1,64");
  EXPECT_EQ(GetVecString(shape.dims), "n,1,64,");
  EXPECT_EQ(GetVecString({}), "");
}

TEST(BaseTypesCoverageTest, ExpressionHelpersHandleMovedFromEmptyAndNamedExpressions) {
  Expr moved_from = af::Symbol("axis_size");
  const Expr retained(std::move(moved_from));
  ASSERT_FALSE(moved_from.IsValid());
  EXPECT_FALSE(IsValid(moved_from));
  EXPECT_EQ(Str(moved_from), "");
  EXPECT_EQ(GetSymbolName(moved_from), "");

  EXPECT_TRUE(IsValid(retained));
  EXPECT_EQ(Str(retained), "axis_size");
  const af::Symbol named("axis_size");
  EXPECT_EQ(GetSymbolName(named), "axis_size");
  EXPECT_FALSE(IsValid(Expr{}));
  EXPECT_EQ(Str(Expr{}), "");
  EXPECT_EQ(Str(CreateExpr(-7)), "-7");
}

TEST(BaseTypesCoverageTest, ExpressionComparatorHandlesInvalidAndLexicographicOrder) {
  Expr moved_from = af::Symbol("a");
  const Expr first(std::move(moved_from));
  const Expr second = af::Symbol("b");
  ASSERT_FALSE(moved_from.IsValid());

  const ExprCmp compare;
  EXPECT_TRUE(compare(moved_from, first));
  EXPECT_FALSE(compare(first, moved_from));
  EXPECT_TRUE(compare(first, second));
  EXPECT_FALSE(compare(second, first));
  EXPECT_FALSE(compare(first, first));
}

TEST(BaseTypesCoverageTest, VectorDebugStringPreservesValuesAndTrailingSeparator) {
  EXPECT_EQ(DebugString(std::vector<int32_t>{}), "[]");
  EXPECT_EQ(DebugString(std::vector<int32_t>{0, -1, 42}), "[0,-1,42,]");
  EXPECT_EQ(DebugString(std::vector<std::string>{}), "[]");
  EXPECT_EQ(DebugString(std::vector<std::string>{"axis", "", "size"}), "[axis,,size,]");
}

TEST(BaseTypesCoverageTest, ScheduleGroupIdentityAndPrefixesUseTheSpecifiedIds) {
  const ScheduleGroupIdent group{3UL, 2UL, 4UL};
  const ScheduleGroupIdent other_asc_graph{9UL, 2UL, 4UL};
  const ScheduleGroupIdent earlier_impl{3UL, 1UL, 9UL};
  const ScheduleGroupIdent later_group{3UL, 2UL, 5UL};

  EXPECT_TRUE(group == other_asc_graph);
  EXPECT_FALSE(group != other_asc_graph);
  EXPECT_FALSE(group < other_asc_graph);
  EXPECT_TRUE(earlier_impl < group);
  EXPECT_FALSE(group < earlier_impl);
  EXPECT_TRUE(group < later_group);
  EXPECT_FALSE(later_group < group);
  EXPECT_TRUE(group != earlier_impl);
  EXPECT_TRUE(group != later_group);
  EXPECT_EQ(group.GetGroupPrefix(), "AscGraph3ScheduleResult2G4");
  EXPECT_EQ(group.GetGroupPrefixSnakeCase(), "asc_graph3_schedule_result2_g4");
  EXPECT_EQ(group.GetItemPrefix(), "graph3_result2_g4");
}

TEST(BaseTypesCoverageTest, ReuseGroupRequiresMatchingInputSearchAndTilingKeyCounts) {
  ReuseScheduleGroup reuse;
  reuse.reuse_group_ident = {0UL, 0UL, 0UL};
  reuse.info = {{"input"}, {"search"}, {1U}};
  const ScheduleGroupIdent candidate{0UL, 0UL, 1UL};
  EXPECT_FALSE(reuse.IsReuseGroup(reuse.reuse_group_ident));
  EXPECT_FALSE(reuse.IsReuseGroup(candidate));

  reuse.schedule_group_to_info[candidate] = {{"other_input"}, {"other_search"}, {2U}};
  EXPECT_TRUE(reuse.IsReuseGroup(candidate));
  auto &candidate_info = reuse.schedule_group_to_info[candidate];
  candidate_info.reuse_search_axes.clear();
  EXPECT_FALSE(reuse.IsReuseGroup(candidate));
  candidate_info.reuse_search_axes = {"other_search"};
  candidate_info.reuse_input_axes.clear();
  EXPECT_FALSE(reuse.IsReuseGroup(candidate));
  candidate_info.reuse_input_axes = {"other_input"};
  candidate_info.tiling_keys.clear();
  EXPECT_FALSE(reuse.IsReuseGroup(candidate));
  candidate_info.tiling_keys = {2U};
  EXPECT_TRUE(reuse.IsReuseGroup(candidate));

  reuse.info = {};
  candidate_info = {};
  EXPECT_TRUE(reuse.IsReuseGroup(candidate));
}

TEST(BaseTypesCoverageTest, ScheduleConfigDebugStringIncludesAllConfigurationFields) {
  TradeOffConfig trade_off;
  trade_off.ub_ratio = af::Symbol(0);
  trade_off.core_num_ratio = af::Symbol(1);
  EXPECT_EQ(trade_off.DebugString(), "is_enable: 0, ub_ratio: 0, core_num_ratio: 1");
  trade_off.is_enable = true;
  EXPECT_EQ(trade_off.DebugString(), "is_enable: 1, ub_ratio: 0, core_num_ratio: 1");

  TilingScheduleConfig config;
  config.trade_off_config = trade_off;
  config.cache_line_size = 128U;
  config.vector_len_size = 256U;
  EXPECT_EQ(config.DebugString(),
            "trade_off_config: {is_enable: 1, ub_ratio: 0, core_num_ratio: 1}, cache_line_size: 128, "
            "vector_len_size: 256, is_penalty_config: 0");
  config.is_penalty_config = true;
  EXPECT_EQ(config.DebugString(),
            "trade_off_config: {is_enable: 1, ub_ratio: 0, core_num_ratio: 1}, cache_line_size: 128, "
            "vector_len_size: 256, is_penalty_config: 1");
}

TEST(BaseTypesCoverageTest, CacheLineConfigFormatsTransferAndRecognizesBothDirections) {
  CacheLineConfig unknown("load", af::Symbol("bytes"), 64U);
  EXPECT_EQ(unknown.ToString(), "node: load, expr: bytes, size: 64");
  EXPECT_FALSE(unknown.IsCacheLineConflictCandidate());
  EXPECT_FALSE(IsValid(unknown.solver_cache_line_expr));

  const CacheLineConfig load("input", af::Symbol(32), 64U, CacheLineDirection::kGmToUb, af::Symbol("total"));
  EXPECT_TRUE(load.IsCacheLineConflictCandidate());
  EXPECT_EQ(Str(load.solver_cache_line_expr), "total");
  EXPECT_EQ(load.ToString(), "node: input, expr: 32, size: 64");
  const CacheLineConfig store("output", af::Symbol(16), 128U, CacheLineDirection::kUbToGm);
  EXPECT_TRUE(store.IsCacheLineConflictCandidate());
  EXPECT_EQ(store.ToString(), "node: output, expr: 16, size: 128");
}

TEST(BaseTypesCoverageTest, ScheduleConfigTableSuppliesDefaultPriorityThresholdAndVectorLength) {
  const DefaultTilingScheduleConfigTableForTest table;
  const TilingScheduleConfigTable &base = table;
  EXPECT_EQ(base.GetConfigPriority(), TilingScheduleConfigPriority::kDefaultPriority);
  EXPECT_DOUBLE_EQ(base.GetPerfEffectVal(), 5000.0);
  EXPECT_EQ(base.GetVectorLenSize(), arch_param::kDefaultVectorLenSize);
}

}  // namespace
}  // namespace att
