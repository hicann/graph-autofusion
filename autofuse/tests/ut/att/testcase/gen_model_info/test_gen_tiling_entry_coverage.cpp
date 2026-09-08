/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <gtest/gtest.h>
#include "autofuse_config/auto_fuse_config.h"
#include "base/att_const_values.h"
#include "gen_tiling_impl.h"
#include "graph_construct_utils.h"
#include "test_fa_ascir_graph.h"
#include "util/duration.h"
#include "util/thread_local_context.h"

namespace att {
namespace {
class GenTilingEntryCoverageTest : public testing::Test {
 protected:
  void SetUp() override {
    for (const auto name : {"AUTOFUSE_FLAGS", "AUTOFUSE_DFX_FLAGS"}) {
      const auto value = std::getenv(name);
      if (value != nullptr) {
        environment_[name] = value;
      }
      unsetenv(name);
    }
    AutoFuseConfig::MutableAttStrategyConfig().Reset();
    AutoFuseConfig::MutablePgoStrategyConfig() = PgoStrategyConfig();
    GetThreadLocalContext().SetOption({});
    kg_duration_level = 0U;
  }

  void TearDown() override {
    for (const auto name : {"AUTOFUSE_FLAGS", "AUTOFUSE_DFX_FLAGS"}) {
      const auto iter = environment_.find(name);
      if (iter == environment_.end()) {
        unsetenv(name);
      } else {
        setenv(name, iter->second.c_str(), 1);
      }
    }
    AutoFuseConfig::MutableAttStrategyConfig() = saved_att_;
    AutoFuseConfig::MutablePgoStrategyConfig() = saved_pgo_;
    GetThreadLocalContext() = saved_context_;
    kg_duration_level = saved_duration_;
  }

  af::AscGraph MakeGraph() const {
    af::AscGraph graph("entry_coverage");
    FaBeforeAutoFuse(graph);
    FaAfterScheduler(graph);
    FaAfterQueBufAlloc(graph);
    GraphConstructUtils::UpdateGraphVectorizedStride(graph);
    return graph;
  }

  ascir::FusedScheduledResult Schedule(const af::AscGraph &graph) const {
    ascir::ScheduleGroup group;
    group.impl_graphs.emplace_back(graph);
    ascir::ScheduledResult schedule;
    schedule.schedule_groups.emplace_back(group);
    ascir::FusedScheduledResult fused;
    fused.node_idx_to_scheduled_results = {{schedule}};
    return fused;
  }

 private:
  AttStrategyConfig saved_att_ = AutoFuseConfig::GetAttStrategyConfig();
  PgoStrategyConfig saved_pgo_ = AutoFuseConfig::GetPgoStrategyConfig();
  ThreadLocalContext saved_context_ = GetThreadLocalContext();
  uint32_t saved_duration_ = kg_duration_level;
  std::map<std::string, std::string> environment_;
};

TEST_F(GenTilingEntryCoverageTest, RejectsEmptyGraphListWithoutChangingOptions) {
  std::map<std::string, std::string> options{{kGenConfigType, "HighPerf"}};
  const auto original = options;
  EXPECT_FALSE(GenTilingImpl("Empty", {}, options));
  EXPECT_EQ(options, original);
}

TEST_F(GenTilingEntryCoverageTest, RejectsGraphWithInvalidAxisIdentifier) {
  auto graph = MakeGraph();
  ASSERT_FALSE(graph.GetAllAxis().empty());
  graph.GetAllAxis().front()->id = -1;
  ASSERT_FALSE(graph.CheckValid());
  std::map<std::string, std::string> options;
  EXPECT_FALSE(GenTilingImpl("InvalidAxis", {graph}, options));
}

TEST_F(GenTilingEntryCoverageTest, RejectsUnregisteredOptionBeforeGeneratingFiles) {
  auto graph = MakeGraph();
  ASSERT_TRUE(graph.CheckValid());
  std::map<std::string, std::string> options{{"unknown_option", "1"}};
  EXPECT_FALSE(GenTilingImpl("InvalidOption", {graph}, options));
  EXPECT_EQ(options.size(), 1U);
}

TEST_F(GenTilingEntryCoverageTest, InvalidDurationFallsBackToUnprofiledGeneratedCode) {
  auto graph = MakeGraph();
  const auto schedules = Schedule(graph);
  std::map<std::string, std::string> options{
      {kGenConfigType, "HighPerf"}, {kTilingDataTypeName, "EntryTilingData"}, {kDurationLevelName, "invalid"}};
  std::map<std::string, std::string> generated;
  ASSERT_TRUE(GenTilingImplAutoFuseV3("Entry", schedules, options, generated, false));
  ASSERT_NE(generated.find(kTilingHeadIdentify), generated.end());
  for (const auto &entry : generated) {
    EXPECT_EQ(entry.second.find("namespace duration_utils"), std::string::npos);
  }
  EXPECT_EQ(kg_duration_level, 0U);
  options[kDurationLevelName] = "999999999999999999999999";
  generated.clear();
  ASSERT_TRUE(GenTilingImplAutoFuseV3("Entry", schedules, options, generated, false));
  ASSERT_NE(generated.find(kTilingHeadIdentify), generated.end());
  for (const auto &entry : generated) {
    EXPECT_EQ(entry.second.find("namespace duration_utils"), std::string::npos);
  }
  EXPECT_EQ(kg_duration_level, 0U);
}

TEST_F(GenTilingEntryCoverageTest, ExplicitPgoOptionOverridesEnvironmentInBothDirections) {
  auto graph = MakeGraph();
  const auto schedules = Schedule(graph);
  for (const bool enabled : {false, true}) {
    SCOPED_TRACE(enabled);
    setenv("AUTOFUSE_FLAGS", enabled ? "--autofuse_enable_pgo=false" : "--autofuse_enable_pgo=true", 1);
    AutoFuseConfig::MutableAttStrategyConfig().Reset();
    AutoFuseConfig::MutablePgoStrategyConfig() = PgoStrategyConfig();
    std::map<std::string, std::string> options{{kGenConfigType, "AxesReorder"},
                                               {kTilingDataTypeName, "EntryTilingData"},
                                               {kInternalEnableAutofusePgo, enabled ? "true" : "false"}};
    std::map<std::string, std::string> generated;
    ASSERT_TRUE(GenTilingImplAutoFuseV3("Entry", schedules, options, generated, false));
    std::string combined;
    for (const auto &entry : generated) {
      combined += entry.second;
    }
    EXPECT_EQ(combined.find("PGOByCoreNumSearchTilingKey") != std::string::npos, enabled);
    EXPECT_EQ(AutoFuseConfig::GetPgoStrategyConfig().enable_autofuse_pgo, enabled ? "false" : "true");
  }
}

TEST_F(GenTilingEntryCoverageTest, ConvertsMissingWorkspaceOutputExceptionToFailure) {
  auto graph = MakeGraph();
  af::Operator workspace("missing_output", "Workspace");
  const auto node = graph.AddNode(workspace);
  ASSERT_NE(node, nullptr);
  ASSERT_TRUE(node->outputs().empty());
  ASSERT_TRUE(graph.CheckValid());
  std::map<std::string, std::string> options{{kGenConfigType, "HighPerf"}};
  EXPECT_FALSE(GenTilingImpl("MissingOutput", {graph}, options));
  EXPECT_EQ(kg_duration_level, 0U);
}
}  // namespace
}  // namespace att
