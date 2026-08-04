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

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "runtime_stub.h"
#include "graph_utils_ex.h"

#define private public
#include "buffer_allocate/buf_que_allocator.h"
#include "asc_graph_builder.h"
#include "ascgraph_info_complete.h"
#include "common/l2_cache_hint_manager.h"
#undef private
#include "schedule_utils.h"
#include "platform_context.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace optimize;
using af::testing::AscGraphBuilder;

namespace optimize {
class SetL2CtrlTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v1 = std::make_shared<RuntimeStub>();
    RuntimeStub::SetInstance(stub_v1);
  }
  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
  }
};
}  // namespace optimize

namespace {
af::AscGraph MakeStaticLoadStoreGraph(const std::string &name, int64_t size, int64_t data_index = 0) {
  auto graph = AscGraphBuilder(name)
                   .Loops({size})
                   .Data("data" + std::to_string(data_index), data_index, af::DT_UINT8)
                   .Load("load", "data" + std::to_string(data_index))
                   .Store("store", "load")
                   .Output("output" + std::to_string(data_index), "store", data_index, af::DT_UINT8)
                   .Build();
  AscGraphInfoComplete::CompleteApiInfo(graph);
  return graph;
}

ascir::FusedScheduledResult MakeFusedScheduledResultWithGraphs(std::vector<af::AscGraph> &&impl_graphs) {
  ascir::FusedScheduledResult fused_result{};
  fused_result.node_idx_to_scheduled_results.resize(1UL);
  auto &scheduled_result = fused_result.node_idx_to_scheduled_results[0].emplace_back();
  auto &group = scheduled_result.schedule_groups.emplace_back();
  group.impl_graphs = std::move(impl_graphs);
  return fused_result;
}

struct L2CtrlTestResult {
  af::AscGraph hint_graph;
  ascir::FusedScheduledResult fused_result;
};
L2CtrlTestResult BuildResult(std::vector<af::AscGraph> &&impl_graphs) {
  af::AscGraph hint_graph = impl_graphs.front();
  auto fused_result = MakeFusedScheduledResultWithGraphs(std::move(impl_graphs));
  BufQueAllocator allocator;
  EXPECT_EQ(allocator.PrepareImplGraphMemoryPlan(fused_result), af::SUCCESS);
  EXPECT_EQ(allocator.CollectFusedIoNodes(fused_result), af::SUCCESS);
  return {std::move(hint_graph), std::move(fused_result)};
}

af::ComputeGraphPtr ToComputeGraph(const af::AscGraph &hint_graph) {
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(hint_graph);
  EXPECT_NE(compute_graph, nullptr);
  return compute_graph;
}

bool GetDataSkipHintFromGraph(const af::AscGraph &impl_graph, const std::string &data_name) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (node != nullptr && node->GetName() == data_name && af::ops::IsOps<af::ascir_op::Data>(node)) {
      bool value = false;
      af::AttrUtils::GetBool(node->GetOpDesc(), "_skip_l2_cache_hint", value);
      return value;
    }
  }
  return false;
}

bool GetDataSkipHintFromScheduledResult(const ascir::ScheduledResult &scheduled_result, const std::string &data_name) {
  for (const auto &schedule_group : scheduled_result.schedule_groups) {
    for (const auto &impl_graph : schedule_group.impl_graphs) {
      if (GetDataSkipHintFromGraph(impl_graph, data_name)) {
        return true;
      }
    }
  }
  return false;
}

bool GetDataSkipHint(ascir::FusedScheduledResult &fsr, const std::string &data_name) {
  for (auto &scheduled_results : fsr.node_idx_to_scheduled_results) {
    for (auto &scheduled_result : scheduled_results) {
      if (GetDataSkipHintFromScheduledResult(scheduled_result, data_name)) {
        return true;
      }
    }
  }
  return false;
}

af::AscGraph MakeMultiSuccessorGraph(const std::string &name, int64_t size) {
  auto graph = AscGraphBuilder(name)
                   .Loops({size})
                   .Data("data0", 0, af::DT_UINT8)
                   .Load("load0", "data0")
                   .Store("store0", "load0")
                   .Output("output0", "store0", 0, af::DT_UINT8)
                   .Load("load1", "data0")
                   .Store("store1", "load1")
                   .Output("output1", "store1", 1, af::DT_UINT8)
                   .Build();
  AscGraphInfoComplete::CompleteApiInfo(graph);
  return graph;
}

af::AscGraph MakeMultiDataSameIndexGraph(const std::string &name, int64_t size) {
  auto graph = AscGraphBuilder(name)
                   .Loops({size})
                   .Data("data0", 0, af::DT_UINT8)
                   .Load("load0", "data0")
                   .Store("store0", "load0")
                   .Output("output0", "store0", 0, af::DT_UINT8)
                   .Data("data1", 0, af::DT_UINT8)
                   .Load("load1", "data1")
                   .Store("store1", "load1")
                   .Output("output1", "store1", 1, af::DT_UINT8)
                   .Build();
  AscGraphInfoComplete::CompleteApiInfo(graph);
  return graph;
}
}  // namespace

TEST_F(SetL2CtrlTest, CalcTensorSizesBasic) {
  constexpr int64_t kInputSize = 16;
  auto [hint_graph, fused_result] = BuildResult({MakeStaticLoadStoreGraph("g0", kInputSize, 0)});

  ascir::GmTensorSizes sizes;
  EXPECT_EQ(optimize::L2CacheHintManager::CalcTensorSizes(*ToComputeGraph(hint_graph), fused_result, sizes),
            af::SUCCESS);
  ASSERT_EQ(sizes.input_sizes.size(), 1UL);
  ASSERT_EQ(sizes.output_sizes.size(), 1UL);
  EXPECT_TRUE(sizes.total_size.IsValid());
  EXPECT_TRUE(sizes.input_sizes[0].IsValid());
  EXPECT_TRUE(sizes.output_sizes[0].IsValid());
  EXPECT_EQ(sizes.min_total_size, 2LL * kInputSize);
  EXPECT_TRUE(sizes.total_size.IsConstExpr());
  EXPECT_EQ(sizes.input_sizes[0].IsConstExpr(), true);
  EXPECT_EQ(sizes.output_sizes[0].IsConstExpr(), true);
}

TEST_F(SetL2CtrlTest, CalcTensorSizesWithSymbolicDim) {
  auto graph = AscGraphBuilder("sym_g")
                   .Loops({af::testing::Sym("sym_n")})
                   .Data("sym_data", 0, af::DT_UINT8)
                   .Load("sym_load", "sym_data")
                   .Store("sym_store", "sym_load")
                   .Output("sym_output", "sym_store", 0, af::DT_UINT8)
                   .Build();
  AscGraphInfoComplete::CompleteApiInfo(graph);

  af::AscGraph hint_graph = graph;
  auto fused_result = MakeFusedScheduledResultWithGraphs({graph});
  BufQueAllocator allocator;
  EXPECT_EQ(allocator.PrepareImplGraphMemoryPlan(fused_result), af::SUCCESS);
  EXPECT_EQ(allocator.CollectFusedIoNodes(fused_result), af::SUCCESS);

  ascir::GmTensorSizes sizes;
  EXPECT_EQ(optimize::L2CacheHintManager::CalcTensorSizes(*ToComputeGraph(hint_graph), fused_result, sizes),
            af::SUCCESS);
  ASSERT_EQ(sizes.input_sizes.size(), 1UL);
  ASSERT_EQ(sizes.output_sizes.size(), 1UL);
  EXPECT_FALSE(sizes.input_sizes[0].IsConstExpr());
  EXPECT_FALSE(sizes.output_sizes[0].IsConstExpr());
  EXPECT_FALSE(sizes.total_size.IsConstExpr());
  EXPECT_EQ(sizes.min_total_size, 2);
}

TEST_F(SetL2CtrlTest, MarkSkipHintMultiSuccessor) {
  auto fused_result = MakeFusedScheduledResultWithGraphs({MakeMultiSuccessorGraph("g0", 16)});
  EXPECT_EQ(optimize::L2CacheHintManager::MarkInputsNeedSkipL2CacheHint(fused_result), af::SUCCESS);
  EXPECT_TRUE(GetDataSkipHint(fused_result, "data0"));
}

TEST_F(SetL2CtrlTest, MarkSkipHintNormalNotSet) {
  auto fused_result = MakeFusedScheduledResultWithGraphs({MakeStaticLoadStoreGraph("g0", 3 * 1024 * 1024, 0)});
  EXPECT_EQ(optimize::L2CacheHintManager::MarkInputsNeedSkipL2CacheHint(fused_result), af::SUCCESS);
  EXPECT_FALSE(GetDataSkipHint(fused_result, "data0"));
}

TEST_F(SetL2CtrlTest, MarkSkipHintMultiDataSameIndex) {
  auto fused_result = MakeFusedScheduledResultWithGraphs({MakeMultiDataSameIndexGraph("g0", 16)});
  EXPECT_EQ(optimize::L2CacheHintManager::MarkInputsNeedSkipL2CacheHint(fused_result), af::SUCCESS);
  EXPECT_TRUE(GetDataSkipHint(fused_result, "data0"));
  EXPECT_TRUE(GetDataSkipHint(fused_result, "data1"));
}
