/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <exception>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>

#include "backend_common.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen.h"
#include "common/platform_context.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "indirect_load_utils.h"
#include "optimize.h"
#include "runtime_stub.h"
#include "share_graph.h"

namespace {
constexpr size_t kRank = IL_RANK;
constexpr int64_t kAxis = IL_AXIS;
constexpr bool kHasInputPre = IL_HAS_INPUT_PRE;
constexpr bool kUseExp2 = IL_USE_EXP2;

const af::Axis *FindDerivedAxis(const af::AscGraph &graph, af::Axis::Type type, af::AxisId from) {
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis != nullptr && axis->type == type && axis->from == std::vector<af::AxisId>{from}) {
      return axis.get();
    }
  }
  return nullptr;
}

void CollectOriginAxes(af::AscGraph &graph, af::AxisId axis_id, std::vector<af::AxisId> &origins) {
  const auto *axis = graph.FindAxis(axis_id);
  ASSERT_NE(axis, nullptr);
  if (axis->from.empty()) {
    origins.emplace_back(axis_id);
    return;
  }
  for (af::AxisId from : axis->from) {
    CollectOriginAxes(graph, from, origins);
  }
}

void ExpectAxisOrigins(af::AscGraph &graph, af::AxisId axis_id, const std::vector<af::AxisId> &expected) {
  std::vector<af::AxisId> origins;
  CollectOriginAxes(graph, axis_id, origins);
  EXPECT_EQ(origins, expected);
}

void ExpectLoopFramework(af::AscGraph &graph, const af::AscNodePtr &indirect_load) {
  ascgen_utils::indirect_load::TemplateAxes axes;
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);

  const auto *outer = graph.FindAxis(axes.outer_axis);
  ASSERT_NE(outer, nullptr);
  const auto *tile_outer = FindDerivedAxis(graph, af::Axis::Type::kAxisTypeTileOuter, outer->id);
  const auto *tile_inner = FindDerivedAxis(graph, af::Axis::Type::kAxisTypeTileInner, outer->id);
  ASSERT_NE(tile_outer, nullptr);
  ASSERT_NE(tile_inner, nullptr);
  EXPECT_EQ(tile_outer->split_pair_other_id, tile_inner->id);
  EXPECT_EQ(tile_inner->split_pair_other_id, tile_outer->id);

  const auto *block_outer = FindDerivedAxis(graph, af::Axis::Type::kAxisTypeBlockOuter, tile_outer->id);
  const auto *block_inner = FindDerivedAxis(graph, af::Axis::Type::kAxisTypeBlockInner, tile_outer->id);
  ASSERT_NE(block_outer, nullptr);
  ASSERT_NE(block_inner, nullptr);
  EXPECT_EQ(block_outer->split_pair_other_id, block_inner->id);
  EXPECT_EQ(block_inner->split_pair_other_id, block_outer->id);

  const int64_t normalized_axis = kAxis < 0L ? kAxis + static_cast<int64_t>(kRank) : kAxis;
  const auto template_id = ascir::GetTemplateIdOrDefault(*indirect_load);
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    ExpectAxisOrigins(graph, axes.outer_axis, logical_view.output.axis_ids);
    EXPECT_EQ(axes.inner_axis, af::kIdNone);
    return;
  }

  ASSERT_EQ(template_id, ascir::TemplateId::kIndirectLoadSimd);
  const size_t split = static_cast<size_t>(normalized_axis);
  if (split == 0UL) {
    EXPECT_TRUE(outer->from.empty());
    EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(outer->size, af::ops::One), af::TriBool::kTrue);
  } else {
    ExpectAxisOrigins(
        graph, axes.outer_axis,
        std::vector<af::AxisId>(logical_view.output.axis_ids.begin(), logical_view.output.axis_ids.begin() + split));
  }
  ExpectAxisOrigins(
      graph, axes.inner_axis,
      std::vector<af::AxisId>(logical_view.output.axis_ids.begin() + split, logical_view.output.axis_ids.end()));
  if (kHasInputPre) {
    ExpectAxisOrigins(
        graph, axes.input_inner_axis,
        std::vector<af::AxisId>(logical_view.data.axis_ids.begin() + split, logical_view.data.axis_ids.end()));
  }
}

void CheckScheduledLoopFramework(ascir::FusedScheduledResult &result) {
  size_t simd_count = 0UL;
  size_t simt_count = 0UL;
  for (auto &candidates : result.node_idx_to_scheduled_results) {
    for (auto &candidate : candidates) {
      for (auto &group : candidate.schedule_groups) {
        for (auto &graph : group.impl_graphs) {
          const af::AscNodePtr indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(graph);
          if (indirect_load == nullptr) {
            continue;
          }
          ExpectLoopFramework(graph, indirect_load);
          if (ascir::GetTemplateIdOrDefault(*indirect_load) == ascir::TemplateId::kIndirectLoadSimd) {
            ++simd_count;
          } else {
            ++simt_count;
          }
        }
      }
    }
  }
  EXPECT_GT(simd_count, 0UL);
  EXPECT_GT(simt_count, 0UL);
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find("// IndirectLoad SIMD"), std::string::npos);
  EXPECT_NE(kernel.find("// IndirectLoad SIMT"), std::string::npos);
}
}  // namespace

class TestBackendIndirectLoadStoreE2e : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    ge::RuntimeStub::SetInstance(std::make_shared<af::RuntimeStubV2>());
  }

  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
  }
};

TEST_F(TestBackendIndirectLoadStoreE2e, IndirectLoadStoreCodegen) {
  const std::string tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  auto graph = ascir::ShareGraph::IndirectLoadStoreFusedGraph(kRank, kAxis, af::DT_FLOAT16, kHasInputPre, kUseExp2);
  ASSERT_NE(graph, nullptr);
  std::map<std::string, std::string> shape_info;
  for (size_t i = 0UL; i < 2UL * kRank; ++i) {
    shape_info.emplace("s" + std::to_string(i), "stub_s" + std::to_string(i));
  }

  const std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  ASSERT_EQ(parts.size(), 3U);
  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{.graph_type = optimize::GraphType::kFusedAscBackend});
    codegen::Codegen codegen(codegen::CodegenOptions{});
    ascir::FusedScheduledResult fused_schedule_result;
    codegen::CodegenResult result;
    testing::internal::CaptureStdout();
    const auto optimize_status = optimizer.Optimize(graph, fused_schedule_result);
    if (optimize_status == 0) {
      CheckScheduledLoopFramework(fused_schedule_result);
    }
    const auto codegen_status = optimize_status == 0 ? codegen.Generate(shape_info, fused_schedule_result, result) : -1;
    const std::string logs = testing::internal::GetCapturedStdout();
    EXPECT_EQ(logs.find("[ERROR]"), std::string::npos) << logs;
    ASSERT_EQ(optimize_status, 0) << logs;
    ASSERT_EQ(codegen_status, 0) << logs;
    CheckGeneratedKernel(result.kernel);

    std::fstream kernel_file(parts[0], std::ios::out);
    std::fstream tiling_file(parts[1], std::ios::out);
    std::fstream tiling_data_file(parts[2], std::ios::out);
    ASSERT_TRUE(kernel_file.is_open());
    ASSERT_TRUE(tiling_file.is_open());
    ASSERT_TRUE(tiling_data_file.is_open());
    kernel_file << tiling_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
    EXPECT_TRUE(kernel_file.good());
    EXPECT_TRUE(tiling_file.good());
    EXPECT_TRUE(tiling_data_file.good());
  } catch (const std::exception &e) {
    FAIL() << e.what();
  } catch (...) {
    FAIL() << "Unknown exception";
  }
}
