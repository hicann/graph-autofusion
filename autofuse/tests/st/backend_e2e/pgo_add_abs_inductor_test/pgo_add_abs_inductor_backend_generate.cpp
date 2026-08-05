/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <exception>
#include <fstream>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "backend_common.h"
#include "codegen.h"
#include "common_utils.h"
#include "optimize.h"
#include "share_graph.h"
#include "tests/common/inductor_pgo_codegen_test_utils.h"

namespace {
class TestBackendPgoAddAbsInductorE2e : public testing::Test {};

TEST_F(TestBackendPgoAddAbsInductorE2e, PgoAddAbsInductorE2eCodegen) {
  auto graph = ascir::ShareGraph::AddAbsFusedGraph(3);
  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});
    ascir::FusedScheduledResult fused_schedule_result;
    std::vector<::ascir::ScheduledResult> schedule_results;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

    codegen::CodegenResult result;
    ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), 0);

    EXPECT_FALSE(result.tiling_data.empty());
    EXPECT_FALSE(result.tiling.empty());
    EXPECT_FALSE(result.kernel.empty());
    EXPECT_NE(result.tiling.find("extern \"C\" int64_t GenerateTopnSolutions("), std::string::npos);
    EXPECT_NE(result.tiling.find("std::string GetTilingDataRepr("), std::string::npos);
    EXPECT_NE(result.kernel.find("AutofuseLaunch"), std::string::npos);

    autofuse::tests::WriteCodegenResult(result, splitString(KERNEL_SRC_LIST, ':'));
  } catch (const std::exception &e) {
    FAIL() << e.what();
  } catch (...) {
    FAIL() << "pgo_add_abs inductor codegen failed";
  }
}

TEST_F(TestBackendPgoAddAbsInductorE2e, PgoAddAbsInductorPgoCodegen) {
  autofuse::tests::ScopedAutofusePgoFlag pgo_flag;
  auto graph = ascir::ShareGraph::AddAbsFusedConstGraph(3, {32, 16, 16});
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), af::SUCCESS);
  ASSERT_EQ(fused_schedule_result.node_idx_to_scheduled_results.size(), 1U);
  ASSERT_FALSE(fused_schedule_result.node_idx_to_scheduled_results[0].empty());

  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);
  ASSERT_EQ(fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1U);
  EXPECT_FALSE(ascgen_utils::IsCubeFusedScheduled(fused_schedule_result));
  EXPECT_TRUE(ascgen_utils::IsStaticSchedResult(fused_schedule_result));
  EXPECT_TRUE(ascgen_utils::IsSingleGroup(fused_schedule_result));
  EXPECT_TRUE(fused_schedule_result.workspace_nodes.empty());
  EXPECT_FALSE(fused_schedule_result.node_idx_to_scheduled_results[0][0].enable_group_parallel);
  EXPECT_TRUE(ascgen_utils::CanUseTilingKey(fused_schedule_result));

  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;
  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);
  const auto runner_start = result.tiling.find("AUTOFUSE_SPLIT_FILE_BEGIN: PgoRunner");
  const auto runner_end = result.tiling.find("AUTOFUSE_SPLIT_FILE_END: PgoRunner", runner_start);
  ASSERT_NE(runner_start, std::string::npos);
  ASSERT_NE(runner_end, std::string::npos);
  const auto runner = result.tiling.substr(runner_start, runner_end - runner_start);
  EXPECT_NE(result.tiling.find("PGOByCoreNumSearchTilingKey(measured_tiling_datas"), std::string::npos);
  EXPECT_EQ(result.tiling.find("GetBuiltinTfPgoConfigs()"), std::string::npos);
  EXPECT_NE(runner.find("struct ResLimit {"), std::string::npos);
  EXPECT_EQ(runner.find("size_t input0_size = (4 * s0"), std::string::npos);
  EXPECT_FALSE(result.kernel.empty());

  autofuse::tests::WriteCodegenResult(result, splitString(PGO_KERNEL_SRC_LIST, ':'));
}
}  // namespace
