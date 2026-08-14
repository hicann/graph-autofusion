/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_BACKEND_CODEGEN_COMMON_H_
#define AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_BACKEND_CODEGEN_COMMON_H_

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "backend_common.h"
#include "codegen.h"
#include "common_utils.h"
#include "optimize.h"
#include "share_graph.h"

template <typename PrepareSchedule, typename CheckKernel>
inline void GenerateBackendKernelWithScheduleCheck(
    const af::ComputeGraphPtr &graph, const std::map<std::string, std::string> &shape_info,
    const std::string &tiling_stub, const std::string &kernel_src_file_name, const std::string &tiling_src_file_name,
    const std::string &tiling_data_src_file_name, PrepareSchedule prepare_schedule, CheckKernel check_kernel) {
  bool gen_success = true;
  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    if (prepare_schedule(fused_schedule_result)) {
      codegen::CodegenResult result;
      EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);
      const std::string kernel = RemoveSubDirInclude(result.kernel);
      check_kernel(kernel);
      kernel_file << tiling_stub << kernel;
      tiling_file << result.tiling;
      tiling_data_file << result.tiling_data;
    } else {
      gen_success = false;
    }
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

template <typename CheckKernel>
inline void GenerateBackendKernelWithCheck(const af::ComputeGraphPtr &graph,
                                           const std::map<std::string, std::string> &shape_info,
                                           const std::string &tiling_stub, CheckKernel check_kernel) {
  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  GenerateBackendKernelWithScheduleCheck(
      graph, shape_info, tiling_stub, parts[0], parts[1], parts[2], [](ascir::FusedScheduledResult &) { return true; },
      check_kernel);
}

template <typename CheckKernel>
inline void GenerateCvBackendUbKernelWithCheck(const af::ComputeGraphPtr &graph,
                                               const std::string &kernel_src_file_name,
                                               const std::string &tiling_src_file_name,
                                               const std::string &tiling_data_src_file_name, CheckKernel check_kernel) {
  GenerateBackendKernelWithScheduleCheck(
      graph, {}, "", kernel_src_file_name, tiling_src_file_name, tiling_data_src_file_name,
      [](ascir::FusedScheduledResult &fused_schedule_result) {
        const bool is_cube_fused = ascgen_utils::IsCubeFusedScheduled(fused_schedule_result);
        EXPECT_TRUE(is_cube_fused);
        if (!is_cube_fused) {
          return false;
        }
        ascgen_utils::FilterCVFusionUBResult(fused_schedule_result);
        return true;
      },
      check_kernel);
}

#endif  // AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_BACKEND_CODEGEN_COMMON_H_
