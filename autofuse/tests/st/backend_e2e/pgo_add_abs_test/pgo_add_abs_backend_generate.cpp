/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <exception>
#include <filesystem>
#include "codegen.h"
#include "optimize.h"
#include "share_graph.h"
#include "backend_common.h"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

class TestBackendPgoAddAbsE2e : public testing::Test {
 protected:
  void SetUp() override {
    setenv("AUTOFUSE_FLAGS", "--autofuse_enable_pgo=true", 1);
  }
  void TearDown() override {
    unsetenv("AUTOFUSE_FLAGS");
  }
};

// pgo有跨文件共享变量的需求，故不使用CombineTilings接口拼接文件
TEST_F(TestBackendPgoAddAbsE2e, PgoAddAbsE2eCodegen) {
  bool gen_success = true;
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  // shape_info 和 AddAbsFusedGraph入参dims_size匹配（个数相同，命名规则为s开头、编号从0开始）
  std::map<std::string, std::string> shape_info({{"s0", "32"}, {"s1", "16"}, {"s2", "16"}});
  auto graph = ascir::ShareGraph::AddAbsFusedConstGraph(3, {32, 16, 16});
  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name = parts[0];                   // pgo_add_abs_test_kernel.cpp
  std::string tiling_data_src_file_name = parts[1];              // autofuse_tiling_data.h
  std::string tiling_head_file_name = parts[2];                  // pgo_add_abs_tiling_head.h
  std::string asc_graph0_result0_g0_file_name = parts[3];        // pgo_add_abs_asc_graph0_schedule_result0_g0.h
  std::string schedule_group_tail_file_name = parts[4];          // pgo_add_abs_schedule_group_tail.h
  std::string solver_func_file_name = parts[5];                  // pgo_add_abs_solver_func.h
  std::string tiling_def_and_tiling_const_file_name = parts[6];  // pgo_add_abs_tiling_def_and_tiling_const.h

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);
    std::fstream tiling_head_file(tiling_head_file_name, std::ios::out);
    std::fstream asc_graph0_result0_g0_file(asc_graph0_result0_g0_file_name, std::ios::out);
    std::fstream schedule_group_tail_file(schedule_group_tail_file_name, std::ios::out);
    std::fstream solver_func_file(solver_func_file_name, std::ios::out);
    std::fstream tiling_def_and_tiling_const_file(tiling_def_and_tiling_const_file_name, std::ios::out);

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    std::string kernel;
    EXPECT_EQ(codegen.GenerateKernel(fused_schedule_result, kernel), 0);
    std::string tiling_data;
    tiling_data = codegen.GenerateTilingData(fused_schedule_result);
    std::string pgo_src;
    pgo_src = codegen.GeneratorPgo(fused_schedule_result, ".");
    EXPECT_EQ(pgo_src.empty(), false);

    std::map<std::string, std::string> tiling_file_name_to_content;
    tiling_file_name_to_content = codegen.GenerateTiling(fused_schedule_result, shape_info, ".", "10");
    kernel_file << tilig_stub << RemoveSubDirInclude(kernel);
    tiling_data_file << tiling_data;
    tiling_head_file << tiling_file_name_to_content["TilingHead"];
    asc_graph0_result0_g0_file << tiling_file_name_to_content["asc_graph0_schedule_result0_g0"];
    schedule_group_tail_file << tiling_file_name_to_content["schedule_group_tail"];
    solver_func_file << tiling_file_name_to_content["solver_func"];
    tiling_def_and_tiling_const_file << tiling_file_name_to_content["tiling_def_and_tiling_const"];
  } catch (...) {
    gen_success = false;
  }
  EXPECT_EQ(gen_success, true);
}
