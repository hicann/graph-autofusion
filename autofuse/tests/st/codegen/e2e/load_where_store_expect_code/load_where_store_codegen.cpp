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

#include "codegen.h"
#include "e2e_load_where_store.h"
#include "e2e_common.h"

std::vector<std::string> splitString(const std::string &input, char delimiter) {
  std::vector<std::string> result;
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    result.push_back(token);
  }

  return result;
}

class LoadWhereStoreTest : public testing::Test {};

TEST_F(LoadWhereStoreTest, LoadWhereStoreCodegen) {
  bool gen_success = true;
  af::AscGraph test_graph("load_where_store");
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  LoadWhereStore_BeforeAutofuse(test_graph);
  LoadWhereStore_AfterInferOutput(test_graph);

  std::vector<af::AscGraph> test_impl_graphs = {af::AscGraph("load_where_store_general_0_nil_0_nil")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadWhereStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadWhereStore_AfterScheduler(test_impl_graphs[0]);
  LoadWhereStore_AfterQueBufAlloc(test_impl_graphs[0]);

  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name = parts[0];       // load_where_store_kernel.cpp
  std::string tiling_src_file_name = parts[1];       // load_where_store_tiling.cpp
  std::string tiling_data_src_file_name = parts[2];  // autofuse_tiling_data.h

  try {
    auto codegen = codegen::Codegen(codegen::CodegenOptions{.tiling_lib_path = ATT_SO_NAME,
                                                            .tiling_lib_codegen_symbol = "CodegenTiling",
                                                            .using_att_calc_qbt_size = false});

    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);

    ascir::ScheduledResult schedule_result1;
    ascir::ScheduledResult schedule_result2;
    std::vector<ascir::ScheduledResult> schedule_results{schedule_result1, schedule_result2};
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.fused_graph_name = af::AscendString("load_where_store");
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    InitScheduleResultsByImplGraphs(test_impl_graphs, fused_schedule_result);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(fused_schedule_result, result), 0);
    kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
