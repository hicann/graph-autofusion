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
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "codegen.h"
#include "optimize.h"
#include "share_graph.h"
#include "backend_common.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascgraph_info_complete.h"
#include "ascgen_log.h"
#include "common/platform_context.h"
#include "runtime_stub.h"
#include "common_utils.h"

class TestBackendMatmulEleBrc : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<af::RuntimeStubV2>();
    ge::RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
  }
};

TEST_F(TestBackendMatmulEleBrc, MatmulEleBrcCodegen) {
  bool gen_success = true;
  const std::map<std::string, std::string> shape_info;
  auto graph = ascir::ShareGraph::LoadMatmulElewiseBrcFusedGraph();

  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name =
      "matmul_elemwise_brc_test_kernel_ub.cpp";  // matmul_elemwise_brc_test_kernel_ub.cpp
  std::string tiling_src_file_name =
      "matmul_elemwise_brc_test_tiling_ub.cpp";      // matmul_elemwise_brc_test_tiling_ub.cpp
  std::string tiling_data_src_file_name = parts[2];  // autofuse_tiling_data.h

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
    codegen::CodegenResult result;
    ::ascir::FusedScheduledResult ub_schedule_result = fused_schedule_result;
    ::ascir::FusedScheduledResult common_schedule_result = fused_schedule_result;
    if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
      // 过滤CVFusion的UBResult ub模板结果
      ascgen_utils::FilterCVFusionUBResult(ub_schedule_result);
      // 过滤CVFusion的CommonResult 兜底模板结果
      ascgen_utils::FilterCVFusionCommonResult(common_schedule_result);
    }

    // 分别生成ub和common模板的kernel和tiling
    EXPECT_EQ(codegen.Generate(shape_info, ub_schedule_result, result), 0);
    kernel_file << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;

    // 校验RemoveSubDirInclude(result.kernel)中是否包含IncludeMatmulHeadFiles方法返回的所有头文件内容
    std::vector<std::string> expected_headers = {"#include \"arch35/mat_mul_v3_tiling_key_public.h\"",
                                                 "#include \"arch35/mat_mul_tiling_data.h\"",
                                                 "#include \"mat_mul_v3_common.h\"",
                                                 "#include \"arch35/mat_mul_asw_block.h\"",
                                                 "#include \"arch35/mat_mul_asw_kernel.h\"",
                                                 "#include \"arch35/mat_mul_stream_k_block.h\"",
                                                 "#include \"arch35/mat_mul_stream_k_kernel.h\"",
                                                 "#include \"arch35/mat_mul_v3_full_load_kernel_helper.h\"",
                                                 "#include \"arch35/mat_mul_full_load.h\"",
                                                 "#include \"arch35/mm_extension_interface/mm_copy_cube_out.h\"",
                                                 "#include \"arch35/mm_extension_interface/mm_custom_mm_policy.h\"",
                                                 "#include \"arch35/mat_mul_fixpipe_opti.h\"",
                                                 "#include \"arch35/block_scheduler_aswt.h\"",
                                                 "#include \"arch35/block_scheduler_streamk.h\"",
                                                 "#include \"arch35/mat_mul_streamk_basic_cmct.h\"",
                                                 "#include \"arch35/mat_mul_fixpipe_opti_basic_cmct.h\"",
                                                 "#include \"arch35/mat_mul_input_k_eq_zero_clear_output.h\""};

    for (const auto &header : expected_headers) {
      EXPECT_NE(RemoveSubDirInclude(result.kernel).find(header), std::string::npos)
          << "Expected header not found in kernel: " << header;
    }

    kernel_src_file_name = "matmul_elemwise_brc_test_kernel_common.cpp";  // matmul_elemwise_brc_test_kernel_common.cpp
    tiling_src_file_name = "matmul_elemwise_brc_test_tiling_common.cpp";  // matmul_elemwise_brc_test_tiling_common.cpp
    tiling_data_src_file_name = parts[2];                                 // autofuse_tiling_data.h
    std::fstream kernel_file_common(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file_common(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file_common(tiling_data_src_file_name, std::ios::out);
    codegen::CodegenResult result_common;
    EXPECT_EQ(codegen.Generate(shape_info, common_schedule_result, result_common), 0);
    kernel_file_common << RemoveSubDirInclude(result_common.kernel);
    tiling_file_common << result_common.tiling;
    tiling_data_file_common << result_common.tiling_data;

    // 校验result_common.kernel中是否包含IncludeMatmulHeadFiles方法返回的所有头文件内容
    for (const auto &header : expected_headers) {
      EXPECT_NE(RemoveSubDirInclude(result_common.kernel).find(header), std::string::npos)
          << "Expected header not found in common kernel: " << header;
    }
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
