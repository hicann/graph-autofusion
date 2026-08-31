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
#include <cstdio>
#include "ascir_utils.h"
#include "codegen.h"
#include "e2e_common.h"
#include "e2e_broadcast.h"

using namespace ascir;

int main(int argc, char *argv[]) {
  af::AscGraph graph("load_broadcast_store");
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  LoadBroadcastStore_BeforeAutofuse(graph, 0, ge::DT_FLOAT);

  af::AscGraph impl_graph("load_broadcast_store_general_0_nil_0_nil");
  impl_graph.CopyFrom(graph);
  LoadBroadcastStore_AfterAutofuse(impl_graph, 0, ge::DT_FLOAT);

  std::cout << utils::DebugImplGraphStr(impl_graph) << std::endl;

  codegen::Codegen c(codegen::CodegenOptions{
      .tiling_lib_path = "./libtest_load_broadcast_store_codegen_tiling_gen.so",
      .tiling_lib_codegen_symbol = "CodegenTiling",
      .using_att_calc_qbt_size = false,
  });
  ascir::ScheduledResult schedule_result;
  std::vector<ascir::ScheduledResult> schedule_results{schedule_result};
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString("load_broadcast_store");
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  InitScheduleResultsByImplGraphs({impl_graph}, fused_schedule_result);
  codegen::CodegenResult result;
  if (c.Generate(fused_schedule_result, result) != af::SUCCESS) {
    std::cout << "graph generate failed" << std::endl;
  }

  std::cout << "***************************************20250822***0*****" << std::endl;
  std::cout << tilig_stub << RemoveSubDirInclude(result.kernel);
  std::fstream kernel_file("load_broadcast_store_kernel.cpp", std::ios::out);
  std::cout << "***************************************20250822****1****" << std::endl;
  std::cout << result.tiling;
  std::fstream tiling_file("load_broadcast_store_tiling.cpp", std::ios::out);
  std::fstream tiling_head_file("load_broadcast_store_tiling.h", std::ios::out);
  std::cout << "***************************************20250822*****2***" << std::endl;
  std::cout << result.tiling_data;
  std::fstream tiling_data_file("autofuse_tiling_data.h", std::ios::out);
  std::cout << "***************************************20250822*****3***" << std::endl;

  kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
  tiling_file << result.tiling;
  tiling_head_file << "";
  tiling_data_file << result.tiling_data;

  af::AscGraph graph_int64("load_broadcast_store_int64");
  std::string tilig_stub_int64 = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  LoadBroadcastStore_BeforeAutofuse(graph_int64, 0, ge::DT_INT64);

  af::AscGraph impl_graph_int64("load_broadcast_store_int64_general_0_nil_0_nil");
  impl_graph_int64.CopyFrom(graph_int64);
  LoadBroadcastStore_AfterAutofuse(impl_graph_int64, 0, ge::DT_INT64);

  std::cout << utils::DebugImplGraphStr(impl_graph_int64) << std::endl;

  codegen::Codegen c_int64(codegen::CodegenOptions{
      .tiling_lib_path = "./libtest_load_broadcast_store_codegen_tiling_gen.so",
      .tiling_lib_codegen_symbol = "CodegenTiling",
      .using_att_calc_qbt_size = false,
  });
  ascir::ScheduledResult schedule_result_int64;
  std::vector<ascir::ScheduledResult> schedule_results_int64{schedule_result_int64};
  ascir::FusedScheduledResult fused_schedule_result_int64;
  fused_schedule_result_int64.fused_graph_name = af::AscendString("load_broadcast_store_int64");
  fused_schedule_result_int64.node_idx_to_scheduled_results.push_back(schedule_results_int64);
  InitScheduleResultsByImplGraphs({impl_graph_int64}, fused_schedule_result_int64);
  if (c_int64.Generate(fused_schedule_result_int64, result) != af::SUCCESS) {
    std::cout << "graph_uint8 generate failed" << std::endl;
  }

  std::fstream kernel_file_int64("load_broadcast_store_int64_kernel.cpp", std::ios::out);
  std::fstream tiling_data_file_int64("autofuse_tiling_data.h", std::ios::out);
  std::fstream tiling_head_file_int64("load_broadcast_store_int64_tiling.h", std::ios::out);

  kernel_file_int64 << tilig_stub_int64 << RemoveSubDirInclude(result.kernel);
  tiling_head_file_int64 << "";
  tiling_data_file_int64 << result.tiling_data;

  af::AscGraph graph_uint8("load_broadcast_store_uint8");
  std::string tilig_stub_uint8 = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  LoadBroadcastStore_BeforeAutofuse(graph_uint8, 0, ge::DT_UINT8);

  af::AscGraph impl_graph_uint8("load_broadcast_store_uint8_general_0_nil_0_nil");
  impl_graph_uint8.CopyFrom(graph_uint8);
  LoadBroadcastStore_AfterAutofuse(impl_graph_uint8, 0, ge::DT_UINT8);

  std::cout << utils::DebugImplGraphStr(impl_graph_uint8) << std::endl;
  std::cout << "***************************************20250822*****5***" << std::endl;

  codegen::Codegen c_uint8(codegen::CodegenOptions{
      .tiling_lib_path = "./libtest_load_broadcast_store_codegen_tiling_gen.so",
      .tiling_lib_codegen_symbol = "CodegenTiling",
      .using_att_calc_qbt_size = false,
  });
  ascir::ScheduledResult schedule_result_uint8;
  std::vector<ascir::ScheduledResult> schedule_results_uint8{schedule_result_uint8};
  ascir::FusedScheduledResult fused_schedule_result_uint8;
  fused_schedule_result_uint8.fused_graph_name = af::AscendString("load_broadcast_store_uint8");
  fused_schedule_result_uint8.node_idx_to_scheduled_results.push_back(schedule_results_uint8);
  InitScheduleResultsByImplGraphs({impl_graph_uint8}, fused_schedule_result_uint8);
  if (c_uint8.Generate(fused_schedule_result_uint8, result) != af::SUCCESS) {
    std::cout << "graph_uint8 generate failed" << std::endl;
  }

  std::fstream kernel_file_uint8("load_broadcast_store_uint8_kernel.cpp", std::ios::out);
  std::fstream tiling_data_file_uint8("autofuse_tiling_data.h", std::ios::out);
  std::fstream tiling_head_file_uint8("load_broadcast_store_uint8_tiling.h", std::ios::out);

  kernel_file_uint8 << tilig_stub_uint8 << RemoveSubDirInclude(result.kernel);
  tiling_head_file_uint8 << "";
  tiling_data_file_uint8 << result.tiling_data;

  std::cout << "***************************************20250822*****6***" << std::endl;
  return 0;
}
