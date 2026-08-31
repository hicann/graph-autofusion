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

#include "ascir_utils.h"
#include "codegen.h"
#include "e2e_common.h"
#include "e2e_broadcast.h"

using namespace ascir;
int main(int argc, char *argv[]) {
  // BroadCast_A11toABC
  af::AscGraph graph_multi_axis("load_broadcast_multi_axis_store");
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  std::vector<af::AscGraph> impl_graphs_multi_axis;
  ConstructMultiAxisGraph(graph_multi_axis, impl_graphs_multi_axis, {false, false, false, true, true},
                          "load_broadcast_mutli_axis_store_general_1_0_nil_0_nil");
  std::cout << utils::DebugImplGraphStr(impl_graphs_multi_axis[0]) << std::endl;

  codegen::Codegen c(codegen::CodegenOptions{
      .tiling_lib_path = "./libtest_load_broadcast_multi_axis_store_codegen_tiling_gen.so",
      .tiling_lib_codegen_symbol = "CodegenTiling",
      .using_att_calc_qbt_size = false,
  });

  std::fstream kernel_file("load_broadcast_multi_axis_store_kernel.cpp", std::ios::out);
  std::fstream tiling_file("load_broadcast_multi_axis_store_tiling.cpp", std::ios::out);
  std::fstream tiling_head_file("load_broadcast_multi_axis_store_tiling.h", std::ios::out);
  std::fstream tiling_data_file("autofuse_tiling_data.h", std::ios::out);

  ascir::ScheduledResult schedule_result;
  std::vector<ascir::ScheduledResult> schedule_results{schedule_result};
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = ge::AscendString("load_broadcast_multi_axis_store");
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  InitScheduleResultsByImplGraphs(impl_graphs_multi_axis, fused_schedule_result);
  codegen::CodegenResult result;
  c.Generate(fused_schedule_result, result);
  kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
  tiling_data_file << result.tiling_data;
  tiling_head_file << "";

  // BroadCast_11CtoABC
  af::AscGraph graph_multi_axis_2("load_broadcast_multi_axis_store_2");
  std::string tilig_stub_2 = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  std::vector<af::AscGraph> impl_graphs_multi_axis_2;
  ConstructMultiAxisGraph(graph_multi_axis_2, impl_graphs_multi_axis_2, {false, true, true, false, false},
                          "load_broadcast_mutli_axis_store_general_2_0_nil_0_nil");
  std::cout << utils::DebugImplGraphStr(impl_graphs_multi_axis_2[0]) << std::endl;

  codegen::Codegen c_2(codegen::CodegenOptions{
      .tiling_lib_path = "./libtest_load_broadcast_multi_axis_store_codegen_tiling_gen.so",
      .tiling_lib_codegen_symbol = "CodegenTiling",
      .using_att_calc_qbt_size = false,
  });

  std::fstream kernel_file_2("load_broadcast_multi_axis_store_2_kernel.cpp", std::ios::out);
  std::fstream tiling_file_2("load_broadcast_multi_axis_store_2_tiling.cpp", std::ios::out);
  std::fstream tiling_head_file_2("load_broadcast_multi_axis_store_2_tiling.h", std::ios::out);
  std::fstream tiling_data_file_2("autofuse_tiling_data.h", std::ios::out);

  ascir::ScheduledResult schedule_result_2;
  std::vector<ascir::ScheduledResult> schedule_results_2{schedule_result_2};
  ascir::FusedScheduledResult fused_schedule_result_2;
  fused_schedule_result_2.fused_graph_name = ge::AscendString("load_broadcast_multi_axis_store_2");
  fused_schedule_result_2.node_idx_to_scheduled_results.push_back(schedule_results_2);
  InitScheduleResultsByImplGraphs(impl_graphs_multi_axis_2, fused_schedule_result_2);
  c_2.Generate(fused_schedule_result_2, result);
  kernel_file_2 << tilig_stub_2 << RemoveSubDirInclude(result.kernel);
  tiling_data_file_2 << result.tiling_data;
  tiling_head_file_2 << "";

  // BroadCast_A1CtoABC
  af::AscGraph graph_multi_axis_3("load_broadcast_multi_axis_store_3");
  std::string tilig_stub_3 = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  std::vector<af::AscGraph> impl_graphs_multi_axis_3;
  ConstructMultiAxisGraph(graph_multi_axis_3, impl_graphs_multi_axis_3, {false, false, false, true, false},
                          "load_broadcast_mutli_axis_store_general_3_0_nil_0_nil");
  std::cout << utils::DebugImplGraphStr(impl_graphs_multi_axis_3[0]) << std::endl;

  codegen::Codegen c_3(codegen::CodegenOptions{
      .tiling_lib_path = "./libtest_load_broadcast_multi_axis_store_codegen_tiling_gen.so",
      .tiling_lib_codegen_symbol = "CodegenTiling",
      .using_att_calc_qbt_size = false,
  });

  std::fstream kernel_file_3("load_broadcast_multi_axis_store_3_kernel.cpp", std::ios::out);
  std::fstream tiling_file_3("load_broadcast_multi_axis_store_3_tiling.cpp", std::ios::out);
  std::fstream tiling_head_file_3("load_broadcast_multi_axis_store_3_tiling.h", std::ios::out);
  std::fstream tiling_data_file_3("autofuse_tiling_data.h", std::ios::out);

  ascir::ScheduledResult schedule_result_3;
  std::vector<ascir::ScheduledResult> schedule_results_3{schedule_result_3};
  ascir::FusedScheduledResult fused_schedule_result_3;
  fused_schedule_result_3.fused_graph_name = ge::AscendString("load_broadcast_multi_axis_store_3");
  fused_schedule_result_3.node_idx_to_scheduled_results.push_back(schedule_results_3);
  InitScheduleResultsByImplGraphs(impl_graphs_multi_axis_3, fused_schedule_result_3);
  c_3.Generate(fused_schedule_result_3, result);
  kernel_file_3 << tilig_stub_3 << RemoveSubDirInclude(result.kernel);
  tiling_data_file_3 << result.tiling_data;
  tiling_head_file_3 << "";

  return 0;
}
