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
#include "e2e_load_abs_store.h"
#include "e2e_common.h"
#include "ascgraph_info_complete.h"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

std::vector<std::string> splitString(const std::string &input, char delimiter) {
  std::vector<std::string> result;
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    result.push_back(token);
  }

  return result;
}

ascir::FusedScheduledResult GenTestCase(int32_t num_groups) {
  ascir::FusedScheduledResult fused_schedule_result;
  ascir::ScheduledResult schedule_result;
  schedule_result.schedule_groups.resize(num_groups);
  int64_t tensor_id_gen = 0;
  for (size_t i = 0U; i < schedule_result.schedule_groups.size(); ++i) {
    auto &group = schedule_result.schedule_groups[i];
    group.impl_graphs = {af::AscGraph("dynamic_inputs_and_outputs_general_0_nil_0_nil")};
    LoadAbsStore_BeforeAutofuse(group.impl_graphs[0]);
    LoadAbsStore_AfterInferOutput(group.impl_graphs[0]);
    LoadAbsStore_AfterGetApiInfo(group.impl_graphs[0]);
    LoadAbsStore_AfterScheduler(group.impl_graphs[0]);
    LoadAbsStore_AfterQueBufAlloc(group.impl_graphs[0]);
    for (auto node : group.impl_graphs[0].GetAllNodes()) {
      node->outputs[0].attr.mem.tensor_id = tensor_id_gen++;
    }
    auto x = group.impl_graphs[0].FindNode("x");
    auto &x_attr = reinterpret_cast<af::AscDataIrAttrDef &>(*x->attr.ir_attr);
    x_attr.SetIndex(i);
    x->GetOpDesc()->SetName("x_" + std::to_string(i));
    auto y = group.impl_graphs[0].FindNode("y");
    auto &y_attr = reinterpret_cast<af::AscDataIrAttrDef &>(*y->attr.ir_attr);
    y_attr.SetIndex(i);
    y->GetOpDesc()->SetName("y_" + std::to_string(i));
    fused_schedule_result.input_nodes.emplace_back(x);
    fused_schedule_result.output_nodes.emplace_back(y);
  }
  std::vector<ascir::ScheduledResult> schedule_results{schedule_result};
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  return fused_schedule_result;
}

class DynamicInputsAndOutputsST : public testing::Test {};

TEST_F(DynamicInputsAndOutputsST, DynamicInputsAndOutputsCodegen) {
  bool gen_success = true;
  af::AscGraph test_graph("dynamic_inputs_and_outputs");
  std::string tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name = parts[0];       // load_cast_store_kernel.cpp
  std::string tiling_src_file_name = parts[1];       // load_cast_store_tiling.cpp
  std::string tiling_data_src_file_name = parts[2];  // autofuse_tiling_data.h

  try {
    auto codegen = codegen::Codegen(codegen::CodegenOptions{.tiling_lib_path = ATT_SO_NAME,
                                                            .tiling_lib_codegen_symbol = "CodegenTiling",
                                                            .using_att_calc_qbt_size = false});

    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);

    auto fused_schedule_result = GenTestCase(10);
    fused_schedule_result.fused_graph_name = af::AscendString("dynamic_inputs_and_outputs");
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(fused_schedule_result, result), 0);
    EXPECT_TRUE(RemoveSubDirInclude(result.kernel).find("kernel_operator_list_tensor_intf.h") == std::string::npos);
    // cpu编译不支持多进程并法编译，修改kernel为单kernel编译的方式
    std::stringstream ss_in;
    std::stringstream ss_out;
    ss_in << RemoveSubDirInclude(result.kernel);
    std::string line;
    while (std::getline(ss_in, line)) {
      if (line.find("#if TILING") != std::string::npos) {
        ss_out << "#if 1" << std::endl;
      } else {
        ss_out << line << std::endl;
      }
    }
    kernel_file << tiling_stub << ss_out.str();
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;

    // use list tensor desc
    auto fused_schedule_result_64_inputs = GenTestCase(64);
    EXPECT_EQ(codegen.Generate(fused_schedule_result_64_inputs, result), 0);
    EXPECT_TRUE(RemoveSubDirInclude(result.kernel).find("kernel_operator_list_tensor_intf.h") != std::string::npos);
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(DynamicInputsAndOutputsST, FrontendShapeAbiKeepsNaturalKsOrderWhenImplDropsSymbols) {
  auto fused_schedule_result = GenTestCase(1);
  fused_schedule_result.fused_graph_name = af::AscendString("dynamic_inputs_and_outputs_frontend_shape_abi");
  af::AscGraph original_asc_graph("dynamic_inputs_and_outputs_frontend_shape_abi_original");
  original_asc_graph.CreateSizeVar("ks10");
  original_asc_graph.CreateSizeVar("ks2");
  original_asc_graph.CreateSizeVar("ks0");
  original_asc_graph.CreateSizeVar("ks1");
  ASSERT_EQ(optimize::AscGraphInfoComplete::CollectFrontendShapeVars(original_asc_graph,
                                                                     fused_schedule_result.frontend_shape_vars),
            af::SUCCESS);
  ASSERT_EQ(optimize::AscGraphInfoComplete::NormalizeFrontendShapeVars(fused_schedule_result.frontend_shape_vars),
            af::SUCCESS);
  // The optimized implementation graph is intentionally modeled as retaining only one symbol.
  fused_schedule_result.origin_vars = {af::Symbol("ks0")};

  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;
  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);

  const auto signature_begin = result.tiling.find("extern \"C\" int64_t AutofuseTiling(");
  ASSERT_NE(signature_begin, std::string::npos);
  const auto signature_end = result.tiling.find(")", signature_begin);
  ASSERT_NE(signature_end, std::string::npos);
  const auto signature = result.tiling.substr(signature_begin, signature_end - signature_begin);

  size_t previous = 0U;
  for (const auto &symbol : {std::string("ks0"), std::string("ks1"), std::string("ks2"), std::string("ks10")}) {
    const auto current = signature.find(symbol);
    ASSERT_NE(current, std::string::npos) << symbol << " missing from generated AutofuseTiling ABI";
    EXPECT_GE(current, previous) << "generated symbol order does not match frontend order";
    previous = current;
  }
}
