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
#include "e2e_load_isfinite_store.h"
#include "e2e_common.h"
#include "ascir_ops.h"
#include "elewise/unary_bitwidth_change_api_call.h"

std::vector<std::string> splitString(const std::string &input, char delimiter) {
  std::vector<std::string> result;
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    result.push_back(token);
  }

  return result;
}

class LoadIsFiniteStoreTest : public testing::Test {};

namespace {
void ConnectUnaryBitWidthChangeGraph(af::AscGraph &graph, const af::Expression &s0, const af::Expression &s1,
                                     const af::Axis &z0, const af::Axis &z1) {
  af::ascir_op::Data x_op("x", graph);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Isnan isnan_op("isnan");
  graph.AddNode(load_op);
  graph.AddNode(isnan_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, af::ops::One};

  isnan_op.x = load_op.y;
  *isnan_op.y.axis = {z0.id, z1.id};
  *isnan_op.y.repeats = {s0, s1};
  *isnan_op.y.strides = {s1, af::ops::One};
}

void InitUnaryLoadAttrs(const af::AscGraph &graph, const af::Axis &z0, const af::Axis &z1) {
  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {af::ops::One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

void InitUnaryIsnanAttrs(const af::AscGraph &graph, const af::Axis &z0, const af::Axis &z1) {
  auto isnan = graph.FindNode("isnan");
  isnan->attr.api.compute_type = af::ComputeType::kComputeElewise;
  isnan->attr.api.type = af::ApiType::kAPITypeCompute;
  isnan->attr.api.unit = af::ComputeUnit::kUnitVector;
  isnan->attr.sched.loop_axis = z0.id;
  isnan->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  isnan->outputs[0].attr.vectorized_axis = {z1.id};
  isnan->outputs[0].attr.vectorized_strides = {af::ops::One};
  isnan->outputs[0].attr.dtype = ge::DT_INT16;
  isnan->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  isnan->outputs[0].attr.mem.tensor_id = 3;
  isnan->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  isnan->outputs[0].attr.que.id = 2;
  isnan->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

std::string GenerateUnaryBitWidthChangeNoLoopCall() {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  ConnectUnaryBitWidthChangeGraph(graph, s0, s1, z0, z1);
  InitUnaryLoadAttrs(graph, z0, z1);
  InitUnaryIsnanAttrs(graph, z0, z1);

  auto load = graph.FindNode("load");
  auto isnan = graph.FindNode("isnan");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  EXPECT_EQ(tpipe.CollectQues(graph), 0);
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);
  EXPECT_EQ(tpipe.AddTensor(isnan->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::UnaryBitWidthChangeApiCall call("IsnanExtend");
  EXPECT_EQ(call.Init(isnan), 0);
  call.inputs.push_back(&x1);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, {z0.id}, result), 0);
  return result;
}
}  // namespace

TEST_F(LoadIsFiniteStoreTest, UnaryBitWidthChangeNoLoopUsesAlignedSize) {
  const std::string result = GenerateUnaryBitWidthChangeNoLoopCall();
  EXPECT_EQ(result, "IsnanExtend(local_3[0], local_0[0], tmp_buf_0, local_0_actual_size);\n");
}

TEST_F(LoadIsFiniteStoreTest, LoadIsFiniteStoreCodegen) {
  bool gen_success = true;
  af::AscGraph test_graph("load_isfinite_store");
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  LoadIsFiniteStore_BeforeAutofuse(test_graph);
  LoadIsFiniteStore_AfterInferOutput(test_graph);

  std::vector<af::AscGraph> test_impl_graphs = {af::AscGraph("load_isfinite_store_general_0_nil_0_nil")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadIsFiniteStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadIsFiniteStore_AfterScheduler(test_impl_graphs[0]);
  LoadIsFiniteStore_AfterQueBufAlloc(test_impl_graphs[0]);

  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name = parts[0];       // load_isfinite_store_kernel.cpp
  std::string tiling_src_file_name = parts[1];       // load_isfinite_store_tiling.cpp
  std::string tiling_data_src_file_name = parts[2];  // autofuse_tiling_data.h

  try {
    auto codegen = codegen::Codegen(codegen::CodegenOptions{.tiling_lib_path = ATT_SO_NAME,
                                                            .tiling_lib_codegen_symbol = "CodegenTiling",
                                                            .using_att_calc_qbt_size = false});

    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);

    ascir::ScheduledResult schedule_result;
    std::vector<ascir::ScheduledResult> schedule_results{schedule_result};
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.fused_graph_name = af::AscendString("load_isfinite_store");
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
