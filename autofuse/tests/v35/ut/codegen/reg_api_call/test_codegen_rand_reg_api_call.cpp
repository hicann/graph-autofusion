/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascir/generator/v2_ascir_codegen_impl.h"
#include "codegen_kernel.h"
#include "common_utils.h"
#include "utils/api_call_factory.h"
#include "unary_output_api_call.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

namespace codegen {
namespace {

// 创建 Rand 算子的测试图
void BuildRandGraph(af::AscGraph &graph, const af::Expression &s0, const af::Axis &z0, ge::DataType dtype) {
  Rand rand_op("Rand");
  graph.AddNode(rand_op);

  *rand_op.y.axis = {z0.id};
  *rand_op.y.repeats = {s0};
  *rand_op.y.strides = {One};
}

void InitRandAttrs(af::AscGraph &graph, ge::DataType dtype, const af::Axis &z0) {
  auto rand = graph.FindNode("Rand");
  rand->attr.api.compute_type = af::ComputeType::kComputeElewise;
  rand->attr.api.type = af::ApiType::kAPITypeCompute;
  rand->attr.api.unit = af::ComputeUnit::kUnitVector;
  rand->attr.sched.loop_axis = z0.id;

  auto &rand_attr = rand->outputs[0].attr;
  rand_attr.vectorized_axis = {z0.id};
  rand_attr.vectorized_strides = {One};
  rand_attr.dtype = dtype;
  rand_attr.mem.position = af::Position::kPositionVecOut;
  rand_attr.mem.tensor_id = 0;
  rand_attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  rand_attr.que.id = 1;
  rand_attr.opt.merge_scope = af::kIdNone;
}

// 创建 Randn 算子的测试图
void BuildRandnGraph(af::AscGraph &graph, const af::Expression &s0, const af::Axis &z0, ge::DataType dtype) {
  Randn randn_op("Randn");
  graph.AddNode(randn_op);

  *randn_op.y.axis = {z0.id};
  *randn_op.y.repeats = {s0};
  *randn_op.y.strides = {One};
}

void InitRandnAttrs(af::AscGraph &graph, ge::DataType dtype, const af::Axis &z0) {
  auto randn = graph.FindNode("Randn");
  randn->attr.api.compute_type = af::ComputeType::kComputeElewise;
  randn->attr.api.type = af::ApiType::kAPITypeCompute;
  randn->attr.api.unit = af::ComputeUnit::kUnitVector;
  randn->attr.sched.loop_axis = z0.id;

  auto &randn_attr = randn->outputs[0].attr;
  randn_attr.vectorized_axis = {z0.id};
  randn_attr.vectorized_strides = {One};
  randn_attr.dtype = dtype;
  randn_attr.mem.position = af::Position::kPositionVecOut;
  randn_attr.mem.tensor_id = 0;
  randn_attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  randn_attr.que.id = 1;
  randn_attr.opt.merge_scope = af::kIdNone;
}

}  // namespace

// 测试 Rand Codegen 基本信息
TEST(CodegenKernel, TestRandCodegenBasicInfo) {
  af::AscGraph graph("test_rand_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandGraph(graph, s0, z0, DT_FLOAT);
  InitRandAttrs(graph, DT_FLOAT, z0);

  auto rand = graph.FindNode("Rand");

  // 创建 Codegen 实例并测试基本信息
  af::ascir::RandAscIrCodegenImplV2 codegen;

  // 测试 GetApiName - Rand 返回 "Rand"
  std::string api_name = codegen.GetApiName();
  EXPECT_EQ(api_name, "Rand");

  // 测试 GetApiCallName
  std::string api_call_name = codegen.GetApiCallName();
  EXPECT_EQ(api_call_name, "UnaryOutputApiCall");

  // 测试 LoadApiHeaderFiles
  auto headers = codegen.LoadApiHeaderFiles(false);
  EXPECT_EQ(headers.size(), 1);
  EXPECT_EQ(headers[0], "random_reg_base.h");

  // 测试 IncludeApiHeaderFiles（应该包含 AscendC Philox 头文件）
  auto include_headers = codegen.IncludeApiHeaderFiles();
  EXPECT_EQ(include_headers.size(), 1);
  EXPECT_EQ(include_headers[0], "adv_api/math/philox.h");
}

// 测试 Randn Codegen 基本信息
TEST(CodegenKernel, TestRandnCodegenBasicInfo) {
  af::AscGraph graph("test_randn_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandnGraph(graph, s0, z0, DT_UINT32);
  InitRandnAttrs(graph, DT_UINT32, z0);

  auto randn = graph.FindNode("Randn");

  // 创建 Codegen 实例并测试基本信息
  af::ascir::RandnAscIrCodegenImplV2 codegen;

  // 测试 GetApiName - Randn 也返回 "Rand"
  std::string api_name = codegen.GetApiName();
  EXPECT_EQ(api_name, "Rand");

  // 测试 GetApiCallName
  std::string api_call_name = codegen.GetApiCallName();
  EXPECT_EQ(api_call_name, "UnaryOutputApiCall");

  // 测试 LoadApiHeaderFiles
  auto headers = codegen.LoadApiHeaderFiles(false);
  EXPECT_EQ(headers.size(), 1);
  EXPECT_EQ(headers[0], "random_reg_base.h");

  // 测试 IncludeApiHeaderFiles（应该包含 AscendC Philox 头文件）
  auto include_headers = codegen.IncludeApiHeaderFiles();
  EXPECT_EQ(include_headers.size(), 1);
  EXPECT_EQ(include_headers[0], "adv_api/math/philox.h");
}

// 测试 Rand 不同 dtype 支持
TEST(CodegenKernel, TestRandDifferentDtypes) {
  std::vector<ge::DataType> supported_dtypes = {DT_FLOAT};

  for (auto dtype : supported_dtypes) {
    af::AscGraph graph("test_rand_dtype_graph");

    auto s0 = graph.CreateSizeVar("s0");
    auto z0 = graph.CreateAxis("z0", s0);
    BuildRandGraph(graph, s0, z0, dtype);
    InitRandAttrs(graph, dtype, z0);

    auto rand = graph.FindNode("Rand");
    EXPECT_EQ(rand->outputs[0].attr.dtype, dtype);
  }
}

// 测试 Randn 不同 dtype 支持
TEST(CodegenKernel, TestRandnDifferentDtypes) {
  std::vector<ge::DataType> supported_dtypes = {DT_UINT32, DT_INT32};

  for (auto dtype : supported_dtypes) {
    af::AscGraph graph("test_randn_dtype_graph");

    auto s0 = graph.CreateSizeVar("s0");
    auto z0 = graph.CreateAxis("z0", s0);
    BuildRandnGraph(graph, s0, z0, dtype);
    InitRandnAttrs(graph, dtype, z0);

    auto randn = graph.FindNode("Randn");
    EXPECT_EQ(randn->outputs[0].attr.dtype, dtype);
  }
}

// 测试 Rand 生成的 API 调用字符串（简化版）
TEST(CodegenKernel, TestRandApiCallStringGeneration) {
  af::AscGraph graph("test_rand_api_call_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandGraph(graph, s0, z0, DT_FLOAT);
  InitRandAttrs(graph, DT_FLOAT, z0);

  auto rand = graph.FindNode("Rand");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(rand->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddSizeVar(af::SizeVar(s0));

  // 使用 UnaryOutputApiCall 来生成调用字符串
  codegen::UnaryOutputApiCall call("Rand");
  EXPECT_EQ(call.Init(rand), 0);

  std::string result;
  call.Generate(tpipe, {}, result);

  // 验证生成的字符串包含关键信息
  EXPECT_NE(result.find("Rand"), std::string::npos);
  EXPECT_NE(result.find("local_0"), std::string::npos);
  EXPECT_NE(result.find("local_0_actual_size"), std::string::npos);
}

// 测试 Randn 生成的 API 调用字符串
TEST(CodegenKernel, TestRandnApiCallStringGeneration) {
  af::AscGraph graph("test_randn_api_call_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandnGraph(graph, s0, z0, DT_UINT32);
  InitRandnAttrs(graph, DT_UINT32, z0);

  auto randn = graph.FindNode("Randn");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(randn->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddSizeVar(af::SizeVar(s0));

  // 使用 UnaryOutputApiCall 来生成调用字符串
  codegen::UnaryOutputApiCall call("Rand");
  EXPECT_EQ(call.Init(randn), 0);

  std::string result;
  call.Generate(tpipe, {}, result);

  // 验证生成的字符串包含关键信息
  EXPECT_NE(result.find("Rand"), std::string::npos);  // Randn 也调用 Rand
  EXPECT_NE(result.find("local_0"), std::string::npos);
  EXPECT_NE(result.find("local_0_actual_size"), std::string::npos);
}

}  // namespace codegen
