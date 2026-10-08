/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascir/generator/v2_ascir_codegen_impl.h"
#include "codegen_kernel.h"
#include "common_utils.h"
#include "optimize/pre_process/pre_process_config.h"
#include "platform_context.h"
#include "runtime_stub.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

namespace {

// 构造 Load -> Rsqrt 的测试图，返回 Rsqrt 节点
af::AscNodePtr BuildRsqrtGraph(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  Data x_op("x", graph);
  Load load_op("load");
  Rsqrt rsqrt_op("rsqrt");
  graph.AddNode(load_op);
  graph.AddNode(rsqrt_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id};
  *load_op.y.axis = {z0.id};
  *load_op.y.repeats = {s0};
  *load_op.y.strides = {One};
  rsqrt_op.x = load_op.y;
  *rsqrt_op.y.axis = {z0.id};
  *rsqrt_op.y.repeats = {s0};
  *rsqrt_op.y.strides = {One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto rsqrt = graph.FindNode("rsqrt");
  rsqrt->attr.api.compute_type = af::ComputeType::kComputeElewise;
  rsqrt->attr.api.type = af::ApiType::kAPITypeCompute;
  rsqrt->attr.api.unit = af::ComputeUnit::kUnitVector;
  rsqrt->attr.sched.loop_axis = z0.id;
  rsqrt->outputs[0].attr.vectorized_axis = {z0.id};
  rsqrt->outputs[0].attr.vectorized_strides = {One};
  rsqrt->outputs[0].attr.dtype = DT_FLOAT;
  rsqrt->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  rsqrt->outputs[0].attr.mem.tensor_id = 1;
  rsqrt->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  rsqrt->outputs[0].attr.que.id = 2;
  rsqrt->outputs[0].attr.opt.merge_scope = af::kIdNone;
  return rsqrt;
}

}  // namespace

// 默认（未配置精度提升黑名单）：Rsqrt 走 regbase Extend 高精度实现
TEST(CodegenKernel, TestRsqrtExtendCodegenBasicInfo) {
  unsetenv("AUTOFUSE_FLAGS");
  af::pre_process::PreProcessConfig::Instance().Reset();

  af::AscGraph graph("test_rsqrt_graph");
  auto rsqrt = BuildRsqrtGraph(graph);

  af::ascir::RsqrtAscIrCodegenImplV2 codegen;
  EXPECT_EQ(codegen.GetApiCallName(), "UnaryApiTmpCall");
  EXPECT_EQ(codegen.GetApiName(), "RsqrtExtend");

  const auto headers = codegen.LoadApiHeaderFiles(false);
  EXPECT_EQ(headers.size(), 1U);
  EXPECT_EQ(headers[0], "rsqrt_reg_base.h");

  const auto include_headers = codegen.IncludeApiHeaderFiles();
  EXPECT_EQ(include_headers.size(), 2U);
  EXPECT_EQ(include_headers[0], "basic_api/kernel_operator_vec_unary_intf.h");
  EXPECT_EQ(include_headers[1], "basic_api/reg_compute/kernel_reg_compute_intf.h");

  const auto tmp_bufs = codegen.CalcTmpBufSize(*rsqrt);
  EXPECT_EQ(tmp_bufs.size(), 1U);
}

// 配置 Rsqrt 精度提升黑名单：codegen 回落内置 Rsqrt（Vector adv_api）快速路径
TEST(CodegenKernel, TestRsqrtBlacklistCodegenBasicInfo) {
  setenv("AUTOFUSE_FLAGS", "--autofuse_enhance_precision_blacklist=Rsqrt", 1);
  af::pre_process::PreProcessConfig::Instance().Reset();

  af::AscGraph graph("test_rsqrt_blacklist_graph");
  auto rsqrt = BuildRsqrtGraph(graph);

  af::ascir::RsqrtAscIrCodegenImplV2 codegen;
  EXPECT_EQ(codegen.GetApiCallName(), "UnaryApiCall");
  EXPECT_EQ(codegen.GetApiName(), "Rsqrt");

  EXPECT_TRUE(codegen.LoadApiHeaderFiles(false).empty());

  // IncludeApiHeaderFiles 与基线一致，黑名单路径不改变生成的头文件依赖
  const auto include_headers = codegen.IncludeApiHeaderFiles();
  EXPECT_EQ(include_headers.size(), 2U);
  EXPECT_EQ(include_headers[0], "basic_api/kernel_operator_vec_unary_intf.h");
  EXPECT_EQ(include_headers[1], "basic_api/reg_compute/kernel_reg_compute_intf.h");

  EXPECT_TRUE(codegen.CalcTmpBufSize(*rsqrt).empty());

  unsetenv("AUTOFUSE_FLAGS");
  af::pre_process::PreProcessConfig::Instance().Reset();
}

// 配置 all 通配精度提升黑名单：Rsqrt 同样回落快速路径
TEST(CodegenKernel, TestRsqrtBlacklistAllCodegenBasicInfo) {
  setenv("AUTOFUSE_FLAGS", "--autofuse_enhance_precision_blacklist=all", 1);
  af::pre_process::PreProcessConfig::Instance().Reset();

  af::AscGraph graph("test_rsqrt_blacklist_all_graph");
  auto rsqrt = BuildRsqrtGraph(graph);

  af::ascir::RsqrtAscIrCodegenImplV2 codegen;
  EXPECT_EQ(codegen.GetApiCallName(), "UnaryApiCall");
  EXPECT_EQ(codegen.GetApiName(), "Rsqrt");
  EXPECT_TRUE(codegen.LoadApiHeaderFiles(false).empty());
  EXPECT_TRUE(codegen.CalcTmpBufSize(*rsqrt).empty());

  unsetenv("AUTOFUSE_FLAGS");
  af::pre_process::PreProcessConfig::Instance().Reset();
}

// 3510 平台经注册表解析到 V2 Rsqrt impl：默认走 RsqrtExtend 高精度路径。
// 黑名单回落路径由上方直接实例化用例覆盖（注册表路径与直接实例化共享同一 impl 实现）
class RsqrtImplPlatformTest : public testing::Test {
 protected:
  void SetUp() override {
    unsetenv("AUTOFUSE_FLAGS");
    ge::PlatformContext::GetInstance().Reset();
    ge::PlatformContext::GetInstance().SetPlatform("3510");
    ge::RuntimeStub::SetInstance(std::make_shared<ge::RuntimeStubV2Common>());
  }

  void TearDown() override {
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

TEST_F(RsqrtImplPlatformTest, ResolveByPlatform) {
  auto impl = ascgen_utils::GetAscIrCodegenImpl("Rsqrt");
  ASSERT_NE(impl, nullptr);
  EXPECT_EQ(impl->GetApiCallName(), "UnaryApiTmpCall");
  EXPECT_EQ(impl->GetApiName(), "RsqrtExtend");
  const auto headers = impl->LoadApiHeaderFiles(false);
  EXPECT_EQ(headers.size(), 1U);
  EXPECT_EQ(headers[0], "rsqrt_reg_base.h");
}
