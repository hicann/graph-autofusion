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
#include "platform_context.h"
#include "runtime_stub.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

namespace {

// 构造仅含一个 Round 节点的测试图，返回该节点
af::AscNodePtr BuildRoundGraph(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  Round round_op("round");
  graph.AddNode(round_op);
  *round_op.y.axis = {z0.id};
  *round_op.y.repeats = {s0};
  *round_op.y.strides = {One};

  auto round = graph.FindNode("round");
  round->attr.api.compute_type = af::ComputeType::kComputeElewise;
  round->attr.api.type = af::ApiType::kAPITypeCompute;
  round->attr.api.unit = af::ComputeUnit::kUnitVector;
  round->attr.sched.loop_axis = z0.id;
  round->outputs[0].attr.vectorized_axis = {z0.id};
  round->outputs[0].attr.vectorized_strides = {One};
  round->outputs[0].attr.dtype = DT_FLOAT;
  round->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  round->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  round->outputs[0].attr.que.id = 1;
  round->outputs[0].attr.opt.merge_scope = af::kIdNone;
  return round;
}

}  // namespace

// 默认：Round 走 regbase Extend 实现
TEST(CodegenKernel, TestRoundExtendCodegenBasicInfo) {
  af::AscGraph graph("test_round_graph");
  auto round = BuildRoundGraph(graph);

  af::ascir::RoundAscIrCodegenImplV2 codegen;
  EXPECT_EQ(codegen.GetApiCallName(), "UnaryApiTmpCall");
  EXPECT_EQ(codegen.GetApiName(), "RoundExtend");

  const auto headers = codegen.LoadApiHeaderFiles(false);
  EXPECT_EQ(headers.size(), 1U);
  EXPECT_EQ(headers[0], "round_reg_base.h");

  const auto include_headers = codegen.IncludeApiHeaderFiles();
  EXPECT_EQ(include_headers.size(), 2U);
  EXPECT_EQ(include_headers[0], "adv_api/math/round.h");
  EXPECT_EQ(include_headers[1], "basic_api/reg_compute/kernel_reg_compute_intf.h");

  const auto tmp_bufs = codegen.CalcTmpBufSize(*round);
  EXPECT_EQ(tmp_bufs.size(), 1U);
}

// 3510 平台经注册表解析到 V2 Round impl：默认走 RoundExtend 路径
class RoundImplPlatformTest : public testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    ge::PlatformContext::GetInstance().SetPlatform("3510");
    ge::RuntimeStub::SetInstance(std::make_shared<ge::RuntimeStubV2Common>());
  }

  void TearDown() override {
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

TEST_F(RoundImplPlatformTest, ResolveByPlatform) {
  auto impl = ascgen_utils::GetAscIrCodegenImpl("Round");
  ASSERT_NE(impl, nullptr);
  EXPECT_EQ(impl->GetApiCallName(), "UnaryApiTmpCall");
  EXPECT_EQ(impl->GetApiName(), "RoundExtend");
  const auto headers = impl->LoadApiHeaderFiles(false);
  EXPECT_EQ(headers.size(), 1U);
  EXPECT_EQ(headers[0], "round_reg_base.h");
}
