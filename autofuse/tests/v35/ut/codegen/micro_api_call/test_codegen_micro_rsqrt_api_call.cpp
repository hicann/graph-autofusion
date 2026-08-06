/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "codegen_kernel.h"
#include "micro_api_call/micro_api_call_factory.h"
#include "micro_api_call/micro_rsqrt_api_call.h"
#include "platform_context.h"
#include "runtime_stub.h"

namespace codegen {
namespace {
struct RsqrtMicroFixture {
  af::AscGraph graph{"micro_rsqrt"};
  af::AscNodePtr rsqrt;
  TensorManager tensor_manager;
  Tiler tiler;
  TPipe tpipe{"tpipe", tiler};

  RsqrtMicroFixture(ge::DataType input_dtype = ge::DT_FLOAT, ge::DataType output_dtype = ge::DT_FLOAT) {
    const auto size = graph.CreateSizeVar(128);
    const auto axis = graph.CreateAxis("axis", size);

    af::ascir_op::Data data("data", graph);
    data.ir_attr.SetIndex(0);

    af::ascir_op::Load load("load");
    load.x = data.y;
    load.y.dtype = input_dtype;
    *load.y.axis = {axis.id};
    *load.y.repeats = {size};
    *load.y.strides = {af::sym::kSymbolOne};
    *load.y.vectorized_axis = {axis.id};
    *load.y.vectorized_strides = {af::sym::kSymbolOne};

    af::ascir_op::Rsqrt rsqrt_op("rsqrt");
    rsqrt_op.x = load.y;
    rsqrt_op.y.dtype = output_dtype;
    *rsqrt_op.y.axis = {axis.id};
    *rsqrt_op.y.repeats = {size};
    *rsqrt_op.y.strides = {af::sym::kSymbolOne};
    *rsqrt_op.y.vectorized_axis = {axis.id};
    *rsqrt_op.y.vectorized_strides = {af::sym::kSymbolOne};

    rsqrt = graph.FindNode("rsqrt");
    auto load_node = graph.FindNode("load");
    load_node->outputs[0].attr.mem.tensor_id = 0;
    rsqrt->outputs[0].attr.mem.tensor_id = 1;

    std::string input_dtype_name;
    EXPECT_EQ(Tensor::DtypeName(input_dtype, input_dtype_name), af::SUCCESS);
    std::string output_dtype_name;
    EXPECT_EQ(Tensor::DtypeName(output_dtype, output_dtype_name), af::SUCCESS);
    EXPECT_EQ(tensor_manager.AddTensor(MicroApiTensor(load_node->outputs[0], input_dtype_name)), af::SUCCESS);
    EXPECT_EQ(tensor_manager.AddTensor(MicroApiTensor(rsqrt->outputs[0], output_dtype_name)), af::SUCCESS);
  }
};

class MicroRsqrtApiCallTest : public testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
    ge::RuntimeStub::SetInstance(stub_v2);
  }

  void TearDown() override {
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};
}  // namespace

TEST_F(MicroRsqrtApiCallTest, GeneratesFloatRsqrtInstructionSequence) {
  RsqrtMicroFixture fixture;
  MicroRsqrtApiCall call("Rsqrt");
  ASSERT_EQ(call.Init(fixture.rsqrt), af::SUCCESS);
  call.AddInput(0);
  call.AddOutput(1);

  CallParam param{"p_reg", "", "float"};
  std::string result;
  ASSERT_EQ(call.Generate(fixture.tensor_manager, fixture.tpipe, param, result), af::SUCCESS);
  EXPECT_EQ(result,
            "AscendC::MicroAPI::RegTensor<float> vreg_1_rsqrt_one;\n"
            "AscendC::MicroAPI::MaskReg vreg_1_rsqrt_negative_mask;\n"
            "AscendC::MicroAPI::Duplicate(vreg_1_rsqrt_one, static_cast<float>(1.0), p_reg);\n"
            "AscendC::MicroAPI::CompareScalar<float, AscendC::CMPMODE::LT>(vreg_1_rsqrt_negative_mask, vreg_0, "
            "static_cast<float>(0.0), p_reg);\n"
            "AscendC::MicroAPI::Sqrt(vreg_1, vreg_0, p_reg);\n"
            "AscendC::MicroAPI::Div(vreg_1_rsqrt_one, vreg_1_rsqrt_one, vreg_1, p_reg);\n"
            "AscendC::MicroAPI::Select(vreg_1, vreg_1, vreg_1_rsqrt_one, vreg_1_rsqrt_negative_mask);\n");
  EXPECT_EQ(result.find("Sqrt(vreg_0,"), std::string::npos);
}

TEST_F(MicroRsqrtApiCallTest, FactoryCreatesRsqrtMicroCallForRsqrtNode) {
  RsqrtMicroFixture fixture;
  std::unique_ptr<MicroApiCall> call(CreateMicroApiCallObject(fixture.rsqrt));
  ASSERT_NE(call, nullptr);
  EXPECT_NE(dynamic_cast<MicroRsqrtApiCall *>(call.get()), nullptr);
  EXPECT_EQ(call->GetMicroApiName(), "Rsqrt");
}

TEST_F(MicroRsqrtApiCallTest, RejectsMismatchedInputOutputDtypes) {
  RsqrtMicroFixture fixture(ge::DT_FLOAT, ge::DT_FLOAT16);
  MicroRsqrtApiCall call("Rsqrt");
  ASSERT_EQ(call.Init(fixture.rsqrt), af::SUCCESS);
  call.AddInput(0);
  call.AddOutput(1);

  CallParam param{"p_reg", "", "float"};
  std::string result;
  EXPECT_NE(call.Generate(fixture.tensor_manager, fixture.tpipe, param, result), af::SUCCESS);
}
}  // namespace codegen
