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
#include "micro_api_call/micro_arange_api_call.h"
#include "platform_context.h"
#include "runtime_stub.h"

namespace codegen {
namespace {
struct ArangeMicroFixture {
  af::AscGraph graph{"micro_arange"};
  af::AscNodePtr arange;
  TensorManager tensor_manager;
  Tiler tiler;
  TPipe tpipe{"tpipe", tiler};

  ArangeMicroFixture(const af::Expression &base, const af::Expression &step, ge::DataType dtype = ge::DT_INT32) {
    const auto size = graph.CreateSizeVar("size");
    const auto axis = graph.CreateAxis("axis", size);
    af::ascir_op::Arange arange_op("arange");
    graph.AddNode(arange_op);
    arange_op.ir_attr.SetBase(base);
    arange_op.ir_attr.SetStep(step);
    arange_op.y.dtype = dtype;
    *arange_op.y.axis = {axis.id};
    *arange_op.y.repeats = {size};
    *arange_op.y.strides = {af::sym::kSymbolOne};
    *arange_op.y.vectorized_axis = {axis.id};
    *arange_op.y.vectorized_strides = {af::sym::kSymbolOne};

    arange = graph.FindNode("arange");
    arange->outputs[0].attr.mem.tensor_id = 0;
    std::string dtype_name;
    EXPECT_EQ(Tensor::DtypeName(dtype, dtype_name), af::SUCCESS);
    EXPECT_EQ(tensor_manager.AddTensor(MicroApiTensor(arange->outputs[0], dtype_name)), af::SUCCESS);
    tiler.AddSizeVar(af::SizeVar(size));
  }
};

class MicroArangeApiCallTest : public testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    ge::RuntimeStub::SetInstance(std::make_shared<ge::RuntimeStubV2Common>());
  }

  void TearDown() override {
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};
}  // namespace

TEST_F(MicroArangeApiCallTest, GeneratesDirectArangeForUnitStep) {
  ArangeMicroFixture fixture(af::Symbol(3), af::Symbol(1));
  MicroArangeApiCall call("Arange");
  ASSERT_EQ(call.Init(fixture.arange), af::SUCCESS);
  call.AddOutput(0);

  CallParam param{"p_reg", "", "int32_t"};
  std::string result;
  ASSERT_EQ(call.Generate(fixture.tensor_manager, fixture.tpipe, param, result), af::SUCCESS);
  EXPECT_EQ(result, "AscendC::Reg::Arange(vreg_0, static_cast<int32_t>(3));\n");
}

TEST_F(MicroArangeApiCallTest, GeneratesInt64WithVectorBlockOffset) {
  ArangeMicroFixture fixture(af::Symbol(3), af::Symbol(2), ge::DT_INT64);
  MicroArangeApiCall call("Arange");
  ASSERT_EQ(call.Init(fixture.arange), af::SUCCESS);
  call.AddOutput(0);

  CallParam param{"p_reg", "axis * ELEMENT_PER_VECTOR_LENGTH * 3", "int64_t"};
  std::string result;
  ASSERT_EQ(call.Generate(fixture.tensor_manager, fixture.tpipe, param, result), af::SUCCESS);
  EXPECT_NE(result.find("axis * ELEMENT_PER_VECTOR_LENGTH * 3"), std::string::npos);
  EXPECT_NE(result.find("3 + (axis * ELEMENT_PER_VECTOR_LENGTH * 3) * (2)"), std::string::npos);
  EXPECT_NE(result.find("static_cast<int64_t>(2)"), std::string::npos);
}

TEST_F(MicroArangeApiCallTest, GeneratesInt32WithVectorBlockOffset) {
  ArangeMicroFixture fixture(af::Symbol(3), af::Symbol(2));
  MicroArangeApiCall call("Arange");
  ASSERT_EQ(call.Init(fixture.arange), af::SUCCESS);
  call.AddOutput(0);

  CallParam param{"p_reg", "axis * ELEMENT_PER_VECTOR_LENGTH * 3", "int32_t"};
  std::string result;
  ASSERT_EQ(call.Generate(fixture.tensor_manager, fixture.tpipe, param, result), af::SUCCESS);
  EXPECT_NE(result.find("axis * ELEMENT_PER_VECTOR_LENGTH * 3"), std::string::npos);
  EXPECT_NE(result.find("3 + (axis * ELEMENT_PER_VECTOR_LENGTH * 3) * (2)"), std::string::npos);
  EXPECT_NE(result.find("static_cast<int32_t>(2)"), std::string::npos);
}

TEST_F(MicroArangeApiCallTest, GeneratesScaledArangeWithDynamicBaseAndStep) {
  ArangeMicroFixture fixture(af::Expression::Parse("size + 1"), af::Expression::Parse("size * 2"));
  MicroArangeApiCall call("Arange");
  ASSERT_EQ(call.Init(fixture.arange), af::SUCCESS);
  call.AddOutput(0);

  CallParam param{"p_reg", "", "int32_t"};
  std::string result;
  ASSERT_EQ(call.Generate(fixture.tensor_manager, fixture.tpipe, param, result), af::SUCCESS);
  EXPECT_EQ(result,
            "AscendC::Reg::Arange(vreg_0, static_cast<int32_t>(0));\n"
            "AscendC::Reg::Muls(vreg_0, vreg_0, static_cast<int32_t>((2 * t->size)), p_reg);\n"
            "AscendC::Reg::Adds(vreg_0, vreg_0, static_cast<int32_t>((1 + t->size)), p_reg);\n");
}

TEST_F(MicroArangeApiCallTest, UsesVfArangeParamsWhenProvided) {
  ArangeMicroFixture fixture(af::Symbol(3), af::Symbol(2));
  MicroArangeApiCall call("Arange");
  ASSERT_EQ(call.Init(fixture.arange), af::SUCCESS);
  call.AddOutput(0);

  CallParam param{"p_reg", "", "int32_t", {true, "arange_base_0", "arange_step_0"}};
  std::string result;
  ASSERT_EQ(call.Generate(fixture.tensor_manager, fixture.tpipe, param, result), af::SUCCESS);
  EXPECT_NE(result.find("arange_base_0"), std::string::npos);
  EXPECT_NE(result.find("arange_step_0"), std::string::npos);
}

TEST_F(MicroArangeApiCallTest, FactoryCreatesArangeCall) {
  ArangeMicroFixture fixture(af::Symbol(0), af::Symbol(1));
  std::unique_ptr<MicroApiCall> call(CreateMicroApiCallObject(fixture.arange));
  ASSERT_NE(call, nullptr);
  EXPECT_NE(dynamic_cast<MicroArangeApiCall *>(call.get()), nullptr);
  EXPECT_EQ(call->Init(fixture.arange), af::SUCCESS);
}

TEST_F(MicroArangeApiCallTest, RejectsUnsupportedDtype) {
  ArangeMicroFixture fixture(af::Symbol(0), af::Symbol(1), ge::DT_FLOAT);
  MicroArangeApiCall call("Arange");
  EXPECT_NE(call.Init(fixture.arange), af::SUCCESS);
}
}  // namespace codegen
