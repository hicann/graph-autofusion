/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "api_perf_register/api_perf.h"
#include "api_perf_register/ascendc_api_perf.h"
#include "api_perf_register/v1/perf_param_v1.h"
#include "common_utils.h"

#define private public
#include "api_perf_register/api_perf_factory.h"
#undef private

namespace att {
namespace ascendcperf {
af::Status RptElementwisePerf(const NodeDetail &node_info, const Expr &aligned_res, const Expr &unaligned_res,
                              PerfOutputInfo &perf);
}  // namespace ascendcperf
namespace {

class ScopedApiPerfCreators {
 public:
  ScopedApiPerfCreators() {
    auto &factory = ApiPerfFactory::Instance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    saved_ = factory.creator_map_;
  }
  ~ScopedApiPerfCreators() {
    auto &factory = ApiPerfFactory::Instance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    factory.creator_map_.swap(saved_);
  }
  ScopedApiPerfCreators(const ScopedApiPerfCreators &) = delete;
  ScopedApiPerfCreators &operator=(const ScopedApiPerfCreators &) = delete;

 private:
  std::map<std::string, ApiPerfCreatorFun> saved_;
};

class DefaultPerfParams : public PerfParamTable {
 public:
  const std::string *GetAscendCApiPerfTable() const override {
    return nullptr;
  }
  PipeHeadPerfFunc GetPipeHeadPerfFunc(PipeType) const override {
    return nullptr;
  }
};

TensorShapeInfo MakePerfShape(const std::vector<Expr> &dims) {
  TensorShapeInfo shape;
  shape.data_type = "float16";
  shape.data_type_size = 2U;
  shape.loc = HardwareDef::UB;
  shape.dims = dims;
  return shape;
}

class ApiPerfCoverageTest : public testing::Test {};

TEST_F(ApiPerfCoverageTest, NodeDetailsFormatEmptyAndPopulatedDtypes) {
  NodeDetail detail;
  detail.name = "load";
  detail.optype = "Load";
  EXPECT_EQ(detail.ToString(), "load[Load], input_dtype[], output_dtype[], dim_info[], gm_stride[0], ub_stride[0]");
  detail.input_dtype = {"float16", "int32"};
  detail.output_dtype = {"float32", "int16"};
  detail.input_dims = {CreateExpr(2), CreateExpr(3)};
  detail.gm_stride = CreateExpr(8);
  detail.ub_stride = CreateExpr(4);
  const auto formatted = detail.ToString();
  EXPECT_NE(formatted.find("input_dtype[float16], output_dtype[float32]"), std::string::npos);
  EXPECT_NE(formatted.find("dim_info[" + GetVecString(detail.input_dims) + "]"), std::string::npos);
  EXPECT_NE(formatted.find("gm_stride[8], ub_stride[4]"), std::string::npos);
  EXPECT_EQ(formatted.find("int32"), std::string::npos);
}

TEST_F(ApiPerfCoverageTest, PerformanceOutputFormatsPipesAndResolvedVariables) {
  PerfOutputInfo output;
  EXPECT_TRUE(output.ToString().empty());
  output.pipe_res[PipeType::AIV_VEC] = CreateExpr(7);
  EXPECT_EQ(output.ToString(), PipeType2Str.at(PipeType::AIV_VEC) + ":7,");
  const auto variable = CreateExpr("api_perf_format_variable");
  TernaryOp value(CreateExpr(9));
  value.SetVariable(variable);
  output.ternary_ops.emplace(variable, value);
  output.pipe_res[PipeType::AIV_VEC] = variable;
  const auto formatted = output.ToString();
  EXPECT_EQ(formatted.find("api_perf_format_variable"), std::string::npos);
  EXPECT_NE(formatted.find("9"), std::string::npos);
}

TEST_F(ApiPerfCoverageTest, DefaultParametersDescribeZeroOverheadAndNoMicroInstructions) {
  const DefaultPerfParams params;
  EXPECT_TRUE(params.GetVfInstructPerfTable("unknown").empty());
  EXPECT_TRUE(params.GetVfInstructDtypeMappingPerfTable("unknown").empty());
  EXPECT_EQ(params.GetVectorFunctionHeadCost(), CreateExpr(0));
  EXPECT_EQ(params.GetOpHeadCost(), CreateExpr(0));
  EXPECT_EQ(params.GetMicroApiLen(), arch_param::kDefaultVectorLenSize);
  EXPECT_TRUE(params.GetApiRegisterVerName().empty());
}

TEST_F(ApiPerfCoverageTest, FactoryRejectsUnknownTypeAndPreservesFirstRegistration) {
  ScopedApiPerfCreators restore;
  auto &factory = ApiPerfFactory::Instance();
  const std::string name = "Issue211ApiPerfFactory";
  EXPECT_EQ(factory.Create(name), nullptr);
  DefaultPerfParams params;
  TilingScheduleConfigTableV1 config;
  ApiPerfRegister<ApiPerf> registration(name, &DefaultGetPerf, nullptr, &params, &config);
  bool duplicate_called = false;
  ApiPerfFactory::Registerar duplicate(name, [&duplicate_called](const std::string &) {
    duplicate_called = true;
    return std::unique_ptr<ApiPerf>{};
  });
  auto api = factory.Create(name);
  ASSERT_NE(api, nullptr);
  EXPECT_FALSE(duplicate_called);
  EXPECT_EQ(api->GetApiName(), name);
  EXPECT_EQ(api->GetPerfFunc(), &DefaultGetPerf);
  EXPECT_EQ(api->GetMicroPerfFunc(), nullptr);
  EXPECT_EQ(api->GetPerfParam(), &params);
  EXPECT_EQ(api->GetTilingScheduleConfigTable(), &config);
  PerfOutputInfo output;
  output.pipe_res.emplace(PipeType::AIV_VEC, CreateExpr(13));
  EXPECT_EQ(api->GetPerfFunc()({}, {}, NodeInfo{}, output), af::SUCCESS);
  EXPECT_EQ(output.pipe_res.at(PipeType::AIV_VEC), CreateExpr(13));
}

TEST_F(ApiPerfCoverageTest, FactoryResolvesNamedPerformanceFunction) {
  ScopedApiPerfCreators restore;
  const std::string name = "Issue211NamedApiPerfFactory";
  ApiPerfRegister<ApiPerf> registration(name, kUnitVector, nullptr, nullptr, nullptr);
  auto api = ApiPerfFactory::Instance().Create(name);
  ASSERT_NE(api, nullptr);
  ASSERT_NE(api->GetPerfFunc(), nullptr);
  EXPECT_EQ(api->GetPerfFunc(), GetPerfFunc(kUnitVector));
  const auto shape = MakePerfShape({CreateExpr(16)});
  PerfOutputInfo output;
  EXPECT_EQ(api->GetPerfFunc()({shape}, {shape}, NodeInfo{}, output), af::SUCCESS);
  EXPECT_EQ(output.pipe_res.at(PipeType::AIV_VEC), CreateExpr(1));
  EXPECT_EQ(GetPerfFunc("Issue211UnknownPerf"), nullptr);
  EXPECT_EQ(GetAscendCPerfFunc("Issue211UnknownAscendCPerf"), nullptr);
}

TEST_F(ApiPerfCoverageTest, UnitPipeFunctionsEnforceRequiredShapesAndWriteOnlyTheirPipe) {
  const std::vector<std::pair<std::string, PipeType>> cases = {{kUnitMTE1, PipeType::AICORE_MTE1},
                                                               {kUnitMTE2, PipeType::AICORE_MTE2},
                                                               {kUnitMTE3, PipeType::AICORE_MTE3},
                                                               {kUnitVector, PipeType::AIV_VEC},
                                                               {kUnitCube, PipeType::AICORE_CUBE}};
  const auto shape = MakePerfShape({CreateExpr(16)});
  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.first);
    const auto function = GetPerfFunc(test_case.first);
    ASSERT_NE(function, nullptr);
    PerfOutputInfo inputless_output;
    if (test_case.first == kUnitVector) {
      ASSERT_EQ(function({}, {shape}, NodeInfo{}, inputless_output), af::SUCCESS);
      ASSERT_EQ(inputless_output.pipe_res.size(), 1U);
      EXPECT_EQ(inputless_output.pipe_res.at(PipeType::AIV_VEC), CreateExpr(1));
    } else {
      EXPECT_NE(function({}, {shape}, NodeInfo{}, inputless_output), af::SUCCESS);
      EXPECT_TRUE(inputless_output.pipe_res.empty());
    }
    PerfOutputInfo missing_output;
    EXPECT_NE(function({shape}, {}, NodeInfo{}, missing_output), af::SUCCESS);
    EXPECT_TRUE(missing_output.pipe_res.empty());
    PerfOutputInfo output;
    ASSERT_EQ(function({shape}, {shape}, NodeInfo{}, output), af::SUCCESS);
    ASSERT_EQ(output.pipe_res.size(), 1U);
    EXPECT_EQ(output.pipe_res.at(test_case.second), CreateExpr(1));
  }
}

TEST_F(ApiPerfCoverageTest, VectorComputeHandlesUnitInnermostDimensionWithoutDivisionByZero) {
  const auto function = GetPerfFunc(kComputeVector);
  ASSERT_NE(function, nullptr);
  const auto shape = MakePerfShape({CreateExpr(2), CreateExpr(1)});
  PerfOutputInfo output;
  ASSERT_EQ(function({shape}, {shape}, NodeInfo{}, output), af::SUCCESS);
  EXPECT_EQ(output.pipe_res.at(PipeType::AIV_VEC), CreateExpr(12));
  PerfOutputInfo invalid_output;
  EXPECT_NE(function({MakePerfShape({})}, {MakePerfShape({})}, NodeInfo{}, invalid_output), af::SUCCESS);
  EXPECT_TRUE(invalid_output.pipe_res.empty());
}

TEST_F(ApiPerfCoverageTest, RepeatElementwiseSelectsAlignedAndUnalignedCosts) {
  auto node = GenNodeDetail("float16", "float16", {CreateExpr(4)});
  PerfOutputInfo aligned;
  ASSERT_EQ(ascendcperf::RptElementwisePerf(node, CreateExpr(11), CreateExpr(23), aligned), af::SUCCESS);
  EXPECT_EQ(aligned.pipe_res.at(PipeType::AIV_VEC), CreateExpr(11));
  EXPECT_TRUE(aligned.ternary_ops.empty());
  node.input_dims = {CreateExpr(1)};
  PerfOutputInfo unaligned;
  ASSERT_EQ(ascendcperf::RptElementwisePerf(node, CreateExpr(11), CreateExpr(23), unaligned), af::SUCCESS);
  EXPECT_EQ(unaligned.pipe_res.at(PipeType::AIV_VEC), CreateExpr(23));
  EXPECT_TRUE(unaligned.ternary_ops.empty());
}

TEST_F(ApiPerfCoverageTest, RepeatElementwiseRetainsBothBranchesForSymbolicShape) {
  const auto dimension = CreateExpr("api_perf_repeat_extent");
  const auto node = GenNodeDetail("float16", "float16", {dimension});
  PerfOutputInfo output;
  ASSERT_EQ(ascendcperf::RptElementwisePerf(node, CreateExpr(11), CreateExpr(23), output), af::SUCCESS);
  ASSERT_EQ(output.ternary_ops.size(), 1U);
  const auto &result = output.pipe_res.at(PipeType::AIV_VEC);
  const auto iter = output.ternary_ops.find(result);
  ASSERT_NE(iter, output.ternary_ops.end());
  const auto condition = iter->second.DeepCopyIfCase();
  ASSERT_NE(condition, nullptr);
  EXPECT_EQ(condition->GetCondType(), CondType::K_EQ);
  EXPECT_EQ(condition->GetCondRight(), CreateExpr(0));
  EXPECT_EQ(condition->GetCondLeft().Replace({{dimension, CreateExpr(4)}}), CreateExpr(0));
  EXPECT_EQ(condition->GetCondLeft().Replace({{dimension, CreateExpr(1)}}), CreateExpr(128));
  ASSERT_NE(condition->GetChoiceA(), nullptr);
  ASSERT_NE(condition->GetChoiceB(), nullptr);
  EXPECT_EQ(condition->GetChoiceA()->GetExpr(), CreateExpr(11));
  EXPECT_EQ(condition->GetChoiceB()->GetExpr(), CreateExpr(23));
}

TEST_F(ApiPerfCoverageTest, RepeatElementwiseRejectsUnknownDtypeWithoutWritingOutput) {
  const auto node = GenNodeDetail("unknown", "float16", {CreateExpr(4)});
  PerfOutputInfo output;
  EXPECT_NE(ascendcperf::RptElementwisePerf(node, CreateExpr(11), CreateExpr(23), output), af::SUCCESS);
  EXPECT_TRUE(output.pipe_res.empty());
  EXPECT_TRUE(output.ternary_ops.empty());
}

TEST_F(ApiPerfCoverageTest, V1HeadCostsIgnoreMalformedLoadsAndRetainHardwareDefaults) {
  PerfParamTableV1 params;
  NodeInfo malformed;
  malformed.node_type = kLoad;
  malformed.name = "missing_input";
  TernaryOpMap ternaries;
  const auto mte2 = params.GetPipeHeadPerfFunc(PipeType::AIV_MTE2);
  ASSERT_NE(mte2, nullptr);
  const auto head = mte2({malformed}, ternaries);
  EXPECT_TRUE(ternaries.empty());
  EXPECT_EQ(head.Replace({{CreateExpr("block_dim"), CreateExpr(2)}}),
            CreateExpr(15.89f) * CreateExpr(2) + CreateExpr(882.09f));
  const auto mte3 = params.GetPipeHeadPerfFunc(PipeType::AIV_MTE3);
  const auto vec = params.GetPipeHeadPerfFunc(PipeType::AIV_VEC);
  ASSERT_NE(mte3, nullptr);
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(mte3({}, ternaries), CreateExpr(497.36f));
  EXPECT_EQ(vec({}, ternaries), CreateExpr(37.37f));
  EXPECT_EQ(params.GetOpHeadCost(), CreateExpr(300.0));
  TilingScheduleConfigTableV1 config;
  EXPECT_FALSE(config.IsCoreNumThresholdPenaltyEnable());
  const StrideResult stride(CreateExpr(64), 2);
  EXPECT_EQ(stride.stride, CreateExpr(64));
  EXPECT_EQ(stride.block_count_idx, 2);
  EXPECT_TRUE(stride.ternary_ops.empty());
}

}  // namespace
}  // namespace att
