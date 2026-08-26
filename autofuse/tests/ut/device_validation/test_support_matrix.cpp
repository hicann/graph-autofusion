/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "case_contract.h"
#include "support_matrix.h"

#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <fstream>

namespace {
using namespace device_validation;
CaseConfig Config() {
  CaseConfig c;
  c.schema_version = 1;
  c.case_id = "add";
  c.inputs = {{{2}, "float16"}};
  c.support_matrix.push_back({"add",
                              "ascendc",
                              "910b",
                              {{Capability::kCompile, SupportStatus::kRequired},
                               {Capability::kFunctional, SupportStatus::kRequired},
                               {Capability::kPrecision, SupportStatus::kOptional},
                               {Capability::kPerformance, SupportStatus::kOptional}},
                              {"float16"},
                              {{2}}});
  return c;
}

BackendCapabilities Backend() {
  return {true, true, true, true, false};
}
SocCapabilities Soc() {
  return {"910b", {"float16"}, {{{2}}}};
}

TEST(SupportMatrixTest, RequiredCapabilityFailureIsFailed) {
  auto c = Config();
  auto b = Backend();
  b.compile = false;
  auto d = ResolveSupport(c, b, Soc(), {"ascendc", "910b", false, "float16", {{2}}});
  EXPECT_EQ(d.capabilities.at(Capability::kCompile).result, CapabilityResult::kFailed);
}

TEST(SupportMatrixTest, OptionalProfilerMissingIsSkipped) {
  auto d = ResolveSupport(Config(), Backend(), Soc(), {"ascendc", "910b", true, "float16", {{2}}});
  EXPECT_EQ(d.capabilities.at(Capability::kPerformance).result, CapabilityResult::kSkipped);
}

TEST(SupportMatrixTest, RequiredProfilerMissingIsFailed) {
  auto c = Config();
  c.support_matrix[0].capabilities[Capability::kPerformance] = SupportStatus::kRequired;
  auto d = ResolveSupport(c, Backend(), Soc(), {"ascendc", "910b", true, "float16", {{2}}});
  EXPECT_EQ(d.capabilities.at(Capability::kPerformance).result, CapabilityResult::kFailed);
}

TEST(SupportMatrixTest, MissingCapabilityDefaultsToNotApplicable) {
  auto c = Config();
  c.support_matrix[0].capabilities.erase(Capability::kPrecision);
  auto d = ResolveSupport(c, Backend(), Soc(), {"ascendc", "910b", false, "float16", {{2}}});
  EXPECT_EQ(d.capabilities.at(Capability::kPrecision).result, CapabilityResult::kNotApplicable);
}

TEST(SupportMatrixTest, SocDtypeAndShapeRestrictionIsChecked) {
  auto soc = Soc();
  soc.dtypes = {"float16"};
  soc.shapes = {{{4}}};
  auto d = ResolveSupport(Config(), Backend(), soc, {"ascendc", "910b", false, "float16", {{2}}});
  EXPECT_EQ(d.capabilities.at(Capability::kCompile).result, CapabilityResult::kFailed);
}

TEST(SupportMatrixTest, UnsupportedCombinationIsNotApplicable) {
  auto c = Config();
  c.support_matrix[0].capabilities[Capability::kFunctional] = SupportStatus::kUnsupported;
  auto d = ResolveSupport(c, Backend(), Soc(), {"ascendc", "910b", true, "float16", {{2}}});
  EXPECT_EQ(d.capabilities.at(Capability::kFunctional).result, CapabilityResult::kNotApplicable);
}

TEST(SupportMatrixTest, UnlistedCaseSocCombinationIsRejected) {
  EXPECT_THROW(ResolveSupport(Config(), Backend(), Soc(), {"acl", "910b", true, "float16", {{2}}}),
               std::invalid_argument);
}

TEST(SupportMatrixTest, LocalAbiAndVariantContractIsEnforced) {
  auto c = Config();
  c.variants = {"fused"};
  c.support_matrix[0].allowed_abi = "AutofuseLaunchV2";
  c.support_matrix[0].variants = {"fused"};
  auto request = RunRequest{"ascendc", "910b", false, "float16", {{2}}};
  request.config = c;
  request.abi = "AutofuseLaunch";
  request.variant = "fused";
  EXPECT_THROW(ResolveSupport(c, Backend(), Soc(), request), std::invalid_argument);
  request.abi = "AutofuseLaunchV2";
  request.variant = "unfused";
  EXPECT_THROW(ResolveSupport(c, Backend(), Soc(), request), std::invalid_argument);
}

TEST(SupportMatrixTest, LocalDtypeSignatureIsEnforced) {
  auto c = Config();
  c.support_matrix[0].input_dtypes = {"uint8"};
  auto request = RunRequest{"ascendc", "910b", false, "float16", {{2}}};
  request.config = c;
  request.variant = "fused";
  EXPECT_EQ(ResolveSupport(c, Backend(), Soc(), request).capabilities.at(Capability::kCompile).result,
            CapabilityResult::kFailed);
}

TEST(SupportMatrixTest, DtypeAndShapeRestrictionIsCheckedBeforeRun) {
  auto d = ResolveSupport(Config(), Backend(), Soc(), {"ascendc", "910b", true, "float32", {{3}}});
  EXPECT_EQ(d.capabilities.at(Capability::kCompile).result, CapabilityResult::kFailed);
  EXPECT_FALSE(d.capabilities.at(Capability::kCompile).reason.empty());
}

TEST(SupportMatrixTest, MultiInputShapeLimitChecksEachTensorIndependently) {
  auto c = Config();
  c.inputs = {{{128, 128}, "float16"}, {{128, 128}, "float16"}, {{128, 128}, "float16"}};
  c.support_matrix[0].input_dtypes = {"float16", "float16", "float16"};
  c.support_matrix[0].shapes = {{128, 128}};
  auto soc = Soc();
  soc.dtypes = {"float16"};
  soc.shapes = {{128, 128}};
  soc.max_shape_elements = 128 * 128;
  RunRequest request{"ascendc", "910b", false, "float16", {{128, 128}, {128, 128}, {128, 128}}};
  request.config = c;
  const auto decision = ResolveSupport(c, Backend(), soc, request);
  EXPECT_EQ(decision.capabilities.at(Capability::kCompile).result, CapabilityResult::kPassed);
}

TEST(SupportMatrixTest, CaseJsonDtypeRestrictionFlowsIntoResolver) {
  static std::atomic<unsigned> counter{0};
  const auto dir =
      std::filesystem::temp_directory_path() / ("device_validation_case_json_" + std::to_string(counter++));
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "case.json") << R"({
    "schema_version": 1, "case_id": "add",
    "inputs": [{"shape": [2], "dtype": "float16"}],
    "support_matrix": [{"backend": "ascendc", "soc": "910b",
      "compile": "required", "dtypes": ["float16"], "shapes": [[2]]}]
  })";
  const auto config = LoadCaseConfig(dir);
  const auto backend = Backend();
  auto soc = Soc();
  soc.dtypes = {"float16", "float32"};
  soc.shapes = {{{2}}};
  const auto unsupported = ResolveSupport(config, backend, soc, {"ascendc", "910b", false, "float32", {{2}}});
  EXPECT_EQ(unsupported.capabilities.at(Capability::kCompile).result, CapabilityResult::kFailed);
  const auto supported = ResolveSupport(config, backend, soc, {"ascendc", "910b", false, "float16", {{2}}});
  EXPECT_EQ(supported.capabilities.at(Capability::kCompile).result, CapabilityResult::kPassed);
  std::filesystem::remove_all(dir);
}
}  // namespace
