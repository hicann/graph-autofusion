/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flat_request_validation.h"

#include <gtest/gtest.h>

#include <string>

#include <nlohmann/json.hpp>

namespace device_validation {
namespace {

nlohmann::json FlatSpec() {
  return {{"dtype", "float16"}, {"shape", nlohmann::json::array({2})}};
}

nlohmann::json FlatSpecList() {
  return nlohmann::json::array({FlatSpec()});
}

nlohmann::json ValidFlatRequest() {
  return {{"case_dir", "/tmp/case"},
          {"case_id", "add"},
          {"step", 0},
          {"profile", "ascend950"},
          {"soc_profile", "ascend950"},
          {"module", "host_fake_module"},
          {"abi", "AutofuseLaunch"},
          {"launch_abi", "AutofuseLaunch"},
          {"abi_metadata",
           {{"launch_abi", "AutofuseLaunch"},
            {"input_count", 1},
            {"output_count", 1},
            {"input_dtypes", nlohmann::json::array({"float16"})},
            {"output_dtypes", nlohmann::json::array({"float16"})}}},
          {"input_count", 1},
          {"output_count", 1},
          {"tensor_files", nlohmann::json::array({"input.bin"})},
          {"inputs", nlohmann::json::array({"input.bin"})},
          {"outputs", nlohmann::json::array({"output.bin"})},
          {"tensor_specs", FlatSpecList()},
          {"output_specs", FlatSpecList()},
          {"shape", nlohmann::json::array({2})},
          {"selected_shape", nlohmann::json::array({2})},
          {"variant", "fused"},
          {"contract_schema",
           {{"schema_version", 1},
            {"case_id", "add"},
            {"variant", "fused"},
            {"step", 0},
            {"inputs", FlatSpecList()},
            {"outputs", FlatSpecList()}}},
          {"artifact_dir", "/tmp"},
          {"device", 0},
          {"warmup", 0},
          {"repeat", 1},
          {"profile_dir", ""},
          {"profiler", false}};
}

nlohmann::json MultiInputValidFlatRequest() {
  auto request = ValidFlatRequest();
  const auto inputs = nlohmann::json::array({"in0.bin", "in1.bin", "in2.bin", "in3.bin"});
  const auto specs = nlohmann::json::array({FlatSpec(), FlatSpec(), FlatSpec(), FlatSpec()});
  request["inputs"] = inputs;
  request["tensor_files"] = inputs;
  request["tensor_specs"] = specs;
  request["input_count"] = 4;
  request["abi_metadata"]["input_count"] = 4;
  request["abi_metadata"]["input_dtypes"] = nlohmann::json::array({"float16", "float16", "float16", "float16"});
  request["contract_schema"]["inputs"] = specs;
  return request;
}

std::string ErrorCode(const nlohmann::json &json) {
  return ValidateFlatRequest(json).error_code;
}

TEST(FlatRequestValidationTest, ValidRequestPasses) {
  EXPECT_TRUE(ValidateFlatRequest(ValidFlatRequest()).ok());
}

TEST(FlatRequestValidationTest, MissingCaseIdIsStepContract) {
  auto request = ValidFlatRequest();
  request.erase("case_id");
  EXPECT_EQ(ErrorCode(request), "step_contract");
}

TEST(FlatRequestValidationTest, MissingContractSchemaIsStepContract) {
  auto request = ValidFlatRequest();
  request.erase("contract_schema");
  EXPECT_EQ(ErrorCode(request), "step_contract");
}

TEST(FlatRequestValidationTest, MissingVariantIsStepContract) {
  auto request = ValidFlatRequest();
  request.erase("variant");
  EXPECT_EQ(ErrorCode(request), "step_contract");
}

TEST(FlatRequestValidationTest, MissingInputArraysIsInputCount) {
  auto request = ValidFlatRequest();
  request.erase("inputs");
  EXPECT_EQ(ErrorCode(request), "input_count");
}

TEST(FlatRequestValidationTest, MissingOutputArraysIsOutputCount) {
  auto request = ValidFlatRequest();
  request.erase("outputs");
  EXPECT_EQ(ErrorCode(request), "output_count");
}

TEST(FlatRequestValidationTest, MissingTensorSpecsIsInputSpecs) {
  auto request = ValidFlatRequest();
  request.erase("tensor_specs");
  EXPECT_EQ(ErrorCode(request), "input_specs");
}

TEST(FlatRequestValidationTest, MissingOutputSpecsIsOutputSpecs) {
  auto request = ValidFlatRequest();
  request.erase("output_specs");
  EXPECT_EQ(ErrorCode(request), "output_specs");
}

TEST(FlatRequestValidationTest, WrongTypeInputsIsInputCount) {
  auto request = ValidFlatRequest();
  request["inputs"] = "not-an-array";
  EXPECT_EQ(ErrorCode(request), "input_count");
}

TEST(FlatRequestValidationTest, WrongTypeStepIsStepContract) {
  auto request = ValidFlatRequest();
  request["step"] = "zero";
  EXPECT_EQ(ErrorCode(request), "step_contract");
}

TEST(FlatRequestValidationTest, WrongTypeTensorSpecsIsInputSpecs) {
  auto request = ValidFlatRequest();
  request["tensor_specs"] = "not-an-array";
  EXPECT_EQ(ErrorCode(request), "input_specs");
}

TEST(FlatRequestValidationTest, UnsupportedInputDtypeIsInputSpecs) {
  auto request = ValidFlatRequest();
  request["tensor_specs"][0]["dtype"] = "float64";
  EXPECT_EQ(ErrorCode(request), "input_specs");
}

TEST(FlatRequestValidationTest, UnsupportedOutputDtypeIsOutputSpecs) {
  auto request = ValidFlatRequest();
  request["output_specs"][0]["dtype"] = "float64";
  EXPECT_EQ(ErrorCode(request), "output_specs");
}

TEST(FlatRequestValidationTest, SelectedShapeMismatchIsSelectedShape) {
  auto request = ValidFlatRequest();
  request["selected_shape"] = nlohmann::json::array({4});
  EXPECT_EQ(ErrorCode(request), "selected_shape");
}

TEST(FlatRequestValidationTest, WrongTypeProfileDirIsProfileDir) {
  auto request = ValidFlatRequest();
  request["profile_dir"] = 42;
  EXPECT_EQ(ErrorCode(request), "profile_dir");
}

TEST(FlatRequestValidationTest, UnknownAbiIsAbiMetadata) {
  auto request = ValidFlatRequest();
  request["abi"] = "OtherLaunch";
  request["launch_abi"] = "OtherLaunch";
  request["abi_metadata"]["launch_abi"] = "OtherLaunch";
  EXPECT_EQ(ErrorCode(request), "abi_metadata");
}

TEST(FlatRequestValidationTest, ZeroDimensionShapeIsShape) {
  auto request = ValidFlatRequest();
  request["tensor_specs"][0]["shape"] = nlohmann::json::array({0});
  EXPECT_EQ(ErrorCode(request), "shape");
}

TEST(FlatRequestValidationTest, NegativeDimensionShapeIsShape) {
  auto request = ValidFlatRequest();
  request["tensor_specs"][0]["shape"] = nlohmann::json::array({-1, 2});
  EXPECT_EQ(ErrorCode(request), "shape");
}

TEST(FlatRequestValidationTest, NonIntegerDimensionShapeIsShape) {
  auto request = ValidFlatRequest();
  request["tensor_specs"][0]["shape"] = nlohmann::json::array({2, "wide"});
  EXPECT_EQ(ErrorCode(request), "shape");
}

TEST(FlatRequestValidationTest, OutOfRangeUnsignedDimensionIsShape) {
  auto request = ValidFlatRequest();
  request["tensor_specs"][0]["shape"] = nlohmann::json::array({18446744073709551615ULL});
  EXPECT_EQ(ErrorCode(request), "shape");
}

TEST(FlatRequestValidationTest, OverflowingDimensionProductIsShape) {
  auto request = ValidFlatRequest();
  request["tensor_specs"][0]["shape"] = nlohmann::json::array({4294967296LL, 4294967296LL});
  EXPECT_EQ(ErrorCode(request), "shape");
}

TEST(FlatRequestValidationTest, OverflowingDtypeByteSizeIsInputSpecs) {
  auto request = ValidFlatRequest();
  request["tensor_specs"][0]["shape"] = nlohmann::json::array({2305843009213693952LL, 2, 2});
  EXPECT_EQ(ErrorCode(request), "input_specs");
}

TEST(FlatRequestValidationTest, LegacyAbiRejectsInputArityOverThree) {
  auto request = MultiInputValidFlatRequest();
  EXPECT_EQ(ErrorCode(request), "unsupported_abi_arity");
}

TEST(FlatRequestValidationTest, LegacyAbiRejectsZeroInputs) {
  auto request = ValidFlatRequest();
  request["inputs"] = nlohmann::json::array();
  request["tensor_files"] = nlohmann::json::array();
  request["tensor_specs"] = nlohmann::json::array();
  request["input_count"] = 0;
  request["abi_metadata"]["input_count"] = 0;
  request["abi_metadata"]["input_dtypes"] = nlohmann::json::array();
  request["contract_schema"]["inputs"] = nlohmann::json::array();
  EXPECT_EQ(ErrorCode(request), "unsupported_abi_arity");
}

TEST(FlatRequestValidationTest, V2AbiAcceptsFourInputs) {
  auto request = MultiInputValidFlatRequest();
  request["abi"] = "AutofuseLaunchV2";
  request["launch_abi"] = "AutofuseLaunchV2";
  request["abi_metadata"]["launch_abi"] = "AutofuseLaunchV2";
  EXPECT_TRUE(ValidateFlatRequest(request).ok());
}

TEST(FlatRequestValidationTest, ContractInputsMismatchIsInputSpecs) {
  auto request = ValidFlatRequest();
  request["contract_schema"]["inputs"][0]["dtype"] = "int32";
  EXPECT_EQ(ErrorCode(request), "input_specs");
}

TEST(FlatRequestValidationTest, ContractOutputsMismatchIsOutputSpecs) {
  auto request = ValidFlatRequest();
  request["contract_schema"]["outputs"][0]["dtype"] = "int32";
  EXPECT_EQ(ErrorCode(request), "output_specs");
}

TEST(FlatRequestValidationTest, ContractSchemaVersionMismatchIsStepContract) {
  auto request = ValidFlatRequest();
  request["contract_schema"]["schema_version"] = 2;
  EXPECT_EQ(ErrorCode(request), "step_contract");
}

TEST(FlatRequestValidationTest, ContractCaseIdMismatchIsStepContract) {
  auto request = ValidFlatRequest();
  request["contract_schema"]["case_id"] = "sub";
  EXPECT_EQ(ErrorCode(request), "step_contract");
}

TEST(FlatRequestValidationTest, ContractStepMismatchIsStepContract) {
  auto request = ValidFlatRequest();
  request["contract_schema"]["step"] = 3;
  EXPECT_EQ(ErrorCode(request), "step_contract");
}

TEST(FlatRequestValidationTest, ContractMissingInputsIsStepContract) {
  auto request = ValidFlatRequest();
  request["contract_schema"].erase("inputs");
  EXPECT_EQ(ErrorCode(request), "step_contract");
}

TEST(FlatRequestValidationTest, ContractOutputsWrongTypeIsStepContract) {
  auto request = ValidFlatRequest();
  request["contract_schema"]["outputs"] = "not-an-array";
  EXPECT_EQ(ErrorCode(request), "step_contract");
}

TEST(FlatRequestValidationTest, AbiMetadataInputDtypeMismatchIsAbiMetadata) {
  auto request = ValidFlatRequest();
  request["abi_metadata"]["input_dtypes"][0] = "int32";
  EXPECT_EQ(ErrorCode(request), "abi_metadata");
}

TEST(FlatRequestValidationTest, AbiMetadataOutputDtypeMismatchIsAbiMetadata) {
  auto request = ValidFlatRequest();
  request["abi_metadata"]["output_dtypes"][0] = "int32";
  EXPECT_EQ(ErrorCode(request), "abi_metadata");
}

nlohmann::json AclnnFlatRequest() {
  auto request = ValidFlatRequest();
  request["aclnn_op"] = "IsInf";
  request["module"] = "";
  request["abi"] = "";
  request["launch_abi"] = "";
  request["abi_metadata"] = nlohmann::json::object();
  return request;
}

TEST(FlatRequestValidationTest, AclnnRequestWithEmptyAbiPasses) {
  EXPECT_TRUE(ValidateFlatRequest(AclnnFlatRequest()).ok());
}

TEST(FlatRequestValidationTest, AclnnRequestWithoutAbiMetadataOrModulePasses) {
  auto request = AclnnFlatRequest();
  request.erase("abi_metadata");
  request.erase("module");
  EXPECT_TRUE(ValidateFlatRequest(request).ok());
}

TEST(FlatRequestValidationTest, AclnnRequestSkipsAbiValueChecks) {
  auto request = AclnnFlatRequest();
  request["abi"] = "OtherLaunch";
  request["launch_abi"] = "OtherLaunch";
  EXPECT_TRUE(ValidateFlatRequest(request).ok());
}

TEST(FlatRequestValidationTest, AclnnRequestStillValidatesContractTensors) {
  auto request = AclnnFlatRequest();
  request["contract_schema"]["inputs"][0]["dtype"] = "int32";
  EXPECT_EQ(ErrorCode(request), "input_specs");
}

TEST(FlatRequestValidationTest, WrongTypeAclnnOpIsStepContract) {
  auto request = AclnnFlatRequest();
  request["aclnn_op"] = 42;
  EXPECT_EQ(ErrorCode(request), "step_contract");
}

TEST(FlatRequestValidationTest, EmptyAclnnOpKeepsLegacyAbiValidation) {
  auto request = ValidFlatRequest();
  request["aclnn_op"] = "";
  request["abi"] = "OtherLaunch";
  request["launch_abi"] = "OtherLaunch";
  request["abi_metadata"]["launch_abi"] = "OtherLaunch";
  EXPECT_EQ(ErrorCode(request), "abi_metadata");
}

}  // namespace
}  // namespace device_validation
