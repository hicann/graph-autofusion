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

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <atomic>

namespace {
using namespace device_validation;
class CaseContractTest : public testing::Test {
 protected:
  void TearDown() override {
    std::filesystem::remove_all(dir_);
  }
  std::filesystem::path Write(const std::string &name, const std::string &text) {
    static std::atomic<unsigned> counter{0};
    dir_ = std::filesystem::temp_directory_path() / ("device_validation_" + std::to_string(counter++) + "_" + name);
    std::filesystem::create_directories(dir_);
    std::ofstream(dir_ / (name == "legacy.json" || name == "legacy_multi.json" ? "ascir.json" : "case.json")) << text;
    return dir_;
  }
  std::filesystem::path dir_;
};

TEST_F(CaseContractTest, LoadsVersionOneSupportMatrix) {
  auto dir = Write("v1.json", R"({
    "schema_version": 1, "case_id": "add", "graph_name": "add_graph",
    "inputs": [{"shape": [2, 3], "dtype": "float16"}],
    "outputs": [{"shape": [2, 3], "dtype": "float16"}],
    "verification": {"functional": true, "precision": true},
    "performance": {"required": false},
    "support_matrix": [{"backend": "ascendc", "soc": "910b",
      "compile": "required", "functional": "required", "precision": "optional", "performance": "optional"}]
  })");
  auto config = LoadCaseConfig(dir);
  EXPECT_EQ(config.schema_version, 1);
  EXPECT_EQ(config.inputs[0].shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(config.support_matrix.size(), 1U);
}

TEST_F(CaseContractTest, LoadsAllPerformanceFields) {
  auto dir = Write("performance.json", R"({
    "schema_version": 1, "case_id": "add",
    "inputs": [{"shape": [2], "dtype": "float16"}],
    "outputs": [{"shape": [2], "dtype": "float16"}],
    "performance": {"required": true, "metric": "latency_ms", "warmup_count": 3,
      "kernel_count": 7, "tiling_time_us": 11.5, "workspace_size": 64,
      "block_dimension": 8, "profiler_available": true},
    "support_matrix": [{"backend": "ascendc_real_device", "soc": "ascend950",
      "compile": "required", "functional": "required", "performance": "required"}]
  })");
  const auto config = LoadCaseConfig(dir);
  EXPECT_TRUE(config.performance.required);
  EXPECT_EQ(config.performance.metric, "latency_ms");
  EXPECT_EQ(config.performance.warmup_count, 3U);
  EXPECT_EQ(config.performance.kernel_count, 7U);
  EXPECT_DOUBLE_EQ(config.performance.tiling_time_us, 11.5);
  EXPECT_EQ(config.performance.workspace_size, 64U);
  EXPECT_EQ(config.performance.block_dimension, 8U);
  EXPECT_TRUE(config.performance.profiler_available);
}

TEST_F(CaseContractTest, RejectsMissingSchemaVersion) {
  auto dir = Write("missing.json", R"({"case_id":"x"})");
  EXPECT_THROW(LoadCaseConfig(dir), std::invalid_argument);
}

TEST_F(CaseContractTest, ConvertsLegacyAscirJson) {
  auto dir = Write("legacy.json", R"({
    "kernel_config": {"graph_name":"legacy", "input_num":"1", "output_num":"1",
    "input_shapes":"2,3", "output_shapes":"2,3",
    "input_data_types":"float16", "output_data_types":"float16"}
  })");
  auto config = LoadCaseConfig(dir);
  EXPECT_EQ(config.schema_version, 1);
  EXPECT_EQ(config.graph_name, "legacy");
  EXPECT_EQ(config.inputs[0].shape, (std::vector<int64_t>{2, 3}));
}

TEST_F(CaseContractTest, RejectsShapeDtypeCountMismatch) {
  auto dir = Write("mismatch.json", R"({"kernel_config": {
    "graph_name":"x", "input_num":"2", "output_num":"1",
    "input_shapes":"2", "output_shapes":"2",
    "input_data_types":"float16", "output_data_types":"float16"}})");
  EXPECT_THROW(LoadCaseConfig(dir), std::invalid_argument);
}

TEST_F(CaseContractTest, ConvertsLegacyMultipleInputsAndOutputs) {
  auto dir = Write("legacy_multi.json", R"({"kernel_config": {
    "graph_name":"legacy_multi", "input_num":"2", "output_num":"2",
    "input_shapes":"128,128;128,128", "output_shapes":"128,128;128,128",
    "input_data_types":"float16;float32", "output_data_types":"float16;float32"}})");
  auto config = LoadCaseConfig(dir);
  ASSERT_EQ(config.inputs.size(), 2U);
  ASSERT_EQ(config.outputs.size(), 2U);
  EXPECT_EQ(config.inputs[1].dtype, "float32");
  EXPECT_EQ(config.outputs[1].shape, (std::vector<int64_t>{128, 128}));
}

TEST_F(CaseContractTest, RejectsOverflowingLegacyShape) {
  auto dir = Write("legacy_overflow.json", R"({"kernel_config": {
    "graph_name":"overflow", "input_num":"1", "output_num":"1",
    "input_shapes":"9223372036854775807,2", "output_shapes":"2",
    "input_data_types":"float16", "output_data_types":"float16"}})");
  EXPECT_THROW(LoadCaseConfig(dir), std::invalid_argument);
}

TEST_F(CaseContractTest, LoadsAndValidatesVerificationTolerances) {
  auto dir = Write("tolerance.json", R"({
    "schema_version": 1, "case_id": "x",
    "inputs": [{"shape": [2], "dtype": "float32"}],
    "outputs": [{"shape": [2], "dtype": "float32"}],
    "verification": {"atol": 0.25, "rtol": 0.5}
  })");
  const auto config = LoadCaseConfig(dir);
  EXPECT_DOUBLE_EQ(config.verification.atol, 0.25);
  EXPECT_DOUBLE_EQ(config.verification.rtol, 0.5);
}

TEST_F(CaseContractTest, LoadsExplicitDynamicShapeWithoutChangingFixedShape) {
  auto dir = Write("dynamic.json", R"({
    "schema_version": 1, "case_id": "x",
    "inputs": [{"shape": [1], "dtype": "float32", "dynamic": true}],
    "outputs": [{"shape": [1], "dtype": "float32"}]
  })");
  const auto config = LoadCaseConfig(dir);
  EXPECT_TRUE(config.inputs[0].dynamic);
  EXPECT_FALSE(config.outputs[0].dynamic);
  EXPECT_EQ(config.outputs[0].shape, (std::vector<int64_t>{1}));
}

TEST_F(CaseContractTest, RejectsInvalidVerificationTolerances) {
  for (const auto tolerance : {"-1.0", "1e100"}) {
    auto dir = Write("bad_tolerance.json", std::string(R"({
      "schema_version": 1, "case_id": "x",
      "inputs": [{"shape": [2], "dtype": "float32"}],
      "outputs": [{"shape": [2], "dtype": "float32"}],
      "verification": {"atol": )") + tolerance +
                                               R"(, "rtol": 0.1}
    })");
    EXPECT_THROW(LoadCaseConfig(dir), std::invalid_argument);
  }
}
}  // namespace
