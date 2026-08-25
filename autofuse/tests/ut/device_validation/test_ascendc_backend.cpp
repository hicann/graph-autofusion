/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_backend.h"
#include "flat_output.h"
#include "profile.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <unordered_map>

namespace device_validation {
namespace {

class FlatOutputHooksGuard final {
 public:
  explicit FlatOutputHooksGuard(const FlatOutputTestHooks &hooks) {
    SetFlatOutputTestHooks(hooks);
  }
  ~FlatOutputHooksGuard() {
    ClearFlatOutputTestHooks();
  }
};

CaseConfig Config() {
  CaseConfig config;
  config.case_id = "add";
  config.inputs = {{{2}, "float16"}};
  config.outputs = {{{2}, "float16"}};
  config.support_matrix.push_back({"add",
                                   "ascendc_real_device",
                                   "ascend950",
                                   {{Capability::kCompile, SupportStatus::kRequired},
                                    {Capability::kFunctional, SupportStatus::kRequired},
                                    {Capability::kPerformance, SupportStatus::kOptional}},
                                   {"float16"},
                                   {{2}}});
  return config;
}

class SuccessRuntime final : public RuntimeApi {
 public:
  Status Init() override {
    return Status::kOk;
  }
  Status SetDevice(int32_t) override {
    return Status::kOk;
  }
  Status CreateStream(void **stream) override {
    *stream = this;
    return Status::kOk;
  }
  Status Synchronize(void *) override {
    return Status::kOk;
  }
  Status DestroyStream(void *) override {
    return Status::kOk;
  }
  Status ResetDevice(int32_t) override {
    return Status::kOk;
  }
  Status Finalize() override {
    return Status::kOk;
  }
  Status Malloc(void **ptr, size_t size) override {
    auto allocation = std::make_unique<uint8_t[]>(size);
    *ptr = allocation.get();
    allocations_.emplace(*ptr, std::move(allocation));
    return Status::kOk;
  }
  Status Free(void *ptr) override {
    allocations_.erase(ptr);
    return Status::kOk;
  }
  Status CopyToDevice(void *destination, const void *source, size_t size) override {
    std::copy_n(static_cast<const uint8_t *>(source), size, static_cast<uint8_t *>(destination));
    return Status::kOk;
  }
  Status CopyToHost(void *destination, const void *source, size_t size) override {
    std::copy_n(static_cast<const uint8_t *>(source), size, static_cast<uint8_t *>(destination));
    return Status::kOk;
  }

 private:
  std::unordered_map<void *, std::unique_ptr<uint8_t[]>> allocations_;
};

class SuccessLoader final : public ModuleLoader {
 public:
  void *Load(const std::filesystem::path &) override {
    return this;
  }
  void *Symbol(void *, const char *name) override {
    if (std::string(name) == "GetTilingDataSize") return reinterpret_cast<void *>(&GetSize);
    if (std::string(name) == "AutofuseTiling") return reinterpret_cast<void *>(&Tiling);
    if (std::string(name) == "AutofuseLaunchV2") return reinterpret_cast<void *>(&Launch);
    return nullptr;
  }
  Status Unload(void *) override {
    return Status::kOk;
  }

 private:
  static size_t GetSize() {
    return 8;
  }
  static ge::graphStatus Tiling(void *, uint32_t *workspace, uint32_t *block, void *) {
    *workspace = 0;
    *block = 1;
    return ge::GRAPH_SUCCESS;
  }
  static uint32_t Launch(uint32_t, void *, void **, int32_t, void **, int32_t, void *, void *) {
    return 0;
  }
};

TEST(AscendCBackendTest, ExposesAscend950Capabilities) {
  AscendCBackend backend(LoadDeviceProfile("autofuse/tests/st/device_validation/profiles/ascend950.json"));
  EXPECT_EQ(backend.GetCapabilities().name, "ascendc_real_device");
  EXPECT_EQ(backend.GetCapabilities().profile, "ascend950");
  EXPECT_EQ(backend.GetCapabilities().abi, "AutofuseLaunchV2");
}

TEST(AscendCBackendTest, DecodesFloat16OutputValuesForPrecisionVerification) {
  const std::vector<uint8_t> bytes = {0x00, 0x3c, 0x00, 0xc0, 0x00, 0x7c, 0x00, 0xfc};
  const auto output = DecodeOutputForTest("float16", {4}, bytes);
  ASSERT_EQ(output.values.size(), 4U);
  EXPECT_DOUBLE_EQ(output.values[0], 1.0);
  EXPECT_DOUBLE_EQ(output.values[1], -2.0);
  EXPECT_TRUE(std::isinf(output.values[2]));
  EXPECT_TRUE(std::isinf(output.values[3]));
  EXPECT_LT(output.values[3], 0.0);
}

TEST(AscendCBackendTest, DecodesSignedNarrowIntegersWithSignExtension) {
  EXPECT_EQ(DecodeOutputForTest("int8", {2}, {0xff, 0x80}).integer_values, (std::vector<int64_t>{-1, -128}));
  std::vector<uint8_t> bytes = {0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x80};
  EXPECT_EQ(DecodeOutputForTest("int32", {2}, bytes).integer_values,
            (std::vector<int64_t>{-1, std::numeric_limits<int32_t>::min()}));
}

TEST(AscendCBackendTest, DecodesNegativeInt64WithoutShiftingBy64) {
  std::vector<uint8_t> bytes = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
                                0x00, 0x00, 0x00, 0x80, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f};
  EXPECT_EQ(DecodeOutputForTest("int64", {3}, bytes).integer_values,
            (std::vector<int64_t>{-1, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()}));
}

TEST(AscendCBackendTest, RejectsUndeclaredBackendBeforeExecution) {
  auto config = Config();
  config.support_matrix[0].backend = "simulator";
  AscendCBackend backend(LoadDeviceProfile("autofuse/tests/st/device_validation/profiles/ascend950.json"));
  EXPECT_EQ(backend.Validate(config), Status::kUnsupported);
}

TEST(AscendCBackendTest, RejectsUnsupportedDtypeAndShapeBeforeExecution) {
  auto config = Config();
  config.inputs[0].dtype = "float32";
  config.inputs[0].shape = {4};
  AscendCBackend backend(LoadDeviceProfile("autofuse/tests/st/device_validation/profiles/ascend950.json"));
  EXPECT_EQ(backend.Validate(config), Status::kUnsupported);
}

TEST(AscendCBackendTest, FactoryRegistersRealDeviceBackend) {
  auto backend =
      CreateExecutionBackend("ascendc_real_device", "autofuse/tests/st/device_validation/profiles/ascend950.json");
  ASSERT_NE(backend, nullptr);
  EXPECT_EQ(backend->GetCapabilities().profile, "ascend950");
  EXPECT_EQ(CreateExecutionBackend("ascendc_simulator"), nullptr);
}

TEST(AscendCBackendTest, FactoryRejectsMissingRealDeviceProfile) {
  EXPECT_THROW(CreateExecutionBackend("ascendc_real_device"), std::invalid_argument);
}

TEST(AscendCBackendTest, RejectsProfileShapeLimit) {
  auto profile = LoadDeviceProfile("autofuse/tests/st/device_validation/profiles/ascend950.json");
  profile.capabilities.max_shape_elements = 1;
  AscendCBackend backend(profile);
  EXPECT_EQ(backend.Validate(Config()), Status::kUnsupported);
}

TEST(AscendCBackendTest, RejectsMultiOutputWhenProfileDoesNotSupportIt) {
  auto config = Config();
  config.outputs.push_back({{2}, "float16"});
  AscendCBackend backend(LoadDeviceProfile("autofuse/tests/st/device_validation/profiles/ascend950.json"));
  EXPECT_EQ(backend.Validate(config), Status::kUnsupported);
}

TEST(AscendCBackendTest, RejectsZeroKernelCountBeforeRuntime) {
  AscendCBackend backend(LoadDeviceProfile("autofuse/tests/st/device_validation/profiles/ascend950.json"));
  RunRequest request;
  request.config = Config();
  request.backend = "ascendc_real_device";
  request.soc = "ascend950";
  request.abi = "AutofuseLaunchV2";
  request.module_path = "not-a-module";
  request.kernel_count = 0;
  RunResult result;
  EXPECT_EQ(backend.Run(request, &result), Status::kInvalidArgument);
  EXPECT_EQ(result.report.stage_status, "failed");
}

TEST(AscendCBackendTest, RequiredProfilerFailureStopsInValidate) {
  auto config = Config();
  config.performance.required = true;
  config.performance.profiler = true;
  config.support_matrix[0].capabilities[Capability::kPerformance] = SupportStatus::kRequired;
  auto profile = LoadDeviceProfile("autofuse/tests/st/device_validation/profiles/ascend950.json");
  profile.profiler_tool.clear();
  profile.capabilities.profiler_available = false;
  AscendCBackend backend(profile);
  EXPECT_EQ(backend.Validate(config), Status::kUnsupported);
}

TEST(AscendCBackendTest, RequiredProfilerFailureStopsRunBeforeRuntime) {
  auto config = Config();
  config.performance.profiler = true;
  config.support_matrix[0].capabilities[Capability::kPerformance] = SupportStatus::kRequired;
  auto profile = LoadDeviceProfile("autofuse/tests/st/device_validation/profiles/ascend950.json");
  profile.profiler_tool.clear();
  profile.capabilities.profiler_available = false;
  AscendCBackend backend(profile);
  RunRequest request;
  request.config = config;
  request.backend = "ascendc_real_device";
  request.soc = "ascend950";
  request.abi = "AutofuseLaunchV2";
  request.profiler = true;
  request.module_path = "not-a-module";
  request.dtype = "float16";
  request.shapes = {{2}};
  request.kernel_count = 1;
  RunResult result;
  EXPECT_EQ(backend.Run(request, &result), Status::kUnsupported);
}

TEST(AscendCBackendTest, PreservesPerformanceManifestInRunRequest) {
  auto config = Config();
  config.performance.required = true;
  config.performance.profiler = true;
  config.performance.metric = "latency_ms";
  config.performance.warmup_count = 3;
  config.performance.kernel_count = 7;
  const auto info = AscendCBackend(LoadDeviceProfile("autofuse/tests/st/device_validation/profiles/ascend950.json"))
                        .GetCapabilities();
  const auto request = BuildRunRequest(config, info, "kernel.so");
  EXPECT_EQ(request.config.performance.metric, "latency_ms");
  EXPECT_EQ(request.warmup_count, 3U);
  EXPECT_EQ(request.kernel_count, 7U);
  EXPECT_EQ(request.config.performance.kernel_count, 7U);
  EXPECT_EQ(request.config.performance.warmup_count, 3U);
  EXPECT_TRUE(request.config.performance.required);
  EXPECT_TRUE(request.profiler);
}

TEST(AscendCBackendTest, KeepsDeclaredProfilerSeparateFromActualAvailability) {
  auto config = Config();
  config.performance.required = true;
  config.performance.profiler = true;
  const auto info = AscendCBackend(LoadDeviceProfile("autofuse/tests/st/device_validation/profiles/ascend950.json"))
                        .GetCapabilities();
  const auto request = BuildRunRequest(config, info, "kernel.so");
  RunReport report;
  report.performance.declared = request.config.performance;
  report.performance.actual.profiler_tool_available = false;
  EXPECT_TRUE(report.performance.declared.required);
  EXPECT_TRUE(report.performance.declared.profiler);
  EXPECT_FALSE(report.performance.actual.profiler_tool_available);
}

void AssertSuccessfulRunReportState(bool profiler_requested, bool profiler_tool_available) {
  auto profile = LoadDeviceProfile("autofuse/tests/st/device_validation/profiles/ascend950.json");
  profile.profiler_tool = profiler_tool_available ? "sh" : "";
  SuccessRuntime runtime;
  SuccessLoader loader;
  AscendCBackend backend(profile, &runtime, &loader);
  auto config = Config();
  config.performance.warmup_count = 1;
  config.performance.kernel_count = 2;
  auto request = BuildRunRequest(config, backend.GetCapabilities(), "fake-module.so");
  request.profiler = profiler_requested;
  request.host_inputs = {{0, 0, 0, 0}};
  RunResult result;

  ASSERT_EQ(backend.Run(request, &result), Status::kOk);
  EXPECT_EQ(result.report.performance.actual.profiler_requested, profiler_requested);
  EXPECT_EQ(result.report.performance.actual.profiler_tool_available, profiler_tool_available);
  EXPECT_FALSE(result.report.performance.actual.profiler_collected);
  const std::string expected_timing = "runner_wall_clock";
  const std::string expected_metric = "runner_wall_clock";
  EXPECT_EQ(result.report.performance.actual.timing_source, expected_timing);
  EXPECT_EQ(result.report.performance.actual.metric, expected_metric);
  EXPECT_EQ(result.report.performance.actual.unit, "ms");
  const auto json = SerializeReport(result.report).at("performance").at("actual");
  EXPECT_EQ(json.at("profiler_requested"), profiler_requested);
  EXPECT_EQ(json.at("profiler_tool_available"), profiler_tool_available);
  EXPECT_FALSE(json.at("profiler_collected"));
  EXPECT_EQ(json.at("timing_source"), expected_timing);
  EXPECT_EQ(json.at("metric"), expected_metric);
  EXPECT_EQ(json.at("unit"), "ms");
}

TEST(AscendCBackendTest, SuccessfulRunPreservesRequestedProfilerStateThroughSerializedReport) {
  AssertSuccessfulRunReportState(true, true);
}

TEST(AscendCBackendTest, SerializesActualAbiInReport) {
  RunReport report;
  report.abi = "AutofuseLaunch";
  report.variant = "unfused";
  EXPECT_EQ(SerializeReport(report).at("abi"), "AutofuseLaunch");
  EXPECT_EQ(SerializeReport(report).at("variant"), "unfused");
}

TEST(AscendCBackendTest, RejectsRequestedAbiAfterLoadingModule) {
  auto profile = LoadDeviceProfile("autofuse/tests/st/device_validation/profiles/ascend950.json");
  SuccessRuntime runtime;
  SuccessLoader loader;
  AscendCBackend backend(profile, &runtime, &loader);
  auto request = BuildRunRequest(Config(), backend.GetCapabilities(), "fake-module.so");
  request.abi = "AutofuseLaunch";
  request.host_inputs = {{0, 0, 0, 0}};
  RunResult result;

  EXPECT_NE(backend.Run(request, &result), Status::kOk);
  EXPECT_TRUE(result.error_code == "abi_mismatch" || result.error_code == "runtime_or_module_init");
}

TEST(AscendCBackendTest, SuccessfulRunPreservesWallClockStateThroughSerializedReport) {
  AssertSuccessfulRunReportState(false, false);
}

TEST(AscendCBackendTest, WritesFlatOutputBytesToRequestedFile) {
  const auto output_path = std::filesystem::temp_directory_path() / "device_validation_flat_output.bin";
  std::filesystem::remove(output_path);
  const std::vector<uint8_t> expected = {0x01, 0x02, 0xa5, 0xff};
  std::string error;

  ASSERT_TRUE(WriteFlatOutputs({output_path.string()}, {expected}, output_path.parent_path().string(), &error))
      << error;
  std::ifstream output(output_path, std::ios::binary);
  const std::vector<uint8_t> actual((std::istreambuf_iterator<char>(output)), std::istreambuf_iterator<char>());
  EXPECT_EQ(actual, expected);
  std::filesystem::remove(output_path);
}

TEST(AscendCBackendTest, FailedSecondOutputLeavesEarlierOutputsUntouchedAndCleansTemps) {
  const auto test_dir = std::filesystem::temp_directory_path() / "device_validation_flat_atomic_failure";
  std::filesystem::remove_all(test_dir);
  ASSERT_TRUE(std::filesystem::create_directories(test_dir));
  const auto first_path = test_dir / "first.bin";
  const auto missing_parent_path = test_dir / "missing" / "second.bin";
  std::string error;

  ASSERT_FALSE(WriteFlatOutputs({first_path.string(), missing_parent_path.string()}, {{0x01}, {0x02}},
                                test_dir.string(), &error));
  EXPECT_EQ(error, "output parent directory is unavailable");
  EXPECT_FALSE(std::filesystem::exists(first_path));
  EXPECT_FALSE(std::filesystem::exists(test_dir / "missing" / "second.bin"));
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(test_dir), std::filesystem::directory_iterator()), 0);
  std::filesystem::remove_all(test_dir);
}

TEST(AscendCBackendTest, RejectsOutputCountMismatchWithoutCreatingFiles) {
  const auto test_dir = std::filesystem::temp_directory_path() / "device_validation_flat_count_mismatch";
  std::filesystem::remove_all(test_dir);
  ASSERT_TRUE(std::filesystem::create_directories(test_dir));
  std::string error;

  ASSERT_FALSE(WriteFlatOutputs({(test_dir / "first.bin").string()}, {}, test_dir.string(), &error));
  EXPECT_EQ(error, "output count does not match result");
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(test_dir), std::filesystem::directory_iterator()), 0);
  std::filesystem::remove_all(test_dir);
}

TEST(AscendCBackendTest, RejectsDuplicateOutputPathsWithoutCreatingFiles) {
  const auto test_dir = std::filesystem::temp_directory_path() / "device_validation_flat_duplicate";
  std::filesystem::remove_all(test_dir);
  ASSERT_TRUE(std::filesystem::create_directories(test_dir));
  std::string error;

  const auto output_path = test_dir / "output.bin";
  ASSERT_FALSE(
      WriteFlatOutputs({output_path.string(), output_path.string()}, {{0x01}, {0x02}}, test_dir.string(), &error));
  EXPECT_EQ(error, "output path is duplicated");
  EXPECT_FALSE(std::filesystem::exists(output_path));
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(test_dir), std::filesystem::directory_iterator()), 0);
  std::filesystem::remove_all(test_dir);
}

TEST(AscendCBackendTest, RejectsExistingRegularFileWithoutChangingIt) {
  const auto test_dir = std::filesystem::temp_directory_path() / "device_validation_flat_existing_file";
  std::filesystem::remove_all(test_dir);
  ASSERT_TRUE(std::filesystem::create_directories(test_dir));
  const auto output_path = test_dir / "output.bin";
  ASSERT_TRUE(std::ofstream(output_path, std::ios::binary).put('\x55'));
  std::string error;

  ASSERT_FALSE(WriteFlatOutputs({output_path.string()}, {{0x01}}, test_dir.string(), &error));
  EXPECT_EQ(error, "output file already exists");
  std::ifstream output(output_path, std::ios::binary);
  EXPECT_EQ(std::vector<uint8_t>((std::istreambuf_iterator<char>(output)), std::istreambuf_iterator<char>()),
            std::vector<uint8_t>({0x55}));
  std::filesystem::remove_all(test_dir);
}

TEST(AscendCBackendTest, RejectsExistingSymlinkWithoutChangingTarget) {
  const auto test_dir = std::filesystem::temp_directory_path() / "device_validation_flat_symlink";
  std::filesystem::remove_all(test_dir);
  ASSERT_TRUE(std::filesystem::create_directories(test_dir));
  const auto target_path = test_dir / "target.bin";
  const auto output_path = test_dir / "output.bin";
  ASSERT_TRUE(std::ofstream(target_path, std::ios::binary).put('\x66'));
  std::filesystem::create_symlink(target_path, output_path);
  std::string error;

  ASSERT_FALSE(WriteFlatOutputs({output_path.string()}, {{0x01}}, test_dir.string(), &error));
  EXPECT_EQ(error, "output file already exists");
  std::ifstream target(target_path, std::ios::binary);
  EXPECT_EQ(std::vector<uint8_t>((std::istreambuf_iterator<char>(target)), std::istreambuf_iterator<char>()),
            std::vector<uint8_t>({0x66}));
  EXPECT_TRUE(std::filesystem::is_symlink(output_path));
  std::filesystem::remove_all(test_dir);
}

TEST(AscendCBackendTest, RejectsParentSymlinkEscapingArtifactDirectory) {
  const auto test_dir = std::filesystem::temp_directory_path() / "device_validation_flat_symlink_escape";
  std::filesystem::remove_all(test_dir);
  const auto artifact_dir = test_dir / "artifact" / "linked";
  const auto outside_dir = test_dir / "outside";
  ASSERT_TRUE(std::filesystem::create_directories(artifact_dir));
  ASSERT_TRUE(std::filesystem::create_directories(outside_dir));
  std::filesystem::create_symlink(outside_dir, artifact_dir / "escape");
  const auto output_path = artifact_dir / "escape" / "output.bin";
  std::string error;

  ASSERT_FALSE(WriteFlatOutputs({output_path.string()}, {{0x01}}, (test_dir / "artifact").string(), &error));
  EXPECT_EQ(error, "output path is outside artifact directory");
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(outside_dir), std::filesystem::directory_iterator()), 0);
  std::filesystem::remove_all(test_dir);
}

TEST(AscendCBackendTest, FlushFailureCleansCurrentTemporaryFile) {
  const auto test_dir = std::filesystem::temp_directory_path() / "device_validation_flat_flush_failure";
  std::filesystem::remove_all(test_dir);
  ASSERT_TRUE(std::filesystem::create_directories(test_dir));
  FlatOutputHooksGuard hooks({1, 0, 0});
  std::string error;

  ASSERT_FALSE(WriteFlatOutputs({(test_dir / "output.bin").string()}, {{0x01}}, test_dir.string(), &error));
  EXPECT_EQ(error, "output file cannot be written");
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(test_dir), std::filesystem::directory_iterator()), 0);
  std::filesystem::remove_all(test_dir);
}

TEST(AscendCBackendTest, CloseFailureCleansCurrentTemporaryFile) {
  const auto test_dir = std::filesystem::temp_directory_path() / "device_validation_flat_close_failure";
  std::filesystem::remove_all(test_dir);
  ASSERT_TRUE(std::filesystem::create_directories(test_dir));
  FlatOutputHooksGuard hooks({0, 1, 0});
  std::string error;

  ASSERT_FALSE(WriteFlatOutputs({(test_dir / "output.bin").string()}, {{0x01}}, test_dir.string(), &error));
  EXPECT_EQ(error, "output file cannot be written");
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(test_dir), std::filesystem::directory_iterator()), 0);
  std::filesystem::remove_all(test_dir);
}

TEST(AscendCBackendTest, WritesMultipleFlatOutputsConsistently) {
  const auto test_dir = std::filesystem::temp_directory_path() / "device_validation_flat_multiple_success";
  std::filesystem::remove_all(test_dir);
  ASSERT_TRUE(std::filesystem::create_directories(test_dir));
  const auto first_path = test_dir / "first.bin";
  const auto second_path = test_dir / "second.bin";
  const std::vector<std::vector<uint8_t>> expected = {{0x01, 0x02}, {0xa5, 0xff}};
  std::string error;

  ASSERT_TRUE(WriteFlatOutputs({first_path.string(), second_path.string()}, expected, test_dir.string(), &error))
      << error;
  const std::vector<std::filesystem::path> output_paths = {first_path, second_path};
  for (size_t i = 0; i < output_paths.size(); ++i) {
    std::ifstream output(output_paths[i], std::ios::binary);
    EXPECT_EQ(std::vector<uint8_t>((std::istreambuf_iterator<char>(output)), std::istreambuf_iterator<char>()),
              expected[i]);
  }
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(test_dir), std::filesystem::directory_iterator()), 2);
  std::filesystem::remove_all(test_dir);
}

TEST(AscendCBackendTest, ReportsRenameFailureAndPreservesExistingDirectory) {
  const auto test_dir = std::filesystem::temp_directory_path() / "device_validation_flat_rename_failure";
  std::filesystem::remove_all(test_dir);
  ASSERT_TRUE(std::filesystem::create_directories(test_dir));
  const auto output_path = test_dir / "existing_directory";
  ASSERT_TRUE(std::filesystem::create_directory(output_path));
  std::string error;

  ASSERT_FALSE(WriteFlatOutputs({output_path.string()}, {{0x01, 0x02}}, test_dir.string(), &error));
  EXPECT_EQ(error, "output file already exists");
  EXPECT_TRUE(std::filesystem::is_directory(output_path));
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(test_dir), std::filesystem::directory_iterator()), 1);
  std::filesystem::remove_all(test_dir);
}

TEST(AscendCBackendTest, InjectedNthRenameFailureCleansTempsWithoutCreatingTargets) {
  const auto test_dir = std::filesystem::temp_directory_path() / "device_validation_flat_nth_rename_failure";
  std::filesystem::remove_all(test_dir);
  ASSERT_TRUE(std::filesystem::create_directories(test_dir));
  FlatOutputHooksGuard hooks({0, 0, 2});
  const auto first_path = test_dir / "first.bin";
  const auto second_path = test_dir / "second.bin";
  std::string error;

  ASSERT_FALSE(
      WriteFlatOutputs({first_path.string(), second_path.string()}, {{0x01}, {0x02}}, test_dir.string(), &error));
  EXPECT_EQ(error, "output file cannot be written");
  EXPECT_TRUE(std::filesystem::exists(first_path));
  EXPECT_FALSE(std::filesystem::exists(second_path));
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(test_dir), std::filesystem::directory_iterator()), 1);
  std::filesystem::remove_all(test_dir);
}

TEST(AscendCBackendTest, TargetCreatedBeforeCommitIsNotOverwritten) {
  const auto test_dir = std::filesystem::temp_directory_path() / "device_validation_flat_target_race";
  std::filesystem::remove_all(test_dir);
  ASSERT_TRUE(std::filesystem::create_directories(test_dir));
  FlatOutputHooksGuard hooks({0, 0, 0, 1});
  const auto output_path = test_dir / "output.bin";
  std::string error;

  ASSERT_FALSE(WriteFlatOutputs({output_path.string()}, {{0x01}}, test_dir.string(), &error));
  EXPECT_EQ(error, "output file cannot be written");
  std::ifstream output(output_path, std::ios::binary);
  EXPECT_EQ(std::vector<uint8_t>((std::istreambuf_iterator<char>(output)), std::istreambuf_iterator<char>()),
            std::vector<uint8_t>({0x55}));
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(test_dir), std::filesystem::directory_iterator()), 1);
  std::filesystem::remove_all(test_dir);
}

TEST(AscendCBackendTest, TemporaryFileCollisionSelectsNewPathWithoutResidue) {
  const auto test_dir = std::filesystem::temp_directory_path() / "device_validation_flat_temp_race";
  std::filesystem::remove_all(test_dir);
  ASSERT_TRUE(std::filesystem::create_directories(test_dir));
  FlatOutputHooksGuard hooks({0, 0, 0, 0, 1});
  const auto output_path = test_dir / "output.bin";
  std::string error;

  ASSERT_TRUE(WriteFlatOutputs({output_path.string()}, {{0x01, 0x02}}, test_dir.string(), &error)) << error;
  std::ifstream output(output_path, std::ios::binary);
  EXPECT_EQ(std::vector<uint8_t>((std::istreambuf_iterator<char>(output)), std::istreambuf_iterator<char>()),
            std::vector<uint8_t>({0x01, 0x02}));
  std::filesystem::path competing_path;
  for (const auto &entry : std::filesystem::directory_iterator(test_dir)) {
    if (entry.path().filename().string().find(".tmp.") != std::string::npos) {
      competing_path = entry.path();
    }
  }
  ASSERT_FALSE(competing_path.empty());
  std::ifstream competing(competing_path, std::ios::binary);
  EXPECT_EQ(std::vector<uint8_t>((std::istreambuf_iterator<char>(competing)), std::istreambuf_iterator<char>()),
            std::vector<uint8_t>({0x33}));
  std::filesystem::remove(competing_path);
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(test_dir), std::filesystem::directory_iterator()), 1);
  std::filesystem::remove_all(test_dir);
}

TEST(AscendCBackendTest, TemporarySymlinkCollisionDoesNotTruncateSymlinkTarget) {
  const auto test_dir = std::filesystem::temp_directory_path() / "device_validation_flat_temp_symlink_race";
  std::filesystem::remove_all(test_dir);
  ASSERT_TRUE(std::filesystem::create_directories(test_dir));
  FlatOutputHooksGuard hooks({0, 0, 0, 0, 0, 1});
  const auto output_path = test_dir / "output.bin";
  const auto symlink_target = test_dir / "output.bin.symlink-target";
  std::string error;

  ASSERT_TRUE(WriteFlatOutputs({output_path.string()}, {{0xa1}}, test_dir.string(), &error)) << error;
  std::ifstream target(symlink_target, std::ios::binary);
  EXPECT_EQ(std::vector<uint8_t>((std::istreambuf_iterator<char>(target)), std::istreambuf_iterator<char>()),
            std::vector<uint8_t>({0x44}));
  size_t temporary_symlink_count = 0;
  for (const auto &entry : std::filesystem::directory_iterator(test_dir)) {
    if (std::filesystem::is_symlink(entry.path()) &&
        entry.path().filename().string().find(".tmp.") != std::string::npos) {
      ++temporary_symlink_count;
    }
  }
  EXPECT_EQ(temporary_symlink_count, 1U);
  std::filesystem::remove(output_path);
  std::filesystem::remove(symlink_target);
  for (const auto &entry : std::filesystem::directory_iterator(test_dir)) {
    std::filesystem::remove(entry.path());
  }
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(test_dir), std::filesystem::directory_iterator()), 0);
  std::filesystem::remove_all(test_dir);
}

TEST(AscendCBackendTest, SuccessfulRunProvidesBytesForFlatOutputAndReportMetadata) {
  auto profile = LoadDeviceProfile("autofuse/tests/st/device_validation/profiles/ascend950.json");
  SuccessRuntime runtime;
  SuccessLoader loader;
  AscendCBackend backend(profile, &runtime, &loader);
  auto request = BuildRunRequest(Config(), backend.GetCapabilities(), "fake-module.so");
  request.host_inputs = {{0, 0, 0, 0}};
  RunResult result;

  ASSERT_EQ(backend.Run(request, &result), Status::kOk);
  ASSERT_EQ(result.output_bytes.size(), 1U);
  ASSERT_EQ(result.outputs.size(), 1U);
  ASSERT_EQ(result.output_bytes[0].size(), 4U);

  const auto output_path = std::filesystem::temp_directory_path() / "device_validation_backend_output.bin";
  std::filesystem::remove(output_path);
  std::string error;
  ASSERT_TRUE(WriteFlatOutputs({output_path.string()}, result.output_bytes, output_path.parent_path().string(), &error))
      << error;
  std::ifstream output(output_path, std::ios::binary);
  const std::vector<uint8_t> actual((std::istreambuf_iterator<char>(output)), std::istreambuf_iterator<char>());
  EXPECT_EQ(actual, result.output_bytes[0]);

  auto report = SerializeReport(result.report);
  report["outputs"] = nlohmann::json::array();
  for (const auto &output_view : result.outputs) {
    report["outputs"].push_back({{"dtype", output_view.dtype}, {"shape", output_view.shape}});
  }
  ASSERT_EQ(report["stage_status"], "passed");
  ASSERT_EQ(report["outputs"].size(), 1U);
  EXPECT_EQ(report["outputs"][0]["dtype"], "float16");
  EXPECT_EQ(report["outputs"][0]["shape"], std::vector<int64_t>({2}));
  std::filesystem::remove(output_path);
}

TEST(AscendCBackendTest, SuccessfulRunPreservesRequestedButNotCollectedStateThroughSerializedReport) {
  AssertSuccessfulRunReportState(true, true);
}
}  // namespace
}  // namespace device_validation
