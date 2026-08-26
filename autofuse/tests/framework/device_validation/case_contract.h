/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace device_validation {
enum class Capability { kCompile, kFunctional, kPrecision, kPerformance };
enum class SupportStatus { kRequired, kOptional, kUnsupported };

struct InputSpec {
  std::vector<int64_t> shape;
  std::string dtype;
  bool dynamic = false;
};
using OutputSpec = InputSpec;
struct VerificationSpec {
  bool functional = true;
  bool precision = false;
  double atol = 1e-5;
  double rtol = 1e-5;
};
struct PerformanceSpec {
  bool required = false;
  bool profiler = false;
  std::string metric = "latency_ms";
  size_t warmup_count = 0;
  size_t kernel_count = 1;
  double tiling_time_us = 0.0;
  size_t workspace_size = 0;
  size_t block_dimension = 0;
  bool profiler_available = false;
};
struct SupportEntry {
  std::string case_id;
  std::string backend;
  std::string soc;
  std::map<Capability, SupportStatus> capabilities;
  std::vector<std::string> dtypes;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<std::string> input_dtypes;
  std::vector<std::string> output_dtypes;
  std::vector<std::string> variants;
  std::string allowed_abi;
};
struct CaseConfig {
  int schema_version = 0;
  std::string case_id;
  std::string graph_name;
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  VerificationSpec verification;
  PerformanceSpec performance;
  std::vector<SupportEntry> support_matrix;
  std::vector<std::string> variants;
};

CaseConfig LoadCaseConfig(const std::filesystem::path &case_dir);
}  // namespace device_validation
