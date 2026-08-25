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

#include "support_matrix.h"
#include "verifier.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace device_validation {

struct PerformanceActual {
  size_t sample_count = 0;
  std::vector<double> samples;
  size_t warmup_count = 0;
  double min_ms = 0.0;
  double mean_ms = 0.0;
  double p50_ms = 0.0;
  double p90_ms = 0.0;
  double p99_ms = 0.0;
  double max_ms = 0.0;
  double min_us = 0.0;
  double mean_us = 0.0;
  double p50_us = 0.0;
  double p90_us = 0.0;
  double p99_us = 0.0;
  double max_us = 0.0;
  size_t kernel_count = 0;
  double tiling_time_us = 0.0;
  size_t workspace_size = 0;
  size_t block_dimension = 0;
  bool profiler_requested = false;
  bool profiler_tool_available = false;
  bool profiler_collected = false;
  std::string profiler_collect_dir;
  std::string metric = "runner_wall_clock";
  std::string unit = "ms";
  std::string timing_source = "runner_wall_clock";
};

struct PerformanceReport {
  PerformanceSpec declared;
  PerformanceActual actual;
};

PerformanceReport SummarizeSamples(const std::vector<double> &samples, const PerformanceSpec &spec);
PerformanceReport SummarizeSamplesWithRuntimeState(const std::vector<double> &samples, const PerformanceSpec &spec,
                                                   bool profiler_requested, bool profiler_tool_available,
                                                   bool profiler_collected, const std::string &timing_source);

struct RunReport {
  int schema_version = 2;
  std::string case_id;
  std::string backend;
  std::string abi = "AutofuseLaunchV2";
  std::string requested_abi;
  nlohmann::json abi_metadata = nlohmann::json::object();
  std::string variant;
  std::string soc_profile;
  std::string metric = "latency_ms";
  std::string stage_status;
  std::string stage = "preflight";
  std::string error_code;
  std::string profile;
  nlohmann::json run_parameters = nlohmann::json::object();
  std::string reason;
  SupportDecision support;
  PrecisionReport precision;
  PerformanceReport performance;
  std::vector<std::string> artifact_paths;
};

nlohmann::json SerializeReport(const RunReport &report);

}  // namespace device_validation
