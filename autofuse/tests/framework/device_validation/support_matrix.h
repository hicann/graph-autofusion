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

#include "case_contract.h"

#include <map>
#include <string>
#include <cstdint>

namespace device_validation {
enum class CapabilityResult { kPassed, kFailed, kSkipped, kNotApplicable };
struct BackendCapabilities {
  bool compile = false;
  bool functional = false;
  bool precision = false;
  bool performance = false;
  bool profiler = false;
  bool dynamic_shape = false;
  bool multi_output = false;
  bool workspace = false;
};
struct SocCapabilities {
  std::string soc;
  std::vector<std::string> dtypes;
  std::vector<std::vector<int64_t>> shapes;
  uint64_t max_shape_elements = 0;
  std::string allowed_abi;
  bool profiler_optional = false;
  bool profiler_available = false;
};
struct RunRequest {
  std::string backend;
  std::string soc;
  bool profiler = false;
  std::string dtype;
  std::vector<std::vector<int64_t>> shapes;
  std::string module_path;
  int32_t device_id = 0;
  size_t warmup_count = 0;
  size_t kernel_count = 1;
  std::string abi = "AutofuseLaunchV2";
  std::string variant;
  bool dynamic_shape = false;
  bool multi_output = false;
  std::vector<std::vector<uint8_t>> host_inputs;
  CaseConfig config;
  CaseConfig capability_config;
};
struct CapabilityDecision {
  CapabilityResult result;
  std::string reason;
};
struct SupportDecision {
  std::map<Capability, CapabilityDecision> capabilities;
};

SupportDecision ResolveSupport(const CaseConfig &, const BackendCapabilities &, const SocCapabilities &,
                               const RunRequest &);
}  // namespace device_validation
