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
#include "inprocess_profiler.h"
#include "kernel_module.h"
#include "profile.h"
#include "report.h"
#include "support_matrix.h"

#include <memory>
#include <string>

namespace device_validation {
struct BackendInfo {
  std::string name;
  std::string profile;
  std::string abi;
  BackendCapabilities capabilities;
};

struct RunResult {
  Status status = Status::kRuntimeError;
  RunReport report;
  std::vector<TensorView> outputs;
  std::vector<std::vector<uint8_t>> output_bytes;
  std::string reason;
  std::string stage = "preflight";
  std::string error_code;
};

RunRequest BuildRunRequest(const CaseConfig &, const BackendInfo &, const std::string &module_path);
TensorView DecodeOutputForTest(const std::string &, const std::vector<int64_t> &, const std::vector<uint8_t> &);

class ExecutionBackend {
 public:
  virtual ~ExecutionBackend() = default;
  virtual BackendInfo GetCapabilities() const = 0;
  virtual Status Validate(const CaseConfig &) const = 0;
  virtual Status Run(RunRequest &, RunResult *) = 0;
};

class AscendCBackend final : public ExecutionBackend {
 public:
  explicit AscendCBackend(DeviceProfile profile, RuntimeApi *runtime_api = nullptr,
                          ModuleLoader *module_loader = nullptr);
  BackendInfo GetCapabilities() const override;
  Status Validate(const CaseConfig &) const override;
  Status Run(RunRequest &, RunResult *) override;
  Status InitializeReport(RunResult &result, const RunRequest &request, const BackendInfo &info) const;
  void set_profiler(InProcessProfiler *profiler, std::string profile_dir) {
    profiler_ = profiler;
    profile_dir_ = std::move(profile_dir);
  }

 private:
  DeviceProfile profile_;
  RuntimeApi *runtime_api_ = nullptr;
  ModuleLoader *module_loader_ = nullptr;
  InProcessProfiler *profiler_ = nullptr;
  std::string profile_dir_;
};

std::unique_ptr<ExecutionBackend> CreateExecutionBackend(const std::string &name,
                                                         const std::filesystem::path &profile_path = {});
}  // namespace device_validation
