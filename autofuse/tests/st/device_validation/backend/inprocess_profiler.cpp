/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inprocess_profiler.h"

#include "acl/acl_prof.h"

#include <filesystem>
#include <string>

namespace device_validation {
namespace {
std::string AclProfErrorText(aclError status, const char *call) {
  return std::string(call) + " failed with error " + std::to_string(static_cast<int>(status));
}

class CANNDeviceProfiler final : public DeviceProfiler {
 public:
  ~CANNDeviceProfiler() override {
    if (started_ && config_ != nullptr) (void)aclprofStop(config_);
    Cleanup();
  }

  Status Start(int32_t device_id, const std::string &output_dir) override {
    error_.clear();
    if (initialized_) {
      error_ = "in-process profiler is already initialized";
      return Status::kInvalidArgument;
    }
    std::error_code directory_error;
    std::filesystem::create_directories(output_dir, directory_error);
    if (directory_error) {
      error_ = "cannot create profiler output directory: " + directory_error.message();
      return Status::kRuntimeError;
    }
    const aclError init_status = aclprofInit(output_dir.c_str(), output_dir.size());
    if (init_status != ACL_SUCCESS) {
      error_ = AclProfErrorText(init_status, "aclprofInit");
      return Status::kRuntimeError;
    }
    initialized_ = true;
    uint32_t device_ids[1] = {static_cast<uint32_t>(device_id)};
    config_ = aclprofCreateConfig(device_ids, 1, ACL_AICORE_NONE, nullptr, ACL_PROF_TASK_TIME);
    if (config_ == nullptr) {
      error_ = "aclprofCreateConfig failed";
      Cleanup();
      return Status::kRuntimeError;
    }
    const aclError start_status = aclprofStart(config_);
    if (start_status != ACL_SUCCESS) {
      error_ = AclProfErrorText(start_status, "aclprofStart");
      Cleanup();
      return Status::kRuntimeError;
    }
    started_ = true;
    return Status::kOk;
  }

  Status Stop() override {
    error_.clear();
    if (!started_ || config_ == nullptr) {
      error_ = "in-process profiler is not started";
      return Status::kInvalidArgument;
    }
    started_ = false;
    std::string failures;
    const aclError stop_status = aclprofStop(config_);
    if (stop_status != ACL_SUCCESS) failures += AclProfErrorText(stop_status, "aclprofStop") + "; ";
    const aclError destroy_status = aclprofDestroyConfig(config_);
    config_ = nullptr;
    if (destroy_status != ACL_SUCCESS) failures += AclProfErrorText(destroy_status, "aclprofDestroyConfig") + "; ";
    const aclError finalize_status = aclprofFinalize();
    initialized_ = false;
    if (finalize_status != ACL_SUCCESS) failures += AclProfErrorText(finalize_status, "aclprofFinalize") + "; ";
    if (!failures.empty()) {
      error_ = failures.substr(0, failures.size() - 2);
      return Status::kRuntimeError;
    }
    return Status::kOk;
  }

  const std::string &last_error() const override {
    return error_;
  }

 private:
  void Cleanup() {
    if (config_ != nullptr) (void)aclprofDestroyConfig(config_);
    config_ = nullptr;
    if (initialized_) (void)aclprofFinalize();
    initialized_ = false;
    started_ = false;
  }

  bool initialized_ = false;
  bool started_ = false;
  aclprofConfig *config_ = nullptr;
  std::string error_;
};
}  // namespace

std::unique_ptr<DeviceProfiler> CreateInProcessProfiler() {
  return std::make_unique<CANNDeviceProfiler>();
}
}  // namespace device_validation
