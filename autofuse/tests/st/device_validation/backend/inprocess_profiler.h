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

#include "acl_runtime.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace device_validation {

class DeviceProfiler {
 public:
  virtual ~DeviceProfiler() = default;
  virtual Status Start(int32_t device_id, const std::string &output_dir) = 0;
  virtual Status Stop() = 0;
  virtual const std::string &last_error() const = 0;
};

class InProcessProfiler {
 public:
  InProcessProfiler() = default;
  explicit InProcessProfiler(std::unique_ptr<DeviceProfiler> impl) : impl_(std::move(impl)) {}
  ~InProcessProfiler();
  InProcessProfiler(const InProcessProfiler &) = delete;
  InProcessProfiler &operator=(const InProcessProfiler &) = delete;
  InProcessProfiler(InProcessProfiler &&other) noexcept
      : impl_(std::move(other.impl_)), started_(other.started_), last_error_(std::move(other.last_error_)) {
    other.started_ = false;
  }
  InProcessProfiler &operator=(InProcessProfiler &&other) noexcept {
    if (this != &other) {
      Stop();
      impl_ = std::move(other.impl_);
      started_ = other.started_;
      last_error_ = std::move(other.last_error_);
      other.started_ = false;
    }
    return *this;
  }

  Status Start(int32_t device_id, const std::string &output_dir);
  Status Stop();
  bool started() const {
    return started_;
  }
  const std::string &last_error() const {
    return last_error_;
  }

 private:
  std::unique_ptr<DeviceProfiler> impl_;
  bool started_ = false;
  std::string last_error_;
};

std::unique_ptr<DeviceProfiler> CreateInProcessProfiler();

inline InProcessProfiler::~InProcessProfiler() {
  Stop();
}

inline Status InProcessProfiler::Start(int32_t device_id, const std::string &output_dir) {
  last_error_.clear();
  if (impl_ == nullptr) {
    last_error_ = "in-process profiler is unavailable";
    return Status::kRuntimeError;
  }
  if (started_) {
    last_error_ = "in-process profiler is already started";
    return Status::kInvalidArgument;
  }
  const Status status = impl_->Start(device_id, output_dir);
  if (status != Status::kOk) {
    last_error_ = impl_->last_error().empty() ? "in-process profiler start failed" : impl_->last_error();
    return status;
  }
  started_ = true;
  return Status::kOk;
}

inline Status InProcessProfiler::Stop() {
  last_error_.clear();
  if (impl_ == nullptr) {
    last_error_ = "in-process profiler is unavailable";
    return Status::kRuntimeError;
  }
  if (!started_) return Status::kOk;
  started_ = false;
  const Status status = impl_->Stop();
  if (status != Status::kOk) {
    last_error_ = impl_->last_error().empty() ? "in-process profiler stop failed" : impl_->last_error();
    return status;
  }
  return Status::kOk;
}

}  // namespace device_validation
