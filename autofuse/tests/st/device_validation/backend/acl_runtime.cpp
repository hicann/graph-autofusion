/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl_runtime.h"

namespace device_validation {
RuntimeLifetime::~RuntimeLifetime() {
  (void)Cleanup();
}

Status RuntimeLifetime::Cleanup() {
  if (!active || api == nullptr) return Status::kOk;
  if (stream != nullptr) {
    const auto status = api->Synchronize(stream);
    if (status != Status::kOk) return status;
  }
  for (auto it = allocations.begin(); it != allocations.end();) {
    const auto status = api->Free(*it);
    if (status != Status::kOk) return status;
    it = allocations.erase(it);
  }
  if (stream != nullptr) {
    const auto status = api->DestroyStream(stream);
    if (status != Status::kOk) return status;
    stream = nullptr;
  }
  if (device_set) {
    const auto status = api->ResetDevice(device_id);
    if (status != Status::kOk) return status;
    device_set = false;
  }
  if (initialized) {
    const auto status = api->Finalize();
    if (status != Status::kOk) return status;
    initialized = false;
  }
  active = false;
  return Status::kOk;
}

AclRuntime::~AclRuntime() {
  if (allocation_count_ == 0) (void)Finalize();
  if (lifetime_ != nullptr) {
    lifetime_->runtime = nullptr;
  }
}

Status AclRuntime::Initialize(int32_t device_id) {
  if (api_ == nullptr || initialized_) return Status::kInvalidArgument;
  auto status = api_->Init();
  if (status != Status::kOk) return status;
  status = api_->SetDevice(device_id);
  if (status != Status::kOk) {
    api_->Finalize();
    return status;
  }
  device_set_ = true;
  status = api_->CreateStream(&stream_);
  if (status != Status::kOk) {
    api_->ResetDevice(device_id);
    api_->Finalize();
    device_set_ = false;
    return status;
  }
  device_id_ = device_id;
  initialized_ = true;
  lifetime_ = std::make_shared<RuntimeLifetime>();
  lifetime_->api = api_;
  lifetime_->active = true;
  lifetime_->runtime = this;
  lifetime_->stream = stream_;
  lifetime_->device_id = device_id_;
  lifetime_->device_set = device_set_;
  lifetime_->initialized = true;
  return Status::kOk;
}

Status AclRuntime::Synchronize() {
  if (!initialized_ || stream_ == nullptr) return Status::kInvalidArgument;
  return api_->Synchronize(stream_);
}

Status AclRuntime::Finalize() {
  if (allocation_count_ != 0) return Status::kInvalidArgument;
  if (lifetime_ == nullptr) return Status::kInvalidArgument;
  const auto result = lifetime_->Cleanup();
  stream_ = lifetime_->stream;
  device_set_ = lifetime_->device_set;
  initialized_ = lifetime_->initialized;
  if (result != Status::kOk) return result;
  initialized_ = false;
  device_id_ = -1;
  if (lifetime_ != nullptr) {
    lifetime_->initialized = false;
    lifetime_->active = false;
  }
  return result;
}
}  // namespace device_validation
