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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace device_validation {
enum class Status { kOk, kInvalidArgument, kOverflow, kUnsupported, kNotFound, kRuntimeError };

class RuntimeApi {
 public:
  virtual ~RuntimeApi() = default;
  virtual Status Init() = 0;
  virtual Status SetDevice(int32_t) = 0;
  virtual Status CreateStream(void **) = 0;
  virtual Status Synchronize(void *) = 0;
  virtual Status DestroyStream(void *) = 0;
  virtual Status ResetDevice(int32_t) = 0;
  virtual Status Finalize() = 0;
  virtual Status Malloc(void **, size_t) = 0;
  virtual Status Free(void *) = 0;
  virtual Status CopyToDevice(void *, const void *, size_t) = 0;
  virtual Status CopyToHost(void *, const void *, size_t) = 0;
};

class AclRuntime;
struct RuntimeLifetime {
  ~RuntimeLifetime();
  RuntimeApi *api = nullptr;
  bool active = false;
  AclRuntime *runtime = nullptr;
  void *stream = nullptr;
  int32_t device_id = -1;
  bool device_set = false;
  bool initialized = false;
  std::vector<void *> allocations;
  Status Cleanup();
};

RuntimeApi *CreateAclRuntimeApi() __attribute__((weak));

class AclRuntime {
 public:
  explicit AclRuntime(RuntimeApi *api) : api_(api) {}
  ~AclRuntime();
  Status Initialize(int32_t device_id);
  Status Synchronize();
  Status Finalize();
  bool initialized() const {
    return initialized_;
  }
  bool HasAllocations() const {
    return allocation_count_ != 0;
  }
  void OnAllocation() {
    ++allocation_count_;
  }
  void OnFree() {
    if (allocation_count_ != 0) --allocation_count_;
  }
  RuntimeApi *api() const {
    return api_;
  }
  void *stream() const {
    return stream_;
  }
  int32_t device_id() const {
    return device_id_;
  }
  std::shared_ptr<RuntimeLifetime> lifetime() const {
    return lifetime_;
  }

 private:
  RuntimeApi *api_;
  void *stream_ = nullptr;
  int32_t device_id_ = -1;
  bool initialized_ = false;
  bool device_set_ = false;
  size_t allocation_count_ = 0;
  std::shared_ptr<RuntimeLifetime> lifetime_;
};
}  // namespace device_validation
