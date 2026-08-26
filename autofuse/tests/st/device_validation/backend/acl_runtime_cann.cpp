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

#include "acl/acl.h"

namespace device_validation {
namespace {
class CANNRuntimeApi final : public RuntimeApi {
 public:
  Status Init() override {
    return Convert(aclInit(nullptr));
  }
  Status SetDevice(int32_t device_id) override {
    return Convert(aclrtSetDevice(device_id));
  }
  Status CreateStream(void **stream) override {
    return Convert(aclrtCreateStream(reinterpret_cast<aclrtStream *>(stream)));
  }
  Status Synchronize(void *stream) override {
    return Convert(aclrtSynchronizeStream(stream));
  }
  Status DestroyStream(void *stream) override {
    return Convert(aclrtDestroyStream(stream));
  }
  Status ResetDevice(int32_t device_id) override {
    return Convert(aclrtResetDevice(device_id));
  }
  Status Finalize() override {
    return Convert(aclFinalize());
  }
  Status Malloc(void **ptr, size_t size) override {
    return Convert(aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
  }
  Status Free(void *ptr) override {
    return Convert(aclrtFree(ptr));
  }
  Status CopyToDevice(void *dst, const void *src, size_t size) override {
    return Convert(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE));
  }
  Status CopyToHost(void *dst, const void *src, size_t size) override {
    return Convert(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST));
  }

 private:
  static Status Convert(aclError status) {
    return status == ACL_SUCCESS ? Status::kOk : Status::kRuntimeError;
  }
};
}  // namespace

RuntimeApi *CreateAclRuntimeApi() {
  static CANNRuntimeApi api;
  return &api;
}
}  // namespace device_validation
