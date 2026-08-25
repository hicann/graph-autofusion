/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensor_buffer.h"

#include <algorithm>
#include <limits>

namespace device_validation {
namespace {
size_t DtypeSize(const std::string &dtype) {
  if (dtype == "float16" || dtype == "bfloat16") return 2;
  if (dtype == "float32" || dtype == "int32" || dtype == "uint32") return 4;
  if (dtype == "int64" || dtype == "uint64") return 8;
  if (dtype == "int8" || dtype == "uint8" || dtype == "bool") return 1;
  return 0;
}
}  // namespace

TensorBuffer::~TensorBuffer() {
  (void)Release();
}

Status TensorBuffer::Allocate(const TensorSpec &spec, AclRuntime *runtime) {
  if (runtime == nullptr || runtime->stream() == nullptr || spec.shape.empty()) return Status::kInvalidArgument;
  if (device_ptr_ != nullptr) return Status::kInvalidArgument;
  const auto element_size = DtypeSize(spec.dtype);
  if (element_size == 0) return Status::kUnsupported;
  size_t elements = 1;
  for (const auto dim : spec.shape) {
    if (dim <= 0 || static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() / elements)
      return dim == 0 ? Status::kInvalidArgument : Status::kOverflow;
    elements *= static_cast<size_t>(dim);
  }
  if (elements > std::numeric_limits<size_t>::max() / element_size) return Status::kOverflow;
  const auto bytes = elements * element_size;
  if (bytes == 0) return Status::kInvalidArgument;
  void *ptr = nullptr;
  const auto status = runtime->api()->Malloc(&ptr, bytes);
  if (status != Status::kOk || ptr == nullptr) return status == Status::kOk ? Status::kRuntimeError : status;
  device_ptr_ = ptr;
  runtime_ = runtime;
  lifetime_ = runtime->lifetime();
  lifetime_->allocations.push_back(device_ptr_);
  size_ = bytes;
  runtime_->OnAllocation();
  return Status::kOk;
}

Status TensorBuffer::Release() {
  if (device_ptr_ == nullptr) return Status::kOk;
  if (lifetime_ == nullptr || !lifetime_->active) return Status::kInvalidArgument;
  const auto lifetime = lifetime_;
  if (lifetime->stream != nullptr) {
    const auto status = lifetime->api->Synchronize(lifetime->stream);
    if (status != Status::kOk) return status;
  }
  const auto status = lifetime->api->Free(device_ptr_);
  if (status != Status::kOk) return status;
  const auto it = std::find(lifetime->allocations.begin(), lifetime->allocations.end(), device_ptr_);
  if (it != lifetime->allocations.end()) lifetime->allocations.erase(it);
  if (lifetime->runtime != nullptr) lifetime->runtime->OnFree();
  const bool runtime_gone = lifetime->runtime == nullptr;
  if (runtime_gone) lifetime->active = true;
  device_ptr_ = nullptr;
  runtime_ = nullptr;
  lifetime_.reset();
  size_ = 0;
  if (runtime_gone && lifetime->allocations.empty()) (void)lifetime->Cleanup();
  return Status::kOk;
}

Status TensorBuffer::CopyToDevice(const void *src, size_t bytes) {
  if (device_ptr_ == nullptr || src == nullptr || bytes != size_) return Status::kInvalidArgument;
  return runtime_->api()->CopyToDevice(device_ptr_, src, bytes);
}

Status TensorBuffer::CopyToHost(void *dst, size_t bytes) {
  if (device_ptr_ == nullptr || dst == nullptr || bytes != size_) return Status::kInvalidArgument;
  return runtime_->api()->CopyToHost(dst, device_ptr_, bytes);
}
}  // namespace device_validation
