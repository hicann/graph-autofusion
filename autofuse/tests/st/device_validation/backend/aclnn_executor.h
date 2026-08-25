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
#include "tensor_buffer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace device_validation {

enum class AclnnOpKind { kUnknown, kIsInf, kLogicalOr, kMaskedFillScalar, kMaskedFillTensor };

inline AclnnOpKind AclnnOpKindFromName(const std::string &name) {
  if (name == "IsInf") return AclnnOpKind::kIsInf;
  if (name == "LogicalOr") return AclnnOpKind::kLogicalOr;
  if (name == "MaskedFillScalar") return AclnnOpKind::kMaskedFillScalar;
  if (name == "MaskedFillTensor") return AclnnOpKind::kMaskedFillTensor;
  return AclnnOpKind::kUnknown;
}

inline std::string AclnnOpName(AclnnOpKind kind) {
  switch (kind) {
    case AclnnOpKind::kIsInf:
      return "IsInf";
    case AclnnOpKind::kLogicalOr:
      return "LogicalOr";
    case AclnnOpKind::kMaskedFillScalar:
      return "MaskedFillScalar";
    case AclnnOpKind::kMaskedFillTensor:
      return "MaskedFillTensor";
    default:
      return "unknown";
  }
}

inline bool AclnnOpArityValid(AclnnOpKind kind, size_t input_count, size_t output_count) {
  switch (kind) {
    case AclnnOpKind::kIsInf:
      return input_count == 1 && output_count == 1;
    case AclnnOpKind::kLogicalOr:
      return input_count == 2 && output_count == 1;
    case AclnnOpKind::kMaskedFillScalar:
    case AclnnOpKind::kMaskedFillTensor:
      return input_count == 3 && output_count == 1;
    default:
      return false;
  }
}

struct AclnnRunResult {
  Status status = Status::kRuntimeError;
  std::string error_code;
  std::string reason;
  std::vector<double> samples;
  std::vector<std::vector<uint8_t>> output_bytes;
};

class AclnnExecutor {
 public:
  explicit AclnnExecutor(AclRuntime *runtime);
  Status Run(const std::string &op_name, const std::vector<TensorSpec> &input_specs, std::vector<TensorBuffer> &inputs,
             const std::vector<std::vector<uint8_t>> &host_inputs, const std::vector<TensorSpec> &output_specs,
             std::vector<TensorBuffer> &outputs, size_t warmup, size_t repeat, AclnnRunResult *result);

 private:
  Status ExecuteOnce(AclnnOpKind kind, const std::vector<TensorSpec> &input_specs, std::vector<TensorBuffer> &inputs,
                     const std::vector<std::vector<uint8_t>> &host_inputs, const std::vector<TensorSpec> &output_specs,
                     std::vector<TensorBuffer> &outputs, AclnnRunResult *result);
  AclRuntime *runtime_;
};

}  // namespace device_validation
