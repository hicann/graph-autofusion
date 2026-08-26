/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_executor.h"

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_is_inf.h"
#include "aclnnop/aclnn_logical_or.h"
#include "aclnnop/aclnn_masked_fill_scalar.h"
#include "aclnnop/aclnn_masked_fill_tensor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace device_validation {
namespace {

aclDataType AclnnDataType(const std::string &dtype) {
  if (dtype == "float16") return ACL_FLOAT16;
  if (dtype == "bfloat16") return ACL_BF16;
  if (dtype == "float32") return ACL_FLOAT;
  if (dtype == "int8") return ACL_INT8;
  if (dtype == "int32") return ACL_INT32;
  if (dtype == "int64") return ACL_INT64;
  if (dtype == "uint8") return ACL_UINT8;
  if (dtype == "uint32") return ACL_UINT32;
  if (dtype == "uint64") return ACL_UINT64;
  if (dtype == "bool") return ACL_BOOL;
  return ACL_DT_UNDEFINED;
}

size_t AclnnDtypeSize(const std::string &dtype) {
  if (dtype == "float16" || dtype == "bfloat16") return 2;
  if (dtype == "float32" || dtype == "int32" || dtype == "uint32") return 4;
  if (dtype == "int64" || dtype == "uint64") return 8;
  if (dtype == "int8" || dtype == "uint8" || dtype == "bool") return 1;
  return 0;
}

aclTensor *CreateAclnnTensor(const TensorSpec &spec, aclDataType dtype, void *device_ptr) {
  return aclCreateTensor(spec.shape.data(), spec.shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, spec.shape.data(),
                         spec.shape.size(), device_ptr);
}

aclTensor *CreateAclnnScalarTensor(aclDataType dtype, void *device_ptr) {
  return aclCreateTensor(nullptr, 0, dtype, nullptr, 0, ACL_FORMAT_ND, nullptr, 0, device_ptr);
}

bool TensorValid(const aclTensor *tensor) {
  return tensor != nullptr;
}

double DecodeFp16Bits(uint16_t bits) {
  const uint32_t sign = (bits >> 15) & 1U;
  const uint32_t exponent = (bits >> 10) & 0x1fU;
  const uint32_t fraction = bits & 0x3ffU;
  if (exponent == 0) return (sign ? -1.0 : 1.0) * std::ldexp(static_cast<double>(fraction), -24);
  if (exponent == 31) return fraction == 0 ? (sign ? -INFINITY : INFINITY) : NAN;
  return (sign ? -1.0 : 1.0) * std::ldexp(1.0 + fraction / 1024.0, static_cast<int>(exponent) - 15);
}

Status MakeValueScalar(const std::vector<uint8_t> &bytes, const std::string &dtype, aclScalar **scalar,
                       std::string *reason) {
  const size_t element_size = AclnnDtypeSize(dtype);
  if (element_size == 0 || bytes.size() < element_size) {
    *reason = "aclnn value tensor must be non-empty and use a supported dtype";
    return Status::kInvalidArgument;
  }
  if (dtype == "float16") {
    uint16_t bits = 0;
    std::copy_n(bytes.data(), sizeof(bits), reinterpret_cast<uint8_t *>(&bits));
    auto value = static_cast<float>(DecodeFp16Bits(bits));
    *scalar = aclCreateScalar(&value, ACL_FLOAT);
  } else if (dtype == "float32") {
    float value = 0.0F;
    std::copy_n(bytes.data(), sizeof(value), reinterpret_cast<uint8_t *>(&value));
    *scalar = aclCreateScalar(&value, ACL_FLOAT);
  } else if (dtype == "int64") {
    int64_t value = 0;
    std::copy_n(bytes.data(), sizeof(value), reinterpret_cast<uint8_t *>(&value));
    *scalar = aclCreateScalar(&value, ACL_INT64);
  } else if (dtype == "int32") {
    int32_t value = 0;
    std::copy_n(bytes.data(), sizeof(value), reinterpret_cast<uint8_t *>(&value));
    *scalar = aclCreateScalar(&value, ACL_INT32);
  } else {
    *reason = "aclnn scalar value dtype is unsupported";
    return Status::kInvalidArgument;
  }
  if (*scalar == nullptr) {
    *reason = "aclCreateScalar failed";
    return Status::kRuntimeError;
  }
  return Status::kOk;
}

std::string AclnnFailureReason(const std::string &op_name, const std::string &stage, aclnnStatus status) {
  std::ostringstream message;
  message << "aclnn " << op_name << " " << stage << " failed with status " << static_cast<int32_t>(status);
  return message.str();
}

struct AclnnOpTensors {
  std::vector<aclTensor *> inputs;
  std::vector<aclTensor *> outputs;
  aclScalar *value_scalar = nullptr;
};

void DestroyAclnnTensors(AclnnOpTensors *tensors) {
  for (auto *tensor : tensors->inputs) (void)aclDestroyTensor(tensor);
  for (auto *tensor : tensors->outputs) (void)aclDestroyTensor(tensor);
  if (tensors->value_scalar != nullptr) (void)aclDestroyScalar(tensors->value_scalar);
}

Status SetupAclnnTensors(AclnnOpKind kind, const std::string &op_name, const std::vector<TensorSpec> &input_specs,
                         const std::vector<std::vector<uint8_t>> &host_inputs, std::vector<TensorBuffer> &inputs,
                         const std::vector<TensorSpec> &output_specs, std::vector<TensorBuffer> &outputs,
                         AclnnOpTensors *tensors, AclnnRunResult *result) {
  for (size_t index = 0; index < input_specs.size(); ++index) {
    aclDataType dtype = AclnnDataType(input_specs[index].dtype);
    const bool is_masked_fill = kind == AclnnOpKind::kMaskedFillScalar || kind == AclnnOpKind::kMaskedFillTensor;
    if (is_masked_fill && index == 1) dtype = ACL_BOOL;
    if (dtype == ACL_DT_UNDEFINED) {
      result->error_code = "aclnn_tensor_dtype";
      result->reason = "aclnn " + op_name + " input dtype is unsupported";
      return Status::kInvalidArgument;
    }
    auto *tensor = kind == AclnnOpKind::kMaskedFillTensor && index == 2
                       ? CreateAclnnScalarTensor(dtype, inputs[index].device_ptr())
                       : CreateAclnnTensor(input_specs[index], dtype, inputs[index].device_ptr());
    if (!TensorValid(tensor)) {
      result->error_code = "aclnn_tensor_setup";
      result->reason = "aclCreateTensor failed for " + op_name + " input";
      return Status::kRuntimeError;
    }
    tensors->inputs.push_back(tensor);
  }
  for (size_t index = 0; index < output_specs.size(); ++index) {
    aclDataType dtype = AclnnDataType(output_specs[index].dtype);
    if (kind == AclnnOpKind::kIsInf) dtype = ACL_BOOL;
    if (dtype == ACL_DT_UNDEFINED) {
      result->error_code = "aclnn_tensor_dtype";
      result->reason = "aclnn " + op_name + " output dtype is unsupported";
      return Status::kInvalidArgument;
    }
    auto *tensor = CreateAclnnTensor(output_specs[index], dtype, outputs[index].device_ptr());
    if (!TensorValid(tensor)) {
      result->error_code = "aclnn_tensor_setup";
      result->reason = "aclCreateTensor failed for " + op_name + " output";
      return Status::kRuntimeError;
    }
    tensors->outputs.push_back(tensor);
  }
  if (kind == AclnnOpKind::kMaskedFillScalar) {
    if (MakeValueScalar(host_inputs[2], input_specs[2].dtype, &tensors->value_scalar, &result->reason) != Status::kOk) {
      result->error_code = "aclnn_scalar_setup";
      return Status::kRuntimeError;
    }
  }
  return Status::kOk;
}

using AclnnExecuteFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

Status ExecuteAclnnOp(AclRuntime *runtime, const std::string &op_name, AclnnExecuteFn execute, uint64_t workspace_size,
                      aclOpExecutor *op_executor, AclnnRunResult *result) {
  void *workspace = nullptr;
  if (workspace_size != 0 && runtime->api()->Malloc(&workspace, static_cast<size_t>(workspace_size)) != Status::kOk) {
    result->error_code = "aclnn_workspace_alloc";
    result->reason = "aclrtMalloc failed for " + op_name + " workspace";
    return Status::kRuntimeError;
  }
  aclnnStatus status = execute(workspace, workspace_size, op_executor, runtime->stream());
  const Status sync_status = status == 0 ? runtime->Synchronize() : Status::kOk;
  if (workspace != nullptr) (void)runtime->api()->Free(workspace);
  if (status != 0) {
    result->error_code = "aclnn_execution";
    result->reason = AclnnFailureReason(op_name, "execution", status);
    return Status::kRuntimeError;
  }
  if (sync_status != Status::kOk) {
    result->error_code = "aclnn_execution";
    result->reason = "aclnn " + op_name + " stream synchronization failed";
    return Status::kRuntimeError;
  }
  return Status::kOk;
}

Status RunAclnnIsInf(AclRuntime *runtime, const AclnnOpTensors &tensors, const std::string &op_name,
                     AclnnRunResult *result) {
  uint64_t workspace_size = 0;
  aclOpExecutor *op_executor = nullptr;
  aclnnStatus status = aclnnIsInfGetWorkspaceSize(tensors.inputs[0], tensors.outputs[0], &workspace_size, &op_executor);
  if (status != 0) {
    result->error_code = "aclnn_workspace_size";
    result->reason = AclnnFailureReason(op_name, "workspace size query", status);
    return Status::kRuntimeError;
  }
  return ExecuteAclnnOp(runtime, op_name, aclnnIsInf, workspace_size, op_executor, result);
}

Status RunAclnnLogicalOr(AclRuntime *runtime, const AclnnOpTensors &tensors, const std::string &op_name,
                         AclnnRunResult *result) {
  uint64_t workspace_size = 0;
  aclOpExecutor *op_executor = nullptr;
  aclnnStatus status = aclnnLogicalOrGetWorkspaceSize(tensors.inputs[0], tensors.inputs[1], tensors.outputs[0],
                                                      &workspace_size, &op_executor);
  if (status != 0) {
    result->error_code = "aclnn_workspace_size";
    result->reason = AclnnFailureReason(op_name, "workspace size query", status);
    return Status::kRuntimeError;
  }
  return ExecuteAclnnOp(runtime, op_name, aclnnLogicalOr, workspace_size, op_executor, result);
}

Status RunAclnnMaskedFillScalar(AclRuntime *runtime, const AclnnOpTensors &tensors, const std::string &op_name,
                                AclnnRunResult *result) {
  uint64_t workspace_size = 0;
  aclOpExecutor *op_executor = nullptr;
  aclnnStatus status = aclnnInplaceMaskedFillScalarGetWorkspaceSize(
      tensors.inputs[0], tensors.inputs[1], tensors.value_scalar, &workspace_size, &op_executor);
  if (status != 0) {
    result->error_code = "aclnn_workspace_size";
    result->reason = AclnnFailureReason(op_name, "workspace size query", status);
    return Status::kRuntimeError;
  }
  return ExecuteAclnnOp(runtime, op_name, aclnnInplaceMaskedFillScalar, workspace_size, op_executor, result);
}

Status RunAclnnMaskedFillTensor(AclRuntime *runtime, const AclnnOpTensors &tensors, const std::string &op_name,
                                AclnnRunResult *result) {
  uint64_t workspace_size = 0;
  aclOpExecutor *op_executor = nullptr;
  aclnnStatus status = aclnnInplaceMaskedFillTensorGetWorkspaceSize(tensors.inputs[0], tensors.inputs[1],
                                                                    tensors.inputs[2], &workspace_size, &op_executor);
  if (status != 0) {
    result->error_code = "aclnn_workspace_size";
    result->reason = AclnnFailureReason(op_name, "workspace size query", status);
    return Status::kRuntimeError;
  }
  return ExecuteAclnnOp(runtime, op_name, aclnnInplaceMaskedFillTensor, workspace_size, op_executor, result);
}

}  // namespace

AclnnExecutor::AclnnExecutor(AclRuntime *runtime) : runtime_(runtime) {}

Status AclnnExecutor::ExecuteOnce(AclnnOpKind kind, const std::vector<TensorSpec> &input_specs,
                                  std::vector<TensorBuffer> &inputs,
                                  const std::vector<std::vector<uint8_t>> &host_inputs,
                                  const std::vector<TensorSpec> &output_specs, std::vector<TensorBuffer> &outputs,
                                  AclnnRunResult *result) {
  const std::string op_name = AclnnOpName(kind);
  AclnnOpTensors tensors;
  Status status =
      SetupAclnnTensors(kind, op_name, input_specs, host_inputs, inputs, output_specs, outputs, &tensors, result);
  if (status == Status::kOk) {
    switch (kind) {
      case AclnnOpKind::kIsInf:
        status = RunAclnnIsInf(runtime_, tensors, op_name, result);
        break;
      case AclnnOpKind::kLogicalOr:
        status = RunAclnnLogicalOr(runtime_, tensors, op_name, result);
        break;
      case AclnnOpKind::kMaskedFillScalar:
        status = RunAclnnMaskedFillScalar(runtime_, tensors, op_name, result);
        break;
      case AclnnOpKind::kMaskedFillTensor:
        status = RunAclnnMaskedFillTensor(runtime_, tensors, op_name, result);
        break;
      default:
        result->error_code = "unknown_aclnn_op";
        result->reason = "unrecognized aclnn op '" + op_name + "'";
        status = Status::kRuntimeError;
        break;
    }
  }
  DestroyAclnnTensors(&tensors);
  return result->status = status;
}

Status AclnnExecutor::Run(const std::string &op_name, const std::vector<TensorSpec> &input_specs,
                          std::vector<TensorBuffer> &inputs, const std::vector<std::vector<uint8_t>> &host_inputs,
                          const std::vector<TensorSpec> &output_specs, std::vector<TensorBuffer> &outputs,
                          size_t warmup, size_t repeat, AclnnRunResult *result) {
  result->error_code.clear();
  result->reason.clear();
  result->samples.clear();
  result->output_bytes.clear();
  const AclnnOpKind kind = AclnnOpKindFromName(op_name);
  if (kind == AclnnOpKind::kUnknown) {
    result->error_code = "unknown_aclnn_op";
    result->reason = "unrecognized aclnn op '" + op_name + "'";
    return result->status = Status::kNotFound;
  }
  if (!AclnnOpArityValid(kind, input_specs.size(), output_specs.size())) {
    result->error_code = "aclnn_arity_mismatch";
    result->reason = "aclnn " + op_name + " expects " + std::to_string(input_specs.size()) + " inputs and " +
                     std::to_string(output_specs.size()) + " outputs";
    return result->status = Status::kInvalidArgument;
  }
  for (size_t iteration = 0; iteration < warmup; ++iteration) {
    if (ExecuteOnce(kind, input_specs, inputs, host_inputs, output_specs, outputs, result) != Status::kOk)
      return result->status;
  }
  for (size_t iteration = 0; iteration < repeat; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    if (ExecuteOnce(kind, input_specs, inputs, host_inputs, output_specs, outputs, result) != Status::kOk)
      return result->status;
    result->samples.push_back(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
  }
  const size_t fetch_index =
      kind == AclnnOpKind::kMaskedFillScalar || kind == AclnnOpKind::kMaskedFillTensor ? 0 : SIZE_MAX;
  for (size_t index = 0; index < output_specs.size(); ++index) {
    TensorBuffer &buffer = fetch_index == SIZE_MAX ? outputs[index] : inputs[fetch_index];
    std::vector<uint8_t> bytes(buffer.size());
    if (buffer.CopyToHost(bytes.data(), bytes.size()) != Status::kOk) {
      result->error_code = "aclnn_d2h";
      result->reason = "aclnn " + op_name + " output D2H copy failed";
      return result->status = Status::kRuntimeError;
    }
    result->output_bytes.push_back(std::move(bytes));
  }
  return result->status = Status::kOk;
}

}  // namespace device_validation
