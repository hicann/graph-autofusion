/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "support_matrix.h"

#include <stdexcept>
#include <algorithm>
#include <limits>

namespace device_validation {
namespace {
bool Has(const std::vector<std::string> &values, const std::string &value) {
  return values.empty() || std::find(values.begin(), values.end(), value) != values.end();
}
bool AbiAllowed(const std::string &allowed, const std::string &requested) {
  if (allowed.empty()) return true;
  size_t begin = 0;
  while (begin <= allowed.size()) {
    const size_t end = allowed.find(',', begin);
    if (allowed.substr(begin, end == std::string::npos ? std::string::npos : end - begin) == requested) return true;
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return false;
}
bool Available(Capability cap, const BackendCapabilities &b, const RunRequest &r) {
  if (cap == Capability::kCompile) return b.compile;
  if (cap == Capability::kFunctional) return b.functional;
  if (cap == Capability::kPrecision) return b.precision;
  return b.performance && (!r.profiler || b.profiler);
}
bool ShapeAllowed(const std::vector<std::vector<int64_t>> &allowed,
                  const std::vector<std::vector<int64_t>> &requested) {
  if (allowed.empty()) return true;
  if (allowed.size() == requested.size() && allowed == requested) return true;
  for (const auto &shape : requested) {
    if (std::find(allowed.begin(), allowed.end(), shape) == allowed.end()) return false;
  }
  return true;
}
}  // namespace

namespace {
bool ShapeExceedsLimit(const SocCapabilities &soc, const std::vector<std::vector<int64_t>> &shapes) {
  for (const auto &shape : shapes) {
    uint64_t elements = 1;
    for (const auto dim : shape) {
      if (dim <= 0 || elements > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim))
        throw std::invalid_argument("invalid shape");
      elements *= static_cast<uint64_t>(dim);
    }
    if (soc.max_shape_elements != 0 && elements > soc.max_shape_elements) return true;
  }
  return false;
}
template <typename T>
bool TensorDtypesContained(const std::vector<T> &tensors, const std::vector<std::string> &allowed);
bool RestrictionViolated(const SupportEntry &entry, const SocCapabilities &soc, const RunRequest &request) {
  if (!Has(entry.dtypes, request.dtype)) return true;
  if (!entry.input_dtypes.empty() && (request.config.inputs.size() > entry.input_dtypes.size() ||
                                      !TensorDtypesContained(request.config.inputs, entry.input_dtypes)))
    return true;
  if (!entry.output_dtypes.empty() && (request.config.outputs.size() > entry.output_dtypes.size() ||
                                       !TensorDtypesContained(request.config.outputs, entry.output_dtypes)))
    return true;
  if (!ShapeAllowed(entry.shapes, request.shapes)) return true;
  if (!soc.dtypes.empty() && !Has(soc.dtypes, request.dtype)) return true;
  if (!ShapeAllowed(soc.shapes, request.shapes)) return true;
  return false;
}
template <typename T>
bool TensorDtypesContained(const std::vector<T> &tensors, const std::vector<std::string> &allowed) {
  for (const auto &tensor : tensors) {
    if (std::find(allowed.begin(), allowed.end(), tensor.dtype) == allowed.end()) return false;
  }
  return true;
}
const SupportEntry *FindSupportEntry(const CaseConfig &config, const RunRequest &request) {
  const auto entry =
      std::find_if(config.support_matrix.begin(), config.support_matrix.end(), [&](const SupportEntry &e) {
        return e.case_id == config.case_id && e.backend == request.backend && e.soc == request.soc;
      });
  if (entry == config.support_matrix.end()) throw std::invalid_argument("undeclared case/backend/SoC combination");
  return &*entry;
}
void EnforceVariantDeclared(const std::vector<std::string> &declared, const std::string &variant, const char *message) {
  if (!declared.empty() && std::find(declared.begin(), declared.end(), variant) == declared.end())
    throw std::invalid_argument(message);
}
CapabilityDecision ResolveCapability(const Capability cap, const SupportEntry &entry, const bool restricted,
                                     const BackendCapabilities &backend, const RunRequest &request) {
  const auto status_it = entry.capabilities.find(cap);
  const auto status = status_it == entry.capabilities.end() ? SupportStatus::kUnsupported : status_it->second;
  if (status == SupportStatus::kUnsupported) {
    return {CapabilityResult::kNotApplicable, "capability is unsupported"};
  }
  if (restricted) {
    return {status == SupportStatus::kRequired ? CapabilityResult::kFailed : CapabilityResult::kSkipped,
            "dtype or shape is outside support restriction"};
  }
  if (!Available(cap, backend, request)) {
    return {status == SupportStatus::kRequired ? CapabilityResult::kFailed : CapabilityResult::kSkipped,
            "backend capability is unavailable"};
  }
  return {CapabilityResult::kPassed, "supported"};
}
void ApplyBackendOverrides(SupportDecision &result, const SupportEntry &entry, const RunRequest &request,
                           const BackendCapabilities &backend, const SocCapabilities &soc) {
  if (request.multi_output && !backend.multi_output)
    result.capabilities[Capability::kFunctional] = {CapabilityResult::kFailed, "multi-output is unsupported"};
  if (request.dynamic_shape && !backend.dynamic_shape)
    result.capabilities[Capability::kCompile] = {CapabilityResult::kFailed, "dynamic-shape is unsupported"};
  if (request.profiler && !soc.profiler_available) {
    const auto status = entry.capabilities.find(Capability::kPerformance);
    if (status != entry.capabilities.end() && status->second == SupportStatus::kRequired)
      result.capabilities[Capability::kPerformance] = {CapabilityResult::kFailed, "profiler is unavailable"};
    else
      result.capabilities[Capability::kPerformance] = {CapabilityResult::kSkipped, "optional profiler is unavailable"};
  }
}
}  // namespace
SupportDecision ResolveSupport(const CaseConfig &config, const BackendCapabilities &backend, const SocCapabilities &soc,
                               const RunRequest &request) {
  if (soc.soc != request.soc) throw std::invalid_argument("SoC mismatch");
  if (!AbiAllowed(soc.allowed_abi, request.abi)) throw std::invalid_argument("ABI mismatch");
  const auto *entry = FindSupportEntry(config, request);
  if (!AbiAllowed(entry->allowed_abi, request.abi)) throw std::invalid_argument("local ABI mismatch");
  EnforceVariantDeclared(config.variants, request.variant, "variant mismatch");
  EnforceVariantDeclared(entry->variants, request.variant, "local variant mismatch");
  SupportDecision result;
  const bool shape_exceeds_limit = ShapeExceedsLimit(soc, request.shapes);
  const bool restricted = RestrictionViolated(*entry, soc, request) || shape_exceeds_limit;
  for (const auto cap :
       {Capability::kCompile, Capability::kFunctional, Capability::kPrecision, Capability::kPerformance}) {
    result.capabilities[cap] = ResolveCapability(cap, *entry, restricted, backend, request);
  }
  ApplyBackendOverrides(result, *entry, request, backend, soc);
  return result;
}
}  // namespace device_validation
