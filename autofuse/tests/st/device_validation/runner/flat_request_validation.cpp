/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flat_request_validation.h"

#include <limits>
#include <string>
#include <vector>

namespace device_validation {
namespace {

bool IsSupportedDtype(const std::string &dtype) {
  return dtype == "float16" || dtype == "bfloat16" || dtype == "uint8" || dtype == "float32" || dtype == "int8" ||
         dtype == "int32" || dtype == "int64" || dtype == "uint32" || dtype == "uint64" || dtype == "bool";
}

size_t DtypeElementSize(const std::string &dtype) {
  if (dtype == "float16" || dtype == "bfloat16") return 2;
  if (dtype == "float32" || dtype == "int32" || dtype == "uint32") return 4;
  if (dtype == "int64" || dtype == "uint64") return 8;
  if (dtype == "int8" || dtype == "uint8" || dtype == "bool") return 1;
  return 0;
}

std::string ValidateFlatShapeSpec(const nlohmann::json &spec, const char *shape_error_code,
                                  const char *size_error_code) {
  if (!spec.contains("shape") || !spec.at("shape").is_array() || spec.at("shape").empty()) return shape_error_code;
  size_t elements = 1;
  for (const auto &dimension : spec.at("shape")) {
    int64_t value = 0;
    if (!TryGetInt64(dimension, &value)) return shape_error_code;
    if (value <= 0 || static_cast<uint64_t>(value) > std::numeric_limits<size_t>::max() / elements)
      return shape_error_code;
    elements *= static_cast<size_t>(value);
  }
  const auto element_size = DtypeElementSize(spec.at("dtype").get<std::string>());
  if (element_size == 0 || elements > std::numeric_limits<size_t>::max() / element_size) return size_error_code;
  return "";
}

std::string ValidateFlatIdentity(const nlohmann::json &json) {
  if (!json.contains("case_id") || !json.at("case_id").is_string() || !json.contains("step") ||
      !json.at("step").is_number_integer())
    return "step_contract";
  if (!json.contains("selected_shape") || !json.at("selected_shape").is_array() ||
      json.at("selected_shape") != json.value("shape", nlohmann::json::array()))
    return "selected_shape";
  if (!json.contains("variant") || !json.at("variant").is_string() || json.at("variant").get<std::string>().empty())
    return "step_contract";
  return "";
}

std::string ValidateFlatContractSchema(const nlohmann::json &json) {
  if (!json.contains("contract_schema") || !json.at("contract_schema").is_object()) return "step_contract";
  const auto &contract = json.at("contract_schema");
  if (contract.value("schema_version", 0) != 1 || contract.value("case_id", "") != json.at("case_id") ||
      contract.value("variant", "") != json.at("variant") || contract.value("step", -2) != json.at("step") ||
      !contract.contains("inputs") || !contract.at("inputs").is_array() || !contract.contains("outputs") ||
      !contract.at("outputs").is_array())
    return "step_contract";
  return "";
}

std::string ValidateFlatContractTensors(const nlohmann::json &json) {
  const auto &contract = json.at("contract_schema");
  if (contract.at("inputs") != json.at("tensor_specs")) return "input_specs";
  if (contract.at("outputs") != json.at("output_specs")) return "output_specs";
  return "";
}

bool InputArraysValid(const nlohmann::json &json) {
  if (!json.contains("inputs") || !json.contains("tensor_files") || !json.at("inputs").is_array() ||
      !json.at("tensor_files").is_array() || json.at("inputs").size() != json.at("tensor_files").size())
    return false;
  for (const auto &path : json.at("inputs")) {
    if (!path.is_string()) return false;
  }
  for (const auto &path : json.at("tensor_files")) {
    if (!path.is_string()) return false;
  }
  return true;
}

template <typename ShapeValidator>
std::string InputSpecsMatch(const nlohmann::json &json, ShapeValidator validate_shape) {
  if (!json.contains("tensor_specs") || !json.at("tensor_specs").is_array() ||
      json.at("tensor_specs").size() != json.at("inputs").size())
    return "input_specs";
  for (const auto &spec : json.at("tensor_specs")) {
    if (!spec.is_object() || !spec.contains("dtype") || !spec.at("dtype").is_string() ||
        !IsSupportedDtype(spec.at("dtype").get<std::string>()))
      return "input_specs";
    if (const auto error = validate_shape(spec, "shape", "input_specs"); !error.empty()) return error;
  }
  return "";
}

bool OutputArraysValid(const nlohmann::json &json) {
  if (!json.contains("outputs") || !json.at("outputs").is_array() || json.at("outputs").empty()) return false;
  for (const auto &path : json.at("outputs")) {
    if (!path.is_string()) return false;
  }
  return true;
}

template <typename ShapeValidator>
std::string OutputSpecsMatch(const nlohmann::json &json, ShapeValidator validate_shape) {
  if (!json.contains("output_specs") || !json.at("output_specs").is_array() ||
      json.at("output_specs").size() != json.at("outputs").size())
    return "output_specs";
  for (const auto &spec : json.at("output_specs")) {
    if (!spec.is_object() || !spec.contains("dtype") || !spec.at("dtype").is_string() ||
        !IsSupportedDtype(spec.at("dtype").get<std::string>()))
      return "output_specs";
    if (const auto error = validate_shape(spec, "output_specs", "output_specs"); !error.empty()) return error;
  }
  return "";
}

template <typename ShapeValidator>
std::string ValidateFlatTensorArrays(const nlohmann::json &json, ShapeValidator validate_shape) {
  if (!InputArraysValid(json)) return "input_count";
  if (const auto error = InputSpecsMatch(json, validate_shape); !error.empty()) return error;
  if (!OutputArraysValid(json)) return "output_count";
  if (const auto error = OutputSpecsMatch(json, validate_shape); !error.empty()) return error;
  return "";
}

bool AbiMetadataPresent(const nlohmann::json &json, const std::string &abi) {
  if (json.value("launch_abi", "") != abi || !json.contains("input_count") || !json.contains("output_count") ||
      !json.at("input_count").is_number_integer() || !json.at("output_count").is_number_integer() ||
      !json.contains("abi_metadata") || !json.at("abi_metadata").is_object())
    return false;
  return true;
}

bool AbiMetadataCountsMatch(const nlohmann::json &json, const nlohmann::json &metadata, const std::string &abi) {
  if (metadata.value("launch_abi", "") != abi || metadata.value("input_count", -1) != json.at("input_count") ||
      metadata.value("output_count", -1) != json.at("output_count") || !metadata.contains("input_dtypes") ||
      !metadata.contains("output_dtypes") || !metadata.at("input_dtypes").is_array() ||
      !metadata.at("output_dtypes").is_array() || metadata.at("input_dtypes").size() != json.at("inputs").size() ||
      metadata.at("output_dtypes").size() != json.at("outputs").size())
    return false;
  return true;
}

bool AbiAritySupported(const std::string &abi, const nlohmann::json &json) {
  if (abi == "AutofuseLaunch" &&
      (json.at("input_count").get<int32_t>() < 1 || json.at("input_count").get<int32_t>() > 3 ||
       json.at("output_count").get<int32_t>() != 1))
    return false;
  return true;
}

bool AbiCountsMatchArrays(const nlohmann::json &json) {
  if (json.at("input_count").get<int32_t>() != static_cast<int32_t>(json.at("inputs").size()) ||
      json.at("output_count").get<int32_t>() != static_cast<int32_t>(json.at("outputs").size()))
    return false;
  return true;
}

bool AbiDtypesMatchSpecs(const nlohmann::json &json, const nlohmann::json &metadata) {
  for (size_t index = 0; index < json.at("tensor_specs").size(); ++index) {
    if (!metadata.at("input_dtypes").at(index).is_string() ||
        metadata.at("input_dtypes").at(index) != json.at("tensor_specs").at(index).at("dtype"))
      return false;
  }
  for (size_t index = 0; index < json.at("output_specs").size(); ++index) {
    if (!metadata.at("output_dtypes").at(index).is_string() ||
        metadata.at("output_dtypes").at(index) != json.at("output_specs").at(index).at("dtype"))
      return false;
  }
  return true;
}

std::string ValidateFlatAbi(const nlohmann::json &json, const std::string &abi) {
  if (!AbiMetadataPresent(json, abi)) return "abi_metadata";
  const auto &metadata = json.at("abi_metadata");
  if (!AbiMetadataCountsMatch(json, metadata, abi)) return "abi_metadata";
  if (!AbiAritySupported(abi, json)) return "unsupported_abi_arity";
  if (!AbiCountsMatchArrays(json)) return "abi_metadata";
  if (!AbiDtypesMatchSpecs(json, metadata)) return "abi_metadata";
  return "";
}

}  // namespace

bool TryGetInt64(const nlohmann::json &value, int64_t *result) {
  if (!value.is_number_integer()) return false;
  if (value.is_number_unsigned() && value.get<uint64_t>() > std::numeric_limits<int64_t>::max()) return false;
  *result = value.get<int64_t>();
  return true;
}

FlatValidationResult ValidateFlatRequest(const nlohmann::json &json) {
  if (const auto error = ValidateFlatTensorArrays(json, ValidateFlatShapeSpec); !error.empty()) return {error};
  if (const auto error = ValidateFlatIdentity(json); !error.empty()) return {error};
  if (const auto error = ValidateFlatContractSchema(json); !error.empty()) return {error};
  if (const auto error = ValidateFlatContractTensors(json); !error.empty()) return {error};
  if (json.contains("aclnn_op") && !json.at("aclnn_op").is_string()) return {"step_contract"};
  const std::string aclnn_op = json.value("aclnn_op", "");
  if (aclnn_op.empty()) {
    const auto abi = json.value("abi", "");
    if (abi != "AutofuseLaunch" && abi != "AutofuseLaunchV2") return {"abi_metadata"};
    if (const auto error = ValidateFlatAbi(json, abi); !error.empty()) return {error};
  }
  if (json.contains("profile_dir") && !json.at("profile_dir").is_string()) return {"profile_dir"};
  return {};
}

}  // namespace device_validation
