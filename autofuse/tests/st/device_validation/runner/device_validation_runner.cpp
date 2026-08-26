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
#include "ascendc_backend.h"
#include "flat_output.h"
#include "flat_request_validation.h"
#include "inprocess_profiler.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace device_validation;

namespace {
std::string PreflightStage(const RunReport &report) {
  bool has_skipped = false;
  bool has_passed = false;
  for (const auto &[capability, decision] : report.support.capabilities) {
    if (decision.result == CapabilityResult::kFailed) return "failed";
    if (decision.result == CapabilityResult::kSkipped) has_skipped = true;
    if (decision.result == CapabilityResult::kPassed) has_passed = true;
  }
  if (has_skipped) return "skipped";
  if (!has_passed) return "not_applicable";
  return "failed";
}
}  // namespace

namespace {
struct FlatRequest {
  std::string case_id;
  int32_t step = -1;
  std::string case_dir;
  std::string profile;
  std::string soc_profile;
  std::string module;
  std::string abi;
  std::string launch_abi;
  nlohmann::json abi_metadata;
  std::string aclnn_op;
  int32_t input_count = -1;
  int32_t output_count = -1;
  std::vector<std::string> tensor_files;
  std::vector<std::string> inputs;
  std::vector<std::string> output_files;
  nlohmann::json input_specs;
  nlohmann::json output_specs;
  bool has_input_specs = false;
  bool has_output_specs = false;
  bool has_inputs = false;
  bool has_outputs = false;
  std::string artifact_dir;
  int device = 0;
  size_t warmup = 0;
  size_t repeat = 1;
  std::vector<int64_t> shape;
  std::vector<int64_t> selected_shape;
  bool profiler = false;
  std::string variant;
  std::string profile_dir;
  nlohmann::json contract_schema;
};

template <typename T>
std::vector<T> ReadStringArray(const nlohmann::json &json, const char *name) {
  if (!json.contains(name) || !json.at(name).is_array()) {
    throw std::runtime_error(std::string(name) + " must be an array");
  }
  for (const auto &item : json.at(name)) {
    if (!item.is_string()) throw std::runtime_error(std::string(name) + " must be an array of strings");
  }
  return json.at(name).get<std::vector<T>>();
}

nlohmann::json ReadSpecArray(const nlohmann::json &json, const char *name) {
  if (!json.contains(name) || !json.at(name).is_array()) {
    throw std::runtime_error(std::string(name) + " must be an array");
  }
  for (const auto &spec : json.at(name)) {
    if (!spec.is_object() || !spec.contains("dtype") || !spec.contains("shape") || !spec.at("dtype").is_string() ||
        !spec.at("shape").is_array())
      throw std::runtime_error(std::string(name) + " must contain tensor specs");
  }
  return json.at(name);
}

nlohmann::json ReadCaseJson(const std::string &case_dir) {
  std::ifstream stream(std::filesystem::path(case_dir) / "case.json");
  if (!stream) throw std::runtime_error("case contract is unavailable");
  nlohmann::json json;
  stream >> json;
  return json;
}

std::string ValidateCaseShape(const nlohmann::json &case_json, const FlatRequest &request,
                              const nlohmann::json &case_inputs, const nlohmann::json &case_outputs,
                              const nlohmann::json &support_matrix) {
  bool supported_shape = false;
  for (const auto &entry : support_matrix) {
    const auto shapes = entry.value("shapes", nlohmann::json::array());
    if (entry.value("backend", "") == "ascendc_real_device" && entry.value("soc", "") == request.soc_profile &&
        std::find(shapes.begin(), shapes.end(), request.selected_shape) != shapes.end()) {
      supported_shape = true;
      break;
    }
  }
  if (request.module != "host_fake_module" && !supported_shape) return "unsupported_shape";
  for (size_t i = 0; i < case_inputs.size() && i < request.input_specs.size(); ++i) {
    if (case_inputs.at(i).value("dynamic", false) && request.input_specs.at(i).at("shape") != request.selected_shape)
      return "selected_shape";
  }
  for (size_t i = 0; i < case_outputs.size() && i < request.output_specs.size(); ++i) {
    if (case_outputs.at(i).value("dynamic", false) && request.output_specs.at(i).at("shape") != request.selected_shape)
      return "selected_shape";
  }
  return "";
}

std::string ValidateDeclaredSpecs(const nlohmann::json &declared, const nlohmann::json &actual,
                                  const nlohmann::json &case_specs, const char *error_code) {
  if (!declared.is_array() || !case_specs.is_array() || declared.size() != actual.size() ||
      case_specs.size() != actual.size())
    return error_code;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!declared.at(i).is_object() || !actual.at(i).is_object() ||
        declared.at(i).value("dtype", "") != actual.at(i).value("dtype", ""))
      return error_code;
    if (!case_specs.at(i).value("dynamic", false) &&
        declared.at(i).value("shape", nlohmann::json::array()) != actual.at(i).value("shape", nlohmann::json::array()))
      return error_code;
  }
  return "";
}

std::string ValidateUnfusedStepSpecs(const nlohmann::json &step, const FlatRequest &request) {
  if (!step.is_object() || !step.contains("inputs") || !step.contains("outputs") || !step.at("inputs").is_array() ||
      !step.at("outputs").is_array() || step.at("outputs").size() != request.output_specs.size() ||
      step.at("inputs").size() != request.input_specs.size())
    return "step_specs";
  return "";
}

std::string ValidatePreviousStepInput(const nlohmann::json &variant, const nlohmann::json &case_outputs,
                                      const FlatRequest &request, size_t index) {
  if (request.step == 0) return "previous_input";
  const auto &previous_step = variant.at("steps").at(request.step - 1);
  if (!previous_step.is_object() || !previous_step.contains("outputs") || !previous_step.at("outputs").is_array() ||
      previous_step.at("outputs").empty())
    return "previous_input";
  const auto &previous = previous_step.at("outputs").back();
  if (!previous.is_object() || previous.value("dtype", "") != request.input_specs.at(index).at("dtype") ||
      (case_outputs.size() > index && !case_outputs.at(index).value("dynamic", false) &&
       previous.value("shape", nlohmann::json::array()) != request.input_specs.at(index).at("shape")))
    return "previous_input";
  return "";
}

std::string ValidateStepInputAt(const nlohmann::json &variant, const nlohmann::json &step,
                                const nlohmann::json &case_inputs, const nlohmann::json &case_outputs,
                                const FlatRequest &request, size_t index) {
  const auto &declared = step.at("inputs").at(index);
  if (declared.is_string() && declared == "$previous")
    return ValidatePreviousStepInput(variant, case_outputs, request, index);
  if (!declared.is_object() || declared.value("dtype", "") != request.input_specs.at(index).at("dtype"))
    return "input_specs";
  if (!case_inputs.at(index).value("dynamic", false) &&
      declared.value("shape", nlohmann::json::array()) != request.input_specs.at(index).at("shape"))
    return "input_specs";
  return "";
}

std::string ValidateUnfusedVariant(const nlohmann::json &variant, const nlohmann::json &case_inputs,
                                   const nlohmann::json &case_outputs, const FlatRequest &request) {
  if (request.step < 0 || !variant.is_object() || !variant.contains("steps") || !variant.at("steps").is_array() ||
      static_cast<size_t>(request.step) >= variant.at("steps").size())
    return "invalid_step";
  const auto &step = variant.at("steps").at(request.step);
  if (const auto error = ValidateUnfusedStepSpecs(step, request); !error.empty()) return error;
  for (size_t i = 0; i < request.input_specs.size(); ++i) {
    if (const auto error = ValidateStepInputAt(variant, step, case_inputs, case_outputs, request, i); !error.empty())
      return error;
  }
  return ValidateDeclaredSpecs(step.at("outputs"), request.output_specs, case_outputs, "output_specs");
}

std::string ValidateCaseVariantStep(const nlohmann::json &case_json, const FlatRequest &request) {
  if (request.variant != "fused" && request.variant != "unfused") return "invalid_variant";
  if (!case_json.contains("variants") || !case_json.at("variants").is_object() ||
      !case_json.at("variants").contains(request.variant))
    return "invalid_variant";
  const auto &variant = case_json.at("variants").at(request.variant);
  const auto case_inputs = case_json.value("inputs", nlohmann::json::array());
  const auto case_outputs = case_json.value("outputs", nlohmann::json::array());
  const auto support_matrix = case_json.value("support_matrix", nlohmann::json::array());
  if (const auto error = ValidateCaseShape(case_json, request, case_inputs, case_outputs, support_matrix);
      !error.empty())
    return error;
  if (request.variant == "unfused") {
    return ValidateUnfusedVariant(variant, case_inputs, case_outputs, request);
  } else if (request.step != -1) {
    return "invalid_step";
  }
  if (request.variant != "unfused") {
    if (case_inputs.size() != request.input_specs.size() || case_outputs.size() != request.output_specs.size())
      return "case_specs";
    if (const auto error = ValidateDeclaredSpecs(case_inputs, request.input_specs, case_inputs, "input_specs");
        !error.empty())
      return error;
    if (const auto error = ValidateDeclaredSpecs(case_outputs, request.output_specs, case_outputs, "output_specs");
        !error.empty())
      return error;
  }
  return "";
}

void ParseRequestIdentity(const nlohmann::json &json, FlatRequest *request) {
  request->aclnn_op = json.value("aclnn_op", "");
  if (request->aclnn_op.empty()) {
    request->module = json.at("module").get<std::string>();
    request->abi = json.at("abi").get<std::string>();
  } else {
    request->module = json.value("module", "");
    request->abi = json.value("abi", "");
  }
  request->launch_abi =
      request->aclnn_op.empty() ? json.value("launch_abi", request->abi) : json.value("launch_abi", "");
  request->abi_metadata =
      request->aclnn_op.empty() ? json.at("abi_metadata") : json.value("abi_metadata", nlohmann::json::object());
}

void ParseRequestShape(const nlohmann::json &json, FlatRequest *request) {
  if (!json.contains("shape")) return;
  if (!json.at("shape").is_array()) throw std::runtime_error("shape");
  for (const auto &dimension : json.at("shape")) {
    int64_t value = 0;
    if (!TryGetInt64(dimension, &value)) throw std::runtime_error("shape");
  }
  try {
    request->shape = json.at("shape").get<std::vector<int64_t>>();
    request->selected_shape = request->shape;
  } catch (const nlohmann::json::exception &) {
    throw std::runtime_error("shape");
  }
}

FlatRequest LoadFlatRequest(const std::string &path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("request file is unavailable");
  nlohmann::json json;
  stream >> json;
  const auto validation = ValidateFlatRequest(json);
  if (!validation.ok()) throw std::runtime_error(validation.error_code);
  FlatRequest request;
  request.case_dir = json.at("case_dir").get<std::string>();
  request.case_id = json.at("case_id").get<std::string>();
  request.step = json.at("step").get<int32_t>();
  request.profile = json.at("profile").get<std::string>();
  request.soc_profile = json.value("soc_profile", "");
  ParseRequestIdentity(json, &request);
  request.input_count = json.value("input_count", -1);
  request.output_count = json.value("output_count", -1);
  request.tensor_files = ReadStringArray<std::string>(json, "tensor_files");
  request.inputs = ReadStringArray<std::string>(json, "inputs");
  request.output_files = ReadStringArray<std::string>(json, "outputs");
  request.input_specs = ReadSpecArray(json, "tensor_specs");
  request.output_specs = ReadSpecArray(json, "output_specs");
  request.has_inputs = request.has_outputs = request.has_input_specs = request.has_output_specs = true;
  request.artifact_dir = json.at("artifact_dir").get<std::string>();
  request.device = json.at("device").get<int>();
  request.warmup = json.at("warmup").get<size_t>();
  request.repeat = json.at("repeat").get<size_t>();
  request.variant = json.value("variant", "fused");
  request.contract_schema = json.at("contract_schema");
  request.profiler = json.value("profiler", false);
  request.profile_dir = json.value("profile_dir", "");
  ParseRequestShape(json, &request);
  return request;
}

size_t DtypeElementSize(const std::string &dtype) {
  if (dtype == "float16" || dtype == "bfloat16") return 2;
  if (dtype == "float32" || dtype == "int32" || dtype == "uint32") return 4;
  if (dtype == "int64" || dtype == "uint64") return 8;
  if (dtype == "int8" || dtype == "uint8" || dtype == "bool") return 1;
  return 0;
}

std::vector<uint8_t> FakeOutputBytes(const nlohmann::json &spec) {
  const auto dtype = spec.at("dtype").get<std::string>();
  const auto element_size = DtypeElementSize(dtype);
  if (element_size == 0) throw std::runtime_error("output_specs: unsupported dtype");
  if (spec.at("shape").empty()) throw std::runtime_error("output_specs: invalid shape");
  size_t elements = 1;
  for (const auto &dimension : spec.at("shape")) {
    int64_t value = 0;
    if (!TryGetInt64(dimension, &value)) throw std::runtime_error("output_specs: invalid shape");
    if (value <= 0 || static_cast<uint64_t>(value) > std::numeric_limits<size_t>::max() / elements)
      throw std::runtime_error("output_specs: invalid shape");
    elements *= static_cast<size_t>(value);
  }
  if (elements > std::numeric_limits<size_t>::max() / element_size)
    throw std::runtime_error("output_specs: invalid size");
  return std::vector<uint8_t>(elements * element_size, 0);
}

nlohmann::json FakeOutputMetadata(const std::vector<uint8_t> &bytes, const nlohmann::json &spec) {
  nlohmann::json data = nlohmann::json::array();
  for (size_t i = 0; i < bytes.size() / DtypeElementSize(spec.at("dtype").get<std::string>()); ++i) data.push_back(0);
  return nlohmann::json{{"dtype", spec.at("dtype")}, {"shape", spec.at("shape")}, {"data", data}};
}

nlohmann::json HostFakeFailureReport(const FlatRequest &request, const std::string &reason) {
  return {{"schema_version", 2},       {"case", request.case_id},      {"case_id", request.case_id},
          {"backend", "host_fake"},    {"variant", request.variant},   {"step", request.step},
          {"actual_abi", request.abi}, {"stage_status", "failed"},     {"stage", "execution"},
          {"reason", reason},          {"error_code", "output_specs"}, {"outputs", nlohmann::json::array()}};
}

bool BuildFakeOutputs(const FlatRequest &request, std::vector<std::vector<uint8_t>> *outputs,
                      nlohmann::json *output_metadata, nlohmann::json *report) {
  for (const auto &spec : request.output_specs) {
    try {
      outputs->push_back(FakeOutputBytes(spec));
    } catch (const std::exception &error) {
      *report = HostFakeFailureReport(request, error.what());
      return false;
    }
    output_metadata->push_back(FakeOutputMetadata(outputs->back(), spec));
  }
  return true;
}

nlohmann::json BuildFakeInputMarker(const FlatRequest &request) {
  nlohmann::json input_marker = nlohmann::json::array();
  for (const auto &input_path : request.inputs) {
    std::ifstream input(input_path, std::ios::binary);
    if (!input) throw std::runtime_error("fake input is unavailable");
    input_marker.push_back(input_path);
  }
  return input_marker;
}

nlohmann::json HostFakeSuccessReport(const FlatRequest &request, const nlohmann::json &output_metadata,
                                     const nlohmann::json &input_marker) {
  nlohmann::json report = {{"schema_version", 2},
                           {"case", request.case_id},
                           {"case_id", request.case_id},
                           {"backend", "host_fake"},
                           {"variant", request.variant},
                           {"step", request.step},
                           {"requested_abi", request.abi},
                           {"actual_abi", request.abi},
                           {"stage_status", "passed"},
                           {"stage", "execution"},
                           {"reason", ""},
                           {"error_code", ""},
                           {"outputs", output_metadata}};
  report["input_marker"] = input_marker;
  report["output_marker"] = "step=" + std::to_string(request.step);
  return report;
}

bool RunHostFakeFlat(const FlatRequest &request, nlohmann::json *report) {
  if (request.module != "host_fake_module") return false;
  std::vector<std::vector<uint8_t>> outputs;
  nlohmann::json output_metadata = nlohmann::json::array();
  if (!BuildFakeOutputs(request, &outputs, &output_metadata, report)) return false;
  const auto input_marker = BuildFakeInputMarker(request);
  std::string output_error;
  if (!WriteFlatOutputs(request.output_files, outputs, request.artifact_dir, &output_error))
    throw std::runtime_error(output_error);
  *report = HostFakeSuccessReport(request, output_metadata, input_marker);
  return true;
}

std::string FlatParseErrorCode(const nlohmann::json &json) {
  const auto validation = ValidateFlatRequest(json);
  if (!validation.ok()) return validation.error_code;
  return "runner_failure";
}

void EmitFailure(const std::string &case_id, const std::string &backend, const std::string &reason,
                 const std::string &stage, const std::string &variant = "",
                 const std::string &error_code = "runner_failure", int32_t step = -1,
                 const std::string &actual_abi = "") {
  nlohmann::json report = {
      {"schema_version", 2},
      {"case", case_id},
      {"case_id", case_id},
      {"backend", backend},
      {"variant", variant},
      {"step", step},
      {"requested_abi", actual_abi},
      {"actual_abi", actual_abi},
      {"soc_profile", ""},
      {"soc", ""},
      {"stage_status", stage},
      {"stage", stage == "failed" ? "execution" : "preflight"},
      {"error_code", error_code},
      {"reason", reason},
      {"support_decisions", nlohmann::json::object()},
      {"profile", ""},
      {"run_parameters", nlohmann::json::object()},
      {"artifact_paths", nlohmann::json::array()},
      {"performance", {{"declared", nlohmann::json::object()}, {"actual", nlohmann::json::object()}}},
      {"precision", {"passed", false}},
      {"outputs", nlohmann::json::array()}};
  std::cout << report.dump() << '\n';
}

bool TryLoadFlatRequest(const std::string &request_path, FlatRequest *request) {
  try {
    *request = LoadFlatRequest(request_path);
    return true;
  } catch (const std::exception &error) {
    std::string variant;
    std::string error_code = "runner_failure";
    try {
      std::ifstream stream(request_path);
      nlohmann::json json;
      stream >> json;
      variant = json.value("variant", "fused");
      error_code = FlatParseErrorCode(json);
    } catch (const std::exception &) {
      // Keep the original parse error as the externally visible failure.
    }
    std::string abi;
    int32_t step = -1;
    try {
      std::ifstream stream(request_path);
      nlohmann::json json;
      stream >> json;
      abi = json.value("abi", "");
      step = json.value("step", -1);
    } catch (const std::exception &) {
    }
    EmitFailure("unknown", "ascendc_real_device", error.what(), "failed", variant, error_code, step, abi);
    return false;
  }
}

std::optional<int> RunHostFakeExecution(const FlatRequest &request) {
  if (request.module != "host_fake_module") return std::nullopt;
  try {
    const auto case_json = ReadCaseJson(request.case_dir);
    if (request.case_id != case_json.value("case_id", "")) throw std::runtime_error("case_id");
    const auto contract_error = ValidateCaseVariantStep(case_json, request);
    if (!contract_error.empty()) throw std::runtime_error(contract_error);
    nlohmann::json report;
    if (!RunHostFakeFlat(request, &report)) {
      std::cout << report.dump() << '\n';
      return 4;
    }
    std::cout << report.dump() << '\n';
    return 0;
  } catch (const std::exception &error) {
    const std::string reason = error.what();
    const std::string error_code =
        reason == "case_id" || reason == "invalid_variant" || reason == "invalid_step" ? reason : "fake_execution";
    EmitFailure(request.case_id, "host_fake", reason, "failed", request.variant, error_code, request.step, request.abi);
    return error_code == reason ? 2 : 4;
  }
}

std::optional<int> ValidateFlatRequestForBackend(const FlatRequest &request, const CaseConfig &config) {
  if (request.variant.empty()) {
    EmitFailure(config.case_id, "ascendc_real_device", "variant must not be empty", "failed", request.variant);
    return 2;
  }
  if (request.step < -1) {
    EmitFailure(config.case_id, "ascendc_real_device", "step must be -1 or non-negative", "failed", request.variant,
                "invalid_step");
    return 2;
  }
  if (request.case_id != config.case_id) {
    EmitFailure(config.case_id, "ascendc_real_device", "flat case_id does not match case", "failed", request.variant,
                "case_id");
    return 2;
  }
  const auto contract_error = ValidateCaseVariantStep(ReadCaseJson(request.case_dir), request);
  if (!contract_error.empty()) {
    EmitFailure(config.case_id, "ascendc_real_device", "flat variant or step is not declared by case", "failed",
                request.variant, contract_error);
    return 2;
  }
  if (request.tensor_files != request.inputs || request.tensor_files.size() != request.input_specs.size() ||
      request.inputs.size() != request.input_specs.size() ||
      request.input_count != static_cast<int32_t>(request.inputs.size()) ||
      request.output_count != static_cast<int32_t>(request.output_files.size()) || request.launch_abi != request.abi) {
    EmitFailure(config.case_id, "ascendc_real_device", "flat input count does not match case", "failed",
                request.variant, "input_count");
    return 2;
  }
  if (request.output_files.size() != request.output_specs.size() || request.output_files.empty()) {
    EmitFailure(config.case_id, "ascendc_real_device", "flat output/spec count does not match case", "failed",
                request.variant, "output_count");
    return 2;
  }
  return std::nullopt;
}

std::optional<int> ApplyFlatRequestToRunRequest(const FlatRequest &request, const CaseConfig &config,
                                                RunRequest *run_request) {
  run_request->config.inputs.clear();
  run_request->config.outputs.clear();
  for (const auto &spec : request.input_specs) {
    run_request->config.inputs.push_back(
        {spec.at("shape").get<std::vector<int64_t>>(), spec.at("dtype").get<std::string>(), false});
  }
  for (const auto &spec : request.output_specs) {
    run_request->config.outputs.push_back(
        {spec.at("shape").get<std::vector<int64_t>>(), spec.at("dtype").get<std::string>(), false});
  }
  run_request->capability_config = config;
  run_request->device_id = request.device;
  run_request->warmup_count = request.warmup;
  run_request->kernel_count = request.repeat;
  run_request->abi = request.abi;
  if (request.aclnn_op.empty() && (request.abi_metadata.value("launch_abi", "") != run_request->abi ||
                                   request.abi_metadata.value("input_count", -1) != request.input_count ||
                                   request.abi_metadata.value("output_count", -1) != request.output_count)) {
    EmitFailure(config.case_id, "ascendc_real_device", "flat ABI metadata mismatch", "failed", request.variant,
                "abi_metadata");
    return 2;
  }
  run_request->variant = request.variant;
  run_request->shapes.clear();
  for (const auto &input : run_request->config.inputs) run_request->shapes.push_back(input.shape);
  return std::nullopt;
}

std::optional<int> ApplyRunOptions(int argc, char **argv, int input_end, bool is_flat, const CaseConfig &config,
                                   const BackendInfo &backend_info, RunRequest *request) {
  for (int i = input_end; !is_flat && i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--device" && i + 1 < argc) {
      ++i;
      request->device_id = std::stoi(argv[i]);
      if (request->device_id < 0) {
        EmitFailure(config.case_id, backend_info.name, "device_id must be non-negative", "preflight");
        return 2;
      }
    } else if (option == "--warmup" && i + 1 < argc) {
      ++i;
      request->warmup_count = static_cast<size_t>(std::stoul(argv[i]));
    } else if (option == "--repeat" && i + 1 < argc) {
      ++i;
      request->kernel_count = static_cast<size_t>(std::stoul(argv[i]));
    } else if (option == "--shape" && i + 2 < argc) {
      ++i;
      const int64_t rows = std::stoll(argv[i]);
      ++i;
      const int64_t cols = std::stoll(argv[i]);
      const std::vector<int64_t> shape = {rows, cols};
      for (auto &input : request->config.inputs) {
        if (input.dynamic) input.shape = shape;
      }
      for (auto &output : request->config.outputs) {
        if (output.dynamic) output.shape = shape;
      }
      request->shapes.clear();
      for (const auto &input : request->config.inputs) request->shapes.push_back(input.shape);
    } else if (option == "--profiler") {
      request->profiler = true;
    } else {
      EmitFailure(config.case_id, backend_info.name, "invalid runner option", "failed");
      return 2;
    }
  }
  return std::nullopt;
}

int RunPreflight(ExecutionBackend *backend, RunRequest *request) {
  RunResult preflight;
  backend->Run(*request, &preflight);
  auto report = SerializeReport(preflight.report);
  report["stage_status"] = PreflightStage(preflight.report);
  report["stage"] = "preflight";
  report["reason"] = preflight.reason.empty() ? "capability validation failed" : preflight.reason;
  report["error_code"] = preflight.error_code.empty() ? "capability_unavailable" : preflight.error_code;
  report["stage"] = preflight.stage;
  report["profile"] = request->soc;
  report["outputs"] = nlohmann::json::array();
  std::cout << report.dump() << '\n';
  return report["stage_status"] == "failed" ? 3 : 0;
}

bool LoadHostInputs(RunRequest *request, bool is_flat, const FlatRequest *flat, char **argv) {
  for (size_t i = 0; i < request->config.inputs.size(); ++i) {
    const std::string &input_path = is_flat ? flat->inputs[i] : argv[4 + i];
    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
      EmitFailure(request->config.case_id, request->backend, "input file is unavailable", "failed");
      return false;
    }
    request->host_inputs.emplace_back(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  }
  return true;
}

int ConfigureInProcessProfiler(ExecutionBackend *backend, const CaseConfig &config, const FlatRequest *flat,
                               const std::string &collect_dir, std::unique_ptr<InProcessProfiler> *profiler) {
  if (collect_dir.empty()) return 0;
  auto *ascendc_backend = dynamic_cast<AscendCBackend *>(backend);
  if (ascendc_backend == nullptr) {
    EmitFailure(config.case_id, backend->GetCapabilities().name, "backend does not support in-process profiling",
                "failed", flat != nullptr ? flat->variant : "", "profiler_unsupported",
                flat != nullptr ? flat->step : -1, flat != nullptr ? flat->abi : "");
    return 4;
  }
  *profiler = std::make_unique<InProcessProfiler>(CreateInProcessProfiler());
  ascendc_backend->set_profiler(profiler->get(), collect_dir);
  return 0;
}

void AppendOutputsToReport(nlohmann::json *report, const RunResult &result) {
  for (const auto &output : result.outputs) {
    nlohmann::json data = output.values;
    if (!output.integer_values.empty()) data = output.integer_values;
    if (!output.unsigned_values.empty()) data = output.unsigned_values;
    if (!output.raw_bits.empty()) data = output.raw_bits;
    (*report)["outputs"].push_back({{"dtype", output.dtype}, {"shape", output.shape}, {"data", data}});
  }
}

int WriteFlatExecutionOutputs(const FlatRequest *flat, const RunResult &result, nlohmann::json *report) {
  std::string output_error;
  if (!WriteFlatOutputs(flat->output_files, result.output_bytes, flat->artifact_dir, &output_error)) {
    (*report)["stage_status"] = "failed";
    (*report)["stage"] = "execution";
    (*report)["error_code"] = output_error == "output count does not match result"          ? "output_count"
                              : output_error == "output path is outside artifact directory" ? "output_path"
                                                                                            : "output_write";
    (*report)["reason"] = output_error;
    std::cout << report->dump() << '\n';
    return 4;
  }
  return 0;
}

int DispatchRealExecution(ExecutionBackend *backend, RunRequest *request, const CaseConfig &config,
                          const FlatRequest *flat, bool is_flat, int argc, char **argv, int input_end,
                          const std::string &collect_dir) {
  request->config.performance.warmup_count = request->warmup_count;
  request->config.performance.kernel_count = request->kernel_count;
  request->config.performance.profiler = request->profiler;
  if (backend->Validate(request->config) != Status::kOk) return RunPreflight(backend, request);
  if (!LoadHostInputs(request, is_flat, flat, argv)) return 4;
  std::unique_ptr<InProcessProfiler> profiler;
  if (const auto profiler_error = ConfigureInProcessProfiler(backend, config, flat, collect_dir, &profiler))
    return profiler_error;
  RunResult result;
  Status status = Status::kRuntimeError;
  status = backend->Run(*request, &result);
  auto report = SerializeReport(result.report);
  report["stage"] = result.stage;
  report["reason"] = result.reason;
  report["error_code"] = result.error_code;
  report["requested_abi"] = request->abi;
  report["actual_abi"] = result.report.abi.empty() ? request->abi : result.report.abi;
  report["variant"] = request->variant;
  report["step"] = is_flat ? flat->step : -1;
  report["profile"] = request->soc;
  report["outputs"] = nlohmann::json::array();
  if (status != Status::kOk) {
    std::cout << report.dump() << '\n';
    return 4;
  }
  AppendOutputsToReport(&report, result);
  if (is_flat) {
    if (const auto write_error = WriteFlatExecutionOutputs(flat, result, &report)) return write_error;
  }
  if (is_flat) report["abi_metadata"] = flat->abi_metadata;
  std::cout << report.dump() << '\n';
  return 0;
}

struct AclnnRunSpecs {
  std::vector<TensorSpec> input_specs;
  std::vector<TensorSpec> output_specs;
  std::vector<std::vector<uint8_t>> host_inputs;
  std::vector<TensorBuffer> inputs;
  std::vector<TensorBuffer> outputs;
  std::unique_ptr<InProcessProfiler> profiler;
};

void ParseAclnnSpecs(const FlatRequest &flat, AclnnRunSpecs *specs) {
  for (const auto &spec : flat.input_specs) {
    specs->input_specs.push_back({spec.at("shape").get<std::vector<int64_t>>(), spec.at("dtype").get<std::string>()});
  }
  for (const auto &spec : flat.output_specs) {
    specs->output_specs.push_back({spec.at("shape").get<std::vector<int64_t>>(), spec.at("dtype").get<std::string>()});
  }
}

int LoadAclnnHostInputs(const FlatRequest &flat, const CaseConfig &config, AclnnRunSpecs *specs) {
  for (const auto &path : flat.inputs) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      EmitFailure(config.case_id, "ascendc_real_device", "input file is unavailable", "failed", flat.variant,
                  "input_read", flat.step);
      return 4;
    }
    specs->host_inputs.emplace_back(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  }
  return 0;
}

int AllocateAclnnBuffers(const FlatRequest &flat, const CaseConfig &config, AclRuntime *runtime, AclnnRunSpecs *specs) {
  specs->inputs.resize(specs->input_specs.size());
  specs->outputs.resize(specs->output_specs.size());
  for (size_t index = 0; index < specs->input_specs.size(); ++index) {
    if (specs->inputs[index].Allocate(specs->input_specs[index], runtime) != Status::kOk ||
        specs->inputs[index].CopyToDevice(specs->host_inputs[index].data(), specs->host_inputs[index].size()) !=
            Status::kOk) {
      EmitFailure(config.case_id, "ascendc_real_device", "input allocation or H2D copy failed", "failed", flat.variant,
                  "input_allocation_or_h2d", flat.step);
      return 4;
    }
  }
  for (size_t index = 0; index < specs->output_specs.size(); ++index) {
    if (specs->outputs[index].Allocate(specs->output_specs[index], runtime) != Status::kOk) {
      EmitFailure(config.case_id, "ascendc_real_device", "output allocation failed", "failed", flat.variant,
                  "output_allocation", flat.step);
      return 4;
    }
  }
  return 0;
}

int StartAclnnProfiler(const FlatRequest &flat, const CaseConfig &config, const std::string &collect_dir,
                       AclnnRunSpecs *specs) {
  if (!flat.profiler || collect_dir.empty()) return 0;
  specs->profiler = std::make_unique<InProcessProfiler>(CreateInProcessProfiler());
  if (specs->profiler->Start(flat.device, collect_dir) != Status::kOk) {
    EmitFailure(config.case_id, "ascendc_real_device",
                "in-process profiler start failed: " + specs->profiler->last_error(), "failed", flat.variant,
                "profiler_start_failed", flat.step);
    return 4;
  }
  return 0;
}

int DecodeAclnnOutputs(const FlatRequest &flat, const CaseConfig &config, const AclnnRunSpecs &specs,
                       AclnnRunResult &run_result, RunResult *result) {
  for (size_t index = 0; index < specs.output_specs.size(); ++index) {
    try {
      result->outputs.push_back(DecodeOutputForTest(specs.output_specs[index].dtype, specs.output_specs[index].shape,
                                                    run_result.output_bytes[index]));
      result->output_bytes.push_back(std::move(run_result.output_bytes[index]));
    } catch (const std::invalid_argument &error) {
      EmitFailure(config.case_id, "ascendc_real_device", error.what(), "failed", flat.variant, "output_decode",
                  flat.step);
      return 4;
    }
  }
  return 0;
}

int SummarizeAclnnTiming(const FlatRequest &flat, const CaseConfig &config, const BackendInfo &backend_info,
                         bool profiler_collected, const std::string &profiler_collect_dir,
                         const AclnnRunResult &run_result, RunReport *run_report, nlohmann::json *report) {
  run_report->case_id = config.case_id;
  run_report->backend = "ascendc_real_device";
  run_report->abi = "";
  run_report->requested_abi = "";
  run_report->variant = flat.variant;
  run_report->soc_profile = backend_info.profile;
  run_report->stage_status = "passed";
  run_report->stage = "execution";
  run_report->profile = backend_info.profile;
  run_report->run_parameters = {{"device_id", flat.device}, {"warmup", flat.warmup}, {"repeat", flat.repeat}};
  auto performance_config = config.performance;
  performance_config.warmup_count = flat.warmup;
  performance_config.kernel_count = flat.repeat;
  performance_config.profiler = flat.profiler;
  try {
    run_report->performance =
        SummarizeSamplesWithRuntimeState(run_result.samples, performance_config, flat.profiler, profiler_collected,
                                         !profiler_collect_dir.empty(), "runner_wall_clock");
  } catch (const std::invalid_argument &error) {
    EmitFailure(config.case_id, "ascendc_real_device", error.what(), "failed", flat.variant, "timing_summary",
                flat.step);
    return 4;
  }
  run_report->performance.actual.warmup_count = flat.warmup;
  run_report->performance.actual.samples = run_result.samples;
  run_report->performance.actual.kernel_count = flat.repeat;
  run_report->performance.actual.metric = "runner_wall_clock";
  run_report->performance.actual.unit = "ms";
  run_report->performance.actual.timing_source = "runner_wall_clock";
  run_report->performance.actual.profiler_collect_dir = profiler_collect_dir;
  run_report->metric = "runner_wall_clock";
  *report = SerializeReport(*run_report);
  (*report)["reason"] = "";
  (*report)["error_code"] = "";
  (*report)["requested_abi"] = "";
  (*report)["actual_abi"] = "";
  (*report)["variant"] = flat.variant;
  (*report)["step"] = flat.step;
  (*report)["profile"] = backend_info.profile;
  (*report)["outputs"] = nlohmann::json::array();
  return 0;
}

int DispatchAclnnExecution(const FlatRequest &flat, const CaseConfig &config, const BackendInfo &backend_info,
                           const std::string &collect_dir) {
  AclRuntime runtime(CreateAclRuntimeApi());
  if (runtime.Initialize(flat.device) != Status::kOk) {
    EmitFailure(config.case_id, "ascendc_real_device", "ACL runtime initialization failed", "failed", flat.variant,
                "runtime_init", flat.step);
    return 4;
  }
  AclnnRunSpecs specs;
  ParseAclnnSpecs(flat, &specs);
  if (const int error = LoadAclnnHostInputs(flat, config, &specs)) return error;
  if (const int error = AllocateAclnnBuffers(flat, config, &runtime, &specs)) return error;
  if (const int error = StartAclnnProfiler(flat, config, collect_dir, &specs)) return error;
  AclnnExecutor executor(&runtime);
  AclnnRunResult run_result;
  const Status status = executor.Run(flat.aclnn_op, specs.input_specs, specs.inputs, specs.host_inputs,
                                     specs.output_specs, specs.outputs, flat.warmup, flat.repeat, &run_result);
  const bool profiler_collected = specs.profiler != nullptr;
  std::string profiler_collect_dir;
  if (profiler_collected && specs.profiler->Stop() == Status::kOk) profiler_collect_dir = collect_dir;
  if (status != Status::kOk) {
    EmitFailure(config.case_id, "ascendc_real_device", run_result.reason, "failed", flat.variant, run_result.error_code,
                flat.step);
    return 4;
  }
  RunResult result;
  if (const int error = DecodeAclnnOutputs(flat, config, specs, run_result, &result)) return error;
  RunReport run_report;
  nlohmann::json report;
  if (const int error = SummarizeAclnnTiming(flat, config, backend_info, profiler_collected, profiler_collect_dir,
                                             run_result, &run_report, &report))
    return error;
  AppendOutputsToReport(&report, result);
  if (const auto write_error = WriteFlatExecutionOutputs(&flat, result, &report)) return write_error;
  report["abi_metadata"] = flat.abi_metadata;
  std::cout << report.dump() << '\n';
  return 0;
}
}  // namespace

void ParseLaunchOptions(int argc, char **argv, bool *is_flat, std::string *request_path, std::string *collect_dir) {
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--request" && i + 1 < argc) {
      *is_flat = true;
      *request_path = argv[i + 1];
      ++i;
    } else if (option == "--collect-profile" && i + 1 < argc) {
      *collect_dir = argv[i + 1];
      ++i;
    }
  }
}

void ResolveLaunchPaths(bool is_flat, const FlatRequest &flat, int argc, char **argv,
                        const std::string &collect_dir_cli, std::string *case_dir, std::string *profile,
                        std::string *collect_dir) {
  *case_dir = is_flat ? flat.case_dir : (argc > 1 ? argv[1] : "");
  *profile = is_flat ? flat.profile : (argc > 2 ? argv[2] : "");
  *collect_dir = collect_dir_cli.empty() ? (is_flat ? flat.profile_dir : "") : collect_dir_cli;
}

int RejectInvalidInvocation(int argc, char **argv) {
  std::cerr << "usage: device_validation_runner <case_dir> <profile.json> <module.so> <input.bin>...\n";
  EmitFailure(argc > 1 ? argv[1] : "unknown", "ascendc_real_device", "invalid invocation", "failed");
  return 2;
}

RunRequest BuildRealRequest(const FlatRequest *flat, const CaseConfig &config, const BackendInfo &backend_info,
                            char **argv) {
  auto request = BuildRunRequest(config, backend_info, flat != nullptr ? flat->module : argv[3]);
  request.profiler = flat != nullptr && flat->profiler;
  return request;
}

int RunRealDevice(const FlatRequest *flat, const std::string &case_dir, const std::string &profile,
                  const std::string &collect_dir, int argc, char **argv) {
  const bool is_flat = flat != nullptr;
  try {
    const auto config = LoadCaseConfig(case_dir);
    auto backend = CreateExecutionBackend("ascendc_real_device", profile);
    if (backend == nullptr) {
      EmitFailure(config.case_id, "ascendc_real_device", "backend is unavailable", "not_applicable");
      return 3;
    }
    const int input_end = 4 + static_cast<int>(config.inputs.size());
    if (!is_flat && argc < input_end) {
      EmitFailure(config.case_id, backend->GetCapabilities().name, "invalid invocation: input count", "failed");
      return 2;
    }
    const auto backend_info = backend->GetCapabilities();
    if (is_flat) {
      if (const auto flat_error = ValidateFlatRequestForBackend(*flat, config)) return *flat_error;
      if (!flat->aclnn_op.empty()) return DispatchAclnnExecution(*flat, config, backend_info, collect_dir);
    }
    auto request = BuildRealRequest(flat, config, backend_info, argv);
    if (is_flat) {
      if (const auto flat_error = ApplyFlatRequestToRunRequest(*flat, config, &request)) return *flat_error;
    }
    if (const auto option_error = ApplyRunOptions(argc, argv, input_end, is_flat, config, backend_info, &request))
      return *option_error;
    return DispatchRealExecution(backend.get(), &request, config, flat, is_flat, argc, argv, input_end, collect_dir);
  } catch (const std::exception &error) {
    EmitFailure(argc > 1 ? argv[1] : "unknown", "ascendc_real_device", error.what(), "failed");
    std::cerr << error.what() << '\n';
    return 4;
  }
}

int main(int argc, char **argv) {
  FlatRequest flat;
  std::string request_path;
  std::string collect_dir_cli;
  bool is_flat = false;
  ParseLaunchOptions(argc, argv, &is_flat, &request_path, &collect_dir_cli);
  if (is_flat && !TryLoadFlatRequest(request_path, &flat)) return 2;
  if (const auto fake_exit = RunHostFakeExecution(flat)) return *fake_exit;
  std::string case_dir;
  std::string profile;
  std::string collect_dir;
  ResolveLaunchPaths(is_flat, flat, argc, argv, collect_dir_cli, &case_dir, &profile, &collect_dir);
  if (!is_flat && argc < 4) return RejectInvalidInvocation(argc, argv);
  return RunRealDevice(is_flat ? &flat : nullptr, case_dir, profile, collect_dir, argc, argv);
}
