/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {
using Json = nlohmann::json;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    std::string pattern = (std::filesystem::temp_directory_path() / "device-validation-flat-contract-XXXXXX").string();
    if (mkdtemp(pattern.data()) == nullptr) throw std::runtime_error("temporary directory is unavailable");
    path_ = pattern;
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  const std::filesystem::path &path() const {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

Json Read(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("request is unavailable");
  Json request;
  input >> request;
  return request;
}

void Write(const std::filesystem::path &path, Json value, bool synchronize_contract = true) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("request cannot be written");
  if (!value.contains("contract_schema") || !value.at("contract_schema").is_object())
    throw std::runtime_error("request contract schema is unavailable");
  if (synchronize_contract) {
    value["contract_schema"]["case_id"] = value.value("case_id", "");
    value["contract_schema"]["variant"] = value.value("variant", "");
    value["contract_schema"]["step"] = value.value("step", -1);
    if (value.at("tensor_specs").is_array()) value["contract_schema"]["inputs"] = value.at("tensor_specs");
    if (value.at("output_specs").is_array()) value["contract_schema"]["outputs"] = value.at("output_specs");
  }
  output << value.dump();
}

int Run(const std::filesystem::path &runner, const std::filesystem::path &request, Json *report) {
  const auto report_path = request.parent_path() / "runner_report.json";
  const pid_t child = fork();
  if (child < 0) throw std::runtime_error("runner process cannot be created");
  if (child == 0) {
    const int output = open(report_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (output < 0) _exit(127);
    if (dup2(output, STDOUT_FILENO) < 0) _exit(127);
    close(output);
    execl(runner.c_str(), runner.c_str(), "--request", request.c_str(), static_cast<char *>(nullptr));
    _exit(127);
  }
  int status = 0;
  if (waitpid(child, &status, 0) < 0 || !WIFEXITED(status)) throw std::runtime_error("runner process failed");
  *report = Read(report_path);
  return WEXITSTATUS(status);
}

void AddInputSpecs(Json &request, const std::filesystem::path &root, size_t input_count, size_t step);
void ApplyStepDtypes(Json &request, size_t step);

Json MakeRequest(const std::filesystem::path &root, size_t input_count, size_t step) {
  Json request = {{"case_id", "isinf_maskedfill_fusion"},
                  {"step", step},
                  {"case_dir", "autofuse/tests/st/device_validation/cases/isinf_maskedfill_fusion"},
                  {"profile", "autofuse/tests/st/device_validation/profiles/ascend950.json"},
                  {"soc_profile", "ascend950"},
                  {"module", "host_fake_module"},
                  {"abi", "AutofuseLaunch"},
                  {"launch_abi", "AutofuseLaunch"},
                  {"input_count", input_count},
                  {"output_count", 1},
                  {"abi_metadata",
                   {{"launch_abi", "AutofuseLaunch"},
                    {"input_count", input_count},
                    {"output_count", 1},
                    {"input_dtypes", Json::array()},
                    {"output_dtypes", {"float16"}}}},
                  {"tensor_files", Json::array()},
                  {"inputs", Json::array()},
                  {"outputs", Json::array({(root / ("step_" + std::to_string(step) + ".bin")).string()})},
                  {"tensor_specs", Json::array()},
                  {"output_specs", {{{"dtype", "float16"}, {"shape", {1}}}}},
                  {"artifact_dir", root.string()},
                  {"device", 0},
                  {"warmup", 0},
                  {"repeat", 1},
                  {"variant", "unfused"}};
  request["shape"] = {1};
  request["selected_shape"] = request["shape"];
  if (step > 2) {
    request["variant"] = "fused";
    request["step"] = -1;
  }
  AddInputSpecs(request, root, input_count, step);
  ApplyStepDtypes(request, step);
  request["contract_schema"] = {
      {"schema_version", 1},     {"case_id", request["case_id"]},     {"variant", request["variant"]},
      {"step", request["step"]}, {"inputs", request["tensor_specs"]}, {"outputs", request["output_specs"]}};
  return request;
}

void AddInputSpecs(Json &request, const std::filesystem::path &root, size_t input_count, size_t step) {
  for (size_t input = 0; input < input_count; ++input) {
    const auto path = input == 0 && step > 0 ? root / ("step_" + std::to_string(step - 1) + ".bin")
                                             : root / ("input_" + std::to_string(step) + "_" + std::to_string(input));
    request["inputs"].push_back(path.string());
    request["tensor_files"].push_back(path.string());
    request["tensor_specs"].push_back({{"dtype", "float16"}, {"shape", {1}}});
    request["abi_metadata"]["input_dtypes"].push_back("float16");
  }
}

void ApplyStepDtypes(Json &request, size_t step) {
  if (step < 2) {
    request["output_specs"][0]["dtype"] = "uint8";
    request["abi_metadata"]["output_dtypes"][0] = "uint8";
  }
  if (step == 1) {
    request["tensor_specs"][0]["dtype"] = "uint8";
    request["abi_metadata"]["input_dtypes"][0] = "uint8";
    request["tensor_specs"][1]["dtype"] = "uint8";
    request["abi_metadata"]["input_dtypes"][1] = "uint8";
  }
  if (step == 2) {
    request["tensor_specs"][1]["dtype"] = "uint8";
    request["abi_metadata"]["input_dtypes"][1] = "uint8";
  }
}

Json MakeRequestWithOutputs(const std::filesystem::path &root) {
  auto request = MakeRequest(root, 3, 6);
  request["variant"] = "fused";
  request["step"] = -1;
  request["shape"] = {2, 3};
  request["selected_shape"] = request["shape"];
  request["abi"] = "AutofuseLaunchV2";
  request["launch_abi"] = "AutofuseLaunchV2";
  request["inputs"] =
      Json::array({(root / "input_0.bin").string(), (root / "input_1.bin").string(), (root / "input_2.bin").string()});
  request["tensor_files"] = request["inputs"];
  request["outputs"] = Json::array({(root / "float32.bin").string()});
  request["output_count"] = 1;
  request["abi_metadata"] = {{"launch_abi", "AutofuseLaunchV2"},
                             {"input_count", 3},
                             {"output_count", 1},
                             {"input_dtypes", {"float16", "uint8", "float16"}},
                             {"output_dtypes", {"float16"}}};
  request["output_specs"] = {{{"dtype", "float16"}, {"shape", {2, 3}}}};
  request["tensor_specs"] = {{{"dtype", "float16"}, {"shape", {2, 3}}},
                             {{"dtype", "uint8"}, {"shape", {2, 3}}},
                             {{"dtype", "float16"}, {"shape", {2, 3}}}};
  return request;
}
void ExpectErrorExit(const std::filesystem::path &runner, const Json &request,
                     const std::filesystem::path &request_path, const std::string &expected_code,
                     const std::string &failure_message, const std::string &expected_stage = "",
                     bool synchronize_contract = true) {
  Write(request_path, request, synchronize_contract);
  Json report;
  if (Run(runner, request_path, &report) != 2 || report.at("error_code") != expected_code ||
      (!expected_stage.empty() && report.at("stage_status") != expected_stage))
    throw std::runtime_error(failure_message);
}

void ValidateFlatRequestMetadata(const Json &request, size_t step) {
  if (request.at("input_count") != step + 1 || request.at("output_count") != 1 ||
      request.at("output_specs").size() != 1 ||
      request.at("output_specs").at(0).at("dtype") != (step < 2 ? "uint8" : "float16") ||
      request.at("output_specs").at(0).at("shape") != Json::array({1}))
    throw std::runtime_error("flat request output metadata mismatch");
}

void VerifyStepResult(const Json &request, const Json &report, size_t step, const std::string &previous) {
  const auto output_path = request.at("outputs").at(0).get<std::string>();
  std::ifstream output(output_path, std::ios::binary);
  if (!output) throw std::runtime_error("flat output is unavailable");
  const std::string content((std::istreambuf_iterator<char>(output)), std::istreambuf_iterator<char>());
  const size_t expected_bytes = step < 2 ? 1 : 2;
  if (content.size() != expected_bytes || content[0] != 0 || (expected_bytes == 2 && content[1] != 0))
    throw std::runtime_error("flat output tensor contract mismatch");
  if (report.at("outputs").size() != 1 || report.at("outputs").at(0).at("dtype") != (step < 2 ? "uint8" : "float16") ||
      report.at("outputs").at(0).at("shape") != Json::array({1}) ||
      report.at("outputs").at(0).at("data") != Json::array({0}))
    throw std::runtime_error("flat report output metadata mismatch");
  if (report.at("actual_abi") != "AutofuseLaunch" || report.at("variant") != "unfused" || report.at("step") != step)
    throw std::runtime_error("flat report execution metadata mismatch");
  if (report.at("output_marker") != "step=" + std::to_string(step))
    throw std::runtime_error("flat output marker mismatch");
  if (step > 0 && report.at("input_marker").at(0) != previous)
    throw std::runtime_error("previous output marker was not consumed");
}

void RunStepContract(const std::filesystem::path &runner, const std::filesystem::path &root) {
  std::string previous;
  for (size_t step = 0; step < 3; ++step) {
    const auto request = MakeRequest(root, step + 1, step);
    ValidateFlatRequestMetadata(request, step);
    for (const auto &input : request.at("inputs")) {
      if (!std::filesystem::exists(input.get<std::string>())) {
        std::ofstream(input.get<std::string>(), std::ios::binary) << "input";
      }
    }
    const auto request_path = root / ("request_" + std::to_string(step) + ".json");
    Write(request_path, request);
    Json report;
    if (Run(runner, request_path, &report) != 0 || report.at("stage_status") != "passed")
      throw std::runtime_error("valid arity did not execute fake backend: " + report.dump());
    VerifyStepResult(request, report, step, previous);
    previous = request.at("outputs").at(0).get<std::string>();
  }
}

void RunArityErrorCases(const std::filesystem::path &runner, const std::filesystem::path &root) {
  auto malformed = MakeRequest(root, 1, 3);
  malformed["output_specs"] = "malformed";
  ExpectErrorExit(runner, malformed, root / "malformed.json", "output_specs", "malformed spec error code mismatch");
  auto unsupported = MakeRequest(root, 4, 4);
  unsupported["input_count"] = 4;
  ExpectErrorExit(runner, unsupported, root / "unsupported.json", "unsupported_abi_arity",
                  "unsupported arity error code mismatch");
  auto unsupported_output_arity = MakeRequest(root, 1, 5);
  unsupported_output_arity["output_count"] = 2;
  unsupported_output_arity["outputs"].push_back((root / "extra.bin").string());
  unsupported_output_arity["output_specs"].push_back({{"dtype", "float16"}, {"shape", {1}}});
  ExpectErrorExit(runner, unsupported_output_arity, root / "unsupported_output_arity.json", "abi_metadata",
                  "unsupported output arity error code mismatch");
}

void RunMultiOutputCase(const std::filesystem::path &runner, const std::filesystem::path &root) {
  const auto multi_output = MakeRequestWithOutputs(root);
  const auto multi_output_path = root / "multi_output.json";
  Write(multi_output_path, multi_output);
  for (const auto &input : multi_output.at("inputs"))
    std::ofstream(input.get<std::string>(), std::ios::binary) << "input";
  Json report;
  if (Run(runner, multi_output_path, &report) != 0 || report.at("stage_status") != "passed")
    throw std::runtime_error("non-default output specs did not execute fake backend: " + report.dump());
  if (std::filesystem::file_size(multi_output.at("outputs").at(0).get<std::string>()) != 12)
    throw std::runtime_error("non-default output size mismatch");
  if (report.at("outputs") != Json::array({{{"dtype", "float16"}, {"shape", {2, 3}}, {"data", {0, 0, 0, 0, 0, 0}}}}))
    throw std::runtime_error("non-default output metadata mismatch");
}

void RunBfloat16OutputCase(const std::filesystem::path &runner, const std::filesystem::path &root) {
  auto bfloat16 = MakeRequestWithOutputs(root);
  bfloat16["outputs"] = Json::array({(root / "bfloat16.bin").string()});
  const auto bfloat16_path = root / "bfloat16.json";
  for (const auto &input : bfloat16.at("inputs")) std::ofstream(input.get<std::string>(), std::ios::binary) << "input";
  Write(bfloat16_path, bfloat16);
  Json report;
  if (Run(runner, bfloat16_path, &report) != 0 || report.at("stage_status") != "passed" ||
      std::filesystem::file_size(bfloat16.at("outputs").at(0).get<std::string>()) != 12 ||
      report.at("outputs").at(0).at("dtype") != "float16")
    throw std::runtime_error("float16 output contract mismatch");
}

void RunInvalidSpecErrorCase(const std::filesystem::path &runner, const std::filesystem::path &root) {
  auto invalid_spec = MakeRequestWithOutputs(root);
  invalid_spec["output_specs"][0]["dtype"] = "float64";
  ExpectErrorExit(runner, invalid_spec, root / "invalid_spec.json", "output_specs",
                  "invalid output spec error contract mismatch", "failed");
}

void RunShapeErrorCases(const std::filesystem::path &runner, const std::filesystem::path &root) {
  const std::vector<Json> invalid_shapes = {Json::array(), Json::array({0}), Json::array({-1}),
                                            Json::array({static_cast<int64_t>(1ULL << 62), 4})};
  for (size_t shape_index = 0; shape_index < invalid_shapes.size(); ++shape_index) {
    const auto &shape = invalid_shapes[shape_index];
    auto invalid_shape = MakeRequest(root, 1, 8 + shape_index);
    invalid_shape["tensor_specs"][0]["shape"] = shape;
    if (shape_index == 0 && !invalid_shape["tensor_specs"][0]["shape"].empty())
      throw std::runtime_error("empty shape test construction failed");
    std::ofstream(invalid_shape.at("inputs").at(0).get<std::string>(), std::ios::binary) << "input";
    ExpectErrorExit(runner, invalid_shape, root / ("invalid_input_shape_" + std::to_string(shape_index) + ".json"),
                    "shape", "invalid input shape error contract mismatch");
  }
  for (size_t shape_index = 0; shape_index < invalid_shapes.size(); ++shape_index) {
    const auto &shape = invalid_shapes[shape_index];
    auto invalid_shape = MakeRequest(root, 1, 12 + shape_index);
    invalid_shape["output_specs"][0]["shape"] = shape;
    ExpectErrorExit(runner, invalid_shape, root / ("invalid_output_shape_" + std::to_string(shape_index) + ".json"),
                    "output_specs", "invalid output shape error contract mismatch");
  }
  const std::vector<Json> invalid_dimension_shapes = {
      Json::array({true}), Json::array({static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1})};
  for (size_t shape_index = 0; shape_index < invalid_dimension_shapes.size(); ++shape_index) {
    auto invalid_input = MakeRequest(root, 1, 20 + shape_index);
    invalid_input["tensor_specs"][0]["shape"] = invalid_dimension_shapes[shape_index];
    ExpectErrorExit(runner, invalid_input, root / ("invalid_input_dimension_" + std::to_string(shape_index) + ".json"),
                    "shape", "invalid input dimension error contract mismatch");
    auto invalid_output = MakeRequest(root, 1, 24 + shape_index);
    invalid_output["output_specs"][0]["shape"] = invalid_dimension_shapes[shape_index];
    ExpectErrorExit(runner, invalid_output,
                    root / ("invalid_output_dimension_" + std::to_string(shape_index) + ".json"), "output_specs",
                    "invalid output dimension error contract mismatch");
  }
}

void RunMetadataErrorCases(const std::filesystem::path &runner, const std::filesystem::path &root) {
  auto invalid_variant = MakeRequest(root, 1, 0);
  invalid_variant["variant"] = "invalid";
  ExpectErrorExit(runner, invalid_variant, root / "invalid_variant.json", "invalid_variant",
                  "invalid variant error contract mismatch");
  auto invalid_step = MakeRequest(root, 1, 9);
  invalid_step["step"] = 9;
  ExpectErrorExit(runner, invalid_step, root / "invalid_step.json", "invalid_step",
                  "invalid step error contract mismatch");
  auto mismatched_spec = MakeRequest(root, 1, 0);
  mismatched_spec["tensor_specs"][0]["dtype"] = "float32";
  ExpectErrorExit(runner, mismatched_spec, root / "mismatched_spec.json", "input_specs",
                  "mismatched input spec error contract mismatch", "", false);
  auto mismatched_case = MakeRequest(root, 1, 0);
  mismatched_case["case_id"] = "other_case";
  ExpectErrorExit(runner, mismatched_case, root / "mismatched_case.json", "case_id",
                  "mismatched case error contract mismatch");
}

void RunOverflowErrorCases(const std::filesystem::path &runner, const std::filesystem::path &root) {
  auto max_int64_input = MakeRequest(root, 1, 28);
  max_int64_input["tensor_specs"][0] = {{"dtype", "uint64"}, {"shape", {std::numeric_limits<int64_t>::max()}}};
  ExpectErrorExit(runner, max_int64_input, root / "max_int64_input.json", "input_specs",
                  "int64 byte overflow error contract mismatch");
  auto input_byte_overflow = MakeRequest(root, 1, 30);
  input_byte_overflow["tensor_specs"][0] = {{"dtype", "float16"}, {"shape", {3037000500LL, 3037000500LL}}};
  ExpectErrorExit(runner, input_byte_overflow, root / "input_byte_overflow.json", "input_specs",
                  "input byte overflow error contract mismatch");
  auto output_byte_overflow = MakeRequest(root, 1, 31);
  output_byte_overflow["output_specs"][0] = {{"dtype", "uint64"}, {"shape", {3037000500LL, 3037000500LL}}};
  ExpectErrorExit(runner, output_byte_overflow, root / "output_byte_overflow.json", "output_specs",
                  "output byte overflow error contract mismatch");
}

void RunEscapedOutputCase(const std::filesystem::path &runner, const std::filesystem::path &root) {
  auto escaped_output = MakeRequest(root, 1, 32);
  escaped_output["outputs"] = Json::array({(root / ".." / "escaped.bin").string()});
  std::ofstream(escaped_output.at("inputs").at(0).get<std::string>(), std::ios::binary) << "input";
  const auto escaped_output_path = root / "escaped_output.json";
  Write(escaped_output_path, escaped_output);
  Json report;
  const auto escaped_status = Run(runner, escaped_output_path, &report);
  if (escaped_status == 0 || (report.at("error_code") != "output_path" && report.at("error_code") != "output_write" &&
                              report.at("error_code") != "fake_execution"))
    throw std::runtime_error("output path containment error contract mismatch: " + std::to_string(escaped_status) +
                             " " + report.dump());
}
}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("runner path is required");
    TemporaryDirectory temporary;
    const std::filesystem::path runner = argv[1];
    const auto root = temporary.path();
    RunStepContract(runner, root);
    RunArityErrorCases(runner, root);
    RunMultiOutputCase(runner, root);
    RunBfloat16OutputCase(runner, root);
    RunInvalidSpecErrorCase(runner, root);
    RunShapeErrorCases(runner, root);
    RunMetadataErrorCases(runner, root);
    RunOverflowErrorCases(runner, root);
    RunEscapedOutputCase(runner, root);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
