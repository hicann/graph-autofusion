/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inductor_host_helper_common.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <utility>

#include "nlohmann/json.hpp"

namespace autofuse::tests {
namespace {

constexpr int64_t kTopnInvalid = 0;
constexpr int64_t kTopnOne = 1;

struct DlHandle {
  void *ptr = nullptr;
  explicit DlHandle(void *handle) : ptr(handle) {}
  ~DlHandle() {
    if (ptr != nullptr) {
      (void)dlclose(ptr);
    }
  }
  bool Close() {
    if (ptr == nullptr) {
      return true;
    }
    dlerror();
    if (dlclose(ptr) != 0) {
      const char *error = dlerror();
      std::cerr << "dlclose host failed";
      if (error != nullptr) {
        std::cerr << ": " << error;
      }
      std::cerr << std::endl;
      return false;
    }
    ptr = nullptr;
    return true;
  }
};

bool Check(bool condition, const std::string &message) {
  if (condition) {
    return true;
  }
  std::cerr << message << std::endl;
  return false;
}

bool ParseInt64(const std::string &text, int64_t *value) {
  try {
    size_t pos = 0;
    *value = std::stoll(text, &pos);
    return pos == text.size();
  } catch (const std::exception &) {
    return false;
  }
}

std::vector<std::string> Split(const std::string &text, char delimiter) {
  std::vector<std::string> result;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, delimiter)) {
    if (!item.empty()) {
      result.push_back(item);
    }
  }
  return result;
}

bool ParseDynamicShapeArgs(const std::string &text, std::vector<int64_t> *args) {
  args->clear();
  for (const auto &item : Split(text, ',')) {
    int64_t value = 0;
    if (!ParseInt64(item, &value)) {
      std::cerr << "invalid dynamic shape arg: " << item << std::endl;
      return false;
    }
    args->push_back(value);
  }
  return true;
}

std::string JsonValueToString(const nlohmann::json &value) {
  if (value.is_string()) {
    return value.get<std::string>();
  }
  return value.dump();
}

bool LoadInputConfigs(const std::string &path, InputConfigs *configs) {
  configs->clear();
  if (path.empty()) {
    return true;
  }

  std::ifstream in(path);
  if (!in.is_open()) {
    std::cerr << "open input configs failed: " << path << std::endl;
    return false;
  }
  nlohmann::json json_value;
  in >> json_value;
  if (!json_value.is_array()) {
    std::cerr << "input configs should be a json array" << std::endl;
    return false;
  }
  for (const auto &item : json_value) {
    if (!item.is_object()) {
      std::cerr << "input config item should be a json object" << std::endl;
      return false;
    }
    std::map<std::string, std::string> config;
    for (auto iter = item.begin(); iter != item.end(); ++iter) {
      config[iter.key()] = JsonValueToString(iter.value());
    }
    configs->push_back(std::move(config));
  }
  return true;
}

bool ParseOptionValue(int argc, char **argv, int *index, std::string *value) {
  if (*index + 1 >= argc) {
    std::cerr << "missing value for " << argv[*index] << std::endl;
    return false;
  }
  *value = argv[++(*index)];
  return true;
}

bool ApplyOption(const std::string &name, const std::string &value, HostHelperOptions *options) {
  if (name == "--host-so") {
    options->host_so = value;
    return true;
  }
  if (name == "--tiling-repr-out") {
    options->tiling_repr_out = value;
    return true;
  }
  if (name == "--input-configs") {
    options->input_configs_file = value;
    return true;
  }
  if (name == "--dynamic-shape-args") {
    return ParseDynamicShapeArgs(value, &options->dynamic_shape_args);
  }
  if (name == "--topn") {
    return ParseInt64(value, &options->topn);
  }
  std::cerr << "unsupported option: " << name << std::endl;
  return false;
}

bool VerifyInvalidTopn(HostCaseRunner *runner, const InputConfigs &input_configs) {
  return Check(runner->GenerateTopn(input_configs, kTopnInvalid) == -1, "invalid topn should fail") &&
         Check(runner->ResultSize() == 0U, "invalid topn result should be empty");
}

bool VerifyAutofuseTiling(HostCaseRunner *runner) {
  if (!Check(runner->RunAutofuseTiling() == 0, "AutofuseTiling failed")) {
    return false;
  }
  return Check(runner->DefaultBlockDim() > 0U, "default block_dim should be positive") &&
         Check(runner->DefaultRepr().find("AutofuseTilingData{") != std::string::npos, "tiling repr format mismatch");
}

bool VerifyTopnResultShape(HostCaseRunner *runner, int64_t topn) {
  return Check(runner->ResultSize() > 0U, "topn result should not be empty") &&
         Check(runner->ResultSize() <= static_cast<size_t>(topn), "topn result size too large");
}

bool VerifyTopnResults(HostCaseRunner *runner) {
  std::vector<std::string> reprs;
  for (size_t i = 0; i < runner->ResultSize(); ++i) {
    const std::string repr = runner->ResultRepr(i);
    if (!Check(!repr.empty(), "topn repr should not be empty")) {
      return false;
    }
    reprs.push_back(repr);
    if (!Check(runner->ResultWorkspace(i) >= 0, "topn workspace should be non-negative") ||
        !Check(runner->ResultBlockDim(i) > 0, "topn block_dim should be positive")) {
      return false;
    }
  }
  std::sort(reprs.begin(), reprs.end());
  if (!Check(std::unique(reprs.begin(), reprs.end()) == reprs.end(), "topn repr should be unique")) {
    return false;
  }
  return true;
}

bool VerifyTopnUniqueness(HostCaseRunner *runner, const InputConfigs &input_configs, const HostHelperOptions &options) {
  if (!Check(runner->GenerateTopn(input_configs, options.topn) == 0, "multi-candidate GenerateTopnSolutions failed") ||
      !VerifyTopnResultShape(runner, options.topn)) {
    return false;
  }
  return VerifyTopnResults(runner);
}

bool VerifyEmptyConfigPath(HostCaseRunner *runner) {
  const InputConfigs empty_configs;
  return Check(runner->GenerateTopn(empty_configs, kTopnOne) == 0, "empty config GenerateTopnSolutions failed") &&
         Check(runner->ResultSize() == 1U, "empty config result size should be 1") && VerifyTopnResults(runner);
}

bool WriteFile(const std::string &path, const std::string &content) {
  std::ofstream out(path);
  if (!out.is_open()) {
    std::cerr << "open output failed: " << path << std::endl;
    return false;
  }
  out << content;
  return true;
}

}  // namespace

bool ParseHostHelperOptions(int argc, char **argv, HostHelperOptions *options) {
  for (int i = 1; i < argc; ++i) {
    const std::string name = argv[i];
    if (name == "--verify-empty-config") {
      options->verify_empty_config = true;
      continue;
    }
    if (name == "--check-z0t-positive") {
      options->check_z0t_positive = true;
      continue;
    }
    std::string value;
    if (!ParseOptionValue(argc, argv, &i, &value) || !ApplyOption(name, value, options)) {
      return false;
    }
  }
  return Check(!options->host_so.empty(), "--host-so is required") &&
         Check(!options->tiling_repr_out.empty(), "--tiling-repr-out is required") &&
         Check(options->topn > 0, "--topn should be positive");
}

int RunHostCheck(const HostHelperOptions &options, HostCaseRunner *runner) {
  InputConfigs input_configs;
  DlHandle host_handle(dlopen(options.host_so.c_str(), RTLD_LAZY | RTLD_LOCAL));
  if (host_handle.ptr == nullptr) {
    std::cerr << "dlopen host failed: " << dlerror() << std::endl;
    return 1;
  }

  const bool loaded = LoadInputConfigs(options.input_configs_file, &input_configs);
  const bool passed = loaded && runner->Resolve(host_handle.ptr) && VerifyInvalidTopn(runner, input_configs) &&
                      VerifyAutofuseTiling(runner) && VerifyTopnUniqueness(runner, input_configs, options) &&
                      (!options.check_z0t_positive || runner->VerifyExtraTopnResult()) &&
                      (!options.verify_empty_config || VerifyEmptyConfigPath(runner)) &&
                      WriteFile(options.tiling_repr_out, runner->DefaultRepr());
  if (!host_handle.Close()) {
    return 1;
  }
  return passed ? 0 : 1;
}

}  // namespace autofuse::tests
