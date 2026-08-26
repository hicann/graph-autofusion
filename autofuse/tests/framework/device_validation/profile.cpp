/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profile.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace device_validation {
namespace {
using Json = nlohmann::json;
}

DeviceProfile LoadDeviceProfile(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) throw std::invalid_argument("device profile not found");
  Json json;
  input >> json;
  DeviceProfile profile;
  profile.capabilities.soc = json.at("profile").get<std::string>();
  profile.capabilities.allowed_abi = json.at("allowed_abi").get<std::string>();
  profile.capabilities.dtypes = json.at("dtypes").get<std::vector<std::string>>();
  profile.capabilities.max_shape_elements = json.value("max_shape_elements", uint64_t{0});
  profile.simulator_backend = json.at("simulator_backend").get<std::string>();
  profile.real_device_backend = json.at("real_device_backend").get<std::string>();
  profile.capabilities.profiler_optional = json.value("profiler", "unsupported") == "optional";
  const auto tools = json.value("tools", Json::object());
  profile.toolkit_env = tools.value("toolkit", "");
  profile.profiler_tool = tools.value("profiler", "");
  const auto resources = json.value("resources", Json::object());
  profile.max_tiling_bytes = resources.value("max_tiling_bytes", uint64_t{0});
  profile.max_workspace_bytes = resources.value("max_workspace_bytes", uint64_t{0});
  profile.max_block_dimension = resources.value("max_block_dimension", uint32_t{0});
  return profile;
}

bool IsProfilerAvailable(const DeviceProfile &profile) {
  if (profile.profiler_tool.empty()) return false;
  const auto path = std::getenv("PATH");
  if (path == nullptr) return false;
  std::string paths(path);
  size_t start = 0;
  while (start <= paths.size()) {
    const auto end = paths.find(':', start);
    const auto dir = paths.substr(start, end == std::string::npos ? end : end - start);
    if (std::filesystem::exists(std::filesystem::path(dir) / profile.profiler_tool)) return true;
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return false;
}
}  // namespace device_validation
