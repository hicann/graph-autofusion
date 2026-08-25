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

#include "support_matrix.h"

#include <filesystem>
#include <string>

namespace device_validation {
struct DeviceProfile {
  SocCapabilities capabilities;
  std::string simulator_backend;
  std::string real_device_backend;
  std::string toolkit_env;
  std::string profiler_tool;
  uint64_t max_tiling_bytes = 0;
  uint64_t max_workspace_bytes = 0;
  uint32_t max_block_dimension = 0;
};

DeviceProfile LoadDeviceProfile(const std::filesystem::path &path);
bool IsProfilerAvailable(const DeviceProfile &profile);
}  // namespace device_validation
