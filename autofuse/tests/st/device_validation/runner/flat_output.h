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

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace device_validation {

struct FlatOutputTestHooks {
  int fail_flush_at = 0;
  int fail_close_at = 0;
  int fail_rename_at = 0;
  int create_target_at = 0;
  int create_temp_file_at = 0;
  int create_temp_symlink_at = 0;
};

void SetFlatOutputTestHooks(const FlatOutputTestHooks &hooks);
void ClearFlatOutputTestHooks();

bool WriteFlatOutputs(const std::vector<std::string> &output_files,
                      const std::vector<std::vector<uint8_t>> &output_bytes, const std::string &artifact_dir,
                      std::string *error);

}  // namespace device_validation
