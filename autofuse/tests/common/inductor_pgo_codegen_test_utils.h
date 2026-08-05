/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TESTS_COMMON_INDUCTOR_PGO_CODEGEN_TEST_UTILS_H_
#define AUTOFUSE_TESTS_COMMON_INDUCTOR_PGO_CODEGEN_TEST_UTILS_H_

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "autofuse_config/auto_fuse_config.h"
#include "codegen.h"

namespace autofuse::tests {

class ScopedAutofusePgoFlag {
 public:
  explicit ScopedAutofusePgoFlag(bool enabled = true) : original_config_(att::AutoFuseConfig::GetPgoStrategyConfig()) {
    const char *flags = std::getenv("AUTOFUSE_FLAGS");
    if (flags != nullptr) {
      original_flags_ = flags;
      had_original_flags_ = true;
    }
    SetFlag(enabled);
  }

  ~ScopedAutofusePgoFlag() {
    had_original_flags_ ? setenv("AUTOFUSE_FLAGS", original_flags_.c_str(), 1) : unsetenv("AUTOFUSE_FLAGS");
    att::AutoFuseConfig::MutablePgoStrategyConfig() = original_config_;
  }

 private:
  static void SetFlag(bool enabled) {
    const char *value = enabled ? "--autofuse_enable_pgo=true" : "--autofuse_enable_pgo=false";
    setenv("AUTOFUSE_FLAGS", value, 1);
    att::AutoFuseConfig::MutablePgoStrategyConfig().is_first_init = true;
  }

  std::string original_flags_;
  att::PgoStrategyConfig original_config_;
  bool had_original_flags_ = false;
};

inline void WriteCodegenResult(const codegen::CodegenResult &result, const std::vector<std::string> &paths) {
  constexpr size_t kExpectedPathCount = 3U;
  ASSERT_EQ(paths.size(), kExpectedPathCount);
  std::fstream tiling_data_file(paths[0], std::ios::out);
  std::fstream host_file(paths[1], std::ios::out);
  std::fstream device_file(paths[2], std::ios::out);
  ASSERT_TRUE(tiling_data_file.is_open() && host_file.is_open() && device_file.is_open());
  tiling_data_file << result.tiling_data;
  host_file << result.tiling;
  device_file << result.kernel;
}

}  // namespace autofuse::tests

#endif  // AUTOFUSE_TESTS_COMMON_INDUCTOR_PGO_CODEGEN_TEST_UTILS_H_
