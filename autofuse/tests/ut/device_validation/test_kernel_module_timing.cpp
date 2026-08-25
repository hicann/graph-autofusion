/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_module.h"

#include <gtest/gtest.h>

namespace device_validation {
namespace {

class TimingLoader final : public ModuleLoader {
 public:
  explicit TimingLoader(bool legacy, bool fail_launch = false) : legacy_(legacy), fail_launch_(fail_launch) {}

  void *Load(const std::filesystem::path &) override {
    return this;
  }

  void *Symbol(void *, const char *name) override {
    if (std::string(name) == "GetTilingDataSize") return reinterpret_cast<void *>(&GetSize);
    if (std::string(name) == "AutofuseTiling") return reinterpret_cast<void *>(&Tiling);
    if (!legacy_ && std::string(name) == "AutofuseLaunchV2") {
      failing_ = fail_launch_;
      return reinterpret_cast<void *>(&LaunchV2);
    }
    if (legacy_ && std::string(name) == "AutofuseLaunch") {
      failing_ = fail_launch_;
      return reinterpret_cast<void *>(&LaunchLegacy);
    }
    return nullptr;
  }

  Status Unload(void *) override {
    return Status::kOk;
  }

 private:
  static size_t GetSize() {
    return 1;
  }
  static ge::graphStatus Tiling(void *, uint32_t *workspace, uint32_t *block, void *) {
    *workspace = 0;
    *block = 1;
    return ge::GRAPH_SUCCESS;
  }
  static uint32_t LaunchV2(uint32_t, void *, void **, int32_t, void **, int32_t, void *, void *) {
    return failing_ ? 1 : 0;
  }
  static int64_t LaunchLegacy(uint32_t, void *, void *, void *, void *, void *) {
    return failing_ ? 1 : 0;
  }

  bool legacy_;
  bool fail_launch_;
  static inline bool failing_ = false;
};

void ExpectTimingSamples(bool legacy) {
  TimingLoader loader(legacy);
  KernelModule module(&loader);
  const AbiSpec abi = legacy ? AbiSpec{"GetTilingDataSize", "AutofuseTiling", "AutofuseLaunch", 1, 1} : AbiSpec{};
  ASSERT_EQ(module.Load("timing-module.so", abi), Status::kOk);
  void *input = nullptr;
  void *output = nullptr;
  void *tiling_data = nullptr;
  module.EnableKernelTiming(false);
  ASSERT_EQ(legacy ? module.LaunchLegacy(1, nullptr, &input, 1, &output, 1, nullptr, tiling_data)
                   : module.LaunchV2(1, nullptr, nullptr, 0, nullptr, 0, nullptr, tiling_data),
            Status::kOk);
  EXPECT_TRUE(module.TakeKernelTimingSamples().empty());

  module.EnableKernelTiming(true);
  ASSERT_EQ(legacy ? module.LaunchLegacy(1, nullptr, &input, 1, &output, 1, nullptr, tiling_data)
                   : module.LaunchV2(1, nullptr, nullptr, 0, nullptr, 0, nullptr, tiling_data),
            Status::kOk);
  const auto samples = module.TakeKernelTimingSamples();
  ASSERT_EQ(samples.size(), 1U);
  EXPECT_GE(samples.front(), 0U);
  EXPECT_LT(samples.front(), 1000000U);
  EXPECT_TRUE(module.TakeKernelTimingSamples().empty());

  module.EnableKernelTiming(false);
  ASSERT_EQ(legacy ? module.LaunchLegacy(1, nullptr, &input, 1, &output, 1, nullptr, tiling_data)
                   : module.LaunchV2(1, nullptr, nullptr, 0, nullptr, 0, nullptr, tiling_data),
            Status::kOk);
  EXPECT_TRUE(module.TakeKernelTimingSamples().empty());
}

TEST(KernelModuleTimingTest, RecordsMicrosecondsForV2WhenEnabledOnly) {
  ExpectTimingSamples(false);
}

TEST(KernelModuleTimingTest, RecordsMicrosecondsForLegacyWhenEnabledOnly) {
  ExpectTimingSamples(true);
}

TEST(KernelModuleTimingTest, FailedLaunchDoesNotRecordTimingSample) {
  TimingLoader loader(false, true);
  KernelModule module(&loader);
  ASSERT_EQ(module.Load("timing-module.so", AbiSpec{}), Status::kOk);
  module.EnableKernelTiming(true);
  EXPECT_EQ(module.LaunchV2(1, nullptr, nullptr, 0, nullptr, 0, nullptr, nullptr), Status::kRuntimeError);
  EXPECT_TRUE(module.TakeKernelTimingSamples().empty());
}

TEST(KernelModuleTimingTest, InvalidLegacyCountDoesNotRecordTimingSample) {
  TimingLoader loader(true);
  KernelModule module(&loader);
  ASSERT_EQ(module.Load("timing-module.so", AbiSpec{"GetTilingDataSize", "AutofuseTiling", "AutofuseLaunch", 1, 1}),
            Status::kOk);
  module.EnableKernelTiming(true);
  EXPECT_EQ(module.LaunchLegacy(1, nullptr, nullptr, 2, nullptr, 1, nullptr, nullptr), Status::kInvalidArgument);
  EXPECT_TRUE(module.TakeKernelTimingSamples().empty());
}

}  // namespace
}  // namespace device_validation
