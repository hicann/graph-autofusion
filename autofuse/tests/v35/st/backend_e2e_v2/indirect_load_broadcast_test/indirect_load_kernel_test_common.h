/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_INDIRECT_LOAD_BROADCAST_TEST_INDIRECT_LOAD_KERNEL_TEST_COMMON_H_
#define AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_INDIRECT_LOAD_BROADCAST_TEST_INDIRECT_LOAD_KERNEL_TEST_COMMON_H_

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"

extern "C" int64_t AutofuseTiling(AutofuseTilingData *, uint32_t *, uint32_t *, uint32_t, uint32_t);

namespace indirect_load_test {
inline void GmFree(void *ptr) {
  AscendC::GmFree(ptr);
}

template <typename DataType, typename IndexType>
struct KernelData {
  KernelData(int64_t input_count, int64_t index_count, int64_t output_count)
      : input(reinterpret_cast<DataType *>(AscendC::GmAlloc(input_count * sizeof(DataType))), GmFree),
        index(reinterpret_cast<IndexType *>(AscendC::GmAlloc(index_count * sizeof(IndexType))), GmFree),
        output(reinterpret_cast<DataType *>(AscendC::GmAlloc(output_count * sizeof(DataType))), GmFree),
        expected(static_cast<size_t>(output_count)) {}

  [[nodiscard]] bool IsValid() const {
    return input != nullptr && index != nullptr && output != nullptr;
  }

  std::unique_ptr<DataType, decltype(&GmFree)> input;
  std::unique_ptr<IndexType, decltype(&GmFree)> index;
  std::unique_ptr<DataType, decltype(&GmFree)> output;
  std::vector<DataType> expected;
};

struct KernelTiling {
  KernelTiling() : workspace(nullptr, GmFree) {
    EXPECT_EQ(AutofuseTiling(&data, &workspace_size, &block_dim, 48U, 192U * 1024U), 0);
    EXPECT_GT(data.block_dim, 0U);
    if (workspace_size != 0U) {
      workspace.reset(reinterpret_cast<uint8_t *>(AscendC::GmAlloc(workspace_size)));
    }
  }

  [[nodiscard]] bool IsValid() const {
    return workspace_size == 0U || workspace != nullptr;
  }

  AutofuseTilingData data{};
  uint32_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  std::unique_ptr<uint8_t, decltype(&GmFree)> workspace;
};
}  // namespace indirect_load_test

#endif  // AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_INDIRECT_LOAD_BROADCAST_TEST_INDIRECT_LOAD_KERNEL_TEST_COMMON_H_
