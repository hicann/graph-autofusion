/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"

extern "C" __global__ __aicore__ void indirect_load_broadcast_index_where_simt_test(GM_ADDR input0, GM_ADDR input1,
                                                                                    GM_ADDR input2, GM_ADDR output,
                                                                                    GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(AutofuseTilingData *, uint32_t *, uint32_t *, uint32_t, uint32_t);

namespace {
constexpr int32_t kRows = 6400;
constexpr int32_t kColumns = 32;
constexpr int32_t kTableRows = 315511;

TEST(E2EIndirectLoadBroadcastWhere, GeneratedKernelMatchesReference) {
  const int64_t table_count = static_cast<int64_t>(kTableRows) * kColumns;
  const int64_t output_count = static_cast<int64_t>(kRows) * kColumns;
  auto *index0 = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * kRows));
  auto *table = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * table_count));
  auto *index2 = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * kRows));
  auto *output = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * output_count));
  ASSERT_NE(index0, nullptr);
  ASSERT_NE(table, nullptr);
  ASSERT_NE(index2, nullptr);
  ASSERT_NE(output, nullptr);

  for (int32_t row = 0; row < kTableRows; ++row) {
    for (int32_t column = 0; column < kColumns; ++column) {
      table[static_cast<int64_t>(row) * kColumns + column] =
          static_cast<float>((row % 97) * 0.25F + (column % 31) * 0.03125F);
    }
  }
  for (int32_t row = 0; row < kRows; ++row) {
    index0[row] = row % 2 == 0 ? -1 : static_cast<int64_t>((row * 17 + 3) % kTableRows);
    index2[row] = static_cast<int64_t>((row * 29 + 7) % kTableRows);
  }

  AutofuseTilingData tiling_data{};
  uint32_t workspace_size = 0;
  uint32_t block_dim = 48;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48U, 192U * 1024U), 0);
  void *workspace = workspace_size == 0U ? nullptr : AscendC::GmAlloc(workspace_size);
  ASSERT_TRUE(workspace_size == 0U || workspace != nullptr);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(indirect_load_broadcast_index_where_simt_test, block_dim, reinterpret_cast<uint8_t *>(index0),
              reinterpret_cast<uint8_t *>(table), reinterpret_cast<uint8_t *>(index2),
              reinterpret_cast<uint8_t *>(output), reinterpret_cast<uint8_t *>(workspace),
              reinterpret_cast<uint8_t *>(&tiling_data));

  for (int32_t row = 0; row < kRows; ++row) {
    const int64_t selected = index0[row] == -1 ? index2[row] : index0[row];
    for (int32_t column = 0; column < kColumns; ++column) {
      const int64_t offset = static_cast<int64_t>(row) * kColumns + column;
      const float expected = table[selected * kColumns + column];
      EXPECT_FLOAT_EQ(output[offset], expected) << "row=" << row << ", column=" << column;
    }
  }

  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
  AscendC::GmFree(index0);
  AscendC::GmFree(table);
  AscendC::GmFree(index2);
  AscendC::GmFree(output);
}
}  // namespace
