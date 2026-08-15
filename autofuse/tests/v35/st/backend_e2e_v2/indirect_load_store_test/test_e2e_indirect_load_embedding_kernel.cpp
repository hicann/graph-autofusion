/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>

#include "indirect_load_kernel_test_common.h"

extern "C" __global__ __aicore__ void indirect_load_embedding_test(GM_ADDR input, GM_ADDR index, GM_ADDR output,
                                                                   GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr int32_t kInputRows = 100;
constexpr int32_t kEmbeddingSize = 16;
constexpr int32_t kIndexRows = 92;

void InitializeData(float *input, int64_t *index, float *expected) {
  for (int32_t row = 0; row < kInputRows; ++row) {
    for (int32_t col = 0; col < kEmbeddingSize; ++col) {
      input[row * kEmbeddingSize + col] = static_cast<float>(row * 0.25F + col * 0.03125F);
    }
  }
  for (int32_t row = 0; row < kIndexRows; ++row) {
    index[row] = static_cast<int64_t>((row * 7 + 3) % kInputRows);
    for (int32_t col = 0; col < kEmbeddingSize; ++col) {
      expected[row * kEmbeddingSize + col] = input[index[row] * kEmbeddingSize + col];
    }
  }
}
}  // namespace

TEST(E2EIndirectLoadEmbedding, GeneratedKernelMatchesReference) {
  constexpr int64_t input_count = static_cast<int64_t>(kInputRows) * kEmbeddingSize;
  constexpr int64_t index_count = kIndexRows;
  constexpr int64_t output_count = static_cast<int64_t>(kIndexRows) * kEmbeddingSize;
  indirect_load_test::KernelData<float, int64_t> buffers(input_count, index_count, output_count);
  ASSERT_TRUE(buffers.IsValid());
  InitializeData(buffers.input.get(), buffers.index.get(), buffers.expected.data());
  std::fill_n(buffers.output.get(), output_count, 0.0F);

  indirect_load_test::KernelTiling tiling;
  ASSERT_TRUE(tiling.IsValid());
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(indirect_load_embedding_test, tiling.data.block_dim, reinterpret_cast<uint8_t *>(buffers.input.get()),
              reinterpret_cast<uint8_t *>(buffers.index.get()), reinterpret_cast<uint8_t *>(buffers.output.get()),
              tiling.workspace.get(), reinterpret_cast<uint8_t *>(&tiling.data));

  for (int64_t i = 0; i < output_count; ++i) {
    EXPECT_FLOAT_EQ(buffers.output.get()[i], buffers.expected[static_cast<size_t>(i)]) << "offset=" << i;
  }
}
