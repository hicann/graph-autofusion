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
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"

extern "C" __global__ __aicore__ void indirect_load_torch_gather_strided_test(GM_ADDR data, GM_ADDR index,
                                                                              GM_ADDR output, GM_ADDR workspace,
                                                                              GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(AutofuseTilingData *, uint32_t *, uint32_t *, uint32_t, uint32_t);

namespace {
constexpr int32_t kInputStride0 = IL_INPUT_STRIDE0;
constexpr int32_t kInputStride1 = IL_INPUT_STRIDE1;
constexpr int32_t kInputStride2 = IL_INPUT_STRIDE2;
constexpr int32_t kIndexStride0 = IL_INDEX_STRIDE0;
constexpr int32_t kIndexStride1 = IL_INDEX_STRIDE1;
constexpr int32_t kIndexStride2 = IL_INDEX_STRIDE2;
#ifdef IL_INDEX_SELECT_CASE
constexpr int32_t kEffectiveInputStride0 = 138;
constexpr int32_t kEffectiveInputStride1 = 23;
constexpr int32_t kEffectiveInputStride2 = 1;
#else
constexpr int32_t kEffectiveInputStride0 = kInputStride0;
constexpr int32_t kEffectiveInputStride1 = kInputStride1;
constexpr int32_t kEffectiveInputStride2 = kInputStride2;
#endif
#ifdef IL_INDEX_SELECT_CASE
constexpr int32_t kDim0 = 30;
constexpr int32_t kInputDim1 = 6;
constexpr int32_t kOutputDim1 = 3;
constexpr int32_t kDim2 = 23;
#else
constexpr int32_t kDim0 = 8;
constexpr int32_t kInputDim1 = 32;
constexpr int32_t kOutputDim1 = 16;
constexpr int32_t kDim2 = 5;
#endif
#ifdef IL_INDEX_SELECT_CASE
constexpr int32_t kInputStorageSize = kDim0 * kInputDim1 * kDim2;
constexpr int32_t kIndexStorageSize = kOutputDim1;
constexpr int32_t kOutputSize = kDim0 * kOutputDim1 * kDim2;
#else
constexpr int32_t kInputStorageSize = (8 - 1) * kInputStride0 + (32 - 1) * kInputStride1 + (5 - 1) * kInputStride2 + 1;
constexpr int32_t kIndexStorageSize = (8 - 1) * kIndexStride0 + (16 - 1) * kIndexStride1 + (5 - 1) * kIndexStride2 + 1;
constexpr int32_t kOutputSize = 8 * 16 * 5;
#endif

void Initialize(float *data, int64_t *index, std::vector<float> &expected) {
  for (int32_t i = 0; i < kInputStorageSize; ++i) {
    data[i] = 1.0F + static_cast<float>((i * 37) % 997) / 997.0F;
  }
  std::fill_n(index, kIndexStorageSize, int64_t{0});
  for (int32_t a0 = 0; a0 < kDim0; ++a0) {
    for (int32_t a1 = 0; a1 < kOutputDim1; ++a1) {
      for (int32_t a2 = 0; a2 < kDim2; ++a2) {
        const int32_t output_offset = kOutputDim1 * kDim2 * a0 + kDim2 * a1 + a2;
#ifdef IL_INDEX_SELECT_CASE
        const int32_t index_offset = a1;
        const int64_t index_value = (a1 * 2 + 1) % kInputDim1;
#else
        const int32_t index_offset = kIndexStride0 * a0 + kIndexStride1 * a1 + kIndexStride2 * a2;
        const int64_t index_value = (a0 * 17 + a1 * 7 + a2 * 3) % 32;
#endif
        index[index_offset] = index_value;
        expected[static_cast<size_t>(output_offset)] =
            data[kEffectiveInputStride0 * a0 + kEffectiveInputStride1 * index_value + kEffectiveInputStride2 * a2];
      }
    }
  }
}
}  // namespace

TEST(E2EIndirectLoadTorchGatherStrided, GeneratedKernelMatchesReference) {
  const auto gm_free = [](void *ptr) { AscendC::GmFree(ptr); };
  std::unique_ptr<float, decltype(gm_free)> data(
      reinterpret_cast<float *>(AscendC::GmAlloc(kInputStorageSize * sizeof(float))), gm_free);
  std::unique_ptr<int64_t, decltype(gm_free)> index(
      reinterpret_cast<int64_t *>(AscendC::GmAlloc(kIndexStorageSize * sizeof(int64_t))), gm_free);
  std::unique_ptr<float, decltype(gm_free)> output(
      reinterpret_cast<float *>(AscendC::GmAlloc(kOutputSize * sizeof(float))), gm_free);
  ASSERT_NE(data, nullptr);
  ASSERT_NE(index, nullptr);
  ASSERT_NE(output, nullptr);

  std::vector<float> expected(kOutputSize);
  Initialize(data.get(), index.get(), expected);
  std::fill_n(output.get(), kOutputSize, 0.0F);

  AutofuseTilingData tiling_data{};
  uint32_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48U, 192U * 1024U), 0);
#if IL_EXPECT_SIMT
  ASSERT_EQ(tiling_data.get_tiling_key(), static_cast<uint32_t>(IL_EXPECT_TILING_KEY));
#else
  ASSERT_EQ(tiling_data.graph0_tiling_key, static_cast<uint32_t>(IL_EXPECT_TILING_KEY));
#endif
  ASSERT_GT(tiling_data.block_dim, 0U);
  std::unique_ptr<uint8_t, decltype(gm_free)> workspace(nullptr, gm_free);
  if (workspace_size > 0U) {
    workspace.reset(reinterpret_cast<uint8_t *>(AscendC::GmAlloc(workspace_size)));
    ASSERT_NE(workspace, nullptr);
    std::fill_n(workspace.get(), workspace_size, uint8_t{0});
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(indirect_load_torch_gather_strided_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(data.get()),
              reinterpret_cast<uint8_t *>(index.get()), reinterpret_cast<uint8_t *>(output.get()), workspace.get(),
              reinterpret_cast<uint8_t *>(&tiling_data));
  int32_t mismatch_count = 0;
  int32_t first_mismatch = -1;
  for (int32_t i = 0; i < kOutputSize; ++i) {
    if (output.get()[i] != expected[static_cast<size_t>(i)]) {
      if (first_mismatch < 0) {
        first_mismatch = i;
      }
      ++mismatch_count;
    }
  }
  if (mismatch_count > 0) {
    ADD_FAILURE() << "mismatch count=" << mismatch_count << ", first mismatch at offset=" << first_mismatch
                  << ", actual=" << output.get()[first_mismatch]
                  << ", expected=" << expected[static_cast<size_t>(first_mismatch)];
  }
}
