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
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"

extern "C" __global__ __aicore__ void indirect_load_store_test(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace,
                                                               GM_ADDR tiling);
#if IL_RANK == 2
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, AutofuseTilingData *, uint32_t *,
                                  uint32_t *, uint32_t, uint32_t);
constexpr std::array<int32_t, 2> kInputShape = {IL_X_S0, IL_X_S1};
constexpr std::array<int32_t, 2> kIndexShape = {IL_INDEX_S0, IL_INDEX_S1};
#elif IL_RANK == 3
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5,
                                  AutofuseTilingData *, uint32_t *, uint32_t *, uint32_t, uint32_t);
constexpr std::array<int32_t, 3> kInputShape = {IL_X_S0, IL_X_S1, IL_X_S2};
constexpr std::array<int32_t, 3> kIndexShape = {IL_INDEX_S0, IL_INDEX_S1, IL_INDEX_S2};
#elif IL_RANK == 4
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5,
                                  uint32_t s6, uint32_t s7, AutofuseTilingData *, uint32_t *, uint32_t *, uint32_t,
                                  uint32_t);
constexpr std::array<int32_t, 4> kInputShape = {IL_X_S0, IL_X_S1, IL_X_S2, IL_X_S3};
constexpr std::array<int32_t, 4> kIndexShape = {IL_INDEX_S0, IL_INDEX_S1, IL_INDEX_S2, IL_INDEX_S3};
#endif

namespace {
constexpr int32_t kAxis = IL_AXIS < 0 ? IL_AXIS + IL_RANK : IL_AXIS;

void RunTiling(AutofuseTilingData &tiling, uint32_t &workspace_size, uint32_t &block_dim) {
#if IL_RANK == 2
  AutofuseTiling(kInputShape[0], kInputShape[1], kIndexShape[0], kIndexShape[1], &tiling, &workspace_size, &block_dim,
                 48, 192 * 1024);
#elif IL_RANK == 3
  AutofuseTiling(kInputShape[0], kInputShape[1], kInputShape[2], kIndexShape[0], kIndexShape[1], kIndexShape[2],
                 &tiling, &workspace_size, &block_dim, 48, 192 * 1024);
#elif IL_RANK == 4
  AutofuseTiling(kInputShape[0], kInputShape[1], kInputShape[2], kInputShape[3], kIndexShape[0], kIndexShape[1],
                 kIndexShape[2], kIndexShape[3], &tiling, &workspace_size, &block_dim, 48, 192 * 1024);
#endif
}

template <size_t N>
int32_t ElementCount(const std::array<int32_t, N> &shape) {
  int32_t count = 1;
  for (int32_t dim : shape) {
    count *= dim;
  }
  return count;
}

void InitializeData(half *x, int32_t *index, half *expected, int32_t input_count, int32_t output_count) {
  std::array<int32_t, IL_RANK> input_strides{};
  input_strides.back() = 1;
  for (size_t i = input_strides.size() - 1UL; i > 0UL; --i) {
    input_strides[i - 1UL] = kInputShape[i] * input_strides[i];
  }
  for (int32_t i = 0; i < input_count; ++i) {
    x[i] = static_cast<half>(static_cast<float>((i % 29) - 14) * 0.25F);
  }
  for (int32_t i = 0; i < output_count; ++i) {
    const int32_t index_value = (i * 3 + 1) % kInputShape[kAxis];
    index[i] = i % 2 == 0 ? index_value : -index_value;
    int32_t remaining = i;
    int32_t input_offset = 0;
    for (size_t dim = kIndexShape.size(); dim-- > 0UL;) {
      const int32_t coord = remaining % kIndexShape[dim];
      remaining /= kIndexShape[dim];
      input_offset += (static_cast<int32_t>(dim) == kAxis ? index_value : coord) * input_strides[dim];
    }
    float value = static_cast<float>(x[input_offset]);
#if IL_HAS_INPUT_PRE
    value = std::max(value, 0.0F);
#endif
#if IL_USE_EXP2
    expected[i] = static_cast<half>(-std::exp2(value));
#else
    expected[i] = static_cast<half>(-std::exp(value));
#endif
  }
}
}  // namespace

TEST(E2EIndirectLoadStore, GeneratedKernelMatchesReference) {
  const int32_t input_count = ElementCount(kInputShape);
  const int32_t output_count = ElementCount(kIndexShape);
  const auto gm_free = [](void *ptr) { AscendC::GmFree(ptr); };
  std::unique_ptr<half, decltype(gm_free)> x(reinterpret_cast<half *>(AscendC::GmAlloc(input_count * sizeof(half))),
                                             gm_free);
  std::unique_ptr<int32_t, decltype(gm_free)> index(
      reinterpret_cast<int32_t *>(AscendC::GmAlloc(output_count * sizeof(int32_t))), gm_free);
  std::unique_ptr<half, decltype(gm_free)> output(
      reinterpret_cast<half *>(AscendC::GmAlloc(output_count * sizeof(half))), gm_free);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(index, nullptr);
  ASSERT_NE(output, nullptr);
  std::vector<half> expected(static_cast<size_t>(output_count));
  InitializeData(x.get(), index.get(), expected.data(), input_count, output_count);
  std::fill_n(output.get(), output_count, static_cast<half>(0.0F));

  AutofuseTilingData tiling_data{};
  uint32_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  RunTiling(tiling_data, workspace_size, block_dim);
  ASSERT_EQ(tiling_data.graph0_tiling_key, IL_TILING_KEY);
  ASSERT_GT(tiling_data.block_dim, 0U);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(indirect_load_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(x.get()),
              reinterpret_cast<uint8_t *>(index.get()), reinterpret_cast<uint8_t *>(output.get()), nullptr,
              reinterpret_cast<uint8_t *>(&tiling_data));
  for (int32_t i = 0; i < output_count; ++i) {
    EXPECT_NEAR(static_cast<float>(output.get()[i]), static_cast<float>(expected[static_cast<size_t>(i)]), 0.0625F)
        << "offset=" << i;
  }
}
