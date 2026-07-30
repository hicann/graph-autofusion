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
#ifdef IL_STATIC_SHAPE
extern "C" int64_t AutofuseTiling(AutofuseTilingData *, uint32_t *, uint32_t *, uint32_t, uint32_t);
#endif
#if IL_RANK == 2
#ifndef IL_STATIC_SHAPE
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, AutofuseTilingData *, uint32_t *,
                                  uint32_t *, uint32_t, uint32_t);
#endif
constexpr std::array<int32_t, 2> kInputShape = {IL_X_S0, IL_X_S1};
constexpr std::array<int32_t, 2> kIndexShape = {IL_INDEX_S0, IL_INDEX_S1};
#elif IL_RANK == 3
#ifndef IL_STATIC_SHAPE
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5,
                                  AutofuseTilingData *, uint32_t *, uint32_t *, uint32_t, uint32_t);
#endif
constexpr std::array<int32_t, 3> kInputShape = {IL_X_S0, IL_X_S1, IL_X_S2};
constexpr std::array<int32_t, 3> kIndexShape = {IL_INDEX_S0, IL_INDEX_S1, IL_INDEX_S2};
#elif IL_RANK == 4
#ifndef IL_STATIC_SHAPE
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5,
                                  uint32_t s6, uint32_t s7, AutofuseTilingData *, uint32_t *, uint32_t *, uint32_t,
                                  uint32_t);
#endif
constexpr std::array<int32_t, 4> kInputShape = {IL_X_S0, IL_X_S1, IL_X_S2, IL_X_S3};
constexpr std::array<int32_t, 4> kIndexShape = {IL_INDEX_S0, IL_INDEX_S1, IL_INDEX_S2, IL_INDEX_S3};
#endif

namespace {
constexpr int32_t kAxis = IL_AXIS < 0 ? IL_AXIS + IL_RANK : IL_AXIS;
#if defined(IL_DATA_BF16)
using DataType = bfloat16_t;
#elif defined(IL_DATA_UINT32)
using DataType = uint32_t;
#elif defined(IL_DATA_FLOAT)
using DataType = float;
#else
using DataType = half;
#endif
#ifdef IL_INDEX_INT64
using IndexType = int64_t;
#else
using IndexType = int32_t;
#endif

float BesselK0Reference(float x) {
  const float y = x * x / 4.0F;
  float series_term = 1.0F;
  float series_sum = 1.0F;
  for (int32_t order = 1; order <= 20; ++order) {
    const float scale = x * 0.5F / static_cast<float>(order);
    series_term *= scale * scale;
    series_sum += series_term;
  }
  const float poly =
      -0.57721566F +
      y * (0.42278420F +
           y * (0.23069756F + y * (0.03488590F + y * (0.00262698F + y * (0.00010750F + y * 0.00000740F)))));
  return -std::log(x * 0.5F) * series_sum + poly;
}

void RunTiling(AutofuseTilingData &tiling, uint32_t &workspace_size, uint32_t &block_dim) {
#ifdef IL_STATIC_SHAPE
  AutofuseTiling(&tiling, &workspace_size, &block_dim, 48, 192 * 1024);
#elif IL_RANK == 2
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

void InitializeData(DataType *x, IndexType *index, DataType *expected, int32_t input_count, int32_t output_count) {
  std::array<int32_t, IL_RANK> input_strides{};
  input_strides.back() = 1;
  for (size_t i = input_strides.size() - 1UL; i > 0UL; --i) {
    input_strides[i - 1UL] = kInputShape[i] * input_strides[i];
  }
  for (int32_t i = 0; i < input_count; ++i) {
#if defined(IL_DATA_UINT32)
    x[i] = static_cast<DataType>(i % 5);
#elif defined(IL_OUTPUT_POST_TYPE) && IL_OUTPUT_POST_TYPE == 1
    x[i] = static_cast<DataType>(static_cast<float>((i % 8) + 1) * 0.25F);
#else
    x[i] = static_cast<DataType>(static_cast<float>((i % 29) - 14) * 0.25F);
#endif
  }
  for (int32_t i = 0; i < output_count; ++i) {
    const int32_t index_value = (i * 3 + 1) % kInputShape[kAxis];
    index[i] = static_cast<IndexType>(i % 2 == 0 ? index_value : -index_value);
    int32_t remaining = i;
    int32_t input_offset = 0;
    for (size_t dim = kIndexShape.size(); dim-- > 0UL;) {
      const int32_t coord = remaining % kIndexShape[dim];
      remaining /= kIndexShape[dim];
      input_offset += (static_cast<int32_t>(dim) == kAxis ? index_value : coord) * input_strides[dim];
    }
    float value = static_cast<float>(x[input_offset]);
#if defined(IL_OUTPUT_POST_TYPE) && IL_OUTPUT_POST_TYPE == 1
    expected[i] = static_cast<DataType>(BesselK0Reference(value));
#elif defined(IL_OUTPUT_POST_TYPE) && IL_OUTPUT_POST_TYPE == 2
    expected[i] = static_cast<DataType>(value);
#elif defined(IL_OUTPUT_POST_TYPE) && IL_OUTPUT_POST_TYPE == 3
    expected[i] = static_cast<DataType>(std::exp2(value));
#elif IL_INPUT_PRE_TYPE == 5
    expected[i] = static_cast<DataType>(IL_USE_EXP2 ? std::exp2(value) : std::exp(value));
#else
#if IL_INPUT_PRE_TYPE == 1 || IL_INPUT_PRE_TYPE == 4
    value = std::max(value, 0.0F);
#elif IL_INPUT_PRE_TYPE == 3
    value = std::exp2(std::max(value, 0.0F));
#endif
#if IL_INPUT_PRE_TYPE == 2
    expected[i] = static_cast<DataType>(-std::exp2(value));
#elif IL_INPUT_PRE_TYPE == 3
    expected[i] = static_cast<DataType>(-value);
#elif IL_USE_EXP2
    expected[i] = static_cast<DataType>(-std::exp2(value));
#else
    expected[i] = static_cast<DataType>(-std::exp(value));
#endif
#endif
  }
}
}  // namespace

TEST(E2EIndirectLoadStore, GeneratedKernelMatchesReference) {
  const int32_t input_count = ElementCount(kInputShape);
  const int32_t output_count = ElementCount(kIndexShape);
  const auto gm_free = [](void *ptr) { AscendC::GmFree(ptr); };
  std::unique_ptr<DataType, decltype(gm_free)> x(
      reinterpret_cast<DataType *>(AscendC::GmAlloc(input_count * sizeof(DataType))), gm_free);
  std::unique_ptr<IndexType, decltype(gm_free)> index(
      reinterpret_cast<IndexType *>(AscendC::GmAlloc(output_count * sizeof(IndexType))), gm_free);
  std::unique_ptr<DataType, decltype(gm_free)> output(
      reinterpret_cast<DataType *>(AscendC::GmAlloc(output_count * sizeof(DataType))), gm_free);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(index, nullptr);
  ASSERT_NE(output, nullptr);
  std::vector<DataType> expected(static_cast<size_t>(output_count));
  InitializeData(x.get(), index.get(), expected.data(), input_count, output_count);
  std::fill_n(output.get(), output_count, static_cast<DataType>(0.0F));

  AutofuseTilingData tiling_data{};
  uint32_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  RunTiling(tiling_data, workspace_size, block_dim);
#if !defined(IL_EXPECT_SIMT) || IL_EXPECT_SIMT
  ASSERT_EQ(tiling_data.graph0_tiling_key, IL_TILING_KEY);
#endif
  ASSERT_GT(tiling_data.block_dim, 0U);
#ifdef IL_EXPECT_SIMT_MULTI_ROUND
  ASSERT_GT((static_cast<uint32_t>(output_count) + tiling_data.block_dim - 1U) / tiling_data.block_dim, 1024U);
#endif

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(indirect_load_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(x.get()),
              reinterpret_cast<uint8_t *>(index.get()), reinterpret_cast<uint8_t *>(output.get()), nullptr,
              reinterpret_cast<uint8_t *>(&tiling_data));
  for (int32_t i = 0; i < output_count; ++i) {
#if defined(IL_DATA_BF16) || defined(IL_DATA_UINT32)
    EXPECT_EQ(static_cast<float>(output.get()[i]), static_cast<float>(expected[static_cast<size_t>(i)]))
        << "offset=" << i;
#else
    EXPECT_NEAR(static_cast<float>(output.get()[i]), static_cast<float>(expected[static_cast<size_t>(i)]), 0.0625F)
        << "offset=" << i;
#endif
  }
}
