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
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"

#if defined(IL_MIXED_ELEMENTWISE) && IL_EXPECT_SIMT
extern "C" __global__ __aicore__ void indirect_load_mixed_elementwise_test(GM_ADDR x, GM_ADDR index0, GM_ADDR index1,
                                                                           GM_ADDR addend, GM_ADDR sign, GM_ADDR scale,
                                                                           GM_ADDR output, GM_ADDR workspace,
                                                                           GM_ADDR tiling);
#elif defined(IL_MIXED_ELEMENTWISE)
extern "C" __global__ __aicore__ void indirect_load_mixed_elementwise_test(GM_ADDR x0, GM_ADDR x1, GM_ADDR index0,
                                                                           GM_ADDR index1, GM_ADDR addend, GM_ADDR sign,
                                                                           GM_ADDR scale, GM_ADDR output,
                                                                           GM_ADDR workspace, GM_ADDR tiling);
#elif defined(IL_POST_REDUCE_ADD)
extern "C" __global__ __aicore__ void indirect_load_store_test(GM_ADDR x, GM_ADDR index, GM_ADDR addend, GM_ADDR y,
                                                               GM_ADDR workspace, GM_ADDR tiling);
#else
extern "C" __global__ __aicore__ void indirect_load_store_test(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace,
                                                               GM_ADDR tiling);
#endif
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
#elif defined(IL_DATA_INT16)
using DataType = int16_t;
#elif defined(IL_DATA_UINT32)
using DataType = uint32_t;
#elif defined(IL_DATA_FLOAT)
using DataType = float;
#else
using DataType = half;
#endif
#ifdef IL_OUTPUT_FLOAT
using OutputType = float;
#else
using OutputType = DataType;
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
constexpr int32_t ElementCount(const std::array<int32_t, N> &shape) {
  int32_t count = 1;
  for (int32_t dim : shape) {
    count *= dim;
  }
  return count;
}

constexpr int32_t ResultCount() {
#ifdef IL_POST_REDUCE
#ifdef IL_REDUCE_LAST_AXIS
  return kIndexShape[0] * kIndexShape[1] * kIndexShape[2];
#elif defined(IL_REDUCE_WITH_A)
  return kIndexShape[0] * kIndexShape[1] * kIndexShape[3];
#else
  return kIndexShape[0] * kIndexShape[1];
#endif
#else
  return ElementCount(kIndexShape);
#endif
}

void InitializeData(DataType *x, IndexType *index, DataType *addend, OutputType *expected, int32_t input_count,
                    int32_t output_count) {
#ifdef IL_RANDOM_INPUT_INDEX
  std::mt19937 generator(20260805U);
  std::uniform_real_distribution<float> input_distribution(1.0F, 2.0F);
  std::uniform_int_distribution<int64_t> index_distribution(1L, 9L);
#endif
  std::array<int32_t, IL_RANK> input_strides{};
  input_strides.back() = 1;
  for (size_t i = input_strides.size() - 1UL; i > 0UL; --i) {
    input_strides[i - 1UL] = kInputShape[i] * input_strides[i];
  }
  for (int32_t i = 0; i < input_count; ++i) {
#if defined(IL_DATA_UINT32)
    x[i] = static_cast<DataType>(i % 5);
#elif defined(IL_POST_REDUCE_SIMD)
    x[i] = static_cast<DataType>(static_cast<float>((i % 8) + 1) * 0.25F);
#elif defined(IL_OUTPUT_POST_TYPE) && IL_OUTPUT_POST_TYPE == 1
    x[i] = static_cast<DataType>(static_cast<float>((i % 8) + 1) * 0.25F);
#elif defined(IL_RANDOM_INPUT_INDEX)
    x[i] = static_cast<DataType>(input_distribution(generator));
#else
    x[i] = static_cast<DataType>(static_cast<float>((i % 29) - 14) * 0.25F);
#endif
  }
  for (int32_t i = 0; i < output_count; ++i) {
#ifdef IL_POST_REDUCE_ADD
    addend[i] = static_cast<DataType>(static_cast<float>((i % 7) - 3) * 0.125F);
#else
    (void)addend;
#endif
    const int32_t index_value =
#ifdef IL_RANDOM_INPUT_INDEX
        static_cast<int32_t>(index_distribution(generator));
#else
        (i * 3 + 1) % kInputShape[kAxis];
#endif
#ifdef IL_RANDOM_INPUT_INDEX
    index[i] = static_cast<IndexType>(index_value);
#else
    index[i] = static_cast<IndexType>(i % 2 == 0 ? index_value : -index_value);
#endif
    int32_t remaining = i;
    int32_t input_offset = 0;
    for (size_t dim = kIndexShape.size(); dim-- > 0UL;) {
      const int32_t coord = remaining % kIndexShape[dim];
      remaining /= kIndexShape[dim];
      input_offset += (static_cast<int32_t>(dim) == kAxis ? index_value : coord) * input_strides[dim];
    }
    float value = static_cast<float>(x[input_offset]);
#ifdef IL_POST_REDUCE
#if IL_INPUT_PRE_TYPE == 3
    value = std::exp2(std::max(value, 0.0F));
#endif
#ifdef IL_POST_REDUCE_SIMD
    value = BesselK0Reference(value);
#endif
#ifdef IL_POST_REDUCE_ABS
    value = std::abs(value);
#endif
#ifdef IL_POST_REDUCE_EXP2
    value = std::exp2(value);
#endif
#ifdef IL_POST_REDUCE_ADD
    value += static_cast<float>(addend[i]);
#endif
#ifdef IL_REDUCE_LAST_AXIS
    const int32_t result_index = i / kIndexShape[3];
#elif defined(IL_REDUCE_WITH_A)
    const int32_t result_index = i / (kIndexShape[2] * kIndexShape[3]) * kIndexShape[3] + i % kIndexShape[3];
#else
    const int32_t result_index = i / (kIndexShape[2] * kIndexShape[3]);
#endif
    expected[result_index] = static_cast<OutputType>(static_cast<float>(expected[result_index]) + value);
#elif defined(IL_OUTPUT_POST_TYPE) && IL_OUTPUT_POST_TYPE == 1
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

#ifdef IL_MIXED_ELEMENTWISE
template <typename T>
std::unique_ptr<T, void (*)(void *)> Allocate(size_t count) {
  return {reinterpret_cast<T *>(AscendC::GmAlloc(count * sizeof(T))), AscendC::GmFree};
}

std::array<int32_t, IL_RANK> InputStrides() {
  std::array<int32_t, IL_RANK> strides{};
  strides.back() = 1;
  for (size_t i = strides.size() - 1UL; i > 0UL; --i) {
    strides[i - 1UL] = kInputShape[i] * strides[i];
  }
  return strides;
}

int32_t GatherInputOffset(int32_t output_index, int32_t gather_index,
                          const std::array<int32_t, IL_RANK> &input_strides) {
  int32_t remaining = output_index;
  int32_t input_offset = 0;
  for (size_t dim = kIndexShape.size(); dim-- > 0UL;) {
    const int32_t coord = remaining % kIndexShape[dim];
    remaining /= kIndexShape[dim];
    input_offset += (static_cast<int32_t>(dim) == kAxis ? gather_index : coord) * input_strides[dim];
  }
  return input_offset;
}

bool IsGatherShapeValid() {
  if (kInputShape == kIndexShape) {
    return false;
  }
  for (size_t dim = 0UL; dim < kInputShape.size(); ++dim) {
    if (static_cast<int32_t>(dim) != kAxis && kInputShape[dim] < kIndexShape[dim]) {
      return false;
    }
  }
  return true;
}

void RunMixedElementwiseCase() {
  ASSERT_TRUE(IsGatherShapeValid());
  const int32_t input_count = ElementCount(kInputShape);
  const int32_t output_count = ElementCount(kIndexShape);
  const auto input_strides = InputStrides();
  auto x0 = Allocate<DataType>(input_count);
#if !IL_EXPECT_SIMT
  auto x1 = Allocate<DataType>(input_count);
#endif
  auto index0 = Allocate<IndexType>(output_count);
  auto index1 = Allocate<IndexType>(output_count);
  auto addend = Allocate<DataType>(output_count);
  auto sign = Allocate<DataType>(output_count);
  auto scale = Allocate<DataType>(output_count);
  auto output = Allocate<DataType>(output_count);
  ASSERT_NE(x0, nullptr);
#if !IL_EXPECT_SIMT
  ASSERT_NE(x1, nullptr);
#endif
  ASSERT_NE(index0, nullptr);
  ASSERT_NE(index1, nullptr);
  ASSERT_NE(addend, nullptr);
  ASSERT_NE(sign, nullptr);
  ASSERT_NE(scale, nullptr);
  ASSERT_NE(output, nullptr);
  for (int32_t i = 0; i < input_count; ++i) {
    x0.get()[i] = static_cast<DataType>(static_cast<float>((i % 13) + 1) * 0.25F);
#if !IL_EXPECT_SIMT
    x1.get()[i] = static_cast<DataType>(i % 3 == 0 ? -1.0F : 1.0F);
#endif
  }
  std::vector<DataType> expected(static_cast<size_t>(output_count));
  for (int32_t i = 0; i < output_count; ++i) {
    const int32_t gather_index = (i * 3 + 1) % kInputShape[kAxis];
    index0.get()[i] = static_cast<IndexType>(gather_index / 2);
    index1.get()[i] = static_cast<IndexType>(gather_index - static_cast<int32_t>(index0.get()[i]));
    addend.get()[i] = static_cast<DataType>(static_cast<float>((i % 5) - 2) * 0.125F);
    sign.get()[i] = static_cast<DataType>(i % 3 == 0 ? -1.0F : 1.0F);
    scale.get()[i] = static_cast<DataType>(static_cast<float>((i % 4) + 1) * 0.25F);
#ifdef IL_SIMT_ELEMENTWISE_COVERAGE
    const int32_t effective_index =
        std::max(static_cast<int32_t>(index0.get()[i]), static_cast<int32_t>(index1.get()[i]));
#else
    const int32_t effective_index = gather_index;
#endif
    const int32_t input_offset = GatherInputOffset(i, effective_index, input_strides);
#if IL_EXPECT_SIMT
    const float gathered = static_cast<float>(x0.get()[input_offset]);
#else
    const float gathered =
        std::copysign(static_cast<float>(x0.get()[input_offset]), static_cast<float>(x1.get()[input_offset]));
#endif
#ifdef IL_SIMT_ELEMENTWISE_COVERAGE
    const float post =
        std::copysign((gathered - static_cast<float>(addend.get()[i])) / static_cast<float>(scale.get()[i]),
                      static_cast<float>(sign.get()[i]));
    expected[static_cast<size_t>(i)] = static_cast<DataType>(post);
#else
    const float post = std::copysign(gathered + static_cast<float>(addend.get()[i]), static_cast<float>(sign.get()[i]));
    expected[static_cast<size_t>(i)] = static_cast<DataType>(post * static_cast<float>(scale.get()[i]));
#endif
    output.get()[i] = static_cast<DataType>(0.0F);
  }
  AutofuseTilingData tiling_data{};
  uint32_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  RunTiling(tiling_data, workspace_size, block_dim);
#if IL_EXPECT_SIMT
  ASSERT_EQ(tiling_data.graph0_tiling_key, IL_TILING_KEY);
#endif
  ASSERT_GT(tiling_data.block_dim, 0U);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
#if IL_EXPECT_SIMT
  ICPU_RUN_KF(indirect_load_mixed_elementwise_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(x0.get()),
              reinterpret_cast<uint8_t *>(index0.get()), reinterpret_cast<uint8_t *>(index1.get()),
              reinterpret_cast<uint8_t *>(addend.get()), reinterpret_cast<uint8_t *>(sign.get()),
              reinterpret_cast<uint8_t *>(scale.get()), reinterpret_cast<uint8_t *>(output.get()), nullptr,
              reinterpret_cast<uint8_t *>(&tiling_data));
#else
  ICPU_RUN_KF(indirect_load_mixed_elementwise_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(x0.get()),
              reinterpret_cast<uint8_t *>(x1.get()), reinterpret_cast<uint8_t *>(index0.get()),
              reinterpret_cast<uint8_t *>(index1.get()), reinterpret_cast<uint8_t *>(addend.get()),
              reinterpret_cast<uint8_t *>(sign.get()), reinterpret_cast<uint8_t *>(scale.get()),
              reinterpret_cast<uint8_t *>(output.get()), nullptr, reinterpret_cast<uint8_t *>(&tiling_data));
#endif
  for (int32_t i = 0; i < output_count; ++i) {
    EXPECT_NEAR(static_cast<float>(output.get()[i]), static_cast<float>(expected[static_cast<size_t>(i)]), 0.0625F)
        << "offset=" << i;
  }
}
#endif
}  // namespace

TEST(E2EIndirectLoadStore, GeneratedKernelMatchesReference) {
#ifdef IL_MIXED_ELEMENTWISE
  RunMixedElementwiseCase();
#else
  const int32_t input_count = ElementCount(kInputShape);
  const int32_t output_count = ElementCount(kIndexShape);
  const int32_t result_count = ResultCount();
  const auto gm_free = [](void *ptr) { AscendC::GmFree(ptr); };
  std::unique_ptr<DataType, decltype(gm_free)> x(
      reinterpret_cast<DataType *>(AscendC::GmAlloc(input_count * sizeof(DataType))), gm_free);
  std::unique_ptr<IndexType, decltype(gm_free)> index(
      reinterpret_cast<IndexType *>(AscendC::GmAlloc(output_count * sizeof(IndexType))), gm_free);
#ifdef IL_POST_REDUCE_ADD
  std::unique_ptr<DataType, decltype(gm_free)> addend(
      reinterpret_cast<DataType *>(AscendC::GmAlloc(output_count * sizeof(DataType))), gm_free);
#else
  DataType *addend = nullptr;
#endif
  std::unique_ptr<OutputType, decltype(gm_free)> output(
      reinterpret_cast<OutputType *>(AscendC::GmAlloc(result_count * sizeof(OutputType))), gm_free);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(index, nullptr);
  ASSERT_NE(output, nullptr);
  std::vector<OutputType> expected(static_cast<size_t>(result_count));
#ifdef IL_POST_REDUCE_ADD
  ASSERT_NE(addend, nullptr);
  InitializeData(x.get(), index.get(), addend.get(), expected.data(), input_count, output_count);
#else
  InitializeData(x.get(), index.get(), addend, expected.data(), input_count, output_count);
#endif
  std::fill_n(output.get(), result_count, static_cast<OutputType>(0.0F));

  AutofuseTilingData tiling_data{};
  uint32_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  RunTiling(tiling_data, workspace_size, block_dim);
#ifdef IL_POST_REDUCE
#ifdef IL_EXPECT_ONLY_SIMT
  static_assert(IL_RANK == 4 && kAxis == 1);
  static_assert(IL_INPUT_PRE_TYPE == 3);
  static_assert(IL_REDUCE_WITH_A);
  ASSERT_EQ(tiling_data.tiling_key, IL_TILING_KEY);
#elif defined(IL_POST_REDUCE_SIMD)
  ASSERT_EQ(tiling_data.graph0_tiling_key, 0U);
  ASSERT_EQ(tiling_data.graph0_result0_g0_tiling_data.tiling_key, IL_TILING_KEY);
#else
#ifdef IL_EXPECT_SIMD_SELECTED
  ASSERT_EQ(tiling_data.graph0_tiling_key, 0U);
  ASSERT_EQ(tiling_data.graph0_result0_g0_tiling_data.tiling_key, IL_TILING_KEY);
#elif defined(IL_EXPECT_SIMT_SELECTED)
  ASSERT_EQ(tiling_data.graph0_tiling_key, 2U);
  ASSERT_EQ(tiling_data.graph0_result2_g0_tiling_data.tiling_key, IL_TILING_KEY);
#else
  ASSERT_TRUE(tiling_data.graph0_tiling_key == 0U || tiling_data.graph0_tiling_key == 2U);
  if (tiling_data.graph0_tiling_key == 0U) {
    ASSERT_EQ(tiling_data.graph0_result0_g0_tiling_data.tiling_key, IL_TILING_KEY);
  } else {
    const auto &simt_tiling_data = tiling_data.graph0_result2_g0_tiling_data;
    ASSERT_EQ(simt_tiling_data.tiling_key, IL_TILING_KEY);
    ASSERT_EQ(simt_tiling_data.block_dim, 48U);
#ifdef IL_REDUCE_LAST_AXIS
    ASSERT_EQ(simt_tiling_data.indirect_load_outerTb_size, 6U);
#else
    ASSERT_EQ(simt_tiling_data.indirect_load_outerTb_size, 2U);
#endif
  }
#endif
#endif
#elif !defined(IL_EXPECT_SIMT) || IL_EXPECT_SIMT
  ASSERT_EQ(tiling_data.graph0_tiling_key, IL_TILING_KEY);
#endif
  ASSERT_GT(tiling_data.block_dim, 0U);
  std::unique_ptr<uint8_t, decltype(gm_free)> workspace(nullptr, gm_free);
  if (workspace_size > 0U) {
    workspace.reset(reinterpret_cast<uint8_t *>(AscendC::GmAlloc(workspace_size)));
    ASSERT_NE(workspace, nullptr);
    std::fill_n(workspace.get(), workspace_size, uint8_t{0});
  }
#ifdef IL_EXPECT_SIMT_MULTI_ROUND
  ASSERT_GT((static_cast<uint32_t>(output_count) + tiling_data.block_dim - 1U) / tiling_data.block_dim, 1024U);
#endif

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
#ifdef IL_POST_REDUCE_ADD
  ICPU_RUN_KF(indirect_load_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(x.get()),
              reinterpret_cast<uint8_t *>(index.get()), reinterpret_cast<uint8_t *>(addend.get()),
              reinterpret_cast<uint8_t *>(output.get()), nullptr, reinterpret_cast<uint8_t *>(&tiling_data));
#else
  ICPU_RUN_KF(indirect_load_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(x.get()),
              reinterpret_cast<uint8_t *>(index.get()), reinterpret_cast<uint8_t *>(output.get()), workspace.get(),
              reinterpret_cast<uint8_t *>(&tiling_data));
#endif
  for (int32_t i = 0; i < result_count; ++i) {
#if defined(IL_DATA_BF16) || defined(IL_DATA_INT16) || defined(IL_DATA_UINT32)
    EXPECT_EQ(static_cast<float>(output.get()[i]), static_cast<float>(expected[static_cast<size_t>(i)]))
        << "offset=" << i;
#else
    EXPECT_NEAR(static_cast<float>(output.get()[i]), static_cast<float>(expected[static_cast<size_t>(i)]), 0.0625F)
        << "offset=" << i;
#endif
  }
#endif
}
