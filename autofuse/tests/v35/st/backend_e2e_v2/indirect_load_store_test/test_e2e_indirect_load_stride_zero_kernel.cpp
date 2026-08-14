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

#include "indirect_load_kernel_test_common.h"

extern "C" __global__ __aicore__ void indirect_load_stride_zero_test(GM_ADDR x, GM_ADDR index, GM_ADDR y,
                                                                     GM_ADDR workspace, GM_ADDR tiling);

namespace {
using DataType = half;
using IndexType = int64_t;
constexpr std::array<int64_t, 4> kShape = {4, 5, 4, 16};
constexpr uint32_t kInputZeroStrideMask = IL_INPUT_ZERO_STRIDE_MASK;
constexpr uint32_t kIndexZeroStrideMask = IL_INDEX_ZERO_STRIDE_MASK;
constexpr bool kHasInputElement = IL_HAS_INPUT_ELEMENT > 0;
constexpr bool kHasIndexElement = IL_HAS_INDEX_ELEMENT > 0;

std::array<int64_t, 4> MakeStrides(uint32_t zero_stride_mask) {
  std::array<int64_t, 4> strides{};
  int64_t stride = 1;
  for (size_t index = kShape.size(); index > 0UL; --index) {
    const size_t dim = index - 1UL;
    if ((zero_stride_mask & (1U << dim)) == 0U) {
      strides[dim] = stride;
      stride *= kShape[dim];
    }
  }
  return strides;
}

int64_t StorageSpan(const std::array<int64_t, 4> &strides) {
  int64_t span = 1;
  for (size_t dim = 0; dim < kShape.size(); ++dim) {
    span += (kShape[dim] - 1) * strides[dim];
  }
  return span;
}

int64_t Offset(const std::array<int64_t, 4> &coordinate, const std::array<int64_t, 4> &strides) {
  int64_t offset = 0;
  for (size_t dim = 0; dim < coordinate.size(); ++dim) {
    offset += coordinate[dim] * strides[dim];
  }
  return offset;
}

void InitializeData(DataType *x, IndexType *index, DataType *expected, const std::array<int64_t, 4> &input_strides,
                    const std::array<int64_t, 4> &index_strides) {
  const int64_t input_count = StorageSpan(input_strides);
  const int64_t index_count = StorageSpan(index_strides);
  for (int64_t i = 0; i < input_count; ++i) {
    x[i] = static_cast<DataType>(static_cast<float>((i % 37) - 18) * 0.25F);
  }
  for (int64_t i = 0; i < index_count; ++i) {
    const int64_t gathered_axis = (i * 3 + 1) % kShape[2];
    index[i] = kHasIndexElement && (i % 2 == 1) ? -gathered_axis : gathered_axis;
  }

  const int64_t output_count = kShape[0] * kShape[1] * kShape[2] * kShape[3];
  for (int64_t output_offset = 0; output_offset < output_count; ++output_offset) {
    int64_t linear = output_offset;
    const int64_t d = linear % kShape[3];
    linear /= kShape[3];
    const int64_t c = linear % kShape[2];
    linear /= kShape[2];
    const int64_t b = linear % kShape[1];
    const int64_t a = linear / kShape[1];
    std::array<int64_t, 4> coordinate = {a, b, c, d};
    int64_t gathered_axis = index[Offset(coordinate, index_strides)];
    if (kHasIndexElement) {
      gathered_axis = std::abs(gathered_axis);
    }
    coordinate[2] = gathered_axis;
    float value = static_cast<float>(x[Offset(coordinate, input_strides)]);
    if (kHasInputElement) {
      value = std::abs(value);
    }
    expected[output_offset] = static_cast<DataType>(std::max(value, 0.0F));
  }
}
}  // namespace

TEST(E2EIndirectLoadStrideZero, GeneratedKernelMatchesPhysicalStrideReference) {
  const auto dense_strides = MakeStrides(0U);
  const auto input_strides = MakeStrides(kInputZeroStrideMask);
  const auto index_strides = MakeStrides(kIndexZeroStrideMask);
  const int64_t input_count = StorageSpan(input_strides);
  const int64_t index_count = StorageSpan(index_strides);
  const int64_t output_count = StorageSpan(dense_strides);
  indirect_load_test::KernelData<DataType, IndexType> buffers(input_count, index_count, output_count);
  ASSERT_TRUE(buffers.IsValid());
  InitializeData(buffers.input.get(), buffers.index.get(), buffers.expected.data(), input_strides, index_strides);
  std::fill_n(buffers.output.get(), output_count, static_cast<DataType>(0.0F));
  indirect_load_test::KernelTiling tiling;
  ASSERT_TRUE(tiling.IsValid());

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(indirect_load_stride_zero_test, tiling.data.block_dim, reinterpret_cast<uint8_t *>(buffers.input.get()),
              reinterpret_cast<uint8_t *>(buffers.index.get()), reinterpret_cast<uint8_t *>(buffers.output.get()),
              tiling.workspace.get(), reinterpret_cast<uint8_t *>(&tiling.data));
  for (int64_t i = 0; i < output_count; ++i) {
    EXPECT_NEAR(static_cast<float>(buffers.output.get()[i]),
                static_cast<float>(buffers.expected[static_cast<size_t>(i)]), 0.0625F)
        << "offset=" << i;
  }
}
