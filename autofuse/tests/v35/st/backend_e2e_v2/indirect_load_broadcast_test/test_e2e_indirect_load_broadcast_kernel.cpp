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

extern "C" __global__ __aicore__ void indirect_load_broadcast_test(GM_ADDR x, GM_ADDR index, GM_ADDR y,
                                                                   GM_ADDR workspace, GM_ADDR tiling);

namespace {
using DataType = half;
using IndexType = int64_t;
constexpr std::array<int32_t, 4> kOutputShape = {4, 5, 4, 16};
constexpr bool kInputBroadcast = IL_INPUT_BROADCAST;
constexpr bool kIndexBroadcast = IL_INDEX_BROADCAST;
constexpr uint32_t kBroadcastAxesMask = IL_BROADCAST_AXES_MASK;

constexpr std::array<int32_t, 4> MakeBroadcastSourceShape() {
  auto shape = kOutputShape;
  for (size_t dim = 0UL; dim < shape.size(); ++dim) {
    if ((kBroadcastAxesMask & (1U << dim)) != 0U) {
      shape[dim] = 1;
    }
  }
  return shape;
}

constexpr std::array<int32_t, 4> kBroadcastSourceShape = MakeBroadcastSourceShape();
constexpr std::array<int32_t, 4> kInputShape = kInputBroadcast ? kBroadcastSourceShape : kOutputShape;
constexpr std::array<int32_t, 4> kIndexShape = kIndexBroadcast ? kBroadcastSourceShape : kOutputShape;
constexpr int32_t kInputElementCount = IL_HAS_INPUT_ELEMENT;
constexpr int32_t kIndexElementCount = IL_HAS_INDEX_ELEMENT;
constexpr bool kHasOutputRelu = IL_HAS_OUTPUT_RELU;

template <size_t N>
int32_t ElementCount(const std::array<int32_t, N> &shape) {
  int32_t count = 1;
  for (const int32_t dim : shape) {
    count *= dim;
  }
  return count;
}

int32_t DenseOffset(const std::array<int32_t, 4> &coordinate, const std::array<int32_t, 4> &shape) {
  int32_t offset = 0;
  for (size_t dim = 0UL; dim < coordinate.size(); ++dim) {
    offset = offset * shape[dim] + coordinate[dim];
  }
  return offset;
}

void InitializeData(DataType *x, IndexType *index, DataType *expected) {
  for (int32_t i = 0; i < ElementCount(kInputShape); ++i) {
    x[i] = static_cast<DataType>(static_cast<float>((i % 29) - 14) * 0.25F);
  }
  for (int32_t i = 0; i < ElementCount(kIndexShape); ++i) {
    const int32_t gathered_axis = (i * 3 + 1) % kInputShape[2];
    index[i] = static_cast<IndexType>(kIndexElementCount == 0 || i % 2 == 0 ? gathered_axis : -gathered_axis);
  }
  for (int32_t i = 0; i < ElementCount(kOutputShape); ++i) {
    int32_t coordinate = i;
    const int32_t d = coordinate % kOutputShape[3];
    coordinate /= kOutputShape[3];
    const int32_t c = coordinate % kOutputShape[2];
    coordinate /= kOutputShape[2];
    const int32_t b = coordinate % kOutputShape[1];
    coordinate /= kOutputShape[1];
    const int32_t a = coordinate;
    std::array<int32_t, 4> index_coordinate = {a, b, c, d};
    if (kIndexBroadcast) {
      for (size_t dim = 0UL; dim < index_coordinate.size(); ++dim) {
        if ((kBroadcastAxesMask & (1U << dim)) != 0U) {
          index_coordinate[dim] = 0;
        }
      }
    }
    const int32_t index_offset = DenseOffset(index_coordinate, kIndexShape);
    int64_t gathered_index = static_cast<int64_t>(index[index_offset]);
    for (int32_t element = 0; element < kIndexElementCount; ++element) {
      gathered_index = std::abs(gathered_index);
    }
    const int32_t gathered_axis = static_cast<int32_t>(gathered_index);
    std::array<int32_t, 4> input_coordinate = {a, b, gathered_axis, d};
    if (kInputBroadcast) {
      for (size_t dim = 0UL; dim < input_coordinate.size(); ++dim) {
        if ((kBroadcastAxesMask & (1U << dim)) != 0U) {
          input_coordinate[dim] = 0;
        }
      }
    }
    const int32_t input_offset = DenseOffset(input_coordinate, kInputShape);
    float value = static_cast<float>(x[input_offset]);
    for (int32_t element = 0; element < kInputElementCount; ++element) {
      value = std::abs(value);
    }
    if (kHasOutputRelu) {
      value = std::max(value, 0.0F);
    }
    expected[i] = static_cast<DataType>(value);
  }
}
}  // namespace

TEST(E2EIndirectLoadBroadcast, GeneratedKernelMatchesReference) {
  const int32_t input_count = ElementCount(kInputShape);
  const int32_t index_count = ElementCount(kIndexShape);
  const int32_t output_count = ElementCount(kOutputShape);
  indirect_load_test::KernelData<DataType, IndexType> buffers(input_count, index_count, output_count);
  ASSERT_TRUE(buffers.IsValid());
  InitializeData(buffers.input.get(), buffers.index.get(), buffers.expected.data());
  std::fill_n(buffers.output.get(), output_count, static_cast<DataType>(0.0F));
  indirect_load_test::KernelTiling tiling;
  ASSERT_TRUE(tiling.IsValid());

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(indirect_load_broadcast_test, tiling.data.block_dim, reinterpret_cast<uint8_t *>(buffers.input.get()),
              reinterpret_cast<uint8_t *>(buffers.index.get()), reinterpret_cast<uint8_t *>(buffers.output.get()),
              tiling.workspace.get(), reinterpret_cast<uint8_t *>(&tiling.data));
  for (int32_t i = 0; i < output_count; ++i) {
    EXPECT_NEAR(static_cast<float>(buffers.output.get()[i]),
                static_cast<float>(buffers.expected[static_cast<size_t>(i)]), 0.0625F)
        << "offset=" << i;
  }
}
