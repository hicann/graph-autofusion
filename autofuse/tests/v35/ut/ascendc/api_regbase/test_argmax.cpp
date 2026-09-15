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
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/argmax.h"

using namespace AscendC;

namespace af {
namespace {
constexpr uint32_t kRows = 6;

constexpr uint32_t AlignBlock(uint32_t bytes) {
  return (bytes + 31U) / 32U * 32U;
}

template <typename T, bool isRa>
void FillInput(T *src, uint32_t columns) {
  for (uint32_t row = 0; row < kRows; ++row) {
    for (uint32_t column = 0; column < columns; ++column) {
      T value = static_cast<T>(column);
      if (row == 1) {
        value = static_cast<T>(-3);
      } else if (row == 2) {
        const uint32_t second = columns > 1 ? columns - 2 : 0;
        value = column == std::min(5U, columns - 1) || column == second ? static_cast<T>(99) : static_cast<T>(-100);
      } else if (row == 3) {
        value = static_cast<T>(-static_cast<int64_t>(column));
        const uint32_t first = std::min(5U, columns - 1);
        const uint32_t second = std::max(first, columns > 1 ? columns - 2 : 0);
        if (column == first || column == second) {
          if constexpr (std::is_same_v<T, float>) {
            value = std::numeric_limits<float>::quiet_NaN();
          } else {
            value = (1 << 24) + 1;
          }
        }
      } else if constexpr (std::is_same_v<T, float>) {
        if (row == 4) {
          value = column % 2 == 0 ? -0.0F : 0.0F;
        } else if (row == 5) {
          value = column == 0 ? 0.0F : -0.0F;
        }
      }
      src[isRa ? column * kRows + row : row * columns + column] = value;
    }
  }
}

template <typename T>
bool IsLarger(T current, T best) {
  if constexpr (std::is_same_v<T, float>) {
    return !std::isnan(best) && (std::isnan(current) || current > best);
  }
  return current > best;
}

template <typename T, typename IndexT, bool isRa>
void RunArgMax(uint32_t columns) {
  T *src = static_cast<T *>(GmAlloc(sizeof(T) * kRows * columns));
  IndexT *index = static_cast<IndexT *>(GmAlloc(sizeof(IndexT) * (kRows + 2)));
  std::fill_n(index, kRows + 2, static_cast<IndexT>(-1));
  FillInput<T, isRa>(src, columns);

  auto kernel = [=] {
    TPipe pipe;
    TBuf<TPosition::VECCALC> srcBuf, indexBuf;
    pipe.InitBuffer(srcBuf, AlignBlock(sizeof(T) * kRows * columns));
    pipe.InitBuffer(indexBuf, AlignBlock(sizeof(IndexT) * (kRows + 2)));
    LocalTensor<T> srcLocal = srcBuf.Get<T>();
    LocalTensor<IndexT> indexLocal = indexBuf.Get<IndexT>();
    GmToUb(srcLocal, src, kRows * columns);
    // Nonzero high bits expose partial int64 stores; guard both ends of dst.
    GmToUb(indexLocal, index, kRows + 2);
    uint32_t srcShape[] = {isRa ? columns : kRows, isRa ? kRows : columns};
    if constexpr (isRa) {
      ArgMaxExtend<IndexT, T, Pattern::Reduce::RA>(indexLocal[1], srcLocal, LocalTensor<uint8_t>(), srcShape, false);
    } else {
      ArgMaxExtend<IndexT, T, Pattern::Reduce::AR>(indexLocal[1], srcLocal, LocalTensor<uint8_t>(), srcShape, false);
    }
    UbToGm(index, indexLocal, kRows + 2);
  };

  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  EXPECT_EQ(index[0], static_cast<IndexT>(-1));
  EXPECT_EQ(index[kRows + 1], static_cast<IndexT>(-1));
  for (uint32_t row = 0; row < kRows; ++row) {
    T best = src[isRa ? row : row * columns];
    uint32_t bestIndex = 0;
    for (uint32_t column = 1; column < columns; ++column) {
      T current = src[isRa ? column * kRows + row : row * columns + column];
      if (IsLarger(current, best)) {
        best = current;
        bestIndex = column;
      }
    }
    EXPECT_EQ(index[row + 1], static_cast<IndexT>(bestIndex));
  }
  GmFree(src);
  GmFree(index);
}

template <bool isRa>
void RunTypeMatrix() {
  RunArgMax<float, int32_t, isRa>(67);
  RunArgMax<float, int64_t, isRa>(67);
  RunArgMax<int32_t, int32_t, isRa>(67);
  RunArgMax<int32_t, int64_t, isRa>(67);
}

template <bool isRa>
void RunBoundaries() {
  // Scalar, full VL, unaligned tail, aligned tail, paired repeats, and large input.
  for (uint32_t columns : {1U, 64U, 65U, 136U, 256U, 4096U}) {
    RunArgMax<float, int32_t, isRa>(columns);
    RunArgMax<float, int64_t, isRa>(columns);
    RunArgMax<int32_t, int32_t, isRa>(columns);
    RunArgMax<int32_t, int64_t, isRa>(columns);
  }
}
}  // namespace

TEST(TestApiArgMax, ARTypeMatrix) {
  RunTypeMatrix<false>();
}

TEST(TestApiArgMax, RATypeMatrix) {
  RunTypeMatrix<true>();
}

TEST(TestApiArgMax, ARBoundaries) {
  RunBoundaries<false>();
}

TEST(TestApiArgMax, RABoundaries) {
  RunBoundaries<true>();
}

TEST(TestApiArgMax, InvalidArguments) {
  auto call = [](const uint32_t *shape, bool innerPad) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> srcBuf, dstBuf;
    pipe.InitBuffer(srcBuf, 256);
    pipe.InitBuffer(dstBuf, 32);
    ArgMaxExtend<int32_t, float, Pattern::Reduce::AR>(dstBuf.Get<int32_t>(), srcBuf.Get<float>(),
                                                      LocalTensor<uint8_t>(), shape, innerPad);
  };
  uint32_t emptyShape[] = {1, 0};
  uint32_t overflowShape[] = {65536, 65536};
  uint32_t shortSrcShape[] = {1, 65};
  uint32_t shortDstShape[] = {9, 1};
  uint32_t validShape[] = {1, 8};
  EXPECT_DEATH(call(nullptr, false), "");
  EXPECT_DEATH(call(emptyShape, false), "");
  EXPECT_DEATH(call(overflowShape, false), "");
  EXPECT_DEATH(call(shortSrcShape, false), "");
  EXPECT_DEATH(call(shortDstShape, false), "");
  EXPECT_DEATH(call(validShape, true), "");
}

}  // namespace af
