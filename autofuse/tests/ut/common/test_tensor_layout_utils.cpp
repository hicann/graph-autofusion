/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <vector>

#include "tensor_layout_utils.h"

namespace ascgen_utils {
namespace {
af::AscTensorAttr CreateAttr(const std::vector<int64_t> &axis, const std::vector<int64_t> &repeats,
                             const std::vector<int64_t> &strides, const std::vector<int64_t> &vectorized_axis) {
  af::AscTensorAttr attr;
  for (const auto value : axis) {
    attr.axis.emplace_back(value);
  }
  for (const auto value : repeats) {
    attr.repeats.emplace_back(af::Symbol(value));
  }
  for (const auto value : strides) {
    attr.strides.emplace_back(af::Symbol(value));
  }
  attr.vectorized_axis = vectorized_axis;
  return attr;
}
}  // namespace

TEST(TensorLayoutUtilsTest, AnalyzeContinuousLayout) {
  const auto attr = CreateAttr({0, 1}, {2, 4}, {4, 1}, {0, 1});
  DiscontinuityInfo info;

  EXPECT_EQ(TensorLayoutUtils::AnalyzeLoadDiscontinuity(attr, info), af::SUCCESS);
  EXPECT_FALSE(info.is_tail_axis_discontinuous);
  EXPECT_FALSE(info.has_multiple_discontinuities);
}

TEST(TensorLayoutUtilsTest, AnalyzeSingleDiscontinuity) {
  const auto attr = CreateAttr({0, 1}, {2, 4}, {8, 1}, {0, 1});
  DiscontinuityInfo info;

  EXPECT_EQ(TensorLayoutUtils::AnalyzeLoadDiscontinuity(attr, info), af::SUCCESS);
  EXPECT_FALSE(info.is_tail_axis_discontinuous);
  EXPECT_FALSE(info.has_multiple_discontinuities);
}

TEST(TensorLayoutUtilsTest, AnalyzeMultipleDiscontinuities) {
  const auto attr = CreateAttr({0, 1, 2}, {2, 2, 4}, {16, 3, 1}, {0, 1, 2});
  DiscontinuityInfo info;

  EXPECT_EQ(TensorLayoutUtils::AnalyzeLoadDiscontinuity(attr, info), af::SUCCESS);
  EXPECT_FALSE(info.is_tail_axis_discontinuous);
  EXPECT_TRUE(info.has_multiple_discontinuities);
}

TEST(TensorLayoutUtilsTest, AnalyzeTailAxisDiscontinuity) {
  const auto attr = CreateAttr({0, 1}, {2, 4}, {4, 2}, {1});
  DiscontinuityInfo info;

  EXPECT_EQ(TensorLayoutUtils::AnalyzeLoadDiscontinuity(attr, info), af::SUCCESS);
  EXPECT_TRUE(info.is_tail_axis_discontinuous);
  EXPECT_FALSE(info.has_multiple_discontinuities);
}

TEST(TensorLayoutUtilsTest, IgnoreBroadcastAxisWhenAnalyzingDiscontinuity) {
  const auto attr = CreateAttr({0, 1, 2}, {2, 3, 4}, {12, 0, 1}, {0, 1, 2});
  DiscontinuityInfo info;

  EXPECT_EQ(TensorLayoutUtils::AnalyzeLoadDiscontinuity(attr, info), af::SUCCESS);
  EXPECT_FALSE(info.is_tail_axis_discontinuous);
  EXPECT_FALSE(info.has_multiple_discontinuities);
}

TEST(TensorLayoutUtilsTest, ReturnFailureWhenVectorizedAxisIsMissing) {
  const auto attr = CreateAttr({0, 1}, {2, 4}, {4, 1}, {2});
  DiscontinuityInfo info;

  EXPECT_NE(TensorLayoutUtils::AnalyzeLoadDiscontinuity(attr, info), af::SUCCESS);
}

TEST(TensorLayoutUtilsTest, ReturnFailureWhenLayoutAttributeLengthsAreInsufficient) {
  const auto attr = CreateAttr({0, 1}, {2}, {4}, {0, 1});
  DiscontinuityInfo info;

  EXPECT_NE(TensorLayoutUtils::AnalyzeLoadDiscontinuity(attr, info), af::SUCCESS);
}
}  // namespace ascgen_utils
