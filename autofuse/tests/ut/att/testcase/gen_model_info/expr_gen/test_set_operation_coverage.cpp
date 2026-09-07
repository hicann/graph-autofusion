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
#include <gtest/gtest.h>
#include "expr_gen/set_operation.h"

namespace att {
namespace {
DimRange Range(int64_t lower, int64_t upper) {
  return {CreateExpr(upper), CreateExpr(lower)};
}

TEST(SetOperationCoverageTest, DimDifferencePreservesBothBoundaryRemainders) {
  auto whole = Range(0, 10);
  auto left = Range(0, 4);
  auto right = Range(6, 10);
  auto middle = Range(3, 7);
  EXPECT_EQ(SetOperation::Diff(whole, whole), std::vector<DimRange>{});
  EXPECT_EQ(SetOperation::Diff(whole, left), (std::vector<DimRange>{Range(4, 10)}));
  EXPECT_EQ(SetOperation::Diff(whole, right), (std::vector<DimRange>{Range(0, 6)}));
  EXPECT_EQ(SetOperation::Diff(whole, middle), (std::vector<DimRange>{Range(0, 3), Range(7, 10)}));
}

TEST(SetOperationCoverageTest, TensorDifferencePartitionsAreaWithoutRemovedRectangle) {
  TensorRange outer{{Range(0, 10), Range(0, 8)}};
  TensorRange inner{{Range(2, 6), Range(1, 5)}};
  auto difference = SetOperation::Diff(outer, inner);
  ASSERT_EQ(difference.size(), 8U);
  EXPECT_TRUE(SetOperation::SetComputation(difference) == 64);
  EXPECT_EQ(std::count(difference.begin(), difference.end(), inner.front()), 0);
  auto intersection = SetOperation::Intersection(outer, inner);
  ASSERT_EQ(intersection, inner);
  EXPECT_TRUE(SetOperation::SetComputation(intersection) == 16);
  EXPECT_TRUE(SetOperation::Diff(inner, inner).empty());
}

TEST(SetOperationCoverageTest, EmptyAndMismatchedTensorRangesProduceEmptyResults) {
  TensorRange empty;
  TensorRange single{{Range(0, 4)}};
  TensorRange pair{{Range(0, 4)}, {Range(8, 12)}};
  EXPECT_TRUE(SetOperation::Diff(single, pair).empty());
  EXPECT_TRUE(SetOperation::Diff(empty, empty).empty());
  EXPECT_TRUE(SetOperation::Intersection(empty, single).empty());
  EXPECT_TRUE(SetOperation::Intersection(single, empty).empty());
  EXPECT_TRUE(SetOperation::SetComputation(empty) == 0);
}

TEST(SetOperationCoverageTest, IntersectionClipsBoundsAndClampsDisjointVolumeToZero) {
  TensorRange left{{Range(0, 4), Range(1, 8)}};
  TensorRange right{{Range(2, 6), Range(3, 9)}};
  auto overlap = SetOperation::Intersection(left, right);
  ASSERT_EQ(overlap, (TensorRange{{Range(2, 4), Range(3, 8)}}));
  EXPECT_TRUE(SetOperation::SetComputation(overlap) == 10);
  TensorRange distant{{Range(10, 12), Range(3, 9)}};
  auto disjoint = SetOperation::Intersection(left, distant);
  EXPECT_TRUE(SetOperation::SetComputation(disjoint) == 0);
}

TEST(SetOperationCoverageTest, CartesianProductEnumeratesCombinationsAndPreservesPrefix) {
  std::vector<std::vector<uint32_t>> sequence{{1, 2}, {3, 4}};
  std::vector<std::vector<uint32_t>> result;
  std::vector<uint32_t> prefix{9};
  SetOperation::ProductImplement(sequence, result, 0, prefix);
  EXPECT_EQ(result, (std::vector<std::vector<uint32_t>>{{9, 1, 3}, {9, 1, 4}, {9, 2, 3}, {9, 2, 4}}));
  EXPECT_EQ(prefix, (std::vector<uint32_t>{9}));
  result.clear();
  sequence.clear();
  SetOperation::ProductImplement(sequence, result, 0, prefix);
  EXPECT_TRUE(result.empty());
  sequence = {{1}, {}};
  SetOperation::ProductImplement(sequence, result, 0, prefix);
  EXPECT_TRUE(result.empty());
}
}  // namespace
}  // namespace att
