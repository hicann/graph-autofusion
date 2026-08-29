/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include <cstdint>

#include "gtest/gtest.h"
#include "test_api_utils.h"
#include "tikicpulib.h"
#include "api_regbase/softmax_af.h"

using namespace AscendC;

namespace {

template <typename T>
uint32_t CalcRAligned(uint32_t r) {
  return SoftmaxRegBase::CalcRAligned<T>(r);
}

class RegbaseApiSoftmaxTest : public testing::Test {};

TEST_F(RegbaseApiSoftmaxTest, HelperCoverage) {
  EXPECT_EQ(SoftmaxRegBase::CeilDiv(33U, 32U), 2U);
  EXPECT_EQ(SoftmaxRegBase::CeilDiv(33U, 0U), 0U);
  EXPECT_EQ(SoftmaxRegBase::AlignUp(33U, 32U), 64U);
  EXPECT_EQ(SoftmaxRegBase::FindNearestPower2(0U), 0U);
  EXPECT_EQ(SoftmaxRegBase::FindNearestPower2(1U), 0U);
  EXPECT_EQ(SoftmaxRegBase::FindNearestPower2(2U), 1U);
  EXPECT_EQ(SoftmaxRegBase::FindNearestPower2(4U), 2U);
  EXPECT_EQ(CalcRAligned<float>(33U), 40U);
  EXPECT_EQ(CalcRAligned<half>(33U), 48U);

  const uint32_t twoVl = 2U * SoftmaxRegBase::VL_FP32;
  EXPECT_EQ(SoftmaxRegBase::CalcReduceTmpStride(twoVl), 0U);
  EXPECT_EQ(SoftmaxRegBase::CalcSmallRExpCacheElems(4U), 4U * SoftmaxRegBase::VL_FP32);
  EXPECT_EQ(SoftmaxRegBase::CalcSmallRExpCacheBytes(4U), 4U * SoftmaxRegBase::VL_FP32 * sizeof(float));

  const uint32_t rSize = twoVl;
  const auto tiling = SoftmaxRegBase::GetSoftmaxARTilingInfo<float>(2U, rSize);
  EXPECT_EQ(tiling.aSize, 2U);
  EXPECT_EQ(tiling.rSize, rSize);
  EXPECT_EQ(tiling.rAligned, SoftmaxRegBase::CalcRAligned<float>(rSize));
  EXPECT_EQ(tiling.expBufElems, 2U * SoftmaxRegBase::CalcRAligned<float>(rSize));
  EXPECT_EQ(tiling.reduceTmpStride, 0U);
  EXPECT_EQ(tiling.reduceTmpElems, 0U);
  EXPECT_EQ(tiling.requiredTmpBytes, (tiling.expBufElems + tiling.reduceTmpElems) * sizeof(float));
}

TEST_F(RegbaseApiSoftmaxTest, TilingInfoForReduceTemp) {
  const uint32_t rSize = 3U * SoftmaxRegBase::VL_FP32;
  const uint32_t expectedStride = SoftmaxRegBase::BLOCK_SIZE / sizeof(float);
  const auto tiling = SoftmaxRegBase::GetSoftmaxARTilingInfo<float>(3U, rSize);

  EXPECT_EQ(SoftmaxRegBase::CalcReduceTmpStride(rSize), expectedStride);
  EXPECT_EQ(tiling.aSize, 3U);
  EXPECT_EQ(tiling.rSize, rSize);
  EXPECT_EQ(tiling.rAligned, rSize);
  EXPECT_EQ(tiling.expBufElems, 3U * rSize);
  EXPECT_EQ(tiling.reduceTmpStride, expectedStride);
  EXPECT_EQ(tiling.reduceTmpElems, 3U * expectedStride);
  EXPECT_EQ(tiling.requiredTmpBytes, (tiling.expBufElems + tiling.reduceTmpElems) * sizeof(float));
}

TEST_F(RegbaseApiSoftmaxTest, PathTypeBoundaryCoverage) {
  EXPECT_EQ(SoftmaxRegBase::GetSoftmaxARPathType(8U), SoftmaxRegBase::SoftmaxARPathType::SMALL_R);
  EXPECT_EQ(SoftmaxRegBase::GetSoftmaxARPathType(9U), SoftmaxRegBase::SoftmaxARPathType::MID_R);
  EXPECT_EQ(SoftmaxRegBase::GetSoftmaxARPathType(SoftmaxRegBase::VL_FP32), SoftmaxRegBase::SoftmaxARPathType::MID_R);
  EXPECT_EQ(SoftmaxRegBase::GetSoftmaxARPathType(SoftmaxRegBase::VL_FP32 + 1U),
            SoftmaxRegBase::SoftmaxARPathType::FULL_LOAD);
}

TEST_F(RegbaseApiSoftmaxTest, MidRtilingInfoCoverage) {
  const auto tiling9 = SoftmaxRegBase::GetSoftmaxARTilingInfo<float>(2U, 9U);
  EXPECT_EQ(tiling9.aSize, 2U);
  EXPECT_EQ(tiling9.rSize, 9U);
  EXPECT_EQ(tiling9.rAligned, SoftmaxRegBase::CalcRAligned<float>(9U));
  EXPECT_EQ(tiling9.expBufElems, 2U * SoftmaxRegBase::CalcRAligned<float>(9U));
  EXPECT_EQ(tiling9.reduceTmpStride, 0U);
  EXPECT_EQ(tiling9.reduceTmpElems, 0U);
  EXPECT_EQ(tiling9.requiredTmpBytes, tiling9.expBufElems * sizeof(float));

  const auto tiling31 = SoftmaxRegBase::GetSoftmaxARTilingInfo<float>(3U, 31U);
  EXPECT_EQ(tiling31.aSize, 3U);
  EXPECT_EQ(tiling31.rSize, 31U);
  EXPECT_EQ(tiling31.rAligned, SoftmaxRegBase::CalcRAligned<float>(31U));
  EXPECT_EQ(tiling31.expBufElems, 3U * SoftmaxRegBase::CalcRAligned<float>(31U));
  EXPECT_EQ(tiling31.reduceTmpStride, 0U);
  EXPECT_EQ(tiling31.reduceTmpElems, 0U);
  EXPECT_EQ(tiling31.requiredTmpBytes, tiling31.expBufElems * sizeof(float));
}

}  // namespace
