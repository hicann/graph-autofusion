/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "test_api_utils.h"
#include "tikicpulib.h"
#include "api_regbase/randn.h"

using namespace AscendC;

namespace af {

struct RandnResult {
  std::vector<float> values;
  bool allFinite{true};
  double mean{0.0};
  double variance{0.0};
};

template <RandnPhiloxRounds Rounds>
static void InvokeRandn(float *dst, uint32_t count, const PhiloxKey &key, const PhiloxCounter &counter) {
  TPipe tpipe;
  constexpr uint32_t kVectorCapacity = B32_DATA_NUM_PER_REPEAT;
  const uint32_t bufferCount = AlignUp(count, kVectorCapacity);
  TBuf<TPosition::VECCALC> dstBuf, uniformBuf;
  tpipe.InitBuffer(dstBuf, sizeof(float) * bufferCount);
  tpipe.InitBuffer(uniformBuf, sizeof(float) * bufferCount);

  LocalTensor<float> dstTensor = dstBuf.Get<float>();
  LocalTensor<float> uniformTensor = uniformBuf.Get<float>();
  RandnExtend<float, Rounds>(dstTensor, uniformTensor, key, counter, static_cast<uint16_t>(count));
  if (count < B32_DATA_NUM_PER_REPEAT) {
    for (uint32_t i = 0U; i < count; ++i) {
      dst[i] = dstTensor.GetValue(i);
    }
  } else {
    UbToGm(dst, dstTensor, count);
  }
}

template <RandnPhiloxRounds Rounds>
static RandnResult RunRandn(uint32_t count, const PhiloxKey &key, const PhiloxCounter &counter) {
  RandnResult result;
  float *allocatedDst = static_cast<float *>(GmAlloc(sizeof(float) * count + 32U));
  const uintptr_t alignedAddress = (reinterpret_cast<uintptr_t>(allocatedDst) + 31U) & ~uintptr_t(31U);
  float *dst = reinterpret_cast<float *>(alignedAddress);
  auto kernel = [dst, count, &key, &counter] { InvokeRandn<Rounds>(dst, count, key, counter); };

  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);

  result.values.assign(dst, dst + count);
  GmFree(allocatedDst);

  double sum = 0.0;
  for (float value : result.values) {
    result.allFinite = result.allFinite && std::isfinite(value);
    sum += static_cast<double>(value);
  }
  if (!result.allFinite || count < 2U) {
    return result;
  }

  result.mean = sum / static_cast<double>(count);
  double squareError = 0.0;
  for (float value : result.values) {
    const double delta = static_cast<double>(value) - result.mean;
    squareError += delta * delta;
  }
  result.variance = squareError / static_cast<double>(count - 1U);
  return result;
}

class TestApiRandn : public testing::Test {
 protected:
  static constexpr PhiloxKey kKey = {0x12345678U, 0x9ABCDEF0U};
  static constexpr PhiloxCounter kCounter = {0x0U, 0x13579BDFU, 0x2468ACE0U, 0x10203040U};
};

TEST_F(TestApiRandn, RandnExtend_MinimumCountIsFinite) {
  const RandnResult result = RunRandn<RandnPhiloxRounds::ROUNDS_10>(4U, kKey, kCounter);
  ASSERT_TRUE(result.allFinite);
  ASSERT_EQ(result.values.size(), 4U);
}

TEST_F(TestApiRandn, RandnExtend_RoundsProduceFiniteNormalSamples) {
  const RandnResult result7 = RunRandn<RandnPhiloxRounds::ROUNDS_7>(1024U, kKey, kCounter);
  const RandnResult result10 = RunRandn<RandnPhiloxRounds::ROUNDS_10>(1024U, kKey, kCounter);

  ASSERT_TRUE(result7.allFinite);
  ASSERT_TRUE(result10.allFinite);
  EXPECT_NE(result7.values, result10.values);
  EXPECT_NEAR(result7.mean, 0.0, 0.25);
  EXPECT_NEAR(result10.mean, 0.0, 0.25);
  EXPECT_NEAR(result7.variance, 1.0, 0.4);
  EXPECT_NEAR(result10.variance, 1.0, 0.4);
}

TEST_F(TestApiRandn, RandnExtend_SameKeyAndCounterIsDeterministic) {
  const RandnResult first = RunRandn<RandnPhiloxRounds::ROUNDS_10>(256U, kKey, kCounter);
  const RandnResult second = RunRandn<RandnPhiloxRounds::ROUNDS_10>(256U, kKey, kCounter);

  EXPECT_EQ(first.values, second.values);
}

TEST_F(TestApiRandn, RandnExtend_KeyChangeChangesSequence) {
  PhiloxKey changedKey = {kKey[0] ^ 1U, kKey[1]};
  const RandnResult first = RunRandn<RandnPhiloxRounds::ROUNDS_10>(256U, kKey, kCounter);
  const RandnResult second = RunRandn<RandnPhiloxRounds::ROUNDS_10>(256U, changedKey, kCounter);

  EXPECT_NE(first.values, second.values);
}

}  // namespace af
