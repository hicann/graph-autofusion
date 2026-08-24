/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "tikicpulib.h"
#include "kernel_operator.h"
using namespace AscendC;

#include "utils.h"
#include <cstdint>
#include <climits>

class TestUtilsApi : public testing::Test {};
TEST(TestUtilsApi, TestCeilRational) {
  ASSERT_EQ(Ceiling(1.1f), 2);
  ASSERT_EQ((32 * Ceiling(Rational(1, 32) * 31)), 32);
  ASSERT_EQ((32 * Ceiling(Rational(1, 32) * 60)), 64);
}

TEST(TestUtilsApi, Floor_ShouldReturnInt64_WhenNumIsFloat) {
  ASSERT_EQ(Floor(2.3f), 2);
  ASSERT_EQ(Floor(-2.3f), -3);
  ASSERT_EQ(Floor(2.0f), 2);
  ASSERT_EQ(Floor(-2.0f), -2);
}

TEST(TestUtilsApi, Floor_ShouldReturnInt64_WhenNumIsInt64) {
  ASSERT_EQ(Floor(INT64_MAX), 9223372036854775807);
  ASSERT_EQ(Floor(INT64_MIN), -9223372036854775808);
}

TEST(TestUtilsApi, Floor_ShouldReturnInt_WhenNumIsInt) {
  ASSERT_EQ(Floor(INT32_MAX), 2147483647);
  ASSERT_EQ(Floor(INT32_MIN), -2147483648);
}

TEST(TestUtilsApi, Floor_ShouldReturnInt_WhenNumIsUnsignedInt) {
  ASSERT_EQ(Floor(UINT_MAX), 4294967295);
  ASSERT_EQ(Floor(0U), 0);
}

TEST(TestUtilsApi, IsInteger_ShouldCheckExactFloatValue) {
  EXPECT_TRUE(IsIntegerWithinUint32Magnitude(0.0f));
  EXPECT_TRUE(IsIntegerWithinUint32Magnitude(-0.0f));
  EXPECT_TRUE(IsIntegerWithinUint32Magnitude(2.0f));
  EXPECT_TRUE(IsIntegerWithinUint32Magnitude(-2.0f));
  EXPECT_TRUE(IsIntegerWithinUint32Magnitude(16777216.0f));
  EXPECT_TRUE(IsIntegerWithinUint32Magnitude(4294967040.0f));
  EXPECT_TRUE(IsIntegerWithinUint32Magnitude(-4294967040.0f));

  EXPECT_FALSE(IsIntegerWithinUint32Magnitude(0.5f));
  EXPECT_FALSE(IsIntegerWithinUint32Magnitude(-2.25f));
  EXPECT_FALSE(IsIntegerWithinUint32Magnitude(123456.75f));
  EXPECT_FALSE(IsIntegerWithinUint32Magnitude(4294967296.0f));
  EXPECT_FALSE(IsIntegerWithinUint32Magnitude(-4294967296.0f));
}

TEST(TestUtilsApi, IsInteger_ShouldRejectNanAndInfinity) {
  EXPECT_FALSE(IsIntegerWithinUint32Magnitude(AfInfinity<float>()));
  EXPECT_FALSE(IsIntegerWithinUint32Magnitude(GetScalarValueByBitCode<uint32_t, float>(0x7FC00000U)));
}

TEST(TestUtilsApi, TestFindNearestPower2) {
  uint64_t n1 = KernelUtils::FindNearestPower2(0);
  ASSERT_EQ(n1, 0);
  uint64_t n2 = KernelUtils::FindNearestPower2(1);
  ASSERT_EQ(n2, 1);
  uint64_t n3 = KernelUtils::FindNearestPower2(3);
  ASSERT_EQ(n3, 2);
}

TEST(TestUtilsApi, TestCalLog2) {
  uint64_t n = KernelUtils::CalLog2(8);
  ASSERT_EQ(n, 3);
}

TEST(TestUtilsApi, Mod_ShouldReturnInt_WhenNumIsInt) {
  ASSERT_EQ(Mod(5, 2), 1);
  ASSERT_EQ(Mod(5, -2), 1);
  ASSERT_EQ(Mod(-5, -2), -1);
  ASSERT_EQ(Mod(-5, 2), -1);
  ASSERT_EQ(Mod(5u, 2u), 1);
  ASSERT_EQ(Mod(0u, 2u), 0);
  ASSERT_NEAR(Mod(9.2f, 2.1f), 0.8, 1e-6);
  ASSERT_NEAR(Mod(-4.2f, 2.5f), -1.7, 1e-6);
  ASSERT_NEAR(Mod(10.0f, -3.5f), 3.0, 1e-6);
  ASSERT_NEAR(Mod(-10.0f, -3.5f), -3.0, 1e-6);
}

TEST(TestUtilsApi, Mod_ShouldReturnInt64_WhenNumIsInt64) {
  ASSERT_EQ(Mod(5ll, 2ll), 1ll);
  ASSERT_EQ(Mod(5ll, -2ll), 1ll);
  ASSERT_EQ(Mod(-5ll, -2ll), -1ll);
  ASSERT_EQ(Mod(-5ll, 2ll), -1ll);
  ASSERT_EQ(Mod(5ull, 2ull), 1ull);
  ASSERT_EQ(Mod(0ull, 2ull), 0);
}

TEST(TestUtilsApi, Mod_MixType) {
  ASSERT_EQ(Mod((int32_t)5, 2u), (int32_t)1);
  ASSERT_EQ(Mod(5u, (int32_t)-2), 1u);
  ASSERT_EQ(Mod((int32_t)5, 2ll), (int32_t)1);
  ASSERT_EQ(Mod(5ll, (int32_t)-2), 1ll);
}

TEST(TestUtilsApi, AfInfinityTest) {
  union ScalarBitcodeFloat {
    __aicore__ ScalarBitcodeFloat() {}
    float input;
    uint32_t output;
  } data_float;

  union ScalarBitcodeHalf {
    __aicore__ ScalarBitcodeHalf() {}
    half input;
    uint16_t output;
  } data_half;

  data_float.input = AfInfinity<float>();
  ASSERT_EQ(data_float.output, 0x7F800000U);

  data_half.input = AfInfinity<half>();
  ASSERT_EQ(data_half.output, 0x7C00U);
}
