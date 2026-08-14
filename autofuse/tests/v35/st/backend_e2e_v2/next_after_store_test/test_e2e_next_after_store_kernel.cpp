/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void next_after_store_test(GM_ADDR x1, GM_ADDR x2, GM_ADDR y1, GM_ADDR workspace,
                                                            GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendNextAfterStoreCode : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

// Reference implementation of nextafter for float
static float NextAfterReference(float x, float y) {
  // Handle NaN cases
  if (std::isnan(x) || std::isnan(y)) {
    return std::nan("");
  }

  // If x equals y, return x
  if (x == y) {
    return x;
  }

  // Get bit representation
  uint32_t x_bits;
  std::memcpy(&x_bits, &x, sizeof(uint32_t));

  const uint32_t sign_mask = 0x80000000U;
  const uint32_t abs_mask = 0x7FFFFFFFU;
  const uint32_t one = 0x00000001U;

  bool x_sign = (x_bits & sign_mask) != 0;

  // Determine direction
  bool go_up = (x < y) ? !x_sign : x_sign;

  if (go_up) {
    // Increment (toward positive infinity)
    if (x_bits == 0x7F7FFFFFU) {  // FLT_MAX
      return std::numeric_limits<float>::infinity();
    }
    x_bits += one;
  } else {
    // Decrement (toward negative infinity)
    if (x_bits == 0xFF7FFFFFU) {  // -FLT_MAX
      return -std::numeric_limits<float>::infinity();
    }
    x_bits -= one;
  }

  float result;
  std::memcpy(&result, &x_bits, sizeof(float));
  return result;
}

TEST_P(E2EBackendNextAfterStoreCode, CalculateCorrect) {
  auto test_shape = GetParam();
  uint64_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  float *x1 = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  float *x2 = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  float *y = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  float *expect = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));

  // Generate test data covering different scenarios
  for (int i = 0; i < test_size; i++) {
    // Test various interesting cases for nextafter
    int case_type = i % 10;
    switch (case_type) {
      case 0:  // Zero
        x1[i] = 0.0f;
        x2[i] = 1.0f;
        break;
      case 1:  // Positive number
        x1[i] = 1.5f;
        x2[i] = 2.0f;
        break;
      case 2:  // Negative number
        x1[i] = -1.5f;
        x2[i] = -1.0f;
        break;
      case 3:  // Subnormal range
        x1[i] = 1.0e-40f;
        x2[i] = 2.0e-40f;
        break;
      case 4:  // Large numbers
        x1[i] = 1.0e30f;
        x2[i] = 2.0e30f;
        break;
      case 5:  // Equal values
        x1[i] = 3.14159f;
        x2[i] = 3.14159f;
        break;
      case 6:  // Direction change
        x1[i] = 1.0f;
        x2[i] = -1.0f;
        break;
      case 7:  // Small positive
        x1[i] = 1.0e-10f;
        x2[i] = 2.0e-10f;
        break;
      case 8:  // Negative toward zero
        x1[i] = -0.5f;
        x2[i] = 0.5f;
        break;
      case 9:  // Mixed
        x1[i] = static_cast<float>((i % 100) - 50) / 10.0f;
        x2[i] = x1[i] + 0.1f;
        break;
    }
    expect[i] = NextAfterReference(x1[i], x2[i]);
  }

  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(next_after_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(x1),
              reinterpret_cast<uint8_t *>(x2), reinterpret_cast<uint8_t *>(y), nullptr,
              reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t diff_count = 0;
  float max_diff = 0.0F;
  for (int i = 0; i < test_size; i++) {
    // For nextafter, we need exact bit equality or special handling for NaN
    if (std::isnan(expect[i])) {
      if (!std::isnan(y[i])) {
        diff_count++;
      }
    } else if (std::isnan(y[i])) {
      diff_count++;
    } else if (y[i] != expect[i]) {
      float diff = std::fabs(y[i] - expect[i]);
      max_diff = std::max(max_diff, diff);
      // For nextafter, we need exact match except for NaN cases
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0U) << "Max diff: " << max_diff << " of " << test_size;

  AscendC::GmFree(x1);
  AscendC::GmFree(x2);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendNextAfterStoreCode,
                         ::testing::Values(std::vector<int>{32, 16}, std::vector<int>{32, 18},
                                           std::vector<int>{512, 15}));
