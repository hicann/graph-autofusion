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

extern "C" __global__ __aicore__ void poly_gamma_store_test(GM_ADDR x, GM_ADDR n, GM_ADDR y, GM_ADDR workspace,
                                                            GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendPolyGammaStoreCode : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

// Reference implementation of polygamma for integer order n
// polygamma(n, x) = (-1)^(n+1) * n! * zeta(n+1, x)
static float PolyGammaReference(int n, float x) {
  // Handle special cases
  if (std::isnan(x) || std::isinf(x)) {
    return std::nan("");
  }
  if (x == 0.0f && n > 0) {
    return (n % 2 == 1) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
  }
  if (n < 0) {
    return std::nan("");  // Only non-negative integer orders supported
  }

  // n=0: Digamma (psi function)
  if (n == 0) {
    // Simple approximation for digamma
    if (x > 0) {
      float result = -0.5772156649f;  // Euler-Mascheroni constant
      float term = x;
      for (int i = 1; i < 100; i++) {
        float x_plus_i = x + i;
        result += 1.0f / x_plus_i;
        term += 1.0f / (x_plus_i * x_plus_i);
      }
      return result - 1.0f / x;
    }
    return std::nan("");
  }

  // n=1: Trigamma
  if (n == 1) {
    if (x > 0) {
      float sum = 0.0f;
      for (int i = 0; i < 1000; i++) {
        float xi = x + i;
        sum += 1.0f / (xi * xi);
      }
      return sum;
    }
    return std::nan("");
  }

  // For n>=2: use approximation polygamma(n, x) ≈ (-1)^(n+1) * n! / x^(n+1) for large x
  // This is a simplified reference for validation
  if (n >= 2 && x > 1.0f) {
    float sign = (n % 2 == 0) ? -1.0f : 1.0f;

    // Calculate factorial
    float factorial = 1.0f;
    for (int i = 2; i <= n; i++) {
      factorial *= i;
    }

    // Approximate for large x: polygamma(n, x) ≈ (-1)^(n+1) * n! / x^(n+1)
    float x_pow = x;
    for (int i = 1; i <= n + 1; i++) {
      x_pow *= x;
    }

    return sign * factorial / x_pow;
  }

  // For other cases, return nan as this is a simplified reference
  return std::nan("");
}

TEST_P(E2EBackendPolyGammaStoreCode, CalculateCorrect) {
  auto test_shape = GetParam();
  uint64_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];
  constexpr int32_t n_value = 1;

  AutofuseTilingData tiling_data;
  float *x = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  int32_t *n = static_cast<int32_t *>(AscendC::GmAlloc(test_size * sizeof(int32_t) + 32));
  float *y = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  float *expect = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));

  // PolyGamma applies one scalar order to the entire input tensor.
  for (int i = 0; i < test_size; i++) {
    x[i] = 0.5F + static_cast<float>(i % 20) * 0.5F;
    n[i] = n_value;
    expect[i] = PolyGammaReference(n_value, x[i]);
  }

  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(poly_gamma_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(x),
              reinterpret_cast<uint8_t *>(n), reinterpret_cast<uint8_t *>(y), nullptr,
              reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t diff_count = 0;
  float max_diff = 0.0F;
  for (int i = 0; i < test_size; i++) {
    if (std::isnan(expect[i])) {
      if (!std::isnan(y[i])) {
        diff_count++;
      }
    } else if (std::isnan(y[i])) {
      diff_count++;
    } else if (std::isinf(expect[i])) {
      if (!std::isinf(y[i]) || (expect[i] > 0) != (y[i] > 0)) {
        diff_count++;
      }
    } else {
      float diff = std::fabs(y[i] - expect[i]);
      max_diff = std::max(max_diff, diff);
      // Use appropriate tolerance for polygamma
      float tolerance = std::max(1e-3F, std::fabs(expect[i]) * 1e-1F);
      if (diff > tolerance) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0U) << "Max diff: " << max_diff << " of " << test_size;

  AscendC::GmFree(x);
  AscendC::GmFree(n);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendPolyGammaStoreCode,
                         ::testing::Values(std::vector<int>{32, 16}, std::vector<int>{32, 18},
                                           std::vector<int>{512, 15}));
