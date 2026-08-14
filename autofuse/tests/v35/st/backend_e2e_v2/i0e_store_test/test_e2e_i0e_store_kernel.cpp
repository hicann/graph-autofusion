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
#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void i0e_store_test(GM_ADDR x1, GM_ADDR y1, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendI0eStoreCode : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

static float I0eReference(float x) {
  // Reference implementation using series expansion
  // I0e(x) = exp(-|x|) * I0(x), where I0 is the modified Bessel function
  const float abs_x = std::fabs(x);
  const float half_x = abs_x * 0.5F;
  float term = 1.0F;
  float sum = 1.0F;

  // Calculate I0(abs_x) using series expansion
  for (int k = 1; k <= 20; ++k) {
    const float scale = half_x / static_cast<float>(k);
    term *= scale * scale;
    sum += term;
  }

  // Apply exponential scaling: I0e(x) = exp(-|x|) * I0(x)
  return std::exp(-abs_x) * sum;
}

TEST_P(E2EBackendI0eStoreCode, CalculateCorrect) {
  auto test_shape = GetParam();
  uint64_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  float *x = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  float *y = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  float *expect = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));

  // Generate test data covering different ranges
  for (int i = 0; i < test_size; i++) {
    // Test values: small to medium inputs for numerical stability
    float val = static_cast<float>((i % 9) - 4) / 4.0F;
    x[i] = val;
    expect[i] = I0eReference(val);
  }

  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(i0e_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(x), reinterpret_cast<uint8_t *>(y),
              nullptr, reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (std::fabs(y[i] - expect[i]) > 1e-4F) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0U) << " of " << test_size;

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendI0eStoreCode,
                         ::testing::Values(std::vector<int>{32, 16}, std::vector<int>{32, 18},
                                           std::vector<int>{512, 15}));
