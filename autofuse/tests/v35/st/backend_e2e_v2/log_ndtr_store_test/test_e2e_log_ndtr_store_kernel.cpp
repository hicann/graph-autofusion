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

extern "C" __global__ __aicore__ void log_ndtr_store_test(GM_ADDR x1, GM_ADDR y1, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendLogNdtrStoreCode : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

static float LogNdtrReference(float x) {
  // Handle special values before the left-tail formula to avoid inf * 0.
  if (std::isnan(x)) {
    return std::nan("");
  }
  if (std::isinf(x) && x > 0) {
    return -0.0f;  // log(ndtr(+inf)) = log(1) = 0
  }
  if (std::isinf(x) && x < 0) {
    return -INFINITY;  // log(ndtr(-inf)) = log(0) = -inf
  }

  const double dx = static_cast<double>(x);
  const double t = dx * 0.7071067811865475;
  if (dx < -1.0) {
    // Left tail: log(erfcx(-t) / 2) - t^2.
    const double erfcx = std::exp(t * t) * std::erfc(-t);
    return static_cast<float>(std::log(erfcx * 0.5) - t * t);
  }

  // Right tail: log(1 - erfc(t) / 2).
  return static_cast<float>(std::log1p(-std::erfc(t) * 0.5));
}

TEST_P(E2EBackendLogNdtrStoreCode, CalculateCorrect) {
  auto test_shape = GetParam();
  uint64_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  float *x = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  float *y = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  float *expect = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));

  // Generate test data covering different ranges
  for (int i = 0; i < test_size; i++) {
    // Test values: negative, zero, positive including large values
    float val = static_cast<float>((i % 30) - 15) / 2.0f;  // Range: [-7.5, 7.0]
    x[i] = val;
    expect[i] = LogNdtrReference(val);
  }

  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(log_ndtr_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(x),
              reinterpret_cast<uint8_t *>(y), nullptr, reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t diff_count = 0;
  float max_diff = 0.0F;
  for (int i = 0; i < test_size; i++) {
    float diff = std::fabs(y[i] - expect[i]);
    max_diff = std::max(max_diff, diff);
    // Use appropriate tolerance for log_ndtr due to numerical precision
    float tolerance = std::max(1e-3F, std::fabs(expect[i]) * 1e-2F);
    if (diff > tolerance) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0U) << "Max diff: " << max_diff << " of " << test_size;

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendLogNdtrStoreCode,
                         ::testing::Values(std::vector<int>{32, 16}, std::vector<int>{32, 18},
                                           std::vector<int>{512, 15}));
