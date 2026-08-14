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

extern "C" __global__ __aicore__ void hermite_polynomial_h_store_test(GM_ADDR x1, GM_ADDR n1, GM_ADDR y1,
                                                                      GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendHermitePolynomialHStoreCode : public testing::Test,
                                              public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2EBackendHermitePolynomialHStoreCode, CalculateCorrect) {
  auto test_shape = GetParam();
  uint64_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];
  constexpr int32_t n_value = 3;

  AutofuseTilingData tiling_data;
  float *x = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  int32_t *n = static_cast<int32_t *>(AscendC::GmAlloc(test_size * sizeof(int32_t) + 32));
  float *y = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));

  for (int i = 0; i < test_size; i++) {
    x[i] = static_cast<float>(i % 100) / 100.0F;  // x in [0, 1]
    n[i] = n_value;
  }

  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(hermite_polynomial_h_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(x),
              reinterpret_cast<uint8_t *>(n), reinterpret_cast<uint8_t *>(y), nullptr,
              reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    // H_3(x) = 8x^3 - 12x.
    const float expect = 8.0F * x[i] * x[i] * x[i] - 12.0F * x[i];
    if (!std::isfinite(y[i]) || std::fabs(y[i] - expect) > 1e-4F) {
      diff_count++;
    }
  }
  EXPECT_EQ(diff_count, 0U) << " of " << test_size;

  AscendC::GmFree(x);
  AscendC::GmFree(n);
  AscendC::GmFree(y);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendHermitePolynomialHStoreCode,
                         ::testing::Values(std::vector<int>{32, 16}, std::vector<int>{32, 18},
                                           std::vector<int>{512, 15}));
