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

extern "C" __global__ __aicore__ void chebyshev_polynomial_u_store_test(GM_ADDR x1, GM_ADDR y1, GM_ADDR workspace,
                                                                        GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendChebyshevPolynomialUStoreCode : public testing::Test,
                                                public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2EBackendChebyshevPolynomialUStoreCode, CalculateCorrect) {
  auto test_shape = GetParam();
  uint64_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  float *x = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));
  float *y = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));

  for (int i = 0; i < test_size; i++) {
    x[i] = static_cast<float>(i % 100) / 100.0F;  // x in [0, 1]
  }

  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(chebyshev_polynomial_u_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(x),
              reinterpret_cast<uint8_t *>(y), nullptr, reinterpret_cast<uint8_t *>(&tiling_data));

  // Basic sanity: output should not be NaN for valid inputs
  uint32_t invalid_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (std::isnan(y[i]) || std::isinf(y[i])) {
      invalid_count++;
    }
  }
  EXPECT_EQ(invalid_count, 0U) << " of " << test_size;

  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendChebyshevPolynomialUStoreCode,
                         ::testing::Values(std::vector<int>{32, 16}, std::vector<int>{32, 18},
                                           std::vector<int>{512, 15}));
