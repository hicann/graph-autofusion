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

extern "C" __global__ __aicore__ void rand_store_test(GM_ADDR y1, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, AutofuseTilingData *tiling, uint32_t *workspaceSize, uint64_t *blockDim,
                                  uint32_t aiv_num, uint32_t ub_size);

class E2EBackendRandStoreCode : public testing::Test, public testing::WithParamInterface<int> {};

TEST_P(E2EBackendRandStoreCode, GenerateRandomNumbers) {
  auto test_size = GetParam();
  uint64_t block_dim = 48;

  AutofuseTilingData tiling_data;
  float *y = static_cast<float *>(AscendC::GmAlloc(test_size * sizeof(float) + 32));

  uint32_t ws_size = 0;
  AutofuseTiling(test_size, &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(rand_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(y), nullptr,
              reinterpret_cast<uint8_t *>(&tiling_data));

  // Verify generated random numbers are in valid range [0, 1)
  uint32_t invalid_count = 0;
  uint32_t zero_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (y[i] < 0.0F || y[i] >= 1.0F) {
      invalid_count++;
    }
    if (y[i] == 0.0F) {
      zero_count++;
    }
  }

  // All values should be in [0, 1)
  EXPECT_EQ(invalid_count, 0U) << "Generated values outside [0, 1) range";

  // Since it's random, we expect some non-zero values (extremely unlikely all are zero)
  EXPECT_LT(zero_count, test_size) << "All generated values are zero (unlikely for random generator)";

  AscendC::GmFree(y);
}

INSTANTIATE_TEST_SUITE_P(GenerateWithDifferentSize, E2EBackendRandStoreCode, ::testing::Values(32, 64, 128, 256, 512));
