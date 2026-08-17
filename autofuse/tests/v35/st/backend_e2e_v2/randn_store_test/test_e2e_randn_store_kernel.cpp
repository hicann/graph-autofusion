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

#include <gtest/gtest.h>
#include <limits>

#include "autofuse_tiling_data.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void randn_store_test(GM_ADDR y1, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, AutofuseTilingData *tiling, uint32_t *workspaceSize, uint64_t *blockDim,
                                  uint32_t aiv_num, uint32_t ub_size);

class E2EBackendRandnStoreCode : public testing::Test, public testing::WithParamInterface<int> {};

TEST_P(E2EBackendRandnStoreCode, GenerateRandomIntegers) {
  auto test_size = GetParam();
  uint64_t block_dim = 48;

  AutofuseTilingData tiling_data;
  uint32_t *y = static_cast<uint32_t *>(AscendC::GmAlloc(test_size * sizeof(uint32_t) + 32));

  uint32_t ws_size = 0;
  AutofuseTiling(test_size, &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(randn_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(y), nullptr,
              reinterpret_cast<uint8_t *>(&tiling_data));

  // Verify generated random integers are in valid uint32_t range
  uint32_t zero_count = 0;
  for (int i = 0; i < test_size; i++) {
    // All values should be valid uint32_t (automatically true by type)
    if (y[i] == 0U) {
      zero_count++;
    }
  }

  // Since it's random, we expect some non-zero values (extremely unlikely all are zero)
  EXPECT_LT(zero_count, test_size) << "All generated values are zero (unlikely for random generator)";

  AscendC::GmFree(y);
}

INSTANTIATE_TEST_SUITE_P(GenerateWithDifferentSize, E2EBackendRandnStoreCode, ::testing::Values(32, 64, 128, 256, 512));
