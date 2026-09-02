/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <gtest/gtest.h>
#include <cmath>
#include "tikicpulib.h"
#include "autofuse_tiling_data.h"

extern "C" __global__ __aicore__ void fused_backend_elewise_test(GM_ADDR x0, GM_ADDR x1, GM_ADDR x2, GM_ADDR y0,
                                                                 GM_ADDR y1, GM_ADDR workspace, GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

namespace {
class E2E_FusedBackendElewise_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_FusedBackendElewise_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;

  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  float *x0 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *x1 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *x2 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);

  float *y0 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *y1 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);

  float *expect0 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *expect1 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);

  // Prepare test and expect data
  for (int i = 0; i < test_size; i++) {
    x0[i] = static_cast<float>(i);
    x1[i] = static_cast<float>(i);
    x2[i] = static_cast<float>(i);
    expect1[i] = x0[i] + x1[i];
    expect0[i] = (x0[i] + x1[i]) * 2 + x2[i];
  }

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  printf("g0_tiling_key: %d\n", tiling_data.graph0_tiling_key);
  printf("g1_tiling_key: %d\n", tiling_data.graph1_tiling_key);
  printf("g2_tiling_key: %d\n", tiling_data.graph2_tiling_key);
  printf("ws_size: %d, ws3_size: %d, ws5_size: %d\n", ws_size, tiling_data.workspace3, tiling_data.workspace5);
  float *workspace = (float *)AscendC::GmAlloc(ws_size + 32);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(fused_backend_elewise_test, tiling_data.block_dim, (uint8_t *)x0, (uint8_t *)x1, (uint8_t *)x2,
              (uint8_t *)y0, (uint8_t *)y1, (uint8_t *)workspace, (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    auto diff0 = (double)(y0[i] - expect0[i]);
    if (diff0 < -1e-5 || diff0 > 1e-5) {
      printf("i: %d, y0: %f, expect0: %f\n", i, y0[i], expect0[i]);
      diff_count++;
    }
  }
  for (int i = 0; i < test_size; i++) {
    auto diff1 = (double)(y1[i] - expect1[i]);
    if (diff1 < -1e-5 || diff1 > 1e-5) {
      printf("i: %d, y1: %f, expect1: %f\n", i, y1[i], expect1[i]);
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x0);
  AscendC::GmFree(x1);
  AscendC::GmFree(x2);
  AscendC::GmFree(y0);
  AscendC::GmFree(y1);
  AscendC::GmFree(expect0);
  AscendC::GmFree(expect1);
  AscendC::GmFree(workspace);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_FusedBackendElewise_Code,
                         ::testing::Values(std::vector<int>{2, 64}));

}  // namespace
