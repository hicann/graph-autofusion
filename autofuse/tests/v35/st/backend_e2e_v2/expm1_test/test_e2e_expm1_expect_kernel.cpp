/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void expm1_bf16_test(GM_ADDR x1, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, AutofuseTilingData *tiling,
                                  uint32_t *workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendExpm1Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

static size_t PrepareData(bfloat16_t *input, bfloat16_t *expect, int test_size) {
  std::mt19937 generator(1);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  for (int i = 0; i < test_size; i++) {
    input[i] = static_cast<bfloat16_t>(distribution(generator));
  }
  const std::vector<float> precision_inputs = {
      0.0f,
      -0.0f,
      std::ldexp(1.0f, -8),
      -std::ldexp(1.0f, -8),
      std::ldexp(1.0f, -11),
      -std::ldexp(1.0f, -11),
      std::ldexp(1.0f, -14),
      -std::ldexp(1.0f, -14),
      std::ldexp(1.0f, -24),
      -std::ldexp(1.0f, -24),
      std::ldexp(1.0f, -25),
      -std::ldexp(1.0f, -25),
  };
  for (size_t i = 0; i < precision_inputs.size(); ++i) {
    input[i] = static_cast<bfloat16_t>(precision_inputs[i]);
  }
  for (int i = 0; i < test_size; i++) {
    float exp_val = std::expm1(static_cast<float>(input[i]));
    expect[i] = static_cast<bfloat16_t>(exp_val);
  }
  return precision_inputs.size();
}

static uint32_t CountDifferences(const bfloat16_t *actual, const bfloat16_t *expect, int test_size,
                                 size_t precision_input_count) {
  const float EPS = 1e-2;
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    float actual_val = static_cast<float>(actual[i]);
    float expect_val = static_cast<float>(expect[i]);
    if (i < static_cast<int>(precision_input_count)) {
      if (actual_val != expect_val || (expect_val == 0.0f && std::signbit(actual_val) != std::signbit(expect_val))) {
        diff_count++;
      }
    } else if (std::fabs(actual_val - expect_val) > EPS) {
      diff_count++;
    }
  }
  return diff_count;
}

TEST_P(E2EBackendExpm1Code, CalculateCorrect) {
  auto test_shape = GetParam();
  uint64_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1] * test_shape[2];

  AutofuseTilingData tiling_data;
  bfloat16_t *input1 = (bfloat16_t *)AscendC::GmAlloc(test_size * sizeof(bfloat16_t) + 32);
  bfloat16_t *y = (bfloat16_t *)AscendC::GmAlloc(test_size * sizeof(bfloat16_t) + 32);
  bfloat16_t *expect = (bfloat16_t *)AscendC::GmAlloc(test_size * sizeof(bfloat16_t) + 32);
  size_t precision_input_count = PrepareData(input1, expect, test_size);

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], test_shape[2], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(expm1_bf16_test, tiling_data.block_dim, (uint8_t *)input1, (uint8_t *)y, nullptr,
              (uint8_t *)&tiling_data);

  uint32_t diff_count = CountDifferences(y, expect, test_size, precision_input_count);
  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(input1);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendExpm1Code, ::testing::Values(std::vector<int>{32, 16, 16}));
