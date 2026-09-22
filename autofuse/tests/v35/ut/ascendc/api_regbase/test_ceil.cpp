/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * test_ceil.cpp
 */

#include <cmath>
#include <random>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/ceil.h"

using namespace AscendC;

namespace af {

class TestRegbaseApiCeilUT : public testing::Test {
 protected:
  template <typename T>
  static void InvokeTensorTensorKernel(UnaryInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_x = xbuf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, param.x1, param.size);
    CeilExtend(l_y, l_x, l_tmp, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void AllocParam(UnaryInputParam<T> &param, uint32_t size) {
    param.size = size;
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
  }

  template <typename T>
  static void FreeTensorInput(UnaryInputParam<T> &param) {
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.x1);
  }

  // 期望值统一从存储后的输入（half 会先舍入）转 double 计算；ceil 为精确取整，期望应逐位一致
  template <typename T>
  static void FillExpect(UnaryInputParam<T> &param, uint32_t idx) {
    double src = static_cast<double>(param.x1[idx]);
    if (std::isnan(src)) {
      param.exp[idx] = static_cast<T>(std::numeric_limits<float>::quiet_NaN());
    } else {
      param.exp[idx] = static_cast<T>(std::ceil(src));
    }
  }

  template <typename T>
  static uint32_t Valid(UnaryInputParam<T> &param, double tol, double &maxRelErr) {
    uint32_t diff_count = 0;
    maxRelErr = 0.0;
    for (uint32_t i = 0; i < param.size; i++) {
      double y_val = static_cast<double>(param.y[i]);
      double exp_val = static_cast<double>(param.exp[i]);

      if (std::isnan(exp_val)) {
        if (!std::isnan(y_val)) {
          diff_count++;
          printf("diff at index %d: x: %e, y: %e, expect: NaN\n", i, static_cast<double>(param.x1[i]), y_val);
        }
      } else if (std::isinf(exp_val)) {
        if (y_val != exp_val) {
          diff_count++;
          printf("diff at index %d: x: %e, y: %e, expect: %cInf\n", i, static_cast<double>(param.x1[i]), y_val,
                 exp_val > 0 ? '+' : '-');
        }
      } else {
        double rel_err = std::abs(y_val - exp_val) / std::max(std::abs(exp_val), 1.0);
        if (rel_err > maxRelErr) {
          maxRelErr = rel_err;
        }
        if (rel_err > tol) {
          diff_count++;
          printf("diff at index %d: x: %.20e, y: %.20e, expect: %.20e, rel_err: %e\n", i,
                 static_cast<double>(param.x1[i]), y_val, exp_val, rel_err);
        }
      }
    }
    return diff_count;
  }

  template <typename T>
  static void RunAndCheck(UnaryInputParam<T> &param, double tol, const std::string &caseName) {
    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    double maxRelErr = 0.0;
    uint32_t diff_count = Valid(param, tol, maxRelErr);
    printf("[precision] Ceil<%s> %s size=%u maxRelErr=%.6e tol=%.1e\n", sizeof(T) == sizeof(half) ? "half" : "float",
           caseName.c_str(), param.size, maxRelErr, tol);
    EXPECT_EQ(diff_count, 0);

    FreeTensorInput(param);
  }

  // 常规区间 [-100, 100]：随机值几乎必含小数部分，正负对称
  template <typename T>
  static void CeilTest(uint32_t size) {
    UnaryInputParam<T> param{};
    AllocParam(param, size);
    std::mt19937 eng(1);
    for (uint32_t i = 0; i < size; i++) {
      std::uniform_real_distribution<double> distr(-100.0, 100.0);
      param.x1[i] = static_cast<T>(distr(eng));
      FillExpect(param, i);
    }
    RunAndCheck<T>(param, 1e-6, "TensorTensor");
  }

  // 整数边界：±n、±n±0.5、±n±1e-3 精确覆盖取整切换点。
  // 负小数（如 -2.5 → -2）是符号回填路径的关键校验点：若实现退化为 floor 会得到 -3
  template <typename T>
  static void CeilNearIntegerTest(uint32_t size) {
    UnaryInputParam<T> param{};
    AllocParam(param, size);
    std::mt19937 eng(2);
    for (uint32_t i = 0; i < size; i++) {
      std::uniform_int_distribution<int> baseDistr(-500, 500);
      double base = static_cast<double>(baseDistr(eng));
      // 偏移循环覆盖：+0.5 / -0.5 / +小量 / -小量 / 恰为整数
      const double offsets[] = {0.5, -0.5, 1e-3, -1e-3, 0.0};
      param.x1[i] = static_cast<T>(base + offsets[i % 5]);
      FillExpect(param, i);
    }
    RunAndCheck<T>(param, 1e-6, "NearInteger");
  }

  // 特殊值：±0（期望 ±0 透传为整数 0）、±0.5、±1、±2.5、大整数（已无可整部分）、±Inf、NaN
  template <typename T>
  static void CeilSpecialValuesTest(uint32_t size) {
    UnaryInputParam<T> param{};
    AllocParam(param, size);

    std::vector<float> special_values = {0.0f,
                                         -0.0f,
                                         0.5f,
                                         -0.5f,
                                         1.0f,
                                         -1.0f,
                                         2.5f,
                                         -2.5f,
                                         1024.0f,
                                         -1024.0f,
                                         2047.5f,
                                         -2047.5f,
                                         65504.0f,
                                         -65504.0f,
                                         1e20f,
                                         -1e20f,
                                         std::numeric_limits<float>::infinity(),
                                         -std::numeric_limits<float>::infinity(),
                                         std::numeric_limits<float>::quiet_NaN()};

    uint32_t idx = 0;
    for (auto val : special_values) {
      if (idx >= param.size) {
        break;
      }
      param.x1[idx] = static_cast<T>(val);
      FillExpect(param, idx);
      idx++;
    }

    std::mt19937 eng(3);
    while (idx < param.size) {
      std::uniform_real_distribution<double> distr(-10.0, 10.0);
      param.x1[idx] = static_cast<T>(distr(eng));
      FillExpect(param, idx);
      idx++;
    }
    RunAndCheck<T>(param, 1e-6, "SpecialValues");
  }
};

TEST_F(TestRegbaseApiCeilUT, Ceil_TensorTensor_Test) {
  CeilTest<float>(ONE_BLK_SIZE / sizeof(float));
  CeilTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  CeilTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  CeilTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  CeilTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  CeilTest<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  CeilTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                   (ONE_BLK_SIZE - sizeof(float))) /
                  2 / sizeof(float));
  CeilTest<half>(ONE_BLK_SIZE / sizeof(half));
  CeilTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  CeilTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  CeilTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  CeilTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));
}

TEST_F(TestRegbaseApiCeilUT, Ceil_NearInteger_Test) {
  CeilNearIntegerTest<float>(ONE_BLK_SIZE / sizeof(float));
  CeilNearIntegerTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  CeilNearIntegerTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  CeilNearIntegerTest<half>(ONE_BLK_SIZE / sizeof(half));
  CeilNearIntegerTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
}

TEST_F(TestRegbaseApiCeilUT, Ceil_SpecialValues_Test) {
  CeilSpecialValuesTest<float>(ONE_BLK_SIZE / sizeof(float));
  CeilSpecialValuesTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  CeilSpecialValuesTest<half>(ONE_BLK_SIZE / sizeof(half));
  CeilSpecialValuesTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
}

}  // namespace af
