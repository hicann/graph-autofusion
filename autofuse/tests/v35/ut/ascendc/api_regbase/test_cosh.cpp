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
 * test_cosh.cpp
 */

#include <cmath>
#include <random>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/cosh.h"

using namespace AscendC;

namespace af {

class TestRegbaseApiCoshUT : public testing::Test {
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
    CoshExtend(l_y, l_x, l_tmp, param.size);
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

  // 期望值统一从存储后的输入（half 会先舍入）转 double 计算；fp32/half 溢出自然反映到期望值
  template <typename T>
  static void FillExpect(UnaryInputParam<T> &param, uint32_t idx) {
    double src = static_cast<double>(param.x1[idx]);
    if (std::isnan(src)) {
      param.exp[idx] = static_cast<T>(std::numeric_limits<float>::quiet_NaN());
    } else {
      param.exp[idx] = static_cast<T>(std::cosh(src));
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
    printf("[precision] Cosh<%s> %s size=%u maxRelErr=%.6e tol=%.1e\n", sizeof(T) == sizeof(half) ? "half" : "float",
           caseName.c_str(), param.size, maxRelErr, tol);
    EXPECT_EQ(diff_count, 0);

    FreeTensorInput(param);
  }

  // 常规区间 [-5, 5]
  template <typename T>
  static void CoshTest(uint32_t size) {
    UnaryInputParam<T> param{};
    AllocParam(param, size);
    std::mt19937 eng(1);
    for (uint32_t i = 0; i < size; i++) {
      std::uniform_real_distribution<double> distr(-5.0, 5.0);
      param.x1[i] = static_cast<T>(distr(eng));
      FillExpect(param, i);
    }
    RunAndCheck<T>(param, sizeof(T) == sizeof(half) ? 5e-3 : 1e-4, "TensorTensor");
  }

  // 宽幅区间 [-89.5, 89.5]：正负对称（偶函数）、大参数 Cody-Waite 缩减
  template <typename T>
  static void CoshWideRangeTest(uint32_t size) {
    UnaryInputParam<T> param{};
    AllocParam(param, size);
    std::mt19937 eng(2);
    for (uint32_t i = 0; i < size; i++) {
      std::uniform_real_distribution<double> distr(-89.5, 89.5);
      param.x1[i] = static_cast<T>(distr(eng));
      FillExpect(param, i);
    }
    RunAndCheck<T>(param, sizeof(T) == sizeof(half) ? 5e-3 : 1e-4, "WideRange");
  }

  // 大参数分支 [88.5, 89.9]：n = trunc(|x|·log2e) 超过 126 触发钳位，结果逼近 fp32 溢出
  template <typename T>
  static void CoshLargeBranchTest(uint32_t size) {
    UnaryInputParam<T> param{};
    AllocParam(param, size);
    std::mt19937 eng(3);
    for (uint32_t i = 0; i < size; i++) {
      std::uniform_real_distribution<double> distr(88.5, 89.9);
      double mag = distr(eng);
      param.x1[i] = static_cast<T>((i % 2 == 0) ? mag : -mag);
      FillExpect(param, i);
    }
    RunAndCheck<T>(param, sizeof(T) == sizeof(half) ? 5e-3 : 1e-4, "LargeBranch");
  }

  template <typename T>
  static void CoshSpecialValuesTest(uint32_t size) {
    UnaryInputParam<T> param{};
    AllocParam(param, size);

    // 覆盖：±0（期望 1）、偶函数对称、90 溢出阈值两侧、inf/NaN
    std::vector<float> special_values = {0.0f,
                                         -0.0f,
                                         1.0f,
                                         -1.0f,
                                         89.9f,
                                         -89.9f,
                                         90.0f,
                                         -90.0f,
                                         100.0f,
                                         -100.0f,
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

    std::mt19937 eng(4);
    while (idx < param.size) {
      std::uniform_real_distribution<double> distr(-5.0, 5.0);
      param.x1[idx] = static_cast<T>(distr(eng));
      FillExpect(param, idx);
      idx++;
    }
    RunAndCheck<T>(param, sizeof(T) == sizeof(half) ? 5e-3 : 1e-4, "SpecialValues");
  }

  // 溢出阈值边界：|x| ≥ 90 强制 +inf，[88.7, 90) 依赖 2·(e^x/4) 自然溢出
  template <typename T>
  static void CoshOverflowBoundaryTest(uint32_t size) {
    UnaryInputParam<T> param{};
    AllocParam(param, size);

    std::vector<float> boundary_values = {87.5f, 88.0f,  88.5f,  89.0f,  89.5f,  89.9f, 90.0f,
                                          90.1f, -87.5f, -88.5f, -89.9f, -90.0f, -90.1f};

    uint32_t idx = 0;
    for (auto val : boundary_values) {
      if (idx >= param.size) {
        break;
      }
      param.x1[idx] = static_cast<T>(val);
      FillExpect(param, idx);
      idx++;
    }

    std::mt19937 eng(5);
    while (idx < param.size) {
      std::uniform_real_distribution<double> distr(-5.0, 5.0);
      param.x1[idx] = static_cast<T>(distr(eng));
      FillExpect(param, idx);
      idx++;
    }
    RunAndCheck<T>(param, sizeof(T) == sizeof(half) ? 5e-3 : 1e-4, "OverflowBoundary");
  }
};

TEST_F(TestRegbaseApiCoshUT, Cosh_TensorTensor_Test) {
  CoshTest<float>(ONE_BLK_SIZE / sizeof(float));
  CoshTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  CoshTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  CoshTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  CoshTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  CoshTest<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  CoshTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                   (ONE_BLK_SIZE - sizeof(float))) /
                  2 / sizeof(float));
  CoshTest<half>(ONE_BLK_SIZE / sizeof(half));
  CoshTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  CoshTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  CoshTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  CoshTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));
}

TEST_F(TestRegbaseApiCoshUT, Cosh_WideRange_Test) {
  CoshWideRangeTest<float>(ONE_BLK_SIZE / sizeof(float));
  CoshWideRangeTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  CoshWideRangeTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
}

TEST_F(TestRegbaseApiCoshUT, Cosh_LargeBranch_Test) {
  CoshLargeBranchTest<float>(ONE_BLK_SIZE / sizeof(float));
  CoshLargeBranchTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  CoshLargeBranchTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
}

TEST_F(TestRegbaseApiCoshUT, Cosh_SpecialValues_Test) {
  CoshSpecialValuesTest<float>(ONE_BLK_SIZE / sizeof(float));
  CoshSpecialValuesTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  CoshSpecialValuesTest<half>(ONE_BLK_SIZE / sizeof(half));
  CoshSpecialValuesTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
}

TEST_F(TestRegbaseApiCoshUT, Cosh_OverflowBoundary_Test) {
  CoshOverflowBoundaryTest<float>(ONE_BLK_SIZE / sizeof(float));
  CoshOverflowBoundaryTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  CoshOverflowBoundaryTest<half>(ONE_BLK_SIZE / sizeof(half));
}

}  // namespace af
