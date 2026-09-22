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
 * test_acosh.cpp
 */

#include <cmath>
#include <random>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/acosh.h"

using namespace AscendC;

namespace af {

class TestRegbaseApiAcoshUT : public testing::Test {
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
    AcoshExtend(l_y, l_x, l_tmp, param.size);
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

  // 期望值统一从存储后的输入（half 会先舍入）转 double 计算
  template <typename T>
  static void FillExpect(UnaryInputParam<T> &param, uint32_t idx) {
    double src = static_cast<double>(param.x1[idx]);
    if (src < 1.0) {
      param.exp[idx] = static_cast<T>(std::numeric_limits<float>::quiet_NaN());
    } else {
      param.exp[idx] = static_cast<T>(std::acosh(src));
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
        double rel_err = std::abs(y_val - exp_val) / std::max(std::abs(exp_val), 1e-6);
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
    printf("[precision] Acosh<%s> %s size=%u maxRelErr=%.6e tol=%.1e\n", sizeof(T) == sizeof(half) ? "half" : "float",
           caseName.c_str(), param.size, maxRelErr, tol);
    EXPECT_EQ(diff_count, 0);

    FreeTensorInput(param);
  }

  // 常规区间 [1, 100]，覆盖补偿分支（x < 10）与直接公式分支
  template <typename T>
  static void AcoshTest(uint32_t size) {
    UnaryInputParam<T> param{};
    AllocParam(param, size);
    std::mt19937 eng(1);
    for (uint32_t i = 0; i < size; i++) {
      std::uniform_real_distribution<double> distr(1.0, 100.0);
      param.x1[i] = static_cast<T>(distr(eng));
      FillExpect(param, i);
    }
    RunAndCheck<T>(param, sizeof(T) == sizeof(half) ? 5e-3 : 1e-4, "TensorTensor");
  }

  // 近 1 区间 [1, 1.01]，acosh 在 x→1+ 时导数无界，主要验证 log1p 补偿分支
  template <typename T>
  static void AcoshNearOneBranchTest(uint32_t size) {
    UnaryInputParam<T> param{};
    AllocParam(param, size);
    std::mt19937 eng(2);
    for (uint32_t i = 0; i < size; i++) {
      std::uniform_real_distribution<double> distr(0.0, 0.01);
      param.x1[i] = static_cast<T>(1.0 + distr(eng));
      FillExpect(param, i);
    }
    RunAndCheck<T>(param, sizeof(T) == sizeof(half) ? 5e-3 : 1e-4, "NearOneBranch");
  }

  // 大 x 区间（对数均匀分布到 3e38），验证 x*x 溢出后的渐近式分支 ln(x)+ln(2)
  template <typename T>
  static void AcoshLargeBranchTest(uint32_t size) {
    UnaryInputParam<T> param{};
    AllocParam(param, size);
    std::mt19937 eng(3);
    for (uint32_t i = 0; i < size; i++) {
      std::uniform_real_distribution<double> distr(0.0, 38.5);
      param.x1[i] = static_cast<T>(std::pow(10.0, distr(eng)));
      FillExpect(param, i);
    }
    RunAndCheck<T>(param, sizeof(T) == sizeof(half) ? 5e-3 : 1e-4, "LargeBranch");
  }

  template <typename T>
  static void AcoshSpecialValuesTest(uint32_t size) {
    UnaryInputParam<T> param{};
    AllocParam(param, size);

    // 覆盖：定义域边界 1、域外（期望 NaN）、分段阈值、渐近分支、inf/NaN
    std::vector<float> special_values = {1.0f,
                                         0.9999f,
                                         10.0f,
                                         1e19f,
                                         1.8446744e19f,  // ~sqrt(FLT_MAX) 分段阈值
                                         1e20f,
                                         std::numeric_limits<float>::max(),
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
      std::uniform_real_distribution<double> distr(1.0, 100.0);
      param.x1[idx] = static_cast<T>(distr(eng));
      FillExpect(param, idx);
      idx++;
    }
    RunAndCheck<T>(param, sizeof(T) == sizeof(half) ? 5e-3 : 1e-4, "SpecialValues");
  }

  // 分段阈值边界：acosh 值域阈值 acosh(10) 与 acosh(sqrt(FLT_MAX)) 附近
  template <typename T>
  static void AcoshThresholdBoundaryTest(uint32_t size) {
    UnaryInputParam<T> param{};
    AllocParam(param, size);

    std::vector<float> boundary_values = {9.5f,  9.999f,  10.0f,         10.001f,  10.5f,
                                          1e19f, 1.8e19f, 1.8446744e19f, 1.85e19f, 1e20f};

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
      std::uniform_real_distribution<double> distr(1.0, 100.0);
      param.x1[idx] = static_cast<T>(distr(eng));
      FillExpect(param, idx);
      idx++;
    }
    RunAndCheck<T>(param, sizeof(T) == sizeof(half) ? 5e-3 : 1e-4, "ThresholdBoundary");
  }
};

TEST_F(TestRegbaseApiAcoshUT, Acosh_TensorTensor_Test) {
  AcoshTest<float>(ONE_BLK_SIZE / sizeof(float));
  AcoshTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  AcoshTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  AcoshTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  AcoshTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  AcoshTest<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  AcoshTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                    (ONE_BLK_SIZE - sizeof(float))) /
                   2 / sizeof(float));
  AcoshTest<half>(ONE_BLK_SIZE / sizeof(half));
  AcoshTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  AcoshTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  AcoshTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  AcoshTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));
}

TEST_F(TestRegbaseApiAcoshUT, Acosh_NearOneBranch_Test) {
  AcoshNearOneBranchTest<float>(ONE_BLK_SIZE / sizeof(float));
  AcoshNearOneBranchTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  AcoshNearOneBranchTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  AcoshNearOneBranchTest<half>(ONE_BLK_SIZE / sizeof(half));
  AcoshNearOneBranchTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
}

TEST_F(TestRegbaseApiAcoshUT, Acosh_LargeBranch_Test) {
  AcoshLargeBranchTest<float>(ONE_BLK_SIZE / sizeof(float));
  AcoshLargeBranchTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  AcoshLargeBranchTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
}

TEST_F(TestRegbaseApiAcoshUT, Acosh_SpecialValues_Test) {
  AcoshSpecialValuesTest<float>(ONE_BLK_SIZE / sizeof(float));
  AcoshSpecialValuesTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  AcoshSpecialValuesTest<half>(ONE_BLK_SIZE / sizeof(half));
  AcoshSpecialValuesTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
}

TEST_F(TestRegbaseApiAcoshUT, Acosh_ThresholdBoundary_Test) {
  AcoshThresholdBoundaryTest<float>(ONE_BLK_SIZE / sizeof(float));
  AcoshThresholdBoundaryTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  AcoshThresholdBoundaryTest<half>(ONE_BLK_SIZE / sizeof(half));
}

}  // namespace af
