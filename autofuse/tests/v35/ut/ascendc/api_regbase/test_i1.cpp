/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * test_i1.cpp
 */

#include <cmath>
#include <random>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/i1.h"

using namespace AscendC;

namespace af {

class TestRegbaseApiI1UT : public testing::Test {
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
    I1Extend(l_y, l_x, l_tmp, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void CreateTensorInput(UnaryInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    std::mt19937 eng(1);

    for (uint32_t i = 0; i < param.size; i++) {
      std::uniform_real_distribution distr(-20.0f, 20.0f);
      param.x1[i] = static_cast<T>(distr(eng));
      double bessel_val = std::cyl_bessel_i(1, std::fabs(param.x1[i]));
      param.exp[i] = static_cast<T>(std::copysign(bessel_val, param.x1[i]));
    }
  }

  template <typename T>
  static void CreateSmallBranchInput(UnaryInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    std::mt19937 eng(2);

    for (uint32_t i = 0; i < param.size; i++) {
      std::uniform_real_distribution distr(-8.9f, 8.9f);
      param.x1[i] = static_cast<T>(distr(eng));
      double bessel_val = std::cyl_bessel_i(1, std::fabs(param.x1[i]));
      param.exp[i] = static_cast<T>(std::copysign(bessel_val, param.x1[i]));
    }
  }

  template <typename T>
  static void CreateLargeBranchInput(UnaryInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    std::mt19937 eng(3);

    for (uint32_t i = 0; i < param.size; i++) {
      if (i % 2 == 0) {
        std::uniform_real_distribution distr(9.1f, 20.0f);
        param.x1[i] = static_cast<T>(distr(eng));
      } else {
        std::uniform_real_distribution distr(-20.0f, -9.1f);
        param.x1[i] = static_cast<T>(distr(eng));
      }
      double bessel_val = std::cyl_bessel_i(1, std::fabs(param.x1[i]));
      param.exp[i] = static_cast<T>(std::copysign(bessel_val, param.x1[i]));
    }
  }

  template <typename T>
  static void CreateSpecialValuesInput(UnaryInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::vector<float> special_values = {0.0f,
                                         -0.0f,
                                         1.0f,
                                         -1.0f,
                                         9.0f,
                                         -9.0f,
                                         std::numeric_limits<float>::infinity(),
                                         -std::numeric_limits<float>::infinity(),
                                         std::numeric_limits<float>::quiet_NaN()};

    uint32_t idx = 0;
    for (auto val : special_values) {
      if (idx < param.size) {
        param.x1[idx] = static_cast<T>(val);
        if (std::isnan(val)) {
          param.exp[idx] = static_cast<T>(std::numeric_limits<float>::quiet_NaN());
        } else if (std::isinf(val)) {
          param.exp[idx] = static_cast<T>(val);
        } else {
          double bessel_val = std::cyl_bessel_i(1, std::fabs(val));
          param.exp[idx] = static_cast<T>(std::copysign(bessel_val, val));
        }
        idx++;
      }
    }

    std::mt19937 eng(4);
    std::uniform_real_distribution distr(-20.0f, 20.0f);
    while (idx < param.size) {
      param.x1[idx] = static_cast<T>(distr(eng));
      double bessel_val = std::cyl_bessel_i(1, std::fabs(param.x1[idx]));
      param.exp[idx] = static_cast<T>(std::copysign(bessel_val, param.x1[idx]));
      idx++;
    }
  }

  template <typename T>
  static uint32_t Valid(UnaryInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      double y_val = static_cast<double>(param.y[i]);
      double exp_val = static_cast<double>(param.exp[i]);

      if (std::isnan(exp_val)) {
        if (!std::isnan(y_val)) {
          diff_count++;
          printf("diff at index %d: x: %.20e, y: %.20e, expect: NaN\n", i, static_cast<float>(param.x1[i]),
                 static_cast<float>(param.y[i]));
        }
      } else if (std::isinf(exp_val)) {
        if (y_val != exp_val) {
          diff_count++;
          printf("diff at index %d: x: %.20e, y: %.20e, expect: %cInf\n", i, static_cast<float>(param.x1[i]),
                 static_cast<float>(param.y[i]), exp_val > 0 ? '+' : '-');
        }
      } else {
        double rel_err = std::abs(y_val - exp_val) / std::max(std::abs(exp_val), 1.0);
        if (rel_err > 1e-2) {
          diff_count++;
          printf("diff at index %d: x: %.20e, y: %.20e, expect: %.20e, rel_err: %f\n", i,
                 static_cast<float>(param.x1[i]), static_cast<float>(param.y[i]), static_cast<float>(param.exp[i]),
                 static_cast<float>(rel_err));
        }
      }
    }
    return diff_count;
  }

  template <typename T>
  static void FreeTensorInput(UnaryInputParam<T> &param) {
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.x1);
  }

  template <typename T>
  static void I1Test(uint32_t size) {
    UnaryInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);

    FreeTensorInput(param);
  }

  template <typename T>
  static void I1SmallBranchTest(uint32_t size) {
    UnaryInputParam<T> param{};
    param.size = size;
    CreateSmallBranchInput(param);

    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);

    FreeTensorInput(param);
  }

  template <typename T>
  static void I1LargeBranchTest(uint32_t size) {
    UnaryInputParam<T> param{};
    param.size = size;
    CreateLargeBranchInput(param);

    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);

    FreeTensorInput(param);
  }

  template <typename T>
  static void I1SpecialValuesTest(uint32_t size) {
    UnaryInputParam<T> param{};
    param.size = size;
    CreateSpecialValuesInput(param);

    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);

    FreeTensorInput(param);
  }
};

TEST_F(TestRegbaseApiI1UT, I1_TensorTensor_Test) {
  I1Test<float>(ONE_BLK_SIZE / sizeof(float));
  I1Test<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  I1Test<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  I1Test<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  I1Test<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  I1Test<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  I1Test<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                 (ONE_BLK_SIZE - sizeof(float))) /
                2 / sizeof(float));
}

TEST_F(TestRegbaseApiI1UT, I1_SmallBranch_Test) {
  I1SmallBranchTest<float>(ONE_BLK_SIZE / sizeof(float));
  I1SmallBranchTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  I1SmallBranchTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
}

TEST_F(TestRegbaseApiI1UT, I1_LargeBranch_Test) {
  I1LargeBranchTest<float>(ONE_BLK_SIZE / sizeof(float));
  I1LargeBranchTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  I1LargeBranchTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
}

TEST_F(TestRegbaseApiI1UT, I1_SpecialValues_Test) {
  I1SpecialValuesTest<float>(ONE_BLK_SIZE / sizeof(float));
  I1SpecialValuesTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
}

TEST_F(TestRegbaseApiI1UT, I1_ThresholdBoundary_Test) {
  UnaryInputParam<float> param{};
  param.size = ONE_BLK_SIZE / sizeof(float);
  param.y = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));
  param.exp = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));
  param.x1 = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));

  std::vector<float> boundary_values = {8.9f, 9.0f, 9.1f, -8.9f, -9.0f, -9.1f};

  uint32_t idx = 0;
  for (auto val : boundary_values) {
    if (idx < param.size) {
      param.x1[idx] = val;
      double bessel_val = std::cyl_bessel_i(1, std::fabs(val));
      param.exp[idx] = static_cast<float>(std::copysign(bessel_val, val));
      idx++;
    }
  }

  std::mt19937 eng(5);
  std::uniform_real_distribution distr(-20.0f, 20.0f);
  while (idx < param.size) {
    param.x1[idx] = distr(eng);
    double bessel_val = std::cyl_bessel_i(1, std::fabs(param.x1[idx]));
    param.exp[idx] = static_cast<float>(std::copysign(bessel_val, param.x1[idx]));
    idx++;
  }

  auto kernel = [&param] { InvokeTensorTensorKernel(param); };

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);

  uint32_t diff_count = Valid(param);
  EXPECT_EQ(diff_count, 0);

  FreeTensorInput(param);
}

}  // namespace af
