/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <random>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "modified_bessel_utils.h"
#include "modified_bessel_i1.h"
#include "i1e.h"

using namespace AscendC;

namespace af {

class TestRegbaseApiI1eUT : public testing::Test {
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
    I1eExtend(l_y, l_x, l_tmp, param.size);
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
      double abs_x = static_cast<double>(std::fabs(param.x1[i]));
      // I1 is odd: I1(-x) = -I1(x), cyl_bessel_i requires x >= 0
      double i1_val = std::cyl_bessel_i(1, abs_x);
      if (param.x1[i] < 0) i1_val = -i1_val;
      param.exp[i] = static_cast<T>(std::exp(-abs_x) * i1_val);
    }
  }

  template <typename T>
  static uint32_t Valid(UnaryInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      double y_val = static_cast<double>(param.y[i]);
      double exp_val = static_cast<double>(param.exp[i]);
      double rel_err = std::abs(y_val - exp_val) / std::max(std::abs(exp_val), 1.0);
      if (rel_err > 1e-5) {
        diff_count++;
        printf("diff at index %d: x: %.20e, y: %.20e, expect: %.20e, rel_err: %f\n", i, static_cast<float>(param.x1[i]),
               static_cast<float>(param.y[i]), static_cast<float>(param.exp[i]), static_cast<float>(rel_err));
      }
    }
    return diff_count;
  }

  template <typename T>
  static void I1eTest(uint32_t size) {
    UnaryInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);

    AscendC::GmFree(param.x1);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.y);
  }
};

TEST_F(TestRegbaseApiI1eUT, I1e_TensorTensor_Test) {
  I1eTest<float>(ONE_BLK_SIZE / sizeof(float));
  I1eTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  I1eTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  I1eTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  I1eTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  I1eTest<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  I1eTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                  (ONE_BLK_SIZE - sizeof(float))) /
                 2 / sizeof(float));
}

}  // namespace af
