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
#include "api_regbase/rsqrt.h"

using namespace AscendC;

namespace af {

template <typename T>
struct RsqrtInputParam {
  T *x{};
  T *y{};
  T *exp{};
  uint32_t size{};
};

class TestRegbaseApiRsqrtUT : public testing::Test {
 protected:
  // Tensor - Tensor 场景
  template <typename T>
  static void InvokeTensorTensorKernel(RsqrtInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_x = xbuf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, param.x, param.size);
    RsqrtExtend(l_y, l_x, l_tmp, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  // In-place 场景：dst 与 src 共用同一 buffer
  template <typename T>
  static void InvokeInplaceKernel(RsqrtInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_x = xbuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, param.x, param.size);
    RsqrtExtend(l_x, l_x, l_tmp, param.size);
    UbToGm(param.y, l_x, param.size);
  }

  template <typename T>
  static void CreateTensorInput(RsqrtInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);

    for (uint32_t i = 0; i < param.size; i++) {
      // 生成正数测试数据，范围 (0, 100]，避免零和负数
      std::uniform_real_distribution<float> distr(0.1f, 100.0f);
      double x = static_cast<double>(distr(eng));
      param.x[i] = static_cast<T>(x);
      param.exp[i] = static_cast<T>(1.0 / std::sqrt(x));
    }
  }

  template <typename T>
  static uint32_t Valid(RsqrtInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      double y_val = static_cast<double>(param.y[i]);
      double exp_val = static_cast<double>(param.exp[i]);
      double diff = std::abs(y_val - exp_val);
      // 使用相对容差，rsqrt 的结果量级跨度大
      double rel = exp_val > 1e-10 ? diff / std::abs(exp_val) : diff;
      if (rel > 1e-3) {
        diff_count++;
        printf("diff at index %d: x: %f, y: %f, expect: %f, diff: %f\n", i, static_cast<float>(param.x[i]),
               static_cast<float>(param.y[i]), static_cast<float>(param.exp[i]), static_cast<float>(diff));
      }
    }
    return diff_count;
  }

  // 特殊输入(0、+Inf、-Inf、NaN、大数等)构造与验证，覆盖 rsqrt 数学边界行为
  template <typename T>
  static void CreateSpecialInput(RsqrtInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    T nan_val = std::numeric_limits<T>::quiet_NaN();
    T inf_val = std::numeric_limits<T>::infinity();

    // idx 0: rsqrt(1) = 1
    param.x[0] = T(1.0);
    param.exp[0] = T(1.0);
    // idx 1: rsqrt(4) = 0.5
    param.x[1] = T(4.0);
    param.exp[1] = T(0.5);
    // idx 2: rsqrt(0.25) = 2
    param.x[2] = T(0.25);
    param.exp[2] = T(2.0);
    // idx 3: rsqrt(0) = +Inf
    param.x[3] = T(0.0);
    param.exp[3] = inf_val;
    // idx 4: rsqrt(+Inf) = 0
    param.x[4] = inf_val;
    param.exp[4] = T(0.0);
    // idx 5: rsqrt(负数) = NaN (sqrt(-1) = NaN)
    param.x[5] = T(-1.0);
    param.exp[5] = nan_val;
    // idx 6: rsqrt(NaN) = NaN
    param.x[6] = nan_val;
    param.exp[6] = nan_val;
    // idx 7: rsqrt(100) = 0.1
    param.x[7] = T(100.0);
    param.exp[7] = T(0.1);
    // idx 8: rsqrt(极小正数) = 大数
    param.x[8] = T(1e-20);
    param.exp[8] = static_cast<T>(1.0 / std::sqrt(1e-20));
    // idx 9: rsqrt(极大正数) = 极小数
    param.x[9] = T(1e20);
    param.exp[9] = static_cast<T>(1.0 / std::sqrt(1e20));
  }

  template <typename T>
  static uint32_t ValidSpecial(RsqrtInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      T exp_val = param.exp[i];
      T y_val = param.y[i];
      bool match = false;
      if (std::isnan(exp_val)) {
        match = std::isnan(y_val);
      } else if (std::isinf(exp_val)) {
        match = std::isinf(y_val) && (std::signbit(y_val) == std::signbit(exp_val));
      } else {
        double diff = std::abs(static_cast<double>(y_val) - static_cast<double>(exp_val));
        double rel =
            std::abs(static_cast<double>(exp_val)) > 1e-10 ? diff / std::abs(static_cast<double>(exp_val)) : diff;
        match = (rel <= 1e-3);
      }
      if (!match) {
        diff_count++;
        printf("special diff at index %d: x: %f, y: %f, expect: %f\n", i, static_cast<float>(param.x[i]),
               static_cast<float>(y_val), static_cast<float>(exp_val));
      }
    }
    return diff_count;
  }

  template <typename T>
  static void FreeTensorInput(RsqrtInputParam<T> &param) {
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.x);
    AscendC::GmFree(param.exp);
  }

  // Tensor - Tensor 测试
  template <typename T>
  static void RsqrtTest(uint32_t size) {
    RsqrtInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0) << " of " << size;
    // 释放内存
    FreeTensorInput(param);
  }

  // In-place 测试：dst == src
  template <typename T>
  static void RsqrtInplaceTest(uint32_t size) {
    RsqrtInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    auto kernel = [&param] { InvokeInplaceKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0) << " of " << size;
    FreeTensorInput(param);
  }

  template <typename T>
  static void RsqrtSpecialTest() {
    RsqrtInputParam<T> param{};
    param.size = 10;
    CreateSpecialInput(param);

    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = ValidSpecial(param);
    EXPECT_EQ(diff_count, 0);

    FreeTensorInput(param);
  }
};

TEST_F(TestRegbaseApiRsqrtUT, Rsqrt_TensorTensor_Test) {
  // half
  RsqrtTest<half>(ONE_BLK_SIZE / sizeof(half));
  RsqrtTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  RsqrtTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  RsqrtTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  RsqrtTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));
  RsqrtTest<half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  // float
  RsqrtTest<float>(ONE_BLK_SIZE / sizeof(float));
  RsqrtTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  RsqrtTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  RsqrtTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  RsqrtTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  RsqrtTest<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  RsqrtTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                    (ONE_BLK_SIZE - sizeof(float))) /
                   2 / sizeof(float));
}

TEST_F(TestRegbaseApiRsqrtUT, Rsqrt_Special_Test) {
  RsqrtSpecialTest<float>();
}

TEST_F(TestRegbaseApiRsqrtUT, Rsqrt_Inplace_Test) {
  // float
  RsqrtInplaceTest<float>(ONE_BLK_SIZE / sizeof(float));
  RsqrtInplaceTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  RsqrtInplaceTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  RsqrtInplaceTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                           (ONE_BLK_SIZE - sizeof(float))) /
                          2 / sizeof(float));
  // half
  RsqrtInplaceTest<half>(ONE_BLK_SIZE / sizeof(half));
  RsqrtInplaceTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  RsqrtInplaceTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
}

}  // namespace af
