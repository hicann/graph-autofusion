/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <random>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/silu.h"

using namespace AscendC;

namespace af {

template <typename T>
struct SiluInputParam {
  T *x{};
  T *y{};
  T *exp{};
  uint32_t size{};
};

class TestRegbaseApiSiluUT : public testing::Test {
 protected:
  template <typename T>
  static uint32_t CalcSiluTmpSize(uint32_t size) {
    if constexpr (std::is_same_v<T, float>) {
      return size * sizeof(float);
    } else {
      return AlignUp(size * static_cast<uint32_t>(sizeof(float)), ONE_BLK_SIZE) + size * sizeof(float);
    }
  }

  // Tensor - Tensor 场景
  template <typename T>
  static void InvokeTensorTensorKernel(SiluInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
    tpipe.InitBuffer(tmp, CalcSiluTmpSize<T>(param.size));

    LocalTensor<T> l_x = xbuf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, param.x, param.size);
    SiluExtend(l_y, l_x, l_tmp, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  // In-place 场景：dst 与 src 共用同一 buffer
  template <typename T>
  static void InvokeInplaceKernel(SiluInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
    tpipe.InitBuffer(tmp, CalcSiluTmpSize<T>(param.size));

    LocalTensor<T> l_x = xbuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, param.x, param.size);
    SiluExtend(l_x, l_x, l_tmp, param.size);
    UbToGm(param.y, l_x, param.size);
  }

  template <typename T>
  static void CreateTensorInput(SiluInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);

    for (uint32_t i = 0; i < param.size; i++) {
      std::uniform_real_distribution distr(-5.0f, 5.0f);
      param.x[i] = static_cast<T>(distr(eng));
      double x = static_cast<double>(param.x[i]);
      param.exp[i] = static_cast<T>(x / (1.0 + std::exp(-x)));
    }
  }

  template <typename T>
  static uint32_t Valid(SiluInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      auto diff = (double)(param.y[i] - param.exp[i]);
      if (diff < -1e-2 || diff > 1e-2) {
        diff_count++;
        printf("diff at index %d: x: %f, y: %f, expect: %f, diff: %f\n", i, static_cast<float>(param.x[i]),
               static_cast<float>(param.y[i]), static_cast<float>(param.exp[i]),
               static_cast<float>(param.y[i] - param.exp[i]));
      }
    }
    return diff_count;
  }

  template <typename T>
  static void FreeTensorInput(SiluInputParam<T> &param) {
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.x);
  }

  // Tensor - Tensor 测试
  template <typename T>
  static void SiluTest(uint32_t size) {
    SiluInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0) << " of " << size;
    FreeTensorInput(param);
  }

  // In-place 测试：dst == src
  template <typename T>
  static void SiluInplaceTest(uint32_t size) {
    SiluInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    auto kernel = [&param] { InvokeInplaceKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0) << " of " << size;
    FreeTensorInput(param);
  }

  // 特殊输入(0、+Inf、-Inf、NaN、大数等)构造与验证，覆盖 silu.h 注释声明的 edge case
  template <typename T>
  static void CreateSpecialInput(SiluInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    T nan_val = std::numeric_limits<T>::quiet_NaN();
    T inf_val = std::numeric_limits<T>::infinity();

    param.x[0] = T(0.0);
    param.exp[0] = T(0.0);
    param.x[1] = T(1.0);
    param.exp[1] = static_cast<T>(1.0 / (1.0 + std::exp(-1.0)));
    param.x[2] = T(-1.0);
    param.exp[2] = static_cast<T>(-1.0 / (1.0 + std::exp(1.0)));
    param.x[3] = inf_val;
    param.exp[3] = inf_val;
    param.x[4] = -inf_val;
    param.exp[4] = nan_val;
    param.x[5] = nan_val;
    param.exp[5] = nan_val;
    param.x[6] = T(100.0);
    param.exp[6] = T(100.0);
    param.x[7] = T(-100.0);
    param.exp[7] = T(0.0);
    param.x[8] = T(0.5);
    param.exp[8] = static_cast<T>(0.5 / (1.0 + std::exp(-0.5)));
    param.x[9] = T(-0.5);
    param.exp[9] = static_cast<T>(-0.5 / (1.0 + std::exp(0.5)));
  }

  template <typename T>
  static uint32_t ValidSpecial(SiluInputParam<T> &param) {
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
        match = (diff <= 1e-2);
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
  static void SiluSpecialTest() {
    SiluInputParam<T> param{};
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

TEST_F(TestRegbaseApiSiluUT, Silu_TensorTensor_Test) {
  SiluTest<half>(ONE_BLK_SIZE / sizeof(half));
  SiluTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  SiluTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  SiluTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  SiluTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));
  SiluTest<half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  SiluTest<float>(ONE_BLK_SIZE / sizeof(float));
  SiluTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  SiluTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  SiluTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  SiluTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  SiluTest<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  SiluTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                   (ONE_BLK_SIZE - sizeof(float))) /
                  2 / sizeof(float));
}

TEST_F(TestRegbaseApiSiluUT, Silu_Special_Test) {
  SiluSpecialTest<float>();
}

TEST_F(TestRegbaseApiSiluUT, Silu_Inplace_Test) {
  // float
  SiluInplaceTest<float>(ONE_BLK_SIZE / sizeof(float));
  SiluInplaceTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  SiluInplaceTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  SiluInplaceTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                          (ONE_BLK_SIZE - sizeof(float))) /
                         2 / sizeof(float));
  // half
  SiluInplaceTest<half>(ONE_BLK_SIZE / sizeof(half));
  SiluInplaceTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  SiluInplaceTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
}

}  // namespace af
