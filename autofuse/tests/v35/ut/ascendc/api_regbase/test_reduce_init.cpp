/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "api_regbase/reduce_init.h"

using namespace AscendC;

template <typename T>
struct TensorReduceInitInputParam {
  T *y{};
  T *exp{};
  T *src0{};
  uint32_t size{0};
  uint32_t dim_a{0};
  uint32_t dim_r{0};
  uint32_t dim_r_current{0};
  uint32_t inner_r{0};
};

class TestApiReduceInit : public testing::Test {
 protected:
  template <typename T, int32_t ReduceType, int32_t isTailLast>
  static void InvokeKernelWithTwoTensorInput(TensorReduceInitInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, ybuf;

    tpipe.InitBuffer(x1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));

    LocalTensor<T> l_x1 = x1buf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();

    GmToUb(l_x1, param.src0, param.size);
    ReduceInit<T, ReduceType, isTailLast>(l_x1, param.dim_a, param.dim_r, param.dim_r_current, param.inner_r);
    UbToGm(param.y, l_x1, param.size);
  }

  template <typename T, int32_t ReduceType, int32_t isTailLast>
  static void CreateTensorInput(TensorReduceInitInputParam<T> &param) {
    // 构造测试输入和预期结果
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src0 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    int input_range = 10;

    std::mt19937 eng(1);
    std::uniform_int_distribution distr(0, input_range);  // Define the range

    for (int i = 0; i < param.size; i++) {
      T input = distr(eng);  // Use the secure random number generator
      param.src0[i] = input;
      param.exp[i] = input;
    }

    int itemsize = sizeof(T);
    int padding_length = 32 / itemsize;

    T pad_value = GetPaddingValue<T, ReduceType>();

    int repeatStride = ((param.inner_r * itemsize + 31) / 32) * 32;  // inner_r up to 32B
    int strideElement = repeatStride / itemsize;                     // inner_r_up
    int pad_len = strideElement - param.inner_r;
    int repeat_times = param.dim_r_current / strideElement;

    for (int i = param.dim_r_current; i < param.dim_r; ++i) {
      param.exp[i] = pad_value;
    }

    for (int i = 0; i < repeat_times; ++i) {
      for (int j = 0; j < pad_len; ++j) {
        param.exp[param.inner_r + j + i * strideElement] = pad_value;
      }
    }
  }

  template <typename T>
  static uint32_t Valid(T *y, T *exp, size_t comp_size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < comp_size; i++) {
      if (y[i] != exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T, int32_t ReduceType, int32_t isTailLast>
  static void ReduceInitTest(uint32_t dim_a, uint32_t dim_r, uint32_t dim_r_current, uint32_t inner_r) {
    TensorReduceInitInputParam<T> param{};
    param.size = dim_a * dim_r;
    param.dim_a = dim_a;
    param.dim_r = dim_r;
    param.dim_r_current = dim_r_current;
    param.inner_r = inner_r;
    CreateTensorInput<T, ReduceType, isTailLast>(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeKernelWithTwoTensorInput<T, ReduceType, isTailLast>(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
    // 释放内存
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src0);
  }
};

TEST_F(TestApiReduceInit, ReduceInit_Test) {
  ReduceInitTest<float, 3, 1>(1, 32, 16, 3);
}

TEST_F(TestApiReduceInit, ReduceInitSingleRowFullVlTest) {
  ReduceInitTest<float, 3, 1>(1, 1024, 512, 3);
}

TEST_F(TestApiReduceInit, ReduceInitSingleRowTailVlTest) {
  ReduceInitTest<float, 2, 1>(1, 1024, 520, 3);
}
