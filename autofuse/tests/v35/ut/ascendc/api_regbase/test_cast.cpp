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
#include <algorithm>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/cast.h"

using namespace AscendC;

template <typename InT, typename OutT, uint8_t dim>
struct TensorCastInputParam {
  OutT *y{};
  OutT *exp{};
  InT *src0{};
  uint32_t output_dims[2];
  uint32_t output_stride[2];
  uint32_t input_stride[2];
  uint32_t size{0};
};

class TestApiCast : public testing::Test {
 protected:
  template <typename InT, typename OutT, uint8_t dim>
  static void InvokeKernelWithTwoTensorInput(TensorCastInputParam<InT, OutT, dim> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, ybuf;
    tpipe.InitBuffer(x1buf, sizeof(InT) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(OutT) * param.size);
    LocalTensor<InT> l_x1 = x1buf.Get<InT>();
    LocalTensor<OutT> l_y = ybuf.Get<OutT>();

    GmToUb(l_x1, param.src0, param.size);
    CastExtend<InT, OutT, dim>(l_y, l_x1, param.output_dims, param.output_stride, param.input_stride);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename InT, typename OutT, uint8_t dim>
  static void CreateTensorInput(TensorCastInputParam<InT, OutT, dim> &param) {
    // 构造测试输入和预期结果
    uint32_t max_T = std::max(sizeof(InT), sizeof(OutT));
    param.y = static_cast<OutT *>(AscendC::GmAlloc(sizeof(max_T) * param.size));
    param.exp = static_cast<OutT *>(AscendC::GmAlloc(sizeof(max_T) * param.size));
    param.src0 = static_cast<InT *>(AscendC::GmAlloc(sizeof(InT) * param.size));
    int input_range = 10;

    std::mt19937 eng(1);
    std::uniform_int_distribution distr(0, input_range);  // Define the range

    for (int i = 0; i < param.size; i++) {
      InT input = distr(eng);
      param.src0[i] = input;
      param.exp[i] = static_cast<OutT>(param.src0[i]);
    }
  }

  template <typename InT, typename OutT, uint8_t dim>
  static uint32_t Valid(OutT *y, OutT *exp, size_t comp_size, const uint32_t (&output_dims)[dim]) {
    uint32_t diff_count = 0;
    printf("dim1 = %d, dim2 = %d\n", output_dims[0], output_dims[1]);
    for (uint32_t i = 0; i < output_dims[0]; i++) {
      for (uint32_t j = 0; j < output_dims[1]; j++) {
        if (y[i * output_dims[1] + j] != exp[i * output_dims[1] + j]) {
          diff_count++;
        }
      }
    }
    return diff_count;
  }

  template <typename InT, typename OutT, uint8_t dim>
  static void CastTest(const uint32_t (&output_dims)[dim], const uint32_t (&output_stride)[dim],
                       const uint32_t (&input_stride)[dim]) {
    TensorCastInputParam<InT, OutT, dim> param{};
    static_assert(dim == 2, "dim must be 2, if dim < 2, set stride as {0, 1}");
    param.size = output_dims[0] * output_dims[1];
    for (int i = 0; i < 2; ++i) {
      param.output_dims[i] = output_dims[i];
    }
    CreateTensorInput(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeKernelWithTwoTensorInput(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid<InT, OutT, dim>(param.y, param.exp, param.size, output_dims);
    EXPECT_EQ(diff_count, 0);
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src0);
  }
};

TEST_F(TestApiCast, Cast_Test_Float_Int32) {
  CastTest<float, int32_t, 2>({1, 32}, {0, 1}, {0, 1});
}

TEST_F(TestApiCast, Cast_Test_BFloat16_Float) {
  CastTest<bfloat16_t, float, 2>({1, 128}, {0, 1}, {0, 1});
}

TEST_F(TestApiCast, Cast_Test_Float_BFloat16) {
  CastTest<float, bfloat16_t, 2>({1, 128}, {0, 1}, {0, 1});
}

TEST_F(TestApiCast, Cast_Test_Int64_BFloat16) {
  CastTest<int64_t, bfloat16_t, 2>({1, 128}, {0, 1}, {0, 1});
}

TEST_F(TestApiCast, Cast_Test_Uint8_Float16) {
  CastTest<uint8_t, half, 2>({1, 128}, {0, 1}, {0, 1});
}

TEST_F(TestApiCast, Cast_Test_Int8_Float) {
  CastTest<int8_t, float, 2>({1, 128}, {0, 1}, {0, 1});
}

TEST_F(TestApiCast, Cast_Test_Float_Int8) {
  CastTest<float, int8_t, 2>({1, 128}, {0, 1}, {0, 1});
}

TEST_F(TestApiCast, Cast_Test_Int8_Int16) {
  CastTest<int8_t, int16_t, 2>({1, 128}, {0, 1}, {0, 1});
}

TEST_F(TestApiCast, Cast_Test_Int16_Int8) {
  CastTest<int16_t, int8_t, 2>({1, 128}, {0, 1}, {0, 1});
}

TEST_F(TestApiCast, Cast_Test_Int16_Uint8) {
  CastTest<int16_t, uint8_t, 2>({1, 128}, {0, 1}, {0, 1});
}
