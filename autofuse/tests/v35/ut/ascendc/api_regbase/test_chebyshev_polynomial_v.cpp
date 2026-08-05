/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include <random>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "chebyshev_polynomial_v.h"

using namespace AscendC;

namespace ge {

class TestRegbaseApiChebyshevPolynomialVUT : public testing::Test {
 protected:
  template <typename T, int64_t N>
  static void InvokeTensorTensorKernel(UnaryInputParam<T> &v_param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, temp;
    tpipe.InitBuffer(xbuf, sizeof(T) * v_param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * v_param.size);
    tpipe.InitBuffer(temp, TMP_UB_SIZE);

    LocalTensor<T> v_x = xbuf.Get<T>();
    LocalTensor<T> v_y = ybuf.Get<T>();
    LocalTensor<uint8_t> v_tmp = temp.Get<uint8_t>();

    GmToUb(v_x, v_param.x1, v_param.size);
    ChebyshevPolynomialVExtend<T, N>(v_y, v_x, v_tmp, v_param.size);
    UbToGm(v_param.y, v_y, v_param.size);
  }

  template <typename T>
  static T Chebyshev_v(int64_t n, T x) {
    if (n < 0) {
      return T(0.0);
    }

    if (n == 0) {
      return T(1.0);
    }

    if (n == 1) {
      return (x + x) - T(1.0);
    }

    T p = T(1.0);
    T q = (x + x) - T(1.0);
    T r;

    for (int64_t k = 2; k <= n; k++) {
      r = (x + x) * q - p;
      p = q;
      q = r;
    }

    return r;
  }

  template <typename T, int64_t N>
  static void CreateTensorInput(UnaryInputParam<T> &v_param) {
    v_param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * v_param.size));
    v_param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * v_param.size));
    v_param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * v_param.size));

    std::mt19937 eng(1);

    for (uint32_t i = 0; i < v_param.size; i++) {
      std::uniform_real_distribution<float> distr(-1.0f, 1.0f);
      v_param.x1[i] = static_cast<T>(distr(eng));
      v_param.exp[i] = static_cast<T>(Chebyshev_v(N, v_param.x1[i]));
    }
  }

  template <typename T>
  static uint32_t Valid(UnaryInputParam<T> &v_param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < v_param.size; i++) {
      double y_val = static_cast<double>(v_param.y[i]);
      double exp_val = static_cast<double>(v_param.exp[i]);
      double rel_err = std::abs(y_val - exp_val) / std::max(std::abs(exp_val), 1.0);
      if (rel_err > 1e-2) {
        diff_count++;
        printf("diff at index %d: x: %.20e, y: %.20e, expect: %.20e, rel_err: %f\n", i,
               static_cast<float>(v_param.x1[i]), static_cast<float>(v_param.y[i]), static_cast<float>(v_param.exp[i]),
               static_cast<float>(rel_err));
      }
    }
    return diff_count;
  }

  template <typename T, int64_t N>
  static void ChebyshevPolynomialVTensorTensorTest(uint32_t size) {
    UnaryInputParam<T> v_param{};
    v_param.size = size;
    CreateTensorInput<T, N>(v_param);

    auto v_kernel = [&v_param] { InvokeTensorTensorKernel<T, N>(v_param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(v_kernel, 1);

    uint32_t diff_count = Valid(v_param);
    EXPECT_EQ(diff_count, 0);
  }
};

TEST_F(TestRegbaseApiChebyshevPolynomialVUT, ChebyshevPolynomialV_TensorTensor_Test_N0) {
  ChebyshevPolynomialVTensorTensorTest<float, 0>(ONE_BLK_SIZE / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 0>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 0>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 0>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 0>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 0>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                  (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                  (ONE_BLK_SIZE - sizeof(float))) /
                                                 2 / sizeof(float));
}

TEST_F(TestRegbaseApiChebyshevPolynomialVUT, ChebyshevPolynomialV_TensorTensor_Test_N1) {
  ChebyshevPolynomialVTensorTensorTest<float, 1>(ONE_BLK_SIZE / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 1>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 1>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 1>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 1>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 1>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                  (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                  (ONE_BLK_SIZE - sizeof(float))) /
                                                 2 / sizeof(float));
}

TEST_F(TestRegbaseApiChebyshevPolynomialVUT, ChebyshevPolynomialV_TensorTensor_Test_N2) {
  ChebyshevPolynomialVTensorTensorTest<float, 2>(ONE_BLK_SIZE / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 2>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 2>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 2>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 2>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 2>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                  (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                  (ONE_BLK_SIZE - sizeof(float))) /
                                                 2 / sizeof(float));
}

TEST_F(TestRegbaseApiChebyshevPolynomialVUT, ChebyshevPolynomialV_TensorTensor_Test_N3) {
  ChebyshevPolynomialVTensorTensorTest<float, 3>(ONE_BLK_SIZE / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 3>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 3>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 3>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 3>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 3>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                  (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                  (ONE_BLK_SIZE - sizeof(float))) /
                                                 2 / sizeof(float));
}

TEST_F(TestRegbaseApiChebyshevPolynomialVUT, ChebyshevPolynomialV_TensorTensor_Test_N5) {
  ChebyshevPolynomialVTensorTensorTest<float, 5>(ONE_BLK_SIZE / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 5>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 5>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 5>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 5>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 5>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                  (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                  (ONE_BLK_SIZE - sizeof(float))) /
                                                 2 / sizeof(float));
}

TEST_F(TestRegbaseApiChebyshevPolynomialVUT, ChebyshevPolynomialV_TensorTensor_Test_N7) {
  ChebyshevPolynomialVTensorTensorTest<float, 7>(ONE_BLK_SIZE / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 7>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 7>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 7>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 7>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 7>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                  (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                  (ONE_BLK_SIZE - sizeof(float))) /
                                                 2 / sizeof(float));
}

TEST_F(TestRegbaseApiChebyshevPolynomialVUT, ChebyshevPolynomialV_TensorTensor_Test_N10) {
  ChebyshevPolynomialVTensorTensorTest<float, 10>(ONE_BLK_SIZE / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 10>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 10>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 10>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 10>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ChebyshevPolynomialVTensorTensorTest<float, 10>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                   (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                   (ONE_BLK_SIZE - sizeof(float))) /
                                                  2 / sizeof(float));
}

}  // namespace ge
