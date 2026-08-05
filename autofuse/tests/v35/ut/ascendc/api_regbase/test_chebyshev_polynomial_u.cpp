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
#include "chebyshev_polynomial_u.h"
#include "boost/math/special_functions/chebyshev.hpp"

using namespace AscendC;

namespace ge {

class TestRegbaseApiChebyshevPolynomialUUT : public testing::Test {
 protected:
  template <typename T, int64_t N>
  static void InvokeTensorTensorKernel(UnaryInputParam<T> &u_param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, temp;
    tpipe.InitBuffer(xbuf, sizeof(T) * u_param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * u_param.size);
    tpipe.InitBuffer(temp, TMP_UB_SIZE);

    LocalTensor<T> u_x = xbuf.Get<T>();
    LocalTensor<T> u_y = ybuf.Get<T>();
    LocalTensor<uint8_t> u_tmp = temp.Get<uint8_t>();

    GmToUb(u_x, u_param.x1, u_param.size);
    ChebyshevPolynomialUExtend<T, N>(u_y, u_x, u_tmp, u_param.size);
    UbToGm(u_param.y, u_y, u_param.size);
  }

  template <typename T, int64_t N>
  static void CreateTensorInput(UnaryInputParam<T> &u_param) {
    u_param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * u_param.size));
    u_param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * u_param.size));
    u_param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * u_param.size));

    std::mt19937 eng(1);

    for (uint32_t i = 0; i < u_param.size; i++) {
      std::uniform_real_distribution<float> distr(-1.0f, 1.0f);
      u_param.x1[i] = static_cast<T>(distr(eng));
      u_param.exp[i] = static_cast<T>(boost::math::chebyshev_u(N, static_cast<double>(u_param.x1[i])));
    }
  }

  template <typename T>
  static uint32_t Valid(UnaryInputParam<T> &u_param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < u_param.size; i++) {
      double y_val = static_cast<double>(u_param.y[i]);
      double exp_val = static_cast<double>(u_param.exp[i]);
      double rel_err = std::abs(y_val - exp_val) / std::max(std::abs(exp_val), 1.0);
      if (rel_err > 1e-2) {
        diff_count++;
        printf("diff at index %d: x: %.20e, y: %.20e, expect: %.20e, rel_err: %f\n", i,
               static_cast<float>(u_param.x1[i]), static_cast<float>(u_param.y[i]), static_cast<float>(u_param.exp[i]),
               static_cast<float>(rel_err));
      }
    }
    return diff_count;
  }

  template <typename T, int64_t N>
  static void ChebyshevPolynomialUTensorTensorTest(uint32_t size) {
    UnaryInputParam<T> u_param{};
    u_param.size = size;
    CreateTensorInput<T, N>(u_param);

    auto u_kernel = [&u_param] { InvokeTensorTensorKernel<T, N>(u_param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(u_kernel, 1);

    uint32_t diff_count = Valid(u_param);
    EXPECT_EQ(diff_count, 0);
  }
};

TEST_F(TestRegbaseApiChebyshevPolynomialUUT, ChebyshevPolynomialU_TensorTensor_Test_N0) {
  ChebyshevPolynomialUTensorTensorTest<float, 0>(ONE_BLK_SIZE / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 0>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 0>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 0>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 0>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 0>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                  (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                  (ONE_BLK_SIZE - sizeof(float))) /
                                                 2 / sizeof(float));
}

TEST_F(TestRegbaseApiChebyshevPolynomialUUT, ChebyshevPolynomialU_TensorTensor_Test_N1) {
  ChebyshevPolynomialUTensorTensorTest<float, 1>(ONE_BLK_SIZE / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 1>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 1>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 1>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 1>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 1>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                  (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                  (ONE_BLK_SIZE - sizeof(float))) /
                                                 2 / sizeof(float));
}

TEST_F(TestRegbaseApiChebyshevPolynomialUUT, ChebyshevPolynomialU_TensorTensor_Test_N2) {
  ChebyshevPolynomialUTensorTensorTest<float, 2>(ONE_BLK_SIZE / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 2>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 2>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 2>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 2>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 2>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                  (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                  (ONE_BLK_SIZE - sizeof(float))) /
                                                 2 / sizeof(float));
}

TEST_F(TestRegbaseApiChebyshevPolynomialUUT, ChebyshevPolynomialU_TensorTensor_Test_N3) {
  ChebyshevPolynomialUTensorTensorTest<float, 3>(ONE_BLK_SIZE / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 3>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 3>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 3>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 3>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 3>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                  (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                  (ONE_BLK_SIZE - sizeof(float))) /
                                                 2 / sizeof(float));
}

TEST_F(TestRegbaseApiChebyshevPolynomialUUT, ChebyshevPolynomialU_TensorTensor_Test_N5) {
  ChebyshevPolynomialUTensorTensorTest<float, 5>(ONE_BLK_SIZE / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 5>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 5>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 5>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 5>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 5>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                  (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                  (ONE_BLK_SIZE - sizeof(float))) /
                                                 2 / sizeof(float));
}

TEST_F(TestRegbaseApiChebyshevPolynomialUUT, ChebyshevPolynomialU_TensorTensor_Test_N7) {
  ChebyshevPolynomialUTensorTensorTest<float, 7>(ONE_BLK_SIZE / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 7>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 7>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 7>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 7>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 7>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                  (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                  (ONE_BLK_SIZE - sizeof(float))) /
                                                 2 / sizeof(float));
}

TEST_F(TestRegbaseApiChebyshevPolynomialUUT, ChebyshevPolynomialU_TensorTensor_Test_N10) {
  ChebyshevPolynomialUTensorTensorTest<float, 10>(ONE_BLK_SIZE / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 10>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 10>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 10>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 10>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ChebyshevPolynomialUTensorTensorTest<float, 10>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                   (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                   (ONE_BLK_SIZE - sizeof(float))) /
                                                  2 / sizeof(float));
}

}  // namespace ge
