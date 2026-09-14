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
#include <cstdint>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/trigonometric_function_utils.h"
#include "api_regbase/zeta.h"
#include "api_regbase/polygamma.h"

using namespace AscendC;

namespace af {

// RAII wrapper around AscendC::GmAlloc / GmFree. Frees on scope exit so tests
// never leak GM memory, even when an EXPECT_* assertion fails mid-function.
// The implicit operator T*() lets a GmBuffer be passed anywhere a raw T* is
// expected (kernels, Valid, subscripting).
template <typename T>
class GmBuffer {
 public:
  explicit GmBuffer(uint32_t count) : ptr_(static_cast<T *>(AscendC::GmAlloc(sizeof(T) * count))) {}
  ~GmBuffer() {
    AscendC::GmFree(ptr_);
  }
  GmBuffer(const GmBuffer &) = delete;
  GmBuffer &operator=(const GmBuffer &) = delete;
  operator T *() const {
    return ptr_;
  }

 private:
  T *ptr_;
};

class TestRegbaseApiPolyGammaUT : public testing::Test {
 protected:
  template <typename T>
  static void InvokePolyGammaKernel(T *y, T *x, int32_t n, uint32_t size) {
    TPipe tpipe;
    // InitBuffer requires a length > 0, but calCount==0 is a valid no-op input;
    // allocate at least one element so the empty case exercises PolyGamma safely.
    uint32_t bufLen = (size == 0) ? 1 : size;
    TBuf<TPosition::VECCALC> xBuf, yBuf, tmpBuf;
    tpipe.InitBuffer(xBuf, sizeof(T) * bufLen);
    tpipe.InitBuffer(yBuf, sizeof(T) * bufLen);
    tpipe.InitBuffer(tmpBuf, TMP_UB_SIZE);
    LocalTensor<T> xTensor = xBuf.Get<T>();
    LocalTensor<T> yTensor = yBuf.Get<T>();
    LocalTensor<T> tmpTensor = tmpBuf.Get<T>();

    GmToUb(xTensor, x, size);
    PolyGammaExtend<T>(yTensor, xTensor, n, tmpTensor, size);
    UbToGm(y, yTensor, size);
  }

  template <typename T>
  static uint32_t Valid(T *y, T *exp, size_t comp_size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < comp_size; i++) {
      bool match = false;
      if (std::isnan(exp[i])) {
        match = std::isnan(y[i]);
      } else if (std::isinf(exp[i])) {
        match = (std::isinf(y[i]) && std::signbit(y[i]) == std::signbit(exp[i]));
      } else {
        match = std::abs(y[i] - exp[i]) <= 1e-5f || std::abs(y[i] - exp[i]) <= std::abs(exp[i]) * 1e-5f;
      }
      if (!match) {
        diff_count++;
        printf("Test[%u] FAIL: exp=%.9g, act=%.9g, rdiff=%.6g\n", i, static_cast<double>(exp[i]),
               static_cast<double>(y[i]), std::abs(y[i] - exp[i]) / std::max(std::abs(exp[i]), 1e-10f));
      }
    }
    return diff_count;
  }

  template <typename T>
  static void DigammaCorrectnessTest() {
    std::vector<T> x_list = {0.5f, 1.0f, 2.0f, 10.0f, 100.0f};
    std::vector<T> exp_list = {-1.963510f, -0.577216f, 0.422784f, 2.251752f, 4.600162f};

    uint32_t n = static_cast<uint32_t>(x_list.size());
    GmBuffer<T> x(n), y(n), exp(n);
    for (uint32_t i = 0; i < n; ++i) {
      x[i] = x_list[i];
      exp[i] = exp_list[i];
    }

    auto kernel = [&] { InvokePolyGammaKernel<T>(y, x, 0, n); };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    EXPECT_EQ(Valid<T>(y, exp, n), 0);
  }

  template <typename T>
  static void TrigammaCorrectnessTest() {
    // positive x, negative non-integer x (reflection formula), small positive x (near pole)
    std::vector<T> x_list = {0.5f, 1.0f, 2.0f, 10.0f, -0.3f, -1.3f, -5.7f, 0.001f};
    std::vector<T> exp_list = {4.934802f,  1.644934f,  0.644934f,  0.105166f,
                               13.945159f, 14.536883f, 14.918462f, 999999.94f};

    uint32_t n = static_cast<uint32_t>(x_list.size());
    GmBuffer<T> x(n), y(n), exp(n);
    for (uint32_t i = 0; i < n; ++i) {
      x[i] = x_list[i];
      exp[i] = exp_list[i];
    }

    auto kernel = [&] { InvokePolyGammaKernel<T>(y, x, 1, n); };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    EXPECT_EQ(Valid<T>(y, exp, n), 0);
  }

  template <typename T>
  static void PolyGammaN2CorrectnessTest() {
    std::vector<T> x_list = {0.5f, 1.0f, 2.0f, 10.0f, -0.3f, -0.7f, -1.3f, -5.3f, -9.7f};
    std::vector<T> exp_list = {-16.828796f, -2.404114f, -0.404114f, -0.011049f, 67.639076f,
                               -69.441628f, 68.549454f, 68.807884f, -68.846985f};

    uint32_t n = static_cast<uint32_t>(x_list.size());
    GmBuffer<T> x(n), y(n), exp(n);
    for (uint32_t i = 0; i < n; ++i) {
      x[i] = x_list[i];
      exp[i] = exp_list[i];
    }

    auto kernel = [&] { InvokePolyGammaKernel<T>(y, x, 2, n); };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    EXPECT_EQ(Valid<T>(y, exp, n), 0);
  }

  template <typename T>
  static void PolyGammaHighOrderTest() {
    std::vector<T> x_list = {0.5f, 1.0f, 3.0f};
    std::vector<T> exp_n3 = {97.409088f, 6.493939f, 0.118939f};
    std::vector<T> exp_n5 = {7691.11377f, 122.081161f, 0.206167f};

    uint32_t m = static_cast<uint32_t>(x_list.size());

    for (auto &pair : std::vector<std::pair<int32_t, std::vector<T>>>{{3, exp_n3}, {5, exp_n5}}) {
      int32_t nval = pair.first;
      auto &exp_vals = pair.second;
      GmBuffer<T> x(m), y(m), exp(m);
      for (uint32_t i = 0; i < m; ++i) {
        x[i] = x_list[i];
        exp[i] = exp_vals[i];
      }
      auto kernel = [&] { InvokePolyGammaKernel<T>(y, x, nval, m); };
      AscendC::SetKernelMode(KernelMode::AIV_MODE);
      ICPU_RUN_KF(kernel, 1);
      EXPECT_EQ(Valid<T>(y, exp, m), 0);
    }
  }

  template <typename T>
  static void PolyGammaZeroPoleTest() {
    std::pair<int32_t, bool> cases[] = {
        {0, true},   // digamma(0) = -inf
        {1, false},  // trigamma(0) = +inf
        {2, true},   // n even → -inf
        {3, false},  // n odd  → +inf
        {4, true},
    };

    for (auto &c : cases) {
      GmBuffer<T> x(1), y(1);
      x[0] = static_cast<T>(0.0);
      auto kernel = [&] { InvokePolyGammaKernel<T>(y, x, c.first, 1); };
      AscendC::SetKernelMode(KernelMode::AIV_MODE);
      ICPU_RUN_KF(kernel, 1);
      EXPECT_TRUE(std::isinf(y[0]));
      EXPECT_EQ(std::signbit(y[0]), c.second);
    }
  }

  template <typename T>
  static void PolyGammaNegIntPoleTest() {
    {
      GmBuffer<T> x(1), y(1);
      x[0] = static_cast<T>(-1.0);
      auto kernel = [&] { InvokePolyGammaKernel<T>(y, x, 1, 1); };
      AscendC::SetKernelMode(KernelMode::AIV_MODE);
      ICPU_RUN_KF(kernel, 1);
      EXPECT_GT(std::abs(y[0]), static_cast<T>(1e12));
    }

    // n=2: psi2(-1) = -inf
    {
      GmBuffer<T> x(1), y(1);
      x[0] = static_cast<T>(-1.0);
      auto kernel = [&] { InvokePolyGammaKernel<T>(y, x, 2, 1); };
      AscendC::SetKernelMode(KernelMode::AIV_MODE);
      ICPU_RUN_KF(kernel, 1);
      EXPECT_TRUE(std::isinf(y[0]) && y[0] < static_cast<T>(0));
    }
  }

  template <typename T>
  static void PolyGammaPosInfTest() {
    auto check = [](int32_t n, bool expectNaN, bool expectInf, bool expectZero) {
      GmBuffer<T> x(1), y(1);
      x[0] = std::numeric_limits<T>::infinity();
      auto kernel = [&] { InvokePolyGammaKernel<T>(y, x, n, 1); };
      AscendC::SetKernelMode(KernelMode::AIV_MODE);
      ICPU_RUN_KF(kernel, 1);

      if (expectNaN) {
        EXPECT_TRUE(std::isnan(y[0]));
      } else if (expectInf) {
        EXPECT_TRUE(std::isinf(y[0]) && y[0] > static_cast<T>(0));
      } else if (expectZero) {
        EXPECT_TRUE(y[0] == static_cast<T>(0.0) || y[0] == static_cast<T>(-0.0));
      }
    };

    check(0, false, true, false);  // digamma(+inf) = +inf (CANN SDK)
    check(1, false, false, true);  // trigamma(+inf) = 0
    check(2, true, false, false);  // Zeta NaN propagation
  }

  template <typename T>
  static void PolyGammaNegInfTest() {
    auto check = [](int32_t n, bool expectNaN, int sign) {
      GmBuffer<T> x(1), y(1);
      x[0] = -std::numeric_limits<T>::infinity();
      auto kernel = [&] { InvokePolyGammaKernel<T>(y, x, n, 1); };
      AscendC::SetKernelMode(KernelMode::AIV_MODE);
      ICPU_RUN_KF(kernel, 1);

      if (expectNaN) {
        EXPECT_TRUE(std::isnan(y[0]));
      } else {
        EXPECT_TRUE(std::isinf(y[0]));
        if (sign > 0) {
          EXPECT_GT(y[0], static_cast<T>(0));
        } else if (sign < 0) {
          EXPECT_LT(y[0], static_cast<T>(0));
        }
      }
    };

    check(1, true, 0);    // trigamma(-inf) = NaN (PyTorch reference)
    check(2, false, -1);  // n even → -inf
    check(3, false, 1);   // n odd  → +inf
  }

  template <typename T>
  static void PolyGammaNaNPropagationTest() {
    GmBuffer<T> x(1), y(1);
    x[0] = std::numeric_limits<T>::quiet_NaN();

    for (int32_t n : {0, 1, 2, 5}) {
      auto kernel = [&] { InvokePolyGammaKernel<T>(y, x, n, 1); };
      AscendC::SetKernelMode(KernelMode::AIV_MODE);
      ICPU_RUN_KF(kernel, 1);
      EXPECT_TRUE(std::isnan(y[0])) << "n=" << n;
    }
  }

  template <typename T>
  static void PolyGammaEmptyTest() {
    GmBuffer<T> x(1), y(1);
    x[0] = static_cast<T>(1.0);
    for (int32_t n : {0, 1, 2}) {
      auto kernel = [&] { InvokePolyGammaKernel<T>(y, x, n, 0); };
      AscendC::SetKernelMode(KernelMode::AIV_MODE);
      ICPU_RUN_KF(kernel, 1);
    }
    SUCCEED();
  }

  template <typename T>
  static void PolyGammaVectorSizeTest() {
    constexpr uint32_t kSize = 256;
    GmBuffer<T> x(kSize), y(kSize), exp(kSize);

    const T x_vals[] = {static_cast<T>(0.5f), static_cast<T>(1.0f), static_cast<T>(2.0f), static_cast<T>(10.0f)};
    const T exp_tri[] = {static_cast<T>(4.934802f), static_cast<T>(1.644934f), static_cast<T>(0.644934f),
                         static_cast<T>(0.105166f)};
    const T exp_n2[] = {static_cast<T>(-16.828796f), static_cast<T>(-2.404114f), static_cast<T>(-0.404114f),
                        static_cast<T>(-0.011049f)};

    // n=1 (Trigamma)
    for (uint32_t i = 0; i < kSize; ++i) {
      x[i] = x_vals[i % 4];
      exp[i] = exp_tri[i % 4];
    }
    auto k1 = [&] { InvokePolyGammaKernel<T>(y, x, 1, kSize); };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(k1, 1);
    EXPECT_EQ(Valid<T>(y, exp, kSize), 0);

    // n=2 (Zeta path)
    for (uint32_t i = 0; i < kSize; ++i) {
      exp[i] = exp_n2[i % 4];
    }
    auto k2 = [&] { InvokePolyGammaKernel<T>(y, x, 2, kSize); };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(k2, 1);
    EXPECT_EQ(Valid<T>(y, exp, kSize), 0);
  }

  template <typename T>
  static void PolyGammaN2NegativeVectorTest() {
    constexpr uint32_t kSize = 128;
    GmBuffer<T> x(kSize), y(kSize), exp(kSize);

    // Negative x values that trigger 1..10 correction terms
    // Expected values from torch.special.polygamma(2, x)
    const T neg_vals[] = {
        static_cast<T>(-0.3f), static_cast<T>(-0.7f), static_cast<T>(-1.3f), static_cast<T>(-2.3f),
        static_cast<T>(-3.3f), static_cast<T>(-5.3f), static_cast<T>(-7.3f), static_cast<T>(-9.7f),
    };
    const T neg_exp[] = {
        static_cast<T>(67.639076f), static_cast<T>(-69.441628f), static_cast<T>(68.549454f),
        static_cast<T>(68.713829f), static_cast<T>(68.769485f),  static_cast<T>(68.807884f),
        static_cast<T>(68.821030f), static_cast<T>(-68.846985f),
    };
    const uint32_t nv = sizeof(neg_vals) / sizeof(neg_vals[0]);

    for (uint32_t i = 0; i < kSize; ++i) {
      x[i] = neg_vals[i % nv];
      exp[i] = neg_exp[i % nv];
    }
    auto kernel = [&] { InvokePolyGammaKernel<T>(y, x, 2, kSize); };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);
    EXPECT_EQ(Valid<T>(y, exp, kSize), 0);
  }
};

// ====================================================================
// Test Cases
// ====================================================================
TEST_F(TestRegbaseApiPolyGammaUT, PolyGamma_Digamma_Correctness) {
  DigammaCorrectnessTest<float>();
}

TEST_F(TestRegbaseApiPolyGammaUT, PolyGamma_Trigamma_Correctness) {
  TrigammaCorrectnessTest<float>();
}

TEST_F(TestRegbaseApiPolyGammaUT, PolyGamma_N2_Correctness) {
  PolyGammaN2CorrectnessTest<float>();
}

TEST_F(TestRegbaseApiPolyGammaUT, PolyGamma_HighOrder) {
  PolyGammaHighOrderTest<float>();
}

TEST_F(TestRegbaseApiPolyGammaUT, PolyGamma_Zero_Pole) {
  PolyGammaZeroPoleTest<float>();
}

TEST_F(TestRegbaseApiPolyGammaUT, PolyGamma_NegInt_Pole) {
  PolyGammaNegIntPoleTest<float>();
}

TEST_F(TestRegbaseApiPolyGammaUT, PolyGamma_PosInf) {
  PolyGammaPosInfTest<float>();
}

TEST_F(TestRegbaseApiPolyGammaUT, PolyGamma_NegInf) {
  PolyGammaNegInfTest<float>();
}

TEST_F(TestRegbaseApiPolyGammaUT, PolyGamma_NaN) {
  PolyGammaNaNPropagationTest<float>();
}

TEST_F(TestRegbaseApiPolyGammaUT, PolyGamma_Empty) {
  PolyGammaEmptyTest<float>();
}

TEST_F(TestRegbaseApiPolyGammaUT, PolyGamma_VectorSize) {
  PolyGammaVectorSizeTest<float>();
}

TEST_F(TestRegbaseApiPolyGammaUT, PolyGamma_N2_NegativeVector) {
  PolyGammaN2NegativeVectorTest<float>();
}

}  // namespace af
