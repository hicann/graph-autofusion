/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cmath>
#include <type_traits>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/tanh.h"

using namespace AscendC;

namespace af {
namespace {

constexpr std::array<float, 26> TANH_INPUTS = {
    0.0f,  -0.0f, 0.1f,  -0.1f, 0.5f,  -0.5f,   0.599f,   -0.599f, 0.6f,    -0.6f, 0.601f, -0.601f, 1.0f,
    -1.0f, 8.0f,  -8.0f, 9.0f,  -9.0f, 9.0109f, -9.0109f, 9.011f,  -9.011f, 20.0f, -20.0f, 100.0f,  -100.0f,
};

template <typename T>
struct TanhExtendInputParam {
  T *dst{};
  T *src{};
  T *expect{};
  uint32_t size{};
};

template <typename T>
constexpr double GetTolerance() {
  return 1e-5;
}

template <typename T>
bool IsClose(T actual, T expect) {
  const double actualValue = static_cast<double>(actual);
  const double expectValue = static_cast<double>(expect);
  if (std::isnan(expectValue)) {
    return std::isnan(actualValue);
  }
  if (std::isinf(expectValue)) {
    return std::isinf(actualValue) && std::signbit(actualValue) == std::signbit(expectValue);
  }
  if (expectValue == 0.0) {
    return actualValue == 0.0 && std::signbit(actualValue) == std::signbit(expectValue);
  }
  const double tolerance = GetTolerance<T>();
  return std::abs(actualValue - expectValue) <= tolerance + tolerance * std::abs(expectValue);
}

template <typename T>
void InvokeTanhExtend(TanhExtendInputParam<T> &param) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> srcBuf, dstBuf, tmpBuf;
  pipe.InitBuffer(srcBuf, sizeof(T) * param.size);
  pipe.InitBuffer(dstBuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
  pipe.InitBuffer(tmpBuf, TMP_UB_SIZE);

  LocalTensor<T> src = srcBuf.Get<T>();
  LocalTensor<T> dst = dstBuf.Get<T>();
  LocalTensor<uint8_t> tmp = tmpBuf.Get<uint8_t>();
  GmToUb(src, param.src, param.size);
  TanhExtend(dst, src, tmp, param.size);
  UbToGm(param.dst, dst, param.size);
}

template <typename T>
void CreateTanhInput(TanhExtendInputParam<T> &param) {
  param.dst = static_cast<T *>(GmAlloc(sizeof(T) * param.size));
  param.src = static_cast<T *>(GmAlloc(sizeof(T) * param.size));
  param.expect = static_cast<T *>(GmAlloc(sizeof(T) * param.size));
  for (uint32_t i = 0; i < param.size; ++i) {
    param.src[i] = static_cast<T>(TANH_INPUTS[i % TANH_INPUTS.size()]);
    param.expect[i] = static_cast<T>(std::tanh(static_cast<double>(param.src[i])));
  }
}

template <typename T>
void FreeTanhInput(TanhExtendInputParam<T> &param) {
  GmFree(param.dst);
  GmFree(param.src);
  GmFree(param.expect);
}

template <typename T>
void RunTanhExtendTest(uint32_t size) {
  TanhExtendInputParam<T> param{};
  param.size = size;
  CreateTanhInput(param);

  auto kernel = [&param] { InvokeTanhExtend(param); };
  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);

  uint32_t mismatchCount = 0;
  for (uint32_t i = 0; i < param.size; ++i) {
    if (!IsClose(param.dst[i], param.expect[i])) {
      ++mismatchCount;
      printf("diff at index %u: src=%f, actual=%f, expect=%f\n", i, static_cast<float>(param.src[i]),
             static_cast<float>(param.dst[i]), static_cast<float>(param.expect[i]));
    }
  }
  EXPECT_EQ(mismatchCount, 0U) << " of " << size;
  FreeTanhInput(param);
}

template <typename T>
void RunTanhExtendSizes() {
  RunTanhExtendTest<T>((ONE_BLK_SIZE - sizeof(T)) / sizeof(T));
  RunTanhExtendTest<T>(ONE_REPEAT_BYTE_SIZE / sizeof(T));
  RunTanhExtendTest<T>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(T)) / sizeof(T));
  RunTanhExtendTest<T>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(T));
}

}  // namespace

class TestApiTanhExtendUT : public testing::Test {};

TEST_F(TestApiTanhExtendUT, TanhExtend_Half_Success) {
  RunTanhExtendSizes<half>();
}

TEST_F(TestApiTanhExtendUT, TanhExtend_Float_Success) {
  RunTanhExtendSizes<float>();
}

TEST_F(TestApiTanhExtendUT, TanhExtend_Bfloat16_Success) {
  RunTanhExtendSizes<bfloat16_t>();
}

}  // namespace af
