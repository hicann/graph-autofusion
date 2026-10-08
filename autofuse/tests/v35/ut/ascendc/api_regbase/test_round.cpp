/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. This file is provided on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied.
 */

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/round.h"

using namespace AscendC;

namespace af {
namespace {

constexpr std::array<float, 24> kRoundInputs = {
    -4.5F, -3.5F, -2.5F, -1.5F, -0.75F, -0.5F, -0.25F, -0.0F,   0.0F,   0.25F,   0.5F,    0.75F,
    1.5F,  2.5F,  3.5F,  4.5F,  10.49F, 10.5F, 10.51F, -10.49F, -10.5F, -10.51F, 123.75F, -123.75F,
};

constexpr uint16_t kBfloat16ExpMask = 0x7f80u;
constexpr uint16_t kBfloat16MantMask = 0x007fu;
constexpr uint16_t kBfloat16CanonicalNan = 0xffffu;

template <typename T>
struct RoundInputParam {
  T *src{};
  T *dst{};
  T *expect{};
  uint32_t size{};
};

double RoundToEven(double value) {
  if (!std::isfinite(value) || value == 0.0) {
    return value;
  }

  const double lower = std::floor(value);
  const double fraction = value - lower;
  double rounded = lower;
  if (fraction > 0.5 || (fraction == 0.5 && std::fmod(std::abs(lower), 2.0) != 0.0)) {
    rounded += 1.0;
  }
  return rounded == 0.0 ? std::copysign(0.0, value) : rounded;
}

template <typename T>
uint16_t GetRawBits(T value) {
  static_assert(sizeof(T) == sizeof(uint16_t));
  uint16_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

template <typename T>
T FromRawBits(uint16_t bits) {
  static_assert(sizeof(T) == sizeof(uint16_t));
  T value{};
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

template <typename T>
bool IsClose(T actual, T expect) {
  if constexpr (std::is_same_v<T, bfloat16_t>) {
    const uint16_t actual_bits = GetRawBits(actual);
    const uint16_t expect_bits = GetRawBits(expect);
    const bool expect_nan =
        (expect_bits & kBfloat16ExpMask) == kBfloat16ExpMask && (expect_bits & kBfloat16MantMask) != 0;
    if (expect_nan) {
      return actual_bits == kBfloat16CanonicalNan;
    }
    const bool expect_inf =
        (expect_bits & kBfloat16ExpMask) == kBfloat16ExpMask && (expect_bits & kBfloat16MantMask) == 0;
    if (expect_inf || expect_bits == 0 || expect_bits == 0x8000u) {
      return actual_bits == expect_bits;
    }
  }

  const double actual_value = static_cast<double>(actual);
  const double expect_value = static_cast<double>(expect);
  if (std::isnan(expect_value)) {
    return std::isnan(actual_value);
  }
  if (std::isinf(expect_value)) {
    return std::isinf(actual_value) && std::signbit(actual_value) == std::signbit(expect_value);
  }
  if (expect_value == 0.0) {
    return actual_value == 0.0 && std::signbit(actual_value) == std::signbit(expect_value);
  }
  return actual_value == expect_value;
}

template <typename T>
void InvokeRound(RoundInputParam<T> &param, bool inplace) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> src_buf, dst_buf, tmp_buf;
  const uint32_t aligned_size = AlignUp(param.size, ONE_BLK_SIZE / sizeof(T));
  pipe.InitBuffer(src_buf, sizeof(T) * aligned_size);
  if (!inplace) {
    pipe.InitBuffer(dst_buf, sizeof(T) * aligned_size);
  }
  pipe.InitBuffer(tmp_buf, TMP_UB_SIZE);

  LocalTensor<T> src = src_buf.Get<T>();
  LocalTensor<T> dst = inplace ? src : dst_buf.Get<T>();
  LocalTensor<uint8_t> tmp = tmp_buf.Get<uint8_t>();

  GmToUb(src, param.src, param.size);
  RoundExtend(dst, src, tmp, param.size);
  UbToGm(param.dst, dst, param.size);
}

template <typename T>
void CreateInput(RoundInputParam<T> &param) {
  param.src = static_cast<T *>(GmAlloc(sizeof(T) * param.size));
  param.dst = static_cast<T *>(GmAlloc(sizeof(T) * param.size));
  param.expect = static_cast<T *>(GmAlloc(sizeof(T) * param.size));

  for (uint32_t i = 0; i < param.size; ++i) {
    param.src[i] = static_cast<T>(kRoundInputs[i % kRoundInputs.size()]);
    param.expect[i] = static_cast<T>(RoundToEven(static_cast<double>(param.src[i])));
  }
}

template <typename T>
void CreateSpecialInput(RoundInputParam<T> &param) {
  param.src = static_cast<T *>(GmAlloc(sizeof(T) * param.size));
  param.dst = static_cast<T *>(GmAlloc(sizeof(T) * param.size));
  param.expect = static_cast<T *>(GmAlloc(sizeof(T) * param.size));

  if constexpr (std::is_same_v<T, bfloat16_t>) {
    const std::array<uint16_t, 5> input_bits = {0x7f80u, 0xff80u, 0x7fc1u, 0x0000u, 0x8000u};
    const std::array<uint16_t, 5> expect_bits = {0x7f80u, 0xff80u, kBfloat16CanonicalNan, 0x0000u, 0x8000u};
    for (uint32_t i = 0; i < param.size; ++i) {
      param.src[i] = FromRawBits<T>(input_bits[i % input_bits.size()]);
      param.expect[i] = FromRawBits<T>(expect_bits[i % expect_bits.size()]);
    }
  } else {
    const T nan_value = std::numeric_limits<T>::quiet_NaN();
    const T inf_value = std::numeric_limits<T>::infinity();
    const std::array<T, 5> inputs = {inf_value, -inf_value, nan_value, T(0.0), T(-0.0)};
    for (uint32_t i = 0; i < param.size; ++i) {
      param.src[i] = inputs[i % inputs.size()];
      param.expect[i] = static_cast<T>(RoundToEven(static_cast<double>(param.src[i])));
    }
  }
}

template <typename T>
void FreeInput(RoundInputParam<T> &param) {
  GmFree(param.src);
  GmFree(param.dst);
  GmFree(param.expect);
}

template <typename T>
void ValidateOutput(const RoundInputParam<T> &param) {
  uint32_t mismatch_count = 0;
  for (uint32_t i = 0; i < param.size; ++i) {
    if (!IsClose(param.dst[i], param.expect[i])) {
      ++mismatch_count;
      if constexpr (std::is_same_v<T, bfloat16_t>) {
        printf(
            "diff at index %u: src=%f, actual=%f, expect=%f, src_bits=0x%04x, actual_bits=0x%04x, "
            "expect_bits=0x%04x\n",
            i, static_cast<float>(param.src[i]), static_cast<float>(param.dst[i]), static_cast<float>(param.expect[i]),
            GetRawBits(param.src[i]), GetRawBits(param.dst[i]), GetRawBits(param.expect[i]));
      } else {
        printf("diff at index %u: src=%f, actual=%f, expect=%f\n", i, static_cast<float>(param.src[i]),
               static_cast<float>(param.dst[i]), static_cast<float>(param.expect[i]));
      }
    }
  }
  EXPECT_EQ(mismatch_count, 0U) << " of " << param.size;
}

template <typename T>
void RunRoundTest(uint32_t size, bool inplace, bool special_input = false) {
  RoundInputParam<T> param{};
  param.size = size;
  if (special_input) {
    CreateSpecialInput(param);
  } else {
    CreateInput(param);
  }

  auto kernel = [&param, inplace] { InvokeRound(param, inplace); };
  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);

  ValidateOutput(param);
  FreeInput(param);
}

template <typename T>
void RunRoundSizes(bool inplace) {
  const uint32_t one_rep_size = ONE_REPEAT_BYTE_SIZE / sizeof(T);
  RunRoundTest<T>(1, inplace);
  RunRoundTest<T>(one_rep_size - 1, inplace);
  RunRoundTest<T>(one_rep_size + 1, inplace);
  RunRoundTest<T>((ONE_BLK_SIZE - sizeof(T)) / sizeof(T), inplace);
  RunRoundTest<T>(one_rep_size, inplace);
  RunRoundTest<T>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(T)) / sizeof(T), inplace);
  RunRoundTest<T>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(T), inplace);
}

}  // namespace

class TestRegbaseApiRoundUT : public testing::Test {};

TEST_F(TestRegbaseApiRoundUT, Round_TensorTensor_Half) {
  RunRoundSizes<half>(false);
}

TEST_F(TestRegbaseApiRoundUT, Round_TensorTensor_Float) {
  RunRoundSizes<float>(false);
}

TEST_F(TestRegbaseApiRoundUT, Round_TensorTensor_Bfloat16) {
  RunRoundSizes<bfloat16_t>(false);
}

TEST_F(TestRegbaseApiRoundUT, Round_Inplace_Half) {
  RunRoundSizes<half>(true);
}

TEST_F(TestRegbaseApiRoundUT, Round_Inplace_Float) {
  RunRoundSizes<float>(true);
}

TEST_F(TestRegbaseApiRoundUT, Round_Inplace_Bfloat16) {
  RunRoundSizes<bfloat16_t>(true);
}

TEST_F(TestRegbaseApiRoundUT, Round_Special_Float) {
  RunRoundTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), false, true);
}

TEST_F(TestRegbaseApiRoundUT, Round_Special_Half) {
  RunRoundTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half), false, true);
}

TEST_F(TestRegbaseApiRoundUT, Round_Special_Bfloat16) {
  RunRoundTest<bfloat16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t), false, true);
}

}  // namespace af
