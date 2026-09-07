/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * test_compare.cpp
 */

#include <cmath>
#include "gtest/gtest.h"
#include "test_api_utils.h"
#include "tikicpulib.h"
#include "utils.h"
// 保持在utils.h之后
#include "duplicate.h"
// 保持在duplicate.h之后
#include "compare.h"

using namespace AscendC;

namespace af {
template <typename O, typename I>
struct CompareInputParam {
  O *y{};
  bool *exp{};
  I *src0{};
  I src1;
  CMPMODE mode{CMPMODE::EQ};
  uint32_t size{0};
  uint32_t out_size{0};
  BinaryRepeatParams a;
};

template <typename O, typename I>
struct TensorCompareInputParam {
  O *y{};
  bool *exp{};
  I *src0{};
  I *src1{};
  CMPMODE mode{CMPMODE::EQ};
  uint32_t size{0};
  uint32_t out_size{0};
  BinaryRepeatParams a;
};

class TestApiCompareUT : public testing::Test {
 protected:
  template <typename O, typename I>
  static void CreateInput(CompareInputParam<O, I> &param, float def_src1) {
    // 构造测试输入和预期结果
    param.y = static_cast<O *>(AscendC::GmAlloc(sizeof(O) * param.size));
    param.exp = static_cast<bool *>(AscendC::GmAlloc(sizeof(bool) * param.size));
    param.src0 = static_cast<I *>(AscendC::GmAlloc(sizeof(I) * param.size));

    if constexpr (std::is_same_v<I, int64_t>) {
      param.src1 = 0xAAAAAAAABBBBBBBB;
    } else {
      param.src1 = def_src1;
    }

    int input_range = 10;
    std::mt19937 eng(1);                                  // Seed the generator
    std::uniform_int_distribution distr(0, input_range);  // Define the range

    for (int i = 0; i < param.size; i++) {
      auto input = distr(eng);  // Use the secure random number generator
      param.src0[i] = input;
      switch (param.mode) {
        case CMPMODE::EQ:
          if (input > 5 || i == param.size - 1) {
            param.src0[i] = param.src1;
            param.exp[i] = true;
          } else {
            param.exp[i] = DefaultCompare(input, param.src1);
          }
          break;
        case CMPMODE::NE:
          if (input > 5 || i == param.size - 1) {
            param.src0[i] = param.src1;
            param.exp[i] = false;
          } else {
            param.exp[i] = !DefaultCompare(param.src0[i], param.src1);
          }
          break;
        case CMPMODE::GE:
          if constexpr (std::is_same_v<I, half>) {
            param.exp[i] = static_cast<half>(input) >= param.src1;
          } else {
            param.exp[i] = input >= param.src1;
          }
          break;
        case CMPMODE::LE:
          if constexpr (std::is_same_v<I, half>) {
            param.exp[i] = static_cast<half>(input) <= param.src1;
          } else {
            param.exp[i] = input <= param.src1;
          }
          break;
        case CMPMODE::GT:
          if constexpr (std::is_same_v<I, half>) {
            param.exp[i] = static_cast<half>(input) > param.src1;
          } else {
            param.exp[i] = input > param.src1;
          }
          break;
        case CMPMODE::LT:
          if constexpr (std::is_same_v<I, half>) {
            param.exp[i] = static_cast<half>(input) < param.src1;
          } else {
            param.exp[i] = input < param.src1;
          }
          break;
        default:
          break;
      }
    }
  }

  template <typename O, typename I>
  static uint32_t Valid(CompareInputParam<O, I> &param) {
    uint32_t diff_count = 0;

    for (uint32_t i = 0; i < param.size; i++) {
      if (static_cast<bool>(param.y[i]) != param.exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename O, typename I>
  static void InvokeKernel(CompareInputParam<O, I> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, ybuf, tmp;
    tpipe.InitBuffer(x1buf, sizeof(I) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(O) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(O)));
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<I> l_x1 = x1buf.Get<I>();
    LocalTensor<O> l_y = ybuf.Get<O>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x1, param.src0, param.size);
    CompareScalarExtend(l_y, l_x1, param.src1, param.mode, param.size, l_tmp);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename O, typename I>
  static void CompareTest(uint32_t size, CMPMODE mode, float def_src1 = 4.5) {
    CompareInputParam<O, I> param{};
    param.size = size;
    param.mode = mode;

    CreateInput(param, def_src1);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeKernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src0);
  }

  /* -------------------- 输入是两个tensor相关的测试基础方法定义-------------------- */

  template <typename O, typename I>
  static void InvokeKernelWithTwoTensorInput(TensorCompareInputParam<O, I> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;

    tpipe.InitBuffer(x1buf, sizeof(I) * param.size);
    tpipe.InitBuffer(x2buf, sizeof(I) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(O) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(O)));
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<I> l_x1 = x1buf.Get<I>();
    LocalTensor<I> l_x2 = x2buf.Get<I>();

    LocalTensor<O> l_y = ybuf.Get<O>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x1, param.src0, param.size);
    GmToUb(l_x2, param.src1, param.size);
    CompareExtend(l_y, l_x1, l_x2, param.mode, param.size, l_tmp);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename O, typename I>
  static void CreateTensorInput(TensorCompareInputParam<O, I> &param, float def_src1) {
    // 构造测试输入和预期结果
    param.y = static_cast<O *>(AscendC::GmAlloc(sizeof(O) * param.size));
    param.exp = static_cast<bool *>(AscendC::GmAlloc(sizeof(bool) * param.size));
    param.src0 = static_cast<I *>(AscendC::GmAlloc(sizeof(I) * param.size));
    param.src1 = static_cast<I *>(AscendC::GmAlloc(sizeof(I) * param.size));
    I src1_val;
    if constexpr (std::is_same_v<I, int64_t>) {
      src1_val = 0xAAAAAAAABBBBBBBB;
    } else {
      src1_val = def_src1;
    }
    (void)src1_val;

    int input_range = 10;

    // 构造src0的随机生成器
    std::mt19937 eng(1);
    std::uniform_int_distribution distr(0, input_range);  // Define the range

    // 构造src1的随机生成器
    std::mt19937 eng1(3);                                  // Seed the generator
    std::uniform_int_distribution distr1(0, input_range);  // Define the range

    for (int i = 0; i < param.size; i++) {
      auto input = distr(eng);  // Use the secure random number generator
      auto input1 = distr1(eng1);
      param.src0[i] = input;
      param.src1[i] = input1;

      switch (param.mode) {
        case CMPMODE::EQ:
          if (input > 5 || i == param.size - 1) {
            param.src0[i] = param.src1[i];
            param.exp[i] = true;
          } else {
            param.exp[i] = DefaultCompare(param.src0[i], param.src1[i]);
          }
          break;
        case CMPMODE::NE:
          if (input > 5 || i == param.size - 1) {
            param.src0[i] = param.src1[i];
            param.exp[i] = false;
          } else {
            param.exp[i] = !DefaultCompare(param.src0[i], param.src1[i]);
          }
          break;
        case CMPMODE::GE:
          if constexpr (std::is_same_v<I, half>) {
            param.exp[i] = static_cast<half>(param.src0[i]) >= param.src1[i];
          } else {
            param.exp[i] = param.src0[i] >= param.src1[i];
          }
          break;
        case CMPMODE::LE:
          if constexpr (std::is_same_v<I, half>) {
            param.exp[i] = static_cast<half>(param.src0[i]) <= param.src1[i];
          } else {
            param.exp[i] = param.src0[i] <= param.src1[i];
          }
          break;
        case CMPMODE::GT:
          param.exp[i] = param.src0[i] > param.src1[i];
          break;
        case CMPMODE::LT:
          if constexpr (std::is_same_v<I, half>) {
            param.exp[i] = static_cast<half>(param.src0[i]) < param.src1[i];
          } else {
            param.exp[i] = param.src0[i] < param.src1[i];
          }
          break;
        default:
          break;
      }
    }
  }

  template <typename O>
  static uint32_t Valid(O *y, bool *exp, size_t comp_size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < comp_size; i++) {
      if (static_cast<bool>(y[i]) != exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename O, typename I>
  static void TensorCompareTest(uint32_t size, CMPMODE mode, float def_src1 = 4.5) {
    TensorCompareInputParam<O, I> param{};
    param.size = size;
    param.mode = mode;

    CreateTensorInput(param, def_src1);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeKernelWithTwoTensorInput(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0) << "of  " << param.size;
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src0);
    AscendC::GmFree(param.src1);
  }
};

// 场景01：EQ-float
TEST_F(TestApiCompareUT, Compare_Eq_float_uint8) {
  CompareTest<uint8_t, float>(ONE_BLK_SIZE / sizeof(float), CMPMODE::EQ);
  CompareTest<uint8_t, float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::EQ);
  CompareTest<uint8_t, float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::EQ);
  CompareTest<uint8_t, float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::EQ);
  CompareTest<uint8_t, float>((ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::EQ);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - sizeof(float)) / sizeof(float), CMPMODE::EQ);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + sizeof(float)) / sizeof(float), CMPMODE::EQ);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), CMPMODE::EQ);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(float), CMPMODE::EQ);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::EQ);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::EQ);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::EQ);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::EQ);
  CompareTest<uint8_t, float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::EQ);
  CompareTest<uint8_t, float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                               (ONE_BLK_SIZE - sizeof(float))) /
                                  sizeof(float),
                              CMPMODE::EQ);
}

TEST_F(TestApiCompareUT, Compare_Eq_float_float_with_small_tail) {
  const uint32_t max_block_cnt = KernelUtils::BlkAlign<uint8_t>((64 * MAX_REPEAT_TIMES) / 8) / sizeof(float);
  CompareTest<float, float>(max_block_cnt + ONE_BLK_SIZE / sizeof(float), CMPMODE::EQ);
}

// 场景02：EQ-half
TEST_F(TestApiCompareUT, Compare_Eq_half_uint8) {
  CompareTest<uint8_t, half>(ONE_BLK_SIZE / sizeof(half), CMPMODE::EQ);
  CompareTest<uint8_t, half>(ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::EQ);
  CompareTest<uint8_t, half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::EQ);
  CompareTest<uint8_t, half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::EQ);
  CompareTest<uint8_t, half>((ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::EQ);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - sizeof(half)) / sizeof(half), CMPMODE::EQ);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + sizeof(half)) / sizeof(half), CMPMODE::EQ);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), CMPMODE::EQ);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(half), CMPMODE::EQ);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::EQ);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::EQ);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::EQ);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::EQ);
  CompareTest<uint8_t, half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::EQ);
  CompareTest<uint8_t, half>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                              (ONE_BLK_SIZE - sizeof(half))) /
                                 sizeof(half),
                             CMPMODE::EQ);
}

// 场景03：EQ-int32
TEST_F(TestApiCompareUT, Compare_Eq_int32_uint8) {
  CompareTest<uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t), CMPMODE::EQ);
  CompareTest<uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::EQ);
  CompareTest<uint8_t, int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::EQ);
  CompareTest<uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::EQ);
  CompareTest<uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::EQ);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::EQ);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::EQ);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::EQ);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::EQ);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::EQ);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::EQ);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::EQ);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::EQ);
  CompareTest<uint8_t, int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::EQ);
  CompareTest<uint8_t, int32_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                 (ONE_BLK_SIZE - sizeof(int32_t))) /
                                    sizeof(int32_t),
                                CMPMODE::EQ);
}

// 场景04：EQ-int64
TEST_F(TestApiCompareUT, Compare_Eq_int64_uint8) {
  CompareTest<uint8_t, int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::EQ);
  CompareTest<uint8_t, int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::EQ);
  CompareTest<uint8_t, int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::EQ);
  CompareTest<uint8_t, int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::EQ);
  CompareTest<uint8_t, int64_t>((ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::EQ);
  CompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::EQ);
  CompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::EQ);
  CompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), CMPMODE::EQ);
  CompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int64_t), CMPMODE::EQ);
  CompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::EQ);
  CompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::EQ);
  CompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::EQ);
  CompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::EQ);
  CompareTest<uint8_t, int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::EQ);
  CompareTest<uint8_t, int64_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                 (ONE_BLK_SIZE - sizeof(int64_t))) /
                                    sizeof(int64_t),
                                CMPMODE::EQ);
  CompareTest<uint8_t, int64_t>((13 * ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                CMPMODE::EQ);
}

// 场景21：NE-float
TEST_F(TestApiCompareUT, Compare_Ne_float_uint8) {
  CompareTest<uint8_t, float>(ONE_BLK_SIZE / sizeof(float), CMPMODE::NE);
  CompareTest<uint8_t, float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::NE);
  CompareTest<uint8_t, float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::NE);
  CompareTest<uint8_t, float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::NE);
  CompareTest<uint8_t, float>((ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::NE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - sizeof(float)) / sizeof(float), CMPMODE::NE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + sizeof(float)) / sizeof(float), CMPMODE::NE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), CMPMODE::NE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(float), CMPMODE::NE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::NE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::NE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::NE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::NE);
  CompareTest<uint8_t, float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::NE);
  CompareTest<uint8_t, float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                               (ONE_BLK_SIZE - sizeof(float))) /
                                  sizeof(float),
                              CMPMODE::NE);
}

// 场景22：NE-half
TEST_F(TestApiCompareUT, Compare_Ne_half_uint8) {
  CompareTest<uint8_t, half>(ONE_BLK_SIZE / sizeof(half), CMPMODE::NE);
  CompareTest<uint8_t, half>(ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::NE);
  CompareTest<uint8_t, half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::NE);
  CompareTest<uint8_t, half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::NE);
  CompareTest<uint8_t, half>((ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::NE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - sizeof(half)) / sizeof(half), CMPMODE::NE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + sizeof(half)) / sizeof(half), CMPMODE::NE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), CMPMODE::NE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(half), CMPMODE::NE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::NE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::NE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::NE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::NE);
  CompareTest<uint8_t, half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::NE);
  CompareTest<uint8_t, half>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                              (ONE_BLK_SIZE - sizeof(half))) /
                                 sizeof(half),
                             CMPMODE::NE);
}

// 场景23：NE-int32 not support

// 场景24：NE-int64
TEST_F(TestApiCompareUT, Compare_Ne_int64_uint8) {
  CompareTest<uint8_t, int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::NE);
  CompareTest<uint8_t, int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::NE);
  CompareTest<uint8_t, int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::NE);
  CompareTest<uint8_t, int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::NE);
  CompareTest<uint8_t, int64_t>((ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::NE);
  CompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::NE);
  CompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::NE);
  CompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), CMPMODE::NE);
  CompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int64_t), CMPMODE::NE);
  CompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::NE);
  CompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::NE);
  CompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::NE);
  CompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::NE);
  CompareTest<uint8_t, int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::NE);
  CompareTest<uint8_t, int64_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                 (ONE_BLK_SIZE - sizeof(int64_t))) /
                                    sizeof(int64_t),
                                CMPMODE::NE);
  CompareTest<uint8_t, int64_t>((13 * ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                CMPMODE::NE);
}

// 场景41：GE-float
TEST_F(TestApiCompareUT, Compare_Ge_float_uint8) {
  CompareTest<uint8_t, float>(ONE_BLK_SIZE / sizeof(float), CMPMODE::GE);
  CompareTest<uint8_t, float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::GE);
  CompareTest<uint8_t, float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::GE);
  CompareTest<uint8_t, float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::GE);
  CompareTest<uint8_t, float>((ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::GE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - sizeof(float)) / sizeof(float), CMPMODE::GE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + sizeof(float)) / sizeof(float), CMPMODE::GE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), CMPMODE::GE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(float), CMPMODE::GE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::GE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::GE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::GE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::GE);
  CompareTest<uint8_t, float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::GE);
  CompareTest<uint8_t, float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                               (ONE_BLK_SIZE - sizeof(float))) /
                                  sizeof(float),
                              CMPMODE::GE);
}

// 场景42：GE-half
TEST_F(TestApiCompareUT, Compare_Ge_half_uint8) {
  CompareTest<uint8_t, half>(ONE_BLK_SIZE / sizeof(half), CMPMODE::GE);
  CompareTest<uint8_t, half>(ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::GE);
  CompareTest<uint8_t, half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::GE);
  CompareTest<uint8_t, half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::GE);
  CompareTest<uint8_t, half>((ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::GE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - sizeof(half)) / sizeof(half), CMPMODE::GE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + sizeof(half)) / sizeof(half), CMPMODE::GE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), CMPMODE::GE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(half), CMPMODE::GE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::GE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::GE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::GE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::GE);
  CompareTest<uint8_t, half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::GE);
  CompareTest<uint8_t, half>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                              (ONE_BLK_SIZE - sizeof(half))) /
                                 sizeof(half),
                             CMPMODE::GE);
}

// 场景43：GE-int32 not support

// 场景61：LE-float
TEST_F(TestApiCompareUT, Compare_Le_float_uint8) {
  CompareTest<uint8_t, float>(ONE_BLK_SIZE / sizeof(float), CMPMODE::LE);
  CompareTest<uint8_t, float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::LE);
  CompareTest<uint8_t, float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::LE);
  CompareTest<uint8_t, float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::LE);
  CompareTest<uint8_t, float>((ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::LE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - sizeof(float)) / sizeof(float), CMPMODE::LE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + sizeof(float)) / sizeof(float), CMPMODE::LE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), CMPMODE::LE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(float), CMPMODE::LE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::LE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::LE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::LE);
  CompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::LE);
  CompareTest<uint8_t, float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::LE);
  CompareTest<uint8_t, float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                               (ONE_BLK_SIZE - sizeof(float))) /
                                  sizeof(float),
                              CMPMODE::LE);
}

// 场景62：LE-half
TEST_F(TestApiCompareUT, Compare_Le_half_uint8) {
  CompareTest<uint8_t, half>(ONE_BLK_SIZE / sizeof(half), CMPMODE::LE);
  CompareTest<uint8_t, half>(ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::LE);
  CompareTest<uint8_t, half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::LE);
  CompareTest<uint8_t, half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::LE);
  CompareTest<uint8_t, half>((ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::LE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - sizeof(half)) / sizeof(half), CMPMODE::LE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + sizeof(half)) / sizeof(half), CMPMODE::LE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), CMPMODE::LE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(half), CMPMODE::LE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::LE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::LE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::LE);
  CompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::LE);
  CompareTest<uint8_t, half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::LE);
  CompareTest<uint8_t, half>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                              (ONE_BLK_SIZE - sizeof(half))) /
                                 sizeof(half),
                             CMPMODE::LE);
}

// 场景63：GT-int32
TEST_F(TestApiCompareUT, Compare_Gt_int32_uint8) {
  CompareTest<uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t), CMPMODE::GT);
  CompareTest<uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::GT);
  CompareTest<uint8_t, int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::GT);
  CompareTest<uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GT);
  CompareTest<uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GT);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GT);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GT);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::GT);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::GT);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GT);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GT);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GT);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GT);
  CompareTest<uint8_t, int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::GT);
  CompareTest<uint8_t, int32_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                 (ONE_BLK_SIZE - sizeof(int32_t))) /
                                    sizeof(int32_t),
                                CMPMODE::GT);
}

// 场景64：GE-int32
TEST_F(TestApiCompareUT, Compare_Ge_int32_uint8) {
  CompareTest<uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t), CMPMODE::GE);
  CompareTest<uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::GE);
  CompareTest<uint8_t, int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::GE);
  CompareTest<uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GE);
  CompareTest<uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::GE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::GE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GE);
  CompareTest<uint8_t, int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::GE);
  CompareTest<uint8_t, int32_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                 (ONE_BLK_SIZE - sizeof(int32_t))) /
                                    sizeof(int32_t),
                                CMPMODE::GE);
}

// 场景65：LE-int32
TEST_F(TestApiCompareUT, Compare_Le_int32_uint8) {
  CompareTest<uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t), CMPMODE::LE);
  CompareTest<uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::LE);
  CompareTest<uint8_t, int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::LE);
  CompareTest<uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LE);
  CompareTest<uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::LE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::LE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LE);
  CompareTest<uint8_t, int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::LE);
  CompareTest<uint8_t, int32_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                 (ONE_BLK_SIZE - sizeof(int32_t))) /
                                    sizeof(int32_t),
                                CMPMODE::LE);
}

// 场景66：LT-int32
TEST_F(TestApiCompareUT, Compare_Lt_int32_uint8) {
  CompareTest<uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t), CMPMODE::LT);
  CompareTest<uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::LT);
  CompareTest<uint8_t, int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::LT);
  CompareTest<uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LT);
  CompareTest<uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LT);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LT);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LT);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::LT);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::LT);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LT);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LT);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LT);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LT);
  CompareTest<uint8_t, int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::LT);
  CompareTest<uint8_t, int32_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                 (ONE_BLK_SIZE - sizeof(int32_t))) /
                                    sizeof(int32_t),
                                CMPMODE::LT);
}

// 场景67：NE-int32
TEST_F(TestApiCompareUT, Compare_Ne_int32_uint8) {
  CompareTest<uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t), CMPMODE::NE);
  CompareTest<uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::NE);
  CompareTest<uint8_t, int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::NE);
  CompareTest<uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::NE);
  CompareTest<uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::NE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::NE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::NE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::NE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::NE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::NE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::NE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::NE);
  CompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::NE);
  CompareTest<uint8_t, int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::NE);
  CompareTest<uint8_t, int32_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                 (ONE_BLK_SIZE - sizeof(int32_t))) /
                                    sizeof(int32_t),
                                CMPMODE::NE);
}

/* Begin: 输入是两个tensor的 compare 测试--------------------------------------- */

// Tensor NE-int64
TEST_F(TestApiCompareUT, Compare_Ne_input_tensor_int64_output_tensor_uint8) {
  TensorCompareTest<uint8_t, int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int64_t>((ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int64_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                      CMPMODE::NE);
  TensorCompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t),
                                      CMPMODE::NE);
  TensorCompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                      CMPMODE::NE);
  TensorCompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t),
                                      CMPMODE::NE);
  TensorCompareTest<uint8_t, int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int64_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                       (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(int64_t))) /
                                          sizeof(int64_t),
                                      CMPMODE::NE);
  TensorCompareTest<uint8_t, int64_t>((13 * ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                      CMPMODE::NE);
}
// Tensor EQ-int64
TEST_F(TestApiCompareUT, Compare_Eq_input_tensor_int64_output_uint8) {
  TensorCompareTest<uint8_t, int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int64_t>((ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int64_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                      CMPMODE::EQ);
  TensorCompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t),
                                      CMPMODE::EQ);
  TensorCompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                      CMPMODE::EQ);
  TensorCompareTest<uint8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t),
                                      CMPMODE::EQ);
  TensorCompareTest<uint8_t, int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int64_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                       (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(int64_t))) /
                                          sizeof(int64_t),
                                      CMPMODE::EQ);
  TensorCompareTest<uint8_t, int64_t>((13 * ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                      CMPMODE::EQ);
}

// Tensor GT-int64
TEST_F(TestApiCompareUT, Compare_GT_input_tensor_int64_output_tensor_int8) {
  TensorCompareTest<int8_t, int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::GT);
  TensorCompareTest<int8_t, int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::GT);
  TensorCompareTest<int8_t, int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::GT);
  TensorCompareTest<int8_t, int64_t>((ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::GT);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::GT);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::GT);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), CMPMODE::GT);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int64_t), CMPMODE::GT);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                     CMPMODE::GT);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t),
                                     CMPMODE::GT);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                     CMPMODE::GT);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t),
                                     CMPMODE::GT);
  TensorCompareTest<int8_t, int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::GT);
  TensorCompareTest<int8_t, int64_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                      (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(int64_t))) /
                                         sizeof(int64_t),
                                     CMPMODE::GT);
  TensorCompareTest<int8_t, int64_t>((13 * ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                     CMPMODE::GT);
}

// Tensor GE-int64
TEST_F(TestApiCompareUT, Compare_GE_input_tensor_int64_output_tensor_int8) {
  TensorCompareTest<int8_t, int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::GE);
  TensorCompareTest<int8_t, int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::GE);
  TensorCompareTest<int8_t, int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::GE);
  TensorCompareTest<int8_t, int64_t>((ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::GE);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::GE);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::GE);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), CMPMODE::GE);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int64_t), CMPMODE::GE);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                     CMPMODE::GE);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t),
                                     CMPMODE::GE);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                     CMPMODE::GE);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t),
                                     CMPMODE::GE);
  TensorCompareTest<int8_t, int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::GE);
  TensorCompareTest<int8_t, int64_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                      (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(int64_t))) /
                                         sizeof(int64_t),
                                     CMPMODE::GE);
  TensorCompareTest<int8_t, int64_t>((13 * ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                     CMPMODE::GE);
}

// Tensor LE-int64
TEST_F(TestApiCompareUT, Compare_LE_input_tensor_int64_output_tensor_int8) {
  TensorCompareTest<int8_t, int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::LE);
  TensorCompareTest<int8_t, int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::LE);
  TensorCompareTest<int8_t, int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::LE);
  TensorCompareTest<int8_t, int64_t>((ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::LE);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int64_t)) / sizeof(int64_t), CMPMODE::LE);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int64_t)) / sizeof(int64_t), CMPMODE::LE);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), CMPMODE::LE);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int64_t), CMPMODE::LE);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                     CMPMODE::LE);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t),
                                     CMPMODE::LE);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                     CMPMODE::LE);
  TensorCompareTest<int8_t, int64_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int64_t)) / sizeof(int64_t),
                                     CMPMODE::LE);
  TensorCompareTest<int8_t, int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), CMPMODE::LE);
  TensorCompareTest<int8_t, int64_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                      (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(int64_t))) /
                                         sizeof(int64_t),
                                     CMPMODE::LE);
  TensorCompareTest<int8_t, int64_t>((13 * ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                                     CMPMODE::LE);
}

// Tensor EQ-float
TEST_F(TestApiCompareUT, Compare_Eq_input_tensor_float_output_uint8) {
  TensorCompareTest<uint8_t, float>(ONE_BLK_SIZE / sizeof(float), CMPMODE::EQ);
  TensorCompareTest<uint8_t, float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::EQ);
  TensorCompareTest<uint8_t, float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::EQ);
  TensorCompareTest<uint8_t, float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::EQ);
  TensorCompareTest<uint8_t, float>((ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::EQ);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - sizeof(float)) / sizeof(float), CMPMODE::EQ);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + sizeof(float)) / sizeof(float), CMPMODE::EQ);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), CMPMODE::EQ);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(float), CMPMODE::EQ);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::EQ);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::EQ);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::EQ);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::EQ);
  TensorCompareTest<uint8_t, float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::EQ);
  TensorCompareTest<uint8_t, float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                     (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(float))) /
                                        sizeof(float),
                                    CMPMODE::EQ);
}

TEST_F(TestApiCompareUT, Compare_Eq_input_tensor_float_output_float_with_small_tail) {
  const uint32_t max_block_cnt = KernelUtils::BlkAlign<uint8_t>((64 * MAX_REPEAT_TIMES) / 8) / sizeof(float);
  TensorCompareTest<float, float>(max_block_cnt + ONE_BLK_SIZE / sizeof(float), CMPMODE::EQ);
}

// Tensor EQ-half
TEST_F(TestApiCompareUT, Compare_Eq_input_tensor_half_output_uint8) {
  TensorCompareTest<uint8_t, half>(ONE_BLK_SIZE / sizeof(half), CMPMODE::EQ);
  TensorCompareTest<uint8_t, half>(ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::EQ);
  TensorCompareTest<uint8_t, half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::EQ);
  TensorCompareTest<uint8_t, half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::EQ);
  TensorCompareTest<uint8_t, half>((ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::EQ);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - sizeof(half)) / sizeof(half), CMPMODE::EQ);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + sizeof(half)) / sizeof(half), CMPMODE::EQ);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), CMPMODE::EQ);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(half), CMPMODE::EQ);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::EQ);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::EQ);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::EQ);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::EQ);
  TensorCompareTest<uint8_t, half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::EQ);
  TensorCompareTest<uint8_t, half>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                    (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(half))) /
                                       sizeof(half),
                                   CMPMODE::EQ);
}

// Tensor EQ-int32
TEST_F(TestApiCompareUT, Compare_Eq_input_tensor_int32_output_uint8) {
  TensorCompareTest<uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::EQ);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::EQ);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::EQ);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::EQ);
  TensorCompareTest<uint8_t, int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::EQ);
  TensorCompareTest<uint8_t, int32_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                       (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(int32_t))) /
                                          sizeof(int32_t),
                                      CMPMODE::EQ);
}

// 场景21：NE-float
TEST_F(TestApiCompareUT, Compare_Ne_input_tensor_float_output_uint8) {
  TensorCompareTest<uint8_t, float>(ONE_BLK_SIZE / sizeof(float), CMPMODE::NE);
  TensorCompareTest<uint8_t, float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::NE);
  TensorCompareTest<uint8_t, float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::NE);
  TensorCompareTest<uint8_t, float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::NE);
  TensorCompareTest<uint8_t, float>((ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::NE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - sizeof(float)) / sizeof(float), CMPMODE::NE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + sizeof(float)) / sizeof(float), CMPMODE::NE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), CMPMODE::NE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(float), CMPMODE::NE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::NE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::NE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::NE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::NE);
  TensorCompareTest<uint8_t, float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::NE);
  TensorCompareTest<uint8_t, float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                     (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(float))) /
                                        sizeof(float),
                                    CMPMODE::NE);
}

// 场景22：NE-half
TEST_F(TestApiCompareUT, Compare_Ne_input_tensor_half_output_uint8) {
  TensorCompareTest<uint8_t, half>(ONE_BLK_SIZE / sizeof(half), CMPMODE::NE);
  TensorCompareTest<uint8_t, half>(ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::NE);
  TensorCompareTest<uint8_t, half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::NE);
  TensorCompareTest<uint8_t, half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::NE);
  TensorCompareTest<uint8_t, half>((ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::NE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - sizeof(half)) / sizeof(half), CMPMODE::NE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + sizeof(half)) / sizeof(half), CMPMODE::NE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), CMPMODE::NE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(half), CMPMODE::NE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::NE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::NE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::NE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::NE);
  TensorCompareTest<uint8_t, half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::NE);
  TensorCompareTest<uint8_t, half>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                    (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(half))) /
                                       sizeof(half),
                                   CMPMODE::NE);
}

// 场景41：GE-float
TEST_F(TestApiCompareUT, Compare_Ge_input_tensor_float_output_uint8) {
  TensorCompareTest<uint8_t, float>(ONE_BLK_SIZE / sizeof(float), CMPMODE::GE);
  TensorCompareTest<uint8_t, float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::GE);
  TensorCompareTest<uint8_t, float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::GE);
  TensorCompareTest<uint8_t, float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::GE);
  TensorCompareTest<uint8_t, float>((ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::GE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - sizeof(float)) / sizeof(float), CMPMODE::GE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + sizeof(float)) / sizeof(float), CMPMODE::GE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), CMPMODE::GE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(float), CMPMODE::GE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::GE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::GE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::GE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::GE);
  TensorCompareTest<uint8_t, float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::GE);
  TensorCompareTest<uint8_t, float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                     (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(float))) /
                                        sizeof(float),
                                    CMPMODE::GE);
}

// 场景42：GE-half
TEST_F(TestApiCompareUT, Compare_Ge_input_tensor_half_output_uint8) {
  TensorCompareTest<uint8_t, half>(ONE_BLK_SIZE / sizeof(half), CMPMODE::GE);
  TensorCompareTest<uint8_t, half>(ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::GE);
  TensorCompareTest<uint8_t, half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::GE);
  TensorCompareTest<uint8_t, half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::GE);
  TensorCompareTest<uint8_t, half>((ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::GE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - sizeof(half)) / sizeof(half), CMPMODE::GE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + sizeof(half)) / sizeof(half), CMPMODE::GE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), CMPMODE::GE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(half), CMPMODE::GE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::GE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::GE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::GE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::GE);
  TensorCompareTest<uint8_t, half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::GE);
  TensorCompareTest<uint8_t, half>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                    (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(half))) /
                                       sizeof(half),
                                   CMPMODE::GE);
}

// 场景61：LE-float
TEST_F(TestApiCompareUT, Compare_Le_input_tensor_float_output_uint8) {
  TensorCompareTest<uint8_t, float>(ONE_BLK_SIZE / sizeof(float), CMPMODE::LE);
  TensorCompareTest<uint8_t, float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::LE);
  TensorCompareTest<uint8_t, float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::LE);
  TensorCompareTest<uint8_t, float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::LE);
  TensorCompareTest<uint8_t, float>((ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::LE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - sizeof(float)) / sizeof(float), CMPMODE::LE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + sizeof(float)) / sizeof(float), CMPMODE::LE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), CMPMODE::LE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(float), CMPMODE::LE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::LE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::LE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(float)) / sizeof(float), CMPMODE::LE);
  TensorCompareTest<uint8_t, float>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(float)) / sizeof(float), CMPMODE::LE);
  TensorCompareTest<uint8_t, float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float), CMPMODE::LE);
  TensorCompareTest<uint8_t, float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                     (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(float))) /
                                        sizeof(float),
                                    CMPMODE::LE);
}

// 场景62：LE-half
TEST_F(TestApiCompareUT, Compare_Le_input_tensor_half_output_uint8) {
  TensorCompareTest<uint8_t, half>(ONE_BLK_SIZE / sizeof(half), CMPMODE::LE);
  TensorCompareTest<uint8_t, half>(ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::LE);
  TensorCompareTest<uint8_t, half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::LE);
  TensorCompareTest<uint8_t, half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::LE);
  TensorCompareTest<uint8_t, half>((ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::LE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - sizeof(half)) / sizeof(half), CMPMODE::LE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + sizeof(half)) / sizeof(half), CMPMODE::LE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), CMPMODE::LE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(half), CMPMODE::LE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::LE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::LE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(half)) / sizeof(half), CMPMODE::LE);
  TensorCompareTest<uint8_t, half>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(half)) / sizeof(half), CMPMODE::LE);
  TensorCompareTest<uint8_t, half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(half), CMPMODE::LE);
  TensorCompareTest<uint8_t, half>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                    (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(half))) /
                                       sizeof(half),
                                   CMPMODE::LE);
}

// 场景63：GT-int32
TEST_F(TestApiCompareUT, Compare_Gt_input_tensor_int32_output_uint8) {
  TensorCompareTest<uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t), CMPMODE::GT);
  TensorCompareTest<uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::GT);
  TensorCompareTest<uint8_t, int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::GT);
  TensorCompareTest<uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GT);
  TensorCompareTest<uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GT);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GT);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GT);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::GT);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::GT);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::GT);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::GT);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::GT);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::GT);
  TensorCompareTest<uint8_t, int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::GT);
  TensorCompareTest<uint8_t, int32_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                       (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(int32_t))) /
                                          sizeof(int32_t),
                                      CMPMODE::GT);
}

// 场景64：NE-int32
TEST_F(TestApiCompareUT, Compare_Ne_input_tensor_int32_output_uint8) {
  TensorCompareTest<uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::NE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::NE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::NE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::NE);
  TensorCompareTest<uint8_t, int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::NE);
  TensorCompareTest<uint8_t, int32_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                       (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(int32_t))) /
                                          sizeof(int32_t),
                                      CMPMODE::NE);
}

// 场景65：LE-int32
TEST_F(TestApiCompareUT, Compare_Le_input_tensor_int32_output_uint8) {
  TensorCompareTest<uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t), CMPMODE::LE);
  TensorCompareTest<uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::LE);
  TensorCompareTest<uint8_t, int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::LE);
  TensorCompareTest<uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LE);
  TensorCompareTest<uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::LE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::LE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::LE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::LE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::LE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::LE);
  TensorCompareTest<uint8_t, int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::LE);
  TensorCompareTest<uint8_t, int32_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                       (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(int32_t))) /
                                          sizeof(int32_t),
                                      CMPMODE::LE);
}

// 场景66：GE-int32
TEST_F(TestApiCompareUT, Compare_Ge_input_tensor_int32_output_uint8) {
  TensorCompareTest<uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t), CMPMODE::GE);
  TensorCompareTest<uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::GE);
  TensorCompareTest<uint8_t, int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::GE);
  TensorCompareTest<uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GE);
  TensorCompareTest<uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::GE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::GE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::GE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::GE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::GE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::GE);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::GE);
  TensorCompareTest<uint8_t, int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::GE);
  TensorCompareTest<uint8_t, int32_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                       (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(int32_t))) /
                                          sizeof(int32_t),
                                      CMPMODE::GE);
}

// 场景67：LT-int32
TEST_F(TestApiCompareUT, Compare_Lt_input_tensor_int32_output_uint8) {
  TensorCompareTest<uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t), CMPMODE::LT);
  TensorCompareTest<uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::LT);
  TensorCompareTest<uint8_t, int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::LT);
  TensorCompareTest<uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LT);
  TensorCompareTest<uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LT);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LT);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t), CMPMODE::LT);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::LT);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t), CMPMODE::LT);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::LT);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::LT);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::LT);
  TensorCompareTest<uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t),
                                      CMPMODE::LT);
  TensorCompareTest<uint8_t, int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), CMPMODE::LT);
  TensorCompareTest<uint8_t, int32_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                       (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(int32_t))) /
                                          sizeof(int32_t),
                                      CMPMODE::LT);
}

/* End: 输入是两个tensor的 compare 测试--------------------------------------- */

}  // namespace af
