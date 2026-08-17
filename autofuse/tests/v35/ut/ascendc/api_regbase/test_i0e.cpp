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
#include "modified_bessel_utils.h"
#include "modified_bessel_i0.h"
#include "i0e.h"

using namespace AscendC;

namespace af {

class TestRegbaseApiI0eUT : public testing::Test {
 protected:
  template <typename T>
  static void InvokeTensorTensorKernel(UnaryInputParam<T> &param) {
    TPipe tp;
    TBuf<TPosition::VECCALC> src_buf, dst_buf, work_buf;
    tp.InitBuffer(src_buf, sizeof(T) * param.size);
    tp.InitBuffer(dst_buf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
    tp.InitBuffer(work_buf, TMP_UB_SIZE);

    LocalTensor<T> local_x = src_buf.Get<T>();
    LocalTensor<T> local_y = dst_buf.Get<T>();
    LocalTensor<uint8_t> local_work = work_buf.Get<uint8_t>();

    GmToUb(local_x, param.x1, param.size);
    I0eExtend(local_y, local_x, local_work, param.size);
    UbToGm(param.y, local_y, param.size);
  }

  template <typename T>
  static void CreateTensorInput(UnaryInputParam<T> &param) {
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    std::mt19937 rand_eng(1);

    for (uint32_t idx = 0; idx < param.size; idx++) {
      std::uniform_real_distribution dist(-20.0f, 20.0f);
      param.x1[idx] = static_cast<T>(dist(rand_eng));
      double abs_val = static_cast<double>(std::fabs(param.x1[idx]));
      param.exp[idx] = static_cast<T>(std::exp(-abs_val) * std::cyl_bessel_i(0, abs_val));
    }
  }

  template <typename T>
  static uint32_t Valid(UnaryInputParam<T> &param) {
    uint32_t err_count = 0;
    for (uint32_t idx = 0; idx < param.size; idx++) {
      double out_val = static_cast<double>(param.y[idx]);
      double ref_val = static_cast<double>(param.exp[idx]);
      double rel_error = std::abs(out_val - ref_val) / std::max(std::abs(ref_val), 1.0);
      if (rel_error > 1e-5) {
        err_count++;
        printf("diff at index %d: x: %.20e, y: %.20e, expect: %.20e, rel_err: %f\n", idx,
               static_cast<float>(param.x1[idx]), static_cast<float>(param.y[idx]), static_cast<float>(param.exp[idx]),
               static_cast<float>(rel_error));
      }
    }
    return err_count;
  }

  template <typename T>
  static void I0eTest(uint32_t size) {
    UnaryInputParam<T> input_param{};
    input_param.size = size;
    CreateTensorInput(input_param);

    auto run_kernel = [&input_param] { InvokeTensorTensorKernel(input_param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(run_kernel, 1);

    uint32_t err_count = Valid(input_param);
    EXPECT_EQ(err_count, 0);

    AscendC::GmFree(input_param.exp);
    AscendC::GmFree(input_param.y);
    AscendC::GmFree(input_param.x1);
  }
};

TEST_F(TestRegbaseApiI0eUT, I0e_TensorTensor_Test) {
  I0eTest<float>(ONE_BLK_SIZE / sizeof(float));
  I0eTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  I0eTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  I0eTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  I0eTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  I0eTest<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  I0eTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                  (ONE_BLK_SIZE - sizeof(float))) /
                 2 / sizeof(float));
}

}  // namespace af
