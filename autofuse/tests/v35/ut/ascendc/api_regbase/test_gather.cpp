/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/gather.h"

using namespace AscendC;

namespace {
constexpr uint32_t kBufferSize = 32;
constexpr uint32_t kDstOffset = 1;
constexpr uint64_t kCopyLength = 1;
constexpr uint8_t kSentinel = 0xA5;
constexpr uint8_t kSourceValue = 0x3C;
}  // namespace

TEST(TestApiGather, DataCopySimdSimtSmallLengthWithinPadding) {
  auto *src = static_cast<uint8_t *>(AscendC::GmAlloc(kBufferSize));
  auto *dst = static_cast<uint8_t *>(AscendC::GmAlloc(kBufferSize));
  for (uint32_t i = 0; i < kBufferSize; ++i) {
    src[i] = kSourceValue + i;
    dst[i] = kSentinel;
  }

  auto kernel = [](uint8_t *src, uint8_t *dst) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> dst_buf;
    pipe.InitBuffer(dst_buf, kBufferSize);
    LocalTensor<uint8_t> local_dst = dst_buf.Get<uint8_t>();
    GmToUb(local_dst, dst, kBufferSize);

    GlobalTensor<uint8_t> global_src;
    global_src.SetGlobalBuffer((__gm__ uint8_t *)src);
    DataCopySimdSimt(local_dst, global_src, kDstOffset, 0, kCopyLength, false);
    UbToGm(dst, local_dst, kBufferSize);
  };

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, src, dst);

  EXPECT_EQ(dst[kDstOffset], kSourceValue);
  for (uint32_t i = 0; i < kBufferSize; ++i) {
    if (i != kDstOffset) {
      EXPECT_EQ(dst[i], kSentinel) << "unexpected write at index " << i;
    }
  }

  AscendC::GmFree(src);
  AscendC::GmFree(dst);
}
