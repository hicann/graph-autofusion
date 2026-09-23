/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_FMOD_H__
#define __ASCENDC_API_REGBASE_FMOD_H__

constexpr uint32_t FMOD_THREAD_NUM = 1024;

// fmod 为截断语义余数，fmodf 为位分解精确算法（大商场景仍精确），对齐单算子参考实现
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(FMOD_THREAD_NUM) inline void FmodSimtCompute(__ubuf__ T *dst, __ubuf__ T *src1,
                                                                                 __ubuf__ T *src2,
                                                                                 const int64_t total_num) {
  for (int64_t i = threadIdx.x; i < total_num; i += blockDim.x) {
    dst[i] = fmodf(src1[i], src2[i]);
  }
}

template <typename T>
__aicore__ inline void FmodExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src1,
                                  const AscendC::LocalTensor<T> &src2, const LocalTensor<uint8_t> &tmp_buf,
                                  const uint32_t size) {
  AscendC::Simt::VF_CALL<FmodSimtCompute<T>>(AscendC::Simt::Dim3(FMOD_THREAD_NUM), (__ubuf__ T *)dst.GetPhyAddr(),
                                             (__ubuf__ T *)src1.GetPhyAddr(), (__ubuf__ T *)src2.GetPhyAddr(), size);
}

#endif  // __ASCENDC_API_REGBASE_FMOD_H__
