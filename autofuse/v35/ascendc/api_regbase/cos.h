/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_COS_H__
#define __ASCENDC_API_REGBASE_COS_H__

constexpr uint32_t COS_THREAD_NUM = 1024;

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(COS_THREAD_NUM) inline void CosSimtCompute(__ubuf__ T *x, __ubuf__ T *y,
                                                                               const int64_t total_num) {
  for (int64_t i = threadIdx.x; i < total_num; i += blockDim.x) {
    y[i] = Simt::Cos(x[i]);
  }
}

template <typename T>
__aicore__ inline void CosExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                 const LocalTensor<uint8_t> &tmp_buf, const uint32_t calc_cnt) {
  AscendC::Simt::VF_CALL<CosSimtCompute<T>>(AscendC::Simt::Dim3(COS_THREAD_NUM), (__ubuf__ T *)src.GetPhyAddr(),
                                            (__ubuf__ T *)dst.GetPhyAddr(), calc_cnt);
}
#endif  // __ASCENDC_API_REGBASE_COS_H__
