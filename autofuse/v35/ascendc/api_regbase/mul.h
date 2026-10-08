
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_REGBASE_API_MUL_H__
#define __ASCENDC_REGBASE_API_MUL_H__

template <typename T>
inline __aicore__ void MulExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src1, const LocalTensor<T> &src2,
                                 const uint32_t calc_cnt) {
  AscendC::Mul(dst, src1, src2, calc_cnt);
}

template <typename T, bool IS_SCALAR_LATTER = true>
inline __aicore__ void MulExtends(const LocalTensor<T> &dst, const LocalTensor<T> &src, const T constant_x,
                                  const uint32_t calc_cnt) {
  if constexpr (IS_SCALAR_LATTER) {
    AscendC::Muls(dst, src, constant_x, calc_cnt);
  } else {
    AscendC::Muls(dst, constant_x, src, calc_cnt);
  }
}

template <typename T>
inline __aicore__ void MulExtends(const LocalTensor<T> &dst, const T x, const T y) {
  T res = x * y;
  AscendC::Duplicate(dst, res, dst.GetSize());
}

inline __aicore__ void MulExtend(const LocalTensor<bool> &dst, const LocalTensor<bool> &src1,
                                 const LocalTensor<bool> &src2, const uint32_t calc_cnt) {
  AscendC::LocalTensor<uint8_t> src1_uint8 = src1.template ReinterpretCast<uint8_t>();
  AscendC::LocalTensor<uint8_t> src2_uint8 = src2.template ReinterpretCast<uint8_t>();
  AscendC::LocalTensor<uint8_t> dst_uint8 = dst.template ReinterpretCast<uint8_t>();
  AscendC::And(dst_uint8, src1_uint8, src2_uint8, calc_cnt);
}

#endif  // __ASCENDC_REGBASE_API_MUL_H__
