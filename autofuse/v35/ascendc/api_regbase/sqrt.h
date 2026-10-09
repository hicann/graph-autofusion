/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_SQRT_H__
#define __ASCENDC_API_REGBASE_SQRT_H__

/**
 * @brief SqrtExtend - compute the square root: y = sqrt(x)
 *
 * 对齐单算子 arch35 sqrt（Vec::Sqrt0ULP）实现：float 输入走
 * PRECISION_0ULP_FTZ_FALSE 精确开方，其余类型走默认 Sqrt。
 *
 * @tparam T data type, supports float, half, and bfloat16_t
 * @param dst output tensor
 * @param src input tensor
 * @param calCount number of elements to compute
 */
template <typename T>
__aicore__ inline void SqrtExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                  const uint32_t calCount) {
  static_assert(AscendC::SupportType<T, half, float, bfloat16_t>(),
                "SqrtExtend only supports half, float and bfloat16_t on current device!");

  if constexpr (std::is_same_v<T, float>) {
    static constexpr AscendC::SqrtConfig config = {AscendC::SqrtAlgo::PRECISION_0ULP_FTZ_FALSE};
    AscendC::Sqrt<T, config>(dst, src, calCount);
  } else {
    AscendC::Sqrt(dst, src, calCount);
  }
}

#endif  // __ASCENDC_API_REGBASE_SQRT_H__
