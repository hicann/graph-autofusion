
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_SCALAR_MINIMUM_H__
#define __ASCENDC_API_SCALAR_MINIMUM_H__

namespace AscendC {
template <typename T>
inline __aicore__ void Mins(const LocalTensor<T> &dst, const T x, const T y) {
  T res = x < y ? x : y;
  AscendC::Duplicate(dst, res, dst.GetSize());
}
}  // namespace AscendC

#endif  // __ASCENDC_API_SCALAR_MINIMUM_H__
