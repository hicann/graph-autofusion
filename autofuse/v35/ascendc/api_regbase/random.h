/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_RANDOM_H__
#define __ASCENDC_API_REGBASE_RANDOM_H__

/**
 * @brief Rand - unified random number generation using AscendC PhiloxRandom API
 * Uses fixed philoxKey = {0, 0} and philoxCounter = {0, 0, 0, 0} for continuous mode
 * Supports float, float16, uint32_t, int32_t data types
 *
 * @tparam T data type, supports float, float16, uint32_t, int32_t
 * @param dst output tensor
 * @param size number of elements to generate
 */
template <typename T>
inline __aicore__ void Rand(const AscendC::LocalTensor<T> &dst, const uint32_t size) {
  // PhiloxKey and PhiloxCounter are plain C arrays: uint32_t[2] and uint32_t[4]
  AscendC::PhiloxKey philoxKey = {0, 0};
  AscendC::PhiloxCounter philoxCounter = {0, 0, 0, 0};

  AscendC::PhiloxRandom<10>(dst, philoxKey, philoxCounter, size);
}

#endif  // __ASCENDC_API_REGBASE_RANDOM_H__
