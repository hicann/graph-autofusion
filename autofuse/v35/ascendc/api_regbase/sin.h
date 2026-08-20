/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_SIN_H__
#define __ASCENDC_API_REGBASE_SIN_H__

static constexpr AscendC::SinAlgo sin_algo = AscendC::SinAlgo::RADIAN_REDUCTION;
static constexpr AscendC::SinConfig sin_config = {sin_algo};

template <typename T>
inline __aicore__ void SinExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                 AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t calCount) {
  Sin<T, false, sin_config>(dst, src, tmp_buf, calCount);
}
#endif  // __ASCENDC_API_REGBASE_SIN_H__
