/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Use the same include guard as adv_api/detail/common/common.h to avoid duplicate CastTrait definitions when UT or ST
// includes both headers.
#ifndef IMPL_COMMON_COMMON_H
#define IMPL_COMMON_COMMON_H

#include "kernel_basic_intf.h"

namespace AscendC {
namespace Internal {

constexpr Reg::CastTrait castTraitB16ToB32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                              RoundMode::UNKNOWN};
constexpr Reg::CastTrait castTraitB32ToB16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                              RoundMode::CAST_RINT};

}  // namespace Internal
}  // namespace AscendC

#endif  // IMPL_COMMON_COMMON_H
