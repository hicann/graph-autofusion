/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "default_reg_func.h"

namespace af {
namespace ascir {
constexpr uint32_t SIGMOID_TEMP_BUFFER_SIZE = 256;
std::vector<std::unique_ptr<TmpBufDesc>> CalcSigmoidTmpSize([[maybe_unused]] const AscNode &node) {
  return GetTmpBuffer(Symbol(SIGMOID_TEMP_BUFFER_SIZE));
}
}  // namespace ascir
}  // namespace af
