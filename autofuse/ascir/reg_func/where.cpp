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

std::vector<std::unique_ptr<TmpBufDesc>> CalcWhereTmpSize(const AscNode &node) {
  auto inputs = node.inputs;
  int32_t data_size = 0;
  for (uint32_t i = 0U; i < inputs.Size(); ++i) {
    GELOGD("Node %s[%s] input[%u] data type size is: %d", node.GetTypePtr(), node.GetNamePtr(), i,
           GetSizeByDataType(inputs[i].attr.dtype));
    data_size += GetSizeByDataType(inputs[i].attr.dtype);
  }
  Expression total_size = GetInputSize(inputs) * Symbol(data_size);
  return GetTmpBuffer(total_size);
}
std::vector<std::unique_ptr<TmpBufDesc>> CalcSelectTmpSize(const AscNode &node) {
  return CalcWhereTmpSize(node);
}
}  // namespace ascir
}  // namespace af
