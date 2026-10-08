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
std::vector<std::unique_ptr<TmpBufDesc>> CalcLogicalNotTmpSize(const AscNode &node) {
  auto node_inputs = node.inputs;
  GE_ASSERT_TRUE(node_inputs.Size() > 0, "Node %s[%s] inputs size is 0.", node.GetTypePtr(), node.GetNamePtr());
  auto input_size = GetInputSize(node_inputs);
  GELOGD("Node %s[%s] inputs[0] data type size is: %d", node.GetTypePtr(), node.GetNamePtr(),
         GetSizeByDataType(node_inputs[0].attr.dtype));
  constexpr uint32_t duplciate_size = 32U;
  Expression total_size = Symbol(2) * input_size + Symbol(duplciate_size);
  return GetTmpBuffer(total_size);
}
}  // namespace ascir
}  // namespace af
