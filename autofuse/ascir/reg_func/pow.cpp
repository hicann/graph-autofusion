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
#include "graph/symbolizer/symbolic_utils.h"
namespace af {
namespace ascir {
std::vector<std::unique_ptr<TmpBufDesc>> CalcPowTmpSize(const AscNode &node) {
  auto node_inputs = node.inputs;
  GE_ASSERT_TRUE(node_inputs.Size() > 0U, "Node %s[%s] inputs size is 0.", node.GetTypePtr(), node.GetNamePtr());
  const auto input_size = GetInputSize(node_inputs);
  uint32_t input_id = GetNonScalarAxisId(node_inputs);
  if (input_id == UINT32_MAX) {
    input_id = node_inputs.Size() - 1U;
  }
  const auto data_type_size = GetSizeByDataType(node_inputs[input_id].attr.dtype);
  GELOGD("Node %s[%s] inputs[%u] data type size is: %d", node.GetTypePtr(), node.GetNamePtr(), input_id,
         data_type_size);
  if (IsAllScalarOrUbScalar(node_inputs)) {
    // 当两个输入都为scalar，或都为ub scalar，或一个scalar，一个ub scalar时，需要额外分配2个blockSize的tmp buffer
    const Expression total_size = Symbol(data_type_size) * input_size + Symbol(32 * 2);
    GELOGD("Node %s[%s] inputs are all scalar or ub scalar, return size= %s.", node.GetTypePtr(), node.GetNamePtr(),
           total_size.Str().get());
    return GetTmpBuffer(total_size);
  } else {
    const Expression total_size = Symbol(data_type_size) * input_size;
    GELOGD("Node %s[%s] inputs are not all scalar or ub scalar, return size= %s.", node.GetTypePtr(), node.GetNamePtr(),
           total_size.Str().get());
    return GetTmpBuffer(total_size);
  }
}
}  // namespace ascir
}  // namespace af
