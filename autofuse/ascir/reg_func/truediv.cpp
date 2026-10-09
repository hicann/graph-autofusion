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
std::vector<std::unique_ptr<TmpBufDesc>> CalcTrueDivTmpSize(const AscNode &node) {
  AscNodeInputs node_inputs = node.inputs;
  // 获取输入数据的元素个数
  const auto input_size = GetInputSize(node_inputs);
  // 根据数据类型计算临时buffer大小
  // 对于int32类型：需要临时buffer用于非原地计算的中间结果存储
  auto data_type = node_inputs[0].attr.dtype;
  const auto data_type_size = GetSizeByDataType(data_type);

  if (HasScalarOrUbScalar(node_inputs)) {
    return CalcDefaultTmpSize(node);
  }
  if (data_type == ge::DT_INT32) {
    // int32类型：tmp_buf大小需要 >= 2* size * sizeof(int32_t)
    // 用于原地计算时的数据暂存或中间结果存储
    const Expression total_size = Symbol(data_type_size) * input_size * Symbol(2);
    GELOGD("Node %s[%s] TrueDivExtend int32 tmp buffer size: %s (input_size: %s, type_size: %d)", node.GetTypePtr(),
           node.GetNamePtr(), total_size.Str().get(), input_size.Str().get(), data_type_size);
    return GetTmpBuffer(total_size);
  } else {
    // 其他数据类型：返回默认大小
    return CalcDefaultTmpSize(node);
  }
}
}  // namespace ascir
}  // namespace af
