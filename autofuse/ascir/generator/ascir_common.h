/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_COMMON_H_
#define ASCIR_COMMON_H_

#include <vector>
#include <map>
#include "ascendc_ir.h"

namespace af {
namespace ascir {
struct BroadcastCapability {
  bool support_inline_brc;            // 是否支持内联广播
  std::vector<size_t> scalar_inputs;  // 支持ub_scalar输入的索引列表
};
bool OnlySecondInputSupportScalar(const std::vector<bool> &is_scalar_list);
[[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionFromDtypeMap(
    const AscNode &node, const std::map<ge::DataType, ge::DataType> &dtype_conversion_map);
bool IsAllVecAxisContinuous(const AscNode &node);
Status ValidateShapeConsistencyWithOutput(const AscNode &node, size_t output_id,
                                          const BroadcastCapability &broadcast_capability = {false, {}});
Status ValidateShapeConsistencyWithSingleOutput(const AscNode &node,
                                                const BroadcastCapability &broadcast_capability = {false, {}});
bool IsNodeHasScalarInput(const AscNode &node);
bool IsNodeFirstInputScalar(const AscNode &node);
bool IsAnyInputNodeScalar(const AscNode &node);
}  // namespace ascir
}  // namespace af

#endif  // ASCIR_COMMON_H_
