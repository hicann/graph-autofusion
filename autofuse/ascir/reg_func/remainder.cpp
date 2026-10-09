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

constexpr uint32_t REMAINDER_TMP_BUF_FACTOR = 3U;  // Need 3 temp buffers for div_res, floor_res, mul_res

/**
 * Calculate temp buffer size for Remainder operation
 * Remainder needs 3 intermediate buffers:
 * 1. div_res = dividend / divisor
 * 2. floor_res = Cast(div_res, CAST_FLOOR)
 * 3. mul_res = floor_res * divisor
 * Then: dst = dividend - mul_res
 *
 * Supported data types: float, int32
 * For int32, computation is performed in float (same size), so temp buffer size is the same.
 */
std::vector<std::unique_ptr<TmpBufDesc>> CalcRemainderTmpSize(const AscNode &node) {
  auto nodeInputs = node.inputs;

  // Check if second input is scalar or ub_scalar
  if (nodeInputs[1].attr.repeats.empty()) {
    GELOGD("Node %s[%s] input[1] repeats is empty, using default temp size", node.GetTypePtr(), node.GetNamePtr());
    return CalcDefaultTmpSize(node);
  }

  // Check if second input is ub_scalar (all repeats are 1)
  bool isUbScalar = true;
  for (uint32_t i = 0; i < nodeInputs[1].attr.repeats.size(); i++) {
    if (SymbolicUtils::StaticCheckEq(nodeInputs[1].attr.repeats[i], sym::kSymbolOne) != TriBool::kTrue) {
      isUbScalar = false;
      break;
    }
  }

  if (isUbScalar) {
    GELOGD("Node %s[%s] input[1] is ub scalar, using default temp size", node.GetTypePtr(), node.GetNamePtr());
    return CalcDefaultTmpSize(node);
  }

  // For tensor input, need 3 * input_size temp buffer
  GELOGD("Node %s[%s] input[1] is tensor, calculating temp size", node.GetTypePtr(), node.GetNamePtr());

  // Get input size
  Expression inputSize = GetInputSize(nodeInputs);
  uint32_t inputId = GetNonScalarAxisId(nodeInputs);
  if (inputId == UINT32_MAX) {
    inputId = 0;
  }

  uint32_t dataTypeSize = GetSizeByDataType(nodeInputs[inputId].attr.dtype);
  GELOGD("Node %s[%s] inputs[%u] data type size is: %d", node.GetTypePtr(), node.GetNamePtr(), inputId, dataTypeSize);

  // Total temp size = 3 * input_size * sizeof(T)
  // Need to store div_res, floor_res, mul_res
  // For both float and int32, sizeof(T) = 4, so same calculation applies
  Expression totalSize = Symbol(REMAINDER_TMP_BUF_FACTOR * dataTypeSize) * inputSize;

  return GetTmpBuffer(totalSize);
}

}  // namespace ascir
}  // namespace af
