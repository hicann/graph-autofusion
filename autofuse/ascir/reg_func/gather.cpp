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
constexpr int32_t MAX_TMP_SIZE = 133 * 1024;  // 最大能存储param(4000),datatype为float, indices(100,100),datatype为int64
constexpr uint32_t BASIC_TMP_SIZE = 80U * 1024U;  // 最小能存储indices(100,100),datatype为int64
constexpr uint32_t INT32_SIZE = 4U;
constexpr uint32_t INT64_SIZE = 8U;
constexpr uint32_t TWO = 2U;

std::vector<std::unique_ptr<TmpBufDesc>> CalcGatherTmpSize(const AscNode &node) {
  std::vector<std::unique_ptr<TmpBufDesc>> tmpBufDescs;
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;
  GE_CHK_BOOL_RET_SPECIAL_STATUS(node_inputs.Size() < TWO, tmpBufDescs, "node.inputs.Size less than TWO");
  auto typeSizeT1 = Expression(Symbol(GetSizeByDataType(node_inputs[0].attr.dtype)));
  // calc param last axis
  Expression paramLastSize = sym::Mul(node_inputs[0].attr.repeats[node_inputs[0].attr.repeats.size() - 1], typeSizeT1);
  // calc indices
  Expression indicesSize = Symbol(0);
  auto typeSizeT2 = Expression(Symbol(GetSizeByDataType(node_inputs[1].attr.dtype)));
  if (SymbolicUtils::StaticCheckEq(typeSizeT2, Symbol(INT32_SIZE)) == TriBool::kTrue) {
    indicesSize = sym::Mul(node_outputs[0].attr.repeats[node_outputs[0].attr.repeats.size() - 1], typeSizeT2);
  } else if (SymbolicUtils::StaticCheckEq(typeSizeT2, Symbol(INT64_SIZE)) == TriBool::kTrue) {
    indicesSize = sym::Mul(node_outputs[0].attr.repeats[node_outputs[0].attr.repeats.size() - 1],
                           (typeSizeT2 + sym::Div(typeSizeT2, Symbol(TWO))));
  }
  Expression TempSize = paramLastSize + indicesSize;
  TempSize = sym::Max(Symbol(BASIC_TMP_SIZE), TempSize);
  TempSize = sym::Min(Symbol(MAX_TMP_SIZE), TempSize);  // BASIC_TMP_SIZE <= TempSize <= MAX_TMP_SIZE
  TmpBufDesc desc = {TempSize, -1};
  tmpBufDescs.emplace_back(std::make_unique<TmpBufDesc>(desc));
  return tmpBufDescs;
}
}  // namespace ascir
}  // namespace af
