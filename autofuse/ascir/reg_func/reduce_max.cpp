/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_ir.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "default_reg_func.h"

namespace af {
namespace ascir {

// Constants for int32_t ReduceMax
constexpr int32_t kInt32Bytes = 4;
constexpr int32_t kInt32ElemPerBlk = 8;        // ONE_BLK_SIZE / INT32_BYTES = 32 / 4 = 8
constexpr int32_t kBrcbTmpSize = 288;          // 32 bytes prep + 256 bytes Brcb output
constexpr int32_t kArFinalReduceTmpSize = 32;  // Extra 8 int32_t scratch used by final AR reduction

static AscGraphAttr *GetOrCreateGraphAttrsGroup(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr;);
  auto attr = graph->GetOrCreateAttrsGroup<AscGraphAttr>();
  GE_CHECK_NOTNULL_EXEC(attr, return nullptr;);
  return attr;
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcReduceMaxTmpSize(const AscNode &node) {
  std::vector<std::unique_ptr<TmpBufDesc>> tmp_buf_desc;
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;

  if (node_inputs.Size() <= 0) {
    return tmp_buf_desc;
  }

  // Only handle int32 data type
  auto data_type = node_inputs[0].attr.dtype;
  if (data_type != ge::DT_INT32) {
    return CalcDefaultTmpSize(node);
  }

  if (node_outputs[0].attr.vectorized_strides.size() <= 0) {
    return tmp_buf_desc;
  }

  bool isAr =
      SymbolicUtils::StaticCheckEq(node_outputs[0].attr.vectorized_strides.back(), sym::kSymbolZero) == TriBool::kTrue;

  auto attr = GetOrCreateGraphAttrsGroup(node.GetOwnerComputeGraph());

  Expression first_exp = attr->axis[0]->size;
  Expression last_exp = attr->axis[1]->size;

  if (isAr) {
    // AR mode: Tmp buffer = first * 32 + 288 + 32 bytes
    Expression tmp_size = sym::Add(sym::Mul(first_exp, Symbol(kInt32ElemPerBlk * kInt32Bytes)),
                                   Symbol(kBrcbTmpSize + kArFinalReduceTmpSize));

    GELOGD("Node %s[%s] ReduceMax AR mode int32 tmp buffer size: first * 32 + 320", node.GetTypePtr(),
           node.GetNamePtr());

    TmpBufDesc desc = {tmp_size, -1};
    tmp_buf_desc.emplace_back(std::make_unique<TmpBufDesc>(desc));
  } else {
    // RA mode: Tmp buffer = input_size * 4 + 288 bytes
    const auto input_size = GetInputSize(node_inputs);

    Expression tmp_size = sym::Add(sym::Mul(input_size, Symbol(kInt32Bytes)), Symbol(kBrcbTmpSize));

    GELOGD("Node %s[%s] ReduceMax RA mode int32 tmp buffer size: input_size * 4 + 288", node.GetTypePtr(),
           node.GetNamePtr());

    TmpBufDesc desc = {tmp_size, -1};
    tmp_buf_desc.emplace_back(std::make_unique<TmpBufDesc>(desc));
  }

  return tmp_buf_desc;
}

}  // namespace ascir
}  // namespace af
