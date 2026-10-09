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
std::vector<std::unique_ptr<TmpBufDesc>> GetExtraTmpBuf(const AscNode &node) {
  AscNodeOutputs node_outputs = node.outputs;
  AscNodeInputs node_inputs = node.inputs;
  std::vector<std::unique_ptr<TmpBufDesc>> tmp_buf;
  GELOGD("Node %s[%s] need extra buffer size.", node.GetTypePtr(), node.GetNamePtr());
  Expression total_size = Symbol(8192);
  GE_CHECK_NOTNULL_EXEC(node.GetOwnerComputeGraph(), return tmp_buf;);
  auto attr = node.GetOwnerComputeGraph()->GetOrCreateAttrsGroup<AscGraphAttr>();
  GE_CHECK_NOTNULL_EXEC(attr, return tmp_buf;);
  Expression a_axis_size = Symbol(1);
  Expression b_axis_size = Symbol(1);
  for (size_t i = 0; i < node_outputs[0].attr.vectorized_strides.size(); i++) {
    uint64_t vectorized_axis_id = node_outputs[0].attr.vectorized_axis[i];
    Expression size_exp = attr->axis[vectorized_axis_id]->size;
    if (i == node_outputs[0].attr.vectorized_strides.size() - 1) {
      const int32_t align_size = 32 / ge::GetSizeByDataType(node_outputs[0].attr.dtype);
      size_exp = sym::Align(size_exp, align_size);
    }

    if (SymbolicUtils::StaticCheckEq(node_outputs[0].attr.vectorized_strides[i], sym::kSymbolZero) != TriBool::kTrue &&
        SymbolicUtils::StaticCheckEq(node_inputs[0].attr.vectorized_strides[i], sym::kSymbolZero) == TriBool::kTrue) {
      b_axis_size = sym::Mul(b_axis_size, size_exp);
    } else {
      a_axis_size = sym::Mul(a_axis_size, size_exp);
    }
  }
  if (node_inputs[0].attr.dtype == ge::DT_UINT8 || node_inputs[0].attr.dtype == ge::DT_INT8) {
    constexpr uint32_t align_size = 16U;
    constexpr uint32_t half_size = 2U;
    // when input is u8/s8, we need to cast input before calc and cast to output after calc,
    // so the tmp_buffer size need to be the sum of input_size and output_size and 8192
    auto input_aligned_size = sym::Mul(sym::Align(a_axis_size, align_size), Symbol(half_size));
    auto output_aligned_size = sym::Mul(sym::Align(sym::Mul(a_axis_size, b_axis_size), align_size), Symbol(half_size));
    total_size = input_aligned_size + output_aligned_size + total_size;
  } else {
    total_size =
        sym::Mul(sym::Mul(a_axis_size, b_axis_size), Symbol(ge::GetSizeByDataType(node_outputs[0].attr.dtype))) +
        total_size;
  }
  GELOGD("Get temp buffer size: %s", total_size.Str().get());
  TmpBufDesc desc = {total_size, -1};
  std::vector<std::unique_ptr<TmpBufDesc>> tmp_buf_descs;
  tmp_buf_descs.emplace_back(std::make_unique<TmpBufDesc>(desc));
  return tmp_buf_descs;
}

bool NeedExtraTmpBuf(const AscNode &node) {
  constexpr size_t dim_three = 3U;
  AscNodeOutputs node_outputs = node.outputs;
  AscNodeInputs node_inputs = node.inputs;
  if (node_inputs.Size() == 0 || node_inputs[0].attr.vectorized_strides.empty()) {
    return false;
  }
  auto data_type = node_inputs[0].attr.dtype;
  if (SymbolicUtils::StaticCheckEq(node_inputs[0].attr.vectorized_strides.back(), sym::kSymbolZero) == TriBool::kTrue &&
      (data_type == ge::DT_UINT8 || data_type == ge::DT_INT8)) {
    return true;
  }
  if (!(node_outputs[0].attr.vectorized_axis.size() >= dim_three &&
        node_outputs[0].attr.vectorized_axis.size() == node_inputs[0].attr.vectorized_axis.size()) ||
      SymbolicUtils::StaticCheckEq(node_inputs[0].attr.vectorized_strides.back(), sym::kSymbolZero) != TriBool::kTrue) {
    return false;
  }
  bool prev_status = false;
  uint32_t brc_num = 0;
  for (uint32_t i = 0; i < node_outputs[0].attr.repeats.size(); ++i) {
    bool cur_status =
        SymbolicUtils::StaticCheckEq(node_outputs[0].attr.repeats[i], node_inputs[0].attr.repeats[i]) != TriBool::kTrue;
    if (cur_status != prev_status) {
      brc_num = prev_status ? brc_num + 1 : brc_num;
      prev_status = cur_status;
    }
  }
  brc_num = prev_status ? brc_num + 1 : brc_num;
  constexpr uint32_t double_axes = 2U;
  return brc_num == double_axes;
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcBroadCastTmpSize(const AscNode &node) {
  if (NeedExtraTmpBuf(node)) {
    return GetExtraTmpBuf(node);
  } else {
    return CalcDefaultTmpSize(node);
  }
}
}  // namespace ascir
}  // namespace af
