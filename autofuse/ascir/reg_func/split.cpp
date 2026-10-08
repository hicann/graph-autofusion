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
namespace {
constexpr int32_t TWO = 2;
constexpr int32_t FOUR = 4;

constexpr int32_t BASIC_TMP_SIZE = 16384;
constexpr int32_t MAX_TMP_SIZE = 65536;
constexpr int32_t MAX_TMP_SIZE_FOR_SMALL_TAIL = 96 * 1024;

constexpr int32_t TYPESIZEEQ8 = 8;
constexpr int32_t TYPESIZEEQ4 = 4;
constexpr int32_t TYPESIZEEQ2 = 2;
constexpr int32_t TYPESIZEEQ1 = 1;

constexpr int32_t ALIGNSIZE8 = 8;
constexpr int32_t ALIGNSIZE16 = 16;
constexpr int32_t ALIGNSIZE32 = 32;

constexpr int32_t ALIGNPAD_8 = 29;
constexpr int32_t ALIGNPAD_4 = 29;
constexpr int32_t ALIGNPAD_2 = 45;
constexpr int32_t ALIGNPAD_1 = 93;

constexpr int32_t TMPSIZEOF8_4 = 128;
constexpr int32_t TMPSIZEOF2 = 64;
constexpr int32_t TMPSIZEOF1 = 48;

bool IsAllStaticAligned(AscNodeOutputs &node_outputs, uint32_t split_dim, int32_t align_size) {
  for (uint32_t i = 0; i < node_outputs().size(); ++i) {
    auto axis = node_outputs[i].attr.repeats[split_dim];
    for (uint32_t j = split_dim + 1; j < node_outputs[i].attr.repeats.size(); ++j) {
      axis = sym::Mul(axis, node_outputs[i].attr.repeats[j]);
    }

    if (SymbolicUtils::StaticCheckEq(sym::Mod(axis, Symbol(align_size)), sym::kSymbolZero) != TriBool::kTrue) {
      GELOGD("The product of dims after split_dim is %s, not aligned.",
             SymbolicUtils::ToString(sym::Mod(axis, Symbol(align_size))).c_str());
      return false;
    }
  }
  return true;
}
}  // namespace
std::vector<std::unique_ptr<TmpBufDesc>> CalcSplitTmpSizeV2(const AscNode &node) {
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;

  GE_ASSERT_TRUE(node_inputs.Size() > 0);
  uint32_t split_dim = 0;
  for (uint32_t i = 0; i < node_inputs[0].attr.repeats.size(); ++i) {
    if (node_outputs[0].attr.repeats[i] != node_inputs[0].attr.repeats[i]) {
      split_dim = i;
      break;
    }
  }
  auto type_size = GetSizeByDataType(node_inputs[0].attr.dtype);
  GE_ASSERT_TRUE(type_size > 0, "%s Invalid node inputs dtype: %d", node.GetNamePtr(),
                 static_cast<int32_t>(node_inputs[0].attr.dtype));
  Expression min_tmp_buf_size = Symbol(0);
  bool is_aligned = IsAllStaticAligned(node_outputs, split_dim, ALIGNSIZE32 / type_size);
  if (is_aligned) {
    GELOGD("%s is all aligned", node.GetNamePtr());
    return {};
  }
  constexpr int64_t kTmpBufSizeForSplitByScatter = 1024L;
  return GetTmpBuffer(Symbol(kTmpBufSizeForSplitByScatter));
}
}  // namespace ascir
}  // namespace af
