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

std::vector<std::unique_ptr<TmpBufDesc>> CalcWelfordUpdateTmpSize(const AscNode &node) {
  AscNodeInputs node_inputs = node.inputs;

  // WelfordUpdate接口的输入shape为[rn_len, ab_len]
  // 获取输入tensor的shape信息
  uint32_t input_id = GetNonScalarAxisId(node_inputs);
  const auto &attr = node_inputs[input_id].attr;

  // 与asc-devkit GetWelfordUpdateMaxMinTmpSize的float场景maxValue对齐:
  //   minValue = 2 * WEL_UP_REP_SIZE = 512
  //   maxValue = ceil(rn_len * ab_len / WEL_UP_FLOAT_SIZE) * minValue
  constexpr uint32_t WEL_UP_REP_SIZE = 256;                                // 单次repeat字节数
  constexpr uint32_t WEL_UP_FLOAT_SIZE = WEL_UP_REP_SIZE / sizeof(float);  // 64
  constexpr uint32_t WEL_UP_MIN_VALUE = 2 * WEL_UP_REP_SIZE;               // 512

  // 使用Expression进行符号计算，支持动态shape
  const Expression rn_len = attr.repeats[0];
  const Expression ab_len = attr.repeats.size() >= 2 ? attr.repeats[1] : Symbol(1);

  // maxValue = ceil(rn_len * ab_len / 64) * 512
  const Expression total_elements = sym::Mul(rn_len, ab_len);
  const Expression blocks =
      sym::Div(sym::Add(total_elements, Symbol(WEL_UP_FLOAT_SIZE - 1)), Symbol(WEL_UP_FLOAT_SIZE));
  const Expression total_size = sym::Mul(blocks, Symbol(WEL_UP_MIN_VALUE));

  GELOGD("Node %s[%s] WelfordUpdate tmp buffer size: %s (rn_len: %s, ab_len: %s)", node.GetTypePtr(), node.GetNamePtr(),
         total_size.Str().get(), rn_len.Str().get(), ab_len.Str().get());

  return GetTmpBuffer(total_size);
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcWelfordFinalizeTmpSize(const AscNode &node) {
  AscNodeInputs node_inputs = node.inputs;

  // WelfordFinalize接口的输入shape为[ab_len]
  // 获取输入tensor的shape信息
  uint32_t input_id = GetNonScalarAxisId(node_inputs);
  if (input_id == UINT32_MAX) {
    input_id = 0;  // 如果没有非scalar输入，默认使用第一个输入
  }

  const auto &attr = node_inputs[input_id].attr;

  // 与asc-devkit GetWelfordFinalizeMaxMinTmpSize的float场景maxValue对齐:
  //   minValue = BASICBLOCK_UNIT * 4 * sizeof(float) = 1024
  //   ab_len <= 64: maxValue = minValue
  //   ab_len > 64:  maxValue = (BASICBLOCK_UNIT*2 + ab_len*2) * sizeof(float) = 512 + ab_len * 8
  // 统一表达为: maxValue = max(minValue, 512 + ab_len * 8)
  constexpr uint32_t BASICBLOCK_UNIT = 64;
  constexpr uint32_t MIN_VALUE = BASICBLOCK_UNIT * 4 * sizeof(float);  // 1024

  // 使用Expression进行符号计算，支持动态shape
  const Expression ab_len = attr.repeats[0];

  // (BASICBLOCK_UNIT * 2 + ab_len * 2) * FLOAT_SIZE = 512 + ab_len * 8
  const Expression var_size =
      sym::Add(Symbol(BASICBLOCK_UNIT * 2 * sizeof(float)), sym::Mul(ab_len, Symbol(2 * sizeof(float))));
  // 取max确保ab_len较小时不低于MIN_VALUE
  const Expression total_size = sym::Max(Symbol(MIN_VALUE), var_size);

  GELOGD("Node %s[%s] WelfordFinalize tmp buffer size: %s (ab_len: %s)", node.GetTypePtr(), node.GetNamePtr(),
         total_size.Str().get(), ab_len.Str().get());

  return GetTmpBuffer(total_size);
}

}  // namespace ascir
}  // namespace af
