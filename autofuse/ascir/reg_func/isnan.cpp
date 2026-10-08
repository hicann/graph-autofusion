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
#include "default_reg_func.h"
#include "common/checker.h"

namespace af {
namespace ascir {

constexpr int32_t ONE_BLK_SIZE = 32;
constexpr int32_t ONE_REPEAT_BYTE_SIZE = 256;
constexpr int32_t MAX_REPEAT_NUM = 255;
constexpr int32_t BASIC_TMP_SIZE = 8192;

// isnan tmp buf has two part
// 1.sign_mask = ONE_BLK_SIZE
// 2.half_isnan
// if input_size > MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE
//   half_isnan = MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE
// else
//   half_isnan = input_size
std::vector<std::unique_ptr<TmpBufDesc>> CalcIsnanTmpSize(const AscNode &node) {
  AscNodeInputs node_inputs = node.inputs;
  auto type_size = Expression(Symbol(GetSizeByDataType(node_inputs[0].attr.dtype)));

  Expression input_size = GetInputSize(node_inputs);

  Expression min_temp_size = sym::Max(type_size * input_size + Symbol(ONE_BLK_SIZE), Symbol(BASIC_TMP_SIZE));
  min_temp_size = sym::Min(min_temp_size, Symbol(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE));

  TmpBufDesc desc = {min_temp_size, -1};
  std::vector<std::unique_ptr<TmpBufDesc>> tmp_buf_descs;
  tmp_buf_descs.emplace_back(std::make_unique<TmpBufDesc>(desc));
  return tmp_buf_descs;
}
}  // namespace ascir
}  // namespace af
