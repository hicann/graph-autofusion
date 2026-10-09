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
const Expression kPadOneBlkSize = Symbol(32);
const Expression kPadMaxRepeatTimes = Symbol(255);
const Expression kPadNCHWConvAddrListSize = Symbol(16);
const Expression kTmpBufferNum = Symbol(2);
const Expression kOne = Symbol(1);

std::vector<std::unique_ptr<TmpBufDesc>> CalcPadTmpSize(const AscNode &node) {
  Expression min_value;
  auto inputs = node.inputs;
  int32_t data_size_int = GetSizeByDataType(inputs[0].attr.dtype);
  GE_ASSERT_TRUE(data_size_int != 0, "Data type size zero");
  Expression data_size = Symbol(data_size_int);
  min_value = kPadNCHWConvAddrListSize * (kPadOneBlkSize / data_size) * kTmpBufferNum * data_size;

  return GetTmpBuffer(min_value);
}
}  // namespace ascir
}  // namespace af
