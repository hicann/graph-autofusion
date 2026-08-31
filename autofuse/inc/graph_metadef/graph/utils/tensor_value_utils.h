/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_UTILS_TENSOR_VALUE_UTILS_H_
#define INC_GRAPH_UTILS_TENSOR_VALUE_UTILS_H_

#include <vector>

#include "graph/attr_value_serializable.h"
#include "graph/def_types.h"
#include "graph/ge_error_codes_af.h"
#include "graph/ge_tensor.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/type_utils.h"
#include "common/checker.h"

namespace af {
static constexpr int32_t kAttrTensorShowNum = 6;
static constexpr int32_t kAttrTensorShowNumHalf = static_cast<int32_t>(kAttrTensorShowNum / 2);
// FP16 常量定义
static constexpr uint16_t kFp16ManMask = 0x03FFU;
static constexpr int16_t kFp16ExpBias = 15;
static constexpr int16_t kFp32ExpBias = 127;
static constexpr uint16_t kFp16ManHideBit = 0x0400U;
static constexpr int16_t kFp16FractionMove = 13;
static constexpr int16_t kFp32FractionMove = 23;
static constexpr int16_t kFp32Fraction = 31;

// Union for type punning to avoid strict aliasing violation
union Fp32Bits {
  uint32_t u;
  float f;
};

class TensorValueUtils {
 public:
  static std::string ConvertTensorValue(const Tensor &tensor, DataType value_type, const std::string &sep = ",",
                                        const bool is_mid_skipped = true);
};
}  // namespace af
#endif  // INC_GRAPH_UTILS_TENSOR_VALUE_UTILS_H_
