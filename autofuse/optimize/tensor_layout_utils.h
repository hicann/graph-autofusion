/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_TENSOR_LAYOUT_UTILS_H_
#define OPTIMIZE_TENSOR_LAYOUT_UTILS_H_

#include "ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"

namespace optimize {
struct DiscontinuityInfo {
  bool is_tail_axis_discontinuous{false};    //  最后一根有效轴不连续
  bool has_multiple_discontinuities{false};  // 多根轴不连续场景
};
class TensorLayoutUtils {
 public:
  static af::Status AnalyzeLoadDiscontinuity(const af::AscTensorAttr &attr, DiscontinuityInfo &info);
};
}  // namespace optimize

#endif  // OPTIMIZE_TENSOR_LAYOUT_UTILS_H_
