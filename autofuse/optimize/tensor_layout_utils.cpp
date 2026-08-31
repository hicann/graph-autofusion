/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensor_layout_utils.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "ascir_ops_utils.h"
#include "ascgen_log.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "schedule_utils.h"

namespace optimize {
af::Status TensorLayoutUtils::AnalyzeLoadDiscontinuity(const af::AscTensorAttr &attr, DiscontinuityInfo &info) {
  const auto &axis = attr.axis;
  const auto &repeats = attr.repeats;
  const auto &strides = attr.strides;
  const auto &vectorized_axis = attr.vectorized_axis;

  int32_t total_discontinuous_cnt = 0;
  af::Expression expected_stride = af::sym::kSymbolOne;
  bool tail_checked = false;
  for (auto axis_it = vectorized_axis.rbegin(); axis_it != vectorized_axis.rend(); ++axis_it) {
    auto axis_iter = std::find(axis.begin(), axis.end(), *axis_it);
    GE_ASSERT(axis_iter != axis.end(), "Failed to find vectorized_axis id [%ld] from axis.", *axis_it);

    const size_t idx = std::distance(axis.begin(), axis_iter);
    GE_ASSERT_TRUE(idx < strides.size() && idx < repeats.size());
    const auto &stride = strides[idx];
    const auto &repeat = repeats[idx];

    bool is_stride_zero = ascgen_utils::ExpressEq(stride, af::sym::kSymbolZero);
    if (!tail_checked && !is_stride_zero) {
      tail_checked = true;
      // 有效尾轴不连续
      if (!ascgen_utils::ExpressEq(stride, af::sym::kSymbolOne)) {
        info.is_tail_axis_discontinuous = true;
      }
    }

    bool is_repeat_one = ascgen_utils::ExpressEq(repeat, af::sym::kSymbolOne);
    if (is_repeat_one || is_stride_zero) {
      continue;
    }
    // 不连续点
    if (!ascgen_utils::ExpressEq(stride, expected_stride)) {
      ++total_discontinuous_cnt;
      expected_stride = stride;
    }

    expected_stride = expected_stride * repeat;
  }

  if (total_discontinuous_cnt > 1) {
    info.has_multiple_discontinuities = true;
  }

  return af::SUCCESS;
}
}  // namespace optimize
