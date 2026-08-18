/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HINT_GRAPH_INFO_COMPLETE_H
#define HINT_GRAPH_INFO_COMPLETE_H

#include <vector>
#include <string>

#include "ascendc_ir.h"
#include "ascgen_log.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace optimize {
struct ExpressionComparator {
  bool operator()(const af::Expression &lhs, const af::Expression &rhs) const {
    return af::SymbolicUtils::ToString(lhs) < af::SymbolicUtils::ToString(rhs);
  }
};
using SizeVarSet = std::set<af::Expression, ExpressionComparator>;

class AscGraphInfoComplete {
 public:
  /**
   * 根据HintGraph，设置ImplGraph中的Api信息
   * @param [in,out] optimize_graph 优化后的图，同时也将api信息设置在这个图上
   */
  static Status CompleteApiInfo(const af::AscGraph &optimize_graph);

  static void AppendOriginalSizeVar(const af::AscGraph &graph, SizeVarSet &size_vars);

  // Capture the complete frontend shape ABI before optimization can remove or
  // rewrite size vars.  This includes symbols embedded directly in ASC axis
  // and tensor expressions.  For Inductor's ksN naming convention this restores
  // the frontend order using the numeric suffix; legacy/non-Inductor symbols
  // keep the existing deterministic expression order.
  static Status CollectFrontendShapeVars(const af::AscGraph &graph, std::vector<af::Expression> &frontend_shape_vars);

  static Status NormalizeFrontendShapeVars(std::vector<af::Expression> &frontend_shape_vars);
};
}  // namespace optimize

#endif  // HINT_GRAPH_INFO_COMPLETE_H
