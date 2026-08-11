/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_PASS_UTILS_H
#define CANN_GRAPH_ENGINE_PASS_UTILS_H

#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace optimize {
class PassUtils {
 public:
  // 图剪枝，data会连一个控制边到输出节点上
  static af::Status PruneGraph(af::AscGraph &graph);

  static af::Status RelinkAllOutNodeToSrc(const af::OutDataAnchorPtr &old_src, const af::OutDataAnchorPtr &new_src);

  static bool IsExprVectorEqual(const std::vector<af::Expression> &lhs, const std::vector<af::Expression> &rhs);

  // 创建和目标节点相同大小的brc(1)结构
  static af::AscNodePtr CreateOneScalarBrc(af::AscGraph &graph, const af::AscNodePtr &ref_node);
};
}  // namespace optimize

#endif  // CANN_GRAPH_ENGINE_PASS_UTILS_H
