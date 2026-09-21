/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "softmax_pattern_fusion_pass.h"

#include <set>

#include "optimize/graph_pass/pass_utils.h"
#include "schedule_utils.h"
#include "softmax_pattern_fusion_utils.h"

namespace optimize {

Status SoftmaxPatternFusionPass::RunPass(af::AscGraph &graph) {
  bool changed = false;
  std::set<af::AscNodePtr> visited_nodes;
  for (const auto &node : graph.GetAllNodes()) {
    if (visited_nodes.find(node) != visited_nodes.end()) {
      continue;
    }
    softmax_pattern::MatchResult pattern;
    if (!softmax_pattern::MatchStable(node, pattern)) {
      continue;
    }
    GELOGD("Stable Softmax pattern found at node [%s].", node->GetNamePtr());
    GE_ASSERT_SUCCESS(softmax_pattern::ReplaceWithSoftmax(graph, pattern));
    changed = true;
    visited_nodes.insert(pattern.true_div_node);
    visited_nodes.insert(pattern.sum_broadcast_node);
    visited_nodes.insert(pattern.sum_node);
    visited_nodes.insert(pattern.exp_node);
    visited_nodes.insert(pattern.sub_node);
    visited_nodes.insert(pattern.max_broadcast_node);
    visited_nodes.insert(pattern.max_node);
  }

  if (changed) {
    GE_ASSERT_SUCCESS(PassUtils::PruneGraph(graph));
    GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  }
  return af::SUCCESS;
}
}  // namespace optimize
