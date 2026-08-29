/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "split_concat_optimization_pass.h"

#include "ascir_ops.h"
#include "schedule_utils.h"
#include "ascir/meta/ascir_ops_utils.h"
#include "task_generator/concat_schedule_case_generator.h"
#include "task_generator/split_schedule_case_generator.h"

namespace optimize {
namespace {
constexpr size_t kExpectedNodeNum = 1UL;
}  // namespace

Status SplitConcatOptimizationPass::RunPass(af::AscGraph &graph) {
  std::vector<af::AscNodePtr> split_nodes;
  std::vector<af::AscNodePtr> concat_nodes;
  FindSplitAndConcatNodes(graph, split_nodes, concat_nodes);
  if (split_nodes.empty() || concat_nodes.empty()) {
    GELOGI("graph[%s] does not have split concat fusion", graph.GetName().c_str());
    return af::SUCCESS;
  }
  GE_ASSERT_TRUE(concat_nodes.size() == kExpectedNodeNum, "expect just 1 Concat node, but got %zu ",
                 concat_nodes.size());
  const auto &split_node = split_nodes.front();
  size_t split_dim = 0UL;
  bool is_first_dim_split = false;
  GE_ASSERT_SUCCESS(ScheduleUtils::ResolveDiffDim(split_node, split_dim, is_first_dim_split));
  if (is_first_dim_split) {
    GELOGI("%s is first dim split, optimize out concat", split_node->GetNamePtr());
    GE_ASSERT_SUCCESS(OptimizeOutSplit(graph));
    return af::SUCCESS;
  }

  const auto &concat_node = concat_nodes.front();
  size_t concat_dim = 0UL;
  bool is_first_dim_concat = false;
  GE_ASSERT_SUCCESS(ScheduleUtils::ResolveDiffDim(concat_node, concat_dim, is_first_dim_concat));
  GE_ASSERT_TRUE(is_first_dim_concat, "%s: concat_dim = %zu, not the first dim", concat_node->GetNamePtr(), concat_dim);
  GE_ASSERT_SUCCESS(OptimizeOutConcat(graph));
  return af::SUCCESS;
}

Status SplitConcatOptimizationPass::OptimizeOutSplit(ascir::HintGraph &owner_graph) {
  // first-dim concat
  std::vector<ascir::ImplGraph> graphs;
  std::vector<std::string> unused_score_funcs;
  // graph被原地修改
  GE_ASSERT_SUCCESS(SplitFusionCaseGenerator().Generate(owner_graph, graphs, unused_score_funcs));
  GE_ASSERT_TRUE(graphs.size() == 1UL, "first dim concat should generate only one template, but got %zu",
                 graphs.size());
  return af::SUCCESS;
}

Status SplitConcatOptimizationPass::OptimizeOutConcat(ascir::HintGraph &owner_graph) {
  // first-dim concat
  std::vector<ascir::ImplGraph> graphs;
  std::vector<std::string> unused_score_funcs;
  // graph被原地修改
  GE_ASSERT_SUCCESS(
      ConcatFusionCaseGenerator().SetConvertToStoreMode().Generate(owner_graph, graphs, unused_score_funcs));
  GE_ASSERT_TRUE(graphs.size() <= 1UL, "first dim concat should generate only one template, but got %zu",
                 graphs.size());
  return af::SUCCESS;
}

void SplitConcatOptimizationPass::FindSplitAndConcatNodes(const ascir::HintGraph &owner_graph,
                                                          std::vector<af::AscNodePtr> &split_nodes,
                                                          std::vector<af::AscNodePtr> &concat_nodes) {
  for (const auto &node : owner_graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Concat>(node)) {
      concat_nodes.emplace_back(node);
    } else if (af::ops::IsOps<af::ascir_op::Split>(node)) {
      split_nodes.emplace_back(node);
    } else {
      // do nothing
    }
  }
}
}  // namespace optimize
