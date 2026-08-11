/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "same_source_broadcast_cse_pass.h"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "ascir/meta/ascir_ops_utils.h"
#include "ascir_ops.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "graph/utils/graph_utils.h"
#include "optimize/graph_pass/pass_utils.h"
#include "optimize/schedule_utils.h"

namespace optimize {
namespace {
using BroadcastGroup = std::vector<af::AscNodePtr>;
using BroadcastGroups = std::map<af::OutDataAnchorPtr, BroadcastGroup>;

bool IsTensorViewEqual(const af::AscTensorAttr &lhs, const af::AscTensorAttr &rhs) {
  return static_cast<ge::DataType>(lhs.dtype) == static_cast<ge::DataType>(rhs.dtype) && lhs.axis == rhs.axis &&
         PassUtils::IsExprVectorEqual(lhs.repeats, rhs.repeats) &&
         PassUtils::IsExprVectorEqual(lhs.strides, rhs.strides) && lhs.vectorized_axis == rhs.vectorized_axis &&
         PassUtils::IsExprVectorEqual(lhs.vectorized_strides, rhs.vectorized_strides);
}

bool IsScalarBroadcast(const af::AscNodePtr &node) {
  const auto &strides = node->inputs[0].attr.strides;
  return !strides.empty() && std::all_of(strides.begin(), strides.end(), [](const af::Expression &stride) {
    return af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolZero) == af::TriBool::kTrue;
  });
}

bool HasBoundaryConsumer(const af::AscNodePtr &node) {
  for (const auto &out_node : node->GetOutDataNodes()) {
    if (af::ops::IsOps<af::ascir_op::Store>(out_node) || af::ops::IsOps<af::ascir_op::Output>(out_node) ||
        af::ops::IsOps<af::ascir_op::Workspace>(out_node)) {
      return true;
    }
  }
  return false;
}

// 仅把真正发生降维的 Reduce 输出作为统计量候选，避免普通 Reduce 扩大优化范围。
bool IsReducedOutput(const af::AscNodePtr &node) {
  if (node == nullptr || !ScheduleUtils::IsReduce(node) || node->inputs.Size() == 0U ||
      node->GetAllOutDataAnchorsSize() == 0U) {
    return false;
  }

  const auto &input_repeats = node->inputs[0].attr.repeats;
  const auto &output_repeats = node->outputs[0].attr.repeats;
  if (input_repeats.size() != output_repeats.size() || input_repeats.empty()) {
    return false;
  }

  for (size_t index = 0UL; index < input_repeats.size(); ++index) {
    if (af::SymbolicUtils::StaticCheckEq(input_repeats[index], output_repeats[index]) != af::TriBool::kTrue) {
      GELOGD("Reduce node [%s] changes the shape, so its output is eligible for Broadcast CSE.", node->GetNamePtr());
      return true;
    }
  }
  return false;
}

// 候选必须是普通 Broadcast，且不能直接连接边界节点；来源限制为实际降维的 Reduce 输出。
bool IsCandidate(const af::AscNodePtr &node) {
  if (!af::ops::IsOps<af::ascir_op::Broadcast>(node) || node->GetAllInDataAnchorsSize() != 1U ||
      node->GetAllOutDataAnchorsSize() != 1U || node->GetInControlNodesSize() != 0U ||
      node->GetOutControlNodesSize() != 0U || IsScalarBroadcast(node) || HasBoundaryConsumer(node)) {
    return false;
  }

  const auto input_anchor = node->GetInDataAnchor(0);
  if (input_anchor == nullptr) {
    return false;
  }
  const auto source_anchor = input_anchor->GetPeerOutAnchor();
  if (source_anchor == nullptr) {
    return false;
  }
  const auto source = std::dynamic_pointer_cast<af::AscNode>(source_anchor->GetOwnerNode());
  if (!IsReducedOutput(source)) {
    GELOGD("Skip Broadcast candidate [%s]: its source is not a dimension-reducing Reduce node.", node->GetNamePtr());
    return false;
  }
  GELOGD("Accept Broadcast candidate [%s], source Reduce node [%s].", node->GetNamePtr(),
         source == nullptr ? "unknown" : source->GetNamePtr());
  return true;
}

// Broadcast 等价不仅要求 Tensor 视图一致，还要求调度和向量化属性一致。
bool IsEquivalentBroadcast(const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) {
  if (lhs->attr.sched.axis != rhs->attr.sched.axis || lhs->attr.sched.loop_axis != rhs->attr.sched.loop_axis ||
      lhs->attr.sched.exec_condition != rhs->attr.sched.exec_condition) {
    return false;
  }
  return IsTensorViewEqual(lhs->inputs[0].attr, rhs->inputs[0].attr) &&
         IsTensorViewEqual(lhs->outputs[0].attr, rhs->outputs[0].attr);
}

// 合并后 Tensor 生命周期可能延长，但 canonical 必须早于所有 duplicate 消费者。
bool CanMerge(const af::AscNodePtr &canonical, const af::AscNodePtr &duplicate) {
  const int64_t canonical_start = canonical->GetOpDescBarePtr()->GetId();
  for (const auto &out_node : duplicate->GetOutDataNodes()) {
    const int64_t consumer_id = out_node->GetOpDescBarePtr()->GetId();
    if (consumer_id <= canonical_start) {
      GELOGD("Skip Broadcast merge: canonical [%s] with ID=%ld is not earlier than consumer [%s] with ID=%ld.",
             canonical->GetNamePtr(), canonical_start, out_node->GetNamePtr(), consumer_id);
      return false;
    }
  }
  return true;
}

// 保留拓扑较早的 Broadcast，确保跨阶段共享结果时不会引用尚未计算的数据。
af::AscNodePtr SelectCanonical(const std::vector<af::AscNodePtr> &nodes) {
  return *std::min_element(nodes.begin(), nodes.end(), [](const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) {
    return lhs->GetOpDescBarePtr()->GetId() < rhs->GetOpDescBarePtr()->GetId();
  });
}

// 使用图结构判定限制作用域，避免 R 轴大小等性能阈值影响 Norm 结构识别。
bool IsNormLikeScope(const af::AscGraph &graph) {
  const bool is_norm_struct = ScheduleUtils::IsNormStruct(graph);
  GELOGD("Norm-like scope check for graph [%s]: IsNormStruct=%d.", graph.GetName().c_str(),
         static_cast<int32_t>(is_norm_struct));
  return is_norm_struct;
}

Status MergeBroadcast(const af::AscNodePtr &canonical, const af::AscNodePtr &duplicate) {
  auto canonical_out = canonical->GetOutDataAnchor(0);
  auto duplicate_out = duplicate->GetOutDataAnchor(0);
  auto duplicate_in = duplicate->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(canonical_out);
  GE_ASSERT_NOTNULL(duplicate_out);
  GE_ASSERT_NOTNULL(duplicate_in);
  auto source_out = duplicate_in->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(source_out);

  GE_ASSERT_SUCCESS(PassUtils::RelinkAllOutNodeToSrc(duplicate_out, canonical_out));
  GE_ASSERT_SUCCESS(af::GraphUtils::RemoveEdge(source_out, duplicate_in));
  auto owner_graph = duplicate->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph);
  GE_ASSERT_SUCCESS(owner_graph->RemoveNode(duplicate));
  GELOGI("Merged equivalent Broadcast node [%s] into canonical node [%s].", duplicate->GetNamePtr(),
         canonical->GetNamePtr());
  return af::SUCCESS;
}
Status CollectBroadcastGroups(const af::AscGraph &graph, BroadcastGroups &source_to_broadcasts,
                              size_t &candidate_count) {
  for (const auto &node : graph.GetAllNodes()) {
    if (!IsCandidate(node)) {
      continue;
    }
    ++candidate_count;
    auto input_anchor = node->GetInDataAnchor(0);
    GE_ASSERT_NOTNULL(input_anchor);
    auto source_anchor = input_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(source_anchor);
    source_to_broadcasts[source_anchor].emplace_back(node);
  }
  return af::SUCCESS;
}

Status MergeEquivalentBroadcastGroup(const BroadcastGroup &broadcasts, size_t &equivalent_group_count,
                                     size_t &merged_count) {
  std::set<af::AscNodePtr> processed;
  for (const auto &broadcast : broadcasts) {
    if (processed.count(broadcast) > 0UL) {
      continue;
    }
    BroadcastGroup equivalent_nodes;
    for (const auto &candidate : broadcasts) {
      if (processed.count(candidate) == 0UL && IsEquivalentBroadcast(broadcast, candidate)) {
        equivalent_nodes.emplace_back(candidate);
      }
    }
    processed.insert(equivalent_nodes.begin(), equivalent_nodes.end());
    if (equivalent_nodes.size() <= 1UL) {
      continue;
    }

    ++equivalent_group_count;
    const auto canonical = SelectCanonical(equivalent_nodes);
    GELOGD("Found equivalent Broadcast group: size=%zu, canonical=[%s].", equivalent_nodes.size(),
           canonical->GetNamePtr());
    for (const auto &duplicate : equivalent_nodes) {
      if (duplicate == canonical || !CanMerge(canonical, duplicate)) {
        continue;
      }
      GE_ASSERT_SUCCESS(MergeBroadcast(canonical, duplicate));
      ++merged_count;
    }
  }
  return af::SUCCESS;
}

Status MergeBroadcastGroups(BroadcastGroups &source_to_broadcasts, size_t &equivalent_group_count,
                            size_t &merged_count) {
  for (auto &[source_anchor, broadcasts] : source_to_broadcasts) {
    const auto source_node = std::dynamic_pointer_cast<af::AscNode>(source_anchor->GetOwnerNode());
    GELOGD("Inspect source [%s] with %zu Broadcast candidates.",
           source_node == nullptr ? "unknown" : source_node->GetNamePtr(), broadcasts.size());
    GE_ASSERT_SUCCESS(MergeEquivalentBroadcastGroup(broadcasts, equivalent_group_count, merged_count));
  }
  return af::SUCCESS;
}
}  // namespace

Status SameSourceBroadcastCsePass::RunPass(af::AscGraph &graph) {
  if (!IsNormLikeScope(graph)) {
    GELOGD("Skip same-source Broadcast CSE for graph [%s]: not a Norm-like graph.", graph.GetName().c_str());
    return af::SUCCESS;
  }

  // 先按共同输入源分组，再在每个分组内比较 Broadcast 的完整视图和调度属性。
  BroadcastGroups source_to_broadcasts;
  size_t candidate_count = 0UL;
  GE_ASSERT_SUCCESS(CollectBroadcastGroups(graph, source_to_broadcasts, candidate_count));
  GELOGI("Same-source Broadcast CSE: graph [%s] has %zu candidates in %zu source groups.", graph.GetName().c_str(),
         candidate_count, source_to_broadcasts.size());

  size_t equivalent_group_count = 0UL;
  size_t merged_count = 0UL;
  GE_ASSERT_SUCCESS(MergeBroadcastGroups(source_to_broadcasts, equivalent_group_count, merged_count));
  GELOGI("Same-source Broadcast CSE finished: equivalent groups=%zu, merged nodes=%zu.", equivalent_group_count,
         merged_count);
  return af::SUCCESS;
}
}  // namespace optimize
