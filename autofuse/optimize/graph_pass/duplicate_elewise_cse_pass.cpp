/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "duplicate_elewise_cse_pass.h"

#include <algorithm>
#include <map>
#include <unordered_set>
#include <vector>

#include "ascir_ops.h"
#include "graph/utils/graph_utils.h"
#include "node_utils.h"
#include "optimize/graph_pass/pass_utils.h"

namespace optimize {
namespace {
// 同源等价 elewise 节点集合，即两个输入分别来自相同源输出端口的同类算子节点列表
using ElewiseGroup = std::vector<af::AscNodePtr>;
// key 为 (input0 源输出锚点, input1 源输出锚点)，value 为共享该输入源对的全部候选节点；
// map 的指针比较语义确保只有真正连到同一个源输出端口的重复消费才归入同一组
using ElewiseGroups = std::map<std::pair<af::OutDataAnchorPtr, af::OutDataAnchorPtr>, ElewiseGroup>;
static const std::unordered_set<std::string> kCseSupportedTypes = {af::ascir_op::Sub::Type};

// 比较 dtype、axis、repeats、strides 是否完全一致
bool IsOutputViewEqual(const af::AscTensorAttr &lhs, const af::AscTensorAttr &rhs) {
  return lhs.dtype == rhs.dtype && lhs.axis == rhs.axis && PassUtils::IsExprVectorEqual(lhs.repeats, rhs.repeats) &&
         PassUtils::IsExprVectorEqual(lhs.strides, rhs.strides);
}

// 判断节点是否为 CSE 候选：类型在支持表内，且恰好 2 输入 1 输出无控制边
bool IsCandidate(const af::AscNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  if (kCseSupportedTypes.find(node->GetType()) == kCseSupportedTypes.end()) {
    return false;
  }
  if (node->GetAllInDataAnchorsSize() != 2U || node->GetAllOutDataAnchorsSize() != 1U ||
      node->GetInControlNodesSize() != 0U || node->GetOutControlNodesSize() != 0U) {
    return false;
  }
  return true;
}

// 判断两个同源节点的输出 view 和调度轴是否等价
bool IsOutputEquivalent(const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) {
  if (lhs->attr.sched.axis != rhs->attr.sched.axis || lhs->attr.sched.loop_axis != rhs->attr.sched.loop_axis) {
    return false;
  }
  return IsOutputViewEqual(lhs->outputs[0].attr, rhs->outputs[0].attr);
}

// 选择拓扑序最早（OpDesc ID 最小）的节点作为保留节点
af::AscNodePtr SelectCanonical(const ElewiseGroup &nodes) {
  return *std::min_element(nodes.begin(), nodes.end(), [](const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) {
    return lhs->GetOpDescBarePtr()->GetId() < rhs->GetOpDescBarePtr()->GetId();
  });
}

// 判断 canonical 是否早于 duplicate 的所有消费者，保证合并后不引用尚未计算的数据
bool CanMerge(const af::AscNodePtr &canonical, const af::AscNodePtr &duplicate) {
  const int64_t canonical_start = canonical->GetOpDescBarePtr()->GetId();
  for (const auto &out_node : duplicate->GetOutDataNodes()) {
    if (out_node->GetOpDescBarePtr()->GetId() <= canonical_start) {
      return false;
    }
  }
  return true;
}

// 将 duplicate 的下游边重定向到 canonical，断开输入边后删除冗余节点
Status MergeElewise(const af::AscNodePtr &canonical, const af::AscNodePtr &duplicate) {
  auto canonical_out = canonical->GetOutDataAnchor(0);
  auto duplicate_out = duplicate->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(canonical_out);
  GE_ASSERT_NOTNULL(duplicate_out);
  GE_ASSERT_SUCCESS(PassUtils::RelinkAllOutNodeToSrc(duplicate_out, canonical_out));
  af::NodeUtils::UnlinkAll(*duplicate);
  auto owner_graph = duplicate->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph);
  GE_ASSERT_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(owner_graph, duplicate));
  GELOGI("Merged duplicate elewise node [%s] into canonical node [%s].", duplicate->GetNamePtr(),
         canonical->GetNamePtr());
  return af::SUCCESS;
}

// 遍历图中所有候选节点，按输入源锚点对分组收集
Status CollectElewiseGroups(const af::AscGraph &graph, ElewiseGroups &source_to_elewise, size_t &candidate_count) {
  for (const auto &node : graph.GetAllNodes()) {
    if (!IsCandidate(node)) {
      continue;
    }
    const auto in_anchor0 = node->GetInDataAnchor(0U);
    const auto in_anchor1 = node->GetInDataAnchor(1U);
    if (in_anchor0 == nullptr || in_anchor1 == nullptr) {
      continue;
    }
    const auto src0 = in_anchor0->GetPeerOutAnchor();
    const auto src1 = in_anchor1->GetPeerOutAnchor();
    if (src0 == nullptr || src1 == nullptr) {
      continue;
    }
    ++candidate_count;
    source_to_elewise[{src0, src1}].emplace_back(node);
  }
  return af::SUCCESS;
}

// 在同源分组内选择 canonical，将输出等价的冗余节点合并到 canonical
Status MergeEquivalentElewiseGroup(const ElewiseGroup &nodes, size_t &merged_count) {
  if (nodes.size() <= 1UL) {
    return af::SUCCESS;
  }
  const auto canonical = SelectCanonical(nodes);
  for (const auto &duplicate : nodes) {
    if (duplicate == canonical || !IsOutputEquivalent(canonical, duplicate) || !CanMerge(canonical, duplicate)) {
      continue;
    }
    GE_ASSERT_SUCCESS(MergeElewise(canonical, duplicate));
    ++merged_count;
  }
  return af::SUCCESS;
}
}  // namespace

// Pass 入口：收集同源候选分组，逐组合并等价冗余节点
Status DuplicateElewiseCsePass::RunPass(af::AscGraph &graph) {
  ElewiseGroups source_to_elewise;
  size_t candidate_count = 0UL;
  GE_ASSERT_SUCCESS(CollectElewiseGroups(graph, source_to_elewise, candidate_count));
  GELOGI("Duplicate elewise CSE: graph [%s] has %zu candidates in %zu source groups.", graph.GetName().c_str(),
         candidate_count, source_to_elewise.size());

  size_t merged_count = 0UL;
  for (const auto &entry : source_to_elewise) {
    GE_ASSERT_SUCCESS(MergeEquivalentElewiseGroup(entry.second, merged_count));
  }
  GELOGI("Duplicate elewise CSE finished: merged nodes=%zu.", merged_count);
  return af::SUCCESS;
}
}  // namespace optimize
