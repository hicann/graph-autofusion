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
#include <vector>

#include "ascir_ops.h"
#include "graph/utils/graph_utils.h"
#include "node_utils.h"
#include "optimize/graph_pass/pass_utils.h"
#include "schedule_utils.h"

using namespace af::ops;
using namespace af::ascir_op;

namespace optimize {
namespace {
constexpr const char *kSoftmaxType = "Softmax";

class SoftmaxOp : public af::Operator {
 public:
  explicit SoftmaxOp(const std::string &name) : af::Operator(name.c_str(), kSoftmaxType) {
    InputRegister("x", "T");
    OutputRegister("y", "T");
  }
};

struct SoftmaxPattern {
  af::OutDataAnchorPtr input_anchor;
  af::AscNodePtr max_node;
  af::AscNodePtr max_broadcast_node;
  af::AscNodePtr sub_node;
  af::AscNodePtr exp_node;
  af::AscNodePtr sum_node;
  af::AscNodePtr sum_broadcast_node;
  af::AscNodePtr true_div_node;
};

af::AscNodePtr GetInputNode(const af::AscNodePtr &node, const size_t index) {
  if (node == nullptr) {
    return nullptr;
  }
  const auto in_anchor = node->GetInDataAnchor(index);
  if (in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<af::AscNode>(in_anchor->GetPeerOutAnchor()->GetOwnerNode());
}

af::OutDataAnchorPtr GetInputSrcAnchor(const af::AscNodePtr &node, const size_t index) {
  if (node == nullptr) {
    return nullptr;
  }
  const auto in_anchor = node->GetInDataAnchor(index);
  if (in_anchor == nullptr) {
    return nullptr;
  }
  return in_anchor->GetPeerOutAnchor();
}

bool HasOnlyConsumers(const af::AscNodePtr &node, const std::set<af::AscNodePtr> &expected_consumers) {
  if (node == nullptr || node->GetOutDataAnchor(0) == nullptr) {
    return false;
  }
  const auto &peer_in_anchors = node->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  if (peer_in_anchors.size() != expected_consumers.size()) {
    return false;
  }
  for (const auto &peer_in_anchor : peer_in_anchors) {
    if (peer_in_anchor == nullptr || peer_in_anchor->GetOwnerNode() == nullptr) {
      return false;
    }
    const auto consumer = std::dynamic_pointer_cast<af::AscNode>(peer_in_anchor->GetOwnerNode());
    if (expected_consumers.find(consumer) == expected_consumers.end()) {
      return false;
    }
  }
  return true;
}

bool ExprVectorEqual(const std::vector<af::Expression> &lhs, const std::vector<af::Expression> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0UL; i < lhs.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(lhs[i], rhs[i]) != af::TriBool::kTrue) {
      return false;
    }
  }
  return true;
}

bool HasSameTensorLayout(const af::AscTensorAttr &lhs, const af::AscTensorAttr &rhs) {
  return lhs.axis == rhs.axis && ExprVectorEqual(lhs.repeats, rhs.repeats) && ExprVectorEqual(lhs.strides, rhs.strides);
}

template <typename T>
bool IsOpsSafe(const af::AscNodePtr &node) {
  return node != nullptr && IsOps<T>(node);
}

bool IsSameReduceLayout(const af::AscNodePtr &max_node, const af::AscNodePtr &sum_node) {
  return max_node != nullptr && sum_node != nullptr &&
         HasSameTensorLayout(max_node->outputs[0].attr, sum_node->outputs[0].attr);
}

bool MatchStableSoftmaxStructure(const af::AscNodePtr &true_div_node, SoftmaxPattern &pattern) {
  if (!IsOpsSafe<TrueDiv>(true_div_node)) {
    return false;
  }

  const auto exp_node = GetInputNode(true_div_node, 0UL);
  const auto sum_broadcast_node = GetInputNode(true_div_node, 1UL);
  if (!IsOpsSafe<Exp>(exp_node) || !IsOpsSafe<Broadcast>(sum_broadcast_node)) {
    return false;
  }

  const auto sum_node = GetInputNode(sum_broadcast_node, 0UL);
  if (!IsOpsSafe<Sum>(sum_node) || GetInputSrcAnchor(sum_node, 0UL) != exp_node->GetOutDataAnchor(0)) {
    return false;
  }

  const auto sub_node = GetInputNode(exp_node, 0UL);
  if (!IsOpsSafe<Sub>(sub_node)) {
    return false;
  }

  const auto max_broadcast_node = GetInputNode(sub_node, 1UL);
  if (!IsOpsSafe<Broadcast>(max_broadcast_node)) {
    return false;
  }

  const auto max_node = GetInputNode(max_broadcast_node, 0UL);
  const auto input_anchor = GetInputSrcAnchor(sub_node, 0UL);
  if (!IsOpsSafe<Max>(max_node) || input_anchor == nullptr || GetInputSrcAnchor(max_node, 0UL) != input_anchor) {
    return false;
  }

  pattern = {input_anchor, max_node, max_broadcast_node, sub_node,
             exp_node,     sum_node, sum_broadcast_node, true_div_node};
  return true;
}

bool HasStableSoftmaxLayout(const SoftmaxPattern &pattern) {
  if (!IsSameReduceLayout(pattern.max_node, pattern.sum_node)) {
    return false;
  }
  return HasSameTensorLayout(pattern.sum_broadcast_node->outputs[0].attr, pattern.true_div_node->outputs[0].attr) &&
         HasSameTensorLayout(pattern.max_broadcast_node->outputs[0].attr, pattern.sub_node->outputs[0].attr) &&
         HasSameTensorLayout(pattern.exp_node->outputs[0].attr, pattern.true_div_node->outputs[0].attr);
}

bool HasStableSoftmaxConsumers(const SoftmaxPattern &pattern) {
  return HasOnlyConsumers(pattern.max_node, {pattern.max_broadcast_node}) &&
         HasOnlyConsumers(pattern.max_broadcast_node, {pattern.sub_node}) &&
         HasOnlyConsumers(pattern.sub_node, {pattern.exp_node}) &&
         HasOnlyConsumers(pattern.exp_node, {pattern.sum_node, pattern.true_div_node}) &&
         HasOnlyConsumers(pattern.sum_node, {pattern.sum_broadcast_node}) &&
         HasOnlyConsumers(pattern.sum_broadcast_node, {pattern.true_div_node});
}

bool MatchStableSoftmax(const af::AscNodePtr &true_div_node, SoftmaxPattern &pattern) {
  return MatchStableSoftmaxStructure(true_div_node, pattern) && HasStableSoftmaxLayout(pattern) &&
         HasStableSoftmaxConsumers(pattern);
}

Status RemoveMatchedNodes(const SoftmaxPattern &pattern) {
  const std::vector<af::AscNodePtr> nodes_to_remove = {
      pattern.true_div_node, pattern.sum_broadcast_node, pattern.sum_node, pattern.exp_node,
      pattern.sub_node,      pattern.max_broadcast_node, pattern.max_node};
  for (const auto &node : nodes_to_remove) {
    GE_CHECK_NOTNULL(node);
    af::NodeUtils::UnlinkAll(*node);
    GE_CHECK_NOTNULL(node->GetOwnerComputeGraph());
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(node->GetOwnerComputeGraph(), node));
  }
  return af::SUCCESS;
}

Status ReplaceWithSoftmax(af::AscGraph &graph, const SoftmaxPattern &pattern) {
  GE_CHECK_NOTNULL(pattern.true_div_node);
  GE_CHECK_NOTNULL(pattern.input_anchor);
  SoftmaxOp softmax_op(pattern.true_div_node->GetName() + "_softmax");
  auto softmax_node = graph.AddNode(softmax_op);
  GE_CHECK_NOTNULL(softmax_node);

  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(pattern.input_anchor, softmax_node->GetInDataAnchor(0)));

  softmax_node->attr = pattern.true_div_node->attr;
  softmax_node->inputs[0].attr = pattern.sub_node->inputs[0].attr;
  softmax_node->outputs[0].attr = pattern.true_div_node->outputs[0].attr;
  softmax_node->attr.api.compute_type = af::ComputeType::kComputeReduce;
  softmax_node->attr.api.type = af::ApiType::kAPITypeCompute;

  const auto true_div_out_anchor = pattern.true_div_node->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(true_div_out_anchor);
  const auto peer_in_anchors = true_div_out_anchor->GetPeerInDataAnchors();
  for (const auto &peer_in_anchor : peer_in_anchors) {
    GE_CHECK_NOTNULL(peer_in_anchor);
    GE_ASSERT_GRAPH_SUCCESS(
        af::GraphUtils::ReplaceEdgeSrc(true_div_out_anchor, peer_in_anchor, softmax_node->GetOutDataAnchor(0)));
  }
  return RemoveMatchedNodes(pattern);
}
}  // namespace

Status SoftmaxPatternFusionPass::RunPass(af::AscGraph &graph) {
  bool changed = false;
  std::set<af::AscNodePtr> visited_nodes;
  for (const auto &node : graph.GetAllNodes()) {
    if (visited_nodes.find(node) != visited_nodes.end()) {
      continue;
    }
    SoftmaxPattern pattern;
    if (!MatchStableSoftmax(node, pattern)) {
      continue;
    }
    GELOGD("Stable Softmax pattern found at node [%s].", node->GetNamePtr());
    GE_ASSERT_SUCCESS(ReplaceWithSoftmax(graph, pattern));
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
