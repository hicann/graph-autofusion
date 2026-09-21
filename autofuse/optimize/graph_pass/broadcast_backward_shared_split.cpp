/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0.
 */
#include "broadcast_backward_shared_split.h"

#include <algorithm>
#include <string>
#include <vector>

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "graph_utils.h"
#include "schedule_utils.h"

using namespace ascir;
using namespace af::ascir_op;
using namespace af::ops;

namespace optimize {
namespace broadcast_backward_shared_split {
namespace {

using af::AscGraph;
using af::AscNode;
using af::AscNodePtr;
using af::AscTensorAttr;
using af::FAILED;
using af::SUCCESS;
using NodePtr = AscNodePtr;

constexpr const char *kStoreType = Store::Type;
constexpr const char *kScalarType = Scalar::Type;
constexpr const char *kBroadcastType = Broadcast::Type;
constexpr const char *kCastType = Cast::Type;

NodePtr ToAscNode(const af::NodePtr &node) {
  return std::dynamic_pointer_cast<AscNode>(node);
}

bool IsSingleInAndOutNode(const NodePtr &node) {
  return node != nullptr && node->GetInDataNodesSize() == 1UL && node->GetOutDataNodesSize() == 1UL;
}

Status GetPeerOutNodeSafe(const NodePtr &node, NodePtr &peer, int32_t index) {
  GE_ASSERT_NOTNULL(node);
  if (index < 0 || static_cast<size_t>(index) >= node->GetAllInDataAnchorsSize()) {
    return FAILED;
  }
  auto in_anchor = node->GetInDataAnchor(index);
  GE_ASSERT_NOTNULL(in_anchor);
  auto out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(out_anchor);
  peer = ToAscNode(out_anchor->GetOwnerNode());
  return SUCCESS;
}

Status GetOutputTensorAttr(const NodePtr &node, AscTensorAttr *&attr) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(node->GetAllOutDataAnchorsSize() > 0U);
  attr = &node->outputs[0].attr;
  return SUCCESS;
}

bool IsScalarInput(const NodePtr &node) {
  auto current = node;
  while (current != nullptr) {
    if (current->GetType() == kScalarType) {
      return true;
    }
    if (current->GetType() != kBroadcastType || GetPeerOutNodeSafe(current, current, 0) != SUCCESS) {
      break;
    }
  }
  return false;
}

bool IsViewOp(const NodePtr &node) {
  static const std::vector<std::string> kViewTypes = {
      Transpose::Type, Broadcast::Type, "Slice", Split::Type, Concat::Type, Gather::Type, "Sum",
      "Mean",          "Max",           "Min",   "Prod",      "Any",        "All"};
  return std::find(kViewTypes.begin(), kViewTypes.end(), node->GetType()) != kViewTypes.end();
}

bool IsDtypeNotSupported(const NodePtr &node) {
  if (node->GetType() != kCastType || node->GetOpDesc() == nullptr) {
    return false;
  }
  const auto output_desc = node->GetOpDesc()->MutableOutputDesc(0);
  if (output_desc == nullptr) {
    return true;
  }
  const auto dtype = output_desc->GetDataType();
  const std::vector<af::DataType> input_dtypes = {dtype};
  std::vector<af::DataType> output_dtypes = {dtype};
  return ScheduleUtils::CallAscirInferDataType<Broadcast>(input_dtypes, output_dtypes) != SUCCESS;
}

bool CanBackward(const NodePtr &node) {
  if (node == nullptr || node->GetType() == kStoreType || IsViewOp(node) || IsDtypeNotSupported(node)) {
    return false;
  }
  return node->GetAllOutDataAnchorsSize() <= 1U && node->GetInDataNodesSize() == 1UL;
}

bool CanSharedConsumer(const NodePtr &node) {
  return node != nullptr && node->GetType() != kStoreType && !IsViewOp(node) && !IsDtypeNotSupported(node);
}

Status TraceMergeNode(const NodePtr &start, NodePtr &merge) {
  NodePtr current = start;
  std::vector<NodePtr> visited;
  while (current != nullptr) {
    if (std::find(visited.begin(), visited.end(), current) != visited.end()) {
      return FAILED;
    }
    visited.push_back(current);
    if (current->GetInDataNodesSize() != 1UL || current->GetType() == kStoreType) {
      merge = current;
      return SUCCESS;
    }
    if (current->GetAllOutDataAnchorsSize() != 1U ||
        current->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() != 1U) {
      return FAILED;
    }
    current = ToAscNode(current->GetOutDataNodes().at(0));
  }
  return FAILED;
}

Status CollectCandidates(const AscGraph &graph, std::vector<NodePtr> &candidates) {
  for (const auto &node : graph.GetAllNodes()) {
    auto candidate = ToAscNode(node);
    if (candidate != nullptr && IsOps<Broadcast>(candidate) && candidate->GetOutDataNodesSize() > 1U &&
        !IsScalarInput(candidate)) {
      candidates.push_back(candidate);
    }
  }
  return SUCCESS;
}

Status CollectBroadcastChain(NodePtr tail, std::vector<NodePtr> &chain, NodePtr &source) {
  NodePtr current = tail;
  while (IsOps<Broadcast>(current) && current->GetInDataNodesSize() == 1UL) {
    chain.push_back(current);
    GE_ASSERT_SUCCESS(GetPeerOutNodeSafe(current, current, 0));
  }
  std::reverse(chain.begin(), chain.end());
  source = current;
  return chain.empty() || source == nullptr || IsOps<Broadcast>(source) ? FAILED : SUCCESS;
}

Status CloneBroadcast(AscGraph &graph, const NodePtr &source, const std::string &name, NodePtr &clone) {
  Broadcast op(name.c_str());
  clone = graph.AddNode(op);
  GE_ASSERT_NOTNULL(clone);
  AscTensorAttr *attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(source, attr));
  op.attr.sched = source->attr.sched;
  op.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  op.attr.api.type = af::ApiType::kAPITypeCompute;
  op.y.dtype = attr->dtype;
  *op.y.axis = attr->axis;
  *op.y.repeats = attr->repeats;
  *op.y.strides = attr->strides;
  return SUCCESS;
}

bool AllSame(const std::vector<NodePtr> &nodes) {
  return nodes.empty() ||
         std::all_of(nodes.begin() + 1, nodes.end(), [&](const NodePtr &node) { return node == nodes.front(); });
}

struct BranchPlan {
  std::vector<NodePtr> consumers;
  std::vector<NodePtr> successors;
  std::vector<int32_t> successor_inputs;
};

Status BuildBranchPlan(const NodePtr &tail, BranchPlan &plan) {
  auto out = tail->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out);
  const auto &peers = out->GetPeerInDataAnchors();
  if (peers.size() < 2U || peers.size() > 8U) {
    return FAILED;
  }
  for (const auto &peer : peers) {
    if (peer == nullptr) {
      return FAILED;
    }
    auto consumer = ToAscNode(peer->GetOwnerNode());
    if (!IsSingleInAndOutNode(consumer) || !CanBackward(consumer)) {
      return FAILED;
    }
    auto successor = ToAscNode(consumer->GetOutDataNodes().at(0));
    if (successor == nullptr || successor->GetAllInDataAnchorsSize() <= 1U) {
      return FAILED;
    }
    int32_t input = -1;
    for (uint32_t i = 0U; i < successor->GetAllInDataAnchorsSize(); ++i) {
      auto anchor = successor->GetInDataAnchor(static_cast<int32_t>(i));
      if (anchor != nullptr && anchor->GetPeerOutAnchor() != nullptr &&
          ToAscNode(anchor->GetPeerOutAnchor()->GetOwnerNode()) == consumer) {
        input = static_cast<int32_t>(i);
        break;
      }
    }
    if (input < 0) {
      return FAILED;
    }
    plan.consumers.push_back(consumer);
    plan.successors.push_back(successor);
    plan.successor_inputs.push_back(input);
  }
  return SUCCESS;
}

Status BuildBranchChains(AscGraph &graph, const std::vector<NodePtr> &original, const BranchPlan &plan,
                         std::vector<std::vector<NodePtr>> &chains) {
  chains.resize(plan.consumers.size());
  chains[0] = original;
  for (size_t branch = 1U; branch < chains.size(); ++branch) {
    for (const auto &node : original) {
      NodePtr clone;
      GE_ASSERT_SUCCESS(
          CloneBroadcast(graph, node, node->GetName() + "_branch_split_" + std::to_string(branch), clone));
      chains[branch].push_back(clone);
    }
  }
  return SUCCESS;
}

Status ApplyBranchPlan(const NodePtr &source, const std::vector<NodePtr> &original, const BranchPlan &plan,
                       const std::vector<std::vector<NodePtr>> &chains) {
  GE_ASSERT_GRAPH_SUCCESS(
      af::GraphUtils::RemoveEdge(source->GetOutDataAnchor(0), original.front()->GetInDataAnchor(0)));
  for (size_t i = 0U; i + 1U < original.size(); ++i) {
    GE_ASSERT_GRAPH_SUCCESS(
        af::GraphUtils::RemoveEdge(original[i]->GetOutDataAnchor(0), original[i + 1U]->GetInDataAnchor(0)));
  }
  AscTensorAttr *source_attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(source, source_attr));
  for (size_t branch = 0U; branch < plan.consumers.size(); ++branch) {
    auto consumer = plan.consumers[branch];
    auto successor = plan.successors[branch];
    GE_ASSERT_GRAPH_SUCCESS(
        af::GraphUtils::RemoveEdge(original.back()->GetOutDataAnchor(0), consumer->GetInDataAnchor(0)));
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(consumer->GetOutDataAnchor(0),
                                                       successor->GetInDataAnchor(plan.successor_inputs[branch])));
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(source->GetOutDataAnchor(0), consumer->GetInDataAnchor(0)));
    const auto &chain = chains[branch];
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(consumer->GetOutDataAnchor(0), chain.front()->GetInDataAnchor(0)));
    for (size_t i = 0U; i + 1U < chain.size(); ++i) {
      GE_ASSERT_GRAPH_SUCCESS(
          af::GraphUtils::AddEdge(chain[i]->GetOutDataAnchor(0), chain[i + 1U]->GetInDataAnchor(0)));
    }
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(chain.back()->GetOutDataAnchor(0),
                                                    successor->GetInDataAnchor(plan.successor_inputs[branch])));
    AscTensorAttr *consumer_attr = nullptr;
    GE_ASSERT_SUCCESS(GetOutputTensorAttr(consumer, consumer_attr));
    consumer_attr->axis = source_attr->axis;
    consumer_attr->repeats = source_attr->repeats;
    consumer_attr->strides = source_attr->strides;
  }
  return SUCCESS;
}

Status SplitOneBranch(AscGraph &graph, const NodePtr &tail, bool &changed) {
  changed = false;
  std::vector<NodePtr> original;
  NodePtr source;
  if (CollectBroadcastChain(tail, original, source) != SUCCESS) {
    return SUCCESS;
  }
  BranchPlan plan;
  if (BuildBranchPlan(tail, plan) != SUCCESS || AllSame(plan.successors)) {
    return SUCCESS;
  }
  std::vector<std::vector<NodePtr>> chains;
  GE_ASSERT_SUCCESS(BuildBranchChains(graph, original, plan, chains));
  GE_ASSERT_SUCCESS(ApplyBranchPlan(source, original, plan, chains));
  changed = true;
  return SUCCESS;
}

struct ConsumerPlan {
  NodePtr source;
  std::vector<NodePtr> consumers;
  std::vector<int32_t> inputs;
};

Status BuildConsumerPlan(const NodePtr &broadcast, ConsumerPlan &plan) {
  GE_ASSERT_SUCCESS(GetPeerOutNodeSafe(broadcast, plan.source, 0));
  if (plan.source == nullptr || IsOps<Broadcast>(plan.source)) {
    return FAILED;
  }
  auto out = broadcast->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out);
  const auto &peers = out->GetPeerInDataAnchors();
  if (peers.size() <= 1U) {
    return FAILED;
  }
  for (const auto &peer : peers) {
    auto consumer = peer == nullptr ? nullptr : ToAscNode(peer->GetOwnerNode());
    if (consumer == nullptr || !CanSharedConsumer(consumer)) {
      return FAILED;
    }
    plan.consumers.push_back(consumer);
    plan.inputs.push_back(peer->GetIdx());
  }
  if (AllSame(plan.consumers)) {
    return FAILED;
  }
  std::vector<NodePtr> merges;
  bool trace_failed = false;
  for (const auto &consumer : plan.consumers) {
    NodePtr merge;
    if (TraceMergeNode(consumer, merge) != SUCCESS) {
      trace_failed = true;
      break;
    }
    merges.push_back(merge);
  }
  if (!trace_failed && AllSame(merges)) {
    return FAILED;
  }
  return SUCCESS;
}

Status SplitOneConsumer(AscGraph &graph, const NodePtr &broadcast, bool &changed) {
  changed = false;
  ConsumerPlan plan;
  if (BuildConsumerPlan(broadcast, plan) != SUCCESS) {
    return SUCCESS;
  }
  for (size_t i = 1U; i < plan.consumers.size(); ++i) {
    NodePtr clone;
    GE_ASSERT_SUCCESS(
        CloneBroadcast(graph, broadcast, broadcast->GetName() + "_consumer_split_" + std::to_string(i), clone));
    auto consumer_input = plan.consumers[i]->GetInDataAnchor(plan.inputs[i]);
    GE_ASSERT_NOTNULL(consumer_input);
    auto peer_out = consumer_input->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out);
    auto source_out = plan.source->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(source_out);
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(peer_out, consumer_input));
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(source_out, clone->GetInDataAnchor(0)));
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(clone->GetOutDataAnchor(0), consumer_input));
  }
  changed = true;
  return SUCCESS;
}

}  // namespace

Status SplitSharedBroadcastBranches(AscGraph &graph) {
  std::vector<NodePtr> candidates;
  GE_ASSERT_SUCCESS(CollectCandidates(graph, candidates));
  bool changed = false;
  for (const auto &candidate : candidates) {
    bool candidate_changed = false;
    GE_ASSERT_SUCCESS(SplitOneBranch(graph, candidate, candidate_changed));
    changed = changed || candidate_changed;
  }
  if (changed) {
    GE_ASSERT_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  }
  return SUCCESS;
}

Status SplitSharedBroadcastConsumers(AscGraph &graph) {
  std::vector<NodePtr> candidates;
  GE_ASSERT_SUCCESS(CollectCandidates(graph, candidates));
  bool changed = false;
  for (const auto &candidate : candidates) {
    bool candidate_changed = false;
    GE_ASSERT_SUCCESS(SplitOneConsumer(graph, candidate, candidate_changed));
    changed = changed || candidate_changed;
  }
  if (changed) {
    GE_ASSERT_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  }
  return SUCCESS;
}

}  // namespace broadcast_backward_shared_split
}  // namespace optimize
