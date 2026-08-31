/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

#include "broadcast_backward_pass.h"
#include "broadcast_backward_shared_split.h"

#include <algorithm>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascgraph_info_complete.h"
#include "common_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "graph_utils.h"
#include "node_utils.h"
#include "schedule_utils.h"

using namespace ascir;
using namespace af::ascir_op;
using namespace af::ops;

namespace optimize {
namespace {

using af::AscGraph;
using af::AscNode;
using af::AscNodePtr;
using af::AscTensorAttr;
using af::Expression;
using af::FAILED;
using af::SUCCESS;
using NodePtr = af::AscNodePtr;

constexpr const char *kStoreType = Store::Type;
constexpr const char *kScalarType = Scalar::Type;
constexpr const char *kBroadcastType = Broadcast::Type;
constexpr const char *kCastType = Cast::Type;

std::vector<std::string> view_op_type = {Transpose::Type, Broadcast::Type, "Slice", Split::Type, Concat::Type,
                                         Gather::Type,    "Sum",           "Mean",  "Max",       "Min",
                                         "Prod",          "Any",           "All"};

// -------------------- compat shim（替代 GE asc_adapt:: / BackendUtils:: / AutofuseUtils::） --------------------

AscNodePtr ToAscNode(const af::NodePtr &node) {
  return std::dynamic_pointer_cast<AscNode>(node);
}

bool IsEqOne(const Expression &expr) {
  return af::SymbolicUtils::StaticCheckEq(expr, af::sym::kSymbolOne) == af::TriBool::kTrue;
}

bool IsEqZero(const Expression &expr) {
  return af::SymbolicUtils::StaticCheckEq(expr, af::sym::kSymbolZero) == af::TriBool::kTrue;
}

std::string VectorToStr(const std::vector<bool> &vec) {
  std::string str = "[";
  for (size_t i = 0U; i < vec.size(); ++i) {
    if (i > 0U) {
      str += ", ";
    }
    str += vec[i] ? "true" : "false";
  }
  str += "]";
  return str;
}

Status GetPeerInNodes(const NodePtr &node, std::vector<NodePtr> &vec, int32_t idx) {
  auto out_anchor = node->GetOutDataAnchor(idx);
  GE_ASSERT_NOTNULL(out_anchor);
  for (const auto &peer_in : out_anchor->GetPeerInDataAnchors()) {
    GE_ASSERT_NOTNULL(peer_in);
    vec.push_back(ToAscNode(peer_in->GetOwnerNode()));
  }
  return SUCCESS;
}

Status GetPeerOutNode(const NodePtr &node, NodePtr &peer, int32_t idx) {
  auto in_anchor = node->GetInDataAnchor(idx);
  GE_ASSERT_NOTNULL(in_anchor);
  auto out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(out_anchor);
  peer = ToAscNode(out_anchor->GetOwnerNode());
  return SUCCESS;
}

Status GetPeerOutNodes(const NodePtr &node, std::vector<NodePtr> &vec) {
  auto in_size = node->GetAllInDataAnchorsSize();
  for (uint32_t i = 0U; i < in_size; ++i) {
    auto in_anchor = node->GetInDataAnchor(static_cast<int32_t>(i));
    if (in_anchor == nullptr) {
      continue;
    }
    auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }
    vec.push_back(ToAscNode(out_anchor->GetOwnerNode()));
  }
  return SUCCESS;
}

Status GetOutputTensorAttr(const NodePtr &node, AscTensorAttr *&attr) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(node->GetAllOutDataAnchorsSize() > 0U);
  attr = &node->outputs[0].attr;
  return SUCCESS;
}

bool IsSingleInAndOutNode(const NodePtr &node) {
  return node->GetInDataNodesSize() == 1UL && node->GetOutDataNodesSize() == 1UL;
}

bool IsSingleInNode(const NodePtr &node) {
  return node->GetInDataNodesSize() == 1UL;
}

bool IsSingleOutNode(const NodePtr &node) {
  if (node->GetAllOutDataAnchorsSize() != 1U) {
    return false;
  }
  auto out_anchor = node->GetOutDataAnchor(0);
  if (out_anchor == nullptr) {
    return false;
  }
  return out_anchor->GetPeerInDataAnchors().size() == 1U;
}

bool HasNoControlEdges(const NodePtr &node) {
  return node != nullptr && node->GetInControlNodesSize() == 0U && node->GetOutControlNodesSize() == 0U;
}

bool IsIndirectLoadBoundary(const NodePtr &node) {
  if (IsOps<af::ascir_op::IndirectLoad>(node)) {
    return true;
  }
  auto out_anchor_size = node->GetAllOutDataAnchorsSize();
  for (uint32_t i = 0U; i < out_anchor_size; ++i) {
    std::vector<NodePtr> peer_in_nodes;
    if (GetPeerInNodes(node, peer_in_nodes, static_cast<int32_t>(i)) == SUCCESS) {
      for (const auto &peer_in_node : peer_in_nodes) {
        if (peer_in_node != nullptr && IsOps<af::ascir_op::IndirectLoad>(peer_in_node)) {
          return true;
        }
      }
    }
  }
  return false;
}

void RemoveDuplicates(std::vector<NodePtr> &vec) {
  std::unordered_set<NodePtr> seen;
  seen.reserve(vec.size());
  vec.erase(std::remove_if(vec.begin(), vec.end(), [&seen](const NodePtr &node) { return !seen.insert(node).second; }),
            vec.end());
}

// -------------------- 算法逻辑（移植自 GE broadcast_backward_pass.cpp） --------------------

Status GetSingleNextNode(NodePtr &node, NodePtr &peer_in_node) {
  std::vector<NodePtr> peer_in_nodes;
  GE_ASSERT_SUCCESS(GetPeerInNodes(node, peer_in_nodes, 0));

  if (peer_in_nodes.size() != 1U) {
    GELOGI("node:%s(%s) has %zu peer out nodes", node->GetName().c_str(), node->GetType().c_str(),
           peer_in_nodes.size());
    return FAILED;
  }
  peer_in_node = peer_in_nodes.at(0);
  return SUCCESS;
}

Status GetPeerOutNodeSafe(const NodePtr &node, NodePtr &peer_out_node, int32_t idx) {
  GE_ASSERT_NOTNULL(node);

  if (node->GetAllInDataAnchorsSize() <= static_cast<size_t>(idx)) {
    return FAILED;
  }

  auto in_anchor = node->GetInDataAnchor(idx);
  GE_ASSERT_NOTNULL(in_anchor);

  auto out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(out_anchor);

  return GetPeerOutNode(node, peer_out_node, idx);
}

bool IsNextViewOp(const NodePtr &next_node) {
  std::string type = next_node->GetType();
  return std::find(view_op_type.begin(), view_op_type.end(), type) != view_op_type.end();
}

bool IsDtypeNotSupportOp(const NodePtr &next_node, af::DataType &output_dtype) {
  std::vector<af::DataType> input_dtypes;
  std::vector<af::DataType> expect_output_dtypes;
  const auto output_tensor_desc = next_node->GetOpDesc()->MutableOutputDesc(0);
  output_dtype = output_tensor_desc->GetDataType();
  expect_output_dtypes.push_back(output_dtype);
  input_dtypes.push_back(output_dtype);
  return (next_node->GetType() == kCastType) &&
         (ScheduleUtils::CallAscirInferDataType<Broadcast>(input_dtypes, expect_output_dtypes) != SUCCESS);
}

Status ReverseCollectBrcNodes(const NodePtr &node, std::vector<NodePtr> &bro_nodes) {
  NodePtr cur_node = node;
  while ((cur_node->GetType() == kBroadcastType) && IsSingleInAndOutNode(cur_node)) {
    bro_nodes.push_back(cur_node);
    GE_ASSERT_SUCCESS(GetPeerOutNodeSafe(cur_node, cur_node, 0));
  }
  return SUCCESS;
}

Status GetBroAxisFromNode(const NodePtr &bro_node, int64_t &bro_axis) {
  bro_axis = -1;
  NodePtr pre_bro_node;
  GE_ASSERT_SUCCESS(GetPeerOutNodeSafe(bro_node, pre_bro_node, 0));
  AscTensorAttr *pre_bro_output_attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(pre_bro_node, pre_bro_output_attr));
  auto pre_bro_repeats = pre_bro_output_attr->repeats;
  auto pre_bro_strides = pre_bro_output_attr->strides;
  auto pre_bro_axis = pre_bro_output_attr->axis;

  AscTensorAttr *bro_output_attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(bro_node, bro_output_attr));
  auto bro_repeats = bro_output_attr->repeats;
  auto bro_strides = bro_output_attr->strides;
  auto bro_attr_axis = bro_output_attr->axis;
  GE_ASSERT_TRUE(pre_bro_repeats.size() == pre_bro_strides.size());
  GE_ASSERT_TRUE(bro_repeats.size() == bro_attr_axis.size());
  GE_ASSERT_TRUE(bro_strides.size() == bro_attr_axis.size());
  GE_ASSERT_TRUE(pre_bro_repeats.size() == pre_bro_axis.size());
  for (size_t index = 0U; index < bro_attr_axis.size(); index++) {
    if (IsEqOne(bro_repeats[index])) {
      continue;
    }

    const auto pre_axis_iter = std::find(pre_bro_axis.begin(), pre_bro_axis.end(), bro_attr_axis[index]);
    if (pre_axis_iter == pre_bro_axis.end()) {
      // A missing input axis is a scalar/implicit broadcast dimension.
      bro_axis = bro_attr_axis[index];
      return SUCCESS;
    }

    const size_t pre_index = static_cast<size_t>(std::distance(pre_bro_axis.begin(), pre_axis_iter));
    if (IsEqOne(pre_bro_repeats[pre_index]) && IsEqZero(pre_bro_strides[pre_index])) {
      bro_axis = bro_attr_axis[index];
      return SUCCESS;
    }
  }
  GELOGW("Cannot infer broadcast axis: broadcast[%s], source[%s].", bro_node->GetName().c_str(),
         pre_bro_node->GetName().c_str());
  return FAILED;
}

Status GetBroAxises(const std::vector<NodePtr> &bro_nodes, std::vector<int64_t> &bro_axis_idx) {
  for (const auto &bro_node : bro_nodes) {
    int64_t bro_axis = -1;
    if (GetBroAxisFromNode(bro_node, bro_axis) != SUCCESS) {
      GELOGI("GetBroAxisFromNode failed for node %s(%s), skipping.", bro_node->GetName().c_str(),
             bro_node->GetType().c_str());
      continue;
    }
    if (bro_axis >= 0) {
      bro_axis_idx.push_back(bro_axis);
    }
  }
  return SUCCESS;
}

Status GetBroAxisesIndex(std::vector<size_t> &bro_axis_idx, const std::vector<Expression> &pre_bro_repeats,
                         const std::vector<Expression> &pre_bro_strides,
                         const std::vector<Expression> &last_bro_repeats) {
  GE_ASSERT_TRUE(pre_bro_repeats.size() == pre_bro_strides.size());
  GE_ASSERT_TRUE(pre_bro_repeats.size() == last_bro_repeats.size());
  for (size_t index = 0U; index < pre_bro_repeats.size(); index++) {
    if (IsEqOne(pre_bro_repeats[index]) && IsEqZero(pre_bro_strides[index])) {
      if (IsEqOne(last_bro_repeats[index])) {
        continue;
      }
      bro_axis_idx.push_back(index);
    }
  }
  return SUCCESS;
}

bool IsSameBroNodes(const std::vector<NodePtr> &bro_nodes1, const std::vector<NodePtr> &bro_nodes2) {
  if (bro_nodes1.size() != bro_nodes2.size()) {
    return false;
  }
  std::vector<int64_t> bro_axis_idx1;
  std::vector<int64_t> bro_axis_idx2;
  GetBroAxises(bro_nodes1, bro_axis_idx1);
  GetBroAxises(bro_nodes2, bro_axis_idx2);
  return bro_axis_idx1 == bro_axis_idx2;
}

Status RemoveAndRelinkNodeEdge(af::InDataAnchorPtr &bro_in_anchor, af::OutDataAnchorPtr &bro_out_anchor) {
  GE_ASSERT_NOTNULL(bro_in_anchor);
  GE_ASSERT_NOTNULL(bro_out_anchor);
  GE_ASSERT_TRUE(!bro_out_anchor->GetPeerInDataAnchors().empty());
  auto before_bro_out_anchor = bro_in_anchor->GetPeerOutAnchor();
  auto after_bro_in_anchor = bro_out_anchor->GetPeerInDataAnchors().at(0);
  GE_ASSERT_NOTNULL(before_bro_out_anchor);
  GE_ASSERT_NOTNULL(after_bro_in_anchor);
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(before_bro_out_anchor, bro_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(bro_out_anchor, after_bro_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(before_bro_out_anchor, after_bro_in_anchor));
  return SUCCESS;
}

Status RemoveBroadcastOneByOne(std::vector<NodePtr> &bro_nodes, AscGraph &graph) {
  for (auto &node : bro_nodes) {
    auto bro_in_anchor = node->GetInDataAnchor(0);
    auto bro_out_anchor = node->GetOutDataAnchor(0);
    GE_ASSERT_SUCCESS(RemoveAndRelinkNodeEdge(bro_in_anchor, bro_out_anchor));
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(af::AscGraphUtils::GetComputeGraph(graph), node));
    af::NodeUtils::UnlinkAll(*node);
  }
  return SUCCESS;
}

Status RemoveBroadcasts(std::vector<NodePtr> &bro_nodes, AscGraph &graph) {
  auto bro_in_anchor = bro_nodes.front()->GetInDataAnchor(0);
  auto bro_out_anchor = bro_nodes.back()->GetOutDataAnchor(0);
  GE_ASSERT_SUCCESS(RemoveAndRelinkNodeEdge(bro_in_anchor, bro_out_anchor));
  for (auto &node : bro_nodes) {
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(af::AscGraphUtils::GetComputeGraph(graph), node));
    af::NodeUtils::UnlinkAll(*node);
  }
  return SUCCESS;
}

std::set<int64_t> FindSubSet(std::vector<int64_t> &bro_axis_idx1, std::vector<int64_t> &bro_axis_idx2) {
  std::set<int64_t> common_elements;
  if (bro_axis_idx1.empty() || bro_axis_idx2.empty()) {
    return common_elements;
  }
  std::sort(bro_axis_idx1.begin(), bro_axis_idx1.end(), std::greater<int64_t>());
  std::sort(bro_axis_idx2.begin(), bro_axis_idx2.end(), std::greater<int64_t>());
  auto it1 = bro_axis_idx1.begin();
  auto it2 = bro_axis_idx2.begin();
  while (it1 != bro_axis_idx1.end() && it2 != bro_axis_idx2.end()) {
    if (*it1 == *it2) {
      common_elements.insert(*it1);
      ++it1;
      ++it2;
    } else if (*it1 > *it2) {
      ++it1;
    } else {
      ++it2;
    }
  }
  return common_elements;
}

bool CollectSameBrcAxis(NodePtr &cur_node, NodePtr &next_node, std::vector<std::vector<NodePtr>> &bro_nodes_list,
                        std::set<int64_t> &common_axes, std::vector<NodePtr> &origin_bro_nodes) {
  std::vector<NodePtr> peer_out_nodes;
  GE_ASSERT_SUCCESS(GetPeerOutNodes(next_node, peer_out_nodes));
  for (const auto &node : peer_out_nodes) {
    if ((cur_node != nullptr) && (node == cur_node)) {
      continue;
    }
    if (node->GetType() != kBroadcastType) {
      bro_nodes_list.clear();
      common_axes.clear();
      return false;
    }
    if (origin_bro_nodes.empty()) {
      ReverseCollectBrcNodes(node, origin_bro_nodes);
      std::reverse(origin_bro_nodes.begin(), origin_bro_nodes.end());
      bro_nodes_list.push_back(origin_bro_nodes);
      continue;
    }
    std::vector<NodePtr> temp_bro_nodes;
    ReverseCollectBrcNodes(node, temp_bro_nodes);
    std::reverse(temp_bro_nodes.begin(), temp_bro_nodes.end());
    bro_nodes_list.push_back(temp_bro_nodes);

    std::vector<int64_t> bro_axis_idx1;
    std::vector<int64_t> bro_axis_idx2;
    if (common_axes.empty()) {
      GetBroAxises(origin_bro_nodes, bro_axis_idx1);
    } else {
      bro_axis_idx1.assign(common_axes.begin(), common_axes.end());
    }
    GetBroAxises(temp_bro_nodes, bro_axis_idx2);
    std::set<int64_t> temp_common_axes = FindSubSet(bro_axis_idx1, bro_axis_idx2);

    if (temp_common_axes.empty()) {
      bro_nodes_list.clear();
      common_axes.clear();
      return false;
    }
    common_axes = temp_common_axes;
    origin_bro_nodes = origin_bro_nodes.size() < temp_bro_nodes.size() ? origin_bro_nodes : temp_bro_nodes;
  }
  return true;
}

Status GetNodeScalarInputList(const af::AscNodePtr &asc_node, std::vector<bool> &is_scalar_list) {
  is_scalar_list.resize(asc_node->GetInDataNodesSize(), false);
  for (size_t i = 0UL; i < is_scalar_list.size(); ++i) {
    const std::vector<Expression> repeats = asc_node->inputs[i].attr.repeats;
    is_scalar_list[i] = ascgen_utils::IsScalarInput(repeats);
  }
  return SUCCESS;
}

Status ProcessOtherInputBranches(const NodePtr &next_comp_op, size_t current_idx, const std::vector<int64_t> &bro_axes,
                                 std::vector<bool> &is_scalar_list);

bool CheckBackwardCommon(const NodePtr &next_node) {
  if (!HasNoControlEdges(next_node)) {
    return false;
  }
  if (next_node->GetType() == kStoreType) {
    return false;
  }
  // IndirectLoad 的输出轴属于独立的物理视图，Broadcast 后移不能跨过该边界。
  // 直接喂给 IndirectLoad 的 compute 同样不能后移，否则 broadcast 会紧贴 IL。
  if (IsIndirectLoadBoundary(next_node)) {
    return false;
  }
  if (ScheduleUtils::IsRemovePad(next_node)) {
    return false;
  }
  if (IsNextViewOp(next_node)) {
    return false;
  }
  af::DataType output_dtype;
  if (IsDtypeNotSupportOp(next_node, output_dtype)) {
    GELOGI("Node %s(%s) cannot backward with dtype(%s)", next_node->GetName().c_str(), next_node->GetType().c_str(),
           af::TypeUtils::DataTypeToSerialString(output_dtype).c_str());
    return false;
  }
  return true;
}

bool CanBackwardSimplified(const NodePtr &next_node) {
  if (!CheckBackwardCommon(next_node)) {
    return false;
  }
  if (next_node->GetAllOutDataAnchorsSize() > 1U) {
    return false;
  }
  if (!IsSingleInNode(next_node)) {
    return false;
  }
  return true;
}

bool IsScalarInput(const NodePtr &input_node) {
  NodePtr temp_node = input_node;
  while (temp_node != nullptr) {
    if (temp_node->GetType() == kScalarType) {
      return true;
    }
    if (temp_node->GetType() != kBroadcastType) {
      break;
    }
    NodePtr pre_node;
    GE_ASSERT_SUCCESS(GetPeerOutNodeSafe(temp_node, pre_node, 0));
    temp_node = pre_node;
  }
  return false;
}

Status CheckNodeSupportsScalarInput(const NodePtr &compute_node, int32_t input_idx,
                                    const std::vector<int64_t> &bro_axes, bool &is_support) {
  is_support = false;
  GE_ASSERT_NOTNULL(std::dynamic_pointer_cast<AscNode>(compute_node));
  const auto &asc_node = std::dynamic_pointer_cast<AscNode>(compute_node);

  std::vector<bool> is_scalar_list;
  GE_ASSERT_SUCCESS(GetNodeScalarInputList(asc_node, is_scalar_list));

  if (input_idx >= 0 && static_cast<size_t>(input_idx) < is_scalar_list.size()) {
    is_scalar_list[input_idx] = true;
  }

  GE_ASSERT_SUCCESS(ProcessOtherInputBranches(compute_node, input_idx, bro_axes, is_scalar_list));

  is_support = ascgen_utils::IsNodeSupportsScalarInput(asc_node, is_scalar_list);
  if (!is_support) {
    GELOGD("Compute node %s does not support scalar input, is_scalar_list: %s", compute_node->GetName().c_str(),
           VectorToStr(is_scalar_list).c_str());
  }
  return SUCCESS;
}

bool CheckScalarInputSupport(const NodePtr &next_node, const std::vector<NodePtr> &bro_nodes) {
  GE_ASSERT_NOTNULL(std::dynamic_pointer_cast<AscNode>(next_node));

  auto in_data_anchor_size = next_node->GetAllInDataAnchorsSize();
  for (uint32_t i = 0U; i < in_data_anchor_size; ++i) {
    NodePtr input_node;
    GE_ASSERT_SUCCESS(GetPeerOutNodeSafe(next_node, input_node, i));

    if (IsScalarInput(input_node)) {
      std::vector<int64_t> bro_axes;
      if (!bro_nodes.empty()) {
        GE_ASSERT_SUCCESS(GetBroAxises(bro_nodes, bro_axes));
      }
      bool is_support = false;
      GE_ASSERT_SUCCESS(CheckNodeSupportsScalarInput(next_node, i, bro_axes, is_support));
      if (!is_support) {
        return false;
      }
    }
  }
  return true;
}

bool IsMulInputsCanBackward(NodePtr &cur_node, NodePtr &next_node, std::vector<NodePtr> &bro_nodes, AscGraph &graph,
                            std::set<NodePtr> &mul_input_nodes) {
  auto in_data_anchor_size = next_node->GetAllInDataAnchorsSize();
  if (in_data_anchor_size == 1U) {
    return false;
  }

  std::vector<NodePtr> peer_out_nodes;
  GE_ASSERT_SUCCESS(GetPeerOutNodes(next_node, peer_out_nodes));
  std::vector<std::vector<NodePtr>> remove_bro_nodes_list;
  for (const auto &node : peer_out_nodes) {
    if (node == cur_node) {
      continue;
    }
    if (node->GetType() != kBroadcastType) {
      return false;
    }

    std::vector<NodePtr> temp_bro_nodes;
    GE_ASSERT_SUCCESS(ReverseCollectBrcNodes(node, temp_bro_nodes));
    std::reverse(temp_bro_nodes.begin(), temp_bro_nodes.end());
    if (!IsSameBroNodes(bro_nodes, temp_bro_nodes)) {
      std::vector<std::vector<NodePtr>> bro_nodes_list;
      std::set<int64_t> common_axes;
      std::vector<NodePtr> origin_bro_nodes;
      origin_bro_nodes.assign(bro_nodes.begin(), bro_nodes.end());
      bro_nodes_list.push_back(origin_bro_nodes);
      if (CollectSameBrcAxis(cur_node, next_node, bro_nodes_list, common_axes, origin_bro_nodes)) {
        mul_input_nodes.insert(next_node);
      }
      return false;
    }
    remove_bro_nodes_list.push_back(temp_bro_nodes);
  }

  if (!CheckScalarInputSupport(next_node, bro_nodes)) {
    return false;
  }

  for (std::vector<NodePtr> &remove_nodes : remove_bro_nodes_list) {
    RemoveBroadcasts(remove_nodes, graph);
  }
  return true;
}

bool CanBackward(NodePtr &cur_node, NodePtr &next_node, std::vector<NodePtr> &bro_nodes, AscGraph &graph,
                 std::set<NodePtr> &mul_input_nodes) {
  if (!CheckBackwardCommon(next_node)) {
    return false;
  }
  if (!IsSingleOutNode(next_node)) {
    return false;
  }
  if (!IsSingleInNode(next_node) && !IsMulInputsCanBackward(cur_node, next_node, bro_nodes, graph, mul_input_nodes)) {
    return false;
  }
  return true;
}

Status CollectBroNodes(NodePtr &cur_node, NodePtr &next_node, std::vector<NodePtr> &nodes) {
  if (cur_node->GetType() == kBroadcastType && !HasNoControlEdges(cur_node)) {
    return SUCCESS;
  }
  if (IsSingleInAndOutNode(cur_node) && (cur_node->GetType() == kBroadcastType)) {
    nodes.push_back(cur_node);
  }
  while (IsSingleInAndOutNode(next_node) && (next_node->GetType() == kBroadcastType)) {
    if (!HasNoControlEdges(next_node)) {
      return SUCCESS;
    }
    nodes.push_back(next_node);
    cur_node = next_node;
    GE_ASSERT_SUCCESS(GetSingleNextNode(cur_node, next_node));
  }
  return SUCCESS;
}

Status CollectCmpNodes(NodePtr &cur_node, NodePtr &next_node, std::vector<NodePtr> &nodes,
                       std::vector<NodePtr> &bro_nodes, AscGraph &graph, std::set<NodePtr> &mul_input_nodes) {
  while (CanBackward(cur_node, next_node, bro_nodes, graph, mul_input_nodes)) {
    nodes.push_back(next_node);
    cur_node = next_node;
    GE_ASSERT_SUCCESS(GetSingleNextNode(cur_node, next_node));
  }
  return SUCCESS;
}

Status ReorderBroadcasts(std::vector<NodePtr> &compute_nodes, std::vector<NodePtr> &bro_nodes) {
  auto bro_in_anchor = bro_nodes.front()->GetInDataAnchor(0);
  auto bro_out_anchor = bro_nodes.back()->GetOutDataAnchor(0);
  auto comp_out_anchor = compute_nodes.back()->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(bro_in_anchor);
  GE_ASSERT_NOTNULL(bro_out_anchor);
  GE_ASSERT_NOTNULL(comp_out_anchor);
  GE_ASSERT_TRUE(!comp_out_anchor->GetPeerInDataAnchors().empty());
  GE_ASSERT_TRUE(!bro_out_anchor->GetPeerInDataAnchors().empty());
  auto before_bro_out_anchor = bro_in_anchor->GetPeerOutAnchor();
  auto after_comp_in_anchor = comp_out_anchor->GetPeerInDataAnchors().at(0);
  auto comp_in_anchor = bro_out_anchor->GetPeerInDataAnchors().at(0);
  GE_ASSERT_NOTNULL(before_bro_out_anchor);
  GE_ASSERT_NOTNULL(after_comp_in_anchor);
  GE_ASSERT_NOTNULL(comp_in_anchor);

  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(before_bro_out_anchor, bro_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(bro_out_anchor, comp_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(comp_out_anchor, after_comp_in_anchor));

  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(before_bro_out_anchor, comp_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(comp_out_anchor, bro_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(bro_out_anchor, after_comp_in_anchor));
  return SUCCESS;
}

Status UpdateComputeNodesAscTensorAttr(std::vector<NodePtr> &bro_nodes, std::vector<NodePtr> &compute_nodes,
                                       const NodePtr &pre_bro_node) {
  AscTensorAttr *pre_bro_output_attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(pre_bro_node, pre_bro_output_attr));

  AscTensorAttr *last_bro_output_attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(bro_nodes.back(), last_bro_output_attr));

  std::vector<size_t> bro_axis_idx;
  auto pre_bro_axis = pre_bro_output_attr->axis;
  auto pre_bro_repeats = pre_bro_output_attr->repeats;
  auto pre_bro_strides = pre_bro_output_attr->strides;
  auto last_bro_repeats = last_bro_output_attr->repeats;
  GE_ASSERT_SUCCESS(GetBroAxisesIndex(bro_axis_idx, pre_bro_repeats, pre_bro_strides, last_bro_repeats));
  GELOGD("Broadcast chain contains %zu broadcast axes.", bro_axis_idx.size());
  for (const auto &compute_node : compute_nodes) {
    AscTensorAttr *compute_output_attr = nullptr;
    GE_ASSERT_SUCCESS(GetOutputTensorAttr(compute_node, compute_output_attr));
    compute_output_attr->axis = last_bro_output_attr->axis;
    compute_output_attr->repeats = pre_bro_repeats;
    compute_output_attr->strides = pre_bro_strides;
    GE_ASSERT_TRUE(compute_output_attr->strides.size() > 0U);
    auto it = std::find(bro_axis_idx.begin(), bro_axis_idx.end(), pre_bro_strides.size() - 1U);
    if (it != bro_axis_idx.end()) {
      compute_output_attr->strides[compute_output_attr->strides.size() - 1U] = af::sym::kSymbolZero;
    }
    if (pre_bro_node->GetType() != kScalarType) {
      GE_ASSERT_SUCCESS(
          ScheduleUtils::RecalculateStridesFromRepeats(compute_output_attr->repeats, compute_output_attr->strides));
    }
  }
  return SUCCESS;
}

Status UpdateBroadcastNodesDataType(std::vector<NodePtr> &bro_nodes, const NodePtr &last_comp_node) {
  const auto last_comp_opdesc = last_comp_node->GetOpDesc();
  GE_ASSERT_NOTNULL(last_comp_opdesc);
  const auto comp_output_tensor_desc = last_comp_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(comp_output_tensor_desc);
  auto last_comp_dtype = comp_output_tensor_desc->GetDataType();
  for (const auto &bro_node : bro_nodes) {
    const auto bro_opdesc = bro_node->GetOpDesc();
    GE_ASSERT_NOTNULL(bro_opdesc);
    const auto bro_output_tensor_desc = bro_opdesc->MutableOutputDesc(0);
    GE_ASSERT_NOTNULL(bro_output_tensor_desc);
    bro_output_tensor_desc->SetDataType(last_comp_dtype);
  }
  return SUCCESS;
}

Status BroadcastBackwardReally(std::vector<NodePtr> &compute_nodes, std::vector<NodePtr> &bro_nodes,
                               const NodePtr &pre_bro_node) {
  GE_ASSERT_SUCCESS(ReorderBroadcasts(compute_nodes, bro_nodes));
  GE_ASSERT_SUCCESS(UpdateComputeNodesAscTensorAttr(bro_nodes, compute_nodes, pre_bro_node));
  GE_ASSERT_SUCCESS(UpdateBroadcastNodesDataType(bro_nodes, compute_nodes.back()));
  return SUCCESS;
}

Status CollectBranchBroadcastNodes(const NodePtr &input_node, std::vector<NodePtr> &branch_bro_nodes) {
  NodePtr temp_node = input_node;
  while (temp_node->GetType() == kBroadcastType && IsSingleInAndOutNode(temp_node)) {
    branch_bro_nodes.push_back(temp_node);
    NodePtr next_temp_node;
    if (GetPeerOutNodeSafe(temp_node, next_temp_node, 0) != SUCCESS) {
      break;
    }
    temp_node = next_temp_node;
  }
  return SUCCESS;
}

bool HasCommonBroadcastAxis(const std::vector<int64_t> &axes1, const std::vector<int64_t> &axes2) {
  for (const auto &axis : axes1) {
    if (std::find(axes2.begin(), axes2.end(), axis) != axes2.end()) {
      return true;
    }
  }
  return false;
}

NodePtr GetPreBroadcastNode(const NodePtr &branch_bro_node) {
  auto bro_in_anchor = branch_bro_node->GetInDataAnchor(0);
  if (bro_in_anchor == nullptr) {
    return nullptr;
  }
  auto peer_out_anchor = bro_in_anchor->GetPeerOutAnchor();
  if (peer_out_anchor == nullptr) {
    return nullptr;
  }
  return ToAscNode(peer_out_anchor->GetOwnerNode());
}

Status ProcessSingleInputBranch(const NodePtr &input_node, const std::vector<int64_t> &bro_axes, bool &is_scalar) {
  std::vector<NodePtr> branch_bro_nodes;
  GE_ASSERT_SUCCESS(CollectBranchBroadcastNodes(input_node, branch_bro_nodes));

  if (!branch_bro_nodes.empty()) {
    std::vector<int64_t> branch_bro_axes;
    GE_ASSERT_SUCCESS(GetBroAxises(branch_bro_nodes, branch_bro_axes));
    if (HasCommonBroadcastAxis(bro_axes, branch_bro_axes)) {
      NodePtr pre_bro_node = GetPreBroadcastNode(branch_bro_nodes.front());
      if (pre_bro_node != nullptr) {
        AscTensorAttr *pre_bro_attr = nullptr;
        if (GetOutputTensorAttr(pre_bro_node, pre_bro_attr) == SUCCESS) {
          const std::vector<Expression> repeats = pre_bro_attr->repeats;
          is_scalar = ascgen_utils::IsScalarInput(repeats);
        }
      }
    }
  }
  return SUCCESS;
}

Status ProcessOtherInputBranches(const NodePtr &next_comp_op, size_t current_idx, const std::vector<int64_t> &bro_axes,
                                 std::vector<bool> &is_scalar_list) {
  for (size_t i = 0; i < is_scalar_list.size(); ++i) {
    if (i == current_idx) {
      continue;
    }
    NodePtr input_node;
    if (GetPeerOutNodeSafe(next_comp_op, input_node, static_cast<int32_t>(i)) != SUCCESS) {
      continue;
    }
    bool scalar_flag = is_scalar_list[i];
    GE_ASSERT_SUCCESS(ProcessSingleInputBranch(input_node, bro_axes, scalar_flag));
    is_scalar_list[i] = scalar_flag;
  }
  return SUCCESS;
}

Status JudgeNextCompOpSupportsScalarInput(const NodePtr &node, bool &is_next_support_scalar) {
  is_next_support_scalar = false;

  GE_ASSERT_TRUE(node->GetAllOutDataAnchorsSize() == 1U);
  auto out_anchor = node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out_anchor);

  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  GE_ASSERT_TRUE(!peer_in_anchors.empty());

  bool all_branches_support_scalar = true;
  for (const auto &peer_in_anchor : peer_in_anchors) {
    NodePtr branch_start_node = ToAscNode(peer_in_anchor->GetOwnerNode());
    NodePtr cur_node = branch_start_node;

    std::vector<NodePtr> bro_nodes;
    NodePtr temp_cur_node = branch_start_node;
    NodePtr temp_next_node = branch_start_node;
    GE_ASSERT_SUCCESS(CollectBroNodes(temp_cur_node, temp_next_node, bro_nodes));

    NodePtr compute_node = temp_next_node;
    if (compute_node == nullptr || bro_nodes.empty()) {
      all_branches_support_scalar = false;
      break;
    }

    bool is_support = false;
    std::vector<int64_t> bro_axes;
    if (!bro_nodes.empty()) {
      GE_ASSERT_SUCCESS(GetBroAxises(bro_nodes, bro_axes));
    }
    GE_ASSERT_SUCCESS(CheckNodeSupportsScalarInput(compute_node, peer_in_anchor->GetIdx(), bro_axes, is_support));
    if (!is_support) {
      all_branches_support_scalar = false;
      break;
    }
  }

  is_next_support_scalar = all_branches_support_scalar;
  return SUCCESS;
}

bool ContainsBroadcastNode(const NodePtr &node) {
  // Graph input nodes (for example Data) legitimately have no predecessor.
  // Check the anchor before calling GetPeerOutNodeSafe(), whose assertion is
  // reserved for paths that require a connected input.
  if (node == nullptr || node->GetAllInDataAnchorsSize() == 0U) {
    return false;
  }
  auto in_anchor = node->GetInDataAnchor(0);
  if (in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr) {
    return false;
  }

  NodePtr pre_node;
  if (GetPeerOutNodeSafe(node, pre_node, 0) != SUCCESS || pre_node == nullptr) {
    return false;
  }
  return pre_node->GetType() == kBroadcastType;
}

Status CollectCandidateMultiRefNodes(const AscGraph &graph, std::vector<NodePtr> &candidate_nodes) {
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetAllOutDataAnchorsSize() != 1U) {
      continue;
    }
    auto out_anchor = node->GetOutDataAnchor(0);
    if (out_anchor == nullptr || out_anchor->GetPeerInDataAnchors().size() <= 1) {
      continue;
    }
    if (node->GetType() == kBroadcastType || ContainsBroadcastNode(node)) {
      candidate_nodes.push_back(node);
    }
  }
  return SUCCESS;
}

Status ExtractBroadcastChainFromNode(const NodePtr &node, std::vector<NodePtr> &bro_nodes) {
  NodePtr cur_node = node;

  if (cur_node != nullptr && cur_node->GetType() != kBroadcastType) {
    NodePtr pre_node;
    GE_ASSERT_SUCCESS(GetPeerOutNodeSafe(cur_node, pre_node, 0));
    cur_node = pre_node;
  }

  while (cur_node != nullptr && cur_node->GetType() == kBroadcastType) {
    bro_nodes.push_back(cur_node);
    NodePtr pre_node;
    if (GetPeerOutNodeSafe(cur_node, pre_node, 0) != SUCCESS) {
      break;
    }
    cur_node = pre_node;
  }

  std::reverse(bro_nodes.begin(), bro_nodes.end());
  return SUCCESS;
}

Status TraceBranchToMergeNode(const NodePtr &start_node, NodePtr &merge_node, std::vector<NodePtr> &branch_nodes) {
  NodePtr cur_node = start_node;
  std::unordered_set<NodePtr> visited_nodes;

  while (cur_node != nullptr) {
    GE_ASSERT_TRUE(visited_nodes.count(cur_node) == 0, "Found cycle dependency in TraceBranchToMergeNode, node: %s",
                   cur_node->GetName().c_str());
    visited_nodes.insert(cur_node);

    if (!IsSingleInNode(cur_node)) {
      merge_node = cur_node;
      return SUCCESS;
    }

    if (cur_node->GetType() == kStoreType) {
      merge_node = cur_node;
      return SUCCESS;
    }

    if (!IsSingleOutNode(cur_node)) {
      return FAILED;
    }

    branch_nodes.push_back(cur_node);

    NodePtr next_node;
    if (GetSingleNextNode(cur_node, next_node) != SUCCESS) {
      return FAILED;
    }
    cur_node = next_node;
  }
  return FAILED;
}

bool CheckBranchNodesSupportBackward(const std::vector<NodePtr> &branch_nodes) {
  for (const auto &next_node : branch_nodes) {
    if (!CanBackwardSimplified(next_node)) {
      return false;
    }
  }
  return true;
}

bool CheckAllBranchesCanBackward(const NodePtr &multi_ref_node, NodePtr &merge_node,
                                 std::vector<std::vector<NodePtr>> &all_branch_nodes) {
  auto out_anchor = multi_ref_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out_anchor);
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();

  GE_ASSERT_TRUE(peer_in_anchors.size() > 1U);
  NodePtr first_merge_node = nullptr;

  for (const auto &in_anchor : peer_in_anchors) {
    NodePtr branch_start_node = ToAscNode(in_anchor->GetOwnerNode());
    NodePtr current_merge_node = nullptr;
    std::vector<NodePtr> branch_nodes;

    if (TraceBranchToMergeNode(branch_start_node, current_merge_node, branch_nodes) != SUCCESS) {
      return false;
    }

    if (first_merge_node == nullptr) {
      first_merge_node = current_merge_node;
    } else if (first_merge_node != current_merge_node) {
      return false;
    }
    all_branch_nodes.push_back(branch_nodes);
  }

  merge_node = first_merge_node;
  return true;
}

bool CheckAllBranchesSupportBackward(const std::vector<std::vector<NodePtr>> &all_branch_nodes,
                                     const NodePtr &candidate_node) {
  if (!HasNoControlEdges(candidate_node)) {
    return false;
  }
  if (candidate_node->GetType() != kBroadcastType) {
    if (!CanBackwardSimplified(candidate_node)) {
      return false;
    }
  }
  for (const auto &branch_nodes : all_branch_nodes) {
    if (!CheckBranchNodesSupportBackward(branch_nodes)) {
      return false;
    }
  }
  return true;
}

Status DisconnectBranchesFromBroadcast(const NodePtr &last_bro_node,
                                       std::vector<af::OutDataAnchorPtr> &branch_out_anchors,
                                       std::vector<af::InDataAnchorPtr> &branch_in_anchors) {
  auto bro_out_anchor = last_bro_node->GetOutDataAnchor(0);
  auto peer_in_anchors = bro_out_anchor->GetPeerInDataAnchors();

  for (const auto &in_anchor : peer_in_anchors) {
    auto branch_node = in_anchor->GetOwnerNode();
    auto branch_in_anchor = branch_node->GetInDataAnchor(in_anchor->GetIdx());
    auto branch_out_anchor = branch_in_anchor->GetPeerOutAnchor();

    branch_out_anchors.push_back(branch_out_anchor);
    branch_in_anchors.push_back(branch_in_anchor);

    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(branch_out_anchor, branch_in_anchor));
  }
  return SUCCESS;
}

Status MoveBroadcastAfterMerge(const NodePtr &merge_node, const NodePtr &first_bro_node, const NodePtr &last_bro_node) {
  auto merge_out_anchor = merge_node->GetOutDataAnchor(0);
  if (merge_out_anchor == nullptr || merge_out_anchor->GetPeerInDataAnchors().empty()) {
    return FAILED;
  }
  auto peer_in_anchors = merge_out_anchor->GetPeerInDataAnchors();
  std::vector<af::InDataAnchorPtr> merge_next_in_anchors(peer_in_anchors.begin(), peer_in_anchors.end());
  for (const auto &in_anchor : merge_next_in_anchors) {
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(merge_out_anchor, in_anchor));
  }

  auto bro_in_anchor = first_bro_node->GetInDataAnchor(0);
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(merge_out_anchor, bro_in_anchor));

  auto bro_out_anchor = last_bro_node->GetOutDataAnchor(0);
  for (const auto &in_anchor : merge_next_in_anchors) {
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(bro_out_anchor, in_anchor));
  }
  return SUCCESS;
}

Status BackwardMultiRefBroadcast(const NodePtr &candidate_node, const NodePtr &merge_node,
                                 const std::vector<std::vector<NodePtr>> &all_branch_nodes,
                                 std::vector<NodePtr> &bro_nodes, [[maybe_unused]] AscGraph &graph) {
  auto bro_in_anchor = bro_nodes.front()->GetInDataAnchor(0);
  auto pre_bro_out_anchor = bro_in_anchor->GetPeerOutAnchor();
  NodePtr pre_bro_node = ToAscNode(pre_bro_out_anchor->GetOwnerNode());
  bool is_pre_scalar = (pre_bro_node->GetType() == kScalarType);

  if (is_pre_scalar) {
    std::vector<int64_t> bro_axes;
    if (!bro_nodes.empty()) {
      GE_ASSERT_SUCCESS(GetBroAxises(bro_nodes, bro_axes));
    }
    auto bro_out_anchor = bro_nodes.back()->GetOutDataAnchor(0);
    auto peer_in_anchors = bro_out_anchor->GetPeerInDataAnchors();
    for (const auto &in_anchor : peer_in_anchors) {
      NodePtr compute_node = ToAscNode(in_anchor->GetOwnerNode());
      int32_t input_idx = in_anchor->GetIdx();
      bool is_support = false;
      GE_ASSERT_SUCCESS(CheckNodeSupportsScalarInput(compute_node, input_idx, bro_axes, is_support));
      if (!is_support) {
        return FAILED;
      }
    }
  }

  std::vector<af::OutDataAnchorPtr> branch_out_anchors;
  std::vector<af::InDataAnchorPtr> branch_in_anchors;
  GE_ASSERT_SUCCESS(DisconnectBranchesFromBroadcast(bro_nodes.back(), branch_out_anchors, branch_in_anchors));

  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(pre_bro_out_anchor, bro_in_anchor));
  GE_ASSERT_SUCCESS(MoveBroadcastAfterMerge(merge_node, bro_nodes.front(), bro_nodes.back()));

  for (size_t i = 0; i < branch_out_anchors.size(); ++i) {
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(pre_bro_out_anchor, branch_in_anchors[i]));
  }

  std::vector<NodePtr> compute_nodes;
  if (candidate_node->GetType() != kBroadcastType) {
    compute_nodes.push_back(candidate_node);
  }
  for (const auto &branch : all_branch_nodes) {
    compute_nodes.insert(compute_nodes.end(), branch.begin(), branch.end());
  }
  compute_nodes.push_back(merge_node);
  GE_ASSERT_SUCCESS(UpdateComputeNodesAscTensorAttr(bro_nodes, compute_nodes, pre_bro_node));

  if (!compute_nodes.empty()) {
    GE_ASSERT_SUCCESS(UpdateBroadcastNodesDataType(bro_nodes, compute_nodes.back()));
  }
  return SUCCESS;
}

Status ProcessMultiRefBroadcastBackward(AscGraph &graph, bool &is_changed) {
  std::vector<NodePtr> candidate_nodes;
  GE_ASSERT_SUCCESS(CollectCandidateMultiRefNodes(graph, candidate_nodes));

  for (const auto &candidate_node : candidate_nodes) {
    std::vector<NodePtr> bro_nodes;
    GE_ASSERT_SUCCESS(ExtractBroadcastChainFromNode(candidate_node, bro_nodes));

    if (bro_nodes.empty()) {
      continue;
    }

    if (std::any_of(bro_nodes.begin(), bro_nodes.end(), [](const NodePtr &node) { return !HasNoControlEdges(node); })) {
      continue;
    }

    NodePtr bro_node = bro_nodes.front();
    NodePtr merge_node = nullptr;
    std::vector<std::vector<NodePtr>> all_branch_nodes;

    if (!CheckAllBranchesCanBackward(candidate_node, merge_node, all_branch_nodes)) {
      continue;
    }
    if (!CheckAllBranchesSupportBackward(all_branch_nodes, candidate_node)) {
      continue;
    }
    GELOGI("Move shared broadcast from node[%s] to merge[%s], broadcasts=%zu, branches=%zu.",
           candidate_node->GetName().c_str(), merge_node->GetName().c_str(), bro_nodes.size(), all_branch_nodes.size());

    is_changed = BackwardMultiRefBroadcast(candidate_node, merge_node, all_branch_nodes, bro_nodes, graph) == SUCCESS;
  }
  return SUCCESS;
}

Status CollectBackwardStartNodes(const AscGraph &graph, std::vector<NodePtr> &pre_brc_nodes) {
  for (const auto &node : graph.GetAllNodes()) {
    NodePtr cur_node = node;
    while ((cur_node->GetType() == kBroadcastType) && IsSingleOutNode(cur_node)) {
      GE_ASSERT_SUCCESS(GetPeerOutNodeSafe(cur_node, cur_node, 0));
    }
    bool is_next_support_scalar = true;
    if (cur_node->GetType() == kScalarType) {
      GE_ASSERT_SUCCESS(JudgeNextCompOpSupportsScalarInput(cur_node, is_next_support_scalar));
    }

    if ((cur_node != node) && is_next_support_scalar) {
      pre_brc_nodes.push_back(cur_node);
    }
  }
  return SUCCESS;
}

Status CollectBackwardSatisfyStartNodes(const NodePtr &node, std::vector<NodePtr> &peer_in_nodes) {
  auto output_size = node->GetAllOutDataAnchorsSize();
  for (uint32_t idx = 0U; idx < output_size; ++idx) {
    std::vector<NodePtr> temp_nodes;
    GE_ASSERT_SUCCESS(GetPeerInNodes(node, temp_nodes, static_cast<int32_t>(idx)));
    if (!temp_nodes.empty() && (temp_nodes.front()->GetType() == kBroadcastType)) {
      peer_in_nodes.insert(peer_in_nodes.end(), temp_nodes.begin(), temp_nodes.end());
    }
  }
  return SUCCESS;
}

Status RemoveBroadcasts(AscGraph &graph, std::vector<std::vector<NodePtr>> &bro_nodes_list,
                        const std::set<int64_t> &common_axises) {
  for (std::vector<NodePtr> &bro_nodes : bro_nodes_list) {
    std::vector<NodePtr> remove_nodes;
    for (auto it = bro_nodes.begin(); it != bro_nodes.end();) {
      int64_t bro_axis;
      GE_ASSERT_SUCCESS(GetBroAxisFromNode(*it, bro_axis));
      if ((bro_axis != -1) && common_axises.count(bro_axis) != 0) {
        remove_nodes.push_back(*it);
        it = bro_nodes.erase(it);
      } else {
        ++it;
      }
    }
    GE_ASSERT_SUCCESS(RemoveBroadcastOneByOne(remove_nodes, graph));
  }
  return SUCCESS;
}

Status GetBackwardBrcNodes(const std::vector<NodePtr> &origin_bro_nodes,
                           std::vector<NodePtr> &origin_need_move_bro_nodes, const std::set<int64_t> &common_axises) {
  for (auto &bro_node : origin_bro_nodes) {
    int64_t bro_axis;
    GE_ASSERT_SUCCESS(GetBroAxisFromNode(bro_node, bro_axis));
    if ((bro_axis != -1) && common_axises.count(bro_axis) != 0) {
      origin_need_move_bro_nodes.push_back(bro_node);
    }
  }
  return SUCCESS;
}

struct TensorInfo {
  std::vector<int64_t> axis;
  std::vector<Expression> repeats;
  std::vector<Expression> strides;
  af::DataType dtype;
  int64_t sched_axis;
  std::vector<int64_t> broadcast_info;
};

Status GetTensorInfo(const NodePtr &node, TensorInfo &tensor_info) {
  AscTensorAttr *attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(node, attr));
  tensor_info.axis = attr->axis;
  tensor_info.repeats = attr->repeats;
  tensor_info.strides = attr->strides;
  tensor_info.dtype = attr->dtype;
  tensor_info.sched_axis = attr->axis.back();
  return SUCCESS;
}

Status UpdateBroadcastNodeAttrs(const NodePtr &b_node, const std::vector<int64_t> &axis,
                                const std::vector<Expression> &repeats, const std::vector<Expression> &strides,
                                int64_t broadcast_axis) {
  AscTensorAttr *attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(b_node, attr));
  GE_ASSERT_TRUE(axis.size() == repeats.size(), "Broadcast axis/repeats size mismatch: %zu vs %zu", axis.size(),
                 repeats.size());
  GE_ASSERT_TRUE(axis.size() == strides.size(), "Broadcast axis/strides size mismatch: %zu vs %zu", axis.size(),
                 strides.size());
  GE_ASSERT_TRUE(std::find(axis.begin(), axis.end(), broadcast_axis) != axis.end(),
                 "Broadcast axis[%ld] is not present in tensor axes.", broadcast_axis);
  attr->axis = axis;
  attr->repeats = repeats;
  attr->strides = strides;
  // The output of Broadcast keeps the expanded shape. The compact repeat/stride
  // (repeat=1, stride=0) belongs to the input edge and is supplied by the
  // preceding tensor's output attributes.
  GE_ASSERT_SUCCESS(ScheduleUtils::RecalculateStridesFromRepeats(attr->repeats, attr->strides));
  return SUCCESS;
}

Status UpdateBroadcastNodeSchedInfo(const NodePtr &b_node, const NodePtr &ref_node) {
  b_node->attr.sched = ref_node->attr.sched;
  return SUCCESS;
}

Status FromDtypeToOtherDtype(const NodePtr &b_node, af::DataType from_dtype, af::DataType to_dtype) {
  std::vector<af::DataType> input_dtypes = {from_dtype};
  std::vector<af::DataType> expect_output_dtypes = {to_dtype};
  GE_ASSERT_SUCCESS(ScheduleUtils::CallAscirInferDataType<Broadcast>(input_dtypes, expect_output_dtypes));
  b_node->outputs[0].attr.dtype = to_dtype;
  return SUCCESS;
}

Status CreateAndUpdateBroadcastNode(AscGraph &asc_graph, const NodePtr &node, NodePtr &connect_node,
                                    const TensorInfo &tensor_info, const TensorInfo &expanded_tensor_info) {
  const std::vector<int64_t> &broadcast_info = tensor_info.broadcast_info;
  GE_ASSERT_TRUE(broadcast_info.size() > 0U);
  GE_ASSERT_TRUE(tensor_info.axis == expanded_tensor_info.axis);
  GE_ASSERT_TRUE(tensor_info.repeats.size() == tensor_info.axis.size());
  GE_ASSERT_TRUE(expanded_tensor_info.repeats.size() == expanded_tensor_info.axis.size());

  TensorInfo current_tensor_info = tensor_info;
  for (size_t index = 0U; index < broadcast_info.size(); index++) {
    const std::string brc_name = "backward_broadcast_" + node->GetName() + "_" + std::to_string(index);
    Broadcast brc_op(brc_name.c_str());
    auto b_node = asc_graph.AddNode(brc_op);
    GE_ASSERT_NOTNULL(b_node);

    brc_op.attr.sched = node->attr.sched;
    brc_op.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
    brc_op.attr.api.type = af::ApiType::kAPITypeCompute;

    int32_t anchor_idx = 0;
    GE_ASSERT_TRUE(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() == 1U);
    anchor_idx = node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetIdx();
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceNodeDataAnchors(b_node, connect_node, {anchor_idx}, {}));
    GE_ASSERT_GRAPH_SUCCESS(
        af::GraphUtils::AddEdge(b_node->GetOutDataAnchor(0), connect_node->GetInDataAnchor(anchor_idx)));

    auto next_repeats = current_tensor_info.repeats;
    auto axis_iter = std::find(current_tensor_info.axis.begin(), current_tensor_info.axis.end(), broadcast_info[index]);
    auto expanded_axis_iter =
        std::find(expanded_tensor_info.axis.begin(), expanded_tensor_info.axis.end(), broadcast_info[index]);
    GE_ASSERT_TRUE(axis_iter != current_tensor_info.axis.end());
    GE_ASSERT_TRUE(expanded_axis_iter != expanded_tensor_info.axis.end());
    auto axis_index = static_cast<size_t>(std::distance(current_tensor_info.axis.begin(), axis_iter));
    auto expanded_axis_index =
        static_cast<size_t>(std::distance(expanded_tensor_info.axis.begin(), expanded_axis_iter));
    next_repeats[axis_index] = expanded_tensor_info.repeats[expanded_axis_index];
    auto next_strides = std::vector<Expression>();
    GE_ASSERT_SUCCESS(ScheduleUtils::RecalculateStridesFromRepeats(next_repeats, next_strides));
    GE_ASSERT_SUCCESS(
        UpdateBroadcastNodeAttrs(b_node, current_tensor_info.axis, next_repeats, next_strides, broadcast_info[index]));
    GE_ASSERT_SUCCESS(UpdateBroadcastNodeSchedInfo(b_node, node));
    GE_ASSERT_SUCCESS(FromDtypeToOtherDtype(b_node, tensor_info.dtype, tensor_info.dtype));
    current_tensor_info.repeats = std::move(next_repeats);
    current_tensor_info.strides = std::move(next_strides);
    connect_node = b_node;
  }
  return SUCCESS;
}

Status InsertBroadcastNode(NodePtr &pre_bro_node, AscGraph &graph, std::set<int64_t> &broadcast_axis,
                           const TensorInfo &expanded_tensor_info) {
  std::vector<int64_t> broadcast_info(broadcast_axis.begin(), broadcast_axis.end());
  TensorInfo tensor_info;
  GE_ASSERT_SUCCESS(GetTensorInfo(pre_bro_node, tensor_info));
  tensor_info.broadcast_info = broadcast_info;
  NodePtr connect_node;
  GE_ASSERT_SUCCESS(GetSingleNextNode(pre_bro_node, connect_node));
  GE_ASSERT_SUCCESS(CreateAndUpdateBroadcastNode(graph, pre_bro_node, connect_node, tensor_info, expanded_tensor_info));
  return SUCCESS;
}

Status UpdateOutputTensor(std::vector<NodePtr> &nodes, std::set<int64_t> &common_axises) {
  for (auto &node : nodes) {
    AscTensorAttr *compute_output_attr = nullptr;
    GE_ASSERT_SUCCESS(GetOutputTensorAttr(node, compute_output_attr));
    auto &repeats = compute_output_attr->repeats;
    auto &strides = compute_output_attr->strides;
    auto &attr_axis = compute_output_attr->axis;
    for (auto common_axis : common_axises) {
      auto it = std::find(attr_axis.begin(), attr_axis.end(), common_axis);
      GE_ASSERT_TRUE(it != attr_axis.end());
      auto index = std::distance(attr_axis.begin(), it);
      repeats[index] = af::sym::kSymbolOne;
      strides[index] = af::sym::kSymbolZero;
    }
    GE_ASSERT_SUCCESS(ScheduleUtils::RecalculateStridesFromRepeats(repeats, strides));
  }
  return SUCCESS;
}

Status JudgePartBackward(std::set<NodePtr> &mul_input_nodes, bool &is_changed, AscGraph &graph) {
  std::set<NodePtr> next_mul_input_nodes;
  for (auto mul_input_node : mul_input_nodes) {
    std::vector<std::vector<NodePtr>> bro_nodes_list;
    std::vector<NodePtr> origin_bro_nodes;
    std::vector<NodePtr> origin_need_move_bro_nodes;
    std::vector<NodePtr> compute_nodes;
    std::set<int64_t> common_axises;
    NodePtr peer_out_node = nullptr;
    if (!CollectSameBrcAxis(peer_out_node, mul_input_node, bro_nodes_list, common_axises, origin_bro_nodes)) {
      continue;
    }
    if (bro_nodes_list.empty() || origin_bro_nodes.size() > 1U) {
      GELOGD("Skip partial broadcast backward at node[%s]: branches=%zu, origin_broadcasts=%zu.",
             mul_input_node->GetName().c_str(), bro_nodes_list.size(), origin_bro_nodes.size());
      continue;
    }

    TensorInfo expanded_tensor_info;
    GE_ASSERT_SUCCESS(GetTensorInfo(origin_bro_nodes.back(), expanded_tensor_info));

    GE_ASSERT_SUCCESS(GetBackwardBrcNodes(origin_bro_nodes, origin_need_move_bro_nodes, common_axises));

    auto cur_node = mul_input_node;
    auto next_node = mul_input_node;
    GE_ASSERT_SUCCESS(GetSingleNextNode(cur_node, next_node));
    compute_nodes.push_back(mul_input_node);
    GE_ASSERT_SUCCESS(
        CollectCmpNodes(cur_node, next_node, compute_nodes, origin_bro_nodes, graph, next_mul_input_nodes));
    GELOGD("Move partial broadcast at node[%s]: common_axes=%zu, compute_nodes=%zu, broadcasts=%zu.",
           mul_input_node->GetName().c_str(), common_axises.size(), compute_nodes.size(), origin_bro_nodes.size());

    is_changed = true;
    GE_ASSERT_SUCCESS(RemoveBroadcasts(graph, bro_nodes_list, common_axises));
    GE_ASSERT_SUCCESS(InsertBroadcastNode(compute_nodes.back(), graph, common_axises, expanded_tensor_info));

    std::vector<NodePtr> merged_nodes;
    for (const auto &row : bro_nodes_list) {
      merged_nodes.insert(merged_nodes.end(), row.begin(), row.end());
    }
    merged_nodes.insert(merged_nodes.end(), compute_nodes.begin(), compute_nodes.end());
    GE_ASSERT_SUCCESS(UpdateOutputTensor(merged_nodes, common_axises));
  }
  if (!next_mul_input_nodes.empty()) {
    return JudgePartBackward(next_mul_input_nodes, is_changed, graph);
  }
  return SUCCESS;
}

Status ProcessOriginalBackwardLogic(AscGraph &graph, bool &is_changed, std::set<NodePtr> &mul_input_nodes) {
  std::vector<NodePtr> start_nodes;
  GE_ASSERT_SUCCESS(CollectBackwardStartNodes(graph, start_nodes));
  RemoveDuplicates(start_nodes);
  for (const auto &start_node : start_nodes) {
    std::vector<NodePtr> peer_in_nodes;
    GE_ASSERT_SUCCESS(CollectBackwardSatisfyStartNodes(start_node, peer_in_nodes));
    for (const auto &peer_in_node : peer_in_nodes) {
      NodePtr cur_node = start_node;
      NodePtr next_node = peer_in_node;

      std::vector<NodePtr> bro_nodes;
      std::vector<NodePtr> compute_nodes;
      NodePtr pre_bro_node = start_node;
      GE_ASSERT_SUCCESS(CollectBroNodes(cur_node, next_node, bro_nodes));
      GE_ASSERT_SUCCESS(CollectCmpNodes(cur_node, next_node, compute_nodes, bro_nodes, graph, mul_input_nodes));

      GELOGI("Move broadcast backward from node[%s]: broadcasts=%zu, computes=%zu.", peer_in_node->GetName().c_str(),
             bro_nodes.size(), compute_nodes.size());

      if (!bro_nodes.empty() && !compute_nodes.empty()) {
        is_changed = true;
        GE_ASSERT_SUCCESS(BroadcastBackwardReally(compute_nodes, bro_nodes, pre_bro_node));
      }
    }
  }
  return SUCCESS;
}

Status BroadcastBackward(AscGraph &graph) {
  if (ScheduleUtils::HasComputeType(graph, af::ComputeType::kComputeCube)) {
    GELOGI("graph %s fuse type is cube, don't backward broadcast.", graph.GetName().c_str());
    return SUCCESS;
  }

  GE_ASSERT_SUCCESS(broadcast_backward_shared_split::SplitSharedBroadcastBranches(graph));
  GE_ASSERT_SUCCESS(broadcast_backward_shared_split::SplitSharedBroadcastConsumers(graph));

  bool is_changed = false;
  bool has_multi_ref_change = true;
  while (has_multi_ref_change) {
    has_multi_ref_change = false;

    std::set<NodePtr> mul_input_nodes;
    GE_ASSERT_SUCCESS(ProcessOriginalBackwardLogic(graph, is_changed, mul_input_nodes));

    if (!mul_input_nodes.empty()) {
      GE_ASSERT_SUCCESS(JudgePartBackward(mul_input_nodes, is_changed, graph));
    }

    bool multi_ref_changed = false;
    GE_ASSERT_SUCCESS(ProcessMultiRefBroadcastBackward(graph, multi_ref_changed));
    if (multi_ref_changed) {
      is_changed = true;
      has_multi_ref_change = true;
    }
  }

  if (is_changed) {
    GE_ASSERT_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  }
  return SUCCESS;
}
}  // namespace

Status BroadcastBackwardPass::RunPass(af::AscGraph &graph) {
  GE_ASSERT_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  GE_ASSERT_SUCCESS(BroadcastBackward(graph));
  GELOGI("Graph %s completed BroadcastBackward successfully.", graph.GetName().c_str());
  return SUCCESS;
}
}  // namespace optimize
