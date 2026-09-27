/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "redundant_broadcast_eliminate_pass.h"

#include <vector>

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "common_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "graph/types_af.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "optimize/graph_pass/pass_utils.h"

namespace optimize {
namespace {

using af::AscGraph;
using af::AscNode;
using af::AscNodePtr;

// scalar-like 生产者（Scalar/ScalarData/IndexExpr）输出视图为空（无 axis/repeats/strides），
// 与 Broadcast 输出的完整视图不可比，且其 Broadcast 承担"标量接上有效输入视图"的职责，
// 即使输出 repeats 全 1 也不属于可剔除的纯冗余。
bool IsScalarLikeProducer(const AscNodePtr &node) {
  return node->GetType() == af::ascir_op::Scalar::Type || node->GetType() == af::ascir_op::ScalarData::Type ||
         node->GetType() == af::ascir_op::IndexExpr::Type;
}

// 1) 必须是 Broadcast 节点，单数据输入、无控制边、有后继（无后继时删除会令前驱输出悬空）；
// 2) 前驱节点输出与 Broadcast 输出完整视图一致（dtype/axis/repeats/strides/vectorized），
//    说明 Broadcast 未产生任何扩维或布局变化，纯冗余，删除后后继看到的视图与删除前完全一致。
bool IsRedundantBroadcast(const AscNodePtr &node) {
  if (!af::ops::IsOps<af::ascir_op::Broadcast>(node) || node->GetAllInDataAnchorsSize() != 1U ||
      node->GetInControlNodesSize() != 0U || node->GetOutControlNodesSize() != 0U ||
      node->GetAllOutDataAnchorsSize() != 1U) {
    return false;
  }
  const auto in_anchor = node->GetInDataAnchor(0);
  if (in_anchor == nullptr) {
    return false;
  }
  const auto pre_out_anchor = in_anchor->GetPeerOutAnchor();
  if (pre_out_anchor == nullptr) {
    return false;
  }
  const auto pre_node = std::dynamic_pointer_cast<AscNode>(pre_out_anchor->GetOwnerNode());
  if (pre_node == nullptr || pre_node->GetAllOutDataAnchorsSize() != 1U || IsScalarLikeProducer(pre_node)) {
    return false;
  }
  const auto &pre_attr = pre_node->outputs[0].attr;
  const auto &bro_attr = node->outputs[0].attr;
  return static_cast<ge::DataType>(pre_attr.dtype) == static_cast<ge::DataType>(bro_attr.dtype) &&
         pre_attr.axis == bro_attr.axis && PassUtils::IsExprVectorEqual(pre_attr.repeats, bro_attr.repeats) &&
         PassUtils::IsExprVectorEqual(pre_attr.strides, bro_attr.strides) &&
         pre_attr.vectorized_axis == bro_attr.vectorized_axis &&
         PassUtils::IsExprVectorEqual(pre_attr.vectorized_strides, bro_attr.vectorized_strides);
}

// 删除冗余 Broadcast：前驱节点输出直连所有后继，移除 Broadcast 节点。
Status RemoveRedundantBroadcast(const AscNodePtr &bro_node) {
  const auto bro_in_anchor = bro_node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(bro_in_anchor);
  const auto pre_out_anchor = bro_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(pre_out_anchor);
  const auto bro_out_anchor = bro_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(bro_out_anchor);

  GE_ASSERT_SUCCESS(PassUtils::RelinkAllOutNodeToSrc(bro_out_anchor, pre_out_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(pre_out_anchor, bro_in_anchor));
  const auto owner_graph = bro_node->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph);
  af::NodeUtils::UnlinkAll(*bro_node);
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(owner_graph, bro_node));
  const auto pre_node = std::dynamic_pointer_cast<AscNode>(pre_out_anchor->GetOwnerNode());
  GELOGI("Removed redundant Broadcast node [%s], predecessor [%s] directly connected to its consumers.",
         bro_node->GetNamePtr(), pre_node == nullptr ? "unknown" : pre_node->GetNamePtr());
  return af::SUCCESS;
}
}  // namespace

Status RedundantBroadcastEliminatePass::RunPass(AscGraph &graph) {
  // 先收集冗余节点，遍历结束后统一删除，避免遍历过程中图结构变化影响迭代。
  std::vector<AscNodePtr> redundant_nodes;
  for (const auto &node : graph.GetAllNodes()) {
    if (IsRedundantBroadcast(node)) {
      redundant_nodes.emplace_back(node);
    }
  }
  for (const auto &node : redundant_nodes) {
    GE_ASSERT_SUCCESS(RemoveRedundantBroadcast(node));
  }
  return af::SUCCESS;
}
}  // namespace optimize
