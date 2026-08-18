/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pass_utils.h"
#include <vector>
#include <queue>
#include <unordered_set>
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/node.h"
#include "ascir/meta/ascir_ops_utils.h"
#include "ascir_ops.h"
#include "graph/utils/graph_utils.h"
#include "symbolizer/symbolic_utils.h"

namespace optimize {
using af::ops::IsOps;
using namespace af::ascir_op;
af::Status PassUtils::PruneGraph(af::AscGraph &graph) {
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  const std::string &graph_name = compute_graph->GetName();

  std::vector<af::NodePtr> out_nodes;
  auto all_nodes = compute_graph->GetDirectNode();
  for (const auto &node_ptr : all_nodes) {
    GE_CHECK_NOTNULL(node_ptr);
    if (IsOps<Output>(node_ptr) || (IsOps<Workspace>(node_ptr) && node_ptr->GetOutDataNodesSize() == 0U)) {
      out_nodes.push_back(node_ptr);
    }
  }
  if (out_nodes.empty()) {
    GELOGW("Graph [%s] does not contain valid output nodes, skip pruning.", graph_name.c_str());
    return af::SUCCESS;
  }

  std::unordered_set<af::NodePtr> reserved_nodes;
  std::queue<af::NodePtr> traverse_queue;
  for (const auto &node_ptr : out_nodes) {
    reserved_nodes.insert(node_ptr);
    traverse_queue.push(node_ptr);
  }

  while (!traverse_queue.empty()) {
    auto curr_node = traverse_queue.front();
    traverse_queue.pop();
    GE_ASSERT_NOTNULL(curr_node);

    const auto in_nodes = curr_node->GetInDataNodes();
    for (const auto &in_node_ptr : in_nodes) {
      GE_ASSERT_NOTNULL(in_node_ptr);
      if (reserved_nodes.insert(in_node_ptr).second) {
        traverse_queue.push(in_node_ptr);
      }
    }
  }

  const af::NodePtr first_out_node = out_nodes.front();
  for (const auto &node_ptr : all_nodes) {
    GE_ASSERT_NOTNULL(node_ptr);
    if (reserved_nodes.count(node_ptr) > 0UL) {
      continue;
    }

    // Data/ScalarData节点需要保留,否则会影响codegen签名
    if (IsOps<Data>(node_ptr) || IsOps<ScalarData>(node_ptr)) {
      auto out_control_anchor = node_ptr->GetOutControlAnchor();
      auto in_control_anchor = first_out_node->GetInControlAnchor();
      GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(out_control_anchor, in_control_anchor));
      GELOGI("Add extra control edge between data node[%s] and output node[%s].", node_ptr->GetNamePtr(),
             first_out_node->GetNamePtr());
      continue;
    }

    GE_ASSERT_GRAPH_SUCCESS(compute_graph->RemoveNode(node_ptr));
    GELOGD("Remove redundant graph node [%s] in graph [%s].", node_ptr->GetNamePtr(), graph_name.c_str());
  }

  return af::SUCCESS;
}

af::Status PassUtils::RelinkAllOutNodeToSrc(const af::OutDataAnchorPtr &old_src, const af::OutDataAnchorPtr &new_src) {
  GE_ASSERT_NOTNULL(new_src);
  GE_ASSERT_NOTNULL(old_src);
  for (const auto &cur_next_in_anchor : old_src->GetPeerInDataAnchors()) {
    GE_ASSERT_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(old_src, cur_next_in_anchor, new_src));
  }
  return af::SUCCESS;
}

bool PassUtils::IsExprVectorEqual(const std::vector<af::Expression> &lhs, const std::vector<af::Expression> &rhs) {
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

af::AscNodePtr PassUtils::CreateOneScalarBrc(af::AscGraph &graph, const af::AscNodePtr &ref_node) {
  std::string scalar_name = ref_node->GetName() + "_One";
  af::ascir_op::Scalar scalar_one(scalar_name.c_str(), graph);
  scalar_one.ir_attr.SetValue(af::SymbolicUtils::ToString(af::sym::kSymbolOne));
  scalar_one.y.dtype = static_cast<ge::DataType>(ref_node->outputs[0].attr.dtype);
  // 动态创建的 Scalar 在 CompleteApiInfo() 之后生成，api.type 保持默认 kAPITypeInvalid，
  // 会导致 IsBuffer() 返回 false，Scalar 错误进入 ReplaceAxis 触发断言。
  scalar_one.attr.api.type = af::ApiType::kAPITypeBuffer;

  std::string brc_name = ref_node->GetName() + "_Brc";
  af::ascir_op::Broadcast brc(brc_name.c_str());
  auto brc_node = graph.AddNode(brc);
  GE_ASSERT_NOTNULL(brc_node);
  brc_node->attr.sched = ref_node->attr.sched;
  brc_node->outputs[0].attr = ref_node->outputs[0].attr;
  brc.x = scalar_one.y;

  return brc_node;
}

}  // namespace optimize
