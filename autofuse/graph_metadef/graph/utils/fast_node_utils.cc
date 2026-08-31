/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/fast_node_utils.h"

#include "common/checker.h"
#include "graph/compiler_options.h"
#include "graph/ge_tensor.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/fast_graph/fast_graph_utils.h"
#include "graph/normal_graph/node_impl.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph/debug/ge_op_types.h"
#include "framework/common/debug/ge_log.h"

namespace af {
FastNode *FastNodeUtils::GetInDataNodeByIndex(const FastNode *const node, const int32_t index) {
  GE_ASSERT_NOTNULL(node);
  const auto in_data_edge = node->GetInDataEdgeByIndex(index);
  GE_ASSERT_NOTNULL(in_data_edge);
  return in_data_edge->src;
}

FastNode *FastNodeUtils::GetParentInput(const FastNode *const node) {
  GE_ASSERT_NOTNULL(node);
  uint32_t parent_index = 0U;
  if (!AttrUtils::GetInt(node->GetOpDescPtr(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    return nullptr;
  }

  // Subgraph Data Node, check for constant input.
  GE_ASSERT_NOTNULL(node->GetExtendInfo(), "ExtendInfo of node %s is null.", node->GetNamePtr());
  const auto graph = node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);

  const auto parent_node = graph->GetParentNodeBarePtr();
  if (parent_node == nullptr) {
    GELOGW("Node {%s %s} has attr %s but has no parent node.", node->GetNamePtr(), node->GetTypePtr(),
           ATTR_NAME_PARENT_NODE_INDEX.c_str());
    return nullptr;
  }

  const auto edge = parent_node->GetInDataEdgeByIndex(static_cast<int32_t>(parent_index));
  GE_ASSERT_NOTNULL(edge);
  const auto src_node = edge->src;
  GE_ASSERT_NOTNULL(src_node);

  if (src_node->GetType() == DATA) {
    GE_ASSERT_NOTNULL(src_node->GetOpDescBarePtr());
    if (src_node->GetOpDescBarePtr()->HasAttr(ATTR_NAME_PARENT_NODE_INDEX)) {
      return GetParentInput(src_node);
    }
  }
  return src_node;
}

bool FastNodeUtils::GetConstOpType(const FastNode *const node) {
  if (node == nullptr) {
    return false;
  }

  const auto &node_type = node->GetType();
  if ((node_type == CONSTANT) || (node_type == CONSTANTOP) || (node_type == FILECONSTANT)) {
    return true;
  }

  if (node_type != DATA) {
    return false;  // not subgraph input node
  }

  const auto parent_node = GetParentInput(node);
  return GetConstOpType(parent_node);
}

graphStatus FastNodeUtils::AppendSubgraphToNode(FastNode *const node, const std::string &subgraph_name,
                                                const ExecuteGraphPtr &subgraph) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(subgraph);
  auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);

  GE_ASSERT_GRAPH_SUCCESS(op_desc->AddSubgraphName(subgraph_name));
  const auto &subgraph_names_to_index = op_desc->GetSubgraphNameIndexes();
  const auto &iter = subgraph_names_to_index.find(subgraph_name);
  GE_ASSERT_TRUE(iter != subgraph_names_to_index.cend());

  return MountSubgraphToNode(node, iter->second, subgraph);
}

ExecuteGraph *FastNodeUtils::GetSubgraphFromNode(const FastNode *const node, const uint32_t index) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);

  GE_ASSERT_NOTNULL(node->GetExtendInfo(), "ExtendInfo of node %s is null.", node->GetNamePtr());
  const auto root_graph = ExecuteGraphUtils::FindRootGraph(node->GetExtendInfo()->GetOwnerGraphBarePtr());
  GE_ASSERT_NOTNULL(root_graph);
  return root_graph->GetSubGraph(op_desc->GetSubgraphInstanceName(index));
}

graphStatus FastNodeUtils::MountSubgraphToNode(FastNode *const node, const uint32_t index,
                                               const ExecuteGraphPtr &subgraph) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(subgraph, "[Check][Param] Failed to set subgraph to node %s index %u, null subgraph",
                    node->GetNamePtr(), index);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);

  GE_ASSERT_NOTNULL(node->GetExtendInfo(), "ExtendInfo of node %s is null.", node->GetNamePtr());
  const auto root_graph = ExecuteGraphUtils::FindRootGraph(node->GetExtendInfo()->GetOwnerGraphBarePtr());
  GE_ASSERT_NOTNULL(root_graph, "[Get][Graph] Failed to add subgraph to node %s, null root graph", node->GetNamePtr());

  const auto ret = op_desc->SetSubgraphInstanceName(index, subgraph->GetName());
  GE_CHK_GRAPH_STATUS_RET(ret, "[Set][Name] Failed to set subgraph to node %s index %u", node->GetNamePtr(), index);

  subgraph->SetParentNode(node);
  GE_ASSERT_NOTNULL(node->GetExtendInfo(), "ExtendInfo of node %s is null.", node->GetNamePtr());
  subgraph->SetParentGraph(node->GetExtendInfo()->GetOwnerGraphBarePtr());

  return (root_graph->AddSubGraph(const_cast<ExecuteGraphPtr &>(subgraph)) != nullptr) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus FastNodeUtils::AppendInputEdgeInfo(FastNode *const node,
                                                                                              const uint32_t num) {
  GE_ASSERT_NOTNULL(node);

  const GeTensorDesc data_desc(GeShape(), FORMAT_ND, DT_FLOAT);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  for (size_t i = op_desc->GetAllInputsSize(); i < num; ++i) {
    GE_CHK_GRAPH_STATUS_RET(op_desc->AddInputDesc(data_desc), "[Add][InputDesc] failed, op:%s", op_desc->GetNamePtr());
  }
  node->UpdateDataInNum(num);

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus FastNodeUtils::AppendOutputEdgeInfo(FastNode *const node,
                                                                                               const uint32_t num) {
  GE_ASSERT_NOTNULL(node);

  const GeTensorDesc data_desc(GeShape(), FORMAT_ND, DT_FLOAT);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  for (size_t i = op_desc->GetOutputsSize(); i < num; ++i) {
    GE_CHK_GRAPH_STATUS_RET(op_desc->AddOutputDesc(data_desc), "[Add][OutputDesc] failed, op:%s",
                            op_desc->GetNamePtr());
  }
  node->UpdateDataOutNum(num);

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool FastNodeUtils::ClearInputDesc(const OpDesc *const op_desc,
                                                                                  const uint32_t index) {
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_NOTNULL(op_desc->impl_);
  GE_ASSERT_TRUE((index < op_desc->impl_->inputs_desc_.size()),
                 "[Check][Param] index %u is invalid, out of range(0, %zu).", index,
                 op_desc->impl_->inputs_desc_.size());

  const auto iter = op_desc->impl_->inputs_desc_.begin() + static_cast<int64_t>(index);
  (void)op_desc->impl_->inputs_desc_.erase(iter);
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus FastNodeUtils::RemoveInputEdgeInfo(FastNode *const node,
                                                                                              const uint32_t num) {
  GE_ASSERT_NOTNULL(node);

  const auto &op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  uint32_t input_size = op_desc->GetAllInputsSize();
  while (input_size > num) {
    if (!FastNodeUtils::ClearInputDesc(op_desc, input_size - 1)) {
      return GRAPH_FAILED;
    }
    --input_size;
  }

  const auto &input_names = op_desc->GetAllInputName();
  (void)op_desc->UpdateInputName(input_names);
  auto is_input_consts = op_desc->GetIsInputConst();
  is_input_consts.resize(static_cast<std::size_t>(num));
  op_desc->SetIsInputConst(is_input_consts);

  node->UpdateDataInNum(num);

  return GRAPH_SUCCESS;
}

void FastNodeUtils::UnlinkAll(FastNode *const node) {
  if (node == nullptr || node->GetExtendInfo() == nullptr) {
    GELOGW("param node is null or node's extend info is null.");
    return;
  }
  const auto owner_graph = node->GetExtendInfo()->GetOwnerGraphBarePtr();
  const auto remove_edge_func = [&owner_graph](FastEdge *e) {
    if (e->src != nullptr) {
      e->src->EraseEdge(e, DirectionType::kDirectionOutType);
      e->src = nullptr;
    }
    if (e->dst != nullptr) {
      e->dst->EraseEdge(e, DirectionType::kDirectionInType);
      e->dst = nullptr;
    }
    if (FastGraphUtils::GetListElementAddr(e)->owner != nullptr) {
      FastGraphUtils::GetListElementAddr(e)->owner->erase(FastGraphUtils::GetListElementAddr(e));
    }
    auto ret = owner_graph->RecycleQuickEdge(e);
    if ((ret != GRAPH_SUCCESS) && (e != nullptr)) {
      delete e;
    }
  };
  node->RemoveAllEdge(remove_edge_func);
}

EdgeDstEndpoint FastNodeUtils::GetDstEndpoint(const FastEdge *const edge) {
  GE_ASSERT_NOTNULL(edge);
  return {edge->dst, edge->dst_input};
}

EdgeSrcEndpoint FastNodeUtils::GetSrcEndpoint(const FastEdge *const edge) {
  GE_ASSERT_NOTNULL(edge);
  return {edge->src, edge->src_output};
}
}  // namespace af
