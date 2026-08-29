/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ref_relation.h"

#include <unordered_set>
#include <set>
#include <unordered_map>

#include "common/util/mem_utils.h"
#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "debug/ge_op_types.h"
#include "graph/utils/graph_utils.h"
#include "graph/def_types.h"
#include "graph_metadef/common/ge_common/util.h"

namespace af {
namespace {
const char_t *kRefIdx = "_parent_node_index";
const char_t *kWhile = "While";
const char_t *kIf = "If";
const char_t *kCase = "Case";
const char_t *kStatelessWhile = "StatelessWhile";
std::set<std::string> function_op = {kWhile, kIf, kCase};
}  // namespace

/* Impl */
class RefRelations::Impl {
 public:
  graphStatus LookUpRefRelations(const RefCell &key, std::unordered_set<RefCell, RefCellHash> &result) {
    const auto iter = look_up_table_.find(key.hash_key);
    if (iter != look_up_table_.end()) {
      for (auto &c : iter->second) {
        (void)result.insert(c);
      }
      return GRAPH_SUCCESS;
    }
    GELOGD("[RefRelations][Check] can not find any relations! key value of dest relation is %s", key.hash_key.c_str());
    return GRAPH_SUCCESS;
  };
  graphStatus BuildRefRelations(af::ComputeGraph &graph);
  graphStatus Clear() {
    GELOGD("Start clear boundary reflections between main graph and sub graph!");
    look_up_table_.clear();
    values_.clear();
    return GRAPH_SUCCESS;
  };

 private:
  friend class RefRelations;
  graphStatus BuildLookUpTables();
  graphStatus BuildRefRelationsForBranch(
      const NodePtr &root_node, const std::vector<std::vector<NodePtr>> &classed_data_nodes,
      const std::vector<std::vector<std::pair<NodePtr, size_t>>> &classed_netoutput_nodes,
      std::vector<std::vector<RefCell>> &node_refs) const;
  graphStatus BuildRefRelationsForWhile(
      const NodePtr &root_node, const std::vector<std::vector<NodePtr>> &classed_data_nodes,
      const std::vector<std::vector<std::pair<NodePtr, size_t>>> &classed_netoutput_nodes,
      std::vector<std::vector<RefCell>> &node_refs) const;
  graphStatus BuildRelationsWithFuncNodeType(
      const NodePtr &root_node, const std::vector<std::vector<NodePtr>> &classed_data_nodes,
      const std::vector<std::vector<std::pair<NodePtr, size_t>>> &classed_netoutput_nodes,
      std::vector<std::vector<RefCell>> &node_refs) const;
  void GetDataAndNetoutputOfSubGraph(const af::ComputeGraph &root_graph, std::vector<NodePtr> &graph_data_nodes,
                                     std::vector<NodePtr> &netoutput_nodes,
                                     const std::vector<std::string> &sub_graph_names,
                                     const std::string &node_type) const;

  graphStatus GetRootGraph(af::ComputeGraph &graph, af::ComputeGraph &root_graph) const;
  graphStatus ProcessSubgraphDataNodes(std::vector<NodePtr> &graph_data_nodes,
                                       std::vector<std::vector<NodePtr>> &classed_data_nodes) const;
  graphStatus ProcessSubgraphNetoutput(
      const std::vector<NodePtr> &netoutput_nodes,
      std::vector<std::vector<std::pair<NodePtr, size_t>>> &classed_netoutput_nodes) const;
  void BuildRelationsForVariables(const af::ComputeGraph &root_graph);

  std::unordered_map<std::string, std::vector<RefCell>> look_up_table_;
  std::vector<std::vector<std::vector<RefCell>>> values_;
};

// Node Level
graphStatus RefRelations::Impl::BuildRefRelationsForBranch(
    const NodePtr &root_node, const std::vector<std::vector<NodePtr>> &classed_data_nodes,
    const std::vector<std::vector<std::pair<NodePtr, size_t>>> &classed_netoutput_nodes,
    std::vector<std::vector<RefCell>> &node_refs) const {
  GELOGD("Enter BuildRefRelationsForBranch!");

  size_t ref_i = 0UL;
  for (const auto &ref_i_data_nodes : classed_data_nodes) {
    std::vector<RefCell> in_ref_i_all_refs;
    RefCell cell_root(root_node->GetName(), root_node, NODE_IN, static_cast<int32_t>(ref_i));
    in_ref_i_all_refs.emplace_back(cell_root);
    for (const auto &data : ref_i_data_nodes) {
      RefCell cell_in(data->GetName(), data, NODE_IN, 0);
      RefCell cell_out(data->GetName(), data, NODE_OUT, 0);
      in_ref_i_all_refs.emplace_back(cell_in);
      in_ref_i_all_refs.emplace_back(cell_out);
    }
    node_refs.emplace_back(in_ref_i_all_refs);
    ref_i++;
  }

  size_t ref_o = 0UL;
  for (const auto &ref_o_net_nodes : classed_netoutput_nodes) {
    std::vector<RefCell> out_ref_i_all_refs;
    RefCell cell_root(root_node->GetName(), root_node, NODE_OUT, static_cast<int32_t>(ref_o));
    out_ref_i_all_refs.emplace_back(cell_root);
    for (const auto &ele : ref_o_net_nodes) {
      RefCell cell_netoutput_in((ele.first)->GetName(), ele.first, NODE_IN, static_cast<int32_t>(ele.second));
      out_ref_i_all_refs.emplace_back(cell_netoutput_in);
    }
    node_refs.emplace_back(out_ref_i_all_refs);
    ref_o++;
  }
  return GRAPH_SUCCESS;
}

graphStatus RefRelations::Impl::BuildLookUpTables() {
  GELOGD("start to build look up table!");
  for (size_t i = 0UL; i < values_.size(); i++) {
    std::vector<std::vector<RefCell>> &val = values_[i];
    for (const auto &ele : val) {
      for (const auto &ref_cell : ele) {
        look_up_table_.emplace(ref_cell.hash_key, ele);
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus RefRelations::Impl::BuildRefRelationsForWhile(
    const NodePtr &root_node, const std::vector<std::vector<NodePtr>> &classed_data_nodes,
    const std::vector<std::vector<std::pair<NodePtr, size_t>>> &classed_netoutput_nodes,
    std::vector<std::vector<RefCell>> &node_refs) const {
  GELOGD("Enter BuildRefRelations for while op!");
  // data_nodes has been sorted
  // for while, input num must be same as output num
  const auto input_num = root_node->GetAllInDataAnchorsSize();
  NodePtr netoutput = nullptr;
  // Boundary check: prevent out-of-bounds access when indexing classed nodes with input_num.
  GE_ASSERT(classed_data_nodes.size() >= input_num, "classed_data_nodes size %zu must be >= input_num %zu",
            classed_data_nodes.size(), input_num);
  GE_ASSERT(classed_netoutput_nodes.size() >= input_num, "classed_netoutput_nodes size %zu must be >= input_num %zu",
            classed_netoutput_nodes.size(), input_num);
  size_t ref_i = 0UL;
  while (ref_i < input_num) {
    auto &ref_i_data_nodes = classed_data_nodes[ref_i];
    auto &ref_i_net_nodes = classed_netoutput_nodes[ref_i];

    std::vector<RefCell> ref_i_all_refs;
    RefCell cell_root_i(root_node->GetName(), root_node, NODE_IN, static_cast<int32_t>(ref_i));
    RefCell cell_root_o(root_node->GetName(), root_node, NODE_OUT, static_cast<int32_t>(ref_i));
    ref_i_all_refs.emplace_back(cell_root_i);
    ref_i_all_refs.emplace_back(cell_root_o);
    for (const auto &data : ref_i_data_nodes) {
      RefCell cell_in(data->GetName(), data, NODE_IN, 0);
      RefCell cell_out(data->GetName(), data, NODE_OUT, 0);
      ref_i_all_refs.emplace_back(cell_in);
      ref_i_all_refs.emplace_back(cell_out);
    }

    for (const auto &ele : ref_i_net_nodes) {
      RefCell cell_netoutput_in((ele.first)->GetName(), ele.first, NODE_IN, static_cast<int32_t>(ele.second));
      ref_i_all_refs.emplace_back(cell_netoutput_in);
      netoutput = ele.first;
    }
    node_refs.emplace_back(ref_i_all_refs);
    ref_i++;
  }
  /* There exist scene like the follows, it means data0 data1 netoutput 0'th
   * and 1'th tensor should be the same addr.
   * Data0  Data1
   *      \/
   *      /\
   *   netoutput
   */
  if (netoutput == nullptr) {
    return GRAPH_SUCCESS;
  }
  for (const auto &in_anchor : netoutput->GetAllInDataAnchorsPtr()) {
    const auto peer_out_data_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_data_anchor == nullptr) {
      continue;
    }
    const auto peer_out_data_node = peer_out_data_anchor->GetOwnerNodeBarePtr();
    if ((peer_out_data_node == nullptr) || (peer_out_data_node->GetOpDesc() == nullptr)) {
      GELOGW("[RefRelations][Check] Node[%s]\'s peer_out_data_node or peer_out_data_node desc is null",
             netoutput->GetName().c_str());
      continue;
    }
    if (peer_out_data_node->GetType() != DATA) {
      continue;
    }
    const auto in_data_anchor_idx = in_anchor->GetIdx();
    const auto net_in_desc = netoutput->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(in_data_anchor_idx));
    int32_t ref_d = 0;
    int32_t ref_n = 0;
    (void)AttrUtils::GetInt(peer_out_data_node->GetOpDesc(), kRefIdx, ref_d);
    (void)AttrUtils::GetInt(net_in_desc, kRefIdx, ref_n);
    const size_t ref_desc = static_cast<size_t>(ref_d);
    const size_t ref_in = static_cast<size_t>(ref_n);

    const size_t idx1 = node_refs[ref_in].size();  // 注意，不要删除idx1、idx2，存在ref_desc=ref_in的情况
    for (size_t i = 0U; i < idx1; ++i) {
      node_refs[ref_desc].emplace_back(node_refs[ref_in][i]);
    }
    const size_t idx2 = node_refs[ref_desc].size();
    for (size_t i = 0U; i < idx2; ++i) {
      node_refs[ref_in].emplace_back(node_refs[ref_desc][i]);
    }
  }

  return GRAPH_SUCCESS;
}
// build ref relations according to diff func op type
graphStatus RefRelations::Impl::BuildRelationsWithFuncNodeType(
    const NodePtr &root_node, const std::vector<std::vector<NodePtr>> &classed_data_nodes,
    const std::vector<std::vector<std::pair<NodePtr, size_t>>> &classed_netoutput_nodes,
    std::vector<std::vector<RefCell>> &node_refs) const {
  // data_nodes has been sorted
  const auto &node_type = root_node->GetType();

  auto status = GRAPH_SUCCESS;
  if ((node_type != kWhile) && (node_type != kStatelessWhile)) {
    status = BuildRefRelationsForBranch(root_node, classed_data_nodes, classed_netoutput_nodes, node_refs);
  } else {
    status = BuildRefRelationsForWhile(root_node, classed_data_nodes, classed_netoutput_nodes, node_refs);
  }
  return status;
}

void RefRelations::Impl::GetDataAndNetoutputOfSubGraph(const af::ComputeGraph &root_graph,
                                                       std::vector<NodePtr> &graph_data_nodes,
                                                       std::vector<NodePtr> &netoutput_nodes,
                                                       const std::vector<std::string> &sub_graph_names,
                                                       const std::string &node_type) const {
  int32_t sub_graph_idx = 0;
  for (const auto &name : sub_graph_names) {
    const auto &sub_graph = root_graph.GetSubgraph(name);
    if (sub_graph == nullptr) {
      GELOGW("[RefRelations][Check] Can not find sub graph %s, root graph: %s.", name.c_str(),
             root_graph.GetName().c_str());
      continue;
    }
    for (const auto &sub_graph_node : sub_graph->GetDirectNode()) {
      const auto &sub_graph_node_type = sub_graph_node->GetType();
      if (sub_graph_node_type == DATA) {
        graph_data_nodes.emplace_back(sub_graph_node);
      }
      if (sub_graph_node_type == NETOUTPUT) {
        // if while, the first subgraph must be cond subgraph.
        // There is no meaning for refs ,so continue
        if (((node_type == kWhile) || (node_type == kStatelessWhile)) && (sub_graph_idx == 0)) {
          continue;
        }
        netoutput_nodes.emplace_back(sub_graph_node);
      }
    }
    sub_graph_idx++;
  }
}

graphStatus RefRelations::Impl::GetRootGraph(af::ComputeGraph &graph, af::ComputeGraph &root_graph) const {
  const auto parent_graph_ptr = graph.GetParentGraph();
  if (parent_graph_ptr == nullptr) {
    root_graph = graph;
    return GRAPH_SUCCESS;
  }
  const auto root_graph_ptr = GraphUtils::FindRootGraph(parent_graph_ptr);
  if (root_graph_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Get null root graph, graph:%s", parent_graph_ptr->GetName().c_str());
    GE_LOGE("[Find][Graph] Get null root graph");
    return GRAPH_PARAM_INVALID;
  }
  root_graph = *root_graph_ptr;
  return GRAPH_SUCCESS;
}

graphStatus RefRelations::Impl::ProcessSubgraphDataNodes(std::vector<NodePtr> &graph_data_nodes,
                                                         std::vector<std::vector<NodePtr>> &classed_data_nodes) const {
  GELOGD("start to process subgraph data nodes!");
  int32_t max_ref_idx = 0;
  for (const auto &e : graph_data_nodes) {
    int32_t i;
    bool is_exist = true;
    is_exist = AttrUtils::GetInt(e->GetOpDesc(), kRefIdx, i);
    if (!is_exist) {
      REPORT_INNER_ERR_MSG("E18888", "Invalid SubGraph NetOutput node[%s]. no attr %s", e->GetName().c_str(), kRefIdx);
      GELOGE(GRAPH_FAILED, "[Get][Int] Invalid SubGraph NetOutput node[%s]. no attr %s", e->GetName().c_str(), kRefIdx);
      return GRAPH_FAILED;
    }
    max_ref_idx = (i > max_ref_idx) ? i : max_ref_idx;
  }
  classed_data_nodes.resize(static_cast<size_t>(max_ref_idx) + 1UL);
  while (!graph_data_nodes.empty()) {
    auto data = graph_data_nodes.back();
    graph_data_nodes.pop_back();
    int32_t ref_idx = 0;
    (void)AttrUtils::GetInt(data->GetOpDesc(), kRefIdx, ref_idx);
    if (ref_idx >= static_cast<int32_t>(classed_data_nodes.size())) {
      return GRAPH_FAILED;
    }
    classed_data_nodes[static_cast<size_t>(ref_idx)].emplace_back(data);
  }
  return GRAPH_SUCCESS;
}

graphStatus RefRelations::Impl::ProcessSubgraphNetoutput(
    const std::vector<NodePtr> &netoutput_nodes,
    std::vector<std::vector<std::pair<NodePtr, size_t>>> &classed_netoutput_nodes) const {
  GELOGD("[RefRelations]Start to process subgraph netoutput!");
  // calc  netoutput max_ref_idx
  int32_t max_ref_idx = 0;
  for (const auto &sub_netoutput_node : netoutput_nodes) {
    const auto op_desc = sub_netoutput_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    for (const auto &in_data_anchor : sub_netoutput_node->GetAllInDataAnchorsPtr()) {
      const auto in_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(in_data_anchor->GetIdx()));
      if (in_desc == nullptr) {
        REPORT_INNER_ERR_MSG("E18888", "Invalid NetOutput node [%s] idx [%d], no tensor on it",
                             sub_netoutput_node->GetName().c_str(), in_data_anchor->GetIdx());
        GELOGE(GRAPH_FAILED, "[Get][Tensor] Invalid NetOutput node [%s] idx [%d], no tensor on it",
               sub_netoutput_node->GetName().c_str(), in_data_anchor->GetIdx());
        return GRAPH_FAILED;
      }
      int32_t ref_o;
      if (AttrUtils::GetInt(in_desc, kRefIdx, ref_o)) {
        max_ref_idx = (ref_o > max_ref_idx) ? ref_o : max_ref_idx;
      } else {
        REPORT_INNER_ERR_MSG("E18888", "Invalid NetOutput node [%s] idx [%d], no attr[_parent_node_index] on it",
                             sub_netoutput_node->GetName().c_str(), in_data_anchor->GetIdx());
        GELOGE(GRAPH_FAILED, "[Get][Int] Invalid NetOutput node [%s] idx [%d], no attr[_parent_node_index] on it",
               sub_netoutput_node->GetName().c_str(), in_data_anchor->GetIdx());
        return GRAPH_FAILED;
      }
    }
  }
  classed_netoutput_nodes.resize(static_cast<size_t>(max_ref_idx) + 1UL);
  // re-sort according ref idx
  for (const auto &sub_netoutput_node : netoutput_nodes) {
    const auto op_desc = sub_netoutput_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    for (const auto &in_data_anchor : sub_netoutput_node->GetAllInDataAnchorsPtr()) {
      const auto in_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(in_data_anchor->GetIdx()));
      int32_t ref_o;
      if (AttrUtils::GetInt(in_desc, kRefIdx, ref_o)) {
        if (ref_o >= static_cast<int32_t>(classed_netoutput_nodes.size())) {
          return GRAPH_FAILED;
        }
        classed_netoutput_nodes[static_cast<size_t>(ref_o)].emplace_back(
            std::pair<NodePtr, size_t>({sub_netoutput_node, static_cast<size_t>(in_data_anchor->GetIdx())}));
      }
    }
  }
  return GRAPH_SUCCESS;
}

void RefRelations::Impl::BuildRelationsForVariables(const af::ComputeGraph &root_graph) {
  if (root_graph.GetAllSubgraphs().empty()) {
    return;
  }

  std::map<std::string, std::vector<NodePtr>> variables;
  for (const auto &node : root_graph.GetAllNodes()) {
    if (node->GetType() == VARIABLE) {
      variables[node->GetName()].emplace_back(node);
    }
  }

  for (const auto &it : variables) {
    const auto &instances = it.second;
    if (instances.size() <= 1UL) {
      continue;
    }

    GELOGD("Variable [%s] has %zu instances", it.first.c_str(), instances.size());
    std::vector<RefCell> variable_all_refs;
    for (const auto &variable : instances) {
      RefCell variable_ref(it.first, variable, NODE_OUT, 0);
      variable_all_refs.emplace_back(std::move(variable_ref));
    }

    std::vector<std::vector<RefCell>> refs{variable_all_refs};
    values_.emplace_back(std::move(refs));
  }
}

graphStatus RefRelations::Impl::BuildRefRelations(af::ComputeGraph &graph) {
  GELOGD("Start to build ref relations!");
  /* First Step: Get root graph */
  af::ComputeGraph &root_graph = graph;
  auto status = GetRootGraph(graph, root_graph);
  if (status != GRAPH_SUCCESS) {
    return status;
  }

  for (const auto &node : graph.GetAllNodes()) {
    const auto &node_type = node->GetType();
    const auto &op_desc = node->GetOpDesc();
    const auto &sub_graph_names = op_desc->GetSubgraphInstanceNames();
    if (sub_graph_names.empty()) {
      continue;
    }
    std::vector<NodePtr> graph_data_nodes;
    std::vector<NodePtr> netoutput_nodes;
    // Get data and netoutput of sub_graph
    GetDataAndNetoutputOfSubGraph(root_graph, graph_data_nodes, netoutput_nodes, sub_graph_names, node_type);
    std::vector<std::vector<NodePtr>> classed_data_nodes;                          // resize according to ref_idx
    std::vector<std::vector<std::pair<NodePtr, size_t>>> classed_netoutput_nodes;  // resize according to ref_idx
    status = ProcessSubgraphDataNodes(graph_data_nodes, classed_data_nodes);
    if (status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Process][SubgraphDataNodes] failed! ret:%d", status);
      return status;
    }

    // for netoutput
    // check netoutput
    // here main graph output number must be the same as every sub_graph netoutput node
    // key: netoutput node_ptr ,<ref_idx, net_in_idx>
    status = ProcessSubgraphNetoutput(netoutput_nodes, classed_netoutput_nodes);
    if (status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Process][SubgraphNetoutput] failed! ret:%d", status);
      return status;
    }

    std::vector<std::vector<RefCell>> node_refs;
    status = BuildRelationsWithFuncNodeType(node, classed_data_nodes, classed_netoutput_nodes, node_refs);
    if (status != GRAPH_SUCCESS) {
      GELOGE(status, "[Build][Relations] WithFuncNodeType Failed! Node is [%s]!", node->GetName().c_str());
      return status;
    }
    if (!node_refs.empty()) {
      values_.push_back(node_refs);
    }
  }

  BuildRelationsForVariables(root_graph);
  /* Seconde Step: generate map */
  status = BuildLookUpTables();
  if (status != GRAPH_SUCCESS) {
    GELOGE(status, "[Build][LookUpTables] failed! ret:%d", status);
    return status;
  }
  return GRAPH_SUCCESS;
}

/* Ref Relations Interface */
RefRelations::RefRelations() {
  impl_ = MakeShared<Impl>();
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "new impl failed.");
    GELOGE(GRAPH_FAILED, "[New][Impl] MakeShared failed!");
    return;
  }
}

graphStatus RefRelations::LookUpRefRelations(const RefCell &key, std::unordered_set<RefCell, RefCellHash> &result) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->LookUpRefRelations(key, result);
}

graphStatus RefRelations::BuildRefRelations(af::ComputeGraph &graph) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->BuildRefRelations(graph);
}

graphStatus RefRelations::Clear() {
  GE_CHECK_NOTNULL(impl_);
  return impl_->Clear();
}
}  // namespace af
