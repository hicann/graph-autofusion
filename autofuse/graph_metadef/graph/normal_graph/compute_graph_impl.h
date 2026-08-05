/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_COMPUTE_GRAPH_IMPL_H_
#define GRAPH_COMPUTE_GRAPH_IMPL_H_

#include "graph/compute_graph.h"

namespace af {
inline const ge::char_t *GetTopoSortingModeStr(const TopoSortingMode &mode) {
  static const ge::char_t *topo_sorting_mode_strs[static_cast<int32_t>(TopoSortingMode::kInvalid) + 1U] = {
      "BFS", "DFS", "RDFS", "StableRDFS", "RDFSV2", "Invalid"};
  if ((mode >= TopoSortingMode::kInvalid) || (mode < TopoSortingMode::kBFS)) {
    return topo_sorting_mode_strs[static_cast<int32_t>(TopoSortingMode::kInvalid)];
  }
  return topo_sorting_mode_strs[static_cast<size_t>(mode)];
}

enum class WalkStatus { kNotWalked, kWalking, kWalked };
class ComputeGraphImpl {
 public:
  using ConstComputeGraphPtr = std::shared_ptr<ConstComputeGraph>;
  template <class T>
  using Vistor = RangeVistor<T, std::shared_ptr<ConstComputeGraph>>;

  explicit ComputeGraphImpl(const std::string &name);

  ~ComputeGraphImpl() = default;

  std::string GetName() const;
  void SetName(const std::string &name);

  size_t GetAllNodesSize(const ConstComputeGraphPtr &compute_graph) const;
  Vistor<NodePtr> GetAllNodes(const ConstComputeGraphPtr &compute_graph) const;
  Vistor<NodePtr> GetAllNodes(const NodeFilter &node_filter, const GraphFilter &graph_filter,
                              const ConstComputeGraphPtr &compute_graph) const;
  Vistor<NodePtr> AllGraphNodes(std::vector<ComputeGraphPtr> &subgraphs,
                                const ConstComputeGraphPtr &compute_graph) const;
  std::vector<Node *> AllGraphNodesPtr(std::vector<ComputeGraphPtr> &subgraphs) const;
  Vistor<NodePtr> GetNodes(const bool is_unknown_shape, const ConstComputeGraphPtr &compute_graph) const;
  Vistor<NodePtr> GetNodes(const bool is_unknown_shape, const NodeFilter &node_filter, const GraphFilter &graph_filter,
                           const ConstComputeGraphPtr &compute_graph) const;
  size_t GetDirectNodesSize() const;
  Vistor<NodePtr> GetDirectNode(const ConstComputeGraphPtr &compute_graph) const;
  std::vector<Node *> GetDirectNodePtr() const;
  Vistor<NodePtr> GetInputNodes(const ConstComputeGraphPtr &compute_graph) const;
  Vistor<NodePtr> GetOutputNodes(const ConstComputeGraphPtr &compute_graph) const;
  NodePtr FindNode(const std::string &name) const;
  NodePtr FindFirstNodeMatchType(const std::string &type) const;

  bool GraphAttrsAreEqual(const ComputeGraphImpl &r_graph) const;
  bool VectorInputNodePtrIsEqual(const std::vector<NodePtr> &left_nodes, const std::vector<NodePtr> &right_nodes) const;
  bool GraphMembersAreEqual(const ComputeGraphImpl &r_graph) const;

  bool operator==(const ComputeGraphImpl &r_graph) const;

  NodePtr AddNodeFront(const NodePtr node);
  NodePtr AddNodeFront(const OpDescPtr &op, const ComputeGraphPtr &compute_graph);
  NodePtr AddNode(const NodePtr node);
  NodePtr AddNode(const OpDescPtr op, const ComputeGraphPtr &compute_graph);
  NodePtr AddNode(const OpDescPtr op, const int64_t id, const ComputeGraphPtr &compute_graph);

  std::vector<NodePtr> InsertNodes(const NodePtr &node, const std::vector<OpDescPtr> &insert_ops,
                                   const ComputeGraphPtr &compute_graph);

  NodePtr InsertNode(const NodePtr &node, const OpDescPtr &insert_op, const ComputeGraphPtr &compute_graph);

  NodePtr InsertNodeBefore(const NodePtr &node, const OpDescPtr &insert_op, const ComputeGraphPtr &compute_graph);

  static bool IsSupportFuse(const std::vector<NodePtr> &nodes, std::string &reason_not_support);
  std::vector<NodePtr> FuseNodeKeepTopo(const std::vector<NodePtr> &ori_nodes, const std::vector<OpDescPtr> &fusion_ops,
                                        const ComputeGraphPtr &compute_graph);

  NodePtr AddInputNode(const NodePtr node);
  NodePtr AddOutputNode(const NodePtr node);
  NodePtr AddOutputNodeByIndex(const NodePtr node, const int32_t index);

  graphStatus RemoveConstInput(const NodePtr &node);
  graphStatus RemoveNode(const NodePtr &node);
  graphStatus RemoveInputNode(const NodePtr &node);
  graphStatus RemoveOutputNode(const NodePtr &node);

  std::shared_ptr<ComputeGraph> AddSubGraph(const std::shared_ptr<ComputeGraph> &sub_graph);
  graphStatus RemoveSubGraph(const std::shared_ptr<ComputeGraph> &sub_graph);
  graphStatus AddSubgraph(const std::string &name, const std::shared_ptr<ComputeGraph> &subgraph);
  void RemoveSubgraph(const std::string &name);

  std::shared_ptr<ComputeGraph> GetSubgraph(const std::string &name) const;
  std::vector<std::shared_ptr<ComputeGraph>> GetAllSubgraphs() const;
  void SetAllSubgraphs(const std::vector<std::shared_ptr<ComputeGraph>> &subgraphs);

  std::shared_ptr<ComputeGraph> GetParentGraph() const;
  const ComputeGraph *GetParentGraphBarePtr() const;
  void SetParentGraph(const std::shared_ptr<ComputeGraph> &parent);
  std::shared_ptr<Node> GetParentNode() const;
  const Node *GetParentNodeBarePtr() const;
  void SetParentNode(const std::shared_ptr<Node> &parent);
  std::shared_ptr<Node> GetOrUpdateNetOutputNode();
  void SetNetOutputNode(const std::shared_ptr<Node> &netoutput_node);
  const std::map<std::string, std::vector<int32_t>> &GetGraphOutNodes() const {
    return out_nodes_map_;
  }

  void SetOrigGraph(const ComputeGraphPtr &orig_graph) {
    origGraph_ = orig_graph;
  }
  ComputeGraphPtr GetOrigGraph(void) {
    return origGraph_;
  }
  void SetOutputSize(const uint32_t size) {
    output_size_ = size;
  }
  uint32_t GetOutputSize() const {
    return output_size_;
  }
  void SetInputSize(const uint32_t size) {
    input_size_ = size;
  }
  uint32_t GetInputSize() const {
    return input_size_;
  }

  void SetNeedIteration(const bool need_iteration) {
    need_iteration_ = need_iteration;
  }
  bool GetNeedIteration() const {
    return need_iteration_;
  }

  const std::map<std::vector<std::string>, std::vector<std::string>> &GetShareParamLayer() const {
    return params_share_map_;
  }
  void SetShareParamLayer(const std::map<std::vector<std::string>, std::vector<std::string>> &params_share_map) {
    params_share_map_ = params_share_map;
  }

  void SetInputsOrder(const std::vector<std::string> &inputs_order) {
    inputs_order_ = inputs_order;
  }
  void SetGraphOutNodes(const std::map<std::string, std::vector<int32_t>> &out_nodes_map) {
    out_nodes_map_ = out_nodes_map;
  }
  void AppendGraphOutNodes(const std::map<std::string, std::vector<int32_t>> out_nodes_map) {
    for (auto &item : out_nodes_map) {
      (void)out_nodes_map_.emplace(item.first, item.second);
    }
  }

  void SetGraphOpName(const std::map<uint32_t, std::string> &op_name_map) {
    op_name_map_ = op_name_map;
  }
  const std::map<uint32_t, std::string> &GetGraphOpName() const {
    return op_name_map_;
  }
  void SetAllNodesInfo(const std::map<OperatorImplPtr, NodePtr> &nodes) {
    all_nodes_infos_ = nodes;
  }

  void SetGraphOutNodesInfo(const std::vector<std::pair<NodePtr, int32_t>> &out_nodes_info);

  void AppendGraphOutNodesInfo(std::vector<std::pair<NodePtr, int32_t>> &out_nodes_info) {
    (void)output_nodes_info_.insert(output_nodes_info_.cend(), out_nodes_info.cbegin(), out_nodes_info.cend());
  }

  const std::vector<std::pair<NodePtr, int32_t>> &GetGraphOutNodesInfo();

  void SetGraphTargetNodesInfo(const std::vector<NodePtr> &target_nodes_info);
  const std::vector<NodePtr> &GetGraphTargetNodesInfo() const {
    return target_nodes_info_;
  }

  void SetSessionID(const uint64_t session_id) {
    session_id_ = session_id;
  }
  uint64_t GetSessionID() const {
    return session_id_;
  }

  void SetGraphID(const uint32_t graph_id) {
    graph_id_ = graph_id;
  }
  uint32_t GetGraphID() const {
    return graph_id_;
  }

  void SaveDataFormat(const ge::Format data_format) {
    data_format_ = data_format;
  }
  ge::Format GetDataFormat() const {
    return data_format_;
  }
  bool IsSummaryGraph() const {
    return is_summary_graph_;
  }
  void SetSummaryFlag(const bool is_summary_graph) {
    is_summary_graph_ = is_summary_graph;
  }

  graphStatus UpdateInputMapping(const std::map<uint32_t, uint32_t> &input_mapping);
  graphStatus UpdateOutputMapping(const std::map<uint32_t, uint32_t> &output_mapping) const;
  graphStatus ReorderEventNodes(const ConstComputeGraphPtr &compute_graph);
  graphStatus InsertGraphEvents(const ConstComputeGraphPtr &compute_graph);

  graphStatus DFSTopologicalSorting(std::vector<NodePtr> &node_vec, const bool reverse,
                                    const ConstComputeGraphPtr &compute_graph) const;
  graphStatus BFSTopologicalSorting(std::vector<NodePtr> &node_vec, const bool reverse,
                                    const ConstComputeGraphPtr &compute_graph) const;
  /**
   * 从模型输出节点开始反向DFS遍历
   * @param node_vec
   * @param reverse
   * @param compute_graph
   * @return
   */
  graphStatus RDFSTopologicalSorting(std::vector<NodePtr> &node_vec, const bool reverse,
                                     const ConstComputeGraphPtr &compute_graph) const;
  graphStatus RDFSTopologicalSortingV2(std::vector<NodePtr> &node_vec, const bool reverse,
                                       const ConstComputeGraphPtr &compute_graph) const;
  /**
   * 基于调用此接口之前的原始topo顺序，仅对拓扑错误的节点做部分调整，部分调整的算法RDFS
   * @param node_vec
   * @param reverse
   * @param compute_graph
   * @return
   */
  graphStatus StableRDFSTopologicalSorting(std::vector<NodePtr> &node_vec, const bool reverse,
                                           const ConstComputeGraphPtr &compute_graph) const;
  graphStatus CollectBreadthOutNode(const NodePtr &node, std::map<NodePtr, uint32_t> &map_in_edge_num,
                                    std::map<std::string, NodePtr> &breadth_node_map) const;
  void TopologicalSorting(const std::function<bool(const NodePtr &, const NodePtr &)> comp);
  graphStatus TopologicalSorting(const ComputeGraphPtr &const_graph_ptr,
                                 const ConstComputeGraphPtr &const_compute_graph);
  graphStatus TopologicalSortingGraph(const ConstComputeGraphPtr &compute_graph, const bool dfs_reverse = false);
  graphStatus TopologicalSortingGraph(const ConstComputeGraphPtr &compute_graph, TopoSortingMode topo_sorting_mode);
  graphStatus SortNodes(std::vector<NodePtr> &stack, std::map<NodePtr, uint32_t> &map_in_edge_num,
                        const ConstComputeGraphPtr &compute_graph) const;

  size_t GetInEdgeSize(const NodePtr &node) const;
  size_t GetOutEdgeSize(const NodePtr &node) const;

  bool IsValid() const;
  void InValid();
  void Dump(const ConstComputeGraphPtr &graph) const;
  void Swap(ComputeGraphImpl &graph);

  void SetNodesOwner(const ComputeGraphPtr &compute_graph);
  void SetTopParentGraph(const ComputeGraphPtr &compute_graph);
  graphStatus IsolateNode(const NodePtr &node) const;
  graphStatus RemoveExtraOutEdge(const NodePtr &node) const;

  ProtoAttrMap &MutableAttrMap();
  ConstProtoAttrMap &GetAttrMap() const;

  const std::map<OperatorImplPtr, NodePtr> &GetAllNodesInfo() const;
  void SetUserDefOutput(const std::string &output_name);
  const std::string GetOutput();

  void EraseFromNodeList(const std::list<NodePtr>::iterator &position);
  void InsertToNodeList(const std::list<NodePtr>::iterator &position, const NodePtr &node);

  void PushBackToNodeList(const NodePtr &node);

  void EmplaceBackToNodeList(const NodePtr &node);
  void ClearNodeList();
  void ReorderByNodeId();
  graphStatus CreateOrUpdateNetoutput(const ComputeGraphPtr &compute_graph, bool update_data_edge);

 private:
  void inline AddInputDataNode(const NodePtr &node);
  NodePtr CreateNodeFromOpDesc(const OpDescPtr &op_desc, const ComputeGraphPtr &compute_graph, const int64_t topo_id);
  void inline GetAllNodesFromOpdesc(const OpDesc &op_desc, const GraphFilter &graph_filter,
                                    std::deque<NodePtr> &candidates, const NodePtr node) const;
  void inline GetAllNodesFromOpdesc(std::vector<ComputeGraphPtr> &subgraphs, const OpDesc &op_desc,
                                    std::deque<NodePtr> &candidates) const;
  void inline GetAllNodesPtrFromOpdesc(std::vector<ComputeGraphPtr> &subgraphs, const OpDesc &op_desc,
                                       std::deque<Node *> &candidates) const;

  template <typename AnchorPtr>
  void inline GetOutNodesFromAnchor(const AnchorPtr &peer_in_anchor, std::map<NodePtr, uint32_t> &map_in_edge_num,
                                    std::vector<NodePtr> &out_nodes) const {
    const auto iter = map_in_edge_num.find(peer_in_anchor->GetOwnerNode());
    if (iter != map_in_edge_num.end()) {
      --iter->second;
      if (iter->second == 0U) {
        out_nodes.push_back(peer_in_anchor->GetOwnerNode());
      }
    }
  }
  graphStatus DoTopologicalSorting(const ConstComputeGraphPtr &compute_graph, TopoSortingMode sorting_mode,
                                   bool dfs_reverse);

 private:
  // 该private用于生成Netoutput节点
  struct RetvalInfo {
    NodePtr output_node;
    int32_t node_output_index;
    int32_t parent_node_index;
  };
  graphStatus AddNetOutputNodeToGraph(const ComputeGraphPtr &compute_graph, NodePtr &output_node);
  graphStatus CreateNetOutputNode(OpDescPtr &net_output_desc) const;
  graphStatus CollectOutputNode(const ComputeGraphPtr &compute_graph, std::vector<RetvalInfo> &output_nodes_info);
  graphStatus GetRetvalOutputInfo(const NodePtr &node, std::map<int32_t, RetvalInfo> &retval_node_index_map);
  graphStatus CheckOutputNodeInfo(const std::vector<RetvalInfo> &outputs) const;

  graphStatus AddCtrlEdgeForTargets(const NodePtr &net_out_node);
  graphStatus AddInOutForNetOutputOp(const OpDescPtr &net_output_desc, std::vector<RetvalInfo> &output_nodes_info);
  graphStatus AddDataEdgesForNetOutput(const ComputeGraphPtr &compute_graph, const NodePtr &net_out_node,
                                       const std::vector<RetvalInfo> &output_nodes_info);
  graphStatus RemoveUnusedRetvalNode(const ComputeGraphPtr &compute_graph);
  graphStatus UpdateNetOutput(const ComputeGraphPtr &compute_graph, const NodePtr &output_node, bool update_data_edge);
  graphStatus UpdateNetOutputDesc(const NodePtr &net_output) const;
  graphStatus UpdateNetOutputParentNodeIndex(const NodePtr &net_output,
                                             const std::vector<RetvalInfo> &output_nodes_info) const;
  graphStatus UnLinkAnchorsOfNetoutput(const NodePtr &net_out_node);
  graphStatus UnLinkDataAnchorOfNetoutput(const NodePtr &net_out_node);
  graphStatus UnLinkControlAnchorOfNetoutput(const NodePtr &net_out_node);
  bool CheckNodeIsInOutputNodes(const NodePtr &node) const;

  bool is_include_special_node_ = false;
  std::set<NodePtr> targets_;
  std::set<NodePtr> old_targets_;  // 在多次SetGraphTargetNodesInfo中保存上一次的信息，用于删除上一次的控制边

 private:
  friend class ModelSerializeImp;
  friend class GraphUtils;
  friend class ExecuteGraphAdapter;
  std::string name_;
  std::list<NodePtr> nodes_;
  uint32_t graph_id_ = 0U;
  AttrStore attrs_;
  size_t direct_nodes_size_ = 0UL;
  std::map<OperatorImplPtr, NodePtr> all_nodes_infos_;
  std::vector<NodePtr> target_nodes_info_;

  std::vector<NodePtr> input_nodes_;
  std::vector<std::string> inputs_order_;
  uint32_t input_size_ = 1U;
  std::map<std::string, std::vector<int32_t>> out_nodes_map_;
  uint32_t output_size_ = 1U;
  std::vector<std::pair<NodePtr, int32_t>> output_nodes_info_;
  std::map<int32_t, RetvalInfo> retval_node_index_map_;

  std::vector<std::shared_ptr<ComputeGraph>> sub_graph_;
  std::map<std::string, std::shared_ptr<ComputeGraph>> names_to_subgraph_;
  std::weak_ptr<ComputeGraph> parent_graph_;
  std::weak_ptr<Node> parent_node_;

  // the members followed should not in the ComputeGraph class
  bool is_valid_flag_;
  bool is_summary_graph_ = false;
  // Indicates whether it is need iteration
  bool need_iteration_ = false;
  std::map<std::vector<std::string>, std::vector<std::string>> params_share_map_;
  // TaskIdx -> op_name Map
  std::map<uint32_t, std::string> op_name_map_;
  uint64_t session_id_ = 0UL;
  ge::Format data_format_ = ge::FORMAT_ND;
  // Graph Before BFE
  ComputeGraphPtr origGraph_;
  std::weak_ptr<Node> graph_netoutput_;
  Node *parent_node_bare_ptr_ = nullptr;
  ComputeGraph *parent_graph_bare_ptr_ = nullptr;
};
}  // namespace af
#endif  // GRAPH_COMPUTE_GRAPH_IMPL_H_
