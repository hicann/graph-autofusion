/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_EXECUTE_GRAPH_H
#define INC_GRAPH_EXECUTE_GRAPH_H

#include <deque>
#include <queue>
#include <unordered_map>
#include "graph/attr_store.h"
#include "graph/fast_graph/fast_node.h"
#include "graph/fast_graph/list_element.h"
#include "graph/attr_store.h"

namespace af {
template <class T, class G>
class FastGraphImpl;

using FastNodeFilter = std::function<bool(const FastNode *)>;

class ExecuteGraph : public std::enable_shared_from_this<ExecuteGraph>, public AttrHolder {
 public:
  struct SubGraphInfo {
    std::shared_ptr<ExecuteGraph> sub_graph;
    ListElement<ExecuteGraph *> *quick_graph;
  };

  explicit ExecuteGraph(const std::string &name);
  ~ExecuteGraph() override {};

  /**
   * The function is shallow copy for ExecuteGraph
   */
  ExecuteGraph &operator=(af::ExecuteGraph &exec_graph);

  /**
   * The function is deep copy for ExecuteGraph
   */
  ExecuteGraph &CompleteCopy(af::ExecuteGraph &exec_graph);

  /**
   * The function is used to add node of graph.
   * The node push back to container in graph.
   */
  FastNode *AddNode(const OpDescPtr &op);
  FastNode *AddNode(FastNode *const fast_node);
  FastNode *AddNode(const OpDescPtr &op, int64_t id);

  /**
   * The function is used to add node of graph.
   * The node push front to container in graph.
   */
  FastNode *AddNodeFront(const OpDescPtr &op);
  FastNode *AddNodeFront(FastNode *const fast_node);

  /**
   * The function is used to remove node of graph.
   * The node don`t release and it will push to free container which is used to store free obj.
   */
  graphStatus RemoveJustNode(const FastNode *const fast_node);

  /**
   * The function is used to add input node of graph.
   */
  FastNode *AddInputNode(FastNode *const fast_node);

  /**
   * The function is used to remove input node of graph.
   */
  graphStatus RemoveInputNode(FastNode *const fast_node);

  /**
   * The function is used to add output node of graph.
   */
  FastNode *AddOutputNodeByIndex(FastNode *const fast_node, int32_t index);

  /**
   * The function is used to remove output node of graph.
   */
  graphStatus RemoveOutputNode(const FastNode *const fast_node);

  /**
   * The function is used to add edge of graph.
   */
  FastEdge *AddEdge(FastNode *const src, int32_t src_index, FastNode *const dst, int32_t dst_index);

  /**
   * The function is used to remove edge of graph.
   * The edge don`t release and it will push to free container which is used to store free obj.
   */
  graphStatus RemoveEdge(const FastEdge *const edge);

  const FastNode *GetParentNodeBarePtr() const;
  FastNode *GetParentNodeBarePtr();
  void SetParentNode(FastNode *const node);

  /**
   * The function is used to directly add subgraph of graph without any check.
   * The shared pointer of subgraph will record in graph.
   */
  ExecuteGraph *AddSubGraph(const std::shared_ptr<ExecuteGraph> &sub_graph);

  /**
   * The function will add subgraph After strict checking the valid of subgraph.
   * The shared pointer of subgraph will record in graph.
   */
  ExecuteGraph *AddSubGraph(const std::shared_ptr<ExecuteGraph> &sub_graph_ptr, const std::string &name);

  /**
   * The function is used to remove subgraph of graph.
   * The shared pointer of subgraph will clear in graph.
   */
  graphStatus RemoveSubGraph(const ExecuteGraph *const sub_graph);
  graphStatus RemoveSubGraph(const std::string &name);

  /**
   * The function is used to get subgraph with name.
   */
  ExecuteGraph *GetSubGraph(const std::string &name) const;

  /**
   * remove all subgraph from parent graph.
   */
  void ClearAllSubGraph();

  /**
   * get the number of direct nodes form graph.
   */
  size_t GetDirectNodesSize() const;

  /**
   * get direct nodes from graph (it is convert to vector which is long time).
   * external modifications don`t affect internal nodes.
   */
  std::vector<FastNode *> GetDirectNode() const;

  /**
   * get all edges from graph (it is convert to vector which is long time).
   * external modifications don`t affect internal edges.
   */
  std::vector<FastEdge *> GetAllEdges() const;

  /**
   * get all sub graph from graph (it is convert to vector which is long time).
   * external modifications don`t affect internal edges.
   */
  std::vector<ExecuteGraph *> GetAllSubgraphs() const;

  /**
   * find the node with node token in the graph.
   */
  const FastNode *FindNode(size_t token) const;

  /**
   * is is topo sort (include dfs, bfs, DFS_POSTORDER).
   */
  graphStatus TopologicalSortingGraph(const ExecuteGraph *const execute_graph, const bool dfs_reverse);

  /**
   * get name of graph.
   */
  std::string GetName() const;

  /**
   * set name of graph.
   */
  void SetName(const std::string &name);

  void SetParentGraph(ExecuteGraph *const parent_graph);

  const ExecuteGraph *GetParentGraphBarePtr(void) const;
  ExecuteGraph *GetParentGraphBarePtr(void);

  /**
   * topo sort in the graph (include sub graph).
   */
  graphStatus TopologicalSorting();

  /**
   * push edge to free edge.
   */
  graphStatus RecycleQuickEdge(const FastEdge *const fast_edge);

  /**
   * push node to free edge.
   */
  graphStatus RecycleQuickNode(const FastNode *const fast_node);

  /**
   * get all of nodes in graph (include subgraph).
   */
  std::vector<FastNode *> GetAllNodes() const;
  std::vector<FastNode *> GetAllNodes(const FastNodeFilter &fast_node_filter) const;

  /**
   * It is used to set input order which is used in topo sorting
   */
  void SetInputsOrder(const std::vector<std::string> &inputs_order);

  void ReorderByNodeId();

  void SetGraphId(size_t graph_id);

  size_t GetGraphId() const;

  bool CheckNodeIsInGraph(const FastNode *const node) const;

  bool CheckEdgeIsInGraph(const FastEdge *const edge) const;

  /**
   * The edge belong to graph.
   * sometime, we need to change the owner of edge to correct graph.
   */
  graphStatus MoveEdgeToGraph(const FastEdge *const edge);

 protected:
  ProtoAttrMap &MutableAttrMap() override;
  ConstProtoAttrMap &GetAttrMap() const override;

 private:
  std::vector<FastNode *> AllGraphNodes(std::vector<std::shared_ptr<ExecuteGraph>> &subgraphs,
                                        const FastNodeFilter &fast_node_filter) const;
  void GetAllNodesFromOpdesc(std::vector<std::shared_ptr<ExecuteGraph>> &subgraphs, const OpDesc &op_desc,
                             std::deque<FastNode *> &candidates) const;
  void RemoveNodeFromNodesFree(const FastNode *const fast_node) const;
  graphStatus SortNodes(std::vector<FastNode *> &stack, std::map<FastNode *, uint32_t> &map_in_edge_num) const;
  void GetOutNodesFromEdgesToMap(std::map<FastNode *, uint32_t> &map_in_edge_num, FastNode *node,
                                 std::map<std::string, FastNode *> &breadth_node_map) const;
  graphStatus CollectBreadthOutNode(const FastNode *const node, std::map<FastNode *, uint32_t> &map_in_edge_num,
                                    std::map<std::string, FastNode *> &breadth_node_map) const;
  graphStatus BFSTopologicalSorting(std::vector<FastNode *> &node_vec, const bool reverse,
                                    const ExecuteGraph *const compute_graph) const;
  graphStatus DFSTopologicalSorting(std::vector<FastNode *> &node_vec, const bool reverse,
                                    const ExecuteGraph *const compute_graph) const;
  graphStatus RDFSTopologicalSorting(std::vector<FastNode *> &node_vec, const bool reverse,
                                     const ExecuteGraph *const compute_graph) const;
  void GetInNodes(const FastNode *const current, std::vector<FastNode *> &input_nodes) const;

 private:
  std::shared_ptr<FastGraphImpl<FastNode, ExecuteGraph>> graph_shared_;
  std::unordered_map<std::string, SubGraphInfo> names_to_subgraph_;
  std::vector<std::string> inputs_order_;
  AttrStore attrs_;

  friend class ExecuteGraphAdapter;
  friend class ExecuteGraphUtils;
};
using ExecuteGraphPtr = std::shared_ptr<ExecuteGraph>;
}  // namespace af
#endif  // INC_GRAPH_EXECUTE_GRAPH_H
