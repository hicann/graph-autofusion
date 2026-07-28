/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef D_INC_GRAPH_FAST_NODE_H
#define D_INC_GRAPH_FAST_NODE_H

#include <map>
#include <mutex>
#include <set>
#include <vector>
#include "graph/op_desc.h"
#include "graph/fast_graph/edge.h"

namespace af {
struct OutDataEdgeStatisticsInfo {
  size_t total_num = 0UL;             // The total number edge of all inputs or all outputs
  std::vector<size_t> per_edges_num;  // The number of one input or one output.
};
constexpr uint64_t kInvalidSymbol = UINT64_MAX;

class ExecuteGraph;
class FastNode;
using FastEdge = Edge<FastNode>;

class ExtendInfo {
 public:
  virtual ~ExtendInfo() {}
  /**
   * set the input index of graph.
   */
  void SetInputIndex(int32_t idx);

  /**
   * get the input index of graph.
   * Don`t use this function unless it is explicitly required.
   */
  int32_t GetInputIndex() const;

  /**
   * add the output index of graph.
   * Don`t use this function unless it is explicitly required.
   */
  void AddOneOutputIndex(int32_t idx);

  /**
   * get the output index of graph.
   * Don`t use this function unless it is explicitly required.
   */
  std::vector<int32_t> &GetOutputIndex();

  /**
   * get the owner graph of node.
   */
  ExecuteGraph *GetOwnerGraphBarePtr() const;

  /**
   * set the owner graph of node.
   */
  graphStatus SetOwnerGraph(ExecuteGraph *const graph, const FastNode *const fast_node);

  /**
   * check the extend information is same with r_info.
   */
  bool operator==(const ExtendInfo &r_info) const;

  /**
   * get the flag of host node.
   */
  bool GetHostNode() const;

  /**
   * set the flag of host node.
   */
  void SetHostNode(const bool is_host);

  /**
   * clear members.
   */
  void Clear();

  /**
   * update size of input_symbols.
   */
  void UpdateInputSymbols(size_t data_in_num);

  /**
   * update size of output_symbols.
   */
  void UpdateOutputSymbols(size_t data_out_num);

  /**
   * set symbol of the idx data input
   */
  graphStatus SetInputSymbol(size_t idx, uint64_t symbol);

  /**
   * set symbol of the idx data output
   */
  graphStatus SetOutputSymbol(size_t idx, uint64_t symbol);

  /**
   * get symbol of the idx data input
   */
  uint64_t GetInputSymbol(size_t idx);

  /**
   * get symbol of the idx data output
   */
  uint64_t GetOutputSymbol(size_t idx);

 private:
  bool IsDataIndexValid(size_t idx, const std::vector<uint64_t> &symbols) const;
  ExecuteGraph *execute_graph_ = nullptr;
  std::vector<int32_t> output_index_;
  int32_t input_index_ = kControlEdgeIndex;
  bool host_node_ = false;
  std::vector<uint64_t> input_symbols_{};
  std::vector<uint64_t> output_symbols_{};
};

class FastNode {
 public:
  /**
   * construct a fastnode.
   * please call Init after construction
   */
  FastNode();

  ~FastNode();
  /**
   * The function is used to init node with opdesc.
   */
  graphStatus Init(const OpDescPtr &op);

  /**
   * get the bare pointer of op desc.
   */
  OpDesc *GetOpDescBarePtr() const;

  /**
   * get the shared pointer of op desc.
   */
  OpDescPtr GetOpDescPtr() const;

  /**
   * get the type of node.
   */
  std::string GetType() const;

  /**
   * get the type of node.
   */
  const char *GetTypePtr() const;

  /**
   * get the name of node.
   */
  std::string GetName() const;

  /**
   * get the name of node.
   */
  const char *GetNamePtr() const;

  /**
   * record the edge info to node.
   * The function is not recommended, please used the AddEdge function instead.
   */
  graphStatus RecordEdge(Edge<FastNode> *const edge, DirectionType type);

  /**
   * clear the edge info to node.
   * The function is not recommended, please used the RemoveEdge function instead.
   */
  graphStatus EraseEdge(const Edge<FastNode> *const edge, DirectionType type);

  /**
   * adjust the position of the edge in the node record.
   */
  graphStatus MoveEdge(DirectionType type, int32_t io_idx, int32_t cur_array_index, int32_t replace_array_index);

  /**
   * get a unique identifier of node.
   * please notes the unique identifier is in the graph.
   */
  size_t GetNodeToken() const;

  /**
   * get the number of input data in the node.
   */
  size_t GetDataInNum() const;

  /**
   * get the number of output data in the node.
   */
  size_t GetDataOutNum() const;

  /**
   * update the number of data input in the node.
   */
  void UpdateDataInNum(size_t new_num);

  /**
   * update the number of data output in the node.
   */
  void UpdateDataOutNum(size_t new_num);

  /**
   * get the total number of output edges from the node.
   */
  size_t GetAllOutEdgesSize() const;
  size_t GetAllOutDataEdgesSize() const;
  size_t GetAllOutControlEdgesSize() const;

  /**
   * get the total number of input edges from the node.
   */
  size_t GetAllInDataEdgesSize() const;
  size_t GetAllInControlEdgesSize() const;

  /**
   * check the node is same with r_node.
   */
  bool operator==(const FastNode &r_node) const;

  /**
   * get the total number of in edge from the node.
   * the number include data edge and control edge.
   */
  size_t GetAllInEdgeSize() const;

  /**
   * collecting all input edge.
   * please check the item, the item from vector may be nullptr.
   * if the item is nullptr, it just continue to get next, no error handing is required.
   */
  const std::vector<Edge<FastNode> *> &GetAllInDataEdgesRef() const;

  /**
   * collecting all output edge.
   * please check the item, the item from vector may be nullptr.
   * if the item is nullptr, it just continue to get next, no error handing is required.
   */
  const std::vector<Edge<FastNode> *> &GetAllOutControlEdgesRef() const;
  const std::vector<std::vector<Edge<FastNode> *>> &GetAllOutDataEdgesRef() const;

  /**
   * collecting all output or input edge.
   * it already filter the nullptr item.
   */
  std::vector<Edge<FastNode> *> GetAllOutDataEdges() const;
  std::vector<Edge<FastNode> *> GetAllOutControlEdges() const;
  std::vector<Edge<FastNode> *> GetAllInDataEdges() const;
  std::vector<Edge<FastNode> *> &MutableAllInDataEdges();

  /**
   * collecting input control edges with input index.
   * please check the item, the item from vector may be nullptr.
   * if the item is nullptr, it just continue to get next, no error handing is required.
   */
  std::vector<Edge<FastNode> *> GetAllInControlEdges() const;
  const std::vector<Edge<FastNode> *> &GetAllInControlEdgesRef() const;

  /**
   * Check the number of out edge is zero.
   */
  bool OutNodesIsEmpty() const;

  /**
   * Set the relative node information.
   * Don`t use this function unless it is explicitly required.
   */
  void SetNodePtr(const std::shared_ptr<Node> &node);

  /**
   * clear the relative node information.
   * Don`t use this function unless it is explicitly required.
   */
  void ClearNodePtr();
  void ClearNodeBarePtr();

  /**
   * get the relative node information.
   * Don`t use this function unless it is explicitly required.
   */
  std::shared_ptr<Node> GetNodePtr() const;
  Node *GetNodeBarePtr() const;

  /**
   * get the total number of edge with input index.
   */
  size_t GetInEdgesSizeByIndex(int32_t idx) const;

  /**
   * get the total number of edge with output index.
   */
  size_t GetOutEdgesSizeByIndex(int32_t idx) const;

  /**
   * collecting input data edge with input index.
   * please check the item, the item from vector may be nullptr.
   * if the item is nullptr, it just continue to get next, no error handing is required.
   */
  Edge<FastNode> *GetInDataEdgeByIndex(int32_t idx) const;

  bool IsDirectlyControlledByNode(FastNode const *node) const;

  /**
   * collecting all output edge with output index.
   * please check the item, the item from vector may be nullptr.
   * if the item is nullptr, it just continue to get next, no error handing is required.
   */
  std::vector<Edge<FastNode> *> GetOutEdgesByIndex(int32_t idx) const;
  const std::vector<Edge<FastNode> *> &GetOutEdgesRefByIndex(int32_t idx) const;

  /**
   * remove all of edge in the node.
   * please define remove_edge_func to delete edge.
   * The general process is as follow:
   *    1. clear the edge information in src node and dst node (use EraseEdge);
   *    2. remove edge in container.
   *    3. add the free edge to free container.
   * example:
   *   node[1]->RemoveAllEdge([&compute_graph](FastEdge *e) {
   *     auto src_node = e->src;
   *     auto dst_node = e->dst;
   *     Utils::GetNode(src_node).EraseEdge(e, DirectionType::kDirectionOutType);
   *     Utils::GetNode(dst_node).EraseEdge(e, DirectionType::kDirectionInType);
   *     if (Utils::GetListElementAddr(e)->owner != nullptr) {
   *       Utils::GetListElementAddr(e)->owner->erase(Utils::GetListElementAddr(e));
   *     }
   *     auto ret = compute_graph->RecycleQuickEdge(e);
   *     if ((ret != GRAPH_SUCCESS) && (e != nullptr)) {
   *       delete e;
   *     }
   *   });
   */
  void RemoveAllEdge(std::function<void(Edge<FastNode> *)> const &remove_edge_func);

  /**
   * get the extend info of node.
   */
  ExtendInfo *GetExtendInfo() const;

  /**
   * get the numbers input edges which peer node is not NEXTITERATION or REFNEXTITERATION.
   */
  size_t GetInEdgeSize() const;

  void UpdateOpDesc(const OpDescPtr &new_opdesc);

  /**
   * get peer nodes from all input data edges.
   */
  std::vector<FastNode *> GetInDataNodes() const;

  /**
   * get peer nodes from out data edges with index.
   */
  std::vector<FastNode *> GetOutDataNodesByIndex(int32_t index) const;

  /**
   * get peer nodes from all out data edges.
   */
  std::vector<FastNode *> GetOutDataNodes() const;

  /**
   * get peer nodes from all out control edges.
   */
  std::vector<FastNode *> GetOutControlNodes() const;

  /**
   * get peer nodes from all in control edges.
   */
  std::vector<FastNode *> GetInControlNodes() const;

  /**
   * get peer nodes from all out edges.
   */
  std::vector<FastNode *> GetAllOutNodes() const;

  /**
   * get peer nodes from all in edges.
   */
  std::vector<FastNode *> GetAllInNodes() const;

 private:
  graphStatus CheckAllInputParamter(DirectionType type, int32_t io_idx, int32_t cur_array_index,
                                    int32_t replace_array_index) const;
  inline bool CheckDataIndexIsValid(int32_t index, DirectionType type) const;
  graphStatus Reset();
  void UpdateDataForIoNumChange();
  graphStatus RecordInControlEdge(FastEdge *const edge);
  graphStatus RecordOutControlEdge(FastEdge *const edge);
  graphStatus RecordInDataEdge(FastEdge *const edge, int32_t index);
  graphStatus RecordOutDataEdge(FastEdge *const edge, int32_t index);
  graphStatus EraseInControlEdge(const FastEdge *const edge);
  graphStatus EraseOutControlEdge(const FastEdge *const edge);
  graphStatus EraseInDataEdge(const FastEdge *const edge);
  graphStatus EraseOutDataEdge(const FastEdge *const edge, int32_t index);
  graphStatus ModifySizeByNodeType(const FastEdge *const fast_edge, size_t &in_edge_size) const;

 private:
  std::string name_;
  size_t node_token_ = 0UL;
  OpDescPtr opdesc_ = nullptr;
  std::shared_ptr<Node> self_ptr_ = nullptr;
  Node *node_bare_ptr_ = nullptr;

  size_t data_in_num_ = 0UL;
  size_t data_out_num_ = 0UL;

  mutable std::vector<Edge<FastNode> *> in_data_edges_;
  std::vector<Edge<FastNode> *> in_control_edges_;
  std::vector<Edge<FastNode> *> out_control_edges_;
  std::vector<std::vector<Edge<FastNode> *>> out_data_edges_;

  size_t in_data_edges_count_ = 0UL;
  size_t in_control_edge_count_ = 0UL;
  size_t out_control_edges_count_ = 0UL;
  OutDataEdgeStatisticsInfo out_data_edges_info_;

  std::unique_ptr<ExtendInfo> extend_info_ = nullptr;
};

}  // namespace af
#endif  // D_INC_GRAPH_FAST_NODE_H
