/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_UTILS_FAST_NODE_UTILS_H
#define INC_GRAPH_UTILS_FAST_NODE_UTILS_H

#include "graph/fast_graph/fast_node.h"
#include "graph/fast_graph/edge.h"
#include "graph/fast_graph/execute_graph.h"
#include "graph/ge_error_codes_af.h"

namespace af {
class FastNodeUtils {
 public:
  // node utils
  /**
   * @brief 获取给定节点的父节点的输入。
   *
   * @param node 指向待查询节点的指针。
   * @return 返回指向父节点的输入的指针，如果父节点不存在，则返回 nullptr。
   */
  static FastNode *GetParentInput(const FastNode *const node);

  /**
   * @brief 获取指定索引处的输入数据节点。
   *
   * @param node 指向要访问其输入数据节点的指针。
   * @param index 要检索的输入数据节点的索引。
   * @return 如果找到指定索引处的输入数据节点，则返回该节点的指针；否则返回 nullptr。
   */
  static FastNode *GetInDataNodeByIndex(const FastNode *const node, const int32_t index);

  /**
   * @brief 获取给定节点是否为常量 Op 节点。
   *
   * @param node 指向待查询节点的指针。
   * @return 如果给定节点是常量 Op 节点，则返回 true；否则返回 false。
   */
  static bool GetConstOpType(const FastNode *const node);

  // subgraph utils
  /**
   * @brief 向给定节点追加子图。
   *
   * @param node 指向待添加子图的节点的指针。
   * @param subgraph_name 子图的名称。
   * @param subgraph 指向待添加的子图的智能指针，子图生命周期由 root_graph 管理。
   * @return 返回添加子图的状态。
   */
  static graphStatus AppendSubgraphToNode(FastNode *const node, const std::string &subgraph_name,
                                          const ExecuteGraphPtr &subgraph);

  /**
   * @brief 获取给定节点的指定索引处的子图。
   *
   * @param node 指向待查询子图的节点的指针。
   * @param index 子图的索引。
   * @return 返回指向子图的指针，如果索引超出范围或者子图不存在，则返回 nullptr。
   */
  static ExecuteGraph *GetSubgraphFromNode(const FastNode *const node, const uint32_t index);

  /**
   * @brief 在给定节点的指定索引 index 处挂载子图。若原先存在子图，该接口会替换 node 下索引为 index
   * 的子图，但是不会从root_graph中移除原子图。 若需移除原子图，建议调用ExecuteGraph::RemoveSubGraph。
   *
   * @param node 指向待设置子图的节点的指针。
   * @param index 子图的索引。
   * @param subgraph 指向待设置的子图的智能指针。
   * @return 返回设置子图的状态。
   */
  static graphStatus MountSubgraphToNode(FastNode *const node, const uint32_t index, const ExecuteGraphPtr &subgraph);

  // edge utils
  /**
   * @brief 向给定节点追加输入边信息，直至输入边信息数量达到 num。
   * 注意：该接口不会实际建立输入边，若需要为节点连边，建议调用ExecuteGraph::AddEdge。
   *
   * @param node 待追加输入边信息的节点的指针。
   * @param num 追加操作后，node 所拥有的输入边信息数量。
   * @return 返回追加输入边信息的状态。
   */
  static graphStatus AppendInputEdgeInfo(FastNode *const node, const uint32_t num);

  /**
   * @brief 向给定节点追加输出边信息，直至输出边信息数量达到 num。
   * 注意：该接口不会实际建立输出边，若需要为节点连边，建议调用ExecuteGraph::AddEdge。
   *
   * @param node 待追加输出边信息的节点的指针。
   * @param num 追加操作后，node 所拥有的输出边信息数量。
   * @return 返回追加输出边信息的状态。
   */
  static graphStatus AppendOutputEdgeInfo(FastNode *const node, const uint32_t num);

  /**
   * @brief 清除给定 OpDesc 的指定索引处的 InputDesc。
   *
   * @param op_desc OpDesc 的指针。
   * @param index 要清除的 InputDesc 的索引。
   * @return 如果成功清除 InputDesc，则返回 true；否则返回 false。
   */
  static bool ClearInputDesc(const OpDesc *const op_desc, const uint32_t index);

  /**
   * @brief 移除给定节点的输入边信息，直至输入边信息数量减少到 num。
   * 注意：该接口不会从执行图中移除输入边，若需要移除输入边，建议调用ExecuteGraph::RemoveEdge。
   *
   * @param node 待移除输入边信息的节点的指针。
   * @param num 移除操作后，node 所拥有的输入边信息数量。
   * @return 返回追加输出边信息的状态。
   */
  static graphStatus RemoveInputEdgeInfo(FastNode *const node, const uint32_t num);

  /**
   * @brief 断开给定节点与其所有相连节点之间的输入边和输出边。
   *
   * @param node 目标节点。
   */
  static void UnlinkAll(FastNode *const node);

  /**
   * @brief 获取给定边的输入端点，包含 dst 节点指针和输入 index。
   * (SrcNode:[OutEndpoint])->Edge->([InEndpoint]:DstNode)
   *
   * @param edge 指向待查询输入端点的边的指针。
   * @return 返回输入端点。
   */
  static EdgeDstEndpoint GetDstEndpoint(const FastEdge *const edge);

  /**
   * @brief 获取给定边的输出端点，包含 src 节点指针和输出 index。
   * (SrcNode:[OutEndpoint])->Edge->([InEndpoint]:DstNode)
   *
   * @param edge 指向待查询输出端点的边的指针。
   * @return 返回输出端点。
   */
  static EdgeSrcEndpoint GetSrcEndpoint(const FastEdge *const edge);
};

struct FastNodeCompareKey {
  bool operator()(const FastNode *const n0, const FastNode *const n1) const {
    if ((n0 == nullptr) || (n1 == nullptr)) {
      return false;
    }
    if (n0->GetName() == n1->GetName()) {
      const ExtendInfo *const extend_info0 = n0->GetExtendInfo();
      const ExtendInfo *const extend_info1 = n1->GetExtendInfo();
      if ((extend_info0 == nullptr) || (extend_info1 == nullptr)) {
        return false;
      }
      const ExecuteGraph *const g0 = extend_info0->GetOwnerGraphBarePtr();
      const ExecuteGraph *const g1 = extend_info1->GetOwnerGraphBarePtr();
      if ((g0 == nullptr) || (g1 == nullptr)) {
        return false;
      }
      return (g0->GetName() < g1->GetName());
    }
    return (n0->GetName() < n1->GetName());
  }
};
}  // namespace af

#endif  // INC_GRAPH_UTILS_FAST_NODE_UTILS_H
