/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_V2_VECTOR_FUNC_PARTITIONER_H
#define OPTIMIZE_PLATFORM_V2_VECTOR_FUNC_PARTITIONER_H

#include "graph/compute_graph.h"
#include "cluster.h"
#include "ascir_ops.h"
#include "cluster_dict.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "ascir_register.h"

namespace optimize {
class VectorFuncPartitioner {
 public:
  explicit VectorFuncPartitioner(af::AscGraph &impl_graph) : impl_graph_(impl_graph) {};
  // VF input graph node names must be unique to keep node and boundary-anchor ordering deterministic.
  af::Status Partition();

 private:
  using InsertOrderMap = std::vector<std::pair<af::OutDataAnchorPtr, std::vector<af::InDataAnchorPtr>>>;
  void DebugMergeLog() const;
  af::Status InitClusters();
  ClusterPtr CreateAndInitCluster(const af::AscNodePtr &node, size_t &rank);
  void EstablishClusterConnections(ClusterPtr &cluster, const af::AscNodePtr &node);
  void FixAllCompareClusterConnections();
  void RefineEnableVFFlag(const af::AscNodePtr &node, bool &enable_vf) const;
  static bool HasReduceNodeInGraph(const af::AscGraph &impl_graph);
  static af::Status InitClusterAttr(const std::unique_ptr<af::ascir::AscIrCodegen> &codegen_impl,
                                    const af::AscNodePtr &node, ClusterPtr &cluster);
  af::Status MergeClusters();
  static bool CanMergeClusters(const Cluster &from, const Cluster &to);
  af::Status SortClustersForBuildSubgraph();
  af::Status BuildSubgraphs();
  af::Status ModifySubgraphAttrs(af::AscGraph &vf_graph);
  static af::Status SetSubGraphAttrs(af::AscGraph &vf_graph);
  af::Status MergeContinuousVectorAxis(af::AscGraph &vf_graph);
  af::Status BuildSubgraph(const ClusterPtr &cluster, af::AscGraph &vf_graph, af::ascir_op::VectorFunc &vf_op);
  static af::Status InsertDataAndLoadNode(af::AscGraph &asc_graph, const af::OutDataAnchorPtr &out_anchor,
                                          const std::vector<af::InDataAnchorPtr> &in_anchors, int64_t parent_in_index);
  static af::Status InsertScalarNode(af::AscGraph &asc_graph, const af::OutDataAnchorPtr &out_anchor,
                                     const std::vector<af::InDataAnchorPtr> &in_anchors, int64_t parent_in_index);
  static af::Status InsertStoreAndOutputNode(af::AscGraph &asc_graph, af::AscNode &pre_node, size_t out_anchor_index,
                                             int64_t parent_out_index);

  static af::Status TopologicalSortingForVfGraph(af::AscGraph &graph);

  static af::Status ReorderAxesForBrcInline(const af::AscGraph &graph);

  static af::Status AddRemovePadForBrcInline(af::AscGraph &graph);

  static af::Status AddInputDataAnchors(const af::NodePtr &node, InsertOrderMap &out_data_to_peer_in_anchors);

  static af::Status AddOutputDataAnchors(const af::NodePtr &node, InsertOrderMap &out_data_to_peer_in_anchors);

  static bool HasDetectedCycle(const Cluster *const src, const Cluster *const dst);
  static bool IsCompareOp(const af::AscNodePtr &node);
  void FixCompareClusterConnections(const ClusterPtr &cluster, const af::AscNodePtr &compare_node);

  af::AscGraph &impl_graph_;
  af::ComputeGraphPtr root_graph_;
  std::vector<af::AscGraph> sub_graphs_;
  // need merged axis id to new axis id
  std::map<std::vector<af::AxisId>, af::AxisPtr> from_id_to_merged_axis_;
  // dictionary for node->cluster
  ClusterDict cluster_dict_;
  size_t subgraph_id_ = 0UL;
  bool graph_has_reduce_node_ = false;  // 缓存图是否有reduce节点
};
}  // namespace optimize
#endif  // OPTIMIZE_PLATFORM_V2_VECTOR_FUNC_PARTITIONER_H
