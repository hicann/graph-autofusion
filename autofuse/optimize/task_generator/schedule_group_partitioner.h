/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_SCHEDULE_GROUP_PARTITIONER_H_
#define ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_SCHEDULE_GROUP_PARTITIONER_H_

#include "ascir.h"
#include "ascgen_log.h"

namespace optimize {
struct MergeableGraphs {
  std::vector<size_t> graph_indices;
  std::vector<size_t> node_counts;
};

struct MergePlan {
  std::vector<std::vector<size_t>> merge_groups;
};

class ScheduleGroupGraphPartitioner {
 public:
  static Status PartitionByConnectivity(const ::ascir::ImplGraph &optimize_graph,
                                        std::vector<::ascir::ImplGraph> &sub_optimize_graphs,
                                        std::vector<af::AscNodePtr> node_order = {});
  static Status NeedRefreshAxisSize(const ::ascir::ImplGraph &optimize_graph, bool &need_refresh);
  static Status RefreshAxisSize(const ::ascir::ImplGraph &sub_graph);
  static Status ReduceGraphCount(std::vector<::ascir::ImplGraph> &grouped_graphs, size_t target_count = 5);

 private:
  static std::vector<MergeableGraphs> FindMergeableGraphs(const std::vector<::ascir::ImplGraph> &grouped_graphs);
  static MergePlan ResolveMergePlan(const std::vector<MergeableGraphs> &mergeable_groups, size_t reductions_needed);
  static bool IsSimpleComputeGraph(const ::ascir::ImplGraph &graph, size_t &node_count);
  static Status MergeGraphs(::ascir::ImplGraph &dst, const std::vector<::ascir::ImplGraph> &grouped_graphs,
                            const std::vector<size_t> &group);
  static Status AddConnectedNodes(const af::AscNodePtr &root_node, ::ascir::ImplGraph &sub_graph,
                                  std::set<af::NodePtr> &all_visited);
  static Status CollectConnectedNodes(const af::AscNodePtr &root_node, std::set<af::NodePtr> &visited,
                                      std::vector<af::AscNodePtr> &asc_nodes);
  static bool CompareByNodeId(const af::AscNodePtr &lhs, const af::AscNodePtr &rhs);
  static Status RecordAxisSizes(const std::vector<af::Expression> &repeats, const std::vector<int64_t> &axis_ids,
                                std::map<af::AxisId, af::Expression> &axis_id_to_size);
  static Status SortSubGraphsByDependency(std::vector<::ascir::ImplGraph> &sub_optimize_graphs);
  static bool HasDataDependency(const ::ascir::ImplGraph &producer, const ::ascir::ImplGraph &consumer);
};
}  // namespace optimize

#endif  // ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_SCHEDULE_GROUP_PARTITIONER_H_
