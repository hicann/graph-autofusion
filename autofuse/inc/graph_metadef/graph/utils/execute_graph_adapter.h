/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_UTILS_EXECUTE_GRAPH_ADAPTER_H
#define INC_GRAPH_UTILS_EXECUTE_GRAPH_ADAPTER_H

#include "graph/fast_graph/execute_graph.h"
#include "graph/ge_error_codes_af.h"

namespace af {
class ExecuteGraphAdapter {
 public:
  ~ExecuteGraphAdapter() = default;
  ExecuteGraphAdapter(const ExecuteGraphAdapter &adapter) = delete;
  ExecuteGraphAdapter &operator=(const ExecuteGraphAdapter &adapter) = delete;

  // 返回的ComputeGraph复用了原图src_graph的OpDesc对象，返回后src_graph不能释放
  // src_graph和返回的ComputeGraph的生命周期需要保证一致
  static ComputeGraphPtr ConvertExecuteGraphToComputeGraph(ExecuteGraph *src_graph);

 private:
  ExecuteGraphAdapter() = default;
  static graphStatus ConvertExecuteGraphToComputeGraph(ExecuteGraph *src_graph, const ComputeGraphPtr &dst_graph,
                                                       const int32_t depth);
  static graphStatus CopyOpAndSubgraph(ExecuteGraph *src_graph, const ComputeGraphPtr &dst_graph,
                                       std::unordered_map<FastNode *, Node *> &all_new_nodes, const int32_t depth);
  static graphStatus RelinkGraphEdges(FastNode *old_node, const std::unordered_map<FastNode *, Node *> &all_new_nodes);
  static graphStatus CopyMembers(ExecuteGraph *src_graph, const ComputeGraphPtr &dst_graph,
                                 const std::unordered_map<FastNode *, Node *> &all_new_nodes);
  static void InheritOriginalAttr(ExecuteGraph *src_graph, const ComputeGraphPtr &dst_graph);
};
}  // namespace af
#endif  // INC_GRAPH_UTILS_EXECUTE_GRAPH_ADAPTER_H
