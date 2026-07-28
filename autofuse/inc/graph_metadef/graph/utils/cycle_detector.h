/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_CYCLE_DETECTOR_H_
#define GRAPH_CYCLE_DETECTOR_H_

#include "graph/node.h"
#include "graph/compute_graph.h"
#include "connection_matrix.h"

namespace af {
class CycleDetector {
  friend class GraphUtils;

 public:
  CycleDetector() = default;
  ~CycleDetector() = default;
  /* Detect whether there are cycles in graph
   * after fusing all nodes in param fusion_nodes.
   * Before call this func, you should call GenerateConnectionMatrix first
   * to generate connection_matrix based on current graph.
   *
   * Compared with Cycle Detection
   * @param fusion_nodes: each vector in fusion_nodes
   * will be fused into an entity(which could contains
   * more than one node). The caller should put all original
   * nodes which are expected to be fused into one larger node
   * into each sub-vector of fusion_nodes.
   *
   * This function can tell whether there are a cycle after
   * fusing all nodes in fusion_nodes. Each vector in 2-d
   * vector fusion_nodes will be fused into an entity.
   *
   *
   * This interface cannot detect whether there are cycles
   * inside the fused nodes.
   *
   * e.g. {a, b, c, d} -> {e, f}
   * Because the edge information is not given for e and f
   * so this function we cannot tell if e and f are in a
   * cycle.
   * */
  bool HasDetectedCycle(const std::vector<std::vector<NodePtr>> &fusion_nodes);

  /**
   * Update connection matrix based on graph.
   * Connection matrix is served for cycle detection.
   *
   * The first param graph, it should be the same one graph when contribute cycle_detector
   */
  void Update(const ComputeGraphPtr &graph, const std::vector<NodePtr> &fusion_nodes);

  /**
   * Expand dim and update connection matrix based on graph.
   */
  void ExpandAndUpdate(const vector<NodePtr> &fusion_nodes, const std::string &node_name);

 private:
  graphStatus Init(const ComputeGraphPtr &graph);
  std::unique_ptr<ConnectionMatrix> connectivity_{nullptr};
};

using CycleDetectorPtr = std::unique_ptr<CycleDetector>;
using CycleDetectorSharedPtr = std::shared_ptr<CycleDetector>;
}  // namespace af
#endif  // GRAPH_CYCLE_DETECTOR_H_
