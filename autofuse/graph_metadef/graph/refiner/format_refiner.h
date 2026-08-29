/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_GRAPH_FORMAT_REFINER_H_
#define COMMON_GRAPH_FORMAT_REFINER_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define METADEF_FUNC_VISIBILITY _declspec(dllexport)
#else
#define METADEF_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define METADEF_FUNC_VISIBILITY
#else
#define METADEF_FUNC_VISIBILITY __attribute__((visibility("hidden")))
#endif
#endif

#include <deque>
#include <string>
#include <unordered_map>
#include <vector>
#include "graph/compute_graph.h"
#include "graph/types_af.h"
#include "graph/ge_error_codes_af.h"

namespace af {
// ShapeRefiner performs shape inference for compute graphs
class METADEF_FUNC_VISIBILITY FormatRefiner {
 public:
  static graphStatus InferOrigineFormat(const ComputeGraphPtr &graph);

 private:
  static graphStatus RefreshConstantOutProcess(const ComputeGraphPtr &com_graph, const OpDescPtr &op_desc);
  static graphStatus GetAnchorPoints(const ComputeGraphPtr &com_graph, std::vector<NodePtr> &anchor_points,
                                     std::vector<NodePtr> &anchor_data_nodes);
  static graphStatus AnchorProcess(const NodePtr &anchor_node);
  static void RefreshOriginFormatOfAnchor(const std::vector<NodePtr> &anchor_points);
  static graphStatus BackInferProcess(std::deque<NodePtr> &nodes, const NodePtr &node);
  static graphStatus ForwardInferProcess(std::deque<NodePtr> &nodes, const NodePtr &node);
  static graphStatus DataNodeFormatProcess(const ComputeGraphPtr &graph, const std::vector<NodePtr> &anchor_data_nodes,
                                           const Format data_format);
  static bool IsGraphInferred(const ComputeGraphPtr &graph);
};
}  // namespace af
#endif  // COMMON_GRAPH_FORMAT_REFINER_H_
