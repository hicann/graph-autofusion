/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_SPLIT_SCHEDULE_CASE_GENERATOR_H_
#define ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_SPLIT_SCHEDULE_CASE_GENERATOR_H_

#include "ascir_ops.h"
#include "ascir/meta/ascir.h"
#include "common/ascgen_log.h"
#include "optimize/task_generator/schedule_case_generator.h"

namespace optimize {
class SplitFusionCaseGenerator : public FusionCaseGenerator {
 public:
  Status Generate(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                  std::vector<std::string> &score_functions) override;

 private:
  static std::vector<af::AscNodePtr> FindSplitNodes(const ascir::HintGraph &owner_graph);
  static Status ResolveSplitDim(const af::AscNodePtr &split_node, size_t &split_dim, bool &is_first_dim);
  Status ConvertSplitToLoads(ascir::HintGraph &owner_graph, const af::AscNodePtr &split_node, size_t split_dim);
  Status SplitSplits(const ascir::HintGraph &owner_graph, const af::AscNodePtr &split_node, size_t split_dim,
                     const bool &split);
  Status Prepare(const af::AscNodePtr &split_node, size_t split_dim);
  Status ReplaceWithLoad(::ascir::ImplGraph &owner_graph, const af::AscNodePtr &split_node,
                         const af::OutDataAnchorPtr &split_out_anchor);
  Status ReplaceWithSplit(::ascir::ImplGraph &owner_graph, const af::AscNodePtr &split_node, size_t split_dim,
                          size_t start, size_t end);
  Status RemoveUnusedNodes(const af::AscNodePtr &split_node) const;
  static Status UpdateSplitAxis(ascir::ImplGraph &owner_graph, af::AscNodePtr &node, uint32_t split_dim,
                                size_t start_index);
  static af::Status SetSplitOpAttr(af::ascir_op::Split &split_op, const af::AscNodePtr &split_node, size_t split_dim,
                                   size_t start, size_t end);
  af::Status SetLoadOpAttr(af::ascir_op::Store &store_op, const af::ascir_op::Split &split_op,
                           size_t start_index) const;
  af::Status SplitOutReplaceAxis(ascir::ImplGraph &owner_graph, std::vector<af::AscNodePtr> &nodes,
                                 const af::AscNodePtr &load_node_new, int32_t out_index,
                                 af::AscNodePtr &broadcast_node);
  af::Status CollectBackwardNodes(const af::AscNodePtr &load_node, std::vector<af::AscNodePtr> &nodes,
                                  af::AscNodePtr &broadcast_node) const;
  af::Status SplitDataForConvertLoad(ascir::ImplGraph &owner_graph, const af::AscNodePtr &split_node,
                                     const af::OutDataAnchorPtr &split_out_anchor, af::AscNodePtr &new_load_node);
  void IsBroadcastNode(const af::NodePtr &origin_node, af::AscNodePtr &broadcast_node, bool &has_broadcast_node) const;
  std::vector<af::Expression> offsets_;
  af::AscNodePtr ori_load_node_;
  af::AscNodePtr ori_in_data_node_;
  std::map<af::AscNodePtr, size_t> split_node_to_start_index_;
  ascir::AxisId split_axis_id_ = -1;
  size_t split_dim_ = std::numeric_limits<size_t>::max();
  af::AscNodePtr split_node_;
  [[nodiscard]] bool HasLoadStoreConversion() const override {
    return true;
  }
};
}  // namespace optimize

#endif  // ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_SPLIT_SCHEDULE_CASE_GENERATOR_H_
