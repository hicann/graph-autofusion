/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AF_INC_GRAPH_UTILS_OP_DESC_UTILS_H_
#define AF_INC_GRAPH_UTILS_OP_DESC_UTILS_H_

#include <string>
#include <vector>
#include "graph/def_types.h"
#include "graph/node.h"
#include "graph/operator_af.h"

/*lint -e148*/
namespace af {
class OpDescUtils {
 public:
  OpDescUtils() = default;
  ~OpDescUtils() = default;
  static std::vector<GeTensorPtr> MutableWeights(const Node &node);
  static std::vector<GeTensorPtr> MutableWeights(const NodePtr node);
  static graphStatus SetWeights(Node &node, const std::vector<af::GeTensorPtr> &weights);
  static graphStatus SetWeights(NodePtr node, const std::vector<af::GeTensorPtr> &weights);
  static graphStatus SetNoneConstNodeWeights(Node &node, const std::vector<af::GeTensorPtr> &weights);
  static bool ClearInputDesc(const OpDescPtr op_desc, const uint32_t index);
  static std::vector<NodePtr> GetConstInputs(const Node &node, const uint32_t depth = 64U);
  static std::vector<NodePtr> GetConstInputs(const ConstNodePtr &node);

  static Operator CreateOperatorFromOpDesc(OpDescPtr op_desc);
  static Operator CreateOperatorFromNode(ConstNodePtr node_ptr);
  static OpDescPtr GetOpDescFromOperator(const Operator &oprt);
  static graphStatus CopyOperatorLinks(const std::map<std::string, Operator> &src_op_list,
                                       std::map<std::string, Operator> &dst_op_list);
  static graphStatus CopyOperators(const ComputeGraphPtr &dst_compute_graph,
                                   const std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                   const std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new,
                                   const std::map<std::string, Operator> &src_op_list,
                                   std::map<std::string, Operator> &dst_op_list);
  __attribute__((weak)) static OpDescPtr CopyOpDesc(const ConstOpDescPtr &org_op_desc);
  __attribute__((weak)) static OpDescPtr CreateConstOp(const GeTensorPtr &tensor_ptr);
  static OpDescPtr CreateConstOp(const GeTensorPtr &tensor_ptr, const bool copy);
  static OpDescPtr CreateConstOpZeroCopy(const GeTensorPtr &tensor_ptr);
  static std::map<size_t, std::pair<size_t, size_t>> GetInputIrIndexes2InstanceIndexesPairMap(const OpDescPtr &op_desc);
  static std::map<size_t, std::pair<size_t, size_t>> GetOutputIrIndexes2InstanceIndexesPairMap(
      const OpDescPtr &op_desc);

  static graphStatus GetIrInputInstanceDescRange(const OpDescPtr &op,
                                                 std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range);

  __attribute__((weak)) static graphStatus GetIrInputRawDescRange(
      const OpDescPtr &op, std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range);

  static graphStatus GetIrOutputDescRange(const OpDescPtr &op,
                                          std::map<size_t, std::pair<size_t, size_t>> &ir_output_2_range);

 private:
  static GeTensorPtr MutableWeights(OpDesc &op_desc);
  static GeTensorPtr MutableWeights(const OpDescPtr op_desc);
  static graphStatus SetWeights(OpDesc &op_desc, const GeTensorPtr weight);
  static graphStatus SetWeights(OpDescPtr op_desc, const GeTensorPtr weight);
};
}  // namespace af
/*lint +e148*/
#endif  // INC_GRAPH_UTILS_OP_DESC_UTILS_H_
