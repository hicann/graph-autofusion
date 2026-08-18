/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_PASS_BASE_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_PASS_BASE_H_

#include <map>
#include <string>
#include <vector>
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_constant.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pattern.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "register/graph_optimizer/fusion_common/op_slice_info.h"

namespace fe {
enum class BufferFusionPassType {
  BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
  BUILT_IN_VECTOR_CORE_BUFFER_FUSION_PASS,
  CUSTOM_AI_CORE_BUFFER_FUSION_PASS,
  CUSTOM_VECTOR_CORE_BUFFER_FUSION_PASS,
  BUFFER_FUSION_PASS_TYPE_RESERVED
};

class BufferFusionPassBase {
 public:
  explicit BufferFusionPassBase();
  virtual ~BufferFusionPassBase();
  virtual std::vector<BufferFusionPattern *> DefinePatterns() = 0;
  virtual Status GetFusionNodes(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes);
  virtual Status GetMixl2FusionNodes(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes);
  virtual Status PostFusion(const ge::NodePtr &fused_node);
  virtual Status CalcFusionOpSliceInfo(std::vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_slice_info);
  virtual Status CheckNodeCanFusion(const BufferFusionNodeDescMap &fusion_nodes, const ge::NodePtr &next_node);
  static std::vector<ge::NodePtr> GetMatchedNodes(const BufferFusionMapping &mapping);
  static std::vector<ge::NodePtr> GetMatchedNodesByDescName(const std::string &desc_name,
                                                            const BufferFusionMapping &mapping);
  static ge::NodePtr GetMatchedHeadNode(const std::vector<ge::NodePtr> &matched_nodes);
  static bool CheckNodeIsDynamicImpl(const ge::NodePtr &node);
  static bool CheckTwoNodesImplConsistent(const ge::NodePtr &src_node, const ge::NodePtr &dst_node);
  static bool CheckNodesImplConsistent(const BufferFusionMapping &mapping);
  static bool CheckNodesImplConsistent(const std::vector<ge::NodePtr> &fusion_nodes);
  static bool CheckNodeIsDynamicShape(const ge::NodePtr &node);
  static bool CheckNodesIncDynamicShape(const BufferFusionMapping &mapping);
  static bool CheckNodesIncDynamicShape(const std::vector<ge::NodePtr> &fusion_nodes);
  void SetName(const std::string &name) {
    name_ = name;
  }

  std::string GetName() {
    return name_;
  }

 private:
  std::string name_;
};

}  // namespace fe

#endif  // INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_PASS_BASE_H_
