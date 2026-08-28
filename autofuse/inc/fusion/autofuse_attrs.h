/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_INC_FUSION_AUTOFUSE_ATTRS_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_INC_FUSION_AUTOFUSE_ATTRS_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "fusion/fuse_type.h"
#include "fusion/fusion_decider.h"
#include "autoschedule/axis_group.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "attribute_group/af_attr_group_base.h"

#include "ascir_ops.h"
#include "ascendc_ir/utils/asc_graph_utils.h"

#ifdef AUTOFUSE_USE_GE_METADEF
namespace af {
using ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME;
using ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX;
using ge::AttrUtils;
}  // namespace af
#endif

namespace af {

const std::string kSplitTypeStub = "Split";
constexpr int64_t kNonSplitGlobalId = -1L;
constexpr float kSplitLowFusionRatioThreshold = 0.2000F;

#define REDUCE_ALL_LOAD_INIT (-1)
#define REDUCE_ALL_LOAD_NOT_ALL (0)
#define REDUCE_ALL_LOAD_ALL (1)

enum class SplitFusionRatioRequirementState : uint32_t {
  NOT_DETERMINED = 0U,  // 尚未计算融合比例
  NOT_SATISFIED = 1U,   // 融合比例不满足阈值要求
  SATISFIED = 2U        // 融合比例满足阈值要求
};

struct AutofuseInnerAttrs {
  std::vector<const af::Node *> origin_nodes;  // Asc节点对应的原始节点，用于Dfx打印、获取融合前ComputeGraph片段等
  std::vector<af::OutDataAnchor *> output_buffers;  // Asc节点负责写入的原始输出anchor，用于lifting
  // Asc输入anchor对应的原始Edge，由于Asc会在Lowering阶段，进行Load的Case，因此一个输出可能对应多个原始Edge
  std::map<size_t, std::set<const af::InDataAnchor *>> concrete_edges;
  std::set<const af::OutDataAnchor *> optimized_input_buffers;  // Asc节点优化掉的读入anchor，例如ZerosLike的输入
  std::set<std::pair<af::NodePtr, af::NodePtr>>
      possible_fusion_nodes;  // 两个fused subgraph融合后可能出现的循环合并的nodes
  std::unique_ptr<FusionDecider> decider;
  std::vector<af::NodePtr> fused_subgraph_outputs;  // 字图融合记录的保序输出信息
  optimize::autoschedule::AxisGroup axis_group;     // 记录一个子图融合产生的axis group信息
  uint64_t fuse_type;                               // 融合后的type，同步bit位记录融合后的ascgraph类型
  size_t fusion_nodes_size_;                        // 被融合的节点数
  std::vector<std::pair<std::string, int32_t>> origin_output_names_;  // 融合节点与原始ge节点的输出映射关系
  std::vector<std::pair<std::string, int32_t>> origin_input_names_;  // 融合节点与原始ge节点的输入映射关系
  int32_t vector_core_num;                                           // user set vector vore num scope
  size_t reduce_fused_elementwise_node_num = 0U;  // reduce节点向后融合的elementwise节点数量
  int64_t split_global_id = kNonSplitGlobalId;  // split op 在 lowering 之前的全局编号，不是split节点的话，这个编号为-1
  SplitFusionRatioRequirementState split_fusion_ratio_requirement_state =
      SplitFusionRatioRequirementState::NOT_DETERMINED;  // 缓存对split融合比例是否超过阈值的预测结果
  bool is_split_complete = false;                        // 缓存原split节点是否完全恢复的判断结果
  bool is_fuse_from_lowering = false;                    // 标识融合节点来自lowering还是can_fuse
  std::vector<int64_t> reduce_original_axis;             // reduce操作前的原始轴信息
  std::vector<Expression> reduce_original_repeats;       // reduce操作前的原始repeats信息
  int32_t is_reduce_all_load = REDUCE_ALL_LOAD_INIT;     // 标识reduce是否所有load都是norm-like

  bool IsReduction() const {
    return HasFuseType(loop::FuseType::kReduction);
  }

  bool HasFuseType(const loop::FuseType type) const {
    return (fuse_type & (1UL << static_cast<uint64_t>(type))) != 0UL;
  }
};

using AfAttrGroupsBase = af::AttrGroupsBase;

class AutoFuseAttrs : public AfAttrGroupsBase {
 public:
  AutoFuseAttrs() = default;
  AutoFuseAttrs(const AutoFuseAttrs &other) : fuse_type_(other.fuse_type_), asc_graph_(other.asc_graph_) {}
  AutoFuseAttrs &operator=(const AutoFuseAttrs &other) = delete;

  [[nodiscard]] const std::shared_ptr<AscGraph> &GetAscGraph() const {
    return asc_graph_;
  }

  void SetFuseType(const loop::FuseType fuse_type) {
    fuse_type_ = fuse_type;
    inner_attrs_.fuse_type = (1UL << static_cast<uint64_t>(fuse_type));
  }

  void SetAscGraph(const std::shared_ptr<AscGraph> &asc_graph,
                   const loop::FuseType fuse_type = loop::FuseType::kExtern) {
    asc_graph_ = asc_graph;
    SetFuseType(fuse_type);
    if (fuse_type == loop::FuseType::kSplit) {
      InitSplitGlobalIdFromGraph();
    }
  }

  [[nodiscard]] const ComputeGraphPtr &GetFuseComputeGraph() const {
    return fused_compute_graph_;
  }

  void SetFuseComputeGraph(const ComputeGraphPtr &fused_compute_graph) {
    fused_compute_graph_ = fused_compute_graph;
  }

  [[nodiscard]] loop::FuseType GetFuseType() const {
    return fuse_type_;
  }

  bool HasFuseType(const loop::FuseType fuse_type) const {
    return (fuse_type_ == fuse_type) || (inner_attrs_.HasFuseType(fuse_type));
  }

  uint64_t GetAllFuseType() const {
    return (1UL << static_cast<uint64_t>(fuse_type_)) | static_cast<uint64_t>(inner_attrs_.fuse_type);
  }

  std::unique_ptr<AfAttrGroupsBase> CloneAf() override {
    return std::unique_ptr<AutoFuseAttrs>(new (std::nothrow) AutoFuseAttrs(*this));
  }

  AutofuseInnerAttrs &GetMutableInterAttrs() {
    return inner_attrs_;
  }

  // ---- 以下为前端使用的便捷方法，后端一般不直接调用 ----

  void SetOriginNodes(const std::vector<const af::Node *> &nodes) {
    inner_attrs_.origin_nodes = nodes;
  }

  void SetVectorCoreNum(const int32_t vector_core_num) {
    inner_attrs_.vector_core_num = vector_core_num;
  }

  int32_t GetVectorCoreNum() const {
    return inner_attrs_.vector_core_num;
  }

  void AddConcreteEdges(const size_t index, const af::InDataAnchor *dst) {
    inner_attrs_.concrete_edges[index].insert(dst);
  }

  void SetOptimizedInputBuffers(const std::set<const af::OutDataAnchor *> &input_buffers) {
    inner_attrs_.optimized_input_buffers = input_buffers;
  }

  [[nodiscard]] const std::set<const af::OutDataAnchor *> &GetOptimizedInputBuffers() const {
    return inner_attrs_.optimized_input_buffers;
  }

  [[nodiscard]] const std::map<size_t, std::set<const af::InDataAnchor *>> &GetConcreteEdges() const {
    return inner_attrs_.concrete_edges;
  }

  void SetOriginOutputBuffers(const std::vector<af::OutDataAnchor *> &buffers) {
    inner_attrs_.output_buffers = buffers;
  }

  [[nodiscard]] const std::vector<af::OutDataAnchor *> &GetOriginOutputBuffers() const {
    return inner_attrs_.output_buffers;
  }

  [[nodiscard]] const std::vector<const af::Node *> &GetOriginNodes() const {
    return inner_attrs_.origin_nodes;
  }

  uint64_t GetFusionNodesSize() const {
    return inner_attrs_.fusion_nodes_size_;
  }

  void SetFusionNodesSize(const uint64_t fusion_nodes_size) {
    inner_attrs_.fusion_nodes_size_ = fusion_nodes_size;
  }

  size_t GetReduceFusedElementwiseNodeNum() const {
    return inner_attrs_.reduce_fused_elementwise_node_num;
  }

  void SetReduceFusedElementwiseNodeNum(const size_t elementwise_node_num) {
    inner_attrs_.reduce_fused_elementwise_node_num = elementwise_node_num;
  }

  void SetSplitGlobalId(const size_t global_id) {
    inner_attrs_.split_global_id = static_cast<int64_t>(global_id);
  }

  int32_t GetSplitGlobalId() const {
    GE_ASSERT_TRUE(inner_attrs_.split_global_id != -1 || !this->HasFuseType(loop::FuseType::kSplit),
                   "Split global id not initialized for split node.");
    return inner_attrs_.split_global_id;
  }

  SplitFusionRatioRequirementState GetSplitFusionRatioRequirementState() const {
    return inner_attrs_.split_fusion_ratio_requirement_state;
  }

  void SetSplitLowFusionRatioRequirementState(SplitFusionRatioRequirementState state) {
    inner_attrs_.split_fusion_ratio_requirement_state = state;
  }

  void SetSplitComplete() {
    inner_attrs_.is_split_complete = true;
  }

  bool GetSplitComplete() const {
    return inner_attrs_.is_split_complete;
  }

  void SetReduceOriginalAxis(const std::vector<int64_t> &axis) {
    inner_attrs_.reduce_original_axis = axis;
  }

  [[nodiscard]] const std::vector<int64_t> &GetReduceOriginalAxis() const {
    return inner_attrs_.reduce_original_axis;
  }

  void SetReduceOriginalRepeats(const std::vector<Expression> &repeats) {
    inner_attrs_.reduce_original_repeats = repeats;
  }

  [[nodiscard]] const std::vector<Expression> &GetReduceOriginalRepeats() const {
    return inner_attrs_.reduce_original_repeats;
  }

  void SetReduceAllLoadState(const int32_t state) {
    inner_attrs_.is_reduce_all_load = state;
  }

  [[nodiscard]] int32_t GetReduceAllLoadState() const {
    return inner_attrs_.is_reduce_all_load;
  }

  Status SetAndPrintOriginNames(const OpDescPtr &op_desc, const std::string &graph_name,
                                const std::vector<const OutDataAnchor *> &origin_inputs,
                                const af::OutDataAnchor *anchor) {
    std::vector<std::pair<std::string, int32_t>> origin_output_names;
    std::vector<std::pair<std::string, int32_t>> origin_input_names;
    origin_input_names.reserve(origin_inputs.size());
    for (auto &origin_input : origin_inputs) {
      origin_input_names.emplace_back(origin_input->GetOwnerNode()->GetName(), origin_input->GetIdx());
    }
    origin_output_names.emplace_back(anchor->GetOwnerNode()->GetName(), anchor->GetIdx());
    inner_attrs_.origin_output_names_ = origin_output_names;
    inner_attrs_.origin_input_names_ = origin_input_names;
    // input
    uint32_t index = 0U;
    for (const auto &origin_input : origin_input_names) {
      GELOGD("ascbc_dfx_log(lowering), %s, input_idx: %u, origin_ge_node: %s, input_idx: %d.", graph_name.c_str(),
             index, origin_input.first.c_str(), origin_input.second);
      ++index;
    }
    // output
    index = 0U;
    const auto output_desc = op_desc->GetAllOutputsDescPtr();
    GE_ASSERT_TRUE(output_desc.size() == origin_output_names.size(),
                   "output desc size(%zu) not equal to origin output names size(%zu).", output_desc.size(),
                   origin_output_names.size());
    for (const auto &origin_output : origin_output_names) {
      GELOGD("ascbc_dfx_log(lowering), %s, output_idx: %u, origin_ge_node: %s, output_idx: %d.", graph_name.c_str(),
             index, origin_output.first.c_str(), origin_output.second);
      af::AttrUtils::SetStr(output_desc.at(index), af::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, origin_output.first);
      af::AttrUtils::SetInt(output_desc.at(index), af::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, origin_output.second);
      ++index;
    }
    return SUCCESS;
  }

 private:
  loop::FuseType fuse_type_ = loop::FuseType::kExtern;
  std::shared_ptr<AscGraph> asc_graph_;
  ComputeGraphPtr fused_compute_graph_;
  AutofuseInnerAttrs inner_attrs_;

  void InitSplitGlobalIdFromGraph() {
    if (asc_graph_ == nullptr) return;
    for (const auto &ir_node : AscGraphUtils::GetComputeGraph(*asc_graph_)->GetAllNodes()) {
      if (ir_node->GetType() == kSplitTypeStub) {
        const auto &ir_desc = ir_node->GetOpDesc();
        const auto &ir_attr = ir_desc->GetAttrsGroup<AscNodeAttr>();
        if (ir_attr == nullptr) continue;
        const auto split_attr = dynamic_cast<ascir_op::Split::AscSplitIrAttrDef *>(ir_attr->ir_attr.get());
        if (split_attr == nullptr) continue;
        (void)split_attr->GetGid(inner_attrs_.split_global_id);
        GELOGD("Init split global id from IR node: [node: %s(%s), global id: %d]", ir_node->GetType().c_str(),
               ir_node->GetName().c_str(), inner_attrs_.split_global_id);
      }
    }
  }
};

inline AutoFuseAttrs *GetOrCreateAutoFuseAttrs(const OpDescPtr &op_desc) {
  return op_desc->GetOrCreateAttrsGroup<AutoFuseAttrs>();
}

inline AutoFuseAttrs *GetOrCreateAutoFuseAttrs(OpDesc *op_desc) {
  return op_desc->GetOrCreateAttrsGroup<AutoFuseAttrs>();
}

inline AutoFuseAttrs *GetOrCreateAutoFuseAttrs(const ComputeGraphPtr &graph) {
  return graph->GetOrCreateAttrsGroup<AutoFuseAttrs>();
}

inline AutofuseInnerAttrs &GetInterAttrs(AutoFuseAttrs *attr) {
  return attr->GetMutableInterAttrs();
}

inline uint64_t MergeFuseType(uint64_t fuse_type1, uint64_t fuse_type2) {
  return fuse_type1 | fuse_type2;
}

}  // namespace af

namespace ge {
using af::AutoFuseAttrs;
using af::AutofuseInnerAttrs;
using af::GetInterAttrs;
using af::GetOrCreateAutoFuseAttrs;
using af::kNonSplitGlobalId;
using af::kSplitLowFusionRatioThreshold;
using af::kSplitTypeStub;
using af::MergeFuseType;
using af::SplitFusionRatioRequirementState;
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_INC_FUSION_AUTOFUSE_ATTRS_H_
