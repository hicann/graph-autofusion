/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/indirect_load_schedule_case_generator.h"

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "v35/ascir/ascir_codegen_v2.h"
#include "common_utils.h"
#include "graph_utils.h"
#include "graph/symbolizer/symbolic.h"
#include "indirect_load_utils.h"
#include "schedule_utils.h"
#include "schedule_result.h"

#include <algorithm>
#include <limits>
#include <set>
#include <unordered_set>
#include <vector>

namespace optimize {
namespace {
constexpr int64_t kIndirectLoadSimtDcacheSize = 32 * 1024;
constexpr size_t kIndirectLoadInputCount = 2UL;
constexpr size_t kIndirectLoadOutputCount = 1UL;
constexpr size_t kIndirectLoadInvalidAxisIndex = std::numeric_limits<size_t>::max();
constexpr int64_t kInvalidBroadcastIndex = -1L;
constexpr char kInputInnerAxisName[] = "indirect_load_input_inner";
constexpr char kIndexInnerAxisName[] = "indirect_load_index_inner";
constexpr char kSingleOuterAxisName[] = "indirect_load_single_outer";
constexpr char kOuterAxisName[] = "indirect_load_outer";
constexpr char kInnerAxisName[] = "indirect_load_inner";
using NodePath = std::vector<af::AscNodePtr>;
using NodeSet = std::unordered_set<const af::AscNode *>;

struct TemplateCase {
  ascir::TemplateId template_id;
  ascgen_utils::indirect_load::Implementation implementation;
};

struct RewrittenGraphAnalysis {
  NodePath input_region;  // SIMD: input 侧链；SIMT 不使用
  NodePath index_region;  // SIMD: index 侧链；SIMT: 融合集（index 链 + 输出链）
  NodePath input_path;
  NodePath index_path;
  af::AscNodePtr input_root;
  af::AscNodePtr index_root;
  af::AscNodePtr output_store;
  af::AscNodePtr post_reduce;
  bool align_input_path = false;
  bool align_index_path = false;
  bool simd_index_uses_output_inner_axis = false;
};

struct InputViewPlan {
  NodePath path;
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  int64_t path_broadcast_index = kInvalidBroadcastIndex;
  bool simd_index_uses_output_inner_axis = false;
};

struct PhysicalViewPreparation {
  InputViewPlan input;
  InputViewPlan index;
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
};

enum class ReduceAxisKind : uint8_t { kRetained, kReduced, kIgnored };

struct PostReduceLayout {
  std::vector<af::AxisId> axes;
  std::vector<ReduceAxisKind> kinds;
  size_t first_reduce;
};

bool IsInputRegionBoundary(const af::AscNodePtr &node) {
  return ScheduleUtils::IsDataInput(node) || af::ops::IsOps<af::ascir_op::Scalar>(node);
}

bool HasControlEdge(const af::AscNodePtr &node) {
  return node->GetInControlNodesSize() != 0UL || node->GetOutControlNodesSize() != 0UL;
}

bool IsSingleConsumerWithoutControlEdge(const af::AscNodePtr &node) {
  return node != nullptr && node->GetOutDataNodesSize() == 1UL && !HasControlEdge(node);
}

bool IsBroadcastNode(const af::AscNodePtr &node) {
  return af::ops::IsOps<af::ascir_op::Broadcast>(node);
}

// 检查已选中的 Broadcast 路径是否可以安全折叠；不支持时只淘汰当前 candidate。
bool IsSupportedBroadcastPath(const NodePath &path, size_t broadcast_index, ascir::TemplateId template_id) {
  if (broadcast_index >= path.size()) {
    GELOGI("[IndirectLoad] Reject candidate[%d]: Broadcast path index is out of range.",
           static_cast<int32_t>(template_id));
    return false;
  }
  const af::AscNodePtr &broadcast = path[broadcast_index];
  const auto broadcast_source = ascgen_utils::indirect_load::GetInputProducer(broadcast, 0UL);
  if (!IsSingleConsumerWithoutControlEdge(broadcast)) {
    GELOGI("[IndirectLoad] Reject candidate[%d]: Broadcast path node[%s] is not safely foldable.",
           static_cast<int32_t>(template_id), broadcast->GetNamePtr());
    return false;
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    if (!IsSingleConsumerWithoutControlEdge(broadcast_source)) {
      GELOGI("[IndirectLoad] Reject candidate[%d]: Broadcast source node[%s] is not safely foldable.",
             static_cast<int32_t>(template_id),
             broadcast_source == nullptr ? "<null>" : broadcast_source->GetNamePtr());
      return false;
    }
    return true;
  }
  // CollectInputPaths 只有 SK 会继续回溯 Broadcast 前的单输入链。
  for (size_t i = 0UL; i < broadcast_index; ++i) {
    const af::AscNodePtr &element = path[i];
    if (element->outputs().size() != 1UL || !IsSingleConsumerWithoutControlEdge(element) ||
        !ScheduleUtils::IsElewise(element)) {
      GELOGI("[IndirectLoad] Reject candidate[%d]: Broadcast pre element node[%s] is not safely foldable.",
             static_cast<int32_t>(template_id), element->GetNamePtr());
      return false;
    }
  }
  for (size_t i = broadcast_index + 1UL; i < path.size(); ++i) {
    const af::AscNodePtr &element = path[i];
    if (!IsSingleConsumerWithoutControlEdge(element)) {
      GELOGI("[IndirectLoad] Reject candidate[%d]: Broadcast path node[%s] is not safely foldable.",
             static_cast<int32_t>(template_id), element->GetNamePtr());
      return false;
    }
  }
  return true;
}

af::Status GetBroadcastPhysicalAttr(const af::AscNodePtr &broadcast, ascir::TemplateId template_id,
                                    af::AscTensorAttr &physical_attr) {
  const auto input_anchor = broadcast->GetInDataAnchor(0UL);
  GE_ASSERT_NOTNULL(input_anchor, "IndirectLoad template[%d] Broadcast node[%s] input anchor is invalid.",
                    static_cast<int32_t>(template_id), broadcast->GetNamePtr());
  const auto peer_out_anchor = input_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor, "IndirectLoad template[%d] Broadcast node[%s] source anchor is invalid.",
                    static_cast<int32_t>(template_id), broadcast->GetNamePtr());
  const auto producer = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
  GE_ASSERT_NOTNULL(producer, "IndirectLoad template[%d] Broadcast node[%s] source is invalid.",
                    static_cast<int32_t>(template_id), broadcast->GetNamePtr());
  const size_t output_idx = static_cast<size_t>(peer_out_anchor->GetIdx());
  const auto producer_outputs = producer->outputs();
  GE_ASSERT_TRUE(output_idx < producer_outputs.size(),
                 "IndirectLoad template[%d] Broadcast node[%s] source output index is out of range.",
                 static_cast<int32_t>(template_id), broadcast->GetNamePtr());
  GE_ASSERT_NOTNULL(producer_outputs[output_idx], "IndirectLoad template[%d] Broadcast node[%s] source output is null.",
                    static_cast<int32_t>(template_id), broadcast->GetNamePtr());
  physical_attr = producer_outputs[output_idx]->attr;
  return af::SUCCESS;
}

af::Status InlineBroadcastPath(NodePath &path, int64_t path_broadcast_index, ascir::TemplateId template_id) {
  GE_ASSERT_TRUE(path_broadcast_index >= 0L, "IndirectLoad Broadcast index is invalid.");
  const size_t broadcast_index = static_cast<size_t>(path_broadcast_index);
  GE_ASSERT_TRUE(broadcast_index < path.size(), "IndirectLoad Broadcast index is out of range.");
  const af::AscNodePtr broadcast = path[broadcast_index];
  const auto owner_graph = broadcast->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph);
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::IsolateNodeOneIO(broadcast));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(owner_graph, broadcast));
  GELOGD("[IndirectLoad] Inline Broadcast node[%s] for template[%d] stride-aware physical source window.",
         broadcast->GetNamePtr(), static_cast<int32_t>(template_id));
  path.erase(path.begin() + static_cast<int64_t>(broadcast_index));
  return af::SUCCESS;
}

af::Status FoldBroadcastPath(InputViewPlan &plan, ascir::TemplateId template_id, bool &is_candidate_legal) {
  if (plan.path_broadcast_index == kInvalidBroadcastIndex) {
    return af::SUCCESS;
  }
  const size_t broadcast_index = static_cast<size_t>(plan.path_broadcast_index);
  if (!IsSupportedBroadcastPath(plan.path, broadcast_index, template_id)) {
    is_candidate_legal = false;
    return af::SUCCESS;
  }
  return InlineBroadcastPath(plan.path, plan.path_broadcast_index, template_id);
}

af::Status BuildBroadcastLogicalView(const af::AscTensorAttr &logical_attr, const af::AscTensorAttr &physical_attr,
                                     ascir::TemplateId template_id,
                                     ascgen_utils::indirect_load::LogicalTensorView &view) {
  view = {logical_attr.axis, logical_attr.repeats, logical_attr.strides};
  GE_ASSERT_TRUE(
      physical_attr.repeats.size() == view.sizes.size() && physical_attr.strides.size() == view.strides.size(),
      "IndirectLoad template[%d] Broadcast source layout rank mismatch.", static_cast<int32_t>(template_id));
  view.strides = physical_attr.strides;
  for (size_t dim = 0UL; dim < view.sizes.size(); ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(physical_attr.repeats[dim], af::sym::kSymbolOne) == af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(view.sizes[dim], af::sym::kSymbolOne) != af::TriBool::kTrue) {
      view.strides[dim] = af::sym::kSymbolZero;
    }
  }
  return af::SUCCESS;
}

bool HasMixedInnerBroadcast(const ascgen_utils::indirect_load::LogicalTensorView &logical_view,
                            const af::AscTensorAttr &physical_attr, size_t axis_index) {
  bool has_broadcast_axis = false;
  bool has_regular_axis = false;
  for (size_t dim = axis_index; dim < logical_view.sizes.size(); ++dim) {
    const bool is_broadcast =
        af::SymbolicUtils::StaticCheckEq(physical_attr.repeats[dim], af::sym::kSymbolOne) == af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(logical_view.sizes[dim], af::sym::kSymbolOne) != af::TriBool::kTrue;
    has_broadcast_axis = has_broadcast_axis || is_broadcast;
    has_regular_axis = has_regular_axis || !is_broadcast;
  }
  return has_broadcast_axis && has_regular_axis;
}

bool HasDirectOuterBroadcast(const ascgen_utils::indirect_load::LogicalTensorView &logical_view,
                             const af::AscTensorAttr &physical_attr, size_t axis_index) {
  for (size_t dim = 0UL; dim < axis_index; ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(physical_attr.repeats[dim], af::sym::kSymbolOne) == af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(logical_view.sizes[dim], af::sym::kSymbolOne) != af::TriBool::kTrue) {
      return !HasMixedInnerBroadcast(logical_view, physical_attr, axis_index);
    }
  }
  return false;
}

bool IsSupportedDirectOuterBroadcast(const af::AscNodePtr &broadcast,
                                     const ascgen_utils::indirect_load::LogicalTensorView &logical_view,
                                     const af::AscTensorAttr &physical_attr, size_t axis_index,
                                     ascir::TemplateId template_id) {
  if (template_id != ascir::TemplateId::kIndirectLoadSimd) {
    return true;
  }
  const auto broadcast_producer = ascgen_utils::indirect_load::GetInputProducer(broadcast, 0UL);
  if (broadcast_producer == nullptr || broadcast_producer->inputs.Size() != 1UL) {
    return true;
  }
  if (!HasDirectOuterBroadcast(logical_view, physical_attr, axis_index)) {
    return true;
  }
  GELOGI("[IndirectLoad] Reject candidate[%d]: direct outer Broadcast in the input/index path.",
         static_cast<int32_t>(template_id));
  return false;
}

af::Status ApplyZeroStrideCompactView(const NodePath &path,
                                      const ascgen_utils::indirect_load::IndirectLoadTensorLayout &layout) {
  GE_ASSERT_TRUE(
      layout.axis_ids.size() == layout.physical_repeats.size() && layout.axis_ids.size() == layout.strides.size(),
      "IndirectLoad physical execution view rank mismatch.");
  for (const af::AscNodePtr &node : path) {
    if (IsInputRegionBoundary(node)) {
      continue;
    }
    GE_ASSERT_EQ(node->attr.sched.axis.size(), layout.axis_ids.size());
    node->attr.sched.axis = layout.axis_ids;
    for (const auto &output : node->outputs()) {
      GE_ASSERT_NOTNULL(output);
      GE_ASSERT_EQ(output->attr.repeats.size(), layout.physical_repeats.size());
      GE_ASSERT_EQ(output->attr.strides.size(), layout.strides.size());
      output->attr.axis = layout.axis_ids;
      output->attr.repeats = layout.physical_repeats;
      output->attr.strides = layout.strides;
    }
  }
  return af::SUCCESS;
}

bool NeedsAlignedUbWindow(const ascgen_utils::indirect_load::IndirectLoadTensorLayout &layout, size_t axis_index) {
  if (layout.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kStrided) {
    return false;
  }
  af::Expression physical_span = af::sym::kSymbolOne;
  for (size_t index = layout.sizes.size(); index > axis_index; --index) {
    const size_t dim = index - 1UL;
    if (af::SymbolicUtils::StaticCheckEq(layout.strides[dim], physical_span) != af::TriBool::kTrue) {
      return true;
    }
    physical_span = physical_span + (layout.sizes[dim] - af::sym::kSymbolOne) * layout.strides[dim];
  }
  return false;
}

af::Status AnnotateStridedUbPath(const NodePath &path) {
  for (const af::AscNodePtr &node : path) {
    if (IsInputRegionBoundary(node) || ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    const auto role = ascgen_utils::indirect_load::GetTemplateRole(node);
    GE_ASSERT_TRUE(role == ascgen_utils::indirect_load::TemplateRole::kNone ||
                       role == ascgen_utils::indirect_load::TemplateRole::kStridedUbPath,
                   "IndirectLoad strided path node[%s] already has template role[%ld].", node->GetNamePtr(),
                   static_cast<int64_t>(role));
    GE_ASSERT_SUCCESS(
        ascgen_utils::indirect_load::SetTemplateRole(node, ascgen_utils::indirect_load::TemplateRole::kStridedUbPath));
  }
  return af::SUCCESS;
}

af::Status ApplyIndirectLoadPathLayout(const NodePath &path,
                                       const ascgen_utils::indirect_load::IndirectLoadTensorLayout &layout,
                                       bool needs_alignment) {
  if (layout.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kZeroStrideCompact) {
    return ApplyZeroStrideCompactView(path, layout);
  }
  if (needs_alignment) {
    return AnnotateStridedUbPath(path);
  }
  return af::SUCCESS;
}

void CollectInputRegionMembers(const af::AscNodePtr &indirect_load, size_t input_index, NodeSet &region) {
  const af::AscNodePtr root = ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_index);
  if (root == nullptr) {
    return;
  }
  NodePath pending = {root};
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr node = pending[cursor];
    if (node == nullptr) {
      continue;
    }
    if (!region.emplace(node.get()).second) {
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Load>(node)) {
      continue;
    }
    for (size_t i = 0UL; i < node->inputs.Size(); ++i) {
      const af::AscNodePtr producer = ascgen_utils::indirect_load::GetInputProducer(node, i);
      if (producer != nullptr) {
        pending.emplace_back(producer);
      }
    }
  }
}

bool CollectSimtBackwardRegion(const NodePath &roots, const af::AscNodePtr &indirect_load, NodeSet &region) {
  NodePath pending = roots;
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr node = pending[cursor];
    if (node == nullptr) {
      return false;
    }
    if (node == indirect_load || IsInputRegionBoundary(node)) {
      continue;
    }
    if (!region.emplace(node.get()).second || af::ops::IsOps<af::ascir_op::Load>(node)) {
      continue;
    }
    if (node->inputs.Size() == 0UL) {
      return false;
    }
    for (size_t i = 0UL; i < node->inputs.Size(); ++i) {
      const af::AscNodePtr producer = ascgen_utils::indirect_load::GetInputProducer(node, i);
      if (producer == nullptr) {
        return false;
      }
      pending.emplace_back(producer);
    }
  }
  return true;
}

void CollectSimtFusedRegionMembers(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                   NodeSet &region) {
  region.clear();
  const af::AscNodePtr index_root =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
  const af::AscNodePtr output_terminal = analysis.post_reduce == nullptr
                                             ? analysis.output_store
                                             : ascgen_utils::indirect_load::GetInputProducer(analysis.post_reduce, 0UL);
  if (index_root == nullptr || output_terminal == nullptr) {
    return;
  }
  if (!CollectSimtBackwardRegion({index_root, output_terminal}, indirect_load, region)) {
    region.clear();
  }
}

// Normalize the IndirectLoad axis to a non-negative output axis index. Return the sentinel for invalid metadata.
size_t GetIndirectLoadAxisIndex(const af::AscNodePtr &node) {
  if (node == nullptr) {
    return kIndirectLoadInvalidAxisIndex;
  }
  const auto outputs = node->outputs();
  if (outputs.empty() || outputs[0] == nullptr) {
    return kIndirectLoadInvalidAxisIndex;
  }
  if (node->attr.ir_attr == nullptr) {
    return kIndirectLoadInvalidAxisIndex;
  }
  const auto *ir_attr = node->attr.ir_attr->DownCastTo<af::ascir_op::IndirectLoad::AscIndirectLoadIrAttrDef>();
  if (ir_attr == nullptr) {
    return kIndirectLoadInvalidAxisIndex;
  }
  int64_t axis = 0L;
  if (ir_attr->GetAxis(axis) != af::SUCCESS) {
    return kIndirectLoadInvalidAxisIndex;
  }
  const size_t rank = outputs[0]->attr.axis.size();
  if (rank > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return kIndirectLoadInvalidAxisIndex;
  }
  const int64_t rank_value = static_cast<int64_t>(rank);
  if (axis < -rank_value || axis >= rank_value) {
    return kIndirectLoadInvalidAxisIndex;
  }
  return static_cast<size_t>(axis < 0L ? axis + rank_value : axis);
}

af::Status BuildPostReduceLayout(const af::AscNodePtr &reduce, PostReduceLayout &layout, bool &is_legal) {
  is_legal = false;
  GE_ASSERT_TRUE(reduce->inputs().size() == 1UL && reduce->outputs().size() == 1UL,
                 "IndirectLoad post Reduce must be unary, node[%s] input num:%zu, output num:%zu.",
                 reduce->GetNamePtr(), reduce->inputs().size(), reduce->outputs().size());
  const auto input = reduce->inputs()[0];
  const auto output = reduce->outputs()[0];
  GE_ASSERT_NOTNULL(input);
  GE_ASSERT_NOTNULL(output);
  const auto &input_strides = input->attr.strides;
  const auto &output_strides = output->attr.strides;
  layout.axes = output->attr.axis;
  GE_ASSERT_TRUE(input->attr.axis.size() == layout.axes.size() && input->attr.repeats.size() == layout.axes.size() &&
                     output->attr.repeats.size() == layout.axes.size() &&
                     input_strides.size() == output_strides.size() && output_strides.size() == layout.axes.size(),
                 "IndirectLoad post Reduce metadata rank mismatch.");
  const auto reduce_axes = ScheduleUtils::CalcReduceAxes(input_strides, output_strides, layout.axes);
  layout.kinds.clear();
  layout.kinds.reserve(layout.axes.size());
  layout.first_reduce = layout.axes.size();
  for (size_t i = 0UL; i < layout.axes.size(); ++i) {
    const auto input_zero = af::SymbolicUtils::StaticCheckEq(input_strides[i], af::ops::Zero);
    const auto output_zero = af::SymbolicUtils::StaticCheckEq(output_strides[i], af::ops::Zero);
    if (input_zero == af::TriBool::kUnknown || output_zero == af::TriBool::kUnknown ||
        (input_zero == af::TriBool::kTrue && output_zero == af::TriBool::kFalse)) {
      return af::SUCCESS;
    }
    if (input_zero == af::TriBool::kTrue && output_zero == af::TriBool::kTrue) {
      layout.kinds.emplace_back(ReduceAxisKind::kIgnored);
    } else if (std::find(reduce_axes.begin(), reduce_axes.end(), layout.axes[i]) != reduce_axes.end()) {
      layout.kinds.emplace_back(ReduceAxisKind::kReduced);
      if (layout.first_reduce == layout.axes.size()) {
        layout.first_reduce = i;
      }
    } else {
      layout.kinds.emplace_back(ReduceAxisKind::kRetained);
    }
  }
  is_legal = true;
  return af::SUCCESS;
}

bool HasSupportedReduceSuffix(const PostReduceLayout &layout, size_t begin) {
  bool has_axis = false;
  bool has_reduce_axis = false;
  bool previous_is_reduce = false;
  size_t transitions = 0UL;
  for (size_t i = begin; i < layout.kinds.size(); ++i) {
    if (layout.kinds[i] == ReduceAxisKind::kIgnored) {
      continue;
    }
    const bool is_reduce = layout.kinds[i] == ReduceAxisKind::kReduced;
    if (has_axis && is_reduce != previous_is_reduce) {
      ++transitions;
    }
    has_axis = true;
    has_reduce_axis = has_reduce_axis || is_reduce;
    previous_is_reduce = is_reduce;
  }
  return has_reduce_axis && transitions <= 1UL;
}

af::Status ValidateSimtPostReduceLayout(const af::AscNodePtr &reduce, size_t &boundary, bool &is_legal) {
  is_legal = true;
  if (reduce == nullptr) {
    return af::SUCCESS;
  }
  PostReduceLayout layout;
  GE_ASSERT_SUCCESS(BuildPostReduceLayout(reduce, layout, is_legal));
  if (!is_legal) {
    return af::SUCCESS;
  }
  if (layout.first_reduce == layout.axes.size()) {
    is_legal = false;
    return af::SUCCESS;
  }
  boundary = layout.first_reduce;
  is_legal = HasSupportedReduceSuffix(layout, boundary);
  return af::SUCCESS;
}

af::Status ValidateSimdPostReduceLayout(const af::AscNodePtr &indirect_load, const af::AscNodePtr &reduce,
                                        bool &is_legal) {
  is_legal = true;
  if (reduce == nullptr) {
    return af::SUCCESS;
  }
  PostReduceLayout layout;
  GE_ASSERT_SUCCESS(BuildPostReduceLayout(reduce, layout, is_legal));
  if (!is_legal) {
    return af::SUCCESS;
  }
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                 indirect_load->GetNamePtr());
  const size_t boundary = axis_index;
  GE_ASSERT_TRUE(boundary <= layout.axes.size(), "IndirectLoad axis is out of range for post Reduce output.");

  for (size_t i = 0UL; i < boundary; ++i) {
    if (layout.kinds[i] == ReduceAxisKind::kReduced) {
      is_legal = false;
      return af::SUCCESS;
    }
  }
  is_legal = HasSupportedReduceSuffix(layout, boundary);
  return af::SUCCESS;
}

bool HasBroadcastMultiInputNode(const af::AscNodePtr &node, NodeSet &visited) {
  if (node == nullptr || !visited.emplace(node.get()).second) {
    return false;
  }
  if (node->inputs.Size() > 1UL) {
    for (size_t input_idx = 0UL; input_idx < node->inputs.Size(); ++input_idx) {
      const auto producer = ascgen_utils::indirect_load::GetInputProducer(node, input_idx);
      if (producer != nullptr && af::ops::IsOps<af::ascir_op::Broadcast>(producer)) {
        return true;
      }
    }
  }
  for (size_t input_idx = 0UL; input_idx < node->inputs.Size(); ++input_idx) {
    if (HasBroadcastMultiInputNode(ascgen_utils::indirect_load::GetInputProducer(node, input_idx), visited)) {
      return true;
    }
  }
  return false;
}

bool IsSkTemplateCandidateLegal(const af::AscNodePtr &indirect_load) {
  if (indirect_load == nullptr || indirect_load->GetOutDataNodesSize() != 1UL) {
    return false;
  }
  for (size_t input_idx = 0UL; input_idx < kIndirectLoadInputCount; ++input_idx) {
    const auto input_anchor = indirect_load->GetInDataAnchor(input_idx);
    if (input_anchor == nullptr || input_anchor->GetPeerOutAnchor() == nullptr) {
      return false;
    }
  }
  NodeSet visited;
  for (size_t input_idx = 0UL; input_idx < kIndirectLoadInputCount; ++input_idx) {
    if (HasBroadcastMultiInputNode(ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_idx), visited)) {
      return false;
    }
  }
  const auto output_consumer = ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load);
  if (HasBroadcastMultiInputNode(output_consumer, visited)) {
    return false;
  }
  return true;
}

af::Status ValidateIndirectLoadNode(const af::AscNodePtr &indirect_load) {
  const auto outputs = indirect_load->outputs();
  GE_ASSERT_TRUE(outputs.size() == kIndirectLoadOutputCount,
                 "IndirectLoad node[%s] output count is invalid, actual:%zu.", indirect_load->GetNamePtr(),
                 outputs.size());
  const auto output = outputs[0];
  GE_ASSERT_NOTNULL(output, "IndirectLoad output tensor is null.");
  const size_t output_rank = output->attr.axis.size();
  // 注意：AscNodeInputs/AscNodeOutputs 的 operator() 每次调用都会重建内部快照，先前调用
  // 返回的 tensor 指针会悬垂。因此这里只调用一次 outputs()，并提前拷贝后续要用的数据。
  const auto output_repeats = output->attr.repeats;
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                 indirect_load->GetNamePtr());
  const auto inputs = indirect_load->inputs();
  GE_ASSERT_TRUE(inputs.size() == kIndirectLoadInputCount, "IndirectLoad node[%s] input count is invalid, actual:%zu.",
                 indirect_load->GetNamePtr(), inputs.size());
  const auto input = inputs[ascgen_utils::indirect_load::kInputTensorIndex];
  const auto index = inputs[ascgen_utils::indirect_load::kIndexTensorIndex];
  GE_ASSERT_NOTNULL(input, "IndirectLoad input tensor is null.");
  GE_ASSERT_NOTNULL(index, "IndirectLoad index tensor is null.");
  GE_ASSERT_TRUE(input->attr.repeats.size() == output_rank && index->attr.repeats.size() == output_rank &&
                     output_repeats.size() == output_rank,
                 "IndirectLoad logical shape rank is invalid.");
  for (size_t i = 0UL; i < output_rank; ++i) {
    GE_ASSERT_TRUE(af::SymbolicUtils::StaticCheckEq(index->attr.repeats[i], output_repeats[i]) == af::TriBool::kTrue,
                   "IndirectLoad index and output logical shape must match.");
    if (i != axis_index) {
      GE_ASSERT_TRUE(af::SymbolicUtils::StaticCheckLt(input->attr.repeats[i], output_repeats[i]) != af::TriBool::kTrue,
                     "IndirectLoad input dimension %zu must not be smaller than index/output outside axis %zu.", i,
                     axis_index);
    }
  }
  return af::SUCCESS;
}

af::Status MergeAxesForTemplate(af::AscGraph &graph, const std::vector<af::AxisId> &axes, const std::string &name,
                                af::AxisId &merged_axis) {
  GE_ASSERT_TRUE(!axes.empty(), "IndirectLoad merge axis source is empty, name:%s.", name.c_str());
  merged_axis = axes.size() == 1UL ? axes.front() : graph.MergeAxis(axes, name)->id;
  return af::SUCCESS;
}

af::Status CreateFixedTileSplit(af::AscGraph &graph, af::AxisId axis_id, af::AxisId &outer_id, af::AxisId &inner_id) {
  const auto *axis = graph.FindAxis(axis_id);
  GE_ASSERT_NOTNULL(axis, "IndirectLoad fixed tile axis %ld is not found.", axis_id);
  const af::Expression outer_size = axis->size;
  const af::Expression inner_size = af::sym::kSymbolOne;
  outer_id =
      graph.CreateAxis(axis->name + "T", ascir::Axis::Type::kAxisTypeTileOuter, outer_size, {axis_id}, af::kIdNone).id;
  inner_id =
      graph.CreateAxis(axis->name + "t", ascir::Axis::Type::kAxisTypeTileInner, inner_size, {axis_id}, outer_id).id;
  auto *outer_axis = graph.FindAxis(outer_id);
  GE_ASSERT_NOTNULL(outer_axis, "IndirectLoad fixed tile outer axis %ld is not found.", outer_id);
  outer_axis->split_pair_other_id = inner_id;
  return af::SUCCESS;
}

af::Status CopyBoundaryTensorAttr(const af::AscNodePtr &src_node, size_t src_output_idx,
                                  const af::AscNodePtr &dst_node) {
  GE_ASSERT_TRUE(src_output_idx < src_node->outputs().size(),
                 "IndirectLoad SK boundary output index %zu is out of range for node[%s].", src_output_idx,
                 src_node->GetNamePtr());
  GE_ASSERT_TRUE(!dst_node->outputs().empty(), "IndirectLoad SK boundary node[%s] has no output.",
                 dst_node->GetNamePtr());
  auto dst_op_desc = dst_node->GetOpDesc();
  GE_ASSERT_NOTNULL(dst_op_desc);
  auto dst_node_attr = dst_op_desc->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  auto src_node_attr = src_node->GetOpDesc()->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  GE_ASSERT_NOTNULL(dst_node_attr);
  if (src_node_attr != nullptr) {
    dst_node_attr->sched = src_node_attr->sched;
    if (src_node_attr->ir_attr != nullptr) {
      dst_node_attr->ir_attr = src_node_attr->ir_attr->Clone();
    }
  }
  auto output_desc = dst_op_desc->MutableOutputDesc(0UL);
  GE_ASSERT_NOTNULL(output_desc);
  auto output_attr = output_desc->GetOrCreateAttrsGroup<af::AscTensorAttr>();
  GE_ASSERT_NOTNULL(output_attr);
  *output_attr = src_node->outputs()[src_output_idx]->attr;
  return af::SUCCESS;
}

af::Status CopyWorkspaceTensorAttr(const af::AscNodePtr &boundary_node, const af::AscNodePtr &workspace_node) {
  GE_ASSERT_TRUE(!boundary_node->outputs().empty(), "IndirectLoad SK boundary node[%s] has no output.",
                 boundary_node->GetNamePtr());
  GE_ASSERT_TRUE(!workspace_node->outputs().empty(), "IndirectLoad SK workspace node[%s] has no output.",
                 workspace_node->GetNamePtr());
  workspace_node->outputs()[0]->attr = boundary_node->outputs()[0]->attr;
  return af::SUCCESS;
}

af::Status InsertWorkspaceBoundary(af::AscGraph &graph, const af::AscNodePtr &src_node, size_t src_output_idx,
                                   const af::AscNodePtr &dst_node, size_t dst_input_idx,
                                   const std::string &boundary_name, bool align_store, bool align_load) {
  const auto src_anchor = src_node->GetOutDataAnchor(src_output_idx);
  const auto dst_anchor = dst_node->GetInDataAnchor(dst_input_idx);
  GE_ASSERT_NOTNULL(src_anchor, "IndirectLoad SK source anchor is null for boundary[%s].", boundary_name.c_str());
  GE_ASSERT_NOTNULL(dst_anchor, "IndirectLoad SK destination anchor is null for boundary[%s].", boundary_name.c_str());
  GE_ASSERT_TRUE(dst_anchor->GetPeerOutAnchor() == src_anchor,
                 "IndirectLoad SK boundary[%s] does not match edge %s:%zu -> %s:%zu.", boundary_name.c_str(),
                 src_node->GetNamePtr(), src_output_idx, dst_node->GetNamePtr(), dst_input_idx);

  const std::string workspace_name = boundary_name + "_workspace";
  af::ascir_op::Workspace workspace_pre(workspace_name.c_str());
  af::ascir_op::Workspace workspace_post(workspace_name.c_str());
  af::ascir_op::Load load((boundary_name + "_load").c_str());
  af::ascir_op::Store store((boundary_name + "_store").c_str());
  auto workspace_pre_node = graph.AddNode(workspace_pre);
  auto workspace_post_node = graph.AddNode(workspace_post);
  auto load_node = graph.AddNode(load);
  auto store_node = graph.AddNode(store);
  GE_ASSERT_NOTNULL(workspace_pre_node);
  GE_ASSERT_NOTNULL(workspace_post_node);
  GE_ASSERT_NOTNULL(load_node);
  GE_ASSERT_NOTNULL(store_node);

  GE_ASSERT_SUCCESS(CopyBoundaryTensorAttr(src_node, src_output_idx, load_node));
  GE_ASSERT_SUCCESS(CopyBoundaryTensorAttr(src_node, src_output_idx, store_node));
  GE_ASSERT_SUCCESS(CopyWorkspaceTensorAttr(store_node, workspace_pre_node));
  GE_ASSERT_SUCCESS(CopyWorkspaceTensorAttr(load_node, workspace_post_node));
  if (align_store) {
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
        store_node, ascgen_utils::indirect_load::TemplateRole::kStridedUbPath));
  }
  if (align_load) {
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
        load_node, ascgen_utils::indirect_load::TemplateRole::kStridedUbPath));
  }

  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(src_anchor, dst_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(src_anchor, store_node->GetInDataAnchor(0UL)));
  GE_ASSERT_GRAPH_SUCCESS(
      af::GraphUtils::AddEdge(store_node->GetOutDataAnchor(0UL), workspace_pre_node->GetInDataAnchor(0UL)));
  GE_ASSERT_GRAPH_SUCCESS(
      af::GraphUtils::AddEdge(workspace_post_node->GetOutDataAnchor(0UL), load_node->GetInDataAnchor(0UL)));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(load_node->GetOutDataAnchor(0UL), dst_anchor));
  return af::SUCCESS;
}

af::Status PartitionSkGraph(af::AscGraph &graph, const af::AscNodePtr &indirect_load, bool align_input_path,
                            bool align_index_path) {
  for (size_t input_idx = 0UL; input_idx < kIndirectLoadInputCount; ++input_idx) {
    const auto input_anchor = indirect_load->GetInDataAnchor(input_idx);
    GE_ASSERT_NOTNULL(input_anchor);
    const auto peer_out_anchor = input_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out_anchor);
    auto producer = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
    GE_ASSERT_NOTNULL(producer);
    const std::string role = input_idx == ascgen_utils::indirect_load::kInputTensorIndex ? "input" : "index";
    const bool align_path =
        input_idx == ascgen_utils::indirect_load::kInputTensorIndex ? align_input_path : align_index_path;
    GE_ASSERT_SUCCESS(
        InsertWorkspaceBoundary(graph, producer, static_cast<size_t>(peer_out_anchor->GetIdx()), indirect_load,
                                input_idx, indirect_load->GetName() + "_sk_" + role, align_path,
                                input_idx == ascgen_utils::indirect_load::kIndexTensorIndex && align_path));
  }

  const auto output_anchor = indirect_load->GetOutDataAnchor(0UL);
  GE_ASSERT_NOTNULL(output_anchor);
  const auto peer_input_anchors = output_anchor->GetPeerInDataAnchors();
  GE_ASSERT_TRUE(peer_input_anchors.size() == 1UL,
                 "IndirectLoad SK requires exactly one output consumer, node[%s], consumer count:%zu.",
                 indirect_load->GetNamePtr(), peer_input_anchors.size());
  const auto peer_input_anchor = *peer_input_anchors.begin();
  GE_ASSERT_NOTNULL(peer_input_anchor);
  auto consumer = std::dynamic_pointer_cast<af::AscNode>(peer_input_anchor->GetOwnerNode());
  GE_ASSERT_NOTNULL(consumer);
  GE_ASSERT_SUCCESS(InsertWorkspaceBoundary(graph, indirect_load, 0UL, consumer,
                                            static_cast<size_t>(peer_input_anchor->GetIdx()),
                                            indirect_load->GetName() + "_sk_output", false, false));
  return af::SUCCESS;
}

af::Status BuildSkPartitionOrder(const ascir::ImplGraph &graph, const af::AscNodePtr &indirect_load,
                                 std::vector<af::AscNodePtr> &node_order) {
  std::set<af::NodePtr> ordered_nodes;
  for (const char *role : {"input", "index", "output"}) {
    const std::string workspace_name = indirect_load->GetName() + "_sk_" + role + "_workspace";
    af::AscNodePtr workspace_pre;
    for (const auto &node : graph.GetAllNodes()) {
      if (node->GetName() == workspace_name && node->GetOutDataNodes().empty()) {
        workspace_pre = node;
        break;
      }
    }
    GE_ASSERT_NOTNULL(workspace_pre, "IndirectLoad SK terminal workspace[%s] is not found.", workspace_name.c_str());
    node_order.emplace_back(workspace_pre);
    ordered_nodes.emplace(workspace_pre);
  }
  std::vector<af::AscNodePtr> remaining_outputs;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetOutDataNodes().empty() && ordered_nodes.find(node) == ordered_nodes.end()) {
      remaining_outputs.emplace_back(node);
    }
  }
  std::sort(remaining_outputs.begin(), remaining_outputs.end(),
            [](const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) {
              return lhs->GetOpDescBarePtr()->GetId() < rhs->GetOpDescBarePtr()->GetId();
            });
  node_order.insert(node_order.end(), remaining_outputs.begin(), remaining_outputs.end());
  return af::SUCCESS;
}

af::Status BuildSimdInnerAxis(af::AscGraph &graph, const af::AscNodePtr &input_producer, size_t axis_index,
                              const char *name, ascir::AxisId &input_inner_axis) {
  GE_ASSERT_TRUE(!input_producer->outputs().empty(), "IndirectLoad SIMD input tensor producer has no output.");
  const auto input_axes = input_producer->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(axis_index < input_axes.size(), "IndirectLoad SIMD input axis index is out of range.");
  std::vector<ascir::AxisId> input_inner_axes(input_axes.begin() + static_cast<int64_t>(axis_index), input_axes.end());
  GE_ASSERT_SUCCESS(MergeAxesForTemplate(graph, input_inner_axes, name, input_inner_axis));
  return af::SUCCESS;
}

af::Status BuildSkInputInnerAxis(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t axis_index,
                                 ascir::AxisId &input_inner_axis) {
  const auto input_boundary = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  GE_ASSERT_TRUE(input_boundary != nullptr && af::ops::IsOps<af::ascir_op::Load>(input_boundary),
                 "IndirectLoad SK input boundary must be a Load node, node[%s].", indirect_load->GetNamePtr());
  return BuildSimdInnerAxis(graph, input_boundary, axis_index, kInputInnerAxisName, input_inner_axis);
}

af::Status BuildAxisViewByBoundary(af::AscGraph &graph, const std::vector<af::AxisId> &axes, size_t boundary,
                                   af::AxisId &outer_axis, af::AxisId &inner_axis) {
  GE_ASSERT_TRUE(!axes.empty(), "IndirectLoad output axis is empty.");
  GE_ASSERT_TRUE(boundary <= axes.size(), "IndirectLoad axis boundary is out of range.");
  const size_t split = boundary;
  const std::vector<af::AxisId> outer_axes(axes.begin(), axes.begin() + static_cast<int64_t>(split));
  const std::vector<af::AxisId> inner_axes(axes.begin() + static_cast<int64_t>(split), axes.end());
  if (outer_axes.empty()) {
    outer_axis = graph.CreateAxis(kSingleOuterAxisName, af::sym::kSymbolOne).id;
  } else {
    GE_ASSERT_SUCCESS(MergeAxesForTemplate(graph, outer_axes, kOuterAxisName, outer_axis));
  }
  if (!inner_axes.empty()) {
    GE_ASSERT_SUCCESS(MergeAxesForTemplate(graph, inner_axes, kInnerAxisName, inner_axis));
  }
  return af::SUCCESS;
}

af::Status NormalizeAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t boundary,
                                    ascir::AxisId input_inner_axis, ascir::AxisId index_inner_axis,
                                    bool simd_index_uses_output_inner_axis = false) {
  const auto output_axes = indirect_load->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(!output_axes.empty(), "IndirectLoad output axis is empty.");
  ascir::AxisId outer_axis = af::kIdNone;
  ascir::AxisId inner_axis = af::kIdNone;
  GE_ASSERT_SUCCESS(BuildAxisViewByBoundary(graph, output_axes, boundary, outer_axis, inner_axis));
  if (simd_index_uses_output_inner_axis) {
    index_inner_axis = inner_axis;
  }
  const af::Axis *outer = graph.FindAxis(outer_axis);
  GE_ASSERT_NOTNULL(outer, "IndirectLoad outer axis %ld is not found.", outer_axis);
  const bool synthetic_outer =
      outer->from.empty() && std::find(indirect_load->attr.sched.axis.begin(), indirect_load->attr.sched.axis.end(),
                                       outer_axis) == indirect_load->attr.sched.axis.end();
  af::AxisId tile_outer_axis = af::kIdNone;
  af::AxisId tile_inner_axis = af::kIdNone;
  GE_ASSERT_SUCCESS(CreateFixedTileSplit(graph, outer_axis, tile_outer_axis, tile_inner_axis));
  std::vector<af::AxisId> vectorized_axes;
  if (inner_axis != af::kIdNone) {
    const auto *inner = graph.FindAxis(inner_axis);
    GE_ASSERT_NOTNULL(inner, "IndirectLoad inner axis %ld is not found.", inner_axis);
    vectorized_axes =
        inner->type == ascir::Axis::Type::kAxisTypeMerged ? inner->from : std::vector<af::AxisId>{inner_axis};
  }
  ascgen_utils::indirect_load::TemplateAxes axes;
  axes.outer_axis = outer_axis;
  axes.inner_axis = inner_axis;
  axes.input_inner_axis = input_inner_axis;
  axes.index_inner_axis = index_inner_axis;
  axes.tile_outer_axis = tile_outer_axis;
  axes.tile_inner_axis = tile_inner_axis;
  axes.vectorized_axes = std::move(vectorized_axes);
  axes.synthetic_outer = synthetic_outer;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateAxes(indirect_load, axes));
  return af::SUCCESS;
}

af::Status RestoreSkTemplateAxes(std::vector<ascir::ImplGraph> &grouped_graphs) {
  for (auto &graph : grouped_graphs) {
    af::AscNodePtr indirect_load;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ValidateSingleIndirectLoadNode(graph, indirect_load));
    if (indirect_load == nullptr) {
      continue;
    }
    GE_ASSERT_TRUE(ascir::GetTemplateIdOrDefault(*indirect_load) == ascir::TemplateId::kIndirectLoadSK,
                   "IndirectLoad partitioned graph[%s] has unexpected template.", graph.GetName().c_str());
    const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
    GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                   indirect_load->GetNamePtr());
    ascir::AxisId input_inner_axis = af::kIdNone;
    GE_ASSERT_SUCCESS(BuildSkInputInnerAxis(graph, indirect_load, axis_index, input_inner_axis));
    GE_ASSERT_SUCCESS(NormalizeAxesForTemplate(graph, indirect_load, axis_index, input_inner_axis, af::kIdNone));
  }
  return af::SUCCESS;
}

af::Status ReplaceAxisPrefix(std::vector<af::AxisId> &target_axes, const std::vector<af::AxisId> &output_axes,
                             size_t axis_index) {
  GE_ASSERT_TRUE(target_axes.size() >= axis_index, "IndirectLoad SIMD input region has invalid axis rank.");
  std::copy_n(output_axes.begin(), axis_index, target_axes.begin());
  return af::SUCCESS;
}

af::Status ReplaceRegionAxisPrefix(const NodePath &region, const std::vector<af::AxisId> &output_axes,
                                   size_t axis_index, bool is_input_region) {
  for (const af::AscNodePtr &node : region) {
    if (!is_input_region && IsInputRegionBoundary(node)) {
      continue;
    }
    const auto role = ascgen_utils::indirect_load::GetTemplateRole(node);
    const bool should_replace = is_input_region ? ascgen_utils::indirect_load::ShouldApplyInputInnerVectorization(node)
                                                : role == ascgen_utils::indirect_load::TemplateRole::kSimdIndexPre;
    if (!should_replace) {
      continue;
    }
    GE_ASSERT_SUCCESS(ReplaceAxisPrefix(node->attr.sched.axis, output_axes, axis_index));
    for (const auto &output : node->outputs()) {
      GE_ASSERT_SUCCESS(ReplaceAxisPrefix(output->attr.axis, output_axes, axis_index));
    }
  }
  return af::SUCCESS;
}

af::Status NormalizeSimdAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                        const RewrittenGraphAnalysis &analysis) {
  const auto output_axes = indirect_load->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(!output_axes.empty(), "IndirectLoad SIMD output axis is empty.");
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                 indirect_load->GetNamePtr());
  ascir::AxisId input_inner_axis = af::kIdNone;
  GE_ASSERT_NOTNULL(analysis.input_root, "IndirectLoad SIMD input producer is missing.");
  GE_ASSERT_SUCCESS(BuildSimdInnerAxis(graph, analysis.input_root, axis_index, kInputInnerAxisName, input_inner_axis));
  ascir::AxisId index_inner_axis = af::kIdNone;
  const auto index_root =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
  GE_ASSERT_NOTNULL(index_root, "IndirectLoad SIMD index producer is missing.");
  if (analysis.simd_index_uses_output_inner_axis) {
    GE_ASSERT_SUCCESS(BuildSimdInnerAxis(graph, indirect_load, axis_index, kIndexInnerAxisName, index_inner_axis));
  }
  GE_ASSERT_SUCCESS(ReplaceRegionAxisPrefix(analysis.input_region, output_axes, axis_index, true));
  GE_ASSERT_SUCCESS(ReplaceRegionAxisPrefix(analysis.index_region, output_axes, axis_index, false));
  return NormalizeAxesForTemplate(graph, indirect_load, axis_index, input_inner_axis, index_inner_axis,
                                  analysis.simd_index_uses_output_inner_axis);
}

af::Status NormalizeSimtAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t boundary) {
  const auto output_axes = indirect_load->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(!output_axes.empty(), "IndirectLoad SIMT output axis is empty.");
  GE_ASSERT_TRUE(boundary <= output_axes.size(), "IndirectLoad SIMT boundary is out of range.");
  return NormalizeAxesForTemplate(graph, indirect_load, boundary, af::kIdNone, af::kIdNone);
}

bool CanEmitSimtScalar(const af::AscNodePtr &node) {
  const auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
  const auto *v2_impl = impl == nullptr ? nullptr : dynamic_cast<af::ascir::AscIrCodegenV2 *>(impl.get());
  return v2_impl != nullptr && v2_impl->IsSimtScalarSupported(*node);
}

// 一次下游行走同时拿输出链上的 Store 与 Reduce；多于一个 Reduce 时没有任何模板能支持，直接断言报错。
af::Status CollectOutputBoundaries(const af::AscNodePtr &indirect_load, RewrittenGraphAnalysis &analysis) {
  for (af::AscNodePtr current = ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load); current != nullptr;
       current = ascgen_utils::indirect_load::GetOnlyOutputConsumer(current)) {
    if (analysis.output_store == nullptr && af::ops::IsOps<af::ascir_op::Store>(current)) {
      analysis.output_store = current;
    }
    if (!ScheduleUtils::IsReduce(current)) {
      continue;
    }
    GE_ASSERT_TRUE(
        analysis.post_reduce == nullptr,
        "[IndirectLoad] IndirectLoad node[%s] post chain contains more than one Reduce: node[%s] and node[%s].",
        indirect_load->GetNamePtr(), analysis.post_reduce->GetNamePtr(), current->GetNamePtr());
    analysis.post_reduce = current;
  }
  return af::SUCCESS;
}

af::Status CollectRewrittenBoundaries(const af::AscNodePtr &indirect_load, RewrittenGraphAnalysis &analysis) {
  analysis.input_root =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
  analysis.index_root =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
  GE_ASSERT_SUCCESS(CollectOutputBoundaries(indirect_load, analysis));
  return af::SUCCESS;
}

bool CanMoveInputPreNode(const af::AscNodePtr &node, const af::AscNodePtr &consumer) {
  if (node == nullptr) {
    return false;
  }
  // Data and Load terminate the movable input-pre chain without being moved.
  if (af::ops::IsOps<af::ascir_op::Data>(node) || af::ops::IsOps<af::ascir_op::Load>(node)) {
    return false;
  }
  // Branching, control dependencies, and non-unary nodes cannot be safely reordered across IndirectLoad.
  if (node->inputs.Size() != 1UL || HasControlEdge(node) ||
      ascgen_utils::indirect_load::GetOnlyOutputConsumer(node) != consumer) {
    return false;
  }
  // Ordinary unary elementwise nodes can be moved; other compute types remain on the input side.
  return ScheduleUtils::IsElewise(node);
}

bool IsSimdInputShapeEnlarged(const af::AscNodePtr &indirect_load) {
  af::Expression input_numel = af::sym::kSymbolOne;
  for (const af::Expression &repeat :
       indirect_load->inputs()[ascgen_utils::indirect_load::kInputTensorIndex]->attr.repeats) {
    input_numel = input_numel * repeat;
  }
  af::Expression output_numel = af::sym::kSymbolOne;
  for (const af::Expression &repeat : indirect_load->outputs()[0]->attr.repeats) {
    output_numel = output_numel * repeat;
  }
  return af::SymbolicUtils::StaticCheckGt(output_numel, input_numel) == af::TriBool::kTrue;
}

af::Status MoveInputPreNode(const af::AscNodePtr &node, const af::AscNodePtr &indirect_load) {
  const auto node_in = node->GetInDataAnchor(0UL);
  const auto indirect_in = indirect_load->GetInDataAnchor(ascgen_utils::indirect_load::kInputTensorIndex);
  GE_ASSERT_NOTNULL(node_in);
  GE_ASSERT_NOTNULL(indirect_in);
  const auto producer_out = node_in->GetPeerOutAnchor();
  const auto node_out = indirect_in->GetPeerOutAnchor();
  const auto indirect_out = indirect_load->GetOutDataAnchor(0UL);
  GE_ASSERT_NOTNULL(producer_out);
  GE_ASSERT_NOTNULL(node_out);
  GE_ASSERT_NOTNULL(indirect_out);
  const auto producer = std::dynamic_pointer_cast<af::AscNode>(producer_out->GetOwnerNode());
  GE_ASSERT_NOTNULL(producer);
  GE_ASSERT_TRUE(node_out->GetOwnerNode() == node, "IndirectLoad input-pre edge does not match node[%s].",
                 node->GetNamePtr());
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(node_out, indirect_in, producer_out));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(producer_out, node_in, indirect_out));
  const auto peer_inputs = indirect_out->GetPeerInDataAnchors();
  for (const auto &peer_in : peer_inputs) {
    if (peer_in != node_in) {
      GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(indirect_out, peer_in, node_out));
    }
  }

  const auto indirect_output = indirect_load->outputs().front();
  const size_t producer_output_index = static_cast<size_t>(producer_out->GetIdx());
  const size_t node_output_index = static_cast<size_t>(node_out->GetIdx());
  GE_ASSERT_TRUE(producer_output_index < producer->outputs().size(),
                 "IndirectLoad input-pre producer output index is invalid.");
  GE_ASSERT_TRUE(node_output_index < node->outputs().size(), "IndirectLoad input-pre output index is invalid.");
  const auto node_output = node->outputs()[node_output_index];
  node->attr.sched.axis = indirect_load->attr.sched.axis;
  node_output->attr.axis = indirect_output->attr.axis;
  node_output->attr.repeats = indirect_output->attr.repeats;
  node_output->attr.strides = indirect_output->attr.strides;
  indirect_output->attr.dtype = producer->outputs()[producer_output_index]->attr.dtype;
  GELOGD("[IndirectLoad] Move input pre node[%s] after IndirectLoad[%s].", node->GetNamePtr(),
         indirect_load->GetNamePtr());
  return af::SUCCESS;
}

af::Status RewriteInputPreNodes(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                ascir::TemplateId template_id) {
  if (HasControlEdge(indirect_load)) {
    GELOGI("[IndirectLoad] Skip moving input-pre nodes for node[%s].", indirect_load->GetNamePtr());
    return af::SUCCESS;
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimd && IsSimdInputShapeEnlarged(indirect_load)) {
    const auto &input_shape = indirect_load->inputs()[ascgen_utils::indirect_load::kInputTensorIndex]->attr.repeats;
    const auto &output_shape = indirect_load->outputs()[0]->attr.repeats;
    GELOGI("[IndirectLoad] Skip moving SIMD input-pre nodes for node[%s]: input shape[%s], output shape[%s].",
           indirect_load->GetNamePtr(), af::ToString(input_shape).c_str(), af::ToString(output_shape).c_str());
    return af::SUCCESS;
  }
  NodePath movable_nodes;
  af::AscNodePtr consumer = indirect_load;
  for (af::AscNodePtr node =
           ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
       CanMoveInputPreNode(node, consumer); node = ascgen_utils::indirect_load::GetInputProducer(node, 0UL)) {
    movable_nodes.emplace_back(node);
    consumer = node;
  }
  if (movable_nodes.empty()) {
    return af::SUCCESS;
  }
  for (const af::AscNodePtr &node : movable_nodes) {
    GE_ASSERT_SUCCESS(MoveInputPreNode(node, indirect_load));
  }
  return ScheduleUtils::TopologicalSorting(graph);
}

af::Status RewriteSimtInputBroadcast(const af::AscNodePtr &indirect_load, InputViewPlan &input_plan,
                                     bool &is_candidate_legal) {
  const af::AscNodePtr input_producer =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
  if (!IsBroadcastNode(input_producer)) {
    return af::SUCCESS;
  }

  const auto broadcast_iter = std::find(input_plan.path.begin(), input_plan.path.end(), input_producer);
  GE_ASSERT_TRUE(broadcast_iter != input_plan.path.end(),
                 "SIMT direct input Broadcast is missing from collected path.");
  input_plan.path_broadcast_index = static_cast<int64_t>(std::distance(input_plan.path.begin(), broadcast_iter));
  GE_ASSERT_SUCCESS(FoldBroadcastPath(input_plan, ascir::TemplateId::kIndirectLoadSimt, is_candidate_legal));
  return af::SUCCESS;
}

// 不区分一元/二元路径，沿所有上游数据边收集 Load；Load 节点是输入边界，停止继续回溯。
void CollectReachableLoads(const af::AscNodePtr &root, NodeSet &visited, NodePath &loads) {
  if (root == nullptr) {
    return;
  }
  NodePath pending{root};
  while (!pending.empty()) {
    const af::AscNodePtr current = pending.back();
    pending.pop_back();
    if (current == nullptr || !visited.emplace(current.get()).second) {
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Load>(current)) {
      loads.emplace_back(current);
      continue;
    }
    for (size_t input_index = 0UL; input_index < current->inputs.Size(); ++input_index) {
      pending.emplace_back(ascgen_utils::indirect_load::GetInputProducer(current, input_index));
    }
  }
}

af::Status CompleteInputDataTensorAttrs(const RewrittenGraphAnalysis &analysis) {
  // 仅对可达 Load 补齐对应 Data；没有 Load 的路径无需处理。
  NodeSet visited;
  for (const af::AscNodePtr &root : {analysis.input_root, analysis.index_root}) {
    NodePath loads;
    CollectReachableLoads(root, visited, loads);
    for (const af::AscNodePtr &load : loads) {
      const af::AscNodePtr data = ascgen_utils::indirect_load::GetInputProducer(load, 0UL);
      GE_ASSERT_NOTNULL(data, "IndirectLoad input Load[%s] has no producer.", load->GetNamePtr());
      GE_ASSERT_TRUE(af::ops::IsOps<af::ascir_op::Data>(data), "IndirectLoad input Load[%s] producer is not Data.",
                     load->GetNamePtr());
      const auto load_outputs = load->outputs();
      const auto data_outputs = data->outputs();
      GE_ASSERT_TRUE(!load_outputs.empty() && !data_outputs.empty(),
                     "IndirectLoad input Load[%s] or Data[%s] has no output.", load->GetNamePtr(), data->GetNamePtr());
      const auto load_out = load_outputs.front();
      const auto data_out = data_outputs.front();
      GE_ASSERT_NOTNULL(load_out, "IndirectLoad input Load[%s] output is null.", load->GetNamePtr());
      GE_ASSERT_NOTNULL(data_out, "IndirectLoad input Data[%s] output is null.", data->GetNamePtr());
      GE_ASSERT_TRUE(!load_out->attr.axis.empty() && load_out->attr.axis.size() == load_out->attr.repeats.size() &&
                         load_out->attr.axis.size() == load_out->attr.strides.size(),
                     "IndirectLoad input Load[%s] tensor attributes are incomplete.", load->GetNamePtr());
      GELOGD("[IndirectLoad] Complete tensor attrs from Load[%s] to Data[%s].", load->GetNamePtr(), data->GetNamePtr());
      data_out->attr.axis = load_out->attr.axis;
      data_out->attr.repeats = load_out->attr.repeats;
      data_out->attr.strides = load_out->attr.strides;
      data_out->attr.dtype = load_out->attr.dtype;
    }
  }
  return af::SUCCESS;
}

// 收集 IndirectLoad 两个输入的路径，并按模板规则选出待处理的 Broadcast：SIMD 只看直接生产者，SIMT/SK
// 沿单输入链回溯并查找最近 Broadcast。
void CollectInputPaths(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                       PhysicalViewPreparation &preparation) {
  const bool collect_full_path = template_id != ascir::TemplateId::kIndirectLoadSimd;
  for (size_t input_index :
       {ascgen_utils::indirect_load::kInputTensorIndex, ascgen_utils::indirect_load::kIndexTensorIndex}) {
    InputViewPlan &plan =
        input_index == ascgen_utils::indirect_load::kInputTensorIndex ? preparation.input : preparation.index;
    NodePath &path = plan.path;
    plan.path_broadcast_index = kInvalidBroadcastIndex;
    for (af::AscNodePtr current = ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_index);
         current != nullptr; current = collect_full_path && current->inputs.Size() == 1UL
                                           ? ascgen_utils::indirect_load::GetInputProducer(current, 0UL)
                                           : nullptr) {
      if (plan.path_broadcast_index == kInvalidBroadcastIndex && IsBroadcastNode(current)) {
        plan.path_broadcast_index = static_cast<int64_t>(path.size());
      }
      path.emplace_back(current);
    }
  }
}

af::Status AnalyzeInputPath(const af::AscNodePtr &indirect_load, size_t input_idx, ascir::TemplateId template_id,
                            InputViewPlan *const plans[], bool &is_path_supported) {
  // 返回 SUCCESS 表示分析流程正常；is_path_supported 为 false 表示当前模板淘汰，不中止 Generate。
  is_path_supported = true;
  const auto inputs = indirect_load->inputs();
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index is invalid.");
  InputViewPlan &plan = *plans[input_idx];
  const auto &logical_attr = inputs[input_idx]->attr;
  plan.simd_index_uses_output_inner_axis = false;
  ascgen_utils::indirect_load::LogicalTensorView view;
  const bool has_broadcast = plan.path_broadcast_index != kInvalidBroadcastIndex;
  if (!has_broadcast) {
    // 无直连广播（SIMD/SIMT）或无广播（SK）：按逻辑视图直接分类，非直连广播由通用逻辑承载。
    view = {logical_attr.axis, logical_attr.repeats, logical_attr.strides};
  } else {
    const size_t broadcast_index = static_cast<size_t>(plan.path_broadcast_index);
    const af::AscNodePtr &broadcast = plan.path[broadcast_index];
    // 取源物理属性前保留源存在性检查；SIMD/SK 的路径折叠安全性在 FoldBroadcastPath 阶段统一校验。
    if (ascgen_utils::indirect_load::GetInputProducer(broadcast, 0UL) == nullptr) {
      GELOGI("[IndirectLoad] Reject candidate[%d]: Broadcast node[%s] source is invalid.",
             static_cast<int32_t>(template_id), broadcast->GetNamePtr());
      is_path_supported = false;
      return af::SUCCESS;
    }
    // 折叠源物理属性：以广播源的物理属性重写逻辑视图的 strides，源形状为 1 而逻辑视图非 1 的维度置 stride 0。
    af::AscTensorAttr physical_attr;
    GE_ASSERT_SUCCESS(GetBroadcastPhysicalAttr(broadcast, template_id, physical_attr));
    GE_ASSERT_SUCCESS(BuildBroadcastLogicalView(logical_attr, physical_attr, template_id, view));
    if (!IsSupportedDirectOuterBroadcast(broadcast, view, physical_attr, axis_index, template_id)) {
      is_path_supported = false;
      return af::SUCCESS;
    }
  }

  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(view, plan.layout));
  if (plan.layout.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported) {
    GELOGI("[IndirectLoad] Reject candidate[%d]: input path layout%s is unsupported.",
           static_cast<int32_t>(template_id), has_broadcast ? " after Broadcast folding" : "");
    is_path_supported = false;
  }
  plan.simd_index_uses_output_inner_axis = template_id == ascir::TemplateId::kIndirectLoadSimd &&
                                           input_idx == ascgen_utils::indirect_load::kIndexTensorIndex && has_broadcast;
  return af::SUCCESS;
}

af::Status RewriteInputPathsForTemplate(ascir::TemplateId template_id, PhysicalViewPreparation &preparation,
                                        RewrittenGraphAnalysis &analysis, bool &is_candidate_legal) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    // SIMT 先完成 input-pre 搬移，再统一处理当前直连的 input Broadcast。
    return af::SUCCESS;
  }
  for (InputViewPlan *const plan : {&preparation.input, &preparation.index}) {
    GE_ASSERT_SUCCESS(FoldBroadcastPath(*plan, template_id, is_candidate_legal));
    if (!is_candidate_legal) {
      return af::SUCCESS;
    }
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    // SK 需要保留完整路径，供调用方统一写回物理视图。
    analysis.input_path = std::move(preparation.input.path);
    analysis.index_path = std::move(preparation.index.path);
  }
  return af::SUCCESS;
}

af::Status PreparePhysicalViews(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                                PhysicalViewPreparation &preparation, bool &is_candidate_legal) {
  // 输入/输出个数已在 Generate 入口统一校验，axis 由 AnalyzeRewrittenGraph 统一获取，此处不再重复获取。
  is_candidate_legal = true;
  CollectInputPaths(indirect_load, template_id, preparation);
  InputViewPlan *const plans[] = {&preparation.input, &preparation.index};
  for (size_t input_idx = 0UL; input_idx < kIndirectLoadInputCount; ++input_idx) {
    bool is_path_supported = true;
    GE_ASSERT_SUCCESS(AnalyzeInputPath(indirect_load, input_idx, template_id, plans, is_path_supported));
    if (!is_path_supported) {
      is_candidate_legal = false;
      const char *const path_name = input_idx == ascgen_utils::indirect_load::kIndexTensorIndex ? "index" : "input";
      GELOGI("[IndirectLoad] Reject candidate[%d]: %s path is unsupported.", static_cast<int32_t>(template_id),
             path_name);
      return af::SUCCESS;
    }
  }
  preparation.logical_view.input = preparation.input.layout;
  preparation.logical_view.index = preparation.index.layout;
  const auto outputs = indirect_load->outputs();
  const auto &output_attr = outputs.front()->attr;
  preparation.logical_view.output = {output_attr.axis, output_attr.repeats, output_attr.strides};
  ascgen_utils::indirect_load::IndirectLoadTensorLayout output_layout;
  GE_ASSERT_SUCCESS(
      ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(preparation.logical_view.output, output_layout));
  GE_ASSERT_TRUE(output_layout.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense,
                 "IndirectLoad node[%s] output layout is not dense.", indirect_load->GetNamePtr());
  // 本函数只做只读分析；Broadcast 删除与物理视图写回在 AnalyzeRewrittenGraph 中统一执行。
  return ascgen_utils::indirect_load::SetTemplateLogicalView(indirect_load, preparation.logical_view);
}

af::Status CollectRewrittenRegion(const af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                  ascir::TemplateId template_id, RewrittenGraphAnalysis &analysis) {
  NodeSet input_region_set;
  NodeSet index_region_set;
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    CollectInputRegionMembers(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex, input_region_set);
    CollectInputRegionMembers(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex, index_region_set);
  } else if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    // SIMT 的 index 链与输出链共享统一轴、下游统一处理，融合集整体归入 index_region；region 保持为空。
    CollectSimtFusedRegionMembers(indirect_load, analysis, index_region_set);
  } else {
    GE_ASSERT_TRUE(false, "IndirectLoad template id %d is invalid.", static_cast<int32_t>(template_id));
  }
  // 一次 GetAllNodes 收集 region 与 index_region，保证顺序确定性。
  for (const af::AscNodePtr &node : graph.GetAllNodes()) {
    if (index_region_set.count(node.get()) != 0UL) {
      analysis.index_region.emplace_back(node);
    }
    if (input_region_set.count(node.get()) != 0UL) {
      analysis.input_region.emplace_back(node);
    }
  }
  return af::SUCCESS;
}

af::Status ApplyTemplatePathLayouts(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                                    const PhysicalViewPreparation &preparation,
                                    const RewrittenGraphAnalysis &analysis) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    // SIMT 融合集（index_region）从 index/output root 回溯，不包含 IndirectLoad 的 input
    // root，因此需要单独写回直接输入生产者。
    const InputViewPlan *const m2_plans[] = {&preparation.input, &preparation.index};
    for (size_t input_idx = 0UL; input_idx < kIndirectLoadInputCount; ++input_idx) {
      const af::AscNodePtr producer = ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_idx);
      if (producer != nullptr) {
        GE_ASSERT_SUCCESS(ApplyIndirectLoadPathLayout(NodePath{producer}, m2_plans[input_idx]->layout, false));
      }
    }
    return af::SUCCESS;
  }

  const bool is_simd = template_id == ascir::TemplateId::kIndirectLoadSimd;
  const NodePath &input_path = is_simd ? analysis.input_region : analysis.input_path;
  const NodePath &index_path = is_simd ? analysis.index_region : analysis.index_path;
  const auto &input_layout = preparation.input.layout;
  const auto &index_layout = preparation.index.layout;
  GE_ASSERT_SUCCESS(ApplyIndirectLoadPathLayout(input_path, input_layout, analysis.align_input_path));
  GE_ASSERT_SUCCESS(ApplyIndirectLoadPathLayout(index_path, index_layout, analysis.align_index_path));
  return af::SUCCESS;
}

af::Status AnalyzeRewrittenGraph(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                 ascir::TemplateId template_id, bool &is_candidate_legal,
                                 RewrittenGraphAnalysis &analysis) {
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                 indirect_load->GetNamePtr());
  // 分析阶段：路径、布局分类、对齐标志（只读）
  PhysicalViewPreparation preparation;
  GE_ASSERT_SUCCESS(PreparePhysicalViews(indirect_load, template_id, preparation, is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  analysis.align_index_path = template_id != ascir::TemplateId::kIndirectLoadSimt &&
                              NeedsAlignedUbWindow(preparation.logical_view.index, axis_index);
  analysis.align_input_path = template_id != ascir::TemplateId::kIndirectLoadSimt &&
                              NeedsAlignedUbWindow(preparation.logical_view.input, axis_index);
  analysis.simd_index_uses_output_inner_axis = preparation.index.simd_index_uses_output_inner_axis;

  // 改写阶段：全部拓扑改动一次完成（Broadcast 删除/折叠、input-pre 搬移）
  GE_ASSERT_SUCCESS(RewriteInputPathsForTemplate(template_id, preparation, analysis, is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    // SK 不参与 region 收集，直接处理完整路径并结束。
    GE_ASSERT_SUCCESS(ApplyTemplatePathLayouts(indirect_load, template_id, preparation, analysis));
    return af::SUCCESS;
  }
  // 将 input-pre 单目元素链搬到 IndirectLoad 之后
  GE_ASSERT_SUCCESS(RewriteInputPreNodes(graph, indirect_load, template_id));
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    GE_ASSERT_SUCCESS(RewriteSimtInputBroadcast(indirect_load, preparation.input, is_candidate_legal));
    if (!is_candidate_legal) {
      return af::SUCCESS;
    }
  }

  // 收集阶段：一次遍历收集全部状态（改写定稿后无需重收）
  GE_ASSERT_SUCCESS(CollectRewrittenBoundaries(indirect_load, analysis));
  GE_ASSERT_SUCCESS(CollectRewrittenRegion(graph, indirect_load, template_id, analysis));

  // 路径布局处理：紧凑零 stride 写回物理视图，需对齐的 strided 路径标注 UB role。
  GE_ASSERT_SUCCESS(ApplyTemplatePathLayouts(indirect_load, template_id, preparation, analysis));
  return af::SUCCESS;
}

af::Status AnnotateSimdTemplateRoles(const RewrittenGraphAnalysis &analysis) {
  for (const af::AscNodePtr &node : analysis.input_region) {
    if (IsInputRegionBoundary(node)) {
      continue;
    }
    const auto role = ascgen_utils::indirect_load::GetTemplateRole(node);
    const auto simd_role = role == ascgen_utils::indirect_load::TemplateRole::kStridedUbPath
                               ? ascgen_utils::indirect_load::TemplateRole::kSimdInputPreStridedUbPath
                               : ascgen_utils::indirect_load::TemplateRole::kSimdInputPre;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(node, simd_role));
  }
  if (analysis.simd_index_uses_output_inner_axis) {
    for (const af::AscNodePtr &node : analysis.index_region) {
      if (IsInputRegionBoundary(node)) {
        continue;
      }
      GE_ASSERT_SUCCESS(
          ascgen_utils::indirect_load::SetTemplateRole(node, ascgen_utils::indirect_load::TemplateRole::kSimdIndexPre));
    }
  }
  return af::SUCCESS;
}

af::Status ValidateSimtTemplateRegion(const RewrittenGraphAnalysis &analysis, bool &is_candidate_legal) {
  is_candidate_legal = false;
  if (!af::ops::IsOps<af::ascir_op::Load>(analysis.input_root) || analysis.index_region.empty()) {
    return af::SUCCESS;
  }
  for (const af::AscNodePtr &node : analysis.index_region) {
    if (af::ops::IsOps<af::ascir_op::ScalarData>(node) || af::ops::IsOps<af::ascir_op::Scalar>(node)) {
      return af::SUCCESS;
    }
    if (HasControlEdge(node) || af::ops::IsOps<af::ascir_op::VectorFunc>(node)) {
      return af::SUCCESS;
    }
    const bool is_gm_boundary = af::ops::IsOps<af::ascir_op::Load>(node) || af::ops::IsOps<af::ascir_op::Store>(node);
    if (!is_gm_boundary && !CanEmitSimtScalar(node)) {
      GELOGD("[IndirectLoad] SIMT scalar codegen does not support node[%s, %s].", node->GetNamePtr(),
             node->GetTypePtr());
      return af::SUCCESS;
    }
  }
  is_candidate_legal = true;
  return af::SUCCESS;
}

af::Status AnnotateSimtTemplateRoles(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis) {
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
      analysis.input_root, ascgen_utils::indirect_load::TemplateRole::kSimtInputBoundary));
  GE_ASSERT_SUCCESS(
      ascgen_utils::indirect_load::SetTemplateRole(indirect_load, ascgen_utils::indirect_load::TemplateRole::kSimtOp));
  for (const af::AscNodePtr &node : analysis.index_region) {
    const auto role = (af::ops::IsOps<af::ascir_op::Load>(node) || af::ops::IsOps<af::ascir_op::Store>(node))
                          ? ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary
                          : ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(node, role));
  }
  return af::SUCCESS;
}

af::Status ValidateTemplate(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                            const RewrittenGraphAnalysis &analysis, size_t &boundary, bool &is_candidate_legal) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return ValidateSimdPostReduceLayout(indirect_load, analysis.post_reduce, is_candidate_legal);
  }
  GE_ASSERT_SUCCESS(ValidateSimtPostReduceLayout(analysis.post_reduce, boundary, is_candidate_legal));
  if (is_candidate_legal) {
    GE_ASSERT_SUCCESS(ValidateSimtTemplateRegion(analysis, is_candidate_legal));
  }
  return af::SUCCESS;
}

af::Status AnnotateTemplate(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                            const RewrittenGraphAnalysis &analysis) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return AnnotateSimdTemplateRoles(analysis);
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    return AnnotateSimtTemplateRoles(indirect_load, analysis);
  }
  GE_ASSERT_TRUE(false, "IndirectLoad template id %d is invalid.", static_cast<int32_t>(template_id));
}

af::Status NormalizeTemplateAxes(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                 ascir::TemplateId template_id, const RewrittenGraphAnalysis &analysis,
                                 size_t boundary) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return NormalizeSimdAxesForTemplate(graph, indirect_load, analysis);
  } else {
    return NormalizeSimtAxesForTemplate(graph, indirect_load, boundary);
  }
}

af::Status FinalizeTemplate(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return ::ascir::SetTemplateId(indirect_load, template_id);
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    GE_ASSERT_SUCCESS(::ascir::SetDcacheSize(indirect_load, kIndirectLoadSimtDcacheSize));
    return ::ascir::SetTemplateId(indirect_load, template_id);
  }
  GE_ASSERT_TRUE(false, "IndirectLoad template id %d is invalid.", static_cast<int32_t>(template_id));
}

af::Status ApplySkGraphPass(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                            const RewrittenGraphAnalysis &analysis, bool &is_candidate_legal) {
  is_candidate_legal = false;
  if (!IsSkTemplateCandidateLegal(indirect_load)) {
    GELOGI("[IndirectLoad] Reject SK candidate for node[%s]: candidate legality check failed.",
           indirect_load->GetNamePtr());
    return af::SUCCESS;
  }
  is_candidate_legal = true;
  GE_ASSERT_SUCCESS(::ascir::SetTemplateId(indirect_load, ascir::TemplateId::kIndirectLoadSK));
  GE_ASSERT_SUCCESS(PartitionSkGraph(graph, indirect_load, analysis.align_input_path, analysis.align_index_path));
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                 indirect_load->GetNamePtr());
  ascir::AxisId input_inner_axis = af::kIdNone;
  GE_ASSERT_SUCCESS(BuildSkInputInnerAxis(graph, indirect_load, axis_index, input_inner_axis));
  GE_ASSERT_SUCCESS(NormalizeAxesForTemplate(graph, indirect_load, axis_index, input_inner_axis, af::kIdNone));
  GE_ASSERT_SUCCESS(
      ascgen_utils::indirect_load::SetTemplateRole(indirect_load, ascgen_utils::indirect_load::TemplateRole::kSkOp));
  const auto input_boundary = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  GE_ASSERT_TRUE(input_boundary != nullptr && af::ops::IsOps<af::ascir_op::Load>(input_boundary),
                 "IndirectLoad SK input boundary must be a Load node, node[%s].", indirect_load->GetNamePtr());
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
      input_boundary, ascgen_utils::indirect_load::TemplateRole::kSkInputBoundary));
  return af::SUCCESS;
}

af::Status ApplyGraphPass(af::AscGraph &graph, const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                          bool &is_candidate_legal) {
  GELOGD("[IndirectLoad] Apply graph pass for node[%s], template_id[%d].", indirect_load->GetNamePtr(),
         static_cast<int32_t>(template_id));
  is_candidate_legal = true;
  RewrittenGraphAnalysis analysis;
  GE_ASSERT_SUCCESS(AnalyzeRewrittenGraph(graph, indirect_load, template_id, is_candidate_legal, analysis));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    return ApplySkGraphPass(graph, indirect_load, analysis, is_candidate_legal);
  }
  GE_ASSERT_SUCCESS(CompleteInputDataTensorAttrs(analysis));

  // The SIMT template overwrites this boundary from its post-reduce layout; without a post Reduce the
  // full output rank is the boundary. The SIMD template does not use this value.
  size_t boundary = indirect_load->outputs()[0]->attr.axis.size();
  GE_ASSERT_SUCCESS(ValidateTemplate(indirect_load, template_id, analysis, boundary, is_candidate_legal));
  if (!is_candidate_legal) {
    GELOGI("[IndirectLoad] Reject template[%d] for node[%s]: post-reduce or region validation failed.",
           static_cast<int32_t>(template_id), indirect_load->GetNamePtr());
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(AnnotateTemplate(indirect_load, template_id, analysis));
  GE_ASSERT_SUCCESS(NormalizeTemplateAxes(graph, indirect_load, template_id, analysis, boundary));
  return FinalizeTemplate(indirect_load, template_id);
}

std::string GenerateScoreFunc(const TemplateCase &template_case) {
  (void)template_case;
  return "int32_t CalcScore(AutofuseTilingData &tiling_data) {\n"
         "  (void)tiling_data;\n"
         "  return 0;\n"
         "}\n";
}

af::Status PartitionTaskGraph(const ascir::ImplGraph &graph, const af::AscNodePtr &indirect_load, bool is_sk_template,
                              std::vector<ascir::ImplGraph> &grouped_graphs) {
  if (!is_sk_template) {
    return ScheduleGroupGraphPartitioner::PartitionByConnectivity(graph, grouped_graphs);
  }
  std::vector<af::AscNodePtr> node_order;
  GE_ASSERT_SUCCESS(BuildSkPartitionOrder(graph, indirect_load, node_order));
  GE_ASSERT_SUCCESS(ScheduleGroupGraphPartitioner::PartitionByConnectivity(graph, grouped_graphs, node_order));
  return RestoreSkTemplateAxes(grouped_graphs);
}

af::Status RefreshTaskAxisSizes(std::vector<ascir::ImplGraph> &grouped_graphs, bool need_update_axis,
                                bool is_sk_template) {
  if ((!need_update_axis && !is_sk_template) || grouped_graphs.size() <= 1UL) {
    return af::SUCCESS;
  }
  for (auto &subgraph : grouped_graphs) {
    if (ScheduleUtils::FindFirstNodeOfType<af::ascir_op::Concat>(subgraph) != nullptr) {
      continue;
    }
    af::AscNodePtr subgraph_indirect_load;
    if (is_sk_template) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ValidateSingleIndirectLoadNode(subgraph, subgraph_indirect_load));
    }
    if (subgraph_indirect_load == nullptr) {
      GE_ASSERT_SUCCESS(ScheduleGroupGraphPartitioner::RefreshAxisSize(subgraph));
    }
  }
  return af::SUCCESS;
}

af::Status ReduceTaskGraphCount(std::vector<ascir::ImplGraph> &grouped_graphs, const OptimizerOptions &options) {
  if (options.graph_type != GraphType::kFusedAscBackend) {
    return af::SUCCESS;
  }
  const auto backend_spec = BackendSpec::GetInstance();
  GE_ASSERT_NOTNULL(backend_spec);
  return ScheduleGroupGraphPartitioner::ReduceGraphCount(grouped_graphs, backend_spec->max_group_num_per_compile_unit);
}
}  // namespace

Status IndirectLoadScheduleCaseGenerator::Generate(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                                                   std::vector<std::string> &score_functions) {
  af::AscNodePtr indirect_load;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ValidateSingleIndirectLoadNode(graph, indirect_load));
  if (indirect_load == nullptr) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(ValidateIndirectLoadNode(indirect_load));
  GELOGI("[IndirectLoad] Generate schedule candidates for graph[%s], node[%s].", graph.GetName().c_str(),
         indirect_load->GetNamePtr());
  const std::string indirect_load_name = indirect_load->GetName();
  const TemplateCase cases[] = {
      {ascir::TemplateId::kIndirectLoadSimd, ascgen_utils::indirect_load::Implementation::kDefault},
      {ascir::TemplateId::kIndirectLoadSimd, ascgen_utils::indirect_load::Implementation::kGatherApi},
      {ascir::TemplateId::kIndirectLoadSimt, ascgen_utils::indirect_load::Implementation::kDefault},
      {ascir::TemplateId::kIndirectLoadSK, ascgen_utils::indirect_load::Implementation::kDefault}};
  for (const TemplateCase &template_case : cases) {
    const ascir::TemplateId template_id = template_case.template_id;
    ascir::ImplGraph candidate_graph(graph.GetName().c_str());
    GE_ASSERT_TRUE(candidate_graph.CopyFrom(graph), "Failed to copy graph [%s].", graph.GetName().c_str());
    const af::AscNodePtr candidate_indirect_load = candidate_graph.FindNode(indirect_load_name.c_str());
    GE_ASSERT_NOTNULL(candidate_indirect_load, "Failed to find copied IndirectLoad node[%s].",
                      indirect_load_name.c_str());
    GE_ASSERT_SUCCESS(
        ascgen_utils::indirect_load::SetImplementation(candidate_indirect_load, template_case.implementation));
    bool is_candidate_legal = false;
    GE_ASSERT_SUCCESS(ApplyGraphPass(candidate_graph, candidate_indirect_load, template_id, is_candidate_legal));
    if (!is_candidate_legal) {
      GELOGW("[IndirectLoad] Skip illegal template candidate[%d] for node[%s].", static_cast<int32_t>(template_id),
             candidate_indirect_load->GetNamePtr());
      continue;
    }
    graphs.emplace_back(std::move(candidate_graph));
    score_functions.emplace_back(GenerateScoreFunc(template_case));
    GELOGI("[IndirectLoad] Add schedule candidate[%d, %d] for node[%s].", static_cast<int32_t>(template_id),
           static_cast<int32_t>(template_case.implementation), candidate_indirect_load->GetNamePtr());
  }
  return af::SUCCESS;
}

Status IndirectLoadScheduleCaseGenerator::GeneratorTask(ascir::HintGraph &optimize_graph,
                                                        std::vector<ScheduleTask> &tasks,
                                                        const OptimizerOptions &options) {
  bool need_update_axis = false;
  GE_ASSERT_SUCCESS(ScheduleGroupGraphPartitioner::NeedRefreshAxisSize(optimize_graph, need_update_axis));
  std::vector<ascir::ImplGraph> optimize_graphs;
  std::vector<std::string> score_funcs;
  GE_CHK_STATUS_RET(Generate(optimize_graph, optimize_graphs, score_funcs), "GenerateScheduleCases failed");
  for (size_t i = 0UL; i < optimize_graphs.size(); ++i) {
    const auto &graph = optimize_graphs[i];
    ScheduleTask task{graph, {}, score_funcs[i]};
    task.has_load_store_conversion = HasLoadStoreConversion();
    af::AscNodePtr indirect_load;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ValidateSingleIndirectLoadNode(graph, indirect_load));
    const bool is_sk_template =
        indirect_load != nullptr && ascir::GetTemplateIdOrDefault(*indirect_load) == ascir::TemplateId::kIndirectLoadSK;
    GE_CHK_STATUS_RET(PartitionTaskGraph(graph, indirect_load, is_sk_template, task.grouped_graphs),
                      "Failed to partition graph");
    GE_ASSERT_SUCCESS(RefreshTaskAxisSizes(task.grouped_graphs, need_update_axis, is_sk_template));
    GE_CHK_STATUS_RET(ReduceTaskGraphCount(task.grouped_graphs, options), "Failed to reduce graph count");
    tasks.emplace_back(std::move(task));
  }
  return af::SUCCESS;
}

}  // namespace optimize
