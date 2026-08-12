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
#include <set>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace optimize {
namespace {
constexpr int64_t kIndirectLoadSimtDcacheSize = 32 * 1024;
using NodePath = std::vector<af::AscNodePtr>;
using NodeSet = std::unordered_set<const af::AscNode *>;

struct TemplateCase {
  ascir::TemplateId template_id;
  ascgen_utils::indirect_load::Implementation implementation;
};

struct RewrittenGraphAnalysis {
  NodePath region;
  NodePath input_path;
  NodePath index_path;
  af::AscNodePtr input_root;
  af::AscNodePtr input_load;
  af::AscNodePtr index_load;
  af::AscNodePtr output_store;
  af::AscNodePtr post_reduce;
  bool align_input_path = false;
  bool align_index_path = false;
};

struct IndirectLoadGraphPaths {
  NodePath input;
  NodePath index;
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

af::Status ValidateBroadcastPath(const NodePath &path, const char *template_name, bool &is_candidate_legal) {
  const auto broadcast_iter = std::find_if(path.begin(), path.end(), [](const af::AscNodePtr &node) {
    return af::ops::IsOps<af::ascir_op::Broadcast>(node);
  });
  if (broadcast_iter == path.end()) {
    return af::SUCCESS;
  }
  const af::AscNodePtr &broadcast = *broadcast_iter;
  if (broadcast->inputs.Size() != 1UL || broadcast->outputs().size() != 1UL ||
      broadcast->GetOutDataNodesSize() != 1UL || HasControlEdge(broadcast)) {
    GELOGW("[IndirectLoad] Skip %s candidate because Broadcast pre path node[%s] is invalid.", template_name,
           broadcast->GetNamePtr());
    is_candidate_legal = false;
    return af::SUCCESS;
  }
  for (auto iter = path.begin(); iter != broadcast_iter; ++iter) {
    const af::AscNodePtr &element = *iter;
    if (element->inputs.Size() != 1UL || element->outputs().size() != 1UL || element->GetOutDataNodesSize() != 1UL ||
        HasControlEdge(element) || !ScheduleUtils::IsElewise(element)) {
      GELOGW("[IndirectLoad] Skip %s candidate because Broadcast post element node[%s] is invalid.", template_name,
             element->GetNamePtr());
      is_candidate_legal = false;
      return af::SUCCESS;
    }
  }
  return af::SUCCESS;
}

af::Status ValidateBroadcastPrePaths(const IndirectLoadGraphPaths &paths, const char *template_name,
                                     bool &is_candidate_legal) {
  GE_ASSERT_SUCCESS(ValidateBroadcastPath(paths.input, template_name, is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  return ValidateBroadcastPath(paths.index, template_name, is_candidate_legal);
}

void SetPhysicalWindowLayout(const af::AscTensorAttr &physical_attr, af::AscTensorAttr &attr) {
  attr.axis = physical_attr.axis;
  attr.repeats = physical_attr.repeats;
  attr.strides = physical_attr.strides;
  attr.vectorized_axis = physical_attr.vectorized_axis;
  attr.vectorized_strides = physical_attr.vectorized_strides;
}

af::Status InlineBroadcastPath(NodePath &path, const char *template_name) {
  const auto broadcast_iter = std::find_if(path.begin(), path.end(), [](const af::AscNodePtr &node) {
    return af::ops::IsOps<af::ascir_op::Broadcast>(node);
  });
  if (broadcast_iter == path.end()) {
    return af::SUCCESS;
  }
  const af::AscNodePtr broadcast = *broadcast_iter;
  const af::AscNodePtr producer = ascgen_utils::indirect_load::GetInputProducer(broadcast, 0UL);
  GE_ASSERT_TRUE(producer != nullptr && !producer->outputs().empty(),
                 "IndirectLoad %s Broadcast node[%s] source is invalid.", template_name, broadcast->GetNamePtr());
  const af::AscTensorAttr physical_attr = producer->outputs()[0]->attr;
  for (auto iter = path.begin(); iter != broadcast_iter; ++iter) {
    const af::AscNodePtr &node = *iter;
    node->attr.sched.axis = physical_attr.axis;
    for (const auto &output : node->outputs()) {
      GE_ASSERT_NOTNULL(output);
      SetPhysicalWindowLayout(physical_attr, output->attr);
    }
  }
  const auto owner_graph = broadcast->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph);
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::IsolateNodeOneIO(broadcast));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(owner_graph, broadcast));
  GELOGD("[IndirectLoad] Inline Broadcast node[%s] for stride-aware %s physical source window.",
         broadcast->GetNamePtr(), template_name);
  path.erase(broadcast_iter);
  return af::SUCCESS;
}

af::Status InlineBroadcastPrePaths(IndirectLoadGraphPaths &paths, const char *template_name) {
  GE_ASSERT_SUCCESS(InlineBroadcastPath(paths.input, template_name));
  return InlineBroadcastPath(paths.index, template_name);
}

af::Status ApplyPhysicalExecutionView(const NodePath &path,
                                      const ascgen_utils::indirect_load::IndirectLoadTensorLayout &layout) {
  if (layout.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kZeroStrideCompact) {
    return af::SUCCESS;
  }
  GE_ASSERT_TRUE(
      layout.axis_ids.size() == layout.physical_repeats.size() && layout.axis_ids.size() == layout.strides.size(),
      "IndirectLoad physical execution view rank mismatch.");
  for (const af::AscNodePtr &node : path) {
    if (node->attr.sched.axis.size() == layout.axis_ids.size()) {
      node->attr.sched.axis = layout.axis_ids;
    }
    for (const auto &output : node->outputs()) {
      GE_ASSERT_NOTNULL(output);
      if (output->attr.repeats.size() != layout.physical_repeats.size() ||
          output->attr.strides.size() != layout.strides.size()) {
        continue;
      }
      output->attr.axis = layout.axis_ids;
      output->attr.repeats = layout.physical_repeats;
      output->attr.strides = layout.strides;
    }
  }
  return af::SUCCESS;
}

af::Status ApplyPhysicalExecutionViews(const IndirectLoadGraphPaths &paths,
                                       const ascgen_utils::indirect_load::TemplateLogicalView &view) {
  GE_ASSERT_SUCCESS(ApplyPhysicalExecutionView(paths.input, view.input));
  return ApplyPhysicalExecutionView(paths.index, view.index);
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

void CollectInputRegionMembers(const af::AscNodePtr &indirect_load, size_t input_index, NodeSet &region) {
  const af::AscNodePtr root = ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_index);
  NodePath pending = {root};
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr node = pending[cursor];
    if (!region.emplace(node.get()).second) {
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Load>(node) || IsInputRegionBoundary(node)) {
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
    if (node == indirect_load || !region.emplace(node.get()).second) {
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Load>(node)) {
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

af::Status GetIndirectLoadAxis(const af::AscNodePtr &node, int64_t &axis) {
  GE_ASSERT_NOTNULL(node->attr.ir_attr, "IndirectLoad ir attr is null, node = %s", node->GetNamePtr());
  const auto *ir_attr = node->attr.ir_attr->DownCastTo<af::ascir_op::IndirectLoad::AscIndirectLoadIrAttrDef>();
  GE_ASSERT_NOTNULL(ir_attr, "IndirectLoad ir attr type is invalid, node = %s", node->GetNamePtr());
  GE_ASSERT_GRAPH_SUCCESS(ir_attr->GetAxis(axis), "Failed to get IndirectLoad axis, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

af::Status BuildPostReduceLayout(const af::AscNodePtr &reduce, PostReduceLayout &layout, bool &is_legal) {
  is_legal = false;
  GE_ASSERT_TRUE(!reduce->inputs().empty() && !reduce->outputs().empty(),
                 "IndirectLoad post Reduce tensor is missing.");
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
  int64_t axis = 0L;
  GE_ASSERT_SUCCESS(GetIndirectLoadAxis(indirect_load, axis));
  const int64_t rank = static_cast<int64_t>(layout.axes.size());
  GE_ASSERT_TRUE(axis >= -rank && axis < rank, "IndirectLoad axis is out of range for post Reduce output.");
  const size_t boundary = static_cast<size_t>(axis < 0L ? axis + rank : axis);

  for (size_t i = 0UL; i < boundary; ++i) {
    if (layout.kinds[i] == ReduceAxisKind::kReduced) {
      is_legal = false;
      return af::SUCCESS;
    }
  }
  is_legal = HasSupportedReduceSuffix(layout, boundary);
  return af::SUCCESS;
}

bool IsSkTemplateCandidateLegal(const af::AscNodePtr &indirect_load) {
  if (indirect_load == nullptr || indirect_load->GetOutDataNodesSize() != 1UL) {
    return false;
  }
  for (size_t input_idx = 0UL; input_idx < 2UL; ++input_idx) {
    const auto input_anchor = indirect_load->GetInDataAnchor(input_idx);
    if (input_anchor == nullptr || input_anchor->GetPeerOutAnchor() == nullptr) {
      return false;
    }
  }
  return true;
}

af::Status ValidateIndirectLoadInputRank(const af::AscNodePtr &indirect_load, size_t input_idx, size_t output_rank) {
  auto input_anchor = indirect_load->GetInDataAnchor(input_idx);
  GE_ASSERT_NOTNULL(input_anchor, "IndirectLoad input%zu anchor is null.", input_idx);
  auto peer_out_anchor = input_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor, "IndirectLoad input%zu peer anchor is null.", input_idx);

  auto input_node = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
  GE_ASSERT_NOTNULL(input_node, "IndirectLoad input%zu node is invalid.", input_idx);
  const size_t output_idx = static_cast<size_t>(peer_out_anchor->GetIdx());
  const auto input_outputs = input_node->outputs();
  GE_ASSERT_TRUE(output_idx < input_outputs.size(),
                 "IndirectLoad input%zu output index %zu is out of range, output num:%zu.", input_idx, output_idx,
                 input_outputs.size());
  const auto input_output = input_outputs[output_idx];
  GE_ASSERT_NOTNULL(input_output, "IndirectLoad input%zu output tensor is null.", input_idx);
  const size_t input_rank = input_output->attr.axis.size();
  GE_ASSERT_TRUE(input_rank == output_rank,
                 "IndirectLoad input%zu rank must equal output rank, input rank:%zu, output rank:%zu.", input_idx,
                 input_rank, output_rank);
  return af::SUCCESS;
}

af::Status ValidateIndirectLoadNode(const af::AscNodePtr &indirect_load) {
  GE_ASSERT_NOTNULL(indirect_load, "IndirectLoad node is null.");
  const auto outputs = indirect_load->outputs();
  GE_ASSERT_TRUE(!outputs.empty(), "IndirectLoad graph is invalid.");
  const auto output = outputs[0];
  GE_ASSERT_NOTNULL(output, "IndirectLoad output tensor is null.");
  const size_t output_rank = output->attr.axis.size();
  int64_t axis = 0L;
  GE_ASSERT_SUCCESS(GetIndirectLoadAxis(indirect_load, axis));
  const int64_t rank = static_cast<int64_t>(output_rank);
  GE_ASSERT_TRUE(axis >= -rank && axis < rank, "IndirectLoad axis %ld is out of range for output rank %zu.", axis,
                 output_rank);
  GE_ASSERT_SUCCESS(
      ValidateIndirectLoadInputRank(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex, output_rank));
  GE_ASSERT_SUCCESS(
      ValidateIndirectLoadInputRank(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex, output_rank));
  const auto inputs = indirect_load->inputs();
  const auto input = inputs[ascgen_utils::indirect_load::kInputTensorIndex];
  const auto index = inputs[ascgen_utils::indirect_load::kIndexTensorIndex];
  GE_ASSERT_TRUE(input->attr.repeats.size() == output_rank && index->attr.repeats.size() == output_rank &&
                     output->attr.repeats.size() == output_rank,
                 "IndirectLoad logical shape rank is invalid.");
  const size_t axis_index = static_cast<size_t>(axis < 0L ? axis + rank : axis);
  for (size_t i = 0UL; i < output_rank; ++i) {
    GE_ASSERT_TRUE(
        af::SymbolicUtils::StaticCheckEq(index->attr.repeats[i], output->attr.repeats[i]) == af::TriBool::kTrue,
        "IndirectLoad index and output logical shape must match.");
    if (i != axis_index) {
      GE_ASSERT_TRUE(
          af::SymbolicUtils::StaticCheckLt(input->attr.repeats[i], output->attr.repeats[i]) != af::TriBool::kTrue,
          "IndirectLoad input dimension %zu must not be smaller than index/output outside axis %zu.", i, axis_index);
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
  outer_id =
      graph.CreateAxis(axis->name + "T", ascir::Axis::Type::kAxisTypeTileOuter, axis->size, {axis_id}, af::kIdNone).id;
  inner_id =
      graph
          .CreateAxis(axis->name + "t", ascir::Axis::Type::kAxisTypeTileInner, af::sym::kSymbolOne, {axis_id}, outer_id)
          .id;
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
  for (size_t input_idx = 0UL; input_idx < 2UL; ++input_idx) {
    const auto input_anchor = indirect_load->GetInDataAnchor(input_idx);
    GE_ASSERT_NOTNULL(input_anchor);
    const auto peer_out_anchor = input_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out_anchor);
    auto producer = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
    GE_ASSERT_NOTNULL(producer);
    const std::string role = input_idx == 0UL ? "input" : "index";
    const bool align_path = input_idx == 0UL ? align_input_path : align_index_path;
    GE_ASSERT_SUCCESS(InsertWorkspaceBoundary(graph, producer, static_cast<size_t>(peer_out_anchor->GetIdx()),
                                              indirect_load, input_idx, indirect_load->GetName() + "_sk_" + role,
                                              align_path, input_idx == 1UL && align_path));
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

af::Status BuildSimdInputInnerAxis(af::AscGraph &graph, const af::AscNodePtr &input_producer, size_t axis_index,
                                   ascir::AxisId &input_inner_axis) {
  GE_ASSERT_TRUE(!input_producer->outputs().empty(), "IndirectLoad SIMD input tensor producer has no output.");
  const auto input_axes = input_producer->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(axis_index < input_axes.size(), "IndirectLoad SIMD input axis index is out of range.");
  std::vector<ascir::AxisId> input_inner_axes(input_axes.begin() + static_cast<int64_t>(axis_index), input_axes.end());
  GE_ASSERT_SUCCESS(MergeAxesForTemplate(graph, input_inner_axes, "indirect_load_input_inner", input_inner_axis));
  return af::SUCCESS;
}

af::Status BuildSkInputInnerAxis(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t axis_index,
                                 ascir::AxisId &input_inner_axis) {
  const auto input_boundary = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  GE_ASSERT_TRUE(input_boundary != nullptr && af::ops::IsOps<af::ascir_op::Load>(input_boundary),
                 "IndirectLoad SK input boundary must be a Load node, node[%s].", indirect_load->GetNamePtr());
  return BuildSimdInputInnerAxis(graph, input_boundary, axis_index, input_inner_axis);
}

af::Status BuildAxisViewByBoundary(af::AscGraph &graph, const std::vector<af::AxisId> &axes, size_t boundary,
                                   af::AxisId &outer_axis, af::AxisId &inner_axis) {
  GE_ASSERT_TRUE(!axes.empty(), "IndirectLoad output axis is empty.");
  const size_t split = std::min(boundary, axes.size());
  const std::vector<af::AxisId> outer_axes(axes.begin(), axes.begin() + static_cast<int64_t>(split));
  const std::vector<af::AxisId> inner_axes(axes.begin() + static_cast<int64_t>(split), axes.end());
  if (outer_axes.empty()) {
    outer_axis = graph.CreateAxis("indirect_load_single_outer", af::sym::kSymbolOne).id;
  } else {
    GE_ASSERT_SUCCESS(MergeAxesForTemplate(graph, outer_axes, "indirect_load_outer", outer_axis));
  }
  if (!inner_axes.empty()) {
    GE_ASSERT_SUCCESS(MergeAxesForTemplate(graph, inner_axes, "indirect_load_inner", inner_axis));
  }
  return af::SUCCESS;
}

af::Status NormalizeAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load, int64_t axis,
                                    int64_t rank, ascir::AxisId input_inner_axis) {
  const auto output_axes = indirect_load->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(!output_axes.empty(), "IndirectLoad output axis is empty.");
  const size_t boundary = static_cast<size_t>(axis < 0L ? axis + rank : axis);
  ascir::AxisId outer_axis = af::kIdNone;
  ascir::AxisId inner_axis = af::kIdNone;
  GE_ASSERT_SUCCESS(BuildAxisViewByBoundary(graph, output_axes, boundary, outer_axis, inner_axis));
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
    int64_t axis = 0L;
    GE_ASSERT_SUCCESS(GetIndirectLoadAxis(indirect_load, axis));
    const int64_t rank = static_cast<int64_t>(indirect_load->outputs()[0]->attr.axis.size());
    const size_t axis_index = static_cast<size_t>(axis < 0L ? axis + rank : axis);
    ascir::AxisId input_inner_axis = af::kIdNone;
    GE_ASSERT_SUCCESS(BuildSkInputInnerAxis(graph, indirect_load, axis_index, input_inner_axis));
    GE_ASSERT_SUCCESS(NormalizeAxesForTemplate(graph, indirect_load, axis, rank, input_inner_axis));
  }
  return af::SUCCESS;
}

af::Status ReplaceAxisPrefix(std::vector<af::AxisId> &target_axes, const std::vector<af::AxisId> &output_axes,
                             size_t axis_index) {
  GE_ASSERT_TRUE(target_axes.size() >= axis_index, "IndirectLoad SIMD input region has invalid axis rank.");
  std::copy_n(output_axes.begin(), axis_index, target_axes.begin());
  return af::SUCCESS;
}

af::Status ReplaceInputRegionAxisPrefix(const NodePath &input_region, const std::vector<af::AxisId> &output_axes,
                                        size_t axis_index) {
  for (const auto &node : input_region) {
    if (!ascgen_utils::indirect_load::ShouldApplyInputInnerVectorization(node)) {
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
                                        const af::AscNodePtr &input_root, const NodePath &input_region) {
  const auto output_axes = indirect_load->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(!output_axes.empty(), "IndirectLoad SIMD output axis is empty.");
  int64_t axis = 0L;
  GE_ASSERT_SUCCESS(GetIndirectLoadAxis(indirect_load, axis));
  const int64_t rank = static_cast<int64_t>(output_axes.size());
  const size_t axis_index = static_cast<size_t>(axis < 0L ? axis + rank : axis);
  ascir::AxisId input_inner_axis = af::kIdNone;
  GE_ASSERT_SUCCESS(BuildSimdInputInnerAxis(graph, input_root, axis_index, input_inner_axis));
  GE_ASSERT_SUCCESS(ReplaceInputRegionAxisPrefix(input_region, output_axes, axis_index));
  return NormalizeAxesForTemplate(graph, indirect_load, axis, rank, input_inner_axis);
}

af::Status NormalizeSimtAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t boundary) {
  const auto output_axes = indirect_load->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(!output_axes.empty(), "IndirectLoad SIMT output axis is empty.");
  const int64_t rank = static_cast<int64_t>(output_axes.size());
  GE_ASSERT_TRUE(boundary <= output_axes.size(), "IndirectLoad SIMT boundary is out of range.");
  return NormalizeAxesForTemplate(graph, indirect_load, static_cast<int64_t>(boundary), rank, af::kIdNone);
}

bool CanEmitSimtScalar(const af::AscNodePtr &node) {
  const auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
  const auto *v2_impl = impl == nullptr ? nullptr : dynamic_cast<af::ascir::AscIrCodegenV2 *>(impl.get());
  return v2_impl != nullptr && v2_impl->IsSimtScalarSupported(*node);
}

af::AscNodePtr FindInputLoad(const af::AscNodePtr &indirect_load, size_t input_index) {
  for (af::AscNodePtr current = ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_index);
       current != nullptr;
       current = current->inputs.Size() == 1UL ? ascgen_utils::indirect_load::GetInputProducer(current, 0UL)
                                               : nullptr) {
    if (af::ops::IsOps<af::ascir_op::Load>(current)) {
      return current;
    }
  }
  return nullptr;
}

af::AscNodePtr FindOutputStore(const af::AscNodePtr &indirect_load) {
  for (af::AscNodePtr current = ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load); current != nullptr;
       current = ascgen_utils::indirect_load::GetOnlyOutputConsumer(current)) {
    if (af::ops::IsOps<af::ascir_op::Store>(current)) {
      return current;
    }
  }
  return nullptr;
}

af::AscNodePtr FindPostReduce(const af::AscNodePtr &indirect_load) {
  af::AscNodePtr post_reduce;
  for (af::AscNodePtr current = ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load); current != nullptr;
       current = ascgen_utils::indirect_load::GetOnlyOutputConsumer(current)) {
    if (ScheduleUtils::IsReduce(current)) {
      post_reduce = current;
    }
  }
  return post_reduce;
}

void CollectRewrittenBoundaries(const af::AscNodePtr &indirect_load, RewrittenGraphAnalysis &analysis) {
  analysis.input_root =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
  analysis.input_load = FindInputLoad(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
  analysis.index_load = FindInputLoad(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
  analysis.output_store = FindOutputStore(indirect_load);
  analysis.post_reduce = FindPostReduce(indirect_load);
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

af::Status CompleteInputDataTensorAttrs(const RewrittenGraphAnalysis &analysis) {
  for (const af::AscNodePtr &load : {analysis.input_load, analysis.index_load}) {
    if (load == nullptr) {
      continue;
    }
    const af::AscNodePtr data = ascgen_utils::indirect_load::GetInputProducer(load, 0UL);
    GE_ASSERT_NOTNULL(data, "IndirectLoad input Load[%s] has no producer.", load->GetNamePtr());
    GE_ASSERT_TRUE(af::ops::IsOps<af::ascir_op::Data>(data), "IndirectLoad input Load[%s] producer is not Data.",
                   load->GetNamePtr());
    GE_ASSERT_TRUE(!load->outputs().empty() && !data->outputs().empty(),
                   "IndirectLoad input Load[%s] or Data[%s] has no output.", load->GetNamePtr(), data->GetNamePtr());
    const auto load_out = load->outputs().front();
    const auto data_out = data->outputs().front();
    if (data_out->attr.axis.empty() && !load_out->attr.axis.empty()) {
      GELOGD("[IndirectLoad] Complete tensor attrs from Load[%s] to Data[%s].", load->GetNamePtr(), data->GetNamePtr());
      data_out->attr.axis = load_out->attr.axis;
      data_out->attr.repeats = load_out->attr.repeats;
      data_out->attr.strides = load_out->attr.strides;
      data_out->attr.dtype = load_out->attr.dtype;
    }
  }
  return af::SUCCESS;
}

af::Status RecordTemplateLogicalView(const af::AscNodePtr &indirect_load, bool &is_candidate_legal) {
  GE_ASSERT_TRUE(indirect_load->inputs.Size() == 2UL && indirect_load->outputs().size() == 1UL,
                 "IndirectLoad expects 2 inputs and 1 output.");
  const auto inputs = indirect_load->inputs();
  ascgen_utils::indirect_load::TemplateLogicalView view;
  const ascgen_utils::indirect_load::LogicalTensorView input_view{
      inputs[ascgen_utils::indirect_load::kInputTensorIndex]->attr.axis,
      inputs[ascgen_utils::indirect_load::kInputTensorIndex]->attr.repeats,
      inputs[ascgen_utils::indirect_load::kInputTensorIndex]->attr.strides};
  const ascgen_utils::indirect_load::LogicalTensorView index_view{
      inputs[ascgen_utils::indirect_load::kIndexTensorIndex]->attr.axis,
      inputs[ascgen_utils::indirect_load::kIndexTensorIndex]->attr.repeats,
      inputs[ascgen_utils::indirect_load::kIndexTensorIndex]->attr.strides};
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(input_view, view.input));
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(index_view, view.index));
  view.output.axis_ids = indirect_load->outputs().front()->attr.axis;
  view.output.sizes = indirect_load->outputs().front()->attr.repeats;
  view.output.strides = indirect_load->outputs().front()->attr.strides;
  ascgen_utils::indirect_load::IndirectLoadTensorLayout output_layout;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(view.output, output_layout));
  if (view.input.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported ||
      view.index.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported ||
      output_layout.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense) {
    GELOGW("[IndirectLoad] Skip candidate because input/index layout is unsupported or output is not dense, node[%s].",
           indirect_load->GetNamePtr());
    is_candidate_legal = false;
    return af::SUCCESS;
  }
  return ascgen_utils::indirect_load::SetTemplateLogicalView(indirect_load, view);
}

void CollectIndirectLoadPaths(const af::AscNodePtr &indirect_load, IndirectLoadGraphPaths &paths) {
  for (size_t input_index :
       {ascgen_utils::indirect_load::kInputTensorIndex, ascgen_utils::indirect_load::kIndexTensorIndex}) {
    NodePath &path = input_index == ascgen_utils::indirect_load::kInputTensorIndex ? paths.input : paths.index;
    for (af::AscNodePtr current = ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_index);
         current != nullptr;
         current = current->inputs.Size() == 1UL ? ascgen_utils::indirect_load::GetInputProducer(current, 0UL)
                                                 : nullptr) {
      path.emplace_back(current);
    }
  }
}

af::Status PreparePhysicalViews(const af::AscNodePtr &indirect_load, const char *template_name,
                                bool &is_candidate_legal, IndirectLoadGraphPaths &paths,
                                ascgen_utils::indirect_load::TemplateLogicalView &logical_view) {
  GE_ASSERT_SUCCESS(ValidateBroadcastPrePaths(paths, template_name, is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(RecordTemplateLogicalView(indirect_load, is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(InlineBroadcastPrePaths(paths, template_name));
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view));
  return ApplyPhysicalExecutionViews(paths, logical_view);
}

af::Status CollectRewrittenRegion(const af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                  ascir::TemplateId template_id, RewrittenGraphAnalysis &analysis) {
  NodeSet region_set;
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    CollectInputRegionMembers(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex, region_set);
  } else if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    CollectSimtFusedRegionMembers(indirect_load, analysis, region_set);
  } else {
    GE_ASSERT_TRUE(false, "IndirectLoad template id %d is invalid.", static_cast<int32_t>(template_id));
  }
  for (const af::AscNodePtr &node : graph.GetAllNodes()) {
    if (region_set.count(node.get()) != 0UL) {
      analysis.region.emplace_back(node);
    }
  }
  return af::SUCCESS;
}

af::Status AnalyzeRewrittenGraph(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                 ascir::TemplateId template_id, bool &is_candidate_legal,
                                 RewrittenGraphAnalysis &analysis) {
  const char *template_name = template_id == ascir::TemplateId::kIndirectLoadSimd
                                  ? "SIMD"
                                  : (template_id == ascir::TemplateId::kIndirectLoadSimt ? "SIMT" : "SK");
  IndirectLoadGraphPaths paths;
  CollectIndirectLoadPaths(indirect_load, paths);
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  GE_ASSERT_SUCCESS(PreparePhysicalViews(indirect_load, template_name, is_candidate_legal, paths, logical_view));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  analysis.input_path = paths.input;
  analysis.index_path = paths.index;
  int64_t axis = 0L;
  GE_ASSERT_SUCCESS(GetIndirectLoadAxis(indirect_load, axis));
  const int64_t rank = static_cast<int64_t>(logical_view.output.sizes.size());
  const size_t axis_index = static_cast<size_t>(axis < 0L ? axis + rank : axis);
  analysis.align_index_path =
      template_id != ascir::TemplateId::kIndirectLoadSimt && NeedsAlignedUbWindow(logical_view.index, axis_index);
  analysis.align_input_path =
      template_id != ascir::TemplateId::kIndirectLoadSimt && NeedsAlignedUbWindow(logical_view.input, axis_index);
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(RewriteInputPreNodes(graph, indirect_load, template_id));

  CollectRewrittenBoundaries(indirect_load, analysis);
  return CollectRewrittenRegion(graph, indirect_load, template_id, analysis);
}

af::Status AnnotateSimdTemplateRoles(const RewrittenGraphAnalysis &analysis) {
  for (const af::AscNodePtr &node : analysis.region) {
    if (IsInputRegionBoundary(node)) {
      continue;
    }
    const auto role = ascgen_utils::indirect_load::GetTemplateRole(node);
    const auto simd_role = role == ascgen_utils::indirect_load::TemplateRole::kStridedUbPath
                               ? ascgen_utils::indirect_load::TemplateRole::kSimdInputPreStridedUbPath
                               : ascgen_utils::indirect_load::TemplateRole::kSimdInputPre;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(node, simd_role));
  }
  return af::SUCCESS;
}

af::Status ValidateSimtTemplateRegion(const RewrittenGraphAnalysis &analysis, bool &is_candidate_legal) {
  is_candidate_legal = false;
  if (!af::ops::IsOps<af::ascir_op::Load>(analysis.input_root) || analysis.region.empty()) {
    return af::SUCCESS;
  }
  for (const af::AscNodePtr &node : analysis.region) {
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
  for (const af::AscNodePtr &node : analysis.region) {
    const auto role = (af::ops::IsOps<af::ascir_op::Load>(node) || af::ops::IsOps<af::ascir_op::Store>(node))
                          ? ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary
                          : ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(node, role));
  }
  return af::SUCCESS;
}

af::Status ValidateSimdTemplate(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                bool &is_candidate_legal) {
  return ValidateSimdPostReduceLayout(indirect_load, analysis.post_reduce, is_candidate_legal);
}

af::Status ValidateSimtTemplate(const RewrittenGraphAnalysis &analysis, size_t &boundary, bool &is_candidate_legal) {
  GE_ASSERT_SUCCESS(ValidateSimtPostReduceLayout(analysis.post_reduce, boundary, is_candidate_legal));
  if (is_candidate_legal) {
    GE_ASSERT_SUCCESS(ValidateSimtTemplateRegion(analysis, is_candidate_legal));
  }
  return af::SUCCESS;
}

af::Status ValidateTemplate(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                            const RewrittenGraphAnalysis &analysis, size_t &boundary, bool &is_candidate_legal) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return ValidateSimdTemplate(indirect_load, analysis, is_candidate_legal);
  } else if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    return ValidateSimtTemplate(analysis, boundary, is_candidate_legal);
  } else {
    GE_ASSERT_TRUE(false, "IndirectLoad template id %d is invalid.", static_cast<int32_t>(template_id));
  }
  return af::FAILED;
}

af::Status AnnotateTemplate(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                            const RewrittenGraphAnalysis &analysis) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return AnnotateSimdTemplateRoles(analysis);
  } else if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    return AnnotateSimtTemplateRoles(indirect_load, analysis);
  } else {
    GE_ASSERT_TRUE(false, "IndirectLoad template id %d is invalid.", static_cast<int32_t>(template_id));
  }
  return af::FAILED;
}

af::Status NormalizeTemplateAxes(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                 ascir::TemplateId template_id, const RewrittenGraphAnalysis &analysis,
                                 size_t boundary) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return NormalizeSimdAxesForTemplate(graph, indirect_load, analysis.input_root, analysis.region);
  } else if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    return NormalizeSimtAxesForTemplate(graph, indirect_load, boundary);
  } else {
    GE_ASSERT_TRUE(false, "IndirectLoad template id %d is invalid.", static_cast<int32_t>(template_id));
  }
  return af::FAILED;
}

af::Status FinalizeTemplate(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return ::ascir::SetTemplateId(indirect_load, template_id);
  } else if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    GE_ASSERT_SUCCESS(::ascir::SetDcacheSize(indirect_load, kIndirectLoadSimtDcacheSize));
    return ::ascir::SetTemplateId(indirect_load, template_id);
  } else {
    GE_ASSERT_TRUE(false, "IndirectLoad template id %d is invalid.", static_cast<int32_t>(template_id));
  }
  return af::FAILED;
}

af::Status ApplySkGraphPass(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                            const RewrittenGraphAnalysis &analysis, bool &is_candidate_legal) {
  is_candidate_legal = false;
  if (!IsSkTemplateCandidateLegal(indirect_load)) {
    return af::SUCCESS;
  }
  is_candidate_legal = true;
  GE_ASSERT_SUCCESS(::ascir::SetTemplateId(indirect_load, ascir::TemplateId::kIndirectLoadSK));
  GE_ASSERT_SUCCESS(PartitionSkGraph(graph, indirect_load, analysis.align_input_path, analysis.align_index_path));
  int64_t axis = 0L;
  GE_ASSERT_SUCCESS(GetIndirectLoadAxis(indirect_load, axis));
  const int64_t rank = static_cast<int64_t>(indirect_load->outputs()[0]->attr.axis.size());
  const size_t axis_index = static_cast<size_t>(axis < 0L ? axis + rank : axis);
  ascir::AxisId input_inner_axis = af::kIdNone;
  GE_ASSERT_SUCCESS(BuildSkInputInnerAxis(graph, indirect_load, axis_index, input_inner_axis));
  GE_ASSERT_SUCCESS(NormalizeAxesForTemplate(graph, indirect_load, axis, rank, input_inner_axis));
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
  if (analysis.align_input_path) {
    const auto &aligned_input_path =
        template_id == ascir::TemplateId::kIndirectLoadSimd ? analysis.region : analysis.input_path;
    GE_ASSERT_SUCCESS(AnnotateStridedUbPath(aligned_input_path));
  }
  if (analysis.align_index_path) {
    GE_ASSERT_SUCCESS(AnnotateStridedUbPath(analysis.index_path));
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    return ApplySkGraphPass(graph, indirect_load, analysis, is_candidate_legal);
  }
  GE_ASSERT_SUCCESS(CompleteInputDataTensorAttrs(analysis));

  size_t boundary = indirect_load->outputs()[0]->attr.axis.size();
  GE_ASSERT_SUCCESS(ValidateTemplate(indirect_load, template_id, analysis, boundary, is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(AnnotateTemplate(indirect_load, template_id, analysis));
  GE_ASSERT_SUCCESS(NormalizeTemplateAxes(graph, indirect_load, template_id, analysis, boundary));
  return FinalizeTemplate(indirect_load, template_id);
}

std::string GenerateScoreFunc(const TemplateCase &template_case) {
  (void)template_case;
  std::stringstream ss;
  ss << "int32_t CalcScore(AutofuseTilingData &tiling_data) {" << std::endl;
  ss << "  (void)tiling_data;" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
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
