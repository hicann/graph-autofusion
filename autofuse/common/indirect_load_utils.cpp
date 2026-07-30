/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indirect_load_utils.h"

#include <string>

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "schedule_result.h"

namespace ascgen_utils::indirect_load {
namespace {
constexpr char kTemplateOuterAxisAttr[] = "af.internal.indirect_load.outer_axis";
constexpr char kTemplateInnerAxisAttr[] = "af.internal.indirect_load.inner_axis";
constexpr char kTemplateInputInnerAxisAttr[] = "af.internal.indirect_load.input_inner_axis";
constexpr char kTemplateLogicalViewAttr[] = "af.internal.indirect_load.logical_view";

bool IsValidLogicalTensorView(const LogicalTensorView &view) {
  return !view.axis_ids.empty() && view.axis_ids.size() == view.strides.size();
}

TemplateRole GetAnnotatedTemplateRole(const af::AscNodePtr &node) {
  if (node == nullptr) {
    return TemplateRole::kNone;
  }
  return static_cast<TemplateRole>(::ascir::GetTemplateRoleOrDefault(*node, static_cast<int64_t>(TemplateRole::kNone)));
}

TemplateBehavior GetBehavior(TemplateRole role) {
  TemplateBehavior behavior;
  switch (role) {
    case TemplateRole::kSimdInputPre:
      behavior.excludes_tiling_group = true;
      behavior.preserves_vectorized_axis = true;
      break;
    case TemplateRole::kSimtInputBoundary:
      behavior.skips_main_schedule_tiling = true;
      behavior.skips_api_emit = true;
      behavior.uses_direct_gm_pipeline = true;
      behavior.preserves_vectorized_axis = true;
      break;
    case TemplateRole::kSimtDirectGmBoundary:
    case TemplateRole::kSimtInlineTransform:
      behavior.skips_api_emit = true;
      behavior.uses_direct_gm_pipeline = true;
      behavior.preserves_vectorized_axis = true;
      break;
    case TemplateRole::kSimtOp:
      behavior.uses_direct_gm_pipeline = true;
      behavior.skips_ub_lifecycle = true;
      behavior.preserves_vectorized_axis = true;
      break;
    case TemplateRole::kNone:
      break;
  }
  return behavior;
}
}  // namespace

TemplateRole GetTemplateRole(const af::AscNodePtr &node) {
  return GetAnnotatedTemplateRole(node);
}

TemplateBehavior GetTemplateBehavior(const af::AscNodePtr &node) {
  return GetBehavior(GetTemplateRole(node));
}

af::Status InheritTemplateRoleIfIL(af::AscGraph &graph, const std::string &vf_node_name, const af::AscNodePtr &src) {
  GE_ASSERT_NOTNULL(src);
  if (GetTemplateRole(src) == TemplateRole::kNone) {
    return af::SUCCESS;
  }
  auto vf_node = graph.FindNode(vf_node_name.c_str());
  GE_ASSERT_NOTNULL(vf_node, "IndirectLoad: cannot find new VectorFunc node %s.", vf_node_name.c_str());
  GE_ASSERT_SUCCESS(::ascir::SetTemplateRole(vf_node, ::ascir::GetTemplateRoleOrDefault(*src)));
  return af::SUCCESS;
}

af::Status SetTemplateRole(const af::AscNodePtr &node, TemplateRole role) {
  return ::ascir::SetTemplateRole(node, static_cast<int64_t>(role));
}

af::Status SetTemplateAxes(const af::AscNodePtr &node, const TemplateAxes &axes) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateOuterAxisAttr, static_cast<int64_t>(axes.outer_axis)),
                 "Set IndirectLoad outer axis failed, node = %s", node->GetNamePtr());
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateInnerAxisAttr, static_cast<int64_t>(axes.inner_axis)),
                 "Set IndirectLoad inner axis failed, node = %s", node->GetNamePtr());
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateInputInnerAxisAttr, static_cast<int64_t>(axes.input_inner_axis)),
                 "Set IndirectLoad input inner axis failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

af::Status GetTemplateAxes(const af::AscNodePtr &node, TemplateAxes &axes) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  axes.outer_axis = op_desc->TryGetExtAttr(kTemplateOuterAxisAttr, static_cast<int64_t>(af::kIdNone));
  axes.inner_axis = op_desc->TryGetExtAttr(kTemplateInnerAxisAttr, static_cast<int64_t>(af::kIdNone));
  axes.input_inner_axis = op_desc->TryGetExtAttr(kTemplateInputInnerAxisAttr, static_cast<int64_t>(af::kIdNone));
  GE_ASSERT_TRUE(axes.outer_axis != af::kIdNone, "IndirectLoad template axes are missing, node = %s",
                 node->GetNamePtr());
  return af::SUCCESS;
}

af::Status SetTemplateLogicalView(const af::AscNodePtr &node, const TemplateLogicalView &view) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(IsValidLogicalTensorView(view.data) && IsValidLogicalTensorView(view.index) &&
                     IsValidLogicalTensorView(view.output),
                 "IndirectLoad logical view is invalid, node = %s", node->GetNamePtr());
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateLogicalViewAttr, view), "Set IndirectLoad logical view failed, node = %s",
                 node->GetNamePtr());
  return af::SUCCESS;
}

af::Status GetTemplateLogicalView(const af::AscNodePtr &node, TemplateLogicalView &view) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  view = op_desc->TryGetExtAttr(kTemplateLogicalViewAttr, TemplateLogicalView{});
  GE_ASSERT_TRUE(IsValidLogicalTensorView(view.data) && IsValidLogicalTensorView(view.index) &&
                     IsValidLogicalTensorView(view.output),
                 "IndirectLoad logical view is missing or invalid, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

bool ShouldSkipMainScheduleTiling(const af::AscNodePtr &node) {
  return GetTemplateBehavior(node).skips_main_schedule_tiling;
}

bool ShouldPreserveVectorizedAxis(const af::AscNodePtr &node) {
  return GetTemplateBehavior(node).preserves_vectorized_axis;
}

bool ShouldApplyInputInnerVectorization(const af::AscNodePtr &node) {
  return GetTemplateRole(node) == TemplateRole::kSimdInputPre;
}

bool ShouldDisableRegularVectorFunc(const af::AscNodePtr &node) {
  return GetTemplateBehavior(node).uses_direct_gm_pipeline;
}

af::AscNodePtr GetInputProducer(const af::AscNodePtr &node, size_t input_index) {
  auto input_anchor = node == nullptr ? nullptr : node->GetInDataAnchor(input_index);
  if (input_anchor == nullptr || input_anchor->GetPeerOutAnchor() == nullptr) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<af::AscNode>(input_anchor->GetPeerOutAnchor()->GetOwnerNode());
}

af::AscNodePtr GetOnlyOutputConsumer(const af::AscNodePtr &node) {
  if (node == nullptr || node->GetOutDataNodesSize() != 1UL) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<af::AscNode>(*node->GetOutDataNodes().begin());
}

af::AscNodePtr FindIndirectLoadNode(const af::AscGraph &graph) {
  for (const af::AscNodePtr &node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::IndirectLoad>(node)) {
      return node;
    }
  }
  return nullptr;
}

af::Status ValidateSingleIndirectLoadNode(const af::AscGraph &graph, af::AscNodePtr &node) {
  node = nullptr;
  for (const af::AscNodePtr &candidate : graph.GetAllNodes()) {
    if (!af::ops::IsOps<af::ascir_op::IndirectLoad>(candidate)) {
      continue;
    }
    if (node != nullptr) {
      GELOGE(af::FAILED, "[IndirectLoad] Graph[%s] contains multiple IndirectLoad nodes, first[%s], next[%s].",
             graph.GetName().c_str(), node->GetNamePtr(), candidate->GetNamePtr());
    }
    GE_ASSERT_TRUE(node == nullptr, "Graph contains multiple IndirectLoad nodes, only one is supported.");
    node = candidate;
  }
  if (node != nullptr) {
    GELOGD("[IndirectLoad] Graph[%s] found IndirectLoad node[%s].", graph.GetName().c_str(), node->GetNamePtr());
  }
  return af::SUCCESS;
}

af::Status GetPrebuiltYTilingCase(const af::AscGraph &graph, bool &has_case, af::AxisId &tile_id,
                                  std::pair<af::AxisPtr, af::AxisPtr> &tiling) {
  has_case = false;
  tile_id = af::kIdNone;
  tiling = {nullptr, nullptr};

  const af::AscNodePtr indirect_load = FindIndirectLoadNode(graph);
  if (indirect_load == nullptr) {
    return af::SUCCESS;
  }

  TemplateAxes axes;
  GE_ASSERT_SUCCESS(GetTemplateAxes(indirect_load, axes), "[IndirectLoad] Failed to get template axes for node[%s].",
                    indirect_load->GetNamePtr());

  for (const auto &axis : graph.GetAllAxis()) {
    if (axis == nullptr || axis->type != af::Axis::Type::kAxisTypeTileOuter || axis->from.size() != 1UL ||
        axis->from[0] != axes.outer_axis) {
      continue;
    }
    GE_ASSERT_TRUE(
        axis->split_pair_other_id >= 0L && static_cast<size_t>(axis->split_pair_other_id) < graph.GetAllAxis().size(),
        "[IndirectLoad] Invalid split pair axis[%ld] for graph[%s].", axis->split_pair_other_id,
        graph.GetName().c_str());
    tiling.first = axis;
    tiling.second = graph.GetAllAxis()[static_cast<size_t>(axis->split_pair_other_id)];
    has_case = true;
    tile_id = axes.outer_axis;
    return af::SUCCESS;
  }
  GELOGE(af::FAILED, "[IndirectLoad] Template fixed split is missing for axis[%ld] in graph[%s].", axes.outer_axis,
         graph.GetName().c_str());
  return af::FAILED;
}

}  // namespace ascgen_utils::indirect_load
