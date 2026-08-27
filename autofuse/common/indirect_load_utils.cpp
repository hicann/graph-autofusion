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

#include <algorithm>
#include <string>
#include <unordered_set>

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "schedule_result.h"

namespace ascgen_utils::indirect_load {
namespace {
constexpr char kTemplateOuterAxisAttr[] = "af.internal.indirect_load.outer_axis";
constexpr char kTemplateInnerAxisAttr[] = "af.internal.indirect_load.inner_axis";
constexpr char kTemplateInputInnerAxisAttr[] = "af.internal.indirect_load.input_inner_axis";
constexpr char kTemplateIndexInnerAxisAttr[] = "af.internal.indirect_load.index_inner_axis";
constexpr char kTemplateTileOuterAxisAttr[] = "af.internal.indirect_load.tile_outer_axis";
constexpr char kTemplateTileInnerAxisAttr[] = "af.internal.indirect_load.tile_inner_axis";
constexpr char kTemplateVectorizedAxesAttr[] = "af.internal.indirect_load.vectorized_axes";
constexpr char kTemplateSyntheticOuterAttr[] = "af.internal.indirect_load.synthetic_outer";
constexpr char kImplementationAttr[] = "af.internal.indirect_load.implementation";
bool IsValidLogicalTensorView(const LogicalTensorView &view) {
  return !view.axis_ids.empty() && view.axis_ids.size() == view.sizes.size() &&
         view.axis_ids.size() == view.strides.size();
}

enum class TensorDimKind : int64_t { kIllegal, kZeroStride, kRegular };
TensorDimKind ClassifyTensorDim(const af::Expression &size, const af::Expression &stride) {
  if (af::SymbolicUtils::StaticCheckLe(size, af::sym::kSymbolZero) == af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckLt(stride, af::sym::kSymbolZero) == af::TriBool::kTrue) {
    return TensorDimKind::kIllegal;
  }
  if (af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolZero) == af::TriBool::kTrue) {
    return TensorDimKind::kZeroStride;
  }
  return TensorDimKind::kRegular;
}

bool TryClassifyDynamicShapeLayout(const LogicalTensorView &logical, IndirectLoadTensorLayout &layout) {
  const bool has_dynamic_shape = std::any_of(logical.sizes.begin(), logical.sizes.end(),
                                             [](const af::Expression &size) { return !size.IsConstExpr(); });
  if (!has_dynamic_shape) {
    return false;
  }
  // A dynamic outer dimension does not prevent proving a compact zero-stride
  // view when all non-broadcast dimensions are contiguous. Preserve this
  // producer-side layout so Broadcast and its source use the same tensor view.
  bool has_zero_stride = false;
  af::Expression physical_span = af::sym::kSymbolOne;
  for (size_t dim = 0UL; dim < logical.sizes.size(); ++dim) {
    const auto dim_kind = ClassifyTensorDim(logical.sizes[dim], logical.strides[dim]);
    if (dim_kind == TensorDimKind::kIllegal) {
      return false;
    }
    if (dim_kind == TensorDimKind::kZeroStride) {
      layout.physical_repeats[dim] = af::sym::kSymbolOne;
      has_zero_stride = true;
    }
  }
  if (has_zero_stride) {
    physical_span = af::sym::kSymbolOne;
    for (size_t index = logical.sizes.size(); index > 0UL; --index) {
      const size_t dim = index - 1UL;
      if (ClassifyTensorDim(logical.sizes[dim], logical.strides[dim]) == TensorDimKind::kZeroStride) {
        continue;
      }
      if (af::SymbolicUtils::StaticCheckEq(logical.strides[dim], physical_span) != af::TriBool::kTrue) {
        layout.kind = IndirectLoadLayoutKind::kStrided;
        return true;
      }
      physical_span = physical_span + (logical.sizes[dim] - af::sym::kSymbolOne) * logical.strides[dim];
    }
    layout.kind = IndirectLoadLayoutKind::kZeroStrideCompact;
    return true;
  }
  layout.kind = IndirectLoadLayoutKind::kStrided;
  return true;
}

bool IsValidTensorLayout(const IndirectLoadTensorLayout &layout) {
  return IsValidLogicalTensorView(layout) && layout.kind != IndirectLoadLayoutKind::kUnsupported &&
         layout.physical_repeats.size() == layout.sizes.size();
}

TemplateAxes ReadTemplateAxes(const af::AscNodePtr &node) {
  TemplateAxes axes;
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return axes;
  }
  const auto op_desc = node->GetOpDesc();
  axes.outer_axis = op_desc->TryGetExtAttr(kTemplateOuterAxisAttr, static_cast<int64_t>(af::kIdNone));
  axes.inner_axis = op_desc->TryGetExtAttr(kTemplateInnerAxisAttr, static_cast<int64_t>(af::kIdNone));
  axes.input_inner_axis = op_desc->TryGetExtAttr(kTemplateInputInnerAxisAttr, static_cast<int64_t>(af::kIdNone));
  axes.index_inner_axis = op_desc->TryGetExtAttr(kTemplateIndexInnerAxisAttr, static_cast<int64_t>(af::kIdNone));
  axes.tile_outer_axis = op_desc->TryGetExtAttr(kTemplateTileOuterAxisAttr, static_cast<int64_t>(af::kIdNone));
  axes.tile_inner_axis = op_desc->TryGetExtAttr(kTemplateTileInnerAxisAttr, static_cast<int64_t>(af::kIdNone));
  axes.vectorized_axes = op_desc->TryGetExtAttr(kTemplateVectorizedAxesAttr, std::vector<af::AxisId>{});
  axes.synthetic_outer = op_desc->TryGetExtAttr(kTemplateSyntheticOuterAttr, false);
  return axes;
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
    case TemplateRole::kSimdInputPreStridedUbPath:
    case TemplateRole::kSimdIndexPre:
      behavior.excludes_tiling_group = true;
      behavior.preserves_vectorized_axis = true;
      break;
    case TemplateRole::kSimtInputBoundary:
    case TemplateRole::kSimtDirectGmBoundary:
    case TemplateRole::kSimtInlineTransform:
      behavior.skips_main_schedule_tiling = true;
      behavior.skips_api_emit = true;
      behavior.uses_direct_gm_pipeline = true;
      behavior.preserves_vectorized_axis = true;
      break;
    case TemplateRole::kSimtFanoutBranch:
      behavior.excludes_tiling_group = true;
      behavior.skips_main_schedule_tiling = true;
      behavior.uses_direct_gm_pipeline = true;
      behavior.preserves_vectorized_axis = true;
      break;
    case TemplateRole::kSimtOp:
      behavior.uses_direct_gm_pipeline = true;
      behavior.skips_ub_lifecycle = true;
      behavior.preserves_vectorized_axis = true;
      break;
    case TemplateRole::kSkInputBoundary:
      behavior.skips_main_schedule_tiling = true;
      behavior.skips_api_emit = true;
      behavior.preserves_vectorized_axis = true;
      break;
    default:
      break;
  }
  return behavior;
}
}  // namespace

TemplateRole GetTemplateRole(const af::AscNodePtr &node) {
  return GetAnnotatedTemplateRole(node);
}

TemplateBehavior GetTemplateBehavior(const af::AscNodePtr &node) {
  const TemplateRole role = GetTemplateRole(node);
  TemplateBehavior behavior = GetBehavior(role);
  if (role == TemplateRole::kSimtOp && GetPostReduceConsumer(node) != nullptr) {
    behavior = {};
    behavior.excludes_tiling_group = true;
  }
  behavior.skips_input_lifecycle = node != nullptr && af::ops::IsOps<af::ascir_op::IndirectLoad>(node) &&
                                   ::ascir::GetTemplateIdOrDefault(*node) == ::ascir::TemplateId::kIndirectLoadSimt;
  return behavior;
}

af::AscNodePtr GetOnlyOutputConsumer(const af::AscNodePtr &node) {
  if (node == nullptr || node->GetOutDataNodesSize() != 1UL) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<af::AscNode>(*node->GetOutDataNodes().begin());
}

namespace {
struct PostReduceChain {
  af::AscNodePtr input_producer;
  af::AscNodePtr reduce;
};

PostReduceChain FindPostReduceChain(const af::AscNodePtr &node) {
  if (node == nullptr) {
    return {};
  }
  std::vector<af::AscNodePtr> pending;
  std::unordered_set<af::AscNode *> visited;
  for (const auto &out_node : node->GetOutDataNodes()) {
    const auto out_asc_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
    if (out_asc_node != nullptr && visited.emplace(out_asc_node.get()).second) {
      pending.emplace_back(out_asc_node);
    }
  }

  af::AscNodePtr reduce;
  for (size_t index = 0UL; index < pending.size(); ++index) {
    const auto &current = pending[index];
    if (current->attr.api.compute_type == af::ComputeType::kComputeReduce) {
      if (reduce != nullptr && reduce != current) {
        return {};
      }
      reduce = current;
      continue;
    }
    for (const auto &out_node : current->GetOutDataNodes()) {
      const auto out_asc_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
      if (out_asc_node != nullptr && visited.emplace(out_asc_node.get()).second) {
        pending.emplace_back(out_asc_node);
      }
    }
  }
  if (reduce == nullptr) {
    return {};
  }
  return {GetInputProducer(reduce, 0UL), reduce};
}
}  // namespace

af::AscNodePtr GetPostReduceConsumer(const af::AscNodePtr &node) {
  return FindPostReduceChain(node).reduce;
}

af::AscNodePtr GetPostReduceInputProducer(const af::AscNodePtr &node) {
  return FindPostReduceChain(node).input_producer;
}

bool ShouldSkipTpipeTensorCollection(const af::AscNodePtr &node) {
  // A SIMT IndirectLoad normally bypasses the TPipe because its direct-GM
  // chain consumes the value in registers.  With a user fan-out, however,
  // an ordinary scheduled branch may consume the IndirectLoad result as a
  // UB tensor (for example through a VectorFunc), so keep that output in the
  // tensor table for the branch while retaining the direct-GM path.
  if (node != nullptr && af::ops::IsOps<af::ascir_op::IndirectLoad>(node)) {
    std::vector<af::AscNodePtr> pending;
    std::unordered_set<const af::AscNode *> visited;
    for (const auto &out : node->GetOutDataNodes()) {
      const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out);
      if (consumer != nullptr && visited.emplace(consumer.get()).second) {
        pending.emplace_back(consumer);
      }
    }
    size_t store_count = 0UL;
    for (size_t i = 0UL; i < pending.size(); ++i) {
      const auto &current = pending[i];
      if (af::ops::IsOps<af::ascir_op::Store>(current)) {
        ++store_count;
        if (store_count > 1UL) {
          return false;
        }
        continue;
      }
      for (const auto &out : current->GetOutDataNodes()) {
        const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out);
        if (consumer != nullptr && visited.emplace(consumer.get()).second) {
          pending.emplace_back(consumer);
        }
      }
    }
  }
  const TemplateBehavior behavior = GetTemplateBehavior(node);
  const af::AscNodePtr consumer = GetOnlyOutputConsumer(node);
  return (behavior.skips_api_emit || behavior.skips_ub_lifecycle) &&
         !(GetTemplateRole(node) == TemplateRole::kSimtInlineTransform && consumer != nullptr &&
           consumer->attr.api.compute_type == af::ComputeType::kComputeReduce);
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
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateIndexInnerAxisAttr, static_cast<int64_t>(axes.index_inner_axis)),
                 "Set IndirectLoad index inner axis failed, node = %s", node->GetNamePtr());
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateTileOuterAxisAttr, static_cast<int64_t>(axes.tile_outer_axis)),
                 "Set IndirectLoad tile outer axis failed, node = %s", node->GetNamePtr());
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateTileInnerAxisAttr, static_cast<int64_t>(axes.tile_inner_axis)),
                 "Set IndirectLoad tile inner axis failed, node = %s", node->GetNamePtr());
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateVectorizedAxesAttr, axes.vectorized_axes),
                 "Set IndirectLoad vectorized axes failed, node = %s", node->GetNamePtr());
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateSyntheticOuterAttr, axes.synthetic_outer),
                 "Set IndirectLoad synthetic outer failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

af::Status GetTemplateAxes(const af::AscNodePtr &node, TemplateAxes &axes) {
  GE_ASSERT_NOTNULL(node);
  axes = ReadTemplateAxes(node);
  GE_ASSERT_TRUE(axes.outer_axis != af::kIdNone, "IndirectLoad template axes are missing, node = %s",
                 node->GetNamePtr());
  return af::SUCCESS;
}

af::Status SetTemplateLogicalView(const af::AscNodePtr &node, const TemplateLogicalView &view) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(
      IsValidTensorLayout(view.input) && IsValidTensorLayout(view.index) && IsValidLogicalTensorView(view.output),
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
  GE_ASSERT_TRUE(
      IsValidTensorLayout(view.input) && IsValidTensorLayout(view.index) && IsValidLogicalTensorView(view.output),
      "IndirectLoad logical view is missing or invalid, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

af::Status ClassifyIndirectLoadLayout(const LogicalTensorView &logical, IndirectLoadTensorLayout &layout,
                                      bool allow_non_overlapping_zero_stride) {
  GE_ASSERT_TRUE(IsValidLogicalTensorView(logical), "IndirectLoad input layout rank is invalid.");
  static_cast<LogicalTensorView &>(layout) = logical;
  layout.kind = IndirectLoadLayoutKind::kUnsupported;
  layout.physical_repeats = logical.sizes;

  if (TryClassifyDynamicShapeLayout(logical, layout)) {
    return af::SUCCESS;
  }
  af::Expression physical_span = af::sym::kSymbolOne;
  bool has_zero_stride = false;
  bool has_physical_gap = false;
  for (size_t index = logical.sizes.size(); index > 0UL; --index) {
    const size_t dim = index - 1UL;
    const auto dim_kind = ClassifyTensorDim(logical.sizes[dim], logical.strides[dim]);
    if (dim_kind == TensorDimKind::kIllegal) {
      return af::SUCCESS;
    }
    if (dim_kind == TensorDimKind::kZeroStride) {
      layout.physical_repeats[dim] = af::sym::kSymbolOne;
      has_zero_stride = true;
      continue;
    }
    if (af::SymbolicUtils::StaticCheckLt(logical.strides[dim], physical_span) != af::TriBool::kFalse) {
      return af::SUCCESS;
    }
    has_physical_gap =
        has_physical_gap || af::SymbolicUtils::StaticCheckEq(logical.strides[dim], physical_span) != af::TriBool::kTrue;
    physical_span = physical_span + (logical.sizes[dim] - af::sym::kSymbolOne) * logical.strides[dim];
  }
  if (has_zero_stride && has_physical_gap) {
    if (allow_non_overlapping_zero_stride) {
      layout.kind = IndirectLoadLayoutKind::kStrided;
      layout.physical_repeats = logical.sizes;
    }
    return af::SUCCESS;
  }
  layout.kind = has_zero_stride
                    ? IndirectLoadLayoutKind::kZeroStrideCompact
                    : (has_physical_gap ? IndirectLoadLayoutKind::kStrided : IndirectLoadLayoutKind::kDense);
  return af::SUCCESS;
}

af::Status ValidateIndirectLoadOutputLayout(const LogicalTensorView &output) {
  GE_ASSERT_TRUE(IsValidLogicalTensorView(output), "IndirectLoad output layout rank is invalid.");
  af::Expression expected_stride = af::sym::kSymbolOne;
  for (size_t index = output.sizes.size(); index > 0UL; --index) {
    const size_t dim = index - 1UL;
    if (af::SymbolicUtils::StaticCheckEq(output.sizes[dim], af::sym::kSymbolOne) == af::TriBool::kTrue) {
      continue;
    }
    GE_ASSERT_TRUE(af::SymbolicUtils::StaticCheckEq(output.strides[dim], expected_stride) == af::TriBool::kTrue,
                   "IndirectLoad output must use a dense contiguous layout.");
    expected_stride = af::sym::Mul(expected_stride, output.sizes[dim]);
  }
  return af::SUCCESS;
}

af::Status SetImplementation(const af::AscNodePtr &node, Implementation implementation) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kImplementationAttr, static_cast<int64_t>(implementation)),
                 "Set IndirectLoad implementation failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

af::Status GetImplementation(const af::AscNodePtr &node, Implementation &implementation) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  implementation = static_cast<Implementation>(op_desc->TryGetExtAttr(kImplementationAttr, -1L));
  GE_ASSERT_TRUE(implementation == Implementation::kDefault || implementation == Implementation::kGatherApi,
                 "IndirectLoad implementation is missing or invalid, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

bool ShouldApplyInputInnerVectorization(const af::AscNodePtr &node) {
  const TemplateRole role = GetTemplateRole(node);
  return role == TemplateRole::kSimdInputPre || role == TemplateRole::kSimdInputPreStridedUbPath ||
         role == TemplateRole::kSkInputBoundary;
}

af::AscNodePtr GetInputProducer(const af::AscNodePtr &node, size_t input_index) {
  auto input_anchor = node == nullptr ? nullptr : node->GetInDataAnchor(input_index);
  if (input_anchor == nullptr || input_anchor->GetPeerOutAnchor() == nullptr) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<af::AscNode>(input_anchor->GetPeerOutAnchor()->GetOwnerNode());
}

namespace {
std::vector<af::AscNodePtr> CollectIndirectLoadNodes(const af::AscGraph &graph) {
  std::vector<af::AscNodePtr> nodes;
  for (const af::AscNodePtr &node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::IndirectLoad>(node)) {
      nodes.push_back(node);
    }
  }
  return nodes;
}
}  // namespace

af::AscNodePtr FindIndirectLoadNode(const af::AscGraph &graph) {
  const auto nodes = CollectIndirectLoadNodes(graph);
  return nodes.empty() ? nullptr : nodes.front();
}

af::Status ValidateSingleIndirectLoadNode(const af::AscGraph &graph, af::AscNodePtr &node) {
  node = nullptr;
  const auto nodes = CollectIndirectLoadNodes(graph);
  GE_ASSERT_TRUE(nodes.size() <= 1UL,
                 "[IndirectLoad] Graph[%s] contains multiple IndirectLoad nodes, first[%s], next[%s].",
                 graph.GetName().c_str(), nodes.empty() ? "<null>" : nodes[0]->GetNamePtr(),
                 nodes.size() < 2UL ? "<null>" : nodes[1]->GetNamePtr());
  node = nodes.empty() ? nullptr : nodes.front();
  if (node != nullptr) {
    GELOGD("[IndirectLoad] Graph[%s] found IndirectLoad node[%s].", graph.GetName().c_str(), node->GetNamePtr());
  }
  return af::SUCCESS;
}

bool GetPrebuiltYTilingCase(const af::AscGraph &graph, af::AxisId &tile_id,
                            std::pair<af::AxisPtr, af::AxisPtr> &tiling) {
  tile_id = af::kIdNone;
  tiling = {nullptr, nullptr};

  const af::AscNodePtr indirect_load = FindIndirectLoadNode(graph);
  if (indirect_load == nullptr) {
    return false;
  }

  const TemplateAxes axes = ReadTemplateAxes(indirect_load);
  tile_id = axes.outer_axis;
  const auto all_axes = graph.GetAllAxis();
  if (axes.tile_outer_axis >= 0L && static_cast<size_t>(axes.tile_outer_axis) < all_axes.size()) {
    tiling.first = all_axes[static_cast<size_t>(axes.tile_outer_axis)];
  }
  if (axes.tile_inner_axis >= 0L && static_cast<size_t>(axes.tile_inner_axis) < all_axes.size()) {
    tiling.second = all_axes[static_cast<size_t>(axes.tile_inner_axis)];
  }
  return true;
}

}  // namespace ascgen_utils::indirect_load
