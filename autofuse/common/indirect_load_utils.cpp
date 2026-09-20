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
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "schedule_result.h"
#include "utils/extern_math_util.h"

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
      behavior.skips_main_schedule_tiling = true;
      behavior.skips_api_emit = true;
      behavior.skips_ub_lifecycle = true;
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

ascir::TensorId FindSkippedChainResultTensor(const af::AscNodePtr &root) {
  if (root == nullptr || root->outputs().empty()) {
    return af::kIdNone;
  }
  std::vector<af::AscNodePtr> pending{root};
  std::unordered_set<af::AscNode *> visited;
  ascir::TensorId result = root->outputs()[0]->attr.mem.tensor_id;
  for (size_t i = 0UL; i < pending.size(); ++i) {
    const auto &node = pending[i];
    if (node == nullptr || !visited.emplace(node.get()).second || node->outputs().empty()) {
      continue;
    }
    result = node->outputs()[0]->attr.mem.tensor_id;
    bool has_skipped_consumer = false;
    for (const auto &out_node : node->GetOutDataNodes()) {
      const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
      if (consumer == nullptr) {
        continue;
      }
      const auto behavior = GetTemplateBehavior(consumer);
      if (behavior.skips_api_emit) {
        has_skipped_consumer = true;
        pending.emplace_back(consumer);
      }
    }
    if (!has_skipped_consumer) {
      return result;
    }
  }
  return result;
}

af::AscNodePtr GetPostReduceConsumer(const af::AscNodePtr &node) {
  return FindPostReduceChain(node).reduce;
}

af::AscNodePtr GetPostReduceInputProducer(const af::AscNodePtr &node) {
  return FindPostReduceChain(node).input_producer;
}

bool ShouldSkipTpipeTensorCollection(const af::AscNodePtr &node) {
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

af::Status SetIndirectLoadAccessInfo(const af::AscNodePtr &node, const IndirectLoadAccessInfo &info) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kAccessInfoAttr, info), "Set IndirectLoad access info failed, node = %s",
                 node->GetNamePtr());
  return af::SUCCESS;
}

af::Status GetIndirectLoadAccessInfo(const af::AscNodePtr &node, IndirectLoadAccessInfo &info) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  info = op_desc->TryGetExtAttr(kAccessInfoAttr, IndirectLoadAccessInfo{});
  return af::SUCCESS;
}

af::Status SetLoweringMetadata(const af::AscNodePtr &node, const IndirectLoadLoweringMetadata &metadata) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(metadata.version == 1U && metadata.axis >= 0L && metadata.outer_axis != af::kIdNone,
                 "IndirectLoad lowering metadata is invalid, node = %s", node->GetNamePtr());
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kLoweringMetadataAttr, metadata),
                 "Set IndirectLoad lowering metadata failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

af::Status GetLoweringMetadata(const af::AscNodePtr &node, IndirectLoadLoweringMetadata &metadata) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  metadata = op_desc->TryGetExtAttr(kLoweringMetadataAttr, IndirectLoadLoweringMetadata{});
  GE_ASSERT_TRUE(metadata.version == 1U && metadata.axis >= 0L && metadata.outer_axis != af::kIdNone,
                 "IndirectLoad lowering metadata is missing or invalid, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

bool HasLoweringMetadata(const af::AscNodePtr &node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return false;
  }
  const auto metadata = node->GetOpDesc()->TryGetExtAttr(kLoweringMetadataAttr, IndirectLoadLoweringMetadata{});
  return metadata.version == 1U && metadata.axis >= 0L && metadata.outer_axis != af::kIdNone;
}

namespace {
af::Expression ProductFrom(const LogicalTensorView &view, size_t begin) {
  af::Expression product = af::sym::kSymbolOne;
  for (size_t index = begin; index < view.sizes.size(); ++index) {
    product = product * view.sizes[index];
  }
  return product;
}

// Verifies that the suffix after axis_index is contiguous.  When skip_zero_stride is set,
// zero-stride (broadcast) dimensions are ignored by the stride chain.  has_payload reports
// whether any suffix dimension carries data; axis_stride receives the dense stride of the
// axis dimension on success.
bool IsContiguousSuffixImpl(const LogicalTensorView &view, size_t axis_index, bool skip_zero_stride, bool &has_payload,
                            af::Expression &axis_stride) {
  has_payload = false;
  if (axis_index >= view.sizes.size()) {
    return false;
  }
  af::Expression expected_stride = af::sym::kSymbolOne;
  for (size_t index = view.sizes.size(); index > axis_index + 1UL; --index) {
    const size_t dim = index - 1UL;
    if (skip_zero_stride &&
        af::SymbolicUtils::StaticCheckEq(view.strides[dim], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    if (af::SymbolicUtils::StaticCheckEq(view.strides[dim], expected_stride) != af::TriBool::kTrue) {
      return false;
    }
    has_payload = true;
    expected_stride = expected_stride + (view.sizes[dim] - af::sym::kSymbolOne) * view.strides[dim];
  }
  axis_stride = expected_stride;
  return true;
}

bool IsContiguousSuffix(const LogicalTensorView &view, size_t axis_index) {
  bool has_payload = false;
  af::Expression axis_stride;
  return IsContiguousSuffixImpl(view, axis_index, false, has_payload, axis_stride);
}

bool IsContiguousPayloadSuffix(const LogicalTensorView &view, size_t axis_index) {
  bool has_payload = false;
  af::Expression axis_stride;
  return IsContiguousSuffixImpl(view, axis_index, true, has_payload, axis_stride);
}

bool HasZeroStrideOnInputPayload(const LogicalTensorView &input, const LogicalTensorView &index, size_t axis_index) {
  bool has_payload = false;
  af::Expression axis_stride;
  if (input.sizes.size() != index.sizes.size() ||
      !IsContiguousSuffixImpl(input, axis_index, true, has_payload, axis_stride)) {
    return false;
  }
  if (!has_payload) {
    return false;
  }
  for (size_t dim = axis_index + 1UL; dim < input.sizes.size(); ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(input.strides[dim], af::sym::kSymbolZero) != af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(index.strides[dim], af::sym::kSymbolZero) != af::TriBool::kTrue) {
      return false;
    }
  }
  return true;
}

bool IsPayloadInvariantView(const af::AscTensorAttr &source_attr, const LogicalTensorView &input,
                            const LogicalTensorView &index, size_t axis_index) {
  if (source_attr.axis.size() != source_attr.repeats.size() || source_attr.axis.size() != source_attr.strides.size()) {
    return false;
  }
  for (size_t dim = axis_index + 1UL; dim < index.axis_ids.size(); ++dim) {
    // Only dimensions that carry the gathered payload need an invariant index.
    // Other dimensions may legitimately enumerate independent lookups (for
    // example the sequence axis in [B, D, S] embedding views).
    if (af::SymbolicUtils::StaticCheckEq(input.strides[dim], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    const auto source_axis = std::find(source_attr.axis.begin(), source_attr.axis.end(), index.axis_ids[dim]);
    if (source_axis == source_attr.axis.end()) {
      return false;
    }
    const size_t source_dim = static_cast<size_t>(std::distance(source_attr.axis.begin(), source_axis));
    const auto &source_size = source_attr.repeats[source_dim];
    const auto &source_stride = source_attr.strides[source_dim];
    if (af::SymbolicUtils::StaticCheckEq(source_stride, af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    if (af::SymbolicUtils::StaticCheckGt(source_size, af::sym::kSymbolOne) != af::TriBool::kFalse) {
      return false;
    }
  }
  return true;
}

bool IsIndexPayloadInvariantFromSources(const af::AscNodePtr &indirect_load, const LogicalTensorView &input,
                                        const LogicalTensorView &index, size_t axis_index) {
  const af::AscNodePtr root = GetInputProducer(indirect_load, kIndexTensorIndex);
  if (root == nullptr) {
    return false;
  }
  std::vector<af::AscNodePtr> pending{root};
  std::unordered_set<const af::AscNode *> visited;
  bool saw_source = false;
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr node = pending[cursor];
    if (node == nullptr || !visited.emplace(node.get()).second) {
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Scalar>(node) || af::ops::IsOps<af::ascir_op::ScalarData>(node)) {
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Load>(node)) {
      const auto outputs = node->outputs();
      if (outputs.empty() || outputs.front() == nullptr ||
          !IsPayloadInvariantView(outputs.front()->attr, input, index, axis_index)) {
        return false;
      }
      saw_source = true;
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Data>(node)) {
      const auto outputs = node->outputs();
      if (!outputs.empty() && outputs.front() != nullptr && !outputs.front()->attr.axis.empty()) {
        if (!IsPayloadInvariantView(outputs.front()->attr, input, index, axis_index)) {
          return false;
        }
        saw_source = true;
      }
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::IndirectLoad>(node)) {
      return false;
    }
    for (size_t input_index = 0UL; input_index < node->inputs.Size(); ++input_index) {
      pending.emplace_back(GetInputProducer(node, input_index));
    }
  }
  return saw_source;
}

bool IsSimdEmbeddingAccess(const TemplateLogicalView &logical_view, const IndirectLoadAccessInfo &info) {
  const size_t rank = logical_view.input.sizes.size();
  if (info.kind != IndirectLoadAccessInfo::Kind::kEmbeddingLike || rank < 2UL ||
      logical_view.index.sizes.size() != rank || logical_view.output.sizes.size() != rank || info.axis < 0L ||
      static_cast<size_t>(info.axis + 1L) >= rank) {
    return false;
  }
  const auto &input = logical_view.input;
  const auto &index = logical_view.index;
  const auto &output = logical_view.output;
  const size_t axis = static_cast<size_t>(info.axis);
  for (size_t dim = 0UL; dim < rank; ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(index.sizes[dim], output.sizes[dim]) != af::TriBool::kTrue) {
      return false;
    }
  }
  bool has_payload = false;
  af::Expression input_axis_stride;
  if (!IsContiguousSuffixImpl(input, axis, false, has_payload, input_axis_stride)) {
    return false;
  }
  for (size_t dim = axis + 1UL; dim < rank; ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(input.sizes[dim], output.sizes[dim]) != af::TriBool::kTrue) {
      return false;
    }
  }
  if (af::SymbolicUtils::StaticCheckEq(input.strides[axis], input_axis_stride) != af::TriBool::kTrue) {
    return false;
  }
  for (size_t dim = 0UL; dim < axis; ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(input.strides[dim], af::sym::kSymbolZero) != af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(input.sizes[dim], output.sizes[dim]) != af::TriBool::kTrue) {
      return false;
    }
  }
  return af::SymbolicUtils::StaticCheckGt(input.strides[axis], af::sym::kSymbolZero) == af::TriBool::kTrue;
}
}  // namespace

af::Status AnalyzeIndirectLoadAccess(const af::AscNodePtr &node, const TemplateLogicalView &logical_view,
                                     IndirectLoadAccessInfo &info) {
  GE_ASSERT_NOTNULL(node);
  const auto &input = logical_view.input;
  const auto &index = logical_view.index;
  const auto &output = logical_view.output;
  GE_ASSERT_TRUE(input.sizes.size() == index.sizes.size() && input.sizes.size() == output.sizes.size(),
                 "IndirectLoad access analysis rank mismatch.");
  const size_t rank = input.sizes.size();
  const auto *ir_attr = node->attr.ir_attr == nullptr
                            ? nullptr
                            : node->attr.ir_attr->DownCastTo<af::ascir_op::IndirectLoad::AscIndirectLoadIrAttrDef>();
  GE_ASSERT_NOTNULL(ir_attr, "IndirectLoad access analysis axis attribute is missing.");
  int64_t axis = 0L;
  GE_ASSERT_SUCCESS(ir_attr->GetAxis(axis));
  GE_ASSERT_TRUE(axis >= -static_cast<int64_t>(rank) && axis < static_cast<int64_t>(rank),
                 "IndirectLoad access analysis axis is invalid.");
  const size_t axis_index = static_cast<size_t>(axis < 0L ? axis + static_cast<int64_t>(rank) : axis);

  info = {};
  info.axis = static_cast<int64_t>(axis_index);
  const af::Expression input_inner_span = ProductFrom(input, axis_index + 1UL);
  const auto indirect_load_inputs = node->inputs();
  GE_ASSERT_TRUE(indirect_load_inputs.size() > kInputTensorIndex, "IndirectLoad access analysis input is missing.");
  const af::Expression dtype_bytes =
      af::Symbol(af::GetSizeByDataType(indirect_load_inputs[kInputTensorIndex]->attr.dtype));
  info.input_slice_bytes = input_inner_span * dtype_bytes;
  info.index_varying_extent = af::sym::kSymbolOne;
  for (size_t dim = 0UL; dim < rank; ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(index.strides[dim], af::sym::kSymbolZero) != af::TriBool::kTrue) {
      info.index_varying_extent = info.index_varying_extent * index.sizes[dim];
    }
  }
  // The index is invariant on the payload axes when proven from IL's view or from source
  // Load/Data views reached through the x2 producer subgraph (pointwise outputs can be
  // dense even when their values repeat along a payload axis).
  const bool index_has_zero_stride_on_inner_axes = HasZeroStrideOnInputPayload(input, index, axis_index) ||
                                                   IsIndexPayloadInvariantFromSources(node, input, index, axis_index);
  const bool input_axis_stride_is_positive =
      af::SymbolicUtils::StaticCheckGt(input.strides[axis_index], af::sym::kSymbolZero) == af::TriBool::kTrue;
  // An Embedding-like access must have a non-empty payload suffix after the
  // gather axis.  Without this guard, a last-axis gather vacuously satisfies
  // the zero-stride/contiguous-payload checks and is misclassified as
  // Embedding-like, although it has no payload to copy as a unit.  The suffix
  // must also carry real data: when every suffix dimension of the input is
  // zero-stride (broadcast), the payload-invariance proof is vacuous and the
  // index may legitimately vary along the inner axes, which the Embedding
  // address policy cannot express.
  bool input_suffix_has_payload = false;
  af::Expression input_axis_dense_stride;
  IsContiguousSuffixImpl(input, axis_index, true, input_suffix_has_payload, input_axis_dense_stride);
  const bool has_payload_suffix = axis_index + 1UL < rank && input_suffix_has_payload;
  const bool embedding_like = has_payload_suffix && index_has_zero_stride_on_inner_axes &&
                              IsContiguousPayloadSuffix(input, axis_index) && IsContiguousSuffix(output, axis_index) &&
                              input_axis_stride_is_positive;
  info.kind = embedding_like ? IndirectLoadAccessInfo::Kind::kEmbeddingLike : IndirectLoadAccessInfo::Kind::kGeneric;
  info.can_use_simt_structured = embedding_like;
  info.can_use_simd_embedding = IsSimdEmbeddingAccess(logical_view, info);
  return af::SUCCESS;
}

namespace {
using NodePath = std::vector<af::AscNodePtr>;
using NodeSet = std::unordered_set<const af::AscNode *>;

bool IsSimtDirectGmBoundary(const af::AscNodePtr &node) {
  return af::ops::IsOps<af::ascir_op::Load>(node) ||
         (af::ops::IsOps<af::ascir_op::Nddma>(node) && GetTemplateBehavior(node).uses_direct_gm_pipeline);
}

bool SimtLoadUsesZeroOffset(const af::AscNodePtr &node) {
  if (node == nullptr || node->outputs().empty()) {
    return false;
  }
  const auto &attr = node->outputs()[0]->attr;
  const bool all_zero_strides =
      !attr.strides.empty() && std::all_of(attr.strides.begin(), attr.strides.end(), [](const af::Expression &stride) {
        return af::SymbolicUtils::StaticCheckEq(stride, af::ops::Zero) == af::TriBool::kTrue;
      });
  const bool single_element_shape =
      !attr.repeats.empty() && std::all_of(attr.repeats.begin(), attr.repeats.end(), [](const af::Expression &size) {
        return af::SymbolicUtils::StaticCheckEq(size, af::ops::One) == af::TriBool::kTrue;
      });
  return all_zero_strides || single_element_shape;
}

bool SimtLoadViewsMatch(const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) {
  if (lhs == nullptr || rhs == nullptr || lhs->outputs().empty() || rhs->outputs().empty()) {
    return false;
  }
  const auto &lhs_attr = lhs->outputs()[0]->attr;
  const auto &rhs_attr = rhs->outputs()[0]->attr;
  if (lhs_attr.repeats.empty() || lhs_attr.strides.empty() || lhs_attr.repeats.size() != rhs_attr.repeats.size() ||
      lhs_attr.strides.size() != rhs_attr.strides.size()) {
    return false;
  }
  for (size_t dim = 0UL; dim < lhs_attr.repeats.size(); ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(lhs_attr.repeats[dim], rhs_attr.repeats[dim]) != af::TriBool::kTrue ||
        af::SymbolicUtils::StaticCheckEq(lhs_attr.strides[dim], rhs_attr.strides[dim]) != af::TriBool::kTrue) {
      return false;
    }
  }
  return true;
}

af::Status GetStableGraphNodes(const af::AscNodePtr &node, NodePath &nodes) {
  const auto owner_graph = node->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph, "IndirectLoad node has no owner graph.");
  for (const auto &graph_node : owner_graph->GetDirectNode()) {
    const auto asc_node = std::dynamic_pointer_cast<af::AscNode>(graph_node);
    GE_ASSERT_NOTNULL(asc_node, "IndirectLoad graph contains invalid node.");
    nodes.emplace_back(asc_node);
  }
  return af::SUCCESS;
}

af::Status CollectSimtLoweringBackwardNodes(const af::AscNodePtr &root, const af::AscNodePtr &indirect_load,
                                            NodeSet &nodes) {
  NodePath pending = {root};
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr current = pending[cursor];
    if (current == nullptr || current == indirect_load || !nodes.emplace(current.get()).second) {
      continue;
    }
    if (IsSimtDirectGmBoundary(current) || af::ops::IsOps<af::ascir_op::Scalar>(current) ||
        af::ops::IsOps<af::ascir_op::ScalarData>(current) || af::ops::IsOps<af::ascir_op::Arange>(current)) {
      continue;
    }
    GE_ASSERT_TRUE(current->inputs.Size() > 0UL, "IndirectLoad SIMT node[%s] has no input.", current->GetNamePtr());
    for (size_t i = 0UL; i < current->inputs.Size(); ++i) {
      const auto producer = GetInputProducer(current, i);
      GE_ASSERT_NOTNULL(producer, "IndirectLoad SIMT node[%s] input[%zu] has no producer.", current->GetNamePtr(), i);
      pending.emplace_back(producer);
    }
  }
  return af::SUCCESS;
}

size_t GetProducerOutputIndex(const af::AscNodePtr &consumer) {
  const auto input_anchor = consumer == nullptr ? nullptr : consumer->GetInDataAnchor(0UL);
  const auto peer_anchor = input_anchor == nullptr ? nullptr : input_anchor->GetPeerOutAnchor();
  return peer_anchor == nullptr ? 0UL : static_cast<size_t>(peer_anchor->GetIdx());
}

struct SimtOutputChainBuild {
  SimtOutputChainMetadata metadata;
  NodeSet nodes;
};

af::Status CollectSimtOutputDescendants(const af::AscNodePtr &indirect_load, NodeSet &descendants) {
  NodePath pending;
  for (const auto &out_node : indirect_load->GetOutDataNodes()) {
    const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
    GE_ASSERT_NOTNULL(consumer, "IndirectLoad SIMT output successor is invalid.");
    pending.emplace_back(consumer);
  }
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const auto &node = pending[cursor];
    if (!descendants.emplace(node.get()).second || af::ops::IsOps<af::ascir_op::Store>(node)) {
      continue;
    }
    for (const auto &out_node : node->GetOutDataNodes()) {
      const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
      GE_ASSERT_NOTNULL(consumer, "IndirectLoad SIMT output successor is invalid.");
      pending.emplace_back(consumer);
    }
  }
  return af::SUCCESS;
}

af::Status CollectSimtOutputChainBuilds(const NodePath &graph_nodes, const af::AscNodePtr &indirect_load,
                                        std::vector<SimtOutputChainBuild> &chains) {
  NodeSet descendants;
  GE_ASSERT_SUCCESS(CollectSimtOutputDescendants(indirect_load, descendants));
  std::unordered_map<const af::AscNode *, size_t> cached_node_sets;
  for (const af::AscNodePtr &node : graph_nodes) {
    if (descendants.count(node.get()) == 0UL || !af::ops::IsOps<af::ascir_op::Store>(node)) {
      continue;
    }
    const auto producer = GetInputProducer(node, 0UL);
    GE_ASSERT_NOTNULL(producer, "IndirectLoad SIMT output terminal[%s] has no producer.", node->GetNamePtr());
    const size_t producer_output_index = GetProducerOutputIndex(node);
    GE_ASSERT_TRUE(producer_output_index < producer->outputs().size(),
                   "IndirectLoad SIMT terminal[%s] producer output index[%zu] is invalid.", node->GetNamePtr(),
                   producer_output_index);
    SimtOutputChainBuild chain;
    const auto cached = cached_node_sets.find(producer.get());
    if (cached == cached_node_sets.end()) {
      GE_ASSERT_SUCCESS(CollectSimtLoweringBackwardNodes(producer, indirect_load, chain.nodes));
      cached_node_sets.emplace(producer.get(), chains.size());
    } else {
      chain.nodes = chains[cached->second].nodes;
    }
    chain.metadata.result_tensor_id = producer->outputs()[producer_output_index]->attr.mem.tensor_id;
    chain.metadata.target_tensor_id = node->outputs()[0]->attr.mem.tensor_id;
    chain.metadata.dtype = producer->outputs()[producer_output_index]->attr.dtype;
    chains.emplace_back(std::move(chain));
  }
  return af::SUCCESS;
}

af::Status ValidateSimtLoweringNode(const af::AscNodePtr &node) {
  if (IsSimtDirectGmBoundary(node) || af::ops::IsOps<af::ascir_op::Scalar>(node) ||
      af::ops::IsOps<af::ascir_op::ScalarData>(node) || af::ops::IsOps<af::ascir_op::Arange>(node) ||
      af::ops::IsOps<af::ascir_op::Store>(node) || af::ops::IsOps<af::ascir_op::Transpose>(node)) {
    return af::SUCCESS;
  }
  const auto role = GetTemplateRole(node);
  GE_ASSERT_TRUE(role == TemplateRole::kSimtInlineTransform || role == TemplateRole::kSimtFanoutBranch,
                 "IndirectLoad SIMT node[%s] has no scalar evaluator role.", node->GetNamePtr());
  GE_ASSERT_TRUE(!af::ops::IsOps<af::ascir_op::VectorFunc>(node),
                 "IndirectLoad SIMT transform must use scalar emission, node:%s", node->GetNamePtr());
  return af::SUCCESS;
}

bool TryBuildStaticSpan(const af::Expression &size_expr, uint64_t stride, uint64_t &span) {
  int64_t size = 0L;
  if (!size_expr.GetConstValue(size) || size < 0L) {
    return false;
  }
  return !ge::MulOverflow(static_cast<uint64_t>(size), stride, span);
}

bool TryAccumulateStaticOffset(const af::Expression &size_expr, const af::Expression &stride_expr, uint64_t &offset) {
  int64_t size = 0L;
  int64_t stride = 0L;
  if (!size_expr.GetConstValue(size) || size <= 0L || !stride_expr.GetConstValue(stride) || stride < 0L) {
    return false;
  }
  uint64_t dim_offset = 0U;
  return !ge::MulOverflow(static_cast<uint64_t>(size - 1L), static_cast<uint64_t>(stride), dim_offset) &&
         !ge::AddOverflow(offset, dim_offset, offset);
}

bool TryGetStaticSpans(const LogicalTensorView &input, const LogicalTensorView &output, size_t axis,
                       SimtPolicyMetadata &policy) {
  uint64_t inner = 1U;
  for (size_t i = axis + 1U; i < output.sizes.size(); ++i) {
    if (!TryBuildStaticSpan(output.sizes[i], inner, inner)) {
      return false;
    }
  }
  int64_t input_stride = 0L;
  if (!input.strides[axis].GetConstValue(input_stride) || input_stride < 0L ||
      !TryBuildStaticSpan(output.sizes[axis], inner, policy.output_axis_span) ||
      !TryBuildStaticSpan(input.sizes[axis], static_cast<uint64_t>(input_stride), policy.input_axis_span)) {
    return false;
  }
  policy.inner_span = inner;
  policy.input_axis_stride = static_cast<uint64_t>(input_stride);
  return true;
}

bool TryGetMaxElementOffset(const LogicalTensorView &tensor, uint64_t &max_offset) {
  max_offset = 0U;
  for (size_t i = 0U; i < tensor.sizes.size(); ++i) {
    if (!TryAccumulateStaticOffset(tensor.sizes[i], tensor.strides[i], max_offset)) {
      return false;
    }
  }
  return true;
}

bool IsDense(const LogicalTensorView &tensor) {
  af::Expression expected_stride = af::ops::One;
  for (size_t i = tensor.sizes.size(); i > 0U; --i) {
    const size_t dim = i - 1U;
    if (af::SymbolicUtils::StaticCheckEq(tensor.strides[dim], expected_stride) != af::TriBool::kTrue) {
      return false;
    }
    expected_stride = af::sym::Mul(expected_stride, tensor.sizes[dim]);
  }
  return true;
}

bool IsStructuredSimt(const LogicalTensorView &input, const LogicalTensorView &index, const LogicalTensorView &output,
                      size_t axis) {
  if (!IsDense(input) || !IsDense(index) || !IsDense(output) || index.sizes.size() != output.sizes.size() ||
      input.sizes.size() != output.sizes.size()) {
    return false;
  }
  for (size_t i = 0U; i < output.sizes.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(index.sizes[i], output.sizes[i]) != af::TriBool::kTrue ||
        (i != axis && af::SymbolicUtils::StaticCheckEq(input.sizes[i], output.sizes[i]) != af::TriBool::kTrue)) {
      return false;
    }
  }
  return true;
}

bool IsPowerOfTwo(uint64_t value) {
  return value != 0U && (value & (value - 1U)) == 0U;
}

bool CanUseUint32Offsets(const LogicalTensorView &input, const LogicalTensorView &index,
                         const LogicalTensorView &output) {
  uint64_t input_max = 0U;
  uint64_t index_max = 0U;
  uint64_t output_max = 0U;
  const uint64_t limit = std::numeric_limits<uint32_t>::max();
  return TryGetMaxElementOffset(input, input_max) && input_max <= limit && TryGetMaxElementOffset(index, index_max) &&
         index_max <= limit && TryGetMaxElementOffset(output, output_max) && output_max <= limit;
}

bool CanUseUint32Divisors(const LogicalTensorView &index) {
  for (const af::Expression &size_expr : index.sizes) {
    int64_t size = 0L;
    if (!size_expr.GetConstValue(size) || size < 0L || size > static_cast<int64_t>(INT32_MAX)) {
      return false;
    }
  }
  return true;
}

uint64_t BuildNonZeroStrideMask(const std::vector<af::Expression> &strides) {
  uint64_t mask = 0U;
  for (size_t dim = 0U; dim < strides.size() && dim < 64UL; ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(strides[dim], af::sym::kSymbolZero) != af::TriBool::kTrue) {
      mask |= 1ULL << dim;
    }
  }
  return mask;
}

bool SimtPhysicalStridesEqual(const std::vector<af::Expression> &lhs, const std::vector<af::Expression> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  return std::equal(lhs.begin(), lhs.end(), rhs.begin(), [](const af::Expression &left, const af::Expression &right) {
    return af::SymbolicUtils::StaticCheckEq(left, right) == af::TriBool::kTrue;
  });
}

bool SimtHasZeroStride(const std::vector<af::Expression> &strides) {
  return std::any_of(strides.begin(), strides.end(), [](const af::Expression &stride) {
    return af::SymbolicUtils::StaticCheckEq(stride, af::ops::Zero) == af::TriBool::kTrue;
  });
}

bool ApplySimtIndexPhysicalStrides(const NodePath &index_nodes, LogicalTensorView &index, bool &mixed_views) {
  mixed_views = false;
  const std::vector<af::Expression> *canonical_strides = nullptr;
  bool has_zero_stride = false;
  for (const af::AscNodePtr &node : index_nodes) {
    if (!IsSimtDirectGmBoundary(node) || node->outputs().empty() ||
        node->outputs()[0]->attr.strides.size() != index.strides.size()) {
      continue;
    }
    const auto &strides = node->outputs()[0]->attr.strides;
    if (canonical_strides == nullptr) {
      canonical_strides = &strides;
    } else if (!SimtPhysicalStridesEqual(*canonical_strides, strides)) {
      mixed_views = true;
    }
    has_zero_stride = has_zero_stride || SimtHasZeroStride(strides);
  }
  if (canonical_strides == nullptr) {
    for (const af::AscNodePtr &node : index_nodes) {
      const std::vector<af::Expression> *physical_strides = nullptr;
      if (af::ops::IsOps<af::ascir_op::Broadcast>(node) && !node->inputs().empty() &&
          node->inputs()[0]->attr.strides.size() == index.strides.size()) {
        physical_strides = &node->inputs()[0]->attr.strides;
      } else if (!node->outputs().empty() && node->outputs()[0]->attr.strides.size() == index.strides.size()) {
        physical_strides = &node->outputs()[0]->attr.strides;
      }
      if (physical_strides != nullptr && SimtHasZeroStride(*physical_strides)) {
        canonical_strides = physical_strides;
        has_zero_stride = true;
        break;
      }
    }
  }
  if (canonical_strides != nullptr && has_zero_stride && !mixed_views) {
    index.strides = *canonical_strides;
  }
  return has_zero_stride || mixed_views;
}

void AppendRuntimeParams(std::vector<af::Expression> &target, const std::vector<af::Expression> &source) {
  target.insert(target.end(), source.begin(), source.end());
}

af::Status BuildSimtPolicyMetadata(const TemplateLogicalView &logical_view, const NodePath &index_nodes, size_t axis,
                                   bool embedding_structured, bool &mixed_index_views, SimtPolicyMetadata &metadata) {
  const size_t rank = logical_view.input.sizes.size();
  GE_ASSERT_TRUE(rank < 64UL, "IndirectLoad SIMT rank must be smaller than 64.");
  LogicalTensorView input = logical_view.input;
  LogicalTensorView index = logical_view.index;
  const auto &output = logical_view.output;
  const bool index_broadcast_strided = ApplySimtIndexPhysicalStrides(index_nodes, index, mixed_index_views);
  const bool strided = logical_view.input.kind != IndirectLoadLayoutKind::kDense ||
                       logical_view.index.kind != IndirectLoadLayoutKind::kDense || index_broadcast_strided;
  af::Expression inner_span = af::sym::kSymbolOne;
  for (size_t i = axis + 1U; i < output.sizes.size(); ++i) {
    inner_span = af::sym::Mul(inner_span, output.sizes[i]);
  }
  const af::Expression output_axis_span = af::sym::Mul(output.sizes[axis], inner_span);
  const af::Expression input_axis_stride = input.strides[axis];
  const af::Expression input_axis_span = af::sym::Mul(input.sizes[axis], input_axis_stride);
  const bool structured = IsStructuredSimt(input, index, output, axis);
  const bool static_spans = TryGetStaticSpans(input, output, axis, metadata);
  if (embedding_structured) {
    metadata.policy = SimtAddressPolicy::kEmbedding;
    metadata.input_stride_mask = BuildNonZeroStrideMask(input.strides);
    metadata.index_stride_mask = BuildNonZeroStrideMask(index.strides);
    if (rank == 2UL && axis == 0UL) {
      metadata.input_stride_mask = 0U;
      metadata.index_stride_mask = 0U;
    }
  } else if (strided) {
    metadata.policy = SimtAddressPolicy::kStrided;
    metadata.input_stride_mask = BuildNonZeroStrideMask(input.strides);
    metadata.index_stride_mask = BuildNonZeroStrideMask(index.strides);
  } else if (structured) {
    const bool static_power_of_two =
        static_spans && IsPowerOfTwo(metadata.inner_span) && IsPowerOfTwo(metadata.output_axis_span);
    const bool static_inner = static_spans && IsPowerOfTwo(metadata.inner_span);
    const bool magic_divisors_supported =
        !static_spans || (metadata.inner_span <= static_cast<uint64_t>(INT64_MAX) &&
                          metadata.output_axis_span <= static_cast<uint64_t>(INT64_MAX));
    metadata.policy = static_power_of_two        ? SimtAddressPolicy::kStaticPowerOfTwo
                      : static_inner             ? SimtAddressPolicy::kStaticInner
                      : magic_divisors_supported ? SimtAddressPolicy::kStructuredMagic
                                                 : SimtAddressPolicy::kRecursive;
  }
  const bool structured_magic_supported = metadata.policy != SimtAddressPolicy::kStructuredMagic ||
                                          (static_spans && metadata.inner_span <= static_cast<uint64_t>(INT32_MAX) &&
                                           metadata.output_axis_span <= static_cast<uint64_t>(INT32_MAX));
  const bool static_inner_magic_supported = metadata.policy != SimtAddressPolicy::kStaticInner ||
                                            metadata.output_axis_span <= static_cast<uint64_t>(INT32_MAX);
  const bool general_magic_supported =
      (metadata.policy != SimtAddressPolicy::kRecursive && metadata.policy != SimtAddressPolicy::kStrided &&
       metadata.policy != SimtAddressPolicy::kEmbedding) ||
      CanUseUint32Divisors(index);
  if (CanUseUint32Offsets(input, index, output) && structured_magic_supported && static_inner_magic_supported &&
      general_magic_supported) {
    metadata.offset_width = SimtOffsetWidth::kUint32;
  }
  switch (metadata.policy) {
    case SimtAddressPolicy::kStaticPowerOfTwo:
      break;
    case SimtAddressPolicy::kStaticInner:
      metadata.runtime_params.emplace_back(output_axis_span);
      break;
    case SimtAddressPolicy::kStructuredMagic:
      metadata.runtime_params = {inner_span, output_axis_span, input_axis_stride, input_axis_span};
      break;
    case SimtAddressPolicy::kEmbedding:
      AppendRuntimeParams(metadata.runtime_params, index.sizes);
      AppendRuntimeParams(metadata.runtime_params, input.strides);
      AppendRuntimeParams(metadata.runtime_params, index.strides);
      break;
    case SimtAddressPolicy::kRecursive:
      AppendRuntimeParams(metadata.runtime_params, index.sizes);
      AppendRuntimeParams(metadata.runtime_params, input.strides);
      break;
    case SimtAddressPolicy::kStrided:
      AppendRuntimeParams(metadata.runtime_params, index.sizes);
      AppendRuntimeParams(metadata.runtime_params, input.strides);
      AppendRuntimeParams(metadata.runtime_params, index.strides);
      break;
  }
  return af::SUCCESS;
}

LogicalTensorView GetNodeOutputView(const af::AscNodePtr &node) {
  const auto &attr = node->outputs()[0]->attr;
  return {attr.axis, attr.repeats, attr.strides};
}

void AppendSimtGmTensorMetadata(const af::AscNodePtr &node, SimtLoweringMetadata &metadata,
                                std::unordered_set<ascir::TensorId> &seen_tensor_ids) {
  const auto output = node->outputs()[0];
  if (!seen_tensor_ids.emplace(output->attr.mem.tensor_id).second) {
    return;
  }
  if (IsSimtDirectGmBoundary(node)) {
    metadata.gm_tensors.push_back(
        {output->attr.mem.tensor_id, node->inputs()[0]->attr.mem.tensor_id, output->attr.dtype, false});
  } else {
    metadata.gm_tensors.push_back({output->attr.mem.tensor_id, output->attr.mem.tensor_id, output->attr.dtype, true});
  }
}

af::Status ValidateSimtOutputPlan(const SimtLoweringMetadata &metadata) {
  GE_ASSERT_TRUE(!metadata.output_chains.empty(), "IndirectLoad SIMT output chains are empty.");
  const size_t local_target_count =
      static_cast<size_t>(std::count_if(metadata.output_chains.begin(), metadata.output_chains.end(),
                                        [](const SimtOutputChainMetadata &chain) { return chain.local_target; }));
  GE_ASSERT_TRUE(local_target_count <= 1UL, "IndirectLoad SIMT supports at most one local output target.");
  GE_ASSERT_TRUE(metadata.has_post_reduce == (local_target_count == 1UL),
                 "IndirectLoad SIMT post Reduce and local output target must appear together.");
  if (local_target_count == 1UL) {
    const auto &local_chain = metadata.output_chains.back();
    GE_ASSERT_TRUE(local_chain.local_target && local_chain.result_tensor_id == metadata.output_result_tensor_id &&
                       local_chain.target_tensor_id == metadata.output_result_tensor_id &&
                       local_chain.node_names == metadata.output_node_names,
                   "IndirectLoad SIMT local output must be the synthetic post Reduce chain.");
  }
  return af::SUCCESS;
}

af::Status BuildSimtLoweringMetadata(const af::AscNodePtr &indirect_load, IndirectLoadLoweringMetadata &metadata) {
  NodePath graph_nodes;
  GE_ASSERT_SUCCESS(GetStableGraphNodes(indirect_load, graph_nodes));
  auto &simt = metadata.simt;
  const af::AscNodePtr post_reduce = GetPostReduceConsumer(indirect_load);
  simt.has_post_reduce = post_reduce != nullptr;
  const auto inputs = indirect_load->inputs();
  GE_ASSERT_TRUE(inputs.size() == 2UL, "Invalid IndirectLoad SIMT input number:%zu.", inputs.size());
  simt.index_result_tensor_id = inputs[kIndexTensorIndex]->attr.mem.tensor_id;
  simt.index_dtype = inputs[kIndexTensorIndex]->attr.dtype;
  simt.value_tensor_id = indirect_load->outputs()[0]->attr.mem.tensor_id;

  const auto index_root = GetInputProducer(indirect_load, kIndexTensorIndex);
  GE_ASSERT_NOTNULL(index_root, "IndirectLoad SIMT index input has no producer.");
  NodeSet index_set;
  GE_ASSERT_SUCCESS(CollectSimtLoweringBackwardNodes(index_root, indirect_load, index_set));
  NodeSet output_set;
  af::AscNodePtr output_root;
  if (simt.has_post_reduce) {
    output_root = GetPostReduceInputProducer(indirect_load);
    GE_ASSERT_NOTNULL(output_root, "IndirectLoad post Reduce input has no producer.");
    GE_ASSERT_SUCCESS(CollectSimtLoweringBackwardNodes(output_root, indirect_load, output_set));
  }

  std::vector<SimtOutputChainBuild> chain_builds;
  GE_ASSERT_SUCCESS(CollectSimtOutputChainBuilds(graph_nodes, indirect_load, chain_builds));
  std::vector<bool> keep_chain(chain_builds.size(), true);
  for (size_t chain_index = 0UL; chain_index < chain_builds.size(); ++chain_index) {
    const bool contains_reduce =
        simt.has_post_reduce && std::any_of(chain_builds[chain_index].nodes.begin(),
                                            chain_builds[chain_index].nodes.end(), [](const af::AscNode *node) {
                                              return node != nullptr &&
                                                     node->attr.api.compute_type == af::ComputeType::kComputeReduce;
                                            });
    keep_chain[chain_index] = !contains_reduce;
  }

  NodePath index_nodes;
  NodePath output_load_nodes;
  std::unordered_set<ascir::TensorId> seen_gm_tensors;
  for (const af::AscNodePtr &node : graph_nodes) {
    const bool in_index = index_set.count(node.get()) != 0UL;
    const bool in_output_region = output_set.count(node.get()) != 0UL;
    bool in_output_chain = false;
    for (size_t chain_index = 0UL; chain_index < chain_builds.size(); ++chain_index) {
      if (keep_chain[chain_index] && chain_builds[chain_index].nodes.count(node.get()) != 0UL) {
        chain_builds[chain_index].metadata.node_names.emplace_back(node->GetName());
        in_output_chain = true;
      }
    }
    if (!in_index && !in_output_region && !in_output_chain) {
      continue;
    }
    GE_ASSERT_SUCCESS(ValidateSimtLoweringNode(node));
    if (in_index) {
      index_nodes.emplace_back(node);
      simt.index_node_names.emplace_back(node->GetName());
    }
    if (in_output_region || in_output_chain) {
      simt.output_node_names.emplace_back(node->GetName());
    }
    if ((in_output_region || in_output_chain) && IsSimtDirectGmBoundary(node)) {
      output_load_nodes.emplace_back(node);
    }
    if (IsSimtDirectGmBoundary(node) || af::ops::IsOps<af::ascir_op::ScalarData>(node)) {
      AppendSimtGmTensorMetadata(node, simt, seen_gm_tensors);
    }
  }
  for (size_t chain_index = 0UL; chain_index < chain_builds.size(); ++chain_index) {
    if (keep_chain[chain_index]) {
      simt.output_chains.emplace_back(std::move(chain_builds[chain_index].metadata));
    }
  }
  if (simt.has_post_reduce) {
    GE_ASSERT_TRUE(!output_root->outputs().empty(), "IndirectLoad post Reduce input producer has no output.");
    simt.output_result_tensor_id = output_root->outputs()[0]->attr.mem.tensor_id;
    simt.output_dtype = output_root->outputs()[0]->attr.dtype;
    simt.output_chains.push_back(
        {simt.output_node_names, simt.output_result_tensor_id, simt.output_result_tensor_id, simt.output_dtype, true});
  } else {
    GE_ASSERT_TRUE(!simt.output_chains.empty(), "IndirectLoad SIMT output chains are empty.");
    if (simt.output_chains.size() == 1UL) {
      simt.output_result_tensor_id = simt.output_chains[0].result_tensor_id;
      simt.output_node_names = simt.output_chains[0].node_names;
    }
    simt.output_dtype = simt.output_chains.front().dtype;
  }

  NodePath index_load_nodes;
  for (const auto &node : index_nodes) {
    if (IsSimtDirectGmBoundary(node)) {
      index_load_nodes.emplace_back(node);
    }
  }
  std::unordered_map<std::string, SimtLoadAddressSource> output_sources;
  for (const auto &node : output_load_nodes) {
    auto source = SimtLoadAddressSource::kOutputOffset;
    if (SimtLoadUsesZeroOffset(node)) {
      source = SimtLoadAddressSource::kZeroOffset;
    } else if (std::any_of(index_load_nodes.begin(), index_load_nodes.end(), [&node](const af::AscNodePtr &index_load) {
                 return SimtLoadViewsMatch(node, index_load);
               })) {
      source = SimtLoadAddressSource::kIndexOffset;
    }
    output_sources.emplace(node->GetName(), source);
  }

  bool mixed_index_views = false;
  GE_ASSERT_SUCCESS(BuildSimtPolicyMetadata(metadata.logical_view, index_nodes, static_cast<size_t>(metadata.axis),
                                            metadata.access_info.can_use_simt_structured, mixed_index_views,
                                            simt.policy));
  for (const auto &node : index_load_nodes) {
    simt.index_loads.push_back(
        {node->GetName(),
         SimtLoadUsesZeroOffset(node) ? SimtLoadAddressSource::kZeroOffset : SimtLoadAddressSource::kOutputOffset,
         mixed_index_views, GetNodeOutputView(node)});
  }
  const bool use_output_logical_offset = !simt.has_post_reduce;
  for (const auto &node : output_load_nodes) {
    simt.output_loads.push_back(
        {node->GetName(), output_sources.at(node->GetName()), use_output_logical_offset, GetNodeOutputView(node)});
  }
  return af::SUCCESS;
}
}  // namespace

af::Status FinalizeLoweringMetadata(const af::AscNodePtr &node, bool &is_supported) {
  is_supported = false;
  GE_ASSERT_NOTNULL(node);
  const auto template_id = ::ascir::GetTemplateIdOrDefault(*node);
  GE_ASSERT_TRUE(
      template_id == ::ascir::TemplateId::kIndirectLoadSimd || template_id == ::ascir::TemplateId::kIndirectLoadSimt,
      "IndirectLoad lowering metadata only supports SIMD/SIMT, node = %s", node->GetNamePtr());
  IndirectLoadLoweringMetadata metadata;
  GE_ASSERT_SUCCESS(GetTemplateLogicalView(node, metadata.logical_view));
  GE_ASSERT_SUCCESS(AnalyzeIndirectLoadAccess(node, metadata.logical_view, metadata.access_info));
  TemplateAxes axes;
  GE_ASSERT_SUCCESS(GetTemplateAxes(node, axes));
  metadata.axis = metadata.access_info.axis;
  metadata.outer_axis = axes.outer_axis;
  Implementation implementation;
  GE_ASSERT_SUCCESS(GetImplementation(node, implementation));
  if (template_id == ::ascir::TemplateId::kIndirectLoadSimd) {
    const bool strided = metadata.logical_view.input.kind != IndirectLoadLayoutKind::kDense ||
                         metadata.logical_view.index.kind != IndirectLoadLayoutKind::kDense;
    metadata.simd.fallback = strided                                        ? SimdFallback::kStrided
                             : implementation == Implementation::kGatherApi ? SimdFallback::kGatherApi
                                                                            : SimdFallback::kRegisterGather;
    metadata.simd.try_embedding =
        implementation == Implementation::kDefault && metadata.access_info.can_use_simd_embedding;
  } else {
    if (metadata.logical_view.input.sizes.size() >= 64UL) {
      GELOGI("[IndirectLoad] Reject SIMT lowering metadata for node[%s]: rank[%zu] must be smaller than 64.",
             node->GetNamePtr(), metadata.logical_view.input.sizes.size());
      return af::SUCCESS;
    }
    GE_ASSERT_SUCCESS(BuildSimtLoweringMetadata(node, metadata));
    GE_ASSERT_SUCCESS(ValidateSimtOutputPlan(metadata.simt));
  }
  GE_ASSERT_SUCCESS(SetIndirectLoadAccessInfo(node, metadata.access_info));
  GE_ASSERT_SUCCESS(SetLoweringMetadata(node, metadata));
  is_supported = true;
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
