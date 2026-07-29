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
#include "schedule_result.h"

#include <algorithm>
#include <sstream>
#include <vector>

namespace optimize {
namespace {
constexpr int64_t kIndirectLoadSimtDcacheSize = 32 * 1024;
using NodePath = std::vector<af::AscNodePtr>;

struct IndirectLoadGraphPaths {
  NodePath data_input;
  NodePath index_input;
  NodePath output;
};

NodePath CollectInputPath(const af::AscNodePtr &indirect_load, size_t input_index) {
  NodePath path;
  for (auto current = ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_index); current != nullptr;
       current = current->inputs.Size() == 1UL ? ascgen_utils::indirect_load::GetInputProducer(current, 0UL)
                                               : nullptr) {
    path.emplace_back(current);
  }
  return path;
}

NodePath CollectOutputPath(const af::AscNodePtr &indirect_load) {
  NodePath path;
  for (auto current = ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load); current != nullptr;
       current = ascgen_utils::indirect_load::GetOnlyOutputConsumer(current)) {
    path.emplace_back(current);
  }
  return path;
}

IndirectLoadGraphPaths CollectGraphPaths(const af::AscNodePtr &indirect_load) {
  return {CollectInputPath(indirect_load, 0UL), CollectInputPath(indirect_load, 1UL), CollectOutputPath(indirect_load)};
}

std::string GenerateScoreFunc(ascir::TemplateId template_id, bool prefer_simd) {
  const bool is_simd = template_id == ascir::TemplateId::kIndirectLoadSimd;
  std::stringstream ss;
  ss << "int32_t CalcScore(AutofuseTilingData &tiling_data) {" << std::endl;
  ss << "  (void)tiling_data;" << std::endl;
  ss << "  return " << (is_simd == prefer_simd ? 1 : 0) << ";" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

bool HasExp2(const af::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Exp2>(node)) {
      return true;
    }
  }
  return false;
}

af::Status ValidateTemplateCandidate(ascir::TemplateId template_id, const IndirectLoadGraphPaths &paths,
                                     bool &is_candidate_legal) {
  is_candidate_legal = true;
  if (template_id != ascir::TemplateId::kIndirectLoadSimt) {
    return af::SUCCESS;
  }
  is_candidate_legal = false;
  if (paths.data_input.empty()) {
    return af::SUCCESS;
  }
  const auto &input_node = paths.data_input.front();
  if (af::ops::IsOps<af::ascir_op::Data>(input_node)) {
    is_candidate_legal = true;
  } else if (af::ops::IsOps<af::ascir_op::Load>(input_node)) {
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
        input_node, ascgen_utils::indirect_load::TemplateRole::kSimtInputBoundary));
    is_candidate_legal = true;
  }
  return af::SUCCESS;
}

af::Status GetIndirectLoadAxis(const af::AscNodePtr &node, int64_t &axis) {
  GE_ASSERT_NOTNULL(node, "IndirectLoad node is null.");
  GE_ASSERT_NOTNULL(node->attr.ir_attr, "IndirectLoad ir attr is null, node = %s", node->GetNamePtr());
  const auto *ir_attr = node->attr.ir_attr->DownCastTo<af::ascir_op::IndirectLoad::AscIndirectLoadIrAttrDef>();
  GE_ASSERT_NOTNULL(ir_attr, "IndirectLoad ir attr type is invalid, node = %s", node->GetNamePtr());
  GE_ASSERT_GRAPH_SUCCESS(ir_attr->GetAxis(axis), "Failed to get IndirectLoad axis, node = %s", node->GetNamePtr());
  return af::SUCCESS;
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
  GE_ASSERT_SUCCESS(ValidateIndirectLoadInputRank(indirect_load, 0UL, output_rank));
  GE_ASSERT_SUCCESS(ValidateIndirectLoadInputRank(indirect_load, 1UL, output_rank));
  return af::SUCCESS;
}

af::Status MergeAxesForTemplate(af::AscGraph &graph, const std::vector<af::AxisId> &axes, const std::string &name,
                                af::AxisId &merged_axis) {
  GE_ASSERT_TRUE(!axes.empty(), "IndirectLoad merge axis source is empty, name:%s.", name.c_str());
  merged_axis = axes.size() == 1UL ? axes.front() : graph.MergeAxis(axes, name)->id;
  return af::SUCCESS;
}

af::Status CreateFixedTileSplit(af::AscGraph &graph, af::AxisId axis_id) {
  const auto *axis = graph.FindAxis(axis_id);
  GE_ASSERT_NOTNULL(axis, "IndirectLoad fixed tile axis %ld is not found.", axis_id);
  const af::AxisId outer_id =
      graph.CreateAxis(axis->name + "T", ascir::Axis::Type::kAxisTypeTileOuter, axis->size, {axis_id}, af::kIdNone).id;
  const af::AxisId inner_id =
      graph
          .CreateAxis(axis->name + "t", ascir::Axis::Type::kAxisTypeTileInner, af::sym::kSymbolOne, {axis_id}, outer_id)
          .id;
  auto *outer_axis = graph.FindAxis(outer_id);
  GE_ASSERT_NOTNULL(outer_axis, "IndirectLoad fixed tile outer axis %ld is not found.", outer_id);
  outer_axis->split_pair_other_id = inner_id;
  return af::SUCCESS;
}

af::Status BuildSimdInputInnerAxis(af::AscGraph &graph, const af::AscNodePtr &input_producer, size_t axis_index,
                                   ascir::AxisId &input_inner_axis) {
  GE_ASSERT_NOTNULL(input_producer, "IndirectLoad SIMD input0 producer is null.");
  GE_ASSERT_TRUE(!input_producer->outputs().empty(), "IndirectLoad SIMD input0 producer has no output.");
  const auto input_axes = input_producer->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(axis_index < input_axes.size(), "IndirectLoad SIMD input axis index is out of range.");
  std::vector<ascir::AxisId> input_inner_axes(input_axes.begin() + static_cast<int64_t>(axis_index), input_axes.end());
  GE_ASSERT_SUCCESS(MergeAxesForTemplate(graph, input_inner_axes, "indirect_load_input_inner", input_inner_axis));
  return af::SUCCESS;
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
  GE_ASSERT_SUCCESS(CreateFixedTileSplit(graph, outer_axis));
  GE_ASSERT_SUCCESS(
      ascgen_utils::indirect_load::SetTemplateAxes(indirect_load, {outer_axis, inner_axis, input_inner_axis}));
  return af::SUCCESS;
}

af::Status NormalizeSimdAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                        const IndirectLoadGraphPaths &paths) {
  const auto output_axes = indirect_load->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(!output_axes.empty(), "IndirectLoad SIMD output axis is empty.");
  int64_t axis = 0L;
  GE_ASSERT_SUCCESS(GetIndirectLoadAxis(indirect_load, axis));
  const int64_t rank = static_cast<int64_t>(output_axes.size());
  const size_t axis_index = static_cast<size_t>(axis < 0L ? axis + rank : axis);
  ascir::AxisId input_inner_axis = af::kIdNone;
  const af::AscNodePtr input_producer = paths.data_input.empty() ? nullptr : paths.data_input.front();
  if (input_producer != nullptr && !af::ops::IsOps<af::ascir_op::Data>(input_producer)) {
    GE_ASSERT_SUCCESS(BuildSimdInputInnerAxis(graph, input_producer, axis_index, input_inner_axis));
  }
  for (const auto &node : paths.data_input) {
    if (af::ops::IsOps<af::ascir_op::Data>(node)) {
      for (const auto &output : node->outputs()) {
        std::copy_n(output_axes.begin(), axis_index, output->attr.axis.begin());
      }
      break;
    }
    if (ascgen_utils::indirect_load::GetTemplateRole(node) !=
        ascgen_utils::indirect_load::TemplateRole::kSimdInputPre) {
      break;
    }
    std::copy_n(output_axes.begin(), axis_index, node->attr.sched.axis.begin());
    for (const auto &output : node->outputs()) {
      std::copy_n(output_axes.begin(), axis_index, output->attr.axis.begin());
    }
  }
  return NormalizeAxesForTemplate(graph, indirect_load, axis, rank, input_inner_axis);
}

af::Status NormalizeSimtAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load) {
  const auto output_axes = indirect_load->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(!output_axes.empty(), "IndirectLoad SIMT output axis is empty.");
  const int64_t rank = static_cast<int64_t>(output_axes.size());
  return NormalizeAxesForTemplate(graph, indirect_load, rank, rank, af::kIdNone);
}

bool CanEmitSimtScalar(const af::AscNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  const auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
  const auto *v2_impl = impl == nullptr ? nullptr : dynamic_cast<af::ascir::AscIrCodegenV2 *>(impl.get());
  return v2_impl != nullptr && v2_impl->IsSimtScalarSupported(*node);
}

bool CollectMovableSimtInputPreNodes(const af::AscNodePtr &indirect_load, const NodePath &data_input, NodePath &nodes) {
  af::AscNodePtr consumer = indirect_load;
  for (const auto &current : data_input) {
    if (af::ops::IsOps<af::ascir_op::Data>(current) || af::ops::IsOps<af::ascir_op::Load>(current)) {
      return true;
    }
    if (current->inputs.Size() != 1UL || current->outputs().empty() || !CanEmitSimtScalar(current) ||
        current->GetInControlNodesSize() != 0UL || current->GetOutControlNodesSize() != 0UL ||
        ascgen_utils::indirect_load::GetOnlyOutputConsumer(current) != consumer) {
      return false;
    }
    nodes.emplace_back(current);
    consumer = current;
  }
  return false;
}

bool IsSimdInputShapeEnlarged(const af::AscNodePtr &indirect_load) {
  af::Expression input_numel = af::sym::kSymbolOne;
  for (const af::Expression &repeat : indirect_load->inputs()[0]->attr.repeats) {
    input_numel = input_numel * repeat;
  }
  af::Expression output_numel = af::sym::kSymbolOne;
  for (const af::Expression &repeat : indirect_load->outputs()[0]->attr.repeats) {
    output_numel = output_numel * repeat;
  }
  return af::SymbolicUtils::StaticCheckGt(output_numel, input_numel) == af::TriBool::kTrue;
}

void CollectMovableSimdInputPreNodes(const af::AscNodePtr &indirect_load, const NodePath &data_input, NodePath &nodes) {
  af::AscNodePtr consumer = indirect_load;
  for (const auto &current : data_input) {
    if (af::ops::IsOps<af::ascir_op::Data>(current) || af::ops::IsOps<af::ascir_op::Load>(current) ||
        current->inputs.Size() != 1UL || current->outputs().size() != 1UL ||
        current->attr.api.compute_type != af::ComputeType::kComputeElewise || current->GetInControlNodesSize() != 0UL ||
        current->GetOutControlNodesSize() != 0UL ||
        ascgen_utils::indirect_load::GetOnlyOutputConsumer(current) != consumer) {
      return;
    }
    nodes.emplace_back(current);
    consumer = current;
  }
}

af::Status MoveInputPreNode(const af::AscNodePtr &node, const af::AscNodePtr &indirect_load) {
  const auto producer = ascgen_utils::indirect_load::GetInputProducer(node, 0UL);
  GE_ASSERT_NOTNULL(producer);
  const auto producer_out = producer->GetOutDataAnchor(0UL);
  const auto node_in = node->GetInDataAnchor(0UL);
  const auto node_out = node->GetOutDataAnchor(0UL);
  const auto indirect_in = indirect_load->GetInDataAnchor(0UL);
  const auto indirect_out = indirect_load->GetOutDataAnchor(0UL);
  GE_ASSERT_NOTNULL(producer_out);
  GE_ASSERT_NOTNULL(node_in);
  GE_ASSERT_NOTNULL(node_out);
  GE_ASSERT_NOTNULL(indirect_in);
  GE_ASSERT_NOTNULL(indirect_out);
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(node_out, indirect_in, producer_out));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(producer_out, node_in, indirect_out));
  const auto peer_inputs = indirect_out->GetPeerInDataAnchors();
  for (const auto &peer_in : peer_inputs) {
    if (peer_in != node_in) {
      GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(indirect_out, peer_in, node_out));
    }
  }

  const auto indirect_output = indirect_load->outputs()[0];
  const auto node_output = node->outputs()[0];
  node->attr.sched.axis = indirect_load->attr.sched.axis;
  node_output->attr.axis = indirect_output->attr.axis;
  node_output->attr.repeats = indirect_output->attr.repeats;
  node_output->attr.strides = indirect_output->attr.strides;
  indirect_output->attr.dtype = producer->outputs()[0]->attr.dtype;
  GELOGD("[IndirectLoad] Move input pre node[%s] after IndirectLoad[%s].", node->GetNamePtr(),
         indirect_load->GetNamePtr());
  return af::SUCCESS;
}

af::Status MoveSimtInputPreNodes(const af::AscNodePtr &indirect_load, IndirectLoadGraphPaths &paths) {
  NodePath nodes;
  if (!CollectMovableSimtInputPreNodes(indirect_load, paths.data_input, nodes) ||
      (!nodes.empty() &&
       (indirect_load->GetInControlNodesSize() != 0UL || indirect_load->GetOutControlNodesSize() != 0UL)) ||
      ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load) == nullptr) {
    return af::SUCCESS;
  }
  for (const auto &node : nodes) {
    GE_ASSERT_SUCCESS(MoveInputPreNode(node, indirect_load));
  }
  paths.data_input.erase(paths.data_input.begin(), paths.data_input.begin() + static_cast<int64_t>(nodes.size()));
  paths.output.insert(paths.output.begin(), nodes.rbegin(), nodes.rend());
  return af::SUCCESS;
}

af::Status MoveSimdInputPreNodes(const af::AscNodePtr &indirect_load, IndirectLoadGraphPaths &paths) {
  if (IsSimdInputShapeEnlarged(indirect_load) || indirect_load->GetInControlNodesSize() != 0UL ||
      indirect_load->GetOutControlNodesSize() != 0UL ||
      ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load) == nullptr) {
    return af::SUCCESS;
  }
  NodePath nodes;
  CollectMovableSimdInputPreNodes(indirect_load, paths.data_input, nodes);
  for (const auto &node : nodes) {
    GE_ASSERT_SUCCESS(MoveInputPreNode(node, indirect_load));
  }
  paths.data_input.erase(paths.data_input.begin(), paths.data_input.begin() + static_cast<int64_t>(nodes.size()));
  paths.output.insert(paths.output.begin(), nodes.rbegin(), nodes.rend());
  return af::SUCCESS;
}

af::Status PropagateInputTensorAttrs(const IndirectLoadGraphPaths &paths) {
  for (const NodePath *path : {&paths.data_input, &paths.index_input}) {
    for (size_t i = 0UL; i < path->size(); ++i) {
      const auto &load = (*path)[i];
      if (!af::ops::IsOps<af::ascir_op::Load>(load)) {
        continue;
      }
      const af::AscNodePtr data = i + 1UL < path->size() ? (*path)[i + 1UL] : nullptr;
      if (data == nullptr || !af::ops::IsOps<af::ascir_op::Data>(data) || load->outputs().empty() ||
          data->outputs().empty()) {
        break;
      }
      const auto load_out = load->outputs()[0];
      const auto data_out = data->outputs()[0];
      if (data_out->attr.axis.empty() && !load_out->attr.axis.empty()) {
        GELOGD("[IndirectLoad] Propagate tensor attrs from Load[%s] to Data[%s].", load->GetNamePtr(),
               data->GetNamePtr());
        data_out->attr.axis = load_out->attr.axis;
        data_out->attr.repeats = load_out->attr.repeats;
        data_out->attr.strides = load_out->attr.strides;
        data_out->attr.dtype = load_out->attr.dtype;
      }
      break;
    }
  }
  return af::SUCCESS;
}

af::Status RecordTemplateLogicalView(const af::AscNodePtr &indirect_load) {
  GE_ASSERT_TRUE(indirect_load->inputs.Size() == 2UL && indirect_load->outputs().size() == 1UL,
                 "IndirectLoad expects 2 inputs and 1 output.");
  ascgen_utils::indirect_load::TemplateLogicalView view;
  view.data.axis_ids = indirect_load->inputs()[0]->attr.axis;
  view.data.strides = indirect_load->inputs()[0]->attr.strides;
  view.index.axis_ids = indirect_load->inputs()[1]->attr.axis;
  view.index.strides = indirect_load->inputs()[1]->attr.strides;
  view.output.axis_ids = indirect_load->outputs()[0]->attr.axis;
  view.output.strides = indirect_load->outputs()[0]->attr.strides;
  return ascgen_utils::indirect_load::SetTemplateLogicalView(indirect_load, view);
}

af::Status PrepareCandidateGraph(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                                 IndirectLoadGraphPaths &paths, bool &is_candidate_legal) {
  paths = CollectGraphPaths(indirect_load);
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    GE_ASSERT_SUCCESS(MoveSimdInputPreNodes(indirect_load, paths));
  } else if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    GE_ASSERT_SUCCESS(MoveSimtInputPreNodes(indirect_load, paths));
  }
  GE_ASSERT_SUCCESS(ValidateTemplateCandidate(template_id, paths, is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(PropagateInputTensorAttrs(paths));
  GE_ASSERT_SUCCESS(RecordTemplateLogicalView(indirect_load));
  return af::SUCCESS;
}

af::Status AnnotateSimdTemplateRoles(const af::AscNodePtr &indirect_load, const IndirectLoadGraphPaths &paths) {
  GE_ASSERT_NOTNULL(indirect_load);
  for (const auto &node : paths.data_input) {
    if (af::ops::IsOps<af::ascir_op::Data>(node)) {
      break;
    }
    if (af::ops::IsOps<af::ascir_op::VectorFunc>(node)) {
      if (ascgen_utils::indirect_load::GetOnlyOutputConsumer(node).get() == indirect_load.get()) {
        GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
            node, ascgen_utils::indirect_load::TemplateRole::kSimdInputPre));
      }
      break;
    }
    if (node->inputs.Size() != 1UL || node->outputs().size() != 1UL) {
      break;
    }
    GE_ASSERT_SUCCESS(
        ascgen_utils::indirect_load::SetTemplateRole(node, ascgen_utils::indirect_load::TemplateRole::kSimdInputPre));
  }
  return af::SUCCESS;
}

af::Status AnnotateSimtTransformRoles(const NodePath &path, const std::string &terminal_type,
                                      const std::string &direct_gm_boundary_type, bool &is_candidate_legal) {
  is_candidate_legal = false;
  for (const auto &node : path) {
    if (node->GetType() == direct_gm_boundary_type) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
          node, ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary));
      is_candidate_legal = true;
      return af::SUCCESS;
    }
    if (node->GetType() == terminal_type) {
      is_candidate_legal = true;
      return af::SUCCESS;
    }
    if (af::ops::IsOps<af::ascir_op::VectorFunc>(node) || !CanEmitSimtScalar(node)) {
      return af::SUCCESS;
    }
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
        node, ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform));
  }
  return af::SUCCESS;
}

af::Status AnnotateSimtTemplateRoles(const af::AscNodePtr &indirect_load, const IndirectLoadGraphPaths &paths,
                                     bool &is_candidate_legal) {
  GE_ASSERT_NOTNULL(indirect_load);
  is_candidate_legal = false;
  GE_ASSERT_SUCCESS(
      ascgen_utils::indirect_load::SetTemplateRole(indirect_load, ascgen_utils::indirect_load::TemplateRole::kSimtOp));
  GE_ASSERT_SUCCESS(AnnotateSimtTransformRoles(paths.index_input, af::ascir_op::Data::Type, af::ascir_op::Load::Type,
                                               is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(AnnotateSimtTransformRoles(paths.output, af::ascir_op::Output::Type, af::ascir_op::Store::Type,
                                               is_candidate_legal));
  return af::SUCCESS;
}

af::Status ApplySimdGraphPass(af::AscGraph &graph, const af::AscNodePtr &indirect_load, bool &is_candidate_legal) {
  IndirectLoadGraphPaths paths;
  GE_ASSERT_SUCCESS(
      PrepareCandidateGraph(indirect_load, ascir::TemplateId::kIndirectLoadSimd, paths, is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(AnnotateSimdTemplateRoles(indirect_load, paths));
  GE_ASSERT_SUCCESS(NormalizeSimdAxesForTemplate(graph, indirect_load, paths));
  GE_ASSERT_SUCCESS(::ascir::SetTemplateId(indirect_load, ascir::TemplateId::kIndirectLoadSimd));
  return af::SUCCESS;
}

af::Status ApplySimtGraphPass(af::AscGraph &graph, const af::AscNodePtr &indirect_load, bool &is_candidate_legal) {
  GELOGD("[IndirectLoad] Apply SIMT graph pass for node[%s], dcache_size[%ld].", indirect_load->GetNamePtr(),
         kIndirectLoadSimtDcacheSize);
  IndirectLoadGraphPaths paths;
  GE_ASSERT_SUCCESS(
      PrepareCandidateGraph(indirect_load, ascir::TemplateId::kIndirectLoadSimt, paths, is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(AnnotateSimtTemplateRoles(indirect_load, paths, is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(::ascir::SetDcacheSize(indirect_load, kIndirectLoadSimtDcacheSize));
  GE_ASSERT_SUCCESS(NormalizeSimtAxesForTemplate(graph, indirect_load));
  GE_ASSERT_SUCCESS(::ascir::SetTemplateId(indirect_load, ascir::TemplateId::kIndirectLoadSimt));
  return af::SUCCESS;
}

af::Status ApplyGraphPass(af::AscGraph &graph, const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                          bool &is_candidate_legal) {
  GELOGD("[IndirectLoad] Apply graph pass for node[%s], template_id[%d].", indirect_load->GetNamePtr(),
         static_cast<int32_t>(template_id));
  is_candidate_legal = false;
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return ApplySimdGraphPass(graph, indirect_load, is_candidate_legal);
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    return ApplySimtGraphPass(graph, indirect_load, is_candidate_legal);
  }
  return af::SUCCESS;
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
  const bool prefer_simd = HasExp2(graph);
  for (ascir::TemplateId template_id : {ascir::TemplateId::kIndirectLoadSimd, ascir::TemplateId::kIndirectLoadSimt}) {
    ascir::ImplGraph candidate_graph(graph.GetName().c_str());
    GE_ASSERT_TRUE(candidate_graph.CopyFrom(graph), "Failed to copy graph [%s].", graph.GetName().c_str());
    const af::AscNodePtr candidate_indirect_load = candidate_graph.FindNode(indirect_load_name.c_str());
    GE_ASSERT_NOTNULL(candidate_indirect_load, "Failed to find copied IndirectLoad node[%s].",
                      indirect_load_name.c_str());
    bool is_candidate_legal = false;
    GE_ASSERT_SUCCESS(ApplyGraphPass(candidate_graph, candidate_indirect_load, template_id, is_candidate_legal));
    if (!is_candidate_legal) {
      GELOGW("[IndirectLoad] Skip illegal template candidate[%d] for node[%s].", static_cast<int32_t>(template_id),
             candidate_indirect_load->GetNamePtr());
      continue;
    }
    graphs.emplace_back(std::move(candidate_graph));
    score_functions.emplace_back(GenerateScoreFunc(template_id, prefer_simd));
    GELOGI("[IndirectLoad] Add schedule candidate[%d] for node[%s].", static_cast<int32_t>(template_id),
           candidate_indirect_load->GetNamePtr());
  }
  return af::SUCCESS;
}

}  // namespace optimize
