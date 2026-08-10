/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/cast_optimization_pass.h"

#include "asc_graph_utils.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "ascir_ops_utils.h"
#include "common_utils.h"
#include "graph_utils.h"
#include "schedule_utils.h"
#include "task_generator/concat_inputs_unification_pass.h"

#include <set>

namespace af::optimize {
namespace {
bool IsReverseCast(AscNode &asc_node, DataType src_dtype, DataType dst_dtype) {
  return (asc_node.inputs[0].attr.dtype == dst_dtype) && (asc_node.outputs[0].attr.dtype == src_dtype);
}

Status UpdateDtype(const AscNodePtr &node, DataType dtype) {
  node->outputs[0].attr.dtype = dtype;
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto output_desc = op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_desc);
  output_desc->SetDataType(dtype);
  for (uint32_t i = 0; i < op_desc->GetAllInputsSize(); ++i) {
    const auto input_desc = op_desc->MutableInputDesc(i);
    GE_ASSERT_NOTNULL(input_desc);
    input_desc->SetDataType(dtype);
  }
  return SUCCESS;
}

Expression GetColSize(const af::AscTensor &tensor, size_t concat_dim) {
  const auto &tensor_repeats = tensor.attr.repeats;
  Expression col_size = tensor_repeats[concat_dim];
  for (size_t i = concat_dim + 1; i < tensor_repeats.size(); ++i) {
    col_size = col_size * tensor_repeats[i];
  }
  return col_size;
}

bool TryRemoveOrBypassReverseCast(const ComputeGraphPtr &cg, const AscNode *src_node,
                                  const OutDataAnchorPtr &src_out_anchor, const InDataAnchorPtr &concat_in_anchor) {
  const auto cast_out_nodes = src_node->GetOutDataNodes();
  if (cast_out_nodes.size() <= 1UL) {
    GE_ASSERT_SUCCESS(GraphUtils::IsolateNode(src_out_anchor->GetOwnerNode(), {0}));
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(cg, src_out_anchor->GetOwnerNode()));
    GELOGD("input index = %d, Cast node was removed", concat_in_anchor->GetIdx());
    return true;
  }
  const auto cast_in_anchor = src_node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(cast_in_anchor);
  GE_ASSERT_SUCCESS(GraphUtils::RemoveEdge(src_out_anchor, concat_in_anchor));
  GE_ASSERT_SUCCESS(GraphUtils::AddEdge(cast_in_anchor, concat_in_anchor));
  GELOGD("input index = %d, bypassed reverse Cast with multiple consumers", concat_in_anchor->GetIdx());
  return true;
}
}  // namespace

bool CastOptimizationPass::MayCauseDegradation(const AscNodePtr &concat_node, int32_t src_dtype_size,
                                               int32_t dst_dtype_size) {
  // 如果对齐从4B下降到不足4B, 且无法使用Gather API, 则可能会引入劣化
  size_t concat_dim = 0UL;
  bool unused = false;
  const auto kAlignment = Symbol(sizeof(uint32_t));
  GE_ASSERT_SUCCESS(::optimize::ScheduleUtils::ResolveDiffDim(concat_node, concat_dim, unused));
  bool alignment_changed = false;
  for (uint32_t i = 0U; i < concat_node->inputs.Size(); ++i) {
    const auto &col_size_expr = GetColSize(concat_node->inputs[i], concat_dim);
    const auto src_aligned =
        SymbolicUtils::StaticCheckEq(sym::Mod(col_size_expr * Symbol(src_dtype_size), kAlignment), ops::Zero);
    const auto target_aligned =
        SymbolicUtils::StaticCheckEq(sym::Mod(col_size_expr * Symbol(dst_dtype_size), kAlignment), ops::Zero);
    if ((src_aligned != TriBool::kFalse) && (target_aligned != TriBool::kTrue)) {
      GELOGI("concat input[%u] col_size = %s, changing dtype size from %d to %d may cause alignment degradation", i,
             col_size_expr.Str().get(), src_dtype_size, dst_dtype_size);
      alignment_changed = true;
      break;
    }
  }
  if (alignment_changed && (!::optimize::ConcatInputUnificationPass::CanOptimize(concat_node, concat_dim))) {
    GELOGI("can not use Gather API");
    return true;
  }
  return false;
}

Status CastOptimizationPass::Run(AscGraph &graph, int32_t concat_alg) {
  for (const auto &node : graph.GetAllNodes()) {
    if (!ops::IsOps<ascir_op::Concat>(node)) {
      continue;
    }
    const auto out_nodes = node->GetOutDataNodes();
    if ((out_nodes.size() != 1UL) || (!ops::IsOps<ascir_op::Cast>(out_nodes.at(0)))) {
      continue;
    }
    auto out_cast_node = std::dynamic_pointer_cast<ge::AscNode>(out_nodes.at(0));
    GE_ASSERT_NOTNULL(out_cast_node);
    const auto src_dtype = out_cast_node->inputs[0].attr.dtype;
    const auto dst_dtype = out_cast_node->outputs[0].attr.dtype;
    if (NeedOptimize(node, src_dtype, dst_dtype, concat_alg)) {
      GE_ASSERT_SUCCESS(DoOptimize(graph, node, out_cast_node, src_dtype, dst_dtype));
      GELOGI("Cast nodes around Concat node: %s was optimized", node->GetNamePtr());
    }
  }
  return SUCCESS;
}

bool CastOptimizationPass::NeedOptimize(const AscNodePtr &node, DataType src_dtype, DataType dst_dtype,
                                        int32_t concat_alg) {
  const auto src_dtype_size = GetSizeByDataType(src_dtype);
  const auto dst_dtype_size = GetSizeByDataType(dst_dtype);
  if (dst_dtype_size >= src_dtype_size) {
    return false;
  }
  // dst_dtype_size < src_dtype_size (downcast)
  // for platform V2
  constexpr int32_t kConcatAlgGather = 1;
  if ((concat_alg == kConcatAlgGather) && MayCauseDegradation(node, src_dtype_size, dst_dtype_size)) {
    GELOGI("changing dtype of Concat node: %s may cause degradation, do not optimize", node->GetNamePtr());
    return false;
  }
  GELOGD("Cast from %s(size = %d) to %s(size = %d), need optimize",
         TypeUtils::DataTypeToSerialString(src_dtype).c_str(), src_dtype_size,
         TypeUtils::DataTypeToSerialString(dst_dtype).c_str(), dst_dtype_size);
  return true;
}

Status CastOptimizationPass::DoOptimize(AscGraph &graph, const AscNodePtr &node, const AscNodePtr &out_cast_node,
                                        DataType src_dtype, DataType dst_dtype) {
  const auto &cg = AscGraphUtils::GetComputeGraph(graph);
  std::map<af::OutDataAnchor *, AscNodePtr> out_anchor_to_cast_node;
  for (const auto &concat_in_anchor : node->GetAllInDataAnchors()) {
    const auto &src_out_anchor = concat_in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(src_out_anchor);
    const auto &src_node = dynamic_cast<AscNode *>(src_out_anchor->GetOwnerNodeBarePtr());
    GE_ASSERT_NOTNULL(src_node);
    if (ops::IsOps<ascir_op::Cast>(src_node) && IsReverseCast(*src_node, src_dtype, dst_dtype)) {
      if (TryRemoveOrBypassReverseCast(cg, src_node, src_out_anchor, concat_in_anchor)) {
        continue;
      }
    }
    if (src_node->outputs[0].attr.dtype == dst_dtype) {
      GELOGD("input index = %d, source dtype already matches dst_dtype, skip adding Cast", concat_in_anchor->GetIdx());
      continue;
    }
    const auto it = out_anchor_to_cast_node.find(src_out_anchor.get());
    if (it != out_anchor_to_cast_node.cend()) {
      GE_ASSERT_SUCCESS(GraphUtils::RemoveEdge(src_out_anchor, concat_in_anchor));
      GE_ASSERT_SUCCESS(GraphUtils::AddEdge(it->second->GetOutDataAnchor(0), concat_in_anchor));
      GELOGD("input index = %d, reuse existing Cast node for shared source", concat_in_anchor->GetIdx());
      continue;
    }
    ascir_op::Cast cast_op((src_node->GetName() + "_cast_optimization_pass").c_str());
    cast_op.attr = out_cast_node->attr;
    auto &src_node_output_tensor_attr = src_node->outputs[0].attr;
    cast_op.y.dtype = dst_dtype;
    const auto cast_node = graph.AddNode(cast_op);
    GE_ASSERT_NOTNULL(cast_node);
    out_anchor_to_cast_node[src_out_anchor.get()] = cast_node;
    GE_ASSERT_SUCCESS(AscGraphUtils::InsertNodeAfter(src_out_anchor, {concat_in_anchor}, cast_node));
    ::optimize::ScheduleUtils::GenerateStrides(src_node_output_tensor_attr.repeats, cast_node->outputs[0].attr.strides);
    GELOGD("input index = %d, new Cast node was added", concat_in_anchor->GetIdx());
  }
  GE_ASSERT_SUCCESS(UpdateDtype(node, dst_dtype));
  GE_ASSERT_SUCCESS(GraphUtils::IsolateNode(out_cast_node, {0}));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(cg, out_cast_node));
  GELOGD("Concat output cast node: %s was removed", out_cast_node->GetNamePtr());
  return SUCCESS;
}
}  // namespace af::optimize
