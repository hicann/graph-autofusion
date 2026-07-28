/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reg_indirect_load_api_call.h"

#include <map>
#include <sstream>
#include "api_call/utils/api_call_factory.h"
#include "asc_graph_utils.h"
#include "ascir_ops.h"
#include "common/checker.h"
#include "common_utils.h"
#include "indirect_load_utils.h"
#include "schedule_result.h"
#include "v35/codegen/simt_scalar_call/simt_scalar_emitter.h"

namespace codegen {
namespace {
struct LogicalTensorInfo {
  std::vector<ascir::SizeExpr> sizes;
  std::vector<ascir::SizeExpr> strides;
};

constexpr uint32_t kIndirectLoadSimtThreadNum = 1024;

bool IsAxisDerivedFrom(const TPipe &tpipe, ascir::AxisId axis_id, ascir::AxisId ancestor_axis_id) {
  if (axis_id == ancestor_axis_id) {
    return true;
  }
  for (ascir::AxisId from_axis_id : tpipe.tiler.GetAxis(axis_id).from) {
    if (IsAxisDerivedFrom(tpipe, from_axis_id, ancestor_axis_id)) {
      return true;
    }
  }
  return false;
}

LogicalTensorInfo BuildLogicalTensorInfo(const ascgen_utils::indirect_load::LogicalTensorView &view,
                                         const TPipe &tpipe) {
  LogicalTensorInfo info;
  info.strides = view.strides;
  for (ascir::AxisId axis_id : view.axis_ids) {
    info.sizes.emplace_back(tpipe.tiler.GetAxis(axis_id).size);
  }
  return info;
}

std::string JoinSizeExprs(const std::vector<ascir::SizeExpr> &exprs, const TPipe &tpipe) {
  std::stringstream ss;
  for (size_t i = 0; i < exprs.size(); ++i) {
    if (i > 0) {
      ss << ", ";
    }
    ss << tpipe.tiler.Size(exprs[i]);
  }
  return ss.str();
}

bool FindCurrentAxisVar(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, Axis::Type axis_type,
                        ascir::AxisId ancestor_axis_id, std::string &axis_var) {
  for (ascir::AxisId axis_id : current_axis) {
    const Axis &axis = tpipe.tiler.GetAxis(axis_id);
    if (axis.type == axis_type && IsAxisDerivedFrom(tpipe, axis_id, ancestor_axis_id)) {
      axis_var = axis.Str();
      return true;
    }
  }
  return false;
}

af::Status CheckDenseStrides(const LogicalTensorInfo &tensor) {
  // Current offset generation assumes compact row-major layout. Views such as transpose, slice with gaps,
  // broadcast stride-0, or padded layouts can be supported later by generating stride-aware offsets.
  af::Expression expected_stride = af::ops::One;
  for (int64_t i = static_cast<int64_t>(tensor.sizes.size()) - 1; i >= 0; --i) {
    GE_ASSERT_TRUE(af::SymbolicUtils::StaticCheckEq(tensor.strides[i], expected_stride) == af::TriBool::kTrue,
                   "IndirectLoad only supports dense contiguous tensors.");
    expected_stride = af::sym::Mul(expected_stride, tensor.sizes[i]);
  }
  return af::SUCCESS;
}

af::Status CheckIndirectLoadShape(const LogicalTensorInfo &x_info, const LogicalTensorInfo &index_info,
                                  const LogicalTensorInfo &output_info) {
  GE_ASSERT_TRUE(x_info.sizes.size() == index_info.sizes.size(),
                 "IndirectLoad expects input and index with the same logical rank, input rank %zu, index rank %zu.",
                 x_info.sizes.size(), index_info.sizes.size());
  GE_ASSERT_TRUE(index_info.sizes.size() == output_info.sizes.size(),
                 "IndirectLoad index and output must have the same logical rank.");
  GE_ASSERT_SUCCESS(CheckDenseStrides(x_info));
  GE_ASSERT_SUCCESS(CheckDenseStrides(index_info));
  GE_ASSERT_SUCCESS(CheckDenseStrides(output_info));
  return af::SUCCESS;
}

af::Status CollectVectorFuncNodes(const af::AscNodePtr &vf_node, std::vector<af::AscNodePtr> &nodes) {
  if (vf_node == nullptr || !af::ops::IsOps<af::ascir_op::VectorFunc>(vf_node)) {
    return af::SUCCESS;
  }
  const std::string *graph_name = af::AttrUtils::GetStr(vf_node->GetOpDescBarePtr(), "sub_graph_name");
  GE_ASSERT_NOTNULL(graph_name, "Get sub graph name failed, vf node:%s", vf_node->GetNamePtr());
  af::AscGraph owner_graph("owner_graph");
  GE_ASSERT_SUCCESS(af::AscGraphUtils::FromComputeGraph(vf_node->GetOwnerComputeGraph(), owner_graph),
                    "Get owner graph failed, vf node:%s", vf_node->GetNamePtr());
  af::AscGraph sub_graph("vf_sub_graph");
  GE_ASSERT_SUCCESS(owner_graph.FindSubGraph(*graph_name, sub_graph), "Get sub graph failed, vf node:%s",
                    vf_node->GetNamePtr());
  for (const af::AscNodePtr &sub_node : sub_graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Data>(sub_node) || af::ops::IsOps<af::ascir_op::Load>(sub_node) ||
        af::ops::IsOps<af::ascir_op::Store>(sub_node) || af::ops::IsOps<af::ascir_op::Output>(sub_node)) {
      continue;
    }
    nodes.emplace_back(sub_node);
  }
  return af::SUCCESS;
}

af::Status CollectSimtInlineTransformNode(const af::AscNodePtr &node, std::vector<af::AscNodePtr> &nodes) {
  GE_ASSERT_TRUE(ascgen_utils::indirect_load::GetTemplateRole(node) ==
                     ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform,
                 "IndirectLoad SIMT transform node has no inline-transform role.");
  const size_t node_count = nodes.size();
  GE_ASSERT_SUCCESS(CollectVectorFuncNodes(node, nodes));
  if (nodes.size() != node_count) {
    return af::SUCCESS;
  }
  GE_ASSERT_TRUE(node->GetInDataNodesSize() == 1UL, "IndirectLoad SIMT transform node must have exactly one input.");
  nodes.emplace_back(node);
  return af::SUCCESS;
}

af::Status CollectSimtIndexPreNodes(const ascir::NodeView &node, std::vector<af::AscNodePtr> &nodes) {
  auto index_anchor = node->GetInDataAnchor(1UL);
  GE_ASSERT_NOTNULL(index_anchor, "IndirectLoad expects index input anchor.");
  auto peer_out_anchor = index_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor, "IndirectLoad index input anchor has no peer output.");
  auto current = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
  GE_ASSERT_SUCCESS(CollectVectorFuncNodes(current, nodes));
  if (!nodes.empty()) {
    return af::SUCCESS;
  }
  std::vector<af::AscNodePtr> reversed_nodes;
  while (current != nullptr && !af::ops::IsOps<af::ascir_op::Data>(current)) {
    if (af::ops::IsOps<af::ascir_op::Load>(current)) {
      break;
    }
    GE_ASSERT_SUCCESS(CollectSimtInlineTransformNode(current, reversed_nodes));
    auto prev_nodes = current->GetInDataNodes();
    current = std::dynamic_pointer_cast<af::AscNode>(*prev_nodes.begin());
  }
  nodes.assign(reversed_nodes.rbegin(), reversed_nodes.rend());
  return af::SUCCESS;
}

af::Status CollectSimtOutputMetadata(const ascir::NodeView &node, std::vector<af::AscNodePtr> &nodes,
                                     std::string &output_gm_tensor, af::DataType &output_dtype) {
  for (af::AscNodePtr current = ascgen_utils::indirect_load::GetOnlyOutputConsumer(node); current != nullptr;
       current = ascgen_utils::indirect_load::GetOnlyOutputConsumer(current)) {
    if (af::ops::IsOps<af::ascir_op::Store>(current)) {
      GE_ASSERT_TRUE(!current->outputs().empty(), "IndirectLoad SIMT Store has no output tensor.");
      output_gm_tensor = "global_" + std::to_string(current->outputs()[0]->attr.mem.tensor_id);
      output_dtype = current->outputs()[0]->attr.dtype;
      return af::SUCCESS;
    }
    GE_ASSERT_SUCCESS(CollectSimtInlineTransformNode(current, nodes));
  }
  GELOGE(af::FAILED, "IndirectLoad SIMT output chain must end with a Store node.");
  return af::FAILED;
}

struct SimdCodegenParams {
  const Tensor &x;
  const Tensor &index;
  const Tensor &y;
  af::AscNodePtr input_pre_vf;
  size_t axis_pos;
  std::string x_dtype_name;
  std::string y_dtype_name;
  std::string tmp_buf_name;
  std::string output_offset_expr;
  std::string index_axis_expr;
  std::string input_axis_expr;
  std::string index_inner_expr;
  std::string input_inner_expr;
  std::string index_outer_sizes_expr;
  std::string input_outer_strides_expr;
};

void EmitSimdWindowSetup(const SimdCodegenParams &params, std::stringstream &ss) {
  ss << "  LocalTensor<uint8_t> indirect_load_tmp = " << params.tmp_buf_name << ";" << std::endl;
  ss << "  LocalTensor<" << params.x_dtype_name << "> input_used = indirect_load_tmp.template ReinterpretCast<"
     << params.x_dtype_name << ">();" << std::endl;
  ss << "  int64_t output_offset = " << params.output_offset_expr << ";" << std::endl;
  ss << "  int64_t index_axis = " << params.index_axis_expr << ";" << std::endl;
  ss << "  int64_t input_axis = " << params.input_axis_expr << ";" << std::endl;
  ss << "  int64_t index_inner = " << params.index_inner_expr << ";" << std::endl;
  ss << "  int64_t input_inner = " << params.input_inner_expr << ";" << std::endl;
  ss << "  int64_t input_slice_count = input_axis * input_inner;" << std::endl;
  ss << "  int64_t output_slice_count = index_axis * index_inner;" << std::endl;
  ss << "  int64_t outer_begin = output_offset / output_slice_count;" << std::endl;
  ss << "  int64_t outer_end = (output_offset + " << params.y.actual_size
     << " + output_slice_count - 1) / output_slice_count;" << std::endl;
  ss << "  int64_t outer_count = outer_end - outer_begin;" << std::endl;
  ss << "  int64_t input_window_count = outer_count * input_slice_count;" << std::endl;
  ss << "  constexpr int64_t indirect_load_block_bytes = 32;" << std::endl;
  ss << "  int64_t input_bytes = (input_window_count * sizeof(" << params.x_dtype_name
     << ") + indirect_load_block_bytes - 1) / indirect_load_block_bytes * indirect_load_block_bytes;" << std::endl;
  if (params.input_pre_vf != nullptr) {
    ss << "  int64_t input_pre_bytes = (input_window_count * sizeof(" << params.y_dtype_name
       << ") + indirect_load_block_bytes - 1) / indirect_load_block_bytes * indirect_load_block_bytes;" << std::endl;
  }
}

void EmitSimdInputWindow(const SimdCodegenParams &params, std::stringstream &ss) {
  if (params.axis_pos == 0UL) {
    ss << "  DataCopyPadExtend<" << params.x_dtype_name << ", AscendC::PaddingMode::Normal>(input_used[0], " << params.x
       << "[0], 1, input_window_count, 0, 0);" << std::endl;
  } else {
    ss << "  int64_t index_sizes[] = {" << params.index_outer_sizes_expr << "};" << std::endl;
    ss << "  int64_t input_outer_strides[] = {" << params.input_outer_strides_expr << "};" << std::endl;
    ss << "  for (int64_t outer_local = 0; outer_local < outer_count; ++outer_local) {" << std::endl;
    ss << "    int64_t outer_linear = outer_begin + outer_local;" << std::endl;
    ss << "    int64_t input_window_offset = 0;" << std::endl;
    ss << "    for (int64_t axis = " << params.axis_pos << " - 1; axis >= 0; --axis) {" << std::endl;
    ss << "      int64_t coord = outer_linear % index_sizes[axis];" << std::endl;
    ss << "      outer_linear = outer_linear / index_sizes[axis];" << std::endl;
    ss << "      input_window_offset += coord * input_outer_strides[axis];" << std::endl;
    ss << "    }" << std::endl;
    ss << "    DataCopyPadExtend<" << params.x_dtype_name << ", AscendC::PaddingMode::Normal>("
       << "input_used[outer_local * input_slice_count], " << params.x
       << "[input_window_offset], 1, input_slice_count, 0, 0);" << std::endl;
    ss << "  }" << std::endl;
  }
  if (params.input_pre_vf == nullptr) {
    ss << "  LocalTensor<" << params.x_dtype_name << "> input_pre_used = input_used;" << std::endl;
    return;
  }
  ss << "  LocalTensor<uint8_t> input_pre_buf = indirect_load_tmp[input_bytes];" << std::endl;
  ss << "  LocalTensor<" << params.y_dtype_name << "> input_pre_used = input_pre_buf.template ReinterpretCast<"
     << params.y_dtype_name << ">();" << std::endl;
  ss << "  VFCall" << params.input_pre_vf->GetName() << "((__local_mem__ " << params.y_dtype_name
     << " *)input_pre_used[0].GetPhyAddr(), (__local_mem__ " << params.x_dtype_name
     << " *)input_used[0].GetPhyAddr(), static_cast<uint32_t>(input_window_count));" << std::endl;
}

void EmitSimdGather(const SimdCodegenParams &params, std::stringstream &ss) {
  const bool has_input_pre = params.input_pre_vf != nullptr;
  ss << "  LocalTensor<uint8_t> offset_buf = indirect_load_tmp["
     << (has_input_pre ? "input_bytes + input_pre_bytes" : "input_bytes") << "];" << std::endl;
  ss << "  auto indirect_load_offset = offset_buf.template ReinterpretCast<uint32_t>();" << std::endl;
  ss << "  for (int64_t i = 0; i < " << params.y.actual_size << "; ++i) {" << std::endl;
  ss << "    int64_t global_idx = output_offset + i;" << std::endl;
  ss << "    int64_t outer_local = global_idx / output_slice_count - outer_begin;" << std::endl;
  ss << "    int64_t index_val = static_cast<int64_t>(" << params.index << ".GetValue(i));" << std::endl;
  ss << "    int64_t inner = global_idx % index_inner;" << std::endl;
  ss << "    int64_t src_idx = outer_local * input_slice_count + index_val * input_inner + inner;" << std::endl;
  ss << "    indirect_load_offset.SetValue(i, static_cast<uint32_t>(src_idx * sizeof("
     << (has_input_pre ? params.y_dtype_name : params.x_dtype_name) << ")));" << std::endl;
  ss << "  }" << std::endl;
  ss << "  AscendC::Gather(" << params.y << ", input_pre_used, indirect_load_offset, static_cast<uint32_t>(0), "
     << params.y.actual_size << ");" << std::endl;
}

af::Status GenerateTypedTransform(const std::string &name, const std::string &input_dtype,
                                  const std::vector<af::AscNodePtr> &nodes, std::stringstream &ss) {
  std::string return_dtype = input_dtype;
  if (!nodes.empty()) {
    GE_ASSERT_TRUE(!nodes.back()->outputs().empty(), "SIMT scalar transform node[%s] has no output.",
                   nodes.back()->GetNamePtr());
    GE_ASSERT_SUCCESS(Tensor::DtypeName(nodes.back()->outputs()[0]->attr.dtype, return_dtype));
  }
  ss << "struct " << name << " {" << std::endl;
  ss << "  __simt_callee__ __aicore__ inline static " << return_dtype << " Call(" << input_dtype << " value) {"
     << std::endl;
  if (nodes.empty()) {
    ss << "    return value;" << std::endl;
  }
  for (size_t i = 0; i < nodes.size(); ++i) {
    std::string expr;
    const std::vector<std::string> inputs = {i == 0UL ? "value" : "v" + std::to_string(i - 1UL)};
    GE_ASSERT_SUCCESS(EmitSimtScalarExpr(nodes[i], inputs, expr));
    if (i + 1UL == nodes.size()) {
      ss << "    return " << expr << ";" << std::endl;
      continue;
    }
    GE_ASSERT_TRUE(!nodes[i]->outputs().empty(), "SIMT scalar transform node[%s] has no output.",
                   nodes[i]->GetNamePtr());
    std::string output_dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(nodes[i]->outputs()[0]->attr.dtype, output_dtype));
    ss << "    " << output_dtype << " v" << i << " = " << expr << ";" << std::endl;
  }
  ss << "  }" << std::endl;
  ss << "};" << std::endl;
  return af::SUCCESS;
}

struct SimtKernelDefinitionParams {
  const std::string &kernel_name;
  const std::string &x_dtype_name;
  const std::string &index_dtype_name;
  const std::string &output_dtype_name;
  const std::string &index_transform;
  const std::string &output_transform;
  size_t rank;
  int64_t axis;
};

void EmitSimtKernelDefinition(const SimtKernelDefinitionParams &params, std::stringstream &ss) {
  ss << "__simt_vf__ __aicore__ LAUNCH_BOUND(" << kIndirectLoadSimtThreadNum << ") inline void " << params.kernel_name
     << "(__gm__ " << params.x_dtype_name << " *x, __gm__ " << params.index_dtype_name << " *index, __gm__ "
     << params.output_dtype_name << " *y, uint32_t actual_size, int64_t output_offset, int64_t x_axis_size";
  for (size_t i = 0; i < params.rank; ++i) {
    ss << ", int64_t index_size" << i;
  }
  for (size_t i = 0; i < params.rank; ++i) {
    ss << ", int64_t x_stride" << i;
  }
  ss << ") {" << std::endl;
  ss << "  for (uint32_t i = threadIdx.x; i < actual_size; i += blockDim.x) {" << std::endl;
  ss << "    int64_t output_index = output_offset + i;" << std::endl;
  ss << "    int64_t indirect_index = static_cast<int64_t>(" << params.index_transform
     << "::Call(index[output_index]));" << std::endl;
  ss << "    if (unlikely(indirect_index < 0 || indirect_index >= x_axis_size)) {" << std::endl;
  ss << "      y[output_index] = static_cast<" << params.output_dtype_name << ">(0);" << std::endl;
  ss << "      continue;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    int64_t linear_index = output_index;" << std::endl;
  ss << "    int64_t input_offset = 0;" << std::endl;
  for (int64_t axis = static_cast<int64_t>(params.rank) - 1; axis >= 0; --axis) {
    if (axis != params.axis) {
      ss << "    int64_t coord" << axis << " = linear_index % index_size" << axis << ";" << std::endl;
    }
    if (axis > 0) {
      ss << "    linear_index = linear_index / index_size" << axis << ";" << std::endl;
    }
    if (axis == params.axis) {
      ss << "    input_offset += indirect_index * x_stride" << axis << ";" << std::endl;
    } else {
      ss << "    input_offset += coord" << axis << " * x_stride" << axis << ";" << std::endl;
    }
  }
  ss << "    y[output_index] = " << params.output_transform << "::Call(x[input_offset]);" << std::endl;
  ss << "  }" << std::endl;
  ss << "}" << std::endl;
}

struct SimtCodegenParams {
  const Tensor &x;
  const Tensor &index;
  std::string output_gm_expr;
  std::string x_dtype_name;
  std::string index_dtype_name;
  std::string output_dtype_name;
  std::string kernel_name;
  std::string x_axis_size_expr;
  std::string rank_sizes_expr;
  std::string x_strides_expr;
  std::string outer_tb_var;
};

void EmitSimtLaunch(const SimtCodegenParams &params, std::stringstream &ss) {
  ss << "  __gm__ " << params.x_dtype_name << " *x_ptr = (__gm__ " << params.x_dtype_name << " *)" << params.x
     << ".GetPhyAddr();" << std::endl;
  ss << "  __gm__ " << params.index_dtype_name << " *index_ptr = (__gm__ " << params.index_dtype_name << " *)"
     << params.index << ".GetPhyAddr();" << std::endl;
  ss << "  __gm__ " << params.output_dtype_name << " *y_ptr = (__gm__ " << params.output_dtype_name << " *)"
     << params.output_gm_expr << ".GetPhyAddr();" << std::endl;
  ss << "  AscendC::Simt::VF_CALL<" << params.kernel_name << ">(AscendC::Simt::Dim3(" << kIndirectLoadSimtThreadNum
     << "), x_ptr, index_ptr, y_ptr, static_cast<uint32_t>(" << params.outer_tb_var << "_loop_size), block_dim_offset, "
     << params.x_axis_size_expr << params.rank_sizes_expr << params.x_strides_expr << ");" << std::endl;
  ss << "  int32_t event_id_v_to_mte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));"
     << std::endl;
  ss << "  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id_v_to_mte3);" << std::endl;
  ss << "  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id_v_to_mte3);" << std::endl;
}

}  // namespace

Status IndirectLoadRegApiCall::ParseAttr(const ascir::NodeView &node) {
  int64_t axis = 0;
  GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("axis", axis),
                          "Failed to get IndirectLoad axis attr, node = %s", node->GetNamePtr());
  ascgen_utils::indirect_load::TemplateAxes template_axes;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateAxes(node, template_axes));
  outer_axis_ = template_axes.outer_axis;
  template_id_ = ::ascir::GetTemplateIdOrDefault(*node, ascir::TemplateId::kIndirectLoadSimt);
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateLogicalView(node, logical_view_));
  const int64_t rank = static_cast<int64_t>(logical_view_.data.axis_ids.size());
  GE_ASSERT_TRUE(axis >= -rank && axis < rank, "IndirectLoad axis is out of range.");
  axis_ = axis < 0L ? axis + rank : axis;
  if (template_id_ == ascir::TemplateId::kIndirectLoadSimd) {
    const auto input_producer = ascgen_utils::indirect_load::GetInputProducer(node, 0UL);
    if (af::ops::IsOps<af::ascir_op::VectorFunc>(input_producer)) {
      GE_ASSERT_TRUE(ascgen_utils::indirect_load::GetTemplateRole(input_producer) ==
                         ascgen_utils::indirect_load::TemplateRole::kSimdInputPre,
                     "IndirectLoad SIMD input VectorFunc has no input-pre role.");
      input_pre_vf_ = input_producer;
    }
  } else {
    GE_ASSERT_SUCCESS(CollectSimtIndexPreNodes(node, index_pre_nodes_));
    GE_ASSERT_SUCCESS(CollectSimtOutputMetadata(node, output_post_nodes_, output_gm_tensor_, output_dtype_));
    GE_ASSERT_TRUE(!node->outputs().empty(), "IndirectLoad SIMT node has no output tensor.");
    GE_ASSERT_TRUE(output_post_nodes_.empty() || !output_post_nodes_.back()->outputs().empty(),
                   "IndirectLoad SIMT final output transform node has no output tensor.");
    af::DataType transform_dtype = node->outputs()[0]->attr.dtype;
    if (!output_post_nodes_.empty()) {
      transform_dtype = output_post_nodes_.back()->outputs()[0]->attr.dtype;
    }
    GE_ASSERT_TRUE(transform_dtype == output_dtype_,
                   "IndirectLoad SIMT output transform dtype[%d] does not match Store dtype[%d].",
                   static_cast<int32_t>(transform_dtype), static_cast<int32_t>(output_dtype_));
  }
  GELOGI("[IndirectLoad] Parse codegen attrs for node[%s], axis[%ld], template_id[%d].", node->GetNamePtr(), axis_,
         static_cast<int32_t>(template_id_));
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::GenerateFuncDefinition(const TPipe &tpipe, const Tiler &tiler,
                                                      std::stringstream &ss) const {
  (void)tiler;
  if (template_id_ != ascir::TemplateId::kIndirectLoadSimt) {
    return af::SUCCESS;
  }

  std::vector<std::reference_wrapper<const Tensor>> input_tensors;
  for (const auto &in : inputs) {
    auto tensor_ptr = tpipe.GetTensor(in->id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, af::FAILED, "Check[Param] tensor_ptr is nullptr");
    input_tensors.emplace_back(*tensor_ptr);
  }
  GE_ASSERT_TRUE(input_tensors.size() == 2U, "IndirectLoad SIMT expects 2 inputs.");
  const Tensor &x = input_tensors[0].get();
  const Tensor &index = input_tensors[1].get();
  const LogicalTensorInfo x_info = BuildLogicalTensorInfo(logical_view_.data, tpipe);
  const LogicalTensorInfo index_info = BuildLogicalTensorInfo(logical_view_.index, tpipe);
  GE_ASSERT_TRUE(x_info.sizes.size() == index_info.sizes.size(),
                 "IndirectLoad SIMT input and index rank must be equal.");
  std::string x_dtype;
  std::string index_dtype;
  std::string output_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(x.dtype, x_dtype));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(index.dtype, index_dtype));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(output_dtype_, output_dtype));
  const std::string valid_node_name = ascgen_utils::GenValidName(node_name);
  const std::string index_transform = "IndirectLoadIndexTransform_" + valid_node_name;
  const std::string output_transform = "IndirectLoadOutputTransform_" + valid_node_name;
  const std::string kernel_name =
      "IndirectLoadSimtKernel_" + ascgen_utils::GenValidName(graph_name) + "_" + valid_node_name;

  GELOGI(
      "[IndirectLoad] Generate SIMT wrapper for node[%s], rank[%zu], axis[%ld], index_pre_nodes[%zu], "
      "output_post_nodes[%zu], kernel[%s].",
      node_name.c_str(), x_info.sizes.size(), axis_, index_pre_nodes_.size(), output_post_nodes_.size(),
      kernel_name.c_str());

  GE_ASSERT_SUCCESS(GenerateTypedTransform(index_transform, index_dtype, index_pre_nodes_, ss));
  GE_ASSERT_SUCCESS(GenerateTypedTransform(output_transform, x_dtype, output_post_nodes_, ss));
  const SimtKernelDefinitionParams params{kernel_name,     x_dtype,          index_dtype,         output_dtype,
                                          index_transform, output_transform, x_info.sizes.size(), axis_};
  EmitSimtKernelDefinition(params, ss);
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                        const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                        const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                        std::string &result) const {
  GE_ASSERT_TRUE(inputs.size() == 2U && outputs.size() == 1U, "IndirectLoad expects 2 inputs and 1 output.");
  const LogicalTensorInfo x_info = BuildLogicalTensorInfo(logical_view_.data, tpipe);
  const LogicalTensorInfo index_info = BuildLogicalTensorInfo(logical_view_.index, tpipe);
  const LogicalTensorInfo output_info = BuildLogicalTensorInfo(logical_view_.output, tpipe);
  GE_ASSERT_SUCCESS(CheckIndirectLoadShape(x_info, index_info, output_info));
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs);

  if (template_id_ == ascir::TemplateId::kIndirectLoadSimd) {
    GELOGI("[IndirectLoad] Generate SIMD API body for node[%s].", node_name.c_str());
    return GenerateSimd(tpipe, current_axis, inputs, outputs, result);
  }
  GELOGI("[IndirectLoad] Generate SIMT API body for node[%s].", node_name.c_str());
  return GenerateSimt(tpipe, current_axis, inputs, result);
}

Status IndirectLoadRegApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                        std::string &result) const {
  if (template_id_ != ascir::TemplateId::kIndirectLoadSimt) {
    return ApiCall::Generate(tpipe, current_axis, result);
  }
  std::vector<std::reference_wrapper<const Tensor>> input_tensors;
  for (const auto &in : inputs) {
    auto tensor_ptr = tpipe.GetTensor(in->id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, af::FAILED, "Check[Param] tensor_ptr is nullptr");
    input_tensors.emplace_back(*tensor_ptr);
  }
  return GenerateSimt(tpipe, current_axis, input_tensors, result);
}

Status IndirectLoadRegApiCall::GenerateSimd(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                            const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                            const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                            std::string &result) const {
  const Tensor &x = inputs[0].get();
  const Tensor &index = inputs[1].get();
  const Tensor &y = outputs[0].get();
  const auto tmp_iter = tmp_buf_id.find(-1L);
  GE_ASSERT_TRUE(tmp_iter != tmp_buf_id.end(), "IndirectLoad SIMD requires an API-level tmp buffer.");

  const LogicalTensorInfo x_info = BuildLogicalTensorInfo(logical_view_.data, tpipe);
  const LogicalTensorInfo index_info = BuildLogicalTensorInfo(logical_view_.index, tpipe);
  const size_t axis_pos = static_cast<size_t>(axis_);
  std::string x_dtype_name;
  std::string y_dtype_name;
  Tensor::DtypeName(x.dtype, x_dtype_name);
  Tensor::DtypeName(y.dtype, y_dtype_name);
  const bool has_input_pre = input_pre_vf_ != nullptr;
  GE_ASSERT_TRUE(has_input_pre || x.dtype == y.dtype,
                 "IndirectLoad SIMD without input preprocess requires matching input/output dtype.");
  GELOGD("[IndirectLoad] Generate SIMD body for node[%s], rank[%zu], axis[%zu], has_input_pre[%d].", node_name.c_str(),
         x_info.sizes.size(), axis_pos, static_cast<int32_t>(has_input_pre));

  const SimdCodegenParams params{
      x,
      index,
      y,
      input_pre_vf_,
      axis_pos,
      x_dtype_name,
      y_dtype_name,
      tpipe.tmp_buf.name + "_" + std::to_string(tmp_iter->second),
      tpipe.tiler.Offset(current_axis, y.axis, y.axis_strides),
      tpipe.tiler.Size(index_info.sizes[axis_pos]),
      tpipe.tiler.Size(x_info.sizes[axis_pos]),
      tpipe.tiler.Size(index_info.strides[axis_pos]),
      tpipe.tiler.Size(x_info.strides[axis_pos]),
      JoinSizeExprs(std::vector<ascir::SizeExpr>(index_info.sizes.begin(), index_info.sizes.begin() + axis_pos), tpipe),
      JoinSizeExprs(std::vector<ascir::SizeExpr>(x_info.strides.begin(), x_info.strides.begin() + axis_pos), tpipe)};
  std::stringstream ss;
  ss << "// IndirectLoad SIMD" << std::endl;
  ss << "{" << std::endl;
  EmitSimdWindowSetup(params, ss);
  EmitSimdInputWindow(params, ss);
  EmitSimdGather(params, ss);
  ss << "}" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::GenerateSimt(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                            const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                            std::string &result) const {
  const Tensor &x = inputs[0].get();
  const Tensor &index = inputs[1].get();
  const LogicalTensorInfo x_info = BuildLogicalTensorInfo(logical_view_.data, tpipe);
  const LogicalTensorInfo index_info = BuildLogicalTensorInfo(logical_view_.index, tpipe);
  GELOGD("[IndirectLoad] Generate SIMT body for node[%s], output_gm[%s].", node_name.c_str(),
         output_gm_tensor_.c_str());

  const std::string x_axis_size = tpipe.tiler.Size(x_info.sizes[axis_]);

  std::string outer_tb_var;
  const bool has_outer_tb =
      FindCurrentAxisVar(tpipe, current_axis, Axis::Type::kAxisTypeBlockInner, outer_axis_, outer_tb_var);
  GE_ASSERT_TRUE(has_outer_tb, "IndirectLoad SIMT current axes must contain the output block-inner axis.");
  std::string x_dtype, index_dtype, output_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(x.dtype, x_dtype));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(index.dtype, index_dtype));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(output_dtype_, output_dtype));
  const std::string kernel_name =
      "IndirectLoadSimtKernel_" + ascgen_utils::GenValidName(graph_name) + "_" + ascgen_utils::GenValidName(node_name);
  const std::string rank_sizes = std::string(", ") + JoinSizeExprs(index_info.sizes, tpipe);
  const std::string x_strides = std::string(", ") + JoinSizeExprs(x_info.strides, tpipe);
  const SimtCodegenParams params{x,           index,       output_gm_tensor_, x_dtype,   index_dtype, output_dtype,
                                 kernel_name, x_axis_size, rank_sizes,        x_strides, outer_tb_var};
  std::stringstream ss;
  ss << "// IndirectLoad SIMT" << std::endl;
  ss << "{" << std::endl;
  EmitSimtLaunch(params, ss);
  ss << "}" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<IndirectLoadRegApiCall> register_indirect_load_reg_api_call("IndirectLoadRegApiCall");
}  // namespace codegen
