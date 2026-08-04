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

#include <sstream>
#include "api_call/utils/api_call_factory.h"
#include "ascir_ops.h"
#include "common/checker.h"
#include "indirect_load_utils.h"
#include "v35/codegen/simt_scalar_call/simt_scalar_emitter.h"

namespace codegen {
namespace {
struct LogicalTensorInfo {
  std::vector<ascir::SizeExpr> sizes;
  std::vector<ascir::SizeExpr> strides;
};

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

af::Status CollectSimtInlineTransformNode(const af::AscNodePtr &node, std::vector<af::AscNodePtr> &nodes) {
  GE_ASSERT_TRUE(ascgen_utils::indirect_load::GetTemplateRole(node) ==
                     ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform,
                 "IndirectLoad SIMT transform node has no inline-transform role.");
  GE_ASSERT_TRUE(!af::ops::IsOps<af::ascir_op::VectorFunc>(node),
                 "IndirectLoad SIMT transform must use scalar emission, node:%s", node->GetNamePtr());
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
}  // namespace

Status IndirectLoadRegApiCall::ParseAttr(const ascir::NodeView &node) {
  int64_t axis = 0;
  GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("axis", axis),
                          "Failed to get IndirectLoad axis attr, node = %s", node->GetNamePtr());
  ascgen_utils::indirect_load::TemplateAxes template_axes;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateAxes(node, template_axes));
  outer_axis_ = template_axes.outer_axis;
  template_id_ = ::ascir::GetTemplateIdOrDefault(*node);
  GE_ASSERT_TRUE(
      template_id_ == ascir::TemplateId::kIndirectLoadSK || template_id_ == ascir::TemplateId::kIndirectLoadSimd ||
          template_id_ == ascir::TemplateId::kIndirectLoadSimt,
      "IndirectLoad node[%s] has invalid template id[%d].", node->GetNamePtr(), static_cast<int32_t>(template_id_));
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateLogicalView(node, logical_view_));
  const int64_t rank = static_cast<int64_t>(logical_view_.data.axis_ids.size());
  GE_ASSERT_TRUE(axis >= -rank && axis < rank, "IndirectLoad axis is out of range.");
  axis_ = axis < 0L ? axis + rank : axis;
  if (template_id_ == ascir::TemplateId::kIndirectLoadSimt) {
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
  GE_ASSERT_SUCCESS(Tensor::DtypeName(x.dtype, x_dtype));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(index.dtype, index_dtype));
  const std::string valid_node_name = ascgen_utils::GenValidName(node_name);
  const std::string index_transform = "IndirectLoadIndexTransform_" + valid_node_name;
  const std::string output_transform = "IndirectLoadOutputTransform_" + valid_node_name;
  GELOGI(
      "[IndirectLoad] Generate SIMT transforms for node[%s], rank[%zu], axis[%ld], index_pre_nodes[%zu], "
      "output_post_nodes[%zu].",
      node_name.c_str(), x_info.sizes.size(), axis_, index_pre_nodes_.size(), output_post_nodes_.size());

  GE_ASSERT_SUCCESS(GenerateTypedTransform(index_transform, index_dtype, index_pre_nodes_, ss));
  GE_ASSERT_SUCCESS(GenerateTypedTransform(output_transform, x_dtype, output_post_nodes_, ss));
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                        const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                        const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                        std::string &result) const {
  GE_ASSERT_TRUE(inputs.size() == 2U && outputs.size() == 1U, "IndirectLoad expects 2 inputs and 1 output.");
  GE_ASSERT_TRUE(
      template_id_ == ascir::TemplateId::kIndirectLoadSK || template_id_ == ascir::TemplateId::kIndirectLoadSimd,
      "IndirectLoad tensor-based Generate only supports SK and SIMD.");
  const LogicalTensorInfo x_info = BuildLogicalTensorInfo(logical_view_.data, tpipe);
  const LogicalTensorInfo index_info = BuildLogicalTensorInfo(logical_view_.index, tpipe);
  const LogicalTensorInfo output_info = BuildLogicalTensorInfo(logical_view_.output, tpipe);
  GE_ASSERT_SUCCESS(CheckIndirectLoadShape(x_info, index_info, output_info));
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs);
  if (template_id_ == ascir::TemplateId::kIndirectLoadSK) {
    GELOGI("[IndirectLoad] Generate SK API body for node[%s].", node_name.c_str());
    return GenerateSk(tpipe, current_axis, inputs, outputs, result);
  }
  GELOGI("[IndirectLoad] Generate SIMD API body for node[%s].", node_name.c_str());
  return GenerateSimd(tpipe, current_axis, inputs, outputs, result);
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

Status IndirectLoadRegApiCall::GenerateSk(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                          const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                          const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                          std::string &result) const {
  const Tensor &x = inputs[0].get();
  const Tensor &index = inputs[1].get();
  const Tensor &y = outputs[0].get();
  const auto tmp_iter = tmp_buf_id.find(-1L);
  GE_ASSERT_TRUE(tmp_iter != tmp_buf_id.end(), "IndirectLoad SK requires an API-level tmp buffer.");

  const LogicalTensorInfo x_info = BuildLogicalTensorInfo(logical_view_.data, tpipe);
  const LogicalTensorInfo index_info = BuildLogicalTensorInfo(logical_view_.index, tpipe);
  const size_t axis_pos = static_cast<size_t>(axis_);
  std::string x_dtype_name;
  std::string index_dtype_name;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(x.dtype, x_dtype_name));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(index.dtype, index_dtype_name));
  GE_ASSERT_TRUE(x.dtype == y.dtype, "IndirectLoad SK input/output dtype must match.");
  GELOGD("[IndirectLoad] Generate SK body for node[%s], rank[%zu], axis[%zu].", node_name.c_str(), x_info.sizes.size(),
         axis_pos);

  std::stringstream ss;
  ss << "// IndirectLoad SK" << std::endl;
  ss << "{" << std::endl;
  ss << "  AscendC::IndirectLoadSk<" << x_dtype_name << ", " << index_dtype_name << ", " << x_info.sizes.size() << ", "
     << axis_ << ">(" << std::endl;
  ss << "      " << x << ", " << index << ", " << y << ", " << tpipe.tmp_buf.name << "_" << tmp_iter->second << ", "
     << y.actual_size << ", " << tpipe.tiler.Offset(current_axis, y.axis, y.axis_strides) << ", "
     << tpipe.tiler.Size(x_info.sizes[axis_pos]) << ", " << JoinSizeExprs(index_info.sizes, tpipe) << ", "
     << JoinSizeExprs(x_info.strides, tpipe) << ");" << std::endl;
  ss << "}" << std::endl;
  result = ss.str();
  return af::SUCCESS;
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
  std::string index_dtype_name;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(x.dtype, x_dtype_name));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(index.dtype, index_dtype_name));
  GE_ASSERT_TRUE(x.dtype == y.dtype, "IndirectLoad SIMD input/output dtype must match after input preprocess.");
  GELOGD("[IndirectLoad] Generate SIMD body for node[%s], rank[%zu], axis[%zu].", node_name.c_str(),
         x_info.sizes.size(), axis_pos);

  std::stringstream ss;
  ss << "// IndirectLoad SIMD" << std::endl;
  ss << "{" << std::endl;
  ss << "  AscendC::IndirectLoadSimd<" << x_dtype_name << ", " << index_dtype_name << ", " << x_info.sizes.size()
     << ", " << axis_ << ">(" << std::endl;
  ss << "      " << x << ", " << index << ", " << y << ", " << tpipe.tmp_buf.name << "_" << tmp_iter->second << ", "
     << y.actual_size << ", " << tpipe.tiler.Offset(current_axis, y.axis, y.axis_strides) << ", "
     << tpipe.tiler.Size(x_info.sizes[axis_pos]) << ", " << JoinSizeExprs(index_info.sizes, tpipe) << ", "
     << JoinSizeExprs(x_info.strides, tpipe) << ");" << std::endl;
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

  std::string outer_tb_var;
  const bool has_outer_tb =
      FindCurrentAxisVar(tpipe, current_axis, Axis::Type::kAxisTypeBlockInner, outer_axis_, outer_tb_var);
  GE_ASSERT_TRUE(has_outer_tb, "IndirectLoad SIMT current axes must contain the output block-inner axis.");
  std::string x_dtype, index_dtype, output_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(x.dtype, x_dtype));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(index.dtype, index_dtype));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(output_dtype_, output_dtype));
  const std::string valid_node_name = ascgen_utils::GenValidName(node_name);
  const std::string index_transform = "IndirectLoadIndexTransform_" + valid_node_name;
  const std::string output_transform = "IndirectLoadOutputTransform_" + valid_node_name;
  std::stringstream ss;
  ss << "// IndirectLoad SIMT" << std::endl;
  ss << "{" << std::endl;
  ss << "  __gm__ " << x_dtype << " *x_ptr = (__gm__ " << x_dtype << " *)" << x << ".GetPhyAddr();" << std::endl;
  ss << "  __gm__ " << index_dtype << " *index_ptr = (__gm__ " << index_dtype << " *)" << index << ".GetPhyAddr();"
     << std::endl;
  ss << "  __gm__ " << output_dtype << " *y_ptr = (__gm__ " << output_dtype << " *)" << output_gm_tensor_
     << ".GetPhyAddr();" << std::endl;
  ss << "  AscendC::IndirectLoadSimt<" << x_dtype << ", " << index_dtype << ", " << output_dtype << ", "
     << x_info.sizes.size() << ", " << axis_ << ", " << index_transform << ", " << output_transform << ">("
     << std::endl;
  ss << "      x_ptr, index_ptr, y_ptr, static_cast<uint32_t>(" << outer_tb_var << "_loop_size), block_dim_offset, "
     << tpipe.tiler.Size(x_info.sizes[axis_]) << ", " << JoinSizeExprs(index_info.sizes, tpipe) << ", "
     << JoinSizeExprs(x_info.strides, tpipe) << ");" << std::endl;
  ss << "}" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<IndirectLoadRegApiCall> register_indirect_load_reg_api_call("IndirectLoadRegApiCall");
}  // namespace codegen
