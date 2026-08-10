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
#include <unordered_set>
#include "api_call/utils/api_call_factory.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "common/checker.h"
#include "common_utils.h"
#include "indirect_load_utils.h"
#include "v35/ascir/ascir_codegen_v2.h"

namespace codegen {
namespace {
struct LogicalTensorInfo {
  std::vector<ascir::SizeExpr> sizes;
  std::vector<ascir::SizeExpr> strides;
};

af::Status EmitSimtScalarExpr(const ascir::NodeView &node, const std::vector<std::string> &inputs, std::string &expr) {
  GE_ASSERT_NOTNULL(node, "SIMT scalar node is null.");
  GE_ASSERT_TRUE(!inputs.empty() && inputs.size() == node->inputs.Size(),
                 "SIMT scalar node %s[%s] expects %zu non-empty inputs, but got %zu.", node->GetTypePtr(),
                 node->GetNamePtr(), node->inputs.Size(), inputs.size());
  const auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
  GE_ASSERT_NOTNULL(impl, "SIMT scalar codegen is not registered for node %s[%s].", node->GetTypePtr(),
                    node->GetNamePtr());
  const auto *v2_impl = dynamic_cast<af::ascir::AscIrCodegenV2 *>(impl.get());
  GE_ASSERT_NOTNULL(v2_impl, "SIMT scalar codegen for node %s[%s] is not a V2 implementation.", node->GetTypePtr(),
                    node->GetNamePtr());
  GE_ASSERT_TRUE(v2_impl->IsSimtScalarSupported(*node), "SIMT scalar codegen is not supported for node %s[%s].",
                 node->GetTypePtr(), node->GetNamePtr());
  return v2_impl->GenerateSimtScalarExpr(*node, inputs, expr);
}

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

af::Status CheckIndirectLoadDenseLayout(const LogicalTensorInfo &input_info, const LogicalTensorInfo &index_info,
                                        const LogicalTensorInfo &output_info) {
  GE_ASSERT_SUCCESS(CheckDenseStrides(input_info));
  GE_ASSERT_SUCCESS(CheckDenseStrides(index_info));
  GE_ASSERT_SUCCESS(CheckDenseStrides(output_info));
  return af::SUCCESS;
}

using SimtNodeSet = std::unordered_set<const af::AscNode *>;

bool IsSimtGmInput(const af::AscNodePtr &node) {
  return af::ops::IsOps<af::ascir_op::Load>(node);
}

af::Status CollectSimtBackwardNodes(const af::AscNodePtr &root, const ascir::NodeView &indirect_load,
                                    SimtNodeSet &nodes) {
  std::vector<af::AscNodePtr> pending = {root};
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr current = pending[cursor];
    if (current == nullptr || current == indirect_load || !nodes.emplace(current.get()).second) {
      continue;
    }
    if (IsSimtGmInput(current)) {
      continue;
    }
    GE_ASSERT_TRUE(current->inputs.Size() > 0UL, "IndirectLoad SIMT node[%s] has no input.", current->GetNamePtr());
    for (size_t i = 0UL; i < current->inputs.Size(); ++i) {
      const af::AscNodePtr producer = ascgen_utils::indirect_load::GetInputProducer(current, i);
      GE_ASSERT_NOTNULL(producer, "IndirectLoad SIMT node[%s] input[%zu] has no producer.", current->GetNamePtr(), i);
      pending.emplace_back(producer);
    }
  }
  return af::SUCCESS;
}

af::AscNodePtr FindSimtOutputStore(const ascir::NodeView &indirect_load) {
  for (af::AscNodePtr current = ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load); current != nullptr;
       current = ascgen_utils::indirect_load::GetOnlyOutputConsumer(current)) {
    if (af::ops::IsOps<af::ascir_op::Store>(current)) {
      return current;
    }
  }
  return nullptr;
}

af::Status ValidateSimtRegionNode(const af::AscNodePtr &node) {
  if (IsSimtGmInput(node) || af::ops::IsOps<af::ascir_op::Store>(node)) {
    return af::SUCCESS;
  }
  GE_ASSERT_TRUE(ascgen_utils::indirect_load::IsSimtInlineTransform(node),
                 "IndirectLoad SIMT node[%s] has no inline-transform role.", node->GetNamePtr());
  GE_ASSERT_TRUE(!af::ops::IsOps<af::ascir_op::VectorFunc>(node),
                 "IndirectLoad SIMT transform must use scalar emission, node:%s", node->GetNamePtr());
  return af::SUCCESS;
}

void AppendSimtGmTensor(const af::AscNodePtr &node, std::vector<SimtGmTensor> &gm_tensors) {
  const auto output = node->outputs()[0];
  const auto found = std::find_if(gm_tensors.begin(), gm_tensors.end(), [output](const SimtGmTensor &tensor) {
    return tensor.value_tensor_id == output->attr.mem.tensor_id;
  });
  if (found == gm_tensors.end()) {
    gm_tensors.push_back({output->attr.mem.tensor_id, node->inputs()[0]->attr.mem.tensor_id, output->attr.dtype});
  }
}

af::Status CollectSimtRegionMetadata(const ascir::NodeView &indirect_load, std::vector<af::AscNodePtr> &index_nodes,
                                     std::vector<af::AscNodePtr> &output_nodes, std::vector<SimtGmTensor> &gm_tensors,
                                     af::AscNodePtr &store) {
  const af::AscNodePtr index_root =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
  GE_ASSERT_NOTNULL(index_root, "IndirectLoad SIMT index input has no producer.");
  store = FindSimtOutputStore(indirect_load);
  GE_ASSERT_NOTNULL(store, "IndirectLoad SIMT output chain must end with a Store node.");
  SimtNodeSet index_set;
  SimtNodeSet output_set;
  GE_ASSERT_SUCCESS(CollectSimtBackwardNodes(index_root, indirect_load, index_set));
  GE_ASSERT_SUCCESS(CollectSimtBackwardNodes(store, indirect_load, output_set));
  const auto owner_graph = indirect_load->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph, "IndirectLoad SIMT node has no owner graph.");
  for (const auto &graph_node : owner_graph->GetDirectNode()) {
    const af::AscNodePtr node = std::dynamic_pointer_cast<af::AscNode>(graph_node);
    GE_ASSERT_NOTNULL(node, "IndirectLoad SIMT graph contains invalid node.");
    if (index_set.count(node.get()) != 0UL) {
      GE_ASSERT_SUCCESS(ValidateSimtRegionNode(node));
      index_nodes.emplace_back(node);
    }
    if (output_set.count(node.get()) != 0UL) {
      GE_ASSERT_SUCCESS(ValidateSimtRegionNode(node));
      output_nodes.emplace_back(node);
    }
    if (IsSimtGmInput(node) && (index_set.count(node.get()) != 0UL || output_set.count(node.get()) != 0UL)) {
      AppendSimtGmTensor(node, gm_tensors);
    }
  }
  return af::SUCCESS;
}

af::Status CollectSimtMetadata(const ascir::NodeView &indirect_load, const af::AscNodePtr &root,
                               std::vector<af::AscNodePtr> &nodes, std::vector<SimtGmTensor> &gm_tensors) {
  GE_ASSERT_NOTNULL(root, "IndirectLoad SIMT region root is missing.");
  SimtNodeSet node_set;
  GE_ASSERT_SUCCESS(CollectSimtBackwardNodes(root, indirect_load, node_set));
  const auto owner_graph = indirect_load->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph, "IndirectLoad SIMT node has no owner graph.");
  for (const auto &graph_node : owner_graph->GetDirectNode()) {
    const af::AscNodePtr node = std::dynamic_pointer_cast<af::AscNode>(graph_node);
    if (node == nullptr || node_set.count(node.get()) == 0UL) {
      continue;
    }
    GE_ASSERT_SUCCESS(ValidateSimtRegionNode(node));
    nodes.emplace_back(node);
    if (IsSimtGmInput(node)) {
      AppendSimtGmTensor(node, gm_tensors);
    }
  }
  return af::SUCCESS;
}

af::Status GenerateSimtEvaluator(const std::string &method, const std::string &return_dtype,
                                 const std::string &value_dtype, ascir::TensorId value_tensor_id,
                                 ascir::TensorId result_tensor_id, const std::vector<af::AscNodePtr> &nodes,
                                 std::stringstream &ss) {
  std::map<ascir::TensorId, std::string> values;
  if (value_tensor_id != af::kIdNone) {
    values.emplace(value_tensor_id, "value");
  }
  ss << "  __simt_callee__ __aicore__ inline static " << return_dtype << " " << method << "(";
  if (!value_dtype.empty()) {
    ss << value_dtype << " value, ";
  }
  ss << "int64_t output_index, const Context &context) {" << std::endl;
  for (const af::AscNodePtr &node : nodes) {
    if (IsSimtGmInput(node)) {
      const auto output = node->outputs()[0];
      values[output->attr.mem.tensor_id] =
          "context.gm_" + std::to_string(output->attr.mem.tensor_id) + "[output_index]";
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Store>(node)) {
      continue;
    }
    std::vector<std::string> inputs;
    for (size_t i = 0UL; i < node->inputs.Size(); ++i) {
      const auto found = values.find(node->inputs()[i]->attr.mem.tensor_id);
      GE_ASSERT_TRUE(found != values.end(), "SIMT node[%s] input[%zu] has no scalar value.", node->GetNamePtr(), i);
      inputs.emplace_back(found->second);
    }
    std::string expr;
    GE_ASSERT_SUCCESS(EmitSimtScalarExpr(node, inputs, expr));
    const auto output = node->outputs()[0];
    std::string output_dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(output->attr.dtype, output_dtype));
    const std::string variable = "v_" + std::to_string(output->attr.mem.tensor_id);
    ss << "    " << output_dtype << " " << variable << " = " << expr << ";" << std::endl;
    values[output->attr.mem.tensor_id] = variable;
  }
  const auto result = values.find(result_tensor_id);
  GE_ASSERT_TRUE(result != values.end(), "SIMT %s result tensor[%ld] has no scalar value.", method.c_str(),
                 result_tensor_id);
  ss << "    return " << result->second << ";" << std::endl;
  ss << "  }" << std::endl;
  return af::SUCCESS;
}

af::Status CalcVectorizedElementCount(const Tensor &tensor, af::Expression &element_count) {
  element_count = af::ops::One;
  for (uint32_t axis_pos : tensor.vectorized_axis_pos) {
    GE_ASSERT_TRUE(axis_pos < tensor.axis_size.size(), "IndirectLoad SIMT output axis is invalid.");
    element_count = af::sym::Mul(element_count, tensor.axis_size[axis_pos]);
  }
  return af::SUCCESS;
}

af::Status GenerateSimtContextInitializer(const std::string &context_name, const std::vector<SimtGmTensor> &gm_tensors,
                                          std::stringstream &ss) {
  ss << "  " << context_name << " context{";
  for (size_t i = 0UL; i < gm_tensors.size(); ++i) {
    const SimtGmTensor &gm_tensor = gm_tensors[i];
    std::string dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(gm_tensor.dtype, dtype));
    ss << (i == 0UL ? "" : ", ") << "(__gm__ " << dtype << " *)global_" << gm_tensor.gm_tensor_id << ".GetPhyAddr()";
  }
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
  const af::AscNodePtr post_reduce = ascgen_utils::indirect_load::GetPostReduceConsumer(node);
  has_post_reduce_ = post_reduce != nullptr;
  template_id_ = ::ascir::GetTemplateIdOrDefault(*node);
  GE_ASSERT_TRUE(
      template_id_ == ascir::TemplateId::kIndirectLoadSK || template_id_ == ascir::TemplateId::kIndirectLoadSimd ||
          template_id_ == ascir::TemplateId::kIndirectLoadSimt,
      "IndirectLoad node[%s] has invalid template id[%d].", node->GetNamePtr(), static_cast<int32_t>(template_id_));
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateLogicalView(node, logical_view_));
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetImplementation(node, implementation_));
  const int64_t rank = static_cast<int64_t>(logical_view_.input.axis_ids.size());
  GE_ASSERT_TRUE(axis >= -rank && axis < rank, "IndirectLoad axis is out of range.");
  axis_ = axis < 0L ? axis + rank : axis;
  if (template_id_ == ascir::TemplateId::kIndirectLoadSimt) {
    GE_ASSERT_SUCCESS(ParseSimtAttr(node));
  }
  GELOGI("[IndirectLoad] Parse codegen attrs for node[%s], axis[%ld], template_id[%d].", node->GetNamePtr(), axis_,
         static_cast<int32_t>(template_id_));
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::ParseSimtAttr(const ascir::NodeView &node) {
  GE_ASSERT_TRUE(node->inputs.Size() == 2UL && node->outputs().size() == 1UL,
                 "IndirectLoad SIMT expects 2 inputs and 1 output.");
  const auto node_inputs = node->inputs();
  index_result_tensor_id_ = node_inputs[ascgen_utils::indirect_load::kIndexTensorIndex]->attr.mem.tensor_id;
  index_dtype_ = node_inputs[ascgen_utils::indirect_load::kIndexTensorIndex]->attr.dtype;
  simt_value_tensor_id_ = node->outputs()[0]->attr.mem.tensor_id;
  if (has_post_reduce_) {
    const af::AscNodePtr output_root = ascgen_utils::indirect_load::GetPostReduceInputProducer(node);
    GE_ASSERT_NOTNULL(output_root, "IndirectLoad post Reduce input has no producer.");
    GE_ASSERT_TRUE(!output_root->outputs().empty(), "IndirectLoad post Reduce input producer has no output.");
    output_result_tensor_id_ = output_root->outputs()[0]->attr.mem.tensor_id;
    output_dtype_ = output_root->outputs()[0]->attr.dtype;
    const af::AscNodePtr index_root =
        ascgen_utils::indirect_load::GetInputProducer(node, ascgen_utils::indirect_load::kIndexTensorIndex);
    GE_ASSERT_SUCCESS(CollectSimtMetadata(node, index_root, index_nodes_, simt_gm_tensors_));
    GE_ASSERT_SUCCESS(CollectSimtMetadata(node, output_root, output_nodes_, simt_gm_tensors_));
    GE_ASSERT_TRUE(outputs.size() == 1UL, "IndirectLoad SIMT expects one output.");
    outputs[0].id = output_result_tensor_id_;
    return af::SUCCESS;
  }
  af::AscNodePtr store;
  GE_ASSERT_SUCCESS(CollectSimtRegionMetadata(node, index_nodes_, output_nodes_, simt_gm_tensors_, store));
  GE_ASSERT_TRUE(store->inputs.Size() == 1UL && store->outputs().size() == 1UL,
                 "IndirectLoad SIMT Store expects 1 input and 1 output.");
  output_result_tensor_id_ = store->inputs()[0]->attr.mem.tensor_id;
  output_gm_tensor_ = "global_" + std::to_string(store->outputs()[0]->attr.mem.tensor_id);
  output_dtype_ = store->outputs()[0]->attr.dtype;
  GE_ASSERT_TRUE(store->inputs()[0]->attr.dtype == output_dtype_,
                 "IndirectLoad SIMT output transform dtype[%d] does not match Store dtype[%d].",
                 static_cast<int32_t>(store->inputs()[0]->attr.dtype), static_cast<int32_t>(output_dtype_));
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::GenerateFuncDefinition(const TPipe &tpipe, const Tiler &tiler,
                                                      std::stringstream &ss) const {
  (void)tiler;
  if (template_id_ != ascir::TemplateId::kIndirectLoadSimt) {
    return af::SUCCESS;
  }

  GE_ASSERT_TRUE(inputs.size() == 2U, "IndirectLoad SIMT expects 2 inputs.");
  const Tensor *input_tensor = tpipe.GetTensor(inputs[ascgen_utils::indirect_load::kInputTensorIndex]->id);
  GE_ASSERT_NOTNULL(input_tensor, "IndirectLoad SIMT input tensor is missing.");
  const Tensor &input = *input_tensor;
  const LogicalTensorInfo input_info = BuildLogicalTensorInfo(logical_view_.input, tpipe);
  const LogicalTensorInfo index_info = BuildLogicalTensorInfo(logical_view_.index, tpipe);
  std::string input_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(input.dtype, input_dtype));
  const std::string valid_node_name = ascgen_utils::GenValidName(node_name);
  const std::string context_name = "IndirectLoadSimtContext_" + valid_node_name;
  const std::string body_name = "IndirectLoadSimtBody_" + valid_node_name;
  std::string index_dtype;
  std::string output_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(index_dtype_, index_dtype));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(output_dtype_, output_dtype));
  GELOGI(
      "[IndirectLoad] Generate SIMT body for node[%s], rank[%zu], axis[%ld], index_nodes[%zu], output_nodes[%zu], "
      "gm_inputs[%zu].",
      node_name.c_str(), input_info.sizes.size(), axis_, index_nodes_.size(), output_nodes_.size(),
      simt_gm_tensors_.size());

  ss << "struct " << context_name << " {" << std::endl;
  for (const SimtGmTensor &tensor : simt_gm_tensors_) {
    std::string dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(tensor.dtype, dtype));
    ss << "  __gm__ " << dtype << " *gm_" << tensor.value_tensor_id << ";" << std::endl;
  }
  ss << "};" << std::endl;
  ss << "struct " << body_name << " {" << std::endl;
  ss << "  using Context = " << context_name << ";" << std::endl;
  GE_ASSERT_SUCCESS(
      GenerateSimtEvaluator("Index", index_dtype, "", af::kIdNone, index_result_tensor_id_, index_nodes_, ss));
  GE_ASSERT_SUCCESS(GenerateSimtEvaluator("Output", output_dtype, input_dtype, simt_value_tensor_id_,
                                          output_result_tensor_id_, output_nodes_, ss));
  ss << "};" << std::endl;
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                        const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                        const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                        std::string &result) const {
  GE_ASSERT_TRUE(inputs.size() == 2U && outputs.size() == 1U, "IndirectLoad expects 2 inputs and 1 output.");
  const LogicalTensorInfo input_info = BuildLogicalTensorInfo(logical_view_.input, tpipe);
  const LogicalTensorInfo index_info = BuildLogicalTensorInfo(logical_view_.index, tpipe);
  const LogicalTensorInfo output_info = BuildLogicalTensorInfo(logical_view_.output, tpipe);
  GE_ASSERT_SUCCESS(CheckIndirectLoadDenseLayout(input_info, index_info, output_info));
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs);
  if (template_id_ == ascir::TemplateId::kIndirectLoadSK) {
    GELOGI("[IndirectLoad] Generate SK API body for node[%s].", node_name.c_str());
    return GenerateSk(tpipe, current_axis, inputs, outputs, result);
  } else if (template_id_ == ascir::TemplateId::kIndirectLoadSimd) {
    GELOGI("[IndirectLoad] Generate SIMD API body for node[%s].", node_name.c_str());
    return GenerateSimd(tpipe, current_axis, inputs, outputs, result);
  } else {
    GE_ASSERT_TRUE(false, "IndirectLoad tensor-based Generate only supports SK and SIMD.");
  }
}

Status IndirectLoadRegApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                        std::string &result) const {
  if (template_id_ != ascir::TemplateId::kIndirectLoadSimt) {
    return ApiCall::Generate(tpipe, current_axis, result);
  }
  GE_ASSERT_TRUE(inputs.size() == 2U, "IndirectLoad SIMT expects 2 inputs.");
  const Tensor *input_tensor = tpipe.GetTensor(inputs[ascgen_utils::indirect_load::kInputTensorIndex]->id);
  GE_ASSERT_NOTNULL(input_tensor, "IndirectLoad SIMT input tensor is missing.");
  return GenerateSimt(tpipe, current_axis, *input_tensor, result);
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

  const LogicalTensorInfo x_info = BuildLogicalTensorInfo(logical_view_.input, tpipe);
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
  const LogicalTensorInfo input_info = BuildLogicalTensorInfo(logical_view_.input, tpipe);
  const LogicalTensorInfo index_info = BuildLogicalTensorInfo(logical_view_.index, tpipe);
  const Tensor &input = inputs[ascgen_utils::indirect_load::kInputTensorIndex].get();
  const Tensor &index = inputs[ascgen_utils::indirect_load::kIndexTensorIndex].get();
  const Tensor &output = outputs[0].get();

  const size_t axis_pos = static_cast<size_t>(axis_);
  std::string input_dtype;
  std::string index_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(input.dtype, input_dtype));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(index.dtype, index_dtype));
  GE_ASSERT_TRUE(input.dtype == output.dtype,
                 "IndirectLoad SIMD input/output dtype must match after input preprocess.");
  GELOGD("[IndirectLoad] Generate SIMD body for node[%s], rank[%zu], axis[%zu].", node_name.c_str(),
         input_info.sizes.size(), axis_pos);

  std::stringstream ss;
  ss << "// IndirectLoad SIMD" << std::endl;
  ss << "{" << std::endl;
  const char *api = implementation_ == ascgen_utils::indirect_load::Implementation::kGatherApi
                        ? "IndirectLoadSimdGatherApi"
                        : "IndirectLoadSimd";
  ss << "  AscendC::" << api << "<" << input_dtype << ", " << index_dtype << ", " << input_info.sizes.size() << ", "
     << axis_ << ">(" << std::endl;
  ss << "      " << input << ", " << index << ", " << output << ", ";
  ss << output.actual_size << ", " << tpipe.tiler.Offset(current_axis, output.axis, output.axis_strides) << ", "
     << input.actual_size << ", " << tpipe.tiler.Size(input_info.sizes[axis_pos]) << ", "
     << JoinSizeExprs(index_info.sizes, tpipe) << ", " << JoinSizeExprs(input_info.strides, tpipe);
  ss << ");" << std::endl;
  ss << "}" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::GenerateSimt(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                            const Tensor &input, std::string &result) const {
  const LogicalTensorInfo input_info = BuildLogicalTensorInfo(logical_view_.input, tpipe);
  const LogicalTensorInfo index_info = BuildLogicalTensorInfo(logical_view_.index, tpipe);
  const LogicalTensorInfo output_info = BuildLogicalTensorInfo(logical_view_.output, tpipe);
  GE_ASSERT_SUCCESS(CheckIndirectLoadDenseLayout(input_info, index_info, output_info));
  GELOGD("[IndirectLoad] Generate SIMT body for node[%s], output_gm[%s].", node_name.c_str(),
         output_gm_tensor_.c_str());

  std::string outer_tb_var;
  const bool has_outer_tb =
      FindCurrentAxisVar(tpipe, current_axis, Axis::Type::kAxisTypeBlockInner, outer_axis_, outer_tb_var);
  GE_ASSERT_TRUE(has_outer_tb, "IndirectLoad SIMT current axes must contain the output block-inner axis.");
  std::string input_dtype, output_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(input.dtype, input_dtype));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(output_dtype_, output_dtype));
  const std::string valid_node_name = ascgen_utils::GenValidName(node_name);
  const std::string context_name = "IndirectLoadSimtContext_" + valid_node_name;
  const std::string body_name = "IndirectLoadSimtBody_" + valid_node_name;
  std::stringstream ss;
  ss << "// IndirectLoad SIMT" << std::endl;
  ss << "{" << std::endl;
  ss << "  __gm__ " << input_dtype << " *input_ptr = (__gm__ " << input_dtype << " *)" << input << ".GetPhyAddr();"
     << std::endl;
  const Tensor *output_tensor = has_post_reduce_ ? tpipe.GetTensor(output_result_tensor_id_) : nullptr;
  af::Expression output_element_count = af::ops::One;
  if (has_post_reduce_) {
    GE_ASSERT_NOTNULL(output_tensor, "IndirectLoad SIMT UB output tensor is missing.");
    GE_ASSERT_SUCCESS(CalcVectorizedElementCount(*output_tensor, output_element_count));
  } else {
    ss << "  __gm__ " << output_dtype << " *y_ptr = (__gm__ " << output_dtype << " *)" << output_gm_tensor_
       << ".GetPhyAddr();" << std::endl;
  }
  GE_ASSERT_SUCCESS(GenerateSimtContextInitializer(context_name, simt_gm_tensors_, ss));
  ss << "  AscendC::IndirectLoadSimt<" << input_dtype << ", " << output_dtype << ", " << input_info.sizes.size() << ", "
     << axis_ << ", " << body_name << ">(" << std::endl;
  if (has_post_reduce_) {
    const std::string output_elements = tpipe.tiler.Size(output_element_count);
    ss << "      input_ptr, " << *output_tensor << ", context, static_cast<uint32_t>(" << output_elements
       << "), (block_dim_offset + " << outer_tb_var << ") * " << output_elements << ", ";
  } else {
    ss << "      input_ptr, y_ptr, context, static_cast<uint32_t>(" << outer_tb_var
       << "_loop_size), block_dim_offset, ";
  }
  ss << tpipe.tiler.Size(input_info.sizes[axis_]) << ", " << JoinSizeExprs(index_info.sizes, tpipe) << ", "
     << JoinSizeExprs(input_info.strides, tpipe) << ");" << std::endl;
  ss << "}" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<IndirectLoadRegApiCall> register_indirect_load_reg_api_call("IndirectLoadRegApiCall");
}  // namespace codegen
