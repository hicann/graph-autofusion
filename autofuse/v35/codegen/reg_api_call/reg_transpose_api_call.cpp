/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reg_transpose_api_call.h"

#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "codegen_api_param/codegen_api_param.h"
#include "codegen/expression_convert_struct.h"
#include "api_call/utils/api_call_utils.h"
#include "ascir_node_param/ascir_node_param.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

void GetTensorVectorizedRepeats(const Tensor &y, std::vector<ascir::SizeExpr> &output_vectorized_repeats) {
  for (size_t i = 0; i < y.vectorized_axis.size(); i++) {
    auto axis_pos = y.vectorized_axis_pos[i];
    output_vectorized_repeats.emplace_back(y.axis_size[axis_pos]);
  }
}

uint32_t GetContinuousInnerAxisNum(const Tensor &y, std::vector<ascir::SizeExpr> &output_vectorized_repeats) {
  uint32_t transpose_inner_axis_num = 1;
  if (y.vectorized_axis.size() == 1) {
    return transpose_inner_axis_num;
  }
  for (int32_t i = y.vectorized_axis.size() - 2; i >= 0; i--) {
    af::Expression inner_axis_stride = output_vectorized_repeats[i + 1] * y.vectorized_strides[i + 1];
    if (af::SymbolicUtils::StaticCheckEq(y.vectorized_strides[i], inner_axis_stride) == af::TriBool::kTrue &&
        transpose_inner_axis_num < 3U) {  // transpose api only support 3 inner continuous axis
      transpose_inner_axis_num++;
    } else {
      break;
    }
  }
  return transpose_inner_axis_num;
}

Status ReorderInputStrideByOutputAxisOrder(const Tensor &x, const Tensor &y,
                                           std::vector<ascir::SizeExpr> &reordered_input_stride) {
  for (size_t i = 0; i < y.vectorized_axis.size(); i++) {
    auto it = std::find(x.vectorized_axis.begin(), x.vectorized_axis.end(), y.vectorized_axis[i]);
    GE_ASSERT_TRUE(it != x.vectorized_axis.end(), "InValid axis ID in input vectorized_axis: %zu", i);
    auto axis_pos = static_cast<uint64_t>(std::distance(x.vectorized_axis.begin(), it));
    reordered_input_stride.emplace_back(x.vectorized_strides[axis_pos]);
  }
  return af::SUCCESS;
}

void BuildTransposeLoopParams(TransposeSpecificParams &transpose_specific_params,
                              std::vector<ascir::SizeExpr> &out_vectorized_repeats,
                              std::vector<ascir::SizeExpr> &reordered_in_vectorized_strides,
                              std::vector<ascir::SizeExpr> &out_vectorized_strides, uint32_t transpose_total_axis_num) {
  for (size_t i = out_vectorized_repeats.size() - transpose_total_axis_num; i < out_vectorized_repeats.size(); i++) {
    transpose_specific_params.output_dims.emplace_back(
        CombinedExpression(ExprItemFactory::ActualSize(out_vectorized_repeats[i])));
    transpose_specific_params.input_strides.emplace_back(
        CombinedExpression(ExprItemFactory::Size(reordered_in_vectorized_strides[i])));
    transpose_specific_params.output_strides.emplace_back(
        CombinedExpression(ExprItemFactory::Size(out_vectorized_strides[i])));
  }
}

af::Status FillTransposeNodeParams(const af::AscNodePtr &node,
                                   const std::vector<ascir::SizeExpr> &out_vectorized_repeats,
                                   const std::vector<ascir::SizeExpr> &reordered_input_strides,
                                   const std::vector<ascir::SizeExpr> &out_vectorized_strides,
                                   uint32_t transpose_inner_axis_num, uint32_t transpose_total_axis_num) {
  GE_ASSERT_NOTNULL(node);
  auto params = ascir_param::GetOrCreateAscirNodeParams(node);

  auto *transpose_params = std::get_if<ascir_param::TransposeNodeParams>(&params->specific_params);
  if (transpose_params == nullptr) {
    params->specific_params = ascir_param::TransposeNodeParams{};
    transpose_params = std::get_if<ascir_param::TransposeNodeParams>(&params->specific_params);
  }
  GE_ASSERT_NOTNULL(transpose_params, "Transpose specific params is null, node[%s].", node->GetNamePtr());
  params->api_name = node->GetType();
  params->status = ascir_param::ParamBuildStatus::kBuilt;

  *transpose_params = ascir_param::TransposeNodeParams{};
  transpose_params->valid = true;
  transpose_params->inner_dim = transpose_inner_axis_num;
  transpose_params->total_dim = transpose_total_axis_num;
  const std::vector<ascir::SizeExpr> out_loop_repeats(out_vectorized_repeats.begin(),
                                                      out_vectorized_repeats.end() - transpose_total_axis_num);
  transpose_params->outer_loop_axes.assign(out_loop_repeats.begin(), out_loop_repeats.end());
  const size_t begin = out_vectorized_repeats.size() - transpose_total_axis_num;
  for (size_t i = begin; i < out_vectorized_repeats.size(); ++i) {
    transpose_params->output_dims.emplace_back(out_vectorized_repeats[i]);
    transpose_params->input_strides.emplace_back(reordered_input_strides[i]);
    transpose_params->output_strides.emplace_back(out_vectorized_strides[i]);
  }
  return af::SUCCESS;
}

// 构建带循环偏移的 inner_offset 表达式（简化表达式合并操作）
CombinedExpression BuildInnerOffsetWithLoopOffset(const std::string &base_offset,
                                                  const std::vector<ascir::SizeExpr> &loop_strides) {
  CombinedExpression inner_offset = CombinedExpression(ExprItemFactory::Direct(ge::Symbol(base_offset.c_str())));
  CombinedExpression loop_offset = CalcInnerOffsetExpr(loop_strides);
  inner_offset.AddExpression(loop_offset, "+");
  return inner_offset;
}

Status TransposeRegApiCall::BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                          const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                          const std::vector<std::reference_wrapper<const Tensor>> &outputs) const {
  auto x = inputs[0].get();
  auto y = outputs[0].get();
  auto api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);
  api_param->api_name = api_name_;
  std::vector<ascir::SizeExpr> out_vectorized_repeats;
  GetTensorVectorizedRepeats(y, out_vectorized_repeats);
  uint32_t transpose_inner_axis_num = GetContinuousInnerAxisNum(y, out_vectorized_repeats);
  uint32_t transpose_total_axis_num = y.vectorized_axis.size() > 4U ? 4U : y.vectorized_axis.size();
  api_param->template_params.emplace_back(std::to_string(transpose_inner_axis_num));
  api_param->template_params.emplace_back(std::to_string(transpose_total_axis_num));
  std::vector<ascir::SizeExpr> reordered_in_vectorized_strides;
  GE_CHK_STATUS_RET(ReorderInputStrideByOutputAxisOrder(x, y, reordered_in_vectorized_strides));
  // 构建 inner_offset 表达式
  CombinedExpression input_inner_offset = CombinedExpression(
      ExprItemFactory::Direct(ge::Symbol(tpipe.tiler.TensorVectorizedOffset(current_axis, x).c_str())));
  CombinedExpression output_inner_offset = CombinedExpression(
      ExprItemFactory::Direct(ge::Symbol(tpipe.tiler.TensorVectorizedOffset(current_axis, y).c_str())));
  if (y.vectorized_axis.size() > 4U) {
    std::vector<ascir::SizeExpr> input_loop_stride(reordered_in_vectorized_strides.begin(),
                                                   reordered_in_vectorized_strides.end() - transpose_total_axis_num);
    std::vector<ascir::SizeExpr> output_loop_stride(y.vectorized_strides.begin(),
                                                    y.vectorized_strides.end() - transpose_total_axis_num);
    std::vector<ascir::SizeExpr> out_loop_repeats(out_vectorized_repeats.begin(),
                                                  out_vectorized_repeats.end() - transpose_total_axis_num);
    // 使用辅助函数简化表达式合并
    input_inner_offset.AddExpression(CalcInnerOffsetExpr(input_loop_stride), "+");
    output_inner_offset.AddExpression(CalcInnerOffsetExpr(output_loop_stride), "+");
    for (const auto &repeat : out_loop_repeats) {
      api_param->outer_loop_axes.emplace_back(CombinedExpression(ExprItemFactory::ActualSize(repeat)));
    }
  }
  api_param->input_params.emplace_back(x.Str(), true, input_inner_offset);
  api_param->output_params.emplace_back(y.Str(), true, output_inner_offset);

  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "TernaryApiTmpV2Call cannot find tmp buffer id to use.");
  int64_t id = it->second;
  api_param->tmp_buf_name = tpipe.tmp_buf.name + "_" + std::to_string(id);

  TransposeSpecificParams transpose_specific_params;
  BuildTransposeLoopParams(transpose_specific_params, out_vectorized_repeats, reordered_in_vectorized_strides,
                           y.vectorized_strides, transpose_total_axis_num);
  api_param->specific_params = transpose_specific_params;
  GE_ASSERT_SUCCESS(FillTransposeNodeParams(this->node, out_vectorized_repeats, reordered_in_vectorized_strides,
                                            y.vectorized_strides, transpose_inner_axis_num, transpose_total_axis_num));

  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param));
  return af::SUCCESS;
}

Status GenTransposeDimParam(const CodegenApiParam &api_param, const Tiler &tiler, std::string graph_name,
                            std::string node_name, std::string dtype_name, std::stringstream &ss) {
  auto *transpose_params = std::get_if<TransposeSpecificParams>(&api_param.specific_params);
  GE_ASSERT_NOTNULL(transpose_params, "transpose_params is null, graph name: %s, node name: %s", graph_name.c_str(),
                    node_name.c_str());
  ss << "{";
  for (size_t i = 0; i < transpose_params->output_dims.size(); i++) {
    if (i != 0) {
      ss << ", ";
    }
    ss << "static_cast<" << dtype_name << ">(" << transpose_params->output_dims[i].ToStr(tiler) << ")";
  }
  ss << "}, ";

  ss << "{";
  for (size_t i = 0; i < transpose_params->output_dims.size(); i++) {
    if (i != 0) {
      ss << ", ";
    }
    ss << "static_cast<" << dtype_name << ">(" << transpose_params->input_strides[i].ToStr(tiler) << ")";
  }
  ss << "}, ";

  ss << "{";
  for (size_t i = 0; i < transpose_params->output_dims.size(); i++) {
    if (i != 0) {
      ss << ", ";
    }
    ss << "static_cast<" << dtype_name << ">(" << transpose_params->output_strides[i].ToStr(tiler) << ")";
  }
  ss << "}";
  ss << ");" << std::endl;
  return af::SUCCESS;
}

Status TransposeRegApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler,
                                              std::stringstream &ss) const {
  auto data_type_size = GetSizeByDataType(node->outputs[0].attr.dtype);
  GE_ASSERT_TRUE(data_type_size > 0);
  std::string dtype_name = static_cast<uint32_t>(data_type_size) <= sizeof(int16_t) ? "int16_t" : "int32_t";
  return GenTransposeDimParam(api_param, tiler, graph_name, node_name, dtype_name, ss);
}

static ApiCallRegister<TransposeRegApiCall> register_transpose_api_call("TransposeRegApiCall");
}  // namespace codegen
