/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "compare_v2_api_call.h"

#include <memory>
#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "api_call/utils/api_call_utils.h"
#include "ascir_node_param/ascir_node_param.h"
#include "codegen/expression_convert_struct.h"
#include "reg_api_call_utils.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

namespace {
constexpr const char *kAscirNodeParams = "AscirNodeParams";

af::Status FillCompareNodeParams(const af::AscNodePtr &node, bool is_scalar, const ge::Expression &outer_call_count,
                                 const std::vector<ge::Expression> &output_dims,
                                 const std::vector<ge::Expression> &output_strides,
                                 const std::vector<ge::Expression> &input_strides) {
  GE_ASSERT_NOTNULL(node);
  auto params = ascir_param::GetAscirNodeParams(node);
  if (params == nullptr) {
    auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    params = std::make_shared<ascir_param::AscirNodeParams>();
    GE_ASSERT_TRUE(op_desc->SetExtAttr(kAscirNodeParams, params), "Node:%s SetExtAttr failed", node->GetNamePtr());
  }

  auto *compare_params = std::get_if<ascir_param::CompareNodeParams>(&params->specific_params);
  if (compare_params == nullptr) {
    params->specific_params = ascir_param::CompareNodeParams{};
    compare_params = std::get_if<ascir_param::CompareNodeParams>(&params->specific_params);
  }
  GE_ASSERT_NOTNULL(compare_params, "Compare specific params is null, node[%s].", node->GetNamePtr());
  params->api_name = node->GetType();
  params->status = ascir_param::ParamBuildStatus::kBuilt;

  *compare_params = ascir_param::CompareNodeParams{};
  compare_params->valid = true;
  compare_params->is_scalar = is_scalar;
  compare_params->outer_call_count = outer_call_count;
  compare_params->output_dims = output_dims;
  compare_params->output_strides = output_strides;
  compare_params->input_strides = input_strides;
  return af::SUCCESS;
}

void BuildCompareLoopParams(const VectorizedAxisLoopMergeStatus &merge_info, const ApiLoopParams &param,
                            ge::Expression &outer_call_count, std::vector<ge::Expression> &output_dims,
                            std::vector<ge::Expression> &output_strides, std::vector<ge::Expression> &input_strides) {
  if (param.outer_repeats.empty()) {
    outer_call_count = af::ops::One;
    if (!merge_info.merge_repeats.empty()) {
      output_dims.emplace_back(merge_info.merge_repeats.back());
    }
    output_strides.emplace_back(af::ops::One);
    input_strides.emplace_back(af::ops::One);
    return;
  }

  const size_t outer_repeats_size = param.outer_repeats.size();
  outer_call_count = af::ops::One;
  for (size_t i = 0U; i + 1U < outer_repeats_size; ++i) {
    outer_call_count = outer_call_count * merge_info.merge_repeats[i];
  }
  output_dims.emplace_back(merge_info.merge_repeats[outer_repeats_size - 1]);
  output_dims.emplace_back(param.cal_count);
  output_strides.emplace_back(param.output_second_to_last_stride);
  output_strides.emplace_back(af::ops::One);
  input_strides.emplace_back(param.input_second_to_last_stride);
  input_strides.emplace_back(af::ops::One);
}
}  // namespace

static void CreateComputeNodeOuterForIfRequired(size_t outer_repeats_size, ApiLoopParams param,
                                                const std::stringstream &ss1, std::stringstream &ss) {
  if (outer_repeats_size == 1UL) {
    ss << ss1.str();
  } else {
    CreateComputeNodeOuterFor({param.outer_repeats.begin(), param.outer_repeats.begin() + outer_repeats_size - 1}, ss1,
                              ss, 0);
  }
  return;
}

Status CompareV2ApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                  const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                  std::string &result) const {
  auto x1 = inputs[0].get();
  auto x2 = inputs[1].get();
  auto y = outputs[0].get();
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs);
  stringstream ss;
  ApiLoopParams param;
  VectorizedAxisLoopMergeStatus merge_info;
  std::vector<Tensor> ub_inputs;
  std::vector<Tensor> ub_outputs;

  // 如果第2个输入是ub_scalar场景, 初始化x2为ub_scalar对应的变量
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(x2.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(x2.dtype));
  std::string x2_scalar = x2.Str();
  if (x2.is_ub_scalar && x2.need_gen_get_value_of_ub_scalar) {
    x2_scalar = "(" + dtype_name + ")" + x2.ub_scalar_name;
  }

  if (x2.IsAnyScalar()) {
    const std::string actual_size = x1.actual_size.Str();
    ub_inputs.push_back(x1);
    ub_outputs.push_back(y);
    bool status = GenerateVectorizedAxisMergeStatus(ub_inputs, ub_outputs, merge_info, tpipe);
    GE_ASSERT_TRUE(status, "GenerateVectorizedAxisMergeStatus failed");
    SaveApiLoopAxisParams(merge_info, param);
    std::vector<ge::Expression> output_dims;
    std::vector<ge::Expression> output_strides;
    std::vector<ge::Expression> input_strides;
    ge::Expression outer_call_count = af::ops::One;
    BuildCompareLoopParams(merge_info, param, outer_call_count, output_dims, output_strides, input_strides);
    GE_ASSERT_SUCCESS(
        FillCompareNodeParams(this->node, true, outer_call_count, output_dims, output_strides, input_strides));
    std::string scalar_local_blk_tensor_name_x2 = x2.IsConstScalar() ? "local_blk_tensor_of_" + x2.name : x2.name;
    size_t outer_repeats_size = param.outer_repeats.size();
    if (outer_repeats_size == 0U) {
      if (IsCVFusionStage(this->api_call_context)) {
        const auto cv_params = BuildCvApi2DParams(tpipe, x1, y);
        ss << "CompareScalarExtend<" << dtype_name << ", 2, CMPMODE::" << this->api_name_ << ">(" << y << "["
           << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x1 << "["
           << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], " << x2_scalar << ", "
           << GenCvUint16Dims(cv_params) << ", " << GenCvUint16Stride(cv_params.output_stride) << ", "
           << GenCvUint16Stride(cv_params.input_stride) << ");" << std::endl;
      } else {
        ss << "CompareScalarExtend<" << dtype_name << ", 1, CMPMODE::" << this->api_name_ << ">(" << y << "["
           << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x1 << "["
           << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], " << x2_scalar << ", "
           << "{static_cast<uint16_t>(" << actual_size << ")}, {static_cast<uint16_t>(1)}, {static_cast<uint16_t>(1)});"
           << std::endl;
      }
    } else {
      std::stringstream ss1;
      size_t input0_strides_size = param.inputs_strides[0].size();
      std::vector<ascir::SizeExpr> inner0_input_strides(param.inputs_strides[0].begin(),
                                                        param.inputs_strides[0].begin() + input0_strides_size - 1);
      std::string input0_inner_offset = input0_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner0_input_strides);

      size_t output_strides_size = param.outputs_strides[0].size();
      std::vector<ascir::SizeExpr> inner_output_strides(param.outputs_strides[0].begin(),
                                                        param.outputs_strides[0].begin() + output_strides_size - 1);
      std::string output_inner_offset = output_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner_output_strides);

      ss1 << "CompareExtend<" << dtype_name << ",2, CMPMODE::" << this->api_name_ << ">(" << y << "["
          << output_inner_offset << "], " << x1 << "[" << input0_inner_offset << "], "
          << scalar_local_blk_tensor_name_x2 << "[0], "
          << "{static_cast<uint16_t>(" << param.outer_repeats[outer_repeats_size - 1] << "), static_cast<uint16_t>("
          << tpipe.tiler.ActualSize(param.cal_count) << ")}, "
          << "{static_cast<uint16_t>(" << tpipe.tiler.Size(param.output_second_to_last_stride) << "), "
          << "static_cast<uint16_t>(1)" << "}, "
          << "{static_cast<uint16_t>(" << tpipe.tiler.Size(param.input_second_to_last_stride)
          << "), static_cast<uint16_t>(" << "1" << ")});" << std::endl;
      CreateComputeNodeOuterForIfRequired(outer_repeats_size, param, ss1, ss);
    }
  } else {
    const std::string actual_size = x1.actual_size.Str();
    ub_inputs.push_back(x1);
    ub_inputs.push_back(x2);
    ub_outputs.push_back(y);
    bool status = GenerateVectorizedAxisMergeStatus(ub_inputs, ub_outputs, merge_info, tpipe);
    GE_ASSERT_TRUE(status, "GenerateVectorizedAxisMergeStatus failed");
    SaveApiLoopAxisParams(merge_info, param);
    std::vector<ge::Expression> output_dims;
    std::vector<ge::Expression> output_strides;
    std::vector<ge::Expression> input_strides;
    ge::Expression outer_call_count = af::ops::One;
    BuildCompareLoopParams(merge_info, param, outer_call_count, output_dims, output_strides, input_strides);
    GE_ASSERT_SUCCESS(
        FillCompareNodeParams(this->node, false, outer_call_count, output_dims, output_strides, input_strides));
    size_t outer_repeats_size = param.outer_repeats.size();
    if (outer_repeats_size == 0U) {
      if (IsCVFusionStage(this->api_call_context)) {
        const auto cv_params = BuildCvApi2DParams(tpipe, x1, y);
        ss << "CompareExtend<" << dtype_name << ", 2, CMPMODE::" << this->api_name_ << ">(" << y << "["
           << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x1 << "["
           << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], " << x2 << "["
           << tpipe.tiler.TensorVectorizedOffset(current_axis, x2) << "], " << GenCvUint16Dims(cv_params) << ", "
           << GenCvUint16Stride(cv_params.output_stride) << ", " << GenCvUint16Stride(cv_params.input_stride) << ");"
           << std::endl;
      } else {
        ss << "CompareExtend<" << dtype_name << ", 1, CMPMODE::" << this->api_name_ << ">(" << y << "["
           << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x1 << "["
           << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], " << x2 << "["
           << tpipe.tiler.TensorVectorizedOffset(current_axis, x2) << "], "
           << "{static_cast<uint16_t>(" << actual_size << ")}, {static_cast<uint16_t>(1)}, {static_cast<uint16_t>(1)});"
           << std::endl;
      }
    } else {
      size_t input0_strides_size = param.inputs_strides[0].size();
      std::vector<ascir::SizeExpr> inner0_input_strides(param.inputs_strides[0].begin(),
                                                        param.inputs_strides[0].begin() + input0_strides_size - 1);
      std::string input0_inner_offset = input0_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner0_input_strides);

      size_t input1_strides_size = param.inputs_strides[1].size();
      std::vector<ascir::SizeExpr> inner1_input_strides(param.inputs_strides[1].begin(),
                                                        param.inputs_strides[1].begin() + input1_strides_size - 1);
      std::string input1_inner_offset = input1_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner1_input_strides);

      size_t output_strides_size = param.outputs_strides[0].size();
      std::vector<ascir::SizeExpr> inner_output_strides(param.outputs_strides[0].begin(),
                                                        param.outputs_strides[0].begin() + output_strides_size - 1);
      std::string output_inner_offset = output_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner_output_strides);

      std::stringstream ss1;
      ss1 << "CompareExtend<" << dtype_name << ", 2, CMPMODE::" << this->api_name_ << ">(" << y << "["
          << output_inner_offset << "], " << x1 << "[" << input0_inner_offset << "], " << x2 << "["
          << input1_inner_offset << "], "
          << "{static_cast<uint16_t>(" << param.outer_repeats[outer_repeats_size - 1] << "), static_cast<uint16_t>("
          << tpipe.tiler.ActualSize(param.cal_count) << ")}, "
          << "{static_cast<uint16_t>(" << tpipe.tiler.Size(param.output_second_to_last_stride)
          << "), static_cast<uint16_t>(" << "1" << ")}, "
          << "{static_cast<uint16_t>(" << tpipe.tiler.Size(param.input_second_to_last_stride)
          << "), static_cast<uint16_t>(" << "1" << ")});" << std::endl;
      CreateComputeNodeOuterForIfRequired(outer_repeats_size, param, ss1, ss);
    }
  }

  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<CompareV2ApiCall> register_compare_v2_api_call("CompareV2ApiCall");

}  // namespace codegen
