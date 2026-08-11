/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reg_where_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "api_call/utils/api_call_utils.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

Status WhereRegApiCall::PrepareInputsAndOutputs(const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                                const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                                const Tensor *&x1, const Tensor *&x2, const Tensor *&x3,
                                                const Tensor *&y) const {
  size_t x1_idx = 0;
  size_t x2_idx = 1;
  size_t x3_idx = 2;
  x1 = &inputs[x1_idx].get();
  x2 = &inputs[x2_idx].get();
  x3 = &inputs[x3_idx].get();
  y = &outputs[0].get();

  GELOGD("x2, is_constant:%d, is_ub_scalar:%d, need_gen_get_value_of_ub_scalar:%d",
         static_cast<int32_t>(x2->is_constant), static_cast<int32_t>(x2->is_ub_scalar),
         static_cast<int32_t>(x2->need_gen_get_value_of_ub_scalar));
  GELOGD("x3, is_constant:%d, is_ub_scalar:%d, need_gen_get_value_of_ub_scalar:%d",
         static_cast<int32_t>(x3->is_constant), static_cast<int32_t>(x3->is_ub_scalar),
         static_cast<int32_t>(x3->need_gen_get_value_of_ub_scalar));

  return af::SUCCESS;
}

Status WhereRegApiCall::GenerateLoopParams(const Tensor &x1, const Tensor &x2, const Tensor &x3, const Tensor &y,
                                           const TPipe &tpipe, ApiLoopParams &param) const {
  VectorizedAxisLoopMergeStatus merge_info;
  std::vector<Tensor> ub_inputs;
  std::vector<Tensor> ub_outputs;

  ub_inputs.push_back(x1);
  if (!x2.is_constant && !x2.need_gen_get_value_of_ub_scalar) {
    ub_inputs.push_back(x2);
  }
  if (!x3.is_constant && !x3.need_gen_get_value_of_ub_scalar) {
    ub_inputs.push_back(x3);
  }
  ub_outputs.push_back(y);

  bool status = GenerateVectorizedAxisMergeStatus(ub_inputs, ub_outputs, merge_info, tpipe);
  GE_ASSERT_TRUE(status, "GenerateVectorizedAxisMergeStatus failed");
  SaveApiLoopAxisParams(merge_info, param);

  return af::SUCCESS;
}

Status WhereRegApiCall::GenerateNoLoopCase(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                           const Tensor &x1, const Tensor &x2, const Tensor &x3, const Tensor &y,
                                           const std::string &x2_scalar, const std::string &x3_scalar,
                                           std::stringstream &ss) const {
  const bool x2_is_scalar_scene = x2.IsAnyScalar();
  const bool x3_is_scalar_scene = x3.IsAnyScalar();

  ss << this->api_name_ << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], ";
  ss << x1 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], ";

  if (x2_is_scalar_scene) {
    ss << x2_scalar << ", ";
  } else {
    ss << x2 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x2) << "], ";
  }

  if (x3_is_scalar_scene) {
    ss << x3_scalar << ", ";
  } else {
    ss << x3 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x3) << "], ";
  }

  ss << x1.actual_size << ");" << std::endl;

  return af::SUCCESS;
}

Status WhereRegApiCall::GenerateBothScalarCase(const TPipe &tpipe, const ApiLoopParams &param, const Tensor &x1,
                                               const Tensor &y, const std::string &scalar_local_blk_tensor_name_x2,
                                               const std::string &scalar_local_blk_tensor_name_x3,
                                               std::stringstream &ss) const {
  stringstream ss1;

  size_t output_strides_size = param.outputs_strides[0].size();
  std::vector<ascir::SizeExpr> inner_output_strides(param.outputs_strides[0].begin(),
                                                    param.outputs_strides[0].begin() + output_strides_size - 1);
  std::string output_inner_offset = output_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner_output_strides);

  uint32_t index = 0U;
  size_t input0_strides_size = param.inputs_strides[index].size();
  std::vector<ascir::SizeExpr> inner0_input_strides(param.inputs_strides[index].begin(),
                                                    param.inputs_strides[index].begin() + input0_strides_size - 1);
  std::string input0_inner_offset = input0_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner0_input_strides);

  ss1 << this->api_name_ << "<true, true>(" << y << "[" << output_inner_offset << "], " << x1 << "["
      << input0_inner_offset << "], " << scalar_local_blk_tensor_name_x2 << "[0], " << scalar_local_blk_tensor_name_x3
      << "[0], "
      << "{static_cast<uint16_t>(" << param.outer_repeats[param.outer_repeats.size() - 1] << "), static_cast<uint16_t>("
      << tpipe.tiler.ActualSize(param.cal_count) << ")}, "
      << "{static_cast<uint16_t>(" << tpipe.tiler.Size(param.output_second_to_last_stride)
      << "), static_cast<uint16_t>(1)" << "}, "
      << "{static_cast<uint16_t>(" << tpipe.tiler.Size(param.input_second_to_last_stride)
      << "), static_cast<uint16_t>(1)" << "}, "
      << "{static_cast<uint16_t>(" << tpipe.tiler.Size(param.output_second_to_last_stride)
      << "), static_cast<uint16_t>(1)" << "});" << std::endl;

  if (param.outer_repeats.size() == 1) {
    ss << ss1.str();
  } else {
    CreateComputeNodeOuterFor(param.outer_repeats, ss1, ss, 0);
  }

  return af::SUCCESS;
}

Status WhereRegApiCall::GenerateX2ScalarCase(const TPipe &tpipe, const ApiLoopParams &param, const Tensor &x1,
                                             const Tensor &x3, const Tensor &y,
                                             const std::string &scalar_local_blk_tensor_name_x2,
                                             std::stringstream &ss) const {
  stringstream ss1;

  size_t output_strides_size = param.outputs_strides[0].size();
  std::vector<ascir::SizeExpr> inner_output_strides(param.outputs_strides[0].begin(),
                                                    param.outputs_strides[0].begin() + output_strides_size - 1);
  std::string output_inner_offset = output_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner_output_strides);

  uint32_t index = 0U;
  size_t input0_strides_size = param.inputs_strides[index].size();
  std::vector<ascir::SizeExpr> inner0_input_strides(param.inputs_strides[index].begin(),
                                                    param.inputs_strides[index].begin() + input0_strides_size - 1);
  std::string input0_inner_offset = input0_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner0_input_strides);

  index++;
  size_t input2_strides_size = param.inputs_strides[index].size();
  std::vector<ascir::SizeExpr> inner2_input_strides(param.inputs_strides[index].begin(),
                                                    param.inputs_strides[index].begin() + input2_strides_size - 1);
  std::string input2_inner_offset = input2_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner2_input_strides);

  ss1 << this->api_name_ << "<true, false>(" << y << "[" << output_inner_offset << "], " << x1 << "["
      << input0_inner_offset << "], " << scalar_local_blk_tensor_name_x2 << "[0], " << x3 << "[" << input2_inner_offset
      << "], "
      << "{static_cast<uint16_t>(" << param.outer_repeats[param.outer_repeats.size() - 1] << "), static_cast<uint16_t>("
      << tpipe.tiler.ActualSize(param.cal_count) << ")}, "
      << "{static_cast<uint16_t>(" << tpipe.tiler.Size(param.output_second_to_last_stride)
      << "), static_cast<uint16_t>(1)" << "}, "
      << "{static_cast<uint16_t>(" << tpipe.tiler.Size(param.input_second_to_last_stride)
      << "), static_cast<uint16_t>(1)" << "}, "
      << "{static_cast<uint16_t>(" << tpipe.tiler.Size(param.output_second_to_last_stride)
      << "), static_cast<uint16_t>(1)" << "});" << std::endl;

  if (param.outer_repeats.size() == 1) {
    ss << ss1.str();
  } else {
    CreateComputeNodeOuterFor(param.outer_repeats, ss1, ss, 0);
  }

  return af::SUCCESS;
}

Status WhereRegApiCall::GenerateX3ScalarCase(const TPipe &tpipe, const ApiLoopParams &param, const Tensor &x1,
                                             const Tensor &x2, const Tensor &y,
                                             const std::string &scalar_local_blk_tensor_name_x3,
                                             std::stringstream &ss) const {
  stringstream ss1;

  size_t output_strides_size = param.outputs_strides[0].size();
  std::vector<ascir::SizeExpr> inner_output_strides(param.outputs_strides[0].begin(),
                                                    param.outputs_strides[0].begin() + output_strides_size - 1);
  std::string output_inner_offset = output_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner_output_strides);

  uint32_t index = 0U;
  size_t input0_strides_size = param.inputs_strides[index].size();
  std::vector<ascir::SizeExpr> inner0_input_strides(param.inputs_strides[index].begin(),
                                                    param.inputs_strides[index].begin() + input0_strides_size - 1);
  std::string input0_inner_offset = input0_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner0_input_strides);

  index++;
  size_t input1_strides_size = param.inputs_strides[index].size();
  std::vector<ascir::SizeExpr> inner1_input_strides(param.inputs_strides[index].begin(),
                                                    param.inputs_strides[index].begin() + input1_strides_size - 1);
  std::string input1_inner_offset = input1_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner1_input_strides);

  ss1 << this->api_name_ << "<false, true>(" << y << "[" << output_inner_offset << "], " << x1 << "["
      << input0_inner_offset << "], " << x2 << "[" << input1_inner_offset << "], " << scalar_local_blk_tensor_name_x3
      << "[0], "
      << "{static_cast<uint16_t>(" << param.outer_repeats[param.outer_repeats.size() - 1] << "), static_cast<uint16_t>("
      << tpipe.tiler.ActualSize(param.cal_count) << ")}, "
      << "{static_cast<uint16_t>(" << tpipe.tiler.Size(param.output_second_to_last_stride)
      << "), static_cast<uint16_t>(1)" << "}, "
      << "{static_cast<uint16_t>(" << tpipe.tiler.Size(param.input_second_to_last_stride)
      << "), static_cast<uint16_t>(1)" << "}, "
      << "{static_cast<uint16_t>(" << tpipe.tiler.Size(param.output_second_to_last_stride)
      << "), static_cast<uint16_t>(1)" << "});" << std::endl;

  if (param.outer_repeats.size() == 1) {
    ss << ss1.str();
  } else {
    CreateComputeNodeOuterFor(param.outer_repeats, ss1, ss, 0);
  }

  return af::SUCCESS;
}

Status WhereRegApiCall::GenerateNormalCase(const TPipe &tpipe, const ApiLoopParams &param, const Tensor &x1,
                                           const Tensor &x2, const Tensor &x3, const Tensor &y,
                                           std::stringstream &ss) const {
  stringstream ss1;

  size_t output_strides_size = param.outputs_strides[0].size();
  std::vector<ascir::SizeExpr> inner_output_strides(param.outputs_strides[0].begin(),
                                                    param.outputs_strides[0].begin() + output_strides_size - 1);
  std::string output_inner_offset = output_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner_output_strides);

  uint32_t index = 0U;
  size_t input0_strides_size = param.inputs_strides[index].size();
  std::vector<ascir::SizeExpr> inner0_input_strides(param.inputs_strides[index].begin(),
                                                    param.inputs_strides[index].begin() + input0_strides_size - 1);
  std::string input0_inner_offset = input0_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner0_input_strides);

  index++;
  size_t input1_strides_size = param.inputs_strides[index].size();
  std::vector<ascir::SizeExpr> inner1_input_strides(param.inputs_strides[index].begin(),
                                                    param.inputs_strides[index].begin() + input1_strides_size - 1);
  std::string input1_inner_offset = input1_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner1_input_strides);

  index++;
  size_t input2_strides_size = param.inputs_strides[index].size();
  std::vector<ascir::SizeExpr> inner2_input_strides(param.inputs_strides[index].begin(),
                                                    param.inputs_strides[index].begin() + input2_strides_size - 1);
  std::string input2_inner_offset = input2_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner2_input_strides);

  ss1 << this->api_name_ << "<false, false>(" << y << "[" << output_inner_offset << "], " << x1 << "["
      << input0_inner_offset << "], " << x2 << "[" << input1_inner_offset << "], " << x3 << "[" << input2_inner_offset
      << "], "
      << "{static_cast<uint16_t>(" << param.outer_repeats[param.outer_repeats.size() - 1] << "), static_cast<uint16_t>("
      << tpipe.tiler.ActualSize(param.cal_count) << ")}, "
      << "{static_cast<uint16_t>(" << tpipe.tiler.Size(param.output_second_to_last_stride)
      << "), static_cast<uint16_t>(1)" << "}, "
      << "{static_cast<uint16_t>(" << tpipe.tiler.Size(param.input_second_to_last_stride)
      << "), static_cast<uint16_t>(1)" << "}, "
      << "{static_cast<uint16_t>(" << tpipe.tiler.Size(param.output_second_to_last_stride)
      << "), static_cast<uint16_t>(1)" << "});" << std::endl;

  if (param.outer_repeats.size() == 1) {
    ss << ss1.str();
  } else {
    CreateComputeNodeOuterFor(param.outer_repeats, ss1, ss, 0);
  }

  return af::SUCCESS;
}

Status WhereRegApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                 const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                 const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                 std::string &result) const {
  const Tensor *x1 = nullptr;
  const Tensor *x2 = nullptr;
  const Tensor *x3 = nullptr;
  const Tensor *y = nullptr;

  GE_CHK_STATUS_RET(PrepareInputsAndOutputs(inputs, outputs, x1, x2, x3, y));
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs);

  ApiLoopParams param;
  GE_CHK_STATUS_RET(GenerateLoopParams(*x1, *x2, *x3, *y, tpipe, param));

  stringstream ss;

  const bool x2_is_scalar_scene = x2->IsAnyScalar();
  const bool x3_is_scalar_scene = x3->IsAnyScalar();

  std::string x2_dtype_name;
  std::string x3_dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(x2->dtype, x2_dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(x2->dtype));
  GE_CHK_STATUS_RET(Tensor::DtypeName(x3->dtype, x3_dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(x3->dtype));
  GE_ASSERT_TRUE(x2_dtype_name == x3_dtype_name, "x2_dtype_name:%s, x3_dtype_name:%s", x2_dtype_name.c_str(),
                 x3_dtype_name.c_str());

  std::string x2_scalar =
      x2->need_gen_get_value_of_ub_scalar ? ("(" + x2_dtype_name + ")" + x2->ub_scalar_name) : x2->Str();
  std::string x3_scalar =
      x3->need_gen_get_value_of_ub_scalar ? ("(" + x3_dtype_name + ")" + x3->ub_scalar_name) : x3->Str();

  if (param.outer_repeats.size() == 0) {
    GE_CHK_STATUS_RET(GenerateNoLoopCase(tpipe, current_axis, *x1, *x2, *x3, *y, x2_scalar, x3_scalar, ss));
  } else if (x2_is_scalar_scene && x3_is_scalar_scene) {
    std::string scalar_local_blk_tensor_name_x2 = x2->IsConstScalar() ? "local_blk_tensor_of_" + x2->name : x2->name;
    std::string scalar_local_blk_tensor_name_x3 = x3->IsConstScalar() ? "local_blk_tensor_of_" + x3->name : x3->name;
    GE_CHK_STATUS_RET(GenerateBothScalarCase(tpipe, param, *x1, *y, scalar_local_blk_tensor_name_x2,
                                             scalar_local_blk_tensor_name_x3, ss));
  } else if (x2_is_scalar_scene) {
    std::string scalar_local_blk_tensor_name_x2 = x2->IsConstScalar() ? "local_blk_tensor_of_" + x2->name : x2->name;
    GE_CHK_STATUS_RET(GenerateX2ScalarCase(tpipe, param, *x1, *x3, *y, scalar_local_blk_tensor_name_x2, ss));
  } else if (x3_is_scalar_scene) {
    std::string scalar_local_blk_tensor_name_x3 = x3->IsConstScalar() ? "local_blk_tensor_of_" + x3->name : x3->name;
    GE_CHK_STATUS_RET(GenerateX3ScalarCase(tpipe, param, *x1, *x2, *y, scalar_local_blk_tensor_name_x3, ss));
  } else {
    GE_CHK_STATUS_RET(GenerateNormalCase(tpipe, param, *x1, *x2, *x3, *y, ss));
  }

  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<WhereRegApiCall> register_where_reg_api_call("WhereRegApiCall");
}  // namespace codegen
