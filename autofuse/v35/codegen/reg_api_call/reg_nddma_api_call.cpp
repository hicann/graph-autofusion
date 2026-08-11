/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "reg_api_call_utils.h"
#include "reg_nddma_api_call.h"
#include "codegen/expression_convert_struct.h"

using namespace af::ops;
using namespace af::ascir_op;

namespace {
constexpr uint64_t kNddmaMaxLen = 5UL;
}

namespace codegen {
namespace {
bool IsZeroStride(const Tensor &tensor, size_t index) {
  return index < tensor.axis_strides.size() && tensor.axis_strides[index] == Zero;
}

void AppendCvNddmaCall(std::stringstream &ss, const std::string &api_name, const Tensor &gm, const Tensor &ub,
                       const std::string &gm_offset) {
  ss << api_name << "(" << ub << ", " << gm << "[" << gm_offset << "], " << "output_dims_" << ub.id << ", "
     << "output_stride_" << ub.id << ", " << "input_stride_" << ub.id << ");" << std::endl;
}

Status GenerateInductorUbFuseNddma(const std::string &api_name, const Tensor &gm, const Tensor &ub,
                                   std::stringstream &ss) {
  GE_ASSERT_TRUE(gm.axis_strides.size() >= 2U, "Nddma src axis-strides less than 2 is invalid in CV-Fusion case");
  auto last_index = gm.axis_strides.size() - 1U;
  auto second_to_last_index = gm.axis_strides.size() - 2U;
  ss << "const int64_t output_dims_" << ub.id << "[2] = {curAivM, curAlignN};" << std::endl;
  std::string gm_offset;
  if (IsZeroStride(gm, second_to_last_index) && IsZeroStride(gm, last_index)) {
    ss << "const int64_t input_stride_" << ub.id << "[2] = {0, 0};" << std::endl;
    ss << "const int64_t output_stride_" << ub.id << "[2] = {curAlignN, 1};" << std::endl;
    gm_offset = "batch_num";
  } else if (IsZeroStride(gm, second_to_last_index) && !IsZeroStride(gm, last_index)) {
    ss << "const int64_t input_stride_" << ub.id << "[2] = {0, 1};" << std::endl;
    ss << "const int64_t output_stride_" << ub.id << "[2] = {curAlignN, 1};" << std::endl;
    gm_offset = "offset % shapeN + batch_num * shapeN";
  } else if (!IsZeroStride(gm, second_to_last_index) && IsZeroStride(gm, last_index)) {
    ss << "const int64_t input_stride_" << ub.id << "[2] = {1, 0};" << std::endl;
    ss << "const int64_t output_stride_" << ub.id << "[2] = {curAlignN, 1};" << std::endl;
    gm_offset = "offset / shapeN";
  } else {
    ss << "const int64_t input_stride_" << ub.id << "[2] = {0, 1};" << std::endl;
    ss << "const int64_t output_stride_" << ub.id << "[2] = {curAlignN, 1};" << std::endl;
    gm_offset = "offset % shapeN + batch_num * shapeN";
  }
  AppendCvNddmaCall(ss, api_name, gm, ub, gm_offset);
  return af::SUCCESS;
}

Status GenerateUbFuseNddma(const std::string &api_name, const Tensor &gm, const Tensor &ub, std::stringstream &ss) {
  GE_ASSERT_TRUE(gm.axis_size.size() >= 2U, "Nddma src axis-size less than 2 is invalid in CV-Fusion case");
  auto last_index = gm.axis_size.size() - 1U;
  auto second_to_last_index = gm.axis_size.size() - 2U;
  ss << "const int64_t output_dims_" << ub.id << "[2] = {curAivM, curAlignN};" << std::endl;
  std::string gm_offset;
  if ((gm.axis_size[second_to_last_index] == One) && (gm.axis_size[last_index] == One)) {
    ss << "const int64_t input_stride_" << ub.id << "[2] = {0, 0};" << std::endl;
    ss << "const int64_t output_stride_" << ub.id << "[2] = {curAlignN, 1};" << std::endl;
    gm_offset = "batch_num";
  } else if (gm.axis_size[second_to_last_index] == One) {
    ss << "const int64_t input_stride_" << ub.id << "[2] = {0, 1};" << std::endl;
    ss << "const int64_t output_stride_" << ub.id << "[2] = {curAlignN, 1};" << std::endl;
    gm_offset = "offset % shapeN + batch_num * shapeN";
  } else if (gm.axis_size[last_index] == One) {
    ss << "const int64_t input_stride_" << ub.id << "[2] = {1, 0};" << std::endl;
    ss << "const int64_t output_stride_" << ub.id << "[2] = {curAlignN, 1};" << std::endl;
    gm_offset = "offset / shapeN";
  }
  AppendCvNddmaCall(ss, api_name, gm, ub, gm_offset);
  return af::SUCCESS;
}

Status GenerateDefaultNddma(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                            const std::string &api_name, const Tensor &gm, const Tensor &ub,
                            const af::Expression &offset, std::stringstream &ss) {
  DataCopyParams param;
  GE_ASSERT_TRUE(CalculateDmaParams(tpipe, ub, ub, param, true), "CalculateDmaParams failed");
  const std::string gm_offset =
      ub.is_ub_scalar ? "0" : tpipe.tiler.Offset(current_axis, ub.axis, ub.axis_strides);  // 每次从gm搬到ub的偏移量
  if (param.repeats.size() <= kNddmaMaxLen) {
    NddmaParams nddma_param;
    SetNddmaParams(tpipe, param, nddma_param, ub.id, ss);
    ss << api_name << "(" << ub << ", " << gm << "[" << gm_offset << " + " << tpipe.tiler.Size(offset) << "], "
       << "output_dims_" << ub.id << ", " << "output_stride_" << ub.id << ", " << "input_stride_" << ub.id << ");"
       << std::endl;
  } else {
    CreateNddmaCall(tpipe, gm, ub, gm_offset, param, offset, ss);
  }
  return af::SUCCESS;
}
}  // namespace

Status NddmaApiCall::ParseAttr(const ascir::NodeView &node) {
  (void)node->attr.ir_attr->GetAttrValue("offset", offset_);
  return af::SUCCESS;
}

Status NddmaApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                              const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                              const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                              std::string &result) const {
  std::stringstream ss;
  const auto &gm = inputs[0].get();
  const auto &ub = outputs[0].get();
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs);
  if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse && tpipe.is_inductor) {
    GE_ASSERT_SUCCESS(GenerateInductorUbFuseNddma(api_name_, gm, ub, ss));
  } else if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    GE_ASSERT_SUCCESS(GenerateUbFuseNddma(api_name_, gm, ub, ss));
  } else {
    GE_ASSERT_SUCCESS(GenerateDefaultNddma(tpipe, current_axis, api_name_, gm, ub, offset_, ss));
  }
  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<NddmaApiCall> register_nddma_api_call("NddmaApiCall");
}  // namespace codegen
