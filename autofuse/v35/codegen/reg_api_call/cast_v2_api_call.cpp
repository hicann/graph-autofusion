/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cast_v2_api_call.h"

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
af::Status FillCastNodeParams(const af::AscNodePtr &node, const std::vector<ge::Expression> &output_dims,
                              const std::vector<ge::Expression> &output_strides,
                              const std::vector<ge::Expression> &input_strides) {
  GE_ASSERT_NOTNULL(node);
  auto params = ascir_param::GetOrCreateAscirNodeParams(node);

  auto *cast_params = std::get_if<ascir_param::CastNodeParams>(&params->specific_params);
  if (cast_params == nullptr) {
    params->specific_params = ascir_param::CastNodeParams{};
    cast_params = std::get_if<ascir_param::CastNodeParams>(&params->specific_params);
  }
  GE_ASSERT_NOTNULL(cast_params, "Cast specific params is null, node[%s].", node->GetNamePtr());
  params->api_name = node->GetType();
  params->status = ascir_param::ParamBuildStatus::kBuilt;

  *cast_params = ascir_param::CastNodeParams{};
  cast_params->valid = true;
  cast_params->output_dims = output_dims;
  cast_params->output_strides = output_strides;
  cast_params->input_strides = input_strides;
  return af::SUCCESS;
}
}  // namespace

Status CastV2ApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                               const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                               const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                               std::string &result) const {
  (void)this->api_name_;
  auto x = inputs[0].get();
  auto y = outputs[0].get();
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs);
  GELOGD("x, is_constant:%d", static_cast<int32_t>(x.is_constant));
  GELOGI("cast x_dtype:%d, y.dtype:%d.", static_cast<int32_t>(x.dtype), static_cast<int32_t>(y.dtype));
  GE_ASSERT_TRUE((x.dtype != y.dtype), "cast s_dtype:%d, y.dtype:%d", static_cast<int32_t>(x.dtype),
                 static_cast<int32_t>(y.dtype));
  // 通过 src_dtype 和 dst_dtype 获取 mode
  std::string x_dtype;
  std::string y_dtype;
  GE_CHK_STATUS_RET(Tensor::DtypeName(x.dtype, x_dtype), "get name of dtype:%d failed", static_cast<int32_t>(x.dtype));
  GE_CHK_STATUS_RET(Tensor::DtypeName(y.dtype, y_dtype), "get name of dtype:%d failed", static_cast<int32_t>(y.dtype));
  ApiLoopParams param;
  std::vector<Tensor> ub_inputs;
  std::vector<Tensor> ub_outputs;
  if (!x.is_constant) {
    ub_inputs.push_back(x);
  }
  ub_outputs.push_back(y);
  VectorizedAxisLoopMergeStatus merge_info;
  bool status = GenerateVectorizedAxisMergeStatus(ub_inputs, ub_outputs, merge_info, tpipe);
  GE_ASSERT_TRUE(status, "GenerateVectorizedAxisMergeStatus failed");
  SaveApiLoopAxisParams(merge_info, param);
  std::vector<ge::Expression> output_dims;
  std::vector<ge::Expression> output_strides;
  std::vector<ge::Expression> input_strides;
  if (param.outer_repeats.empty()) {
    if (!merge_info.merge_repeats.empty()) {
      output_dims.emplace_back(merge_info.merge_repeats.back());
    }
    output_strides.emplace_back(af::ops::One);
    input_strides.emplace_back(af::ops::One);
  } else {
    const size_t merge_repeats_size = merge_info.merge_repeats.size();
    ge::Expression outer_call_count = merge_info.merge_repeats[0];
    for (size_t i = 1; i + 1 < merge_repeats_size; ++i) {
      outer_call_count = outer_call_count * merge_info.merge_repeats[i];
    }
    output_dims.emplace_back(outer_call_count);
    output_dims.emplace_back(param.cal_count);
    output_strides.emplace_back(param.output_second_to_last_stride);
    output_strides.emplace_back(af::ops::One);
    input_strides.emplace_back(param.input_second_to_last_stride);
    input_strides.emplace_back(af::ops::One);
  }
  GE_ASSERT_SUCCESS(FillCastNodeParams(this->node, output_dims, output_strides, input_strides));
  stringstream ss;
  if (IsCVFusionStage(this->api_call_context)) {
    const auto cv_params = BuildCvApi2DParams(tpipe, x, y);
    const std::string input_tensor = x.is_constant ? ("local_blk_tensor_of_" + x.name) : x.Str();
    ss << y.actual_size << " = " << cv_params.first_dim << " * " << cv_params.last_dim << ";" << std::endl;
    ss << this->api_name_ << "(" << y << "[0], " << input_tensor << "[0], " << GenCvUint32Dims(cv_params) << ", "
       << GenCvUint32Stride(cv_params.output_stride) << ", " << GenCvUint32Stride(cv_params.input_stride) << ");"
       << std::endl;
    result = ss.str();
    return af::SUCCESS;
  }

  size_t outer_repeats_size = param.outer_repeats.size();
  std::string scalar_local_blk_tensor_name = "local_blk_tensor_of_" + x.name;
  if (outer_repeats_size == 0U) {
    GELOGD("outer_repeats_size is 0, x_dtype = %s, y_dtype = %s", x_dtype.c_str(), y_dtype.c_str());
    if (x.is_constant) {
      ss << this->api_name_ << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
         << scalar_local_blk_tensor_name << "[0], "
         << "{" << "ConvertToUint32(" << y.actual_size << ")" << "}, "
         << "{ConvertToUint32(1)}, {ConvertToUint32(1)});" << std::endl;
    } else {
      ss << this->api_name_ << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x
         << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
         << "{" << "ConvertToUint32(" << x.actual_size << ")" << "}, "
         << "{ConvertToUint32(1)}, {ConvertToUint32(1)});" << std::endl;
    }
  } else {
    GELOGD("enable cast mask mode optimize, x_dtype = %s, y_dtype = %s", x_dtype.c_str(), y_dtype.c_str());
    // 获取输入输出中最大的数据类型max_dtype_size
    std::string dtype_size;
    GE_CHK_BOOL_RET_STATUS(GetMaxDtypeSize(x.dtype, y.dtype, dtype_size) == true, af::FAILED,
                           "get max data type size failed, x_dtype = %s, y_dtype = %s", x_dtype.c_str(),
                           y_dtype.c_str());
    size_t input_strides_size = param.inputs_strides[0].size();
    std::vector<ascir::SizeExpr> inner_input_strides(param.inputs_strides[0].begin(),
                                                     param.inputs_strides[0].begin() + input_strides_size - 1);
    std::string input_inner_offset = input_strides_size == 1U ? "0" : CalcInnerOffset(tpipe, inner_input_strides);

    size_t output_strides_size = param.outputs_strides[0].size();
    std::vector<ascir::SizeExpr> inner_output_strides(param.outputs_strides[0].begin(),
                                                      param.outputs_strides[0].begin() + output_strides_size - 1);
    std::string output_inner_offset = output_strides_size == 1U ? "0" : CalcInnerOffset(tpipe, inner_output_strides);

    std::stringstream ss1;
    ss1 << this->api_name_ << "(" << y << "[" << output_inner_offset << "], " << x << "[" << input_inner_offset << "], "
        << "{" << "ConvertToUint32(" << param.outer_repeats[outer_repeats_size - 1] << ")" << ", " << "ConvertToUint32("
        << tpipe.tiler.ActualSize(param.cal_count) << ")" << "}, "
        << "{" << "ConvertToUint32(" << tpipe.tiler.Size(param.output_second_to_last_stride) << ")" << ", "
        << "ConvertToUint32(1)" << "}, "
        << "{" << "ConvertToUint32(" << tpipe.tiler.Size(param.input_second_to_last_stride) << ")" << ", "
        << "ConvertToUint32(1)" << "});" << std::endl;
    if (outer_repeats_size == 1U) {
      ss << ss1.str();
    } else {
      std::vector<std::string> inner_outer_repeats(param.outer_repeats.begin(),
                                                   param.outer_repeats.begin() + outer_repeats_size - 1);
      CreateComputeNodeOuterFor(inner_outer_repeats, ss1, ss, 0);
    }
  }

  result = ss.str();
  return af::SUCCESS;
}
static ApiCallRegister<CastV2ApiCall> register_cast_v2_api_call("CastV2ApiCall");

}  // namespace codegen
