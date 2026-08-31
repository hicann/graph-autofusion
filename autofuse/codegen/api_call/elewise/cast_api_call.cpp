/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cast_api_call.h"

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

static bool EnableCastMaskModeOptimize(const std::string &input_dtype, const std::string &output_dtype) {
  std::vector<std::pair<std::string, std::string>> src_cast_dst_lists = {
      // first is input dtype, second is output dtype
      {"uint8_t", "half"},    {"int64_t", "float"},    {"int64_t", "int32_t"},    {"half", "float"},
      {"half", "int32_t"},    {"half", "int16_t"},     {"half", "int8_t"},        {"half", "uint8_t"},
      {"half", "int4b_t"},    {"float", "half"},       {"float", "int64_t"},      {"float", "int32_t"},
      {"float", "int16_t"},   {"float", "bfloat16_t"}, {"int4b_t", "half"},       {"int16_t", "half"},
      {"int16_t", "float"},   {"int32_t", "float"},    {"int32_t", "int64_t"},    {"int32_t", "int16_t"},
      {"int32_t", "half"},    {"bfloat16_t", "float"}, {"bfloat16_t", "int32_t"}, {"uint8_t", "float"},
      {"uint8_t", "int32_t"}, {"uint8_t", "int16_t"},  {"uint8_t", "int8_t"},     {"uint8_t", "int4b_t"},
      {"int64_t", "half"},    {"half", "int64_t"}};
  std::pair<std::string, std::string> src_cast_dst = {input_dtype, output_dtype};
  GELOGD("current cast from %s to %s", input_dtype.c_str(), output_dtype.c_str());
  return std::find(src_cast_dst_lists.begin(), src_cast_dst_lists.end(), src_cast_dst) != src_cast_dst_lists.end();
}

Status CastApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                             const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                             const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                             std::string &result) const {
  (void)this->api_name_;
  auto x = inputs[0].get();
  auto y = outputs[0].get();
  GELOGD("x, is_constant:%d", static_cast<int32_t>(x.is_constant));
  GELOGI("cast x_dtype:%d, y.dtype:%d.", static_cast<int32_t>(x.dtype), static_cast<int32_t>(y.dtype));
  GE_ASSERT_TRUE((x.dtype != y.dtype), "cast x_dtype:%d, y.dtype:%d", static_cast<int32_t>(x.dtype),
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
  stringstream ss;

  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "CastApiCall cannot find tmp buffer id to use.");
  id = it->second;

  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, {}, tpipe.tmp_buf.name + "_" + std::to_string(id));

  size_t outer_repeats_size = param.outer_repeats.size();
  std::string scalar_local_blk_tensor_name = "local_blk_tensor_of_" + x.name;
  if (outer_repeats_size == 0U) {
    GELOGD("outer_repeats_size is 0, x_dtype = %s, y_dtype = %s", x_dtype.c_str(), y_dtype.c_str());
    if (x.is_constant) {
      ss << "CastExtend(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
         << scalar_local_blk_tensor_name << "[0], " << tpipe.tmp_buf << "_" << std::to_string(id) << ", "
         << y.actual_size << ");" << std::endl;
    } else {
      ss << "CastExtend(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x << "["
         << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], " << tpipe.tmp_buf << "_" << std::to_string(id)
         << ", " << x.actual_size << ");" << std::endl;
    }
  } else {
    if (EnableCastMaskModeOptimize(x_dtype, y_dtype)) {
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
      ss1 << "CastExtend(" << y << "[" << output_inner_offset << "], " << x << "[" << input_inner_offset << "], "
          << tpipe.tmp_buf << "_" << std::to_string(id) << ", " << param.outer_repeats[outer_repeats_size - 1] << ", "
          << tpipe.tiler.ActualSize(param.cal_count) << ", " << tpipe.tiler.Size(param.input_second_to_last_stride)
          << ", " << tpipe.tiler.Size(param.output_second_to_last_stride) << ", " << dtype_size << ");" << std::endl;
      if (outer_repeats_size == 1U) {
        ss << ss1.str();
      } else {
        std::vector<std::string> inner_outer_repeats(param.outer_repeats.begin(),
                                                     param.outer_repeats.begin() + outer_repeats_size - 1);
        CreateComputeNodeOuterFor(inner_outer_repeats, ss1, ss, 0);
      }
    } else {
      GELOGD("not enable cast mask mode optimize, x_dtype = %s, y_dtype = %s", x_dtype.c_str(), y_dtype.c_str());
      std::string input_inner_offset = CalcInnerOffset(tpipe, param.inputs_strides[0]);
      std::string output_inner_offset = CalcInnerOffset(tpipe, param.outputs_strides[0]);
      std::stringstream ss1;
      ss1 << "CastExtend(" << y << "[" << output_inner_offset << "], " << x << "[" << input_inner_offset << "], "
          << tpipe.tmp_buf << "_" << std::to_string(id) << ", " << tpipe.tiler.ActualSize(param.cal_count) << ");"
          << std::endl;
      CreateComputeNodeOuterFor(param.outer_repeats, ss1, ss, 0);
    }
  }

  result = ss.str();
  return af::SUCCESS;
}
static ApiCallRegister<CastApiCall> register_cast_api_call("CastApiCall");

}  // namespace codegen
