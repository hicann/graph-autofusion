/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "unary_bitwidth_change_api_call.h"

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

Status UnaryBitWidthChangeApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                            const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                            const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                            std::string &result) const {
  auto x = inputs[0].get();
  auto y = outputs[0].get();
  GE_ASSERT_TRUE((x.dtype != y.dtype), "cast s_dtype:%d, y.dtype:%d", static_cast<int32_t>(x.dtype),
                 static_cast<int32_t>(y.dtype));

  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "UnaryBitWidthChangeApiCall cannot find tmp buffer id to use.");
  id = it->second;

  ApiLoopParams param;
  VectorizedAxisLoopMergeStatus merge_info;
  std::vector<Tensor> ub_inputs;
  std::vector<Tensor> ub_outputs;
  ub_inputs.push_back(x);
  ub_outputs.push_back(y);
  bool status = GenerateVectorizedAxisMergeStatus(ub_inputs, ub_outputs, merge_info, tpipe);
  GE_ASSERT_TRUE(status, "GenerateVectorizedAxisMergeStatus failed");
  SaveApiLoopAxisParams(merge_info, param);
  stringstream ss;
  if (param.outer_repeats.size() == 0) {
    (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, CombinedExprFactory::SymbolVar(x.actual_size.Str()),
                                 tpipe.tmp_buf.name + "_" + std::to_string(id));
    ss << this->api_name_ << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x << "["
       << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], " << tpipe.tmp_buf << "_" << std::to_string(id)
       << ", " << x.actual_size << ");" << std::endl;
  } else {
    (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs,
                                 CombinedExprFactory::SymbolVar(tpipe.tiler.ActualSize(param.cal_count)),
                                 tpipe.tmp_buf.name + "_" + std::to_string(id));
    std::string input_inner_offset = CalcInnerOffset(tpipe, param.inputs_strides[0]);
    std::string output_inner_offset = CalcInnerOffset(tpipe, param.outputs_strides[0]);
    std::stringstream ss1;
    ss1 << this->api_name_ << "(" << y << "[" << output_inner_offset << "], " << x << "[" << input_inner_offset << "], "
        << tpipe.tmp_buf << "_" << std::to_string(id) << ", " << tpipe.tiler.ActualSize(param.cal_count) << ");"
        << std::endl;
    CreateComputeNodeOuterFor(param.outer_repeats, ss1, ss, 0);
  }

  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<UnaryBitWidthChangeApiCall> register_unary_bitwidth_change_api_call(
    "UnaryBitWidthChangeApiCall");

}  // namespace codegen
