/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "unary_bitwidth_change_api_call_v2.h"

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

Status UnaryBitWidthChangeApiCallV2::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                              const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                              const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                              std::string &result) const {
  auto x = inputs[0].get();
  auto y = outputs[0].get();
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs);
  stringstream ss;

  ss << "LocalTensor<bool> " << y << "_cast"
     << " = " << y << ".template ReinterpretCast<" << "bool" << ">();" << std::endl;

  ApiLoopParams param;
  VectorizedAxisLoopMergeStatus merge_info;
  std::vector<Tensor> ub_inputs;
  std::vector<Tensor> ub_outputs;
  ub_inputs.push_back(x);
  ub_outputs.push_back(y);
  bool status = GenerateVectorizedAxisMergeStatus(ub_inputs, ub_outputs, merge_info, tpipe);
  GE_ASSERT_TRUE(status, "GenerateVectorizedAxisMergeStatus failed");
  SaveApiLoopAxisParams(merge_info, param);
  if (param.outer_repeats.size() == 0) {
    ss << this->api_name_ << "(" << y << "_cast[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x
       << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], " << x.actual_size << ");" << std::endl;
  } else {
    std::string input_inner_offset = CalcInnerOffset(tpipe, param.inputs_strides[0]);
    std::string output_inner_offset = CalcInnerOffset(tpipe, param.outputs_strides[0]);
    std::stringstream ss1;
    ss1 << this->api_name_ << "(" << y << "_cast[" << output_inner_offset << "], " << x << "[" << input_inner_offset
        << "], " << tpipe.tiler.ActualSize(param.cal_count) << ");" << std::endl;
    CreateComputeNodeOuterFor(param.outer_repeats, ss1, ss, 0);
  }

  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<UnaryBitWidthChangeApiCallV2> register_unary_api_call("UnaryBitWidthChangeApiCallV2");
}  // namespace codegen
