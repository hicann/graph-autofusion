/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/
#include "micro_div_api_call.h"
#include "micro_api_call_factory.h"
#include "ascir_ops.h"
#include "optimize/pre_process/pre_process_config.h"

namespace codegen {
Status MicroDivApiCall::Generate(const codegen::TensorManager &tensor_mng, [[maybe_unused]] const TPipe &tpipe,
                                 CallParam &param, string &result) {
  std::stringstream ss;

  auto input_tensor_id = GetInputTensorIdByIndex(0);
  auto input_tensor0 = tensor_mng.GetTensor(input_tensor_id);
  GE_ASSERT_NOTNULL(input_tensor0, "input tensor is null, id: %d", input_tensor_id);

  auto input_dtype = input_tensor0->dtype_;

  string input_dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(input_dtype, input_dtype_name), "Get data type:%d failed",
                    static_cast<int32_t>(input_dtype));
  ss << "AscendC::MicroAPI::" << this->api_name_;
  if (input_dtype == ge::DT_FLOAT) {
    ss << "<" << input_dtype_name;
    if (!af::pre_process::PreProcessConfig::Instance().IsInImprovePrecisionBlacklist(af::ascir_op::Div::Type)) {
      ss << ", &high_precision_div_mode";
    }
    ss << ">";
  }
  ss << "(";
  for (const auto &out_arg : this->outputs_) {
    GE_ASSERT_NOTNULL(tensor_mng.GetTensor(out_arg.second));
    ss << *(tensor_mng.GetTensor(out_arg.second)) << ", ";
  }
  for (const auto &in_arg : this->inputs_) {
    GE_ASSERT_NOTNULL(tensor_mng.GetTensor(in_arg.second));
    ss << *(tensor_mng.GetTensor(in_arg.second)) << ", ";
  }
  ss << param.p_reg << ");" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

static MicroApiCallRegister<MicroDivApiCall> register_micro_leaky_relu_api_call("MicroDivApiCall");
}  // namespace codegen
