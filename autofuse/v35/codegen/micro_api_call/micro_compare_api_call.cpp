/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "micro_api_call_factory.h"
#include "ascir_ops.h"

#include "micro_compare_api_call.h"

namespace codegen {
Status MicroCompareApiCall::Generate(const codegen::TensorManager &tensor_mng, [[maybe_unused]] const TPipe &tpipe,
                                     CallParam &param, string &result) {
  GE_ASSERT_TRUE(this->inputs_.size() == 2, "Compare api call must have 2 inputs");
  GE_ASSERT_TRUE(this->outputs_.size() == 1, "Compare api call must have 1 output");

  std::stringstream ss;
  auto output_tensor = tensor_mng.GetTensor(this->outputs_[0].second);
  GE_ASSERT_NOTNULL(output_tensor);
  bool is_mask_reg = output_tensor->init_as_mask_reg_;

  // Compare 输出到 MaskReg（实际输出或临时）
  std::string output_name = is_mask_reg ? output_tensor->name : (output_tensor->name + "_temp_mask");

  const auto *first_input = tensor_mng.GetTensor(this->inputs_[0].second);
  GE_ASSERT_NOTNULL(first_input);
  auto dtype = first_input->dtype_;
  string dtype_name;
  Tensor::DtypeName(dtype, dtype_name);
  ss << "AscendC::MicroAPI::" << "Compare" << (this->second_input_scalar_ ? "s" : "");
  ss << "<" << dtype_name << ", CMPMODE::" << this->api_name_ << ">(";
  ss << output_name << ", ";

  ss << *first_input << ", ";
  if (inputs_[1].first != TensorType::REG_TENSOR) {
    GE_ASSERT_NOTNULL(tpipe.GetTensor(this->inputs_[1].second));
    ss << *(tpipe.GetTensor(inputs_[1].second)) << ", ";
  } else {
    GE_ASSERT_NOTNULL(tensor_mng.GetTensor(this->inputs_[1].second));
    ss << *(tensor_mng.GetTensor(inputs_[1].second)) << ", ";
  }
  ss << param.p_reg << ");" << std::endl;
  // 如果不是 MaskReg，Duplicate 转换到实际 RegTensor
  if (!is_mask_reg) {
    // Duplicate 应使用输出 tensor 的类型
    string output_dtype_name;
    Tensor::DtypeName(output_tensor->dtype_, output_dtype_name);
    ss << "AscendC::Reg::Duplicate(" << *output_tensor << ", static_cast<" << output_dtype_name << ">(1), "
       << output_name << ");" << std::endl;
  }
  result = ss.str();
  return af::SUCCESS;
}

Status MicroCompareApiCall::Init(const ascir::NodeView &node) {
  // 判断第二个输入是否是scalar
  if (node->GetInDataNodes().at(1)->GetType() == "Scalar" ||
      node->GetInDataNodes().at(1)->GetType() == af::ascir_op::IndexExpr::Type) {
    this->second_input_scalar_ = true;
  }
  GELOGI("name:%s, second input scalar:%d", node->GetNamePtr(), this->second_input_scalar_);
  return af::SUCCESS;
}

static MicroApiCallRegister<MicroCompareApiCall> register_micro_compare_api_call("MicroCompareApiCall");
}  // namespace codegen
