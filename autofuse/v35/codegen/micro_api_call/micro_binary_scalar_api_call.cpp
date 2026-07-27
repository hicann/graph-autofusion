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

#include "micro_binary_scalar_api_call.h"

namespace codegen {
namespace {
bool IsScalarNodeType(const std::string &node_type) {
  return node_type == af::ascir_op::Scalar::Type || node_type == af::ascir_op::ScalarData::Type;
}

Status GenerateInput(const std::pair<TensorType, ascir::TensorId> &input, const codegen::TensorManager &tensor_mng,
                     const TPipe &tpipe, std::stringstream &ss) {
  if (input.first == TensorType::REG_TENSOR) {
    GE_ASSERT_NOTNULL(tensor_mng.GetTensor(input.second));
    ss << *(tensor_mng.GetTensor(input.second)) << ", ";
  } else {
    GE_ASSERT_NOTNULL(tpipe.GetTensor(input.second));
    ss << *(tpipe.GetTensor(input.second)) << ", ";
  }
  return af::SUCCESS;
}
}  // namespace

Status MicroBinaryScalarApiCall::Generate(const codegen::TensorManager &tensor_mng, [[maybe_unused]] const TPipe &tpipe,
                                          CallParam &param, string &result) {
  GE_ASSERT_TRUE(this->inputs_.size() == 2, "Binary scalar api call must have 2 inputs");
  GE_ASSERT_TRUE(this->outputs_.size() == 1, "Binary scalar api call must have 1 output");

  std::stringstream ss;
  // 构建API调用名称，scalar输入时添加's'后缀
  const auto api_suffix = this->has_scalar_input_ ? "s" : "";
  ss << "AscendC::MicroAPI::" << this->api_name_ << api_suffix << "(";
  GE_ASSERT_NOTNULL(tensor_mng.GetTensor(this->outputs_[0].second));
  ss << *(tensor_mng.GetTensor(this->outputs_[0].second)) << ", ";
  const auto first_input_index = this->need_exchange_inputs_ ? 1 : 0;
  const auto second_input_index = this->need_exchange_inputs_ ? 0 : 1;
  GE_CHK_STATUS_RET(GenerateInput(this->inputs_[first_input_index], tensor_mng, tpipe, ss));
  GE_CHK_STATUS_RET(GenerateInput(this->inputs_[second_input_index], tensor_mng, tpipe, ss));
  ss << param.p_reg << ");" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status MicroBinaryScalarApiCall::Init(const ascir::NodeView &node) {
  const bool first_input_scalar = IsScalarNodeType(node->GetInDataNodes().at(0)->GetType());
  const bool second_input_scalar = IsScalarNodeType(node->GetInDataNodes().at(1)->GetType());
  GE_ASSERT_TRUE(!(first_input_scalar && second_input_scalar),
                 "Binary scalar api call node[%s] does not support both inputs as scalar", node->GetNamePtr());
  this->has_scalar_input_ = first_input_scalar || second_input_scalar;
  this->need_exchange_inputs_ = first_input_scalar && !second_input_scalar;
  GELOGI("name:%s, first input scalar:%d, second input scalar:%d, exchange inputs:%d", node->GetNamePtr(),
         first_input_scalar, second_input_scalar, this->need_exchange_inputs_);
  return af::SUCCESS;
}

static MicroApiCallRegister<MicroBinaryScalarApiCall> register_micro_binary_scalar_api_call("MicroBinaryScalarApiCall");
}  // namespace codegen
