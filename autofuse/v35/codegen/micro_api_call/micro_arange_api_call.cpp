/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "micro_arange_api_call.h"

#include <sstream>

#include "common/checker.h"
#include "micro_api_call_factory.h"

namespace codegen {
Status MicroArangeApiCall::Init(const ascir::NodeView &node) {
  GE_CHK_STATUS_RET(MicroApiCall::Init(node));
  GE_ASSERT_TRUE(node->outputs[0].attr.dtype == ge::DT_INT32 || node->outputs[0].attr.dtype == ge::DT_INT64,
                 "Arange supports only int32 and int64, dtype:%d", static_cast<int32_t>(node->outputs[0].attr.dtype));
  GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("base", base_), "Failed to get Arange base attr");
  GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("step", step_), "Failed to get Arange step attr");
  return af::SUCCESS;
}

void MicroArangeApiCall::GetArangeParams(const TPipe &tpipe, std::string &base, std::string &step) const {
  base = tpipe.tiler.ActualSize(base_);
  step = tpipe.tiler.ActualSize(step_);
}

Status MicroArangeApiCall::Generate(const TensorManager &tensor_mng, const TPipe &tpipe, CallParam &param,
                                    std::string &result) {
  GE_ASSERT_TRUE(outputs_.size() == 1U, "Arange micro api call must have one output");
  const auto *output = tensor_mng.GetTensor(outputs_[0].second);
  GE_ASSERT_NOTNULL(output);

  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(output->dtype_, dtype_name), "Get Arange dtype:%d failed",
                    static_cast<int32_t>(output->dtype_));
  const auto base = tpipe.tiler.ActualSize(base_);
  const auto step = tpipe.tiler.ActualSize(step_);
  GE_ASSERT_TRUE(!base.empty() && !step.empty(), "Failed to generate Arange base or step expression");
  const std::string block_offset = param.offset.empty() ? "0" : param.offset;
  const auto vf_base = param.arange.valid ? param.arange.base : base;
  const auto vf_step = param.arange.valid ? param.arange.step : step;
  GE_ASSERT_TRUE(!vf_base.empty() && !vf_step.empty(), "Failed to generate Arange base or step expression");
  const std::string block_base =
      param.offset.empty() ? vf_base : "(" + vf_base + " + (" + block_offset + ") * (" + vf_step + "))";

  std::stringstream ss;
  if (af::SymbolicUtils::StaticCheckEq(step_, af::sym::kSymbolOne) == af::TriBool::kTrue) {
    ss << "AscendC::Reg::Arange(" << output->name << ", static_cast<" << dtype_name << ">(" << block_base << "));"
       << std::endl;
  } else {
    ss << "AscendC::Reg::Arange(" << output->name << ", static_cast<" << dtype_name << ">(0));" << std::endl;
    ss << "AscendC::Reg::Muls(" << output->name << ", " << output->name << ", static_cast<" << dtype_name << ">("
       << vf_step << "), " << param.p_reg << ");" << std::endl;
    ss << "AscendC::Reg::Adds(" << output->name << ", " << output->name << ", static_cast<" << dtype_name << ">("
       << block_base << "), " << param.p_reg << ");" << std::endl;
  }
  result = ss.str();
  return af::SUCCESS;
}

static MicroApiCallRegister<MicroArangeApiCall> register_micro_arange_api_call("MicroArangeApiCall");
}  // namespace codegen
