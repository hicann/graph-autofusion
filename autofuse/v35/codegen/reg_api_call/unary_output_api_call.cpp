/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "unary_output_api_call.h"
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;

Status UnaryOutputApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                    const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                    const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                    std::string &result) const {
  // 验证：无输入，单输出
  GE_CHK_BOOL_RET_STATUS(inputs.empty(), af::FAILED, "UnaryOutputApiCall expects no inputs, but got %zu inputs",
                         inputs.size());
  GE_CHK_BOOL_RET_STATUS(outputs.size() == 1, af::FAILED,
                         "UnaryOutputApiCall expects exactly 1 output, but got %zu outputs", outputs.size());

  auto y = outputs[0].get();

  stringstream ss;
  // 生成 API 调用：Rand(local_0[0], local_0_actual_size);
  ss << this->api_name_ << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
     << y.actual_size << ");" << std::endl;

  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<UnaryOutputApiCall> register_unary_output_api_call("UnaryOutputApiCall");
}  // namespace codegen
