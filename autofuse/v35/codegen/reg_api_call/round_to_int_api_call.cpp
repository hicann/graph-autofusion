/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "round_to_int_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/checker.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "api_call/utils/api_call_factory.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

Status RoundToIntApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                   const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                   const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                   std::string &result) const {
  (void)this->api_name_;
  auto x = inputs[0].get();
  auto y = outputs[0].get();
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, CombinedExprFactory::SymbolVar(x.actual_size.Str()));

  GELOGI("RoundToInt x_dtype:%d, y.dtype:%d.", static_cast<int32_t>(x.dtype), static_cast<int32_t>(y.dtype));

  stringstream ss;

  // 使用 AscendC::Cast 函数，设置 round 模式为 CAST_RINT
  ss << "AscendC::Cast(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
     << "AscendC::RoundMode::CAST_RINT, " << x.actual_size << ");" << std::endl;

  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<RoundToIntApiCall> register_round_to_int_api_call("RoundToIntApiCall");

}  // namespace codegen
