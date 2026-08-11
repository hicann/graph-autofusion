/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "logical_not_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils//asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

Status LogicalNotApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                   const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                   const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                   std::string &result) const {
  const auto &x = inputs[0].get();
  const auto &y = outputs[0].get();

  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, CombinedExprFactory::SymbolVar(x.actual_size.Str()));

  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "LogicalNotApiCall cannot find tmp buffer id to use.");
  id = it->second;

  stringstream ss;
  ss << this->api_name_ << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], local_blk_tensor_of_half_1, " << tpipe.tmp_buf << "_"
     << std::to_string(id) << ", " << x.actual_size << ");" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}
static ApiCallRegister<LogicalNotApiCall> register_logical_not_api_call("LogicalNotApiCall");
}  // namespace codegen
