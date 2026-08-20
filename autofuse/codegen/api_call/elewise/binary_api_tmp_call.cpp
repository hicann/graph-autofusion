/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "binary_api_tmp_call.h"

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

Status BinaryApiTmpCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                  const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                  std::string &result) const {
  const size_t x1_idx = 0;
  const size_t x2_idx = 1;

  auto x1 = inputs[x1_idx].get();
  auto x2 = inputs[x2_idx].get();

  auto y = outputs[0].get();
  stringstream ss;
  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "BinaryApiTmpCall cannot find tmp buffer id to use.");
  id = it->second;

  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, CombinedExprFactory::SymbolVar(x1.actual_size.Str()),
                               tpipe.tmp_buf.name + "_" + std::to_string(id));

  // 如果第2个输入是ub_scalar场景, 初始化x2为ub_scalar对应的变量
  bool is_scalar_scene = (x2.is_constant) || (x2.is_ub_scalar && x2.need_gen_get_value_of_ub_scalar);
  if (is_scalar_scene && (this->api_name_ == "LogicalOr" || this->api_name_ == "LogicalAnd")) {
    std::string dtype_name;
    GE_CHK_STATUS_RET(Tensor::DtypeName(x2.dtype, dtype_name), "Codegen get data type:%d failed",
                      static_cast<int32_t>(x2.dtype));
    std::string x2_scalar =
        x2.need_gen_get_value_of_ub_scalar ? ("(" + dtype_name + ")" + x2.ub_scalar_name) : x2.Str();
    ss << this->api_name_ << "ScalarExtend(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
       << x1 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], " << x2_scalar << ", "
       << tpipe.tmp_buf << "_" << std::to_string(id) << ", " << x1.actual_size << ");" << std::endl;
  } else {
    ss << this->api_name_ << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x1
       << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], " << x2 << "["
       << tpipe.tiler.TensorVectorizedOffset(current_axis, x2) << "], " << tpipe.tmp_buf << "_" << std::to_string(id)
       << ", " << x1.actual_size << ");" << std::endl;
  }

  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<BinaryApiTmpCall> register_binary_api_tmp_call("BinaryApiTmpCall");

}  // namespace codegen
