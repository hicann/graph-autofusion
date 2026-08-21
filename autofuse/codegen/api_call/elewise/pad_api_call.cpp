/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "pad_api_call.h"

#include <sstream>
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
using namespace ascgen_utils;

Status PadApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                            const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                            const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                            std::string &result) const {
  auto x = inputs[0].get();
  auto y = outputs[0].get();

  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  if (it != this->tmp_buf_id.end()) {
    id = it->second;
  }

  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, {}, tpipe.tmp_buf.name + "_" + std::to_string(id));
  stringstream ss;
  stringstream axis_size_product;
  const size_t axis_num = y.vectorized_axis.size();
  for (size_t i = 0; i < axis_num; i++) {
    auto axis = tpipe.tiler.GetAxis(y.vectorized_axis[i]);
    if (i == (axis_num - 1UL)) {
      axis_size_product << axis.actual_size;
    } else {
      axis_size_product << axis.actual_size << " * ";
    }
  }

  string blk_align;
  GE_CHK_STATUS_RET(KernelUtils::BlkAlign(x.dtype, blk_align), "Codegen blk align failed in Pad api");
  uint32_t tiling_case_id = tpipe.tiler.GetTilingCaseId();
  std::string apiTilingDataString = "t->" + this->api_tiling_data_field + "_" + std::to_string(tiling_case_id);
  auto axis_pos = y.vectorized_axis_pos[axis_num - 1];
  if (tpipe.tiler.Size(y.vectorized_strides[axis_num - 2]) != tpipe.tiler.ActualSize(y.axis_size[axis_pos])) {
    GE_ASSERT_TRUE(id != -1L, "PadApiCall cannot find tmp buffer id to use.");
    ss << "if (" << tpipe.tiler.Size(y.vectorized_strides[axis_num - 2])
       << " != " << tpipe.tiler.ActualSize(y.axis_size[axis_pos]) << ") {"  // 2表示倒数第二个维度
       << std::endl;
    ss << "AscendC::PadParams padParams = {0, static_cast<uint16_t>("
       << tpipe.tiler.Size(y.vectorized_strides[axis_num - 2]) << " - " << tpipe.tiler.ActualSize(y.axis_size[axis_pos])
       << "), 0};"  // 2表示倒数第二个维度
       << std::endl;
    ss << "PadTiling apiTilingData = " << apiTilingDataString << ";" << std::endl;
    ss << "AscendC::Pad(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x << "["
       << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], " << "padParams, " << tpipe.tmp_buf << "_"
       << std::to_string(id) << ", " << "apiTilingData);" << std::endl;
    ss << "} else {" << std::endl;
    ss << "AscendC::DataCopy(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x << "["
       << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], " << blk_align << "(" << axis_size_product.str()
       << "));" << std::endl;
    ss << "}" << std::endl;
  } else {
    ss << "AscendC::DataCopy(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x << "["
       << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], " << blk_align << "(" << axis_size_product.str()
       << "));" << std::endl;
  }
  result = ss.str();
  return af::SUCCESS;
}

Status PadApiCall::ParseAttr(const ascir::NodeView &node) {
  GE_ASSERT_SUCCESS(GetApiTilingFieldName(node, this->api_tiling_data_field));
  GELOGD("Get Pad api tiling field name success, field_name: %s\n", this->api_tiling_data_field.c_str());
  return af::SUCCESS;
}

static ApiCallRegister<PadApiCall> register_pad_api_call("PadApiCall");
}  // namespace codegen
