/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gather_api_call.h"
#include "gather_api_call_base.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "../utils/api_call_factory.h"
#include "../utils/api_call_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;
using namespace gather_base;

std::string GenerateNonLastAxisGather(const std::vector<ascir::AxisId> &current_axis,
                                      const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                      const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                      int64_t gather_axis, const TPipe &tpipe) {
  stringstream ss;
  const auto &x1 = inputs[0].get();  // param_gm
  const auto &x2 = inputs[1].get();  // indices_gm
  const auto &y = outputs[0].get();  // dst_ub
  if (y.vectorized_axis.size() > 1) {
    auto axis0 = tpipe.tiler.GetAxis(y.vectorized_axis[0]);
    auto axis1 = tpipe.tiler.GetAxis(y.vectorized_axis[1]);
    std::vector<ascir::AxisId> param_outer_axes;
    std::vector<ascir::AxisId> param_inner_axes;
    CollectParamOuterAndInnerAxes(x1.axis, gather_axis, param_outer_axes, param_inner_axes);
    std::string outer_axis_offset = CalGatherOuterAxisOffset(current_axis, param_inner_axes, axis0.id, tpipe);
    ss << "for (" << axis0.AsArg() << " = 0; " << axis0 << " < " << axis0.actual_size << "; " << axis0 << "++) {"
       << std::endl;
    ss << CalGatherOuterAxesIndex(outer_axis_offset, param_outer_axes, x2.axis, tpipe);
    ss << "auto indices_index = " << outer_axis_offset << " % " << CalGatherIndicesAxesSize(x2.axis, tpipe) << ";"
       << std::endl;
    std::string indices_value = x2.Str() + ".GetValue(indices_index)";
    ss << "auto param_offset = " << CalGatherParamOffset(x1.axis, indices_value, gather_axis, axis1, tpipe);
    ss << "DataCopyPadExtend(" << y << "[" << axis0 << " * " << tpipe.tiler.Size(y.vectorized_strides[0]) << "], " << x1
       << "[param_offset], 1, " << axis1.actual_size << ", 0, 0);" << std::endl;
    ss << "}" << std::endl;
  } else {
    auto axis0 = tpipe.tiler.GetAxis(y.vectorized_axis[0]);
    std::vector<ascir::AxisId> param_outer_axes;
    std::vector<ascir::AxisId> param_inner_axes;
    CollectParamOuterAndInnerAxes(x1.axis, gather_axis, param_outer_axes, param_inner_axes);
    std::string outer_axis_offset = CalGatherOuterAxisOffset(current_axis, param_inner_axes, af::kIdNone, tpipe);

    ss << "auto indices_index = " << outer_axis_offset << " % " << CalGatherIndicesAxesSize(x2.axis, tpipe) << ";"
       << std::endl;
    std::string indices_value = x2.Str() + ".GetValue(indices_index)";
    ss << "auto param_offset = " << CalGatherParamOffset(x1.axis, indices_value, gather_axis, axis0, tpipe);

    ss << "DataCopyPadExtend(" << y << "[0], " << x1 << "[param_offset], 1, " << axis0.actual_size << ", 0, 0);"
       << std::endl;
  }
  return ss.str();
}

Status GatherApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                               const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                               const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                               std::string &result) const {
  std::stringstream ss;
  const auto &x1 = inputs[0].get();  // param_gm
  const auto &x2 = inputs[1].get();  // indices_gm
  const auto &y = outputs[0].get();  // dst_ub
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs);
  DataCopyParams param_x1;
  DataCopyParams param_x2;
  DataCopyParams param;

  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  if (it != this->tmp_buf_id.end()) {
    id = it->second;
  }

  std::string x1_offset = tpipe.tiler.Offset(current_axis, x1.axis, x1.axis_strides);
  std::string dst_offset = tpipe.tiler.Offset(current_axis, y.axis, y.axis_strides);
  size_t pos = dst_offset.rfind('+');
  std::string x2_offset = (pos != std::string::npos) ? dst_offset.substr(pos + 1) : dst_offset;
  x2_offset.erase(0, x2_offset.find_first_not_of(" "));
  if (this->axis + 1 > static_cast<int64_t>(x1.axis_size.size())) {
    GELOGE(af::FAILED, "gather axis(%d) is larger than x1 axis size(%d)", this->axis, x1.axis_size.size());
    return af::FAILED;
  }
  if (this->axis + 1 != static_cast<int64_t>(x1.axis_size.size())) {
    ss << GenerateNonLastAxisGather(current_axis, inputs, outputs, this->axis, tpipe);
  } else {
    GE_ASSERT_TRUE(id != -1L, "GatherApiCall cannot find tmp buffer id to use.");
    if (x1.axis_size.size() == 1) {
      ss << this->api_name_ << "(" << y << ", " << x1 << ", " << x2 << "[" << dst_offset << "], "
         << tpipe.tiler.Size(x1.axis_size[0], true) << ", " << y.actual_size << ", " << tpipe.tmp_buf << "_"
         << std::to_string(id) << ");" << std::endl;
    } else {
      string first_merge_axis = "0";
      string block_inner_axis;
      for (size_t i = 0; i < current_axis.size(); i++) {
        if (tpipe.tiler.GetAxis(current_axis[i]).type == Axis::Type::kAxisTypeBlockInner) {
          block_inner_axis = tpipe.tiler.GetAxis(current_axis[i]).Str();
        }
      }
      if (block_inner_axis.length() > 1) {
        first_merge_axis = block_inner_axis.substr(0, block_inner_axis.length() - 1);  // 获取合轴之后的首轴未切分的轴
      }
      std::string param_last_axis_size = tpipe.tiler.Size(x1.axis_size[x1.axis_size.size() - 1], true);
      x1_offset = first_merge_axis + " * " + param_last_axis_size;
      ss << this->api_name_ << "(" << y << ", " << x1 << "[" << x1_offset << "], " << x2 << "[" << x2_offset << "], "
         << param_last_axis_size << ", " << y.actual_size << ", " << tpipe.tmp_buf << "_" << std::to_string(id) << ");"
         << std::endl;
    }
  }
  result = ss.str();
  return af::SUCCESS;
}

Status GatherApiCall::ParseAttr(const ascir::NodeView &node) {
  GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("axis", this->axis),
                          "Failed to get Gather axis attr, node = %s", node->GetNamePtr());
  GELOGI("name:%s, axis:%lld", node->GetNamePtr(), this->axis);
  return af::SUCCESS;
}

static ApiCallRegister<GatherApiCall> register_gather_api_call("GatherApiCall");

}  // namespace codegen
