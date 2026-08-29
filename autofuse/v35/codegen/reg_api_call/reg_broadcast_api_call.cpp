/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reg_broadcast_api_call.h"
#include <limits>
#include <sstream>
#include "ascir_ops.h"
#include "common/ge_common/debug/log.h"
#include "api_call/utils/api_call_utils.h"
#include "api_call/utils/api_call_factory.h"
#include "api_call/broadcast/broadcast_api_call.h"
#include "codegen/expression_convert_struct.h"
#include "reg_api_call_utils.h"

namespace codegen {
using namespace std;
using namespace ascgen_utils;

static void GenParams(const TPipe &tpipe, const Tensor &input, const Tensor &output, std::stringstream &ss,
                      bool is_src) {
  // 只保证在仅对张量尾轴做32B对齐的场景下有效，若对中间轴做了对齐，则还需要增加处理逻辑
  auto vectorized_axis_size = input.vectorized_axis.size();
  const char *shape_prefix = is_src ? "src_shape_" : "dst_shape_";
  constexpr char kCompactPddingMode[] = "AscendC::PaddingMode::Compact";
  std::string padding_mode = GetPaddingMode(output);

  ss << "const uint32_t " << shape_prefix << input.id << "_brc_to_" << output.id << "[" << vectorized_axis_size
     << "] = {";
  const char *sep = "";
  for (size_t pos = 0UL; pos < vectorized_axis_size; pos++) {
    ss << sep;
    sep = ", ";
    ss << "static_cast<uint32_t>(";
    // 处理输入广播轴
    if (is_src) {
      auto input_repeat = input.axis_size[input.vectorized_axis_pos[pos]];
      auto output_repeat = output.axis_size[output.vectorized_axis_pos[pos]];
      if (af::SymbolicUtils::StaticCheckEq(input_repeat, output_repeat) != af::TriBool::kTrue) {
        ss << "1)";
        continue;
      }
    }
    // 非尾轴
    if (pos != vectorized_axis_size - 1UL) {
      GetOneAxisSize(tpipe, output, pos, ss);
      ss << ")";
      continue;
    }
    // 找到最近一个stride非0的轴
    size_t pre_pos = std::numeric_limits<size_t>::max();
    for (size_t i = 0UL; i < pos; ++i) {
      if (output.vectorized_strides[i] != 0) {
        pre_pos = i;
      }
    }
    if (pre_pos == std::numeric_limits<size_t>::max()) {
      GetOneAxisSize(tpipe, output, pos, ss);
      ss << ")";
      continue;
    }
    ascir::AxisId axis_id = output.vectorized_axis[pos];
    auto last_dim_size = output.vectorized_strides[pre_pos];
    if (tpipe.tiler.GetAxis(axis_id).type != ascir::Axis::Type::kAxisTypeTileInner ||
        output.vectorized_axis[0] == axis_id || padding_mode == kCompactPddingMode) {
      ss << tpipe.tiler.ActualSize(last_dim_size);
    } else {
      ss << tpipe.tiler.Size(last_dim_size);
    }
    ss << ")";
  }
  ss << "};\n";
}

Status BroadcastRegApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                     const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                     const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                     std::string &result) const {
  const auto &x = inputs[0].get();
  const auto &y = outputs[0].get();
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs);

  if (IsBroadcastConstantTensor(x)) {
    int64_t id = -1;
    BroadcastScalar(tpipe, current_axis, x, y, id, result, false);
    return af::SUCCESS;
  }

  size_t min_vectorized_axis_size = 1UL;
  size_t max_vectorized_axis_size = 9UL;
  if (x.vectorized_axis.size() < min_vectorized_axis_size || x.vectorized_axis.size() > max_vectorized_axis_size) {
    GELOGE(af::FAILED, "Codegen broadcast input vec axis size[%zu] is either 0 or greater than 9",
           x.vectorized_axis.size());
    return af::FAILED;
  }

  if (y.vectorized_axis.size() < min_vectorized_axis_size || y.vectorized_axis.size() > max_vectorized_axis_size) {
    GELOGE(af::FAILED, "Codegen broadcast output vec axis size[%zu] is either 0 or greater than 9",
           y.vectorized_axis.size());
    return af::FAILED;
  }

  if (x.vectorized_axis.size() != y.vectorized_axis.size()) {
    GELOGE(af::FAILED, "Codegen broadcast input vec axis size[%zu] not equal output vec axis size[%zu]",
           x.vectorized_axis.size(), y.vectorized_axis.size());
    return af::FAILED;
  }

  std::stringstream ss;
  // 生成参数 const uint32_t *dst_shape;
  std::stringstream params_name;
  GenParams(tpipe, x, y, ss, false);
  // 生成参数 const uint32_t *src_shape;
  GenParams(tpipe, x, y, ss, true);

  std::string dtype_name;
  Tensor::DtypeName(x.dtype, dtype_name);
  ss << this->api_name_ << "<" << dtype_name << "," << x.vectorized_axis.size() << ">(";
  // 传入参数  const LocalTensor<T> &dst;
  ss << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], ";
  // 传入参数  const LocalTensor<T> &src;
  ss << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], ";

  ss << "dst_shape_" << x.id << "_brc_to_" << y.id << ", src_shape_" << x.id << "_brc_to_" << y.id << ");\n";

  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<BroadcastRegApiCall> register_broadcast_reg_api_call("BroadcastRegApiCall");
}  // namespace codegen
