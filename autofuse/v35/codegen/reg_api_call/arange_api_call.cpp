/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "arange_api_call.h"

#include <algorithm>
#include <sstream>

#include "api_call/utils/api_call_factory.h"
#include "common/checker.h"

namespace codegen {
Status ArangeApiCall::ParseAttr(const ascir::NodeView &node) {
  GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("base", base_), "Failed to get Arange base attr");
  GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("step", step_), "Failed to get Arange step attr");
  return af::SUCCESS;
}

Status ArangeApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                               const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                               const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                               std::string &result) const {
  GE_ASSERT_TRUE(inputs.empty(), "Arange ApiCall must have no inputs, but got %zu inputs", inputs.size());
  GE_ASSERT_TRUE(outputs.size() == 1U, "Arange ApiCall must have exactly one output, but got %zu outputs",
                 outputs.size());
  const auto &output = outputs[0].get();
  GE_ASSERT_TRUE(output.dtype == ge::DT_INT32 || output.dtype == ge::DT_INT64,
                 "Arange supports only int32 and int64, dtype:%d", static_cast<int32_t>(output.dtype));

  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(output.dtype, dtype_name), "Get Arange dtype failed, dtype:%d",
                    static_cast<int32_t>(output.dtype));
  const auto base = tpipe.tiler.ActualSize(base_);
  const auto step = tpipe.tiler.ActualSize(step_);
  GE_ASSERT_TRUE(!base.empty() && !step.empty(), "Failed to generate Arange base or step expression");
  const auto logical_offset = tpipe.tiler.Offset(current_axis, output.axis, output.axis_strides);
  const auto write_offset = tpipe.tiler.TensorVectorizedOffset(current_axis, output);
  // A Loop fixes a prefix of the local view. TensorActualSize(output) still
  // includes that prefix, so only the remaining contiguous suffix is written.
  auto slice = output;
  slice.vectorized_axis.clear();
  slice.vectorized_axis_pos.clear();
  slice.vectorized_strides.clear();
  af::Expression stride = af::Symbol(1);
  bool has_fixed_axis = false;
  std::vector<std::pair<std::string, std::string>> broadcast_frames;
  GE_ASSERT_TRUE(output.vectorized_axis.size() == output.vectorized_axis_pos.size() &&
                     output.vectorized_axis.size() == output.vectorized_strides.size(),
                 "Arange vectorized layout metadata size mismatch");
  for (size_t i = output.vectorized_axis.size(); i > 0U; --i) {
    const auto index = i - 1U;
    const auto id = output.vectorized_axis[index];
    const auto pos = output.vectorized_axis_pos[index];
    GE_ASSERT_TRUE(pos < output.axis_size.size() && pos < output.axis_strides.size(),
                   "Arange vectorized axis position is out of range");
    // 退化轴(size 1 且逻辑 stride 0): 对物理偏移(size 1)和逻辑取值(stride 0)
    // 均无贡献, 物化结果即扁平一维序列; 跳过该轴, 扩维由 Broadcast 通用路径处理。
    if (af::SymbolicUtils::StaticCheckEq(output.axis_size[pos], af::sym::kSymbolOne) == af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(output.axis_strides[pos], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    const auto &axis = tpipe.tiler.GetAxis(id);
    GE_CHK_BOOL_RET_STATUS(
        af::SymbolicUtils::StaticCheckEq(output.axis_size[pos], axis.size) == af::TriBool::kTrue &&
            af::SymbolicUtils::StaticCheckEq(output.vectorized_strides[index], stride) == af::TriBool::kTrue,
        af::FAILED, "Arange ApiCall requires a bounded contiguous vectorized layout");
    const bool fixed = std::find(current_axis.begin(), current_axis.end(), id) != current_axis.end();
    for (const auto current : current_axis) {
      GE_CHK_BOOL_RET_STATUS(current == id || !tpipe.tiler.IsFrom(current, id), af::FAILED,
                             "Arange ApiCall cannot prove a derived current-axis local range");
    }
    if (fixed) {
      GE_CHK_BOOL_RET_STATUS(axis.type == Axis::Type::kAxisTypeOriginal || axis.IsInner(), af::FAILED,
                             "Arange ApiCall cannot prove the current-axis loop bound");
      has_fixed_axis = true;
    } else if (af::SymbolicUtils::StaticCheckEq(output.axis_strides[pos], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      // 广播轴: 取值沿该轴重复, 逻辑偏移不前进。作为连续片段边界处理;
      // 调用点外层循环不迭代该轴时, 在调用内部展开逐帧写入,
      // 物理偏移按连续布局前进, 取值按逻辑 stride(0)保持重复。
      has_fixed_axis = true;
      std::string frame_size;
      const bool size_equal =
          af::SymbolicUtils::StaticCheckEq(output.axis_size[pos], axis.size_expr) == af::TriBool::kTrue;
      if (axis.type == Axis::Type::kAxisTypeTileInner || size_equal) {
        frame_size = axis.actual_size.Str();
      } else {
        frame_size = tpipe.tiler.Size(output.axis_size[pos]);
      }
      broadcast_frames.emplace_back(frame_size, tpipe.tiler.Size(output.vectorized_strides[index]));
    } else {
      GE_CHK_BOOL_RET_STATUS(
          !has_fixed_axis && af::SymbolicUtils::StaticCheckEq(output.axis_strides[pos], stride) == af::TriBool::kTrue,
          af::FAILED, "Arange ApiCall requires a contiguous logical suffix");
      GE_CHK_BOOL_RET_STATUS(
          slice.vectorized_axis.empty() || !tpipe.tiler.GetAxis(slice.vectorized_axis.front()).IsInner(), af::FAILED,
          "Arange ApiCall cannot flatten a multi-axis inner tail");
      slice.vectorized_axis.insert(slice.vectorized_axis.begin(), id);
      slice.vectorized_axis_pos.insert(slice.vectorized_axis_pos.begin(), pos);
      slice.vectorized_strides.insert(slice.vectorized_strides.begin(), output.vectorized_strides[index]);
    }
    stride = af::sym::Mul(stride, output.axis_size[pos]);
  }
  const auto count = has_fixed_axis ? tpipe.tiler.TensorActualSize(slice) : output.actual_size.Str();

  std::stringstream ss;
  std::string write_addr = write_offset;
  for (size_t frame = 0UL; frame < broadcast_frames.size(); ++frame) {
    const auto &var = "arange_b" + std::to_string(frame);
    ss << "for (int64_t " << var << " = 0; " << var << " < " << broadcast_frames[frame].first << "; ++" << var << ") {"
       << std::endl;
    write_addr += " + " + var + " * " + broadcast_frames[frame].second;
  }
  ss << "for (int64_t arange_i = 0; arange_i < " << count << "; ++arange_i) {" << std::endl;
  ss << "  " << output << ".SetValue(static_cast<uint32_t>(" << write_addr << " + arange_i), static_cast<" << dtype_name
     << ">((" << base << ") + (" << logical_offset << " + arange_i) * (" << step << ")));" << std::endl;
  ss << "}" << std::endl;
  for (size_t frame = 0UL; frame < broadcast_frames.size(); ++frame) {
    ss << "}" << std::endl;
  }
  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<ArangeApiCall> register_arange_api_call("ArangeApiCall");
}  // namespace codegen
