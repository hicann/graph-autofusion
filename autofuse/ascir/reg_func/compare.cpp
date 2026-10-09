/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "default_reg_func.h"

namespace af {
namespace ascir {

Expression GetCompareNormalNoLoopTmpSize(const AscNode &node, const std::vector<uint32_t> &axis_ids) {
  auto node_inputs = node.inputs;
  uint8_t typeSize = node_inputs[0].attr.dtype == ge::DataType::DT_FLOAT16 ? 2 : 4;
  EXPECT_SYMBOL_NE(Symbol(typeSize), Symbol(0));
  Expression compare_out_size = Symbol(256 / 8 / typeSize) * node_inputs[0].attr.repeats[axis_ids[0]];
  Expression select_out_size =
      Symbol(128) * ((node_inputs[0].attr.repeats[axis_ids[0]] * Symbol(2) + Symbol(typeSize - 1)) / Symbol(typeSize));
  return compare_out_size + select_out_size;
}

std::vector<std::unique_ptr<TmpBufDesc>> GetCompareNormalTmpSize(const AscNode &node, const std::string &mode) {
  auto node_inputs = node.inputs;
  std::vector<uint32_t> axis_ids = {UINT32_MAX, UINT32_MAX};
  // 在axis中找到vectorized_axis最后一个元素在axis的位置
  uint32_t vec_last_axis_pos_in_axis = std::find(node_inputs[0].attr.axis.begin(), node_inputs[0].attr.axis.end(),
                                                 node_inputs[0].attr.vectorized_axis.back()) -
                                       node_inputs[0].attr.axis.begin();
  std::vector<std::unique_ptr<TmpBufDesc>> tmp_buf_descs;
  for (auto vec_axis : node_inputs[0].attr.vectorized_axis) {
    auto pos = std::find(node_inputs[0].attr.axis.begin(), node_inputs[0].attr.axis.end(), vec_axis);
    GE_ASSERT_TRUE(pos != node_inputs[0].attr.axis.end(), "Incorrect axis ID in vectorized_axis");
    uint32_t axis_id = static_cast<uint32_t>(pos - node_inputs[0].attr.axis.begin());
    if (axis_id < axis_ids[1]) {
      if (axis_id < axis_ids[0]) {
        axis_ids[1] = axis_ids[0];
        axis_ids[0] = axis_id;
      } else {
        axis_ids[1] = axis_id;
      }
    }
  }
  if (axis_ids[1] != UINT32_MAX) {
    if (node_inputs[0].attr.dtype == ge::DataType::DT_FLOAT16 || node_inputs[0].attr.dtype == ge::DataType::DT_FLOAT ||
        (node_inputs[0].attr.dtype == ge::DataType::DT_INT32 && (mode == "EQ" || mode == "NE"))) {
      GELOGD("[GetInputSize] axis id is: %u, %u", axis_ids[0], axis_ids[1]);
      GELOGD("[GetInputSize] inputs[0].repeat is: %s, %s", node_inputs[0].attr.repeats[axis_ids[0]].Str().get(),
             node_inputs[0].attr.repeats[axis_ids[1]].Str().get());
      GELOGD("[GetInputSize] inputs[0].vectorized_strides is: %s, %s",
             node_inputs[0].attr.vectorized_strides[0].Str().get(),
             node_inputs[0].attr.vectorized_strides[1].Str().get());
      Expression input_size =
          Symbol(256) * node_inputs[0].attr.repeats[axis_ids[1]] * node_inputs[0].attr.vectorized_strides[1];
      Expression total_size = input_size + Symbol(32) * node_inputs[0].attr.repeats[axis_ids[1]] *
                                               node_inputs[0].attr.vectorized_strides[1];
      return GetTmpBuffer(sym::Max(total_size, GetCompareNormalNoLoopTmpSize(node, axis_ids)));
    } else if (node_inputs[0].attr.dtype == ge::DataType::DT_INT32) {
      GELOGD("[GetInputSize] axis id is: %u", axis_ids[1]);
      GELOGD("[GetInputSize] inputs[1].repeat is: %s", node_inputs[0].attr.repeats[axis_ids[0]].Str().get());
      GELOGD("[GetInputSize] inputs[1].vectorized_strides is: %s",
             node_inputs[0].attr.vectorized_strides[1].Str().get());
      Expression input_size = node_inputs[0].attr.repeats[axis_ids[0]] * node_inputs[0].attr.vectorized_strides[1];
      Expression total_size = input_size * Symbol(256);
      return GetTmpBuffer(total_size);
    } else if (node_inputs[0].attr.dtype == ge::DataType::DT_INT64) {
      if (mode == "EQ" || mode == "NE") {
        GELOGD("[GetInputSize] axis id is: %u", axis_ids[1]);
        GELOGD("[GetInputSize] inputs[1].repeat is: %s", node_inputs[0].attr.repeats[axis_ids[0]].Str().get());
        GELOGD("[GetInputSize] inputs[1].vectorized_strides is: %s",
               node_inputs[0].attr.vectorized_strides[1].Str().get());
        Expression input_size = node_inputs[0].attr.repeats[axis_ids[0]] * node_inputs[0].attr.vectorized_strides[1];
        Expression total_size = input_size * Symbol(256) * Symbol(2) + Symbol(256);
        return GetTmpBuffer(total_size);
      } else {
        // 找到last axis, 并32字节对齐
        Expression last_axis_size = sym::Align(node_inputs[0].attr.repeats[vec_last_axis_pos_in_axis], 32);
        GELOGD("[GetInputSize] last axis size is: %s", last_axis_size.Str().get());
        GELOGD("[GetInputSize] axis id is: %u", axis_ids[1]);
        GELOGD("[GetInputSize] node_inputs[0].attr.repeats[axis_ids[0]] is: %s",
               node_inputs[0].attr.repeats[axis_ids[0]].Str().get());
        GELOGD("[GetInputSize] node_inputs[0].attr.vectorized_strides[1] is: %s",
               node_inputs[0].attr.vectorized_strides[1].Str().get());
        Expression input_size = node_inputs[0].attr.repeats[axis_ids[0]] * node_inputs[0].attr.vectorized_strides[1];
        GELOGD("[GetInputSize] input_size is: %s", input_size.Str().get());
        Expression total_size = input_size * last_axis_size * Symbol(8) * Symbol(5);
        TmpBufDesc desc = {total_size, -1};
        tmp_buf_descs.emplace_back(std::make_unique<TmpBufDesc>(desc));
        return tmp_buf_descs;
      }
    } else {
      return CalcDefaultTmpSize(node);
    }
  } else {
    GELOGD("[GetInputSize] axis id is: %u", axis_ids[0]);
    GELOGD("[GetInputSize] inputs[0].repeat is: %s", node_inputs[0].attr.repeats[axis_ids[0]].Str().get());
    GELOGD("[GetInputSize] inputs[0].vectorized_strides is: %s", node_inputs[0].attr.vectorized_strides[0].Str().get());
    Expression input_size = node_inputs[0].attr.repeats[axis_ids[0]] * node_inputs[0].attr.vectorized_strides[0];
    Expression total_size = Symbol(ge::GetSizeByDataType(node_inputs[0].attr.dtype)) * input_size;
    if (node_inputs[0].attr.dtype == ge::DataType::DT_INT64) {
      total_size = total_size * Symbol(5);  // Five times the temp buf is required.
    }
    return GetTmpBuffer(total_size);
  }
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcGeTmpSize(const AscNode &node) {
  return GetCompareNormalTmpSize(node, "GE");
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcEqTmpSize(const AscNode &node) {
  return GetCompareNormalTmpSize(node, "EQ");
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcNeTmpSize(const AscNode &node) {
  return GetCompareNormalTmpSize(node, "NE");
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcGtTmpSize(const AscNode &node) {
  return GetCompareNormalTmpSize(node, "GT");
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcLeTmpSize(const AscNode &node) {
  return GetCompareNormalTmpSize(node, "LE");
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcLtTmpSize(const AscNode &node) {
  return GetCompareNormalTmpSize(node, "LT");
}

}  // namespace ascir
}  // namespace af
