/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_INFER_AXIS_SLICE_REGISTRY_H_
#define INC_REGISTER_INFER_AXIS_SLICE_REGISTRY_H_

#include "graph/ge_error_codes_af.h"
#include "graph/operator_af.h"
#include "graph/types_af.h"
#include "graph/axis_type_info.h"

namespace af {
// cut tensor : axis index : slice range
using DataSliceInfo = std::vector<std::vector<std::vector<int64_t>>>;
using InferAxisTypeInfoFunc = std::function<graphStatus(Operator &, std::vector<AxisTypeInfo> &)>;
using InferAxisSliceFunc =
    std::function<graphStatus(Operator &, const AxisTypeInfo &, const DataSliceInfo &, DataSliceInfo &)>;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferAxisTypeInfoFuncRegister {
 public:
  InferAxisTypeInfoFuncRegister(const std::string &operator_type,
                                const InferAxisTypeInfoFunc &infer_axis_type_info_func);
  InferAxisTypeInfoFuncRegister(const char_t *const operator_type,
                                const InferAxisTypeInfoFunc &infer_axis_type_info_func);
  ~InferAxisTypeInfoFuncRegister() = default;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferAxisSliceFuncRegister {
 public:
  InferAxisSliceFuncRegister(const std::string &operator_type, const InferAxisSliceFunc &infer_axis_slice_func);
  InferAxisSliceFuncRegister(const char_t *const operator_type, const InferAxisSliceFunc &infer_axis_slice_func);
  ~InferAxisSliceFuncRegister() = default;
};
}  // namespace af

#define PASTE(g_register, y) g_register##y

// infer axis type info func register
#define IMPLEMT_COMMON_INFER_AXIS_TYPE_INFO(func_name)                              \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static ge::graphStatus(func_name)( \
      ge::Operator & op, std::vector<ge::AxisTypeInfo> & axis_type)

#define IMPLEMT_INFER_AXIS_TYPE_INFO(op_name, func_name)                            \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static ge::graphStatus(func_name)( \
      (op_name) & op, std::vector<AxisTypeInfo> & axis_type)

#define INFER_AXIS_TYPE_INFO_FUNC(op_name, x) \
  [](ge::Operator &v, std::vector<ge::AxisTypeInfo> &axis_type) { return (x)(v, axis_type); }

#define INFER_AXIS_TYPE_INFO_REG_IMPL(op_name, x, n) \
  static const ge::InferAxisTypeInfoFuncRegister PASTE(ids_register, n)(#op_name, (x))

#define INFER_AXIS_TYPE_INFO_REG(op_name, x) \
  INFER_AXIS_TYPE_INFO_REG_IMPL(op_name, INFER_AXIS_TYPE_INFO_FUNC(op_name, x), __COUNTER__)

// infer axis slice func register
#define IMPLEMT_COMMON_INFER_AXIS_SLICE(func_name)                                                 \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static ge::graphStatus(func_name)(                \
      ge::Operator & op, const ge::AxisTypeInfo &axis_info, const ge::DataSliceInfo &output_param, \
      ge::DataSliceInfo &input_param)

#define IMPLEMT_INFER_AXIS_SLICE(op_name, func_name)                                            \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static ge::graphStatus(func_name)(             \
      (op_name) & op, const ge::AxisTypeInfo &axis_info, const ge::DataSliceInfo &output_param, \
      ge::DataSliceInfo &input_param)

#define INFER_AXIS_SLICE_FUNC(op_name, x)                                                       \
  [](ge::Operator &v, const ge::AxisTypeInfo &axis_info, const ge::DataSliceInfo &output_param, \
     ge::DataSliceInfo &input_param) { return (x)(v, axis_info, output_param, input_param); }

#define INFER_AXIS_SLICE_FUNC_REG_IMPL(op_name, x, n) \
  static const ge::InferAxisSliceFuncRegister PASTE(ids_register, n)(#op_name, (x))

#define INFER_AXIS_SLICE_FUNC_REG(op_name, x) \
  INFER_AXIS_SLICE_FUNC_REG_IMPL(op_name, INFER_AXIS_SLICE_FUNC(op_name, x), __COUNTER__)

#endif  // INC_REGISTER_INFER_AXIS_SLICE_REGISTRY_H_
