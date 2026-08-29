/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GRAPH_OPERATOR_FACTORY_H_
#define INC_EXTERNAL_GRAPH_OPERATOR_FACTORY_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "graph/operator_af.h"
#include "graph/ge_error_codes_af.h"
#include "graph/ascend_string_af.h"
#include "graph/types_af.h"

namespace af {
using ge::AscendString;
using ge::char_t;

using OpCreator = std::function<Operator(const std::string &)>;
using OpCreatorV2 = std::function<Operator(const AscendString &)>;
using InferShapeFunc = std::function<graphStatus(Operator &)>;
using InferFormatFunc = std::function<graphStatus(Operator &)>;
using InferValueRangeFunc = std::function<graphStatus(Operator &)>;
using VerifyFunc = std::function<graphStatus(Operator &)>;

enum WHEN_CALL { INPUT_IS_DYNAMIC = 0, INPUT_HAS_VALUE_RANGE = 1 };

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OperatorFactory {
 public:
  ATTRIBUTED_DEPRECATED(static Operator CreateOperator(const char_t *, const char_t *))
  static Operator CreateOperator(const std::string &operator_name, const std::string &operator_type);

  static Operator CreateOperator(const char_t *const operator_name, const char_t *const operator_type);

  ATTRIBUTED_DEPRECATED(graphStatus GetOpsTypeList(std::vector<AscendString> &))
  static graphStatus GetOpsTypeList(std::vector<std::string> &all_ops);

  static graphStatus GetOpsTypeList(std::vector<AscendString> &all_ops);

  ATTRIBUTED_DEPRECATED(bool IsExistOp(const char_t *))
  static bool IsExistOp(const std::string &operator_type);

  static bool IsExistOp(const char_t *const operator_type);
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OperatorCreatorRegister {
 public:
  ATTRIBUTED_DEPRECATED(OperatorCreatorRegister(const char_t *, OpCreatorV2 const &))
  OperatorCreatorRegister(const std::string &operator_type, OpCreator const &op_creator);
  OperatorCreatorRegister(const char_t *const operator_type, OpCreatorV2 const &op_creator);
  ~OperatorCreatorRegister() = default;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferShapeFuncRegister {
 public:
  ATTRIBUTED_DEPRECATED(InferShapeFuncRegister(const char_t *, const InferShapeFunc &))
  InferShapeFuncRegister(const std::string &operator_type, const InferShapeFunc &infer_shape_func);
  InferShapeFuncRegister(const char_t *const operator_type, const InferShapeFunc &infer_shape_func);
  ~InferShapeFuncRegister() = default;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferFormatFuncRegister {
 public:
  ATTRIBUTED_DEPRECATED(InferFormatFuncRegister(const char_t *, const InferFormatFunc &))
  InferFormatFuncRegister(const std::string &operator_type, const InferFormatFunc &infer_format_func);
  InferFormatFuncRegister(const char_t *const operator_type, const InferFormatFunc &infer_format_func);
  ~InferFormatFuncRegister() = default;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferValueRangeFuncRegister {
 public:
  InferValueRangeFuncRegister(const char_t *const operator_type, const WHEN_CALL when_call,
                              const InferValueRangeFunc &infer_value_range_func);
  InferValueRangeFuncRegister(const char_t *const operator_type);
  ~InferValueRangeFuncRegister() = default;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY VerifyFuncRegister {
 public:
  ATTRIBUTED_DEPRECATED(VerifyFuncRegister(const char_t *, const VerifyFunc &))
  VerifyFuncRegister(const std::string &operator_type, const VerifyFunc &verify_func);
  VerifyFuncRegister(const char_t *const operator_type, const VerifyFunc &verify_func);
  ~VerifyFuncRegister() = default;
};
}  // namespace af

#endif  // INC_EXTERNAL_GRAPH_OPERATOR_FACTORY_H_
