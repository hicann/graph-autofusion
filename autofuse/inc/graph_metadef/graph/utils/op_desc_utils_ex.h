/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INC_METADEF_OP_DESC_UTILS_EX_H
#define __INC_METADEF_OP_DESC_UTILS_EX_H

#include "graph/ge_error_codes_af.h"
#include "graph/op_desc.h"

namespace af {
class OpDescUtilsEx {
 public:
  // Detach from OpDesc
  __attribute__((weak)) static graphStatus CallInferFunc(const OpDescPtr &op_desc, Operator &op);
  static graphStatus CallInferFormatFunc(const OpDescPtr &op_desc, Operator &op);
  static graphStatus CallInferValueRangeFunc(const OpDescPtr &op_desc, Operator &op);
  __attribute__((weak)) static graphStatus OpVerify(const OpDescPtr &op_desc);
  static graphStatus InferShapeAndType(const OpDescPtr &op_desc);
  static graphStatus InferDataSlice(const OpDescPtr &op_desc);
  static void SetType(OpDescPtr &op_desc, const std::string &type);
  static void ResetFuncHandle(OpDescPtr &op_desc);
  static void SetTypeAndResetFuncHandle(OpDescPtr &op_desc, const std::string &type);
  static void UpdateShapeAndDType(const GeTensorDescPtr &src, const GeTensorDescPtr &dst);

 private:
  static graphStatus CallInferFuncV1(const OpDescPtr &op_desc, Operator &op);
  static graphStatus CallInferFuncV2(const OpDescPtr &op_desc, Operator &op);
  static graphStatus InferShapeByOutputShapesAttr(const OpDescPtr &op_desc);
  static graphStatus CallInferFormatFuncV1(const OpDescPtr &op_desc, Operator &op);
  static graphStatus CallInferFormatFuncV2(const OpDescPtr &op_desc, Operator &op);
  static graphStatus InferCustomOpShape(const OpDescPtr &op_desc, Operator &op);
};
}  // namespace af
#endif  // __INC_METADEF_OP_DESC_UTILS_EX_H
