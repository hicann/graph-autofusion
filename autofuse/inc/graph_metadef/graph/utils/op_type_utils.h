/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INC_METADEF_OP_TYPE_UTILS_H
#define __INC_METADEF_OP_TYPE_UTILS_H
#include <string>
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/ge_error_codes_af.h"

namespace af {
class OpTypeUtils {
 public:
  static bool IsDataNode(const std::string &type);
  static bool IsInputRefData(const OpDescPtr &op_desc);
  static bool IsAutofuseNode(const std::string &type);
  static bool IsAutofuseNode(const OpDescPtr &op_desc);
  static bool IsEmptyAutofuseNode(const std::string &type);
  static bool IsVariableNode(const std::string &type);
  static bool IsVarLikeNode(const std::string &type);
  static bool IsAssignLikeNode(const std::string &type);
  static bool IsIdentityLikeNode(const std::string &type);
  static bool IsConstPlaceHolderNode(const std::string &type);
  static graphStatus GetOriginalType(const OpDescPtr &op_desc, std::string &type);
  static bool IsSubgraphInnerData(const OpDescPtr &op_desc);
  // CONST/CONSTANT/CONSTPLACEHOLDER
  static bool IsConstNode(const std::string &type);
  // IsDataNode/IsInputRefData/IsVariableNode/IsVarLikeNode/IsConstNode
  static bool IsGraphInputNode(const std::string &type);
  // NETOUTPUT
  static bool IsGraphOutputNode(const std::string &type);
};
}  // namespace af
#endif  // __INC_METADEF_OP_TYPE_UTILS_H
