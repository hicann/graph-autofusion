/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_CUSTOM_OP_REGISTRY_H
#define CANN_GRAPH_ENGINE_CUSTOM_OP_REGISTRY_H
#include "graph/custom_op_af.h"
#include "graph/ascend_string_af.h"
#include "graph/ge_error_codes_af.h"

namespace af {
using BaseOpCreator = std::function<std::unique_ptr<BaseCustomOp>()>;

class CustomOpFactory {
 public:
  static graphStatus RegisterCustomOpCreator(const AscendString &op_type, const BaseOpCreator &op_creator);

  static std::unique_ptr<BaseCustomOp> CreateCustomOp(const AscendString &op_type);

  static graphStatus GetAllRegisteredOps(std::vector<AscendString> &all_registered_ops);

  static bool IsExistOp(const AscendString &op_type);
};
}  // namespace af
#endif  // CANN_GRAPH_ENGINE_CUSTOM_OP_REGISTRY_H
