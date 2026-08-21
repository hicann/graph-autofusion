/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TESTS_FRAMEWORK_IMPROVE_PRECISION_TEST_UTILS_H
#define AUTOFUSE_TESTS_FRAMEWORK_IMPROVE_PRECISION_TEST_UTILS_H

#include <string>

#include "ascgraph_info_complete.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "optimize/pre_process/improve_precision.h"

namespace af::testing {

inline size_t CountNodesByType(AscGraph &graph, const std::string &type) {
  size_t count = 0U;
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetType() == type) {
      ++count;
    }
  }
  return count;
}

inline bool HasCastOutputDtype(AscGraph &graph, ge::DataType expected_dtype) {
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetType() == ascir_op::Cast::Type &&
        node->GetOpDesc()->GetOutputDesc(0).GetDataType() == expected_dtype) {
      return true;
    }
  }
  return false;
}

inline bool CheckNodeOutputDtype(AscGraph &graph, const std::string &node_name, ge::DataType expected_dtype) {
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetName() == node_name) {
      auto desc = node->GetOpDesc();
      if (desc != nullptr && desc->GetOutputDesc(0).GetDataType() == expected_dtype) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace af::testing

#endif  // AUTOFUSE_TESTS_FRAMEWORK_IMPROVE_PRECISION_TEST_UTILS_H
