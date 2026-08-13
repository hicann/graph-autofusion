/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "solver.h"

namespace att {
std::string GetSolverHead(SolverType type) {
  if (type == SolverType::SEARCH_TILE) {
    return GENERAL_SOLVER_CODE;  // 全是inline,放在头文件
  }
  return "";
}

std::string GetSolverFunc(SolverType type) {
  (void)type;
  return "";
}

std::string GetAxesReorderSolverHead(bool enable_equal_order_tiling) {
  return GetAxesSolverSolverHead(enable_equal_order_tiling);
}

std::string GetAxesReorderSolverFunc(bool enable_equal_order_tiling) {
  return GetAxesSolverSolverFunc(enable_equal_order_tiling);
}

std::string GetAxesReorderPgoSolverHead(int64_t pgo_step_max) {
  return GetAxesSolverPgoSolverHead(pgo_step_max);
}

std::string GetAxesReorderPgoSolverFunc() {
  return AXES_SOLVER_PGO_CODE_FUNC;
}
}  // namespace att
