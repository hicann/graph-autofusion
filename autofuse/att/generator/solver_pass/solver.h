/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_SOLVER_H_
#define ATT_SOLVER_H_
#include "general_solver_code.h"
#include "axes_reorder_solver_code.h"
#include "base/base_types.h"

namespace att {
std::string GetSolverHead(SolverType type);
std::string GetSolverFunc(SolverType type);
std::string GetAxesReorderSolverHead(bool enable_equal_order_tiling = false);
std::string GetAxesReorderSolverFunc(bool enable_equal_order_tiling = false);
std::string GetAxesReorderPgoSolverHead(int64_t pgo_step_max);
std::string GetAxesReorderPgoSolverFunc();
}  // namespace att
#endif
