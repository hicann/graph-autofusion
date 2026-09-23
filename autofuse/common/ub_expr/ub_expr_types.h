/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_BASE_COMMON_UB_EXPR_TYPES_H_
#define ASCGEN_DEV_BASE_COMMON_UB_EXPR_TYPES_H_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "graph/symbolizer/symbolic.h"

namespace ascir {

using UbExpr = af::Expression;

struct UbExprCmp {
  bool operator()(const UbExpr &lhs, const UbExpr &rhs) const {
    if (!lhs.IsValid()) {
      return rhs.IsValid();
    }
    if (!rhs.IsValid()) {
      return false;
    }
    return lhs.Compare(rhs) < 0;
  }
};

using UbExprExprMap = std::map<UbExpr, UbExpr, UbExprCmp>;
using UbExprUintMap = std::map<UbExpr, uint32_t, UbExprCmp>;

struct UbExprContext {
  UbExpr ub_expr;
  UbExprExprMap container_expr;
  std::map<UbExpr, std::string, UbExprCmp> container_names;
  std::vector<UbExpr> ub_related_vars;
  UbExprExprMap var_min_values;
  UbExprUintMap const_vars;
  UbExprExprMap static_size_vars;
  std::vector<UbExpr> dynamic_size_vars;
  std::vector<std::pair<UbExpr, UbExpr>> expr_relations;
  std::string graph_name;
  std::string template_name;
  int64_t tiling_case_id = 0;
  // CV UBFuse 的 epilogue 内联公式不走 tiling 求解, tmp 需求按原始公式分配(无 v1 repeat cap),
  // UB 约束须与 device 侧分配保持一致, 否则求解器会选出使 tmp 越界的 stage 尺寸。
  bool use_raw_tmp_size = false;
};

struct UbExprBuildResult {
  bool has_ub_expr = false;
  UbExpr ub_expr;
  std::string origin_expr;
};

}  // namespace ascir

#endif  // ASCGEN_DEV_BASE_COMMON_UB_EXPR_TYPES_H_
