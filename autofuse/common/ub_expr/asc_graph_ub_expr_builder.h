/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_BASE_COMMON_ASC_GRAPH_UB_EXPR_BUILDER_H_
#define ASCGEN_DEV_BASE_COMMON_ASC_GRAPH_UB_EXPR_BUILDER_H_

#include "common/ub_expr/ub_expr_types.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace ascir {

class AscGraphUbExprBuilder {
 public:
  af::Status Build(const af::AscGraph &graph, UbExprContext &context, bool use_raw_tmp_size = false) const;
};

}  // namespace ascir

#endif  // ASCGEN_DEV_BASE_COMMON_ASC_GRAPH_UB_EXPR_BUILDER_H_
