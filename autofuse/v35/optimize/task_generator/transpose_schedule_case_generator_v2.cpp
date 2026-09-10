/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/transpose_schedule_case_generator_v2.h"

namespace optimize {
Status TransposeFusionCaseGeneratorV2::GenerateScoreFuncForUbReorder(const ascir::HintGraph &graph,
                                                                     const af::AscNodePtr &transpose_node,
                                                                     std::string &score_func) {
  (void)graph;
  (void)transpose_node;
  (void)score_func;
  return af::SUCCESS;
}
}  // namespace optimize
