/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_PATTERN_FUSION_UNITTEST_CPP_O_D_PASS_RUNNER_V1_H
#define AUTOFUSE_PATTERN_FUSION_UNITTEST_CPP_O_D_PASS_RUNNER_V1_H

#include "optimize/platform/common/pass_runner.h"
#include "optimize/graph_pass/broadcast_const_to_store.h"
#include "optimize/graph_pass/duplicate_elewise_cse_pass.h"
#include "optimize/graph_pass/scalar_to_1d_tensor.h"
#include "optimize/graph_pass/scalar_broadcast_optimization.h"
#include "optimize/graph_pass/expand_dims_for_all_reduce.h"
#include "optimize/graph_pass/pow_equiv_substitution_pass.h"
#include "optimize/graph_pass/masked_fill_input_reorder_pass.h"

namespace optimize {
class PassRunnerV1 final : public BasePassRunner {
 public:
  explicit PassRunnerV1() : BasePassRunner() {
    this->RegisterPass<PowEquivSubstitutionPass>();
    this->RegisterPass<BroadcastConstToStorePass>();
    this->RegisterPass<ScalarTo1DTensorPass>();
    this->RegisterPass<ScalarBroadcastOptimizationPass>();
    this->RegisterPass<MaskedFillInputReorderPass>();
    this->RegisterPass<ExpandDimsForAllReducePass>();
    this->RegisterPass<DuplicateElewiseCsePass>();
  }
};
}  // namespace optimize

#endif  // AUTOFUSE_PATTERN_FUSION_UNITTEST_CPP_O_D_PASS_RUNNER_V1_H
