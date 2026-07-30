/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "v35/codegen/simt_scalar_call/simt_scalar_emitter.h"

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "v35/ascir/ascir_codegen_v2.h"
#include "common/checker.h"
#include "common_utils.h"

namespace codegen {
af::Status EmitSimtScalarExpr(const ascir::NodeView &node, const std::vector<std::string> &inputs, std::string &expr) {
  GE_ASSERT_NOTNULL(node, "SIMT scalar node is null.");
  GE_ASSERT_TRUE(!inputs.empty() && inputs.size() == node->inputs.Size(),
                 "SIMT scalar node %s[%s] expects %zu non-empty inputs, but got %zu.", node->GetTypePtr(),
                 node->GetNamePtr(), node->inputs.Size(), inputs.size());
  const auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
  GE_ASSERT_NOTNULL(impl, "SIMT scalar codegen is not registered for node %s[%s].", node->GetTypePtr(),
                    node->GetNamePtr());
  const auto *v2_impl = dynamic_cast<af::ascir::AscIrCodegenV2 *>(impl.get());
  GE_ASSERT_NOTNULL(v2_impl, "SIMT scalar codegen for node %s[%s] is not a V2 implementation.", node->GetTypePtr(),
                    node->GetNamePtr());
  GE_ASSERT_TRUE(v2_impl->IsSimtScalarSupported(*node), "SIMT scalar codegen is not supported for node %s[%s].",
                 node->GetTypePtr(), node->GetNamePtr());
  return v2_impl->GenerateSimtScalarExpr(*node, inputs, expr);
}
}  // namespace codegen
