/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "specific_params_builder.h"

#include "ascir_node_param/ascir_node_param.h"
#include "ascir_node_param/ascir_param_builder.h"
#include "base/att_const_values.h"
#include "common/checker.h"

namespace att {
namespace {
af::Status FillReduceParams(const ascir_param::AscirNodeParams &params, NodeInfo &node_info) {
  const auto *reduce = ascir_param::GetSpecificParams<ascir_param::ReduceNodeParams>(params);
  GE_ASSERT_NOTNULL(reduce, "Reduce specific params is null, node[%s].", node_info.name.c_str());
  node_info.reduce_specific_params = *reduce;
  return af::SUCCESS;
}

af::Status FillVectorFuncParams(const ascir_param::AscirNodeParams &params, NodeInfo &node_info) {
  const auto *vector_func_params = ascir_param::GetSpecificParams<ascir_param::VectorFuncNodeParams>(params);
  GE_ASSERT_NOTNULL(vector_func_params, "VectorFunc specific params is null, node[%s].", node_info.name.c_str());
  node_info.vector_func_params = *vector_func_params;
  return af::SUCCESS;
}

af::Status FillCastParams(const ascir_param::AscirNodeParams &params, NodeInfo &node_info) {
  const auto *cast_params = ascir_param::GetSpecificParams<ascir_param::CastNodeParams>(params);
  GE_ASSERT_NOTNULL(cast_params, "Cast specific params is null, node[%s].", node_info.name.c_str());
  node_info.cast_node_params = *cast_params;
  return af::SUCCESS;
}

}  // namespace

af::Status FillSpecificParams(const af::AscNodePtr &ge_node, NodeInfo &node_info) {
  const auto params = ascir_param::GetAscirNodeParams(ge_node);
  if (params == nullptr) {
    return af::SUCCESS;
  }
  if (params->status == ascir_param::ParamBuildStatus::kSkipped) {
    return af::SUCCESS;
  }
  GE_ASSERT_TRUE(params->status == ascir_param::ParamBuildStatus::kBuilt, "Build ascir node params failed, node[%s].",
                 node_info.name.c_str());
  if (ascir_param::IsReduceParamSupported(node_info.node_type)) {
    return FillReduceParams(*params, node_info);
  }
  if (node_info.node_type == kVectorFunc) {
    return FillVectorFuncParams(*params, node_info);
  }
  if (node_info.node_type == kCast) {
    return FillCastParams(*params, node_info);
  }
  return af::SUCCESS;
}
}  // namespace att
