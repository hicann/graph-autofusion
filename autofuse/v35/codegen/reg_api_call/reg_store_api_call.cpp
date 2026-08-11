/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reg_store_api_call.h"

#include <sstream>

#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "reg_api_call_utils.h"
#include "api_call/utils/api_call_factory.h"

using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

namespace codegen {
Status StoreRegApiCall::ParseAttr(const ascir::NodeView &node) {
  // 存在多个Store写同一个Tensor不同offset的场景, repeats用当前Store节点的
  repeats_ = node->outputs[0U].attr.repeats;
  (void)node->attr.ir_attr->GetAttrValue("offset", offset_);
  return af::SUCCESS;
}

Status StoreRegApiCall::PreProcess(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                   const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                   std::string &result) const {
  GE_ASSERT_TRUE(!outputs.empty());
  const_cast<Tensor &>(outputs.front().get()).axis_size = repeats_;
  GE_ASSERT_SUCCESS(ApiCall::PreProcess(tpipe, current_axis, outputs, result));
  return af::SUCCESS;
}

Status StoreRegApiCall::BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                      const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                      const std::vector<std::reference_wrapper<const Tensor>> &outputs) const {
  const auto &gm = outputs[0].get();
  const auto &ub = inputs[0].get();
  auto api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);
  api_param->api_name = api_name_;
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(ub.dtype, dtype_name), "data type:%d failed", static_cast<int32_t>(ub.dtype));
  api_param->template_params.emplace_back(dtype_name);
  DmaSpecificParams dma_specific_params;
  if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    BuildDataCopyApiParamInCVFusion(*api_param, dma_specific_params, gm, ub, dtype_name, false);
  } else {
    std::string gm_offset = tpipe.tiler.Offset(current_axis, gm.axis, gm.axis_strides);
    gm_offset = gm_offset + " + " + tpipe.tiler.Size(offset_);
    BuildDataCopyApiParamInNormal(tpipe, *api_param, dma_specific_params, ub, gm, gm_offset, false);
  }
  api_param->specific_params = dma_specific_params;
  if (IsUnitLastRead(*(this->inputs[0])) && ub.is_load_link_store_and_vec) {
    std::stringstream ss;
    std::string offset = offset_.Str().get();
    offset = GenValidName(offset);
    GenerateLinkStoreEventCode(ub, offset, ss);
    api_param->api_post_process.emplace_back(ss.str());
  }

  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param));
  return af::SUCCESS;
}

Status StoreRegApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler,
                                          std::stringstream &ss) const {
  return GenDataCopyDimParam(api_param, tiler, graph_name, node_name, ss);
}
static ApiCallRegister<StoreRegApiCall> register_store_reg_api_call("StoreRegApiCall");
}  // namespace codegen
