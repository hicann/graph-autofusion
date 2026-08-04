/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AUTOFUSE_REG_INDIRECT_LOAD_API_CALL_H__
#define __AUTOFUSE_REG_INDIRECT_LOAD_API_CALL_H__

#include <string>

#include "codegen_kernel.h"
#include "indirect_load_utils.h"

namespace codegen {
class IndirectLoadRegApiCall final : public ApiCall {
 public:
  explicit IndirectLoadRegApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~IndirectLoadRegApiCall() final = default;
  Status GenerateFuncDefinition(const TPipe &tpipe, const Tiler &tiler, std::stringstream &ss) const override;
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                  const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const override;
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  std::string &result) const override;

 protected:
  Status ParseAttr(const ascir::NodeView &node) override;

 private:
  Status GenerateSk(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                    const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                    const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const;
  Status GenerateSimd(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                      const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                      const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const;
  Status GenerateSimt(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                      const std::vector<std::reference_wrapper<const Tensor>> &inputs, std::string &result) const;

  int64_t axis_ = 0;
  ascir::AxisId outer_axis_ = af::kIdNone;
  ascir::TemplateId template_id_{ascir::TemplateId::kDefault};
  ascgen_utils::indirect_load::TemplateLogicalView logical_view_;
  std::vector<af::AscNodePtr> index_pre_nodes_;
  std::vector<af::AscNodePtr> output_post_nodes_;
  std::string output_gm_tensor_;
  af::DataType output_dtype_{af::DT_UNDEFINED};
};
}  // namespace codegen

#endif  // __AUTOFUSE_REG_INDIRECT_LOAD_API_CALL_H__
