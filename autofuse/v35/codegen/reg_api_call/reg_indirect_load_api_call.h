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
#include <vector>

#include "codegen_kernel.h"
#include "indirect_load_utils.h"

namespace codegen {
struct SimtGmTensor {
  ascir::TensorId value_tensor_id;
  ascir::TensorId gm_tensor_id;
  af::DataType dtype;
};

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
  Status ParseSimtAttr(const ascir::NodeView &node);
  Status GenerateSk(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                    const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                    const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const;
  Status GenerateSimd(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                      const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                      const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const;
  Status GenerateSimtInvocation(const TPipe &tpipe, const std::string &input_dtype, const std::string &output_dtype,
                                const std::string &outer_tb_var, std::stringstream &ss) const;
  Status GenerateSimt(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, const Tensor &input,
                      std::string &result) const;

  int64_t axis_ = 0;
  ascir::AxisId outer_axis_ = af::kIdNone;
  ascir::TemplateId template_id_{ascir::TemplateId::kDefault};
  ascgen_utils::indirect_load::TemplateLogicalView logical_view_;
  bool has_post_reduce_{false};
  std::vector<af::AscNodePtr> index_nodes_;
  std::vector<af::AscNodePtr> output_nodes_;
  std::vector<SimtGmTensor> simt_gm_tensors_;
  ascir::TensorId index_result_tensor_id_{af::kIdNone};
  ascir::TensorId simt_value_tensor_id_{af::kIdNone};
  ascir::TensorId output_result_tensor_id_{af::kIdNone};
  std::string output_gm_tensor_;
  af::DataType index_dtype_{af::DT_UNDEFINED};
  af::DataType output_dtype_{af::DT_UNDEFINED};
  ascgen_utils::indirect_load::Implementation implementation_{ascgen_utils::indirect_load::Implementation::kDefault};
};
}  // namespace codegen

#endif  // __AUTOFUSE_REG_INDIRECT_LOAD_API_CALL_H__
