/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __AUTOFUSE_VEC_FUNC_CALL_H__
#define __AUTOFUSE_VEC_FUNC_CALL_H__

#include "vf_loop.h"
#include "codegen_kernel.h"
#include "vf_loop.h"
#include "common_utils.h"

namespace codegen {

#define MAX_VF_AXIS_MERGE_SIZE 2

class VfCall final : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit VfCall(const std::string &api_name = "VfCall") : ApiCall(api_name), root_loop_(af::kIdNone) {}
  ~VfCall() final;

  // 生成vf func的定义
  Status GenerateFuncDefinition(const TPipe &tpipe, const Tiler &tiler, std::stringstream &ss) const override;

  // 生成vf func的调用
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                  const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const override;

  Status ParseAttr(const ascir::NodeView &node) override;
  Status ParseSubGraph(const ascir::NodeView &vf_node, const ascir::ImplGraph &graph);

 private:
  Status ParseInputOutputInfo(const TPipe &tpipe) const;
  void SetNodeAxisIds(const std::vector<ascir::AxisId> &origin_axis_ids);
  bool ShouldInitAsMaskReg(const ascir::NodeView &node, const af::AscTensor *output) const;

 private:
  std::string vf_call_name_;
  TensorManager tensor_mgr_;
  VFLoop root_loop_;
  std::string max_dtype_size_;
  std::vector<ascir::AxisId> axis_ids_;
  mutable std::vector<Tensor> ub_inputs_;
  mutable std::vector<Tensor> ub_outputs_;
  mutable std::vector<Tensor> scalar_inputs_;
  mutable std::vector<ascir::TensorId> subgraph_scalar_ids_;
};
}  // namespace codegen
#endif  // __AUTOFUSE_VEC_FUNC_CALL_H__
