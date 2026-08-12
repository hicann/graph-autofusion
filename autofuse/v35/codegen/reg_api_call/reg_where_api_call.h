/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __AUTOFUSE_WHERE_REG_API_CALL_H__
#define __AUTOFUSE_WHERE_REG_API_CALL_H__
#include "codegen_kernel.h"
#include "api_call/utils/api_call_utils.h"

namespace codegen {
class WhereRegApiCall : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit WhereRegApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~WhereRegApiCall() final = default;
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                  const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const override;

 private:
  Status PrepareInputsAndOutputs(const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                 const std::vector<std::reference_wrapper<const Tensor>> &outputs, const Tensor *&x1,
                                 const Tensor *&x2, const Tensor *&x3, const Tensor *&y) const;
  Status GenerateLoopParams(const Tensor &x1, const Tensor &x2, const Tensor &x3, const Tensor &y, const TPipe &tpipe,
                            ApiLoopParams &param, VectorizedAxisLoopMergeStatus &merge_info) const;
  Status GenerateNoLoopCase(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, const Tensor &x1,
                            const Tensor &x2, const Tensor &x3, const Tensor &y, const std::string &x2_scalar,
                            const std::string &x3_scalar, std::stringstream &ss) const;
  Status GenerateBothScalarCase(const TPipe &tpipe, const ApiLoopParams &param, const Tensor &x1, const Tensor &y,
                                const std::string &scalar_local_blk_tensor_name_x2,
                                const std::string &scalar_local_blk_tensor_name_x3, std::stringstream &ss) const;
  Status GenerateX2ScalarCase(const TPipe &tpipe, const ApiLoopParams &param, const Tensor &x1, const Tensor &x3,
                              const Tensor &y, const std::string &scalar_local_blk_tensor_name_x2,
                              std::stringstream &ss) const;
  Status GenerateX3ScalarCase(const TPipe &tpipe, const ApiLoopParams &param, const Tensor &x1, const Tensor &x2,
                              const Tensor &y, const std::string &scalar_local_blk_tensor_name_x3,
                              std::stringstream &ss) const;
  Status GenerateNormalCase(const TPipe &tpipe, const ApiLoopParams &param, const Tensor &x1, const Tensor &x2,
                            const Tensor &x3, const Tensor &y, std::stringstream &ss) const;
};
}  // namespace codegen
#endif  // __AUTOFUSE_WHERE_REG_API_CALL_H__
