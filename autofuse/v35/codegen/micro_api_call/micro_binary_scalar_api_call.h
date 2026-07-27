/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __AUTOFUSE_MICRO_BINARY_SCALAR_API_CALL_H__
#define __AUTOFUSE_MICRO_BINARY_SCALAR_API_CALL_H__

#include "micro_api_call.h"

namespace codegen {
class MicroBinaryScalarApiCall final : public MicroApiCall {
 public:
  explicit MicroBinaryScalarApiCall(const std::string &api_name) : MicroApiCall(api_name) {}
  ~MicroBinaryScalarApiCall() override = default;
  Status Init(const ascir::NodeView &node) override;
  Status Generate(const TensorManager &tensor_mng, const TPipe &tpipe, CallParam &param, std::string &result) override;

 private:
  bool has_scalar_input_ = false;
  bool need_exchange_inputs_ = false;
};
}  // namespace codegen

#endif  // __AUTOFUSE_MICRO_BINARY_SCALAR_API_CALL_H__
