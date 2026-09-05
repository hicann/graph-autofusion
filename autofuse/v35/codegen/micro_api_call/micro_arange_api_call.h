/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_MICRO_ARANGE_API_CALL_H
#define AUTOFUSE_MICRO_ARANGE_API_CALL_H

#include "micro_api_call.h"

namespace codegen {
class MicroArangeApiCall final : public MicroApiCall {
 public:
  explicit MicroArangeApiCall(const std::string &api_name) : MicroApiCall(api_name) {}
  ~MicroArangeApiCall() final = default;

  Status Init(const ascir::NodeView &node) override;
  Status Generate(const TensorManager &tensor_mng, const TPipe &tpipe, CallParam &param, std::string &result) override;
  bool HasArangeParam() const override {
    return true;
  }
  void GetArangeParams(const TPipe &tpipe, std::string &base, std::string &step) const override;

 private:
  af::Expression base_;
  af::Expression step_;
};
}  // namespace codegen

#endif  // AUTOFUSE_MICRO_ARANGE_API_CALL_H
