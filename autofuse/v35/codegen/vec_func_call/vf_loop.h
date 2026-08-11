/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __AUTOFUSE_VF_LOOP_H__
#define __AUTOFUSE_VF_LOOP_H__
#include "codegen_kernel.h"
#include "../micro_api_call/micro_api_call.h"

namespace codegen {

class VFLoop;
struct VFLoopBody {
  LoopType type_;
  union {
    MicroApiCall *call_;
    VFLoop *loop_;
  };
};

class VFLoop {
 public:
  explicit VFLoop(const ascir::AxisId axis);
  void AddLoop(VFLoop *loop);
  void AddCall(MicroApiCall *call);

  /* 图解析阶段调用 */
  Status ConstructFromNodes(ascir::NodeViewVisitorConst nodes, const ascir::NodeView &vf_node);
  void Destruct();

  /* kernel生成阶段调用 */
  Status Generate(const TPipe &tpipe, const TensorManager &tensor_mgr, int32_t depth, std::string &result,
                  std::string &loop_size_result, int32_t &only_loop_max_depth,
                  std::vector<std::string> &loop_size_vec) const;
  void SetMaxDtypeSize(std::string dtype);
  void CollectMaskRegTempTensors(const TPipe &tpipe, const TensorManager &tensor_mgr,
                                 std::vector<std::string> &temp_tensors) const;

 private:
  ascir::AxisId axis_id_;
  struct VFLoop *parent_;
  std::vector<VFLoopBody> bodys_;
  std::string max_dtype_size_;

  Status GenerateLoop(const TPipe &tpipe, const TensorManager &tensor_mgr, int32_t depth,
                      std::vector<ascir::AxisId> &current_axis, std::stringstream &ss, std::stringstream &loop_size_ss,
                      int32_t &only_loop_max_depth, std::vector<std::string> &loop_size_vec) const;
  Status GenerateBody(const TPipe &tpipe, const TensorManager &tensor_mgr, int32_t depth,
                      std::vector<ascir::AxisId> &current_axis, std::stringstream &ss, std::stringstream &loop_size_ss,
                      int32_t &only_loop_max_depth, std::vector<std::string> &loop_size_vec) const;
};

}  // namespace codegen
#endif  // __AUTOFUSE_VF_LOOP_H__
