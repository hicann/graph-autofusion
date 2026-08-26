/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef API_PERF_REGISTER_ASCENDC_API_PERF_H_
#define API_PERF_REGISTER_ASCENDC_API_PERF_H_

#include <map>
#include <mutex>
#include <unordered_map>
#include <sstream>
#include "base/model_info.h"
#include "utils/api_perf_utils.h"

namespace att {
// 负责注册所有的评估函数（包括ASCIR API, AscendC Api）
class EvalCosts {
 public:
  static EvalCosts &Instance();

  void RegisterFunc(const std::string &op_type, const Perf &perf_func) {
    std::lock_guard<std::mutex> lock(mutex_);
    func_container_[op_type] = perf_func;
  }
  void RegisterAscendCFunc(const std::string &op_type, const AscendCPerf &ascendc_perf_func) {
    std::lock_guard<std::mutex> lock(mutex_);
    ascendc_func_container_[op_type] = ascendc_perf_func;
  }

  Perf GetFunc(const std::string &op_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = func_container_.find(op_type);
    return iter == func_container_.end() ? nullptr : iter->second;
  }
  AscendCPerf GetAscendCFunc(const std::string &op_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = ascendc_func_container_.find(op_type);
    return iter == ascendc_func_container_.end() ? nullptr : iter->second;
  }

 private:
  EvalCosts() = default;
  ~EvalCosts() = default;

 private:
  std::unordered_map<std::string, Perf> func_container_;
  std::unordered_map<std::string, AscendCPerf> ascendc_func_container_;
  std::mutex mutex_;
};

class FuncRegister {
 public:
  FuncRegister(const std::string &op_type, const Perf &func) {
    EvalCosts::Instance().RegisterFunc(op_type, func);
  }
  ~FuncRegister() = default;
};
class AscendcFuncRegister {
 public:
  AscendcFuncRegister(const std::string &op_type, const AscendCPerf &func) {
    EvalCosts::Instance().RegisterAscendCFunc(op_type, func);
  }
  ~AscendcFuncRegister() = default;
};

// 1.用于获取AscendC API的性能函数
// 2.默认ASCIR的性能函数也通过该接口获取
Perf GetPerfFunc(const std::string &op_type);
AscendCPerf GetAscendCPerfFunc(const std::string &op_type);

namespace ascendcperf {
af::Status LoadPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status GetDatasize(const TensorShapeInfo &shape, Expr &dim_product);
af::Status AbsPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status AddsPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status AddPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status AndPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status BlockReduceMaxPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status BlockReduceMinPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status BrcbPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CastPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CopyUbtoUbPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CopyPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareEQPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareScalarEQPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareGEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareScalarGEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareGTPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareScalarGTPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareLEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareScalarLEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareLTPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareScalarLTPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareNEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareScalarNEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status DivPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status DuplicatePerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status ErfPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status ExpPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status GatherPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status GatherMaskPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status MaxsPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status MaxPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status MinsPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status MinPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status MulsPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status MulPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status OrPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status PairReduceSumPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status ReciprocalPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status ReluPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status RsqrtPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status SelectPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status SigmoidPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status SignPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status SqrtPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status SubPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status TanhPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status WholeReduceSumPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status WholeReduceMaxPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status WholeReduceMinPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
// load store的API在不同形态的拟合形式不一样，暂不放在本文件定义
}  // namespace ascendcperf
// AscendC和ASCIR均可以使用下面的宏注册
#define REGISTER_EVAL_FUNC(op_type, func_name) FuncRegister eval_##op_type(op_type, func_name)
#define REGISTER_ASCENDC_EVAL_FUNC(op_type, func_name) AscendcFuncRegister ascendc_eval_##op_type(op_type, func_name)

#define REGISTER_ASCENDC_EVAL_FUNC_TAG(op_type, tag, func_name) \
  AscendcFuncRegister JOIN_A_B_C(eval_, op_type, tag)((op_type) + (#tag), (func_name))

#define REGISTER_EVAL_FUNC_TAG(op_type, tag, func_name) \
  FuncRegister JOIN_A_B_C(eval_, op_type, tag)((op_type) + (#tag), (func_name))
}  // namespace att
#endif  // API_PERF_REGISTER_ASCENDC_API_PERF_H_
