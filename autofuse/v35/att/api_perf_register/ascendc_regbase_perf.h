/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_ASCENDC_REGBASE_PERF_H
#define AUTOFUSE_ASCENDC_REGBASE_PERF_H
#include "api_perf_register/utils/vf_perf_utils.h"
#include "api_perf_register/utils/api_perf_utils.h"
namespace att {
namespace ascendcperf_v2 {
// 工具函数，提取重复代码
struct RepeatParams {
  Expr repeat_elm;
  Expr repeat_time;
};
RepeatParams CalculateRepeatParams(const std::string &input_dtype, const Expr &cal_count);
// 注册V2性能
af::Status CompareGEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareEQPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareNEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareGTPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareLEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CompareLTPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status AbsPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status ExpPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status LnPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status SqrtPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status RsqrtPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status DivPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status ReciprocalPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status ReluPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status MaxPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status MinPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status ReduceMaxPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status ReduceMinPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status NegPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status MeanPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status AddPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status SubPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status MulPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status LeakyReluPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status CastPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status SumPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status RemovePadPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status WherePerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status PowPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status ErfPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status TanhPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status SigmoidPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status GeluPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status SignPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status LogicalNotPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status LogicalOrPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status LogicalAndPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status ClipByValuePerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status BitwiseAndPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status FloorDivPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status IsNanPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status IsFinitePerf(const NodeDetail &node_info, PerfOutputInfo &perf);
af::Status TransposePerf(const NodeDetail &node_info, PerfOutputInfo &perf);
}  // namespace ascendcperf_v2
}  // namespace att

#endif  // AUTOFUSE_ASCENDC_REGBASE_PERF_H
