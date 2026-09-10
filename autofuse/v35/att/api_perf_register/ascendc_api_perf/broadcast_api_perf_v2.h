/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef AUTOFUSE_ASCENDC_BROADCAST_API_PERF_V2_H_
#define AUTOFUSE_ASCENDC_BROADCAST_API_PERF_V2_H_

#include "api_perf_register/api_perf.h"

namespace att {
namespace ascendcperf_v2 {

af::Status BroadcastPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
bool IsBroadcastFallback(const NodeDetail &node_info);

}  // namespace ascendcperf_v2
}  // namespace att

#endif  // AUTOFUSE_ASCENDC_BROADCAST_API_PERF_V2_H_
