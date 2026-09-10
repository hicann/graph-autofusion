/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms of
 * CANN Open Software License Agreement Version 2.0 of the License.
 */

#ifndef AUTOFUSE_ASCENDC_BROADCAST_LAST_AXIS_PERF_V2_H_
#define AUTOFUSE_ASCENDC_BROADCAST_LAST_AXIS_PERF_V2_H_

#include "broadcast_perf_utils_v2.h"

namespace att {
namespace ascendcapi_v2 {

enum class LastAxisBranch {
  kE2B,
  kE2BLessThanVl,
  kE2BLargerThanVl,
  kGatherOne,
  kGatherTwo,
  kGatherWrapper,
  kGatherWrapperForFourDim,
  kLessThanVlAligned,
  kLessThanVlUnaligned,
  kLargerThanVlAligned,
  kLargerThanVlUnaligned,
  kDynamicLessThanVlUnaligned,
  kDynamicLargerThanVlUnaligned,
  kFallback,
};

LastAxisBranch GetLastAxisBranch(const BroadcastTilingInfo &tiling, const std::string &dtype, int32_t const_rank);

}  // namespace ascendcapi_v2

namespace ascendcperf_v2 {

ascendcapi_v2::LastAxisBranch GetLastAxisPerfBranch(const NodeDetail &node_info);
af::Status BuildLastAxisPerf(const NodeDetail &node_info, const ascendcapi_v2::BroadcastTilingInfo &tiling,
                             PerfOutputInfo &perf);

}  // namespace ascendcperf_v2
}  // namespace att

#endif  // AUTOFUSE_ASCENDC_BROADCAST_LAST_AXIS_PERF_V2_H_
