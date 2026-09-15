/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms of
 * CANN Open Software License Agreement Version 2.0 of the License.
 */

#ifndef AUTOFUSE_ASCENDC_BROADCAST_NLAST_AXIS_PERF_V2_H_
#define AUTOFUSE_ASCENDC_BROADCAST_NLAST_AXIS_PERF_V2_H_

#include "broadcast_perf_utils_v2.h"

namespace att {
namespace ascendcapi_v2 {

enum class NlastAxisBranch {
  kGather,
  kGatherWrapperForFourDim,
  kGatherOne,
  kGatherTwo,
  kGatherBOne,
  kGatherBTwo,
  kLessThanVlAligned,
  kLessThanVlUnaligned,
  kLargerThanVlAlignedWithBlock,
  kLargerThanVlAlignedWithVl,
  kLargerThanVlUnaligned,
  kDynamicGather,
  kDynamicLessThanVlAligned,
  kDynamicLessThanVlUnaligned,
  kDynamicLargerThanVlAlignedWithBlock,
  kDynamicLargerThanVlUnaligned,
  kB64MoreDimGather,
  kFallback,
};

NlastAxisBranch GetNlastAxisBranch(const BroadcastTilingInfo &tiling, const std::string &dtype, int32_t const_rank);

}  // namespace ascendcapi_v2

namespace ascendcperf_v2 {

ascendcapi_v2::NlastAxisBranch GetNlastAxisPerfBranch(const NodeDetail &node_info);
bool HasUnknownNlastCondition(const NodeDetail &node_info, const ascendcapi_v2::BroadcastTilingInfo &tiling);
af::Status BuildNlastAxisPerf(const NodeDetail &node_info, const ascendcapi_v2::BroadcastTilingInfo &tiling,
                              PerfOutputInfo &perf);

}  // namespace ascendcperf_v2
}  // namespace att

#endif  // AUTOFUSE_ASCENDC_BROADCAST_NLAST_AXIS_PERF_V2_H_
