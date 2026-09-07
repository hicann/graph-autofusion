/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_GROUP_PERF_CONFIG_H_
#define ATT_GROUP_PERF_CONFIG_H_

namespace att {

// Values are from two cold 2048x1024, block_dim=52, g0 -> SyncAll -> g1 samples:
// edge median 3208.7/8510.2 cycles and p95 5358.1/9845.5 cycles. They are deliberately
// conservative and must be revalidated with the profile matrix before changing.
struct SerializedEdgePenaltyConfig {
  static constexpr double kPenaltyRatio = 0.10;
  static constexpr double kPenaltyCapCycles = 8000.0;
  static constexpr const char *kModelVersion = "serialized_edge_v1";
};

}  // namespace att

#endif  // ATT_GROUP_PERF_CONFIG_H_
