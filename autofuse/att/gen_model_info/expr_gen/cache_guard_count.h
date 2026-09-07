/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef EXPR_GEN_CACHE_GUARD_COUNT_H_
#define EXPR_GEN_CACHE_GUARD_COUNT_H_

#include "base/base_types.h"

namespace att {
enum class CacheGuardKind { kNone, kOriginBroadcast, kFusedBroadcast };

struct CacheGuardInfo {
  CacheGuardKind kind{CacheGuardKind::kNone};
  Expr guard_axis;
  Expr period;
  Expr block_inner_extent;
  Expr loop_extent;
  bool address_invariant{false};
  // Set only when metadata came from parser/scheduler axis repeats whose
  // positive range is guaranteed by the scheduling contract.  Generic
  // symbolic expressions must leave this false and take the legacy path.
  bool positive_range_proven{false};
};

Expr CountGuardHits(const CacheGuardInfo &info, const Expr &block_idx);
bool IsBlockCountUniform(const CacheGuardInfo &info, const Expr &block_dim);
bool ValidateCacheGuardInfo(const CacheGuardInfo &info);
}  // namespace att

#endif  // EXPR_GEN_CACHE_GUARD_COUNT_H_
