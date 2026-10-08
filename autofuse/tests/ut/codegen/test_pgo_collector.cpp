/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/pgo/pgo_collector.h"
#include "codegen/pgo/pgo_profapi_adapter.h"

#include <dlfcn.h>

#include <gtest/gtest.h>

namespace codegen::pgo {
namespace {

TEST(PgoCollector, NullArgumentsReturnFailure) {
  EXPECT_NE(AutofusePgoCollectorCreate(0, nullptr, nullptr), 0);
  EXPECT_NE(AutofusePgoCollectorRecordLaunch(nullptr, 0), 0);
  EXPECT_NE(AutofusePgoCollectorEnd(nullptr), 0);
  uint64_t duration = 0;
  EXPECT_NE(AutofusePgoCollectorGetDurationNs(nullptr, &duration), 0);
  AutofusePgoCollectorDestroy(nullptr);
}

TEST(PgoCollector, CreatePropagatesCapabilityStatus) {
  void *collector = nullptr;
  EXPECT_EQ(AutofusePgoCollectorCreate(0, nullptr, &collector), PGO_CAPABILITY_UNAVAILABLE);
  EXPECT_EQ(collector, nullptr);
}

TEST(PgoProfApiAdapter, StopStatusPreservesChannelFailure) {
  EXPECT_EQ(CombinePgoStopStatus(PGO_SUCCESS, PGO_SUCCESS), PGO_SUCCESS);
  EXPECT_EQ(CombinePgoStopStatus(PGO_CHANNEL_UNAVAILABLE, PGO_SUCCESS), PGO_CHANNEL_UNAVAILABLE);
  EXPECT_EQ(CombinePgoStopStatus(PGO_SUCCESS, PGO_PROFILING_STOP_FAILED), PGO_PROFILING_STOP_FAILED);
}

TEST(PgoProfApiAdapter, PollPositiveChannelCountIsSuccess) {
  EXPECT_EQ(NormalizePgoPollStatus(0), PGO_SUCCESS);
  EXPECT_EQ(NormalizePgoPollStatus(1), PGO_SUCCESS);
  EXPECT_EQ(NormalizePgoPollStatus(-1), PGO_CHANNEL_UNAVAILABLE);
}

TEST(PgoProfApiAdapter, FlushNotSupportMatchesMsptiNoOp) {
  EXPECT_TRUE(IsPgoFlushUnsupportedStatus(static_cast<int32_t>(0xfffe)));
  EXPECT_FALSE(IsPgoFlushUnsupportedStatus(1));
}

TEST(PgoProfApiAdapter, DlopenModeKeepsProfilerLibrariesResident) {
  const int flags = PgoDlopenFlags();
  EXPECT_EQ(flags & RTLD_NOW, RTLD_NOW);
#if defined(RTLD_LOCAL) && RTLD_LOCAL != 0
  EXPECT_EQ(flags & RTLD_LOCAL, RTLD_LOCAL);
#endif
#if defined(RTLD_NODELETE)
  EXPECT_EQ(flags & RTLD_NODELETE, RTLD_NODELETE);
#endif
}

}  // namespace
}  // namespace codegen::pgo
