/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "asc_graph_builder.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#define private public
#include "optimize/pre_process/improve_precision.h"
#include "optimize/pre_process/pre_process_config.h"
#undef private
#include "ascgraph_info_complete.h"
#include "platform_context.h"
#include "runtime_stub.h"

using namespace af;
using namespace af::testing;
using namespace af::pre_process;
using namespace af::ascir_op;

namespace {

size_t CountNodesByType(AscGraph &graph, const std::string &type) {
  size_t count = 0U;
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetType() == type) {
      ++count;
    }
  }
  return count;
}

bool HasNodeWithName(AscGraph &graph, const std::string &name) {
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetName() == name) {
      return true;
    }
  }
  return false;
}

class TestImprovePrecisionUT : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    PreProcessConfig::Instance().Reset();
    auto stub_v1 = std::make_shared<ge::RuntimeStub>();
    ge::RuntimeStub::SetInstance(stub_v1);
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }

  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
    unsetenv("AUTOFUSE_FLAGS");
    PreProcessConfig::Instance().Reset();
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};
}  // namespace

TEST_F(TestImprovePrecisionUT, Fp16ToFp16CastBeforeStore_CastDeleted) {
  auto graph = AscGraphBuilder("ut_fp16_to_fp16_cast_before_store")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Cast("cast0", "abs0", ge::DT_FLOAT16)
                   .Store("store0", "cast0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  EXPECT_FALSE(HasNodeWithName(graph, "cast0"));
}

TEST_F(TestImprovePrecisionUT, Fp32ToFp16CastBeforeStore_CastDeletedAndAbsPromoted) {
  auto graph = AscGraphBuilder("ut_fp32_to_fp16_before_store")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Cast("cast0", "abs0", ge::DT_FLOAT16)
                   .Store("store0", "cast0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  EXPECT_FALSE(HasNodeWithName(graph, "cast0"));
}
