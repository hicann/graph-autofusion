/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"

#include "graph/ascendc_ir/ascir_registry.h"
#include "v1_ascir_att_impl.h"

namespace af {
namespace ascir {

class AscIrAttImplTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Each AscIrAtt impl exposes the IR name through both perf-table entry points.
#define EXPECT_ATT_IMPL_NAMED(ir_name)                                                \
  {                                                                                   \
    JOIN(ir_name, AscIrAttImpl) impl;                                                 \
    EXPECT_STREQ(static_cast<const char *>(impl.GetApiPerf()), #ir_name);             \
    EXPECT_STREQ(static_cast<const char *>(impl.GetAscendCApiPerfTable()), #ir_name); \
  }

TEST_F(AscIrAttImplTest, GetApiPerfReturnsIrName_ElementWise) {
  EXPECT_ATT_IMPL_NAMED(Add);
  EXPECT_ATT_IMPL_NAMED(Gather);
  EXPECT_ATT_IMPL_NAMED(Abs);
  EXPECT_ATT_IMPL_NAMED(Broadcast);
  EXPECT_ATT_IMPL_NAMED(Cast);
  EXPECT_ATT_IMPL_NAMED(Div);
  EXPECT_ATT_IMPL_NAMED(Erf);
  EXPECT_ATT_IMPL_NAMED(Exp);
  EXPECT_ATT_IMPL_NAMED(LogicalAnd);
  EXPECT_ATT_IMPL_NAMED(LogicalOr);
  EXPECT_ATT_IMPL_NAMED(LogicalNot);
}

TEST_F(AscIrAttImplTest, GetApiPerfReturnsIrName_ReduceArgMax) {
  EXPECT_ATT_IMPL_NAMED(ReduceArgMax);
  EXPECT_ATT_IMPL_NAMED(ReduceArgMaxMultiRPhase1);
  EXPECT_ATT_IMPL_NAMED(ReduceArgMaxMultiRPhase2);
}

TEST_F(AscIrAttImplTest, GetApiPerfReturnsIrName_NoModeling) {
  EXPECT_ATT_IMPL_NAMED(Data);
  EXPECT_ATT_IMPL_NAMED(Scalar);
  EXPECT_ATT_IMPL_NAMED(IndexExpr);
  EXPECT_ATT_IMPL_NAMED(Output);
  EXPECT_ATT_IMPL_NAMED(Workspace);
  EXPECT_ATT_IMPL_NAMED(MatMul);
  EXPECT_ATT_IMPL_NAMED(Conv2D);
  EXPECT_ATT_IMPL_NAMED(Pad);
  EXPECT_ATT_IMPL_NAMED(Nop);
  EXPECT_ATT_IMPL_NAMED(Ln);
  EXPECT_ATT_IMPL_NAMED(Isnan);
}

}  // namespace ascir
}  // namespace af
