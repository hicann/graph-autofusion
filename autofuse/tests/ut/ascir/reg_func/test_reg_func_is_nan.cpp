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

#include "graph/operator_reg_af.h"
#include "graph_utils_ex.h"
#include "node_utils.h"
#include "op_desc_utils.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "../test_util.h"
namespace af {
namespace ascir {
extern std::vector<std::unique_ptr<af::TmpBufDesc>> CalcIsnanTmpSize(const af::AscNode &node);

using namespace testing;

constexpr int32_t ONE_BLK_SIZE = 32;
constexpr int32_t ONE_REPEAT_BYTE_SIZE = 256;
constexpr int32_t MAX_REPEAT_NUM = 255;

class CalcIsnanTmpSizeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST(IsnanInferDtypeTest, DefaultsBoolCompatibleOutputToBool) {
  for (const auto *npu_arch : {"3510", "5102"}) {
    std::vector<af::DataType> output_dtypes;

    EXPECT_EQ(af::ascir_op::Isnan::InferDataType({af::DT_FLOAT16}, output_dtypes, npu_arch), af::SUCCESS);
    ASSERT_EQ(output_dtypes.size(), 1U);
    EXPECT_EQ(output_dtypes[0], af::DT_BOOL);
  }
}

TEST(IsnanInferDtypeTest, AcceptsExplicitUint8Output) {
  std::vector<af::DataType> output_dtypes{af::DT_UINT8};

  EXPECT_EQ(af::ascir_op::Isnan::InferDataType({af::DT_FLOAT16}, output_dtypes, "5102"), af::SUCCESS);
  ASSERT_EQ(output_dtypes.size(), 1U);
  EXPECT_EQ(output_dtypes[0], af::DT_UINT8);
}

TEST(IsnanInferDtypeTest, DefaultsBoolCompatibleOutputToBoolWithoutCheck) {
  std::vector<af::DataType> output_dtypes;

  EXPECT_EQ(af::ascir_op::Isnan::InferDataTypeWithNoCheck({af::DT_FLOAT16}, output_dtypes, "5102"), af::SUCCESS);
  ASSERT_EQ(output_dtypes.size(), 1U);
  EXPECT_EQ(output_dtypes[0], af::DT_BOOL);
}

/**
 * @tc.name:CalcIsnanTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
 * @tc.number: CalcIsnanTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcIsnanTmpSize returns correct size
 */
TEST_F(CalcIsnanTmpSizeTest, CalcIsnanTmpSize_ShouldReturnCorrectSize_WhenNodelsValid) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto zo = graph.CreateAxis("zo", s1 + s2);
  auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, af::kIdNone);
  auto zo_s_1 = graph.CreateAxis("zo_s_1", Axis::Type::kAxisTypeOriginal, s2, {zo.id}, af::kIdNone);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Isnan isnan("isnan");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, zo_s_0.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, zo_s_0.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, zo_s_0.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, zo_s_0.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, zo_s_0.id};

  isnan.x = load1.y;
  isnan.attr.sched.axis = {z0.id, zo_s_0.id};
  isnan.y.dtype = af::DT_FLOAT;
  *isnan.y.axis = {z0.id, zo_s_0.id};
  *isnan.y.repeats = {s0, s1};
  *isnan.y.strides = {s1, Symbol(1)};

  store.x = isnan.y;
  store.attr.sched.axis = {z0.id, zo_s_0.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, zo_s_0.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, zo_s_0.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, zo_s_0.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("isnan");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcIsnanTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, sym::Min(sym::Max(Symbol(4) * s0 * s1 + af::Symbol(ONE_BLK_SIZE), af::Symbol(8192)),
                                      af::Symbol(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE)));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
}  // namespace ascir
}  // namespace af
