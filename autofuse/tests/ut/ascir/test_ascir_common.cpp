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

#include "ascir_common.h"
#include "ascir.h"
#include "ascir_ops.h"

namespace af {
namespace ascir {

using namespace af::ascir_op;

class AscirCommonTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test: dtypes present in the conversion map are replaced on both the input and the output side
TEST_F(AscirCommonTest, GetConversionFromDtypeMap_ShouldConvertMappedDtypes) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Cast cast("cast");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  cast.x = load1.y;
  cast.attr.sched.axis = {z0.id, z1.id};
  cast.y.dtype = af::DT_INT32;
  *cast.y.axis = {z0.id, z1.id};
  *cast.y.repeats = {s0, s1};
  *cast.y.strides = {s1, Symbol(1)};
  *cast.y.vectorized_axis = {z0.id, z1.id};

  store.x = cast.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_INT32;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_INT32;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("cast");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  node->outputs[0].attr.vectorized_strides = {s1, Symbol(1)};

  std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{af::DT_FLOAT, af::DT_BF16},
                                                               {af::DT_INT32, af::DT_FLOAT16}};
  auto conversion = GetConversionFromDtypeMap(*node, dtype_conversion_map);
  ASSERT_EQ(conversion.first.size(), 1U);
  EXPECT_EQ(conversion.first[0], af::DT_BF16);
  ASSERT_EQ(conversion.second.size(), 1U);
  EXPECT_EQ(conversion.second[0], af::DT_FLOAT16);
}

// Test: both input and output have two vectorized axes whose strides are continuous -> true
TEST_F(AscirCommonTest, IsAllVecAxisContinuous_ShouldReturnTrue_WhenStridesContinuous) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Abs abs("abs");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  abs.x = load1.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, Symbol(1)};
  *abs.y.vectorized_axis = {z0.id, z1.id};

  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("abs");
  // repeats[axis_id] * strides[j] == strides[j-1]: s1 * 1 == s1 on both sides
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  node->outputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  EXPECT_TRUE(IsAllVecAxisContinuous(*node));
}

// Test: input strides are continuous but output strides are not -> false (output loop returns early)
TEST_F(AscirCommonTest, IsAllVecAxisContinuous_ShouldReturnFalse_WhenOutputStridesNotContinuous) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Abs abs("abs");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  abs.x = load1.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, Symbol(1)};
  *abs.y.vectorized_axis = {z0.id, z1.id};

  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("abs");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  // s1 * 1 != 2 * s1 -> the output loop detects discontinuity
  node->outputs[0].attr.vectorized_strides = {Symbol(2) * s1, Symbol(1)};
  EXPECT_FALSE(IsAllVecAxisContinuous(*node));
}

// Test: input strides themselves are not continuous -> false (input loop returns early)
TEST_F(AscirCommonTest, IsAllVecAxisContinuous_ShouldReturnFalse_WhenInputStridesNotContinuous) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Abs abs("abs");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  abs.x = load1.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, Symbol(1)};
  *abs.y.vectorized_axis = {z0.id, z1.id};

  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("abs");
  // s1 * 1 != 2 * s1 -> the input loop detects discontinuity
  node->inputs[0].attr.vectorized_strides = {Symbol(2) * s1, Symbol(1)};
  node->outputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  EXPECT_FALSE(IsAllVecAxisContinuous(*node));
}

}  // namespace ascir
}  // namespace af
