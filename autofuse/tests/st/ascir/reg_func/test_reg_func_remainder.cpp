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
#include "default_reg_func.h"

namespace af {
namespace ascir {

using namespace testing;
using namespace af::ascir_op;

class CalcRemainderTmpSizeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(CalcRemainderTmpSizeTest, CalcRemainderTmpSize_ShouldReturnCorrectSize_WhenNodeIsUbScalar) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Data x2("x2", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Load load2("load2");
  af::ascir_op::Remainder remainder("remainder");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  x2.attr.sched.axis = {z0.id, z1.id};
  x2.y.dtype = ge::DT_FLOAT;
  *x2.y.axis = {z0.id, z1.id};
  *x2.y.repeats = {Symbol(1), Symbol(1)};
  *x2.y.strides = {Symbol(1), Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  load2.x = x2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {Symbol(1), Symbol(1)};
  *load2.y.strides = {Symbol(1), Symbol(1)};
  *load2.y.vectorized_axis = {z0.id, z1.id};

  remainder.dividend = load1.y;
  remainder.divisor = load2.y;
  remainder.attr.sched.axis = {z0.id, z1.id};
  remainder.y.dtype = ge::DT_FLOAT;
  *remainder.y.axis = {z0.id, z1.id};
  *remainder.y.repeats = {s0, s1};
  *remainder.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("remainder");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcRemainderTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(8192));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcRemainderTmpSizeTest, CalcRemainderTmpSize_ShouldReturnCorrectSize_WhenNodeIsScalar) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Scalar x2("x2", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Remainder remainder("remainder");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  x2.attr.sched.axis = {z0.id, z1.id};
  x2.y.dtype = DT_FLOAT;
  *x2.y.axis = {};
  *x2.y.repeats = {};
  *x2.y.strides = {};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  remainder.dividend = load1.y;
  remainder.divisor = x2.y;
  remainder.attr.sched.axis = {z0.id, z1.id};
  remainder.y.dtype = ge::DT_FLOAT;
  *remainder.y.axis = {z0.id, z1.id};
  *remainder.y.repeats = {s0, s1};
  *remainder.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("remainder");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcRemainderTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(8192));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcRemainderTmpSizeTest, CalcRemainderTmpSize_ShouldReturnCorrectSize_WhenNodeIsTensor) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Remainder remainder("remainder");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  remainder.dividend = load1.y;
  remainder.divisor = load1.y;
  remainder.attr.sched.axis = {z0.id, z1.id};
  remainder.y.dtype = ge::DT_FLOAT;
  *remainder.y.axis = {z0.id, z1.id};
  *remainder.y.repeats = {s0, s1};
  *remainder.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("remainder");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcRemainderTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  // For tensor input, temp size = 3 * 4 (float) * s1
  ASSERT_EQ(result[0]->size, af::Symbol(12) * s1);
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcRemainderTmpSizeTest, CalcRemainderTmpSize_ShouldReturnCorrectSize_WhenNodeIsInt32) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Remainder remainder("remainder");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_INT32;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_INT32;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  remainder.dividend = load1.y;
  remainder.divisor = load1.y;
  remainder.attr.sched.axis = {z0.id, z1.id};
  // For int32 input, output is float
  remainder.y.dtype = ge::DT_FLOAT;
  *remainder.y.axis = {z0.id, z1.id};
  *remainder.y.repeats = {s0, s1};
  *remainder.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("remainder");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcRemainderTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  // For int32 input, temp size = 3 * 4 (float) * s1
  ASSERT_EQ(result[0]->size, af::Symbol(12) * s1);
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcRemainderTmpSizeTest, GetRemainderTmpBufferFactorSize_ShouldReturnCorrectValues) {
  uint32_t maxLiveNodeCount = 0;
  uint32_t extraBuf = 0;

  GetRemainderTmpBufferFactorSize(4, maxLiveNodeCount, extraBuf);  // float
  ASSERT_EQ(maxLiveNodeCount, 3);
  ASSERT_EQ(extraBuf, 0);
}

}  // namespace ascir
}  // namespace af
