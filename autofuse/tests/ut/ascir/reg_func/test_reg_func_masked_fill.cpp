/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "default_reg_func.h"
namespace af {
namespace ascir {

const Expression MAX_TMP_BUFFER_SIZE = af::Symbol(255 * 256 + 32);
using namespace testing;

class CalcMaskedFillTmpSizeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

struct MaskedFillTmpSizeCase {
  std::shared_ptr<af::AscNode> node;
  Expression s0;
  Expression s1;
};

MaskedFillTmpSizeCase BuildMaskedFillTmpSizeCase(af::AscGraph &graph, const std::vector<Expression> &repeats,
                                                 const std::vector<Expression> &strides, bool use_symbol_shape,
                                                 bool use_default_vectorized_axis) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto zo = graph.CreateAxis("zo", s1 + s2);
  auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, af::kIdNone);
  auto actual_repeats = use_symbol_shape ? std::vector<Expression>{s0, s1} : repeats;
  auto actual_strides = use_symbol_shape ? std::vector<Expression>{s1, Symbol(1)} : strides;
  auto actual_vectorized_axis =
      use_default_vectorized_axis ? std::vector<int64_t>{z0.id, zo_s_0.id} : std::vector<int64_t>{};

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::MaskedFill masked_fill("masked_fill");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, zo_s_0.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, zo_s_0.id};
  *x1.y.repeats = actual_repeats;
  *x1.y.strides = actual_strides;

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, zo_s_0.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, zo_s_0.id};
  *load1.y.repeats = actual_repeats;
  *load1.y.strides = actual_strides;
  *load1.y.vectorized_axis = actual_vectorized_axis;

  masked_fill.x = load1.y;
  masked_fill.mask = load1.y;
  masked_fill.value = load1.y;
  masked_fill.attr.sched.axis = {z0.id, zo_s_0.id};
  masked_fill.y.dtype = af::DT_FLOAT;
  *masked_fill.y.axis = {z0.id, zo_s_0.id};
  *masked_fill.y.repeats = actual_repeats;
  *masked_fill.y.strides = actual_strides;

  store.x = masked_fill.y;
  store.attr.sched.axis = {z0.id, zo_s_0.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, zo_s_0.id};
  *store.y.repeats = actual_repeats;
  *store.y.strides = actual_strides;

  y.x = store.y;
  y.attr.sched.axis = {z0.id, zo_s_0.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, zo_s_0.id};
  *y.y.repeats = actual_repeats;
  *y.y.strides = actual_strides;

  return {graph.FindNode("masked_fill"), s0, s1};
}

/**
 * @tc.name: CalcMaskedFillTmpSize_ShouldReturnCorrectSize_WhenNodeIsValid
 * @tc.number: CalcMaskedFillTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcMaskedFillTmpSize returns correct size
 */
TEST_F(CalcMaskedFillTmpSizeTest, CalcMaskedFillTmpSize_ShouldReturnCorrectSize_WhenNodeIsValid) {
  af::AscGraph graph("test");
  auto test_case = BuildMaskedFillTmpSizeCase(graph, {}, {}, true, true);
  test_case.node->inputs[0].attr.vectorized_strides = {test_case.s1, Symbol(1)};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcMaskedFillTmpSize(*test_case.node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, sym::Min(af::Symbol(12) * test_case.s1 * test_case.s0, MAX_TMP_BUFFER_SIZE));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcMaskedFillTmpSizeTest, CalcMaskedFillTmpSize_ShouldReturnCorrectSize_Scalar) {
  af::AscGraph graph("test");
  auto test_case = BuildMaskedFillTmpSizeCase(graph, {Symbol(1), Symbol(1)}, {Symbol(0), Symbol(0)}, false, true);
  test_case.node->inputs[0].attr.vectorized_strides = {};
  test_case.node->inputs[1].attr.vectorized_strides = {Symbol(0), Symbol(0)};
  test_case.node->inputs[2].attr.vectorized_strides = {Symbol(0), Symbol(0)};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcMaskedFillTmpSize(*test_case.node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(1) * af::Symbol(12));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcMaskedFillTmpSizeTest, CalcMaskedFillTmpSize_ShouldReturnCorrectSize_ConstScalar) {
  af::AscGraph graph("test");
  auto test_case = BuildMaskedFillTmpSizeCase(graph, {}, {}, false, false);
  test_case.node->inputs[0].attr.vectorized_strides = {};
  test_case.node->inputs[1].attr.vectorized_strides = {};
  test_case.node->inputs[2].attr.vectorized_strides = {};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcMaskedFillTmpSize(*test_case.node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(12));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
}  // namespace ascir
}  // namespace af
