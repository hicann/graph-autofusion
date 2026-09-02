/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <string>
#include <fstream>
#define private public
#include "ascir_ops.h"
// #include "ascir.h"
#include "ascir_ops_utils.h"
#include "common_utils.h"
#include "ascir_register.h"
#include "graph/types_af.h"
#include "graph/tensor_af.h"

using namespace af::ops;
using namespace af::ascir_op;
namespace ascgen_utils {
class CommonUtilsTest : public testing::Test {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(CommonUtilsTest, IsStaticSchedResultTest) {
  ascir::FusedScheduledResult static_result;
  static_result.origin_vars.push_back(ge::Symbol(10));
  static_result.origin_vars.push_back(ge::Symbol(20));
  EXPECT_EQ(IsStaticSchedResult({static_result}), true);
}

TEST_F(CommonUtilsTest, ScalarValuePreProcessTest) {
  std::string after_pre_pro_value;
  EXPECT_EQ(ScalarValuePreProcess("inf", "float", after_pre_pro_value), 0);
  EXPECT_EQ(after_pre_pro_value, "AfInfinity<float>()");

  EXPECT_EQ(ScalarValuePreProcess("inf", "half", after_pre_pro_value), 0);
  EXPECT_EQ(after_pre_pro_value, "AfInfinity<half>()");

  EXPECT_NE(ScalarValuePreProcess("inf", "bfloat16_t", after_pre_pro_value), 0);
}

// 测试大于2个输入的node, 不支持brc inline
TEST_F(CommonUtilsTest, GreateTwoInputsNotSupportBrcInline) {
  ge::AscGraph graph("test_graph");
  Data x_op1("x1", graph);
  Data x_op2("x2", graph);
  Data x_op3("x3", graph);
  Load load_op1("load1");
  Load load_op2("load2");
  Load load_op3("load3");
  af::ascir_op::Where where_op("where");
  graph.AddNode(load_op1);
  graph.AddNode(load_op2);
  graph.AddNode(load_op3);
  graph.AddNode(where_op);

  load_op1.x = x_op1.y;
  load_op2.x = x_op2.y;
  load_op3.x = x_op3.y;
  where_op.x1 = load_op1.y;
  where_op.x2 = load_op2.y;
  where_op.x3 = load_op3.y;
  auto where = graph.FindNode("where");
  std::vector<uint8_t> input_idx_2_brc_inline;
  const bool ret = IsGeneralizeBrcInlineScene(where, input_idx_2_brc_inline);
  EXPECT_EQ(ret, false);
  auto BlkSupportFlag = IsSupportBlkTensorInput(where);
  EXPECT_EQ(BlkSupportFlag, true);
  auto load1 = graph.FindNode("load1");
  auto BlkTensorSupportFlag = IsScalarNextNodeSupportBlkTensor(load1);
  EXPECT_EQ(BlkTensorSupportFlag, true);
}

// 测试2个输入都包含广播轴, 不支持brc inline
TEST_F(CommonUtilsTest, TwoInputsHasbrcAxisNotSupportBrcInline) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data x_op("x0", graph);
  Data x_op1("x1", graph);
  Load load_op("load");
  Load load_op1("load1");
  af::ascir_op::Add add_op("add");

  graph.AddNode(load_op);
  graph.AddNode(load_op1);
  graph.AddNode(add_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, One, s3};

  load_op1.x = x_op1.y;
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op1.y.repeats = {s0, s1, s2, One};

  add_op.x1 = load_op.y;
  add_op.x2 = load_op1.y;
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *add_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *add_op.y.repeats = {s0, s1, s2, s3};

  // graph.SetInputs({x_op});

  auto load = graph.FindNode("load");
  load->outputs[0].attr.vectorized_axis = {z2.id, z3.id};

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.vectorized_axis = {z2.id, z3.id};

  auto add = graph.FindNode("add");
  add->outputs[0].attr.vectorized_axis = {z2.id, z3.id};

  std::vector<uint8_t> input_idx_2_brc_inline;
  const bool ret = IsGeneralizeBrcInlineScene(add, input_idx_2_brc_inline);
  EXPECT_EQ(ret, false);
}

// 测试1个输入包含广播轴, 广播轴不连续, 不支持brc inline
TEST_F(CommonUtilsTest, InputsHasNotConinueBrcAxisNotSupportBrcInline) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data x_op("x0", graph);
  Data x_op1("x1", graph);
  Load load_op("load");
  Load load_op1("load1");
  af::ascir_op::Add add_op("add");

  graph.AddNode(load_op);
  graph.AddNode(load_op1);
  graph.AddNode(add_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, One, s2, One};

  load_op1.x = x_op1.y;
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op1.y.repeats = {s0, s1, s2, s3};

  add_op.x1 = load_op.y;
  add_op.x2 = load_op1.y;
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *add_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *add_op.y.repeats = {s0, s1, s2, s3};

  // graph.SetInputs({x_op});

  auto load = graph.FindNode("load");
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {Zero, One, Zero};

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  load1->outputs[0].attr.vectorized_strides = {s1, s2, s3};

  auto add = graph.FindNode("add");
  add->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};

  std::vector<uint8_t> input_idx_2_brc_inline;
  const bool ret = IsGeneralizeBrcInlineScene(add, input_idx_2_brc_inline);
  EXPECT_EQ(ret, false);
}

// 测试1个输入包含广播轴, 广播轴合并之后不是首轴, 不支持brc inline
TEST_F(CommonUtilsTest, AfterMegerBrcAxisNotFirstAxisNotSupportBrcInline) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data x_op("x0", graph);
  Data x_op1("x1", graph);
  Load load_op("load");
  Load load_op1("load1");
  af::ascir_op::Add add_op("add");

  graph.AddNode(load_op);
  graph.AddNode(load_op1);
  graph.AddNode(add_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, One};

  load_op1.x = x_op1.y;
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op1.y.repeats = {s0, s1, s2, s3};

  add_op.x1 = load_op.y;
  add_op.x2 = load_op1.y;
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *add_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *add_op.y.repeats = {s0, s1, s2, s3};

  // graph.SetInputs({x_op});

  auto load = graph.FindNode("load");
  load->outputs[0].attr.vectorized_axis = {z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {One, Zero};

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.vectorized_axis = {z2.id, z3.id};
  load1->outputs[0].attr.vectorized_strides = {s2, s3};

  auto add = graph.FindNode("add");
  add->outputs[0].attr.vectorized_axis = {z2.id, z3.id};

  std::vector<uint8_t> input_idx_2_brc_inline;
  const bool ret = IsGeneralizeBrcInlineScene(add, input_idx_2_brc_inline);
  EXPECT_EQ(ret, false);
}

// 测试2个输入中包含都是1的场景，需要支持brc inline
TEST_F(CommonUtilsTest, AfterMegerBrcAxisisBothOneSupportBrcInline) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data x_op("x0", graph);
  Data x_op1("x1", graph);
  Load load_op("load");
  Load load_op1("load1");
  af::ascir_op::Add add_op("add");

  graph.AddNode(load_op);
  graph.AddNode(load_op1);
  graph.AddNode(add_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {One, One, One, s3};

  load_op1.x = x_op1.y;
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op1.y.repeats = {s0, One, One, s3};

  add_op.x1 = load_op.y;
  add_op.x2 = load_op1.y;
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *add_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *add_op.y.repeats = {s0, One, One, s3};

  auto load = graph.FindNode("load");
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {Zero, Zero, Zero, One};

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
  load1->outputs[0].attr.vectorized_strides = {s3, Zero, Zero, One};

  auto add = graph.FindNode("add");
  add->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};

  std::vector<uint8_t> input_idx_2_brc_inline;
  const bool ret = IsGeneralizeBrcInlineScene(add, input_idx_2_brc_inline);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(input_idx_2_brc_inline.size(), 2);
  EXPECT_EQ(input_idx_2_brc_inline[0], 1);
  EXPECT_EQ(input_idx_2_brc_inline[1], 0);
}

// 测试支持brc inline的BA场景，需要合轴，则需要带着对齐的stride，而不是所有repeats乘积（没对齐）
TEST_F(CommonUtilsTest, MergeRepeatsWithAlignedStrideToBA) {
  using ExpVec = std::vector<ge::Expression>;
  using S = ge::Symbol;
  //                        B      B      A     A      A -> BA
  ExpVec input0_repeats = {S(1), S(1), S(10), S(3), S(11)};
  ExpVec input1_repeats = {S(22), S(2), S(10), S(3), S(11)};
  ExpVec input1_strides = {S(960), S(480), S(48), S(16), S(1)};
  ExpVec i0_meger_repeates;
  ExpVec i1_meger_repeates;
  MergeBrcAxisRepeats(input0_repeats, input1_repeats, input1_strides, i0_meger_repeates, i1_meger_repeates);
  EXPECT_EQ(i0_meger_repeates.size(), 2);
  EXPECT_EQ(i1_meger_repeates.size(), 2);
  EXPECT_EQ(i0_meger_repeates[0], S(1));
  EXPECT_EQ(i0_meger_repeates[1], S(480));
  EXPECT_EQ(i1_meger_repeates[0], S(44));
  EXPECT_EQ(i1_meger_repeates[1], S(480));

  //                  1     1     B     1     B     1      A     A     1     A      1 -> BA
  input0_repeats = {S(1), S(1), S(1), S(1), S(1), S(1), S(10), S(3), S(1), S(11), S(1)};
  input1_repeats = {S(1), S(1), S(2), S(1), S(3), S(1), S(10), S(3), S(1), S(11), S(1)};
  input1_strides = {S(0), S(0), S(960), S(0), S(480), S(0), S(48), S(16), S(0), S(1), S(0)};
  MergeBrcAxisRepeats(input0_repeats, input1_repeats, input1_strides, i0_meger_repeates, i1_meger_repeates);
  EXPECT_EQ(i0_meger_repeates.size(), 2);
  EXPECT_EQ(i1_meger_repeates.size(), 2);
  EXPECT_EQ(i0_meger_repeates[0], S(1));
  EXPECT_EQ(i0_meger_repeates[1], S(480));
  EXPECT_EQ(i1_meger_repeates[0], S(6));
  EXPECT_EQ(i1_meger_repeates[1], S(480));
}

// 测试支持brc inline的ABA场景，需要合轴，则需要带着对齐的stride，而不是所有repeats乘积（没对齐）
TEST_F(CommonUtilsTest, MergeRepeatsWithAlignedStrideToABA) {
  using ExpVec = std::vector<ge::Expression>;
  using S = ge::Symbol;
  //                         A     B     1     A     A     A -> ABA
  ExpVec input0_repeats = {S(2), S(1), S(1), S(2), S(3), S(11)};
  ExpVec input1_repeats = {S(2), S(22), S(1), S(2), S(3), S(11)};
  ExpVec input0_strides = {};
  ExpVec input1_strides = {S(192), S(96), S(0), S(48), S(16), S(1)};
  ExpVec i0_meger_repeates;
  ExpVec i1_meger_repeates;
  MergeBrcAxisRepeats(input0_repeats, input1_repeats, input1_strides, i0_meger_repeates, i1_meger_repeates);
  EXPECT_EQ(i0_meger_repeates.size(), 3);
  EXPECT_EQ(i1_meger_repeates.size(), 3);
  EXPECT_EQ(i0_meger_repeates[0], S(2));
  EXPECT_EQ(i0_meger_repeates[1], S(1));
  EXPECT_EQ(i0_meger_repeates[2], S(96));
  EXPECT_EQ(i1_meger_repeates[0], S(2));
  EXPECT_EQ(i1_meger_repeates[1], S(22));
  EXPECT_EQ(i1_meger_repeates[2], S(96));

  //                  1     A     1     A     1     B     1     B     1     A     1     1     A      1     1 -> ABA
  input0_repeats = {S(1), S(2), S(1), S(2), S(1), S(1), S(1), S(1), S(1), S(3), S(1), S(1), S(11), S(1), S(1)};
  input1_repeats = {S(1), S(2), S(1), S(2), S(1), S(2), S(1), S(2), S(1), S(3), S(1), S(1), S(11), S(1), S(1)};
  input1_strides = {S(0), S(384), S(0), S(192), S(0), S(96), S(0), S(48), S(0), S(16), S(0), S(0), S(1), S(0), S(0)};
  MergeBrcAxisRepeats(input0_repeats, input1_repeats, input1_strides, i0_meger_repeates, i1_meger_repeates);
  EXPECT_EQ(i0_meger_repeates.size(), 3);
  EXPECT_EQ(i1_meger_repeates.size(), 3);
  EXPECT_EQ(i0_meger_repeates[0], S(4));
  EXPECT_EQ(i0_meger_repeates[1], S(1));
  EXPECT_EQ(i0_meger_repeates[2], S(48));
  EXPECT_EQ(i1_meger_repeates[0], S(4));
  EXPECT_EQ(i1_meger_repeates[1], S(4));
  EXPECT_EQ(i1_meger_repeates[2], S(48));

  //                  1     A     1     A     1     B     1     B     1     A     1     1     A      1     1 -> ABA
  input0_repeats = {S(1), S(2), S(1), S(2), S(1), S(1), S(1), S(1), S(1), S(3), S(1), S(1), S(11), S(1), S(1)};
  input0_strides = {S(0), S(96), S(0), S(48), S(0), S(0), S(0), S(0), S(0), S(16), S(0), S(0), S(1), S(0), S(0)};
  input1_repeats = {S(1), S(1), S(1), S(2), S(1), S(2), S(1), S(2), S(1), S(3), S(1), S(1), S(1), S(1), S(1)};
  input1_strides = {S(0), S(0), S(0), S(32), S(0), S(16), S(0), S(8), S(0), S(1), S(0), S(0), S(0), S(0), S(0)};

  MergeBrcAxisParams in0(input0_repeats, input0_strides);
  MergeBrcAxisParams in1(input1_repeats, input1_strides);
  MergeBrcAxisRepeats(in0, in1);
  i0_meger_repeates = in0.merge_repeats;
  i1_meger_repeates = in1.merge_repeats;
  EXPECT_EQ(i0_meger_repeates.size(), 5);
  EXPECT_EQ(i1_meger_repeates.size(), 5);
  EXPECT_EQ(i0_meger_repeates[0], S(2));
  EXPECT_EQ(i0_meger_repeates[1], S(2));
  EXPECT_EQ(i0_meger_repeates[2], S(1));
  EXPECT_EQ(i0_meger_repeates[3], S(3));
  EXPECT_EQ(i0_meger_repeates[4], S(11));

  EXPECT_EQ(i1_meger_repeates[0], S(1));
  EXPECT_EQ(i1_meger_repeates[1], S(2));
  EXPECT_EQ(i1_meger_repeates[2], S(4));
  EXPECT_EQ(i1_meger_repeates[3], S(3));
  EXPECT_EQ(i1_meger_repeates[4], S(1));
}

// 测试全1
TEST_F(CommonUtilsTest, MergeRepeatsWithToAll1) {
  using ExpVec = std::vector<ge::Expression>;
  using S = ge::Symbol;
  ExpVec input0_repeats = {S(1), S(1), S(1)};
  ExpVec input1_repeats = {S(1), S(1), S(1)};
  ExpVec input1_strides = {S(0), S(0), S(0)};
  ExpVec i0_meger_repeates;
  ExpVec i1_meger_repeates;
  MergeBrcAxisRepeats(input0_repeats, input1_repeats, input1_strides, i0_meger_repeates, i1_meger_repeates);
  EXPECT_EQ(i0_meger_repeates.size(), 1);
  EXPECT_EQ(i1_meger_repeates.size(), 1);
  EXPECT_EQ(i0_meger_repeates[0], S(1));
  EXPECT_EQ(i1_meger_repeates[0], S(1));
}

// 测试1个输入包含广播轴, 广播轴合并之后是首轴,且合并轴有2个轴, 支持brc inline
TEST_F(CommonUtilsTest, AfterMegerBrcAxisisFirstAxisSupportBrcInline) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data x_op("x0", graph);
  Data x_op1("x1", graph);
  Load load_op("load");
  Load load_op1("load1");
  af::ascir_op::Add add_op("add");

  graph.AddNode(load_op);
  graph.AddNode(load_op1);
  graph.AddNode(add_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {One, One, s2, s3};

  load_op1.x = x_op1.y;
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op1.y.repeats = {s0, s1, s2, s3};

  add_op.x1 = load_op.y;
  add_op.x2 = load_op1.y;
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *add_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *add_op.y.repeats = {s0, s1, s2, s3};

  // graph.SetInputs({x_op});

  auto load = graph.FindNode("load");
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {Zero, Zero, s3, One};

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
  load1->outputs[0].attr.vectorized_strides = {s1 * s2 * s3, s2 * s3, s3, One};

  auto add = graph.FindNode("add");
  add->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};

  std::vector<uint8_t> input_idx_2_brc_inline;
  const bool ret = IsGeneralizeBrcInlineScene(add, input_idx_2_brc_inline);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(input_idx_2_brc_inline.size(), 2);
  EXPECT_EQ(input_idx_2_brc_inline[0], 1);
  EXPECT_EQ(input_idx_2_brc_inline[1], 0);
}

// 测试合轴逻辑支持两个输入都包含广播轴
TEST_F(CommonUtilsTest, MergeRepeatsWithAllInputHasBrcAxis) {
  using ExpVec = std::vector<ge::Expression>;
  using S = ge::Symbol;
  ExpVec input0_repeats = {S(2), S(1), S(1), S(2), S(3), S(11)};
  ExpVec input1_repeats = {S(2), S(22), S(1), S(1), S(3), S(11)};
  ExpVec input1_strides = {S(192), S(96), S(0), S(0), S(16), S(1)};
  ExpVec i0_merge_repeats;
  ExpVec i1_merge_repeats;
  MergeBrcAxisRepeats(input0_repeats, input1_repeats, input1_strides, i0_merge_repeats, i1_merge_repeats);
  EXPECT_EQ(i0_merge_repeats.size(), 4);
  EXPECT_EQ(i1_merge_repeats.size(), 4);
  EXPECT_EQ(i0_merge_repeats[0], S(2));
  EXPECT_EQ(i0_merge_repeats[1], S(1));
  EXPECT_EQ(i0_merge_repeats[2], S(2));
  EXPECT_EQ(i0_merge_repeats[3], S(48));

  EXPECT_EQ(i1_merge_repeats[0], S(2));
  EXPECT_EQ(i1_merge_repeats[1], S(22));
  EXPECT_EQ(i1_merge_repeats[2], S(1));
  EXPECT_EQ(i1_merge_repeats[3], S(48));
}

TEST_F(CommonUtilsTest, FormatExpressionTrue) {
  EXPECT_EQ(FormatExpression("(s0 * s100 * s3)"),
            "static_cast<int64_t>(tiling_data.get_s0() * tiling_data.get_s100() * tiling_data.get_s3())");
  EXPECT_EQ(FormatExpression("(Rational(1, 2) * s0 * s3)"),
            "static_cast<int64_t>(Rational(1, 2) * tiling_data.get_s0() * tiling_data.get_s3())");
  EXPECT_EQ(FormatExpression("size2"), "tiling_data.get_size2()");
  EXPECT_EQ(FormatExpression("block_size"), "tiling_data.get_block_size()");
}

TEST_F(CommonUtilsTest, GenValidNameTest) {
  std::string name1 = "(128) * s2";
  std::string ret1 = GenValidName(name1);
  EXPECT_EQ(ret1, std::string{"t_128_s2"});

  std::string name2 = "00/*Bc";
  std::string ret2 = GenValidName(name2);
  EXPECT_EQ(ret2, std::string{"t_00_Bc"});

  std::string name3 = "ab*c_d";
  std::string ret3 = GenValidName(name3);
  EXPECT_EQ(ret3, std::string{"ab_c_d"});
}

TEST_F(CommonUtilsTest, CalcReservedTmpBufSizeForAscGraphTest) {
  ge::AscGraph graph("test_graph");
  Data x_op1("x1", graph);
  Data x_op2("x2", graph);
  Data x_op3("x3", graph);
  Load load_op1("load1");
  Load load_op2("load2");
  Load load_op3("load3");
  af::ascir_op::Where where_op("where");
  graph.AddNode(load_op1);
  graph.AddNode(load_op2);
  graph.AddNode(load_op3);
  graph.AddNode(where_op);

  load_op1.x = x_op1.y;
  load_op2.x = x_op2.y;
  load_op3.x = x_op3.y;
  where_op.x1 = load_op1.y;
  where_op.x2 = load_op2.y;
  where_op.x3 = load_op3.y;
  auto expr = CalcReservedTmpBufSizeForAscGraph(graph);
  EXPECT_EQ(expr, 8 * 1024);
}

TEST_F(CommonUtilsTest, CalcExtraTmpBufForAscGraphTest) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::LogicalNot logical_not_op("logical_not");

  graph.AddNode(load_op);
  graph.AddNode(logical_not_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, s3};

  logical_not_op.x = load_op.y;
  logical_not_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *logical_not_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *logical_not_op.y.repeats = {s0, s1, s2, s3};

  auto load = graph.FindNode("load");
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};

  auto logical_not = graph.FindNode("logical_not");
  logical_not->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
  auto buffer_size = CalcExtraTmpBufForAscGraph(graph);
  EXPECT_EQ(buffer_size, ge::Symbol(32));
  auto BlkSupportFlag = IsSupportBlkTensorInput(load);
  EXPECT_EQ(BlkSupportFlag, false);
  auto BlkTensorSupportFlag = IsScalarNextNodeSupportBlkTensor(load);
  EXPECT_EQ(BlkTensorSupportFlag, false);
}

TEST_F(CommonUtilsTest, GetAscIrCodegenImplTest) {
  auto codegen_impl = GetAscIrCodegenImpl("Abs");
  EXPECT_EQ(codegen_impl, nullptr);
}

TEST_F(CommonUtilsTest, GetAscIrAttImplTest) {
  auto att_impl = GetAscIrAttImpl("Abs");
  EXPECT_EQ(att_impl, nullptr);
}

TEST_F(CommonUtilsTest, GetAscIrCodegenImplNotNullTest) {
  class StubWorkspaceAscIrCodegenImpl : public af::ascir::AscIrCodegen {
   public:
    virtual std::string GetApiCallName() const {
      return "ApiCall";
    }
    virtual std::string GetApiName() const {
      return "Workspace";
    }
  };
  class StubWorkspaceAscIrAtt : public af::ascir::AscIrAtt {
    virtual void *GetApiPerf() const {
      return (void *)(0x123456);
    }
    virtual void *GetAscendCApiPerfTable() const {
      return nullptr;
    }
  };

  std::vector<std::string> soc_version_stub{"2201"};
  REG_ASC_IR(StubWorkspace)
      .Input("x", "T")
      .Output("y", "T")
      // .DataType("T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64,
      //                           DT_FLOAT16, DT_FLOAT})
      .Impl(soc_version_stub, {af::ascir::AscIrImplCreator<StubWorkspaceAscIrAtt>(),
                               af::ascir::AscIrImplCreator<StubWorkspaceAscIrCodegenImpl>(),
                               {{"T", ge::TensorType::ALL()}}});

  auto codegen_impl = ascgen_utils::GetAscIrCodegenImpl("StubWorkspace");
  EXPECT_NE(codegen_impl, nullptr);
  EXPECT_EQ(codegen_impl->GetApiCallName(), "ApiCall");
}

TEST_F(CommonUtilsTest, GetAscIrAttImplNotNullTest) {
  auto att_impl = ascgen_utils::GetAscIrAttImpl("StubWorkspace");
  EXPECT_NE(att_impl, nullptr);
  EXPECT_EQ((uint64_t)(uintptr_t)(att_impl->GetApiPerf()), 0x123456);
}

template <typename ConvOp>
void FillConv2DCommonIrAttr(ConvOp &conv) {
  conv.ir_attr.SetStrides({1, 1, 1, 1});
  conv.ir_attr.SetPads({0, 0, 0, 0});
  conv.ir_attr.SetDilations({1, 1, 1, 1});
  conv.ir_attr.SetGroups(1);
  conv.ir_attr.SetPad_mode("SPECIFIC");
  conv.ir_attr.SetData_format("NCHW");
  conv.ir_attr.SetOffset_x(0);
  conv.ir_attr.SetEnable_hf32(false);
  conv.ir_attr.SetFixed_shift_value(0);
  conv.attr.api.compute_type = af::ComputeType::kComputeCube;
}

template <typename ConvOp>
void FillExtendConv2DIrAttr(ConvOp &conv, bool enable_relu0 = false) {
  FillConv2DCommonIrAttr(conv);
  conv.ir_attr.SetRound_mode("rint");
  conv.ir_attr.SetEnable_relu0(enable_relu0);
}

TEST_F(CommonUtilsTest, ExtendConv2DGraphTypeAndAttrParse) {
  af::AscGraph graph("extend_conv2d_graph");
  af::ascir_op::ExtendConv2D conv("extend_conv2d");
  graph.AddNode(conv);
  FillExtendConv2DIrAttr(conv);

  EXPECT_TRUE(IsConv2DGraphType(graph));
  EXPECT_FALSE(IsConv2DTypeWithBias(graph));
  EXPECT_FALSE(IsConv2DTypeWithScale0(graph));
  EXPECT_FALSE(IsConv2DTypeWithOffsetW(graph));

  auto node = graph.FindNode("extend_conv2d");
  ASSERT_NE(node, nullptr);
  Conv2DAttr attr;
  ASSERT_EQ(ParseConv2DAttr(node, attr), af::SUCCESS);
  EXPECT_TRUE(attr.is_extend_conv2d);
  EXPECT_FALSE(attr.has_bias);
  EXPECT_FALSE(attr.has_scale0);
  EXPECT_EQ(attr.round_mode, "rint");
  EXPECT_EQ(attr.pad_mode, "SPECIFIC");
}

TEST_F(CommonUtilsTest, ExtendConv2DBiasGraphTypeAndAttrParse) {
  af::AscGraph graph("extend_conv2d_bias_graph");
  af::ascir_op::ExtendConv2DBias conv("extend_conv2d_bias");
  graph.AddNode(conv);
  FillExtendConv2DIrAttr(conv);

  EXPECT_TRUE(IsConv2DGraphType(graph));
  EXPECT_TRUE(IsConv2DTypeWithBias(graph));
  EXPECT_FALSE(IsConv2DTypeWithScale0(graph));

  auto node = graph.FindNode("extend_conv2d_bias");
  ASSERT_NE(node, nullptr);
  Conv2DAttr attr;
  ASSERT_EQ(ParseConv2DAttr(node, attr), af::SUCCESS);
  EXPECT_TRUE(attr.is_extend_conv2d);
  EXPECT_TRUE(attr.has_bias);
  EXPECT_FALSE(attr.has_scale0);
}

TEST_F(CommonUtilsTest, ExtendConv2DScaleGraphTypeAndAttrParse) {
  af::AscGraph graph("extend_conv2d_scale_graph");
  af::ascir_op::ExtendConv2DScale conv("extend_conv2d_scale");
  graph.AddNode(conv);
  FillExtendConv2DIrAttr(conv);

  EXPECT_TRUE(IsConv2DGraphType(graph));
  EXPECT_TRUE(IsConv2DTypeWithScale0(graph));
  EXPECT_FALSE(IsConv2DTypeWithBias(graph));

  auto node = graph.FindNode("extend_conv2d_scale");
  ASSERT_NE(node, nullptr);
  Conv2DAttr attr;
  ASSERT_EQ(ParseConv2DAttr(node, attr), af::SUCCESS);
  EXPECT_TRUE(attr.is_extend_conv2d);
  EXPECT_TRUE(attr.has_scale0);
  EXPECT_FALSE(attr.has_bias);
}

TEST_F(CommonUtilsTest, ExtendConv2DBiasScaleGraphTypeAndAttrParse) {
  af::AscGraph graph("extend_conv2d_bias_scale_graph");
  af::ascir_op::ExtendConv2DBiasScale conv("extend_conv2d_bias_scale");
  graph.AddNode(conv);
  FillExtendConv2DIrAttr(conv, true);

  EXPECT_TRUE(IsConv2DGraphType(graph));
  EXPECT_TRUE(IsConv2DTypeWithBias(graph));
  EXPECT_TRUE(IsConv2DTypeWithScale0(graph));

  auto node = graph.FindNode("extend_conv2d_bias_scale");
  ASSERT_NE(node, nullptr);
  Conv2DAttr attr;
  ASSERT_EQ(ParseConv2DAttr(node, attr), af::SUCCESS);
  EXPECT_TRUE(attr.is_extend_conv2d);
  EXPECT_TRUE(attr.has_bias);
  EXPECT_TRUE(attr.has_scale0);
  EXPECT_TRUE(attr.enable_relu0);
}

TEST_F(CommonUtilsTest, Conv2DLegacyOptionalInputAttrParse) {
  {
    af::AscGraph graph("conv2d_bias_graph");
    af::ascir_op::Conv2DBias conv("conv2d_bias");
    graph.AddNode(conv);
    FillConv2DCommonIrAttr(conv);
    EXPECT_TRUE(IsConv2DTypeWithBias(graph));
    auto node = graph.FindNode("conv2d_bias");
    ASSERT_NE(node, nullptr);
    Conv2DAttr attr;
    ASSERT_EQ(ParseConv2DAttr(node, attr), af::SUCCESS);
    EXPECT_TRUE(attr.has_bias);
    EXPECT_FALSE(attr.has_offset_w);
  }
  {
    af::AscGraph graph("conv2d_offset_graph");
    af::ascir_op::Conv2DOffset conv("conv2d_offset");
    graph.AddNode(conv);
    FillConv2DCommonIrAttr(conv);
    auto node = graph.FindNode("conv2d_offset");
    ASSERT_NE(node, nullptr);
    Conv2DAttr attr;
    ASSERT_EQ(ParseConv2DAttr(node, attr), af::SUCCESS);
    EXPECT_TRUE(attr.has_offset_w);
    EXPECT_FALSE(attr.has_bias);
  }
  {
    af::AscGraph graph("conv2d_offset_bias_graph");
    af::ascir_op::Conv2DOffsetBias conv("conv2d_offset_bias");
    graph.AddNode(conv);
    FillConv2DCommonIrAttr(conv);
    EXPECT_TRUE(IsConv2DTypeWithBias(graph));
    auto node = graph.FindNode("conv2d_offset_bias");
    ASSERT_NE(node, nullptr);
    Conv2DAttr attr;
    ASSERT_EQ(ParseConv2DAttr(node, attr), af::SUCCESS);
    EXPECT_TRUE(attr.has_bias);
    EXPECT_TRUE(attr.has_offset_w);
  }
}
}  // namespace ascgen_utils
