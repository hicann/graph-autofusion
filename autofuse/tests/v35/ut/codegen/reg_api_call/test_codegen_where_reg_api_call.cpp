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
#include "ascir_ops.h"
#include "utils/api_call_factory.h"
#include "codegen_kernel.h"
#include "reg_where_api_call.h"

using namespace af::ops;
using namespace codegen;
using namespace af::ascir_op;
using namespace testing;
using namespace codegen;

class WhereRegApiCallTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST(WhereRegApiCallTest, WhereRegApiCall_Scalar_x2x3) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

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
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1 * s2, s2, One};

  load_op3.x = x_op3.y;
  load_op3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.repeats = {s0, s1, s2};
  *load_op3.y.strides = {s1 * s2, s2, One};

  where_op.x1 = load_op1.y;
  where_op.x2 = load_op2.y;
  where_op.x3 = load_op3.y;
  *where_op.y.axis = {z0.id, z1.id, z2.id};
  *where_op.y.repeats = {s0, s1, s2};
  *where_op.y.strides = {s1 * s2, s2, One};

  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {s2, One};
  load1->outputs[0].attr.dtype = af::DT_FLOAT;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {s2 + s2, One};
  load2->outputs[0].attr.dtype = af::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load3 = graph.FindNode("load3");
  load3->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load3->attr.api.type = af::ApiType::kAPITypeCompute;
  load3->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load3->attr.sched.loop_axis = z0.id;
  load3->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load3->outputs[0].attr.vectorized_strides = {s2 + s2, One};
  load3->outputs[0].attr.dtype = af::DT_FLOAT;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.tensor_id = 2;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load3->outputs[0].attr.que.id = 1;
  load3->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto where = graph.FindNode("where");
  where->attr.api.compute_type = af::ComputeType::kComputeElewise;
  where->attr.api.type = af::ApiType::kAPITypeCompute;
  where->attr.api.unit = af::ComputeUnit::kUnitVector;
  where->attr.sched.loop_axis = z0.id;
  where->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  where->outputs[0].attr.vectorized_strides = {s2, One};
  where->outputs[0].attr.dtype = af::DT_INT16;
  where->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  where->outputs[0].attr.mem.tensor_id = 3;
  where->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  where->outputs[0].attr.que.id = 2;
  where->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor("1", load2->outputs[0]);
  tpipe.AddTensor("1", load3->outputs[0]);
  tpipe.AddTensor(where->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x3;
  x3.id = load3->outputs[0].attr.mem.tensor_id;
  codegen::WhereRegApiCall call("Where");
  EXPECT_EQ(call.Init(where), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(result, std::string{"Where(local_3[0], local_0[0], local_1, local_2, local_0_actual_size);\n"});
}

TEST(WhereRegApiCallTest, WhereRegApiCall_x2_x3_is_ub_scalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

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
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {One, One, One};
  *load_op2.y.strides = {One, One, One};

  load_op3.x = x_op3.y;
  load_op3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.repeats = {One, One, One};
  *load_op3.y.strides = {One, One, One};

  where_op.x1 = load_op1.y;
  where_op.x2 = load_op2.y;
  where_op.x3 = load_op3.y;
  *where_op.y.axis = {z0.id, z1.id, z2.id};
  *where_op.y.repeats = {s0, s1, s2};
  *where_op.y.strides = {s1 * s2, s2, One};

  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {s2, One};
  load1->outputs[0].attr.dtype = af::DT_FLOAT;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {One, One};
  load2->outputs[0].attr.dtype = af::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load3 = graph.FindNode("load3");
  load3->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load3->attr.api.type = af::ApiType::kAPITypeCompute;
  load3->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load3->attr.sched.loop_axis = z0.id;
  load3->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load3->outputs[0].attr.vectorized_strides = {One, One};
  load3->outputs[0].attr.dtype = af::DT_FLOAT;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.tensor_id = 2;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load3->outputs[0].attr.que.id = 1;
  load3->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto where = graph.FindNode("where");
  where->attr.api.compute_type = af::ComputeType::kComputeElewise;
  where->attr.api.type = af::ApiType::kAPITypeCompute;
  where->attr.api.unit = af::ComputeUnit::kUnitVector;
  where->attr.sched.loop_axis = z0.id;
  where->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  where->outputs[0].attr.vectorized_strides = {s2, One};
  where->outputs[0].attr.dtype = af::DT_INT16;
  where->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  where->outputs[0].attr.mem.tensor_id = 3;
  where->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  where->outputs[0].attr.que.id = 2;
  where->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load1->outputs[0]);

  // begin:构造x2 x3 是ub_scalar的tensor
  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t_x2(load2->outputs[0], dtype_name, "t_x2");
  EXPECT_EQ(t_x2.Init(), 0);
  t_x2.need_gen_get_value_of_ub_scalar = true;
  t_x2.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t_x2), 0);

  codegen::Tensor t_x3(load3->outputs[0], dtype_name, "t_x3");
  EXPECT_EQ(t_x3.Init(), 0);
  t_x3.need_gen_get_value_of_ub_scalar = true;
  t_x3.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t_x3), 0);
  // end
  tpipe.AddTensor(where->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x3;
  x3.id = load3->outputs[0].attr.mem.tensor_id;
  codegen::WhereRegApiCall call("Where");
  EXPECT_EQ(call.Init(where), 0);

  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(
      result,
      std::string{
          "Where(local_3[0], local_0[0], (float)local_1_ub_scalar, (float)local_2_ub_scalar, local_0_actual_size);\n"});
}

TEST(WhereRegApiCallTest, WhereRegApiCall_x2_is_ub_scalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

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
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {One, One, One};
  *load_op2.y.strides = {One, One, One};

  load_op3.x = x_op3.y;
  load_op3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.repeats = {s0, s1, s2};
  *load_op3.y.strides = {s1 * s2, s2, One};

  where_op.x1 = load_op1.y;
  where_op.x2 = load_op2.y;
  where_op.x3 = load_op3.y;
  *where_op.y.axis = {z0.id, z1.id, z2.id};
  *where_op.y.repeats = {s0, s1, s2};
  *where_op.y.strides = {s1 * s2, s2, One};

  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {s2, One};
  load1->outputs[0].attr.dtype = af::DT_FLOAT;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {One, One};
  load2->outputs[0].attr.dtype = af::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load3 = graph.FindNode("load3");
  load3->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load3->attr.api.type = af::ApiType::kAPITypeCompute;
  load3->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load3->attr.sched.loop_axis = z0.id;
  load3->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load3->outputs[0].attr.vectorized_strides = {s2, One};
  load3->outputs[0].attr.dtype = af::DT_FLOAT;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.tensor_id = 2;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load3->outputs[0].attr.que.id = 1;
  load3->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto where = graph.FindNode("where");
  where->attr.api.compute_type = af::ComputeType::kComputeElewise;
  where->attr.api.type = af::ApiType::kAPITypeCompute;
  where->attr.api.unit = af::ComputeUnit::kUnitVector;
  where->attr.sched.loop_axis = z0.id;
  where->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  where->outputs[0].attr.vectorized_strides = {s2, One};
  where->outputs[0].attr.dtype = af::DT_INT16;
  where->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  where->outputs[0].attr.mem.tensor_id = 3;
  where->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  where->outputs[0].attr.que.id = 2;
  where->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load1->outputs[0]);

  // begin:构造x2 是ub_scalar的tensor
  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t_x2(load2->outputs[0], dtype_name, "t_x2");
  EXPECT_EQ(t_x2.Init(), 0);
  t_x2.need_gen_get_value_of_ub_scalar = true;
  t_x2.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t_x2), 0);

  tpipe.AddTensor(load3->outputs[0]);

  // end
  tpipe.AddTensor(where->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x3;
  x3.id = load3->outputs[0].attr.mem.tensor_id;
  codegen::WhereRegApiCall call("Where");
  EXPECT_EQ(call.Init(where), 0);

  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(result,
            std::string{"Where(local_3[0], local_0[0], (float)local_1_ub_scalar, local_2[0], local_0_actual_size);\n"});
}

TEST(WhereRegApiCallTest, WhereRegApiCall_x3_is_ub_scalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

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
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1 * s2, s2, One};

  load_op3.x = x_op3.y;
  load_op3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.repeats = {One, One, One};
  *load_op3.y.strides = {One, One, One};

  where_op.x1 = load_op1.y;
  where_op.x2 = load_op2.y;
  where_op.x3 = load_op3.y;
  *where_op.y.axis = {z0.id, z1.id, z2.id};
  *where_op.y.repeats = {s0, s1, s2};
  *where_op.y.strides = {s1 * s2, s2, One};

  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {s2, One};
  load1->outputs[0].attr.dtype = af::DT_FLOAT;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {s2, One};
  load2->outputs[0].attr.dtype = af::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load3 = graph.FindNode("load3");
  load3->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load3->attr.api.type = af::ApiType::kAPITypeCompute;
  load3->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load3->attr.sched.loop_axis = z0.id;
  load3->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load3->outputs[0].attr.vectorized_strides = {One, One};
  load3->outputs[0].attr.dtype = af::DT_FLOAT;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.tensor_id = 2;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load3->outputs[0].attr.que.id = 1;
  load3->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto where = graph.FindNode("where");
  where->attr.api.compute_type = af::ComputeType::kComputeElewise;
  where->attr.api.type = af::ApiType::kAPITypeCompute;
  where->attr.api.unit = af::ComputeUnit::kUnitVector;
  where->attr.sched.loop_axis = z0.id;
  where->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  where->outputs[0].attr.vectorized_strides = {s2, One};
  where->outputs[0].attr.dtype = af::DT_INT16;
  where->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  where->outputs[0].attr.mem.tensor_id = 3;
  where->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  where->outputs[0].attr.que.id = 2;
  where->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);

  // begin:构造x3 是ub_scalar的tensor
  std::string dtype_name;
  codegen::Tensor::DtypeName(load3->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t_x3(load3->outputs[0], dtype_name, "t_x3");
  EXPECT_EQ(t_x3.Init(), 0);
  t_x3.need_gen_get_value_of_ub_scalar = true;
  t_x3.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t_x3), 0);

  // end
  tpipe.AddTensor(where->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x3;
  x3.id = load3->outputs[0].attr.mem.tensor_id;
  codegen::WhereRegApiCall call("Where");
  EXPECT_EQ(call.Init(where), 0);

  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(result,
            std::string{"Where(local_3[0], local_0[0], local_1[0], (float)local_2_ub_scalar, local_0_actual_size);\n"});
}

TEST(WhereRegApiCallTest, WhereRegApiCall) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

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
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1 * s2, s2, One};

  load_op3.x = x_op3.y;
  load_op3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.repeats = {s0, s1, s2};
  *load_op3.y.strides = {s1 * s2, s2, One};

  where_op.x1 = load_op1.y;
  where_op.x2 = load_op2.y;
  where_op.x3 = load_op3.y;
  *where_op.y.axis = {z0.id, z1.id, z2.id};
  *where_op.y.repeats = {s0, s1, s2};
  *where_op.y.strides = {s1 * s2, s2, One};

  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {s2, One};
  load1->outputs[0].attr.dtype = af::DT_FLOAT;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {s2, One};
  load2->outputs[0].attr.dtype = af::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load3 = graph.FindNode("load3");
  load3->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load3->attr.api.type = af::ApiType::kAPITypeCompute;
  load3->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load3->attr.sched.loop_axis = z0.id;
  load3->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load3->outputs[0].attr.vectorized_strides = {s2, One};
  load3->outputs[0].attr.dtype = af::DT_FLOAT;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.tensor_id = 2;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load3->outputs[0].attr.que.id = 1;
  load3->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto where = graph.FindNode("where");
  where->attr.api.compute_type = af::ComputeType::kComputeElewise;
  where->attr.api.type = af::ApiType::kAPITypeCompute;
  where->attr.api.unit = af::ComputeUnit::kUnitVector;
  where->attr.sched.loop_axis = z0.id;
  where->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  where->outputs[0].attr.vectorized_strides = {s2, One};
  where->outputs[0].attr.dtype = af::DT_INT16;
  where->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  where->outputs[0].attr.mem.tensor_id = 3;
  where->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  where->outputs[0].attr.que.id = 2;
  where->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(load3->outputs[0]);
  tpipe.AddTensor(where->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x3;
  x3.id = load3->outputs[0].attr.mem.tensor_id;

  codegen::WhereRegApiCall call("Where");
  EXPECT_EQ(call.Init(where), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(result, std::string{"Where(local_3[0], local_0[0], local_1[0], local_2[0], local_0_actual_size);\n"});
}

TEST(WhereRegApiCallTest, WhereRegApiCall_Scalar_x2x3_throwfor) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

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
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1 * s2, s2, One};

  load_op3.x = x_op3.y;
  load_op3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.repeats = {s0, s1, s2};
  *load_op3.y.strides = {s1 * s2, s2, One};

  where_op.x1 = load_op1.y;
  where_op.x2 = load_op2.y;
  where_op.x3 = load_op3.y;
  *where_op.y.axis = {z0.id, z1.id, z2.id};
  *where_op.y.repeats = {s0, s1, s2};
  *where_op.y.strides = {s1 * s2, s2, One};

  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {s2 + s2, One};
  load1->outputs[0].attr.dtype = af::DT_FLOAT;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {s2 + s2, One};
  load2->outputs[0].attr.dtype = af::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load3 = graph.FindNode("load3");
  load3->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load3->attr.api.type = af::ApiType::kAPITypeCompute;
  load3->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load3->attr.sched.loop_axis = z0.id;
  load3->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load3->outputs[0].attr.vectorized_strides = {s2 + s2, One};
  load3->outputs[0].attr.dtype = af::DT_FLOAT;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.tensor_id = 2;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load3->outputs[0].attr.que.id = 1;
  load3->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto where = graph.FindNode("where");
  where->attr.api.compute_type = af::ComputeType::kComputeElewise;
  where->attr.api.type = af::ApiType::kAPITypeCompute;
  where->attr.api.unit = af::ComputeUnit::kUnitVector;
  where->attr.sched.loop_axis = z0.id;
  where->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  where->outputs[0].attr.vectorized_strides = {s2, One};
  where->outputs[0].attr.dtype = af::DT_INT16;
  where->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  where->outputs[0].attr.mem.tensor_id = 3;
  where->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  where->outputs[0].attr.que.id = 2;
  where->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor("1", load2->outputs[0]);
  tpipe.AddTensor("1", load3->outputs[0]);
  tpipe.AddTensor(where->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x3;
  x3.id = load3->outputs[0].attr.mem.tensor_id;
  codegen::WhereRegApiCall call("Where");
  EXPECT_EQ(call.Init(where), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(
      result,
      std::string{"Where<true, true>(local_3[0], local_0[0], local_blk_tensor_of_local_1[0], "
                  "local_blk_tensor_of_local_2[0], {static_cast<uint16_t>(t->s1), static_cast<uint16_t>(t->s2)}, "
                  "{static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)}, {static_cast<uint16_t>((2 * t->s2)), "
                  "static_cast<uint16_t>(1)}, {static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)});\n"});
}

TEST(WhereRegApiCallTest, WhereRegApiCall_Scalar_x2_throwfor) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

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
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1 * s2, s2, One};

  load_op3.x = x_op3.y;
  load_op3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.repeats = {s0, s1, s2};
  *load_op3.y.strides = {s1 * s2, s2, One};

  where_op.x1 = load_op1.y;
  where_op.x2 = load_op2.y;
  where_op.x3 = load_op3.y;
  *where_op.y.axis = {z0.id, z1.id, z2.id};
  *where_op.y.repeats = {s0, s1, s2};
  *where_op.y.strides = {s1 * s2, s2, One};

  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {s2, One};
  load1->outputs[0].attr.dtype = af::DT_FLOAT;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {s2, One};
  load2->outputs[0].attr.dtype = af::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load3 = graph.FindNode("load3");
  load3->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load3->attr.api.type = af::ApiType::kAPITypeCompute;
  load3->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load3->attr.sched.loop_axis = z0.id;
  load3->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load3->outputs[0].attr.vectorized_strides = {s2 + s2, One};
  load3->outputs[0].attr.dtype = af::DT_FLOAT;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.tensor_id = 2;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load3->outputs[0].attr.que.id = 1;
  load3->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto where = graph.FindNode("where");
  where->attr.api.compute_type = af::ComputeType::kComputeElewise;
  where->attr.api.type = af::ApiType::kAPITypeCompute;
  where->attr.api.unit = af::ComputeUnit::kUnitVector;
  where->attr.sched.loop_axis = z0.id;
  where->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  where->outputs[0].attr.vectorized_strides = {s2, One};
  where->outputs[0].attr.dtype = af::DT_INT16;
  where->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  where->outputs[0].attr.mem.tensor_id = 3;
  where->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  where->outputs[0].attr.que.id = 2;
  where->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor("1", load2->outputs[0]);
  tpipe.AddTensor(load3->outputs[0]);
  tpipe.AddTensor(where->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x3;
  x3.id = load3->outputs[0].attr.mem.tensor_id;
  codegen::WhereRegApiCall call("Where");
  EXPECT_EQ(call.Init(where), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(result,
            std::string{"Where<true, false>(local_3[0], local_0[0], local_blk_tensor_of_local_1[0], local_2[0], "
                        "{static_cast<uint16_t>(t->s1), static_cast<uint16_t>(t->s2)}, {static_cast<uint16_t>(t->s2), "
                        "static_cast<uint16_t>(1)}, {static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)}, "
                        "{static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)});\n"});
}

TEST(WhereRegApiCallTest, WhereRegApiCall_Scalar_x3_throwfor) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

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
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1 * s2, s2, One};

  load_op3.x = x_op3.y;
  load_op3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.repeats = {s0, s1, s2};
  *load_op3.y.strides = {s1 * s2, s2, One};

  where_op.x1 = load_op1.y;
  where_op.x2 = load_op2.y;
  where_op.x3 = load_op3.y;
  *where_op.y.axis = {z0.id, z1.id, z2.id};
  *where_op.y.repeats = {s0, s1, s2};
  *where_op.y.strides = {s1 * s2, s2, One};

  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {s2, One};
  load1->outputs[0].attr.dtype = af::DT_FLOAT;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {s2 + s2, One};  // 构造不连续，抛for循环的场景
  load2->outputs[0].attr.dtype = af::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load3 = graph.FindNode("load3");
  load3->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load3->attr.api.type = af::ApiType::kAPITypeCompute;
  load3->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load3->attr.sched.loop_axis = z0.id;
  load3->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load3->outputs[0].attr.vectorized_strides = {s2, One};
  load3->outputs[0].attr.dtype = af::DT_FLOAT;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.tensor_id = 2;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load3->outputs[0].attr.que.id = 1;
  load3->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto where = graph.FindNode("where");
  where->attr.api.compute_type = af::ComputeType::kComputeElewise;
  where->attr.api.type = af::ApiType::kAPITypeCompute;
  where->attr.api.unit = af::ComputeUnit::kUnitVector;
  where->attr.sched.loop_axis = z0.id;
  where->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  where->outputs[0].attr.vectorized_strides = {s2, One};
  where->outputs[0].attr.dtype = af::DT_INT16;
  where->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  where->outputs[0].attr.mem.tensor_id = 3;
  where->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  where->outputs[0].attr.que.id = 2;
  where->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor("1", load3->outputs[0]);
  tpipe.AddTensor(where->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x3;
  x3.id = load3->outputs[0].attr.mem.tensor_id;
  codegen::WhereRegApiCall call("Where");
  EXPECT_EQ(call.Init(where), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(result,
            std::string{"Where<false, true>(local_3[0], local_0[0], local_1[0], local_blk_tensor_of_local_2[0], "
                        "{static_cast<uint16_t>(t->s1), static_cast<uint16_t>(t->s2)}, {static_cast<uint16_t>(t->s2), "
                        "static_cast<uint16_t>(1)}, {static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)}, "
                        "{static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)});\n"});
}

TEST(WhereRegApiCallTest, WhereRegApiCall_throwfor) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

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
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1 * s2, s2, One};

  load_op3.x = x_op3.y;
  load_op3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.axis = {z0.id, z1.id, z2.id};
  *load_op3.y.repeats = {s0, s1, s2};
  *load_op3.y.strides = {s1 * s2, s2, One};

  where_op.x1 = load_op1.y;
  where_op.x2 = load_op2.y;
  where_op.x3 = load_op3.y;
  *where_op.y.axis = {z0.id, z1.id, z2.id};
  *where_op.y.repeats = {s0, s1, s2};
  *where_op.y.strides = {s1 * s2, s2, One};

  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {s2, One};
  load1->outputs[0].attr.dtype = af::DT_FLOAT;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {s2 + s2, One};
  load2->outputs[0].attr.dtype = af::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load3 = graph.FindNode("load3");
  load3->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load3->attr.api.type = af::ApiType::kAPITypeCompute;
  load3->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load3->attr.sched.loop_axis = z0.id;
  load3->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load3->outputs[0].attr.vectorized_strides = {s2, One};
  load3->outputs[0].attr.dtype = af::DT_FLOAT;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.tensor_id = 2;
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load3->outputs[0].attr.que.id = 1;
  load3->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto where = graph.FindNode("where");
  where->attr.api.compute_type = af::ComputeType::kComputeElewise;
  where->attr.api.type = af::ApiType::kAPITypeCompute;
  where->attr.api.unit = af::ComputeUnit::kUnitVector;
  where->attr.sched.loop_axis = z0.id;
  where->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  where->outputs[0].attr.vectorized_strides = {s2, One};
  where->outputs[0].attr.dtype = af::DT_INT16;
  where->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  where->outputs[0].attr.mem.tensor_id = 3;
  where->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  where->outputs[0].attr.que.id = 2;
  where->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(load3->outputs[0]);
  tpipe.AddTensor(where->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x3;
  x3.id = load3->outputs[0].attr.mem.tensor_id;

  codegen::WhereRegApiCall call("Where");
  EXPECT_EQ(call.Init(where), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(result,
            std::string{"Where<false, false>(local_3[0], local_0[0], local_1[0], local_2[0], "
                        "{static_cast<uint16_t>(t->s1), static_cast<uint16_t>(t->s2)}, {static_cast<uint16_t>(t->s2), "
                        "static_cast<uint16_t>(1)}, {static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)}, "
                        "{static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)});\n"});
}
