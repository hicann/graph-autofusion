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
#include "node_utils_ex.h"
#include "graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "utils/api_call_factory.h"
#include "broadcast/broadcast_api_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

TEST(CodegenKernel, ConstantBroadCastStore) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  Scalar constant_op("constant", graph);
  constant_op.ir_attr.SetValue("1.0");
  Broadcast broadcast("broadcast");
  broadcast.x = constant_op.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  *broadcast.y.repeats = {s0, s1};
  *broadcast.y.strides = {s1, One};

  Store store("store");
  store.x = broadcast.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  auto broadcast_node = graph.FindNode("broadcast");
  broadcast_node->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast_node->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast_node->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast_node->attr.sched.loop_axis = z0.id;
  broadcast_node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  broadcast_node->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  broadcast_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  broadcast_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast_node->outputs[0].attr.mem.tensor_id = 1;
  broadcast_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast_node->outputs[0].attr.que.id = 1;
  broadcast_node->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store_node = graph.FindNode("store");
  store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
  store_node->attr.api.type = af::ApiType::kAPITypeCompute;
  store_node->attr.api.unit = af::ComputeUnit::kUnitMTE3;
  store_node->attr.sched.loop_axis = z0.id;
  store_node->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  store_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  store_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  store_node->outputs[0].attr.mem.tensor_id = 2;
  store_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store_node->outputs[0].attr.que.id = 1;
  store_node->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 0;
  constant_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(broadcast_node->outputs[0]);
  tpipe.AddTensor("1.0", constant_node->outputs[0], "const_y");

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = constant_node->outputs[0].attr.mem.tensor_id;
  x2.id = broadcast_node->outputs[0].attr.mem.tensor_id;

  codegen::BroadcastApiCall call("Broadcast");
  EXPECT_EQ(call.Init(broadcast_node), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{"Duplicate(local_1[0], scalar_0, local_1_actual_size);\n"});
}

TEST(CodegenKernel, BroadcastApiCallAllCommonAxis) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  broadcast_op.x = load_op.y;
  *broadcast_op.y.axis = {z0.id, z1.id};
  *broadcast_op.y.repeats = {s0, s1};
  *broadcast_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {s1, One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0.id;
  broadcast->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  broadcast->outputs[0].attr.vectorized_strides = {s1, One};
  broadcast->outputs[0].attr.dtype = ge::DT_FLOAT;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastApiCall call("Broadcast");
  EXPECT_EQ(call.Init(broadcast), 0);

  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(
      result,
      std::string{
          "DataCopy(local_1[0], local_0[0], KernelUtils::SizeAlign(local_1_actual_size, 32 / sizeof(float)));\n"});
}

TEST(CodegenKernel, BroadcastApiCallUBScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {One, One};
  *load_op.y.strides = {Zero, Zero};
  broadcast_op.x = load_op.y;
  *broadcast_op.y.axis = {z0.id, z1.id};
  *broadcast_op.y.repeats = {s0, s1};
  *broadcast_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {Zero, Zero};
  load->outputs[0].attr.dtype = ge::DT_INT64;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0.id;
  broadcast->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  broadcast->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  broadcast->outputs[0].attr.vectorized_strides = {s1, One};
  broadcast->outputs[0].attr.dtype = ge::DT_INT64;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastApiCall call("Broadcast");
  EXPECT_EQ(call.Init(broadcast), 0);

  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{"Duplicate(local_1[0], local_0_ub_scalar, local_1_actual_size, tmp_buf_0);\n"});
}
TEST(CodegenKernel, BroadcastApiCallScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, One};
  *load_op.y.strides = {One, Zero};
  broadcast_op.x = load_op.y;
  *broadcast_op.y.axis = {z0.id, z1.id};
  *broadcast_op.y.repeats = {s0, s1};
  *broadcast_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {Zero};
  load->outputs[0].attr.dtype = ge::DT_INT64;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0.id;
  broadcast->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  broadcast->outputs[0].attr.vectorized_axis = {z1.id};
  broadcast->outputs[0].attr.vectorized_strides = {One};
  broadcast->outputs[0].attr.dtype = ge::DT_INT64;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastApiCall call("Broadcast");
  EXPECT_EQ(call.Init(broadcast), 0);

  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{"event_t local_0_to_local_1_event_id = "
                                "static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));\n"
                                "SetFlag<HardEvent::MTE2_S>(local_0_to_local_1_event_id);\n"
                                "WaitFlag<HardEvent::MTE2_S>(local_0_to_local_1_event_id);\n"
                                "Duplicate(local_1[0], local_0.GetValue(0), local_1_actual_size, tmp_buf_0);\n"});
}

TEST(CodegenKernel, BroadcastApiCallVectorizedSizeInvalid) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  broadcast_op.x = load_op.y;
  *broadcast_op.y.axis = {z0.id, z1.id};
  *broadcast_op.y.repeats = {s0, s1};
  *broadcast_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {s1, s1, Zero};
  load->outputs[0].attr.dtype = ge::DT_INT64;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0.id;
  broadcast->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  broadcast->outputs[0].attr.vectorized_strides = {s1, One};
  broadcast->outputs[0].attr.dtype = ge::DT_INT64;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastApiCall call("Broadcast");
  EXPECT_EQ(call.Init(broadcast), 0);

  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, "");
}

TEST(CodegenKernel, BroadcastApiCall_BAB) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {One, s1, One};
  *load_op.y.strides = {Zero, One, Zero};
  broadcast_op.x = load_op.y;
  *broadcast_op.y.axis = {z0.id, z1.id, z2.id};
  *broadcast_op.y.repeats = {s0, s1, s2};
  *broadcast_op.y.strides = {s1 * s2, s2, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {Zero, One, Zero};
  load->outputs[0].attr.dtype = ge::DT_INT64;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0.id;
  broadcast->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  broadcast->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  broadcast->outputs[0].attr.vectorized_strides = {s1 * s2, s1, One};
  broadcast->outputs[0].attr.dtype = ge::DT_INT64;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastApiCall call("Broadcast");
  EXPECT_EQ(call.Init(broadcast), 0);

  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, "Broadcast(local_1[0], local_0[0], 1, 1, t->s1, 1, 1, t->s0, t->s1, t->s1, tmp_buf_0, 1);\n");
}

TEST(CodegenKernel, BroadcastApiCall_ABAB) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load_op.y.repeats = {s0, s1, One, s3, One};
  *load_op.y.strides = {(s1 * s3), s3, Zero, One, Zero};
  broadcast_op.x = load_op.y;
  *broadcast_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *broadcast_op.y.repeats = {s0, s1, s2, s3, s4};
  *broadcast_op.y.strides = {(s1 * s2 * s3 * s4), (s2 * s3 * s4), (s3 * s4), s4, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id, z4.id};
  load->outputs[0].attr.vectorized_strides = {s3, Zero, s1, Zero};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0.id;
  broadcast->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  broadcast->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id, z4.id};
  broadcast->outputs[0].attr.vectorized_strides = {s2 * s3 * s4, s3 * s4, s4, One};
  broadcast->outputs[0].attr.dtype = ge::DT_FLOAT;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddAxis(z4);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddSizeVar(af::SizeVar(s3));
  tiler.AddSizeVar(af::SizeVar(s4));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastApiCall call("Broadcast");
  EXPECT_EQ(call.Init(broadcast), 0);

  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result,
            "Broadcast(local_1[0], local_0[0], t->s1, 1, t->s3, 1, t->s1, t->s2, t->s3, t->s4, tmp_buf_0, t->s1);\n");
}

TEST(CodegenKernel, BroadcastApiCallNotSupportCase) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load_op.y.repeats = {One, s1, One, s3, One};
  *load_op.y.strides = {Zero, (s1 * s3), Zero, One, Zero};
  broadcast_op.x = load_op.y;
  *broadcast_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *broadcast_op.y.repeats = {s0, s1, s2, s3, s4};
  *broadcast_op.y.strides = {(s1 * s2 * s3 * s4), (s2 * s3 * s4), (s3 * s4), s4, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load->outputs[0].attr.vectorized_strides = {Zero, s1, Zero, s3, Zero};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0.id;
  broadcast->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  broadcast->outputs[0].attr.vectorized_strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  broadcast->outputs[0].attr.dtype = ge::DT_FLOAT;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddAxis(z4);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddSizeVar(af::SizeVar(s3));
  tiler.AddSizeVar(af::SizeVar(s4));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastApiCall call("Broadcast");
  EXPECT_EQ(call.Init(broadcast), 0);

  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, "");
}
TEST(CodegenKernel, BroadcastApiCallScalarAfterVectorNode) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Abs abs_op("abs");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, One};
  *load_op.y.strides = {One, Zero};

  abs_op.x = load_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id};
  *abs_op.y.axis = {z0.id, z1.id};
  *abs_op.y.repeats = {s0, One};
  *abs_op.y.strides = {One, Zero};

  broadcast_op.x = abs_op.y;
  broadcast_op.attr.sched.axis = {z0.id, z1.id};
  *broadcast_op.y.axis = {z0.id, z1.id};
  *broadcast_op.y.repeats = {s0, s1};
  *broadcast_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {Zero};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = af::ComputeType::kComputeLoad;
  abs->attr.api.type = af::ApiType::kAPITypeCompute;
  abs->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  abs->attr.sched.loop_axis = z0.id;
  abs->outputs[0].attr.vectorized_axis = {z1.id};
  abs->outputs[0].attr.vectorized_strides = {Zero};
  abs->outputs[0].attr.dtype = ge::DT_FLOAT;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
  abs->outputs[0].attr.mem.tensor_id = 1;
  abs->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.que.id = 2;
  abs->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0.id;
  broadcast->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  broadcast->outputs[0].attr.vectorized_axis = {z1.id};
  broadcast->outputs[0].attr.vectorized_strides = {One};
  broadcast->outputs[0].attr.dtype = ge::DT_FLOAT;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 2;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 3;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(abs->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = abs->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastApiCall call("Broadcast");
  EXPECT_EQ(call.Init(broadcast), 0);

  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(
      result,
      std::string{
          "event_t local_1_to_local_2_event_id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));\n"
          "SetFlag<HardEvent::V_S>(local_1_to_local_2_event_id);\n"
          "WaitFlag<HardEvent::V_S>(local_1_to_local_2_event_id);\n"
          "Duplicate(local_2[0], local_1.GetValue(0), local_2_actual_size);\n"});
}
