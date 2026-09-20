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
#include "micro_api_call_factory.h"
#include "micro_div_api_call.h"
#include "optimize/pre_process/pre_process_config.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

TEST(CodegenKernel, MicroDivApiCall_Load_Div_Float_Store) {
  af::AscGraph graph("test_div_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);

  auto s0 = af::Symbol(16);
  auto s1 = af::Symbol(8);
  auto s2 = af::Symbol(4);
  auto s3 = af::Symbol(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data x_op("x", graph);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Div div_op("div");
  af::ascir_op::Store store_op("store");

  graph.AddNode(load_op);
  graph.AddNode(div_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, s3};
  *load_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  div_op.x1 = load_op.y;
  div_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *div_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *div_op.y.repeats = {s0, s1, s2, s3};
  *div_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  store_op.x = div_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store_op.y.repeats = {s0, s1, s2, s3};
  *store_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto div_node = graph.FindNode("div");
  div_node->attr.api.compute_type = af::ComputeType::kComputeLoad;
  div_node->attr.api.type = af::ApiType::kAPITypeCompute;
  div_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  div_node->attr.sched.loop_axis = z0.id;
  div_node->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  div_node->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  div_node->outputs[0].attr.dtype = af::DT_FLOAT;
  div_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  div_node->outputs[0].attr.mem.tensor_id = 1;
  div_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  div_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  div_node->outputs[0].attr.que.id = 2;
  div_node->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeElewise;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  store->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 1;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddSizeVar(af::SizeVar(s3));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = store->outputs[0].attr.mem.tensor_id;
  codegen::CallParam cp = {"p_reg", ""};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(load->outputs[0], tensor_load);
  auto tensor_store = store->GetName() + "_" + store->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor2(store->outputs[0], tensor_store);
  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor1);
  tensor_mng.AddTensor(tensor2);
  codegen::MicroDivApiCall call("Div");
  EXPECT_EQ(call.Init(div_node), 0);
  call.AddInput(x1.id);
  call.AddOutput(y1.id);

  std::string result;
  call.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::Div<float, &high_precision_div_mode>(vreg_1, vreg_0, p_reg);\n"});
}

TEST(CodegenKernel, MicroDivApiCall_Load_Div_Float_EnhancePrecisionBlacklist) {
  setenv("AUTOFUSE_FLAGS", "--autofuse_enhance_precision_blacklist=Div", 1);
  af::pre_process::PreProcessConfig::Instance().Reset();

  af::AscGraph graph("test_div_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);

  auto s0 = af::Symbol(16);
  auto s1 = af::Symbol(8);
  auto s2 = af::Symbol(4);
  auto s3 = af::Symbol(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data x_op("x", graph);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Div div_op("div");
  af::ascir_op::Store store_op("store");

  graph.AddNode(load_op);
  graph.AddNode(div_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, s3};
  *load_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  div_op.x1 = load_op.y;
  div_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *div_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *div_op.y.repeats = {s0, s1, s2, s3};
  *div_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  store_op.x = div_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store_op.y.repeats = {s0, s1, s2, s3};
  *store_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto div_node = graph.FindNode("div");
  div_node->attr.api.compute_type = af::ComputeType::kComputeLoad;
  div_node->attr.api.type = af::ApiType::kAPITypeCompute;
  div_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  div_node->attr.sched.loop_axis = z0.id;
  div_node->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  div_node->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  div_node->outputs[0].attr.dtype = af::DT_FLOAT;
  div_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  div_node->outputs[0].attr.mem.tensor_id = 1;
  div_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  div_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  div_node->outputs[0].attr.que.id = 2;
  div_node->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeElewise;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  store->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 1;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddSizeVar(af::SizeVar(s3));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = store->outputs[0].attr.mem.tensor_id;
  codegen::CallParam cp = {"p_reg", ""};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(load->outputs[0], tensor_load);
  auto tensor_store = store->GetName() + "_" + store->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor2(store->outputs[0], tensor_store);
  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor1);
  tensor_mng.AddTensor(tensor2);
  codegen::MicroDivApiCall call("Div");
  EXPECT_EQ(call.Init(div_node), 0);
  call.AddInput(x1.id);
  call.AddOutput(y1.id);

  std::string result;
  call.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::Div<float>(vreg_1, vreg_0, p_reg);\n"});

  unsetenv("AUTOFUSE_FLAGS");
  af::pre_process::PreProcessConfig::Instance().Reset();
}

TEST(CodegenKernel, MicroDivApiCall_Load_Div_Half_Store) {
  af::AscGraph graph("test_div_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);

  auto s0 = af::Symbol(16);
  auto s1 = af::Symbol(8);
  auto s2 = af::Symbol(4);
  auto s3 = af::Symbol(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data x_op("x", graph);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Div div_op("div");
  af::ascir_op::Store store_op("store");

  graph.AddNode(load_op);
  graph.AddNode(div_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, s3};
  *load_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  div_op.x1 = load_op.y;
  div_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *div_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *div_op.y.repeats = {s0, s1, s2, s3};
  *div_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  store_op.x = div_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store_op.y.repeats = {s0, s1, s2, s3};
  *store_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  load->outputs[0].attr.dtype = af::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto div_node = graph.FindNode("div");
  div_node->attr.api.compute_type = af::ComputeType::kComputeLoad;
  div_node->attr.api.type = af::ApiType::kAPITypeCompute;
  div_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  div_node->attr.sched.loop_axis = z0.id;
  div_node->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  div_node->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  div_node->outputs[0].attr.dtype = af::DT_FLOAT16;
  div_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  div_node->outputs[0].attr.mem.tensor_id = 1;
  div_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  div_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  div_node->outputs[0].attr.que.id = 2;
  div_node->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeElewise;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  store->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  store->outputs[0].attr.dtype = af::DT_FLOAT16;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 1;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddSizeVar(af::SizeVar(s3));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = store->outputs[0].attr.mem.tensor_id;
  codegen::CallParam cp = {"p_reg", ""};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(load->outputs[0], tensor_load);
  auto tensor_store = store->GetName() + "_" + store->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor2(store->outputs[0], tensor_store);
  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor1);
  tensor_mng.AddTensor(tensor2);
  codegen::MicroDivApiCall call("Div");
  EXPECT_EQ(call.Init(div_node), 0);
  call.AddInput(x1.id);
  call.AddOutput(y1.id);

  std::string result;
  call.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::Div(vreg_1, vreg_0, p_reg);\n"});
}
