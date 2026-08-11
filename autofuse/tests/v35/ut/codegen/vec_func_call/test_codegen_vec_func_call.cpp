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
#include "ascir_utils.h"
#include "ascir_node_param/ascir_node_param.h"
#include "ascir_node_param/ascir_param_builder.h"
#include "graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "utils/api_call_factory.h"
#include "vec_func_call.h"
#include "../common.h"
#include "codegen_graph_check.h"

using namespace std;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

namespace {
template <typename TensorLike>
void SetTwoDimSchedule(TensorLike &tensor, const af::Axis &z0, const af::Axis &z1, const af::Expression &s0,
                       const af::Expression &s1) {
  *tensor.axis = {z0.id, z1.id};
  *tensor.repeats = {s0, s1};
  *tensor.strides = {s1, One};
}

void SetTwoDimVecInAttr(AscTensor &tensor, const af::Axis &z0, const af::Axis &z1, const af::Expression &stride,
                        int64_t tensor_id) {
  tensor.attr.vectorized_axis = {z0.id, z1.id};
  tensor.attr.vectorized_strides = {stride, One};
  tensor.attr.dtype = af::DT_FLOAT;
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.mem.tensor_id = tensor_id;
}

void InitScalarDataVfGraph(VectorFunc &vf_op, Store &store_op, Broadcast &sub_brc_op, Abs &abs_op, Store &sub_store_op,
                           Output &sub_output_op, const ScalarData &scalar_data_op, const Scalar &sub_scalar_op,
                           const af::Axis &z0, const af::Axis &z1, const af::Expression &s0, const af::Expression &s1) {
  vf_op.InstanceOutputy(1);
  vf_op.x = {scalar_data_op.y};
  vf_op.attr.sched.axis = {z0.id, z1.id};
  SetTwoDimSchedule(vf_op.y[0], z0, z1, s0, s1);

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::Symbol(0));
  SetTwoDimSchedule(store_op.y, z0, z1, s0, s1);

  sub_brc_op.x = sub_scalar_op.y;
  sub_brc_op.attr.sched.axis = {z0.id, z1.id};
  SetTwoDimSchedule(sub_brc_op.y, z0, z1, s0, s1);

  abs_op.x = sub_brc_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id};
  SetTwoDimSchedule(abs_op.y, z0, z1, s0, s1);

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {z0.id, z1.id};
  SetTwoDimSchedule(sub_store_op.y, z0, z1, s0, s1);
  sub_output_op.x = sub_store_op.y;
}

void InitScalarDataVfTensorAttrs(AscGraph &graph, AscGraph &vf_sub_graph, const af::Axis &z0, const af::Axis &z1,
                                 const af::Expression &s1) {
  auto scalar_data = graph.FindNode("scalar_data");
  scalar_data->outputs[0].attr.dtype = af::DT_FLOAT;
  scalar_data->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  scalar_data->outputs[0].attr.mem.tensor_id = 1;
  scalar_data->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sub_scalar = vf_sub_graph.FindNode("sub_scalar");
  sub_scalar->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  sub_scalar->outputs[0].attr.vectorized_strides = {Zero, Zero};
  sub_scalar->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_scalar->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_scalar->outputs[0].attr.mem.tensor_id = 1;

  SetTwoDimVecInAttr(vf_sub_graph.FindNode("sub_brc")->outputs[0], z0, z1, s1, 2);
  SetTwoDimVecInAttr(vf_sub_graph.FindNode("abs")->outputs[0], z0, z1, s1, 3);
  SetTwoDimVecInAttr(vf_sub_graph.FindNode("sub_store")->outputs[0], z0, z1, s1, 4);

  auto vf = graph.FindNode("vf");
  SetTwoDimVecInAttr(vf->outputs[0], z0, z1, s1, 5);
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 1;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  store->outputs[0].attr.vectorized_strides = {s1, One};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 6;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

std::string GenerateScalarDataVfCall(AscGraph &graph, const af::Axis &z0, const af::Axis &z1, const af::Expression &s0,
                                     const af::Expression &s1) {
  auto scalar_data = graph.FindNode("scalar_data");
  auto vf = graph.FindNode("vf");
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  EXPECT_EQ(tpipe.AddTensor(scalar_data->outputs[0], "scalar_data_y"), 0);
  tpipe.AddTensor(vf->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = scalar_data->outputs[0].attr.mem.tensor_id;
  x2.id = vf->outputs[0].attr.mem.tensor_id;

  codegen::VfCall call;
  EXPECT_EQ(call.Init(vf), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::stringstream func_def;
  EXPECT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  return result;
}

template <typename TensorLike>
void SetFiveDimSchedule(TensorLike &tensor, const std::vector<af::Axis> &axes,
                        const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  *tensor.axis = {axes[0].id, axes[1].id, axes[2].id, axes[3].id, axes[4].id};
  *tensor.repeats = repeats;
  *tensor.strides = strides;
}

void SetFiveDimVecInAttr(AscTensor &tensor, const std::vector<af::Axis> &axes,
                         const std::vector<af::Expression> &strides, int64_t tensor_id) {
  tensor.attr.vectorized_axis = {axes[0].id, axes[1].id, axes[2].id, axes[3].id, axes[4].id};
  tensor.attr.vectorized_strides = strides;
  tensor.attr.dtype = af::DT_FLOAT;
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.mem.tensor_id = tensor_id;
}

void InitDoubleLoopVfGraph(VectorFunc &vf_op, Store &store_op, Load &sub_load_op, Abs &abs_op, Store &sub_store_op,
                           Output &sub_output_op, const Load &load_op, const Data &sub_x_op,
                           const std::vector<af::Axis> &axes, const std::vector<af::Expression> &repeats,
                           const std::vector<af::Expression> &strides) {
  vf_op.InstanceOutputy(1);
  vf_op.x = {load_op.y};
  vf_op.attr.sched.axis = {axes[0].id, axes[1].id, axes[2].id, axes[3].id, axes[4].id};
  SetFiveDimSchedule(vf_op.y[0], axes, repeats, strides);

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::Symbol(0));
  SetFiveDimSchedule(store_op.y, axes, repeats, strides);

  sub_load_op.x = sub_x_op.y;
  sub_load_op.attr.sched.axis = {axes[3].id, axes[4].id};
  sub_load_op.attr.sched.loop_axis = axes[4].id;
  SetFiveDimSchedule(sub_load_op.y, axes, repeats, strides);

  abs_op.x = sub_load_op.y;
  abs_op.attr.sched.axis = {axes[3].id, axes[4].id};
  abs_op.attr.sched.loop_axis = axes[4].id;
  SetFiveDimSchedule(abs_op.y, axes, repeats, strides);

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {axes[3].id, axes[4].id};
  sub_store_op.attr.sched.loop_axis = axes[4].id;
  SetFiveDimSchedule(sub_store_op.y, axes, repeats, strides);
  sub_output_op.x = sub_store_op.y;
}

void InitDoubleLoopTensorAttrs(AscGraph &graph, AscGraph &vf_sub_graph, const std::vector<af::Axis> &axes,
                               const std::vector<af::Expression> &base_strides,
                               const std::vector<af::Expression> &output_strides) {
  auto load = graph.FindNode("load");
  SetFiveDimVecInAttr(load->outputs[0], axes, base_strides, 0);
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  SetFiveDimVecInAttr(vf_sub_graph.FindNode("sub_load")->outputs[0], axes, base_strides, 0);
  SetFiveDimVecInAttr(vf_sub_graph.FindNode("abs")->outputs[0], axes, base_strides, 1);
  SetFiveDimVecInAttr(vf_sub_graph.FindNode("sub_store")->outputs[0], axes, output_strides, 2);

  auto vf = graph.FindNode("vf");
  SetFiveDimVecInAttr(vf->outputs[0], axes, output_strides, 1);
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 1;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.vectorized_axis = {axes[0].id, axes[1].id, axes[2].id, axes[3].id, axes[4].id};
  store->outputs[0].attr.vectorized_strides = output_strides;
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

void PrepareDoubleLoopCodegen(AscGraph &graph, const std::vector<af::Axis> &axes,
                              const std::vector<af::Expression> &repeats, codegen::Tiler &tiler, codegen::TPipe &tpipe,
                              codegen::VfCall &call, std::stringstream &func_def) {
  auto load = graph.FindNode("load");
  auto vf = graph.FindNode("vf");
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(vf->outputs[0]);
  for (const auto &axis : axes) {
    tiler.AddAxis(axis);
  }
  for (const auto &repeat : repeats) {
    tiler.AddSizeVar(af::SizeVar(repeat));
  }

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  ASSERT_EQ(call.Init(vf), 0);
  call.inputs.push_back(&x1);
  ASSERT_EQ(ascir_param::EnrichAscirGraphNodeParams(graph), af::SUCCESS);
  ASSERT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);
}

void RegisterDoubleLoopNodes(AscGraph &graph, AscGraph &vf_sub_graph, VectorFunc &vf_op, Data &sub_x_op,
                             Output &sub_output_op, Load &load_op, Store &store_op, const std::string &sub_graph_name) {
  vf_op.SetAttr("sub_graph_name", sub_graph_name);
  sub_x_op.ir_attr.SetIndex(0);
  sub_output_op.ir_attr.SetIndex(0);
  graph.AddNode(load_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);
}

void CheckDoubleLoopVectorFuncParams(const af::AscNodePtr &vf) {
  auto params = ascir_param::GetAscirNodeParams(vf);
  ASSERT_NE(params, nullptr);
  const auto *vector_func_params = std::get_if<ascir_param::VectorFuncNodeParams>(&params->specific_params);
  ASSERT_NE(vector_func_params, nullptr);
  EXPECT_TRUE(vector_func_params->is_double_loop);
  ASSERT_EQ(vector_func_params->all_strides.size(), 6U);
  EXPECT_STREQ(vector_func_params->all_strides[0].Serialize().get(), "(s2 * s3 * s4)");
  EXPECT_STREQ(vector_func_params->all_strides[5].Serialize().get(), "(2 * s4)");
  ASSERT_EQ(vector_func_params->output_dims.size(), 4U);
  EXPECT_STREQ(vector_func_params->output_dims[0].Serialize().get(), "s1");
  EXPECT_STREQ(vector_func_params->output_dims[3].Serialize().get(), "s4");
}
}  // namespace

TEST(CodegenKernel, VfCall_TwoDimLoad) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");

  std::string sub_graph_name = "vf_sub_graph1";
  // 创建VectorFunc的子图
  af::AscGraph vf_sub_graph(sub_graph_name.c_str());
  VectorFunc vf_op("vf");
  vf_op.SetAttr("sub_graph_name", sub_graph_name);

  Data sub_x_op("sub_x", vf_sub_graph);
  sub_x_op.ir_attr.SetIndex(0);

  Load sub_load_op("sub_load");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  sub_output_op.ir_attr.SetIndex(0);

  Store store_op("store");
  graph.AddNode(load_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  vf_op.InstanceOutputy(1);
  vf_op.x = {load_op.y};
  vf_op.attr.sched.axis = {z0.id, z1.id};
  *vf_op.y[0].axis = {z0.id, z1.id};
  *vf_op.y[0].repeats = {s0, s1};
  *vf_op.y[0].strides = {s1, One};

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

  auto z0z1 = graph.MergeAxis({z0.id, z1.id});
  for (auto node : vf_sub_graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    vf_sub_graph.ApplyMerge(node, z0z1->id);
  }

  sub_load_op.x = sub_x_op.y;
  sub_load_op.attr.sched.axis = {z0z1->id};
  *sub_load_op.y.axis = {z0z1->id};
  *sub_load_op.y.repeats = {s0 * s1};
  *sub_load_op.y.strides = {One};

  abs_op.x = sub_load_op.y;
  abs_op.attr.sched.axis = {z0z1->id};
  *abs_op.y.axis = {z0z1->id};
  *abs_op.y.repeats = {s0 * s1};
  *abs_op.y.strides = {One};

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {z0z1->id};
  *sub_store_op.y.axis = {z0z1->id};
  *sub_store_op.y.repeats = {s0 * s1};
  *sub_store_op.y.strides = {One};

  sub_output_op.x = sub_store_op.y;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {s1, One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sub_load = vf_sub_graph.FindNode("sub_load");
  sub_load->outputs[0].attr.vectorized_axis = {z0z1->id};
  sub_load->outputs[0].attr.vectorized_strides = {One};
  sub_load->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_load->outputs[0].attr.mem.tensor_id = 0;

  auto abs = vf_sub_graph.FindNode("abs");
  abs->outputs[0].attr.vectorized_axis = {z0z1->id};
  abs->outputs[0].attr.vectorized_strides = {One};
  abs->outputs[0].attr.dtype = af::DT_FLOAT;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;

  auto sub_store = vf_sub_graph.FindNode("sub_store");
  sub_store->outputs[0].attr.vectorized_axis = {z0z1->id};
  sub_store->outputs[0].attr.vectorized_strides = {One};
  sub_store->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_store->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_store->outputs[0].attr.mem.tensor_id = 2;

  auto vf = graph.FindNode("vf");
  vf->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  vf->outputs[0].attr.vectorized_strides = {s1, One};
  vf->outputs[0].attr.dtype = af::DT_FLOAT;
  vf->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  vf->outputs[0].attr.mem.tensor_id = 1;
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 1;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  store->outputs[0].attr.vectorized_strides = {s1, One};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Kernel kernel("test_kernel");
  EXPECT_EQ(IsDataTypeSupported(graph), 0);

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(vf->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(*z0z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::VfCall call;
  EXPECT_EQ(call.Init(vf), 0);
  call.inputs.push_back(&x1);

  std::stringstream func_def;
  EXPECT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(
      result,
      std::string{
          "#if defined(__DAV_C310__) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3510))\n"
          "AscendC::SetCtrlSpr<60, 60>(0);\n"
          "VFCallvf((__local_mem__ float *)local_1[0].GetPhyAddr(), (__local_mem__ float "
          "*)local_0[0].GetPhyAddr(), t->s0 * t->s1);\n"
          "#endif\n"});
}

TEST(CodegenKernel, VfCall_TwoDimLoad_VFLoop) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");

  std::string sub_graph_name = "vf_sub_graph1";
  // 创建VectorFunc的子图
  af::AscGraph vf_sub_graph(sub_graph_name.c_str());
  VectorFunc vf_op("vf");
  vf_op.SetAttr("sub_graph_name", sub_graph_name);

  Data sub_x_op("sub_x", vf_sub_graph);
  sub_x_op.ir_attr.SetIndex(0);

  Load sub_load_op("sub_load");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  sub_output_op.ir_attr.SetIndex(0);

  Store store_op("store");
  graph.AddNode(load_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, Zero};

  vf_op.InstanceOutputy(1);
  vf_op.x = {load_op.y};
  vf_op.attr.sched.axis = {z0.id, z1.id};
  *vf_op.y[0].axis = {z0.id, z1.id};
  *vf_op.y[0].repeats = {s0, s1};
  *vf_op.y[0].strides = {s1, Zero};

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, Zero};

  sub_load_op.x = sub_x_op.y;
  sub_load_op.attr.sched.axis = {z0.id, z1.id};
  *sub_load_op.y.axis = {z0.id, z1.id};
  *sub_load_op.y.repeats = {s0, s1};
  *sub_load_op.y.strides = {s1, Zero};

  abs_op.x = sub_load_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id};
  *abs_op.y.axis = {z0.id, z1.id};
  *abs_op.y.repeats = {s0, s1};
  *abs_op.y.strides = {s1, Zero};

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {z0.id, z1.id};
  *sub_store_op.y.axis = {z0.id, z1.id};
  *sub_store_op.y.repeats = {s0, s1};
  *sub_store_op.y.strides = {s1, Zero};

  sub_output_op.x = sub_store_op.y;

  auto load = graph.FindNode("load");
  load->attr.sched.loop_axis = -1;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {Zero, Zero};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sub_load = vf_sub_graph.FindNode("sub_load");
  sub_load->attr.sched.loop_axis = -1;
  sub_load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  sub_load->outputs[0].attr.vectorized_strides = {Zero, Zero};
  sub_load->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_load->outputs[0].attr.mem.tensor_id = 0;

  auto abs = vf_sub_graph.FindNode("abs");
  abs->attr.sched.loop_axis = -1;
  abs->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  abs->outputs[0].attr.vectorized_strides = {Zero, Zero};
  abs->outputs[0].attr.dtype = af::DT_FLOAT;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;

  auto sub_store = vf_sub_graph.FindNode("sub_store");
  sub_store->attr.sched.loop_axis = -1;
  sub_store->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  sub_store->outputs[0].attr.vectorized_strides = {Zero, Zero};
  sub_store->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_store->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_store->outputs[0].attr.mem.tensor_id = 2;

  auto vf = graph.FindNode("vf");
  vf->attr.sched.loop_axis = -1;
  vf->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  vf->outputs[0].attr.vectorized_strides = {Zero, Zero};
  vf->outputs[0].attr.dtype = af::DT_FLOAT;
  vf->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  vf->outputs[0].attr.mem.tensor_id = 1;
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 1;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.sched.loop_axis = -1;
  store->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  store->outputs[0].attr.vectorized_strides = {Zero, Zero};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Kernel kernel("test_kernel");
  EXPECT_EQ(IsDataTypeSupported(graph), 0);

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(vf->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::VfCall call;
  EXPECT_EQ(call.Init(vf), 0);
  call.inputs.push_back(&x1);

  std::stringstream func_def;
  EXPECT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);

  std::string result = func_def.str();
  EXPECT_TRUE(result.find("AscendC::MicroAPI::MaskReg preg_main = AscendC::MicroAPI::CreateMask") != std::string::npos);
  EXPECT_TRUE(result.find("uint32_t element_count") != std::string::npos);
  EXPECT_TRUE(result.find("uint16_t loop_times") != std::string::npos);
}

TEST(CodegenKernel, VfCall_TwoDim_Scalar) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Scalar x_op("x", graph);
  x_op.ir_attr.SetValue("2.0");

  std::string sub_graph_name = "vf_sub_graph1";
  // 创建VectorFunc的子图
  af::AscGraph vf_sub_graph(sub_graph_name.c_str());
  VectorFunc vf_op("vf");
  vf_op.SetAttr("sub_graph_name", sub_graph_name);

  Scalar sub_x_op("sub_x", vf_sub_graph);
  sub_x_op.ir_attr.SetValue("2.0");
  sub_x_op.ir_attr.SetIndex(0);

  Broadcast sub_brc_op("sub_brc");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  sub_output_op.ir_attr.SetIndex(0);

  Store store_op("store");
  graph.AddNode(x_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);

  vf_op.InstanceOutputy(1);
  vf_op.x = {x_op.y};
  vf_op.attr.sched.axis = {z0.id, z1.id};
  *vf_op.y[0].axis = {z0.id, z1.id};
  *vf_op.y[0].repeats = {s0, s1};
  *vf_op.y[0].strides = {s1, One};

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

  sub_brc_op.x = sub_x_op.y;
  sub_brc_op.attr.sched.axis = {z0.id, z1.id};
  *sub_brc_op.y.axis = {z0.id, z1.id};
  *sub_brc_op.y.repeats = {s0, s1};
  *sub_brc_op.y.strides = {s1, One};

  abs_op.x = sub_brc_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id};
  *abs_op.y.axis = {z0.id, z1.id};
  *abs_op.y.repeats = {s0, s1};
  *abs_op.y.strides = {s1, One};

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {z0.id, z1.id};
  *sub_store_op.y.axis = {z0.id, z1.id};
  *sub_store_op.y.repeats = {s0, s1};
  *sub_store_op.y.strides = {s1, One};

  sub_output_op.x = sub_store_op.y;

  auto x = graph.FindNode("x");
  x->outputs[0].attr.dtype = af::DT_FLOAT;
  x->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x->outputs[0].attr.que.id = 0;
  x->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sub_x = vf_sub_graph.FindNode("sub_x");
  sub_x->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  sub_x->outputs[0].attr.vectorized_strides = {Zero, Zero};
  sub_x->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_x->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_x->outputs[0].attr.mem.tensor_id = 0;

  auto sub_brc = vf_sub_graph.FindNode("sub_brc");
  sub_brc->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  sub_brc->outputs[0].attr.vectorized_strides = {s1, One};
  sub_brc->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_brc->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_brc->outputs[0].attr.mem.tensor_id = 1;

  auto abs = vf_sub_graph.FindNode("abs");
  abs->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  abs->outputs[0].attr.vectorized_strides = {s1, One};
  abs->outputs[0].attr.dtype = af::DT_FLOAT;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 2;

  auto sub_store = vf_sub_graph.FindNode("sub_store");
  sub_store->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  sub_store->outputs[0].attr.vectorized_strides = {s1, One};
  sub_store->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_store->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_store->outputs[0].attr.mem.tensor_id = 3;

  auto vf = graph.FindNode("vf");
  vf->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  vf->outputs[0].attr.vectorized_strides = {s1, One};
  vf->outputs[0].attr.dtype = af::DT_FLOAT;
  vf->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  vf->outputs[0].attr.mem.tensor_id = 1;
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 1;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  store->outputs[0].attr.vectorized_strides = {s1, One};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Kernel kernel("test_kernel");
  EXPECT_EQ(IsDataTypeSupported(graph), 0);

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor("2.0", x->outputs[0], "scalar_x");
  tpipe.AddTensor(vf->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);
  codegen::ApiTensor x1, x2;
  x1.id = x->outputs[0].attr.mem.tensor_id;
  x2.id = vf->outputs[0].attr.mem.tensor_id;

  codegen::VfCall call;
  EXPECT_EQ(call.Init(vf), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::stringstream func_def;
  EXPECT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(
      result,
      std::string{
          "#if defined(__DAV_C310__) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3510))\n"
          "AscendC::SetCtrlSpr<60, 60>(0);\n"
          "VFCallvf((__local_mem__ float *)local_1[0].GetPhyAddr(), (__local_mem__ float "
          "*)local_1[0].GetPhyAddr(), scalar_0, t->s0 * t->s1);\n"
          "#endif\n"});
}

TEST(CodegenKernel, VfCall_TwoDim_ScalarData) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ScalarData scalar_data_op("scalar_data", graph);
  scalar_data_op.ir_attr.SetIndex(0);

  std::string sub_graph_name = "vf_sub_graph1";
  af::AscGraph vf_sub_graph(sub_graph_name.c_str());
  VectorFunc vf_op("vf");
  vf_op.SetAttr("sub_graph_name", sub_graph_name);

  Scalar sub_scalar_op("sub_scalar", vf_sub_graph);
  sub_scalar_op.ir_attr.SetIndex(0);

  Broadcast sub_brc_op("sub_brc");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  sub_output_op.ir_attr.SetIndex(0);

  Store store_op("store");
  graph.AddNode(scalar_data_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);

  InitScalarDataVfGraph(vf_op, store_op, sub_brc_op, abs_op, sub_store_op, sub_output_op, scalar_data_op, sub_scalar_op,
                        z0, z1, s0, s1);
  InitScalarDataVfTensorAttrs(graph, vf_sub_graph, z0, z1, s1);

  auto result = GenerateScalarDataVfCall(graph, z0, z1, s0, s1);
  EXPECT_NE(result.find("scalar_data"), std::string::npos) << result;
  EXPECT_EQ(result.find("scalar_1"), std::string::npos) << result;
}

TEST(CodegenKernel, VfCall_ThreeDimLoad) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Load load_op("load");

  std::string sub_graph_name = "vf_sub_graph1";
  // 创建VectorFunc的子图
  af::AscGraph vf_sub_graph(sub_graph_name.c_str());
  VectorFunc vf_op("vf");
  vf_op.SetAttr("sub_graph_name", sub_graph_name);

  Data sub_x_op("sub_x", vf_sub_graph);
  sub_x_op.ir_attr.SetIndex(0);

  Load sub_load_op("sub_load");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  sub_output_op.ir_attr.SetIndex(0);

  Store store_op("store");
  graph.AddNode(load_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, s1, s2};
  *load_op.y.strides = {Two * s1 * s2, Two * s2, One};

  vf_op.InstanceOutputy(1);
  vf_op.x = {load_op.y};
  vf_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *vf_op.y[0].axis = {z0.id, z1.id, z2.id};
  *vf_op.y[0].repeats = {s0, s1, s2};
  *vf_op.y[0].strides = {Two * s1 * s2, Two * s2, One};

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {Two * s1 * s2, Two * s2, One};

  auto z0z1 = graph.MergeAxis({z0.id, z1.id});
  for (auto node : vf_sub_graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    vf_sub_graph.ApplyMerge(node, z0z1->id);
  }

  sub_load_op.x = sub_x_op.y;
  sub_load_op.attr.sched.axis = {z0z1->id, z2.id};
  *sub_load_op.y.axis = {z0z1->id, z2.id};
  *sub_load_op.y.repeats = {s0 * s1, s2};
  *sub_load_op.y.strides = {Two * s2, One};

  abs_op.x = sub_load_op.y;
  abs_op.attr.sched.axis = {z0z1->id, z2.id};
  *abs_op.y.axis = {z0z1->id, z2.id};
  *abs_op.y.repeats = {s0 * s1, s2};
  *abs_op.y.strides = {Two * s2, One};

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {z0z1->id, z2.id};
  *sub_store_op.y.axis = {z0z1->id, z2.id};
  *sub_store_op.y.repeats = {s0 * s1, s2};
  *sub_store_op.y.strides = {Two * s2, One};

  sub_output_op.x = sub_store_op.y;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {Two * s1 * s2, Two * s2, One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sub_load = vf_sub_graph.FindNode("sub_load");
  sub_load->outputs[0].attr.vectorized_axis = {z0z1->id, z2.id};
  sub_load->outputs[0].attr.vectorized_strides = {Two * s2, One};
  sub_load->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_load->outputs[0].attr.mem.tensor_id = 0;

  auto abs = vf_sub_graph.FindNode("abs");
  abs->outputs[0].attr.vectorized_axis = {z0z1->id, z2.id};
  abs->outputs[0].attr.vectorized_strides = {Two * s2, One};
  abs->outputs[0].attr.dtype = af::DT_FLOAT;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;

  auto sub_store = vf_sub_graph.FindNode("sub_store");
  sub_store->outputs[0].attr.vectorized_axis = {z0z1->id, z2.id};
  sub_store->outputs[0].attr.vectorized_strides = {Two * s2, One};
  sub_store->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_store->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_store->outputs[0].attr.mem.tensor_id = 2;

  auto vf = graph.FindNode("vf");
  vf->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  vf->outputs[0].attr.vectorized_strides = {Two * s1 * s2, Two * s2, One};
  vf->outputs[0].attr.dtype = af::DT_FLOAT;
  vf->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  vf->outputs[0].attr.mem.tensor_id = 1;
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 1;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  store->outputs[0].attr.vectorized_strides = {Two * s1 * s2, Two * s2, One};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Kernel kernel("test_kernel");
  EXPECT_EQ(IsDataTypeSupported(graph), 0);

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(vf->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(*z0z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));

  vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::VfCall call;
  EXPECT_EQ(call.Init(vf), 0);
  call.inputs.push_back(&x1);

  std::stringstream func_def;
  EXPECT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(
      result,
      std::string{
          "#if defined(__DAV_C310__) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3510))\n"
          "AscendC::SetCtrlSpr<60, 60>(0);\n"
          "VFCallvf((__local_mem__ float *)local_1[0].GetPhyAddr(), (__local_mem__ float "
          "*)local_0[0].GetPhyAddr(), t->s0 * t->s1, t->s2, (2 * t->s2), (2 * t->s2));\n"
          "#endif\n"});
}

TEST(CodegenKernel, VfCall_FiveDimLoad) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);
  af::Expression Five = af::Symbol(5);
  af::Expression Six = af::Symbol(6);

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

  std::string sub_graph_name = "vf_sub_graph1";
  // 创建VectorFunc的子图
  af::AscGraph vf_sub_graph(sub_graph_name.c_str());
  VectorFunc vf_op("vf");
  vf_op.SetAttr("sub_graph_name", sub_graph_name);

  Data sub_x_op("sub_x", vf_sub_graph);
  sub_x_op.ir_attr.SetIndex(0);

  Load sub_load_op("sub_load");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  sub_output_op.ir_attr.SetIndex(0);

  Store store_op("store");
  graph.AddNode(load_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load_op.y.repeats = {s0, s1, s2, s3, s4};
  *load_op.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  vf_op.InstanceOutputy(1);
  vf_op.x = {load_op.y};
  vf_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *vf_op.y[0].axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *vf_op.y[0].repeats = {s0, s1, s2, s3, s4};
  *vf_op.y[0].strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store_op.y.repeats = {s0, s1, s2, s3, s4};
  *store_op.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  sub_load_op.x = sub_x_op.y;
  sub_load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *sub_load_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *sub_load_op.y.repeats = {s0, s1, s2, s3, s4};
  *sub_load_op.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  abs_op.x = sub_load_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs_op.y.repeats = {s0, s1, s2, s3, s4};
  *abs_op.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {z0.id, z1.id};
  *sub_store_op.y.axis = {z0.id, z1.id};
  *sub_store_op.y.repeats = {s0, s1, s2, s3, s4};
  *sub_store_op.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  sub_output_op.x = sub_store_op.y;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load->outputs[0].attr.vectorized_strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sub_load = vf_sub_graph.FindNode("sub_load");
  sub_load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  sub_load->outputs[0].attr.vectorized_strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  sub_load->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_load->outputs[0].attr.mem.tensor_id = 0;

  auto abs = vf_sub_graph.FindNode("abs");
  abs->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs->outputs[0].attr.vectorized_strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  abs->outputs[0].attr.dtype = af::DT_FLOAT;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;

  auto sub_store = vf_sub_graph.FindNode("sub_store");
  sub_store->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  sub_store->outputs[0].attr.vectorized_strides = {s1 * s2 * s3 * s4 * Five, s2 * s3 * s4 * Four, s3 * s4 * Three,
                                                   s4 * Two, One};
  sub_store->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_store->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_store->outputs[0].attr.mem.tensor_id = 2;

  auto vf = graph.FindNode("vf");
  vf->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  vf->outputs[0].attr.vectorized_strides = {s1 * s2 * s3 * s4 * Five, s2 * s3 * s4 * Four, s3 * s4 * Three, s4 * Two,
                                            One};
  vf->outputs[0].attr.dtype = af::DT_FLOAT;
  vf->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  vf->outputs[0].attr.mem.tensor_id = 1;
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 1;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store->outputs[0].attr.vectorized_strides = {s1 * s2 * s3 * s4 * Five, s2 * s3 * s4 * Four, s3 * s4 * Three, s4 * Two,
                                               One};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Kernel kernel("test_kernel");
  EXPECT_EQ(IsDataTypeSupported(graph), 0);

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(vf->outputs[0]);

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

  vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::VfCall call;
  EXPECT_EQ(call.Init(vf), 0);
  call.inputs.push_back(&x1);

  std::stringstream func_def;
  EXPECT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(
      result,
      std::string{
          "#if defined(__DAV_C310__) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3510))\n"
          "AscendC::SetCtrlSpr<60, 60>(0);\n"
          "for(int outer_for_0 = 0; outer_for_0 < t->s0; outer_for_0++) {\n"
          "VFCallvf((__local_mem__ float "
          "*)local_1[outer_for_0 * (5 * t->s1 * t->s2 * t->s3 * t->s4)].GetPhyAddr(), (__local_mem__ float "
          "*)local_0[outer_for_0 * (t->s1 * t->s2 * t->s3 * t->s4)].GetPhyAddr(), t->s1, t->s2, t->s3, t->s4, (4 * "
          "t->s2 "
          "* t->s3 * t->s4), (3 * t->s3 * t->s4), (2 * t->s4), (t->s2 * t->s3 * t->s4), (t->s3 * t->s4), t->s4);\n\n"
          "}\n"
          "#endif\n"});
}

TEST(CodegenKernel, VfCall_DoubleLoopWritesVectorFuncParams) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);
  af::Expression Five = af::Symbol(5);

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
  const std::vector<af::Axis> axes = {z0, z1, z2, z3, z4};
  const std::vector<af::Expression> repeats = {s0, s1, s2, s3, s4};
  const std::vector<af::Expression> base_strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  const std::vector<af::Expression> output_strides = {s1 * s2 * s3 * s4 * Five, s2 * s3 * s4 * Four, s3 * s4 * Three,
                                                      s4 * Two, One};

  Data x_op("x", graph);
  Load load_op("load");
  std::string sub_graph_name = "vf_sub_graph_double_loop";
  af::AscGraph vf_sub_graph(sub_graph_name.c_str());
  VectorFunc vf_op("vf");
  Data sub_x_op("sub_x", vf_sub_graph);
  Load sub_load_op("sub_load");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  Store store_op("store");
  RegisterDoubleLoopNodes(graph, vf_sub_graph, vf_op, sub_x_op, sub_output_op, load_op, store_op, sub_graph_name);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  SetFiveDimSchedule(load_op.y, axes, repeats, base_strides);
  InitDoubleLoopVfGraph(vf_op, store_op, sub_load_op, abs_op, sub_store_op, sub_output_op, load_op, sub_x_op, axes,
                        repeats, base_strides);
  InitDoubleLoopTensorAttrs(graph, vf_sub_graph, axes, base_strides, output_strides);

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  codegen::VfCall call;
  std::stringstream func_def;
  PrepareDoubleLoopCodegen(graph, axes, repeats, tiler, tpipe, call, func_def);

  auto vf = graph.FindNode("vf");
  CheckDoubleLoopVectorFuncParams(vf);
}

// 测试单维场景不触发优化逻辑
TEST(CodegenKernel, VfCall_OneDim_NoOptimization) {
  ge::SetupRuntimeStub();
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  Data x_op("x", graph);
  Load load_op("load");

  std::string sub_graph_name = "vf_sub_graph_1d";
  af::AscGraph vf_sub_graph(sub_graph_name.c_str());
  VectorFunc vf_op("vf");
  vf_op.SetAttr("sub_graph_name", sub_graph_name);

  Data sub_x_op("sub_x", vf_sub_graph);
  sub_x_op.ir_attr.SetIndex(0);

  Load sub_load_op("sub_load");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  sub_output_op.ir_attr.SetIndex(0);

  Store store_op("store");
  graph.AddNode(load_op);
  graph.AddSubGraph(vf_sub_graph);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id};
  *load_op.y.axis = {z0.id};
  *load_op.y.repeats = {s0};
  *load_op.y.strides = {One};

  vf_op.InstanceOutputy(1);
  vf_op.x = {load_op.y};
  vf_op.attr.sched.axis = {z0.id};
  *vf_op.y[0].axis = {z0.id};
  *vf_op.y[0].repeats = {s0};
  *vf_op.y[0].strides = {One};

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id};
  *store_op.y.repeats = {s0};
  *store_op.y.strides = {One};

  sub_load_op.x = sub_x_op.y;
  sub_load_op.attr.sched.axis = {z0.id};
  sub_load_op.attr.sched.loop_axis = -1;
  *sub_load_op.y.axis = {z0.id};
  *sub_load_op.y.repeats = {s0};
  *sub_load_op.y.strides = {One};

  abs_op.x = sub_load_op.y;
  abs_op.attr.sched.axis = {z0.id};
  abs_op.attr.sched.loop_axis = -1;
  *abs_op.y.axis = {z0.id};
  *abs_op.y.repeats = {s0};
  *abs_op.y.strides = {One};

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {z0.id};
  sub_store_op.attr.sched.loop_axis = -1;
  *sub_store_op.y.axis = {z0.id};
  *sub_store_op.y.repeats = {s0};
  *sub_store_op.y.strides = {One};

  sub_output_op.x = sub_store_op.y;

  auto load = graph.FindNode("load");
  load->attr.sched.loop_axis = -1;
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sub_load = vf_sub_graph.FindNode("sub_load");
  sub_load->attr.sched.loop_axis = -1;
  sub_load->outputs[0].attr.vectorized_axis = {z0.id};
  sub_load->outputs[0].attr.vectorized_strides = {One};
  sub_load->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_load->outputs[0].attr.mem.tensor_id = 0;

  auto abs = vf_sub_graph.FindNode("abs");
  abs->attr.sched.loop_axis = -1;
  abs->outputs[0].attr.vectorized_axis = {z0.id};
  abs->outputs[0].attr.vectorized_strides = {One};
  abs->outputs[0].attr.dtype = af::DT_FLOAT;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;

  auto sub_store = vf_sub_graph.FindNode("sub_store");
  sub_store->attr.sched.loop_axis = -1;
  sub_store->outputs[0].attr.vectorized_axis = {z0.id};
  sub_store->outputs[0].attr.vectorized_strides = {One};
  sub_store->outputs[0].attr.dtype = af::DT_FLOAT;
  sub_store->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  sub_store->outputs[0].attr.mem.tensor_id = 2;

  auto vf = graph.FindNode("vf");
  vf->attr.sched.loop_axis = -1;
  vf->outputs[0].attr.vectorized_axis = {z0.id};
  vf->outputs[0].attr.vectorized_strides = {One};
  vf->outputs[0].attr.dtype = af::DT_FLOAT;
  vf->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  vf->outputs[0].attr.mem.tensor_id = 1;
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 1;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.sched.loop_axis = -1;
  store->outputs[0].attr.vectorized_axis = {z0.id};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Kernel kernel("test_kernel");
  EXPECT_EQ(IsDataTypeSupported(graph), 0);

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(vf->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddSizeVar(af::SizeVar(s0));

  vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::VfCall call;
  EXPECT_EQ(call.Init(vf), 0);
  call.inputs.push_back(&x1);

  std::stringstream func_def;
  EXPECT_EQ(call.GenerateFuncDefinition(tpipe, tiler, func_def), 0);

  std::string result = func_def.str();

  // 单维场景（dim_size < MAX_VF_AXIS_MERGE_SIZE），不应该触发优化逻辑
  // 检查不应该包含优化相关的代码
  EXPECT_FALSE(result.find("output_dims_1 != strides_align") != std::string::npos)
      << "Optimization code should not be generated for 1D case (dim_size < MAX_VF_AXIS_MERGE_SIZE)";
}
