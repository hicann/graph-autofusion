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
#include "reg_load_api_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

TEST(CodegenKernel, LoadRegApiCall_OneDimLoad) {
  af::AscGraph graph("test_graph");

  auto s0 = af::Symbol(16);
  auto s1 = af::Symbol(8);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  store_op.x = load_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::LoadRegApiCall call_0("DataCopyPadExtend");
  EXPECT_EQ(call_0.Init(load), 0);
  call_0.inputs.push_back(&x1);

  std::string result;
  call_0.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(
      result,
      std::string{"DataCopyPadExtend<float, AscendC::PaddingMode::Normal>(local_0[0], local_0[0 + 0], 1, 8, 0, 0);\n"});
}

TEST(CodegenKernel, NormalModeDataCopyIfDualSplitting) {
  af::AscGraph graph("test_graph");
  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);

  auto s0 = af::Symbol(16);
  auto s1 = af::Symbol(8);
  auto s2 = af::Symbol(4);
  auto s3 = af::Symbol(2);
  auto z0 = graph.CreateAxis("z0", af::Axis::kAxisTypeTileInner, s0, {}, -1);
  auto z1 = graph.CreateAxis("z1", af::Axis::kAxisTypeTileInner, s1, {}, -1);
  auto z2 = graph.CreateAxis("z2", af::Axis::kAxisTypeTileInner, s2, {}, -1);
  auto z3 = graph.CreateAxis("z3", af::Axis::kAxisTypeTileInner, s3, {}, -1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, s3};
  *load_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};
  store_op.x = load_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id};
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

  codegen::LoadRegApiCall call_0("DataCopyPadExtend");
  EXPECT_EQ(call_0.Init(load), 0);
  call_0.inputs.push_back(&x1);

  std::string result;
  call_0.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(
      result,
      std::string{"DataCopyPadExtend<float, AscendC::PaddingMode::Normal>(local_0[0], local_0[0 + 0], 4, 2, 2, (2 "
                  "- 2), {static_cast<uint32_t>(8), "
                  "static_cast<uint32_t>(1), static_cast<uint64_t>(24 * 4), static_cast<uint64_t>(8 * 4), "
                  "static_cast<uint64_t>(0 * 4), static_cast<uint64_t>(0 * 4)});\n"});
}

TEST(CodegenKernel, LoadRegApiCall_ThreeDimLoad) {
  af::AscGraph graph("test_graph");

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

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, s3};
  *load_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};
  store_op.x = load_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id};
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

  codegen::LoadRegApiCall call_0("DataCopyPadExtend");
  EXPECT_EQ(call_0.Init(load), 0);
  call_0.inputs.push_back(&x1);

  std::string result;
  call_0.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(
      result,
      std::string{"DataCopyPadExtend<float, AscendC::PaddingMode::Compact>(local_0[0], local_0[0 + 0], 4, 2, 2, (2 "
                  "- 2), {static_cast<uint32_t>(8), "
                  "static_cast<uint32_t>(1), static_cast<uint64_t>(24 * 4), static_cast<uint64_t>(8 * 4), "
                  "static_cast<uint64_t>(0 * 4), static_cast<uint64_t>(0 * 4)});\n"});
}

TEST(CodegenKernel, LoadRegApiCall_FiveDimLoad) {
  af::AscGraph graph("test_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);
  af::Expression Five = af::Symbol(5);
  af::Expression Six = af::Symbol(6);

  auto s0 = af::Symbol(16);
  auto s1 = af::Symbol(8);
  auto s2 = af::Symbol(4);
  auto s3 = af::Symbol(2);
  auto s4 = af::Symbol(2);
  auto s5 = af::Symbol(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  auto z5 = graph.CreateAxis("z5", s5);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *load_op.y.repeats = {s0, s1, s2, s3, s4, s5};
  *load_op.y.strides = {
      s1 * s2 * s3 * s4 * s5 * Six, s2 * s3 * s4 * s5 * Five, s3 * s4 * s5 * Four, s4 * s5 * Three, s5 * Two, One};
  store_op.x = load_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *store_op.y.repeats = {s0, s1, s2, s3, s4, s5};
  *store_op.y.strides = {
      s1 * s2 * s3 * s4 * s5 * Six, s2 * s3 * s4 * s5 * Five, s3 * s4 * s5 * Four, s4 * s5 * Three, s5 * Two, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id, z4.id, z5.id};
  load->outputs[0].attr.vectorized_strides = {af::Symbol(32), af::Symbol(8), af::Symbol(4), af::Symbol(2), One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddAxis(z4);
  tiler.AddAxis(z5);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddSizeVar(af::SizeVar(s3));
  tiler.AddSizeVar(af::SizeVar(s4));
  tiler.AddSizeVar(af::SizeVar(s5));
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::LoadRegApiCall call_0("DataCopyPadExtend");
  EXPECT_EQ(call_0.Init(load), 0);
  call_0.inputs.push_back(&x1);

  std::string result;
  call_0.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result,
            std::string{"for (int outer_for_0 = 0; outer_for_0 < 8; outer_for_0++) {\nDataCopyPadExtend<float, "
                        "AscendC::PaddingMode::Compact>(local_0[outer_for_0 * "
                        "32], local_0[(0 + 0 + outer_for_0 * 160)], 2, 2, 2, (2 - 2), {static_cast<uint32_t>(2), "
                        "static_cast<uint32_t>(4), static_cast<uint64_t>(12 * 4), static_cast<uint64_t>(4 * 4), "
                        "static_cast<uint64_t>(32 * 4), static_cast<uint64_t>(8 * 4)});\n}\n"});
}
