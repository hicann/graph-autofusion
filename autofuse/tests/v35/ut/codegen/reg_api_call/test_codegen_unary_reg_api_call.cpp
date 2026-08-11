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

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "common_utils.h"
#include "utils/api_call_factory.h"
#include "unary_bitwidth_change_api_call_v2.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

namespace codegen {
TEST(CodegenKernel, UnaryApicallIsNan) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Isnan rsqrt_op("IsNan");
  graph.AddNode(rsqrt_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  rsqrt_op.x = load_op.y;
  *rsqrt_op.y.axis = {z0.id, z1.id};
  *rsqrt_op.y.repeats = {s0, s1};
  *rsqrt_op.y.strides = {s1, One};

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

  auto rsqrt = graph.FindNode("IsNan");
  rsqrt->attr.api.compute_type = af::ComputeType::kComputeElewise;
  rsqrt->attr.api.type = af::ApiType::kAPITypeCompute;
  rsqrt->attr.api.unit = af::ComputeUnit::kUnitVector;
  rsqrt->attr.sched.loop_axis = z0.id;
  rsqrt->outputs[0].attr.vectorized_axis = {z1.id};
  rsqrt->outputs[0].attr.vectorized_strides = {One};
  rsqrt->outputs[0].attr.dtype = af::DT_INT16;
  rsqrt->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  rsqrt->outputs[0].attr.mem.tensor_id = 1;
  rsqrt->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  rsqrt->outputs[0].attr.que.id = 2;
  rsqrt->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(rsqrt->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  auto call = CreateApiCallObject(rsqrt);

  codegen::UnaryBitWidthChangeApiCallV2 call_0("IsNan");
  EXPECT_EQ(call_0.Init(rsqrt), 0);
  call_0.inputs.push_back(&x1);

  std::string result;
  call_0.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{"LocalTensor<bool> local_1_cast = local_1.template ReinterpretCast<bool>();\n"
                                "IsNan(local_1_cast[0], local_0[0], local_0_actual_size);\n"});
  delete call;
}

TEST(CodegenKernel, UnaryApicallIsNanThrowingFor) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Isnan isnan("isnan");
  graph.AddNode(load_op);
  graph.AddNode(isnan);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, s1, s2};
  *load_op.y.strides = {s1 * s2, s2, One};

  isnan.x = load_op.y;
  *isnan.y.axis = {z0.id, z1.id, z2.id};
  *isnan.y.repeats = {s0, s1, s2};
  *isnan.y.strides = {s1 * s2, s2, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;

  auto size = af::GetSizeByDataType(af::DT_FLOAT16);
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {af::sym::Align(z2.size, 16), One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto isnan_node = graph.FindNode("isnan");
  isnan_node->attr.api.compute_type = af::ComputeType::kComputeElewise;
  isnan_node->attr.api.type = af::ApiType::kAPITypeCompute;
  isnan_node->attr.api.unit = af::ComputeUnit::kUnitVector;
  isnan_node->attr.sched.loop_axis = z0.id;
  isnan_node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  isnan_node->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  isnan_node->outputs[0].attr.vectorized_strides = {af::sym::Align(z2.size, 16), One};
  isnan_node->outputs[0].attr.dtype = af::DT_INT16;
  isnan_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  isnan_node->outputs[0].attr.mem.tensor_id = 3;
  isnan_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  isnan_node->outputs[0].attr.que.id = 2;
  isnan_node->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(isnan_node->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::UnaryBitWidthChangeApiCallV2 call("IsNan");
  EXPECT_EQ(call.Init(isnan_node), 0);
  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  std::cout << result << std::endl;
  EXPECT_EQ(result,
            std::string{"LocalTensor<bool> local_3_cast = local_3.template ReinterpretCast<bool>();\n"
                        "for(int outer_for_0 = 0; outer_for_0 < t->s1; outer_for_0++) {\n"
                        "IsNan(local_3_cast[outer_for_0 * ((16 * Ceiling((Rational(1 , 16) * t->s2))))/(1)], "
                        "local_0[outer_for_0 * ((16 * Ceiling((Rational(1 , 16) * t->s2))))/(1)], t->s2);\n\n}\n"});
}

TEST(CodegenKernel, UnaryApicallIsFinite) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  IsFinite rsqrt_op("IsFinite");
  graph.AddNode(rsqrt_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  rsqrt_op.x = load_op.y;
  *rsqrt_op.y.axis = {z0.id, z1.id};
  *rsqrt_op.y.repeats = {s0, s1};
  *rsqrt_op.y.strides = {s1, One};

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

  auto rsqrt = graph.FindNode("IsFinite");
  rsqrt->attr.api.compute_type = af::ComputeType::kComputeElewise;
  rsqrt->attr.api.type = af::ApiType::kAPITypeCompute;
  rsqrt->attr.api.unit = af::ComputeUnit::kUnitVector;
  rsqrt->attr.sched.loop_axis = z0.id;
  rsqrt->outputs[0].attr.vectorized_axis = {z1.id};
  rsqrt->outputs[0].attr.vectorized_strides = {One};
  rsqrt->outputs[0].attr.dtype = af::DT_INT16;
  rsqrt->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  rsqrt->outputs[0].attr.mem.tensor_id = 1;
  rsqrt->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  rsqrt->outputs[0].attr.que.id = 2;
  rsqrt->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(rsqrt->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  auto call = CreateApiCallObject(rsqrt);

  codegen::UnaryBitWidthChangeApiCallV2 call_0("IsFinite");
  EXPECT_EQ(call_0.Init(rsqrt), 0);
  call_0.inputs.push_back(&x1);

  std::string result;
  call_0.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{"LocalTensor<bool> local_1_cast = local_1.template ReinterpretCast<bool>();\n"
                                "IsFinite(local_1_cast[0], local_0[0], local_0_actual_size);\n"});
  delete call;
}
}  // namespace codegen
