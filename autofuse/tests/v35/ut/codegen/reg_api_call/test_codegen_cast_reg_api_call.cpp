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
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common_utils.h"
#include "utils/api_call_factory.h"
#include "cast_v2_api_call.h"
#include "ascir_node_param/ascir_node_param.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

TEST(CastV2ApiCallTest, CastV2ApiCall_Zero_Stride) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Cast cast_op("cast");
  graph.AddNode(load_op);
  graph.AddNode(cast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, One, s2};
  *load_op.y.strides = {s2, Zero, One};
  cast_op.x = load_op.y;
  *cast_op.y.axis = {z0.id, z1.id, z2.id};
  *cast_op.y.repeats = {s0, One, s2};
  *cast_op.y.strides = {s2, Zero, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;

  auto size = af::GetSizeByDataType(af::DT_FLOAT16);
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {Zero, One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  cast->outputs[0].attr.vectorized_strides = {Zero, One};
  cast->outputs[0].attr.dtype = af::DT_INT16;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 1;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 2;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(cast->outputs[0]);

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
  codegen::CastV2ApiCall call("CastExtend");
  EXPECT_EQ(call.Init(cast), 0);
  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result, std::string{"CastExtend(local_1[0], local_0[0], {ConvertToUint32(local_0_actual_size)}, "
                                "{ConvertToUint32(1)}, {ConvertToUint32(1)});\n"});

  const auto params = ascir_param::GetAscirNodeParams(cast);
  ASSERT_NE(params, nullptr);
  const auto *cast_params = std::get_if<ascir_param::CastNodeParams>(&params->specific_params);
  ASSERT_NE(cast_params, nullptr);
  ASSERT_EQ(cast_params->output_dims.size(), 1U);
  ASSERT_EQ(cast_params->output_strides.size(), 1U);
  ASSERT_EQ(cast_params->input_strides.size(), 1U);
  EXPECT_STREQ(cast_params->output_dims[0].Serialize().get(), "s2");
  EXPECT_STREQ(cast_params->output_strides[0].Serialize().get(), "1");
  EXPECT_STREQ(cast_params->input_strides[0].Serialize().get(), "1");
}

TEST(CastV2ApiCallTest, CastV2ApiCallTwoDimension) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Cast cast_op("cast");
  graph.AddNode(load_op);
  graph.AddNode(cast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  cast_op.x = load_op.y;
  *cast_op.y.axis = {z0.id, z1.id};
  *cast_op.y.repeats = {s0, s1};
  *cast_op.y.strides = {s1 + s1, One};

  auto load = graph.FindNode("load");
  auto size = af::GetSizeByDataType(af::DT_FLOAT16);
  auto stride = af::sym::Align(z1.size, 32 / size);
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {stride, One};
  load->outputs[0].attr.dtype = af::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  auto size1 = af::GetSizeByDataType(af::DT_FLOAT);
  auto stride1 = af::sym::Align(z1.size, 32 / size1);
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  cast->outputs[0].attr.vectorized_strides = {stride1, One};
  cast->outputs[0].attr.dtype = af::DT_FLOAT;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 1;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 2;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(cast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::CastV2ApiCall call("CastExtend");
  EXPECT_EQ(call.Init(cast), 0);
  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result,
            std::string{"CastExtend(local_1[0], local_0[0], {ConvertToUint32(t->s0), ConvertToUint32(t->s1)}, "
                        "{ConvertToUint32(((8 * Ceiling((Rational(1 , 8) * t->s1))))/(1)), ConvertToUint32(1)}, "
                        "{ConvertToUint32(((16 * Ceiling((Rational(1 , 16) * t->s1))))/(1)), ConvertToUint32(1)});\n"});

  const auto params = ascir_param::GetAscirNodeParams(cast);
  ASSERT_NE(params, nullptr);
  const auto *cast_params = std::get_if<ascir_param::CastNodeParams>(&params->specific_params);
  ASSERT_NE(cast_params, nullptr);
  ASSERT_EQ(cast_params->output_dims.size(), 2U);
  ASSERT_EQ(cast_params->output_strides.size(), 2U);
  ASSERT_EQ(cast_params->input_strides.size(), 2U);
  EXPECT_STREQ(cast_params->output_dims[0].Serialize().get(), "s0");
  EXPECT_STREQ(cast_params->output_dims[1].Serialize().get(), "s1");
  EXPECT_STREQ(cast_params->output_strides[0].Serialize().get(), "(8 * Ceiling((Rational(1 , 8) * s1)))");
  EXPECT_STREQ(cast_params->output_strides[1].Serialize().get(), "1");
  EXPECT_STREQ(cast_params->input_strides[0].Serialize().get(), "(16 * Ceiling((Rational(1 , 16) * s1)))");
  EXPECT_STREQ(cast_params->input_strides[1].Serialize().get(), "1");
}

TEST(CastV2ApiCallTest, CastV2ApiCallThreeDimension) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Cast cast_op("cast");
  graph.AddNode(load_op);
  graph.AddNode(cast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, s1, s2};
  *load_op.y.strides = {s1 * s2, s2, One};
  cast_op.x = load_op.y;
  *cast_op.y.axis = {z0.id, z1.id, z2.id};
  *cast_op.y.repeats = {s0, s1, s2};
  *cast_op.y.strides = {s1 * s2 + s1 * s2 + s1 * s2, s2 + s2, One};

  auto load = graph.FindNode("load");
  auto size = af::GetSizeByDataType(af::DT_FLOAT16);
  auto stride = af::sym::Align(z2.size, 32 / size);
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {stride * z1.size, stride, One};
  load->outputs[0].attr.dtype = af::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  auto size1 = af::GetSizeByDataType(af::DT_FLOAT);
  auto stride1 = af::sym::Align(z2.size, 32 / size1);
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  cast->outputs[0].attr.vectorized_strides = {stride1 * z1.size + stride1 * z1.size, stride1, One};
  cast->outputs[0].attr.dtype = af::DT_FLOAT;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 1;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 2;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(cast->outputs[0]);

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
  codegen::CastV2ApiCall call("CastExtend");
  EXPECT_EQ(call.Init(cast), 0);
  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result,
            std::string{
                "for(int outer_for_0 = 0; outer_for_0 < t->s0; outer_for_0++) {\nCastExtend(local_1[outer_for_0 * ((16 "
                "* Ceiling((Rational(1 , 8) * t->s2)) * t->s1))/(1)], local_0[outer_for_0 * ((16 * Ceiling((Rational(1 "
                ", 16) * t->s2)) * t->s1))/(1)], {ConvertToUint32(t->s1), ConvertToUint32(t->s2)}, "
                "{ConvertToUint32(((8 * Ceiling((Rational(1 , 8) * t->s2))))/(1)), ConvertToUint32(1)}, "
                "{ConvertToUint32(((16 * Ceiling((Rational(1 , 16) * t->s2))))/(1)), ConvertToUint32(1)});\n\n}\n"});

  const auto params = ascir_param::GetAscirNodeParams(cast);
  ASSERT_NE(params, nullptr);
  const auto *cast_params = std::get_if<ascir_param::CastNodeParams>(&params->specific_params);
  ASSERT_NE(cast_params, nullptr);
  ASSERT_EQ(cast_params->output_dims.size(), 2U);
  EXPECT_STREQ(cast_params->output_dims[0].Serialize().get(), "(s0 * s1)");
  EXPECT_STREQ(cast_params->output_dims[1].Serialize().get(), "s2");
}

TEST(CastV2ApiCallTest, CastV2ApiCall_Offset) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Cast cast_op("cast");
  graph.AddNode(load_op);
  graph.AddNode(cast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, s1, s2};
  *load_op.y.strides = {s1 * s2, s2, One};
  cast_op.x = load_op.y;
  *cast_op.y.axis = {z0.id, z1.id, z2.id};
  *cast_op.y.repeats = {s0, s1, s2};
  *cast_op.y.strides = {s1 * s2, s2, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;

  auto size = af::GetSizeByDataType(af::DT_FLOAT16);
  auto stride = af::sym::Align(z2.size, 32 / size);
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {stride, One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  cast->outputs[0].attr.vectorized_strides = {stride, One};
  cast->outputs[0].attr.dtype = af::DT_INT16;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 1;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 2;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(cast->outputs[0]);

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
  codegen::CastV2ApiCall call("CastExtend");
  EXPECT_EQ(call.Init(cast), 0);
  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result,
            std::string{"CastExtend(local_1[0], local_0[0], {ConvertToUint32(t->s1), ConvertToUint32(t->s2)}, "
                        "{ConvertToUint32(((16 * Ceiling((Rational(1 , 16) * t->s2))))/(1)), ConvertToUint32(1)}, "
                        "{ConvertToUint32(((16 * Ceiling((Rational(1 , 16) * t->s2))))/(1)), ConvertToUint32(1)});\n"});
}

TEST(CastV2ApiCallTest, CastV2ApiCallThreeDimensionNotSupportNormal) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Cast cast_op("cast");
  graph.AddNode(load_op);
  graph.AddNode(cast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, s1, s2};
  *load_op.y.strides = {s1 * s2, s2, One};
  cast_op.x = load_op.y;
  *cast_op.y.axis = {z0.id, z1.id, z2.id};
  *cast_op.y.repeats = {s0, s1, s2};
  *cast_op.y.strides = {s1 * s2 + s1 * s2 + s1 * s2, s2 + s2, One};

  auto load = graph.FindNode("load");
  auto size = af::GetSizeByDataType(af::DT_INT64);
  auto stride = af::sym::Align(z2.size, 32 / size);
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {stride * z1.size, stride, One};
  load->outputs[0].attr.dtype = af::DT_INT64;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  auto size1 = af::GetSizeByDataType(af::DT_UINT8);
  auto stride1 = af::sym::Align(z2.size, 32 / size1);
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  cast->outputs[0].attr.vectorized_strides = {stride1 * z1.size + stride1 * z1.size, stride1, One};
  cast->outputs[0].attr.dtype = af::DT_UINT8;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 1;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 2;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(cast->outputs[0]);

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
  codegen::CastV2ApiCall call("CastExtend");
  EXPECT_EQ(call.Init(cast), 0);
  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  std::cout << "=============search me============" << "\n";
  std::cout << result << "\n";
  EXPECT_EQ(result,
            std::string{
                "for(int outer_for_0 = 0; outer_for_0 < t->s0; outer_for_0++) {\nCastExtend(local_1[outer_for_0 * ((64 "
                "* Ceiling((Rational(1 , 32) * t->s2)) * t->s1))/(1)], local_0[outer_for_0 * ((4 * Ceiling((Rational(1 "
                ", 4) * t->s2)) * t->s1))/(1)], {ConvertToUint32(t->s1), ConvertToUint32(t->s2)}, "
                "{ConvertToUint32(((32 * Ceiling((Rational(1 , 32) * t->s2))))/(1)), ConvertToUint32(1)}, "
                "{ConvertToUint32(((4 * Ceiling((Rational(1 , 4) * t->s2))))/(1)), ConvertToUint32(1)});\n\n}\n"});
}

TEST(CastV2ApiCallTest, CastV2ApiCall_WithMaskMode) {
  std::vector<std::tuple<af::DataType, af::DataType, std::string>> x_y_max_size_list = {
      // x_dtype, y_dtype, max_dtype_size
      std::make_tuple(af::DT_UINT8, af::DT_FLOAT16, "2"),

      std::make_tuple(af::DT_INT64, af::DT_FLOAT, "8"),   std::make_tuple(af::DT_INT64, af::DT_INT32, "8"),

      std::make_tuple(af::DT_FLOAT16, af::DT_FLOAT, "4"), std::make_tuple(af::DT_FLOAT16, af::DT_INT32, "4"),
      std::make_tuple(af::DT_FLOAT16, af::DT_INT16, "2"), std::make_tuple(af::DT_FLOAT16, af::DT_INT8, "2"),
      std::make_tuple(af::DT_FLOAT16, af::DT_UINT8, "2"), std::make_tuple(af::DT_FLOAT16, af::DT_INT4, "2"),

      std::make_tuple(af::DT_FLOAT, af::DT_FLOAT16, "4"), std::make_tuple(af::DT_FLOAT, af::DT_INT64, "8"),
      std::make_tuple(af::DT_FLOAT, af::DT_INT32, "4"),   std::make_tuple(af::DT_FLOAT, af::DT_INT16, "4"),
      std::make_tuple(af::DT_FLOAT, af::DT_BF16, "4"),

      std::make_tuple(af::DT_INT4, af::DT_FLOAT16, "2"),

      std::make_tuple(af::DT_INT16, af::DT_FLOAT16, "2"), std::make_tuple(af::DT_INT16, af::DT_FLOAT, "4"),

      std::make_tuple(af::DT_INT32, af::DT_FLOAT, "4"),   std::make_tuple(af::DT_INT32, af::DT_INT64, "8"),
      std::make_tuple(af::DT_INT32, af::DT_INT16, "4"),   std::make_tuple(af::DT_INT32, af::DT_FLOAT16, "4"),

      std::make_tuple(af::DT_FLOAT16, af::DT_FLOAT, "4"), std::make_tuple(af::DT_FLOAT16, af::DT_INT32, "4"),
  };
  for (const auto &t : x_y_max_size_list) {
    af::AscGraph graph("test_graph");
    std::string max_size = std::get<2>(t);
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2);

    Data x_op("x", graph);
    Load load_op("load");
    af::ascir_op::Cast cast_op("cast");
    graph.AddNode(load_op);
    graph.AddNode(cast_op);

    load_op.x = x_op.y;
    load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
    *load_op.y.axis = {z0.id, z1.id, z2.id};
    *load_op.y.repeats = {s0, s1, s2};
    *load_op.y.strides = {s1 * s2, s2, One};
    cast_op.x = load_op.y;
    *cast_op.y.axis = {z0.id, z1.id, z2.id};
    *cast_op.y.repeats = {s0, s1, s2};
    *cast_op.y.strides = {s1 * s2, s2, One};

    auto load = graph.FindNode("load");
    load->attr.api.compute_type = af::ComputeType::kComputeLoad;
    load->attr.api.type = af::ApiType::kAPITypeCompute;
    load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
    load->attr.sched.loop_axis = z0.id;

    auto size = af::GetSizeByDataType(af::DT_FLOAT16);
    auto stride = af::sym::Align(z2.size, 32 / size);
    load->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
    load->outputs[0].attr.vectorized_strides = {stride, One};
    load->outputs[0].attr.dtype = std::get<0>(t);
    load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    load->outputs[0].attr.mem.tensor_id = 0;
    load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    load->outputs[0].attr.que.id = 1;
    load->outputs[0].attr.opt.merge_scope = af::kIdNone;

    auto cast = graph.FindNode("cast");
    cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
    cast->attr.api.type = af::ApiType::kAPITypeCompute;
    cast->attr.api.unit = af::ComputeUnit::kUnitVector;
    cast->attr.sched.loop_axis = z0.id;
    cast->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
    cast->outputs[0].attr.vectorized_strides = {stride, One};
    cast->outputs[0].attr.dtype = std::get<1>(t);
    cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
    cast->outputs[0].attr.mem.tensor_id = 1;
    cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    cast->outputs[0].attr.que.id = 2;
    cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

    codegen::Tiler tiler;
    codegen::TPipe tpipe("tpipe", tiler);

    tpipe.AddTensor(load->outputs[0]);
    tpipe.AddTensor(cast->outputs[0]);

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
    codegen::CastV2ApiCall call("Cast");
    EXPECT_EQ(call.Init(cast), 0);
    call.inputs.push_back(&x1);

    std::string result;
    call.Generate(tpipe, current_axis, result);
    std::string expect =
        "Cast(local_1[0], local_0[0], {ConvertToUint32(t->s1), ConvertToUint32(t->s2)}, {ConvertToUint32(((16 * "
        "Ceiling((Rational(1 , 16) * t->s2))))/(1)), ConvertToUint32(1)}, {ConvertToUint32(((16 * Ceiling((Rational(1 "
        ", 16) * t->s2))))/(1)), ConvertToUint32(1)});\n";
    EXPECT_EQ(result, expect);
  }
}
