/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <exception>
#include <filesystem>
#include "codegen.h"
#include "optimize.h"
#include "share_graph.h"
#include "backend_common.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascgraph_info_complete.h"
#include "ascgen_log.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

class TestBackendConv2DE2e : public testing::Test, public codegen::TilingLib {
 public:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }

 protected:
  TestBackendConv2DE2e() : codegen::TilingLib("test", "test") {}
};

namespace {

static void CreateElemwiseGraphWithAbs(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  af::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load0.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, af::ops::One};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = abs.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {s1, af::ops::One};
  *store_op.y.repeats = {s0, s1};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto x1Local = graph.FindNode("data0");
  x1Local->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
}

static void CreateElemwiseGraphWithMul(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  af::ascir_op::Mul mul("mul");
  graph.AddNode(mul);
  mul.x1 = load0.y;
  mul.x2 = load0.y;
  mul.attr.sched.axis = {z0.id, z1.id};
  mul.y.dtype = ge::DT_FLOAT16;
  *mul.y.axis = {z0.id, z1.id};
  *mul.y.repeats = {s0, s1};
  *mul.y.strides = {s1, af::ops::One};
  mul.attr.api.compute_type = af::ComputeType::kComputeElewise;

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = mul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {s1, af::ops::One};
  *store_op.y.repeats = {s0, s1};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto x1Local = graph.FindNode("data0");
  x1Local->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
}

static bool OptimizeAndGenerateCode(codegen::TilingLib &tiling_lib, af::AscGraph &graph, af::AscGraph &conv2d_graph,
                                    const std::string &tiling_func_file) {
  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    ascir::FusedScheduledResult fused_schedule_result;
    if (optimizer.Optimize(graph, fused_schedule_result) != 0) {
      return false;
    }

    ascir::ScheduleGroup schedule_group2;
    schedule_group2.impl_graphs.push_back(conv2d_graph);
    fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(schedule_group2);
    fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kUBFuse;

    const std::map<std::string, std::string> shape_info;
    auto res = tiling_lib.Generate(fused_schedule_result, shape_info, "", "0");

    std::fstream tiling_func(tiling_func_file.c_str(), std::ios::out);
    tiling_func << res["tiling_def_and_tiling_const"];

    auto pos = res["tiling_def_and_tiling_const"].find("extern \"C\" int64_t GenCVFusionTilingKey");
    return (pos != std::string::npos);
  } catch (...) {
    return false;
  }
}

static void CreateElemwiseGraphWithRelu(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  af::ascir_op::Relu relu("relu");
  graph.AddNode(relu);
  relu.x = load0.y;
  relu.attr.sched.axis = {z0.id, z1.id};
  relu.y.dtype = ge::DT_FLOAT16;
  *relu.y.axis = {z0.id, z1.id};
  *relu.y.repeats = {s0, s1};
  *relu.y.strides = {s1, af::ops::One};
  relu.attr.api.compute_type = af::ComputeType::kComputeElewise;

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = relu.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {s1, af::ops::One};
  *store_op.y.repeats = {s0, s1};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto x1Local = graph.FindNode("data0");
  x1Local->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
}

static void CreateElemwiseGraphWithMulScalar(af::AscGraph &graph, const std::string &scalar_value) {
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  af::ascir_op::Mul mul("mul");
  graph.AddNode(mul);
  mul.x1 = load0.y;
  af::ascir_op::Scalar scalar("scalar", graph);
  scalar.ir_attr.SetValue(scalar_value);
  scalar.y.dtype = ge::DT_FLOAT16;
  *scalar.y.axis = {z0.id, z1.id};
  *scalar.y.repeats = {af::ops::One, af::ops::One};
  *scalar.y.strides = {af::ops::Zero, af::ops::Zero};
  mul.x2 = scalar.y;
  mul.attr.sched.axis = {z0.id, z1.id};
  mul.y.dtype = ge::DT_FLOAT16;
  *mul.y.axis = {z0.id, z1.id};
  *mul.y.repeats = {s0, s1};
  *mul.y.strides = {s1, af::ops::One};
  mul.attr.api.compute_type = af::ComputeType::kComputeElewise;

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = mul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {s1, af::ops::One};
  *store_op.y.repeats = {s0, s1};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto x1Local = graph.FindNode("data0");
  x1Local->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
}

static void CreateElemwiseGraphWithAddScalar(af::AscGraph &graph, const std::string &scalar_value) {
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  af::ascir_op::Add add("add");
  graph.AddNode(add);
  add.x1 = load0.y;
  af::ascir_op::Scalar scalar("scalar", graph);
  scalar.ir_attr.SetValue(scalar_value);
  scalar.y.dtype = ge::DT_FLOAT16;
  *scalar.y.axis = {z0.id, z1.id};
  *scalar.y.repeats = {af::ops::One, af::ops::One};
  *scalar.y.strides = {af::ops::Zero, af::ops::Zero};
  add.x2 = scalar.y;
  add.attr.sched.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, af::ops::One};
  add.attr.api.compute_type = af::ComputeType::kComputeElewise;

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = add.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {s1, af::ops::One};
  *store_op.y.repeats = {s0, s1};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto x1Local = graph.FindNode("data0");
  x1Local->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
}

}  // namespace

struct ConvGraphAxes {
  af::Expression s0;
  af::Expression s1;
  af::AxisId z0;
  af::AxisId z1;
};

struct ConvInputConfig {
  int64_t index;
  ge::DataType dtype;
  bool use_output_shape;
};

ConvGraphAxes CreateConvGraphAxes(af::AscGraph &graph) {
  const auto s0 = graph.CreateSizeVar(32);
  const auto s1 = graph.CreateSizeVar(32);
  return {s0, s1, graph.CreateAxis("z0", s0).id, graph.CreateAxis("z1", s1).id};
}

void ConfigureConvInput(af::ascir_op::Data &data, af::ascir_op::Load &load, const ConvGraphAxes &axes,
                        const ConvInputConfig &config) {
  data.attr.sched.axis = {axes.z0, axes.z1};
  data.y.dtype = config.dtype;
  *data.y.axis = {axes.z0, axes.z1};
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.ir_attr.SetIndex(config.index);
  load.attr.sched.axis = {axes.z0, axes.z1};
  load.x = data.y;
  load.y.dtype = config.dtype;
  *load.y.axis = {axes.z0, axes.z1};
  if (config.use_output_shape) {
    *data.y.strides = {axes.s1, af::ops::One};
    *data.y.repeats = {axes.s0, axes.s1};
    *load.y.strides = {axes.s1, af::ops::One};
    *load.y.repeats = {axes.s0, axes.s1};
  } else {
    *data.y.strides = {af::ops::Zero, af::ops::Zero};
    *data.y.repeats = {af::ops::One, af::ops::One};
    *load.y.strides = {af::ops::Zero, af::ops::Zero};
    *load.y.repeats = {af::ops::One, af::ops::One};
  }
}

template <typename ConvOp>
void ConfigureExtendConv2D(ConvOp &conv2d, const ConvGraphAxes &axes, bool enable_relu0) {
  conv2d.attr.sched.axis = {axes.z0, axes.z1};
  conv2d.y.dtype = ge::DT_FLOAT16;
  *conv2d.y.axis = {axes.z0, axes.z1};
  *conv2d.y.repeats = {axes.s0, axes.s1};
  *conv2d.y.strides = {axes.s1, af::ops::One};
  conv2d.ir_attr.SetStrides({1, 1});
  conv2d.ir_attr.SetPads({0, 0, 0, 0});
  conv2d.ir_attr.SetDilations({1, 1});
  conv2d.ir_attr.SetGroups(1);
  conv2d.ir_attr.SetData_format("NCHW");
  conv2d.ir_attr.SetOffset_x(0);
  conv2d.ir_attr.SetPad_mode("SPECIFIC");
  conv2d.ir_attr.SetRound_mode("rint");
  conv2d.ir_attr.SetEnable_hf32(false);
  conv2d.ir_attr.SetEnable_relu0(enable_relu0);
}

template <typename Tensor>
void FinishConvGraph(af::AscGraph &graph, Tensor &conv_output, const ConvGraphAxes &axes) {
  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {axes.z0, axes.z1};
  store_op.x = conv_output;
  *store_op.y.axis = {axes.z0, axes.z1};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {axes.s1, af::ops::One};
  *store_op.y.repeats = {axes.s0, axes.s1};
  store_op.ir_attr.SetOffset(af::ops::One);
  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
}

void CreateConv2DGraph(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(32);
  auto s1 = graph.CreateSizeVar(32);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {af::ops::One, af::ops::One};
  *data1.y.strides = {af::ops::Zero, af::ops::Zero};
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {af::ops::Zero, af::ops::Zero};
  *load1.y.repeats = {af::ops::One, af::ops::One};

  af::ascir_op::Conv2D conv2d("conv2d");
  conv2d.attr.sched.axis = {z0.id, z1.id};
  conv2d.x = load0.y;
  conv2d.filter = load1.y;
  conv2d.y.dtype = ge::DT_FLOAT16;
  *conv2d.y.axis = {z0.id, z1.id};
  *conv2d.y.repeats = {s0, s1};
  *conv2d.y.strides = {s1, af::ops::One};
  conv2d.ir_attr.SetStrides({1, 1});
  conv2d.ir_attr.SetPads({0, 0, 0, 0});
  conv2d.ir_attr.SetDilations({1, 1});
  conv2d.ir_attr.SetGroups(1);
  conv2d.ir_attr.SetData_format("NCHW");
  conv2d.ir_attr.SetOffset_x(0);
  conv2d.ir_attr.SetPad_mode("SPECIFIC");
  conv2d.ir_attr.SetEnable_hf32(false);
  FinishConvGraph(graph, conv2d.y, {s0, s1, z0.id, z1.id});
}

void CreateConv2DBiasGraph(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(32);
  auto s1 = graph.CreateSizeVar(32);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {af::ops::One, af::ops::One};
  *data1.y.strides = {af::ops::Zero, af::ops::Zero};
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {af::ops::Zero, af::ops::Zero};
  *load1.y.repeats = {af::ops::One, af::ops::One};

  af::ascir_op::Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT16;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data2.y.repeats = {af::ops::One, af::ops::One};
  *data2.y.strides = {af::ops::Zero, af::ops::Zero};
  data2.ir_attr.SetIndex(2);

  af::ascir_op::Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.strides = {af::ops::Zero, af::ops::Zero};
  *load2.y.repeats = {af::ops::One, af::ops::One};

  af::ascir_op::Conv2DBias conv2d_bias("conv2d_bias");
  conv2d_bias.attr.sched.axis = {z0.id, z1.id};
  conv2d_bias.x = load0.y;
  conv2d_bias.filter = load1.y;
  conv2d_bias.bias = load2.y;
  conv2d_bias.y.dtype = ge::DT_FLOAT16;
  *conv2d_bias.y.axis = {z0.id, z1.id};
  *conv2d_bias.y.repeats = {s0, s1};
  *conv2d_bias.y.strides = {s1, af::ops::One};
  conv2d_bias.ir_attr.SetStrides({1, 1});
  conv2d_bias.ir_attr.SetPads({0, 0, 0, 0});
  conv2d_bias.ir_attr.SetDilations({1, 1});
  conv2d_bias.ir_attr.SetGroups(1);
  conv2d_bias.ir_attr.SetData_format("NCHW");
  conv2d_bias.ir_attr.SetOffset_x(0);
  conv2d_bias.ir_attr.SetPad_mode("SPECIFIC");
  conv2d_bias.ir_attr.SetEnable_hf32(false);
  FinishConvGraph(graph, conv2d_bias.y, {s0, s1, z0.id, z1.id});
}

void CreateConv2DOffsetGraph(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(32);
  auto s1 = graph.CreateSizeVar(32);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {af::ops::One, af::ops::One};
  *data1.y.strides = {af::ops::Zero, af::ops::Zero};
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {af::ops::Zero, af::ops::Zero};
  *load1.y.repeats = {af::ops::One, af::ops::One};

  af::ascir_op::Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT16;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data2.y.repeats = {af::ops::One, af::ops::One};
  *data2.y.strides = {af::ops::Zero, af::ops::Zero};
  data2.ir_attr.SetIndex(2);

  af::ascir_op::Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.strides = {af::ops::Zero, af::ops::Zero};
  *load2.y.repeats = {af::ops::One, af::ops::One};

  af::ascir_op::Conv2DOffset conv2d_offset("conv2d_offset");
  conv2d_offset.attr.sched.axis = {z0.id, z1.id};
  conv2d_offset.x = load0.y;
  conv2d_offset.filter = load1.y;
  conv2d_offset.offset_w = load2.y;
  conv2d_offset.y.dtype = ge::DT_FLOAT16;
  *conv2d_offset.y.axis = {z0.id, z1.id};
  *conv2d_offset.y.repeats = {s0, s1};
  *conv2d_offset.y.strides = {s1, af::ops::One};
  conv2d_offset.ir_attr.SetStrides({1, 1});
  conv2d_offset.ir_attr.SetPads({0, 0, 0, 0});
  conv2d_offset.ir_attr.SetDilations({1, 1});
  conv2d_offset.ir_attr.SetGroups(1);
  conv2d_offset.ir_attr.SetData_format("NCHW");
  conv2d_offset.ir_attr.SetOffset_x(0);
  conv2d_offset.ir_attr.SetPad_mode("SPECIFIC");
  conv2d_offset.ir_attr.SetEnable_hf32(false);
  FinishConvGraph(graph, conv2d_offset.y, {s0, s1, z0.id, z1.id});
}

void CreateConv2DOffsetBiasGraph(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(32);
  auto s1 = graph.CreateSizeVar(32);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {af::ops::One, af::ops::One};
  *data1.y.strides = {af::ops::Zero, af::ops::Zero};
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {af::ops::Zero, af::ops::Zero};
  *load1.y.repeats = {af::ops::One, af::ops::One};

  af::ascir_op::Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT16;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data2.y.repeats = {af::ops::One, af::ops::One};
  *data2.y.strides = {af::ops::Zero, af::ops::Zero};
  data2.ir_attr.SetIndex(2);

  af::ascir_op::Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.strides = {af::ops::Zero, af::ops::Zero};
  *load2.y.repeats = {af::ops::One, af::ops::One};

  af::ascir_op::Data data3("data3", graph);
  data3.y.dtype = ge::DT_FLOAT16;
  data3.attr.sched.axis = {z0.id, z1.id};
  *data3.y.axis = {z0.id, z1.id};
  data3.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data3.y.repeats = {af::ops::One, af::ops::One};
  *data3.y.strides = {af::ops::Zero, af::ops::Zero};
  data3.ir_attr.SetIndex(3);

  af::ascir_op::Load load3("load3");
  load3.x = data3.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = ge::DT_FLOAT16;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.strides = {af::ops::Zero, af::ops::Zero};
  *load3.y.repeats = {af::ops::One, af::ops::One};

  af::ascir_op::Conv2DOffsetBias conv2d_offset_bias("conv2d_offset_bias");
  conv2d_offset_bias.attr.sched.axis = {z0.id, z1.id};
  conv2d_offset_bias.x = load0.y;
  conv2d_offset_bias.filter = load1.y;
  conv2d_offset_bias.bias = load2.y;
  conv2d_offset_bias.offset_w = load3.y;
  conv2d_offset_bias.y.dtype = ge::DT_FLOAT16;
  *conv2d_offset_bias.y.axis = {z0.id, z1.id};
  *conv2d_offset_bias.y.repeats = {s0, s1};
  *conv2d_offset_bias.y.strides = {s1, af::ops::One};
  conv2d_offset_bias.ir_attr.SetStrides({1, 1});
  conv2d_offset_bias.ir_attr.SetPads({0, 0, 0, 0});
  conv2d_offset_bias.ir_attr.SetDilations({1, 1});
  conv2d_offset_bias.ir_attr.SetGroups(1);
  conv2d_offset_bias.ir_attr.SetData_format("NCHW");
  conv2d_offset_bias.ir_attr.SetOffset_x(0);
  conv2d_offset_bias.ir_attr.SetPad_mode("SPECIFIC");
  conv2d_offset_bias.ir_attr.SetEnable_hf32(false);
  FinishConvGraph(graph, conv2d_offset_bias.y, {s0, s1, z0.id, z1.id});
}

TEST_F(TestBackendConv2DE2e, Conv2DE2eCodegen) {
  af::AscGraph graph("conv2d_elemwise_pro");
  CreateElemwiseGraphWithAbs(graph);

  af::AscGraph conv2d_graph("conv2d");
  CreateConv2DGraph(conv2d_graph);

  bool gen_success = OptimizeAndGenerateCode(*this, graph, conv2d_graph, "Conv2d_fuse_tiling_func.cpp");
  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendConv2DE2e, Conv2DBiasE2eCodegen) {
  af::AscGraph graph("conv2d_bias_elemwise_pro");
  CreateElemwiseGraphWithRelu(graph);

  af::AscGraph conv2d_bias_graph("conv2d_bias");
  CreateConv2DBiasGraph(conv2d_bias_graph);

  bool gen_success = OptimizeAndGenerateCode(*this, graph, conv2d_bias_graph, "Conv2dBias_fuse_tiling_func.cpp");

  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::fstream tiling_data_file(parts[0], std::ios::out);
  tiling_data_file << "";

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendConv2DE2e, Conv2DOffsetE2eCodegen) {
  af::AscGraph graph("conv2d_offset_elemwise_pro");
  CreateElemwiseGraphWithMulScalar(graph, "2.0");

  af::AscGraph conv2d_offset_graph("conv2d_offset");
  CreateConv2DOffsetGraph(conv2d_offset_graph);

  bool gen_success = OptimizeAndGenerateCode(*this, graph, conv2d_offset_graph, "Conv2dOffset_fuse_tiling_func.cpp");

  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::fstream tiling_data_file(parts[0], std::ios::out);
  tiling_data_file << "";

  EXPECT_EQ(gen_success, true);
}

void CreateExtendConv2DGraph(af::AscGraph &graph, bool with_bias_scale = false) {
  const auto axes = CreateConvGraphAxes(graph);
  af::ascir_op::Data data0("data0", graph);
  af::ascir_op::Load load0("load0");
  ConfigureConvInput(data0, load0, axes, {0, ge::DT_FLOAT16, true});
  af::ascir_op::Data data1("data1", graph);
  af::ascir_op::Load load1("load1");
  ConfigureConvInput(data1, load1, axes, {1, ge::DT_FLOAT16, false});
  if (!with_bias_scale) {
    af::ascir_op::ExtendConv2D conv2d("extend_conv2d");
    conv2d.x = load0.y;
    conv2d.filter = load1.y;
    ConfigureExtendConv2D(conv2d, axes, false);
    FinishConvGraph(graph, conv2d.y, axes);
    return;
  }

  af::ascir_op::Data data2("data2", graph);
  af::ascir_op::Load load2("load2");
  ConfigureConvInput(data2, load2, axes, {2, ge::DT_FLOAT16, false});
  af::ascir_op::Data data3("data3", graph);
  af::ascir_op::Load load3("load3");
  ConfigureConvInput(data3, load3, axes, {3, ge::DT_UINT64, false});
  af::ascir_op::ExtendConv2DBiasScale conv2d("extend_conv2d_bias_scale");
  conv2d.x = load0.y;
  conv2d.filter = load1.y;
  conv2d.bias = load2.y;
  conv2d.scale0 = load3.y;
  ConfigureExtendConv2D(conv2d, axes, true);
  FinishConvGraph(graph, conv2d.y, axes);
}

TEST_F(TestBackendConv2DE2e, Conv2DOffsetBiasE2eCodegen) {
  af::AscGraph graph("conv2d_offset_bias_elemwise_pro");
  CreateElemwiseGraphWithAddScalar(graph, "1.0");

  af::AscGraph conv2d_offset_bias_graph("conv2d_offset_bias");
  CreateConv2DOffsetBiasGraph(conv2d_offset_bias_graph);

  bool gen_success =
      OptimizeAndGenerateCode(*this, graph, conv2d_offset_bias_graph, "Conv2dOffsetBias_fuse_tiling_func.cpp");

  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::fstream tiling_data_file(parts[0], std::ios::out);
  tiling_data_file << "";

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendConv2DE2e, ExtendConv2DE2eCodegen) {
  af::AscGraph graph("extend_conv2d_elemwise_pro");
  CreateElemwiseGraphWithAbs(graph);

  af::AscGraph conv2d_graph("extend_conv2d");
  CreateExtendConv2DGraph(conv2d_graph);

  bool gen_success = OptimizeAndGenerateCode(*this, graph, conv2d_graph, "ExtendConv2d_fuse_tiling_func.cpp");
  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendConv2DE2e, ExtendConv2DBiasScaleE2eCodegen) {
  af::AscGraph graph("extend_conv2d_bias_scale_elemwise_pro");
  CreateElemwiseGraphWithRelu(graph);

  af::AscGraph conv2d_graph("extend_conv2d_bias_scale");
  CreateExtendConv2DGraph(conv2d_graph, true);

  bool gen_success = OptimizeAndGenerateCode(*this, graph, conv2d_graph, "ExtendConv2dBiasScale_fuse_tiling_func.cpp");
  EXPECT_EQ(gen_success, true);
}
