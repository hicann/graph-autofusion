/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ascendc_ir.h>
#include <ascir_ops.h>
#include <ascir_utils.h>
#include <iostream>

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "graph_utils_ex.h"

#define private public
#include "autoschedule/autoschedule.h"
#include "optimize.h"
#undef private
#include "ascir_ops_utils.h"
#include "autoschedule/tiling_group.h"
#include "schedule_utils.h"
#include "ascir_utils.h"
#include "platform_context.h"
#include "platform/v1/platformv1.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace optimize::autoschedule;

struct FourAxisLoadContext {
  ascir::SizeExpr s0;
  ascir::SizeExpr s1;
  ascir::SizeExpr s2;
  ascir::SizeExpr s3;
  std::vector<af::AxisId> axes;
};

FourAxisLoadContext InitFourAxisLoad(af::AscGraph &graph, Load &load) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  std::vector<af::AxisId> axes = {z0.id, z1.id, z2.id, z3.id};

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = af::ApiType::kAPITypeBuffer;
  arg4_1.y.dtype = ge::DT_FLOAT;

  load.x = arg4_1.y;
  load.attr.sched.axis = axes;
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.attr.api.type = af::ApiType::kAPITypeCompute;
  load.y.dtype = ge::DT_FLOAT;
  *load.y.axis = axes;
  *load.y.repeats = {s0, s1, s2, s3};
  *load.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};
  return {s0, s1, s2, s3, axes};
}

void Construct_Reduce_RARA(af::AscGraph &graph) {
  Load b0_load("b0_load");
  auto input = InitFourAxisLoad(graph, b0_load);

  Abs abs("abs");
  abs.x = b0_load.y;
  abs.attr.sched.axis = input.axes;
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.attr.api.type = af::ApiType::kAPITypeCompute;
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = input.axes;
  *abs.y.repeats = {input.s0, input.s1, input.s2, input.s3};
  *abs.y.strides = {input.s1 * input.s2 * input.s3, input.s2 * input.s3, input.s3, One};

  af::ascir_op::Max b0_max("b0_max");
  b0_max.x = abs.y;
  b0_max.attr.sched.axis = input.axes;
  b0_max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  b0_max.attr.api.type = af::ApiType::kAPITypeCompute;
  b0_max.y.dtype = ge::DT_FLOAT;
  *b0_max.y.axis = input.axes;
  *b0_max.y.repeats = {One, input.s1, One, input.s3};
  *b0_max.y.strides = {Zero, input.s3, Zero, One};

  Abs abs1("abs1");
  abs1.x = b0_max.y;
  abs1.attr.sched.axis = input.axes;
  abs1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs1.attr.api.type = af::ApiType::kAPITypeCompute;
  abs1.y.dtype = ge::DT_FLOAT;
  *abs1.y.axis = input.axes;
  *abs1.y.repeats = {One, input.s1, One, input.s3};
  *abs1.y.strides = {Zero, input.s3, Zero, One};

  Store b3_store("b3_store");
  b3_store.x = abs1.y;
  b3_store.attr.sched.axis = input.axes;
  b3_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  b3_store.attr.api.type = af::ApiType::kAPITypeCompute;
  b3_store.y.dtype = ge::DT_FLOAT;
  *b3_store.y.axis = input.axes;
  *b3_store.y.repeats = {One, input.s1, One, input.s3};
  *b3_store.y.strides = {Zero, input.s3, Zero, One};

  Output buf3("buf3");
  buf3.x = b3_store.y;
  buf3.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  buf3.attr.api.type = af::ApiType::kAPITypeBuffer;
  buf3.y.dtype = ge::DT_FLOAT;
}

void Construct_Reduce_ARAR(af::AscGraph &graph) {
  Load b0_load("b0_load");
  auto input = InitFourAxisLoad(graph, b0_load);

  Abs abs("abs");
  abs.x = b0_load.y;
  abs.attr.sched.axis = input.axes;
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.attr.api.type = af::ApiType::kAPITypeCompute;
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = input.axes;
  *abs.y.repeats = {input.s0, input.s1, input.s2, input.s3};
  *abs.y.strides = {input.s1 * input.s2 * input.s3, input.s2 * input.s3, input.s3, One};

  af::ascir_op::Max b0_max("b0_max");
  b0_max.x = abs.y;
  b0_max.attr.sched.axis = input.axes;
  b0_max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  b0_max.attr.api.type = af::ApiType::kAPITypeCompute;
  b0_max.y.dtype = ge::DT_FLOAT;
  *b0_max.y.axis = input.axes;
  *b0_max.y.repeats = {input.s0, One, input.s2, One};
  *b0_max.y.strides = {input.s2, Zero, One, Zero};

  Abs abs1("abs1");
  abs1.x = b0_max.y;
  abs1.attr.sched.axis = input.axes;
  abs1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs1.attr.api.type = af::ApiType::kAPITypeCompute;
  abs1.y.dtype = ge::DT_FLOAT;
  *abs1.y.axis = input.axes;
  *abs1.y.repeats = {input.s0, One, input.s2, One};
  *abs1.y.strides = {input.s2, Zero, One, Zero};

  Store b3_store("b3_store");
  b3_store.x = abs1.y;
  b3_store.attr.sched.axis = input.axes;
  b3_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  b3_store.attr.api.type = af::ApiType::kAPITypeCompute;
  b3_store.y.dtype = ge::DT_FLOAT;
  *b3_store.y.axis = input.axes;
  *b3_store.y.repeats = {input.s0, One, input.s2, One};
  *b3_store.y.strides = {input.s2, Zero, One, Zero};

  Output buf3("buf3");
  buf3.x = b3_store.y;
  buf3.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  buf3.attr.api.type = af::ApiType::kAPITypeBuffer;
  buf3.y.dtype = ge::DT_FLOAT;
}

void Construct_Reduce_AAAR(af::AscGraph &graph) {
  Load b0_load("b0_load");
  auto input = InitFourAxisLoad(graph, b0_load);

  Max b0_max("b0_max");
  b0_max.x = b0_load.y;
  b0_max.attr.sched.axis = input.axes;
  b0_max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  b0_max.attr.api.type = af::ApiType::kAPITypeCompute;
  b0_max.y.dtype = ge::DT_FLOAT;
  *b0_max.y.axis = input.axes;
  *b0_max.y.repeats = {input.s0, input.s1, input.s2, One};
  *b0_max.y.strides = {input.s1 * input.s2, input.s2, One, Zero};

  Store b3_store("b3_store");
  b3_store.x = b0_max.y;
  b3_store.attr.sched.axis = input.axes;
  b3_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  b3_store.attr.api.type = af::ApiType::kAPITypeCompute;
  b3_store.y.dtype = ge::DT_FLOAT;
  *b3_store.y.axis = input.axes;
  *b3_store.y.repeats = {input.s0, input.s1, input.s2, One};
  *b3_store.y.strides = {input.s1 * input.s2, input.s2, One, Zero};

  Output buf3("buf3");
  buf3.x = b3_store.y;
  buf3.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  buf3.attr.api.type = af::ApiType::kAPITypeBuffer;
  buf3.y.dtype = ge::DT_FLOAT;
}

namespace optimize {
class AutoSchedulerReduceUT : public ::testing::Test {
  void SetUp() override {}

 protected:
  optimize::Optimizer optimizer_;
  AutoSchedulerReduceUT() : optimizer_(optimize::OptimizerOptions{}) {};
};

TEST_F(AutoSchedulerReduceUT, Autoschedule_reduce_rara_axesgroup) {
  af::AscGraph graph("Reduce_RARA");
  Construct_Reduce_RARA(graph);

  af::AscGraph except_graph("Reduce_RARA_1");
  except_graph.CopyFrom(graph);

  auto store = graph.FindNode("b3_store");
  std::vector<af::AxisId> axes = store->attr.sched.axis;
  std::vector<af::AxisId> y_group = {axes[1], axes[3]};
  std::vector<af::AxisId> r_group = {axes[0], axes[2]};

  AxisGroup axis_group;
  ASSERT_EQ(TilingGroup::GenTilingGroup(graph, axis_group), 0);
  EXPECT_EQ(axis_group.x_group.size(), 0);
  EXPECT_EQ(axis_group.y_group.size(), 2);
  EXPECT_EQ(axis_group.y_group, y_group);
  EXPECT_EQ(axis_group.r_group.size(), 2);
  EXPECT_EQ(axis_group.r_group, r_group);
}

TEST_F(AutoSchedulerReduceUT, Autoschedule_reduce_aaar_fullload_axesgroup) {
  af::AscGraph graph("Reduce_AAAR");
  Construct_Reduce_AAAR(graph);

  auto store = graph.FindNode("b3_store");
  std::vector<af::AxisId> axes = store->attr.sched.axis;
  std::vector<af::AxisId> y_group = {axes[0], axes[1], axes[2]};
  std::vector<af::AxisId> n_group = {axes[3]};

  AxisGroup axis_group;
  ASSERT_EQ(TilingGroup::GenTilingGroup(graph, axis_group, true), 0);
  EXPECT_EQ(axis_group.x_group.size(), 0);
  EXPECT_EQ(axis_group.y_group, y_group);
  EXPECT_EQ(axis_group.r_group.size(), 0);
  EXPECT_EQ(axis_group.n_group, n_group);
}

TEST_F(AutoSchedulerReduceUT, Autoschedule_reduce_rara_tilingcase) {
  af::AscGraph graph("Reduce_RARA");
  Construct_Reduce_RARA(graph);

  af::AscGraph except_graph("Reduce_RARA_1");
  except_graph.CopyFrom(graph);

  auto store = graph.FindNode("b3_store");
  std::vector<af::AxisId> axes = store->attr.sched.axis;

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  EXPECT_EQ(TilingGroup::GenTilingGroup(graph, autoschedule.axes_group_), 0);
  TilingGroup::NormGroup(autoschedule.axes_group_);
  std::vector<TilingCase> tiling_cases;
  autoschedule.GenTilingCase(tiling_cases);
  ASSERT_EQ(tiling_cases.size(), 4);
  EXPECT_EQ(tiling_cases[0].ub_tiling_id_x, -1);
  EXPECT_EQ(tiling_cases[0].ub_tiling_id_y, axes[1]);
  EXPECT_EQ(tiling_cases[0].ub_tiling_id_r, axes[0]);
  EXPECT_EQ(tiling_cases[0].block_tiling_id, 0);
  EXPECT_EQ(tiling_cases[0].reduce_is_block, false);
}

TEST_F(AutoSchedulerReduceUT, Autoschedule_reduce_rara_fusion) {
  af::AscGraph graph("Reduce_RARA");
  Construct_Reduce_RARA(graph);

  af::AscGraph except_graph("Reduce_RARA_1");
  except_graph.CopyFrom(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 4);
}

TEST_F(AutoSchedulerReduceUT, Autoschedule_reduce_arar_axesgroup) {
  af::AscGraph graph("Reduce_ARAR");
  Construct_Reduce_ARAR(graph);

  af::AscGraph except_graph("Reduce_ARAR_1");
  except_graph.CopyFrom(graph);

  auto store = graph.FindNode("b3_store");
  std::vector<af::AxisId> axes = store->attr.sched.axis;
  std::vector<af::AxisId> y_group = {axes[0], axes[2]};
  std::vector<af::AxisId> r_group = {axes[1], axes[3]};

  AxisGroup axis_group;
  ASSERT_EQ(TilingGroup::GenTilingGroup(graph, axis_group), 0);
  EXPECT_EQ(axis_group.x_group.size(), 0);
  EXPECT_EQ(axis_group.y_group.size(), 2);
  EXPECT_EQ(axis_group.y_group, y_group);
  EXPECT_EQ(axis_group.r_group.size(), 2);
  EXPECT_EQ(axis_group.r_group, r_group);
}

TEST_F(AutoSchedulerReduceUT, Autoschedule_reduce_arar_tilingcase) {
  af::AscGraph graph("Reduce_ARAR");
  Construct_Reduce_ARAR(graph);

  af::AscGraph except_graph("Reduce_ARAR_1");
  except_graph.CopyFrom(graph);

  auto store = graph.FindNode("b3_store");
  std::vector<af::AxisId> axes = store->attr.sched.axis;

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  EXPECT_EQ(TilingGroup::GenTilingGroup(graph, autoschedule.axes_group_), 0);
  TilingGroup::NormGroup(autoschedule.axes_group_);
  std::vector<TilingCase> tiling_cases;
  autoschedule.GenTilingCase(tiling_cases);
  ASSERT_EQ(tiling_cases.size(), 4);
  EXPECT_EQ(tiling_cases[0].ub_tiling_id_x, -1);
  EXPECT_EQ(tiling_cases[0].ub_tiling_id_y, axes[0]);
  EXPECT_EQ(tiling_cases[0].ub_tiling_id_r, axes[1]);
  EXPECT_EQ(tiling_cases[0].block_tiling_id, 0);
  EXPECT_EQ(tiling_cases[0].reduce_is_block, false);
}

TEST_F(AutoSchedulerReduceUT, Autoschedule_reduce_arar_fusion) {
  af::AscGraph graph("Reduce_ARAR");
  Construct_Reduce_ARAR(graph);

  af::AscGraph except_graph("Reduce_ARAR_1");
  except_graph.CopyFrom(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 4);
}

TEST_F(AutoSchedulerReduceUT, Autoschedule_reduce_arar_fusion_rcore) {
  af::AscGraph graph("Reduce_ARAR");
  Construct_Reduce_ARAR(graph);

  af::AscGraph except_graph("Reduce_ARAR_1");
  except_graph.CopyFrom(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs, true);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 4);
}

TEST_F(AutoSchedulerReduceUT, reduceSorting) {
  af::AscGraph graph("Reduce_ARAR");

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = af::ApiType::kAPITypeBuffer;

  Load b0_load("b0_load");
  b0_load.x = arg4_1.y;
  b0_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  b0_load.attr.api.type = af::ApiType::kAPITypeCompute;

  Abs abs("abs");
  abs.x = b0_load.y;
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Max max("max");
  max.x = abs.y;
  max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  max.attr.api.type = af::ApiType::kAPITypeCompute;
  max.y.dtype = ge::DT_FLOAT;

  Store b3_store("b3_store");
  b3_store.x = max.y;
  b3_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  b3_store.attr.api.type = af::ApiType::kAPITypeCompute;
  b3_store.y.dtype = ge::DT_FLOAT;

  Output buf3("buf3");
  buf3.x = b3_store.y;
  buf3.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  buf3.attr.api.type = af::ApiType::kAPITypeBuffer;
  buf3.y.dtype = ge::DT_FLOAT;

  Data data2("data2", graph);
  data2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data2.attr.api.type = af::ApiType::kAPITypeBuffer;

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2.attr.api.type = af::ApiType::kAPITypeCompute;

  Add add("add");
  add.x1 = b0_load.y;
  add.x2 = load2.y;
  add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add.attr.api.type = af::ApiType::kAPITypeCompute;

  af::ascir_op::Max max2("max2");
  max2.x = add.y;
  max2.attr.api.compute_type = af::ComputeType::kComputeReduce;
  max2.attr.api.type = af::ApiType::kAPITypeCompute;
  max2.y.dtype = ge::DT_FLOAT;

  Store store2("store2");
  store2.x = max2.y;
  store2.attr.api.compute_type = af::ComputeType::kComputeStore;
  store2.attr.api.type = af::ApiType::kAPITypeCompute;
  store2.y.dtype = ge::DT_FLOAT;

  Output out2("out2");
  out2.x = store2.y;
  out2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  out2.attr.api.type = af::ApiType::kAPITypeBuffer;
  out2.y.dtype = ge::DT_FLOAT;

  ASSERT_EQ(ScheduleUtils::TopologicalSorting(graph), af::SUCCESS);

  auto max_node = graph.FindNode("max");
  auto max2_node = graph.FindNode("max2");
  auto add_node = graph.FindNode("add");
  auto abs_node = graph.FindNode("abs");

  EXPECT_TRUE(max_node->GetOpDescBarePtr()->GetId() > abs_node->GetOpDescBarePtr()->GetId());
  EXPECT_TRUE(max_node->GetOpDescBarePtr()->GetId() > add_node->GetOpDescBarePtr()->GetId());
  EXPECT_TRUE(max2_node->GetOpDescBarePtr()->GetId() > abs_node->GetOpDescBarePtr()->GetId());
  EXPECT_TRUE(max2_node->GetOpDescBarePtr()->GetId() > add_node->GetOpDescBarePtr()->GetId());
}

}  // namespace optimize
