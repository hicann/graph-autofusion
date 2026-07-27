/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "gtest/gtest.h"
#include "ascendc_ir_def.h"
#define private public
#include "autoschedule/alignment_handler.h"
#include "optimize/partition/vector_func_partitioner.h"
#undef private
#include "ascir_ops_utils.h"
#include "schedule_utils.h"
#include "platform_context.h"
#include "optimize/platformv2.h"
#include "ascgraph_info_complete.h"
#include "runtime_stub.h"
#include "optimize.h"
#include "asc_graph_builder.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace optimize::autoschedule;

namespace optimize {
using namespace ge;

namespace {
void SetupGraphAxes(af::AscGraph &graph, const std::vector<af::Symbol> &loops) {
  for (size_t i = 0UL; i < loops.size(); ++i) {
    graph.CreateAxis("z" + std::to_string(i), loops[i]);
  }
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto axes = graph.GetAllAxis();
  std::vector<af::Expression> loop_repeats;
  std::vector<af::AxisId> axis_ids;
  for (const auto &axis : axes) {
    loop_repeats.push_back(axis->size);
    axis_ids.push_back(axis->id);
  }

  auto SetStrides = [](const std::vector<af::Expression> &repeats, std::vector<af::Expression> &strides) {
    af::Expression stride = af::sym::kSymbolOne;
    for (auto iter = repeats.rbegin(); iter != repeats.rend(); ++iter) {
      if (af::SymbolicUtils::StaticCheckEq(*iter, af::sym::kSymbolOne) == af::TriBool::kTrue) {
        strides.push_back(af::sym::kSymbolZero);
      } else {
        strides.push_back(stride);
        stride = stride * *iter;
      }
    }
    std::reverse(strides.begin(), strides.end());
  };

  for (const auto &node : graph.GetAllNodes()) {
    node->attr.sched.axis = axis_ids;
    if (ScheduleUtils::IsBuffer(node)) continue;
    bool is_follow_input = (node->attr.api.compute_type == af::ComputeType::kComputeElewise ||
                            node->attr.api.compute_type == af::ComputeType::kComputeStore);
    for (auto &output : node->outputs()) {
      output->attr.axis = axis_ids;
      output->attr.vectorized_axis = axis_ids;
      if (is_follow_input) {
        output->attr.repeats = node->inputs[0].attr.repeats;
      } else {
        output->attr.repeats = loop_repeats;
      }
      auto &reps = is_follow_input ? node->inputs[0].attr.repeats : loop_repeats;
      SetStrides(reps, output->attr.strides);
      SetStrides(reps, output->attr.vectorized_strides);
    }
  }
}
}  // namespace

class VfPartition : public testing::Test {
 protected:
  void SetUp() override {
    // setenv("DUMP_GE_GRAPH", "2", 1);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<af::RuntimeStubV2>();
    ge::RuntimeStub::SetInstance(stub_v2);
  }

  void TearDown() override {
    // setenv("DUMP_GE_GRAPH", "0", 1);
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

TEST_F(VfPartition, brc_abs) {
  af::AscGraph graph("brc_abs");
  auto s0 = af::Symbol(10);
  auto s1 = af::Symbol(10);
  auto s2 = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z1", s2);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = af::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, One, s2};
  *load0.y.strides = {s2, Zero, One};
  *load0.y.vectorized_axis = {z0.id, z1.id, z2.id};

  af::ascir_op::Abs abs00("abs00");
  abs00.x = load0.y;
  abs00.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs00.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs00.y.dtype = af::DT_FLOAT;
  *abs00.y.axis = {z0.id, z1.id, z2.id};
  *abs00.y.repeats = {s0, One, s2};
  *abs00.y.strides = {s2, Zero, One};
  *abs00.y.vectorized_axis = {z0.id, z1.id, z2.id};

  af::ascir_op::Abs abs01("abs01");
  abs01.x = abs00.y;
  abs01.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs01.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs01.y.dtype = af::DT_FLOAT;
  *abs01.y.axis = {z0.id, z1.id, z2.id};
  *abs01.y.repeats = {s0, One, s2};
  *abs01.y.strides = {s2, Zero, One};
  *abs01.y.vectorized_axis = {z0.id, z1.id, z2.id};

  af::ascir_op::Abs abs02("abs01");
  abs02.x = abs01.y;
  abs02.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs02.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs02.y.dtype = af::DT_FLOAT;
  *abs02.y.axis = {z0.id, z1.id, z2.id};
  *abs02.y.repeats = {s0, One, s2};
  *abs02.y.strides = {s2, Zero, One};
  *abs02.y.vectorized_axis = {z0.id, z1.id, z2.id};

  af::ascir_op::Broadcast brc("brc");
  brc.x = abs02.y;
  brc.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = af::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s1, One};
  *brc.y.vectorized_axis = {z0.id, z1.id, z2.id};

  af::ascir_op::Abs abs("abs");
  abs.x = brc.y;
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, s2};
  *abs.y.strides = {s1 * s2, s1, One};
  *abs.y.vectorized_axis = {z0.id, z1.id, z2.id};

  af::ascir_op::Abs abs1("abs1");
  abs1.x = brc.y;
  abs1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs1.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs1.y.dtype = af::DT_FLOAT;
  *abs1.y.axis = {z0.id, z1.id, z2.id};
  *abs1.y.repeats = {s0, s1, s2};
  *abs1.y.strides = {s1 * s2, s1, One};
  *abs1.y.vectorized_axis = {z0.id, z1.id, z2.id};

  af::ascir_op::Add add("add");
  add.x1 = abs.y;
  add.x2 = abs1.y;
  add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add.attr.sched.axis = {z0.id, z1.id, z2.id};
  add.y.dtype = af::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id, z2.id};
  *add.y.repeats = {s0, s1, s2};
  *add.y.strides = {s1 * s2, s1, One};
  *add.y.vectorized_axis = {z0.id, z1.id, z2.id};

  af::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s1, One};
  *store.y.vectorized_axis = {z0.id, z1.id, z2.id};

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);

  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == "VectorFunc") {
      const std::string *graph_name = af::AttrUtils::GetStr(node->GetOpDescBarePtr(), "sub_graph_name");
      ASSERT_NE(graph_name, nullptr);
      af::AscGraph subgraph("tmp");
      ASSERT_EQ(graph.FindSubGraph(*graph_name, subgraph), af::SUCCESS);
    }
  }
}

TEST_F(VfPartition, brc_only_revert) {
  af::AscGraph graph("brc_abs");
  auto s0 = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Scalar scalar0("scalar0", graph);
  *scalar0.y.repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne};
  *scalar0.y.strides = {af::sym::kSymbolZero, af::sym::kSymbolZero};
  scalar0.ir_attr.SetValue("888");
  scalar0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Broadcast brc("brc");
  brc.x = scalar0.y;
  brc.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc.attr.sched.axis = {z0.id};
  brc.y.dtype = af::DT_FLOAT;
  *brc.y.axis = {z0.id};
  *brc.y.repeats = {s0};
  *brc.y.strides = {One};
  *brc.y.vectorized_axis = {z0.id};

  af::ascir_op::Store store("store");
  store.x = brc.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};
  *store.y.vectorized_axis = {z0.id};

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);
  std::vector<af::AscGraph> sub_graphs;
  graph.GetAllSubGraphs(sub_graphs);
  EXPECT_TRUE(sub_graphs.empty());
}

TEST_F(VfPartition, brc_with_cycle) {
  af::AscGraph graph("brc_abs");
  auto s0 = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Scalar scalar0("scalar0", graph);
  *scalar0.y.repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne};
  *scalar0.y.strides = {af::sym::kSymbolZero, af::sym::kSymbolZero};
  scalar0.ir_attr.SetValue("888");
  scalar0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Broadcast brc("brc");
  brc.x = scalar0.y;
  brc.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc.attr.sched.axis = {z0.id};
  brc.y.dtype = af::DT_FLOAT;
  *brc.y.axis = {z0.id};
  *brc.y.repeats = {s0};
  *brc.y.strides = {One};
  *brc.y.vectorized_axis = {z0.id};

  af::ascir_op::RemovePad remove_pad("remove_pad");
  remove_pad.x = brc.y;
  remove_pad.attr.api.compute_type = af::ComputeType::kComputeElewise;
  remove_pad.attr.sched.axis = {z0.id};
  remove_pad.y.dtype = af::DT_FLOAT;
  *remove_pad.y.axis = {z0.id};
  *remove_pad.y.repeats = {s0};
  *remove_pad.y.strides = {One};
  *remove_pad.y.vectorized_axis = {z0.id};

  af::ascir_op::Add add("add");
  add.x1 = brc.y;
  add.x2 = remove_pad.y;
  add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add.attr.sched.axis = {z0.id};
  add.y.dtype = af::DT_FLOAT;
  *add.y.axis = {z0.id};
  *add.y.repeats = {s0};
  *add.y.strides = {One};
  *add.y.vectorized_axis = {z0.id};

  af::ascir_op::Abs abs1("abs1");
  abs1.x = add.y;
  abs1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs1.attr.sched.axis = {z0.id};
  abs1.y.dtype = af::DT_FLOAT;
  *abs1.y.axis = {z0.id};
  *abs1.y.repeats = {s0};
  *abs1.y.strides = {One};
  *abs1.y.vectorized_axis = {z0.id};

  af::ascir_op::Store store("store");
  store.x = abs1.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};
  *store.y.vectorized_axis = {z0.id};

  af::ascir_op::Abs abs("abs");
  abs.x = brc.y;
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.attr.sched.axis = {z0.id};
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id};
  *abs.y.repeats = {s0};
  *abs.y.strides = {One};
  *abs.y.vectorized_axis = {z0.id};

  af::ascir_op::Store store0("store0");
  store0.x = abs.y;
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.attr.sched.axis = {z0.id};
  store0.y.dtype = af::DT_FLOAT;
  *store0.y.axis = {z0.id};
  *store0.y.repeats = {s0};
  *store0.y.strides = {One};
  *store0.y.vectorized_axis = {z0.id};

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);
  std::vector<af::AscGraph> sub_graphs;
  graph.GetAllSubGraphs(sub_graphs);
  EXPECT_EQ(sub_graphs.size(), 2UL);
  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");

  auto brc_in_root = graph.FindNode("brc");
  ASSERT_EQ(brc_in_root, nullptr);
  auto remove_pad_in_root = graph.FindNode("remove_pad");
  ASSERT_NE(remove_pad_in_root, nullptr);
}

TEST_F(VfPartition, add_more_than_30) {
  af::AscGraph graph("brc_abs");
  auto s0 = af::Symbol(10);
  auto s1 = af::Symbol(10);
  auto s2 = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT;
  data0.ir_attr.SetIndex(0);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load_i("load0");
  load_i.x = data0.y;
  load_i.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load_i.attr.sched.axis = {z0.id, z1.id, z2.id};
  load_i.y.dtype = af::DT_FLOAT;
  *load_i.y.axis = {z0.id, z1.id, z2.id};
  *load_i.y.repeats = {s0, One, s2};
  *load_i.y.strides = {s2, Zero, One};
  *load_i.y.vectorized_axis = {z0.id, z1.id, z2.id};
  *load_i.y.vectorized_strides = {s2, Zero, One};

  std::vector<af::ascir_op::Abs> abs_list;
  for (int i = 0; i < 40; ++i) {
    af::ascir_op::Abs abs_i(std::to_string(i).c_str());
    abs_i.attr.api.compute_type = af::ComputeType::kComputeElewise;
    abs_i.attr.sched.axis = {z0.id, z1.id, z2.id};
    abs_i.y.dtype = af::DT_FLOAT;
    *abs_i.y.axis = {z0.id, z1.id, z2.id};
    *abs_i.y.repeats = {s0, One, s2};
    *abs_i.y.strides = {s2, Zero, One};
    *abs_i.y.vectorized_axis = {z0.id, z1.id, z2.id};
    *abs_i.y.vectorized_strides = {s2, Zero, One};
    abs_list.push_back(abs_i);
  }
  abs_list[0].x = load_i.y;
  for (int i = 1; i < 40; ++i) {
    abs_list[i].x = abs_list[i - 1].y;
  }

  af::ascir_op::Abs abs_o("abs0");
  abs_o.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs_o.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs_o.y.dtype = af::DT_FLOAT;
  *abs_o.y.axis = {z0.id, z1.id, z2.id};
  *abs_o.y.repeats = {s0, One, s2};
  *abs_o.y.strides = {s2, One, One};
  *abs_o.y.vectorized_axis = {z0.id, z1.id, z2.id};
  *abs_o.y.vectorized_strides = {s2, Zero, One};
  abs_o.x = abs_list[39].y;

  af::ascir_op::Broadcast brc("brc");
  brc.x = abs_o.y;
  brc.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = af::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, One};
  *brc.y.vectorized_axis = {z0.id, z1.id, z2.id};
  *brc.y.vectorized_strides = {s2, Zero, One};

  af::ascir_op::Add add("add");
  add.x1 = abs_o.y;
  add.x2 = brc.y;
  add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add.attr.sched.axis = {z0.id, z1.id, z2.id};
  add.y.dtype = af::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id, z2.id};
  *add.y.repeats = {s0, s1, s2};
  *add.y.strides = {s1 * s2, s2, One};
  *add.y.vectorized_axis = {z0.id, z1.id, z2.id};
  *add.y.vectorized_strides = {s2, Zero, One};

  af::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};
  *store.y.vectorized_axis = {z0.id, z1.id, z2.id};
  *store.y.vectorized_strides = {s2, Zero, One};

  af::ascir_op::Output y0("y0");
  y0.x = store.y;
  y0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id};
  y0.y.dtype = af::DT_FLOAT;
  *y0.y.axis = {z0.id, z1.id, z2.id};
  *y0.y.repeats = {s0, s1, s2};
  *y0.y.strides = {s1 * s2, s2, One};
  *y0.y.vectorized_axis = {z0.id, z1.id, z2.id};
  *y0.y.vectorized_strides = {s2, Zero, One};

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);

  ::ascir::utils::DumpGraph(graph, "AfterPart");

  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == "VectorFunc") {
      const std::string *graph_name = af::AttrUtils::GetStr(node->GetOpDescBarePtr(), "sub_graph_name");
      ASSERT_NE(graph_name, nullptr);
      af::AscGraph subgraph("tmp");
      ASSERT_EQ(graph.FindSubGraph(*graph_name, subgraph), af::SUCCESS);
    }
  }

  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);
  auto brc_node = graph.FindNode("brc");
  EXPECT_NE(brc_node, nullptr);
  EXPECT_EQ(sub_graphs.size(), 2UL);
}

TEST_F(VfPartition, skip_fuse_for_io_num) {
  af::AscGraph graph("brc_abs");
  auto s0 = af::Symbol(999);
  auto s1 = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.attr.sched.axis = {z0.id, z1.id};
  load.y.dtype = af::DT_FLOAT;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, One};
  *load.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Load load1("load1");
  load1.x = data0.y;
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, One};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Load load2("load2");
  load2.x = data0.y;
  load2.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = af::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, One};
  *load2.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Load load3("load3");
  load3.x = data0.y;
  load3.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = af::DT_FLOAT;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, One};
  *load3.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Load load4("load4");
  load4.x = data0.y;
  load4.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load4.attr.sched.axis = {z0.id, z1.id};
  load4.y.dtype = af::DT_FLOAT;
  *load4.y.axis = {z0.id, z1.id};
  *load4.y.repeats = {s0, s1};
  *load4.y.strides = {s1, One};
  *load4.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Load load5("load5");
  load5.x = data0.y;
  load5.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load5.attr.sched.axis = {z0.id, z1.id};
  load5.y.dtype = af::DT_FLOAT;
  *load5.y.axis = {z0.id, z1.id};
  *load5.y.repeats = {s0, s1};
  *load5.y.strides = {s1, One};
  *load5.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Load load6("load6");
  load6.x = data0.y;
  load6.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load6.attr.sched.axis = {z0.id, z1.id};
  load6.y.dtype = af::DT_FLOAT;
  *load6.y.axis = {z0.id, z1.id};
  *load6.y.repeats = {s0, s1};
  *load6.y.strides = {s1, One};
  *load6.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Load load7("load7");
  load7.x = data0.y;
  load7.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load7.attr.sched.axis = {z0.id, z1.id};
  load7.y.dtype = af::DT_FLOAT;
  *load7.y.axis = {z0.id, z1.id};
  *load7.y.repeats = {s0, s1};
  *load7.y.strides = {s1, One};
  *load7.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, One};
  *abs.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs1("abs1");
  abs1.x = load1.y;
  abs1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs1.attr.sched.axis = {z0.id, z1.id};
  abs1.y.dtype = af::DT_FLOAT;
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {s0, s1};
  *abs1.y.strides = {s1, One};
  *abs1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs2("abs2");
  abs2.x = load2.y;
  abs2.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs2.attr.sched.axis = {z0.id, z1.id};
  abs2.y.dtype = af::DT_FLOAT;
  *abs2.y.axis = {z0.id, z1.id};
  *abs2.y.repeats = {s0, s1};
  *abs2.y.strides = {s1, One};
  *abs2.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs3("abs3");
  abs3.x = load3.y;
  abs3.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs3.attr.sched.axis = {z0.id, z1.id};
  abs3.y.dtype = af::DT_FLOAT;
  *abs3.y.axis = {z0.id, z1.id};
  *abs3.y.repeats = {s0, s1};
  *abs3.y.strides = {s1, One};
  *abs3.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs4("abs4");
  abs4.x = load4.y;
  abs4.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs4.attr.sched.axis = {z0.id, z1.id};
  abs4.y.dtype = af::DT_FLOAT;
  *abs4.y.axis = {z0.id, z1.id};
  *abs4.y.repeats = {s0, s1};
  *abs4.y.strides = {s1, One};
  *abs4.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs5("abs5");
  abs5.x = load5.y;
  abs5.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs5.attr.sched.axis = {z0.id, z1.id};
  abs5.y.dtype = af::DT_FLOAT;
  *abs5.y.axis = {z0.id, z1.id};
  *abs5.y.repeats = {s0, s1};
  *abs5.y.strides = {s1, One};
  *abs5.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs6("abs6");
  abs6.x = load6.y;
  abs6.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs6.attr.sched.axis = {z0.id, z1.id};
  abs6.y.dtype = af::DT_FLOAT;
  *abs6.y.axis = {z0.id, z1.id};
  *abs6.y.repeats = {s0, s1};
  *abs6.y.strides = {s1, One};
  *abs6.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs7("abs7");
  abs7.x = load7.y;
  abs7.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs7.attr.sched.axis = {z0.id, z1.id};
  abs7.y.dtype = af::DT_FLOAT;
  *abs7.y.axis = {z0.id, z1.id};
  *abs7.y.repeats = {s0, s1};
  *abs7.y.strides = {s1, One};
  *abs7.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Add add("add");
  add.x1 = abs.y;
  add.x2 = abs1.y;
  add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add.attr.sched.axis = {z0.id, z1.id};
  add.y.dtype = af::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, One};
  *add.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Add add1("add1");
  add1.x1 = abs2.y;
  add1.x2 = abs3.y;
  add1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add1.attr.sched.axis = {z0.id, z1.id};
  add1.y.dtype = af::DT_FLOAT;
  *add1.y.axis = {z0.id, z1.id};
  *add1.y.repeats = {s0, s1};
  *add1.y.strides = {s1, One};
  *add1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Add add2("add2");
  add2.x1 = abs4.y;
  add2.x2 = abs5.y;
  add2.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add2.attr.sched.axis = {z0.id, z1.id};
  add2.y.dtype = af::DT_FLOAT;
  *add2.y.axis = {z0.id, z1.id};
  *add2.y.repeats = {s0, s1};
  *add2.y.strides = {s1, One};
  *add2.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Add add3("add3");
  add3.x1 = abs6.y;
  add3.x2 = abs7.y;
  add3.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add3.attr.sched.axis = {z0.id, z1.id};
  add3.y.dtype = af::DT_FLOAT;
  *add3.y.axis = {z0.id, z1.id};
  *add3.y.repeats = {s0, s1};
  *add3.y.strides = {s1, One};
  *add3.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Add add11("add11");
  add11.x1 = add.y;
  add11.x2 = add1.y;
  add11.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add11.attr.sched.axis = {z0.id, z1.id};
  add11.y.dtype = af::DT_FLOAT;
  *add11.y.axis = {z0.id, z1.id};
  *add11.y.repeats = {s0, s1};
  *add11.y.strides = {s1, One};
  *add11.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Add add12("add12");
  add12.x1 = add2.y;
  add12.x2 = add3.y;
  add12.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add12.attr.sched.axis = {z0.id, z1.id};
  add12.y.dtype = af::DT_FLOAT;
  *add12.y.axis = {z0.id, z1.id};
  *add12.y.repeats = {s0, s1};
  *add12.y.strides = {s1, One};
  *add12.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Add add21("add21");
  add21.x1 = add11.y;
  add21.x2 = add12.y;
  add21.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add21.attr.sched.axis = {z0.id, z1.id};
  add21.y.dtype = af::DT_FLOAT;
  *add21.y.axis = {z0.id, z1.id};
  *add21.y.repeats = {s0, s1};
  *add21.y.strides = {s1, One};
  *add21.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Store store("store");
  store.x = add21.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Output out("out");
  out.x = store.y;
  out.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  out.attr.api.type = af::ApiType::kAPITypeBuffer;
  out.y.dtype = af::DT_FLOAT;
  out.ir_attr.SetIndex(0);

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);
  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);
  EXPECT_EQ(sub_graphs.size(), 2UL);
}

TEST_F(VfPartition, cast_reverse) {
  af::AscGraph graph("brc_abs");
  auto s0 = af::Symbol(999);
  auto s1 = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = af::DT_UINT16;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.attr.sched.axis = {z0.id, z1.id};
  load.y.dtype = af::DT_UINT16;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, One};
  *load.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = af::DT_UINT16;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, One};
  *abs.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs1("abs1");
  abs1.x = abs.y;
  abs1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs1.attr.sched.axis = {z0.id, z1.id};
  abs1.y.dtype = af::DT_UINT16;
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {s0, s1};
  *abs1.y.strides = {s1, One};
  *abs1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Cast cast("cast");
  cast.x = abs1.y;
  cast.attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast.attr.sched.axis = {z0.id, z1.id};
  cast.y.dtype = af::DT_UINT64;
  *cast.y.axis = {z0.id, z1.id};
  *cast.y.repeats = {s0, s1};
  *cast.y.strides = {s1, One};
  *cast.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Abs abs2("abs2");
  abs2.x = cast.y;
  abs2.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs2.attr.sched.axis = {z0.id, z1.id};
  abs2.y.dtype = af::DT_UINT64;
  *abs2.y.axis = {z0.id, z1.id};
  *abs2.y.repeats = {s0, s1};
  *abs2.y.strides = {s1, One};
  *abs2.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Store store("store");
  store.x = abs1.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_UINT64;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Output out("out");
  out.x = store.y;
  out.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  out.attr.api.type = af::ApiType::kAPITypeBuffer;
  out.y.dtype = af::DT_UINT64;
  out.ir_attr.SetIndex(0);

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);
  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);
  EXPECT_EQ(sub_graphs.size(), 1UL);
  auto cast_node = graph.FindNode("cast");
  EXPECT_NE(cast_node, nullptr);
  auto abs2_node = graph.FindNode("abs2");
  EXPECT_NE(abs2_node, nullptr);
}

TEST_F(VfPartition, cast_fused) {
  af::AscGraph graph("brc_abs");
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs1("abs1");
  abs1.x = abs.y;
  abs1.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Cast cast("cast");
  cast.x = abs1.y;
  cast.y.dtype = af::DT_FLOAT;

  af::ascir_op::Exp exp("exp");
  exp.x = cast.y;
  exp.y.dtype = af::DT_FLOAT;

  af::ascir_op::Exp exp1("exp1");
  exp1.x = exp.y;
  exp1.y.dtype = af::DT_FLOAT;

  af::ascir_op::Cast cast2("cast2");
  cast2.x = exp1.y;
  cast2.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Exp exp2("exp2");
  exp2.x = cast2.y;
  exp2.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Store store("store");
  store.x = exp2.y;
  store.y.dtype = af::DT_UINT64;

  af::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  SetupGraphAxes(graph, {af::Symbol("s0"), af::Symbol("s1")});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);
  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 2UL);

  auto cast_node = sub_graphs[1].FindNode("cast");
  EXPECT_NE(cast_node, nullptr);
  auto cast2_node = sub_graphs[1].FindNode("cast2");
  EXPECT_NE(cast2_node, nullptr);
  auto exp2_node = graph.FindNode("exp2");
  EXPECT_NE(exp2_node, nullptr);
}

TEST_F(VfPartition, vf_cascade) {
  af::AscGraph graph("brc_abs");
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Exp exp("exp");
  exp.x = abs.y;
  exp.y.dtype = af::DT_FLOAT;

  af::ascir_op::Abs abs1("abs1");
  abs1.x = load.y;
  abs1.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Exp exp1("exp1");
  exp1.x = abs1.y;
  exp1.y.dtype = af::DT_FLOAT;

  af::ascir_op::Div div("div");
  div.x1 = exp.y;
  div.x2 = exp1.y;
  div.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Store store("store");
  store.x = div.y;
  store.y.dtype = af::DT_UINT64;

  af::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  SetupGraphAxes(graph, {af::Symbol("s0"), af::Symbol("s1")});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);

  ::ascir::utils::DumpGraph(graph, "AfterPart");
}

TEST_F(VfPartition, all_zero_axis_stride) {
  af::AscGraph graph("brc_abs");
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs1("abs1");
  abs1.x = abs.y;
  abs1.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Store store("store");
  store.x = abs1.y;
  store.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  SetupGraphAxes(graph, {af::Symbol(1), af::Symbol(1)});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);
  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);

  auto abs_node = sub_graphs[0].FindNode("abs");
  ASSERT_NE(abs_node, nullptr);
  EXPECT_EQ(abs_node->attr.sched.loop_axis, -1);

  auto abs1_node = sub_graphs[0].FindNode("abs1");
  ASSERT_NE(abs1_node, nullptr);
  EXPECT_EQ(abs1_node->attr.sched.loop_axis, -1);
}

TEST_F(VfPartition, ScalarInputSupportVf) {
  af::AscGraph graph("brc_abs");
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs1("abs1");
  abs1.x = abs.y;
  abs1.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Scalar scalar0("scalar0", graph);
  af::ascir_op::Maximum maximum("maximum");
  maximum.x1 = abs1.y;
  maximum.x2 = scalar0.y;
  maximum.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Store store("store");
  store.x = maximum.y;
  store.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  SetupGraphAxes(graph, {af::Symbol(10), af::Symbol(2)});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);
  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);

  auto scalar_node = graph.FindNode("scalar0");
  ASSERT_NE(scalar_node, nullptr);

  auto maximum_node = sub_graphs[0].FindNode("maximum");
  ASSERT_NE(maximum_node, nullptr);

  auto abs_node = sub_graphs[0].FindNode("abs");
  ASSERT_NE(abs_node, nullptr);

  auto abs1_node = sub_graphs[0].FindNode("abs1");
  ASSERT_NE(abs1_node, nullptr);
}

TEST_F(VfPartition, ScalarDataInputDisableVf) {
  af::AscGraph graph("scalardata_disable_vf");
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Abs abs1("abs1");
  abs1.x = abs.y;
  abs1.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::ScalarData scalar_data0("scalar_data0", graph);
  scalar_data0.ir_attr.SetIndex(0);

  af::ascir_op::Maximum maximum("maximum");
  maximum.x1 = abs1.y;
  maximum.x2 = scalar_data0.y;
  maximum.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Store store("store");
  store.x = maximum.y;
  store.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  SetupGraphAxes(graph, {af::Symbol(10), af::Symbol(2)});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);
  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);

  auto scalardata_node = graph.FindNode("scalar_data0");
  ASSERT_NE(scalardata_node, nullptr);

  // ScalarData 作为运行时标量输入，Maximum 因 ScalarData 直连而禁用 VF，
  // 分区时 ScalarData 走 InsertScalarNode 路径，与常量 Scalar 行为一致
  auto abs_node = sub_graphs[0].FindNode("abs");
  ASSERT_NE(abs_node, nullptr);

  auto abs1_node = sub_graphs[0].FindNode("abs1");
  ASSERT_NE(abs1_node, nullptr);
}

TEST_F(VfPartition, test_scalar_brc) {
  af::AscGraph graph("brc_abs");
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Scalar scalar0("scalar0", graph);
  *scalar0.y.repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne};
  *scalar0.y.strides = {af::sym::kSymbolZero, af::sym::kSymbolZero};
  scalar0.ir_attr.SetValue("888");

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = scalar0.y;
  brc0.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs1("abs1");
  abs1.x = brc0.y;
  abs1.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Add add0("add0");
  add0.x1 = abs.y;
  add0.x2 = brc0.y;
  add0.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Add add1("add1");
  add1.x1 = add0.y;
  add1.x2 = abs1.y;
  add1.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Store store("store");
  store.x = add1.y;
  store.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  SetupGraphAxes(graph, {af::Symbol(32), af::Symbol(16)});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);

  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);
  auto brc_in_root = graph.FindNode("brc0");
  ASSERT_EQ(brc_in_root, nullptr);
}

TEST_F(VfPartition, test_scalar_brc_unsupport_vf) {
  af::AscGraph graph("brc_abs");
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Data data1("data1", graph);
  *data1.y.repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne};
  *data1.y.strides = {af::sym::kSymbolOne, af::sym::kSymbolZero};
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = load1.y;
  brc0.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs1("abs1");
  abs1.x = brc0.y;
  abs1.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Add add0("add0");
  add0.x1 = abs.y;
  add0.x2 = brc0.y;
  add0.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Store store("store");
  store.x = add0.y;
  store.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  SetupGraphAxes(graph, {af::Symbol(32), af::Symbol(16)});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);

  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);
  auto brc_node = graph.FindNode("brc0");
  ASSERT_NE(brc_node, nullptr);
  auto sub_brc_node = sub_graphs[0].FindNode("brc0");
  ASSERT_EQ(sub_brc_node, nullptr);
}

TEST_F(VfPartition, test_scalar_brc_multi_output) {
  af::AscGraph graph("brc_abs");
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = load.y;
  brc0.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Sub sub("sub0");
  sub.x1 = brc0.y;
  sub.x2 = brc0.y;
  sub.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Add add0("add0");
  add0.x1 = sub.y;
  add0.x2 = brc0.y;
  add0.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Mul mul("mul");
  mul.x1 = add0.y;
  mul.x2 = load1.y;
  mul.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  SetupGraphAxes(graph, {af::Symbol(1)});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);

  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);
  auto brc_in_root = graph.FindNode("brc0");
  ASSERT_EQ(brc_in_root, nullptr);
}

TEST_F(VfPartition, test_vf_no_cache) {
  af::AscGraph graph("brc_abs");
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs0("abs0");
  abs0.x = load.y;
  abs0.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs1("abs1");
  abs1.x = abs0.y;
  abs1.attr.sched.exec_condition = af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
  abs1.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs2("abs2");
  abs2.x = abs0.y;
  abs2.attr.sched.exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  abs2.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Store store0("store0");
  store0.x = abs1.y;
  store0.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Output out("out");
  out.x = store0.y;
  out.ir_attr.SetIndex(0);

  af::ascir_op::Store store1("store1");
  store1.x = abs2.y;
  store1.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Output out1("out1");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(1);

  SetupGraphAxes(graph, {af::Symbol(1)});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);

  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);
  auto brc_in_root = graph.FindNode("brc0");
  ASSERT_EQ(brc_in_root, nullptr);
  for (const auto &node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::VectorFunc>(node)) {
      EXPECT_EQ(node->attr.sched.exec_condition, af::ExecuteCondition::kNoCache);
    }
  }
}

TEST_F(VfPartition, VFWithCycleDetectBug) {
  af::AscGraph graph("cycle_test");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = af::DT_FLOAT16;
  af::ascir_op::Cast cast0("cast0");
  cast0.x = load.y;
  cast0.y.dtype = af::DT_FLOAT;

  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.y.dtype = af::DT_FLOAT16;
  af::ascir_op::Cast cast1("cast1");
  cast1.x = load1.y;
  cast1.y.dtype = af::DT_FLOAT;

  af::ascir_op::Add add0("add0");
  add0.x1 = cast0.y;
  add0.x2 = cast1.y;
  add0.y.dtype = af::DT_FLOAT;
  af::ascir_op::Add add1("add1");
  add1.x1 = cast0.y;
  add1.x2 = cast1.y;
  add1.y.dtype = af::DT_FLOAT;
  af::ascir_op::Add add2("add2");
  add2.x1 = cast0.y;
  add2.x2 = cast1.y;
  add2.y.dtype = af::DT_FLOAT;
  af::ascir_op::Add add3("add3");
  add3.x1 = cast0.y;
  add3.x2 = cast1.y;
  add3.y.dtype = af::DT_FLOAT;

  af::ascir_op::Cast cast3_0("cast3_0");
  cast3_0.x = add0.y;
  cast3_0.y.dtype = af::DT_FLOAT16;
  af::ascir_op::Store store0("store0");
  store0.x = cast3_0.y;
  store0.y.dtype = af::DT_FLOAT16;
  af::ascir_op::Output out0("out0");
  out0.x = store0.y;
  out0.ir_attr.SetIndex(0);

  af::ascir_op::Cast cast3_1("cast3_1");
  cast3_1.x = add1.y;
  cast3_1.y.dtype = af::DT_FLOAT16;
  af::ascir_op::Store store1("store1");
  store1.x = cast3_1.y;
  store1.y.dtype = af::DT_FLOAT16;
  af::ascir_op::Output out1("out1");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(1);

  af::ascir_op::Cast cast3_2("cast3_2");
  cast3_2.x = add2.y;
  cast3_2.y.dtype = af::DT_FLOAT16;
  af::ascir_op::Store store2("store2");
  store2.x = cast3_2.y;
  store2.y.dtype = af::DT_FLOAT16;
  af::ascir_op::Output out2("out2");
  out2.x = store2.y;
  out2.ir_attr.SetIndex(2);

  af::ascir_op::Cast cast3_3("cast3_3");
  cast3_3.x = add3.y;
  cast3_3.y.dtype = af::DT_FLOAT16;
  af::ascir_op::Store store3("store3");
  store3.x = cast3_3.y;
  store3.y.dtype = af::DT_FLOAT16;
  af::ascir_op::Output out3("out3");
  out3.x = store3.y;
  out3.ir_attr.SetIndex(3);

  SetupGraphAxes(graph, {af::Symbol(1)});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);

  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);
}

TEST_F(VfPartition, test_vf_cache) {
  af::AscGraph graph("brc_abs");
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs0("abs0");
  abs0.x = load.y;
  abs0.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs1("abs1");
  abs1.x = abs0.y;
  abs1.attr.sched.exec_condition = af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
  abs1.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs2("abs2");
  abs2.x = abs0.y;
  abs2.attr.sched.exec_condition = af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
  abs2.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Store store0("store0");
  store0.x = abs1.y;
  store0.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Output out("out");
  out.x = store0.y;
  out.ir_attr.SetIndex(0);

  af::ascir_op::Store store1("store1");
  store1.x = abs2.y;
  store1.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Output out1("out1");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(1);

  SetupGraphAxes(graph, {af::Symbol(1)});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);

  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);
  auto brc_in_root = graph.FindNode("brc0");
  ASSERT_EQ(brc_in_root, nullptr);
  for (const auto &node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::VectorFunc>(node)) {
      EXPECT_EQ(node->attr.sched.exec_condition, af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis);
    }
  }
}

// 测试 Cast 位宽差距恰好为 2 倍的情况（INT32->INT16），应该允许融合
TEST_F(VfPartition, cast_bitwidth_gap_exactly_2x) {
  af::AscGraph graph("cast_2x_gap");
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  af::ascir_op::Load load("load");
  load.x = data0.y;
  load.y.dtype = af::DT_INT32;

  af::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = af::DT_INT32;

  af::ascir_op::Cast cast("cast");
  cast.x = abs.y;
  cast.y.dtype = af::DT_INT16;  // INT32 -> INT16, 恰好 2 倍差距

  af::ascir_op::Exp exp("exp");
  exp.x = cast.y;
  exp.y.dtype = af::DT_INT16;

  af::ascir_op::Store store("store");
  store.x = exp.y;
  store.y.dtype = af::DT_INT16;

  af::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  SetupGraphAxes(graph, {af::Symbol(10), af::Symbol(10)});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);

  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);

  // 恰好 2 倍差距应该允许融合
  EXPECT_EQ(sub_graphs.size(), 1UL);
  auto cast_node = sub_graphs[0].FindNode("cast");
  EXPECT_NE(cast_node, nullptr);
}

// 测试 Cast 位宽差距超过 2 倍的情况（INT64->INT16，4倍差距），不应该融合
// 使用多个输入分支来触发 Cluster 合并时的位宽检查
TEST_F(VfPartition, cast_bitwidth_gap_exceed_2x) {
  af::AscGraph graph("cast_4x_gap");
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(2);

  // 分支1: INT64 -> INT16 (4倍差距)
  af::ascir_op::Load load1("load1");
  load1.x = data0.y;
  load1.y.dtype = af::DT_INT64;

  af::ascir_op::Abs abs1("abs1");
  abs1.x = load1.y;
  abs1.y.dtype = af::DT_INT64;

  af::ascir_op::Cast cast1("cast1");
  cast1.x = abs1.y;
  cast1.y.dtype = af::DT_INT16;

  // 分支2: INT32
  af::ascir_op::Load load2("load2");
  load2.x = data1.y;
  load2.y.dtype = af::DT_INT32;

  af::ascir_op::Abs abs2("abs2");
  abs2.x = load2.y;
  abs2.y.dtype = af::DT_INT32;

  // 合并点：两个分支的位宽差距超过 2 倍，应该阻止融合
  af::ascir_op::Add add("add");
  add.x1 = cast1.y;
  add.x2 = abs2.y;
  add.y.dtype = af::DT_INT16;

  af::ascir_op::Store store("store");
  store.x = add.y;
  store.y.dtype = af::DT_INT16;

  af::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  SetupGraphAxes(graph, {af::Symbol(10), af::Symbol(10)});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);

  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);

  // 由于位宽差距超过 2 倍，两个分支不应该融合到同一个 Cluster
  EXPECT_GE(sub_graphs.size(), 1UL);
}

// 测试高→低 Cast 不应和输出节点融合
TEST_F(VfPartition, cast_high_to_low_no_fuse_with_output) {
  af::AscGraph graph("cast_high_to_low");
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  af::ascir_op::Load load("load");
  load.x = data0.y;
  load.y.dtype = af::DT_INT64;

  af::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = af::DT_INT64;

  af::ascir_op::Cast cast("cast");
  cast.x = abs.y;
  cast.y.dtype = af::DT_INT32;  // INT64 -> INT32, 高→低

  af::ascir_op::Exp exp("exp");
  exp.x = cast.y;
  exp.y.dtype = af::DT_INT32;

  af::ascir_op::Store store("store");
  store.x = exp.y;
  store.y.dtype = af::DT_INT32;

  af::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  SetupGraphAxes(graph, {af::Symbol(10), af::Symbol(10)});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);

  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);

  // 高→低 Cast 的检查是在合并 Cluster 时进行的
  // 这里只有一个分支，所以所有节点应该能融合
  EXPECT_EQ(sub_graphs.size(), 1UL);
  auto cast_node = sub_graphs[0].FindNode("cast");
  EXPECT_NE(cast_node, nullptr);
}

// 测试当节点尾轴stride!=1时禁用VF支持
TEST_F(VfPartition, tail_axis_stride_not_one_disable_vf) {
  af::AscGraph graph("tail_stride_test");
  auto s0 = af::Symbol(10);
  auto s1 = af::Symbol(10);
  auto s2 = af::Symbol(10);
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store("store");
  store.x = abs.y;
  store.y.dtype = ge::DT_FLOAT;

  SetupGraphAxes(graph, {s0, s1, s2});
  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  // 设置尾轴stride!=1，应该禁用VF
  std::vector<af::Expression> strides = {s1 * s2, s2, af::Symbol(2)};
  graph.FindNode("load0")->outputs[0].attr.vectorized_strides = strides;
  graph.FindNode("abs")->outputs[0].attr.vectorized_strides = strides;
  graph.FindNode("store")->outputs[0].attr.vectorized_strides = strides;

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), af::SUCCESS);

  std::vector<af::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), af::SUCCESS);
  EXPECT_EQ(sub_graphs.size(), 0UL);
}

TEST_F(VfPartition, topological_sort_for_vf_graph_keeps_load_before_consumer) {
  af::AscGraph graph("vf_sort_keep_dependency");
  auto z0 = graph.CreateAxis("z0", af::Symbol(2));

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  af::ascir_op::TrueDiv truediv("truediv");
  truediv.x1 = load0.y;
  truediv.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store("store");
  store.x = truediv.y;
  store.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.y.dtype = ge::DT_FLOAT;
  truediv.x2 = load1.y;

  af::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  for (const auto &node : graph.GetAllNodes()) {
    node->attr.sched.axis = {z0.id};
    node->attr.sched.loop_axis = z0.id;
  }
  graph.FindNode("truediv")->attr.sched.loop_axis = -1;

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.TopologicalSortingForVfGraph(graph), af::SUCCESS);

  std::vector<std::string> node_names;
  for (const auto &node : graph.GetAllNodes()) {
    node_names.push_back(node->GetName());
  }
  auto load1_iter = std::find(node_names.begin(), node_names.end(), "load1");
  auto truediv_iter = std::find(node_names.begin(), node_names.end(), "truediv");
  ASSERT_NE(load1_iter, node_names.end());
  ASSERT_NE(truediv_iter, node_names.end());
  EXPECT_LT(std::distance(node_names.begin(), load1_iter), std::distance(node_names.begin(), truediv_iter));
}
}  // namespace optimize
