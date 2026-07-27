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
#undef private
#include "ascir_ops_utils.h"
#include "schedule_utils.h"
#include "platform_context.h"
#include "platform/v1/platformv1.h"
#include "platform/v1/alignment_strategy.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace optimize::autoschedule;

namespace optimize {
using namespace ge;

template <typename Op, typename Dtype>
void SetTwoDimNodeAttr(Op &op, Dtype dtype, af::ComputeType compute_type, const std::vector<af::AxisId> &axes,
                       const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  op.y.dtype = dtype;
  op.attr.sched.axis = axes;
  op.attr.api.compute_type = compute_type;
  *op.y.axis = axes;
  *op.y.repeats = repeats;
  *op.y.strides = strides;
  *op.y.vectorized_axis = axes;
}

void ExpectCompactReduceBranch(const af::AscGraph &graph, const char *aligned_input) {
  std::vector<af::Expression> aligned_input_strides = {af::Symbol(24), One};
  std::vector<af::Expression> compact_output_strides = {One, Zero};
  EXPECT_EQ(graph.FindNode(aligned_input)->outputs[0].attr.vectorized_strides, aligned_input_strides);
  EXPECT_EQ(graph.FindNode("max")->outputs[0].attr.vectorized_strides, compact_output_strides);
  EXPECT_EQ(graph.FindNode("sum")->outputs[0].attr.vectorized_strides, compact_output_strides);
  EXPECT_EQ(graph.FindNode("scalar_broadcast")->outputs[0].attr.vectorized_strides, compact_output_strides);
  EXPECT_EQ(graph.FindNode("div")->outputs[0].attr.vectorized_strides, compact_output_strides);
}

class VectorizedAlignmentUT : public testing::Test {
 protected:
  void SetUp() override {}

  class AlignmentStrategyShadow : public AlignmentStrategy {
   public:
    AlignmentStrategyShadow() {
      AlignmentStrategy::InitAlignmentInferFunc();
    }

    af::Status AccessSetAlignWidth(const ImplGraph &impl_graph) {
      return SetAlignWidth(impl_graph);
    }

    af::Status AccessAddRemovePadForTailAxisDiscontinuousLoad(ImplGraph &impl_graph) {
      return AddRemovePadForTailAxisDiscontinuousLoad(impl_graph);
    }
    af::Status AccessAddPadForAlignmentConflictNode(ImplGraph &impl_graph) {
      return AddPadForAlignmentConflictNode(impl_graph);
    }
    af::Status AccessInferAlignmentForOneNode(const af::AscNodePtr &node) {
      return InferAlignmentForOneNode(node);
    }
    // 当前tensor的对齐行为只会出现在尾轴,如果没有新的对齐行为或者类型,该函数不应该修改
    af::Status AccessSetVectorizedStridesForOneNode(const af::AscNodePtr &node) {
      return SetVectorizedStridesForOneNode(node);
    }
  };
};

TEST_F(VectorizedAlignmentUT, concat_with_small_tail_axis) {
  af::AscGraph graph("SmallTailAxisConcat");
  auto s0 = af::Symbol(2);
  auto s1 = af::Symbol(31);
  auto s2 = af::Symbol(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};
  *load0.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Load load1("load1");
  load1.x = data0.y;
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s2};
  *load1.y.strides = {One, Zero};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Concat concat("concat");
  concat.x = {load0.y, load1.y};
  concat.attr.api.compute_type = af::ComputeType::kComputeConcat;
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_FLOAT;
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1 + s2};
  *concat.y.strides = {s1 + s2, One};
  *concat.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1 + s2};
  *store.y.strides = {s1 + s2, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  AlignmentHandler handler;
  ASSERT_EQ(handler.AlignVectorizedStrides(graph), af::SUCCESS);

  auto load0_node = graph.FindNode("load0");
  ASSERT_NE(load0_node, nullptr);
  auto load1_node = graph.FindNode("load1");
  ASSERT_NE(load1_node, nullptr);
  auto concat_node = graph.FindNode("concat");
  ASSERT_NE(concat_node, nullptr);

  std::vector<af::Expression> golden_strides_load0 = {s1, One};
  std::vector<af::Expression> golden_strides_load1 = {One, Zero};
  std::vector<af::Expression> golden_strides_concat = {s1 + s2, One};
  EXPECT_EQ(load0_node->outputs[0].attr.vectorized_strides, golden_strides_load0);
  EXPECT_EQ(load1_node->outputs[0].attr.vectorized_strides, golden_strides_load1);
  EXPECT_EQ(concat_node->outputs[0].attr.vectorized_strides, golden_strides_concat);
}

TEST_F(VectorizedAlignmentUT, brc_align_input_not_need_align) {
  af::AscGraph graph("Concat");
  auto s0 = af::Symbol(999);
  auto s1 = af::Symbol(10);
  auto s2 = af::Symbol(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, One};
  *load0.y.strides = {One, Zero};
  *load0.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Broadcast brc("brc");
  brc.x = load0.y;
  brc.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc.attr.sched.axis = {z0.id, z1.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id};
  *brc.y.repeats = {s0, s1};
  *brc.y.strides = {s1, One};
  *brc.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Load load1("load1");
  load1.x = data0.y;
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s2};
  *load1.y.strides = {One, Zero};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Concat concat("concat");
  concat.x = {load1.y, brc.y};
  concat.attr.api.compute_type = af::ComputeType::kComputeConcat;
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_FLOAT;
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1 + s2};
  *concat.y.strides = {s1 + s2, One};
  *concat.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1 + s2};
  *store.y.strides = {s1 + s2 + s0, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  AlignmentHandler handler;
  ASSERT_EQ(handler.AlignVectorizedStrides(graph), af::SUCCESS);

  auto load0_node = graph.FindNode("load0");
  ASSERT_NE(load0_node, nullptr);
  auto load1_node = graph.FindNode("load1");
  ASSERT_NE(load1_node, nullptr);
  auto brc_node = graph.FindNode("brc");
  ASSERT_NE(brc_node, nullptr);
  auto concat_node = graph.FindNode("concat");
  ASSERT_NE(concat_node, nullptr);

  std::vector<af::Expression> golden_strides_load1 = {One, Zero};
  std::vector<af::Expression> golden_strides_load0 = {One, Zero};
  std::vector<af::Expression> golden_strides_brc = {af::Symbol(16), One};
  EXPECT_EQ(load0_node->outputs[0].attr.vectorized_strides, golden_strides_load0);
  EXPECT_EQ(load1_node->outputs[0].attr.vectorized_strides, golden_strides_load1);
  EXPECT_EQ(concat_node->outputs[0].attr.vectorized_strides, golden_strides_brc);
}

TEST_F(VectorizedAlignmentUT, brc_align_input_need_align) {
  af::AscGraph graph("LoadBrc");
  auto s0 = af::Symbol(10);
  auto s1 = af::Symbol(10);
  auto s2 = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s1, One, s2};
  *load0.y.strides = {s2, Zero, One};
  *load0.y.vectorized_axis = {z0.id, z1.id, z2.id};

  af::ascir_op::Broadcast brc("brc");
  brc.x = load0.y;
  brc.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, One};
  *brc.y.vectorized_axis = {z0.id, z1.id, z2.id};

  AlignmentHandler handler;
  ASSERT_EQ(handler.AlignVectorizedStrides(graph), af::SUCCESS);

  auto load0_node = graph.FindNode("load0");
  ASSERT_NE(load0_node, nullptr);
  auto brc_node = graph.FindNode("brc");
  ASSERT_NE(brc_node, nullptr);

  std::vector<af::Expression> golden_strides_load0 = {af::Symbol(16), Zero, One};
  std::vector<af::Expression> golden_strides_brc = {af::Symbol(160), af::Symbol(16), One};

  EXPECT_EQ(load0_node->outputs[0].attr.vectorized_strides, golden_strides_load0);
  EXPECT_EQ(brc_node->outputs[0].attr.vectorized_strides, golden_strides_brc);
}

TEST_F(VectorizedAlignmentUT, brc_1b1_to_abc_need_align) {
  af::AscGraph graph("LoadBrc");
  auto s0 = af::Symbol(10);
  auto s1 = af::Symbol(10);
  auto s2 = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {One, s1, One};
  *load0.y.strides = {Zero, One, Zero};
  *load0.y.vectorized_axis = {z0.id, z1.id, z2.id};

  af::ascir_op::Broadcast brc("brc");
  brc.x = load0.y;
  brc.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, One};
  *brc.y.vectorized_axis = {z0.id, z1.id, z2.id};

  AlignmentHandler handler;
  ASSERT_EQ(handler.AlignVectorizedStrides(graph), af::SUCCESS);

  auto load0_node = graph.FindNode("load0");
  ASSERT_NE(load0_node, nullptr);
  auto brc_node = graph.FindNode("brc");
  ASSERT_NE(brc_node, nullptr);

  std::vector<af::Expression> golden_strides_load0 = {Zero, af::Symbol(8), Zero};
  std::vector<af::Expression> golden_strides_brc = {af::Symbol(160), af::Symbol(16), One};

  EXPECT_EQ(load0_node->outputs[0].attr.vectorized_strides, golden_strides_load0);
  EXPECT_EQ(brc_node->outputs[0].attr.vectorized_strides, golden_strides_brc);
}

TEST_F(VectorizedAlignmentUT, brc_111_to_abc_no_need_align) {
  af::AscGraph graph("LoadBrc");
  auto s0 = af::Symbol(10);
  auto s1 = af::Symbol(10);
  auto s2 = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {One, One, One};
  *load0.y.strides = {One, One, One};
  *load0.y.vectorized_axis = {z0.id, z1.id, z2.id};

  af::ascir_op::Broadcast brc("brc");
  brc.x = load0.y;
  brc.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, One};
  *brc.y.vectorized_axis = {z0.id, z1.id, z2.id};

  AlignmentHandler handler;
  ASSERT_EQ(handler.AlignVectorizedStrides(graph), af::SUCCESS);

  auto load0_node = graph.FindNode("load0");
  ASSERT_NE(load0_node, nullptr);
  auto brc_node = graph.FindNode("brc");
  ASSERT_NE(brc_node, nullptr);

  std::vector<af::Expression> golden_strides_load0 = {Zero, Zero, Zero};
  std::vector<af::Expression> golden_strides_brc = {af::Symbol(160), af::Symbol(16), One};

  EXPECT_EQ(load0_node->outputs[0].attr.vectorized_strides, golden_strides_load0);
  EXPECT_EQ(brc_node->outputs[0].attr.vectorized_strides, golden_strides_brc);
}

TEST_F(VectorizedAlignmentUT, last_axis_slice_discontinuous) {
  af::AscGraph graph("Concat");
  auto s0 = af::Symbol(2);
  auto s1 = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};
  *load0.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Load load1("load1");
  load1.x = data0.y;
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1 * s0, s0};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Add add("add");
  add.x1 = load0.y;
  add.x2 = load1.y;
  add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add.attr.sched.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1 * s0, s0};
  *add.y.vectorized_axis = {z0.id, z1.id};

  AlignmentHandler handler;
  ASSERT_EQ(handler.AlignVectorizedStrides(graph), af::SUCCESS);

  auto load0_node = graph.FindNode("load0");
  ASSERT_NE(load0_node, nullptr);
  auto load1_node = graph.FindNode("load1");
  ASSERT_NE(load1_node, nullptr);
  auto add_node = graph.FindNode("add");
  ASSERT_NE(add_node, nullptr);

  std::vector<af::Expression> golden_strides1 = {af::Symbol(80), af::Symbol(8)};
  std::vector<af::Expression> golden_strides2 = {af::Symbol(10), af::Symbol(1)};
  EXPECT_EQ(load0_node->outputs[0].attr.vectorized_strides, golden_strides2);
  EXPECT_EQ(load1_node->outputs[0].attr.vectorized_strides, golden_strides1);
  EXPECT_EQ(add_node->outputs[0].attr.vectorized_strides, golden_strides2);
}

TEST_F(VectorizedAlignmentUT, nouse_brc_need_input_align) {
  af::AscGraph graph("Concat");
  auto s0 = af::Symbol(2);
  auto s1 = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, One};
  *load0.y.strides = {One, Zero};
  *load0.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = load0.y;
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.attr.sched.axis = {z0.id, z1.id};
  brc0.y.dtype = ge::DT_FLOAT;
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {s0, s1};
  *brc0.y.strides = {s1, One};
  *brc0.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Load load1("load1");
  load1.x = data0.y;
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, One};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};
  *brc1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Add add("add");
  add.x1 = brc0.y;
  add.x2 = brc1.y;
  add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add.attr.sched.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, One};
  *add.y.vectorized_axis = {z0.id, z1.id};

  AlignmentHandler handler;
  ASSERT_EQ(handler.AlignVectorizedStrides(graph), af::SUCCESS);

  auto load0_node = graph.FindNode("load0");
  ASSERT_NE(load0_node, nullptr);
  auto load1_node = graph.FindNode("load1");
  ASSERT_NE(load1_node, nullptr);
  auto add_node = graph.FindNode("add");
  ASSERT_NE(add_node, nullptr);

  std::vector<af::Expression> golden_strides_load0 = {af::Symbol(1), af::Symbol(0)};
  std::vector<af::Expression> golden_strides = {af::Symbol(16), af::Symbol(1)};
  EXPECT_EQ(load0_node->outputs[0].attr.vectorized_strides, golden_strides_load0);
  EXPECT_EQ(load1_node->outputs[0].attr.vectorized_strides, golden_strides);
  EXPECT_EQ(add_node->outputs[0].attr.vectorized_strides, golden_strides);
}

TEST_F(VectorizedAlignmentUT, discontinuous_write_need_align) {
  af::AscGraph graph("Concat");
  auto s0 = af::Symbol(10);
  auto s1 = af::Symbol(1);
  auto stride_val = af::Symbol(66);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, Zero};
  *load0.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Store store("store");
  store.x = load0.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {stride_val, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  AlignmentHandler handler;
  ASSERT_EQ(handler.AlignVectorizedStrides(graph), af::SUCCESS);

  auto load0_node = graph.FindNode("load0");
  ASSERT_NE(load0_node, nullptr);
  auto stode_node = graph.FindNode("store");
  ASSERT_NE(stode_node, nullptr);

  std::vector<af::Expression> golden_strides = {af::Symbol(8), af::Symbol(0)};
  EXPECT_EQ(load0_node->outputs[0].attr.vectorized_strides, golden_strides);
  EXPECT_EQ(stode_node->outputs[0].attr.vectorized_strides, golden_strides);
}

TEST_F(VectorizedAlignmentUT, transpose_fp32_need_64b_align) {
  af::AscGraph graph("Transpose");
  auto s0 = af::Symbol(10);
  auto s1 = af::Symbol(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z1.id, z0.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};
  *load0.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Transpose transpose("transpose");
  transpose.x = load0.y;
  transpose.attr.api.compute_type = af::ComputeType::kComputeTranspose;
  transpose.attr.sched.axis = {z1.id, z0.id};
  transpose.y.dtype = ge::DT_FLOAT;
  *transpose.y.axis = {z1.id, z0.id};
  *transpose.y.repeats = {s1, s0};
  *transpose.y.strides = {s0, One};
  *transpose.y.vectorized_axis = {z1.id, z0.id};

  af::ascir_op::Store store("store");
  store.x = load0.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z1.id, z0.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z1.id, z0.id};
  *store.y.repeats = {s1, s0};
  *store.y.strides = {s0, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  AlignmentHandler handler;
  ASSERT_EQ(handler.AlignVectorizedStrides(graph), af::SUCCESS);

  auto load0_node = graph.FindNode("load0");
  ASSERT_NE(load0_node, nullptr);
  auto stode_node = graph.FindNode("store");
  ASSERT_NE(stode_node, nullptr);

  std::vector<af::Expression> golden_strides = {af::Symbol(16), af::Symbol(0)};
  std::vector<af::Expression> golden_strides1 = {af::Symbol(16), af::Symbol(0)};
  EXPECT_EQ(load0_node->outputs[0].attr.vectorized_strides, golden_strides);
  EXPECT_EQ(stode_node->outputs[0].attr.vectorized_strides, golden_strides);
}

TEST_F(VectorizedAlignmentUT, removepad_and_add_pad) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = af::Symbol(2);
  auto s1 = af::Symbol(3);
  auto s2 = af::Symbol(10);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x("x", graph);
  x.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x.attr.api.type = af::ApiType::kAPITypeBuffer;
  x.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {s1 * s2 * s2, s2 * s2, s2};

  af::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, s2};
  *abs.y.strides = {s1 * s2 * s2, s2 * s2, s2};

  af::ascir_op::Max max("max");
  max.x = abs.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id};
  max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  max.y.dtype = ge::DT_FLOAT16;
  *max.y.axis = {z0.id, z1.id, z2.id};
  *max.y.repeats = {s0, s1, One};
  *max.y.strides = {s1, One, Zero};

  af::ascir_op::Store store("store");
  store.x = max.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, One};
  *store.y.strides = {s1, One, Zero};

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load1("load1");
  load1.x = x.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_INT64;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2 * s2, s2 * s2, s2};

  af::ascir_op::Abs abs1("abs1");
  abs1.x = load1.y;
  abs1.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs1.y.dtype = ge::DT_INT64;
  *abs1.y.axis = {z0.id, z1.id, z2.id};
  *abs1.y.repeats = {s0, s1, s2};
  *abs1.y.strides = {s1 * s2 * s2, s2 * s2, s2};

  af::ascir_op::Store store1("store1");
  store1.x = abs1.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  store1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_INT64;
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, s1, One};
  *store1.y.strides = {s1, One, Zero};

  af::ascir_op::Output y1("y1");
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id};
  y1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y1.attr.api.type = af::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_INT64;

  for (auto n : graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(n)) {
      continue;
    }
    n->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  }
  // codegen pad算子暂未支持,先在ut/st中模拟整个流程
  AlignmentStrategyShadow handler;
  EXPECT_EQ(handler.AccessSetAlignWidth(graph), af::SUCCESS);
  EXPECT_EQ(handler.AccessAddRemovePadForTailAxisDiscontinuousLoad(graph), af::SUCCESS);
  for (const auto &node : graph.GetAllNodes()) {
    EXPECT_EQ(handler.AccessInferAlignmentForOneNode(node), af::SUCCESS);
  }
  EXPECT_EQ(handler.AccessAddPadForAlignmentConflictNode(graph), af::SUCCESS);

  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    EXPECT_EQ(handler.AccessSetVectorizedStridesForOneNode(node), af::SUCCESS);
  }

  std::vector<af::Expression> golden1 = {af::Symbol(160), af::Symbol(16)};
  std::vector<af::Expression> golden2 = {af::Symbol(16), af::Symbol(1)};
  auto load_node = graph.FindNode("load");
  ASSERT_NE(load_node, nullptr);
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, golden1);

  auto max_node = graph.FindNode("max");
  ASSERT_NE(max_node, nullptr);
  EXPECT_EQ(max_node->inputs[0].attr.vectorized_strides, golden2);

  std::vector<af::Expression> golden3 = {af::Symbol(40), af::Symbol(4)};
  auto load1_node = graph.FindNode("load1");
  ASSERT_NE(load1_node, nullptr);
  EXPECT_EQ(load1_node->outputs[0].attr.vectorized_strides, golden3);
}

TEST_F(VectorizedAlignmentUT, reduce_alignment) {
  af::AscGraph graph("reduce_align");
  auto s0 = af::Symbol(7);
  auto s1 = af::Symbol(8);
  auto s2 = af::Symbol(9);
  auto s3 = af::Symbol(10);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load");
  load.y.dtype = ge::DT_FLOAT16;
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.repeats = {s0, s1, s2, s3};
  *load.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  *load.y.vectorized_axis = {z1.id, z2.id, z3.id};

  af::ascir_op::Max max("max");
  max.y.dtype = ge::DT_FLOAT16;
  max.x = load.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *max.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *max.y.repeats = {s0, s1, One, s3};
  *max.y.strides = {s1 * s3, s3, Zero, One};
  max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  *max.y.vectorized_axis = {z1.id, z2.id, z3.id};

  af::ascir_op::Store store("store");
  store.y.dtype = ge::DT_FLOAT16;
  store.x = max.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.repeats = {s0, s1, One, s3};
  *store.y.strides = {s1 * s3, s3, Zero, One};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  *store.y.vectorized_axis = {z1.id, z2.id, z3.id};

  EXPECT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  auto s16 = af::Symbol(16);
  std::vector<af::Expression> load_golden_stride = {s2 * s16, s16, One};
  std::vector<af::Expression> golden_stride = {s16, Zero, One};

  auto load_node = graph.FindNode("load");
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, load_golden_stride);

  auto reduce_node = graph.FindNode("max");
  EXPECT_EQ(reduce_node->outputs[0].attr.vectorized_strides, golden_stride);

  auto store_node = graph.FindNode("store");
  EXPECT_EQ(store_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(VectorizedAlignmentUT, sibling_reduce_keeps_compact_output_alignment) {
  af::AscGraph graph("sibling_reduce_alignment");
  auto rows = af::Symbol(10);
  auto cols = af::Symbol(22);
  auto row_axis = graph.CreateAxis("row", rows);
  auto col_axis = graph.CreateAxis("col", cols);
  std::vector<af::AxisId> axes = {row_axis.id, col_axis.id};

  af::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load");
  load.x = data.y;
  SetTwoDimNodeAttr(load, ge::DT_FLOAT, af::ComputeType::kComputeLoad, axes, {rows, cols}, {cols, One});

  af::ascir_op::Relu relu("relu");
  relu.x = load.y;
  SetTwoDimNodeAttr(relu, ge::DT_FLOAT, af::ComputeType::kComputeElewise, axes, {rows, cols}, {cols, One});

  af::ascir_op::Scalar scalar("scalar", graph);
  scalar.ir_attr.SetValue("22.0");
  scalar.y.dtype = ge::DT_FLOAT;
  scalar.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Broadcast scalar_broadcast("scalar_broadcast");
  scalar_broadcast.x = scalar.y;
  SetTwoDimNodeAttr(scalar_broadcast, ge::DT_FLOAT, af::ComputeType::kComputeBroadcast, axes, {rows, One}, {One, Zero});

  af::ascir_op::Max max("max");
  max.x = relu.y;
  SetTwoDimNodeAttr(max, ge::DT_FLOAT, af::ComputeType::kComputeReduce, axes, {rows, One}, {One, Zero});

  af::ascir_op::Store max_store("max_store");
  max_store.x = max.y;
  SetTwoDimNodeAttr(max_store, ge::DT_FLOAT, af::ComputeType::kComputeStore, axes, {rows, One}, {One, Zero});

  af::ascir_op::Sum sum("sum");
  sum.x = relu.y;
  SetTwoDimNodeAttr(sum, ge::DT_FLOAT, af::ComputeType::kComputeReduce, axes, {rows, One}, {One, Zero});

  af::ascir_op::Store sum_store("sum_store");
  sum_store.x = sum.y;
  SetTwoDimNodeAttr(sum_store, ge::DT_FLOAT, af::ComputeType::kComputeStore, axes, {rows, One}, {One, Zero});

  af::ascir_op::TrueDiv div("div");
  div.x1 = sum.y;
  div.x2 = scalar_broadcast.y;
  SetTwoDimNodeAttr(div, ge::DT_FLOAT, af::ComputeType::kComputeElewise, axes, {rows, One}, {One, Zero});

  af::ascir_op::Store div_store("div_store");
  div_store.x = div.y;
  SetTwoDimNodeAttr(div_store, ge::DT_FLOAT, af::ComputeType::kComputeStore, axes, {rows, One}, {One, Zero});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);
  ExpectCompactReduceBranch(graph, "relu");
}

}  // namespace optimize
