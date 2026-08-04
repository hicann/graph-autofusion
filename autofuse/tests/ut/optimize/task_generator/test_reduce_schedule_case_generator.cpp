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
#include "ascir_ops_utils.h"
#include "ascir_utils.h"
#include "asc_graph_utils.h"
#include "ascir_ops.h"
#include "task_generator/schedule_case_generator.h"
#include "task_generator/reduce_schedule_case_generator.h"
#include "asc_graph_builder.h"

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

class ReduceScheduleCaseGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }

  static std::string ExpressToStr(std::vector<af::Expression> exprs) {
    std::stringstream ss;
    for (auto &size_expr : exprs) {
      ss << std::string(size_expr.Str().get()) << ", ";
    }
    return ss.str();
  }

  static std::string RepeatsToStr(const af::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    return ExpressToStr(node->outputs[0].attr.repeats);
  }

  static std::string StridesToStr(const af::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    return ExpressToStr(node->outputs[0].attr.strides);
  }

  static std::string AxisToStr(af::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    std::stringstream ss;
    for (auto axis_id : node->outputs[0].attr.axis) {
      ss << graph.FindAxis(axis_id)->name << ", ";
    }
    return ss.str();
  }
};

bool HasReduceTemplateType(const std::vector<ScheduleTask> &tasks, ReduceTemplateType reduce_type) {
  for (const auto &task : tasks) {
    if (task.reduce_type == reduce_type) {
      return true;
    }
  }
  return false;
}

void ConstructFourAxisReduce(af::AscGraph &graph, bool tail_reduce) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto s2 = graph.CreateSizeVar(32);
  auto s3 = graph.CreateSizeVar(16);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  std::vector<af::AxisId> axes = {z0.id, z1.id, z2.id, z3.id};

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = axes;
  load.x = data.y;
  *load.y.axis = axes;
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1 * s2 * s3, s2 * s3, s3, af::ops::One};
  *load.y.repeats = {s0, s1, s2, s3};

  Sum sum("sum");
  sum.attr.sched.axis = axes;
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.x = load.y;
  *sum.y.axis = axes;
  sum.y.dtype = ge::DT_FLOAT;
  if (tail_reduce) {
    *sum.y.repeats = {s0, s1, s2, af::ops::One};
    *sum.y.strides = {s1 * s2, s2, af::ops::One, af::ops::Zero};
  } else {
    *sum.y.repeats = {s0, s1, s2, s3};
    *sum.y.strides = {s1 * s2 * s3, s2 * s3, s3, af::ops::One};
  }

  Store store_op("store");
  store_op.attr.sched.axis = axes;
  store_op.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op.x = sum.y;
  *store_op.y.axis = *sum.y.axis;
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = *sum.y.strides;
  *store_op.y.repeats = *sum.y.repeats;

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructEmptyRepeatReduce(af::AscGraph &graph) {
  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.x = data.y;
  load.y.dtype = ge::DT_FLOAT;

  Sum sum("sum");
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.x = load.y;
  sum.y.dtype = ge::DT_FLOAT;

  Store store_op("store");
  store_op.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op.x = sum.y;
  store_op.y.dtype = ge::DT_FLOAT;

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructFourAxisSoftmax(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto s2 = graph.CreateSizeVar(32);
  auto s3 = graph.CreateSizeVar(16);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  std::vector<af::AxisId> axes = {z0.id, z1.id, z2.id, z3.id};

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = axes;
  load.x = data.y;
  *load.y.axis = axes;
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1 * s2 * s3, s2 * s3, s3, af::ops::One};
  *load.y.repeats = {s0, s1, s2, s3};

  Softmax softmax("softmax");
  softmax.attr.sched.axis = axes;
  softmax.attr.api.compute_type = af::ComputeType::kComputeReduce;
  softmax.x = load.y;
  *softmax.y.axis = axes;
  softmax.y.dtype = ge::DT_FLOAT;
  *softmax.y.repeats = {s0, s1, s2, s3};
  *softmax.y.strides = {s1 * s2 * s3, s2 * s3, s3, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = axes;
  store_op.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op.x = softmax.y;
  *store_op.y.axis = *softmax.y.axis;
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = *softmax.y.strides;
  *store_op.y.repeats = *softmax.y.repeats;

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct3Elewise(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {af::ops::One, af::ops::One};
  *sum.y.strides = {af::ops::Zero, af::ops::Zero};

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {af::ops::Zero, af::ops::Zero};
  *abs.y.repeats = {af::ops::One, af::ops::One};

  Exp exp("exp");
  exp.x = abs.y;
  exp.attr.sched.axis = {z0.id, z1.id};
  exp.attr.api.compute_type = af::ComputeType::kComputeElewise;
  *exp.y.axis = {z0.id, z1.id};
  exp.y.dtype = ge::DT_FLOAT;
  *exp.y.strides = {af::ops::Zero, af::ops::Zero};
  *exp.y.repeats = {af::ops::One, af::ops::One};

  af::ascir_op::Relu b0_relu("b0_relu");
  b0_relu.x = exp.y;
  b0_relu.attr.sched.axis = {z0.id, z1.id};
  b0_relu.attr.api.compute_type = af::ComputeType::kComputeElewise;
  b0_relu.y.dtype = ge::DT_FLOAT;
  *b0_relu.y.axis = {z0.id, z1.id};
  *b0_relu.y.repeats = {af::ops::One, af::ops::One};
  *b0_relu.y.strides = {af::ops::Zero, af::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op1.x = b0_relu.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {af::ops::Zero, af::ops::Zero};
  *store_op1.y.repeats = {af::ops::One, af::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct1Elewise(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {af::ops::One, af::ops::One};
  *sum.y.strides = {af::ops::Zero, af::ops::Zero};

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {af::ops::Zero, af::ops::Zero};
  *abs.y.repeats = {af::ops::One, af::ops::One};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op1.x = abs.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {af::ops::Zero, af::ops::Zero};
  *store_op1.y.repeats = {af::ops::One, af::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStructMultiplyCitations(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.x = data.y;
  *load1.y.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.strides = {s1, af::ops::One};
  *load1.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {af::ops::One, af::ops::One};
  *sum.y.strides = {af::ops::Zero, af::ops::Zero};

  Add add("add");
  add.x1 = sum.y;
  add.x2 = load1.y;
  add.attr.sched.axis = {z0.id, z1.id};
  add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  *add.y.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.strides = {s1, One};
  *add.y.repeats = {s0, s1};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op1.x = add.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {s1, One};
  *store_op1.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct4Elewise(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {af::ops::One, af::ops::One};
  *sum.y.strides = {af::ops::Zero, af::ops::Zero};

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {af::ops::Zero, af::ops::Zero};
  *abs.y.repeats = {af::ops::One, af::ops::One};

  Tanh tanh("tanh");
  tanh.x = abs.y;
  tanh.attr.sched.axis = {z0.id, z1.id};
  tanh.attr.api.compute_type = af::ComputeType::kComputeElewise;
  *tanh.y.axis = {z0.id, z1.id};
  tanh.y.dtype = ge::DT_FLOAT;
  *tanh.y.strides = {af::ops::Zero, af::ops::Zero};
  *tanh.y.repeats = {af::ops::One, af::ops::One};

  Exp exp("exp");
  exp.x = tanh.y;
  exp.attr.sched.axis = {z0.id, z1.id};
  exp.attr.api.compute_type = af::ComputeType::kComputeElewise;
  *exp.y.axis = {z0.id, z1.id};
  exp.y.dtype = ge::DT_FLOAT;
  *exp.y.strides = {af::ops::Zero, af::ops::Zero};
  *exp.y.repeats = {af::ops::One, af::ops::One};

  af::ascir_op::Relu b0_relu("b0_relu");
  b0_relu.x = exp.y;
  b0_relu.attr.sched.axis = {z0.id, z1.id};
  b0_relu.attr.api.compute_type = af::ComputeType::kComputeElewise;
  b0_relu.y.dtype = ge::DT_FLOAT;
  *b0_relu.y.axis = {z0.id, z1.id};
  *b0_relu.y.repeats = {af::ops::One, af::ops::One};
  *b0_relu.y.strides = {af::ops::Zero, af::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op1.x = b0_relu.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {af::ops::Zero, af::ops::Zero};
  *store_op1.y.repeats = {af::ops::One, af::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_Three_Elewise_Store) {
  af::AscGraph graph("reduce_Three_elewise_store");
  ConstructNormStruct3Elewise(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_One_Elewise_Store) {
  af::AscGraph graph("reduce_one_elewise_store");
  ConstructNormStruct1Elewise(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  generator.GeneratorTask(graph, tasks, options);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_Four_Elewise_Store) {
  af::AscGraph graph("reduce_four_elewise_store");
  ConstructNormStruct4Elewise(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  generator.GeneratorTask(graph, tasks, options);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 3UL);
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_AllLoadTailReduceFourAxis) {
  af::AscGraph graph("reduce_all_load_tail_reduce_four_axis");
  ConstructFourAxisReduce(graph, true);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  EXPECT_TRUE(HasReduceTemplateType(tasks, ReduceTemplateType::kAllLoad));
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_NoAllLoadForFourAxisNonTailReduce) {
  af::AscGraph graph("reduce_no_all_load_for_four_axis_non_tail_reduce");
  ConstructFourAxisReduce(graph, false);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  EXPECT_FALSE(HasReduceTemplateType(tasks, ReduceTemplateType::kAllLoad));
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_NoAllLoadForEmptyRepeats) {
  af::AscGraph graph("reduce_no_all_load_for_empty_repeats");
  ConstructEmptyRepeatReduce(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  EXPECT_FALSE(HasReduceTemplateType(tasks, ReduceTemplateType::kAllLoad));
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_AllLoadForSoftmaxWithUnchangedTailRepeat) {
  af::AscGraph graph("reduce_all_load_for_softmax_with_unchanged_tail_repeat");
  ConstructFourAxisSoftmax(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  EXPECT_TRUE(HasReduceTemplateType(tasks, ReduceTemplateType::kAllLoad));
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_Multi_Cita_Store) {
  af::AscGraph graph("reduce_multi_citation_store");
  ConstructNormStructMultiplyCitations(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  generator.GeneratorTask(graph, tasks, options);
  ASSERT_EQ(tasks.size(), 0UL);
}

void ConstructReduceWithScalarData(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  *sum.y.repeats = {af::ops::One, af::ops::One};
  *sum.y.strides = {af::ops::Zero, af::ops::Zero};

  af::ascir_op::ScalarData scalar_data("scalar_data", graph);
  scalar_data.ir_attr.SetIndex(1);

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.strides = {af::ops::Zero, af::ops::Zero};
  *abs.y.repeats = {af::ops::One, af::ops::One};

  Add add("add");
  add.x1 = abs.y;
  add.x2 = scalar_data.y;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  *add.y.strides = {af::ops::Zero, af::ops::Zero};
  *add.y.repeats = {af::ops::One, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = add.y;
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.strides = {af::ops::Zero, af::ops::Zero};
  *store_op.y.repeats = {af::ops::One, af::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.ir_attr.SetIndex(0);
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_ScalarData_Input) {
  af::AscGraph graph("reduce_scalar_data_input");
  ConstructReduceWithScalarData(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  // ScalarData直连算子，reduce拆分时ScalarData走PartitionScalar分支，保持ScalarData类型
  for (const auto &task : tasks) {
    for (const auto &grouped_graph : task.grouped_graphs) {
      auto scalar_data_node = grouped_graph.FindNode("copy_from_scalar_data");
      if (scalar_data_node != nullptr) {
        EXPECT_EQ(std::string(scalar_data_node->GetTypePtr()), "ScalarData");
      }
    }
  }
}
}  // namespace schedule
