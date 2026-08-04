/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "task_generator/reduce_schedule_case_generator.h"

using namespace af::ascir_op;

namespace optimize {
namespace {
bool HasReduceTemplateType(const std::vector<ScheduleTask> &tasks, ReduceTemplateType reduce_type) {
  for (const auto &task : tasks) {
    if (task.reduce_type == reduce_type) {
      return true;
    }
  }
  return false;
}

std::vector<af::AxisId> CreateFourAxes(af::AscGraph &graph, std::vector<af::Expression> &sizes) {
  sizes = {graph.CreateSizeVar(128), graph.CreateSizeVar(64), graph.CreateSizeVar(32), graph.CreateSizeVar(16)};
  auto z0 = graph.CreateAxis("z0", sizes[0]);
  auto z1 = graph.CreateAxis("z1", sizes[1]);
  auto z2 = graph.CreateAxis("z2", sizes[2]);
  auto z3 = graph.CreateAxis("z3", sizes[3]);
  return {z0.id, z1.id, z2.id, z3.id};
}

void InitFourAxisLoad(Load &load, af::AscGraph &graph, const std::vector<af::AxisId> &axes,
                      const std::vector<af::Expression> &sizes) {
  Data data("data", graph);
  data.y.dtype = af::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  load.attr.sched.axis = axes;
  load.x = data.y;
  *load.y.axis = axes;
  load.y.dtype = af::DT_FLOAT;
  *load.y.strides = {sizes[1] * sizes[2] * sizes[3], sizes[2] * sizes[3], sizes[3], af::ops::One};
  *load.y.repeats = sizes;
}

af::AscGraph ConstructReduceTailFullLoadFourAxis(const std::string &name) {
  af::AscGraph graph(name.c_str());
  std::vector<af::Expression> sizes;
  const auto axes = CreateFourAxes(graph, sizes);
  Load load("load");
  InitFourAxisLoad(load, graph, axes, sizes);

  Sum sum("sum");
  sum.attr.sched.axis = axes;
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.x = load.y;
  *sum.y.axis = axes;
  sum.y.dtype = af::DT_FLOAT;
  *sum.y.repeats = {sizes[0], sizes[1], sizes[2], af::ops::One};
  *sum.y.strides = {sizes[1] * sizes[2], sizes[2], af::ops::One, af::ops::Zero};

  Store store_op("store");
  store_op.attr.sched.axis = axes;
  store_op.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op.x = sum.y;
  *store_op.y.axis = *sum.y.axis;
  store_op.y.dtype = af::DT_FLOAT;
  *store_op.y.strides = *sum.y.strides;
  *store_op.y.repeats = *sum.y.repeats;

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = af::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  return graph;
}

af::AscGraph ConstructSoftmaxUnchangedTailRepeatFourAxis(const std::string &name) {
  af::AscGraph graph(name.c_str());
  std::vector<af::Expression> sizes;
  const auto axes = CreateFourAxes(graph, sizes);
  Load load("load");
  InitFourAxisLoad(load, graph, axes, sizes);

  Softmax softmax("softmax");
  softmax.attr.sched.axis = axes;
  softmax.attr.api.compute_type = af::ComputeType::kComputeReduce;
  softmax.x = load.y;
  *softmax.y.axis = axes;
  softmax.y.dtype = af::DT_FLOAT;
  *softmax.y.strides = *load.y.strides;
  *softmax.y.repeats = *load.y.repeats;

  Store store_op("store");
  store_op.attr.sched.axis = axes;
  store_op.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op.x = softmax.y;
  *store_op.y.axis = *softmax.y.axis;
  store_op.y.dtype = af::DT_FLOAT;
  *store_op.y.strides = *softmax.y.strides;
  *store_op.y.repeats = *softmax.y.repeats;

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = af::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  return graph;
}
}  // namespace

TEST(ReduceScheduleCaseGeneratorSt, TestReduce_AllLoadTailReduceFourAxis) {
  auto graph = ConstructReduceTailFullLoadFourAxis("REDUCE_TAIL_FULL_LOAD_FOUR_AXIS");
  std::vector<ScheduleTask> tasks;
  ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  EXPECT_TRUE(HasReduceTemplateType(tasks, ReduceTemplateType::kAllLoad));
}

TEST(ReduceScheduleCaseGeneratorSt, TestReduce_AllLoadSoftmaxUnchangedTailRepeat) {
  auto graph = ConstructSoftmaxUnchangedTailRepeatFourAxis("REDUCE_SOFTMAX_UNCHANGED_TAIL_REPEAT");
  std::vector<ScheduleTask> tasks;
  ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  EXPECT_TRUE(HasReduceTemplateType(tasks, ReduceTemplateType::kAllLoad));
}
}  // namespace optimize
