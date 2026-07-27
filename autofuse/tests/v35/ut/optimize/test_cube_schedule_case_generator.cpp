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
#include "task_generator/cube_schedule_case_generator.h"

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

class CubeScheduleCaseGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    setenv("DUMP_GE_GRAPH", "2", false);
    setenv("DUMP_GRAPH_LEVEL", "1", false);
    setenv("DUMP_GRAPH_PATH", "/home/tl", false);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};

struct TwoDimGraphVars {
  af::Expression s0;
  af::Expression s1;
  int64_t z0_id;
  int64_t z1_id;
};

TwoDimGraphVars CreateTwoDimGraphVars(af::AscGraph &graph, int64_t dim0, int64_t dim1) {
  auto s0 = graph.CreateSizeVar(dim0);
  auto s1 = graph.CreateSizeVar(dim1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  return {s0, s1, z0.id, z1.id};
}

template <typename Op>
void SetSchedAxis2D(Op &op, const TwoDimGraphVars &vars) {
  op.attr.sched.axis = {vars.z0_id, vars.z1_id};
}

template <typename Tensor>
void SetTensor2D(Tensor &tensor, ge::DataType dtype, const TwoDimGraphVars &vars,
                 const std::vector<af::Expression> &strides, const std::vector<af::Expression> &repeats) {
  tensor.dtype = dtype;
  *tensor.axis = {vars.z0_id, vars.z1_id};
  *tensor.strides = strides;
  *tensor.repeats = repeats;
}

void InitData2D(Data &data, ge::DataType dtype, const TwoDimGraphVars &vars, const std::vector<af::Expression> &strides,
                const std::vector<af::Expression> &repeats, int64_t index) {
  SetSchedAxis2D(data, vars);
  SetTensor2D(data.y, dtype, vars, strides, repeats);
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.ir_attr.SetIndex(index);
}

template <typename Tensor>
void InitLoad2D(Load &load, const Tensor &input, ge::DataType dtype, const TwoDimGraphVars &vars,
                const std::vector<af::Expression> &strides, const std::vector<af::Expression> &repeats) {
  load.x = input;
  SetSchedAxis2D(load, vars);
  SetTensor2D(load.y, dtype, vars, strides, repeats);
}

template <typename Op, typename Lhs, typename Rhs>
void InitBinary2D(Op &op, const Lhs &lhs, const Rhs &rhs, const TwoDimGraphVars &vars) {
  SetSchedAxis2D(op, vars);
  op.x1 = lhs;
  op.x2 = rhs;
  SetTensor2D(op.y, af::DT_FLOAT, vars, {vars.s1, af::ops::One}, {vars.s0, vars.s1});
}

template <typename Lhs, typename Rhs>
void InitMatMul2D(MatMul &matmul, const Lhs &lhs, const Rhs &rhs, const TwoDimGraphVars &vars) {
  InitBinary2D(matmul, lhs, rhs, vars);
  matmul.attr.api.compute_type = af::ComputeType::kComputeCube;
}

template <typename Tensor>
void InitStoreOutput2D(Store &store_op, Output &output_op, const Tensor &input, const TwoDimGraphVars &vars) {
  SetSchedAxis2D(store_op, vars);
  store_op.x = input;
  SetTensor2D(store_op.y, af::DT_FLOAT, vars, {vars.s1, af::ops::One}, {vars.s0, vars.s1});

  output_op.x = store_op.y;
  output_op.y.dtype = af::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructJustMatMul(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = af::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = af::DT_FLOAT16;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = af::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.attr.api.compute_type = af::ComputeType::kComputeCube;
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = af::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, af::ops::One};
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = af::DT_FLOAT;
  store_op.attr.api.compute_type = af::ComputeType::kComputeStore;
  *store_op.y.strides = {s1, af::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(af::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = af::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructMatMulAndAdd(af::AscGraph &graph) {
  const auto vars = CreateTwoDimGraphVars(graph, 64, 64);

  Data data0("data0", graph);
  Load load0("load0");
  InitData2D(data0, af::DT_FLOAT, vars, {vars.s1, af::ops::One}, {vars.s0, vars.s1}, 0);
  InitLoad2D(load0, data0.y, af::DT_FLOAT, vars, {vars.s1, af::ops::One}, {vars.s0, vars.s1});

  Data data1("data1", graph);
  Load load1("load1");
  InitData2D(data1, af::DT_FLOAT, vars, {Zero, Zero}, {One, One}, 1);
  InitLoad2D(load1, data1.y, af::DT_FLOAT, vars, {Zero, Zero}, {One, One});

  Data data2("data2", graph);
  Load load2("load2");
  InitData2D(data2, af::DT_FLOAT, vars, {Zero, Zero}, {One, One}, 1);
  InitLoad2D(load2, data2.y, af::DT_FLOAT, vars, {vars.s1, af::ops::One}, {vars.s0, vars.s1});

  MatMul matmul("matmul");
  InitMatMul2D(matmul, load0.y, load1.y, vars);

  af::ascir_op::Add add_op("add");
  InitBinary2D(add_op, matmul.y, load2.y, vars);

  Store store_op("store");
  Output output_op("output");
  InitStoreOutput2D(store_op, output_op, add_op.y, vars);
}

size_t CountDataNodeByNameAndIndex(const ascir::ImplGraph &graph, const std::string &name, int64_t index) {
  size_t count = 0UL;
  for (const auto &node : graph.GetAllNodes()) {
    if (!af::ops::IsOps<af::ascir_op::Data>(node) || node->GetName() != name) {
      continue;
    }
    if (node->attr.ir_attr == nullptr) {
      continue;
    }
    auto ir_attr = node->attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
    if (ir_attr == nullptr) {
      continue;
    }
    int64_t data_index = -1;
    if (ir_attr->GetIndex(data_index) == af::SUCCESS && data_index == index) {
      ++count;
    }
  }
  return count;
}

size_t CountLoadsFromData(const ascir::ImplGraph &graph, const std::string &data_name) {
  size_t load_count = 0UL;
  for (const auto &node : graph.GetAllNodes()) {
    if (!af::ops::IsOps<af::ascir_op::Data>(node) || node->GetName() != data_name) {
      continue;
    }
    for (const auto &out_node : node->GetOutDataNodes()) {
      auto load_asc_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
      if (load_asc_node != nullptr && ScheduleUtils::IsLoad(load_asc_node)) {
        ++load_count;
      }
    }
  }
  return load_count;
}

void ConstructMatMulAddSharedInput(af::AscGraph &graph) {
  const auto vars = CreateTwoDimGraphVars(graph, 64, 64);

  Data data0("data0", graph);
  Load load0("load0");
  Load load2("load2");
  Load load3("load3");
  InitData2D(data0, af::DT_FLOAT, vars, {vars.s1, af::ops::One}, {vars.s0, vars.s1}, 0);
  InitLoad2D(load0, data0.y, af::DT_FLOAT, vars, {vars.s1, af::ops::One}, {vars.s0, vars.s1});
  InitLoad2D(load2, data0.y, af::DT_FLOAT, vars, {vars.s1, af::ops::One}, {vars.s0, vars.s1});
  InitLoad2D(load3, data0.y, af::DT_FLOAT, vars, {vars.s1, af::ops::One}, {vars.s0, vars.s1});

  Data data1("data1", graph);
  Load load1("load1");
  InitData2D(data1, af::DT_FLOAT, vars, {af::ops::Zero, af::ops::Zero}, {af::ops::One, af::ops::One}, 1);
  InitLoad2D(load1, data1.y, af::DT_FLOAT, vars, {af::ops::Zero, af::ops::Zero}, {af::ops::One, af::ops::One});

  MatMul matmul("matmul");
  InitMatMul2D(matmul, load0.y, load1.y, vars);

  af::ascir_op::Add add_op("add");
  InitBinary2D(add_op, matmul.y, load2.y, vars);

  af::ascir_op::Mul mul_op("mul");
  InitBinary2D(mul_op, add_op.y, load3.y, vars);

  Store store_op("store");
  Output output_op("output");
  InitStoreOutput2D(store_op, output_op, mul_op.y, vars);
}

void ConstructJustMatMulBias(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = af::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = af::DT_FLOAT16;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = af::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  data2.y.dtype = af::DT_FLOAT;
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data2.y.strides = {af::ops::Zero, af::ops::One};
  *data2.y.repeats = {af::ops::One, s1};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id};
  load2.y.dtype = af::DT_FLOAT;
  *load2.y.strides = {af::ops::Zero, af::ops::One};
  *load2.y.repeats = {af::ops::One, s1};

  MatMulBias matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.bias = load2.y;
  matmul.attr.api.compute_type = af::ComputeType::kComputeCube;
  matmul.y.dtype = af::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, af::ops::One};
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  store_op.attr.api.compute_type = af::ComputeType::kComputeStore;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = af::DT_FLOAT;
  *store_op.y.strides = {s1, af::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(af::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = af::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

TEST_F(CubeScheduleCaseGeneratorTest, Test_Just_Matmul_Store) {
  af::AscGraph graph("just_matmul");
  ConstructJustMatMul(graph);
  std::vector<ScheduleTask> tasks;
  optimize::CubeFusionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 1UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
}

TEST_F(CubeScheduleCaseGeneratorTest, Test_MatMul_Add_Store) {
  af::AscGraph graph("matmul_add_store");
  ConstructMatMulAndAdd(graph);
  std::vector<ScheduleTask> tasks;
  optimize::CubeFusionCaseGenerator generator;
  OptimizerOptions options;
  generator.GeneratorTask(graph, tasks, options);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
}

TEST_F(CubeScheduleCaseGeneratorTest, Test_MatMul_Add_Shared_Data_Split) {
  af::AscGraph graph("matmul_add_shared_data_split");
  ConstructMatMulAddSharedInput(graph);
  std::vector<ScheduleTask> tasks;
  optimize::CubeFusionCaseGenerator generator;
  OptimizerOptions options;
  ASSERT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);

  size_t data0_graph_count = 0UL;
  size_t data0_cube_graph_count = 0UL;
  size_t data0_vector_graph_load_count = 0UL;
  for (const auto &grouped_graph : tasks[0].grouped_graphs) {
    const auto count = CountDataNodeByNameAndIndex(grouped_graph, "data0", 0);
    if (count == 0UL) {
      continue;
    }
    ++data0_graph_count;
    if (ScheduleUtils::HasComputeType(grouped_graph, af::ComputeType::kComputeCube)) {
      ++data0_cube_graph_count;
    } else {
      data0_vector_graph_load_count = CountLoadsFromData(grouped_graph, "data0");
    }
  }
  EXPECT_EQ(data0_graph_count, 2UL);
  EXPECT_EQ(data0_cube_graph_count, 1UL);
  EXPECT_EQ(data0_vector_graph_load_count, 2UL);
}

TEST_F(CubeScheduleCaseGeneratorTest, Test_MatMul_Bias_Store) {
  af::AscGraph graph("matmul_bias_store");
  ConstructJustMatMulBias(graph);
  std::vector<ScheduleTask> tasks;
  optimize::CubeFusionCaseGenerator generator;
  OptimizerOptions options;
  generator.GeneratorTask(graph, tasks, options);
  ASSERT_EQ(tasks.size(), 1UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
}
}  // namespace schedule
