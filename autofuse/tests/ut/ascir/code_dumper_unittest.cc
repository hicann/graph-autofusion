/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "ascir_ops.h"
#include "graph/symbolizer/symbolic.h"
#include "expression/const_values.h"

#include "ascendc_ir.h"
#include "ascend_graph_code_dumper.h"
#include "ascendc_graph_txt_dumper.h"

#include "graph_utils.h"
#include "ascendc_ir/utils/asc_graph_utils.h"

static af::Expression One = af::Symbol(1);

class AscendGraphCodeDumperUT : public testing::Test {
 protected:
  void SetUp() {
    dlog_setlevel(0, 0, 0);
  }

  void TearDown() {
    dlog_setlevel(0, 3, 0);
  }
};

namespace af {
namespace {
std::string ReadFileContent(const std::string &filePath) {
  std::ifstream file(filePath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filePath << std::endl;
    return "";
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

void ConstructAscGraph(AscGraph &graph) {
  Expression s0 = graph.CreateSizeVar("s0");
  // 也可以创建常量，跟var成图的方式不太一样
  Expression s1 = graph.CreateSizeVar(2);

  Axis &s0_axis = graph.CreateAxis("S0", s0);
  Axis &s1_axis = graph.CreateAxis("S1", s1);

  ascir_op::Scalar scalar("Scalar", graph);
  scalar.attr.sched.axis = {s0_axis.id, s1_axis.id};
  scalar.y.dtype = af::DT_FLOAT16;
  scalar.y.format = af::FORMAT_ND;
  *scalar.y.axis = {s0_axis.id, s1_axis.id};
  *scalar.y.repeats = {s0, s1};
  *scalar.y.strides = {s1, sym::kSymbolOne};
  scalar.ir_attr.SetValue("Test");

  ascir_op::Data data1("data1", graph);
  data1.attr.sched.axis = {s0_axis.id, s1_axis.id};
  data1.y.dtype = af::DT_FLOAT16;
  data1.y.format = af::FORMAT_ND;
  *data1.y.axis = {s0_axis.id, s1_axis.id};
  *data1.y.repeats = {s0, s1};
  *data1.y.strides = {s1, sym::kSymbolOne};
  data1.ir_attr.SetIndex(2);

  ascir_op::Data data2("data2", graph);
  data2.attr.sched.axis = {s0_axis.id, s1_axis.id};
  data2.y.dtype = af::DT_FLOAT16;
  data2.y.format = af::FORMAT_ND;
  *data2.y.axis = {s0_axis.id, s1_axis.id};
  *data2.y.repeats = {s0, s1};
  *data2.y.strides = {s1, sym::kSymbolOne};
  data2.ir_attr.SetIndex(3);

  ascir_op::Add add("add");
  add.x1 = scalar.y;
  add.x2 = scalar.y;
  add.attr.sched.axis = {s0_axis.id, s1_axis.id};
  add.y.dtype = af::DT_FLOAT16;
  add.y.format = af::FORMAT_ND;
  *add.y.axis = {s0_axis.id, s1_axis.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, sym::kSymbolOne};

  ascir_op::Exp exp("exp");
  exp.x = scalar.y;
  exp.attr.sched.axis = {s0_axis.id, s1_axis.id};
  exp.y.dtype = af::DT_FLOAT16;
  exp.y.format = af::FORMAT_ND;
  *exp.y.axis = {s0_axis.id, s1_axis.id};
  *exp.y.repeats = {s0, s1};
  *exp.y.strides = {s1, sym::kSymbolOne};

  ascir_op::Concat concat("concat");
  concat.x = {add.y, exp.y};
  concat.attr.sched.axis = {s0_axis.id, s1_axis.id};
  concat.y.dtype = af::DT_FLOAT16;
  concat.y.format = af::FORMAT_ND;
  *concat.y.axis = {s0_axis.id, s1_axis.id};
  *concat.y.repeats = {s0, s1 * af::Symbol(2)};
  *concat.y.strides = {s1 * af::Symbol(2), sym::kSymbolOne};

  ascir_op::Select fake_opa("select");
  fake_opa.x1 = exp.y;
  fake_opa.x2 = exp.y;  // 可选输入
  fake_opa.x3 = data1.y;
  fake_opa.attr.sched.axis = {s0_axis.id, s1_axis.id};
  fake_opa.y.dtype = af::DT_FLOAT16;
  fake_opa.y.format = af::FORMAT_ND;
  *fake_opa.y.axis = {s0_axis.id, s1_axis.id};
  *fake_opa.y.repeats = {s0, s1 * af::Symbol(2)};
  *fake_opa.y.strides = {s1 * af::Symbol(2), sym::kSymbolOne};

  ascir_op::LeakyRelu leaky_relu("leaky_relu");
  leaky_relu.x = scalar.y;
  leaky_relu.ir_attr.SetNegative_slope(1);

  ascir_op::Output output("output");
  output.x = fake_opa.y;
  output.ir_attr.SetIndex(1);
}
}  // namespace
TEST_F(AscendGraphCodeDumperUT, test_python_gen) {
  AscGraph graph("test");
  ConstructAscGraph(graph);
  af::ascir::PythonCodeDumper dumper;
  EXPECT_EQ(dumper.Dump(graph, "./asc_graph_python.py"), af::SUCCESS);
  const std::string expected_graph_code = R"(# Python code to construct AscGraph
from autofuse.pyautofuse import ascir
from autofuse.pyautofuse import Autofuser, AutofuserOptions

graph = ascir.HintGraph("test")
s0 = graph.create_size("s0")
S0 = graph.create_axis("S0", s0)
S1 = graph.create_axis("S1", 2)
Scalar_0 = ascir.ops.Scalar("Scalar", graph)
Scalar_0.attr.sched.axis = [S0, S1]
Scalar_0.y.dtype = ascir.dtypes.float16
Scalar_0.y.axis = [S0, S1]
Scalar_0.y.size = [s0, 2]
Scalar_0.y.strides = [2, 1]
Scalar_0.attr.ir_attr.value = 'Test'
Data_0 = ascir.ops.Data("data1", graph)
Data_0.attr.sched.axis = [S0, S1]
Data_0.y.dtype = ascir.dtypes.float16
Data_0.y.axis = [S0, S1]
Data_0.y.size = [s0, 2]
Data_0.y.strides = [2, 1]
Data_0.attr.ir_attr.index = 2
Data_1 = ascir.ops.Data("data2", graph)
Data_1.attr.sched.axis = [S0, S1]
Data_1.y.dtype = ascir.dtypes.float16
Data_1.y.axis = [S0, S1]
Data_1.y.size = [s0, 2]
Data_1.y.strides = [2, 1]
Data_1.attr.ir_attr.index = 3
Add_0 = ascir.ops.Add("add")
Add_0.attr.sched.axis = [S0, S1]
Add_0.x1 = Scalar_0.y
Add_0.x2 = Scalar_0.y
Add_0.y.dtype = ascir.dtypes.float16
Add_0.y.axis = [S0, S1]
Add_0.y.size = [s0, 2]
Add_0.y.strides = [2, 1]
Exp_0 = ascir.ops.Exp("exp")
Exp_0.attr.sched.axis = [S0, S1]
Exp_0.x = Scalar_0.y
Exp_0.y.dtype = ascir.dtypes.float16
Exp_0.y.axis = [S0, S1]
Exp_0.y.size = [s0, 2]
Exp_0.y.strides = [2, 1]
Concat_0 = ascir.ops.Concat("concat")
Concat_0.attr.sched.axis = [S0, S1]
Concat_0.x = [Add_0.y, Exp_0.y]
Concat_0.y.dtype = ascir.dtypes.float16
Concat_0.y.axis = [S0, S1]
Concat_0.y.size = [s0, 4]
Concat_0.y.strides = [4, 1]
Select_0 = ascir.ops.Select("select")
Select_0.attr.sched.axis = [S0, S1]
Select_0.x1 = Exp_0.y
Select_0.x2 = Exp_0.y
Select_0.x3 = Data_0.y
Select_0.y.dtype = ascir.dtypes.float16
Select_0.y.axis = [S0, S1]
Select_0.y.size = [s0, 4]
Select_0.y.strides = [4, 1]
LeakyRelu_0 = ascir.ops.LeakyRelu("leaky_relu")
LeakyRelu_0.x = Scalar_0.y
LeakyRelu_0.y.dtype = ascir.dtypes.float32
LeakyRelu_0.attr.ir_attr.negative_slope = 1.000000
Output_0 = ascir.ops.Output("output")
Output_0.x = Select_0.y
Output_0.y.dtype = ascir.dtypes.float32
Output_0.attr.ir_attr.index = 1
fuser = Autofuser(AutofuserOptions())
schedule_results = fuser.schedule(graph)
tiling_def, host_impl, device_impl = fuser.codegen(schedule_results)
)";
  EXPECT_EQ(expected_graph_code, ReadFileContent("./asc_graph_python.py"));
}
void ConstructWrongAscGraph(AscGraph &graph) {
  // 构造环路图
  Expression s0 = graph.CreateSizeVar("s0");
  Expression s1 = graph.CreateSizeVar(2);

  Axis &s0_axis = graph.CreateAxis("S0", s0);
  Axis &s1_axis = graph.CreateAxis("S1", s1);

  ascir_op::Data data("data", graph);
  data.attr.sched.axis = {s0_axis.id, s1_axis.id};
  data.y.dtype = af::DT_FLOAT16;
  data.y.format = af::FORMAT_ND;
  *data.y.axis = {s0_axis.id, s1_axis.id};
  *data.y.repeats = {s0, s1};
  *data.y.strides = {s1, sym::kSymbolOne};
  data.ir_attr.SetIndex(1);

  ascir_op::Add add("add");
  add.x1 = data.y;
  add.attr.sched.axis = {s0_axis.id, s1_axis.id};
  add.y.dtype = af::DT_FLOAT16;
  add.y.format = af::FORMAT_ND;
  *add.y.axis = {s0_axis.id, s1_axis.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, sym::kSymbolOne};

  ascir_op::Exp exp("exp");
  exp.x = add.y;
  exp.attr.sched.axis = {s0_axis.id, s1_axis.id};
  exp.y.dtype = af::DT_FLOAT16;
  exp.y.format = af::FORMAT_ND;
  *exp.y.axis = {s0_axis.id, s1_axis.id};
  *exp.y.repeats = {s0, s1};
  *exp.y.strides = {s1, sym::kSymbolOne};

  add.x2 = exp.y;

  ascir_op::Output output("output");
  output.x = add.y;
  output.ir_attr.SetIndex(1);
}
TEST_F(AscendGraphCodeDumperUT, test_wrong_python_gen) {
  AscGraph graph("wrong_graph");
  ConstructWrongAscGraph(graph);
  af::ascir::PythonCodeDumper dumper;
  EXPECT_EQ(dumper.Dump(graph, "./asc_wrong_graph_python.py"), af::SUCCESS);
  const std::string expected_graph_code = R"(# Python code to construct AscGraph
from autofuse.pyautofuse import ascir
from autofuse.pyautofuse import Autofuser, AutofuserOptions

graph = ascir.HintGraph("wrong_graph")
s0 = graph.create_size("s0")
S0 = graph.create_axis("S0", s0)
S1 = graph.create_axis("S1", 2)
Data_0 = ascir.ops.Data("data", graph)
Data_0.attr.sched.axis = [S0, S1]
Data_0.y.dtype = ascir.dtypes.float16
Data_0.y.axis = [S0, S1]
Data_0.y.size = [s0, 2]
Data_0.y.strides = [2, 1]
Data_0.attr.ir_attr.index = 1
Add_0 = ascir.ops.Add("add")
Add_0.attr.sched.axis = [S0, S1]
Add_0.x1 = Data_0.y
Add_0.x2 = .y
Add_0.y.dtype = ascir.dtypes.float16
Add_0.y.axis = [S0, S1]
Add_0.y.size = [s0, 2]
Add_0.y.strides = [2, 1]
Exp_0 = ascir.ops.Exp("exp")
Exp_0.attr.sched.axis = [S0, S1]
Exp_0.x = Add_0.y
Exp_0.y.dtype = ascir.dtypes.float16
Exp_0.y.axis = [S0, S1]
Exp_0.y.size = [s0, 2]
Exp_0.y.strides = [2, 1]
Output_0 = ascir.ops.Output("output")
Output_0.x = Add_0.y
Output_0.y.dtype = ascir.dtypes.float32
Output_0.attr.ir_attr.index = 1
fuser = Autofuser(AutofuserOptions())
schedule_results = fuser.schedule(graph)
tiling_def, host_impl, device_impl = fuser.codegen(schedule_results)
)";
  EXPECT_EQ(expected_graph_code, ReadFileContent("./asc_wrong_graph_python.py"));
}

void CreateAscBackendGraph(std::shared_ptr<AscGraph> &graph, const std::string &prefix, int64_t axis_num = 2) {
  auto ONE = Symbol(1);
  std::vector<int64_t> axis_ids;
  std::vector<af::Expression> repeats;
  for (int64_t i = 0; i < axis_num; ++i) {
    const Expression exp = graph->CreateSizeVar("s" + std::to_string(i));
    auto axis = graph->CreateAxis("z" + std::to_string(i), exp);
    axis_ids.push_back(i);
    repeats.push_back(exp);
  }

  std::vector<af::Expression> strides(repeats.size(), One);
  if (axis_num > 1) {
    for (int64_t i = axis_num - 2; i >= 0; --i) {
      strides[i] = repeats[i + 1] * strides[i + 1];
    }
  }

  af::ascir_op::Data data(std::string(prefix + "_data").c_str(), *graph);
  data.attr.sched.axis = axis_ids;
  *data.y.axis = axis_ids;
  *data.y.repeats = repeats;
  *data.y.strides = strides;
  data.ir_attr.SetIndex(0);
  data.y.dtype = af::DT_INT8;
  data.ir_attr.SetIndex(0);

  af::ascir_op::Load load(std::string(prefix + "_load").c_str());
  load.x = data.y;
  load.attr.sched.axis = axis_ids;
  *load.y.axis = axis_ids;
  *load.y.repeats = repeats;
  *load.y.strides = strides;
  load.ir_attr.SetOffset(Expression(Symbol(1024)));

  af::ascir_op::Abs abs(std::string(prefix + "_abs").c_str());
  abs.x = load.y;
  abs.attr.sched.axis = axis_ids;
  *abs.y.axis = axis_ids;
  *abs.y.repeats = repeats;
  *abs.y.strides = strides;

  af::ascir_op::Store store(std::string(prefix + "_store").c_str());
  store.x = abs.y;
  store.attr.sched.axis = axis_ids;
  *store.y.axis = axis_ids;
  *store.y.repeats = repeats;
  *store.y.strides = strides;

  af::ascir_op::Output y(std::string(prefix + "_out").c_str());
  y.x = store.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DT_FLOAT16;
  y.ir_attr.SetIndex(0);
}

void CreateAscBackendGraphTwoInTwoOut(std::shared_ptr<AscGraph> &graph, const std::string &prefix,
                                      int64_t axis_num = 2) {
  auto ONE = Symbol(1);
  std::vector<int64_t> axis_ids;
  std::vector<af::Expression> repeats;
  for (int64_t i = 0; i < axis_num; ++i) {
    const Expression exp = graph->CreateSizeVar("s" + std::to_string(i));
    auto axis = graph->CreateAxis("z" + std::to_string(i), exp);
    axis_ids.push_back(i);
    repeats.push_back(exp);
  }

  std::vector<af::Expression> strides(repeats.size(), One);
  if (axis_num > 1) {
    for (int64_t i = axis_num - 2; i >= 0; --i) {
      strides[i] = repeats[i + 1] * strides[i + 1];
    }
  }

  af::ascir_op::Data data0(std::string(prefix + "_data0").c_str(), *graph);
  data0.attr.sched.axis = axis_ids;
  *data0.y.axis = axis_ids;
  *data0.y.repeats = repeats;
  *data0.y.strides = strides;
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = af::DT_INT8;
  data0.ir_attr.SetIndex(3);

  af::ascir_op::Load load0(std::string(prefix + "_load0").c_str());
  load0.x = data0.y;
  load0.attr.sched.axis = axis_ids;
  *load0.y.axis = axis_ids;
  *load0.y.repeats = repeats;
  *load0.y.strides = strides;

  af::ascir_op::Data data1(std::string(prefix + "_data1").c_str(), *graph);
  data1.attr.sched.axis = axis_ids;
  *data1.y.axis = axis_ids;
  *data1.y.repeats = repeats;
  *data1.y.strides = strides;
  data1.ir_attr.SetIndex(1);
  data1.y.dtype = af::DT_INT8;
  data1.ir_attr.SetIndex(4);

  af::ascir_op::Load load1(std::string(prefix + "_load1").c_str());
  load1.x = data1.y;
  load1.attr.sched.axis = axis_ids;
  *load1.y.axis = axis_ids;
  *load1.y.repeats = repeats;
  *load1.y.strides = strides;

  af::ascir_op::Add add(std::string(prefix + "_add").c_str());
  add.x1 = load0.y;
  add.x2 = load1.y;
  add.attr.sched.axis = axis_ids;
  *add.y.axis = axis_ids;
  *add.y.repeats = repeats;
  *add.y.strides = strides;

  af::ascir_op::Store store0(std::string(prefix + "_store0").c_str());
  store0.x = add.y;
  store0.attr.sched.axis = axis_ids;
  *store0.y.axis = axis_ids;
  *store0.y.repeats = repeats;
  *store0.y.strides = strides;

  af::ascir_op::Output y0(std::string(prefix + "_out0").c_str());
  y0.x = store0.y;
  y0.ir_attr.SetIndex(0);
  y0.y.dtype = af::DT_FLOAT16;
  y0.ir_attr.SetIndex(0);

  af::ascir_op::Store store1(std::string(prefix + "_store1").c_str());
  store1.x = add.y;
  store1.attr.sched.axis = axis_ids;
  *store1.y.axis = axis_ids;
  *store1.y.repeats = repeats;
  *store1.y.strides = strides;

  af::ascir_op::Output y1(std::string(prefix + "_out1").c_str());
  y1.x = store1.y;
  y1.ir_attr.SetIndex(1);
  y1.y.dtype = af::DT_FLOAT16;
  y1.ir_attr.SetIndex(1);
}

void CreateAscBackendGraphTwoInOneOut(std::shared_ptr<AscGraph> &graph, const std::string &prefix,
                                      int64_t axis_num = 2) {
  auto ONE = Symbol(1);
  std::vector<int64_t> axis_ids;
  std::vector<af::Expression> repeats;
  for (int64_t i = 0; i < axis_num; ++i) {
    const Expression exp = graph->CreateSizeVar("s" + std::to_string(i));
    auto axis = graph->CreateAxis("z" + std::to_string(i), exp);
    axis_ids.push_back(i);
    repeats.push_back(exp);
  }

  std::vector<af::Expression> strides(repeats.size(), One);
  if (axis_num > 1) {
    for (int64_t i = axis_num - 2; i >= 0; --i) {
      strides[i] = repeats[i + 1] * strides[i + 1];
    }
  }

  af::ascir_op::Data data0(std::string(prefix + "_data0").c_str(), *graph);
  data0.attr.sched.axis = axis_ids;
  *data0.y.axis = axis_ids;
  *data0.y.repeats = repeats;
  *data0.y.strides = strides;
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = af::DT_INT8;
  data0.ir_attr.SetIndex(5);

  af::ascir_op::Load load0(std::string(prefix + "_load0").c_str());
  load0.x = data0.y;
  load0.attr.sched.axis = axis_ids;
  *load0.y.axis = axis_ids;
  *load0.y.repeats = repeats;
  *load0.y.strides = strides;

  af::ascir_op::Data data1(std::string(prefix + "_data1").c_str(), *graph);
  data1.attr.sched.axis = axis_ids;
  *data1.y.axis = axis_ids;
  *data1.y.repeats = repeats;
  *data1.y.strides = strides;
  data1.ir_attr.SetIndex(1);
  data1.y.dtype = af::DT_INT8;
  data1.ir_attr.SetIndex(6);

  af::ascir_op::Load load1(std::string(prefix + "_load1").c_str());
  load1.x = data1.y;
  load1.attr.sched.axis = axis_ids;
  *load1.y.axis = axis_ids;
  *load1.y.repeats = repeats;
  *load1.y.strides = strides;
  load1.ir_attr.SetOffset(Expression(Symbol(1024)));

  af::ascir_op::Add add(std::string(prefix + "_add").c_str());
  add.x1 = load0.y;
  add.x2 = load1.y;
  add.attr.sched.axis = axis_ids;
  *add.y.axis = axis_ids;
  *add.y.repeats = repeats;
  *add.y.strides = strides;

  af::ascir_op::Store store0(std::string(prefix + "_store0").c_str());
  store0.x = add.y;
  store0.attr.sched.axis = axis_ids;
  *store0.y.axis = axis_ids;
  *store0.y.repeats = repeats;
  *store0.y.strides = strides;

  af::ascir_op::Output y0(std::string(prefix + "_out0").c_str());
  y0.x = store0.y;
  y0.ir_attr.SetIndex(0);
  y0.y.dtype = af::DT_FLOAT16;
  y0.ir_attr.SetIndex(0);
}

NodePtr CreateAscbcToAscGraph(const std::string &name, ComputeGraphPtr &compute_graph, int64_t in_num = 1,
                              int64_t out_num = 1) {
  OpDescBuilder op_desc_builder(name, "AscBackend");
  op_desc_builder.AddDynamicInput("x", in_num);
  op_desc_builder.AddDynamicOutput("y", out_num);
  const auto &op_desc = op_desc_builder.Build();
  auto node = compute_graph->AddNode(op_desc);
  node->SetOwnerComputeGraph(compute_graph);
  return node;
}

struct AscBackend : public Operator {
  inline explicit AscBackend(const char *name) : af::Operator(name, "AscBackend") {
    this->DynamicInputRegister("x", 0U, true);
    this->DynamicOutputRegister("y", 0U, true);
  }
};

NodePtr CreateAscbcToAscGraphWithIr(const std::string &name, ComputeGraphPtr &compute_graph, int64_t in_num = 1,
                                    int64_t out_num = 1) {
  AscBackend asc_backend((name.data()));
  asc_backend.DynamicInputRegister("x", in_num, true);
  asc_backend.DynamicOutputRegister("y", out_num, true);
  const auto op_desc = af::OpDescUtils::GetOpDescFromOperator(asc_backend);
  auto node = compute_graph->AddNode(op_desc);
  node->SetOwnerComputeGraph(compute_graph);
  return node;
}

/**
 * Output0
 *    |
 *  AscBc3
 *    |
 *  AscBc2
 *    |
 *  AscBc1
 *    |
 *  data0
 */
ComputeGraphPtr BuildFusedAscbc1() {
  std::shared_ptr<AscGraph> g0 = std::make_shared<af::AscGraph>("g0");
  CreateAscBackendGraph(g0, "g0", 2);
  std::shared_ptr<AscGraph> g1 = std::make_shared<af::AscGraph>("g1");
  CreateAscBackendGraph(g1, "g1", 1);
  std::shared_ptr<AscGraph> g2 = std::make_shared<af::AscGraph>("g2");
  CreateAscBackendGraph(g2, "g2", 2);

  AscGraph fused_asc_graph("fused_graph");

  af::ascir_op::Data data0("data0", fused_asc_graph);
  auto ir_attr = data0.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  ir_attr->SetIndex(0);

  auto fused_graph = af::AscGraphUtils::GetComputeGraph(fused_asc_graph);
  auto data_node = fused_asc_graph.FindNode("data0");

  auto ascbc1 = CreateAscbcToAscGraphWithIr("ascbc1", fused_graph);
  auto ascbc2 = CreateAscbcToAscGraphWithIr("ascbc2", fused_graph);
  auto ascbc3 = CreateAscbcToAscGraphWithIr("ascbc3", fused_graph);
  af::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), ascbc1->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc1->GetOutDataAnchor(0), ascbc2->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc2->GetOutDataAnchor(0), ascbc3->GetInDataAnchor(0));

  af::ascir_op::Output output("output");
  auto out_ir_attr = output.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  out_ir_attr->SetIndex(0);
  auto out_desc = OpDescUtils::GetOpDescFromOperator(output);
  auto output_node = fused_graph->AddNode(out_desc);
  af::GraphUtils::AddEdge(ascbc3->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));

  std::string asc_graph_str_1;
  GE_ASSERT_SUCCESS(af::AscGraphUtils::SerializeToReadable(*g0, asc_graph_str_1));
  auto op_desc_1 = ascbc1->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc_1);
  af::AttrUtils::SetStr(op_desc_1, "ascgraph", asc_graph_str_1);

  std::string asc_graph_str_2;
  GE_ASSERT_SUCCESS(af::AscGraphUtils::SerializeToReadable(*g1, asc_graph_str_2));
  auto op_desc_2 = ascbc2->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc_2);
  af::AttrUtils::SetStr(op_desc_2, "ascgraph", asc_graph_str_2);

  std::string asc_graph_str_3;
  GE_ASSERT_SUCCESS(af::AscGraphUtils::SerializeToReadable(*g2, asc_graph_str_3));
  auto op_desc_3 = ascbc3->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc_3);
  af::AttrUtils::SetStr(op_desc_3, "ascgraph", asc_graph_str_3);

  fused_graph->TopologicalSorting();
  return fused_graph;
}

/**
 *
 *                  Output0
 *                    |
 *                  AscBc3
 *                /     |
 *           AscBc2    / ---Output1
 *        /    \     /
 *     data2  AscBc1
 *            /   \
 *         data0  data1
 */
ComputeGraphPtr BuildFusedAscbc2(const std::string node_type = "") {
  std::shared_ptr<AscGraph> g0 = std::make_shared<af::AscGraph>("g0");
  CreateAscBackendGraphTwoInTwoOut(g0, "g0", 2);
  std::shared_ptr<AscGraph> g1 = std::make_shared<af::AscGraph>("g1");
  CreateAscBackendGraphTwoInOneOut(g1, "g1", 1);
  std::shared_ptr<AscGraph> g2 = std::make_shared<af::AscGraph>("g2");
  CreateAscBackendGraphTwoInOneOut(g2, "g2", 2);

  AscGraph fused_asc_graph("fused_graph");
  af::ascir_op::Data data0("data0", fused_asc_graph);
  auto ir_attr0 = data0.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  ir_attr0->SetIndex(0);

  af::ascir_op::Data data1("data1", fused_asc_graph);
  auto ir_attr1 = data1.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  ir_attr1->SetIndex(1);

  af::ascir_op::Data data2("data2", fused_asc_graph);
  auto ir_attr2 = data2.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  ir_attr2->SetIndex(2);

  auto fused_graph = af::AscGraphUtils::GetComputeGraph(fused_asc_graph);
  auto data0_node = fused_asc_graph.FindNode("data0");
  auto data1_node = fused_asc_graph.FindNode("data1");
  auto data2_node = fused_asc_graph.FindNode("data2");

  auto ascbc1 = CreateAscbcToAscGraph("ascbc1", fused_graph, 2, 2);
  auto ascbc2 = CreateAscbcToAscGraph("ascbc2", fused_graph, 2, 1);
  auto ascbc3 = CreateAscbcToAscGraph("ascbc3", fused_graph, 2, 1);

  af::GraphUtils::AddEdge(data0_node->GetOutDataAnchor(0), ascbc1->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), ascbc1->GetInDataAnchor(1));
  af::GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), ascbc2->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc1->GetOutDataAnchor(0), ascbc2->GetInDataAnchor(1));
  af::GraphUtils::AddEdge(ascbc2->GetOutDataAnchor(0), ascbc3->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc1->GetOutDataAnchor(1), ascbc3->GetInDataAnchor(1));

  af::ascir_op::Output output0("output0");
  auto out0_ir_attr = output0.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  out0_ir_attr->SetIndex(0);
  auto out0_desc = OpDescUtils::GetOpDescFromOperator(output0);
  auto output0_node = fused_graph->AddNode(out0_desc);

  af::ascir_op::Output output1("output1");
  auto out1_ir_attr = output1.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  out1_ir_attr->SetIndex(1);
  auto out1_desc = OpDescUtils::GetOpDescFromOperator(output1);
  auto output1_node = fused_graph->AddNode(out1_desc);
  af::GraphUtils::AddEdge(ascbc3->GetOutDataAnchor(0), output0_node->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc1->GetOutDataAnchor(1), output1_node->GetInDataAnchor(0));

  std::string asc_graph_str_1;
  GE_ASSERT_SUCCESS(af::AscGraphUtils::SerializeToReadable(*g0, asc_graph_str_1));
  auto op_desc_1 = ascbc1->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc_1);
  af::AttrUtils::SetStr(op_desc_1, "ascgraph", asc_graph_str_1);

  std::string asc_graph_str_2;
  GE_ASSERT_SUCCESS(af::AscGraphUtils::SerializeToReadable(*g1, asc_graph_str_2));
  auto op_desc_2 = ascbc2->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc_2);
  af::AttrUtils::SetStr(op_desc_2, "ascgraph", asc_graph_str_2);

  std::string asc_graph_str_3;
  GE_ASSERT_SUCCESS(af::AscGraphUtils::SerializeToReadable(*g2, asc_graph_str_3));
  auto op_desc_3 = ascbc3->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc_3);
  af::AttrUtils::SetStr(op_desc_3, "ascgraph", asc_graph_str_3);

  fused_graph->TopologicalSorting();
  return fused_graph;
}

TEST_F(AscendGraphCodeDumperUT, test_compute_graph1) {
  auto fused_graph = BuildFusedAscbc2();
  af::ascir::PythonCodeDumperFused dumper;
  EXPECT_EQ(dumper.Dump(*fused_graph, "./asc_wrong_graph_python.py"), af::SUCCESS);

  const std::string expected_graph_code = R"(# Python code to construct ComputeGraph
from autofuse.pyautofuse import ascir
from autofuse.pyautofuse import Autofuser, AutofuserOptions

graph = ascir.FusedGraph("fused_graph")
Data_0 = ascir.ops.Data("data0", graph)
Data_0.y.dtype = ascir.dtypes.float32
Data_0.attr.ir_attr.index = 0
Data_1 = ascir.ops.Data("data1", graph)
Data_1.y.dtype = ascir.dtypes.float32
Data_1.attr.ir_attr.index = 1

def GetAscBackend_0():
    graph = ascir.HintGraph("g0")
    s0 = graph.create_size("s0")
    s1 = graph.create_size("s1")
    z0 = graph.create_axis("z0", s0)
    z1 = graph.create_axis("z1", s1)
    Data_2 = ascir.ops.Data("g0_data0", graph)
    Data_2.attr.sched.axis = [z0, z1]
    Data_2.y.dtype = ascir.dtypes.int8
    Data_2.y.axis = [z0, z1]
    Data_2.y.size = [s0, s1]
    Data_2.y.strides = [s1, 1]
    Data_2.attr.ir_attr.index = 3
    Load_0 = ascir.ops.Load("g0_load0")
    Load_0.attr.sched.axis = [z0, z1]
    Load_0.x = Data_2.y
    Load_0.y.dtype = ascir.dtypes.float32
    Load_0.y.axis = [z0, z1]
    Load_0.y.size = [s0, s1]
    Load_0.y.strides = [s1, 1]
    Data_3 = ascir.ops.Data("g0_data1", graph)
    Data_3.attr.sched.axis = [z0, z1]
    Data_3.y.dtype = ascir.dtypes.int8
    Data_3.y.axis = [z0, z1]
    Data_3.y.size = [s0, s1]
    Data_3.y.strides = [s1, 1]
    Data_3.attr.ir_attr.index = 4
    Load_1 = ascir.ops.Load("g0_load1")
    Load_1.attr.sched.axis = [z0, z1]
    Load_1.x = Data_3.y
    Load_1.y.dtype = ascir.dtypes.float32
    Load_1.y.axis = [z0, z1]
    Load_1.y.size = [s0, s1]
    Load_1.y.strides = [s1, 1]
    Add_0 = ascir.ops.Add("g0_add")
    Add_0.attr.sched.axis = [z0, z1]
    Add_0.x1 = Load_0.y
    Add_0.x2 = Load_1.y
    Add_0.y.dtype = ascir.dtypes.float32
    Add_0.y.axis = [z0, z1]
    Add_0.y.size = [s0, s1]
    Add_0.y.strides = [s1, 1]
    Store_0 = ascir.ops.Store("g0_store0")
    Store_0.attr.sched.axis = [z0, z1]
    Store_0.x = Add_0.y
    Store_0.y.dtype = ascir.dtypes.float32
    Store_0.y.axis = [z0, z1]
    Store_0.y.size = [s0, s1]
    Store_0.y.strides = [s1, 1]
    Output_0 = ascir.ops.Output("g0_out0")
    Output_0.x = Store_0.y
    Output_0.y.dtype = ascir.dtypes.float16
    Output_0.attr.ir_attr.index = 0
    Store_1 = ascir.ops.Store("g0_store1")
    Store_1.attr.sched.axis = [z0, z1]
    Store_1.x = Add_0.y
    Store_1.y.dtype = ascir.dtypes.float32
    Store_1.y.axis = [z0, z1]
    Store_1.y.size = [s0, s1]
    Store_1.y.strides = [s1, 1]
    Output_1 = ascir.ops.Output("g0_out1")
    Output_1.x = Store_1.y
    Output_1.y.dtype = ascir.dtypes.float16
    Output_1.attr.ir_attr.index = 1
    return graph

AscBackend_0 = ascir.ops.AscBackend('ascbc1', GetAscBackend_0(), graph)
AscBackend_0.x = [Data_0.y, Data_1.y]

Output_2 = ascir.ops.Output("output1")
Output_2.x = AscBackend_0.y[1]
Output_2.y.dtype = ascir.dtypes.float32
Output_2.attr.ir_attr.index = 1
Data_4 = ascir.ops.Data("data2", graph)
Data_4.y.dtype = ascir.dtypes.float32
Data_4.attr.ir_attr.index = 2

def GetAscBackend_1():
    graph = ascir.HintGraph("g1")
    s0 = graph.create_size("s0")
    z0 = graph.create_axis("z0", s0)
    Data_5 = ascir.ops.Data("g1_data0", graph)
    Data_5.attr.sched.axis = [z0]
    Data_5.y.dtype = ascir.dtypes.int8
    Data_5.y.axis = [z0]
    Data_5.y.size = [s0]
    Data_5.y.strides = [1]
    Data_5.attr.ir_attr.index = 5
    Load_2 = ascir.ops.Load("g1_load0")
    Load_2.attr.sched.axis = [z0]
    Load_2.x = Data_5.y
    Load_2.y.dtype = ascir.dtypes.float32
    Load_2.y.axis = [z0]
    Load_2.y.size = [s0]
    Load_2.y.strides = [1]
    Data_6 = ascir.ops.Data("g1_data1", graph)
    Data_6.attr.sched.axis = [z0]
    Data_6.y.dtype = ascir.dtypes.int8
    Data_6.y.axis = [z0]
    Data_6.y.size = [s0]
    Data_6.y.strides = [1]
    Data_6.attr.ir_attr.index = 6
    Load_3 = ascir.ops.Load("g1_load1")
    Load_3.attr.sched.axis = [z0]
    Load_3.x = Data_6.y
    Load_3.y.dtype = ascir.dtypes.float32
    Load_3.y.axis = [z0]
    Load_3.y.size = [s0]
    Load_3.y.strides = [1]
    Load_3.attr.ir_attr.offset = 1024
    Add_1 = ascir.ops.Add("g1_add")
    Add_1.attr.sched.axis = [z0]
    Add_1.x1 = Load_2.y
    Add_1.x2 = Load_3.y
    Add_1.y.dtype = ascir.dtypes.float32
    Add_1.y.axis = [z0]
    Add_1.y.size = [s0]
    Add_1.y.strides = [1]
    Store_2 = ascir.ops.Store("g1_store0")
    Store_2.attr.sched.axis = [z0]
    Store_2.x = Add_1.y
    Store_2.y.dtype = ascir.dtypes.float32
    Store_2.y.axis = [z0]
    Store_2.y.size = [s0]
    Store_2.y.strides = [1]
    Output_3 = ascir.ops.Output("g1_out0")
    Output_3.x = Store_2.y
    Output_3.y.dtype = ascir.dtypes.float16
    Output_3.attr.ir_attr.index = 0
    return graph

AscBackend_1 = ascir.ops.AscBackend('ascbc2', GetAscBackend_1(), graph)
AscBackend_1.x = [Data_4.y, AscBackend_0.y[0]]


def GetAscBackend_2():
    graph = ascir.HintGraph("g2")
    s0 = graph.create_size("s0")
    s1 = graph.create_size("s1")
    z0 = graph.create_axis("z0", s0)
    z1 = graph.create_axis("z1", s1)
    Data_7 = ascir.ops.Data("g2_data0", graph)
    Data_7.attr.sched.axis = [z0, z1]
    Data_7.y.dtype = ascir.dtypes.int8
    Data_7.y.axis = [z0, z1]
    Data_7.y.size = [s0, s1]
    Data_7.y.strides = [s1, 1]
    Data_7.attr.ir_attr.index = 5
    Load_4 = ascir.ops.Load("g2_load0")
    Load_4.attr.sched.axis = [z0, z1]
    Load_4.x = Data_7.y
    Load_4.y.dtype = ascir.dtypes.float32
    Load_4.y.axis = [z0, z1]
    Load_4.y.size = [s0, s1]
    Load_4.y.strides = [s1, 1]
    Data_8 = ascir.ops.Data("g2_data1", graph)
    Data_8.attr.sched.axis = [z0, z1]
    Data_8.y.dtype = ascir.dtypes.int8
    Data_8.y.axis = [z0, z1]
    Data_8.y.size = [s0, s1]
    Data_8.y.strides = [s1, 1]
    Data_8.attr.ir_attr.index = 6
    Load_5 = ascir.ops.Load("g2_load1")
    Load_5.attr.sched.axis = [z0, z1]
    Load_5.x = Data_8.y
    Load_5.y.dtype = ascir.dtypes.float32
    Load_5.y.axis = [z0, z1]
    Load_5.y.size = [s0, s1]
    Load_5.y.strides = [s1, 1]
    Load_5.attr.ir_attr.offset = 1024
    Add_2 = ascir.ops.Add("g2_add")
    Add_2.attr.sched.axis = [z0, z1]
    Add_2.x1 = Load_4.y
    Add_2.x2 = Load_5.y
    Add_2.y.dtype = ascir.dtypes.float32
    Add_2.y.axis = [z0, z1]
    Add_2.y.size = [s0, s1]
    Add_2.y.strides = [s1, 1]
    Store_3 = ascir.ops.Store("g2_store0")
    Store_3.attr.sched.axis = [z0, z1]
    Store_3.x = Add_2.y
    Store_3.y.dtype = ascir.dtypes.float32
    Store_3.y.axis = [z0, z1]
    Store_3.y.size = [s0, s1]
    Store_3.y.strides = [s1, 1]
    Output_4 = ascir.ops.Output("g2_out0")
    Output_4.x = Store_3.y
    Output_4.y.dtype = ascir.dtypes.float16
    Output_4.attr.ir_attr.index = 0
    return graph

AscBackend_2 = ascir.ops.AscBackend('ascbc3', GetAscBackend_2(), graph)
AscBackend_2.x = [AscBackend_1.y[0], AscBackend_0.y[1]]

Output_5 = ascir.ops.Output("output0")
Output_5.x = AscBackend_2.y[0]
Output_5.y.dtype = ascir.dtypes.float32
Output_5.attr.ir_attr.index = 0
fuser = Autofuser(AutofuserOptions())
schedule_results = fuser.schedule(graph)
tiling_def, host_impl, device_impl = fuser.codegen(schedule_results)
)";
  EXPECT_EQ(expected_graph_code, ReadFileContent("./asc_wrong_graph_python.py"));
}

TEST_F(AscendGraphCodeDumperUT, test_compute_graph2) {
  auto fused_graph = BuildFusedAscbc1();
  af::ascir::PythonCodeDumperFused dumper;
  EXPECT_EQ(dumper.Dump(*fused_graph, "./asc_wrong_graph_python.py"), af::SUCCESS);

  const std::string expected_graph_code = R"(# Python code to construct ComputeGraph
from autofuse.pyautofuse import ascir
from autofuse.pyautofuse import Autofuser, AutofuserOptions

graph = ascir.FusedGraph("fused_graph")
Data_0 = ascir.ops.Data("data0", graph)
Data_0.y.dtype = ascir.dtypes.float32
Data_0.attr.ir_attr.index = 0

def GetAscBackend_0():
    graph = ascir.HintGraph("g0")
    s0 = graph.create_size("s0")
    s1 = graph.create_size("s1")
    z0 = graph.create_axis("z0", s0)
    z1 = graph.create_axis("z1", s1)
    Data_1 = ascir.ops.Data("g0_data", graph)
    Data_1.attr.sched.axis = [z0, z1]
    Data_1.y.dtype = ascir.dtypes.int8
    Data_1.y.axis = [z0, z1]
    Data_1.y.size = [s0, s1]
    Data_1.y.strides = [s1, 1]
    Data_1.attr.ir_attr.index = 0
    Load_0 = ascir.ops.Load("g0_load")
    Load_0.attr.sched.axis = [z0, z1]
    Load_0.x = Data_1.y
    Load_0.y.dtype = ascir.dtypes.float32
    Load_0.y.axis = [z0, z1]
    Load_0.y.size = [s0, s1]
    Load_0.y.strides = [s1, 1]
    Load_0.attr.ir_attr.offset = 1024
    Abs_0 = ascir.ops.Abs("g0_abs")
    Abs_0.attr.sched.axis = [z0, z1]
    Abs_0.x = Load_0.y
    Abs_0.y.dtype = ascir.dtypes.float32
    Abs_0.y.axis = [z0, z1]
    Abs_0.y.size = [s0, s1]
    Abs_0.y.strides = [s1, 1]
    Store_0 = ascir.ops.Store("g0_store")
    Store_0.attr.sched.axis = [z0, z1]
    Store_0.x = Abs_0.y
    Store_0.y.dtype = ascir.dtypes.float32
    Store_0.y.axis = [z0, z1]
    Store_0.y.size = [s0, s1]
    Store_0.y.strides = [s1, 1]
    Output_0 = ascir.ops.Output("g0_out")
    Output_0.x = Store_0.y
    Output_0.y.dtype = ascir.dtypes.float16
    Output_0.attr.ir_attr.index = 0
    return graph

AscBackend_0 = ascir.ops.AscBackend('ascbc1', GetAscBackend_0(), graph)
AscBackend_0.x = [Data_0.y]


def GetAscBackend_1():
    graph = ascir.HintGraph("g1")
    s0 = graph.create_size("s0")
    z0 = graph.create_axis("z0", s0)
    Data_2 = ascir.ops.Data("g1_data", graph)
    Data_2.attr.sched.axis = [z0]
    Data_2.y.dtype = ascir.dtypes.int8
    Data_2.y.axis = [z0]
    Data_2.y.size = [s0]
    Data_2.y.strides = [1]
    Data_2.attr.ir_attr.index = 0
    Load_1 = ascir.ops.Load("g1_load")
    Load_1.attr.sched.axis = [z0]
    Load_1.x = Data_2.y
    Load_1.y.dtype = ascir.dtypes.float32
    Load_1.y.axis = [z0]
    Load_1.y.size = [s0]
    Load_1.y.strides = [1]
    Load_1.attr.ir_attr.offset = 1024
    Abs_1 = ascir.ops.Abs("g1_abs")
    Abs_1.attr.sched.axis = [z0]
    Abs_1.x = Load_1.y
    Abs_1.y.dtype = ascir.dtypes.float32
    Abs_1.y.axis = [z0]
    Abs_1.y.size = [s0]
    Abs_1.y.strides = [1]
    Store_1 = ascir.ops.Store("g1_store")
    Store_1.attr.sched.axis = [z0]
    Store_1.x = Abs_1.y
    Store_1.y.dtype = ascir.dtypes.float32
    Store_1.y.axis = [z0]
    Store_1.y.size = [s0]
    Store_1.y.strides = [1]
    Output_1 = ascir.ops.Output("g1_out")
    Output_1.x = Store_1.y
    Output_1.y.dtype = ascir.dtypes.float16
    Output_1.attr.ir_attr.index = 0
    return graph

AscBackend_1 = ascir.ops.AscBackend('ascbc2', GetAscBackend_1(), graph)
AscBackend_1.x = [AscBackend_0.y[0]]


def GetAscBackend_2():
    graph = ascir.HintGraph("g2")
    s0 = graph.create_size("s0")
    s1 = graph.create_size("s1")
    z0 = graph.create_axis("z0", s0)
    z1 = graph.create_axis("z1", s1)
    Data_3 = ascir.ops.Data("g2_data", graph)
    Data_3.attr.sched.axis = [z0, z1]
    Data_3.y.dtype = ascir.dtypes.int8
    Data_3.y.axis = [z0, z1]
    Data_3.y.size = [s0, s1]
    Data_3.y.strides = [s1, 1]
    Data_3.attr.ir_attr.index = 0
    Load_2 = ascir.ops.Load("g2_load")
    Load_2.attr.sched.axis = [z0, z1]
    Load_2.x = Data_3.y
    Load_2.y.dtype = ascir.dtypes.float32
    Load_2.y.axis = [z0, z1]
    Load_2.y.size = [s0, s1]
    Load_2.y.strides = [s1, 1]
    Load_2.attr.ir_attr.offset = 1024
    Abs_2 = ascir.ops.Abs("g2_abs")
    Abs_2.attr.sched.axis = [z0, z1]
    Abs_2.x = Load_2.y
    Abs_2.y.dtype = ascir.dtypes.float32
    Abs_2.y.axis = [z0, z1]
    Abs_2.y.size = [s0, s1]
    Abs_2.y.strides = [s1, 1]
    Store_2 = ascir.ops.Store("g2_store")
    Store_2.attr.sched.axis = [z0, z1]
    Store_2.x = Abs_2.y
    Store_2.y.dtype = ascir.dtypes.float32
    Store_2.y.axis = [z0, z1]
    Store_2.y.size = [s0, s1]
    Store_2.y.strides = [s1, 1]
    Output_2 = ascir.ops.Output("g2_out")
    Output_2.x = Store_2.y
    Output_2.y.dtype = ascir.dtypes.float16
    Output_2.attr.ir_attr.index = 0
    return graph

AscBackend_2 = ascir.ops.AscBackend('ascbc3', GetAscBackend_2(), graph)
AscBackend_2.x = [AscBackend_1.y[0]]

Output_3 = ascir.ops.Output("output")
Output_3.x = AscBackend_2.y[0]
Output_3.y.dtype = ascir.dtypes.float32
Output_3.attr.ir_attr.index = 0
fuser = Autofuser(AutofuserOptions())
schedule_results = fuser.schedule(graph)
tiling_def, host_impl, device_impl = fuser.codegen(schedule_results)
)";
  EXPECT_EQ(expected_graph_code, ReadFileContent("./asc_wrong_graph_python.py"));
}

// 验证 DumpMemoryLayoutView 对多输出节点的收集：
// 多输出节点的每个 output 的 que/buf 信息都应收集，且命名为 y[i]
TEST_F(AscendGraphCodeDumperUT, test_memory_layout_view_multi_output) {
  AscGraph graph("multi_output_layout");
  Expression s0 = graph.CreateSizeVar("s0");
  Expression s1 = graph.CreateSizeVar("s1");
  Axis &z0 = graph.CreateAxis("z0", s0);
  Axis &z1 = graph.CreateAxis("z1", s1);

  ascir_op::Data data("data", graph);
  data.attr.sched.axis = {z0.id, z1.id};
  data.y.dtype = DT_FLOAT16;

  ascir_op::Load load("load");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = DT_FLOAT16;
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, One};

  ascir_op::Split split("split");
  split.InstanceOutputy(3U);
  split.x = load.y;
  split.attr.sched.axis = {z0.id, z1.id};
  split.attr.api.compute_type = ComputeType::kComputeSplit;
  for (size_t i = 0UL; i < 3UL; ++i) {
    split.y[i].dtype = DT_FLOAT16;
    *split.y[i].axis = {z0.id, z1.id};
    *split.y[i].repeats = {s0, s1};
    *split.y[i].strides = {s1, One};
    // output[0] 用 queue，output[1]/[2] 用 buffer，覆盖两种收集路径
    split.y[i].mem->alloc_type = (i == 0UL) ? AllocType::kAllocTypeQueue : AllocType::kAllocTypeBuffer;
    if (i == 0UL) {
      split.y[i].que->id = 7;
      split.y[i].que->depth = 2;
      split.y[i].que->buf_num = 2;
    } else {
      split.y[i].buf->id = static_cast<int64_t>(i);
    }
  }

  const std::string layout = ::ascir::dumper::DumpMemoryLayoutView(graph, true);
  // Queue 7 来自 split 的 output[0]
  EXPECT_NE(layout.find("Queue 7"), std::string::npos);
  EXPECT_NE(layout.find("split.y[0]"), std::string::npos) << layout;
  // Buffer 1/2 来自 split 的 output[1]/[2]，修复前会缺失
  EXPECT_NE(layout.find("Buffer 1"), std::string::npos) << layout;
  EXPECT_NE(layout.find("Buffer 2"), std::string::npos) << layout;
  EXPECT_NE(layout.find("split.y[1]"), std::string::npos) << layout;
  EXPECT_NE(layout.find("split.y[2]"), std::string::npos) << layout;
}

}  // namespace af
