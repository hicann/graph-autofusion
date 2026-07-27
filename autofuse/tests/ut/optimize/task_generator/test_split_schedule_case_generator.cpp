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
#include "task_generator/split_schedule_case_generator.h"
#include "task_generator/split_group_partitioner.h"
#include "task_generator/split_score_function_generator.h"

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

class SplitScheduleCaseGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }

  static void CreateSplitAscGraph(af::AscGraph &graph, const std::string &head_dim,
                                  const std::vector<std::string> &split_dims,
                                  const std::vector<std::string> &tail_dims) {
    std::vector<af::Expression> split_axis_sizes;
    af::Expression split_axis_size = af::Symbol(0);
    for (const auto &dim : split_dims) {
      if (dim[0] == 's') {
        split_axis_sizes.emplace_back(graph.CreateSizeVar(dim));
      } else {
        split_axis_sizes.emplace_back(af::Symbol(std::strtol(dim.c_str(), nullptr, 10)));
      }
      split_axis_size = split_axis_size + split_axis_sizes.back();
    }
    std::vector<af::Expression> tail_dim_sizes;
    for (const auto &dim : tail_dims) {
      if (dim[0] == 's') {
        tail_dim_sizes.emplace_back(graph.CreateSizeVar(dim));
      } else {
        tail_dim_sizes.emplace_back(af::Symbol(std::strtol(dim.c_str(), nullptr, 10)));
      }
    }

    auto head_dim_size = graph.CreateSizeVar(head_dim);
    auto z0 = graph.CreateAxis("z0", head_dim_size);
    auto z1 = graph.CreateAxis("z1", split_axis_size);
    std::vector<af::Axis> tail_axes;
    tail_axes.reserve(tail_dims.size());
    for (size_t i = 0U; i < tail_dims.size(); ++i) {
      tail_axes.emplace_back(graph.CreateAxis("z" + std::to_string(2 + i), tail_dim_sizes[i]));
    }

    Data data0("data0", graph);
    data0.attr.api.type = af::ApiType::kAPITypeBuffer;
    data0.ir_attr.SetIndex(0);
    data0.y.dtype = ge::DT_FLOAT16;
    *data0.y.axis = {z0.id, z1.id, tail_axes[0].id, tail_axes[1].id};
    *data0.y.repeats = {head_dim_size, split_axis_size, tail_dim_sizes[0], tail_dim_sizes[1]};
    //*data0.y.strides = {s1, af::sym::kSymbolOne};

    Load load1("load");
    load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
    load1.y.dtype = ge::DT_FLOAT16;
    load1.x = data0.y;
    load1.attr.sched.axis = {z0.id, z1.id, tail_axes[0].id, tail_axes[1].id};
    *load1.y.axis = {z0.id, z1.id, tail_axes[0].id, tail_axes[1].id};
    *load1.y.repeats = {head_dim_size, split_axis_size, tail_dim_sizes[0], tail_dim_sizes[1]};
    //*load1.y.strides = {s1, af::sym::kSymbolOne};

    Split split("split");
    split.InstanceOutputy(4U);  // 需要先指定输出个数
    split.attr.api.compute_type = af::ComputeType::kComputeSplit;
    split.x = {load1.y};
    split.attr.sched.axis = {z0.id, z1.id, tail_axes[0].id, tail_axes[1].id};
    split.y[0].dtype = ge::DT_FLOAT16;
    *split.y[0].axis = {z0.id, z1.id, tail_axes[0].id, tail_axes[1].id};
    *split.y[0].repeats = {head_dim_size, split_axis_sizes[0], tail_dim_sizes[0], tail_dim_sizes[1]};
    //*split.y[0].strides = {s1_1, af::sym::kSymbolOne};

    split.y[1].dtype = ge::DT_FLOAT16;
    *split.y[1].axis = {z0.id, z1.id, tail_axes[0].id, tail_axes[1].id};
    *split.y[1].repeats = {head_dim_size, split_axis_sizes[1], tail_dim_sizes[0], tail_dim_sizes[1]};
    //*split.y[1].strides = {s1_2, af::sym::kSymbolOne};

    split.y[2].dtype = ge::DT_FLOAT16;
    *split.y[2].axis = {z0.id, z1.id, tail_axes[0].id, tail_axes[1].id};
    *split.y[2].repeats = {head_dim_size, split_axis_sizes[2], tail_dim_sizes[0], tail_dim_sizes[1]};

    split.y[3].dtype = ge::DT_FLOAT16;
    *split.y[3].axis = {z0.id, z1.id, tail_axes[0].id, tail_axes[1].id};
    *split.y[3].repeats = {head_dim_size, split_axis_sizes[3], tail_dim_sizes[0], tail_dim_sizes[1]};

    Store store1("store");
    store1.x = split.y[0];
    store1.attr.sched.axis = {z0.id, z1.id, tail_axes[0].id, tail_axes[1].id};
    store1.y.dtype = ge::DT_FLOAT16;
    *store1.y.axis = {z0.id, z1.id, tail_axes[0].id, tail_axes[1].id};
    *store1.y.repeats = {head_dim_size, split_axis_sizes[0], tail_dim_sizes[0], tail_dim_sizes[1]};
    //*store1.y.strides = {s1_1, af::sym::kSymbolOne};

    Store store2("store");
    store2.x = split.y[1];
    store2.attr.sched.axis = {z0.id, z1.id, tail_axes[0].id, tail_axes[1].id};
    store2.y.dtype = ge::DT_FLOAT16;
    *store2.y.axis = {z0.id, z1.id, tail_axes[0].id, tail_axes[1].id};
    *store2.y.repeats = {head_dim_size, split_axis_sizes[1], tail_dim_sizes[0], tail_dim_sizes[1]};

    Store store3("store");
    store3.x = split.y[2];
    store3.attr.sched.axis = {z0.id, z1.id, tail_axes[0].id, tail_axes[1].id};
    store3.y.dtype = ge::DT_FLOAT16;
    *store3.y.axis = {z0.id, z1.id, tail_axes[0].id, tail_axes[1].id};
    *store3.y.repeats = {head_dim_size, split_axis_sizes[2], tail_dim_sizes[0], tail_dim_sizes[1]};

    Store store4("store");
    store4.x = split.y[3];
    store4.attr.sched.axis = {z0.id, z1.id, tail_axes[0].id, tail_axes[1].id};
    store4.y.dtype = ge::DT_FLOAT16;
    *store4.y.axis = {z0.id, z1.id, tail_axes[0].id, tail_axes[1].id};
    *store4.y.repeats = {head_dim_size, split_axis_sizes[3], tail_dim_sizes[0], tail_dim_sizes[1]};

    Output out1("out");
    out1.x = store1.y;
    out1.ir_attr.SetIndex(0);

    Output out2("out");
    out2.x = store2.y;
    out2.ir_attr.SetIndex(1);

    Output out3("out");
    out3.x = store3.y;
    out3.ir_attr.SetIndex(2);

    Output out4("out");
    out4.x = store4.y;
    out4.ir_attr.SetIndex(3);
  }

  static std::string ExpressToStr(const std::vector<af::Expression> &exprs) {
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
};

TEST_F(SplitScheduleCaseGeneratorTest, split_data_to_different_axis) {
  // dlog_setlevel(0, 0, 1);
  // setenv("DUMP_GE_GRAPH", "1", 1);
  // setenv("DUMP_GRAPH_LEVEL", "1", 1);
  // setenv("DUMP_GRAPH_PATH", "./", 1);
  af::AscGraph graph("split_last_dim_graph");
  auto s0 = af::Symbol("s0");
  auto s1 = af::Symbol("s1");
  auto s1_1 = af::Symbol("s1_1");
  auto s1_2 = af::Symbol("s1_2");
  auto s1_3 = af::Symbol("s1_3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);
  auto z1_2 = graph.CreateAxis("z1_2", s1_2);
  auto z1_3 = graph.CreateAxis("z1_3", s1_3);

  Data data0("data0", graph);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);

  Load load1("load");
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data0.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::sym::kSymbolOne};

  Split split("split");
  split.InstanceOutputy(3U);  // 需要先指定输出个数
  split.attr.api.compute_type = af::ComputeType::kComputeSplit;

  split.x = {load1.y};
  split.attr.sched.axis = {z0.id, z1.id};
  split.y[0].dtype = ge::DT_FLOAT16;
  *split.y[0].axis = {z0.id, z1_1.id};
  split.y[0].dtype = ge::DT_FLOAT16;
  *split.y[0].repeats = {s0, s1_1};
  *split.y[0].strides = {s1_1, af::sym::kSymbolOne};

  split.y[1].dtype = ge::DT_FLOAT16;
  *split.y[1].axis = {z0.id, z1_2.id};
  split.y[1].dtype = ge::DT_FLOAT16;
  *split.y[1].repeats = {s0, s1_2};
  *split.y[1].strides = {s1_2, af::sym::kSymbolOne};

  split.y[2].dtype = ge::DT_FLOAT16;
  *split.y[2].axis = {z0.id, z1_3.id};
  split.y[2].dtype = ge::DT_FLOAT16;
  *split.y[2].repeats = {s0, s1_3};
  *split.y[2].strides = {s1_3, af::sym::kSymbolOne};

  Store store1("store1");
  store1.y.dtype = ge::DT_FLOAT16;
  store1.x = split.y[0];
  store1.attr.sched.axis = {z0.id, z1_1.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1_1.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.repeats = {s0, s1_1};
  *store1.y.strides = {s1_1, af::sym::kSymbolOne};

  Store store2("store2");
  store2.y.dtype = ge::DT_FLOAT16;
  store2.x = split.y[1];
  store2.attr.sched.axis = {z0.id, z1_2.id};
  store2.y.dtype = ge::DT_FLOAT16;
  *store2.y.axis = {z0.id, z1_2.id};
  store2.y.dtype = ge::DT_FLOAT16;
  *store2.y.repeats = {s0, s1_2};
  *store2.y.strides = {s1_2, af::sym::kSymbolOne};

  Store store3("store3");
  store3.y.dtype = ge::DT_FLOAT16;
  store3.x = split.y[2];
  store3.attr.sched.axis = {z0.id, z1_3.id};
  store3.y.dtype = ge::DT_FLOAT16;
  *store3.y.axis = {z0.id, z1_3.id};
  store3.y.dtype = ge::DT_FLOAT16;
  *store3.y.repeats = {s0, s1_3};
  *store3.y.strides = {s1_3, af::sym::kSymbolOne};

  Output out1("out1");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(0);

  Output out2("out2");
  out2.x = store2.y;
  out2.ir_attr.SetIndex(1);

  Output out3("out3");
  out3.x = store3.y;
  out3.ir_attr.SetIndex(2);

  ::ascir::utils::DumpImplGraphs({graph}, "split_last_dim_graph");
  optimize::SplitFusionCaseGenerator generator;
  std::vector<af::AscGraph> generated_graphs;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.Generate(graph, generated_graphs, score_functions), af::SUCCESS);
  ASSERT_EQ(generated_graphs.size(), 2UL);

  auto cg0 = af::AscGraphUtils::GetComputeGraph(generated_graphs[0]);
  auto cg1 = af::AscGraphUtils::GetComputeGraph(generated_graphs[1]);
  EXPECT_EQ(cg0->GetAllNodesSize(), 9UL);
  EXPECT_EQ(cg1->GetAllNodesSize(), 12UL);
}
TEST_F(SplitScheduleCaseGeneratorTest, split_to_load_axis_connect_broadcast) {
  // dlog_setlevel(0, 0, 1);
  // setenv("DUMP_GE_GRAPH", "1", 1);
  // setenv("DUMP_GRAPH_LEVEL", "1", 1);
  // setenv("DUMP_GRAPH_PATH", "./", 1);
  af::AscGraph graph("split_to_load_axis_connect_broadcast");
  auto s0 = af::Symbol("s0");
  auto s1 = af::Symbol("s1");
  auto s1_1 = af::Symbol("s1_1");
  auto s1_2 = af::Symbol("s1_2");
  auto s1_3 = af::Symbol("s1_3");
  auto s2_brc = af::Symbol("s2_brc");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);
  auto z1_2 = graph.CreateAxis("z1_2", s1_2);
  auto z1_3 = graph.CreateAxis("z1_3", s1_3);
  auto z2_brc = graph.CreateAxis("z2_brc", s2_brc);

  Data data0("data0", graph);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);

  Load load1("load");
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data0.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::sym::kSymbolOne};

  Split split("split");
  split.InstanceOutputy(3U);  // 需要先指定输出个数
  split.attr.api.compute_type = af::ComputeType::kComputeSplit;

  split.x = {load1.y};
  split.attr.sched.axis = {z0.id, z1.id};
  split.y[0].dtype = ge::DT_FLOAT16;
  *split.y[0].axis = {z0.id, z1_1.id};
  split.y[0].dtype = ge::DT_FLOAT16;
  *split.y[0].repeats = {s0, s1_1};
  *split.y[0].strides = {s1_1, af::sym::kSymbolOne};

  split.y[1].dtype = ge::DT_FLOAT16;
  *split.y[1].axis = {z0.id, z1_2.id};
  split.y[1].dtype = ge::DT_FLOAT16;
  *split.y[1].repeats = {s0, s1_2};
  *split.y[1].strides = {s1_2, af::sym::kSymbolOne};

  split.y[2].dtype = ge::DT_FLOAT16;
  *split.y[2].axis = {z0.id, z1_3.id};
  split.y[2].dtype = ge::DT_FLOAT16;
  *split.y[2].repeats = {s0, s1_3};
  *split.y[2].strides = {s1_3, af::sym::kSymbolOne};

  Store store1("store1");
  store1.y.dtype = ge::DT_FLOAT16;
  store1.x = split.y[0];
  store1.attr.sched.axis = {z0.id, z1_1.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1_1.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.repeats = {s0, s1_1};
  *store1.y.strides = {s1_1, af::sym::kSymbolOne};

  Store store2("store2");
  store2.y.dtype = ge::DT_FLOAT16;
  store2.x = split.y[1];
  store2.attr.sched.axis = {z0.id, z1_2.id};
  store2.y.dtype = ge::DT_FLOAT16;
  *store2.y.axis = {z0.id, z1_2.id};
  store2.y.dtype = ge::DT_FLOAT16;
  *store2.y.repeats = {s0, s1_2};
  *store2.y.strides = {s1_2, af::sym::kSymbolOne};

  Broadcast broadcat1("broadcat1");
  broadcat1.y.dtype = ge::DT_FLOAT16;
  broadcat1.x = split.y[2];
  broadcat1.attr.sched.axis = {z0.id, z2_brc.id};
  broadcat1.y.dtype = ge::DT_FLOAT16;
  *broadcat1.y.axis = {z0.id, z2_brc.id};
  broadcat1.y.dtype = ge::DT_FLOAT16;
  *broadcat1.y.repeats = {s0, s2_brc};
  *broadcat1.y.strides = {s2_brc, af::sym::kSymbolOne};

  Store store3("store3");
  store3.y.dtype = ge::DT_FLOAT16;
  store3.x = broadcat1.y;
  store3.attr.sched.axis = {z0.id, z2_brc.id};
  store3.y.dtype = ge::DT_FLOAT16;
  *store3.y.axis = {z0.id, z2_brc.id};
  store3.y.dtype = ge::DT_FLOAT16;
  *store3.y.repeats = {s0, s2_brc};
  *store3.y.strides = {s2_brc, af::sym::kSymbolOne};

  Output out1("out1");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(0);

  Output out2("out2");
  out2.x = store2.y;
  out2.ir_attr.SetIndex(1);

  Output out3("out3");
  out3.x = store3.y;
  out3.ir_attr.SetIndex(2);

  ::ascir::utils::DumpImplGraphs({graph}, "split_to_load_axis_connect_broadcast");
  optimize::SplitFusionCaseGenerator generator;
  std::vector<af::AscGraph> generated_graphs;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.Generate(graph, generated_graphs, score_functions), af::SUCCESS);
  ASSERT_EQ(generated_graphs.size(), 2UL);

  auto cg0 = af::AscGraphUtils::GetComputeGraph(generated_graphs[0]);
  auto cg1 = af::AscGraphUtils::GetComputeGraph(generated_graphs[1]);
  EXPECT_EQ(cg0->GetAllNodesSize(), 10UL);
  EXPECT_EQ(cg1->GetAllNodesSize(), 13UL);
}

// ok
TEST_F(SplitScheduleCaseGeneratorTest, split_data_first_dim) {
  // dlog_setlevel(0, 0, 1);
  af::AscGraph graph("split_data_first_dim");
  auto s0 = af::Symbol("192");
  auto s1 = af::Symbol("64");
  auto s0_1 = af::Symbol("32");
  auto s0_2 = af::Symbol("56");
  auto s0_3 = af::Symbol("114");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z0_1 = graph.CreateAxis("z1_1", s0_1);
  auto z0_2 = graph.CreateAxis("z1_2", s0_2);
  auto z0_3 = graph.CreateAxis("z1_3", s0_3);

  Data data0("data0", graph);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);

  Load load1("load");
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data0.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::sym::kSymbolOne};

  Split split("split");
  split.InstanceOutputy(3U);  // 需要先指定输出个数
  split.attr.api.compute_type = af::ComputeType::kComputeSplit;

  split.x = {load1.y};
  split.attr.sched.axis = {z0.id, z1.id};
  split.y[0].dtype = ge::DT_FLOAT16;
  *split.y[0].axis = {z0_1.id, z1.id};
  split.y[0].dtype = ge::DT_FLOAT16;
  *split.y[0].repeats = {s0_1, s1};
  *split.y[0].strides = {s1, af::sym::kSymbolOne};

  split.y[1].dtype = ge::DT_FLOAT16;
  *split.y[1].axis = {z0_2.id, z1.id};
  split.y[1].dtype = ge::DT_FLOAT16;
  *split.y[1].repeats = {s0_2, s1};
  *split.y[1].strides = {s1, af::sym::kSymbolOne};

  split.y[2].dtype = ge::DT_FLOAT16;
  *split.y[2].axis = {z0_3.id, z1.id};
  split.y[2].dtype = ge::DT_FLOAT16;
  *split.y[2].repeats = {s0_3, s1};
  *split.y[2].strides = {s1, af::sym::kSymbolOne};

  Store store1("store");
  store1.y.dtype = ge::DT_FLOAT16;
  store1.x = split.y[0];
  store1.attr.sched.axis = {z0_1.id, z1.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0_1.id, z1.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.repeats = {s0_1, s1};
  *store1.y.strides = {s1, af::sym::kSymbolOne};

  Store store2("store");
  store2.y.dtype = ge::DT_FLOAT16;
  store2.x = split.y[1];
  store2.attr.sched.axis = {z0_2.id, z1.id};
  store2.y.dtype = ge::DT_FLOAT16;
  *store2.y.axis = {z0_2.id, z1.id};
  store2.y.dtype = ge::DT_FLOAT16;
  *store2.y.repeats = {s0_2, s1};
  *store2.y.strides = {s1, af::sym::kSymbolOne};

  Store store3("store");
  store3.y.dtype = ge::DT_FLOAT16;
  store3.x = split.y[2];
  store3.attr.sched.axis = {z0_3.id, z1.id};
  store3.y.dtype = ge::DT_FLOAT16;
  *store3.y.axis = {z0_3.id, z1.id};
  store3.y.dtype = ge::DT_FLOAT16;
  *store3.y.repeats = {s0_3, s1};
  *store3.y.strides = {s1, af::sym::kSymbolOne};

  Output out1("out");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(0);

  Output out2("out");
  out2.x = store2.y;
  out2.ir_attr.SetIndex(1);

  Output out3("out");
  out3.x = store3.y;
  out3.ir_attr.SetIndex(2);

  ::ascir::utils::DumpImplGraphs({graph}, "split_last_dim_graph");
  optimize::SplitFusionCaseGenerator generator;
  std::vector<af::AscGraph> generated_graphs;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.Generate(graph, generated_graphs, score_functions), af::SUCCESS);
  ASSERT_EQ(generated_graphs.size(), 1UL);

  auto cg0 = af::AscGraphUtils::GetComputeGraph(generated_graphs[0]);
  printf("node size is %ld", cg0->GetAllNodesSize());
  EXPECT_EQ(cg0->GetAllNodesSize(), 12UL);
}

TEST_F(SplitScheduleCaseGeneratorTest, SplitScoreFunc_Aligned) {
  // dlog_setlevel(0, 0, 1);
  af::AscGraph graph("split_last_dim_graph");
  auto s0 = af::Symbol("s0");
  auto s1 = af::Symbol("s1");
  auto s2 = af::Symbol(64);
  // auto s2 = 64U;
  auto s1_1 = af::Symbol("s1_1");
  auto s1_2 = af::Symbol("s1_2");
  auto s1_3 = af::Symbol("s1_3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);
  auto z1_2 = graph.CreateAxis("z1_2", s1_2);
  auto z1_3 = graph.CreateAxis("z1_3", s1_3);

  Data data0("data0", graph);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);

  Load load1("load");
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data0.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::sym::kSymbolOne};

  Split split("split");
  split.InstanceOutputy(3U);  // 需要先指定输出个数
  split.attr.api.compute_type = af::ComputeType::kComputeSplit;

  split.x = {load1.y};
  split.attr.sched.axis = {z0.id, z1.id, z2.id};
  split.y[0].dtype = ge::DT_FLOAT16;
  *split.y[0].axis = {z0.id, z1_1.id, z2.id};
  split.y[0].dtype = ge::DT_FLOAT16;
  *split.y[0].repeats = {s0, s1_1, s2};
  *split.y[0].strides = {s1_1 * s2, s2, af::sym::kSymbolOne};

  split.y[1].dtype = ge::DT_FLOAT16;
  *split.y[1].axis = {z0.id, z1_2.id, z2.id};
  split.y[1].dtype = ge::DT_FLOAT16;
  *split.y[1].repeats = {s0, s1_2, s2};
  *split.y[1].strides = {s1_2 * s2, s2, af::sym::kSymbolOne};

  split.y[2].dtype = ge::DT_FLOAT16;
  *split.y[2].axis = {z0.id, z1_3.id, z2.id};
  split.y[2].dtype = ge::DT_FLOAT16;
  *split.y[2].repeats = {s0, s1_3, s2};
  *split.y[2].strides = {s1_3 * s2, s2, af::sym::kSymbolOne};

  Store store1("store");
  store1.y.dtype = ge::DT_FLOAT16;
  store1.x = split.y[0];
  store1.attr.sched.axis = {z0.id, z1_1.id, z2.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1_1.id, z2.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.repeats = {s0, s1_1, s2};
  *store1.y.strides = {s1_1 * s2, s2, af::sym::kSymbolOne};

  Store store2("store");
  store2.y.dtype = ge::DT_FLOAT16;
  store2.x = split.y[1];
  store2.attr.sched.axis = {z0.id, z1_2.id, z2.id};
  store2.y.dtype = ge::DT_FLOAT16;
  *store2.y.axis = {z0.id, z1_2.id, z2.id};
  store2.y.dtype = ge::DT_FLOAT16;
  *store2.y.repeats = {s0, s1_2, s2};
  *store2.y.strides = {s1_2 * s2, s2, af::sym::kSymbolOne};

  Store store3("store");
  store3.y.dtype = ge::DT_FLOAT16;
  store3.x = split.y[2];
  store3.attr.sched.axis = {z0.id, z1_3.id, z2.id};
  store3.y.dtype = ge::DT_FLOAT16;
  *store3.y.axis = {z0.id, z1_3.id, z2.id};
  store3.y.dtype = ge::DT_FLOAT16;
  *store3.y.repeats = {s0, s1_3, s2};
  *store3.y.strides = {s1_3 * s2, s2, af::sym::kSymbolOne};

  Output out1("out");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(0);

  Output out2("out");
  out2.x = store2.y;
  out2.ir_attr.SetIndex(1);

  Output out3("out");
  out3.x = store3.y;
  out3.ir_attr.SetIndex(2);

  ::ascir::utils::DumpImplGraphs({graph}, "split_last_dim_graph");
  auto split_node = graph.FindNode("split");
  ASSERT_TRUE(split_node != nullptr);
  optimize::SplitScoreFunctionGenerator generator(graph, split_node, 1);
  std::string score_func;
  EXPECT_EQ(generator.Generate(score_func), af::SUCCESS);
  EXPECT_TRUE(score_func.find("return 1;") != std::string::npos);
}

TEST_F(SplitScheduleCaseGeneratorTest, SplitScoreFunc_compile_cannot_decide) {
  // dlog_setlevel(0, 0, 1);
  af::AscGraph graph("split_last_dim_graph");
  auto s0 = af::Symbol("s0");
  auto s1 = af::Symbol("s1");
  auto s2 = af::Symbol("64");
  auto s1_1 = af::Symbol("s1_1");
  auto s1_2 = af::Symbol("s1_2");
  auto s1_3 = af::Symbol("s1_3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);
  auto z1_2 = graph.CreateAxis("z1_2", s1_2);
  auto z1_3 = graph.CreateAxis("z1_3", s1_3);

  Data data0("data0", graph);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);

  Load load1("load");
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data0.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::sym::kSymbolOne};

  Split split("split");
  split.InstanceOutputy(3U);  // 需要先指定输出个数
  split.attr.api.compute_type = af::ComputeType::kComputeSplit;

  split.x = {load1.y};
  split.attr.sched.axis = {z0.id, z1.id, z2.id};
  split.y[0].dtype = ge::DT_FLOAT16;
  *split.y[0].axis = {z0.id, z1_1.id, z2.id};
  split.y[0].dtype = ge::DT_FLOAT16;
  *split.y[0].repeats = {s0, s1_1, s2};
  *split.y[0].strides = {s1_1 * s2, s2, af::sym::kSymbolOne};

  split.y[1].dtype = ge::DT_FLOAT16;
  *split.y[1].axis = {z0.id, z1_2.id, z2.id};
  split.y[1].dtype = ge::DT_FLOAT16;
  *split.y[1].repeats = {s0, s1_2, s2};
  *split.y[1].strides = {s1_2 * s2, s2, af::sym::kSymbolOne};

  split.y[2].dtype = ge::DT_FLOAT16;
  *split.y[2].axis = {z0.id, z1_3.id, z2.id};
  split.y[2].dtype = ge::DT_FLOAT16;
  *split.y[2].repeats = {s0, s1_3, s2};
  *split.y[2].strides = {s1_3 * s2, s2, af::sym::kSymbolOne};

  Store store1("store");
  store1.y.dtype = ge::DT_FLOAT16;
  store1.x = split.y[0];
  store1.attr.sched.axis = {z0.id, z1_1.id, z2.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1_1.id, z2.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.repeats = {s0, s1_1, s2};
  *store1.y.strides = {s1_1 * s2, s2, af::sym::kSymbolOne};

  Store store2("store");
  store2.y.dtype = ge::DT_FLOAT16;
  store2.x = split.y[1];
  store2.attr.sched.axis = {z0.id, z1_2.id, z2.id};
  store2.y.dtype = ge::DT_FLOAT16;
  *store2.y.axis = {z0.id, z1_2.id, z2.id};
  store2.y.dtype = ge::DT_FLOAT16;
  *store2.y.repeats = {s0, s1_2, s2};
  *store2.y.strides = {s1_2 * s2, s2, af::sym::kSymbolOne};

  Store store3("store");
  store3.y.dtype = ge::DT_FLOAT16;
  store3.x = split.y[2];
  store3.attr.sched.axis = {z0.id, z1_3.id, z2.id};
  store3.y.dtype = ge::DT_FLOAT16;
  *store3.y.axis = {z0.id, z1_3.id, z2.id};
  store3.y.dtype = ge::DT_FLOAT16;
  *store3.y.repeats = {s0, s1_3, s2};
  *store3.y.strides = {s1_3 * s2, s2, af::sym::kSymbolOne};

  Output out1("out");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(0);

  Output out2("out");
  out2.x = store2.y;
  out2.ir_attr.SetIndex(1);

  Output out3("out");
  out3.x = store3.y;
  out3.ir_attr.SetIndex(2);

  ::ascir::utils::DumpImplGraphs({graph}, "split_last_dim_graph");
  auto split_node = graph.FindNode("split");
  ASSERT_TRUE(split_node != nullptr);
  optimize::SplitScoreFunctionGenerator generator(graph, split_node, 1);
  std::string score_func;
  EXPECT_EQ(generator.Generate(score_func), af::SUCCESS);
  EXPECT_TRUE(score_func.find("return 1;") != std::string::npos);
}

TEST_F(SplitScheduleCaseGeneratorTest, SplitScoreFunc_Not_Aligned) {
  // dlog_setlevel(0, 0, 1);
  af::AscGraph graph("split_last_dim_graph");
  auto s0 = af::Symbol(128);
  auto s1 = af::Symbol(32);
  auto s2 = af::Symbol(41);
  auto s1_1 = af::Symbol(7);
  auto s1_2 = af::Symbol(10);
  auto s1_3 = af::Symbol(15);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);
  auto z1_2 = graph.CreateAxis("z1_2", s1_2);
  auto z1_3 = graph.CreateAxis("z1_3", s1_3);

  Data data0("data0", graph);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);

  Load load1("load");
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data0.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::sym::kSymbolOne};

  Split split("split");
  split.InstanceOutputy(3U);  // 需要先指定输出个数
  split.attr.api.compute_type = af::ComputeType::kComputeSplit;

  split.x = {load1.y};
  split.attr.sched.axis = {z0.id, z1.id, z2.id};
  split.y[0].dtype = ge::DT_FLOAT16;
  *split.y[0].axis = {z0.id, z1_1.id, z2.id};
  split.y[0].dtype = ge::DT_FLOAT16;
  *split.y[0].repeats = {s0, s1_1, s2};
  *split.y[0].strides = {s1_1 * s2, s2, af::sym::kSymbolOne};

  split.y[1].dtype = ge::DT_FLOAT16;
  *split.y[1].axis = {z0.id, z1_2.id, z2.id};
  split.y[1].dtype = ge::DT_FLOAT16;
  *split.y[1].repeats = {s0, s1_2, s2};
  *split.y[1].strides = {s1_2 * s2, s2, af::sym::kSymbolOne};

  split.y[2].dtype = ge::DT_FLOAT16;
  *split.y[2].axis = {z0.id, z1_3.id, z2.id};
  split.y[2].dtype = ge::DT_FLOAT16;
  *split.y[2].repeats = {s0, s1_3, s2};
  *split.y[2].strides = {s1_3 * s2, s2, af::sym::kSymbolOne};

  Store store1("store");
  store1.y.dtype = ge::DT_FLOAT16;
  store1.x = split.y[0];
  store1.attr.sched.axis = {z0.id, z1_1.id, z2.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1_1.id, z2.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.repeats = {s0, s1_1, s2};
  *store1.y.strides = {s1_1 * s2, s2, af::sym::kSymbolOne};

  Store store2("store");
  store2.y.dtype = ge::DT_FLOAT16;
  store2.x = split.y[1];
  store2.attr.sched.axis = {z0.id, z1_2.id, z2.id};
  store2.y.dtype = ge::DT_FLOAT16;
  *store2.y.axis = {z0.id, z1_2.id, z2.id};
  store2.y.dtype = ge::DT_FLOAT16;
  *store2.y.repeats = {s0, s1_2, s2};
  *store2.y.strides = {s1_2 * s2, s2, af::sym::kSymbolOne};

  Store store3("store");
  store3.y.dtype = ge::DT_FLOAT16;
  store3.x = split.y[2];
  store3.attr.sched.axis = {z0.id, z1_3.id, z2.id};
  store3.y.dtype = ge::DT_FLOAT16;
  *store3.y.axis = {z0.id, z1_3.id, z2.id};
  store3.y.dtype = ge::DT_FLOAT16;
  *store3.y.repeats = {s0, s1_3, s2};
  *store3.y.strides = {s1_3 * s2, s2, af::sym::kSymbolOne};

  Output out1("out");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(0);

  Output out2("out");
  out2.x = store2.y;
  out2.ir_attr.SetIndex(1);

  Output out3("out");
  out3.x = store3.y;
  out3.ir_attr.SetIndex(2);

  ::ascir::utils::DumpImplGraphs({graph}, "split_last_dim_graph");
  auto split_node = graph.FindNode("split");
  ASSERT_TRUE(split_node != nullptr);
  optimize::SplitScoreFunctionGenerator generator(graph, split_node, 1);
  std::string score_func;
  EXPECT_EQ(generator.Generate(score_func), af::SUCCESS);
  EXPECT_TRUE(score_func.find("return -1;") != std::string::npos);
}

// test pass
TEST_F(SplitScheduleCaseGeneratorTest, SplitScoreFunc_stridenotalign_totalalign) {
  // dlog_setlevel(0, 0, 1);
  af::AscGraph graph("split_last_dim_graph");
  auto s0 = af::Symbol(128);
  auto s1 = af::Symbol(128);
  auto s2 = af::Symbol(4);
  auto s1_1 = af::Symbol(16);
  auto s1_2 = af::Symbol(48);
  auto s1_3 = af::Symbol(64);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);
  auto z1_2 = graph.CreateAxis("z1_2", s1_2);
  auto z1_3 = graph.CreateAxis("z1_3", s1_3);

  Data data0("data0", graph);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);

  Load load1("load");
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data0.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::sym::kSymbolOne};

  Split split("split");
  split.InstanceOutputy(3U);  // 需要先指定输出个数
  split.attr.api.compute_type = af::ComputeType::kComputeSplit;

  split.x = {load1.y};
  split.attr.sched.axis = {z0.id, z1.id, z2.id};
  split.y[0].dtype = ge::DT_FLOAT16;
  *split.y[0].axis = {z0.id, z1_1.id, z2.id};
  split.y[0].dtype = ge::DT_FLOAT16;
  *split.y[0].repeats = {s0, s1_1, s2};
  *split.y[0].strides = {s1_1 * s2, s2, af::sym::kSymbolOne};

  split.y[1].dtype = ge::DT_FLOAT16;
  *split.y[1].axis = {z0.id, z1_2.id, z2.id};
  split.y[1].dtype = ge::DT_FLOAT16;
  *split.y[1].repeats = {s0, s1_2, s2};
  *split.y[1].strides = {s1_2 * s2, s2, af::sym::kSymbolOne};

  split.y[2].dtype = ge::DT_FLOAT16;
  *split.y[2].axis = {z0.id, z1_3.id, z2.id};
  split.y[2].dtype = ge::DT_FLOAT16;
  *split.y[2].repeats = {s0, s1_3, s2};
  *split.y[2].strides = {s1_3 * s2, s2, af::sym::kSymbolOne};

  Store store1("store");
  store1.y.dtype = ge::DT_FLOAT16;
  store1.x = split.y[0];
  store1.attr.sched.axis = {z0.id, z1_1.id, z2.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1_1.id, z2.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.repeats = {s0, s1_1, s2};
  *store1.y.strides = {s1_1 * s2, s2, af::sym::kSymbolOne};

  Store store2("store");
  store2.y.dtype = ge::DT_FLOAT16;
  store2.x = split.y[1];
  store2.attr.sched.axis = {z0.id, z1_2.id, z2.id};
  store2.y.dtype = ge::DT_FLOAT16;
  *store2.y.axis = {z0.id, z1_2.id, z2.id};
  store2.y.dtype = ge::DT_FLOAT16;
  *store2.y.repeats = {s0, s1_2, s2};
  *store2.y.strides = {s1_2 * s2, s2, af::sym::kSymbolOne};

  Store store3("store");
  store3.y.dtype = ge::DT_FLOAT16;
  store3.x = split.y[2];
  store3.attr.sched.axis = {z0.id, z1_3.id, z2.id};
  store3.y.dtype = ge::DT_FLOAT16;
  *store3.y.axis = {z0.id, z1_3.id, z2.id};
  store3.y.dtype = ge::DT_FLOAT16;
  *store3.y.repeats = {s0, s1_3, s2};
  *store3.y.strides = {s1_3 * s2, s2, af::sym::kSymbolOne};

  Output out1("out");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(0);

  Output out2("out");
  out2.x = store2.y;
  out2.ir_attr.SetIndex(1);

  Output out3("out");
  out3.x = store3.y;
  out3.ir_attr.SetIndex(2);

  ::ascir::utils::DumpImplGraphs({graph}, "split_last_dim_graph");
  auto split_node = graph.FindNode("split");
  ASSERT_TRUE(split_node != nullptr);
  optimize::SplitScoreFunctionGenerator generator(graph, split_node, 1);
  std::string score_func;
  EXPECT_EQ(generator.Generate(score_func), af::SUCCESS);
  EXPECT_TRUE(score_func.find("return 1;") != std::string::npos);
}
// TEST_F(SplitScheduleCaseGeneratorTest, split_tail_dim1_scene) {
//   af::AscGraph graph("split_last_dim_graph");
//   auto s0 = af::Symbol("s0");
//   auto s2 = af::Symbol("s2");
//   auto sym2 = af::Symbol(2);

//   auto z0 = graph.CreateAxis("z0", s0);
//   auto z1 = graph.CreateAxis("z1", af::sym::kSymbolOne);
//   auto z2 = graph.CreateAxis("z2", s2);
//   auto z3 = graph.CreateAxis("z3", sym2);

//   Data data0("data0", graph);
//   data0.attr.api.type = af::ApiType::kAPITypeBuffer;
//   data0.ir_attr.SetIndex(0);
//   Load load0("load0");
//   load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
//   load0.y.dtype = ge::DT_FLOAT16;
//   load0.x = data0.y;
//   load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
//   load0.y.dtype = ge::DT_FLOAT16;
//   *load0.y.axis = {z0.id, z1.id, z2.id, z3.id};
//   load0.y.dtype = ge::DT_FLOAT16;
//   *load0.y.repeats = {s0, af::sym::kSymbolOne, s2, af::sym::kSymbolOne};
//   *load0.y.strides = {s2, af::sym::kSymbolZero, af::sym::kSymbolOne, af::sym::kSymbolZero};

//   Data data1("data1", graph);
//   data1.ir_attr.SetIndex(1);
//   data1.attr.api.type = af::ApiType::kAPITypeBuffer;
//   Load load1("load1");
//   load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
//   load1.y.dtype = ge::DT_FLOAT16;
//   load1.x = data1.y;
//   load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
//   load1.y.dtype = ge::DT_FLOAT16;
//   *load1.y.axis = {z0.id, z1.id, z2.id, z3.id};
//   load1.y.dtype = ge::DT_FLOAT16;
//   *load1.y.repeats = {s0, af::sym::kSymbolOne, s2, af::sym::kSymbolOne};
//   *load1.y.strides = {s2, af::sym::kSymbolZero, af::sym::kSymbolOne, af::sym::kSymbolZero};

//   Split split("split");
//   split.attr.api.compute_type = af::ComputeType::kComputeSplit;
//   split.y.dtype = ge::DT_FLOAT16;
//   split.x = {load0.y, load1.y};
//   split.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
//   split.y.dtype = ge::DT_FLOAT16;
//   *split.y.axis = {z0.id, z1.id, z2.id, z3.id};
//   split.y.dtype = ge::DT_FLOAT16;
//   *split.y.repeats = {s0, af::sym::kSymbolOne, s2, sym2};
//   *split.y.strides = {s2 * sym2, af::sym::kSymbolZero, sym2, af::sym::kSymbolOne};

//   Store store("store");
//   store.y.dtype = ge::DT_FLOAT16;
//   store.x = split.y;
//   store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
//   store.y.dtype = ge::DT_FLOAT16;
//   *store.y.axis = {z0.id, z1.id, z2.id, z3.id};
//   store.y.dtype = ge::DT_FLOAT16;
//   *store.y.repeats = {s0, af::sym::kSymbolOne, s2, sym2};
//   *store.y.strides = {s2 * sym2, af::sym::kSymbolZero, sym2, af::sym::kSymbolOne};

//   Output out("out");
//   out.x = store.y;
//   out.ir_attr.SetIndex(0);

//   optimize::SplitFusionCaseGenerator generator;
//   std::vector<af::AscGraph> generated_graphs;
//   std::vector<std::string> score_functions;
//   EXPECT_EQ(generator.Generate(graph, generated_graphs, score_functions), af::SUCCESS);
//   EXPECT_EQ(generated_graphs.size(), 2UL);
//   std::string load0_repeats = RepeatsToStr(generated_graphs[0], "load0");
//   EXPECT_EQ(load0_repeats, "s0, 1, s2, 1, ");
//   std::string load0_strides = StridesToStr(generated_graphs[0], "load0");
//   EXPECT_EQ(load0_strides, "s2, 0, 1, 0, ");
// }

TEST_F(SplitScheduleCaseGeneratorTest, SplitScoreFunc) {
  af::AscGraph graph("split_last_dim_graph");

  CreateSplitAscGraph(graph, {"s0"}, {"s1", "s2", "s3", "s4"}, {"2", "1"});
  auto split_node = graph.FindNode("split");
  ASSERT_TRUE(split_node != nullptr);
  optimize::SplitScoreFunctionGenerator generator(graph, split_node, 1);
  std::string score_func;
  EXPECT_EQ(generator.Generate(score_func), af::SUCCESS);
  EXPECT_TRUE(!score_func.empty());
}

TEST_F(SplitScheduleCaseGeneratorTest, SplitScoreFunc_stride_aligned) {
  af::AscGraph graph("split_last_dim_graph");

  CreateSplitAscGraph(graph, {"s0"}, {"s1", "s2", "s3", "s4"}, {"2", "8"});
  auto split_node = graph.FindNode("split");
  ASSERT_TRUE(split_node != nullptr);
  optimize::SplitScoreFunctionGenerator generator(graph, split_node, 1);
  std::string score_func;
  EXPECT_EQ(generator.Generate(score_func), af::SUCCESS);
  EXPECT_TRUE(score_func.find("return 1;") != std::string::npos);
}

TEST_F(SplitScheduleCaseGeneratorTest, SplitScoreFunc_split_dim_aligned) {
  af::AscGraph graph("split_last_dim_graph");

  CreateSplitAscGraph(graph, {"s0"}, {"4", "8", "12", "16"}, {"2", "2"});
  auto split_node = graph.FindNode("split");
  ASSERT_TRUE(split_node != nullptr);
  optimize::SplitScoreFunctionGenerator generator(graph, split_node, 1);
  std::string score_func;
  EXPECT_EQ(generator.Generate(score_func), af::SUCCESS);
  EXPECT_TRUE(score_func.find("return 1;") != std::string::npos);
}

TEST_F(SplitScheduleCaseGeneratorTest, SplitScoreFunc_split_dim_unaligned) {
  af::AscGraph graph("split_last_dim_graph");

  CreateSplitAscGraph(graph, {"s0"}, {"3", "8", "12", "16"}, {"2", "2"});
  auto split_node = graph.FindNode("split");
  ASSERT_TRUE(split_node != nullptr);
  optimize::SplitScoreFunctionGenerator generator(graph, split_node, 1);
  std::string score_func;
  EXPECT_EQ(generator.Generate(score_func), af::SUCCESS);
  EXPECT_TRUE(score_func.find("return -1;") != std::string::npos);
}

TEST_F(SplitScheduleCaseGeneratorTest, SplitScoreFunc_split_dim_dynamic) {
  af::AscGraph graph("split_last_dim_graph");

  CreateSplitAscGraph(graph, {"s0"}, {"3", "8", "12", "16"}, {"s1", "s2"});
  auto split_node = graph.FindNode("split");
  ASSERT_TRUE(split_node != nullptr);
  optimize::SplitScoreFunctionGenerator generator(graph, split_node, 1);
  std::string score_func;
  EXPECT_EQ(generator.Generate(score_func), af::SUCCESS);
  // 验证生成实际的函数
  EXPECT_TRUE(score_func.find("graph0_result0_g0_tiling_data") != std::string::npos);
}

TEST_F(SplitScheduleCaseGeneratorTest, SplitScoreFunc_split_dim_mixed_const_and_dynamic) {
  af::AscGraph graph("split_last_dim_graph");

  // output[0] 为常量 32，output[1..3] 为动态 SizeVar，构造混合 const/symbolic 场景
  CreateSplitAscGraph(graph, {"s0"}, {"32", "s1", "s2", "s3"}, {"2", "1"});
  auto split_node = graph.FindNode("split");
  ASSERT_TRUE(split_node != nullptr);
  optimize::SplitScoreFunctionGenerator generator(graph, split_node, 1);
  std::string score_func;
  EXPECT_EQ(generator.Generate(score_func), af::SUCCESS);
  // output[0] 为 const 但后续 output 非 const，应优雅降级为运行时打分函数
  EXPECT_TRUE(score_func.find("graph0_result0_g0_tiling_data") != std::string::npos);
}
}  // namespace schedule
