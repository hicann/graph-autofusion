/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common_gen_utils.h"
#include <iostream>
#include <fstream>
#include "gtest/gtest.h"
#include "tiling_code_generator.h"
#include "api_tiling_gen/gen_api_tiling.h"
#include "stub_graph.h"
#include "base/att_const_values.h"
#include "gen_tiling_impl.h"
#include "tests/ut/att/utils/graph_construct_utils.h"
#include "common/st_scenario_utils.h"
#include "test_common_utils.h"

using namespace att;
using namespace att::test;
using namespace ge::ascir_op;
class TestSchedule : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 1);
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
  }

  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
    unsetenv("ASCEND_SLOG_PRINT_TO_STDOUT");
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  }
};

/**
 * 测试场景1：按基本块分多核
 * (z0, z1) -> (z0T, z0t, z1T, z1t) -> (z0T, z1T, z0t, z1t) -> (z0Tz1T, z0t, z1t) -> (z0Tz1TB, z0Tz1Tb, z0t, z1t)
 */
void OriGraphBasicBlock(ge::AscGraph &graph) {
  auto ONE = af::sym::kSymbolOne;
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto axis_list = {z0.id, z1.id};
  std::initializer_list<Expr> repeats = {s0, s1};
  std::initializer_list<Expr> strides = {s1, ONE};
  Data data("data", graph);
  data.attr.sched.axis = axis_list;
  data.y.dtype = ge::DT_FLOAT16;
  *data.y.axis = axis_list;
  *data.y.repeats = repeats;
  *data.y.strides = strides;

  Load load("load");
  load.x = data.y;
  load.attr.sched.axis = axis_list;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = axis_list;
  *load.y.repeats = repeats;
  *load.y.strides = strides;

  Output data_out("out");
  data_out.x = load.y;
  data_out.y.dtype = ge::DT_FLOAT16;
  *data_out.y.axis = axis_list;
  *data_out.y.repeats = repeats;
  *data_out.y.strides = strides;
}

void ScheduleBasicBlock(ge::AscGraph &graph) {
  auto z0 = graph.GetAllAxis()[0]->id;
  auto z1 = graph.GetAllAxis()[1]->id;
  auto [z1T, z1t] = graph.TileSplit(z1);
  auto [z0T, z0t] = graph.TileSplit(z0);
  auto z0Tz1T = *(graph.MergeAxis({z0T->id, z1T->id}));
  auto [z0Tz1TB, z0Tz1Tb] = graph.BlockSplit(z0Tz1T.id);

  auto data = graph.FindNode("data");
  data->outputs[0].attr.vectorized_axis = {z0, z1};

  auto load = graph.FindNode("load");
  graph.ApplySplit(load, z1T->id, z1t->id);
  graph.ApplySplit(load, z0T->id, z0t->id);
  graph.ApplyReorder(load, {z0T->id, z1T->id, z0t->id, z1t->id});
  graph.ApplyMerge(load, z0Tz1T.id);
  graph.ApplySplit(load, z0Tz1TB->id, z0Tz1Tb->id);
  load->attr.sched.loop_axis = z0Tz1Tb->id;
  load->outputs[0].attr.vectorized_axis = {z0t->id, z1t->id};
}

void MemoryAllocBasicBlock(ge::AscGraph &graph) {
  int32_t tensorID = 0;
  int32_t loadBuf = 0;
  auto load = graph.FindNode("load");
  load->outputs[0].attr.mem.tensor_id = tensorID++;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  load->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.buf.id = loadBuf;
}

/**
 * 测试场景2：切reduce轴，reduce结果为一个scalar
 */
void OriGraphReduceScalar(ge::AscGraph &graph) {
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto axis_list = {z0.id, z1.id};
  std::initializer_list<Expr> repeats = {s0, s1};
  std::initializer_list<Expr> strides = {s1, ONE};
  Data data("data", graph);
  data.attr.sched.axis = axis_list;
  data.y.dtype = ge::DT_FLOAT16;
  *data.y.axis = axis_list;
  *data.y.repeats = repeats;
  *data.y.strides = strides;

  Load load("load");
  load.x = data.y;
  load.attr.sched.axis = axis_list;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = axis_list;
  *load.y.repeats = repeats;
  *load.y.strides = strides;

  Sum reduce_sum("reduce_sum");
  reduce_sum.x = load.y;
  reduce_sum.attr.sched.axis = axis_list;
  reduce_sum.y.dtype = ge::DT_FLOAT16;
  *reduce_sum.y.axis = axis_list;
  *reduce_sum.y.repeats = repeats;
  *reduce_sum.y.strides = {ONE, ZERO};

  Output data_out("out");
  data_out.x = reduce_sum.y;
  data_out.y.dtype = ge::DT_FLOAT16;
  *data_out.y.axis = axis_list;
  *data_out.y.repeats = repeats;
  *data_out.y.strides = {ONE, ZERO};
}

void ScheduleReduceScalar(ge::AscGraph &graph) {
  auto z0 = graph.GetAllAxis()[0]->id;
  auto z1 = graph.GetAllAxis()[1]->id;
  auto [z1T, z1t] = graph.TileSplit(z1);
  auto [z0B, z0b] = graph.BlockSplit(z0);

  auto data = graph.FindNode("data");
  data->outputs[0].attr.vectorized_axis = {z0, z1};

  auto load = graph.FindNode("load");
  graph.ApplySplit(load, z1T->id, z1t->id);
  graph.ApplySplit(load, z0B->id, z0b->id);
  load->attr.sched.loop_axis = z1T->id;
  load->outputs[0].attr.vectorized_axis = {z1t->id};

  auto reduce_sum = graph.FindNode("reduce_sum");
  graph.ApplySplit(reduce_sum, z1T->id, z1t->id);
  graph.ApplySplit(reduce_sum, z0B->id, z0b->id);
  reduce_sum->attr.sched.loop_axis = z1T->id;
  reduce_sum->outputs[0].attr.vectorized_axis = {z1t->id};
}

void MemoryReduceScalar(ge::AscGraph &graph) {
  int32_t tensorID = 0;
  int32_t loadBuf = 0;
  int32_t reudeBuf = 1;
  auto load = graph.FindNode("load");
  load->outputs[0].attr.mem.tensor_id = tensorID++;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  load->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.buf.id = loadBuf;
  auto reduce_sum = graph.FindNode("reduce_sum");
  reduce_sum->outputs[0].attr.mem.tensor_id = tensorID++;
  reduce_sum->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  reduce_sum->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  reduce_sum->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  reduce_sum->outputs[0].attr.buf.id = reudeBuf;
}

TEST_F(TestSchedule, case0) {
  int32_t ret = 0;
  std::vector<ge::AscGraph> graphs;
  std::vector<att::ModelInfo> model_info_list;

  ge::AscGraph basic_block_graph("BASIC_BLOCK_GRAPH");
  OriGraphBasicBlock(basic_block_graph);
  ScheduleBasicBlock(basic_block_graph);
  MemoryAllocBasicBlock(basic_block_graph);
  graphs.emplace_back(basic_block_graph);

  ge::AscGraph reduce_scalar_graph("REDUCE_SCALAR_GRAPH");
  OriGraphReduceScalar(reduce_scalar_graph);
  ScheduleReduceScalar(reduce_scalar_graph);
  MemoryReduceScalar(reduce_scalar_graph);
  graphs.emplace_back(reduce_scalar_graph);

  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);
  std::map<std::string, std::string> options;
  options.emplace(kTilingDataTypeName, "NpuKernel0TilingData");
  options.emplace(kOutputFilePath, kDefaultFilePath);
  options.emplace(kDumpDebugInfo, kDefaultFilePath);
  options.emplace(kGenConfigType, "HighPerf");
  std::string op_name = "OpTest5";
  options.emplace(kGenTilingDataDef, "1");
  auto res = GenTilingImpl(op_name, graphs, options);
  EXPECT_EQ(res, true);
  AddHeaderGuardToFile("autofuse_tiling_func_common.h", "__AUTOFUSE_TILING_FUNC_COMMON_H__");
  ret = std::system(std::string("cp ").append(TILING_DATA_DIR).append("/tiling_func_main_special.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/stub/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system(
      "g++ -std=c++17 tiling_func_main_special.cpp OpTest5_*tiling_func.cpp -I ./ -o tiling_func_main_special -g");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_special");
}
