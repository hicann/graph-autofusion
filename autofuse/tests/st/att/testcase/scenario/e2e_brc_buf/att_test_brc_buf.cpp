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
#include "ascir_ops.h"
#include "gtest/gtest.h"
#include "gen_tiling_impl.h"
#include "tiling_code_generator.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph_construct_utils.h"
#include "common/st_scenario_utils.h"
#include "test_common_utils.h"

using namespace att;
using namespace att::test;
using namespace ge::ascir_op;

namespace {
template <typename NodeT>
void InitNode(NodeT &node, std::initializer_list<int64_t> axis, ge::DataType dtype,
              std::initializer_list<ge::Expression> repeats, std::initializer_list<ge::Expression> strides) {
  node.attr.sched.axis = axis;
  node.y.dtype = dtype;
  *node.y.axis = axis;
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}

template <typename NodeT, typename InputT>
void InitInputNode(NodeT &node, const InputT &input, std::initializer_list<int64_t> axis, ge::DataType dtype,
                   std::initializer_list<ge::Expression> repeats, std::initializer_list<ge::Expression> strides) {
  node.x = input;
  InitNode(node, axis, dtype, repeats, strides);
}

void ApplySchedulerTransform(ge::AscGraph &graph, const char *name, int64_t z1T, int64_t z1t, int64_t z2T, int64_t z2t,
                             int64_t z0z2T, int64_t z0z2TB, int64_t z0z2Tb, const std::vector<int64_t> &vec_axis) {
  auto node = graph.FindNode(name);
  graph.ApplySplit(node, z1T, z1t);
  graph.ApplySplit(node, z2T, z2t);
  graph.ApplyMerge(node, z0z2T);
  graph.ApplySplit(node, z0z2TB, z0z2Tb);
  graph.ApplyReorder(node, {z0z2TB, z0z2Tb, z1T, z1t, z2t});
  node->attr.sched.loop_axis = z1T;
  node->outputs[0].attr.vectorized_axis = vec_axis;
}

void SetGmNode(ge::AscGraph &graph, const char *name) {
  auto node = graph.FindNode(name);
  node->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  node->outputs[0].attr.mem.position = ge::Position::kPositionGM;
}

void SetQueueNode(ge::AscGraph &graph, const char *name, int32_t &tensor_id, int32_t que_id, ge::Position position,
                  int32_t depth = 1, int32_t reuse_id = ge::kIdNone) {
  auto node = graph.FindNode(name);
  node->outputs[0].attr.mem.tensor_id = tensor_id++;
  node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  node->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  node->outputs[0].attr.mem.position = position;
  node->outputs[0].attr.mem.reuse_id = reuse_id;
  node->outputs[0].attr.buf.id = ge::kIdNone;
  node->outputs[0].attr.que.id = que_id;
  node->outputs[0].attr.que.depth = depth;
  node->outputs[0].attr.que.buf_num = depth;
  node->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
}

void SetCalcBufNode(ge::AscGraph &graph, const char *name, int32_t &tensor_id, int32_t buf_id) {
  auto node = graph.FindNode(name);
  node->outputs[0].attr.mem.tensor_id = tensor_id++;
  node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  node->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  node->outputs[0].attr.mem.position = ge::Position::kPositionVecCalc;
  node->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  node->outputs[0].attr.buf.id = buf_id;
  node->outputs[0].attr.que.id = ge::kIdNone;
  node->outputs[0].attr.que.depth = ge::kIdNone;
  node->outputs[0].attr.que.buf_num = ge::kIdNone;
  node->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
}

void SetGmStoreNode(ge::AscGraph &graph, const char *name, int32_t &tensor_id) {
  auto node = graph.FindNode(name);
  node->outputs[0].attr.mem.tensor_id = tensor_id++;
  node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  node->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  node->outputs[0].attr.mem.position = ge::Position::kPositionGM;
  node->outputs[0].attr.mem.reuse_id = 0;
  node->outputs[0].attr.opt.ref_tensor = 0;
}
}  // namespace

void BrcBufBeforeAutoFuse1(ge::AscGraph &graph) {
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;

  auto Z0 = ge::Symbol("Z0");
  auto Z1 = ge::Symbol("Z1");
  auto Z2 = ge::Symbol("Z2");

  auto z0 = graph.CreateAxis("z0", Z0);
  auto z1 = graph.CreateAxis("z1", Z1);
  auto z2 = graph.CreateAxis("z2", Z2);

  auto normalAxis = {z0.id, z1.id, z2.id};
  Data input_data("input_data", graph);
  InitNode(input_data, normalAxis, ge::DT_FLOAT16, {Z0, ONE, Z2}, {Z2, ZERO, ONE});

  Load load("load");
  InitInputNode(load, input_data.y, normalAxis, ge::DT_FLOAT16, {Z0, ONE, Z2}, {Z2, ZERO, ONE});

  Cast cast0("cast0");
  InitInputNode(cast0, load.y, normalAxis, ge::DT_FLOAT, {Z0, ONE, Z2}, {Z2, ZERO, ONE});

  Broadcast broadcast("broadcast");
  InitInputNode(broadcast, cast0.y, normalAxis, ge::DT_FLOAT, {Z0, Z1, Z2}, {Z1 * Z2, Z2, ONE});

  Sum sum("sum");
  InitInputNode(sum, broadcast.y, normalAxis, ge::DT_FLOAT, {Z0, ONE, Z2}, {Z2, ZERO, ONE});

  Cast cast1("cast1");
  InitInputNode(cast1, sum.y, normalAxis, ge::DT_FLOAT16, {Z0, ONE, Z2}, {Z2, ZERO, ONE});

  Store store("store");
  InitInputNode(store, cast1.y, normalAxis, ge::DT_FLOAT16, {Z0, ONE, Z2}, {Z2, ZERO, ONE});

  Output output_data("output_data");
  output_data.x = store.y;
}

void BrcBufAfterScheduler1(ge::AscGraph &graph) {
  auto z0 = graph.GetAllAxis()[0]->id;
  auto z1 = graph.GetAllAxis()[1]->id;
  auto z2 = graph.GetAllAxis()[2]->id;

  std::tuple<ge::AxisPtr, ge::AxisPtr> split = graph.TileSplit(z1);
  auto z1T = *(std::get<0>(split));
  auto z1t = *(std::get<1>(split));
  split = graph.TileSplit(z2);
  auto z2T = *(std::get<0>(split));
  auto z2t = *(std::get<1>(split));

  auto z0z2T = *graph.MergeAxis({z0, z2T.id});
  split = graph.BlockSplit(z0z2T.id);
  auto z0z2TB = *(std::get<0>(split));
  auto z0z2Tb = *(std::get<1>(split));

  vector<int64_t> VectorizedAxis{z1t.id, z2t.id};
  for (const auto *name : {"load", "cast0", "broadcast", "sum", "cast1", "store"}) {
    ApplySchedulerTransform(graph, name, z1T.id, z1t.id, z2T.id, z2t.id, z0z2T.id, z0z2TB.id, z0z2Tb.id,
                            VectorizedAxis);
  }
}

void BrcBufAfterQueBufAlloc1(ge::AscGraph &graph) {
  int32_t tensorID = 0;
  int32_t queID = 0;
  int32_t bufID = 0;
  int32_t loadQue = queID++;
  int32_t cast0Buf = bufID++;
  int32_t broadcastBuf = bufID++;
  int32_t sumBuf = bufID++;
  int32_t cast1Buf = bufID++;

  SetGmNode(graph, "input_data");
  SetQueueNode(graph, "load", tensorID, loadQue, ge::Position::kPositionVecIn, 2, 0);
  SetCalcBufNode(graph, "cast0", tensorID, cast0Buf);
  SetCalcBufNode(graph, "broadcast", tensorID, broadcastBuf);
  SetCalcBufNode(graph, "sum", tensorID, sumBuf);
  SetCalcBufNode(graph, "cast1", tensorID, cast1Buf);
  SetGmStoreNode(graph, "store", tensorID);
}

void BrcBufBeforeAutoFuse2(ge::AscGraph &graph) {
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;

  auto Z0 = ge::Symbol("Z0");
  auto Z1 = ge::Symbol("Z1");
  auto Z2 = ge::Symbol("Z2");

  auto z0 = graph.CreateAxis("z0", Z0);
  auto z1 = graph.CreateAxis("z1", Z1);
  auto z2 = graph.CreateAxis("z2", Z2);

  auto normalAxis = {z0.id, z1.id, z2.id};
  Data input_data("input_data", graph);
  InitNode(input_data, normalAxis, ge::DT_FLOAT16, {Z0, ONE, Z2}, {Z2, ZERO, ONE});

  Load load("load");
  InitInputNode(load, input_data.y, normalAxis, ge::DT_FLOAT16, {Z0, ONE, Z2}, {Z2, ZERO, ONE});

  Cast cast0("cast0");
  InitInputNode(cast0, load.y, normalAxis, ge::DT_FLOAT, {Z0, ONE, Z2}, {Z2, ZERO, ONE});

  Broadcast broadcast("broadcast");
  InitInputNode(broadcast, cast0.y, normalAxis, ge::DT_FLOAT, {Z0, Z1, Z2}, {Z1 * Z2, Z2, ONE});

  Sum sum("sum");
  InitInputNode(sum, broadcast.y, normalAxis, ge::DT_FLOAT, {Z0, Z1, ONE}, {Z1, ONE, ZERO});

  Cast cast1("cast1");
  InitInputNode(cast1, sum.y, normalAxis, ge::DT_FLOAT16, {Z0, Z1, ONE}, {Z1, ONE, ZERO});

  Store store("store");
  InitInputNode(store, cast1.y, normalAxis, ge::DT_FLOAT16, {Z0, Z1, ONE}, {Z1, ONE, ZERO});

  Output output_data("output_data");
  output_data.x = store.y;
}

void BrcBufAfterScheduler2(ge::AscGraph &graph) {
  auto z0 = graph.GetAllAxis()[0]->id;
  auto z1 = graph.GetAllAxis()[1]->id;
  auto z2 = graph.GetAllAxis()[2]->id;

  std::tuple<ge::AxisPtr, ge::AxisPtr> split = graph.TileSplit(z1);
  auto z1T = *(std::get<0>(split));
  auto z1t = *(std::get<1>(split));
  split = graph.TileSplit(z2);
  auto z2T = *(std::get<0>(split));
  auto z2t = *(std::get<1>(split));

  auto z0z2T = *graph.MergeAxis({z0, z2T.id});
  split = graph.BlockSplit(z0z2T.id);
  auto z0z2TB = *(std::get<0>(split));
  auto z0z2Tb = *(std::get<1>(split));

  vector<int64_t> VectorizedAxis{z1t.id, z2t.id};
  for (const auto *name : {"load", "cast0", "broadcast", "sum", "cast1", "store"}) {
    ApplySchedulerTransform(graph, name, z1T.id, z1t.id, z2T.id, z2t.id, z0z2T.id, z0z2TB.id, z0z2Tb.id,
                            VectorizedAxis);
  }
}

void BrcBufAfterQueBufAlloc2(ge::AscGraph &graph) {
  int32_t tensorID = 0;
  int32_t queID = 0;
  int32_t bufID = 0;
  int32_t loadQue = queID++;
  int32_t cast0Buf = bufID++;
  int32_t broadcastBuf = bufID++;
  int32_t sumBuf = bufID++;
  int32_t cast1Buf = bufID++;

  SetGmNode(graph, "input_data");
  SetQueueNode(graph, "load", tensorID, loadQue, ge::Position::kPositionVecIn, 2, 0);
  SetCalcBufNode(graph, "cast0", tensorID, cast0Buf);
  SetCalcBufNode(graph, "broadcast", tensorID, broadcastBuf);
  SetCalcBufNode(graph, "sum", tensorID, sumBuf);
  SetCalcBufNode(graph, "cast1", tensorID, cast1Buf);
  SetGmStoreNode(graph, "store", tensorID);
}

class TestBrcBuf : public ::testing::Test {
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

TEST_F(TestBrcBuf, case_01) {
  int32_t ret = 0;
  std::vector<ge::AscGraph> graphs;
  ge::AscGraph graph1("graph1");
  graph1.SetTilingKey(1101u);
  BrcBufBeforeAutoFuse1(graph1);
  BrcBufAfterScheduler1(graph1);
  BrcBufAfterQueBufAlloc1(graph1);
  graphs.emplace_back(graph1);

  ge::AscGraph graph2("graph2");
  graph2.SetTilingKey(1102u);
  BrcBufBeforeAutoFuse2(graph2);
  BrcBufAfterScheduler2(graph2);
  BrcBufAfterQueBufAlloc2(graph2);
  graphs.emplace_back(graph2);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["gen_extra_info"] = "1";
  options["solver_type"] = "AxesReorder";
  EXPECT_EQ(GenTilingImpl("BrcBuf", graphs, options), true);
  AddHeaderGuardToFile("autofuse_tiling_func_common.h", "__AUTOFUSE_TILING_FUNC_COMMON_H__");
  ret = std::system(std::string("cp ").append(TILING_DATA_DIR).append("/tiling_func_main_brc.cpp ./ -f").c_str());
  ret = std::system(
      std::string("cp ").append(TOP_DIR).append("/autofuse/tests/st/att/testcase/stub/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "autofuse/tests/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system(
      "g++ -std=c++17 tiling_func_main_brc.cpp BrcBuf_*_tiling_func.cpp -o tiling_func_main_brc -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_brc");
}
