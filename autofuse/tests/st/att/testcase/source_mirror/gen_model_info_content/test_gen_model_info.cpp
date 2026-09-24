/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include "tiling_code_generator.h"
#include "api_tiling_gen/gen_api_tiling.h"
#include "stub_graph.h"
#include "base/att_const_values.h"
#include "gen_tiling_impl.h"
#include "common/st_scenario_utils.h"
#include "test_common_utils.h"

using namespace att;
using namespace ge::ascir_op;
class TestGenModelInfo : public ::testing::Test {
 public:
  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
  }
};

/*
TEST_F(TestGenModelInfo, case0)
{
  std::vector<ge::AscGraph> graphs;
  std::string json_info;
  std::string op_name = "FA";
  std::vector<att::ModelInfo> model_info_list;

  ge::AscGraph graph("graph");
  FaBeforeAutoFuse(graph);
  FaAfterScheduler(graph);
  FaAfterQueBufAlloc(graph);
  graphs.emplace_back(graph);
  att::GenerateModelInfo(graphs, model_info_list);
  att::MakeJson(model_info_list, json_info);
  std::cout << json_info << std::endl;

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = true;
  TilingCodeGenerator generator;
  EXPECT_EQ(generator.GenTilingCode(op_name, model_info_list, config), af::SUCCESS);

  auto ret = std::system(std::string("cp ").append(TILING_DATA_DIR).append("/tiling_func_main_fa.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/stub/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ -std=c++17 tiling_func_main_fa.cpp FA_tiling_func.cpp -I ./ -o tiling_func_main_fa");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_fa");
}

TEST_F(TestGenModelInfo, case0_gen_tiling_impl)
{
  std::vector<ge::AscGraph> graphs;
  std::string json_info;
  std::string op_name = "FA";
  std::vector<att::ModelInfo> model_info_list;

  ge::AscGraph graph("graph");
  FaBeforeAutoFuse(graph);
  FaAfterScheduler(graph);
  FaAfterQueBufAlloc(graph);
  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;
  options.emplace(kGenConfigType, "HighPerf");
  EXPECT_EQ(att::GenTilingImpl(op_name, graphs, options), true);

  auto ret = std::system(std::string("cp ").append(TILING_DATA_DIR).append("/tiling_func_main_fa.cpp ./ -f").c_str());
  ret = std::system("sed -i 's/TilingData/graphTilingData/g' tiling_func_main_fa.cpp");
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/stub/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ -std=c++17 tiling_func_main_fa.cpp FA_tiling_func.cpp -I ./ -o tiling_func_main_fa");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_fa");
}

TEST_F(TestGenModelInfo, case1)
{
  std::vector<ge::AscGraph> graphs;

  ge::AscGraph graph("graph");
  FaBeforeAutoFuse(graph);
  FaAfterScheduler(graph);
  FaAfterQueBufAlloc(graph);
  EXPECT_EQ(GetApiTilingInfo(graph, "TilingData"), af::SUCCESS);
  for (auto node_tiling : ApiTilingMgr::Instance().GetApiTilingFunc(0u)) {
    std::cout  << node_tiling.first << " " << node_tiling.second << std::endl;
  }
  for (auto node_tiling : ApiTilingMgr::Instance().GetApiTilingDataType(0u)) {
    std::cout  << node_tiling.first << " " << node_tiling.second.first << " " << node_tiling.second.second << std::endl;
  }
}

TEST_F(TestGenModelInfo, case2_axes_tiling_data_gen)
{
  std::vector<ge::AscGraph> graphs;
  std::string json_info;
  std::string op_name = "FA";
  std::vector<att::ModelInfo> model_info_list;
  ge::AscGraph graph("graph");
  FaBeforeAutoFuse(graph);
  FaAfterSchedulerSplitBlockFirst(graph);
  FaAfterQueBufAlloc(graph);

  graphs.emplace_back(graph);
  att::GenerateModelInfo(graphs, model_info_list);
  att::MakeJson(model_info_list, json_info);
  std::cout << json_info << std::endl;

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = true;
  TilingCodeGenerator generator;
  EXPECT_EQ(generator.GenTilingCode(op_name, model_info_list, config), af::SUCCESS);
  auto ret = std::system(std::string("cp ").append(TILING_DATA_DIR).append("/tiling_func_main_fa.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/stub/op_log.h ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("g++ -std=c++17 tiling_func_main_fa.cpp FA_tiling_func.cpp -I ./ -o tiling_func_main_fa");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_fa");
}

TEST_F(TestGenModelInfo, test_sketch_gen)
{
  std::vector<ge::AscGraph> graphs;
  std::string json_info;
  std::string op_name = "FA";
  std::vector<att::ModelInfo> model_info_list;
  ge::AscGraph graph("graph");
  FaBeforeAutoFuse(graph);
  FaAfterScheduler(graph);
  FaAfterQueBufAlloc(graph);

  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;
  options[kDumpDebugInfo] = "./";
  EXPECT_EQ(att::GenerateModelInfo(graphs, model_info_list, options), af::SUCCESS);
}

TEST_F(TestGenModelInfo, input_check_01)
{
  std::vector<ge::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  ge::AscGraph graph("graph");
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;

  auto B = ge::Symbol("B");
  auto N = ge::Symbol("N");
  auto G = ge::Symbol("G");
  auto b = graph.CreateAxis("b", B, 1, 100000);
  auto n = graph.CreateAxis("n", N, 1, 100000);
  auto g = graph.CreateAxis("g", G, 1, 100000);

  Data query("query", graph);
  query.attr.sched.axis = {b.id, n.id, g.id};
  query.y.dtype = ge::DT_FLOAT16;
  *query.y.axis = {b.id, n.id, g.id};
  *query.y.repeats = {B, N, G};

  Data key("key", graph);
  key.attr.sched.axis = {b.id, n.id, g.id};
  key.y.dtype = ge::DT_FLOAT16;
  *key.y.axis = {b.id, n.id, g.id};
  *key.y.repeats = {B, N, G};

  query.index = 1;
  query.axis_continuous_map = {{1, 2}, {3}, {4}};
  query.format = ge::FORMAT_ND;

  key.index = 2;
  key.axis_continuous_map = {{1, 2}, {3}, {4}};

  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;

  EXPECT_EQ(att::GenerateModelInfo(graphs, model_info_list, options), af::SUCCESS);
}

TEST_F(TestGenModelInfo, input_check_02)
{
  std::vector<ge::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  ge::AscGraph graph("graph");
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;

  auto B = ge::Symbol("B");
  auto N = ge::Symbol("N");
  auto G = ge::Symbol("G");
  auto b = graph.CreateAxis("b", B, 1, 100000);
  auto n = graph.CreateAxis("n", N, 1, 100000);
  auto g = graph.CreateAxis("g", G, 1, 100000);

  Data query("query", graph);
  query.attr.sched.axis = {b.id, n.id, g.id};
  query.y.dtype = ge::DT_FLOAT16;
  *query.y.axis = {b.id, n.id, g.id};
  *query.y.repeats = {B, N, G};

  Data key("key", graph);
  key.attr.sched.axis = {b.id, n.id, g.id};
  key.y.dtype = ge::DT_FLOAT16;
  *key.y.axis = {b.id, n.id, g.id};
  *key.y.repeats = {B, N, G};

  query.index = 1;
  query.axis_continuous_map = {{1, 2}, {3}};
  query.attr.sched.axis = {b.id, n.id, g.id};

  key.index = 2;
  key.axis_continuous_map = {{1, 2}, {3}, {4}};
  key.attr.sched.axis = {b.id, n.id, g.id};

  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;

  EXPECT_NE(att::GenerateModelInfo(graphs, model_info_list, options), af::SUCCESS);
}

TEST_F(TestGenModelInfo, input_check_03)
{
  std::vector<ge::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  ge::AscGraph graph("graph");
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;

  auto B = ge::Symbol("B");
  auto N = ge::Symbol("N");
  auto G = ge::Symbol("G");
  auto b = graph.CreateAxis("b", B, 1, 100000);
  auto n = graph.CreateAxis("n", N, 1, 100000);
  auto g = graph.CreateAxis("g", G, 1, 100000);

  Data query("query", graph);
  query.attr.sched.axis = {b.id, n.id, g.id};
  query.y.dtype = ge::DT_FLOAT16;
  *query.y.axis = {b.id, n.id, g.id};
  *query.y.repeats = {B, N, G};

  Data key("key", graph);
  key.attr.sched.axis = {b.id, n.id, g.id};
  key.y.dtype = ge::DT_FLOAT16;
  *key.y.axis = {b.id, n.id, g.id};
  *key.y.repeats = {B, N, G};

  query.index = 1;
  query.axis_continuous_map = {{1, 2}, {3}, {4}};
  query.attr.sched.axis = {b.id, n.id, g.id};

  key.index = 1;
  key.axis_continuous_map = {{1, 2}, {3}, {4}};
  key.attr.sched.axis = {b.id, n.id, g.id};

  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;

  EXPECT_NE(att::GenerateModelInfo(graphs, model_info_list, options), af::SUCCESS);
}

TEST_F(TestGenModelInfo, input_check_04)
{
  std::vector<ge::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  ge::AscGraph graph("graph");
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;

  auto B = ge::Symbol("B");
  auto N = ge::Symbol("N");
  auto G = ge::Symbol("G");
  auto b = graph.CreateAxis("b", B, 1, 100000);
  auto n = graph.CreateAxis("n", N, 1, 100000);
  auto g = graph.CreateAxis("g", G, 1, 100000);

  Data query("query", graph);
  query.attr.sched.axis = {b.id, n.id, g.id};
  query.y.dtype = ge::DT_FLOAT16;
  *query.y.axis = {b.id, n.id, g.id};
  *query.y.repeats = {B, N, G};

  Data key("key", graph);
  key.attr.sched.axis = {b.id, n.id, g.id};
  key.y.dtype = ge::DT_FLOAT16;
  *key.y.axis = {b.id, n.id, g.id};
  *key.y.repeats = {B, N, G};

  query.index = 1;
  query.axis_continuous_map = {{1, 2, 3}, {3}, {4}};
  query.attr.sched.axis = {b.id, n.id, g.id};

  key.index = 2;
  key.axis_continuous_map = {{1, 2}, {3}, {4}};
  key.attr.sched.axis = {b.id, n.id, g.id};

  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;

  EXPECT_NE(att::GenerateModelInfo(graphs, model_info_list, options), af::SUCCESS);
}

TEST_F(TestGenModelInfo, input_check_05)
{
  std::vector<ge::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  ge::AscGraph graph("graph");
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;

  auto B = ge::Symbol("B");
  auto N = ge::Symbol("N");
  auto G = ge::Symbol("G");
  auto b = graph.CreateAxis("b", B, 1, 100000);
  auto n = graph.CreateAxis("n", N, 1, 100000);
  auto g = graph.CreateAxis("g", G, 1, 100000);

  Data query("query", graph);
  query.attr.sched.axis = {b.id, n.id, g.id};
  query.y.dtype = ge::DT_FLOAT16;
  *query.y.axis = {b.id, n.id, g.id};
  *query.y.repeats = {B, N, G};

  Data key("key", graph);
  key.attr.sched.axis = {b.id, n.id, g.id};
  key.y.dtype = ge::DT_FLOAT16;
  *key.y.axis = {b.id, n.id, g.id};
  *key.y.repeats = {B, N, G};

  std::tuple<ge::AxisPtr, ge::AxisPtr> split = graph.TileSplit(b.id);
  auto bT = *(std::get<0>(split));
  auto bt = *(std::get<1>(split));

  query.index = 1;
  query.axis_continuous_map = {{1, 2}, {3}, {4}};
  query.attr.sched.axis = {bT.id, n.id, g.id};

  key.index = 2;
  key.axis_continuous_map = {{1, 2}, {3}, {4}};
  key.attr.sched.axis = {b.id, n.id, g.id};

  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;

  EXPECT_NE(att::GenerateModelInfo(graphs, model_info_list, options), af::SUCCESS);
}

TEST_F(TestGenModelInfo, input_check_06)
{
  std::vector<ge::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  ge::AscGraph graph("graph");
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;

  auto B = ge::Symbol("B");
  auto N = ge::Symbol("N");
  auto G = ge::Symbol("G");
  auto M = ge::Symbol("G");
  auto b = graph.CreateAxis("b", B, 1, 100000);
  auto n = graph.CreateAxis("n", N, 1, 100000);
  auto g = graph.CreateAxis("g", G, 1, 100000);
  auto m = graph.CreateAxis("g", M, 1, 100000);

  Data query("query", graph);
  query.attr.sched.axis = {b.id, n.id, g.id};
  query.y.dtype = ge::DT_FLOAT16;
  *query.y.axis = {b.id, n.id, g.id};
  *query.y.repeats = {B, N, G};

  Data key("key", graph);
  key.attr.sched.axis = {b.id, n.id, g.id};
  key.y.dtype = ge::DT_FLOAT16;
  *key.y.axis = {b.id, n.id, g.id};
  *key.y.repeats = {B, N, G};

  query.index = 1;
  query.axis_continuous_map = {{1, 2}, {3}, {4}};
  query.attr.sched.axis = {b.id, n.id, g.id};

  key.index = 2;
  key.axis_continuous_map = {{1, 2}, {3}, {4}};
  key.attr.sched.axis = {b.id, n.id, g.id};

  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;

  EXPECT_NE(att::GenerateModelInfo(graphs, model_info_list, options), af::SUCCESS);
}

TEST_F(TestGenModelInfo, OptionTest01) {
    std::vector<ge::AscGraph> graphs;
    ge::AscGraph graph("graph");
    att::FaBeforeAutoFuse(graph);
    att::FaAfterScheduler(graph);
    att::FaAfterQueBufAlloc(graph);
    graphs.emplace_back(graph);
    std::map<std::string, std::string> options;
    options.emplace(kTilingDataTypeName, "NpuKernel0TilingData");
    options.emplace(kOutputFilePath, kDefaultFilePath);
    options.emplace(kDurationLevelName, "1");
    options.emplace(kGenTilingDataDef, "1");
    options.emplace("invalidOption01", "xxx");
    std::string op_name = "FA";
    bool res = GenTilingImpl(op_name, graphs, options);
    EXPECT_EQ(res, false);
}

TEST_F(TestGenModelInfo, OptionTest02) {
    std::vector<ge::AscGraph> graphs;
    ge::AscGraph graph("graph");
    att::FaBeforeAutoFuse(graph);
    att::FaAfterScheduler(graph);
    att::FaAfterQueBufAlloc(graph);
    graphs.emplace_back(graph);
    std::map<std::string, std::string> options;
    options.emplace(kTilingDataTypeName, "NpuKernel0TilingData");
    options.emplace(kOutputFilePath, kDefaultFilePath);
    options.emplace(kDurationLevelName, "1");
    options.emplace(kGenTilingDataDef, "2");
    options.emplace(kGenConfigType, "HighPerf");
    std::string op_name = "FA";
    bool res = GenTilingImpl(op_name, graphs, options);
    EXPECT_EQ(res, false);
}

TEST_F(TestGenModelInfo, OptionTest03) {
    std::vector<ge::AscGraph> graphs;
    ge::AscGraph graph("graph");
    att::FaBeforeAutoFuse(graph);
    att::FaAfterScheduler(graph);
    att::FaAfterQueBufAlloc(graph);
    graphs.emplace_back(graph);
    std::map<std::string, std::string> options;
    options.emplace(kTilingDataTypeName, "");
    options.emplace(kOutputFilePath, kDefaultFilePath);
    options.emplace(kDurationLevelName, "1");
    options.emplace(kGenTilingDataDef, "1");
    options.emplace(kGenTilingDataDef, "1");
    std::string op_name = "FA";
    bool res = GenTilingImpl(op_name, graphs, options);
    EXPECT_EQ(res, false);
}

TEST_F(TestGenModelInfo, OptionTest04) {
    std::vector<ge::AscGraph> graphs;
    ge::AscGraph graph("graph");
    att::FaBeforeAutoFuse(graph);
    att::FaAfterScheduler(graph);
    att::FaAfterQueBufAlloc(graph);
    graphs.emplace_back(graph);
    std::map<std::string, std::string> options;
    options.emplace(kTilingDataTypeName, "NpuKernel0TilingData");
    options.emplace(kDurationLevelName, "1");
    options.emplace(kGenTilingDataDef, "1");
    options.emplace(kOutputFilePath, "./$#@");
    std::string op_name = "FA";
    bool res = GenTilingImpl(op_name, graphs, options);
    EXPECT_EQ(res, false);
}

TEST_F(TestGenModelInfo, OptionTest05)
{
  std::vector<ge::AscGraph> graphs;
  ge::AscGraph graph("graph");
  FaBeforeAutoFuse(graph);
  FaAfterScheduler(graph);
  FaAfterQueBufAlloc(graph);
  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;
  options.emplace(kTilingDataTypeName, "TilingData");
  options.emplace(kOutputFilePath, "./");
  options.emplace(kDurationLevelName, "1");
  options.emplace(kGenTilingDataDef, "1");
  options.emplace(kGenConfigType, "AxesReorder");
  std::string op_name = "FA";
  EXPECT_EQ(GenTilingImpl(op_name, graphs, options), true);

  auto ret = std::system(std::string("cp ").append(TILING_DATA_DIR).append("/tiling_func_main_fa.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/stub/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ -std=c++17 tiling_func_main_fa.cpp FA_tiling_func.cpp -I ./ -o tiling_func_main_fa");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_fa");
  EXPECT_EQ(ret, 0);
}

TEST_F(TestGenModelInfo, OptionTest06) {
  std::vector<ge::AscGraph> graphs;
  ge::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;
  options.emplace(kTilingDataTypeName, "*_ttt");
  options.emplace(kDurationLevelName, "1");
  options.emplace(kGenTilingDataDef, "1");
  options.emplace(kOutputFilePath, "./$#@");
  std::string op_name = "OpTest";
  bool res = GenTilingImpl(op_name, graphs, options);
  EXPECT_EQ(res, false);
}

TEST_F(TestGenModelInfo, OptionTest07) {
  std::vector<ge::AscGraph> graphs;
  ge::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;
  options.emplace(kTilingDataTypeName, "");
  options.emplace(kDurationLevelName, "1");
  options.emplace(kGenTilingDataDef, "1");
  std::string op_name = "OpTest";
  bool res = GenTilingImpl(op_name, graphs, options);
  EXPECT_EQ(res, false);
}

TEST_F(TestGenModelInfo, OptionTest08) {
  std::vector<ge::AscGraph> graphs;
  ge::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;
  options.emplace(kDurationLevelName, "-1");
  options.emplace(kGenTilingDataDef, "1");
  std::string op_name = "OpTest";
  std::map<std::string, std::string> inner_options;
  bool res = GenTilingImpl(op_name, graphs, options);
  EXPECT_EQ(res, false);
}

TEST_F(TestGenModelInfo, OptionTest09) {
  std::vector<ge::AscGraph> graphs;
  ge::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;
  options.emplace(kDurationLevelName, "");
  options.emplace(kGenTilingDataDef, "1");
  std::string op_name = "OpTest";
  std::map<std::string, std::string> inner_options;
  bool res = GenTilingImpl(op_name, graphs, options);
  EXPECT_EQ(res, false);
}

TEST_F(TestGenModelInfo, case_axes_reorder)
{
  std::vector<ge::AscGraph> graphs;
  std::string json_info;
  std::string op_name = "FA";
  std::vector<att::ModelInfo> model_info_list;

  ge::AscGraph graph("graph");
  FaBeforeAutoFuse(graph);
  FaAfterScheduler(graph);
  FaAfterQueBufAlloc(graph);
  graphs.emplace_back(graph);


  std::map<std::string, std::string> options;
  options[kOutputFilePath] = "./";
  options[kGenConfigType] = "AxesReorder";
  options[kTilingDataTypeName] = "TilingData";
  options[kGenExtraInfo] = kIsTrue;
  std::cout << json_info << std::endl;

  EXPECT_TRUE(GenTilingImpl("FA", graphs, options));

  auto ret = std::system(std::string("cp ").append(TILING_DATA_DIR).append("/tiling_func_main_fa.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/stub/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ -std=c++17 tiling_func_main_fa.cpp FA_tiling_func.cpp -I ./ -o tiling_func_main_fa");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_fa");
}
*/
