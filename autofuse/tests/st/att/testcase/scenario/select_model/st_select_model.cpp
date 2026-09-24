/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include "gtest/gtest.h"
#include "base/base_types.h"
#include "base/model_info.h"
#include "generator/preprocess/args_manager.h"
#include "gen_model_info_content/stub_graph.h"
#include "select_model/stub_regex.h"
#include "preprocess/test_stub.h"
#include "stub_model_info.h"
#include "tiling_code_generator.h"
#include "common/st_scenario_utils.h"
#include "test_common_utils.h"

using namespace att;

class SelectModelST : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    unsetenv("ASCEND_SLOG_PRINT_TO_STDOUT");
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 1);
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
    std::cout << "Test begin." << std::endl;
  }

  void SetUp() override {}

  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
  }
};

/*
TEST_F(SelectModelST, st_test_select_model) {
  std::vector<ge::AscGraph> graphs;
  std::string json_info;
  std::string op_name = "OpTest";
  TilingModelInfo model_info_list;
  TilingCodeGenConfig config;
  TilingCodeGenerator generator;
  ge::AscGraph graph1("graph1");
  ge::AscGraph graph2("graph2");

  FaBeforeAutoFuse(graph1);
  FaAfterSchedulerSplitBlockFirst(graph1);
  FaAfterQueBufAlloc(graph1);

  FaBeforeAutoFuse(graph2);
  FaAfterScheduler(graph2);
  FaAfterQueBufAlloc(graph2);

  graphs.emplace_back(graph1);
  graphs.emplace_back(graph2);
  GenerateModelInfo(graphs, model_info_list);

  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = true;
  config.tiling_data_type_name = "TilingData";

  generator.GenTilingCode(op_name, model_info_list, config);
  std::system("cp ../test/st/testcase/tiling_func_select_main_fa.cpp ./ -f");
  std::system(std::string("cp ").append(ST_DIR).append("/testcase/stub/op_log.h ./ -f").c_str());
  autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  std::system("g++ -std=c++17 -DDEBUG tiling_func_select_main_fa.cpp OpTest_tiling_func.cpp -I ./ "
              "-o tiling_func_select_main_fa");
  auto ret = std::system("./tiling_func_select_main_fa > ./info.log");
  EXPECT_EQ(ret, 0);

  bool valid_tilingkey = CheckValidTilingkey();
  EXPECT_EQ(valid_tilingkey, true);
}
*/
