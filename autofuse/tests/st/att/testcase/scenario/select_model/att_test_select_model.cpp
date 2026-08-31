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
#include <regex>
#include <map>
#include <fstream>
#include "gtest/gtest.h"
#include "base/model_info.h"
#include "select_model/stub_regex.h"
#include "preprocess/test_stub.h"
#include "stub_model_info.h"
#include "tiling_code_generator.h"
#include "reuse_group_utils/reuse_group_utils.h"
#include "result_checker_utils.h"
#include "common/st_scenario_utils.h"
#include "test_common_utils.h"

using namespace att;
using namespace att::test;

class TestSelectModel : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }

  void SetUp() override {
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 1);
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
    TilingModelInfo modelInfos;
    TilingCodeGenConfig config;
    TilingCodeGenerator generator;
    model_info_ = CreateModelInfo();
    auto model_info = GetMatmulL2TileInfo();
    auto op_name = "OpTest3";
    modelInfos.emplace_back(model_info);
    modelInfos.emplace_back(model_info_);
    config.path = "./";
    config.type = TilingImplType::HIGH_PERF;
    config.gen_extra_infos = true;
    config.tiling_data_type_name = "MMTilingData";
    (void)ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, modelInfos);
    generator.GenTilingCode(op_name, modelInfos, config);
    if (ResultCheckerUtils::ReplaceLogMacros()) {
      std::cout << "Replace log macros successfully" << std::endl;
    } else {
      std::cout << "Replace log macros failed" << std::endl;
    }
    AddHeaderGuardToFile("autofuse_tiling_func_common.h", "__AUTOFUSE_TILING_FUNC_COMMON_H__");
    std::system(std::string("cp ").append(TILING_DATA_DIR).append("/tiling_func_select_main.cpp ./ -f").c_str());
    std::system(std::string("cp ").append(ST_DIR).append("/testcase/stub/op_log.h ./ -f").c_str());
    autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
    std::string build_cmd = "g++ -DDEBUG tiling_func_select_main.cpp OpTest3_*_tiling_func.cpp -I ./";
    build_cmd.append(ResultCheckerUtils::GetDependAscendIncPath()).append(" -o tiling_func_select_main");
    std::system(build_cmd.c_str());
  }

  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
    unsetenv("ASCEND_SLOG_PRINT_TO_STDOUT");
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  }
  ModelInfo model_info_;
};

TEST_F(TestSelectModel, att_test_select_model_01) {
  auto ret = std::system("./tiling_func_select_main 1024 2048 2048 -1 > ./info.log");
  EXPECT_EQ(ret, 0);

  std::ifstream file("./info.log");
  std::string str;
  std::string line;

  if (file.is_open()) {
    while (getline(file, line)) {
      str += line + "\n";
    }
    file.close();
  }
  std::map<uint64_t, double> myMap;
  std::regex pattern("The optimal objective for tiling_case_id (\\d+) is (\\d+).");
  std::sregex_iterator it(str.begin(), str.end(), pattern);
  std::sregex_iterator end;
  while (it != end) {
    uint64_t key = stoi(it->str(1));
    double value = stoi(it->str(2));
    myMap[key] = value;
    it++;
  }
  uint64_t used_key = ObtainOutput("tiling_key = (\\d+)");
  bool has_visited = false;
  for (auto it = myMap.begin(); it != myMap.end(); ++it) {
    if (it->first == used_key) {
      has_visited = true;
      break;
    }
  }
  EXPECT_TRUE(has_visited);
  double min_obj = myMap[used_key];
  for (auto it = myMap.begin(); it != myMap.end(); ++it) {
    if (it->first != used_key) {
      EXPECT_GT(it->second, min_obj);
    }
  }
}

TEST_F(TestSelectModel, att_test_select_model_03) {
  auto ret = std::system("./tiling_func_select_main 1 1 0 -1 > ./info.log");
  EXPECT_TRUE(ExistOutput("failed"));
}

TEST_F(TestSelectModel, att_test_select_model_04) {
  auto ret = std::system("./tiling_func_select_main 8192 2048 2048 -1 > ./info.log");
  EXPECT_EQ(ret, 0);
  std::ifstream file("./info.log");
  std::string str;
  std::string line;

  if (file.is_open()) {
    while (getline(file, line)) {
      str += line + "\n";
    }
    file.close();
  }
  std::map<uint64_t, double> myMap;
  std::regex pattern("The optimal objective for tiling_case_id (\\d+) is (\\d+).");
  std::sregex_iterator it(str.begin(), str.end(), pattern);
  std::regex pattern2("Objective value for case(\\d+) is (\\d+).");
  std::sregex_iterator it2(str.begin(), str.end(), pattern2);
  std::sregex_iterator end;
  EXPECT_NE(it, end);
  while (it != end) {
    uint64_t key = stoi(it->str(1));
    double value = stoi(it->str(2));
    myMap[key] = value;
    it++;
  }
  EXPECT_NE(it2, end);
  while (it2 != end) {
    uint64_t key = stoi(it2->str(1));
    double value = stoi(it2->str(2));
    double ratio = (std::abs(myMap[key] - value) / value);
    EXPECT_LE(ratio, 0.05);
    it2++;
  }
}

TEST_F(TestSelectModel, att_test_select_model_05) {
  auto ret = std::system("./tiling_func_select_main 8192 2048 2048 -1 > ./info.log");
  EXPECT_EQ(ret, 0);

  uint64_t corenum = ObtainOutput("block_dim = (\\d+)");
  EXPECT_EQ(corenum, 20);
}
