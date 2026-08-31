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
#include <fstream>
#include <regex>
#include "gtest/gtest.h"
#include "base/model_info.h"
#include "stub_model_info.h"
#include "tiling_code_generator.h"
#include "reuse_group_utils/reuse_group_utils.h"
#include "result_checker_utils.h"
#include "common/st_scenario_utils.h"
#include "test_common_utils.h"

using namespace att;
using namespace att::test;

class TestAttLog : public ::testing::Test {
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

  void SetUp() override {
    TilingModelInfo modelInfos;
    TilingCodeGenConfig config;
    TilingCodeGenerator generator;
    model_info_ = CreateModelInfo();
    auto op_name = "OpTest6";
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
    std::system(std::string("cp ").append(TILING_DATA_DIR).append("/tiling_func_attlog_main.cpp ./ -f").c_str());
    std::system(std::string("cp ").append(ST_DIR).append("/testcase/stub/op_log.h ./ -f").c_str());
    autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
    std::string build_cmd =
        "g++ -DDEBUG tiling_func_attlog_main.cpp OpTest6_*_tiling_func.cpp -DLOG_CPP -I ./ -I ./stub";
    build_cmd.append(ResultCheckerUtils::GetDependAscendIncPath()).append(" -o tiling_func_log_main");
    std::system(build_cmd.c_str());
  }

  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
  }
  ModelInfo model_info_;
};

bool CheckOutput(const std::string &pattern) {
  std::string str;
  std::string line;
  std::ifstream file("./att_info.log");

  if (file.is_open()) {
    while (getline(file, line)) {
      str += line + "\n";
    }
    file.close();
  }
  std::regex cur_pattern(pattern);
  std::sregex_iterator it(str.begin(), str.end(), cur_pattern);
  std::sregex_iterator end;
  return it != end;
}

TEST_F(TestAttLog, test_att_logd) {
  auto ret = std::system("./tiling_func_log_main 1024 2048 -1 > ./att_info.log");
  EXPECT_EQ(ret, 0);

  EXPECT_TRUE(CheckOutput("\\[DEBUG\\]\\[OpTest6\\]Start initializing the input."));
  EXPECT_TRUE(CheckOutput("\\[DEBUG\\]\\[OpTest6\\]The solver executed successfully."));
}

TEST_F(TestAttLog, test_att_logi) {
  auto ret = std::system("./tiling_func_log_main 1024 2048 -1 > ./att_info.log");
  EXPECT_EQ(ret, 0);

  EXPECT_TRUE(CheckOutput("\\[INFO\\]\\[OpTest6\\]The user didn't specify tiling_case_id, iterate all templates."));
}

TEST_F(TestAttLog, test_att_logw) {
  auto ret = std::system("./tiling_func_log_main 8192 2048 -1 > ./att_info.log");
  EXPECT_EQ(ret, 0);

  EXPECT_TRUE(CheckOutput("\\[WARNING\\]\\[OpTest6\\]The solver execution failed."));
}

TEST_F(TestAttLog, test_att_loge) {
  auto ret = std::system("./tiling_func_log_main 8192 2048 -1 > ./att_info.log");
  EXPECT_EQ(ret, 0);

  EXPECT_TRUE(CheckOutput("\\[ERROR\\]\\[OpTest6\\]Failed to execute tiling func."));
}
