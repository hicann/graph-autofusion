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
#include "base/base_types.h"
#include "base/model_info.h"
#include "generator/preprocess/args_manager.h"
#include "stub_model_info.h"
#include "tiling_code_generator.h"
#include "reuse_group_utils/reuse_group_utils.h"
#include "common/st_scenario_utils.h"
#include "test_common_utils.h"

using namespace att;
using namespace att::test;

class TestProfiler : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    unsetenv("ASCEND_SLOG_PRINT_TO_STDOUT");
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 1);
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
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
    EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, modelInfos), af::SUCCESS);
    generator.GenTilingCode(op_name, modelInfos, config);
    AddHeaderGuardToFile("autofuse_tiling_func_common.h", "__AUTOFUSE_TILING_FUNC_COMMON_H__");
    std::system(std::string("cp ").append(TILING_DATA_DIR).append("/tiling_func_attlog_main.cpp ./ -f").c_str());
    std::system(std::string("cp ").append(ST_DIR).append("/testcase/stub/op_log.h ./ -f").c_str());
    autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
    std::system(std::string("cp -r ").append(ST_DIR).append("/testcase/scenario/profiler/profiler.py ./ -f").c_str());
    std::system(
        "g++ -std=c++17 -DDEBUG tiling_func_attlog_main.cpp OpTest6_*tiling_func.cpp -I ./ -o tiling_func_log_main");
  }

  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
  }
  ModelInfo model_info_;
};

bool CheckResult(const std::string &pattern) {
  std::string str;
  std::string line;
  std::ifstream file("./result.txt");

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

TEST_F(TestProfiler, test_profiler_without_sched) {
  int32_t ret = 0;
  ret = std::system("./tiling_func_log_main 1024 2048 -1 > ./info.log");
  EXPECT_EQ(ret, 0);
  ret = std::system("python3 profiler.py --log_path=./info.log > result.txt");
  EXPECT_EQ(ret, 0);
}
