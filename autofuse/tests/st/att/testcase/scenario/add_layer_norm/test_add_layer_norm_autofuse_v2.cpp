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
#include "base/att_const_values.h"
#include "gen_model_info.h"
#include "ascir_ops.h"
#include "tiling_code_generator.h"
#include "api_tiling_gen/gen_api_tiling.h"
#include "autofuse_config/auto_fuse_config.h"
#include "gen_tiling_impl.h"
#include "graph_construct_utils.h"
#include "result_checker_utils.h"
#include "common/st_scenario_utils.h"
#include "test_common_utils.h"

using namespace ge::ascir_op;
namespace ascir {
constexpr int64_t ID_NONE = -1;
using namespace ge;
using HintGraph = AscGraph;
}  // namespace ascir

// Forward declarations of graph construction functions from test_add_layer_norm.cpp
void Add_Layer_Norm_Normal_BeforeAutofuse(ascir::HintGraph &graph, const std::string &ident = "");
void Add_Layer_Norm_Normal_BeforeAutofuseConstInput(ascir::HintGraph &graph);
void Add_Layer_Norm_Normal_AfterScheduler(ascir::HintGraph &graph, const std::string &ident = "");
void Add_Layer_Norm_Normal_AfterQueBufAlloc(ascir::HintGraph &graph);
void Add_Layer_Norm_Slice_BeforeAutofuse(ascir::HintGraph &graph);
void Add_Layer_Norm_Slice_AfterScheduler(ascir::HintGraph &graph);
void Add_Layer_Norm_Slice_AfterQueBufAlloc(ascir::HintGraph &graph);
void Add_Layer_Norm_Welford_BeforeAutofuse(ascir::HintGraph &graph);
void Add_Layer_Norm_Welford_AfterScheduler(ascir::HintGraph &graph);
void Add_Layer_Norm_Welford_AfterQueBufAlloc(ascir::HintGraph &graph);
void CombineTilings(const std::map<std::string, std::string> &tilings, std::string &result);

using namespace att;

namespace {
void SetStatsEnv() {
  setenv("OPEN_COMPILE_STATS", "open", 1);
  setenv("OPEN_TILINGFUNC_STATS", "open", 1);
}

ascir::AscGraph BuildNormalGraph(const std::string &name, uint32_t tiling_key, const std::string &ident = "") {
  ascir::AscGraph graph(name.c_str());
  graph.SetTilingKey(tiling_key);
  Add_Layer_Norm_Normal_BeforeAutofuse(graph, ident);
  Add_Layer_Norm_Normal_AfterScheduler(graph, ident);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph);
  return graph;
}

ascir::AscGraph BuildSliceGraph(const std::string &name, uint32_t tiling_key) {
  ascir::AscGraph graph(name.c_str());
  graph.SetTilingKey(tiling_key);
  Add_Layer_Norm_Slice_BeforeAutofuse(graph);
  Add_Layer_Norm_Slice_AfterScheduler(graph);
  Add_Layer_Norm_Slice_AfterQueBufAlloc(graph);
  return graph;
}

ascir::AscGraph BuildWelfordGraph(const std::string &name, uint32_t tiling_key) {
  ascir::AscGraph graph(name.c_str());
  graph.SetTilingKey(tiling_key);
  Add_Layer_Norm_Welford_BeforeAutofuse(graph);
  Add_Layer_Norm_Welford_AfterScheduler(graph);
  Add_Layer_Norm_Welford_AfterQueBufAlloc(graph);
  return graph;
}

void UpdateScheduleGroupStride(ascir::ScheduleGroup &schedule_group) {
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group.impl_graphs);
}

void WriteFile(const std::string &file_name, const std::string &content) {
  std::ofstream oss(file_name, std::ios::out);
  oss << content;
}

void WriteTilingFiles(const std::map<std::string, std::string> &tiling_funcs, const std::string &head_file,
                      const std::string &source_prefix, const std::string &head_include) {
  for (const auto &[key, value] : tiling_funcs) {
    if (key == "TilingHead") {
      WriteFile(head_file, head_include + value);
      continue;
    }
    WriteFile(source_prefix + key + "_3.cpp", value);
  }
}

void WriteCombinedTilingFile(const std::map<std::string, std::string> &tiling_funcs, const std::string &file_name,
                             const std::string &head_include, std::string *tiling_func = nullptr) {
  std::string combined_tiling_func;
  CombineTilings(tiling_funcs, combined_tiling_func);
  WriteFile(file_name, head_include + combined_tiling_func);
  if (tiling_func != nullptr) {
    *tiling_func = combined_tiling_func;
  }
}

void GenerateTilingDataFile(const std::string &op_name, const ascir::FusedScheduledResult &schedule_results,
                            const std::map<std::string, std::string> &options, const std::string &tiling_data_name,
                            const std::string &output_file) {
  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(schedule_results, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options.at(kTilingDataTypeName);
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), af::SUCCESS);
  WriteFile(output_file, tiling_res[tiling_data_name]);
}

void CopyStubArtifacts() {
  auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/stub/op_log.h ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);
}

void CopyBuildArtifacts(const std::string &main_file) {
  auto ret = std::system(std::string("cp ").append(main_file).append(" ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  CopyStubArtifacts();
}

void BuildAndRunBinary(const std::string &compile_cmd, const std::string &run_cmd) {
  auto ret = std::system(compile_cmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system(run_cmd.c_str());
  EXPECT_EQ(ret, 0);
}

bool IsFileContainsString(const std::string &filename, const std::string &searchString) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "无法打开文件: " << filename << std::endl;
    return false;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.find(searchString) != std::string::npos) {
      file.close();
      return true;
    }
  }
  file.close();
  return false;
}
}  // namespace

class TestGenAddLayerNormalModelInfoV2 : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
  }
  void TearDown() override {
    autofuse::test::CleanupTestArtifacts();
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  }
};

TEST_F(TestGenAddLayerNormalModelInfoV2, test_autofuse_v2_axes_reorder) {
  SetStatsEnv();
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduleGroup schedule_group3;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  ascir::FusedScheduledResult schedule_results;

  schedule_group1.impl_graphs.emplace_back(BuildNormalGraph("graph_normal", 1101u));
  schedule_group2.impl_graphs.emplace_back(BuildSliceGraph("graph_slice", 1111u));
  schedule_group3.impl_graphs.emplace_back(BuildWelfordGraph("graph_welford", 1151u));
  UpdateScheduleGroupStride(schedule_group1);
  UpdateScheduleGroupStride(schedule_group2);
  UpdateScheduleGroupStride(schedule_group3);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.schedule_groups.emplace_back(schedule_group3);
  schedule_results.node_idx_to_scheduled_results.emplace_back(
      std::vector<ascir::ScheduledResult>{schedule_result1, schedule_result2});

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "AxesReorder");

  auto res = GenTilingImplAutoFuseV3(op_name, schedule_results, options, tiling_funcs, true);
  WriteTilingFiles(tiling_funcs, "autofuse_tiling_func_common.h", "add_layer_norm_autofuse_tiling_func_",
                   "#include \"AddLayerNorm_tiling_data.h\"\n");
  EXPECT_EQ(res, true);

  GenerateTilingDataFile(op_name, schedule_results, options, "graph_normalTilingData", "AddLayerNorm_tiling_data.h");
  CopyBuildArtifacts(std::string(TILING_DATA_DIR).append("/tiling_func_main_add_layer_norm_sche.cpp"));
  BuildAndRunBinary(
      "g++ tiling_func_main_add_layer_norm_sche.cpp add_layer_norm_autofuse_tiling_func_*_3.cpp -o "
      "tiling_func_main_add_layer_norm_autofuse -I ./",
      "./tiling_func_main_add_layer_norm_autofuse");
}
std::string RemoveAutoFuseTilingHeadGuards(const std::string &input) {
  std::istringstream iss(input);
  std::ostringstream oss;
  std::string line;
  const std::string guard_token = "__AUTOFUSE_TILING_FUNC_COMMON_H__";

  while (std::getline(iss, line)) {
    // 如果当前行不包含 guard_token，则保留
    if (line.find(guard_token) == std::string::npos) {
      oss << line << "\n";
    }
  }

  return oss.str();
}

void CombineTilings(const std::map<std::string, std::string> &tilings, std::string &result) {
  const std::string tiling_head = "TilingHead";                       // TilingHead作为开头拼接其他文件
  const std::string tiling_data = "TilingData";                       // 要排除的 TilingData 子串
  result += RemoveAutoFuseTilingHeadGuards(tilings.at(tiling_head));  // 删除头文件的宏保护，cpp文件不需要
  const std::string include_str = "#include \"autofuse_tiling_func_common.h\"";

  // 遍历所有非 TilingHead 和 TilingData 的条目，去掉第一行后拼接
  for (const auto &[key, value] : tilings) {
    if (key == tiling_head || key.find(tiling_data) != std::string::npos) {
      continue;
    }

    // 查找并跳过第一行头文件行
    size_t include_pos = value.find(include_str);
    if (include_pos != std::string::npos) {
      // 找到 include 行，跳过它，并去掉后面的换行符
      size_t content_start = include_pos + include_str.length();
      while (content_start < value.size() && (value[content_start] == '\n' || value[content_start] == '\r')) {
        content_start++;
      }
      result += value.substr(content_start);
    } else {
      // 如果没有 include 行，直接拼接整个内容
      result += value;
    }

    if (!result.empty() && result.back() != '\n') {
      result += '\n';
    }
  }
}

const std::string kGroupParallelTilingMain = R"(
 #include <iostream>
 #include "AddLayerNorm_tiling_data.h"
 using namespace optiling;

 void PrintResult(graph_normalTilingData& tilingData) {
   std::cout << "====================================================" << std::endl;
   auto tiling_key = tilingData.get_graph0_tiling_key();
   std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
   MY_ASSERT_EQ(tiling_key, 1);
   std::cout << "====================================================" << std::endl;
 }

 int main() {
   graph_normalTilingData tilingData;
   tilingData.set_block_dim(64);
   tilingData.set_ub_size(245760);
   auto &schedule0_g0_tiling_data = tilingData.graph0_result0_g0_tiling_data;
   auto &schedule0_g1_tiling_data = tilingData.graph0_result0_g1_tiling_data;
   auto &schedule1_g0_tiling_data = tilingData.graph0_result1_g0_tiling_data;
   auto &schedule1_g1_tiling_data = tilingData.graph0_result1_g1_tiling_data;
   schedule0_g0_tiling_data.set_A(1536);
   schedule0_g0_tiling_data.set_R(128);
   schedule0_g0_tiling_data.set_BL(8);
   schedule0_g1_tiling_data.set_A(1536);
   schedule0_g1_tiling_data.set_R(128);
   schedule0_g1_tiling_data.set_BL(8);
   schedule1_g0_tiling_data.set_A(1536);
   schedule1_g0_tiling_data.set_R(128);
   schedule1_g0_tiling_data.set_BL(8);
   schedule1_g1_tiling_data.set_A(1536);
   schedule1_g1_tiling_data.set_R(128);
   if (GetTiling(tilingData)) {
     PrintResult(tilingData);
   } else {
     std::cout << "addlayernorm tiling func execute failed." << std::endl;
     return -1;
   }
   return 0;
 }
)";

// 测试autofuse v2 轴重排开启group parallel
// 测试场景：
// 1. 轴重排开启group parallel，预期tiling key为1
// 2. 轴重排关闭group parallel，预期tiling key为0
// 预期：
// 1. 轴重排开启group parallel，预期tiling key为1
TEST_F(TestGenAddLayerNormalModelInfoV2, test_autofuse_v2_axes_reorder_enable_group_parallel) {
  SetStatsEnv();
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduleGroup schedule_group3;
  ascir::ScheduleGroup schedule_group4;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  ascir::FusedScheduledResult schedule_results;

  schedule_group1.impl_graphs.emplace_back(BuildNormalGraph("graph_normal", 1101u));
  schedule_group2.impl_graphs.emplace_back(BuildNormalGraph("graph_normal1", 1111u));
  schedule_group3.impl_graphs.emplace_back(BuildNormalGraph("graph_normal2", 1151u));
  schedule_group4.impl_graphs.emplace_back(BuildSliceGraph("graph_slice", 1155u));
  UpdateScheduleGroupStride(schedule_group1);
  UpdateScheduleGroupStride(schedule_group2);
  UpdateScheduleGroupStride(schedule_group3);
  UpdateScheduleGroupStride(schedule_group4);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.schedule_groups.emplace_back(schedule_group3);
  schedule_result2.schedule_groups.emplace_back(schedule_group4);
  schedule_result2.enable_group_parallel = true;
  schedule_results.node_idx_to_scheduled_results.emplace_back(
      std::vector<ascir::ScheduledResult>{schedule_result1, schedule_result2});

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  options.emplace(kGenConfigType, "AxesReorder");

  auto res = GenTilingImplAutoFuseV3("AddLayerNorm", schedule_results, options, tiling_funcs, true);
  std::string tiling_func;
  WriteCombinedTilingFile(tiling_funcs, "add_layer_norm_autofuse_*_tiling_func.cpp",
                          "#include \"AddLayerNorm_tiling_data.h\"\n", &tiling_func);
  EXPECT_EQ(res, true);
  EXPECT_TRUE(tiling_func.find("  ArrangeBlockOffsetsAscGraph0Result0(") == std::string::npos);
  EXPECT_TRUE(tiling_func.find("  ArrangeBlockOffsetsAscGraph0Result1(") != std::string::npos);

  GenerateTilingDataFile("AddLayerNorm", schedule_results, options, "graph_normalTilingData",
                         "AddLayerNorm_tiling_data.h");
  WriteFile("tiling_func_main_add_layer_norm_sche.cpp",
            ResultCheckerUtils::DefineCheckerFunction() + kGroupParallelTilingMain);
  CopyStubArtifacts();
  BuildAndRunBinary(
      "g++ tiling_func_main_add_layer_norm_sche.cpp add_layer_norm_autofuse_*_tiling_func.cpp -o "
      "tiling_func_main_add_layer_norm_autofuse -I ./",
      "./tiling_func_main_add_layer_norm_autofuse");
}

TEST_F(TestGenAddLayerNormalModelInfoV2, test_autofuse_v2_axes_reorder_uniq_group) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduledResult schedule_result1;
  ascir::FusedScheduledResult scheduled_results;
  schedule_group1.impl_graphs.emplace_back(BuildNormalGraph("graph_normal", 1101u));
  schedule_group1.impl_graphs.emplace_back(BuildSliceGraph("graph_slice", 1111u));
  schedule_group1.impl_graphs.emplace_back(BuildWelfordGraph("graph_welford", 1151u));
  UpdateScheduleGroupStride(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  scheduled_results.node_idx_to_scheduled_results.emplace_back(std::vector<ascir::ScheduledResult>{schedule_result1});

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "AxesReorder");
  auto res = GenTilingImplAutoFuseV3(op_name, scheduled_results, options, tiling_funcs, true);
  WriteCombinedTilingFile(tiling_funcs, "add_layer_norm_autofuse_*_tiling_func.cpp",
                          "#include \"AddLayerNorm_tiling_data.h\"\n");
  EXPECT_EQ(res, true);
  GenerateTilingDataFile(op_name, scheduled_results, options, "graph_normalTilingData", "AddLayerNorm_tiling_data.h");
  CopyBuildArtifacts(std::string(TILING_DATA_DIR).append("/tiling_func_main_add_layer_norm.cpp"));
  BuildAndRunBinary(
      "g++ tiling_func_main_add_layer_norm.cpp add_layer_norm_autofuse_*_tiling_func.cpp -o "
      "tiling_func_main_add_layer_norm_autofuse -I ./",
      "./tiling_func_main_add_layer_norm_autofuse > ./info.log");
}

TEST_F(TestGenAddLayerNormalModelInfoV2, test_autofuse_v2_axes_reorder_reuse_solver) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  schedule_group1.impl_graphs.emplace_back(BuildNormalGraph("graph_normal", 1101u));
  schedule_group2.impl_graphs.emplace_back(BuildNormalGraph("graph_slice", 1111u, "_slice"));
  UpdateScheduleGroupStride(schedule_group1);
  UpdateScheduleGroupStride(schedule_group2);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func = "int32_t CalcScore(graph_normalTilingData &tiling_data) { return 1;}";
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func = "int32_t CalcScore(graph_normalTilingData &tiling_data) { return 2;}";
  ascir::FusedScheduledResult scheduled_results;
  scheduled_results.node_idx_to_scheduled_results.emplace_back(
      std::vector<ascir::ScheduledResult>{schedule_result1, schedule_result2});

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "AxesReorder");
  options.emplace("enable_score_func", "1");
  auto res = GenTilingImplAutoFuseV3(op_name, scheduled_results, options, tiling_funcs, true);
  WriteCombinedTilingFile(tiling_funcs, "add_layer_norm_autofuse_*_tiling_func.cpp",
                          "#include \"AddLayerNorm_tiling_data.h\"\n");
  EXPECT_EQ(res, true);
  GenerateTilingDataFile(op_name, scheduled_results, options, "graph_normalTilingData", "AddLayerNorm_tiling_data.h");
  CopyBuildArtifacts(std::string(TILING_DATA_DIR).append("/tiling_func_main_add_layer_norm_reuse_solver.cpp"));
  BuildAndRunBinary(
      "g++ -g -O0 tiling_func_main_add_layer_norm_reuse_solver.cpp add_layer_norm_autofuse_*_tiling_func.cpp -o "
      "tiling_func_main_add_layer_norm_autofuse -I ./",
      "./tiling_func_main_add_layer_norm_autofuse > ./info.log");
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(TestGenAddLayerNormalModelInfoV2, test_autofuse_v2_axes_reorder_with_score_funcs) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduleGroup schedule_group3;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  schedule_group1.impl_graphs.emplace_back(BuildNormalGraph("graph_normal", 1101u));
  schedule_group2.impl_graphs.emplace_back(BuildSliceGraph("graph_slice", 1111u));
  schedule_group3.impl_graphs.emplace_back(BuildWelfordGraph("graph_welford", 1151u));
  UpdateScheduleGroupStride(schedule_group1);
  UpdateScheduleGroupStride(schedule_group2);
  UpdateScheduleGroupStride(schedule_group3);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group2);
  schedule_result1.score_func = "int32_t CalcScore(graph_normalTilingData &tiling_data) { return 1;}";
  schedule_result2.schedule_groups.emplace_back(schedule_group3);
  schedule_result2.score_func = "int32_t CalcScore(graph_normalTilingData &tiling_data) { return 2;}";
  ascir::FusedScheduledResult scheduled_results;
  scheduled_results.node_idx_to_scheduled_results.emplace_back(
      std::vector<ascir::ScheduledResult>{schedule_result1, schedule_result2});

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "AxesReorder");
  auto res = GenTilingImplAutoFuseV3(op_name, scheduled_results, options, tiling_funcs, true);
  std::string tiling_func;
  WriteCombinedTilingFile(tiling_funcs, "add_layer_norm_autofuse_*_tiling_func.cpp",
                          "#include \"AddLayerNorm_tiling_data.h\"\n", &tiling_func);
  EXPECT_NE(tiling_func.find("ScheduleResult0::CalcScore"), std::string::npos);
  EXPECT_EQ(res, true);
  GenerateTilingDataFile(op_name, scheduled_results, options, "graph_normalTilingData", "AddLayerNorm_tiling_data.h");
  CopyBuildArtifacts(std::string(TILING_DATA_DIR).append("/tiling_func_main_add_layer_norm_sche.cpp"));
  BuildAndRunBinary(
      "g++ tiling_func_main_add_layer_norm_sche.cpp add_layer_norm_autofuse_*_tiling_func.cpp -o "
      "tiling_func_main_add_layer_norm_autofuse -I ./",
      "./tiling_func_main_add_layer_norm_autofuse > ./info.log");
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(TestGenAddLayerNormalModelInfoV2, test_autofuse_v2_axes_reorder_with_score_funcs_one_group) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group3;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  schedule_group1.impl_graphs.emplace_back(BuildNormalGraph("graph_normal", 1101u));
  schedule_group3.impl_graphs.emplace_back(BuildWelfordGraph("graph_welford", 1151u));
  UpdateScheduleGroupStride(schedule_group1);
  UpdateScheduleGroupStride(schedule_group3);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func = "int32_t CalcScore(graph_normalTilingData &tiling_data) { return 1;}";
  schedule_result2.schedule_groups.emplace_back(schedule_group3);
  schedule_result2.score_func = "int32_t CalcScore(graph_normalTilingData &tiling_data) { return 2;}";
  ascir::FusedScheduledResult scheduled_results;
  scheduled_results.node_idx_to_scheduled_results.emplace_back(
      std::vector<ascir::ScheduledResult>{schedule_result1, schedule_result2});

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "AxesReorder");
  options.emplace("enable_score_func", "1");
  auto res = GenTilingImplAutoFuseV3(op_name, scheduled_results, options, tiling_funcs, true);
  WriteCombinedTilingFile(tiling_funcs, "add_layer_norm_autofuse_*_tiling_func.cpp",
                          "#include \"AddLayerNorm_tiling_data.h\"\n");
  EXPECT_EQ(res, true);
  GenerateTilingDataFile(op_name, scheduled_results, options, "graph_normalTilingData", "AddLayerNorm_tiling_data.h");
  CopyBuildArtifacts(std::string(TILING_DATA_DIR).append("/tiling_func_main_add_layer_norm_sche.cpp"));
  auto ret = std::system("sed -i '/schedule_result0_g1_tiling_data/d' ./tiling_func_main_add_layer_norm_sche.cpp");
  EXPECT_EQ(ret, 0);
  ret = std::system("sed -i '/schedule0_g1_tiling_data/d' ./tiling_func_main_add_layer_norm_sche.cpp");
  EXPECT_EQ(ret, 0);
  BuildAndRunBinary(
      "g++ tiling_func_main_add_layer_norm_sche.cpp add_layer_norm_autofuse_*_tiling_func.cpp -o "
      "tiling_func_main_add_layer_norm_autofuse -I ./",
      "./tiling_func_main_add_layer_norm_autofuse > ./info.log");
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(TestGenAddLayerNormalModelInfoV2, test_autofuse_v2_high_perf_choose_first_according_to_perf) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduleGroup schedule_group3;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  schedule_group1.impl_graphs.emplace_back(BuildNormalGraph("graph_normal", 1101u));
  schedule_group2.impl_graphs.emplace_back(BuildSliceGraph("graph_slice", 1111u));
  schedule_group3.impl_graphs.emplace_back(BuildWelfordGraph("graph_welford", 1151u));
  UpdateScheduleGroupStride(schedule_group1);
  UpdateScheduleGroupStride(schedule_group2);
  UpdateScheduleGroupStride(schedule_group3);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.schedule_groups.emplace_back(schedule_group3);
  ascir::FusedScheduledResult scheduled_results;
  scheduled_results.node_idx_to_scheduled_results.emplace_back(
      std::vector<ascir::ScheduledResult>{schedule_result1, schedule_result2});

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "HighPerf");
  auto res = GenTilingImplAutoFuseV3(op_name, scheduled_results, options, tiling_funcs, true);
  WriteCombinedTilingFile(tiling_funcs, "add_layer_norm_autofuse_*_tiling_func.cpp",
                          "#include \"AddLayerNorm_tiling_data.h\"\n");
  EXPECT_EQ(res, true);
  GenerateTilingDataFile(op_name, scheduled_results, options, "graph_normalTilingData", "AddLayerNorm_tiling_data.h");
  CopyBuildArtifacts(std::string(TILING_DATA_DIR).append("/tiling_func_main_add_layer_norm_sche.cpp"));
  BuildAndRunBinary(
      "g++ tiling_func_main_add_layer_norm_sche.cpp add_layer_norm_autofuse_*_tiling_func.cpp -o "
      "tiling_func_main_add_layer_norm_autofuse -I ./",
      "./tiling_func_main_add_layer_norm_autofuse > ./info.log");
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(TestGenAddLayerNormalModelInfoV2, test_autofuse_v2_high_perf_choose_second_according_to_perf) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduleGroup schedule_group3;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  schedule_group1.impl_graphs.emplace_back(BuildNormalGraph("graph_normal", 1101u));
  schedule_group2.impl_graphs.emplace_back(BuildSliceGraph("graph_slice", 1111u));
  schedule_group3.impl_graphs.emplace_back(BuildWelfordGraph("graph_welford", 1151u));
  UpdateScheduleGroupStride(schedule_group1);
  UpdateScheduleGroupStride(schedule_group2);
  UpdateScheduleGroupStride(schedule_group3);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group3);
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  ascir::FusedScheduledResult scheduled_results;
  scheduled_results.node_idx_to_scheduled_results.emplace_back(
      std::vector<ascir::ScheduledResult>{schedule_result1, schedule_result2});

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "HighPerf");
  auto res = GenTilingImplAutoFuseV3(op_name, scheduled_results, options, tiling_funcs, true);
  WriteCombinedTilingFile(tiling_funcs, "add_layer_norm_autofuse_*_tiling_func.cpp",
                          "#include \"AddLayerNorm_tiling_data.h\"\n");
  EXPECT_EQ(res, true);
  GenerateTilingDataFile(op_name, scheduled_results, options, "graph_normalTilingData", "AddLayerNorm_tiling_data.h");
  CopyBuildArtifacts(std::string(TILING_DATA_DIR).append("/tiling_func_main_add_layer_norm_sche.cpp"));
  BuildAndRunBinary(
      "g++ tiling_func_main_add_layer_norm_sche.cpp add_layer_norm_autofuse_*_tiling_func.cpp -o "
      "tiling_func_main_add_layer_norm_autofuse -I ./",
      "./tiling_func_main_add_layer_norm_autofuse > ./info.log");
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(TestGenAddLayerNormalModelInfoV2, test_autofuse_v2_set_log_debug) {
  setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduleGroup schedule_group3;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  schedule_group1.impl_graphs.emplace_back(BuildNormalGraph("graph_normal", 1101u));
  schedule_group2.impl_graphs.emplace_back(BuildSliceGraph("graph_slice", 1111u));
  schedule_group3.impl_graphs.emplace_back(BuildWelfordGraph("graph_welford", 1151u));
  UpdateScheduleGroupStride(schedule_group1);
  UpdateScheduleGroupStride(schedule_group2);
  UpdateScheduleGroupStride(schedule_group3);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.schedule_groups.emplace_back(schedule_group3);
  ascir::FusedScheduledResult scheduled_results;
  scheduled_results.node_idx_to_scheduled_results.emplace_back(
      std::vector<ascir::ScheduledResult>{schedule_result1, schedule_result2});
  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "HighPerf");
  EXPECT_TRUE(GenTilingImplAutoFuseV3(op_name, scheduled_results, options, tiling_funcs, true));
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_NE(tiling_func.find("OP_LOGD(OP_NAME"), std::string::npos);
  unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
}

TEST_F(TestGenAddLayerNormalModelInfoV2, test_autofuse_v2_axes_reorder_by_env) {
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_att_algorithm=AxesReorder", 1);
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduleGroup schedule_group3;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  schedule_group1.impl_graphs.emplace_back(BuildNormalGraph("graph_normal", 1101u));
  schedule_group2.impl_graphs.emplace_back(BuildSliceGraph("graph_slice", 1111u));
  schedule_group3.impl_graphs.emplace_back(BuildWelfordGraph("graph_welford", 1151u));
  UpdateScheduleGroupStride(schedule_group1);
  UpdateScheduleGroupStride(schedule_group2);
  UpdateScheduleGroupStride(schedule_group3);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.schedule_groups.emplace_back(schedule_group3);
  ascir::FusedScheduledResult scheduled_results;
  scheduled_results.node_idx_to_scheduled_results.emplace_back(
      std::vector<ascir::ScheduledResult>{schedule_result1, schedule_result2});
  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "AddLayerNorm";
  auto res = GenTilingImplAutoFuseV3(op_name, scheduled_results, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_NE(tiling_func.find("axes reorder solver"), std::string::npos);
  WriteCombinedTilingFile(tiling_funcs, "add_layer_norm_autofuse_*_tiling_func.cpp",
                          "#include \"AddLayerNorm_tiling_data.h\"\n");
  EXPECT_EQ(res, true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(TestGenAddLayerNormalModelInfoV2, case_axes_reorder_by_env) {
  setenv("AUTOFUSE_DFX_FLAGS",
         "--att_enable_multicore_ub_tradeoff=true;--autofuse_att_algorithm=AxesReorder;--att_enable_small_shape_"
         "strategy=true;--att_accuracy_level=1",
         1);
  std::vector<ascir::AscGraph> graphs;

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Add_Layer_Norm_Normal_BeforeAutofuseConstInput(graph_normal);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  graphs.emplace_back(graph_normal);

  graphs.emplace_back(BuildSliceGraph("graph_slice", 1111u));
  graphs.emplace_back(BuildWelfordGraph("graph_welford", 1151u));
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["gen_extra_info"] = "1";
  options["solver_type"] = "AxesReorder";
  EXPECT_EQ(GenTilingImpl("AddLayerNorm", graphs, options), true);

  CopyBuildArtifacts(std::string(TILING_DATA_DIR).append("/tiling_func_main_add_layer_norm.cpp"));
  BuildAndRunBinary(
      "g++ tiling_func_main_add_layer_norm.cpp AddLayerNorm_*_tiling_func.cpp -o tiling_func_main_add_layer_norm "
      "-I ./ -DSTUB_LOG",
      "./tiling_func_main_add_layer_norm");
  unsetenv("AUTOFUSE_DFX_FLAGS");
  att::AutoFuseConfig::MutableAttStrategyConfig().enable_multicore_ub_tradeoff = false;
  att::AutoFuseConfig::MutableAttStrategyConfig().enable_small_shape_strategy = false;
  att::AutoFuseConfig::MutableAttStrategyConfig().solution_accuracy_level = 0;
}
