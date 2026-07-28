/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#define private public
#define protected public
#include "tiling_code_generator.h"
#include "high_perf_tiling_code_gen_impl.h"
#include "tiling_code_gen_impl.h"
#undef private
#undef protected
#include "generator_utils/tilingdata_gen_utils.h"

#include <symengine/symengine_rcp.h>
#include <symengine/basic.h>
#include <symengine/symbol.h>
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/integer.h>
#include "stub_solver_model_info.h"
#include "reuse_group_utils/reuse_group_utils.h"
#include "tiling_data_gen/tiling_data_generator.h"
#include "base/base_types.h"
#include "util/base_types_printer.h"

const std::string op_name = "OpTest";

namespace att {
namespace {
size_t CountSubstr(const std::string &text, const std::string &pattern) {
  size_t count = 0U;
  size_t pos = text.find(pattern);
  while (pos != std::string::npos) {
    ++count;
    pos = text.find(pattern, pos + pattern.size());
  }
  return count;
}
}  // namespace

class MockHighPerfTilingCodeGenImpl : public HighPerfTilingCodeGenImpl {
 public:
  MockHighPerfTilingCodeGenImpl(const std::string &mock_op_name, const TilingCodeGenConfig &config,
                                const TilingModelInfo &model_infos, const ScoreFuncs &score_funcs,
                                const bool is_uniq_group)
      : HighPerfTilingCodeGenImpl(mock_op_name, config, model_infos, score_funcs, is_uniq_group) {}
};

class MockTilingCodeGenerator : public TilingCodeGenerator {
 protected:
  TilingCodeGenImplPtr CreateTilingCodeGenImpl(const std::string &mock_op_name, const TilingCodeGenConfig &config,
                                               const TilingModelInfo &model_infos, const ScoreFuncs &score_funcs,
                                               const bool is_uniq_group) override {
    std::shared_ptr<MockHighPerfTilingCodeGenImpl> impl =
        std::make_shared<MockHighPerfTilingCodeGenImpl>(mock_op_name, config, model_infos, score_funcs, is_uniq_group);
    return impl;
  }
};

class GeneratorUT : public testing::Test {};

TEST(GeneratorUT, Normal) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo();
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = true;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), af::SUCCESS);
}

TEST(GeneratorUT, NormalStaticUint32Shape) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo(1, af::ExprType::kExprConstantInteger);
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = false;
  config.gen_tiling_data = false;
  TilingCodeGenerator generator;
  std::map<size_t, std::map<size_t, std::vector<ModelInfo>>> model_infos_new;
  model_infos_new[0][0] = model_infos;
  std::map<std::string, std::string> tiling_res;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), af::SUCCESS);
  ASSERT_EQ(tiling_res.size(), 4);
}

TEST(GeneratorUT, NormalStaticRationShape) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo(1, af::ExprType::kExprConstantRation);
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = false;
  config.gen_tiling_data = false;
  TilingCodeGenerator generator;
  std::map<size_t, std::map<size_t, std::vector<ModelInfo>>> model_infos_new;
  model_infos_new[0][0] = model_infos;
  std::map<std::string, std::string> tiling_res;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), af::SUCCESS);
  ASSERT_EQ(tiling_res.size(), 4);
}

TEST(GeneratorUT, GenTilingSolverSuccess) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo();
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  MockTilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), af::SUCCESS);
}

TEST(GeneratorUT, InvalidConfig) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo;
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::MAX;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  EXPECT_NE(generator.GenTilingCode(op_name, model_infos, config), af::SUCCESS);
}

TEST(GeneratorUT, TestSymengine) {
  using namespace SymEngine;
  using SymEngine::Basic;
  using SymEngine::make_rcp;
  using SymEngine::RCP;
  using SymEngine::Symbol;
  const RCP<const Basic> x = make_rcp<SymEngine::Symbol>("x");
  EXPECT_EQ(x->__str__(), "x");
}

TEST(GeneratorUT, TestSymengine2) {
  using namespace SymEngine;
  RCP<const Basic> x1 = symbol("x1");
  RCP<const Basic> x2 = symbol("x2");
  RCP<const Basic> int1 = integer(1);
  RCP<const Basic> int2 = integer(2);
  RCP<const Basic> y = mul(x2, add(x1, int1));
  RCP<const Basic> z = mul(add(int1, x1), x2);
  EXPECT_EQ(x1->__str__(), "x1");
  EXPECT_EQ(x2->__str__(), "x2");
  EXPECT_EQ(y->__str__(), "x2*(1 + x1)");
  EXPECT_EQ(z->__str__(), "x2*(1 + x1)");
  EXPECT_EQ(is_a<Symbol>(*x1), true);
  EXPECT_EQ(is_a<Symbol>(*x2), true);
  EXPECT_EQ(is_a<Symbol>(*y), false);
  EXPECT_EQ(is_a<Symbol>(*z), false);
  EXPECT_EQ(is_a<Integer>(*int1), true);
  RCP<const Basic> multi_add = add(add(int1, x1), int2);
  EXPECT_EQ(multi_add->__str__(), "3 + x1");
  RCP<const Basic> m = add(mul(add(int1, x1), x2), int2);
  EXPECT_EQ(m->get_args()[0]->__str__(), "2");
  EXPECT_EQ(m->get_args()[1]->__str__(), "x2*(1 + x1)");
}

TEST(GeneratorUT, AddElementInTilingData) {
  ge::CodePrinter dumper;
  TilingDataGenUtils::AddStructElementDefinition(dumper, "TCubeTiling", "mm_tiling");
  EXPECT_TRUE(dumper.GetOutputStr().find("TCubeTiling, mm_tiling") != std::string::npos);
}

TEST(GeneratorUT, TestSchedGroup) {
  ModelInfo modelInfo = CreateModelInfo();
  FusedParsedScheduleResult fused_schedule_result;
  auto &all_model_infos = fused_schedule_result[0];
  std::map<size_t, std::vector<ModelInfo>> model_infos1;

  model_infos1[0] = {modelInfo, modelInfo};
  model_infos1[0][0].schedule_group_ident.impl_graph_id = 0;
  model_infos1[0][0].schedule_group_ident.group_id = 0;
  model_infos1[0][0].tiling_case_id = 0;
  model_infos1[0][1].schedule_group_ident.impl_graph_id = 0;
  model_infos1[0][1].schedule_group_ident.group_id = 0;
  model_infos1[0][1].tiling_case_id = 1;

  model_infos1[1] = {modelInfo};
  model_infos1[1][0].schedule_group_ident.impl_graph_id = 0;
  model_infos1[1][0].schedule_group_ident.group_id = 1;
  model_infos1[1][0].tiling_case_id = 2;
  for (auto &model_info : model_infos1) {
    EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_info.second), af::SUCCESS);
  }
  all_model_infos[0].groups_tiling_model_info = model_infos1;
  all_model_infos[0].impl_graph_id = 0;
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = true;
  config.gen_extra_infos = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  EXPECT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);
}

TEST(GeneratorUT, TestSchedGroupEnableGroupParallel) {
  ModelInfo modelInfo = CreateModelInfo();
  FusedParsedScheduleResult fused_schedule_result;
  auto &all_model_infos = fused_schedule_result[0];
  std::map<size_t, std::vector<ModelInfo>> model_infos1;

  model_infos1[0] = {modelInfo, modelInfo};
  model_infos1[0][0].schedule_group_ident.impl_graph_id = 0;
  model_infos1[0][0].schedule_group_ident.group_id = 0;
  model_infos1[0][0].tiling_case_id = 0;
  model_infos1[0][0].enable_group_parallel = true;
  model_infos1[0][1].schedule_group_ident.impl_graph_id = 0;
  model_infos1[0][1].schedule_group_ident.group_id = 0;
  model_infos1[0][1].tiling_case_id = 1;
  model_infos1[0][1].enable_group_parallel = true;

  model_infos1[1] = {modelInfo};
  model_infos1[1][0].schedule_group_ident.impl_graph_id = 0;
  model_infos1[1][0].schedule_group_ident.group_id = 1;
  model_infos1[1][0].tiling_case_id = 2;
  model_infos1[1][0].enable_group_parallel = true;
  for (auto &model_info : model_infos1) {
    EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_info.second), af::SUCCESS);
  }
  all_model_infos[0].groups_tiling_model_info = model_infos1;
  all_model_infos[0].impl_graph_id = 0;
  all_model_infos[0].enable_group_parallel = true;

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = true;
  config.gen_extra_infos = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  EXPECT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);
  bool flag_arrange = false;
  bool flag_parallel = false;
  for (const auto &[key, value] : tiling_res) {
    if (value.find("  ArrangeBlockOffsetsAscGraph0Result0(") != std::string::npos) {
      flag_arrange = true;
    }
    if (value.find("UpdateCurPerfAndBlockByGroup(") != std::string::npos) {
      flag_parallel = true;
    }
  }
  EXPECT_EQ(flag_arrange && flag_parallel, true);
}

TEST(GeneratorUT, CreateAxesReorderTilingCodeGenImplSuccess) {
  TilingModelInfo model_infos;
  model_infos.emplace_back(CreateModelInfo());
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::AXES_REORDER;
  config.gen_extra_infos = true;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), af::SUCCESS);
}

TEST(GeneratorUT, TilingCodeGenImplConstruct) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;
  MockHighPerfTilingCodeGenImpl impl("test", config, tiling_model_info, score_funcs, true);
  EXPECT_EQ(config.force_template_op_name, "test");
  impl.GenGetAllSchedulesResults({});
  EXPECT_EQ(impl.tiling_func_.GetOutputStr().empty(), true);
}

TEST(GeneratorUT, GenVariableAnnotationShowsReduceBreakdownAndContribSemantics) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  ModelInfo model_info = CreateModelInfo();
  Expr contrib_var = CreateExpr("Min_AIV_VEC_core_contrib");
  TernaryOp contrib_op(CreateExpr("reduce_total_perf") * CreateExpr("core_exe_time"));
  contrib_op.SetVariable(contrib_var);
  contrib_op.SetDescription("AIV_VEC core contribution = node API perf * core exe time");
  model_info.ternary_op_map[contrib_var] = contrib_op;
  model_info.perf_breakdowns = {
      {"Min Reduce API",
       {{"reduce_body_perf", CreateExpr("reduce_body_perf"), "Reduce API body perf", 0U},
        {"reduce_total_perf", CreateExpr("reduce_total_perf"), "Reduce API total perf = body + merge", 0U}}}};
  tiling_model_info.push_back(model_info);
  MockHighPerfTilingCodeGenImpl impl("test", config, tiling_model_info, score_funcs, true);
  ArgsManager args_manager(model_info);
  ASSERT_TRUE(args_manager.Process(false));
  EXPECT_EQ(impl.GenVariableAnnotation(args_manager), af::SUCCESS);

  const std::string tiling_func_output = impl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("Reduce perf breakdown used for tiling case 0"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("reduce_total_perf: Reduce API total perf = body + merge"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("AIV_VEC core contribution = node API perf * core exe time"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("core_exe_time * reduce_total_perf"), std::string::npos);
}

TEST(GeneratorUT, TilingCodeGenImplPGO) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;
  ModelInfo info;
  tiling_model_info.push_back(info);
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);

  genImpl.config_.enable_autofuse_pgo = true;
  EXPECT_EQ(genImpl.GenTilingImplPublicFunc(), af::SUCCESS);

  std::string expectCode = R"rawliteral(  bool GetTiling(TilingData &tiling_data) {
    OP_LOGD(OP_NAME, "Execute DoTiling.");
    if (!DoTiling(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to do tiling.");
      return false;
    }
    if (is_empty_tensor_) {
      OP_LOGW(OP_NAME, "Empty tensor, skip DoApiTiling and GeneralTiling.");
      return true;
    }
    DoApiTiling(tiling_data);
    GeneralTiling(tiling_data);
    TilingSummary(tiling_data);
    return true;
  }
  virtual double GetPerf(TilingData &tiling_data) { (void)tiling_data; return 0.0; }
  virtual const char* GetScheduleName() { return ""; }
  virtual void TilingSummary(TilingData &tiling_data) = 0;
  virtual bool ExecutePGOSolver(TilingData &tiling_data, std::vector<AutofuseTilingDataPerf>& tiling_data_list, AutofuseTilingData* autofuse_tiling_data, void* stream, std::unordered_map<int64_t, uint64_t> &workspace_map, std::vector<uint32_t*> block_dim_vec={}, const SearchConfig *search_cfg=nullptr) {
    (void)tiling_data; (void)tiling_data_list; (void)autofuse_tiling_data; (void)stream; (void)workspace_map; (void)block_dim_vec; (void)search_cfg;
    return false;
  }
  virtual int32_t CalcScore(const TilingData &tiling_data) { (void)tiling_data; return 0;}
  virtual void GetTilingData(TilingDataCopy &from_tiling, TilingData &to_tiling) { (void)from_tiling; (void)to_tiling; }
  virtual void SetTilingData(TilingData &from_tiling, TilingDataCopy &to_tiling) { (void)from_tiling; (void)to_tiling; }
  virtual void SetWorkspaceSize(TilingData &tiling_data, std::unordered_map<int64_t, uint64_t> &workspace_map) { (void)tiling_data; (void)workspace_map; }
)rawliteral";
  EXPECT_EQ(genImpl.tiling_func_.output_.str(), expectCode);
}

TEST(GeneratorUT, GenTilingHeadPGO) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  EnableGroupParallels enable_group_parallels;
  std::map<std::string, std::string> tiling_res;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;
  ModelInfo info;
  ReuseScheduleGroup reuse_schedule_group;
  info.reuse_schedule_group = std::make_shared<ReuseScheduleGroup>();
  tiling_model_info.push_back(info);
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);

  genImpl.config_.enable_autofuse_pgo = true;
  genImpl.GenTilingHead(tiling_res, enable_group_parallels);
  std::string expectCode = R"rawliteral(#include "autofuse_tiling_func_common.h"
namespace optiling {

} // namespace optiling
)rawliteral";
  EXPECT_EQ(genImpl.tiling_func_.output_.str(), expectCode);
}

TEST(GeneratorUT, GenScheduleGroupTilingTailPGOSuccess) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  EnableGroupParallels enable_group_parallels;
  std::map<std::string, std::string> tiling_res;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;

  ModelInfo info;
  ReuseScheduleGroup reuse_schedule_group;
  info.reuse_schedule_group = std::make_shared<ReuseScheduleGroup>();
  tiling_model_info.push_back(info);
  enable_group_parallels[0][0] = true;

  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);
  genImpl.config_.enable_autofuse_pgo = true;
  genImpl.config_.gen_tiling_data = false;
  genImpl.enable_group_parallels_ = enable_group_parallels;
  EXPECT_EQ(genImpl.GenScheduleGroupTilingTail(), af::SUCCESS);

  EXPECT_EQ(genImpl.tiling_func_.GetOutputStr().empty(), false);
}

TEST(GeneratorUT, GenTilingPGOSuccess) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  EnableGroupParallels enable_group_parallels;
  std::map<std::string, std::string> tiling_res;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;

  ModelInfo info;
  ReuseScheduleGroup reuse_schedule_group;
  info.reuse_schedule_group = std::make_shared<ReuseScheduleGroup>();
  tiling_model_info.push_back(info);
  enable_group_parallels[0][0] = true;

  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);
  genImpl.config_.enable_autofuse_pgo = true;
  genImpl.config_.gen_tiling_data = false;
  EXPECT_EQ(genImpl.GenTiling(tiling_res, {}, 0, enable_group_parallels), af::SUCCESS);

  EXPECT_EQ(genImpl.tiling_func_.GetOutputStr().empty(), false);
}

TEST(GeneratorUT, RootGetTilingFailuresUseWarningLogOnlyForPGOPath) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ModelInfo info;
  tiling_model_info.push_back(info);
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);
  genImpl.config_.cache_enabled_at_compile_time = false;
  std::map<size_t, std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>>> namespace_map;
  namespace_map[0][0] = {};

  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenFusedScheduleResultsGetTilingDefine(namespace_map), af::SUCCESS);
  std::string tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("OP_LOGE(OP_NAME, \"Failed to get tiling of AscGraph0.\");"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("OP_LOGW(OP_NAME, \"Failed to get tiling of AscGraph0.\");"), std::string::npos);

  genImpl.config_.is_inductor_scene = true;
  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenFusedScheduleResultsGetTilingDefine(namespace_map), af::SUCCESS);
  tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("OP_LOGW(OP_NAME, \"Failed to get tiling of AscGraph0.\");"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("OP_LOGE(OP_NAME, \"Failed to get tiling of AscGraph0.\");"), std::string::npos);

  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenPGOByCoreNumFusedScheduleResultsGetTilingDefine(namespace_map), af::SUCCESS);
  tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("OP_LOGW(OP_NAME, \"Failed to get tiling of AscGraph0.\");"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("OP_LOGE(OP_NAME, \"Failed to get tiling of AscGraph0.\");"), std::string::npos);

  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenPGOFusedScheduleResultsGetTilingDefine(namespace_map), af::SUCCESS);
  tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("OP_LOGW(OP_NAME, \"Failed to get tiling of AscGraph0.\");"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("OP_LOGE(OP_NAME, \"Failed to get tiling of AscGraph0.\");"), std::string::npos);
}

TEST(GeneratorUT, PGOGetTilingKeyFailureUsesWarningLog) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  ModelInfo info;
  info.schedule_group_ident.group_id = 0;
  tiling_model_info.push_back(info);

  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);
  genImpl.config_.enable_autofuse_pgo = true;
  genImpl.config_.is_inductor_scene = true;
  genImpl.config_.cache_enabled_at_compile_time = false;
  EXPECT_EQ(genImpl.GenGetTilingKeyCall(""), af::SUCCESS);

  const std::string tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("OP_LOGW(OP_NAME, \"GetTiling Failed.\");"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("OP_LOGE(OP_NAME, \"GetTiling Failed.\");"), std::string::npos);
}

TEST(GeneratorUT, GetResultSummaryFailureUsesWarningLogForPGOPath) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ModelInfo info;
  tiling_model_info.push_back(info);
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);

  EXPECT_EQ(genImpl.GenGetResultSummary(0), af::SUCCESS);
  std::string tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("OP_LOGE(OP_NAME, \"GetTiling Failed.\");"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("OP_LOGW(OP_NAME, \"GetTiling Failed.\");"), std::string::npos);

  genImpl.config_.enable_autofuse_pgo = true;
  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenGetResultSummary(0), af::SUCCESS);
  tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("OP_LOGW(OP_NAME, \"GetTiling Failed.\");"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("OP_LOGE(OP_NAME, \"GetTiling Failed.\");"), std::string::npos);

  genImpl.config_.enable_autofuse_pgo = false;
  genImpl.config_.is_inductor_scene = true;
  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenGetResultSummary(0), af::SUCCESS);
  tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("OP_LOGW(OP_NAME, \"GetTiling Failed.\");"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("OP_LOGE(OP_NAME, \"GetTiling Failed.\");"), std::string::npos);
}

TEST(GeneratorUT, PGOGetAllSchedulesResultsDoesNotPushGraphTilingTmpOutsideScheduleResult) {
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.enable_autofuse_pgo = true;
  config.is_inductor_scene = true;
  TilingModelInfo tiling_model_info;
  ModelInfo info;
  tiling_model_info.push_back(info);
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);
  std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>> namespace_map;
  namespace_map[0] = {};
  namespace_map[1] = {};

  genImpl.GenPGOGetAllSchedulesResults(0, namespace_map);

  const std::string tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("if (!AscGraph0::GetTiling(tilingTmp, index, nullptr)) {"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("cur_perf = DBL_MAX;"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("continue;"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("tiling_perf.tiling_data = tilingTmp;"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("tiling_data_list.push_back(tiling_perf);"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("PgoConfig::Instance().single_callback("), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("*tilingData = tilingTmp;"), std::string::npos);
}

static const std::string kExpectPGOCode =
    R"rawliteral(inline bool GetScheduleResult0PGO(std::vector<AutofuseTilingDataPerf>& tiling_data_list, const uint32_t ori_block_dim, const int32_t tiling_case_id,AutofuseTilingData &tiling_data, double &cur_perf, double &best_perf, uint32_t &cur_block_dim,void* stream, uint32_t workspaceSize, std::vector<uint32_t*> multi_group_block_dim_list = {}, const SearchConfig *search_cfg=nullptr) {
  (void)cur_perf; (void)cur_block_dim;
  std::vector<AutofuseTilingDataPerf> tiling_data_list_tmp{};
  workspaceSize = 0;
  std::unordered_map<int64_t, uint64_t> workspace_map_filter_use{};
  tiling_data.set_graph0_tiling_key(0);
  auto &group0_tiling_data = tiling_data.group0_tiling_data;
  group0_tiling_data.set_block_dim(ori_block_dim);
  size_t candidate_begin_index0 = tiling_data_list_tmp.size();
  auto result0 = ScheduleResult0::PGOSearchTilingKey(tiling_data_list_tmp, group0_tiling_data, tiling_case_id, &tiling_data, PgoConfig::Instance().tensor_args, stream, workspaceSize, best_perf, workspace_map_filter_use, multi_group_block_dim_list, search_cfg);
  if (result0) {
    bool has_solution = true;
    std::vector<bool> valid_candidates(tiling_data_list_tmp.size() - candidate_begin_index0, true);
    for (size_t candidate_index = candidate_begin_index0; candidate_index < tiling_data_list_tmp.size(); ++candidate_index) {
      auto &tiling_data_perf = tiling_data_list_tmp[candidate_index];
      auto &tiling_data = tiling_data_perf.tiling_data;
      std::unordered_map<int64_t, uint64_t> workspace_map;
      workspace_map.reserve(workspace_map_filter_use.size());
      workspace_map.insert(workspace_map_filter_use.begin(), workspace_map_filter_use.end());
      tiling_data.group1_tiling_data.set_block_dim(ori_block_dim);
      has_solution = ScheduleResult0::GetTiling(tiling_data.group1_tiling_data, workspace_map, -1);
      if (!has_solution) {
        OP_LOGI(OP_NAME, "No solution for group0 at group1");
        valid_candidates[candidate_index - candidate_begin_index0] = false;
        continue;
      }
      tiling_data.set_block_dim(Max(tiling_data.group0_tiling_data.get_block_dim(), tiling_data.group1_tiling_data.get_block_dim()));
      auto workspaceSizeTmp = GetWorkspaceSize(tiling_data);
      if (workspaceSizeTmp > workspaceSize) {
        workspaceSize = workspaceSizeTmp;
      }
    }
    workspaceSize += 16 * 1024 * 1024;
    if (PgoConfig::Instance().batch_callback) {
      PgoConfig::Instance().batch_callback(PgoConfig::Instance().tensor_args, stream, workspaceSize, &tiling_data_list_tmp);
    }
    for (size_t candidate_index = candidate_begin_index0; candidate_index < tiling_data_list_tmp.size(); ++candidate_index) {
      const size_t candidate_offset = candidate_index - candidate_begin_index0;
      if (candidate_offset >= valid_candidates.size() || !valid_candidates[candidate_offset]) {
        continue;
      }
      auto &tiling_data_perf = tiling_data_list_tmp[candidate_index];
      tiling_data_list.push_back(tiling_data_perf);
      if (tiling_data_perf.best_perf < best_perf) {
        tiling_data = tiling_data_perf.tiling_data;
        best_perf = tiling_data_perf.best_perf;
      }
    }
  }
  auto &group1_tiling_data = tiling_data.group1_tiling_data;
  group1_tiling_data.set_block_dim(ori_block_dim);
  size_t candidate_begin_index1 = tiling_data_list_tmp.size();
  auto result1 = ScheduleResult0::PGOSearchTilingKey(tiling_data_list_tmp, group1_tiling_data, tiling_case_id, &tiling_data, PgoConfig::Instance().tensor_args, stream, workspaceSize, best_perf, workspace_map_filter_use, multi_group_block_dim_list, search_cfg);
  if (result1) {
    bool has_solution = true;
    std::vector<bool> valid_candidates(tiling_data_list_tmp.size() - candidate_begin_index1, true);
    for (size_t candidate_index = candidate_begin_index1; candidate_index < tiling_data_list_tmp.size(); ++candidate_index) {
      auto &tiling_data_perf = tiling_data_list_tmp[candidate_index];
      auto &tiling_data = tiling_data_perf.tiling_data;
      std::unordered_map<int64_t, uint64_t> workspace_map;
      workspace_map.reserve(workspace_map_filter_use.size());
      workspace_map.insert(workspace_map_filter_use.begin(), workspace_map_filter_use.end());
      tiling_data.set_block_dim(Max(tiling_data.group0_tiling_data.get_block_dim(), tiling_data.group1_tiling_data.get_block_dim()));
      auto workspaceSizeTmp = GetWorkspaceSize(tiling_data);
      if (workspaceSizeTmp > workspaceSize) {
        workspaceSize = workspaceSizeTmp;
      }
    }
    workspaceSize += 16 * 1024 * 1024;
    if (PgoConfig::Instance().batch_callback) {
      PgoConfig::Instance().batch_callback(PgoConfig::Instance().tensor_args, stream, workspaceSize, &tiling_data_list_tmp);
    }
    for (size_t candidate_index = candidate_begin_index1; candidate_index < tiling_data_list_tmp.size(); ++candidate_index) {
      const size_t candidate_offset = candidate_index - candidate_begin_index1;
      if (candidate_offset >= valid_candidates.size() || !valid_candidates[candidate_offset]) {
        continue;
      }
      auto &tiling_data_perf = tiling_data_list_tmp[candidate_index];
      tiling_data_list.push_back(tiling_data_perf);
      if (tiling_data_perf.best_perf < best_perf) {
        tiling_data = tiling_data_perf.tiling_data;
        best_perf = tiling_data_perf.best_perf;
      }
    }
  }
  return true;
}
)rawliteral";

TEST(GeneratorUT, GenGetScheduleResultPGOSuccess) {
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;

  TilingModelInfo tiling_model_info;
  ModelInfo group0_info;
  group0_info.schedule_group_ident.asc_graph_id = 0;
  group0_info.schedule_group_ident.impl_graph_id = 0;
  group0_info.schedule_group_ident.group_id = 0;
  tiling_model_info.push_back(group0_info);
  ModelInfo group1_info;
  group1_info.schedule_group_ident.asc_graph_id = 0;
  group1_info.schedule_group_ident.impl_graph_id = 0;
  group1_info.schedule_group_ident.group_id = 1;
  tiling_model_info.push_back(group1_info);

  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);

  std::map<size_t, std::pair<std::string, std::string>> graph_info;
  graph_info[0] = std::make_pair("ScheduleResult0", "group0");
  graph_info[1] = std::make_pair("ScheduleResult0", "group1");

  std::map<std::string, std::set<std::string>> hardware_map;
  hardware_map["group0"].insert("block_dim");
  hardware_map["group1"].insert("block_dim");

  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenPGOGetScheduleResult(0, 0, graph_info, hardware_map), af::SUCCESS);
  EXPECT_EQ(genImpl.tiling_func_.output_.str(), kExpectPGOCode);

  EnableGroupParallels enable_group_parallels;
  enable_group_parallels[0][0] = true;
  genImpl.tiling_func_.Reset();
  genImpl.enable_group_parallels_ = enable_group_parallels;
  EXPECT_EQ(genImpl.GenPGOGetScheduleResult(0, 0, graph_info, hardware_map), af::SUCCESS);
  const std::string tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_EQ(tiling_func_output.empty(), false);
  EXPECT_NE(tiling_func_output.find("tiling_data.set_block_dim(Max(tiling_data.group0_tiling_data.get_block_dim(), "
                                    "tiling_data.group1_tiling_data.get_block_dim()));"),
            std::string::npos);
  EXPECT_NE(tiling_func_output.find("ArrangeBlockOffsetsAscGraph0Result0(tiling_data, ori_block_dim);"),
            std::string::npos);
  EXPECT_EQ(tiling_func_output.find("ArrangeBlockOffsetsAscGraph0Result0(tiling_data, tiling_data.get_block_dim());"),
            std::string::npos);
}

TEST(GeneratorUT, GenPGOGetScheduleResultSetsCurrentGroupVarRelationsBeforeSearch) {
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;

  TilingModelInfo tiling_model_info;
  ModelInfo group0_info;
  group0_info.schedule_group_ident.asc_graph_id = 0;
  group0_info.schedule_group_ident.impl_graph_id = 0;
  group0_info.schedule_group_ident.group_id = 0;
  tiling_model_info.push_back(group0_info);
  ModelInfo group1_info;
  group1_info.schedule_group_ident.asc_graph_id = 0;
  group1_info.schedule_group_ident.impl_graph_id = 0;
  group1_info.schedule_group_ident.group_id = 1;
  tiling_model_info.push_back(group1_info);

  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);
  genImpl.var_relations_[0][0][1][0]["dep_size"] = CreateExpr("src_size") + CreateExpr(1);

  std::map<size_t, std::pair<std::string, std::string>> graph_info;
  graph_info[0] = std::make_pair("ScheduleResult0", "group0");
  graph_info[1] = std::make_pair("ScheduleResult1", "group1");

  std::map<std::string, std::set<std::string>> hardware_map;
  hardware_map["group0"].insert("block_dim");
  hardware_map["group1"].insert("block_dim");

  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenPGOGetScheduleResult(0, 0, graph_info, hardware_map), af::SUCCESS);
  const std::string tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  const auto search_pos =
      tiling_func_output.find("ScheduleResult1::PGOSearchTilingKey(tiling_data_list_tmp, group1_tiling_data");
  const auto group1_entry_pos =
      tiling_func_output.rfind("auto &group1_tiling_data = tiling_data.group1_tiling_data;", search_pos);
  const auto guard_pos =
      tiling_func_output.find("tiling_data.group0_tiling_data.get_src_size() == 0U", group1_entry_pos);
  const auto value_pos = tiling_func_output.find("double group1_dep_size_var_relation_value =", group1_entry_pos);
  const auto set_pos = tiling_func_output.find(
      "tiling_data.group1_tiling_data.set_dep_size(static_cast<uint32_t>(group1_dep_size_var_relation_value))",
      group1_entry_pos);

  ASSERT_NE(search_pos, std::string::npos);
  ASSERT_NE(group1_entry_pos, std::string::npos);
  ASSERT_NE(guard_pos, std::string::npos);
  ASSERT_NE(value_pos, std::string::npos);
  ASSERT_NE(set_pos, std::string::npos);
  EXPECT_LT(group1_entry_pos, set_pos);
  EXPECT_LT(guard_pos, value_pos);
  EXPECT_LT(value_pos, set_pos);
  EXPECT_LT(set_pos, search_pos);
}

TEST(GeneratorUT, GenPGOGetScheduleResultGuardsInvalidVarRelationBeforeSet) {
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;
  config.is_inductor_scene = true;

  TilingModelInfo tiling_model_info;
  ModelInfo group0_info;
  group0_info.schedule_group_ident.asc_graph_id = 0;
  group0_info.schedule_group_ident.impl_graph_id = 0;
  group0_info.schedule_group_ident.group_id = 0;
  tiling_model_info.push_back(group0_info);
  ModelInfo group1_info;
  group1_info.schedule_group_ident.asc_graph_id = 0;
  group1_info.schedule_group_ident.impl_graph_id = 0;
  group1_info.schedule_group_ident.group_id = 1;
  tiling_model_info.push_back(group1_info);

  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);
  genImpl.var_relations_[0][0][1][0]["Rm_org_size"] =
      af::sym::Ceiling(af::sym::Ceiling(CreateExpr(7) / CreateExpr("z2t_size")) / CreateExpr("z2Tt_size"));

  std::map<size_t, std::pair<std::string, std::string>> graph_info;
  graph_info[0] = std::make_pair("ScheduleResult0", "group0");
  graph_info[1] = std::make_pair("ScheduleResult1", "group1");

  std::map<std::string, std::set<std::string>> hardware_map;
  hardware_map["group0"].insert("block_dim");
  hardware_map["group1"].insert("block_dim");

  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenPGOGetScheduleResult(0, 0, graph_info, hardware_map), af::SUCCESS);
  const std::string tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  const auto guard_pos = tiling_func_output.find("get_z2t_size() == 0U");
  const auto value_pos = tiling_func_output.find("double group1_Rm_org_size_var_relation_value =");
  const auto finite_pos = tiling_func_output.find("!std::isfinite(group1_Rm_org_size_var_relation_value)");
  const auto set_pos = tiling_func_output.find("set_Rm_org_size(static_cast<uint32_t>(");
  const auto invalid_pos =
      tiling_func_output.find("valid_candidates[candidate_index - candidate_begin_index0] = false;");
  const auto search_pos =
      tiling_func_output.find("ScheduleResult1::GetTiling(tiling_data.group1_tiling_data, workspace_map, -1)");

  ASSERT_NE(guard_pos, std::string::npos);
  EXPECT_NE(tiling_func_output.find("get_z2Tt_size() == 0U"), std::string::npos);
  ASSERT_NE(value_pos, std::string::npos);
  ASSERT_NE(finite_pos, std::string::npos);
  ASSERT_NE(set_pos, std::string::npos);
  ASSERT_NE(invalid_pos, std::string::npos);
  ASSERT_NE(search_pos, std::string::npos);
  EXPECT_LT(guard_pos, value_pos);
  EXPECT_LT(value_pos, finite_pos);
  EXPECT_LT(finite_pos, set_pos);
  EXPECT_LT(set_pos, search_pos);
}

TEST(GeneratorUT, GenWorkspaceRelatedVarsGuardsDynamicDenominator) {
  std::map<int64_t, Expr> workspace_size_map;
  workspace_size_map[0] = af::sym::Ceiling(CreateExpr(512) / CreateExpr("a1t_size"));

  const auto code = GenWorkspaceRelatedVars(workspace_size_map, {});

  EXPECT_NE(code.find("double a1t_size = tiling_data.get_a1t_size();"), std::string::npos);
  EXPECT_NE(code.find("if (a1t_size <= 0) {"), std::string::npos);
  EXPECT_NE(code.find("return;"), std::string::npos);
  EXPECT_LT(code.find("if (a1t_size <= 0) {"), code.find("static_cast<uint64_t>(Ceiling(512/a1t_size))"));
}

// UT测试：验证tiling_data.set参数溢出修复
// 测试用例1: 验证 MemoryTilingDataGen::GenFuncImpl 使用 static_cast<uint32_t>()
TEST(GeneratorUT, MemoryTilingDataGen_GenFuncImpl_UseStaticCast) {
  ModelInfo model_info;
  // 创建大数值表达式: 70000 * 70000 * 4 > UINT32_MAX(4294967295)
  // 70000 * 70000 = 4900000000 > UINT32_MAX
  Expr large_expr = af::Symbol(70000, "tmp") * af::Symbol(70000, "tmp") * af::Symbol(4, "tmp");
  model_info.container_exprs["LargeContainer"] = large_expr;

  // 创建 MemoryTilingDataGen 对象
  auto memory_gen = att::MemoryTilingDataGen(model_info);
  EXPECT_EQ(memory_gen.Init(), af::SUCCESS);

  // 获取生成的函数实现代码
  const std::vector<std::string> func_impls = memory_gen.GetTilingFuncImpl("TestTilingData");

  // 验证生成的代码包含 "static_cast<uint32_t>("
  bool found_static_cast = false;
  for (const auto &code : func_impls) {
    if (code.find("static_cast<uint32_t>(") != std::string::npos) {
      found_static_cast = true;
      break;
    }
  }
  EXPECT_TRUE(found_static_cast) << "Generated code should contain 'static_cast<uint32_t>(' to prevent overflow";
}

// 测试用例2: 验证硬件约束代码生成使用 double 类型
TEST(GeneratorUT, GenHardwareCheckCode_UseDoubleType) {
  ModelInfo model_info;
  // 创建大数值硬件约束: 102400 * 102400 = 10485760000，可能导致 uint32_t 溢出
  Expr large_hardware_expr = af::Symbol(102400, "tmp") * af::Symbol(102400, "tmp");
  model_info.hardware_cons[HardwareDef::UB] = large_hardware_expr;

  TilingModelInfo model_infos;
  model_infos.emplace_back(model_info);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = false;
  config.gen_tiling_data = false;
  MockTilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), af::SUCCESS);

  // 获取生成的代码
  std::map<std::string, std::string> tiling_res;
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), af::SUCCESS);

  // 验证生成的代码包含 "double " 类型声明
  bool found_double_type = false;
  for (const auto &[key, code] : tiling_res) {
    if (code.find("double ") != std::string::npos) {
      found_double_type = true;
      break;
    }
  }
  EXPECT_TRUE(found_double_type) << "Generated hardware check code should contain 'double ' type to prevent overflow";
}

// Task 3: Inductor scene triggers ATT PGO main search skeleton, PGOSearchTilingKey and perf extraction

TEST(GeneratorUT, InductorSceneTriggersPGOSkeletonAndSearchTilingKey) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  EnableGroupParallels enable_group_parallels;
  std::map<std::string, std::string> tiling_res;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;

  ModelInfo info;
  info.reuse_schedule_group = std::make_shared<ReuseScheduleGroup>();
  tiling_model_info.push_back(info);
  enable_group_parallels[0][0] = true;

  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);
  genImpl.config_.is_inductor_scene = true;
  genImpl.config_.gen_tiling_data = false;
  EXPECT_EQ(genImpl.GenTiling(tiling_res, {}, 0, enable_group_parallels), af::SUCCESS);

  std::string tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_FALSE(tiling_func_output.empty());
  // Inductor scene must generate ExecutePGOSolver override for single-group
  EXPECT_NE(tiling_func_output.find("bool ExecutePGOSolver("), std::string::npos);
  // Inductor scene must generate SearchAllTilingbyCaseId
  EXPECT_NE(tiling_func_output.find("SearchAllTilingbyCaseId("), std::string::npos);
  // PGOSearchTilingKey must be generated
  EXPECT_NE(tiling_func_output.find("PGOSearchTilingKey("), std::string::npos);
  // GetPerf must be called inside ExecutePGOSolver override
  EXPECT_NE(tiling_func_output.find("GetPerf(tiling_data)"), std::string::npos);
}

// ============================================================================
// Cache line conflict detection tests (Task 1 - expected to fail until feature is implemented)
// ============================================================================

// Test 1: Both groups have cache line conflict → should use sum aggregation
TEST(GeneratorUT, GroupParallelCacheLine_AllConflict_UseSumAggregation) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  // Group 0: conflict (expr=4/8, small value → not aligned to cache line)
  auto info0 =
      CreateGroupParallelCacheLineModelInfo(0, 0, CreateExpr(4) / CreateExpr(8), 128, CacheLineDirection::kUbToGm);
  // Group 1: conflict (expr=4/8, same conflict)
  auto info1 =
      CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(4) / CreateExpr(8), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  EXPECT_THAT(all_code, HasSubstr("IsConflictGroup_0_0_0_0"));
  EXPECT_THAT(all_code, HasSubstr("IsConflictGroup_0_0_1_1"));
  EXPECT_THAT(all_code, HasSubstr("conflict_perf_sum"));
  EXPECT_THAT(all_code, HasSubstr("conflict_perf_sum + normal_perf_merged"));
}

// Test 2: Boundary expression exactly equals cache_line_size → stays normal
TEST(GeneratorUT, GroupParallelCacheLine_BoundaryEqualCacheLine_StaysNormal) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  auto info0 =
      CreateGroupParallelCacheLineModelInfo(0, 0, CreateExpr(128) / CreateExpr(256), 128, CacheLineDirection::kUbToGm);
  auto info1 =
      CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(128) / CreateExpr(256), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  EXPECT_THAT(all_code, HasSubstr("< 128"));
  EXPECT_THAT(all_code, HasSubstr("return false"));
}

// Test 3: Group0 conflict, Group1 normal → init from first normal group
TEST(GeneratorUT, GroupParallelCacheLine_FirstConflictSecondNormal_InitFromFirstNormal) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  // Group 0: conflict (expr=4, small value)
  auto info0 = CreateGroupParallelCacheLineModelInfo(0, 0, CreateExpr(4), 128, CacheLineDirection::kUbToGm);
  // Group 1: normal (expr=256, large aligned value)
  auto info1 = CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(256), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  EXPECT_THAT(all_code, HasSubstr("has_normal_group"));
  EXPECT_THAT(all_code, HasSubstr("normal_perf_merged += cur_tmp_perf;"));
  EXPECT_THAT(all_code, HasSubstr("Final normal perf"));
  EXPECT_THAT(all_code, HasSubstr("conflict_perf_sum +="));
}

// Test 4: Multi-case final tiling key dispatch uses case helper
TEST(GeneratorUT, GroupParallelCacheLine_FinalTilingKeyDispatch_UsesFinalCaseHelper) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  // Group 0: 2 cases (case_id 0 and 1), both conflict
  auto info0_case0 =
      CreateGroupParallelCacheLineModelInfo(0, 0, CreateExpr(4) / CreateExpr(8), 128, CacheLineDirection::kUbToGm);
  auto info0_case1 =
      CreateGroupParallelCacheLineModelInfo(0, 1, CreateExpr(4) / CreateExpr(8), 128, CacheLineDirection::kUbToGm);
  // Group 1: 1 case (case_id 2), normal
  auto info1_case2 = CreateGroupParallelCacheLineModelInfo(1, 2, CreateExpr(256), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0_case0, info0_case1};
  schedule_result.groups_tiling_model_info[1] = {info1_case2};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  EXPECT_THAT(all_code, HasSubstr("get_tiling_key())"));
  EXPECT_THAT(all_code, HasSubstr("case 0: return IsConflictGroup_0_0_0_0()"));
  EXPECT_THAT(all_code, HasSubstr("case 1: return IsConflictGroup_0_0_0_1()"));
}

// Test 5: Byte expression does not multiply dtype_size again
TEST(GeneratorUT, GroupParallelCacheLine_ByteExprDoesNotMultiplyDtypeAgain) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  // Composite expression: CreateExpr(64) * CreateExpr(2) = 128 bytes (already in bytes)
  Expr byte_expr = CreateExpr(64) * CreateExpr(2);
  auto info0 = CreateGroupParallelCacheLineModelInfo(0, 0, byte_expr, 128, CacheLineDirection::kUbToGm);
  auto info1 = CreateGroupParallelCacheLineModelInfo(1, 1, byte_expr, 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  using testing::Not;
  EXPECT_THAT(all_code, Not(HasSubstr("dtype_size")));
  EXPECT_THAT(all_code, HasSubstr("< 128"));
}

// Test 6: Missing schedule table → fallback to normal with log
TEST(GeneratorUT, GroupParallelCacheLine_MissingScheduleTable_FallbackToNormalWithLog) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  // Group 0: no schedule table (nullptr), manually created
  ModelInfo info0 = CreateModelInfo();
  info0.schedule_group_ident.asc_graph_id = 0;
  info0.schedule_group_ident.impl_graph_id = 0;
  info0.schedule_group_ident.group_id = 0;
  info0.tiling_case_id = 0;
  info0.enable_group_parallel = true;
  info0.tiling_schedule_config_table = nullptr;
  CacheLineConfig cfg0;
  cfg0.node_name = "test_cache_line_node";
  cfg0.cache_line_expr = CreateExpr(4);
  cfg0.cache_line_size = 128;
  cfg0.direction = CacheLineDirection::kUbToGm;
  info0.cache_line_config = {cfg0};

  // Group 1: normal
  auto info1 = CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(256), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  EXPECT_THAT(all_code, HasSubstr("cache line size is unavailable, fallback to normal group"));
}

// Test 7: Direction is kGmToUb (read) → should also use conflict aggregation
TEST(GeneratorUT, GroupParallelCacheLine_GmToUbConflict_UseSumAggregation) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  // Group 0: kGmToUb direction (read)
  auto info0 = CreateGroupParallelCacheLineModelInfo(0, 0, CreateExpr(4), 128, CacheLineDirection::kGmToUb);
  // Group 1: kUbToGm direction (write, valid)
  auto info1 = CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(4), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  EXPECT_THAT(all_code, HasSubstr("IsConflictGroup_0_0_0_0"));
  EXPECT_THAT(all_code, HasSubstr("IsConflictGroup_0_0_1_1"));
  EXPECT_THAT(all_code, HasSubstr("conflict_perf_sum +="));
  EXPECT_THAT(all_code, HasSubstr("conflict_perf_sum + normal_perf_merged"));
  EXPECT_THAT(all_code, Not(HasSubstr("no valid gm<->ub cache line expr, fallback to normal group")));
}

// Test 8: Duplicate final tiling key → fallback to normal with log
TEST(GeneratorUT, GroupParallelCacheLine_DuplicateFinalKey_FallbackToNormalWithLog) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  // Group 0: 2 cases with SAME tiling_case_id=0 (duplicate key)
  auto info0_case0 =
      CreateGroupParallelCacheLineModelInfo(0, 0, CreateExpr(4) / CreateExpr(8), 128, CacheLineDirection::kUbToGm);
  auto info0_case1 =
      CreateGroupParallelCacheLineModelInfo(0, 0, CreateExpr(4) / CreateExpr(8), 128, CacheLineDirection::kUbToGm);

  // Group 1: normal
  auto info1 = CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(256), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0_case0, info0_case1};
  schedule_result.groups_tiling_model_info[1] = {info1};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  EXPECT_THAT(all_code, HasSubstr("duplicate final tiling key mapping, fallback to normal group"));
  EXPECT_EQ(CountSubstr(all_code, "auto IsConflictGroup_0_0_0_0 ="), 1U);
}

TEST(GeneratorUT, GroupParallelCacheLine_DynamicInputSizeSymbols_GenerateContext) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  Expr s1 = CreateExpr("s1");
  Expr s20 = CreateExpr("s20");
  auto info0 = CreateGroupParallelCacheLineModelInfo(0, 0, s1 * s20 * CreateExpr(4), 128, CacheLineDirection::kUbToGm);
  info0.sizes = {s1, s20};
  auto info1 = CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(256), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  using testing::Not;
  EXPECT_THAT(all_code, HasSubstr("auto s1 = group_tiling_data.get_s1();"));
  EXPECT_THAT(all_code, HasSubstr("auto s20 = group_tiling_data.get_s20();"));
  EXPECT_THAT(all_code, Not(HasSubstr("cache line expr is not codegenable, fallback to normal group")));
}

TEST(GeneratorUT, GroupParallelCacheLine_UnknownDirection_FallbackToNormal) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  auto info0 = CreateGroupParallelCacheLineModelInfo(0, 0, CreateExpr(4), 128, CacheLineDirection::kUnknown);
  auto info1 = CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(256), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};
  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  EXPECT_THAT(all_code, HasSubstr("no valid gm<->ub cache line expr, fallback to normal group"));
}

TEST(GeneratorUT, GroupParallelCacheLine_MultiWriteExprs_DeduplicateContext) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  Expr s1 = CreateExpr("s1");
  Expr s20 = CreateExpr("s20");
  auto info0 = CreateGroupParallelCacheLineModelInfo(0, 0, s1 * s20 * CreateExpr(4), 128, CacheLineDirection::kUbToGm);
  info0.sizes = {s1, s20};
  CacheLineConfig second_cfg = info0.cache_line_config[0];
  second_cfg.node_name = "test_cache_line_node2";
  info0.cache_line_config.push_back(second_cfg);
  auto info1 = CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(256), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};
  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  EXPECT_EQ(CountSubstr(all_code, "auto s1 = group_tiling_data.get_s1();"), 1U);
  EXPECT_EQ(CountSubstr(all_code, "auto s20 = group_tiling_data.get_s20();"), 1U);
}

}  // namespace att
