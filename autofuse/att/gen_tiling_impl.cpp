/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gen_tiling_impl.h"
#include "base/att_const_values.h"
#include "common/checker.h"
#include "util/duration.h"
#include "gen_model_info/gen_model_info.h"
#include "tiling_code_generator.h"
#include "autofuse_config/auto_fuse_config.h"
#include "autofuse_config/auto_fuse_config_utils.h"
#include "util/option_register.h"
#include "reuse_group_utils/reuse_group_utils.h"
#include "common/scope_tracing_recorder.h"
#include "common_utils.h"

namespace att {
namespace {
constexpr uint32_t kPercentageDivisor = 100;
TilingImplType GetTilingAlgorithm(const std::string &algorithm_name) {
  static const std::map<std::string, TilingImplType> kAttTilingAlgorithmMap = {
      {"AxesReorder", TilingImplType::AXES_REORDER},
      {"HighPerf", TilingImplType::HIGH_PERF},
  };
  const auto iter = kAttTilingAlgorithmMap.find(algorithm_name);
  if (iter != kAttTilingAlgorithmMap.cend()) {
    return iter->second;
  }
  return TilingImplType::AXES_REORDER;
}

void PgoEnvConfigInit(TilingCodeGenConfig &generator_config) {
  const auto res_pgo = AutoFuseConfig::MutablePgoStrategyConfig().Init();
  if (res_pgo == af::SUCCESS) {
    if (AutoFuseConfig::GetPgoStrategyConfig().set_env_enable_autofuse_pgo) {
      generator_config.enable_autofuse_pgo = (AutoFuseConfig::GetPgoStrategyConfig().enable_autofuse_pgo == "true");
    }
    if (AutoFuseConfig::GetPgoStrategyConfig().set_env_autofuse_pgo_algo_step_max) {
      generator_config.pgo_step_max = AutoFuseConfig::GetPgoStrategyConfig().autofuse_pgo_algo_step_max;
    }
  }
}

void OverridePgoConfigByOptions(TilingCodeGenConfig &generator_config,
                                const std::map<std::string, std::string> &options) {
  const auto iter = options.find(kInternalEnableAutofusePgo);
  if (iter != options.cend()) {
    generator_config.enable_autofuse_pgo = (iter->second == "true");
  }
}

af::Status InitializeConfigByEnvOrIni(TilingCodeGenConfig &generator_config) {
  // ATT的配置初始化，当前前端不支持传入配置文件的目录，所以这里直接使用默认的配置文件路径
  const auto res = AutoFuseConfig::MutableAttStrategyConfig().Init();
  if (res == af::SUCCESS) {
    if (AutoFuseConfig::GetAttStrategyConfig().set_env_tiling_algorithm) {
      generator_config.type = GetTilingAlgorithm(AutoFuseConfig::GetAttStrategyConfig().tiling_algorithm);
    }
    if (AutoFuseConfig::GetAttStrategyConfig().set_env_solution_accuracy_level) {
      generator_config.high_precision = (AutoFuseConfig::GetAttStrategyConfig().solution_accuracy_level == 1L);
    }
    if (AutoFuseConfig::GetAttStrategyConfig().set_env_ub_threshold) {
      generator_config.ub_threshold =
          (static_cast<double>(AutoFuseConfig::GetAttStrategyConfig().ub_threshold) / kPercentageDivisor);
    }
    if (AutoFuseConfig::GetAttStrategyConfig().set_env_corenum_threshold) {
      generator_config.corenum_threshold =
          (static_cast<double>(AutoFuseConfig::GetAttStrategyConfig().corenum_threshold) / kPercentageDivisor);
    }
    if (AutoFuseConfig::GetAttStrategyConfig().set_env_enable_small_shape_strategy) {
      generator_config.enable_small_shape_strategy =
          (AutoFuseConfig::GetAttStrategyConfig().enable_small_shape_strategy == "true");
    }
    if (AutoFuseConfig::GetAttStrategyConfig().set_env_enable_multicore_ub_tradeoff) {
      generator_config.enable_multicore_ub_tradeoff =
          (AutoFuseConfig::GetAttStrategyConfig().enable_multicore_ub_tradeoff == "true");
    }
    if (AutoFuseConfig::GetAttStrategyConfig().set_force_tiling_case) {
      GE_ASSERT_SUCCESS(ge::AttStrategyConfigUtils::ParseForceTilingCase(
          AutoFuseConfig::GetAttStrategyConfig().force_tiling_case, generator_config.force_tiling_case));
    }
    if (AutoFuseConfig::GetAttStrategyConfig().set_force_schedule_result) {
      generator_config.force_schedule_result = AutoFuseConfig::GetAttStrategyConfig().force_schedule_result;
    }
    if (AutoFuseConfig::GetAttStrategyConfig().set_force_template_op_name) {
      generator_config.force_template_op_name = AutoFuseConfig::GetAttStrategyConfig().force_template_op_name;
    }
  }
  PgoEnvConfigInit(generator_config);
  return af::SUCCESS;
}

uint32_t GetDurationLevel(const std::map<std::string, std::string> &options) {
  uint32_t duration_level = 0U;
  const auto iter_duration_level = options.find(kDurationLevelName);
  if (iter_duration_level != options.end()) {
    try {
      duration_level = static_cast<uint32_t>(std::stoi(iter_duration_level->second));
    } catch (...) {
      GELOGW("Invalid %s[%s], set default value[0].", kDurationLevelName.c_str(), iter_duration_level->second.c_str());
    }
  }
  return duration_level;
}

string GetOptionValue(const std::map<std::string, std::string> &options, const std::string &name) {
  if (options.find(name) != options.cend()) {
    return options.at(name);
  }
  GELOGW("option value not found by name %s", name.c_str());
  return "";
}

void InitializeConfig(TilingCodeGenConfig &generator_config, const std::map<std::string, std::string> &options) {
  generator_config.type = GetTilingAlgorithm(GetOptionValue(options, kGenConfigType));
  generator_config.path = GetOptionValue(options, kOutputFilePath);
  generator_config.tiling_data_type_name = GetOptionValue(options, kTilingDataTypeName);
  generator_config.gen_tiling_data = (GetOptionValue(options, kGenTilingDataDef) == kIsTrue);
  generator_config.high_precision = (GetOptionValue(options, kHighPrecision) == kIsTrue);
  generator_config.gen_extra_infos = (GetOptionValue(options, kGenExtraInfo) == kIsTrue);
  generator_config.do_variable_replace = (GetOptionValue(options, kVariableReplace) == kIsTrue);
}
}  // namespace

bool GenTilingImpl(const std::string &op_name, const std::vector<af::AscGraph> &graphs,
                   std::map<std::string, std::string> &options) {
  try {
    GELOGI("Gen tiling for total [%zu] graphs.", graphs.size());
    if (graphs.empty()) {
      return false;
    }
    for (const auto &graph : graphs) {
      if (!graph.CheckValid()) {
        return false;
      }
    }
    std::map<std::string, std::string> inner_options;
    if (!RegisterOptionsAndInitInnerOptions(inner_options, options, graphs[0].GetName())) {
      return false;
    }
    const auto duration_level = GetDurationLevel(inner_options);
    DurationInitGuard duration_init_guard(duration_level);
    std::vector<ModelInfo> model_info_list;
    GE_ASSERT_SUCCESS(GenerateModelInfo(graphs, model_info_list, inner_options), "Get model info failed.");
    GE_ASSERT_SUCCESS(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_info_list),
                      "Init reuse schedule group failed");
    TilingCodeGenConfig generator_config;
    InitializeConfig(generator_config, inner_options);
    GE_ASSERT_SUCCESS(InitializeConfigByEnvOrIni(generator_config));
    TilingCodeGenerator generator;
    GE_ASSERT_SUCCESS(generator.GenTilingCode(op_name, model_info_list, generator_config), "Get tiling func failed.");
    return true;
  } catch (const af::AscIRException &e) {
    GELOGE(af::FAILED, "Gen tiling failed, exception:%d", static_cast<int32_t>(e.GetInfo().error_code));
    return false;
  }
}

bool GenTilingImplAutoFuseV3(const std::string &op_name, const ascir::FusedScheduledResult &fused_schedule_result,
                             std::map<std::string, std::string> &options,
                             std::map<std::string, std::string> &tiling_func, bool is_inductor_scene) {
  TRACING_PERF_SCOPE(ge::TracingModule::kModelCompile, "GenTilingImpl", op_name);
  GE_ASSERT_TRUE(!fused_schedule_result.node_idx_to_scheduled_results.empty(), "fused schedule results of %s empty.",
                 op_name.c_str());
  size_t id = 0UL;
  for (const auto &schedule_result : fused_schedule_result.node_idx_to_scheduled_results) {
    GE_ASSERT_TRUE(!schedule_result.empty(), "schedule results of %s in asc graph[%zu] empty.", op_name.c_str(), id);
    GELOGI("Gen tiling for total [%zu] schedules for op [%s].",
           fused_schedule_result.node_idx_to_scheduled_results.size(), op_name.c_str());
    id++;
  }
  const auto duration_level = GetDurationLevel(options);
  DurationInitGuard duration_init_guard(duration_level);
  // 四个层级的结构，分别是：
  // asc graphs->schedule results->schedule groups->impl graphs
  std::vector<std::vector<std::vector<std::vector<af::AscGraph>>>> all_graphs_lists;
  std::map<std::string, std::string> all_graph_score_funcs;
  if (options.find(kTilingDataTypeName) == options.cend()) {
    GE_ASSERT_SUCCESS(GetAllSubImplGraphs(fused_schedule_result, all_graphs_lists, all_graph_score_funcs),
                      "Get all sub impl graphs failed of op %s", op_name.c_str());
    options[kTilingDataTypeName] = all_graphs_lists[0][0][0][0].GetName() + kDefaultTilingDataTypeName;
    GELOGD("Set tiling data type name %s", options[kTilingDataTypeName].c_str());
  }
  TilingCodeGenConfig generator_config;
  generator_config.type = GetTilingAlgorithm(options[kGenConfigType]);
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = false;
  generator_config.gen_extra_infos = false;
  generator_config.is_autofuse = true;
  generator_config.is_inductor_scene = is_inductor_scene;
  generator_config.is_cube = ascgen_utils::IsCubeFusedScheduled(fused_schedule_result);
  InitializeConfigByEnvOrIni(generator_config);
  OverridePgoConfigByOptions(generator_config, options);
  TilingCodeGenerator generator;
  FusedParsedScheduleResult fused_parsed_schedule_result;
  GE_ASSERT_SUCCESS(GetModelInfoMap(fused_schedule_result, options, fused_parsed_schedule_result));
  GE_ASSERT_SUCCESS(generator.GenTilingCode(op_name, fused_parsed_schedule_result, generator_config, tiling_func));
  GE_ASSERT_TRUE(tiling_func.find(kTilingHeadIdentify) != tiling_func.cend(), "Get tiling func failed.");
  return true;
}
}  // namespace att
