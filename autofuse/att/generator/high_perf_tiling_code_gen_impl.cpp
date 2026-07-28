/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "high_perf_tiling_code_gen_impl.h"
#include <regex>
#include "args_manager.h"
#include "solver_pass_manager.h"
#include "common/checker.h"
#include "autofuse_config/auto_fuse_config.h"

namespace att {
namespace {
constexpr ge::char_t kDefaultConfigMaxIterHeader[] = "cfg_iterations = ";
constexpr ge::char_t kDefaultConfigMaxIterValue[] = "100";
}  // namespace
af::Status HighPerfTilingCodeGenImpl::GenExternFuncDef() {
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenExternFuncDef(), "Generate extern func definition failed.");
  return af::SUCCESS;
}

af::Status HighPerfTilingCodeGenImpl::GenTilingImplPublicFunc() {
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenTilingImplPublicFunc(), "Generate tiling public func failed.");
  GE_ASSERT_SUCCESS(GenVirtualDataTransferFuncs(), "Generate virtual data transfer funcs failed.");
  return af::SUCCESS;
}

af::Status HighPerfTilingCodeGenImpl::GenToolFuncs() {
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenToolFuncs(), "Generate tool funcs.");
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenStructCopyDef(), "Generate struct copy.");
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenCacheHashMapDef(), "Generate cache hash map.");
  return af::SUCCESS;
}

af::Status HighPerfTilingCodeGenImpl::GenSolverBaseClass() {
  std::vector<ArgsManager> total_models;
  for (const auto &model_info_iter : tiling_model_info_) {
    ArgsManager args_manager(model_info_iter);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    total_models.emplace_back(args_manager);
  }
  std::string basic_solvers_head;
  std::string basic_solvers_func;
  basic_solvers_head = SolverPassManager::GenCommonBaseClassesHead(total_models);
  basic_solvers_func = SolverPassManager::GenCommonBaseClassesFunc(total_models);
  std::regex pattern(std::string(kDefaultConfigMaxIterHeader) + std::string(kDefaultConfigMaxIterValue));
  std::string result_head = std::regex_replace(
      basic_solvers_head, pattern,
      kDefaultConfigMaxIterHeader + std::to_string(AutoFuseConfig::GetAttStrategyConfig().max_iter_num));
  std::string result_func = std::regex_replace(
      basic_solvers_func, pattern,
      kDefaultConfigMaxIterHeader + std::to_string(AutoFuseConfig::GetAttStrategyConfig().max_iter_num));
  tiling_head_.AddLine(result_head);
  tiling_func_.AddLine(result_func);
  return af::SUCCESS;
}

af::Status HighPerfTilingCodeGenImpl::GenSolverTiling(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  SolverPassManager solver_pass_manager(args_manager, {args_manager.GetTilingCaseId()}, config_.tiling_data_type_name);
  tiling_func_.AddLine(solver_pass_manager.GenClassPass());
  return af::SUCCESS;
}

af::Status HighPerfTilingCodeGenImpl::GenDoTiling(const ModelInfo &model_info) {
  ArgsManager manager(model_info);
  GE_ASSERT_TRUE(manager.Process(false), "Args manager process failed.");
  SolverPassManager solver_pass_manager(manager, {manager.GetTilingCaseId()}, config_.tiling_data_type_name);
  GE_ASSERT_SUCCESS(GenGetSetTilingImpl(model_info), "Gen get set tiling impl failed, group[%s], case[%u,%s].",
                    model_info.schedule_group_ident.GetItemPrefix().c_str(), model_info.tiling_case_id,
                    model_info.sub_case_tag.c_str());
  return GenDoTilingCommon(model_info, solver_pass_manager.GenFuncPass());
}
}  // namespace att
