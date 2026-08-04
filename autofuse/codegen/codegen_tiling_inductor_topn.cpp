/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_tiling.h"
#include "codegen_tiling_data.h"

#include "common_utils.h"

namespace codegen {
using namespace ascgen_utils;

namespace {
void AppendTopnEntryInitialization(std::stringstream &ss) {
  ss << "  tiling_datas.clear();" << std::endl;
  ss << "  workspaces.clear();" << std::endl;
  ss << "  block_dims.clear();" << std::endl;
  ss << "  if (topn <= 0) {" << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"GenerateTopnSolutions failed: invalid topn.\");" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
}
}  // namespace

void TilingLib::GenReprScheduleGroupFields(std::stringstream &ss, const ascir::ScheduleGroup &sg,
                                           const std::string &field_prefix, const std::string &emit_fn,
                                           const std::string &indent, bool emit_first_arg) const {
  std::unordered_set<std::string> seen_vars;
  std::set<int64_t> q_ids;
  std::set<int64_t> b_ids;
  std::string first_arg = emit_first_arg ? ", first); first = false;" : ");";
  for (size_t gi = 0; gi < sg.impl_graphs.size(); ++gi) {
    const auto &graph = sg.impl_graphs[gi];
    for (auto size : graph.GetAllSizeVar()) {
      if (!size->expr.IsConstExpr()) {
        std::string var_name = std::string(size->expr.Str().get());
        if (seen_vars.find(var_name) == seen_vars.end()) {
          ss << indent << emit_fn << "(\"" << var_name << "\", " << field_prefix << "get_" << var_name << "()"
             << first_arg << std::endl;
          seen_vars.insert(var_name);
        }
      }
    }
    codegen::TilingData::GetTqueAndTbufId(graph, q_ids, b_ids);
    codegen::TilingData::GetTmpBufName(graph, b_ids);
    GenReprApiTilingFields(ss, graph, gi, field_prefix, emit_first_arg);
  }
  for (auto q_id : q_ids) {
    if (q_id >= 0) {
      ss << indent << emit_fn << "(\"q" << q_id << "_size\", " << field_prefix << "get_q" << q_id << "_size()"
         << first_arg << std::endl;
    }
  }
  for (auto b_id : b_ids) {
    if (b_id >= 0) {
      ss << indent << emit_fn << "(\"b" << b_id << "_size\", " << field_prefix << "get_b" << b_id << "_size()"
         << first_arg << std::endl;
    }
  }
}

void TilingLib::GenReprApiTilingFields(std::stringstream &ss, const ascir::ImplGraph &graph, size_t tiling_case_id,
                                       const std::string &field_prefix, bool top_level) const {
  const std::string indent = top_level ? "  " : "    ";
  const std::string first_flag = top_level ? "first" : "sub_first";
  for (const auto &node : graph.GetAllNodes()) {
    std::string device_type_name;
    std::string api_field_name;
    if (af::SUCCESS == GetApiTilingTypeName(node, device_type_name) &&
        af::SUCCESS == GetApiTilingFieldName(node, api_field_name)) {
      api_field_name = api_field_name + "_" + std::to_string(tiling_case_id);
      ss << indent << "{" << std::endl;
      ss << indent << "  if (!" << first_flag << ") { repr << \",\"; }" << std::endl;
      ss << indent << "  repr << std::endl << \"" << indent << "." << api_field_name << " = {\";" << std::endl;
      std::vector<std::string> api_fields;
      codegen::TilingData::GetApiTilingDataName(node, api_fields);
      bool api_first = true;
      for (const auto &af : api_fields) {
        ss << indent << "  if (!" << (api_first ? "true" : "false") << ") { repr << \",\"; }" << std::endl;
        ss << indent << "  repr << std::endl << \"" << indent << "  ." << af << " = \" << " << field_prefix
           << api_field_name << "." << af << ";" << std::endl;
        api_first = false;
      }
      ss << indent << "  repr << std::endl << \"" << indent << "}\";" << std::endl;
      ss << indent << "  " << first_flag << " = false;" << std::endl;
      ss << indent << "}" << std::endl;
    }
  }
}

std::string TilingLib::GenGetTilingDataReprFuncForInductor(const ascir::FusedScheduledResult &fused_schedule_result,
                                                           const std::string &tiling) const {
  std::stringstream ss;
  ss << "// GetTilingDataRepr returns a valid C++ designated initializer string for " << tiling << "." << std::endl;
  ss << "#pragma GCC diagnostic push" << std::endl;
  ss << "#pragma GCC diagnostic ignored \"-Wreturn-type-c-linkage\"" << std::endl;
  ss << "extern \"C\" std::string GetTilingDataRepr(const " << tiling << " *tiling_data)" << std::endl;
  ss << "{" << std::endl;
  ss << "  if (tiling_data == nullptr) {" << std::endl;
  ss << "    return std::string();" << std::endl;
  ss << "  }" << std::endl;
  ss << "  std::stringstream repr;" << std::endl;
  ss << "  repr << \"" << tiling << "{\" << std::endl;" << std::endl;
  ss << "  auto emit_field = [&](const char *name, const auto &val, bool first) {" << std::endl;
  ss << "    if (!first) { repr << \",\"; }" << std::endl;
  ss << "    repr << std::endl << \"  .\" << name << \" = \" << val;" << std::endl;
  ss << "  };" << std::endl;
  ss << "  bool first = true;" << std::endl;
  ss << "  emit_field(\"block_dim\", tiling_data->get_block_dim(), first); first = false;" << std::endl;
  ss << "  emit_field(\"corenum\", tiling_data->get_corenum(), first); first = false;" << std::endl;
  ss << "  emit_field(\"ub_size\", tiling_data->get_ub_size(), first); first = false;" << std::endl;
  ss << "  emit_field(\"hbm_size\", tiling_data->get_hbm_size(), first); first = false;" << std::endl;

  std::vector<ascir::TensorId> workspace_ids =
      ascgen_utils::GetWorkspaceTensorIdListInOneScheduleResult(fused_schedule_result);
  std::sort(workspace_ids.begin(), workspace_ids.end());
  for (auto workspace_id : workspace_ids) {
    ss << "  emit_field(\"workspace" << workspace_id << "\", tiling_data->get_workspace" << workspace_id
       << "(), first); first = false;" << std::endl;
  }

  if (ascgen_utils::IsSingleGroup(fused_schedule_result)) {
    GenReprSingleGroup(ss, fused_schedule_result);
  } else {
    GenReprMultiGroup(ss, fused_schedule_result);
  }

  ss << "  repr << std::endl << \"}\";" << std::endl;
  ss << "  return repr.str();" << std::endl;
  ss << "}" << std::endl;
  ss << "#pragma GCC diagnostic pop" << std::endl;
  return ss.str();
}

void TilingLib::GenReprSingleGroup(std::stringstream &ss,
                                   const ascir::FusedScheduledResult &fused_schedule_result) const {
  ss << "  emit_field(\"tiling_key\", tiling_data->get_tiling_key(), first); first = false;" << std::endl;
  auto &sg = fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0];
  GenReprScheduleGroupFields(ss, sg, "tiling_data->", "emit_field", "  ", true);
}

void TilingLib::GenReprMultiGroup(std::stringstream &ss,
                                  const ascir::FusedScheduledResult &fused_schedule_result) const {
  for (size_t i = 0; i < fused_schedule_result.node_idx_to_scheduled_results.size(); ++i) {
    ss << "  emit_field(\"graph" << i << "_tiling_key\", tiling_data->get_graph" << i << "_tiling_key(), first);"
       << " first = false;" << std::endl;
  }
  for (size_t i = 0; i < fused_schedule_result.node_idx_to_scheduled_results.size(); ++i) {
    const auto &scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[i];
    for (size_t j = 0; j < scheduled_results.size(); ++j) {
      const auto &schedule_groups = scheduled_results[j].schedule_groups;
      for (size_t k = 0; k < schedule_groups.size(); ++k) {
        std::string sub_name =
            "graph" + std::to_string(i) + "_result" + std::to_string(j) + "_g" + std::to_string(k) + "_tiling_data";
        ss << "  {" << std::endl;
        ss << "    if (!first) { repr << \",\"; }" << std::endl;
        ss << "    repr << std::endl << \"  ." << sub_name << " = {\";" << std::endl;
        ss << "    bool sub_first = true;" << std::endl;
        ss << "    auto emit_sub = [&](const char *name, const auto &val) {" << std::endl;
        ss << "      if (!sub_first) { repr << \",\"; }" << std::endl;
        ss << "      repr << std::endl << \"    .\" << name << \" = \" << val;" << std::endl;
        ss << "      sub_first = false;" << std::endl;
        ss << "    };" << std::endl;
        ss << "    emit_sub(\"block_dim\", tiling_data->" << sub_name << ".get_block_dim());" << std::endl;
        ss << "    emit_sub(\"corenum\", tiling_data->" << sub_name << ".get_corenum());" << std::endl;
        ss << "    emit_sub(\"ub_size\", tiling_data->" << sub_name << ".get_ub_size());" << std::endl;
        ss << "    emit_sub(\"hbm_size\", tiling_data->" << sub_name << ".get_hbm_size());" << std::endl;
        ss << "    emit_sub(\"tiling_key\", tiling_data->" << sub_name << ".get_tiling_key());" << std::endl;
        std::string field_prefix = "tiling_data->" + sub_name + ".";
        GenReprScheduleGroupFields(ss, schedule_groups[k], field_prefix, "emit_sub", "    ", false);
        ss << "    repr << std::endl << \"  }\";" << std::endl;
        ss << "    first = false;" << std::endl;
        ss << "  }" << std::endl;
      }
    }
  }
}

std::string TilingLib::GenUpdateCurPerfAndBlockByGroupHelper() const {
  return ascgen_utils::GenUpdateCurPerfAndBlockByGroupHelper(false, true);
}

std::string TilingLib::GenEvaluateModeledPerfForInductor(
    const std::string &tiling, const ::ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  bool is_single_group = ascgen_utils::IsSingleGroup(fused_schedule_result);
  ss << "static double EvaluateModeledPerf(const " << tiling << " &tiling_data) {" << std::endl;
  if (is_single_group) {
    ss << "  " << tiling << " tmp = tiling_data;" << std::endl;
    ss << "  return optiling::GetPerf(tmp);" << std::endl;
  } else {
    GenMultiGroupPerfAggregation(ss, fused_schedule_result);
  }
  ss << "}" << std::endl;
  return ss.str();
}

void TilingLib::GenMultiGroupPerfAggregation(std::stringstream &ss,
                                             const ::ascir::FusedScheduledResult &fused_schedule_result) const {
  const auto &node_results = fused_schedule_result.node_idx_to_scheduled_results;
  ss << "  double cur_perf = 0.0;" << std::endl;
  ss << "  double cur_tmp_perf = 0.0;" << std::endl;
  ss << "  uint32_t cur_block = 0;" << std::endl;
  ss << "  uint32_t limited_block = tiling_data.get_block_dim();" << std::endl;
  bool first_result = true;
  for (size_t asc_graph_id = 0; asc_graph_id < node_results.size(); ++asc_graph_id) {
    const auto &scheduled_results = node_results[asc_graph_id];
    for (size_t result_id = 0; result_id < scheduled_results.size(); ++result_id) {
      if (first_result) {
        ss << "  if (tiling_data.get_graph" << asc_graph_id << "_tiling_key() == " << result_id << ") {" << std::endl;
        first_result = false;
      } else {
        ss << "  } else if (tiling_data.get_graph" << asc_graph_id << "_tiling_key() == " << result_id << ") {"
           << std::endl;
      }
      GenGroupPerfForScheduleResult(ss, asc_graph_id, result_id, scheduled_results[result_id]);
    }
    if (!scheduled_results.empty()) {
      ss << "  }" << std::endl;
    }
  }
  ss << "  return cur_perf;" << std::endl;
}

void TilingLib::GenGroupPerfForScheduleResult(std::stringstream &ss, size_t asc_graph_id, size_t result_id,
                                              const ::ascir::ScheduledResult &sched_result) const {
  const auto &schedule_groups = sched_result.schedule_groups;
  bool enable_group_parallel = sched_result.enable_group_parallel && schedule_groups.size() > 1;
  if (schedule_groups.size() == 1 || !enable_group_parallel) {
    bool first_group = true;
    for (size_t group_id = 0; group_id < schedule_groups.size(); ++group_id) {
      std::string ns = "AscGraph" + std::to_string(asc_graph_id) + "ScheduleResult" + std::to_string(result_id) + "G" +
                       std::to_string(group_id);
      std::string item = "graph" + std::to_string(asc_graph_id) + "_result" + std::to_string(result_id) + "_g" +
                         std::to_string(group_id) + "_tiling_data";
      ss << "    { auto _tmp = tiling_data." << item << "; ";
      if (first_group) {
        ss << "cur_perf = " << ns << "::GetPerf(_tmp); }" << std::endl;
        first_group = false;
      } else {
        ss << "cur_perf += " << ns << "::GetPerf(_tmp); }" << std::endl;
      }
    }
  } else {
    bool first_group = true;
    for (size_t group_id = 0; group_id < schedule_groups.size(); ++group_id) {
      std::string ns = "AscGraph" + std::to_string(asc_graph_id) + "ScheduleResult" + std::to_string(result_id) + "G" +
                       std::to_string(group_id);
      std::string item = "graph" + std::to_string(asc_graph_id) + "_result" + std::to_string(result_id) + "_g" +
                         std::to_string(group_id) + "_tiling_data";
      if (first_group) {
        ss << "    { auto _tmp = tiling_data." << item << "; "
           << "cur_tmp_perf = " << ns << "::GetPerf(_tmp); }" << std::endl;
        ss << "    cur_block = tiling_data." << item << ".get_block_dim();" << std::endl;
        first_group = false;
      } else {
        ss << "    { auto _tmp = tiling_data." << item << "; "
           << "(void)UpdateCurPerfAndBlockByGroup({tiling_data." << item << ".get_block_dim(), " << ns
           << "::GetPerf(_tmp)}, limited_block, cur_block, cur_perf, "
           << "cur_tmp_perf); }" << std::endl;
      }
    }
    ss << "    cur_perf += cur_tmp_perf;" << std::endl;
  }
}

std::string TilingLib::GenGetTopnSolutionsFuncForInductor(const ascir::FusedScheduledResult &fused_schedule_result,
                                                          const std::string &tiling) const {
  std::stringstream ss;
  codegen::PgoShapeStringStream pgo_shape_dim;
  int symbol_value_count = 0;
  for (auto vars : fused_schedule_result.origin_vars) {
    if (!(vars.IsConstExpr())) {
      std::string var_define = std::string(vars.Str().get());
      pgo_shape_dim.shape_dim_def << "int64_t " << var_define << ", ";
      pgo_shape_dim.shape_dim_use << var_define << ", ";
      TilingSetShapeDim(pgo_shape_dim.tiling_set_shape_dim, var_define, fused_schedule_result);
      symbol_value_count++;
    }
  }

  GenTopnGetTilingFunc(ss, fused_schedule_result, tiling, symbol_value_count);
  GenGenerateTopnSolutionsEntry(ss, fused_schedule_result, tiling, pgo_shape_dim);
  return ss.str();
}

void TilingLib::GenTopnInitSearchTiling(std::stringstream &ss, const ascir::FusedScheduledResult &fused_schedule_result,
                                        const std::string &tiling, int symbol_value_count) const {
  ss << "  const ResLimit *limit = (request.res_limit == nullptr || request.res_limit->aiv_num == 0) "
     << "? &g_no_limit_res : request.res_limit;" << std::endl;
  ss << "  if (request.symbol_values.size() != " << symbol_value_count << "ULL) {" << std::endl;
  ss << "    response.error_message = \"symbol_values size mismatch\";" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << std::endl;
  ss << "  " << tiling << " search_tiling = {};" << std::endl;
  ss << "  search_tiling.set_block_dim(limit->aiv_num);" << std::endl;
  ss << "  search_tiling.set_ub_size(limit->ub_size - 256);" << std::endl;
  {
    int idx = 0;
    for (auto vars : fused_schedule_result.origin_vars) {
      if (!(vars.IsConstExpr())) {
        std::string var_define = std::string(vars.Str().get());
        ss << "  const uint32_t " << var_define << " = static_cast<uint32_t>(request.symbol_values[" << idx << "]);"
           << std::endl;
        TilingSetShapeDim(ss, var_define, fused_schedule_result, "search_tiling.");
        idx++;
      }
    }
  }
  ss << std::endl;
}

void TilingLib::GenTopnGetTilingFunc(std::stringstream &ss, const ascir::FusedScheduledResult &fused_schedule_result,
                                     const std::string &tiling, int symbol_value_count) const {
  ss << "static int64_t GetTopnCandidateSolutions(const GetTilingRequest &request, GetTilingResponse &response) {"
     << std::endl;
  ss << "  response.candidate_solutions.clear();" << std::endl;
  ss << "  response.error_message.clear();" << std::endl;
  ss << "  OP_LOGI(OP_NAME, \"GetTopnCandidateSolutions enter: topn=%ld, symbol_values.size=%zu, input_configs=%s\", "
     << "static_cast<long>(request.topn), request.symbol_values.size(), "
     << "request.input_configs == nullptr ? \"null\" : \"present\");" << std::endl;
  ss << "  if (request.topn <= 0) {" << std::endl;
  GenTopnSetFailureMessage(ss, "    ", "invalid topn");
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;

  GenTopnInitSearchTiling(ss, fused_schedule_result, tiling, symbol_value_count);
  GenTopnDefaultTiling(ss, tiling);

  ss << "  const bool internal_no_config_path = (request.input_configs == nullptr);" << std::endl;
  ss << "  const bool explicit_no_config_path = request.input_configs != nullptr && request.input_configs->size() == 1 "
     << "&& request.input_configs->front().empty();" << std::endl;
  ss << "  const bool original_config_path = internal_no_config_path || explicit_no_config_path;" << std::endl;
  ss << "  std::vector<SearchConfig> configs;" << std::endl;
  ss << "  std::vector<const SearchConfig *> config_ptrs;" << std::endl;
  ss << "  if (original_config_path) {" << std::endl;
  ss << "    config_ptrs.push_back(nullptr);" << std::endl;
  ss << "  } else {" << std::endl;
  ss << "    configs = ParseSearchConfigs(*request.input_configs);" << std::endl;
  ss << "    if (configs.empty()) {" << std::endl;
  GenTopnSetFailureMessage(ss, "      ", "invalid input configs");
  ss << "      return -1;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    config_ptrs.reserve(configs.size());" << std::endl;
  ss << "    for (const auto &cfg : configs) { config_ptrs.push_back(&cfg); }" << std::endl;
  ss << "  }" << std::endl;
  ss << std::endl;

  GenTopnSearchAndFinalChecks(ss, tiling, fused_schedule_result);
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;
  ss << std::endl;
}

void TilingLib::GenTopnSearchTilingSetup(std::stringstream &ss, const std::string &tiling,
                                         const ascir::FusedScheduledResult &fused_schedule_result) const {
  ss << "  for (const auto *cfg : config_ptrs) {" << std::endl;
  ss << "    if (cfg == nullptr) {" << std::endl;
  ss << "      OP_LOGI(OP_NAME, \"config: original tiling config\");" << std::endl;
  ss << "    } else {" << std::endl;
  ss << "      OP_LOGI(OP_NAME, \"config: ub_thresh=%.3f(enabled=%d), corenum_thresh=%.3f(enabled=%d), "
     << "multicore_ub_tradeoff=%d\", cfg->ub_threshold, cfg->ub_threshold_enabled, "
     << "cfg->corenum_threshold, cfg->corenum_threshold_enabled, cfg->enable_multicore_ub_tradeoff);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    std::vector<AutofuseTilingDataPerf> raw_candidates;" << std::endl;
  ss << "    " << tiling << " cur_search_tiling = search_tiling;" << std::endl;
  ss << "    double best_perf = DBL_MAX;" << std::endl;
  ss << "    bool helper_ret = false;" << std::endl;
  const bool is_single_group = ascgen_utils::IsSingleGroup(fused_schedule_result);
  if (is_single_group) {
    ss << "    std::unordered_map<int64_t, uint64_t> workspace_map;" << std::endl;
  }
  GenTopnSearchTilingKeyCall(ss, fused_schedule_result, "cfg");
  ss << "    if (!helper_ret) {" << std::endl;
  ss << "      ++failed_config_count;" << std::endl;
  ss << "      response.error_message = \"PGOSearchTilingKey failed for topn config\";" << std::endl;
  ss << "      OP_LOGW(OP_NAME, \"PGOSearchTilingKey failed for topn config, failed=%zu/%zu.\", "
     << "failed_config_count, config_ptrs.size());" << std::endl;
  ss << "      continue;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    OP_LOGI(OP_NAME, \"PGOSearchTilingKey returned %zu raw_candidates, best_perf=%.6f\", "
     << "raw_candidates.size(), best_perf);" << std::endl;
  ss << "    if (raw_candidates.empty()) {" << std::endl;
  ss << "      response.error_message = \"PGOSearchTilingKey returned no raw candidate\";" << std::endl;
  ss << "      OP_LOGW(OP_NAME, \"PGOSearchTilingKey returned no raw candidate for topn config.\");" << std::endl;
  ss << "      continue;" << std::endl;
  ss << "    }" << std::endl;
}

void TilingLib::GenTopnCollectCandidates(std::stringstream &ss, const std::string &tiling) const {
  (void)tiling;
  ss << "    for (const auto &raw_candidate : raw_candidates) {" << std::endl;
  ss << "      CandidateSolution solution;" << std::endl;
  ss << "      solution.tiling_data = raw_candidate.tiling_data;" << std::endl;
  ss << "      solution.canonical_repr = GetTilingDataRepr(&raw_candidate.tiling_data);" << std::endl;
  ss << "      if (solution.canonical_repr.empty()) { continue; }" << std::endl;
  ss << "      double final_modeled_perf = EvaluateModeledPerf(raw_candidate.tiling_data);" << std::endl;
  ss << "      if (!std::isfinite(final_modeled_perf)) { final_modeled_perf = DBL_MAX; }" << std::endl;
  ss << "      solution.modeled_perf = final_modeled_perf;" << std::endl;
  ss << "      solution.is_default = !default_repr.empty() && (solution.canonical_repr == default_repr);" << std::endl;
  ss << "      if (solution.is_default) { found_default_candidate = true; }" << std::endl;
  ss << "      OP_LOGI(OP_NAME, \"candidate: repr=%s perf=%.6f is_default=%d\", "
     << "solution.canonical_repr.c_str(), solution.modeled_perf, solution.is_default);" << std::endl;
  ss << "      response.candidate_solutions.push_back(solution);" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (!default_repr.empty() && !found_default_candidate) {" << std::endl;
  ss << "    CandidateSolution default_solution;" << std::endl;
  ss << "    default_solution.tiling_data = default_tiling;" << std::endl;
  ss << "    default_solution.canonical_repr = default_repr;" << std::endl;
  ss << "    default_solution.modeled_perf = DBL_MAX;" << std::endl;
  ss << "    default_solution.is_default = true;" << std::endl;
  ss << "    found_default_candidate = true;" << std::endl;
  ss << "    response.candidate_solutions.push_back(default_solution);" << std::endl;
  ss << "  }" << std::endl;
  ss << std::endl;
}

void TilingLib::GenTopnSearchTilingKeyCall(std::stringstream &ss,
                                           const ascir::FusedScheduledResult &fused_schedule_result,
                                           const std::string &search_cfg) const {
  ss << "    helper_ret = optiling::PGOSearchTilingKey(raw_candidates, cur_search_tiling, -1, &cur_search_tiling, ";
  ss << "nullptr, ";
  const bool is_single_group = ascgen_utils::IsSingleGroup(fused_schedule_result);
  if (is_single_group) {
    ss << "nullptr, 0, best_perf, workspace_map, {}, " << search_cfg << ");" << std::endl;
  } else {
    ss << "nullptr, 0, best_perf, " << search_cfg << ");" << std::endl;
  }
}

void TilingLib::GenTopnSetFailureMessage(std::stringstream &ss, const std::string &indent,
                                         const std::string &reason) const {
  ss << indent << "response.error_message = \"" << reason << "\";" << std::endl;
}

void TilingLib::GenTopnDefaultTiling(std::stringstream &ss, const std::string &tiling) const {
  ss << "  std::string default_repr;" << std::endl;
  ss << "  bool found_default_candidate = false;" << std::endl;
  ss << "  " << tiling << " default_tiling = search_tiling;" << std::endl;
  ss << "  if (GetTiling(default_tiling, -1)) {" << std::endl;
  ss << "    default_repr = GetTilingDataRepr(&default_tiling);" << std::endl;
  ss << "  } else {" << std::endl;
  ss << "    OP_LOGW(OP_NAME, \"GetTiling failed for default topn config.\");" << std::endl;
  ss << "    response.error_message = \"GetTiling failed for default topn config\";" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenTopnSearchAndFinalChecks(std::stringstream &ss, const std::string &tiling,
                                            const ascir::FusedScheduledResult &fused_schedule_result) const {
  ss << "  PgoConfig::Instance().ResetRuntimeOverrides();" << std::endl;
  ss << "  size_t failed_config_count = 0U;" << std::endl;
  GenTopnSearchTilingSetup(ss, tiling, fused_schedule_result);
  GenTopnCollectCandidates(ss, tiling);
  ss << "  if (!found_default_candidate) {" << std::endl;
  ss << "    if (response.error_message.empty()) {" << std::endl;
  GenTopnSetFailureMessage(ss, "      ", "default topn candidate not found");
  ss << "    }" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  OP_LOGI(OP_NAME, \"GetTopnCandidateSolutions collected %zu candidates\", "
        "response.candidate_solutions.size());"
     << std::endl;
  ss << "  if (response.candidate_solutions.empty()) {" << std::endl;
  ss << "    if (response.error_message.empty()) {" << std::endl;
  GenTopnSetFailureMessage(ss, "      ", "no topn candidate solution found");
  ss << "    }" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenGenerateTopnSolutionsEntry(std::stringstream &ss,
                                              const ascir::FusedScheduledResult &fused_schedule_result,
                                              const std::string &tiling,
                                              const codegen::PgoShapeStringStream &pgo_shape_dim) const {
  ss << "extern \"C\" int64_t GenerateTopnSolutions(";
  ss << pgo_shape_dim.shape_dim_def.str();
  ss << "const std::vector<std::map<std::string, std::string>> &input_configs, int64_t topn, ";
  ss << "std::vector<" << tiling << "> &tiling_datas, std::vector<int64_t> &workspaces, ";
  ss << "std::vector<int64_t> &block_dims, ResLimit *res_limit = nullptr)" << std::endl;
  ss << "{" << std::endl;
  AppendTopnEntryInitialization(ss);
  ss << "  OP_LOGI(OP_NAME, \"GenerateTopnSolutions enter: topn=%ld, input_configs.size=%zu\", "
     << "static_cast<long>(topn), input_configs.size());" << std::endl;
  ss << "  GetTilingRequest request;" << std::endl;
  (void)fused_schedule_result;  // symbol_values already captured in pgo_shape_dim
  ss << "  request.symbol_values = {" << pgo_shape_dim.shape_dim_use.str() << "};" << std::endl;
  ss << "  if (input_configs.empty()) {" << std::endl;
  ss << "    request.input_configs = nullptr;" << std::endl;
  ss << "  } else {" << std::endl;
  ss << "    request.input_configs = &input_configs;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  request.res_limit = res_limit;" << std::endl;
  ss << "  request.topn = topn;" << std::endl;
  ss << "  GetTilingResponse response;" << std::endl;
  ss << "  if (GetTopnCandidateSolutions(request, response) != 0) {" << std::endl;
  ss << "    if (response.error_message.empty()) { response.error_message = \"unknown topn candidate generation "
        "failure\"; }"
     << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"GenerateTopnSolutions failed: %s\", response.error_message.c_str());" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  SelectTopnCandidateSolutions(response.candidate_solutions, topn);" << std::endl;
  ss << "  if (response.candidate_solutions.empty()) {" << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"GenerateTopnSolutions failed: no candidate after topn selection.\");" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  OP_LOGI(OP_NAME, \"SelectTopn: %zu solutions after dedup+sort+truncate (topn=%ld)\", "
     << "response.candidate_solutions.size(), static_cast<long>(topn));" << std::endl;
  ss << "  for (const auto &sol : response.candidate_solutions) {" << std::endl;
  ss << "    tiling_datas.push_back(sol.tiling_data);" << std::endl;
  ss << "    workspaces.push_back(static_cast<int64_t>(GetWorkspaceSize(sol.tiling_data)));" << std::endl;
  ss << "    block_dims.push_back(static_cast<int64_t>(sol.tiling_data.get_block_dim()));" << std::endl;
  ss << "    OP_LOGI(OP_NAME, \"output[%zu]: perf=%.6f is_default=%d block_dim=%ld repr=%s\", "
     << "tiling_datas.size() - 1, sol.modeled_perf, sol.is_default, "
     << "static_cast<long>(sol.tiling_data.get_block_dim()), sol.canonical_repr.c_str());" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;
}

std::string TilingLib::GenCandidateSolutionProtocolForInductor(const std::string &tiling) const {
  std::stringstream ss;
  ss << "// Candidate solution protocol for Inductor topn selection." << std::endl;
  ss << "struct CandidateSolution {" << std::endl;
  ss << "  " << tiling << " tiling_data;" << std::endl;
  ss << "  double modeled_perf = 0.0;" << std::endl;
  ss << "  bool is_default = false;" << std::endl;
  ss << "  std::string canonical_repr;" << std::endl;
  ss << "};" << std::endl;
  ss << std::endl;
  ss << "struct GetTilingRequest {" << std::endl;
  ss << "  std::vector<int64_t> symbol_values;" << std::endl;
  ss << "  const std::vector<std::map<std::string, std::string>> *input_configs = nullptr;" << std::endl;
  ss << "  ResLimit *res_limit = nullptr;" << std::endl;
  ss << "  int64_t topn = 1;" << std::endl;
  ss << "};" << std::endl;
  ss << std::endl;
  ss << "struct GetTilingResponse {" << std::endl;
  ss << "  std::vector<CandidateSolution> candidate_solutions;" << std::endl;
  ss << "  std::string error_message;" << std::endl;
  ss << "};" << std::endl;
  ss << std::endl;
  return ss.str();
}

void TilingLib::GenDeduplicateCandidateSolutions(std::stringstream &ss) const {
  ss << "inline void DeduplicateCandidateSolutions(std::vector<CandidateSolution> &solutions) {" << std::endl;
  ss << "  std::unordered_map<std::string, size_t> repr_to_index;" << std::endl;
  ss << "  std::vector<CandidateSolution> deduplicated;" << std::endl;
  ss << "  deduplicated.reserve(solutions.size());" << std::endl;
  ss << "  for (const auto &solution : solutions) {" << std::endl;
  ss << "    if (solution.canonical_repr.empty()) { continue; }" << std::endl;
  ss << "    const auto iter = repr_to_index.find(solution.canonical_repr);" << std::endl;
  ss << "    if (iter == repr_to_index.end()) {" << std::endl;
  ss << "      repr_to_index.emplace(solution.canonical_repr, deduplicated.size());" << std::endl;
  ss << "      deduplicated.push_back(solution);" << std::endl;
  ss << "      continue;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    auto &kept = deduplicated[iter->second];" << std::endl;
  ss << "    if (!(std::fabs(kept.modeled_perf - solution.modeled_perf) < 1e-8)) {" << std::endl;
  ss << "      OP_LOGW(OP_NAME, \"same repr with different modeled_perf, keep first: kept=%.6f, current=%.6f, "
        "repr=%s\", "
     << "kept.modeled_perf, solution.modeled_perf, solution.canonical_repr.c_str());" << std::endl;
  ss << "      continue;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    if (!kept.is_default && solution.is_default) {" << std::endl;
  ss << "      kept = solution;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  solutions.swap(deduplicated);" << std::endl;
  ss << "}" << std::endl;
  ss << std::endl;
}

std::string TilingLib::GenTopnSelectorHelpersForInductor() const {
  std::stringstream ss;
  ss << "// Topn selector helpers: default-first, modeled_perf ascending, canonical_repr tiebreak." << std::endl;
  ss << "inline bool CompareCandidateSolution(const CandidateSolution &lhs, const CandidateSolution &rhs) {"
     << std::endl;
  ss << "  if (lhs.is_default != rhs.is_default) { return lhs.is_default; }" << std::endl;
  ss << "  if (lhs.modeled_perf < rhs.modeled_perf || rhs.modeled_perf < lhs.modeled_perf) { return lhs.modeled_perf < "
        "rhs.modeled_perf; }"
     << std::endl;
  ss << "  return lhs.canonical_repr < rhs.canonical_repr;" << std::endl;
  ss << "}" << std::endl;
  ss << std::endl;
  GenDeduplicateCandidateSolutions(ss);
  ss << "inline void SelectTopnCandidateSolutions(std::vector<CandidateSolution> &solutions, int64_t topn) {"
     << std::endl;
  ss << "  const size_t before_dedup = solutions.size();" << std::endl;
  ss << "  DeduplicateCandidateSolutions(solutions);" << std::endl;
  ss << "  OP_LOGI(OP_NAME, \"DeduplicateCandidateSolutions: %zu -> %zu\", before_dedup, solutions.size());"
     << std::endl;
  ss << "  std::sort(solutions.begin(), solutions.end(), CompareCandidateSolution);" << std::endl;
  ss << "  for (size_t i = 0; i < solutions.size(); ++i) {" << std::endl;
  ss << "    OP_LOGI(OP_NAME, \"sorted[%zu]: perf=%.6f is_default=%d repr_len=%zu\", "
     << "i, solutions[i].modeled_perf, solutions[i].is_default, solutions[i].canonical_repr.size());" << std::endl;
  ss << "    const std::string &repr = solutions[i].canonical_repr;" << std::endl;
  ss << "    const size_t chunk = 800;" << std::endl;
  ss << "    for (size_t off = 0; off < repr.size(); off += chunk) {" << std::endl;
  ss << "      OP_LOGI(OP_NAME, \"  repr[%zu..%zu]: %.*s\", off, std::min(off + chunk, repr.size()), "
     << "static_cast<int>(std::min(chunk, repr.size() - off)), repr.c_str() + off);" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (topn > 0 && static_cast<int64_t>(solutions.size()) > topn) {" << std::endl;
  ss << "    OP_LOGI(OP_NAME, \"truncate %zu -> %ld\", solutions.size(), static_cast<long>(topn));" << std::endl;
  ss << "    solutions.resize(static_cast<size_t>(topn));" << std::endl;
  ss << "  }" << std::endl;
  ss << "}" << std::endl;
  ss << std::endl;
  return ss.str();
}

std::string TilingLib::GenSearchConfigProtocolForInductor() const {
  std::stringstream ss;
  ss << "// SearchConfig for dual-path PGO: TF builtin and Inductor request configs." << std::endl;
  ss << "struct SearchConfig {" << std::endl;
  ss << "  bool ub_threshold_enabled = false;" << std::endl;
  ss << "  double ub_threshold = 0.0;" << std::endl;
  ss << "  bool corenum_threshold_enabled = false;" << std::endl;
  ss << "  double corenum_threshold = 1.0;" << std::endl;
  ss << "  bool enable_multicore_ub_tradeoff = false;" << std::endl;
  ss << "};" << std::endl;
  ss << std::endl;
  return ss.str();
}

std::string TilingLib::GenBuiltinTfPgoConfigsForInductor() const {
  std::stringstream ss;
  ss << "// Builtin TF PGO search configs: 5 fixed threshold configurations." << std::endl;
  ss << "inline std::vector<SearchConfig> GetBuiltinTfPgoConfigs() {" << std::endl;
  ss << "  return {" << std::endl;
  ss << "    {true, 0.2, true, 0.4, false}," << std::endl;
  ss << "    {true, 0.1, true, 0.4, false}," << std::endl;
  ss << "    {true, 0.0, true, 1.0, false}," << std::endl;
  ss << "    {true, 0.05, true, 1.0, false}," << std::endl;
  ss << "    {true, 0.1, true, 0.8, false}," << std::endl;
  ss << "  };" << std::endl;
  ss << "}" << std::endl;
  ss << std::endl;
  return ss.str();
}

std::string TilingLib::GenInductorConfigParserForInductor() const {
  std::stringstream ss;
  ss << "// Parse Inductor request configs from interface input." << std::endl;
  ss << "constexpr double kMinUbThreshold = 0.001;" << std::endl;
  ss << "inline bool ParseSearchConfig(const std::map<std::string, std::string> &raw, SearchConfig &out) {"
     << std::endl;
  ss << "  out = SearchConfig();" << std::endl;
  ss << "  auto ub_it = raw.find(\"ub_threshold\");" << std::endl;
  ss << "  if (ub_it != raw.end()) {" << std::endl;
  ss << "    out.ub_threshold_enabled = true;" << std::endl;
  ss << "    try { out.ub_threshold = std::stod(ub_it->second); } catch (...) { return false; }" << std::endl;
  ss << "    if (std::fabs(out.ub_threshold) < 1e-8) { out.ub_threshold = kMinUbThreshold; }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  auto cn_it = raw.find(\"corenum_threshold\");" << std::endl;
  ss << "  if (cn_it != raw.end()) {" << std::endl;
  ss << "    out.corenum_threshold_enabled = true;" << std::endl;
  ss << "    try { out.corenum_threshold = std::stod(cn_it->second); } catch (...) { return false; }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  auto mc_it = raw.find(\"enable_multicore_ub_tradeoff\");" << std::endl;
  ss << "  if (mc_it != raw.end()) {" << std::endl;
  ss << "    if (mc_it->second == \"true\") { out.enable_multicore_ub_tradeoff = true; }" << std::endl;
  ss << "    else if (mc_it->second == \"false\") { out.enable_multicore_ub_tradeoff = false; }" << std::endl;
  ss << "    else { return false; }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  for (const auto &kv : raw) {" << std::endl;
  ss << "    if (kv.first != \"ub_threshold\" && kv.first != \"corenum_threshold\"" << std::endl;
  ss << "        && kv.first != \"enable_multicore_ub_tradeoff\") { return false; }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return true;" << std::endl;
  ss << "}" << std::endl;
  ss << std::endl;
  ss << "inline std::vector<SearchConfig> ParseSearchConfigs(" << std::endl;
  ss << "    const std::vector<std::map<std::string, std::string>> &raws) {" << std::endl;
  ss << "  std::vector<SearchConfig> result;" << std::endl;
  ss << "  for (const auto &raw : raws) {" << std::endl;
  ss << "    SearchConfig cfg;" << std::endl;
  ss << "    if (!ParseSearchConfig(raw, cfg)) { return {};" << std::endl;
  ss << "    }" << std::endl;
  ss << "    result.push_back(cfg);" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return result;" << std::endl;
  ss << "}" << std::endl;
  ss << std::endl;
  return ss.str();
}

}  // namespace codegen
