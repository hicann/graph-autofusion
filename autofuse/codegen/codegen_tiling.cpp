/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_tiling.h"
#include "codegen_tiling_data.h"

#include <algorithm>
#include <initializer_list>
#include <string>
#include <cstdlib>
#include <fstream>
#include <set>
#include <securec.h>
#include "runtime/base.h"
#include "runtime/dev.h"

#include "dlfcn.h"

#include "ascir_ops.h"
#include "ascir_ops_utils.h"

#include "common_utils.h"
#include "gen_tiling_impl.h"
#include "common/ge_common/debug/log.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "autofuse_config/auto_fuse_config.h"
#include "graph/ge_context.h"
#include "common/platform_context.h"
#include "graph/utils/type_utils.h"
#include "backend/backend_spec.h"
#include "common/ascgraph_info_complete.h"
#include "codegen_tiling_cube_wrapper.h"
#include "common/tiling_source_dependencies.h"

namespace codegen {
using optimize::AscGraphInfoComplete;
using optimize::SizeVarSet;
using namespace af::ascir_op;
using namespace ascir;
using namespace codegen;
using namespace af::ops;
using namespace ascgen_utils;
namespace {
std::string GetFullSocVersionForCubeTiling(const std::string &fallback_soc_version) {
  constexpr uint32_t kSocVersionLen = 64U;
  char soc_version[kSocVersionLen] = {};
  auto res = rtGetSocSpec("version", "SoC_version", soc_version, kSocVersionLen);
  if (res == RT_ERROR_NONE && soc_version[0] != '\0') {
    return std::string(soc_version);
  }
  GELOGW("Failed to get full SoC_version for cube tiling, fallback to %s", fallback_soc_version.c_str());
  return fallback_soc_version;
}

std::string GetDeviceIdForCubeTiling() {
  return std::to_string(af::GetContext().DeviceId());
}

bool CheckTilingHeadersValid(const std::map<std::string, std::string> &tiling_file_name_to_content) {
  for (const auto &pair : tiling_file_name_to_content) {
    if (pair.second == INVALID_TILING) {
      GELOGE(af::FAILED, "tilings(%s) is invalid", pair.first.c_str());
      return false;
    }
  }
  return true;
}

void RequireSystemHeaders(autofuse::SourceDependencies &dependencies, std::initializer_list<const char *> headers) {
  for (const auto *header : headers) {
    autofuse::RequireSystemHeader(dependencies, header);
  }
}

void RequireEntrySystemHeaders(autofuse::SourceDependencies &dependencies, bool is_inductor, bool is_cv,
                               bool is_multi_group) {
  if (is_inductor && is_cv) {
    RequireSystemHeaders(dependencies, {"algorithm", "cfloat", "cstddef", "cstdint", "cstring", "ostream", "sstream",
                                        "string", "vector"});
  } else if (is_inductor) {
    RequireSystemHeaders(dependencies, {"algorithm", "cfloat", "cmath", "cstddef", "cstdint", "map", "ostream",
                                        "sstream", "string", "unordered_map", "vector"});
  } else {
    RequireSystemHeaders(dependencies, {"algorithm", "cfloat", "cmath", "cstddef", "cstdint", "cstdlib", "map",
                                        "ostream", "sstream", "string", "unordered_map", "vector"});
  }
  if (is_multi_group) {
    autofuse::RequireSystemHeader(dependencies, "utility");
  }
}

bool ImplGraphHasWorkspace(const ascir::ImplGraph &graph) {
  auto nodes = graph.GetAllNodes();
  for (const auto &node : nodes) {
    if (IsOps<Workspace>(node)) {
      return true;
    }
  }
  return false;
}

bool ScheduleGroupHasWorkspace(const ascir::ScheduleGroup &group) {
  return std::any_of(group.impl_graphs.begin(), group.impl_graphs.end(), ImplGraphHasWorkspace);
}

bool ScheduledResultHasWorkspace(const ascir::ScheduledResult &result) {
  return std::any_of(result.schedule_groups.begin(), result.schedule_groups.end(), ScheduleGroupHasWorkspace);
}

bool EntryWorkspaceUsesSolver(const ascir::FusedScheduledResult &fused_schedule_result) {
  for (const auto &scheduled_results : fused_schedule_result.node_idx_to_scheduled_results) {
    if (std::any_of(scheduled_results.begin(), scheduled_results.end(), ScheduledResultHasWorkspace)) {
      return true;
    }
  }
  return false;
}

struct EntryTranslationUnitOptions {
  bool is_inductor;
  bool is_cv;
  bool include_pgo;
  bool enable_pgo_runtime;
  bool include_solver;
  bool is_multi_group;
  bool include_cube_wrapper = false;
};

EntryTranslationUnitOptions GetInductorEntryTranslationUnitOptions(
    const ascir::FusedScheduledResult &fused_schedule_result, bool is_cv, bool enable_pgo_runtime) {
  return {true,
          is_cv,
          true,
          enable_pgo_runtime,
          enable_pgo_runtime || EntryWorkspaceUsesSolver(fused_schedule_result),
          !ascgen_utils::IsSingleGroup(fused_schedule_result),
          is_cv};
}

std::string RenderEntryTranslationUnit(const std::string &body, const EntryTranslationUnitOptions &options) {
  autofuse::GeneratedCode code;
  code.body = "using namespace optiling;\n\n" + body;
  RequireEntrySystemHeaders(code.dependencies, options.is_inductor, options.is_cv, options.is_multi_group);
  autofuse::RequireGeneratedHeader(code.dependencies, autofuse::GeneratedHeaderId::kTilingData);
  autofuse::RequireGeneratedHeader(code.dependencies, autofuse::GeneratedHeaderId::kLog);
  if (options.include_pgo) {
    autofuse::RequireGeneratedHeader(code.dependencies, autofuse::GeneratedHeaderId::kPgo);
  }
  if (options.enable_pgo_runtime) {
    RequireSystemHeaders(code.dependencies, {"fstream", "securec.h", "unordered_set"});
  }
  if (options.include_solver) {
    autofuse::RequireGeneratedHeader(code.dependencies, autofuse::GeneratedHeaderId::kSolver);
  }
  autofuse::RequireGeneratedHeader(code.dependencies, autofuse::GeneratedHeaderId::kApi);
  if (!options.is_inductor) {
    autofuse::RequireExternalHeaderUnlessCceKtTest(code.dependencies, "exe_graph/runtime/tiling_context.h");
    autofuse::RequireExternalHeaderUnlessCceKtTest(code.dependencies, "tiling/platform/platform_ascendc.h");
  }
  if (options.is_cv && !options.is_inductor) {
    autofuse::RequireExternalHeader(code.dependencies, "autofuse_cube_tiling_data.h");
  }
  if (options.include_cube_wrapper) {
    autofuse::RequireExternalHeader(code.dependencies, "cube_kernel_tiling_wrapper.h");
  }
  std::string output;
  GE_ASSERT_SUCCESS(autofuse::RenderTranslationUnit(code, output));
  return output;
}

void AddFallbackHeader(std::map<std::string, std::string> &headers, const std::string &key, const std::string &guard,
                       autofuse::GeneratedCode code) {
  std::string output;
  if (autofuse::RenderGeneratedHeader(code, guard, output) != af::SUCCESS) {
    return;
  }
  headers.emplace(key, std::move(output));
}

std::string GetFallbackSolverMacros() {
  return "#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))\n"
         "#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))\n"
         "#define Abs(a) ((double)(a) >= 0 ? (a) : -(a))\n"
         "#define Log(a) (log((double)(a)))\n"
         "#define Pow(a, b) pow(a, b)\n"
         "#define Rational(a, b) ((double)(a) / (double)(b))\n"
         "#define ExpectEq(a, b) ((a) == (b))\n"
         "#define ExpectNe(a, b) ((a) != (b))\n"
         "#define ExpectLe(a, b) ((a) <= (b))\n"
         "#define ExpectLt(a, b) ((a) < (b))\n"
         "#define LogicAnd(a, b) ((a) && (b))\n"
         "#define LogicOr(a, b) ((a) || (b))\n"
         "#define True true\n#define False false\n#define MAX_SOLUTION 50\n";
}

std::string GetFallbackSolverFunctions() {
  return R"(namespace optiling {
inline bool IsEqual(double a, double b) {
  constexpr double kEpsilon = 1e-8;
  double abs = (a > b) ? (a - b) : (b - a);
  return abs < kEpsilon;
}
template <typename T1, typename T2>
inline double TernaryOp(bool cond, T1 a, T2 b) {
  return static_cast<double>(cond ? a : b);
}
template <typename T>
inline T Ceiling(T a) {
  T value = static_cast<T>(static_cast<int64_t>(a));
  return IsEqual(value, a) ? value : (value + 1);
}
template <typename T>
inline T Floor(T a) {
  return static_cast<T>(static_cast<int64_t>(a));
}
template <typename T1, typename T2>
inline auto Mod(T1 a, T2 b) -> decltype(a % b) {
  return a % b;
}
template <typename T1, typename T2>
inline auto Mod(T1 a, T2 b) -> typename std::enable_if<std::is_floating_point<T1>::value ||
                                                       std::is_floating_point<T2>::value,
                                                   decltype(std::fmod(a, b))>::type {
  return std::fmod(a, b);
}
template <typename TI, typename TO>
inline TO &RefToRef(TI &value) {
  return *(reinterpret_cast<TO *>(reinterpret_cast<void *>(&value)));
}
}  // namespace optiling
)";
}

void EnsureFallbackAtomicHeaders(std::map<std::string, std::string> &headers, const std::string &pgo_body,
                                 const std::string &api_body) {
  if (headers.find(kTilingStateHeaderIdentify) != headers.end()) {
    return;
  }
  autofuse::GeneratedCode state;
  state.body = "namespace optiling {}\n";
  AddFallbackHeader(headers, kTilingStateHeaderIdentify, "__AUTOFUSE_TILING_FUNC_STATE_H__", std::move(state));

  autofuse::GeneratedCode log;
  log.body =
      "#define OP_LOGD(name, fmt, ...)\n#define OP_LOGI(name, fmt, ...)\n"
      "#define OP_LOGW(name, fmt, ...)\n#define OP_LOGE(name, fmt, ...)\n#define OP_NAME \"Autofuse\"\n";
  AddFallbackHeader(headers, kTilingLogHeaderIdentify, "__AUTOFUSE_TILING_FUNC_LOG_H__", std::move(log));

  autofuse::GeneratedCode solver;
  for (const auto &header : {"cmath", "cstdint", "type_traits"}) {
    autofuse::RequireSystemHeader(solver.dependencies, header);
  }
  solver.body = GetFallbackSolverMacros() + GetFallbackSolverFunctions();
  AddFallbackHeader(headers, kTilingSolverHeaderIdentify, "__AUTOFUSE_TILING_FUNC_SOLVER_H__", std::move(solver));

  autofuse::GeneratedCode api;
  autofuse::RequireSystemHeader(api.dependencies, "cstdint");
  if (!pgo_body.empty()) {
    autofuse::RequireSystemHeader(api.dependencies, "unordered_map");
    autofuse::RequireSystemHeader(api.dependencies, "vector");
  }
  api.body =
      "struct AutofuseTilingData;\nstruct AutofuseTilingDataPerf;\nstruct PgoTensorArgs;\n"
      "namespace optiling {\nstruct SearchConfig;\n" +
      api_body + "}  // namespace optiling\n";
  AddFallbackHeader(headers, kTilingApiHeaderIdentify, "__AUTOFUSE_TILING_FUNC_API_H__", std::move(api));
  if (!pgo_body.empty()) {
    autofuse::GeneratedCode pgo;
    for (const auto &header : {"array", "cstddef", "cstdint", "vector"}) {
      autofuse::RequireSystemHeader(pgo.dependencies, header);
    }
    pgo.body = "struct AutofuseTilingData;\nstruct AutofuseTilingDataPerf;\n" + pgo_body;
    AddFallbackHeader(headers, kTilingPgoHeaderIdentify, "__AUTOFUSE_TILING_FUNC_PGO_H__", std::move(pgo));
  }
}

void AddCvDeclarationsToApiHeader(std::map<std::string, std::string> &headers) {
  auto iter = headers.find(kTilingApiHeaderIdentify);
  if (iter == headers.end()) {
    return;
  }
  const auto guard_end = iter->second.rfind("#endif");
  if (guard_end == std::string::npos) {
    return;
  }
  iter->second.insert(guard_end,
                      "int32_t get_g_basen_basem_align();\nvoid set_g_basen_basem_align(int32_t value);\n\n");
}

void AppendCvSafetyAivOnlyModeDef(std::stringstream &ss, bool is_batch) {
  ss << "static inline bool is_cv_safety_aiv_only_mode(int64_t tiling_key) {" << std::endl;
  ss << "    switch (tiling_key) {" << std::endl;
  if (is_batch) {
    for (auto key : {513, 8192}) {
      ss << "      case " << key << ":" << std::endl;
    }
  } else {
    for (auto key : {8192, 8208, 8256, 8272}) {
      ss << "      case " << key << ":" << std::endl;
    }
  }
  ss << "        return true;" << std::endl;
  ss << "      default:" << std::endl;
  ss << "        return false;" << std::endl;
  ss << "    }" << std::endl;
  ss << "}" << std::endl;
}

void AppendCvSafetyBlockIdxScheduledModeDef(std::stringstream &ss, bool is_batch) {
  ss << "static inline bool is_cv_safety_blockidx_scheduled_mode(int64_t tiling_key) {" << std::endl;
  ss << "    switch (tiling_key) {" << std::endl;
  if (is_batch) {
    for (auto key : {257, 273, 321, 337, 769, 785, 833, 849, 2097409, 2097425, 2097473, 2097489}) {
      ss << "      case " << key << ":" << std::endl;
    }
  }
  ss << "        return true;" << std::endl;
  ss << "      default:" << std::endl;
  ss << "        return false;" << std::endl;
  ss << "    }" << std::endl;
  ss << "}" << std::endl;
}

void AppendCvSafetyMixModeDef(std::stringstream &ss, bool is_batch) {
  ss << "static inline bool is_cv_safety_mix_mode(int64_t tiling_key) {" << std::endl;
  ss << "    switch (tiling_key) {" << std::endl;
  if (is_batch) {
    for (auto key :
         {0,    16,   64,    80,    256,   272,   320,    336,    4096,   4097,   4112,    4113,    4160,    4161,
          4176, 4177, 1,     17,    65,    81,    257,    273,    321,    337,    769,     785,     833,     849,
          513,  8192, 65537, 65553, 65601, 65617, 131073, 131089, 131137, 131153, 2097409, 2097425, 2097473, 2097489}) {
      ss << "      case " << key << ":" << std::endl;
    }
  } else {
    for (auto key :
         {4096,    4097,    4112,    4113,    4160,    4161,    4176,    4177,    8192,    8208,    8256,    8272,
          24577,   24593,   24641,   24657,   28673,   28689,   28737,   28753,   1048576, 1048577, 1048656, 1048657,
          1114113, 1114129, 1114177, 1114193, 1179649, 1179665, 1179713, 1179729, 2097152, 2097153, 2097168, 2097169,
          2097216, 2097217, 2097232, 2097233, 2101248, 2101249, 2101264, 2101265, 2101312, 2101313, 2101328, 2101329,
          2125825, 2125841, 2125889, 2125905, 2162689, 2162705, 2162753, 2162769, 2228225, 2228241, 2228289, 2228305}) {
      ss << "      case " << key << ":" << std::endl;
    }
  }
  ss << "        return true;" << std::endl;
  ss << "      default:" << std::endl;
  ss << "        return false;" << std::endl;
  ss << "    }" << std::endl;
  ss << "}" << std::endl;
}

void AppendCvSafetyMixModeHelperDefs(std::stringstream &ss, bool is_batch) {
  AppendCvSafetyAivOnlyModeDef(ss, is_batch);
  AppendCvSafetyBlockIdxScheduledModeDef(ss, is_batch);
  AppendCvSafetyMixModeDef(ss, is_batch);
}

void AppendTilingKeyBranch(std::stringstream &ss, const std::vector<std::vector<std::string>> &per_group_conditions,
                           std::vector<std::string> &current, uint32_t depth, uint32_t &tiling_key,
                           bool &first_append) {
  if (per_group_conditions.size() == depth) {
    ss << (first_append ? "    if " : " else if ") << "(";
    first_append = false;
    for (uint32_t i = 0; i < current.size(); i++) {
      ss << current[i];
      if (i < (current.size() - 1)) {
        ss << " && ";
      }
    }
    ss << ") {" << std::endl;
    ss << "      return " << tiling_key << ";" << std::endl;
    ss << "    }";
    tiling_key++;
    return;
  }
  for (const auto &condition : per_group_conditions[depth]) {
    current.push_back(condition);
    AppendTilingKeyBranch(ss, per_group_conditions, current, depth + 1, tiling_key, first_append);
    current.pop_back();
  }
}

void GenMulGroupFindBestTilingKey(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) {
  uint32_t tiling_key = 0U;
  for (size_t graph_id = 0; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    for (size_t i = 0; i < scheduled_results.size(); i++) {
      auto schedule_groups = scheduled_results[i].schedule_groups;
      ss << (i == 0 ? "  if " : "  else if ") << "(t." << "graph" << std::to_string(graph_id)
         << "_tiling_key == " << std::to_string(i) << ") {" << std::endl;
      std::vector<std::vector<std::string>> per_group_conditions;
      for (size_t j = 0; j < schedule_groups.size(); j++) {
        std::vector<std::string> conditions;
        auto schedule_graphs = schedule_groups[j].impl_graphs;
        for (size_t k = 0; k < schedule_graphs.size(); k++) {
          std::string filed_name = CamelToLowerSneak("t.graph" + std::to_string(graph_id) + "_result" +
                                                     std::to_string(i) + "_g" + std::to_string(j) + "_tiling_data");
          auto index = std::to_string(k);
          std::string condition;
          condition.append(filed_name).append(".tiling_key").append(" == ").append(index);
          conditions.emplace_back(condition);
        }
        per_group_conditions.emplace_back(std::move(conditions));
      }
      std::vector<std::string> current;
      bool first_append = true;
      AppendTilingKeyBranch(ss, per_group_conditions, current, 0, tiling_key, first_append);
      ss << std::endl;
      ss << "  }";
    }
  }
  ss << std::endl;
}

size_t CalcTilingKeyCount(const ascir::FusedScheduledResult &result) {
  if (!ascgen_utils::CanUseTilingKey(result)) {
    return 1ULL;
  }
  size_t count = 0ULL;
  for (const auto &scheduled_results : result.node_idx_to_scheduled_results) {
    for (const auto &scheduled_result : scheduled_results) {
      size_t per_result_count = 1ULL;
      for (const auto &schedule_group : scheduled_result.schedule_groups) {
        per_result_count *= schedule_group.impl_graphs.size();
      }
      count += per_result_count;
    }
  }
  return count;
}

bool HasWorkSpaceNode(const af::AscGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (node->GetType() == "Workspace") {
      return true;
    }
  }
  return false;
}

void CodegenTilingKeyKerneType(std::stringstream &ss, const std::vector<std::vector<bool>> &per_group_conditions,
                               std::vector<bool> &current, uint32_t depth, uint32_t &tiling_key) {
  if (per_group_conditions.size() == depth) {
    bool has_workspace_node = false;
    for (const auto &workspace_node : current) {
      if (workspace_node) {
        has_workspace_node = true;
        break;
      }
    }
    std::string kernel_type = (has_workspace_node ? kKernelTaskTypeMixAIVOneZero : kKernelTaskTypeAIVOnly);
    ss << "    {" << std::to_string(tiling_key) << ",\"" << kernel_type << "\"}," << std::endl;
    tiling_key++;
    return;
  }
  for (const auto &condition : per_group_conditions[depth]) {
    current.push_back(condition);
    CodegenTilingKeyKerneType(ss, per_group_conditions, current, depth + 1, tiling_key);
    current.pop_back();
  }
}

bool IsNeedFfts() {
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_ASSERT_NOTNULL(backend_spec);
  return backend_spec->pgo_spec.need_ffts;
}

void CollectWorkspaceDenominatorSymbols(const af::Expression &expr, std::set<std::string> &denominator_symbols) {
  if (!expr.IsValid() || expr.IsConstExpr()) {
    return;
  }
  af::Expression numer;
  af::Expression denom;
  expr.AsNumerDenom(numer, denom);
  if (denom.IsValid() && !denom.IsConstExpr()) {
    for (const auto &symbol : denom.FreeSymbols()) {
      if (symbol.GetExprType() == af::ExprType::kExprVariable) {
        denominator_symbols.insert(af::SymbolicUtils::ToString(symbol));
      }
    }
  }
  af::Expression expr_copy = expr;
  for (const auto &arg : expr_copy.GetArgs()) {
    CollectWorkspaceDenominatorSymbols(arg, denominator_symbols);
  }
}

void GenWorkspaceDenominatorGuards(std::stringstream &ss, const af::Expression &expr, const std::string &indent) {
  std::set<std::string> denominator_symbols;
  CollectWorkspaceDenominatorSymbols(expr, denominator_symbols);
  for (const auto &symbol : denominator_symbols) {
    ss << indent << "if (" << symbol << " <= 0) {" << std::endl;
    ss << indent << "  OP_LOGW(OP_NAME, \"Invalid workspace denominator " << symbol << "=%lf.\", static_cast<double>("
       << symbol << "));" << std::endl;
    ss << indent << "  return ws_size;" << std::endl;
    ss << indent << "}" << std::endl;
  }
}
}  // namespace

TilingLib::TilingLib(const std::string &lib_path, const std::string &codegen_symbol_name) {
  af::GetContext().Init();
  auto ret = att::AutoFuseConfig::MutablePgoStrategyConfig().Init();
  if (ret == af::SUCCESS || ret == af::NOT_CHANGED) {
    if (att::AutoFuseConfig::GetPgoStrategyConfig().set_env_enable_autofuse_pgo) {
      enable_autofuse_pgo_ = (att::AutoFuseConfig::GetPgoStrategyConfig().enable_autofuse_pgo == "true");
    }
  } else {
    GELOGE(af::FAILED, "TilingLib function ENV init failed");
    return;
  }
  GELOGI("TilingLib lib_path:%s, symbol_name:%s", lib_path.c_str(), codegen_symbol_name.c_str());
  if (lib_path.empty() || codegen_symbol_name.empty()) {
    GELOGI("TilingLib using default att api: GenTilingImplAutoFuseV3");
    this->codegen_func_ = att::GenTilingImplAutoFuseV3;
    return;
  }

  this->codegen_func_ = nullptr;
  std::string real_lib_path;
  if (!ascgen_utils::GetRealPath(lib_path, real_lib_path)) {
    GELOGE(af::FAILED, "lib_path::%s realpath failed", lib_path.c_str());
    return;
  }
  auto handle = dlopen(real_lib_path.c_str(), RTLD_LAZY);
  GE_CHK_BOOL_EXEC(handle != nullptr, return, "TilingLib lib dlopen fail lib_path:%s", real_lib_path.c_str());

  auto func = dlsym(handle, codegen_symbol_name.c_str());
  if (func == nullptr) {
    GELOGE(af::FAILED, "TilingLib function dlsym fail symbol_name:%s", codegen_symbol_name.c_str());
    dlclose(handle);
    return;
  }

  this->codegen_func_ = reinterpret_cast<TilingLibCodegenFunc>(func);
}

std::map<std::string, std::string> TilingLib::GenerateForInductor(
    const ascir::FusedScheduledResult &fused_schedule_result) const {
  ascir::FusedScheduledResult elemwise_schedule_result = fused_schedule_result;
  const bool is_cube_fused_scheduled = ascgen_utils::IsCubeFusedScheduled(fused_schedule_result);
  if (is_cube_fused_scheduled) {
    GE_ASSERT_SUCCESS(ascgen_utils::ProcessCubeFusionResultDynamic(elemwise_schedule_result));
  }
  std::map<std::string, std::string> tiling_file_name_to_content =
      GetTilingHeaders(elemwise_schedule_result, true, is_cube_fused_scheduled);
  GE_CHK_BOOL_RET_STATUS_NOLOG(CheckTilingHeadersValid(tiling_file_name_to_content), tiling_file_name_to_content);
  std::stringstream ss;

  ss << "#pragma GCC diagnostic push\n" << "#pragma GCC diagnostic ignored \"-Wreturn-type-c-linkage\"\n";
  ss << "extern \"C\" std::string GetTilingDataRepr(const AutofuseTilingData *tiling_data);\n";
  ss << "#pragma GCC diagnostic pop\n";
  ss << TilingFuncDefForInductor(fused_schedule_result, elemwise_schedule_result) << std::endl;
  if (!is_cube_fused_scheduled) {
    ss << this->GenCandidateSolutionProtocolForInductor("AutofuseTilingData") << std::endl;
    ss << this->GenTopnSelectorHelpersForInductor() << std::endl;
    ss << this->GenBuiltinTfPgoConfigsForInductor() << std::endl;
    ss << this->GenInductorConfigParserForInductor() << std::endl;
    ss << GenGetTilingKeyCount(elemwise_schedule_result) << std::endl;
    if (!ascgen_utils::IsSingleGroup(elemwise_schedule_result)) {
      ss << GenUpdateCurPerfAndBlockByGroupHelper() << std::endl;
    }
    ss << this->GenEvaluateModeledPerfForInductor("AutofuseTilingData", elemwise_schedule_result) << std::endl;
    ss << "extern \"C\" double GetModeledPerfForTesting(const AutofuseTilingData *tiling_data) {\n"
       << "  if (tiling_data == nullptr) { return 0.0; }\n"
       << "  double modeled_perf = EvaluateModeledPerf(*tiling_data);\n"
       << "  return std::isfinite(modeled_perf) ? modeled_perf : DBL_MAX;\n"
       << "}\n"
       << std::endl;
    ss << this->GenGetTopnSolutionsFuncForInductor(elemwise_schedule_result, "AutofuseTilingData") << std::endl;
    ss << this->GenGetTilingDataReprFuncForInductor(elemwise_schedule_result, "AutofuseTilingData") << std::endl;
  }
  // 生成GenConstTilingData方法（对所有场景生成，包括CV fusion静态shape）
  ss << TilingData("Autofuse").GenerateConst(fused_schedule_result) << std::endl;
  if (is_cube_fused_scheduled) {
    tiling_file_name_to_content[kCubeKernelTilingWrapperHpp] = kCubeKernelTilingWrapperHppValue;
    tiling_file_name_to_content[kCubeKernelTilingWrapperCpp] = kCubeKernelTilingWrapperInclude;
    tiling_file_name_to_content[kCubeKernelTilingWrapperCpp] += kCubeKernelTilingWrapperCppValue;
  }
  const std::string entry_body = tiling_file_name_to_content[kTilingDefAndConstIdentify] + ss.str();
  const auto entry_options =
      GetInductorEntryTranslationUnitOptions(elemwise_schedule_result, is_cube_fused_scheduled, enable_autofuse_pgo_);
  tiling_file_name_to_content[kTilingDefAndConstIdentify] = RenderEntryTranslationUnit(entry_body, entry_options);

  return tiling_file_name_to_content;
}

std::string DtypeToStr(ge::DataType dtype) {
  const std::map<ge::DataType, const ge::char_t *> kTypeName = {
      {ge::DT_FLOAT, "float32"}, {ge::DT_FLOAT16, "float16"}, {ge::DT_BF16, "bfloat16"}, {ge::DT_INT8, "int8"},
      {ge::DT_UINT8, "uint8"},   {ge::DT_INT16, "int16"},     {ge::DT_UINT16, "uint16"}, {ge::DT_INT32, "int32"},
      {ge::DT_UINT32, "uint32"}, {ge::DT_INT64, "int64"},     {ge::DT_UINT64, "uint64"}, {ge::DT_DOUBLE, "double"}};
  const auto &type_name_iter = kTypeName.find(dtype);
  if (type_name_iter == kTypeName.end()) {
    return "unknown";
  }
  return type_name_iter->second;
}

std::string TilingLib::GenerateForPgo(const ascir::FusedScheduledResult &fused_schedule_result,
                                      const std::string &pgo_dir) const {
  // 生成PGO的头文件和函数定义
  std::stringstream ss;
  GenPgoHeaders(ss);
  // 生成PGO需要的工具函数
  GenPgoToolFunction(fused_schedule_result, pgo_dir, ss);
  // 生成PGO需要的wrapper函数
  GenPgoWrapper(fused_schedule_result, ss);
  // 生成PGO需要的求解代码
  GenPgoProfiling(fused_schedule_result, ss);
  // 生成PGO的main函数
  GenPgoMain(fused_schedule_result, ss);
  return ss.str();
}

void TilingLib::GenPgoHeaders(std::stringstream &ss) const {
  ss << "#include <cinttypes>" << std::endl;
  ss << "#include <unistd.h>" << std::endl;
  ss << "#include <fcntl.h>" << std::endl;
  ss << "#include <sys/file.h>" << std::endl;
  ss << "#include <sys/syscall.h>" << std::endl;
  ss << "#include <sys/wait.h>" << std::endl;
  ss << "#include <dlfcn.h>" << std::endl << std::endl;

  ss << "#include <algorithm>" << std::endl;
  ss << "#include <chrono>" << std::endl;
  ss << "#include <cfloat>" << std::endl;
  ss << "#include <cstdint>" << std::endl;
  ss << "#include <cerrno>" << std::endl;
  ss << "#include <cstring>" << std::endl;
  ss << "#include <securec.h>" << std::endl;
  ss << "#include <fstream>" << std::endl;
  ss << "#include <map>" << std::endl;
  ss << "#include <string>" << std::endl;
  ss << "#include <thread>" << std::endl;
  ss << "#include <unordered_map>" << std::endl;
  ss << "#include <vector>" << std::endl << std::endl;

  ss << "#include \"acl/acl.h\"" << std::endl;
  ss << "#include \"dlog_pub.h\"" << std::endl;
  ss << "#include \"mspti.h\"" << std::endl;
  ss << "#include \"tiling/platform/platform_ascendc.h\"" << std::endl << std::endl;

  ss << "#include \"autofuse_tiling_data.h\"" << std::endl << std::endl;
  ss << PGOTensorArgsDef();
}

void TilingLib::GenDynamicLibraryLoaderCode(std::stringstream &ss) const {
  ss << "static void *handle = nullptr;" << std::endl;
  ss << "static bool initialized = false;" << std::endl;
  ss << R"(
__attribute__((constructor)) void Init() {
  if (initialized) return;
  handle = dlopen(kernel_file, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    DLOGE("Failed to load %s: %s", kernel_file, dlerror());
    return;
  }
  DLOGD("Kernel api lib %s load succeed", kernel_file);
  initialized = true;
})" << std::endl;
  ss << R"(
__attribute__((destructor)) void DeInit() {
  if (handle) {
    dlclose(handle);
    handle = nullptr;
  }
  initialized = false;
})" << std::endl;
  ss << R"(
inline void *GetFunc(const char *func_name) {
  if (handle == nullptr) {
    return nullptr;
  }
  void *func = dlsym(handle, func_name);
  if (func == nullptr) {
    DLOGE("Failed to load wrapper api func: %s", dlerror());
  }
  return func;
})" << std::endl;
}

void TilingLib::GenPgoCardLock(std::stringstream &ss) const {
  ss << R"(
class CardLock {
public:
  CardLock(const char *path) {
    fd_ = open(path, O_RDWR | O_CREAT, 0666);
    if (fd_ == -1) {
      DLOGE("open lock file: %s", std::strerror(errno));
      std::exit(1);
    }
    if (flock(fd_, LOCK_EX) == -1) {
      DLOGE("flock LOCK_EX: %s", std::strerror(errno));
      std::exit(1);
    }
  }

  ~CardLock() {
    if (fd_ != -1) {
      if (flock(fd_, LOCK_UN) == -1) {
        DLOGW("flock LOCK_UN: %s", std::strerror(errno));
      }
      close(fd_);
    }
  }

  CardLock(const CardLock&) = delete;
  CardLock& operator=(const CardLock&) = delete;

private:
  int fd_{-1};
};
)" << std::endl;
}

void TilingLib::GenPgoSaveTilingKey(std::stringstream &ss) const {
  ss << R"(void PgoSaveTilingKey(const AutofuseTilingData &tiling_data, double best_perf, std::ofstream &out_file) {
  const size_t tiling_bytes = sizeof(tiling_data);
  const size_t tiling_bytes_align = (tiling_bytes + sizeof(int32_t) - 1) / sizeof(int32_t);
  std::vector<int32_t> tiling_i32(tiling_bytes_align, 0);
  memcpy_s(tiling_i32.data(), tiling_i32.size() * sizeof(int32_t), &tiling_data, tiling_bytes);
  for (size_t idx = 0; idx < tiling_i32.size(); ++idx) {
    out_file << tiling_i32[idx] << " ";
  }
  out_file << "# " << best_perf << std::endl;
})" << std::endl;
}

void TilingLib::GenPgoAppendSearchTilingData(std::stringstream &ss) const {
  GenPgoSaveTilingKey(ss);
  ss << "void AppendPgoSearchTilingData(const AutofuseTilingData &tiling_data, double best_perf, std::ios::openmode "
        "mode = std::ios::app) {"
     << std::endl;
  ss << "  DLOGD(\"AppendPgoSearchTilingData to file: %s\", search_file);" << std::endl;
  ss << "  std::ofstream out_file(search_file, mode);" << std::endl;
  ss << "  if (!out_file.is_open()) {" << std::endl;
  ss << "    DLOGE(\"Failed to open file:%s\", search_file);" << std::endl;
  ss << "    return;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  PgoSaveTilingKey(tiling_data, best_perf, out_file);" << std::endl;
  ss << "  out_file.close();" << std::endl;
  ss << std::endl;

  ss << "  int fd = ::open(search_file, O_WRONLY);" << std::endl;
  ss << "  if (fd < 0) {" << std::endl;
  ss << "    DLOGE(\"Failed to open file:%s\", search_file);" << std::endl;
  ss << "    return;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (::fsync(fd) < 0) {" << std::endl;
  ss << "    DLOGW(\"Failed to fsync file:%s\", search_file);" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ::close(fd);" << std::endl;
  ss << std::endl;
  ss << "  return;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoKernelLaunchOpArgs(const ascir::FusedScheduledResult &fused_schedule_result,
                                         std::stringstream &ss) const {
  ss << "struct AivKernelLaunchOpArgs {" << std::endl;
  ss << PGOSearchStructInputOutputDef(fused_schedule_result);
  ss << "  uint64_t workspace_addr;" << std::endl;
  ss << "  uint64_t tiling_addr;" << std::endl;
  ss << "};" << std::endl;

  ss << "struct MixKernelLaunchOpArgs {" << std::endl;
  if (IsNeedFfts()) {
    ss << "  uint64_t ffts;" << std::endl;
  }
  ss << PGOSearchStructInputOutputDef(fused_schedule_result);
  ss << "  uint64_t workspace_addr;" << std::endl;
  ss << "  uint64_t tiling_addr;" << std::endl;
  ss << "};" << std::endl;

  ss << "void *g_workspace = nullptr;" << std::endl;
}

void TilingLib::GenPgoMixTilingTable(const ascir::FusedScheduledResult &fused_schedule_result,
                                     std::stringstream &ss) const {
  for (size_t graph_id = 0U; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    const auto &scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    ss << "std::vector<uint32_t> g_mix_graph" << graph_id << "_tiling_keys = {" << std::endl;
    for (size_t result_id = 0U; result_id < scheduled_results.size(); result_id++) {
      const auto &schedule_groups = scheduled_results[result_id].schedule_groups;
      bool has_workspace_node = false;
      for (size_t group_id = 0U; group_id < schedule_groups.size() - 1U; group_id++) {
        const auto &impl_graphs = schedule_groups[group_id].impl_graphs;
        has_workspace_node = std::any_of(impl_graphs.begin(), impl_graphs.end(),
                                         [](const auto &graph) { return HasWorkSpaceNode(graph); });
      }
      if (has_workspace_node) {
        ss << "    " << result_id << "," << std::endl;
      }
    }
    ss << "};" << std::endl;
  }
}
void TilingLib::GenPgoCheckTilingIsMix(const ascir::FusedScheduledResult &fused_schedule_result,
                                       std::stringstream &ss) const {
  ss << "bool IsMixTiling(const AutofuseTilingData &t) {" << std::endl;
  ss << "  if constexpr (!g_is_mix_operator) {" << std::endl;
  ss << "    return false;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (!g_is_static_kernel) {" << std::endl;
  ss << "    return true;" << std::endl;
  ss << "  }" << std::endl;
  if (!ascgen_utils::IsSingleGroup(fused_schedule_result)) {
    for (size_t graph_id = 0U; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
      ss << "  if (!g_mix_graph" << graph_id << "_tiling_keys.empty() && std::find(g_mix_graph" << graph_id
         << "_tiling_keys.begin(), g_mix_graph" << graph_id << "_tiling_keys.end(), t.graph" << graph_id
         << "_tiling_key) != g_mix_graph" << graph_id << "_tiling_keys.end()) {" << std::endl;
      ss << "    return true;" << std::endl;
      ss << "  }" << std::endl;
    }
  }
  ss << "  return false;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoLaunchParamsInit(const ascir::FusedScheduledResult &fused_schedule_result,
                                       std::stringstream &ss) const {
  ss << "aclError LaunchParamsInit(PgoTensorArgs *tensor_args) {" << std::endl;
  ss << "  static void *ffts = nullptr;" << std::endl;
  ss << "  aclError ret = ACL_SUCCESS;" << std::endl;
  ss << "  constexpr uint32_t kPgoInputCount = " << fused_schedule_result.input_nodes.size() << "U;" << std::endl;
  ss << "  constexpr uint32_t kPgoOutputCount = " << PGOSearchFuncGetOutputCount(fused_schedule_result) << "U;"
     << std::endl;
  ss << "  if (tensor_args == nullptr || tensor_args->input_num < kPgoInputCount ||" << std::endl;
  ss << "      tensor_args->output_num < kPgoOutputCount ||" << std::endl;
  ss << "      (kPgoInputCount > 0U && tensor_args->inputs == nullptr) ||" << std::endl;
  ss << "      (kPgoOutputCount > 0U && tensor_args->outputs == nullptr)) {" << std::endl;
  ss << "    DLOGE(\"invalid pgo tensor args\");" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << PGOSearchFuncInputOutputStructAssignDef(fused_schedule_result, "  g_launch_params.aiv_args");
  ss << "  g_launch_params.aiv_args.tiling_addr = reinterpret_cast<uint64_t>(g_tiling_device_addr);" << std::endl;
  if (IsNeedFfts()) {
    ss << "  ret = aclrtGetHardwareSyncAddr(&ffts);" << std::endl;
    ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
    ss << "    DLOGE(\"acl get hardware sync addr failed, ERROR: %d\", ret);" << std::endl;
    ss << "    return FAILED;" << std::endl;
    ss << "  }" << std::endl;
    ss << "  g_launch_params.mix_args.ffts = reinterpret_cast<uint64_t>(ffts);" << std::endl;
  }
  ss << PGOSearchFuncInputOutputStructAssignDef(fused_schedule_result, "  g_launch_params.mix_args");
  ss << "  g_launch_params.mix_args.tiling_addr = reinterpret_cast<uint64_t>(g_tiling_device_addr);" << std::endl;
  ss << "  ret = aclrtMalloc(&g_launch_params.aiv_args_device, sizeof(AivKernelLaunchOpArgs), "
        "ACL_MEM_MALLOC_HUGE_FIRST);"
     << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl malloc aiv args device failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtMalloc(&g_launch_params.mix_args_device, sizeof(MixKernelLaunchOpArgs), "
        "ACL_MEM_MALLOC_HUGE_FIRST);"
     << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl malloc mix args device failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return ACL_SUCCESS;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoLaunchParamsDeInit(std::stringstream &ss) const {
  ss << "void LaunchParamsDeInit() {" << std::endl;
  ss << "  if (g_launch_params.aiv_args_device != nullptr) {" << std::endl;
  ss << "    auto ret = aclrtFree(g_launch_params.aiv_args_device);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGW(\"acl free aiv args device failed, ERROR: %d\", ret);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_launch_params.aiv_args_device = nullptr;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (g_launch_params.mix_args_device != nullptr) {" << std::endl;
  ss << "    auto ret = aclrtFree(g_launch_params.mix_args_device);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGW(\"acl free mix args device failed, ERROR: %d\", ret);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_launch_params.mix_args_device = nullptr;" << std::endl;
  ss << "  }" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoUpdateLaunchParams(std::stringstream &ss) const {
  ss << "aclError UpdateLaunchParam(const AutofuseTilingData &tiling_data) {" << std::endl;
  ss << "  if (IsMixTiling(tiling_data)) {" << std::endl;
  ss << "    auto ret = aclrtMemcpy((void *)g_launch_params.mix_args.tiling_addr, sizeof(AutofuseTilingData), (void "
        "*)&tiling_data, "
     << "sizeof(AutofuseTilingData), ACL_MEMCPY_HOST_TO_DEVICE);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"memcpy tiling data to device failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_launch_params.mix_args.workspace_addr = reinterpret_cast<uint64_t>(g_workspace);" << std::endl;
  ss << "    ret = aclrtMemcpy(g_launch_params.mix_args_device, sizeof(g_launch_params.mix_args), (void "
        "*)&g_launch_params.mix_args, sizeof(g_launch_params.mix_args), ACL_MEMCPY_HOST_TO_DEVICE);"
     << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"memcpy mix_args to device failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  } else {" << std::endl;
  ss << "    auto ret = aclrtMemcpy((void *)g_launch_params.aiv_args.tiling_addr, sizeof(AutofuseTilingData), (void "
        "*)&tiling_data, "
     << "sizeof(AutofuseTilingData), ACL_MEMCPY_HOST_TO_DEVICE);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"memcpy tiling data to device failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_launch_params.aiv_args.workspace_addr = reinterpret_cast<uint64_t>(g_workspace);" << std::endl;
  ss << "    ret = aclrtMemcpy(g_launch_params.aiv_args_device, sizeof(g_launch_params.aiv_args), (void "
        "*)&g_launch_params.aiv_args, sizeof(g_launch_params.aiv_args), ACL_MEMCPY_HOST_TO_DEVICE);"
     << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"memcpy aiv_args to device failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return ACL_SUCCESS;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoLaunchParams(const ascir::FusedScheduledResult &fused_schedule_result,
                                   std::stringstream &ss) const {
  ss << "struct LaunchParams {" << std::endl;
  ss << "  AivKernelLaunchOpArgs aiv_args;" << std::endl;
  ss << "  void *aiv_args_device;" << std::endl;
  ss << "  MixKernelLaunchOpArgs mix_args;" << std::endl;
  ss << "  void *mix_args_device;" << std::endl;
  ss << "} g_launch_params;" << std::endl;

  GenPgoLaunchParamsInit(fused_schedule_result, ss);
  GenPgoLaunchParamsDeInit(ss);
  GenPgoUpdateLaunchParams(ss);
}

void TilingLib::GenPgoToolFunction(const ascir::FusedScheduledResult &fused_schedule_result, const std::string &pgo_dir,
                                   std::stringstream &ss) const {
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  ss << "namespace {" << std::endl;
  ss << "constexpr bool g_is_mix_operator = " << (IsMixKernelTaskType(fused_schedule_result) ? "true;" : "false;")
     << std::endl;
  ss << "static bool g_is_static_kernel = false;" << std::endl;
  GenPgoMixTilingTable(fused_schedule_result, ss);
  GenPgoCheckTilingIsMix(fused_schedule_result, ss);
  ss << "static std::string g_kernel_name;" << std::endl;
  ss << "static std::string g_kernel_o_file;" << std::endl;
  ss << "static std::string g_npu_lock_file;" << std::endl;
  ss << "#define PGO_GRAPH_NAME \"" << graph_name << "\"" << std::endl;
  ss << "const char *pgo_dir = \"" << pgo_dir << "\";" << std::endl;
  ss << "const char *config_file = \"" << pgo_dir << "/" << graph_name << "_config.txt" << "\";" << std::endl;
  ss << "const char *search_file = \"" << pgo_dir << "/" << graph_name << "_search.txt" << "\";" << std::endl;
  ss << "const char *kernel_file = \"" << pgo_dir << "/lib" << graph_name << ".so" << "\";" << std::endl;
  ss << "#define SUCCESS 0" << std::endl;
  ss << "#define FAILED 1" << std::endl;

  ss << "inline uint64_t PgoGetTid() {" << std::endl;
  ss << "  return static_cast<uint64_t>(syscall(__NR_gettid));" << std::endl;
  ss << "}" << std::endl;
  ss << "constexpr int32_t PGO_MODULE_NAME = static_cast<int32_t>(" << GE_MODULE_NAME << ");" << std::endl;
  ss << "#define PGO_LOG_PREFIX \"%\" PRIu64 \" %s:[PGO][\" PGO_GRAPH_NAME \"] \"" << std::endl;
  ss << "#define DLOGD(fmt, ...) do { dlog_debug(PGO_MODULE_NAME, PGO_LOG_PREFIX fmt, PgoGetTid(), &__FUNCTION__[0U], "
        "##__VA_ARGS__); } while (false)"
     << std::endl;
  ss << "#define DLOGI(fmt, ...) do { dlog_info(PGO_MODULE_NAME, PGO_LOG_PREFIX fmt, PgoGetTid(), &__FUNCTION__[0U], "
        "##__VA_ARGS__); } while (false)"
     << std::endl;
  ss << "#define DLOGW(fmt, ...) do { dlog_warn(PGO_MODULE_NAME, PGO_LOG_PREFIX fmt, PgoGetTid(), &__FUNCTION__[0U], "
        "##__VA_ARGS__); } while (false)"
     << std::endl;
  ss << "#define DLOGE(fmt, ...) do { dlog_error(PGO_MODULE_NAME, PGO_LOG_PREFIX fmt, PgoGetTid(), &__FUNCTION__[0U], "
        "##__VA_ARGS__); } while (false)"
     << std::endl;

  GenPgoCardLock(ss);
  GenPgoAppendSearchTilingData(ss);
  GenPgoKernelLaunchOpArgs(fused_schedule_result, ss);

  GenDynamicLibraryLoaderCode(ss);

  ss << "aclrtStream g_stream;" << std::endl;
  ss << PGOSearchTensorInputOutputDef(fused_schedule_result) << std::endl;
  ss << "void *g_tiling_device_addr = nullptr;" << std::endl;

  GenPgoLaunchParams(fused_schedule_result, ss);

  ss << "struct ResLimit {" << std::endl;
  ss << "  uint32_t valid_num = 0;" << std::endl;
  ss << "  uint32_t aiv_num = 0;" << std::endl;
  ss << "  uint32_t aic_num = 0;" << std::endl;
  ss << "  uint32_t ub_size = 0;" << std::endl;
  ss << "  uint32_t resv[10];" << std::endl;
  ss << "};" << std::endl;
  ss << "ResLimit g_res_limit = {1, {}};" << std::endl;
  ss << "inline bool IsEqual(double a, double b) {" << std::endl;
  ss << "  const double epsilon = 1e-8;" << std::endl;
  ss << "  double abs = (a > b) ? (a - b) : (b - a);" << std::endl;
  ss << "  return abs < epsilon;" << std::endl;
  ss << "}" << std::endl;
  ss << "} // namespace" << std::endl;
}

void TilingLib::GenPgoWrapperParmCall(const ascir::FusedScheduledResult &fused_schedule_result,
                                      std::stringstream &ss) const {
  ss << "  if (tiling_data == nullptr) {" << std::endl;
  ss << "    DLOGE(\"tiling_data is null\");" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  uint32_t block_dim = tiling_data->block_dim;" << std::endl;
  ss << "  aclError ret = ACL_SUCCESS;" << std::endl;
  ss << "  int64_t tiling_key = 0;" << std::endl;
  if (CanUseTilingKey(fused_schedule_result)) {
    ss << "  if (find_best_tiling_key_fn != nullptr) {" << std::endl;
    ss << "    tiling_key = find_best_tiling_key_fn(*tiling_data);" << std::endl;
    ss << "    if (tiling_key == -1) {" << std::endl;
    ss << "      DLOGE(\"find best tiling key failed\");" << std::endl;
    ss << "      return FAILED;" << std::endl;
    ss << "    }" << std::endl;
    ss << "  } else {" << std::endl;
    ss << "    DLOGE(\"find best tiling key func is null\");" << std::endl;
    ss << "    return FAILED;" << std::endl;
    ss << "  }" << std::endl;
  }
}

void TilingLib::GenPgoWrapperKernelLaunch(std::stringstream &ss) const {
  ss << "  if (IsMixTiling(*tiling_data)) {" << std::endl;
  const auto backend_spce = optimize::BackendSpec::GetInstance();
  const bool use_local_memory = (backend_spce != nullptr && backend_spce->set_local_memory_size > 0);
  ss << (use_local_memory ? "    ret = aclrtLaunchKernelV2(func_handles[tiling_key], block_dim, "
                            "g_launch_params.mix_args_device, sizeof(g_launch_params.mix_args), &kernel_cfg, g_stream);"
                          : "    ret = aclrtLaunchKernelV2(func_handles[tiling_key], block_dim, "
                            "g_launch_params.mix_args_device, sizeof(g_launch_params.mix_args), nullptr, g_stream);")
     << std::endl;
  ss << "  } else {" << std::endl;
  ss << (use_local_memory ? "    ret = aclrtLaunchKernelV2(func_handles[tiling_key], block_dim, "
                            "g_launch_params.aiv_args_device, sizeof(g_launch_params.aiv_args), &kernel_cfg, g_stream);"
                          : "    ret = aclrtLaunchKernelV2(func_handles[tiling_key], block_dim, "
                            "g_launch_params.aiv_args_device, sizeof(g_launch_params.aiv_args), nullptr, g_stream);")
     << std::endl;
  ss << "  }" << std::endl;
  ss << "  auto ret_async = aclrtSynchronizeStream(g_stream);" << std::endl;
}

void TilingLib::GenPgoWrapper(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "typedef uint64_t (*GetTilingKeyCountType)(void);" << std::endl;
  ss << "GetTilingKeyCountType get_tiling_key_count_fn = "
        "reinterpret_cast<GetTilingKeyCountType>(GetFunc(\"GetTilingKeyCount\"));"
     << std::endl;
  if (CanUseTilingKey(fused_schedule_result)) {
    ss << "typedef int64_t (*FindBestTilingKeyType)(AutofuseTilingData &t);" << std::endl;
    ss << "FindBestTilingKeyType find_best_tiling_key_fn = "
          "reinterpret_cast<FindBestTilingKeyType>(GetFunc(\"FindBestTilingKey\"));"
       << std::endl;
  }
  ss << "int WrapperOnlyLaunch(uint32_t workspace_size, AutofuseTilingData *tiling_data) {" << std::endl;
  ss << "  static bool inited = false;" << std::endl;
  ss << "  static aclrtBinHandle bin_handle = nullptr;" << std::endl;
  const auto backend_spce = optimize::BackendSpec::GetInstance();
  if (backend_spce != nullptr && backend_spce->set_local_memory_size > 0) {
    ss << "  static aclrtLaunchKernelCfg kernel_cfg{};" << std::endl;
    ss << "  static aclrtLaunchKernelAttr local_memory_size_attr{};" << std::endl;
  }
  ss << "  if (get_tiling_key_count_fn == nullptr) {" << std::endl;
  ss << "    DLOGE(\"get_tiling_key_count_fn is nullptr\");" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  static uint64_t tiling_key_count = get_tiling_key_count_fn();" << std::endl;
  ss << "  static std::vector<aclrtFuncHandle> func_handles(tiling_key_count);" << std::endl;

  GenPgoWrapperParmCall(fused_schedule_result, ss);
  GenPgoLaunchKernelInit(ss);
  GenPgoWrapperKernelLaunch(ss);
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"aclrtLaunchKernelV2 failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (ret_async != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"aclrtSynchronizeStream failed, ERROR: %d\", ret_async);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return ret;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoProfilingConstants(std::stringstream &ss) const {
  ss << "#define ALIGN_SIZE (8)" << std::endl;
  ss << "#define ALIGN_BUFFER(buffer, align) \\" << std::endl;
  ss << "    (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : "
     << "(buffer))" << std::endl;
  ss << "constexpr size_t group_size = 1000ULL;" << std::endl;
  ss << "static std::map<uint64_t, msptiActivity*> g_profiling_map;" << std::endl;
  ss << "constexpr uint64_t loop = 20;" << std::endl;
  ss << "constexpr int max_flush_times = 5;" << std::endl;
  ss << "constexpr size_t mspti_buffer_size = 16ULL * 1024 * 1024;" << std::endl;
  ss << "static double best_perf = DBL_MAX;" << std::endl;
}

void TilingLib::GenPgoMsptiStringTable(std::stringstream &ss) const {
  ss << R"(
static const char* GetActivityKindString(msptiActivityKind kind) {
  static const std::unordered_map<msptiActivityKind, const char*> STRING_MAP = {
    {MSPTI_ACTIVITY_KIND_INVALID, "INVALID"},
    {MSPTI_ACTIVITY_KIND_MARKER, "MARKER"},
    {MSPTI_ACTIVITY_KIND_KERNEL, "KERNEL"},
    {MSPTI_ACTIVITY_KIND_API, "API"},
    {MSPTI_ACTIVITY_KIND_MEMORY, "MEMORY"},
    {MSPTI_ACTIVITY_KIND_MEMSET, "MEMSET"},
    {MSPTI_ACTIVITY_KIND_MEMCPY, "MEMCPY"},
    {MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION, "CORRELATION"}
  };
  auto it = STRING_MAP.find(kind);
  return it != STRING_MAP.end() ? it->second : "<unknown>";
})" << std::endl;
  ss << R"(
static const char* GetResultCodeString(msptiResult result) {
  static const std::unordered_map<msptiResult, const char*> STRING_MAP = {
    {MSPTI_SUCCESS, "SUCCESS"},
    {MSPTI_ERROR_INVALID_PARAMETER, "ERROR_INVALID_PARAMETER"},
    {MSPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED, "MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED"},
    {MSPTI_ERROR_DEVICE_OFFLINE, "DEVICE_OFFLINE"},
    {MSPTI_ERROR_QUEUE_EMPTY, "QUEUE_EMPTY"},
    {MSPTI_ERROR_INNER, "ERROR_INNER"}
  };

  auto it = STRING_MAP.find(result);
  return it != STRING_MAP.end() ? it->second : "<unknown>";
})" << std::endl;
}

void TilingLib::GenPgoMsptiRequest(std::stringstream &ss) const {
  ss << R"(
void UserBufferRequest(uint8_t **buffer, size_t *size, size_t *records_num) {
  DLOGD("[mspti] UserBufferRequest...");
  uint8_t *mspti_buffer = reinterpret_cast<uint8_t *>(malloc(mspti_buffer_size + ALIGN_SIZE));
  if (mspti_buffer == nullptr) {
    DLOGE("[mspti] malloc mspti_buffer failed");
    *buffer = nullptr;
    *size = 0;
    *records_num = 0;
    return;
  }
  *buffer = ALIGN_BUFFER(mspti_buffer, ALIGN_SIZE);
  *size = mspti_buffer_size;
  *records_num = 0;
})" << std::endl;
}

void TilingLib::GenPgoMsptiComplete(std::stringstream &ss) const {
  ss << R"(
void UserBufferComplete(uint8_t *buffer, size_t size, size_t valid_size) {
  DLOGD("[mspti] UserBufferComplete, buf addr: %" PRIuPTR ", size: %zu, valid size: %zu", (uintptr_t)buffer, size, valid_size);
  if (valid_size > 0) {
    msptiActivity *mspti_record = NULL;
    msptiResult status = MSPTI_SUCCESS;
    do {
      status = msptiActivityGetNextRecord(buffer, valid_size, &mspti_record);
      if (status == MSPTI_SUCCESS) {
        if (mspti_record->kind == MSPTI_ACTIVITY_KIND_KERNEL) {
          msptiActivityKernel* kernelRecord = (msptiActivityKernel*)mspti_record;
          msptiActivity* pRecordCopy = (msptiActivity *)malloc(sizeof(msptiActivityKernel));
          memset(pRecordCopy, 0, sizeof(msptiActivityKernel));
          memcpy(pRecordCopy, kernelRecord, sizeof(msptiActivityKernel));
          g_profiling_map[kernelRecord->start] = pRecordCopy;

        } else {
          DLOGD("[mspti] [%s] ignored", GetActivityKindString(mspti_record->kind));
        }
      } else if (status == MSPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else {
        DLOGW("[mspti] Consume data fail error is %s", GetResultCodeString(status));
        break;
      }
    } while (1);
  }
  free(buffer);
})" << std::endl;
}

void TilingLib::GenPgoMsptiToolFunction(std::stringstream &ss) const {
  ss << R"(
void SetUpMspti(msptiSubscriberHandle* subscriber) {
  DLOGD("[mspti] setup mspti");
  msptiSubscribe(subscriber, nullptr, nullptr);
  msptiActivityRegisterCallbacks(UserBufferRequest, UserBufferComplete);
  msptiActivityEnable(MSPTI_ACTIVITY_KIND_KERNEL);
})" << std::endl;
  ss << R"(
void TearDownMspti(msptiSubscriberHandle *subscriber) {
  DLOGD("[mspti] tear down mspti");
  msptiUnsubscribe(*subscriber);
  msptiActivityFlushAll(1);
})" << std::endl;
}

void TilingLib::GenPgoMsptiProfiling(std::stringstream &ss) const {
  GenPgoProfilingConstants(ss);
  GenPgoMsptiStringTable(ss);
  GenPgoMsptiRequest(ss);
  GenPgoMsptiComplete(ss);
  GenPgoMsptiToolFunction(ss);
}

void TilingLib::GenPgoBatchCallback(std::stringstream &ss) const {
  ss << "  result = aclrtSynchronizeStream(g_stream);" << std::endl;
  ss << "  TearDownMspti(&subscriber);" << std::endl << std::endl;
  ss << "  int flush_count = 0;" << std::endl;
  ss << "  while (g_profiling_map.size() < batch_size * loop && flush_count < max_flush_times) {" << std::endl;
  ss << "    flush_count++;" << std::endl;
  ss << "    std::this_thread::sleep_for(std::chrono::milliseconds(10 * flush_count));" << std::endl;
  ss << "    msptiActivityFlushAll(1);" << std::endl;
  ss << "  }" << std::endl << std::endl;
  ss << "  if (g_profiling_map.size() < batch_size * loop) {" << std::endl;
  ss << "    DLOGE(\"ProfilingBatchProcess g_profiling_map size %zu is less than batch_size * loop %\" PRIu64 \"\", "
        "g_profiling_map.size(), batch_size * loop);"
     << std::endl;
  ss << "    for (auto &item : g_profiling_map) {" << std::endl;
  ss << "      free(item.second);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl << std::endl;
  ss << "  auto it = g_profiling_map.begin();" << std::endl;
  ss << "  for (uint64_t i = 0; i < batch_size; ++i) {" << std::endl;
  ss << "    uint64_t total_duration = 0;" << std::endl;
  ss << "    std::vector<uint64_t> durations;" << std::endl;
  ss << "    for (uint64_t j = 0; j < loop; ++j) {" << std::endl;
  ss << "      msptiActivityKernel* kernel = reinterpret_cast<msptiActivityKernel*>(it->second);" << std::endl;
  ss << "      durations.push_back(kernel->end - kernel->start);" << std::endl;
  ss << "      std::advance(it, 1);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    std::sort(durations.begin(), durations.end(), std::greater<int>());" << std::endl;
  ss << "    for (size_t k = 1; k < 6; ++k) {" << std::endl;
  ss << "      total_duration += durations[k];" << std::endl;
  ss << "    }" << std::endl;
  ss << "    double average_duration = static_cast<double>(total_duration) / 5;" << std::endl;
  ss << "    (begin + i)->best_perf = average_duration;" << std::endl;
  ss << "    if (best_perf > average_duration) {" << std::endl;
  ss << "      best_perf = average_duration;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    DLOGD(\"average_duration:%f best_perf:%f count:%\" PRId64 \" batch_size:%\" PRIu64 \" flush_count:%d\", "
        "average_duration, best_perf, count, batch_size, flush_count);"
     << std::endl;
  ss << "  }" << std::endl;
  ss << "  for (auto &item : g_profiling_map) {" << std::endl;
  ss << "    free(item.second);" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenPgoBatchProcess(std::stringstream &ss) const {
  ss << "int ProfilingBatchProcess(uint32_t workspace_size, std::vector<AutofuseTilingDataPerf>::iterator begin, "
        "std::vector<AutofuseTilingDataPerf>::iterator end) {"
     << std::endl;
  ss << "  uint64_t batch_size = end - begin;" << std::endl;
  ss << "  g_profiling_map.clear();" << std::endl;
  ss << "  msptiSubscriberHandle subscriber;" << std::endl;
  ss << "  SetUpMspti(&subscriber);" << std::endl << std::endl;
  ss << "  static int64_t count = 0;" << std::endl;
  ss << "  count++;" << std::endl << std::endl;
  ss << "  int64_t result = 0;" << std::endl;
  ss << "  for (auto it = begin; it != end; ++it) {" << std::endl;
  ss << "    it->best_perf = DBL_MAX;" << std::endl;
  ss << "    AutofuseTilingData &tiling_data = it->tiling_data;" << std::endl;
  ss << "    UpdateLaunchParam(tiling_data);" << std::endl;
  ss << "    for (uint64_t i = 0; i < loop; ++i) {" << std::endl;
  ss << "      result = WrapperOnlyLaunch(workspace_size, &tiling_data);" << std::endl;
  ss << "      if (result != 0) {" << std::endl;
  ss << "        DLOGE(\"ProfilingBatchProcess launch failed loop:%\" PRIu64 \"\", i);" << std::endl;
  ss << "        TearDownMspti(&subscriber);" << std::endl;
  ss << "        return -1;" << std::endl;
  ss << "      }" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl << std::endl;
  GenPgoBatchCallback(ss);
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoGetProfilingBatch(const ascir::FusedScheduledResult &fused_schedule_result,
                                        std::stringstream &ss) const {
  ss << "extern \"C\" long int PGOGetProfilingBatch(" << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result)
     << "void* stream, uint32_t workspace_size, std::vector<AutofuseTilingDataPerf> *profiles) {" << std::endl;
  ss << "  int case_num = profiles->size();" << std::endl;
  ss << "  DLOGI(\"PGOGetProfilingBatch case_num:%d\", case_num);" << std::endl;
  ss << "  if (workspace_size > 0) {" << std::endl;
  ss << "    auto ret = aclrtMalloc(&g_workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"malloc workspace failed, size: %u, ERROR: %d\", workspace_size, ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  int64_t result = 0;" << std::endl;
  ss << "  auto it = profiles->begin();" << std::endl;
  ss << "  while (it != profiles->end()) {" << std::endl;
  ss << "    auto end_it = (it + group_size >= profiles->end()) ? profiles->end() : it + group_size;" << std::endl;
  ss << "    size_t start_index = std::distance(profiles->begin(), it);" << std::endl;
  ss << "    for (int i = 0; i < 3; i++) {" << std::endl;
  ss << "      result = ProfilingBatchProcess(workspace_size, it, end_it);" << std::endl;
  ss << "      if (result != 0) {" << std::endl;
  ss << "        DLOGW(\"ProfilingBatchProcess failed at start_index:%zu retry time:%d\", start_index, i);"
     << std::endl;
  ss << "      } else {" << std::endl;
  ss << "        break;" << std::endl;
  ss << "      }" << std::endl;
  ss << "    }" << std::endl;
  ss << "    it = end_it;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (g_workspace != nullptr) {" << std::endl;
  ss << "    auto ret = aclrtFree(g_workspace);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"free workspace failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoProfilingCallback(std::stringstream &ss) const {
  ss << "  result = aclrtSynchronizeStream(g_stream);" << std::endl;
  ss << "  if (result != 0) {" << std::endl;
  ss << "    DLOGE(\"sync stream failed\");" << std::endl;
  ss << "    TearDownMspti(&subscriber);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  TearDownMspti(&subscriber);" << std::endl;
  ss << std::endl;
  ss << "  int flush_count = 0;" << std::endl;
  ss << "  while (g_profiling_map.size() < loop && flush_count < max_flush_times) {" << std::endl;
  ss << "    flush_count++;" << std::endl;
  ss << "    std::this_thread::sleep_for(std::chrono::milliseconds(10 * flush_count));" << std::endl;
  ss << "    msptiActivityFlushAll(1);" << std::endl;
  ss << "  }" << std::endl;
  ss << std::endl;
  ss << "  if (g_profiling_map.size() != loop) {" << std::endl;
  ss << "    DLOGE(\"map size %zu not equals to loop %\" PRIu64 \"\", g_profiling_map.size(), loop);" << std::endl;
  ss << "    for (auto &item : g_profiling_map) {" << std::endl;
  ss << "      free(item.second);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << std::endl;
  ss << "  uint64_t total_duration = 0;" << std::endl;
  ss << "  std::vector<uint64_t> durations;" << std::endl;
  ss << "  for (const auto &pair : g_profiling_map) {" << std::endl;
  ss << "    msptiActivityKernel* kernel = reinterpret_cast<msptiActivityKernel*>(pair.second);" << std::endl;
  ss << "    durations.push_back(kernel->end - kernel->start);" << std::endl;
  ss << "    DLOGD(\"kernel duration:%\" PRIu64 \"\", kernel->end - kernel->start);" << std::endl;
  ss << "  }" << std::endl;
  ss << "  std::sort(durations.begin(), durations.end(), std::greater<int>());" << std::endl;
  ss << "  for (size_t i = 1; i < 6; ++i) {" << std::endl;
  ss << "    total_duration += durations[i];" << std::endl;
  ss << "  }" << std::endl;
  ss << "  double average_duration = static_cast<double>(total_duration) / 5;" << std::endl;
  ss << "  *outCostTime = average_duration;" << std::endl;
  ss << std::endl;
  ss << "  if (best_perf > *outCostTime) {" << std::endl;
  ss << "    best_perf = *outCostTime;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  DLOGD(\"average_duration:%f best_perf:%f count:%\" PRId64 \" flush_count:%d\", *outCostTime, best_perf, "
        "count, flush_count);"
     << std::endl;
  ss << "  for (auto &item : g_profiling_map) {" << std::endl;
  ss << "    free(item.second);" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenPgoGetProfiling(const ascir::FusedScheduledResult &fused_schedule_result,
                                   std::stringstream &ss) const {
  ss << "extern \"C\" long int PGOGetProfiling(" << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result)
     << "void *stream, uint32_t workspace_size, AutofuseTilingData *tiling_data, double *outCostTime) {" << std::endl;
  ss << "  if (workspace_size > 0) {" << std::endl;
  ss << "    auto ret = aclrtMalloc(&g_workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"malloc workspace failed, size: %u, ERROR: %d\", workspace_size, ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  g_profiling_map.clear();" << std::endl;
  ss << "  msptiSubscriberHandle subscriber;" << std::endl;
  ss << "  SetUpMspti(&subscriber);" << std::endl << std::endl;
  ss << "  int64_t result = -1;" << std::endl;
  ss << "  *outCostTime = DBL_MAX;" << std::endl;
  ss << "  static int64_t count = 0;" << std::endl;
  ss << "  count++;" << std::endl << std::endl;

  ss << "  UpdateLaunchParam(*tiling_data);" << std::endl;
  ss << "  for (uint64_t j = 0; j < loop; ++j) {" << std::endl;
  ss << "    result = WrapperOnlyLaunch(workspace_size, tiling_data);" << std::endl;
  ss << "    if (result != 0) {" << std::endl;
  ss << "      DLOGE(\"launch failed loop:%\" PRIu64 \"\", j);" << std::endl;
  ss << "      TearDownMspti(&subscriber);" << std::endl;
  ss << "      return -1;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl << std::endl;

  ss << "  if (g_workspace != nullptr) {" << std::endl;
  ss << "    auto ret = aclrtFree(g_workspace);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"free workspace failed, ERROR: %d\", ret);" << std::endl;
  ss << "      TearDownMspti(&subscriber);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  GenPgoProfilingCallback(ss);
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoFunc(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "int pgo() {" << std::endl;
  ss << "  AutofuseTilingData tiling_data = {0};" << std::endl;
  ss << "  PgoTensorArgs *tensor_args = &g_pgo_tensor_args;" << std::endl;
  ss << "  uint32_t workspace_size = 0;" << std::endl;
  ss << "  uint32_t block_dim = 0;" << std::endl;
  ss << "  if (pgo_search_fn == nullptr) {" << std::endl;
  ss << "    DLOGE(\"pgo search func not found\");" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  int64_t result = pgo_search_fn((char*)search_file, (char *)config_file, &tiling_data, &workspace_size, "
        "&block_dim, &g_res_limit,"
     << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "&g_stream, reinterpret_cast<void*>(PGOGetProfiling), reinterpret_cast<void*>(PGOGetProfilingBatch));"
     << std::endl;
  ss << "  if (result != 0) {" << std::endl;
  ss << "    DLOGE(\"pgo search failed. ERROR: %\" PRId64 \"\", result);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoStaticFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                 std::stringstream &ss) const {
  ss << "int static_pgo(const char* config_file) {" << std::endl;
  ss << "  if (autofuse_tiling_with_config_fn == nullptr) {" << std::endl;
  ss << "    DLOGE(\"autofuse tiling with config func not found\");" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  AutofuseTilingData tiling_data = {0};" << std::endl;
  ss << "  PgoTensorArgs *tensor_args = &g_pgo_tensor_args;" << std::endl;
  ss << "  uint32_t workspace_size = 0;" << std::endl;
  ss << "  uint32_t block_dim = 0;" << std::endl;
  ss << "  int64_t result = autofuse_tiling_with_config_fn(config_file, &tiling_data, &workspace_size, &block_dim, "
        "&g_res_limit);"
     << std::endl;
  ss << "  if (result != 0) {" << std::endl;
  ss << "    DLOGE(\"autofuse tiling with config failed. ERROR: %\" PRId64 \"\", result);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  double out_cost = DBL_MAX;" << std::endl;
  ss << "  for (int i = 0; i < max_flush_times; i++) {" << std::endl;
  ss << "    result = PGOGetProfiling(" << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "g_stream, workspace_size, &tiling_data, &out_cost);" << std::endl;
  ss << "    if (result != 0 || IsEqual(out_cost, DBL_MAX)) {" << std::endl;
  ss << "      DLOGW(\"get profiling failed.\");" << std::endl;
  ss << "    } else {" << std::endl;
  ss << "      break;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  AppendPgoSearchTilingData(tiling_data, out_cost);" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoProfiling(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  GenPgoMsptiProfiling(ss);
  GenPgoBatchProcess(ss);
  GenPgoGetProfilingBatch(fused_schedule_result, ss);
  GenPgoGetProfiling(fused_schedule_result, ss);
  ss << "typedef int64_t (*PGOSearchType)(char *search_file, char *config_file, AutofuseTilingData *tiling_data, "
        "uint32_t *workspace_size, uint32_t *blockDim, void *resource_limit, "
     << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result)
     << "void *stream, void *prof_callback, void *prof_batch_callback);" << std::endl;
  ss << "static PGOSearchType pgo_search_fn = reinterpret_cast<PGOSearchType>(GetFunc(\"PgoTilingSearch\"));"
     << std::endl;
  GenPgoFunc(fused_schedule_result, ss);
  ss << "typedef int64_t (*AutofuseTilingWithConfigType)(const char *config_file, AutofuseTilingData *tiling, uint32_t "
        "*"
     << "workspace_size, uint32_t *blockDim, ResLimit *res_limit);" << std::endl;
  ss << "static AutofuseTilingWithConfigType autofuse_tiling_with_config_fn = "
     << "reinterpret_cast<AutofuseTilingWithConfigType>(GetFunc(\"AutofuseTilingWithConfig\"));" << std::endl;
  GenPgoStaticFunc(fused_schedule_result, ss);
}

void TilingLib::GenPgoMain(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "int main(int argc, char *argv[]) {" << std::endl;
  ss << "  if (argc != 6) {" << std::endl;
  ss << "    DLOGE(\"Usage: %s <type> <device_id> <aiv_num> <ub_size> <kernel_name>\", argv[0]);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  int32_t type = static_cast<int32_t>(atoi(argv[1]));" << std::endl;
  ss << "  int32_t device_id = static_cast<int32_t>(atoi(argv[2]));" << std::endl;
  ss << "  int32_t aiv_num = static_cast<int32_t>(atoi(argv[3]));" << std::endl;
  ss << "  int32_t ub_size = static_cast<int32_t>(atoi(argv[4]));" << std::endl;
  ss << "  g_kernel_name = argv[5];" << std::endl;
  ss << "  DLOGI(\"execute info : type: %d, device_id: %d, kernel_name: %s\", type, device_id, g_kernel_name.c_str());"
     << std::endl;
  ss << "  DLOGI(\"execute limit: aiv_num is %d, ub_size is %d\", aiv_num, ub_size);" << std::endl;
  ss << "  g_npu_lock_file = std::string(pgo_dir) + \"/npu_lock_\" + std::to_string(device_id) + \".lock\";"
     << std::endl;
  ss << "  g_kernel_o_file = std::string(pgo_dir) + \"/\" + g_kernel_name + \".o\";" << std::endl;
  ss << "  CardLock lock(g_npu_lock_file.c_str());" << std::endl;
  GenPgoEnvInit(fused_schedule_result, ss);
  ss << "  if (type == 0) {" << std::endl;
  ss << "    ret = pgo();" << std::endl;
  ss << "  } else if (type == 1) {" << std::endl;
  ss << "    g_is_static_kernel = true;" << std::endl;
  ss << "    ret = static_pgo(config_file);" << std::endl;
  ss << "  } else {" << std::endl;
  ss << "    DLOGE(\"Invalid type: %d\", type);" << std::endl;
  ss << "    ret = -1;" << std::endl;
  ss << "  }" << std::endl;
  GenPgoDeinit(fused_schedule_result, ss);
  ss << "  return ret;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoEnvInit(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "  g_res_limit.aiv_num = aiv_num;" << std::endl;
  ss << "  g_res_limit.ub_size = ub_size;" << std::endl;
  ss << "  auto ret = aclInit(nullptr);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl init failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtSetDevice(device_id);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl set device failed, device id: %d, ERROR: %d\", device_id, ret);" << std::endl;
  ss << "    aclFinalize();" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtCreateStream(&g_stream);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl create stream failed, ERROR: %d\", ret);" << std::endl;
  ss << "    aclrtResetDevice(device_id);" << std::endl;
  ss << "    aclFinalize();" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << PGOSearchTensorMallocDef(fused_schedule_result) << std::endl;
  ss << PGOSearchTensorArgsUpdateDef(fused_schedule_result);
  ss << "  ret = aclrtMalloc(&g_tiling_device_addr, sizeof(AutofuseTilingData), ACL_MEM_MALLOC_HUGE_FIRST);"
     << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl malloc tiling data failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = LaunchParamsInit(&g_pgo_tensor_args);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenPgoLaunchKernelInit(std::stringstream &ss) const {
  ss << "  if (!inited) {" << std::endl;
  ss << "    auto ret = aclrtBinaryLoadFromFile(g_kernel_o_file.c_str(), nullptr, &bin_handle);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"acl load binary from file failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    if (g_is_static_kernel) {" << std::endl;
  ss << "      aclrtFuncHandle func_handle = nullptr;" << std::endl;
  ss << "      ret = aclrtBinaryGetFunction(bin_handle, (g_kernel_name + \"_\" + std::to_string(tiling_key)).c_str(), "
        "&func_handle);"
     << std::endl;
  ss << "      if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "        DLOGE(\"acl get function failed, ERROR: %d\", ret);" << std::endl;
  ss << "        return FAILED;" << std::endl;
  ss << "      }" << std::endl;
  ss << "      func_handles[tiling_key] = func_handle;" << std::endl;
  ss << "    } else {" << std::endl;
  ss << "      for (uint64_t i = 0; i < tiling_key_count; ++i) {" << std::endl;
  ss << "        aclrtFuncHandle func_handle = nullptr;" << std::endl;
  ss << "        ret = aclrtBinaryGetFunction(bin_handle, (g_kernel_name + \"_\" + std::to_string(i)).c_str(), "
        "&func_handle);"
     << std::endl;
  ss << "        if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "          DLOGE(\"acl get function failed, ERROR: %d\", ret);" << std::endl;
  ss << "          return FAILED;" << std::endl;
  ss << "        }" << std::endl;
  ss << "        func_handles[i] = func_handle;" << std::endl;
  ss << "      }" << std::endl;
  ss << "    }" << std::endl;
  const auto backend_spce = optimize::BackendSpec::GetInstance();
  if (backend_spce != nullptr && backend_spce->set_local_memory_size > 0) {
    ss << "    local_memory_size_attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;" << std::endl;
    ss << "    local_memory_size_attr.value.dynUBufSize = " << backend_spce->set_local_memory_size << ";" << std::endl;
    ss << "    kernel_cfg.numAttrs = 1;" << std::endl;
    ss << "    kernel_cfg.attrs = &local_memory_size_attr;" << std::endl;
  }
  ss << "    inited = true;" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenPgoDeinit(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "  LaunchParamsDeInit();" << std::endl;
  ss << PGOSearchTensorFreeDef(fused_schedule_result) << std::endl;
  ss << "  if (g_tiling_device_addr != nullptr) {" << std::endl;
  ss << "    ret = aclrtFree(g_tiling_device_addr);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"acl free tiling data failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_tiling_device_addr = nullptr;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtDestroyStream(g_stream);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl destroy stream failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtResetDevice(device_id);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl reset device failed, device id: %d, ERROR: %d\", device_id, ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclFinalize();" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl finalize failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  DeInit();" << std::endl;
}

std::string TilingLib::GenCVTilingFunc() const {
  std::string find_cv_tiling_func = R"(
static int32_t g_basen_basem_align = 0;

int32_t get_g_basen_basem_align() {
  return g_basen_basem_align;
}

void set_g_basen_basem_align(int32_t value) {
  g_basen_basem_align = value;
}

extern "C" int64_t GenCVFusionTilingKey(char* config_file, int aiv_num, int ub_size) {
  uint32_t workspace_size;
  uint32_t block_dim;
  ResLimit limit;
  limit.aiv_num = aiv_num;
  limit.ub_size = ub_size - 256;
  set_g_basen_basem_align(basen_basem_align);
  OP_LOGI(OP_NAME, "basen_basem_align=%d, basen_align=%d, set_g_basen_basem_align=%d",
          basen_basem_align, basen_align, get_g_basen_basem_align());
  auto ret = AutofuseTilingWithConfig(config_file, &TilingDataValue, &workspace_size, &block_dim, &limit, 0);
  if (ret == -1) {
    uint32_t basen_basem_align_tmp = (uint32_t)basen_basem_align;
    // ub_size必大于 basen_basem_align_tmp
    limit.ub_size = limit.ub_size - basen_basem_align_tmp * cube_output_type_size; // 元素个数 * type_size
    set_g_basen_basem_align(basen_align);
    OP_LOGI(OP_NAME, "set_g_basen_basem_align=%d, ub_size=%u", get_g_basen_basem_align(), ub_size);
    ret = AutofuseTilingWithConfig(config_file, &TilingDataValue, &workspace_size, &block_dim, &limit, 1);
    if (ret == -1) {
      return -1;
    } else {
      return 1; // ub非全载模板返回1
    }
  }
  // need compute tile inner / basen * basem
  return 0; // ub全载模板返回0
}
)";
  return find_cv_tiling_func;
}

std::string TilingLib::GenTilingDataBlockDimAndWss() const {
  std::string get_block_dim_and_wss = R"(
extern "C" int GenTilingDataValueBlockDimAndWss(char* config_file, uint32_t aiv_num, uint32_t ub_size, uint32_t* workspace_size, uint32_t* block_dim) {
    ResLimit limit;
    limit.aiv_num = aiv_num;
    limit.ub_size = ub_size - 256;
    auto ret = AutofuseTilingWithConfig(config_file, &TilingDataValue, workspace_size, block_dim, &limit);
    if (ret == -1) {
        OP_LOGI(OP_NAME, "get_block_dim_and_wss return -1");
        return -1;
    } else {
        return 0;
    }
}
)";
  return get_block_dim_and_wss;
}

Status TilingLib::ExtractMatMulCubeInfoFromImplGraph(const af::AscGraph &impl_graph, MatMulCubeInfo &cube_info) const {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (node->attr.api.compute_type != af::ComputeType::kComputeCube) {
      continue;
    }
    ascgen_utils::MatMulAttr mm_attr_data;
    GE_CHK_STATUS_RET(ascgen_utils::ParseMatmulAttr(node, mm_attr_data), "ParseMatmulAttr failed for node[%s]",
                      node->GetName().c_str());

    cube_info.transpose_x1 = (mm_attr_data.transpose_x1 != 0) || (mm_attr_data.adj_x1 != 0);
    cube_info.transpose_x2 = (mm_attr_data.transpose_x2 != 0) || (mm_attr_data.adj_x2 != 0);
    cube_info.offset_x = mm_attr_data.offset_x;
    cube_info.is_batch = mm_attr_data.is_batch;
    cube_info.has_relu = (mm_attr_data.has_relu != 0);
    cube_info.enable_hf32 = mm_attr_data.enable_hf32;
    cube_info.matmul_node = node;

    GE_CHK_STATUS_RET(ascgen_utils::GetCubeOutputTypeSize(node, cube_info.type_size),
                      "GetMutmulOutputTypeSize failed for node[%s]", node->GetName().c_str());

    GE_CHK_STATUS_RET(ascgen_utils::GetCubeInputNum(node, cube_info.input_num), "GetMutmulInputNum failed for node[%s]",
                      node->GetName().c_str());

    return af::SUCCESS;
  }

  return af::FAILED;
}

Status TilingLib::ExtractMatMulCubeInfoFromFusedResult(const ascir::FusedScheduledResult &fused_schedule_result,
                                                       MatMulCubeInfo &cube_info) const {
  auto extract_from_impl_graphs = [this, &cube_info](const auto &schedule_groups) {
    for (const auto &schedule_group : schedule_groups) {
      for (const auto &impl_graph : schedule_group.impl_graphs) {
        if (ExtractMatMulCubeInfoFromImplGraph(impl_graph, cube_info) == af::SUCCESS) {
          return true;
        }
      }
    }
    return false;
  };

  auto process_scheduled_results = [&extract_from_impl_graphs](const auto &scheduled_results) {
    for (const auto &scheduled_result : scheduled_results) {
      if (scheduled_result.cube_type != ascir::CubeTemplateType::kDefault) {
        if (extract_from_impl_graphs(scheduled_result.schedule_groups)) {
          return true;
        }
      }
    }
    return false;
  };

  for (const auto &scheduled_results : fused_schedule_result.node_idx_to_scheduled_results) {
    if (process_scheduled_results(scheduled_results)) {
      return af::SUCCESS;
    }
  }

  return af::FAILED;
}

Status TilingLib::GetInputTensorInfoFromLoadNode(const ge::NodePtr &load_node, TensorInfo &tensor_info) const {
  GE_ASSERT_NOTNULL(load_node);
  const auto load_node_desc = load_node->GetOpDesc();
  GE_ASSERT_NOTNULL(load_node_desc);
  const auto load_tensor_desc = load_node_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(load_tensor_desc);

  tensor_info.name = load_node->GetName();
  tensor_info.dtype = DtypeToStr(load_tensor_desc->GetDataType());
  tensor_info.format = ge::TypeUtils::FormatToSerialString(load_tensor_desc->GetFormat());

  auto tensor_attr = load_tensor_desc->GetAttrsGroup<ge::AscTensorAttr>();
  GE_ASSERT_NOTNULL(tensor_attr);

  for (const auto &repeat : tensor_attr->repeats) {
    tensor_info.shape.push_back(repeat);
  }

  tensor_info.ori_shape = tensor_info.shape;
  tensor_info.param_name = tensor_info.name;

  return af::SUCCESS;
}

Status TilingLib::ExtractInputsFromMatMulNode(const ge::AscNodePtr &matmul_node,
                                              std::vector<TensorInfo> &inputs) const {
  GE_ASSERT_NOTNULL(matmul_node);
  GE_ASSERT_NOTNULL(matmul_node->GetOpDesc());
  uint32_t input_num = matmul_node->GetOpDesc()->GetInputsSize();

  for (uint32_t i = 0U; i < input_num; ++i) {
    auto in_input_anchor = matmul_node->GetInDataAnchor(i);
    GE_ASSERT_NOTNULL(in_input_anchor);

    auto peer_out_anchor = in_input_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out_anchor);
    auto load_node = peer_out_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(load_node);

    TensorInfo tensor_info;
    GE_CHK_STATUS(GetInputTensorInfoFromLoadNode(load_node, tensor_info), "Get mutmul input info failed.");
    inputs.push_back(tensor_info);
  }

  return inputs.empty() ? af::FAILED : af::SUCCESS;
}

Status TilingLib::ExtractOutputsFromMatMulNode(const ge::AscNodePtr &matmul_node,
                                               std::vector<TensorInfo> &outputs) const {
  const auto mm_node_desc = matmul_node->GetOpDesc();
  GE_ASSERT_NOTNULL(mm_node_desc);
  const auto mm_tensor_desc = mm_node_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(mm_tensor_desc);

  TensorInfo output_info;
  output_info.name = mm_tensor_desc->GetName() + "_output";
  output_info.dtype = DtypeToStr(mm_tensor_desc->GetDataType());
  output_info.format = ge::TypeUtils::FormatToSerialString(mm_tensor_desc->GetFormat());

  auto tensor_attr = mm_tensor_desc->GetAttrsGroup<ge::AscTensorAttr>();
  GE_ASSERT_NOTNULL(tensor_attr);
  for (const auto &repeat : tensor_attr->repeats) {
    output_info.shape.push_back(repeat);
  }

  output_info.ori_shape = output_info.shape;
  output_info.param_name = output_info.name;
  outputs.push_back(output_info);

  return af::SUCCESS;
}

std::string TilingLib::GenerateTensorInfoCode(const TensorInfo &tensor, const std::string &var_name) const {
  std::stringstream ss;
  ss << "TensorInfo " << var_name << ";\n";
  ss << "  " << var_name << ".name = \"" << tensor.name << "\";\n";
  ss << "  " << var_name << ".dtype = \"" << tensor.dtype << "\";\n";
  ss << "  " << var_name << ".format = \"" << tensor.format << "\";\n";
  ss << "  " << var_name << ".shape = " << VectorToStr(tensor.shape, '{', '}') << ";\n";
  ss << "  " << var_name << ".ori_shape = " << VectorToStr(tensor.ori_shape, '{', '}') << ";\n";
  ss << "  " << var_name << ".param_name = \"" << tensor.param_name << "\";\n";
  return ss.str();
}

std::string TilingLib::GenerateAttrInfoCode(const AttrInfo &attr, const std::string &var_name) const {
  std::stringstream ss;
  ss << "AttrInfo " << var_name << ";\n";
  ss << "  " << var_name << ".name = \"" << attr.name << "\";\n";
  ss << "  " << var_name << ".dtype = \"" << attr.dtype << "\";\n";

  if (attr.dtype == "bool") {
    ss << "  " << var_name << ".value_bool = " << (attr.value_bool ? "true" : "false") << ";\n";
  } else if (attr.dtype == "int") {
    ss << "  " << var_name << ".value_int = " << attr.value_int << ";\n";
  } else if (attr.dtype == "string") {
    ss << "  " << var_name << ".value_str = \"" << attr.value_str << "\";\n";
  } else if (attr.dtype == "float") {
    ss << "  " << var_name << ".value_float = " << attr.value_float << ";\n";
  }

  return ss.str();
}

void TilingLib::PrepareMatMulAttrs(const MatMulCubeInfo &cube_info, std::vector<AttrInfo> &attrs) const {
  AttrInfo attr1;
  attr1.name = cube_info.is_batch ? "adj_x1" : "transpose_x1";
  attr1.dtype = "bool";
  attr1.value_bool = cube_info.transpose_x1;
  attrs.push_back(attr1);

  AttrInfo attr2;
  attr2.name = cube_info.is_batch ? "adj_x2" : "transpose_x2";
  attr2.dtype = "bool";
  attr2.value_bool = cube_info.transpose_x2;
  attrs.push_back(attr2);

  AttrInfo attr3;
  attr3.name = "offset_x";
  attr3.dtype = "int";
  attr3.value_int = 0x80;
  attrs.push_back(attr3);

  AttrInfo attr4;
  if (cube_info.is_batch) {
    attr4.name = "enable_hf32";
    attr4.dtype = "bool";
    attr4.value_bool = cube_info.enable_hf32 ? 1 : 0;
  } else {
    attr4.name = "opImplMode";
    attr4.dtype = "int";
    attr4.value_int = cube_info.enable_hf32;
  }
  attrs.push_back(attr4);

  AttrInfo attr5;
  attr5.name = "ascendc_op_para_size";
  attr5.dtype = "int";
  attr5.value_int = 2 * 1024 * 1024;
  attrs.push_back(attr5);
}

void TilingLib::GenerateTensorListCode(std::stringstream &code_ss, const std::vector<TensorInfo> &inputs,
                                       const std::vector<TensorInfo> &outputs) const {
  code_ss << "// Inputs\n";
  code_ss << "std::vector<TensorInfo> inputs;\n";
  for (size_t i = 0U; i < inputs.size(); ++i) {
    std::string var_name = "input_" + std::to_string(i);
    code_ss << GenerateTensorInfoCode(inputs[i], var_name);
    code_ss << "  inputs.push_back(" << var_name << ");\n";
  }
  code_ss << "\n";

  code_ss << "// Outputs\n";
  code_ss << "std::vector<TensorInfo> outputs;\n";
  for (size_t i = 0U; i < outputs.size(); ++i) {
    std::string var_name = "output_" + std::to_string(i);
    code_ss << GenerateTensorInfoCode(outputs[i], var_name);
    code_ss << "  outputs.push_back(" << var_name << ");\n";
  }
  code_ss << "\n";
}

void TilingLib::GenerateTilingCallCode(std::stringstream &code_ss, bool is_batch) const {
  code_ss << "// Call DoMatMulTiling\n";
  code_ss << "CubeKernelTilingWrapper wrapper;\n";
  code_ss << "TilingResult result = wrapper.DoMatMulTiling(compile_info, inputs, outputs, attrs, "
          << (is_batch ? "true" : "false") << ");\n";
  code_ss << "ws_size = result.workspace_size;\n";

  // 保存字节流到全局变量（用于静态shape常量生成）
  code_ss << "// Save tiling bytes for const generation in static shape\n";
  code_ss << "g_matmul_tiling_bytes = result.tiling_data;\n";
  code_ss << "std::memset(tiling_data->matmul_tiling_data, 0, sizeof(tiling_data->matmul_tiling_data));\n";
  code_ss << "size_t copy_size = std::min(result.tiling_data.size(), sizeof(tiling_data->matmul_tiling_data));\n";
  code_ss << "std::memcpy(tiling_data->matmul_tiling_data, result.tiling_data.data(), copy_size);\n";

  code_ss << "cube_block_dim = result.cube_used_core_num;\n";
  code_ss << "basem = result.cube_base_m;\n";
  code_ss << "basen = result.cube_base_n;\n";
  code_ss << "tiling_key = result.tiling_key;\n";
  code_ss << "OP_LOGI(OP_NAME, \"tiling_key=%ld, ws_size=%ld, cube_block_dim=%d, basem=%d, basen=%d\", tiling_key, "
             "ws_size, cube_block_dim, basem, basen);\n";
}

std::string TilingLib::GenerateMatMulTilingCode(const CompileInfo &compile_info, const std::vector<TensorInfo> &inputs,
                                                const std::vector<TensorInfo> &outputs,
                                                const std::vector<AttrInfo> &attrs, bool is_batch) const {
  std::stringstream code_ss;

  // 注意：全局变量g_matmul_tiling_bytes已在CallCubeTiling函数外部单独定义
  // 此处不重复定义

  code_ss << "// CompileInfo\n";
  code_ss << "CompileInfo compile_info;\n";
  code_ss << "compile_info.soc_version = \"" << compile_info.soc_version << "\";\n";
  code_ss << "compile_info.core_type = \"" << compile_info.core_type << "\";\n";
  auto device_id_iter = compile_info.extra_info.find("device_id");
  if (device_id_iter != compile_info.extra_info.end()) {
    code_ss << "compile_info.device_id = \"" << device_id_iter->second << "\";\n";
  }
  code_ss << "compile_info.aicore_num = " << compile_info.aicore_num << ";\n";
  code_ss << "compile_info.aiv_num = " << compile_info.aiv_num << ";\n";
  code_ss << "compile_info.op_kernel_lib = \"" << compile_info.op_kernel_lib << "\";\n";
  code_ss << "compile_info.op_impl_mode = \"" << compile_info.op_impl_mode << "\";\n\n";

  GenerateTensorListCode(code_ss, inputs, outputs);

  code_ss << "// Attributes\n";
  code_ss << "std::vector<AttrInfo> attrs;\n";
  for (size_t i = 0U; i < attrs.size(); ++i) {
    std::string var_name = "attr_" + std::to_string(i);
    code_ss << GenerateAttrInfoCode(attrs[i], var_name);
    code_ss << "  attrs.push_back(" << var_name << ");\n";
  }
  code_ss << "\n";

  GenerateTilingCallCode(code_ss, is_batch);
  return code_ss.str();
}

std::string TilingLib::ProcessCubeKernelTilingFromFusedResult(
    const ascir::FusedScheduledResult &fused_schedule_result) const {
  MatMulCubeInfo cube_info;
  GE_ASSERT_SUCCESS(ExtractMatMulCubeInfoFromFusedResult(fused_schedule_result, cube_info),
                    "[Extract][MatMulCubeInfo]Failed to extract MatMul cube info from FusedScheduledResult");

  std::vector<TensorInfo> inputs;
  GE_ASSERT_SUCCESS(ExtractInputsFromMatMulNode(cube_info.matmul_node, inputs),
                    "[Extract][Inputs]Failed to extract inputs from MatMul node[%s]",
                    cube_info.matmul_node->GetName().c_str());

  std::vector<TensorInfo> outputs;
  GE_ASSERT_SUCCESS(ExtractOutputsFromMatMulNode(cube_info.matmul_node, outputs),
                    "[Extract][Outputs]Failed to extract outputs from MatMul node[%s]",
                    cube_info.matmul_node->GetName().c_str());

  CompileInfo compile_info;
  ge::PlatformInfo platform_info;
  GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetPlatformInfo(platform_info),
                    "Failed to get platform info for cube tiling.");
  compile_info.soc_version = GetFullSocVersionForCubeTiling(platform_info.soc_ver);
  compile_info.core_type = "AiCore";
  compile_info.extra_info["device_id"] = GetDeviceIdForCubeTiling();
  compile_info.aicore_num = 0;
  compile_info.aiv_num = 0;
  compile_info.op_kernel_lib = "";
  compile_info.op_impl_mode = "";

  std::vector<AttrInfo> attrs;
  PrepareMatMulAttrs(cube_info, attrs);

  return GenerateMatMulTilingCode(compile_info, inputs, outputs, attrs, cube_info.is_batch);
}

std::map<std::string, std::string> TilingLib::GenerateCVFusionStatic(
    const ::ascir::FusedScheduledResult &fused_schedule_result,
    const ascir::FusedScheduledResult &elemwise_schedule_result, const std::map<std::string, std::string> &shape_info,
    const std::string &pgo_dir, const std::string &core_num) const {
  std::stringstream ss;
  ss << TilingFuncDef(fused_schedule_result, elemwise_schedule_result, shape_info, pgo_dir, core_num) << std::endl;
  ss << TilingData("Autofuse").GenerateConst(elemwise_schedule_result, false) << std::endl;

  ss << GenCVTilingFunc();
  if (ascgen_utils::IsCubeUBFusedScheduled(elemwise_schedule_result)) {
    std::stringstream get_cv_ub_stage_size_name;
    get_cv_ub_stage_size_name << std::endl;
    get_cv_ub_stage_size_name << "extern \"C\" const char* GetCVUBFusionStageSizeName() {" << std::endl;
    if ((elemwise_schedule_result.node_idx_to_scheduled_results.size() > 0U) &&
        (elemwise_schedule_result.node_idx_to_scheduled_results[0].size() > 0U) &&
        (elemwise_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.size() > 0U) &&
        (elemwise_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size() > 0U)) {
      auto graph = elemwise_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
      for (auto axis : graph.GetAllAxis()) {
        if (axis->type == ascir::Axis::Type::kAxisTypeTileInner) {
          get_cv_ub_stage_size_name << "  return \"" << axis->name << "_size\";" << std::endl;
          GELOGD("gen GetCVUBFusionStageSizeName axis name:%s", axis->name.c_str());
        }
      }
      get_cv_ub_stage_size_name << "}" << std::endl;
      ss << get_cv_ub_stage_size_name.str();
    }
  }
  ss << GenTilingDataBlockDimAndWss();
  return {{kTilingDefAndConstIdentify, ss.str()}};
}

std::map<std::string, std::string> TilingLib::GenerateCVFusionDynamic(
    const ascir::FusedScheduledResult &fused_schedule_result,
    const ascir::FusedScheduledResult &elemwise_schedule_result, const std::map<std::string, std::string> &shape_info,
    const std::string &pgo_dir, const std::string &core_num) const {
  std::stringstream ss;
  std::stringstream call_cube_tiling;
  std::stringstream shape_symbol;
  for (auto vars : fused_schedule_result.origin_vars) {
    if (!(vars.IsConstExpr())) {
      std::string var_define = std::string(vars.Str().get());
      auto it = shape_info.find(var_define);
      if (it != shape_info.end()) {
        shape_symbol << "uint32_t " << var_define << ", ";
      }
    }
  }
  shape_symbol << "int64_t &ws_size, uint32_t &cube_block_dim, int64_t &tiling_key, uint32_t &basem, uint32_t &basen, "
                  "CVAutofuseTilingData *tiling_data";
  call_cube_tiling << "using namespace ge::autofuse;" << std::endl;
  std::string basenm_tiling_func = R"(
static int32_t g_basen_basem_align = 0;

int32_t get_g_basen_basem_align() {
  return g_basen_basem_align;
}

void set_g_basen_basem_align(int32_t value) {
  g_basen_basem_align = value;
}
)";
  call_cube_tiling << basenm_tiling_func;
  MatMulCubeInfo cube_info;
  GE_ASSERT_SUCCESS(ExtractMatMulCubeInfoFromFusedResult(fused_schedule_result, cube_info),
                    "[Extract][MatMulCubeInfo]Failed to extract MatMul cube info from FusedScheduledResult");
  AppendCvSafetyMixModeHelperDefs(call_cube_tiling, cube_info.is_batch);

  // 在CallCubeTiling函数之前定义全局变量（用于静态shape常量生成）
  call_cube_tiling << "// Global variable to store tiling bytes for const generation in static shape\n";
  call_cube_tiling << "std::vector<uint8_t> g_matmul_tiling_bytes;\n\n";

  call_cube_tiling << "extern \"C\" void CallCubeTiling(" << shape_symbol.str() << ") {" << std::endl;
  call_cube_tiling << ProcessCubeKernelTilingFromFusedResult(fused_schedule_result) << std::endl;
  call_cube_tiling << "}" << std::endl;
  ss << call_cube_tiling.str();

  if (ascgen_utils::IsCubeUBFusedScheduled(elemwise_schedule_result)) {
    std::stringstream get_cv_ub_stage_size_name;
    get_cv_ub_stage_size_name << std::endl;
    get_cv_ub_stage_size_name << "extern \"C\" const char* GetCVUBFusionStageSizeName() {" << std::endl;
    if ((elemwise_schedule_result.node_idx_to_scheduled_results.size() > 0U) &&
        (elemwise_schedule_result.node_idx_to_scheduled_results[0].size() > 0U) &&
        (elemwise_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.size() > 0U) &&
        (elemwise_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size() > 0U)) {
      auto graph = elemwise_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
      for (auto axis : graph.GetAllAxis()) {
        if (axis->type == ascir::Axis::Type::kAxisTypeTileInner) {
          get_cv_ub_stage_size_name << "  return \"" << axis->name << "_size\";" << std::endl;
          GELOGD("gen GetCVUBFusionStageSizeName axis name:%s", axis->name.c_str());
        }
      }
      get_cv_ub_stage_size_name << "}" << std::endl;
      ss << get_cv_ub_stage_size_name.str();
    }
  }
  ss << TilingFuncDef(fused_schedule_result, elemwise_schedule_result, shape_info, pgo_dir, core_num) << std::endl;

  std::map<std::string, std::string> result;
  result[kTilingDefAndConstIdentify] = ss.str();
  result[kCubeKernelTilingWrapperHpp] = kCubeKernelTilingWrapperHppValue;
  result[kCubeKernelTilingWrapperCpp] = kCubeKernelTilingWrapperInclude;
  result[kCubeKernelTilingWrapperCpp] += kCubeKernelTilingWrapperCppValue;
  return result;
}

std::map<std::string, std::string> TilingLib::GenerateCVFusion(const ascir::FusedScheduledResult &fused_schedule_result,
                                                               const std::map<std::string, std::string> &shape_info,
                                                               const std::string &pgo_dir,
                                                               const std::string &core_num) const {
  std::map<std::string, std::string> tiling_file_name_to_content;
  ascir::FusedScheduledResult elemwise_schedule_result = fused_schedule_result;
  bool is_static = IsStaticSchedResult(elemwise_schedule_result);
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result) && !is_static) {
    GE_ASSERT_SUCCESS(ascgen_utils::ProcessCubeFusionResultDynamic(elemwise_schedule_result));
  } else {
    if (ascgen_utils::IsCubeUBFusedScheduled(elemwise_schedule_result)) {
      GE_ASSERT_SUCCESS(ascgen_utils::CreateCVFusionResult(elemwise_schedule_result));
    } else if (ascgen_utils::IsCubeCommonFusedScheduled(elemwise_schedule_result)) {
      GE_ASSERT_SUCCESS(ascgen_utils::CreateCVFusionCommonResult(elemwise_schedule_result));
    }
  }
  tiling_file_name_to_content = GetTilingHeaders(elemwise_schedule_result, false, true);
  GE_CHK_BOOL_RET_STATUS_NOLOG(CheckTilingHeadersValid(tiling_file_name_to_content), tiling_file_name_to_content);

  std::map<std::string, std::string> result;
  if (is_static) {
    result = GenerateCVFusionStatic(fused_schedule_result, elemwise_schedule_result, shape_info, pgo_dir, core_num);
  } else {
    result = GenerateCVFusionDynamic(fused_schedule_result, elemwise_schedule_result, shape_info, pgo_dir, core_num);
  }

  const std::string entry_body =
      tiling_file_name_to_content[kTilingDefAndConstIdentify] + result[kTilingDefAndConstIdentify];
  const EntryTranslationUnitOptions entry_options = {
      false,
      true,
      enable_autofuse_pgo_,
      enable_autofuse_pgo_,
      enable_autofuse_pgo_ || EntryWorkspaceUsesSolver(elemwise_schedule_result),
      !ascgen_utils::IsSingleGroup(elemwise_schedule_result),
      !is_static};
  tiling_file_name_to_content[kTilingDefAndConstIdentify] = RenderEntryTranslationUnit(entry_body, entry_options);
  if (!is_static) {
    tiling_file_name_to_content[kCubeKernelTilingWrapperHpp] = result[kCubeKernelTilingWrapperHpp];
    tiling_file_name_to_content[kCubeKernelTilingWrapperCpp] = result[kCubeKernelTilingWrapperCpp];
  }
  return tiling_file_name_to_content;
}

std::map<std::string, std::string> TilingLib::Generate(const ascir::FusedScheduledResult &fused_schedule_result,
                                                       const std::map<std::string, std::string> &shape_info,
                                                       const std::string &pgo_dir, const std::string &core_num) const {
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result) &&
      !ascgen_utils::IsJustCubeFixpip(fused_schedule_result)) {
    return GenerateCVFusion(fused_schedule_result, shape_info, pgo_dir, core_num);
  }

  std::map<std::string, std::string> tiling_file_name_to_content = GetTilingHeaders(fused_schedule_result, false);
  GE_CHK_BOOL_RET_STATUS_NOLOG(CheckTilingHeadersValid(tiling_file_name_to_content), tiling_file_name_to_content);
  std::stringstream ss;
  ss << TilingFuncDef(fused_schedule_result, fused_schedule_result, shape_info, pgo_dir, core_num) << std::endl;
  // 生成GenConstTilingData方法
  ss << TilingData("Autofuse").GenerateConst(fused_schedule_result, false) << std::endl;

  ss << kTilingHeadCceKtTestGuard << std::endl;
  if (!ascgen_utils::IsJustCubeFixpip(fused_schedule_result) && CanUseTilingKey(fused_schedule_result) &&
      IsStaticSchedResult(fused_schedule_result)) {
    ss << GenGetTilingKeyForStatic();
    ss << GenGetTilingKeyKernelTypeForStatic(fused_schedule_result);
  }
  ss << "#endif" << std::endl;
  const std::string entry_body = tiling_file_name_to_content[kTilingDefAndConstIdentify] + ss.str();
  const EntryTranslationUnitOptions entry_options = {
      false,
      false,
      enable_autofuse_pgo_,
      enable_autofuse_pgo_,
      enable_autofuse_pgo_ || EntryWorkspaceUsesSolver(fused_schedule_result),
      !ascgen_utils::IsSingleGroup(fused_schedule_result)};
  tiling_file_name_to_content[kTilingDefAndConstIdentify] = RenderEntryTranslationUnit(entry_body, entry_options);

  return tiling_file_name_to_content;
}

std::string TilingLib::StubHeadersWithoutCodegenFunc() const {
  std::stringstream ss;
  ss << "#include <iostream>" << std::endl;
  ss << "#include <fstream>" << std::endl;
  ss << "#include <cinttypes>" << std::endl;
  ss << "#include <sys/syscall.h>" << std::endl;
  ss << "#include <unistd.h>" << std::endl;
  ss << "#include <securec.h>" << std::endl;
  ss << "#include \"dlog_pub.h\"" << std::endl;
  ss << "#define OP_LOGD(name, fmt, ...)" << std::endl;
  ss << "#define OP_LOGI(name, fmt, ...)" << std::endl;
  ss << "#define GE_MODULE_NAME static_cast<int32_t>(45)" << std::endl;
  ss << "inline uint64_t GetTid() {" << std::endl;
  ss << "     return static_cast<uint64_t>(syscall(__NR_gettid));" << std::endl;
  ss << "}" << std::endl;

  ss << "#define GELOGE(ERROR_CODE, fmt, ...)" << std::endl;

  ss << "#define OP_LOGE(name, fmt, ...)" << std::endl;
  ss << "#define OP_NAME \"asc0000_autofused_abs\"" << std::endl;
  ss << "#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))" << std::endl;
  ss << "#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))" << std::endl;
  ss << "#define Log(a) (log((double)(a)))" << std::endl;
  ss << "#define Pow(a, b) pow(a, b)" << std::endl;
  ss << "#define Rational(a, b) ((double)(a) / (double)(b))" << std::endl;
  ss << "" << std::endl;

  return ss.str();
}

std::string TilingLib::GetStubTilingHeaders(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  ss << StubHeadersWithoutCodegenFunc();
  ss << "namespace optiling {" << std::endl;
  ss << "extern \"C\" bool GetTiling(AutofuseTilingData& tiling_data, int32_t tilingCaseId=-1, double *perf=nullptr) {"
     << std::endl;
  ss << "  (void)perf;" << std::endl;
  ss << "  return true;" << std::endl;
  ss << "}" << std::endl;
  ss << "inline bool IsEqual(double a, double b) {" << std::endl;
  ss << "  return true;" << std::endl;
  ss << "}" << std::endl;
  if (enable_autofuse_pgo_) {
    ss << "struct SearchConfig;" << std::endl;
    ss << "bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, "
       << "AutofuseTilingData &tiling_data, int32_t tilingCaseId, AutofuseTilingData* output_tiling_data, "
       << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result)
       << "void* stream, uint32_t workspaceSize, double& out_best_perf, "
       << "std::unordered_map<int64_t, uint64_t> &workspace_map, "
       << "std::vector<uint32_t*> block_dim_vec={}, const SearchConfig *search_cfg=nullptr) {" << std::endl;
    ss << "  return true;" << std::endl;
    ss << "}" << std::endl;
    ss << "bool PGOByCoreNumSearchTilingKey(std::vector<AutofuseTilingData>& tiling_data_list, "
       << "AutofuseTilingData* tiling_data, uint32_t max_block_dim=48) {" << std::endl;
    ss << "  return true;" << std::endl;
    ss << "}" << std::endl;
  }
  ss << "}" << std::endl;
  ss << std::endl;
  return ss.str();
}

std::string TilingLib::GetStubTilingApi(const ascir::FusedScheduledResult &fused_schedule_result,
                                        bool include_pgo) const {
  std::stringstream ss;
  ss << "extern \"C\" inline bool GetTiling(AutofuseTilingData &tiling_data, int32_t tiling_case_id = -1, "
        "double *perf = nullptr) {\n";
  ss << "  (void)tiling_data; (void)tiling_case_id; (void)perf; return true;\n}\n";
  if (!include_pgo) {
    return ss.str();
  }
  const std::string common_params =
      "std::vector<AutofuseTilingDataPerf> &tiling_data_list, AutofuseTilingData &tiling_data, "
      "int32_t tiling_case_id, AutofuseTilingData *output_tiling_data, " +
      PGOSearchFuncInputOutputCallBackDef(fused_schedule_result) +
      "void *stream, uint32_t workspace_size, double &out_best_perf";
  ss << "inline bool PGOSearchTilingKey(" << common_params
     << ", std::unordered_map<int64_t, uint64_t> &workspace_map, "
        "std::vector<uint32_t *> block_dim_vec = {}, const SearchConfig *search_cfg = nullptr) {\n";
  ss << "  (void)tiling_data_list; (void)tiling_data; (void)tiling_case_id; (void)output_tiling_data;\n"
        "  (void)tensor_args; (void)stream; (void)workspace_size; (void)out_best_perf; (void)workspace_map;\n"
        "  (void)block_dim_vec; (void)search_cfg; return true;\n}\n";
  ss << "inline bool PGOSearchTilingKey(" << common_params << ", const SearchConfig *search_cfg = nullptr) {\n";
  ss << "  (void)tiling_data_list; (void)tiling_data; (void)tiling_case_id; (void)output_tiling_data;\n"
        "  (void)tensor_args; (void)stream; (void)workspace_size; (void)out_best_perf; (void)search_cfg; return true;\n"
        "}\n";
  ss << "inline bool PGOByCoreNumSearchTilingKey(std::vector<AutofuseTilingData> &tiling_data_list, "
        "AutofuseTilingData *tiling_data, uint32_t max_block_dim = 48) {\n";
  ss << "  (void)tiling_data_list; (void)tiling_data; (void)max_block_dim; return true;\n}\n";
  return ss.str();
}

std::string TilingLib::GetTilingIncludeHead(bool is_cv) const {
  std::stringstream ss;
  ss << "#ifndef __AUTOFUSE_TILING_FUNC_COMMON_H__" << std::endl;
  ss << "#define __AUTOFUSE_TILING_FUNC_COMMON_H__" << std::endl;
  ss << "#include <stdexcept>" << std::endl;
  ss << "#include <sstream>" << std::endl;
  ss << "#include <cmath>" << std::endl;
  ss << "#include <cstdint>" << std::endl;
  ss << "#include \"autofuse_tiling_data.h\"" << std::endl;
  if (is_cv) {
    ss << "int32_t get_g_basen_basem_align();" << std::endl;
    ss << "void set_g_basen_basem_align(int32_t value);" << std::endl;
  }
  ss << kTilingHeadCceKtTestGuard << std::endl;
  ss << "#include \"exe_graph/runtime/infer_shape_context.h\"" << std::endl;
  ss << "#include \"exe_graph/runtime/kernel_context.h\"" << std::endl;
  ss << "#include \"exe_graph/runtime/continuous_vector.h\"" << std::endl;
  ss << "#include \"platform/platform_infos_def.h\"" << std::endl;
  ss << "#include \"platform_ascendc.h\"" << std::endl;
  ss << "#include \"acl/acl.h\"" << std::endl;

  return ss.str();
}

void TilingLib::PopulateFallbackAtomicHeaders(std::map<std::string, std::string> &tiling_file_name_to_content,
                                              const ascir::FusedScheduledResult &fused_schedule_result,
                                              bool use_att_codegen, bool include_pgo) const {
  std::string fallback_pgo_body;
  std::string fallback_api_body;
  if (!use_att_codegen) {
    fallback_api_body = GetStubTilingApi(fused_schedule_result, include_pgo);
    if (include_pgo) {
      fallback_pgo_body = PGOProfilingCallbackDef(fused_schedule_result, "AutofuseTilingData", false);
      fallback_pgo_body +=
          "namespace optiling {\nstruct SearchConfig {\n"
          "  bool ub_threshold_enabled = true;\n  double ub_threshold = 0.0;\n"
          "  bool corenum_threshold_enabled = true;\n  double corenum_threshold = 1.0;\n"
          "  bool enable_multicore_ub_tradeoff = true;\n};\n}  // namespace optiling\n";
    }
  }
  EnsureFallbackAtomicHeaders(tiling_file_name_to_content, fallback_pgo_body, fallback_api_body);
}

std::map<std::string, std::string> TilingLib::GetTilingHeaders(const ascir::FusedScheduledResult &fused_schedule_result,
                                                               bool is_inductor_scene, bool is_cv) const {
  std::stringstream ss;
  std::string graph_name = GenValidName(fused_schedule_result.fused_graph_name.GetString());
  ss << GetTilingIncludeHead(is_cv);
  ss << "#endif" << std::endl;
  ss << std::endl;

  std::map<std::string, std::string> tiling_file_name_to_content;
  std::string tiling_name = "AutofuseTilingData";

  // just cube kernel skip GetTiling
  if (ascgen_utils::IsJustCubeFixpip(fused_schedule_result)) {
    ss << "#endif // __AUTOFUSE_TILING_FUNC_COMMON_H__" << std::endl;
    tiling_file_name_to_content[kTilingHeadIdentify] += ss.str();
    EnsureFallbackAtomicHeaders(tiling_file_name_to_content, "", GetStubTilingApi(fused_schedule_result, false));
    if (is_cv) {
      AddCvDeclarationsToApiHeader(tiling_file_name_to_content);
    }
    return tiling_file_name_to_content;
  }

  const bool use_att_codegen = this->codegen_func_ != nullptr && !IsEmptyTensorSence(fused_schedule_result);
  if ((enable_autofuse_pgo_ || is_inductor_scene) && !use_att_codegen) {
    ss << PGOProfilingCallbackDef(fused_schedule_result, tiling_name);
  }
  if (use_att_codegen) {
    std::map<std::string, std::string> options;
    tiling_file_name_to_content[kTilingHeadIdentify] += ss.str();
    options.emplace("tiling_data_type_name", tiling_name);
    options.emplace("solver_type", "AxesReorder");
    GE_CHK_BOOL_EXEC(
        this->codegen_func_(fused_schedule_result.fused_graph_name.GetString(), fused_schedule_result, options,
                            tiling_file_name_to_content, is_inductor_scene),
        GELOGE(af::FAILED, "Codegen Gen tiling func failed, graph:%s", graph_name.c_str());
        tiling_file_name_to_content[kTilingHeadIdentify] += "#endif // __AUTOFUSE_TILING_FUNC_COMMON_H__\n";
        tiling_file_name_to_content[kTilingDefAndConstIdentify] = INVALID_TILING; return tiling_file_name_to_content);
  } else {
    GELOGI("TilingLib generate stub GetTiling func start");
    ss << GetStubTilingHeaders(fused_schedule_result);
    tiling_file_name_to_content[kTilingHeadIdentify] += ss.str();
  }
  std::stringstream ss_end;
  ss_end << "#endif // __AUTOFUSE_TILING_FUNC_COMMON_H__" << std::endl;
  tiling_file_name_to_content[kTilingHeadIdentify] += ss_end.str();
  const bool include_pgo = enable_autofuse_pgo_ || is_inductor_scene;
  PopulateFallbackAtomicHeaders(tiling_file_name_to_content, fused_schedule_result, use_att_codegen, include_pgo);
  if (is_cv) {
    AddCvDeclarationsToApiHeader(tiling_file_name_to_content);
  }
  return tiling_file_name_to_content;
}

std::string TilingLib::TilingFuncDefForInductor(const ascir::FusedScheduledResult &fused_schedule_result,
                                                const ::ascir::FusedScheduledResult &elemwise_schedule_result) const {
  std::stringstream ss;
  std::string graph_name = ascgen_utils::GenValidName(elemwise_schedule_result.fused_graph_name.GetString());
  std::string tiling_func_name = "AutofuseTiling";
  std::string tiling_data_name = "AutofuseTilingData";

  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    ss << this->GenGetTilingSizeFunc(fused_schedule_result, graph_name, "CVAutofuseTilingData", true) << std::endl;
  } else {
    ss << this->GenGetTilingSizeFunc(fused_schedule_result, graph_name, tiling_data_name, true) << std::endl;
  }
  ss << this->GenGetWorkspaceSizeFunc(tiling_data_name, elemwise_schedule_result) << std::endl;
  ss << this->GenTilingFuncForInductor(fused_schedule_result, elemwise_schedule_result, tiling_func_name,
                                       tiling_data_name)
     << std::endl;
  ss << kTilingHeadCceKtTestGuard << std::endl;
  ss << this->ExternFunctionDeclare(elemwise_schedule_result, tiling_data_name) << std::endl;
  ss << "#endif" << std::endl;

  return ss.str();
}

std::string TilingLib::TilingFuncDef(const ascir::FusedScheduledResult &fused_schedule_result,
                                     const ::ascir::FusedScheduledResult &elemwise_schedule_result,
                                     const std::map<std::string, std::string> &shape_info, const std::string &pgo_dir,
                                     const std::string &core_num) const {
  std::stringstream ss;
  std::string graph_name = ascgen_utils::GenValidName(fused_schedule_result.fused_graph_name.GetString());
  std::string tiling_func_name = "AutofuseTiling";
  std::string tiling_data_name = "AutofuseTilingData";

  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result) && !IsStaticSchedResult(fused_schedule_result)) {
    ss << this->GenGetTilingSizeFunc(fused_schedule_result, graph_name, "CVAutofuseTilingData") << std::endl;
  } else {
    ss << this->GenGetTilingSizeFunc(fused_schedule_result, graph_name, tiling_data_name) << std::endl;
  }
  ss << this->GenGetWorkspaceSizeFunc(tiling_data_name, elemwise_schedule_result) << std::endl;
  ss << this->GenTilingFunc(shape_info, elemwise_schedule_result, tiling_func_name, tiling_data_name, core_num)
     << std::endl;
  ss << kTilingHeadCceKtTestGuard << std::endl;
  // 生成判断是否为静态shape的接口
  bool is_static = IsStaticSchedResult(elemwise_schedule_result);
  ss << GenCheckStaticShapeFunc(is_static);
  if (ascgen_utils::CanUseTilingKey(elemwise_schedule_result)) {
    ss << this->GenFindBestTilingKeyFunc(elemwise_schedule_result, tiling_data_name);
  }
  if (enable_autofuse_pgo_) {
    ss << GenGetTilingKeyCount(elemwise_schedule_result);
  }
  ss << this->GenExternTilingFunc(elemwise_schedule_result, shape_info, tiling_data_name, pgo_dir, core_num)
     << std::endl;
  ss << this->GenTilingCacheFunc(elemwise_schedule_result, shape_info);
  ss << this->GenDfxInputSymbolInfo(elemwise_schedule_result, shape_info);
  ss << "#endif" << std::endl;

  return ss.str();
}

void TilingLib::TilingProcessSymbolToTiling(const ascir::ImplGraph &graph, size_t graph_num, size_t res_num,
                                            size_t group_num,
                                            std::unordered_map<std::string, std::string> &ori_sym_tiling_map) const {
  for (auto size : graph.GetAllSizeVar()) {
    if (size->expr.IsConstExpr()) {
      continue;
    }
    std::string ori_sym = af::SymbolicUtils::ToString(size->expr);
    std::string tiling_var = "t.graph" + std::to_string(graph_num) + "_result" + std::to_string(res_num) + "_g" +
                             std::to_string(group_num) + "_tiling_data";
    ori_sym_tiling_map[ori_sym] = tiling_var;
    GELOGD("TilingProcessSymbolToTiling make tiling var set [%s:%s]", ori_sym.c_str(), tiling_var.c_str());
  }
}

void TilingLib::TilingMappingSymbolToTiling(const ascir::FusedScheduledResult &fused_schedule_result,
                                            std::unordered_map<std::string, std::string> &ori_sym_tiling_map) const {
  for (size_t i = 0; i < fused_schedule_result.node_idx_to_scheduled_results.size(); i++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[i];
    if ((scheduled_results.size() == 0) ||
        ((scheduled_results.size() == 1) && (scheduled_results[0].schedule_groups.size() == 1))) {
      ori_sym_tiling_map.clear();
    } else {
      for (size_t j = 0; j < scheduled_results.size(); j++) {
        for (size_t k = 0; k < scheduled_results[j].schedule_groups.size(); k++) {
          for (auto graph : scheduled_results[j].schedule_groups[k].impl_graphs) {
            TilingProcessSymbolToTiling(graph, i, j, k, ori_sym_tiling_map);
          }
        }
      }
    }
  }
}

std::string TilingLib::GenImplGraphWorkspaceSize(const ascir::ImplGraph &graph, const std::string &tiling_data,
                                                 uint32_t index) const {
  std::stringstream ss;
  std::vector<af::AscNodePtr> ws_nodes;
  af::Expression ws_size = af::Symbol(0);

  for (const auto &node : graph.GetAllNodes()) {
    if (IsOps<Workspace>(node)) {
      ws_nodes.push_back(node);
    }
  }

  ss << (index == 0U ? "    if (" : " else if(") << tiling_data << ".tiling_key == " << std::to_string(index) << ") {"
     << std::endl;
  ws_size = ascgen_utils::CalculateWorkspaceSize(ws_nodes);
  std::vector<af::Expression> ori_symbols = ws_size.FreeSymbols();
  std::vector<std::pair<af::Expression, af::Expression>> sizes;
  for (auto &ori : ori_symbols) {
    if (!(ori.IsConstExpr())) {
      std::string tiling_var = tiling_data + "." + af::SymbolicUtils::ToString(ori);
      af::Expression tiling_sizevar = af::Symbol(tiling_var.c_str());
      GELOGD("GenImplGraphWorkspaceSize make tiling var set[%s:%s]", af::SymbolicUtils::ToString(ori).c_str(),
             tiling_var.c_str());
      sizes.emplace_back(std::make_pair(ori, tiling_sizevar));
    }
  }
  af::Expression replaced_ws_size = ws_size.Replace(sizes);
  std::string ws_size_str = af::SymbolicUtils::ToString(replaced_ws_size);

  GenWorkspaceDenominatorGuards(ss, replaced_ws_size, "      ");
  ss << "      ws_size += " << ws_size_str << ";" << std::endl;
  ss << "    }" << std::endl;
  return ss.str();
}

std::string TilingLib::GenGetWorkspaceSizeFunc(const std::string &tiling,
                                               const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;

  std::unordered_map<std::string, std::string> ori_sym_tiling_map;
  TilingMappingSymbolToTiling(fused_schedule_result, ori_sym_tiling_map);

  ss << "uint32_t GetWorkspaceSize(const " << tiling << " &t) {" << std::endl;

  if (!ascgen_utils::IsJustCubeFixpip(fused_schedule_result)) {
    ss << "  using namespace optiling;" << std::endl;
  }
  ss << "  uint32_t ws_size = 0;" << std::endl;
  for (size_t graph_id = 0; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    if ((fused_schedule_result.node_idx_to_scheduled_results.size() == 1) && (scheduled_results.size() == 1) &&
        (scheduled_results[0].schedule_groups.size() == 1)) {
      auto schedule_graphs = scheduled_results[0].schedule_groups[0].impl_graphs;
      for (uint32_t i = 0; i < schedule_graphs.size(); i++) {
        ss << GenImplGraphWorkspaceSize(schedule_graphs[i], "t", i);
      }
    } else {
      for (uint32_t i = 0; i < scheduled_results.size(); i++) {
        auto schedule_groups = scheduled_results[i].schedule_groups;
        ss << (i == 0 ? "  if " : "  else if ") << "(t." << "graph" << std::to_string(graph_id)
           << "_tiling_key == " << std::to_string(i) << ") {" << std::endl;
        for (uint32_t j = 0; j < schedule_groups.size(); j++) {
          auto schedule_graphs = schedule_groups[j].impl_graphs;
          for (uint32_t k = 0; k < schedule_graphs.size(); k++) {
            std::string filed_name = "t.graph" + std::to_string(graph_id) + "_result" + std::to_string(i) + "_g" +
                                     std::to_string(j) + "_tiling_data";
            ss << GenImplGraphWorkspaceSize(schedule_graphs[k], filed_name, k);
          }
        }
        ss << "  }";
      }
    }
  }

  ss << std::endl;
  ss << "  ws_size = (ws_size + 512 - 1) / 512 * 512;" << std::endl;
  ss << "  return ws_size;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

bool TilingLib::IsVarUsedInScheduleGroup(const std::string &var_define,
                                         const ::ascir::ScheduleGroup &schedule_group) const {
  SizeVarSet used_vars;
  for (const auto &impl_graph : schedule_group.impl_graphs) {
    AscGraphInfoComplete::AppendOriginalSizeVar(impl_graph, used_vars);
  }

  // 检查 var_define 是否在 used_vars 中
  for (const auto &var : used_vars) {
    if (auto var_str = var.Str()) {
      if (std::string(var_str.get()) == var_define) {
        return true;
      }
    }
  }
  return false;
}

void TilingLib::TilingSetShapeDim(std::stringstream &tiling_set_shape_dim, const std::string &var_define,
                                  const ascir::FusedScheduledResult &fused_schedule_result,
                                  const std::string &tiling_expr) const {
  for (size_t i = 0; i < fused_schedule_result.node_idx_to_scheduled_results.size(); i++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[i];
    if ((scheduled_results.empty()) ||
        ((scheduled_results.size() == 1) && (scheduled_results[0].schedule_groups.size() == 1))) {
      // 检查变量是否被此 schedule_group 使用
      if (!IsVarUsedInScheduleGroup(var_define, scheduled_results[0].schedule_groups[0])) {
        continue;
      }
      // 简单情况：直接设置（保持原逻辑）
      tiling_set_shape_dim << "  " << tiling_expr << "set_" << var_define << "(" << var_define << ");" << std::endl;
    } else {
      for (size_t j = 0; j < scheduled_results.size(); j++) {
        for (size_t k = 0; k < scheduled_results[j].schedule_groups.size(); k++) {
          // 新增：检查变量是否被此 schedule_group 使用
          if (!IsVarUsedInScheduleGroup(var_define, scheduled_results[j].schedule_groups[k])) {
            continue;
          }
          // 原有的 var_relations 检查
          if (scheduled_results[j].var_relations.find(k) != scheduled_results[j].var_relations.end()) {
            continue;
          }
          tiling_set_shape_dim << "  " << tiling_expr << "graph" << i << "_result" << j << "_g" << k
                               << "_tiling_data.set_" << var_define << "(" << var_define << ");" << std::endl;
        }
      }
    }
  }
}

std::string TilingLib::GenPgoTilingFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                        const std::string &tiling, codegen::PgoShapeStringStream &pgo_shape_dim,
                                        bool is_inductor_scene, const std::string &core_num) const {
  std::stringstream ss;
  // 生成 AutofuseTilingWithConfig 函数
  ss << GenPgoAutofuseTiling(fused_schedule_result, pgo_shape_dim, tiling, is_inductor_scene);
  // 生成 PgoSaveTilingKey 函数
  GenPgoSaveTilingKey(ss);
  // 生成 SavePGOSearchTilingDataFunc 函数
  ss << GenSavePGOSearchTilingDataFunc(tiling);
  // 生成 SavePGOConfigTilingDataFunc 函数
  ss << GenSavePGOConfigTilingDataFunc();

  // 生成 PgoByCoreNumTilingSearch函数
  ss << GenPgoTilingSearchByCoreNum(fused_schedule_result, pgo_shape_dim, tiling, is_inductor_scene, core_num);

  // 生成 PgoTilingSearch 函数
  ss << GenPgoTilingSearchPGO(fused_schedule_result, pgo_shape_dim, tiling, is_inductor_scene, core_num);

  ss << GenPgoTilingSearch(fused_schedule_result, pgo_shape_dim, tiling);

  return ss.str();
}

std::string TilingLib::GenPgoAutofuseTiling(const ascir::FusedScheduledResult &fused_schedule_result,
                                            codegen::PgoShapeStringStream &pgo_shape_dim, const std::string &tiling,
                                            bool is_inductor_scene) const {
  std::stringstream ss;

  ss << "extern \"C\" int64_t AutofuseTilingWithConfig(const char *config_file, ";
  ss << pgo_shape_dim.shape_dim_def.str();
  ss << tiling << " *tiling, uint32_t *workspaceSize, uint32_t *blockDim,";
  ss << " ResLimit *res_limit = nullptr, int32_t tiling_case_id = -1)" << std::endl;
  ss << "{" << std::endl;

  ss << " const ResLimit *limit = (res_limit == nullptr) ? &g_no_limit_res : res_limit;" << std::endl;
  ss << pgo_shape_dim.tiling_set_shape_dim.str();
  ss << "  tiling->set_block_dim(limit->aiv_num);" << std::endl;
  if (is_inductor_scene) {
    ss << "  tiling->set_ub_size(limit->ub_size - 256);" << std::endl;
  } else {
    ss << "  tiling->set_ub_size(limit->ub_size);" << std::endl;
  }
  if (!ascgen_utils::IsJustCubeFixpip(fused_schedule_result)) {
    if (enable_autofuse_pgo_) {
      ss << "  if (!PGOGetTilingKey(config_file, *tiling)) {" << std::endl;
      ss << "    if (!optiling::GetTiling(*tiling, tiling_case_id, nullptr)) {" << std::endl;
      ss << "      return -1;" << std::endl;
      ss << "    }" << std::endl;
      ss << "  }" << std::endl;
    } else {
      ss << "  (void)config_file;" << std::endl;
      ss << "  if (!optiling::GetTiling(*tiling, tiling_case_id, nullptr)) {" << std::endl;
      ss << "    return -1;" << std::endl;
      ss << "  }" << std::endl;
    }
    ss << "  *blockDim = tiling->get_block_dim();" << std::endl;
    ss << "  using namespace optiling;" << std::endl;
  }
  ss << "  *workspaceSize = GetWorkspaceSize(*tiling);" << std::endl;
  if (!is_inductor_scene) {
    ss << "  *workspaceSize += 16 * 1024 * 1024;" << std::endl;
  }
  ss << std::endl;

  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenProfilingAllTilingData(std::string tiling_data_list_name,
                                                 std::string tiling_data_perf_list_name,
                                                 const ascir::FusedScheduledResult &fused_schedule_result,
                                                 bool is_inductor_scene) const {
  std::stringstream ss;
  ss << "  double out_cost = DBL_MAX;" << std::endl;
  ss << "  *workspaceSize = 0;" << std::endl;
  ss << "  std::unordered_set<std::string> solver_filter;" << std::endl;
  ss << "  for (const auto &tiling_data_item : " << tiling_data_list_name << ") {" << std::endl;
  ss << "    const char *ptr = reinterpret_cast<const char*>(&tiling_data_item);" << std::endl;
  ss << "    std::string key(ptr, ptr + sizeof(AutofuseTilingData));" << std::endl;
  ss << "    if (!solver_filter.insert(key).second) {" << std::endl;
  ss << "      continue;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    *workspaceSize = std::max(GetWorkspaceSize(tiling_data_item), *workspaceSize);" << std::endl;
  ss << "    AutofuseTilingDataPerf tiling_data_perf;" << std::endl;
  ss << "    tiling_data_perf.tiling_data = tiling_data_item;" << std::endl;
  ss << "    tiling_data_perf.best_perf = DBL_MAX;" << std::endl;
  ss << "    " << tiling_data_perf_list_name << ".push_back(tiling_data_perf);" << std::endl;
  ss << "  }" << std::endl;
  if (!is_inductor_scene) {
    ss << "  *workspaceSize += 16 * 1024 * 1024;" << std::endl;
  }
  ss << "  PgoConfig::Instance().batch_callback(" << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "stream, *workspaceSize, &" << tiling_data_perf_list_name << ");" << std::endl;
  return ss.str();
}

std::string TilingLib::GenPgoTilingSearchByCoreNum(const ascir::FusedScheduledResult &fused_schedule_result,
                                                   codegen::PgoShapeStringStream &pgo_shape_dim,
                                                   const std::string &tiling, bool is_inductor_scene,
                                                   const std::string &core_num) const {
  std::stringstream ss;
  ss << "extern \"C\" int64_t PgoTilingSearchByCoreNum(char *search_file, char *config_file, ";
  ss << pgo_shape_dim.shape_dim_def.str();
  ss << tiling << " *tiling, uint32_t *workspaceSize, uint32_t *blockDim,";
  ss << " ResLimit *res_limit = nullptr, ";
  ss << PGOSearchFuncInputOutputDef(fused_schedule_result);
  ss << "void *stream=nullptr, ProfilingCallback prof_callback=nullptr, ProfilingBatchCallback "
        "prof_batch_callback=nullptr) {"
     << std::endl;
  ss << "  const ResLimit *limit = (res_limit == nullptr) ? &g_no_limit_res : res_limit;" << std::endl;
  ss << pgo_shape_dim.tiling_set_shape_dim.str();
  ss << "  double best_perf = DBL_MAX;" << std::endl;
  ss << "  uint32_t max_block_dim = limit->aiv_num;" << std::endl;
  ss << GenGetMaxBlockDimFromInput(core_num);
  ss << "  using namespace optiling;" << std::endl;
  ss << "  std::vector<AutofuseTilingData> tiling_data_list;" << std::endl;
  ss << "  std::vector<AutofuseTilingDataPerf> tiling_data_perf_list;" << std::endl;
  ss << "  double axeorder_cost = DBL_MAX;" << std::endl;
  ss << "  AutofuseTiling(";
  ss << pgo_shape_dim.shape_dim_use.str();
  ss << GenGetAutoFuseTilingInput(is_inductor_scene);
  ss << "  PgoConfig::Instance().single_callback(";
  ss << PGOSearchFuncInputOutputCall(fused_schedule_result);
  ss << "stream, *workspaceSize, tiling, &axeorder_cost);" << std::endl;
  ss << "  AutofuseTilingDataPerf tiling_data_axereorder_perf;" << std::endl;
  ss << "  tiling_data_axereorder_perf.tiling_data = *tiling;" << std::endl;
  ss << "  tiling_data_axereorder_perf.best_perf = axeorder_cost;" << std::endl;
  ss << "  tiling_data_perf_list.push_back(tiling_data_axereorder_perf);" << std::endl;
  ss << "  PgoConfig::Instance().need_change_solver_run = true;" << std::endl;
  ss << "  PgoConfig::Instance().pgo_threshold_index = 0;" << std::endl;
  ss << "  while (PgoConfig::Instance().pgo_threshold_index < PgoConfig::Instance().pgo_threshold_list_size) {"
     << std::endl;
  ss << "    if (!optiling::PGOByCoreNumSearchTilingKey(tiling_data_list, tiling, max_block_dim)) {" << std::endl;
  ss << "      return -1;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    PgoConfig::Instance().pgo_threshold_index++;" << std::endl;
  ss << "  }" << std::endl;
  ss << GenProfilingAllTilingData("tiling_data_list", "tiling_data_perf_list", fused_schedule_result,
                                  is_inductor_scene);
  ss << "  best_perf = DBL_MAX;" << std::endl;
  ss << "  SavePGOSearchTilingData(search_file, tiling_data_perf_list);" << std::endl;
  ss << "  SavePGOConfigTilingData(config_file, tiling_data_perf_list, best_perf);" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

std::string TilingLib::GenPgoTilingSearch(const ascir::FusedScheduledResult &fused_schedule_result,
                                          codegen::PgoShapeStringStream &pgo_shape_dim,
                                          const std::string &tiling) const {
  std::stringstream ss;

  ss << "extern \"C\" int64_t PgoTilingSearch(char *search_file, char *config_file, ";
  ss << pgo_shape_dim.shape_dim_def.str();
  ss << tiling << " *tiling, uint32_t *workspaceSize, uint32_t *blockDim,";
  ss << " ResLimit *res_limit = nullptr, ";
  ss << PGOSearchFuncInputOutputDef(fused_schedule_result);
  ss << "void *stream=nullptr, ProfilingCallback prof_callback=nullptr, ProfilingBatchCallback "
        "prof_batch_callback=nullptr) {"
     << std::endl;
  ss << "  const char* var = std::getenv(\"AUTOFUSE_DFX_FLAGS\");" << std::endl;
  ss << "  if ((var != nullptr) && (std::string(var).find(\"autofuse_pgo_algo=pruning\") != std::string::npos)) {"
     << std::endl;
  ss << "    PgoConfig::Instance().pgo_algorithm = 0;" << std::endl;
  ss << "  } else {" << std::endl;
  ss << "    PgoConfig::Instance().pgo_algorithm = 1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  PgoConfig::Instance().single_callback = prof_callback;" << std::endl;
  ss << "  PgoConfig::Instance().batch_callback = prof_batch_callback;" << std::endl;
  ss << "  if (PgoConfig::Instance().pgo_algorithm == 0) {" << std::endl;
  ss << "    PgoTilingSearchPGO(search_file, config_file, " << pgo_shape_dim.shape_dim_use.str()
     << " tiling, workspaceSize, blockDim, res_limit, ";
  ss << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "stream, PgoConfig::Instance().single_callback, PgoConfig::Instance().batch_callback);" << std::endl;
  ss << "  } else if (PgoConfig::Instance().pgo_algorithm == 1) {" << std::endl;
  ss << "    PgoTilingSearchByCoreNum(search_file, config_file, " << pgo_shape_dim.shape_dim_use.str()
     << " tiling, workspaceSize, blockDim, res_limit, ";
  ss << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "stream, PgoConfig::Instance().single_callback, PgoConfig::Instance().batch_callback);" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenGetMaxBlockDimFromInput(const std::string &core_num) const {
  std::stringstream ss;
  if (std::stoi(core_num) != 0) {
    ss << "  auto max_core_num = " << core_num << ";" << std::endl;
    ss << "  tiling->set_block_dim(max_core_num);" << std::endl;
    ss << "  max_block_dim = max_core_num;" << std::endl;
  }
  return ss.str();
}

std::string TilingLib::GenGetAutoFuseTilingInput(bool is_inductor_scene) const {
  std::stringstream ss;
  ss << "tiling, workspaceSize, blockDim, ";
  if (is_inductor_scene) {
    ss << "res_limit);" << std::endl;
  } else {
    ss << "limit->aiv_num, limit->ub_size - 256);" << std::endl;
  }

  return ss.str();
}

std::string TilingLib::GenPgoTilingSearchPGO(const ascir::FusedScheduledResult &fused_schedule_result,
                                             codegen::PgoShapeStringStream &pgo_shape_dim, const std::string &tiling,
                                             bool is_inductor_scene, const std::string &core_num) const {
  std::stringstream ss;

  ss << "extern \"C\" int64_t PgoTilingSearchPGO(char *search_file, char *config_file, "
     << pgo_shape_dim.shape_dim_def.str() << tiling << " *tiling, uint32_t *workspaceSize, uint32_t *blockDim,"
     << " ResLimit *res_limit = nullptr, " << PGOSearchFuncInputOutputDef(fused_schedule_result)
     << "void *stream=nullptr, ProfilingCallback prof_callback=nullptr, ProfilingBatchCallback "
     << "prof_batch_callback=nullptr) {" << std::endl;

  ss << "  const ResLimit *limit = (res_limit == nullptr) ? &g_no_limit_res : res_limit;" << std::endl;
  ss << "  std::vector<AutofuseTilingDataPerf> tiling_data_list;" << std::endl;
  ss << pgo_shape_dim.tiling_set_shape_dim.str();
  ss << "  double best_perf = DBL_MAX;" << std::endl;
  ss << "  uint32_t max_block_dim = limit->aiv_num;" << std::endl;
  ss << GenGetMaxBlockDimFromInput(core_num);
  ss << "  AutofuseTiling(" << pgo_shape_dim.shape_dim_use.str() << GenGetAutoFuseTilingInput(is_inductor_scene);
  ss << "  PgoConfig::Instance().single_callback(" << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "stream, *workspaceSize, tiling, &best_perf);" << std::endl;
  ss << "  if (optiling::IsEqual(best_perf, DBL_MAX)) {" << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"axesreorder solution get perf failed %lf\", best_perf);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  AutofuseTilingDataPerf tiling_perf;" << std::endl;
  ss << "  tiling_perf.tiling_data = *tiling;" << std::endl;
  ss << "  tiling_perf.best_perf = best_perf;" << std::endl;
  ss << "  tiling_data_list.push_back(tiling_perf);" << std::endl;
  ss << "  OP_LOGD(OP_NAME, \"axesreorder solution base perf is %lf\", best_perf);" << std::endl;
  ss << "  tiling->set_block_dim(max_block_dim);" << std::endl;
  if (ascgen_utils::IsSingleGroup(fused_schedule_result)) {
    ss << "  // 不使用，仅保持接口一致" << std::endl;
    ss << "  std::unordered_map<int64_t, uint64_t> workspace_map;" << std::endl;
    ss << "  if (!optiling::PGOSearchTilingKey(tiling_data_list, *tiling, -1, tiling, "
       << PGOSearchFuncInputOutputCall(fused_schedule_result) << "stream, *workspaceSize, best_perf, workspace_map)) {"
       << std::endl;
  } else {
    ss << "  if (!optiling::PGOSearchTilingKey(tiling_data_list, *tiling, -1, tiling, "
       << PGOSearchFuncInputOutputCall(fused_schedule_result) << "stream, *workspaceSize, best_perf)) {" << std::endl;
  }
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (optiling::IsEqual(best_perf, DBL_MAX)) {" << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"pgo solution get perf failed %lf\", best_perf);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  SavePGOSearchTilingData(search_file, tiling_data_list);" << std::endl;
  ss << "  SavePGOConfigTilingData(config_file, tiling_data_list, best_perf);" << std::endl;
  ss << "  OP_LOGD(OP_NAME, \"pgo solution best perf is %lf\", best_perf);" << std::endl;
  ss << std::endl;

  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenGetResLimitStru(void) const {
  std::stringstream ss;
  ss << "struct ResLimit {" << std::endl;
  ss << "  uint32_t valid_num = 0;" << std::endl;
  ss << "  uint32_t aiv_num = 0;" << std::endl;
  ss << "  uint32_t aic_num = 0;" << std::endl;
  ss << "  uint32_t ub_size = 0;" << std::endl;
  ss << "  uint32_t resv[10];" << std::endl;
  ss << "};" << std::endl;

  ge::PlatformInfo platform_info;
  GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetPlatformInfo(platform_info));
  ss << "constexpr ResLimit g_no_limit_res = {1, " << platform_info.aiv_num << ", 0, " << platform_info.ub_size
     << ", {}};" << std::endl;

  return ss.str();
}

bool TilingLib::IsMixKernelTaskType(const ascir::FusedScheduledResult &fused_schedule_result) const {
  return fused_schedule_result.workspace_nodes.size() != 0;
}

std::string TilingLib::GenCubeFusionTilingBodyInductor(const ascir::FusedScheduledResult &fused_schedule_result,
                                                       const ::ascir::FusedScheduledResult &elemwise_schedule_result,
                                                       const std::string &shape_dim_param) const {
  std::stringstream ss;
  MatMulCubeInfo cube_info;
  GE_ASSERT_SUCCESS(ExtractMatMulCubeInfoFromFusedResult(fused_schedule_result, cube_info),
                    "[Extract][MatMulCubeInfo]Failed to extract MatMul cube info from FusedScheduledResult");
  size_t count = CalcTilingKeyCount(elemwise_schedule_result);
  ss << "  int64_t ws_size = 0;" << std::endl;
  ss << "  int64_t cube_tiling_key = 0;" << std::endl;
  ss << "  uint32_t cube_block_dim = 0;" << std::endl;
  ss << "  uint32_t basem = 0;" << std::endl;
  ss << "  uint32_t basen = 0;" << std::endl;
  ss << "  CallCubeTiling(" << shape_dim_param << "ws_size, cube_block_dim, cube_tiling_key, basem, basen, tiling);"
     << std::endl;
  ss << "  tiling->cube_tiling_key = cube_tiling_key;" << std::endl;
  ss << "  int64_t cube_tiling_key_ub = cube_tiling_key & ~0xF0;" << std::endl;
  ss << "  const int32_t ub_align_value = 32 / " << cube_info.type_size << ";" << std::endl;
  ss << "  const int32_t basen_align = (basen + ub_align_value - 1) / ub_align_value * ub_align_value;" << std::endl;
  ss << "  const int32_t basen_basem_align = (basem * basen_align) / 2 + basen_align;" << std::endl;
  ss << "  set_g_basen_basem_align(basen_basem_align);" << std::endl;
  ss << "  tiling->cube_ub_stage_size = (uint32_t)basen_basem_align;" << std::endl;
  ss << "  tiling->tiling_data.set_block_dim(limit->aiv_num);" << std::endl;
  ss << "  tiling->tiling_data.set_ub_size(limit->ub_size - 256);" << std::endl;

  ss << "  if (cube_tiling_key_ub != 1) {" << std::endl;
  ss << "    set_g_basen_basem_align(1);" << std::endl;
  ss << "    uint32_t vec_core_num = limit->aiv_num;" << std::endl;
  ss << "    tiling->tiling_data.set_block_dim(vec_core_num);" << std::endl;
  ss << "    tiling->tiling_data.set_ub_size(limit->ub_size - 256);" << std::endl;
  ss << "    double min_perf = DBL_MAX;" << std::endl;
  ss << "    size_t choice_case_id = 2U;" << std::endl;
  ss << "    for (size_t i = 2U; i < " << count << "; i++) {" << std::endl;
  ss << "      double cur_perf;" << std::endl;
  ss << "      if (!optiling::GetTiling(tiling->tiling_data, i, &cur_perf)) {" << std::endl;
  ss << "        return -1;" << std::endl;
  ss << "      }" << std::endl;
  ss << "      if (cur_perf < min_perf) {" << std::endl;
  ss << "        min_perf = cur_perf;" << std::endl;
  ss << "        choice_case_id = i;" << std::endl;
  ss << "      }" << std::endl;
  ss << "    }" << std::endl;
  ss << "    if (!optiling::GetTiling(tiling->tiling_data, choice_case_id)) {" << std::endl;
  ss << "      return -1;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    tiling->stage_size_name = tiling->tiling_data.STAGE_SIZE_NAME;" << std::endl;
  ss << "    tiling->tiling_data.set_tiling_key(tiling->tiling_data.get_tiling_key() - 2);" << std::endl;
  ss << "    // Subtract 2 from tiling_key because case 0/1 are reserved for CV UB normal/fallback tiling."
     << std::endl;
  ss << "    const bool is_cv_safety_aiv_only = is_cv_safety_aiv_only_mode(cube_tiling_key);" << std::endl;
  ss << "    const bool is_cv_safety_mix = is_cv_safety_mix_mode(cube_tiling_key);" << std::endl;
  ss << "    const bool use_launch_aic_num = is_cv_safety_blockidx_scheduled_mode(cube_tiling_key);" << std::endl;
  ss << "    uint32_t vec_block_dim = tiling->tiling_data.get_block_dim();" << std::endl;
  ss << "    int64_t vec_wss = GetWorkspaceSize(tiling->tiling_data);" << std::endl;
  ss << "    *blockDim = is_cv_safety_aiv_only ? vec_block_dim : "
        "((cube_block_dim * 2 < vec_block_dim) ? (vec_block_dim + 1) / 2 : cube_block_dim);"
     << std::endl;
  ss << "    *workspaceSize = vec_wss + ws_size;" << std::endl;
  ss << "    tiling->cv_tiling_data.fusion_mode = 1;" << std::endl;
  ss << "    tiling->cv_tiling_data.ub_mode = 0;" << std::endl;
  ss << "    tiling->cv_tiling_data.mix_mode = is_cv_safety_aiv_only ? 2 : (is_cv_safety_mix ? 1 : 0);" << std::endl;
  ss << "    tiling->cv_tiling_data.cv_aic_num = use_launch_aic_num ? *blockDim : cube_block_dim;" << std::endl;
  ss << "    tiling->cv_tiling_data.cv_aiv_num = vec_block_dim;" << std::endl;
  ss << "    tiling->cv_tiling_data.cv_vec_wss = vec_wss;" << std::endl;
  ss << "    return 0;" << std::endl;
  ss << "  }" << std::endl;

  ss << "  if (!optiling::GetTiling(tiling->tiling_data, 0)) {" << std::endl;
  ss << "    const uint32_t basen_basem_align_tmp = (uint32_t)basen_basem_align;" << std::endl;
  ss << "    set_g_basen_basem_align(basen_align);" << std::endl;
  ss << "    tiling->tiling_data.set_ub_size(limit->ub_size - 256 - basen_basem_align_tmp * " << cube_info.type_size
     << ");" << std::endl;
  ss << "    if (!optiling::GetTiling(tiling->tiling_data, 1)) {" << std::endl;
  ss << "      return -1;" << std::endl;
  ss << "    } else {" << std::endl;
  ss << "      tiling->stage_size_name = tiling->tiling_data.STAGE_SIZE_NAME;" << std::endl;
  ss << "      tiling->cv_tiling_data.fusion_mode = 0;" << std::endl;
  ss << "      tiling->cv_tiling_data.ub_mode = 1;" << std::endl;
  ss << "      tiling->cv_tiling_data.mix_mode = 0;" << std::endl;
  ss << "      tiling->cv_tiling_data.cv_aic_num = 0;" << std::endl;
  ss << "      tiling->cv_tiling_data.cv_aiv_num = 0;" << std::endl;
  ss << "      tiling->cv_tiling_data.cv_vec_wss = 0;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  } else {" << std::endl;
  ss << "    tiling->stage_size_name = tiling->tiling_data.STAGE_SIZE_NAME;" << std::endl;
  ss << "    tiling->cv_tiling_data.fusion_mode = 0;" << std::endl;
  ss << "    tiling->cv_tiling_data.ub_mode = 0;" << std::endl;
  ss << "    tiling->cv_tiling_data.mix_mode = 0;" << std::endl;
  ss << "    tiling->cv_tiling_data.cv_aic_num = 0;" << std::endl;
  ss << "    tiling->cv_tiling_data.cv_aiv_num = 0;" << std::endl;
  ss << "    tiling->cv_tiling_data.cv_vec_wss = 0;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  *blockDim = cube_block_dim;" << std::endl;
  ss << "  *workspaceSize = GetWorkspaceSize(tiling->tiling_data) + ws_size;" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

std::string TilingLib::GenTilingFuncForInductor(const ascir::FusedScheduledResult &fused_schedule_result,
                                                const ::ascir::FusedScheduledResult &elemwise_schedule_result,
                                                const std::string func, const std::string tiling) const {
  std::stringstream ss;
  codegen::PgoShapeStringStream pgo_shape_dim;
  std::vector<std::string> dynamic_shape_vars;
  std::string tiling_var = "tiling->";
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    tiling_var = "tiling->tiling_data.";
  }
  for (auto vars : elemwise_schedule_result.origin_vars) {
    if (!(vars.IsConstExpr())) {
      std::string var_define = std::string(vars.Str().get());
      dynamic_shape_vars.push_back(var_define);
      pgo_shape_dim.shape_dim_def << "uint32_t " << var_define << ", ";
      pgo_shape_dim.shape_dim_use << var_define << ", ";
      TilingSetShapeDim(pgo_shape_dim.tiling_set_shape_dim, var_define, elemwise_schedule_result, tiling_var);
    }
  }

  ss << GenGetResLimitStru();
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    MatMulCubeInfo cube_info;
    GE_ASSERT_SUCCESS(ExtractMatMulCubeInfoFromFusedResult(fused_schedule_result, cube_info),
                      "[Extract][MatMulCubeInfo]Failed to extract MatMul cube info from FusedScheduledResult");
    std::stringstream call_cube_tiling;
    call_cube_tiling << "using namespace ge::autofuse;" << std::endl;
    std::string basenm_tiling_func = R"(
static int32_t g_basen_basem_align = 0;

int32_t get_g_basen_basem_align() {
  return g_basen_basem_align;
}

void set_g_basen_basem_align(int32_t value) {
  g_basen_basem_align = value;
}
)";
    call_cube_tiling << basenm_tiling_func;
    AppendCvSafetyMixModeHelperDefs(call_cube_tiling, cube_info.is_batch);

    // 在CallCubeTiling函数之前定义全局变量（用于静态shape常量生成）
    call_cube_tiling << "// Global variable to store tiling bytes for const generation in static shape\n";
    call_cube_tiling << "std::vector<uint8_t> g_matmul_tiling_bytes;\n\n";

    call_cube_tiling << "extern \"C\" void CallCubeTiling(" << pgo_shape_dim.shape_dim_def.str()
                     << "int64_t &ws_size, uint32_t &cube_block_dim, int64_t &tiling_key, uint32_t &basem, uint32_t "
                        "&basen, CVAutofuseTilingData *tiling_data) {"
                     << std::endl;
    call_cube_tiling << "static bool g_cube_tiling_cache_valid = false;\n";
    for (const auto &var_name : dynamic_shape_vars) {
      call_cube_tiling << "static uint32_t g_cube_tiling_cache_" << var_name << " = 0;\n";
    }
    call_cube_tiling << "static int64_t g_cube_tiling_cache_ws_size = 0;\n";
    call_cube_tiling << "static uint32_t g_cube_tiling_cache_block_dim = 0;\n";
    call_cube_tiling << "static int64_t g_cube_tiling_cache_tiling_key = 0;\n";
    call_cube_tiling << "static uint32_t g_cube_tiling_cache_basem = 0;\n";
    call_cube_tiling << "static uint32_t g_cube_tiling_cache_basen = 0;\n";
    call_cube_tiling << "static uint8_t g_cube_tiling_cache_bytes[sizeof(tiling_data->matmul_tiling_data)] = {};\n";
    call_cube_tiling << "if (g_cube_tiling_cache_valid";
    for (const auto &var_name : dynamic_shape_vars) {
      call_cube_tiling << " && g_cube_tiling_cache_" << var_name << " == " << var_name;
    }
    call_cube_tiling << ") {\n";
    call_cube_tiling << "  ws_size = g_cube_tiling_cache_ws_size;\n";
    call_cube_tiling << "  cube_block_dim = g_cube_tiling_cache_block_dim;\n";
    call_cube_tiling << "  tiling_key = g_cube_tiling_cache_tiling_key;\n";
    call_cube_tiling << "  basem = g_cube_tiling_cache_basem;\n";
    call_cube_tiling << "  basen = g_cube_tiling_cache_basen;\n";
    call_cube_tiling << "  std::memcpy(tiling_data->matmul_tiling_data, g_cube_tiling_cache_bytes, "
                        "sizeof(tiling_data->matmul_tiling_data));\n";
    call_cube_tiling << "  return;\n";
    call_cube_tiling << "}\n";
    call_cube_tiling << ProcessCubeKernelTilingFromFusedResult(fused_schedule_result) << std::endl;
    call_cube_tiling << "g_cube_tiling_cache_valid = true;\n";
    for (const auto &var_name : dynamic_shape_vars) {
      call_cube_tiling << "g_cube_tiling_cache_" << var_name << " = " << var_name << ";\n";
    }
    call_cube_tiling << "g_cube_tiling_cache_ws_size = ws_size;\n";
    call_cube_tiling << "g_cube_tiling_cache_block_dim = cube_block_dim;\n";
    call_cube_tiling << "g_cube_tiling_cache_tiling_key = tiling_key;\n";
    call_cube_tiling << "g_cube_tiling_cache_basem = basem;\n";
    call_cube_tiling << "g_cube_tiling_cache_basen = basen;\n";
    call_cube_tiling << "std::memcpy(g_cube_tiling_cache_bytes, tiling_data->matmul_tiling_data, "
                        "sizeof(tiling_data->matmul_tiling_data));\n";
    call_cube_tiling << "}" << std::endl;
    ss << call_cube_tiling.str();
  }

  // AutofuseTiling
  ss << "extern \"C\" int64_t " << func << "(";
  ss << pgo_shape_dim.shape_dim_def.str();
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    ss << "CVAutofuseTilingData* tiling, uint32_t* workspaceSize, uint32_t *blockDim,";
  } else {
    ss << tiling << "* tiling, uint32_t* workspaceSize, uint32_t *blockDim,";
  }
  ss << " ResLimit *res_limit = nullptr)" << std::endl;
  ss << "{" << std::endl;

  ss << " const ResLimit *limit = (res_limit == nullptr || res_limit->aiv_num == 0) ? &g_no_limit_res : res_limit;"
     << std::endl;

  // Use first input shape pass all size variable value
  ss << pgo_shape_dim.tiling_set_shape_dim.str();

  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    return ss.str() + GenCubeFusionTilingBodyInductor(fused_schedule_result, elemwise_schedule_result,
                                                      pgo_shape_dim.shape_dim_use.str());
  }
  ss << "  tiling->set_block_dim(limit->aiv_num);" << std::endl;
  ss << "  tiling->set_ub_size(limit->ub_size - 256);" << std::endl;
  ss << "  if (!optiling::GetTiling(*tiling, -1, nullptr)) {return -1;}" << std::endl;
  ss << "  *blockDim = tiling->get_block_dim();" << std::endl;  // Only consider 48 for now
  ss << "  using namespace optiling;" << std::endl;
  ss << "  *workspaceSize = GetWorkspaceSize(*tiling);" << std::endl;
  ss << std::endl;

  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;
  if (enable_autofuse_pgo_) {
    // PGOGetTilingKey
    ss << GenPGOGetTilingKey(tiling);
    // AutofuseTilingWithConfig
    ss << GenPgoTilingFunc(elemwise_schedule_result, tiling, pgo_shape_dim, true);
  } else {
    // 生成 AutofuseTilingWithConfig 函数
    ss << GenPgoAutofuseTiling(elemwise_schedule_result, pgo_shape_dim, tiling, true);
  }
  return ss.str();
}

std::string TilingLib::GenPGOGetTilingKey(const std::string tiling) const {
  std::stringstream ss;
  ss << "bool PGOGetTilingKey(const char *config_file_path, " << tiling << " &tiling_data) {" << std::endl;
  ss << "  OP_LOGD(OP_NAME, \"PGOGetTilingKey from file:%s.\", config_file_path);" << std::endl;
  ss << "  static int best_config = 0;" << std::endl;
  ss << "  static " + tiling + " best_tiling;" << std::endl;
  ss << "  if (best_config == 0) {" << std::endl;
  ss << "    std::ifstream config_file(config_file_path);" << std::endl;
  ss << "    if (!config_file.is_open()) {" << std::endl;
  ss << "      OP_LOGD(OP_NAME, \"failed to open or not exist: %s.\", config_file_path);" << std::endl;
  ss << "      return false;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    OP_LOGD(OP_NAME, \"[Start to use tiling result]: %s.\", config_file_path);" << std::endl;
  ss << "    std::string line;" << std::endl;
  ss << "    // first line: 0:read every time; 1:read first time" << std::endl;
  ss << "    std::getline(config_file, line);" << std::endl;
  ss << "    std::istringstream iss0(line);" << std::endl;
  ss << "    int flag = -1;" << std::endl;
  ss << "    iss0 >> flag;" << std::endl;
  ss << "    OP_LOGD(OP_NAME, \"best_config %d.\", flag);" << std::endl;
  ss << "    // second line: tiling_data dumped as int32 decimals, space-separated" << std::endl;
  ss << "    std::getline(config_file, line);" << std::endl;
  ss << "    if (line.find('#') != std::string::npos) {" << std::endl;
  ss << "        line = line.substr(0, line.find('#'));" << std::endl;
  ss << "    }" << std::endl;
  ss << "    std::istringstream iss1(line);" << std::endl;
  ss << "    std::vector<int32_t> tiling_i32;" << std::endl;
  ss << "    tiling_i32.reserve((sizeof(tiling_data) + sizeof(int32_t) - 1) / sizeof(int32_t));" << std::endl;
  ss << "    int64_t tmp = 0;" << std::endl;
  ss << "    while (iss1 >> tmp) {" << std::endl;
  ss << "      tiling_i32.push_back(static_cast<int32_t>(tmp));" << std::endl;
  ss << "    }" << std::endl;
  ss << "    const size_t expect_num = (sizeof(tiling_data) + sizeof(int32_t) - 1) / sizeof(int32_t);" << std::endl;
  ss << "    tiling_i32.resize(expect_num, 0);" << std::endl;
  ss << "    memcpy_s(&tiling_data, sizeof(tiling_data), tiling_i32.data(), sizeof(tiling_data));" << std::endl;
  ss << "    config_file.close();" << std::endl;
  ss << "    if (flag == 1) {" << std::endl;
  ss << "      best_tiling = tiling_data;" << std::endl;
  ss << "      best_config = flag;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  } else {" << std::endl;
  ss << "    tiling_data = best_tiling;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return true;" << std::endl;
  ss << "}" << std::endl;
  ss << "" << std::endl;
  return ss.str();
}

std::string TilingLib::GenSavePGOSearchTilingDataFunc(const std::string tiling) const {
  std::stringstream ss;

  // SavePGOSearchTilingData
  ss << "void SavePGOSearchTilingData(char *search_file, std::vector<" << tiling << "Perf> &tiling_data_list, "
     << "std::ios::openmode mode = std::ios::out) {" << std::endl;
  ss << "  OP_LOGI(OP_NAME, \"SavePGOSearchTilingData to file:%s.\", search_file);" << std::endl;
  ss << "  std::ofstream out_file(search_file, mode);" << std::endl;
  ss << "  if (!out_file.is_open()) {" << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"Failed to open file:%s.\", search_file);" << std::endl;
  ss << "    return;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  for (auto item = tiling_data_list.rbegin(); item != tiling_data_list.rend(); ++item) {" << std::endl;
  ss << "    PgoSaveTilingKey(item->tiling_data, item->best_perf, out_file);" << std::endl;
  ss << "  }" << std::endl;
  ss << "  out_file.close();" << std::endl;
  ss << std::endl;

  ss << "  return;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenSavePGOConfigTilingDataFunc() const {
  std::stringstream ss;

  // SavePGOConfigTilingData
  ss << "void SavePGOConfigTilingData(char *file, std::vector<AutofuseTilingDataPerf> &tiling_data_list, "
     << "double best_perf, std::ios::openmode mode = std::ios::out) {" << std::endl;
  ss << "  OP_LOGI(OP_NAME, \"SavePGOConfigTilingData to file:%s.\", file);" << std::endl;
  ss << "  std::ofstream out_file(file, mode);" << std::endl;
  ss << "  if (!out_file.is_open()) {" << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"Failed to open file:%s.\", file);" << std::endl;
  ss << "    return;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (PgoConfig::Instance().pgo_algorithm == 1) {" << std::endl;
  ss << "    for (auto item : tiling_data_list) {" << std::endl;
  ss << "      if (item.best_perf < best_perf) {" << std::endl;
  ss << "        best_perf = item.best_perf;" << std::endl;
  ss << "      }" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  out_file << \"1\" << std::endl;" << std::endl;
  ss << "  for (auto item = tiling_data_list.rbegin(); item != tiling_data_list.rend(); ++item) {" << std::endl;
  ss << "    if (optiling::IsEqual(item->best_perf, best_perf)) {" << std::endl;
  ss << "      PgoSaveTilingKey(item->tiling_data, item->best_perf, out_file);" << std::endl;
  ss << "      break;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  out_file.close();" << std::endl;
  ss << std::endl;

  ss << "  return;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenTilingFunc(const std::map<std::string, std::string> &shape_info,
                                     const ascir::FusedScheduledResult &fused_schedule_result, const std::string func,
                                     const std::string tiling, const std::string &core_num) const {
  std::stringstream ss;
  codegen::PgoShapeStringStream pgo_shape_dim;
  std::string tiling_var = "tiling->";
  for (auto vars : fused_schedule_result.origin_vars) {
    if (!(vars.IsConstExpr())) {
      std::string var_define = std::string(vars.Str().get());
      auto it = shape_info.find(var_define);
      if (it != shape_info.end()) {
        // shape dim参数和tiling set shape dim匹配
        pgo_shape_dim.shape_dim_def << "uint32_t " << var_define << ", ";
        pgo_shape_dim.shape_dim_use << var_define << ", ";
        TilingSetShapeDim(pgo_shape_dim.tiling_set_shape_dim, var_define, fused_schedule_result, tiling_var);
      }
    }
  }
  ss << GenGetResLimitStru();
  // AutofuseTiling
  ss << "extern \"C\" int64_t " << func << "(";
  ss << pgo_shape_dim.shape_dim_def.str();
  ss << tiling << "* tiling, uint32_t* workspaceSize, uint32_t *blockDim,";
  ss << " uint32_t aiv_num, uint32_t ub_size)" << std::endl;
  ss << "{" << std::endl;

  // Use first input shape pass all size variable value
  ss << pgo_shape_dim.tiling_set_shape_dim.str();
  ss << "  tiling->set_block_dim(aiv_num);" << std::endl;
  ss << "  tiling->set_ub_size(ub_size);" << std::endl;

  if (!ascgen_utils::IsJustCubeFixpip(fused_schedule_result)) {
    ss << "  if (!optiling::GetTiling(*tiling, -1, nullptr)) {" << std::endl;
    ss << "      return -1;" << std::endl;
    ss << "  }" << std::endl;
  }
  ss << "  *blockDim = tiling->get_block_dim();" << std::endl;  // Only consider 48 for now
  ss << "  *workspaceSize = GetWorkspaceSize(*tiling);" << std::endl;
  ss << "  *workspaceSize += 16 * 1024 * 1024;" << std::endl;
  ss << std::endl;

  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;

  if (enable_autofuse_pgo_) {
    // PGOGetTilingKey
    ss << GenPGOGetTilingKey(tiling);
    // AutofuseTilingWithConfig
    ss << GenPgoTilingFunc(fused_schedule_result, tiling, pgo_shape_dim, false, core_num);
  } else {
    // 生成 AutofuseTilingWithConfig 函数
    ss << GenPgoAutofuseTiling(fused_schedule_result, pgo_shape_dim, tiling, false);
  }
  return ss.str();
}

static void GetTilingParse(std::string &tiling_parse, int &vector_core_num) {
  std::stringstream ss;
  ss << "bool version_is_ASCEND950 = false;" << std::endl;
  ss << "struct AfTilingParseData{" << std::endl;
  ss << " uint32_t aiv_num;" << std::endl;
  ss << " uint64_t ub_size;" << std::endl;
  ss << "};" << std::endl;

  ss << "extern \"C\" ge::graphStatus TilingParse(gert::SymbolTilingParseContext *context) {" << std::endl;
  ss << " auto platform = context->GetPlatFormInfos();" << std::endl;
  ss << " if (platform == nullptr) {" << std::endl;
  ss << " return ge::GRAPH_FAILED;" << std::endl;
  ss << " }" << std::endl;
  ss << " auto ascendc_platform = platform_ascendc::PlatformAscendC(platform);" << std::endl;
  ss << " uint32_t platform_core_num = ascendc_platform.GetCoreNumAiv();" << std::endl;

  ss << " uint32_t aiv_num = 0;" << std::endl;
  ss << " uint64_t ub_size = (184 * 1024);" << std::endl;
  if (vector_core_num == 0) {
    ss << " aiv_num = platform_core_num;" << std::endl;
  } else {
    ss << " aiv_num = std::min(platform_core_num, static_cast<uint32_t>(" << vector_core_num << "));" << std::endl;
  }

  ss << " ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);" << std::endl;

  ss << " auto extend_context = reinterpret_cast<gert::KernelContext *>(context);" << std::endl;
  ss << " auto tiling_parse_data_av = extend_context->GetOutput(0);" << std::endl;
  ss << " if (tiling_parse_data_av == nullptr) {" << std::endl;
  ss << " return ge::GRAPH_FAILED;" << std::endl;
  ss << " }" << std::endl;
  ss << " auto tiling_parse_data_ptr = new (std::nothrow) uint8_t[sizeof(AfTilingParseData)];" << std::endl;
  ss << " if (tiling_parse_data_ptr == nullptr) {" << std::endl;
  ss << " return ge::GRAPH_FAILED;" << std::endl;
  ss << " }" << std::endl;
  ss << " tiling_parse_data_av->SetWithDefaultDeleter<uint8_t[]>(tiling_parse_data_ptr);" << std::endl;

  ss << " auto tiling_parse_data = extend_context->GetOutputPointer<AfTilingParseData *>(0);" << std::endl;
  ss << " (*tiling_parse_data)->aiv_num = aiv_num;" << std::endl;
  // 当前A5获取ubsize没减256，和静态编译获取的不一致，临时规避
  ss << " if (ascendc_platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {" << std::endl;
  ss << " version_is_ASCEND950 = true;" << std::endl;
  ss << " }" << std::endl;
  ss << " ub_size -= (ascendc_platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND910 && "
        "ascendc_platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND910B && ub_size % 1024 == 0) ? "
        "256 : 0;"
     << std::endl;
  ss << " (*tiling_parse_data)->ub_size = ub_size;" << std::endl;
  ss << " return ge::GRAPH_SUCCESS;" << std::endl;
  ss << "}" << std::endl;
  tiling_parse = ss.str();
}

static void FillShapeDimInfo(const ascir::FusedScheduledResult &fused_schedule_result,
                             const std::map<std::string, std::string> &shape_info, std::stringstream &shape_dim_def,
                             std::stringstream &shape_dim_param) {
  for (const auto &vars : fused_schedule_result.origin_vars) {
    if (!vars.IsConstExpr()) {
      std::string var_define = std::string(vars.Str().get());
      auto it = shape_info.find(var_define);
      if (it != shape_info.end()) {
        shape_dim_def << "  auto " << it->first << " = " << it->second << ";" << std::endl;
        shape_dim_param << it->first << ", ";
      }
    }
  }
}

static bool HasWorkspaceInNonLastGroup(const ascir::ScheduledResult &schedule_result) {
  const auto &schedule_groups = schedule_result.schedule_groups;
  for (size_t j = 0; j < schedule_groups.size() - 1; j++) {
    for (const auto &impl_graph : schedule_groups[j].impl_graphs) {
      for (const auto &node : impl_graph.GetAllNodes()) {
        if (IsOps<Workspace>(node)) {
          return true;
        }
      }
    }
  }
  return false;
}

static std::set<size_t> GetWorkspaceNodeResultSet(const ascir::FusedScheduledResult &fused_schedule_result) {
  std::set<size_t> result;
  for (const auto &schedule_result_list : fused_schedule_result.node_idx_to_scheduled_results) {
    for (size_t i = 0; i < schedule_result_list.size(); i++) {
      if (HasWorkspaceInNonLastGroup(schedule_result_list[i])) {
        result.insert(i);
      }
    }
  }
  return result;
}

static std::string GenWorkspaceNodeCheckCode(const ascir::FusedScheduledResult &fused_schedule_result) {
  std::stringstream ss;
  std::set<size_t> schedule_result_has_workspace_node = GetWorkspaceNodeResultSet(fused_schedule_result);

  if (schedule_result_has_workspace_node.empty()) {
    return ss.str();
  }

  ss << "  std::set<size_t> schedule_result_has_workspace_node = {";
  bool first = true;
  for (const auto &result_idx : schedule_result_has_workspace_node) {
    if (!first) {
      ss << ", ";
    }
    ss << result_idx;
    first = false;
  }
  ss << "};" << std::endl;

  ss << "  if (version_is_ASCEND950 && ";
  ss << "schedule_result_has_workspace_node.count(tiling_data->graph0_tiling_key) > 0) {" << std::endl;
  ss << "    context->SetScheduleMode(1);" << std::endl;
  ss << "  }" << std::endl;

  return ss.str();
}

static std::string GenLocalMemorySizeCode() {
  std::stringstream ss;
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_ASSERT_NOTNULL(backend_spec);
  if (backend_spec->set_local_memory_size > 0) {
    ss << "  #ifdef CV_RELU_FIXPIP_MODE" << std::endl;
    ss << "  context->SetLocalMemorySize(0);" << std::endl;
    ss << "  #else" << std::endl;
    ss << "  context->SetLocalMemorySize(" << backend_spec->set_local_memory_size << ");" << std::endl;
    ss << "  #endif" << std::endl;
  }
  return ss.str();
}

static void AppendCubeFusionInitCode(std::stringstream &ss, const std::string &shape_dim_param,
                                     const MatMulCubeInfo &cube_info) {
  ss << "  auto tiling_data =  context->GetTilingData<CVAutofuseTilingData>();" << std::endl;
  ss << "  int64_t ws_size = 0;" << std::endl;
  ss << "  int64_t cube_tiling_key = 0;" << std::endl;
  ss << "  uint32_t cube_block_dim = 0;" << std::endl;
  ss << "  uint32_t basem = 0;" << std::endl;
  ss << "  uint32_t basen = 0;" << std::endl;
  ss << "  CallCubeTiling(" << shape_dim_param
     << "ws_size, cube_block_dim, cube_tiling_key, basem, basen, tiling_data);" << std::endl;
  ss << "  int64_t cube_tiling_key_ub = cube_tiling_key & ~0xF0;" << std::endl;
  ss << "  ResLimit limit;" << std::endl << "  limit.aiv_num = parse->aiv_num;" << std::endl;
  ss << "  limit.ub_size = (uint32_t)parse->ub_size;" << std::endl;
  ss << "  auto ret = ge::GRAPH_SUCCESS;" << std::endl;
  ss << "  const int32_t ub_align_value = 32 / " << cube_info.type_size << ";" << std::endl;
  ss << "  const int32_t basen_align = (basen + ub_align_value - 1) / ub_align_value * ub_align_value;" << std::endl;
  ss << "  const int32_t basen_basem_align = (basem * basen_align) / 2 + basen_align;" << std::endl;
  ss << "  tiling_data->cube_ub_stage_size = (uint32_t)basen_basem_align;" << std::endl;
}

static void AppendCubeFusionUbModeCode(std::stringstream &ss) {
  ss << "  if (cube_tiling_key_ub != 1) {" << std::endl;
  ss << "    set_g_basen_basem_align(1);" << std::endl;
  ss << "    uint32_t vec_core_num = limit.aiv_num;" << std::endl;
  ss << "    tiling_data->tiling_data.set_block_dim(vec_core_num);" << std::endl;
  ss << "    tiling_data->tiling_data.set_ub_size(limit.ub_size - 256);" << std::endl;
  ss << "    if (!optiling::GetTiling(tiling_data->tiling_data, 2)) {return ge::GRAPH_FAILED;}" << std::endl;
  ss << "    tiling_data->stage_size_name = tiling_data->tiling_data.STAGE_SIZE_NAME;" << std::endl;
  ss << "    tiling_data->tiling_data.set_tiling_key(tiling_data->tiling_data.get_tiling_key() - 2);" << std::endl;
  ss << "    // Subtract 2 from tiling_key because case 0/1 are reserved for CV UB normal/fallback tiling."
     << std::endl;
  ss << "    uint32_t vec_block_dim = tiling_data->tiling_data.get_block_dim();" << std::endl;
  ss << "    uint32_t vec_wss = GetWorkspaceSize(tiling_data->tiling_data);" << std::endl;
  ss << "    uint32_t new_block_dim = (cube_block_dim * 2 < vec_block_dim) ? (vec_block_dim + 1) / 2 : cube_block_dim;"
     << std::endl;
  ss << "    const bool is_cv_safety_mix = is_cv_safety_mix_mode(cube_tiling_key);" << std::endl;
  ss << "    const bool use_launch_aic_num = is_cv_safety_blockidx_scheduled_mode(cube_tiling_key);" << std::endl;
  ss << "    context->SetBlockDim(new_block_dim);" << std::endl;
  ss << "    *context->GetWorkspaceSizes(1) = vec_wss + ws_size;" << std::endl;
  ss << "    tiling_data->cv_tiling_data.fusion_mode = 1;" << std::endl;
  ss << "    tiling_data->cv_tiling_data.ub_mode = 0;" << std::endl;
  ss << "    tiling_data->cv_tiling_data.mix_mode = is_cv_safety_mix ? 1 : 0;" << std::endl;
  ss << "    tiling_data->cv_tiling_data.cv_aic_num = use_launch_aic_num ? new_block_dim : cube_block_dim;"
     << std::endl;
  ss << "    tiling_data->cv_tiling_data.cv_aiv_num = vec_block_dim;" << std::endl;
  ss << "    tiling_data->cv_tiling_data.cv_vec_wss = vec_wss;" << std::endl;
  ss << "  } else {" << std::endl;
}

static void AppendCubeFusionFallbackCode(std::stringstream &ss, const std::string &shape_dim_param) {
  ss << "  set_g_basen_basem_align(basen_basem_align);" << std::endl;
  ss << "  ret = AutofuseTilingWithConfig(config_file, ";
  ss << shape_dim_param;
  ss << "&(tiling_data->tiling_data), &workspace_size, &block_dim, &limit);" << std::endl;
  ss << "  if (ret == 0) {" << std::endl;
  ss << "  tiling_data->stage_size_name = tiling_data->tiling_data.STAGE_SIZE_NAME;" << std::endl;
  ss << "  context->SetBlockDim(cube_block_dim);" << std::endl;
  ss << "  *context->GetWorkspaceSizes(1) = 16 * 1024 * 1024 + ws_size;" << std::endl;
  ss << "  tiling_data->cv_tiling_data.fusion_mode = 0;" << std::endl;
  ss << "  tiling_data->cv_tiling_data.ub_mode = 0;" << std::endl;
  ss << "  tiling_data->cv_tiling_data.mix_mode = 0;" << std::endl;
  ss << "  tiling_data->cv_tiling_data.cv_aic_num = 0;" << std::endl;
  ss << "  tiling_data->cv_tiling_data.cv_aiv_num = 0;" << std::endl;
  ss << "  tiling_data->cv_tiling_data.cv_vec_wss = 0;" << std::endl;
  ss << "    } else {" << std::endl;
  ss << "      ret = ge::GRAPH_SUCCESS;" << std::endl;
  ss << "      set_g_basen_basem_align(1);" << std::endl;
  ss << "      tiling_data->tiling_data.set_block_dim(limit.aiv_num);" << std::endl;
  ss << "      tiling_data->tiling_data.set_ub_size(limit.ub_size - 256);" << std::endl;
  ss << "      if (!optiling::GetTiling(tiling_data->tiling_data, 1)) {return ge::GRAPH_FAILED;}" << std::endl;
  ss << "      tiling_data->stage_size_name = tiling_data->tiling_data.STAGE_SIZE_NAME;" << std::endl;
  ss << "      context->SetBlockDim(cube_block_dim);" << std::endl;
  ss << "      *context->GetWorkspaceSizes(1) = 16 * 1024 * 1024 + ws_size;" << std::endl;
  ss << "      tiling_data->cv_tiling_data.fusion_mode = 0;" << std::endl;
  ss << "      tiling_data->cv_tiling_data.ub_mode = 1;" << std::endl;
  ss << "      tiling_data->cv_tiling_data.mix_mode = 0;" << std::endl;
  ss << "      tiling_data->cv_tiling_data.cv_aic_num = 0;" << std::endl;
  ss << "      tiling_data->cv_tiling_data.cv_aiv_num = 0;" << std::endl;
  ss << "      tiling_data->cv_tiling_data.cv_vec_wss = 0;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  }" << std::endl;
}

static void AppendCubeFusionTilingKeyCode(std::stringstream &ss,
                                          const ascir::FusedScheduledResult &fused_schedule_result) {
  if (ascgen_utils::CanUseTilingKey(fused_schedule_result)) {
    ss << R"(
  auto tiling_key = FindBestTilingKey(tiling_data->tiling_data);
  if (tiling_key < 0) {
    return ge::GRAPH_FAILED;
  }
  context->SetTilingKey(static_cast<uint64_t>(cube_tiling_key));
)";
  }
}

std::string TilingLib::GenCubeFusionTilingBody(const ascir::FusedScheduledResult &fused_schedule_result,
                                               const std::string &shape_dim_param) const {
  std::stringstream ss;
  MatMulCubeInfo cube_info;
  GE_ASSERT_SUCCESS(ExtractMatMulCubeInfoFromFusedResult(fused_schedule_result, cube_info),
                    "[Extract][MatMulCubeInfo]Failed to extract MatMul cube info from FusedScheduledResult");
  AppendCubeFusionInitCode(ss, shape_dim_param, cube_info);
  AppendCubeFusionUbModeCode(ss);
  AppendCubeFusionFallbackCode(ss, shape_dim_param);
  ss << GenLocalMemorySizeCode();
  ss << GenWorkspaceNodeCheckCode(fused_schedule_result);
  AppendCubeFusionTilingKeyCode(ss, fused_schedule_result);
  return ss.str();
}

std::string TilingLib::GenNonCubeFusionTilingBody(const ascir::FusedScheduledResult &fused_schedule_result,
                                                  const std::string &tiling, const std::string &shape_dim_param) const {
  std::stringstream ss;
  ss << "  auto tiling_data =  context->GetTilingData<" << tiling << ">();" << std::endl;
  ss << "  ResLimit limit;" << std::endl << "  limit.aiv_num = parse->aiv_num;" << std::endl;
  ss << "  limit.ub_size = (uint32_t)parse->ub_size;" << std::endl;
  ss << "  auto ret = AutofuseTilingWithConfig(config_file, ";
  ss << shape_dim_param;
  ss << "tiling_data, &workspace_size, &block_dim, &limit);" << std::endl;
  ss << "  context->SetBlockDim(block_dim);" << std::endl;

  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result) &&
      !ascgen_utils::IsJustCubeFixpip(fused_schedule_result) &&
      !ascgen_utils::IsCubeCommonFusedScheduled(fused_schedule_result)) {
    ss << "  *context->GetWorkspaceSizes(1) = 16 * 1024 * 1024;" << std::endl;
  } else {
    ss << "  *context->GetWorkspaceSizes(1) = workspace_size;" << std::endl;
  }
  ss << GenLocalMemorySizeCode();
  ss << GenWorkspaceNodeCheckCode(fused_schedule_result);

  if (ascgen_utils::CanUseTilingKey(fused_schedule_result)) {
    ss << R"(
  auto tiling_key = FindBestTilingKey(*tiling_data);
  if (tiling_key < 0) {
    return ge::GRAPH_FAILED;
  }
  context->SetTilingKey(static_cast<uint64_t>(tiling_key));
)";
  }
  return ss.str();
}

std::string TilingLib::GenExternTilingFuncBody(const ascir::FusedScheduledResult &fused_schedule_result,
                                               const std::map<std::string, std::string> &shape_info,
                                               const std::string &tiling, const std::string &pgo_dir) const {
  std::stringstream ss;
  std::stringstream shape_dim_def;
  std::stringstream shape_dim_param;

  FillShapeDimInfo(fused_schedule_result, shape_info, shape_dim_def, shape_dim_param);
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  ss << "  auto extend_context = reinterpret_cast<const gert::KernelContext *>(context);" << std::endl;
  ss << "  auto input_data_num =  extend_context->GetInputValue<size_t>(0U);" << std::endl;
  ss << "  auto parse = extend_context->GetInputValue<AfTilingParseData*>(input_data_num + 1);" << std::endl;
  ss << shape_dim_def.str();
  ss << "  uint32_t workspace_size;" << std::endl << "  uint32_t block_dim;" << std::endl;
  if (enable_autofuse_pgo_) {
    ss << "  static const char* config_file = \"" << pgo_dir << "/" << graph_name << "_config.txt\";" << std::endl;
  } else {
    ss << "  static const char* config_file = nullptr;" << std::endl;
  }
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result) && !IsStaticSchedResult(fused_schedule_result)) {
    return ss.str() + GenCubeFusionTilingBody(fused_schedule_result, shape_dim_param.str());
  }
  return ss.str() + GenNonCubeFusionTilingBody(fused_schedule_result, tiling, shape_dim_param.str());
}

std::string TilingLib::GenExternTilingFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                           const std::map<std::string, std::string> &shape_info,
                                           const std::string tiling, const std::string &pgo_dir,
                                           const std::string &core_num) const {
  std::stringstream ss;
  std::string extern_c = "extern \"C\"";
  std::string tiling_context = R"(
namespace gert {
  class TilingSymbolEvalContext : public TilingContext {
    public:
      const gert::Tensor *GetGraphInputTensor(size_t data_index) const {
        auto *tensor = GetInputPointer<gert::Tensor>(data_index + 1);
        if (tensor == nullptr) {
          return nullptr;
        }
        return tensor;
      }
  };

  class SymbolTilingParseContext : public KernelContext {
    public:
      fe::PlatFormInfos *GetPlatFormInfos() const {
        auto platform = GetInputValue<fe::PlatFormInfos *>(0);
        if (platform == nullptr) {
          return nullptr;
        }
        return platform;
      }
  };
})";

  ss << tiling_context << std::endl;
  std::string tiling_parse_def;
  int vector_core_num = std::atoi(core_num.c_str());
  GetTilingParse(tiling_parse_def, vector_core_num);
  ss << tiling_parse_def << std::endl;
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result) && IsStaticSchedResult(fused_schedule_result)) {
    ss << extern_c << " ge::graphStatus TilingFunc(gert::TilingSymbolEvalContext *context)" << std::endl;
    ss << "{" << std::endl;
    ss << GenExternTilingFuncBody(fused_schedule_result, shape_info, tiling, pgo_dir);
    ss << "  context->SetBlockDim(CUBE_BLOCK_DIM);" << std::endl;
    ss << "  context->SetTilingKey(static_cast<uint64_t>(CUBE_TILING_KEY));" << std::endl;
    ss << "  return ret;" << std::endl;
    ss << "}" << std::endl;
    ss << extern_c << " ge::graphStatus TilingFuncVec(gert::TilingSymbolEvalContext *context)" << std::endl;
  } else {
    ss << extern_c << " ge::graphStatus TilingFunc(gert::TilingSymbolEvalContext *context)" << std::endl;
  }
  ss << "{" << std::endl;
  if (!IsEmptyTensorSence(fused_schedule_result)) {
    ss << GenExternTilingFuncBody(fused_schedule_result, shape_info, tiling, pgo_dir);
    ss << "  return ret;" << std::endl;
  } else {
    ss << "  context->SetBlockDim(1);" << std::endl;
    ss << "  *context->GetWorkspaceSizes(1) = 0;" << std::endl;
    ss << "  return ge::GRAPH_SUCCESS;" << std::endl;
  }
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenGetTilingSizeFunc(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                            const std::string graph_name, const std::string tiling,
                                            bool is_inductor) const {
  std::stringstream ss;
  GELOGI("start %s Gen GetTilingDataSize function", graph_name.c_str());
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    bool is_batch = false;
    bool is_conv = false;
    std::string input_type;
    std::string output_type;
    GE_ASSERT_SUCCESS(ascgen_utils::GetCubeInfo(fused_schedule_result, is_batch, is_conv, input_type, output_type),
                      "Failed to get cube info from FusedScheduledResult");
    std::string struct_name = is_batch ? "BatchMatMulV3BasicTilingData" : "MatMulV3BasicTilingData";
    if (is_conv) {
      struct_name = "Conv2DTilingData";
    }
    ss << "extern \"C\" size_t GetTilingDataSize()" << std::endl;
    ss << "{" << std::endl;
    if (is_inductor) {
      ss << "  return sizeof(" << tiling << ");" << std::endl;
    } else {
      ss << "  return sizeof(" << struct_name << ");" << std::endl;
    }
    ss << "}" << std::endl;
    ss << "extern \"C\" size_t GetTilingDataSizeVec()" << std::endl;
    ss << "{" << std::endl;
    ss << "  return sizeof(" << tiling << ");" << std::endl;
    ss << "}" << std::endl;
    return ss.str();
  }

  ss << "extern \"C\" size_t GetTilingDataSize()" << std::endl;
  ss << "{" << std::endl;
  ss << "  return sizeof(" << tiling << ");" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::ExternFunctionDeclare(const ascir::FusedScheduledResult &fused_schedule_result,
                                             const std::string tiling) const {
  (void)tiling;
  std::stringstream ss;

  // 生成判断是否为静态shape的接口
  bool is_static = IsStaticSchedResult(fused_schedule_result);
  ss << GenCheckStaticShapeFunc(is_static);
  return ss.str();
}

std::string TilingLib::PGOTensorArgsDef() const {
  std::stringstream ss;
  ss << "#ifndef AUTOFUSE_PGO_TENSOR_ARGS_DEFINED" << std::endl;
  ss << "#define AUTOFUSE_PGO_TENSOR_ARGS_DEFINED" << std::endl;
  ss << "struct PgoTensorArgs {" << std::endl;
  ss << "  void **inputs = nullptr;" << std::endl;
  ss << "  uint32_t input_num = 0;" << std::endl;
  ss << "  void **outputs = nullptr;" << std::endl;
  ss << "  uint32_t output_num = 0;" << std::endl;
  ss << "};" << std::endl;
  ss << "#endif" << std::endl;
  return ss.str();
}

void TilingLib::AppendPgoConfigDef(std::stringstream &ss) const {
  ss << "class PgoConfig {" << std::endl;
  ss << "public:" << std::endl;
  ss << "  static PgoConfig& Instance() {" << std::endl;
  ss << "    static PgoConfig instance;" << std::endl;
  ss << "    return instance;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  void ResetRuntimeOverrides() {" << std::endl;
  ss << "    need_change_solver_run = false;" << std::endl;
  ss << "    pgo_threshold_index = 0;" << std::endl;
  ss << "    pgo_ub_threshold_list = {0.2, 0.1, 0, 0.05, 0.1};" << std::endl;
  ss << "    pgo_corenum_threshold_list = {0.4, 0.4, 1, 1, 0.8};" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ProfilingCallback single_callback;" << std::endl;
  ss << "  ProfilingBatchCallback batch_callback;" << std::endl;
  ss << "  PgoTensorArgs *tensor_args = nullptr;" << std::endl;
  ss << "  int32_t pgo_algorithm = 1; // 0 for pruning, 1 for core num" << std::endl;
  ss << "  bool need_change_solver_run = false;" << std::endl;
  ss << "  size_t pgo_threshold_index = 0;" << std::endl;
  ss << "  constexpr static size_t pgo_threshold_list_size = 5;" << std::endl;
  ss << "  std::array<double, pgo_threshold_list_size> pgo_ub_threshold_list{0.2, 0.1, 0, 0.05, 0.1};" << std::endl;
  ss << "  std::array<double, pgo_threshold_list_size> pgo_corenum_threshold_list{0.4, 0.4, 1, 1, 0.8};" << std::endl;
  ss << "private:" << std::endl;
  ss << "  PgoConfig() = default;" << std::endl;
  ss << "  ~PgoConfig() = default;" << std::endl;
  ss << "  PgoConfig(const PgoConfig &) = delete;" << std::endl;
  ss << "  PgoConfig &operator=(const PgoConfig &) = delete;" << std::endl;
  ss << "};" << std::endl;
  ss << "class PgoConfigRuntimeGuard {" << std::endl;
  ss << "public:" << std::endl;
  ss << "  PgoConfigRuntimeGuard() { PgoConfig::Instance().ResetRuntimeOverrides(); }" << std::endl;
  ss << "  ~PgoConfigRuntimeGuard() { PgoConfig::Instance().ResetRuntimeOverrides(); }" << std::endl;
  ss << "};" << std::endl;
}

std::string TilingLib::PGOProfilingCallbackDef(const ascir::FusedScheduledResult &fused_schedule_result,
                                               const std::string tiling, bool include_headers) const {
  std::stringstream ss;

  if (include_headers) {
    ss << "#include <cfloat>" << std::endl;
    ss << "#include <cstdint>" << std::endl;
    ss << "#include <vector>" << std::endl;
    ss << "#include <unordered_set>" << std::endl;
    ss << "#include <array>" << std::endl;
    ss << std::endl;
  }
  ss << PGOTensorArgsDef();
  ss << "typedef long int (*ProfilingCallback)(";
  ss << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result);
  ss << "void *stream, uint32_t workspaceSize, " << tiling << " *tiling_data, double *cost_time);" << std::endl;
  ss << "typedef long int (*ProfilingBatchCallback)(";
  ss << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result);
  ss << "void *stream, uint32_t workspaceSize, std::vector<AutofuseTilingDataPerf> *profiles);" << std::endl;
  AppendPgoConfigDef(ss);
  ss << std::endl;

  return ss.str();
}

std::string TilingLib::PGOSearchFuncInputOutputCallBackDef(
    const ascir::FusedScheduledResult &fused_schedule_result) const {
  (void)fused_schedule_result;
  return "PgoTensorArgs *tensor_args, ";
}

std::string TilingLib::PGOSearchFuncInputOutputDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  (void)fused_schedule_result;
  return "PgoTensorArgs *tensor_args = nullptr, ";
}

std::string TilingLib::PGOSearchFuncInputOutputCall(const ascir::FusedScheduledResult &fused_schedule_result) const {
  (void)fused_schedule_result;
  return "tensor_args, ";
}

std::string TilingLib::PGOSearchStructInputOutputDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for ([[maybe_unused]] auto &input : fused_schedule_result.input_nodes) {
    ss << "  uint64_t input" << index++ << ";" << std::endl;
  }
  index = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(node)) {
      ss << "  uint64_t output" << index++ << ";" << std::endl;
    }
  }

  return ss.str();
}

std::string TilingLib::PGOSearchTensorInputOutputDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for ([[maybe_unused]] auto &input : fused_schedule_result.input_nodes) {
    ss << "void* input" << index++ << ";" << std::endl;
  }
  index = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(node)) {
      ss << "void* output" << index++ << ";" << std::endl;
    }
  }
  ss << "uint64_t ffts;" << std::endl;
  const size_t input_array_size = std::max<size_t>(fused_schedule_result.input_nodes.size(), 1UL);
  const uint32_t output_count = PGOSearchFuncGetOutputCount(fused_schedule_result);
  const uint32_t output_array_size = std::max<uint32_t>(output_count, 1U);
  ss << "void *g_pgo_inputs[" << input_array_size << "] = {nullptr};" << std::endl;
  ss << "void *g_pgo_outputs[" << output_array_size << "] = {nullptr};" << std::endl;
  ss << "PgoTensorArgs g_pgo_tensor_args = {g_pgo_inputs, " << fused_schedule_result.input_nodes.size()
     << "U, g_pgo_outputs, " << output_count << "U};" << std::endl;

  return ss.str();
}

std::string TilingLib::PGOSearchTensorArgsUpdateDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for ([[maybe_unused]] auto &input : fused_schedule_result.input_nodes) {
    ss << "  g_pgo_inputs[" << index << "] = input" << index << ";" << std::endl;
    index++;
  }
  index = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(node)) {
      ss << "  g_pgo_outputs[" << index << "] = output" << index << ";" << std::endl;
      index++;
    }
  }
  ss << "  g_pgo_tensor_args.inputs = g_pgo_inputs;" << std::endl;
  ss << "  g_pgo_tensor_args.input_num = " << fused_schedule_result.input_nodes.size() << "U;" << std::endl;
  ss << "  g_pgo_tensor_args.outputs = g_pgo_outputs;" << std::endl;
  ss << "  g_pgo_tensor_args.output_num = " << PGOSearchFuncGetOutputCount(fused_schedule_result) << "U;" << std::endl;
  return ss.str();
}

std::string TilingLib::PGOSearchFuncInputOutputStructAssignDef(const ascir::FusedScheduledResult &fused_schedule_result,
                                                               const std::string &struct_var_name) const {
  std::stringstream ss;
  int index = 0;
  for ([[maybe_unused]] auto &input : fused_schedule_result.input_nodes) {
    ss << struct_var_name << ".input" << index << " = reinterpret_cast<uint64_t>(tensor_args->inputs[" << index << "]);"
       << std::endl;
    index++;
  }
  index = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(node)) {
      ss << struct_var_name << ".output" << index << " = reinterpret_cast<uint64_t>(tensor_args->outputs[" << index
         << "]);" << std::endl;
      index++;
    }
  }
  return ss.str();
}

uint32_t TilingLib::PGOSearchFuncGetInputOutputCount(const ascir::FusedScheduledResult &fused_schedule_result) const {
  return fused_schedule_result.input_nodes.size() + PGOSearchFuncGetOutputCount(fused_schedule_result);
}

uint32_t TilingLib::PGOSearchFuncGetOutputCount(const ascir::FusedScheduledResult &fused_schedule_result) const {
  uint32_t count = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(node)) {
      count++;
    }
  }
  return count;
}

std::string TilingLib::CalculateTensorMemorySizeStr(const ascir::TensorAttr &tensor) const {
  static const std::unordered_map<ge::DataType, af::Expression> type_size_map = {
      {ge::DT_FLOAT, af::Expression::Parse("4")},    // sizeof(float)
      {ge::DT_FLOAT16, af::Expression::Parse("2")},  // fp16 is 2 bytes
      {ge::DT_INT8, af::Expression::Parse("1")},     // sizeof(int8_t)
      {ge::DT_INT16, af::Expression::Parse("2")},    // sizeof(int16_t)
      {ge::DT_INT32, af::Expression::Parse("4")},    // sizeof(int32_t)
      {ge::DT_INT64, af::Expression::Parse("8")},    // sizeof(int64_t)
      {ge::DT_UINT8, af::Expression::Parse("1")},    // sizeof(uint8_t)
      {ge::DT_UINT16, af::Expression::Parse("2")},   // sizeof(uint16_t)
      {ge::DT_UINT32, af::Expression::Parse("4")},   // sizeof(uint32_t)
      {ge::DT_UINT64, af::Expression::Parse("8")},   // sizeof(uint64_t)
      {ge::DT_DOUBLE, af::Expression::Parse("8")},   // sizeof(double)
      {ge::DT_BF16, af::Expression::Parse("2")},     // bf16 is 2 bytes
      {ge::DT_BOOL, af::Expression::Parse("1")}      // sizeof(bool)
  };
  const auto dtype = tensor.attr.dtype.operator ge::DataType();
  auto it = type_size_map.find(dtype);
  if (it == type_size_map.end()) {
    GELOGE(ge::GRAPH_FAILED, "Unsupported data type: %d", static_cast<int32_t>(dtype));
    return "0";
  }
  af::Expression type_size = it->second;
  if (tensor.attr.repeats.empty() || tensor.attr.strides.empty()) {
    GELOGE(ge::GRAPH_FAILED, "Empty repeats or strides for tensor when calculating memory size");
    return "0";
  }

  // 跳过brc场景下的0 strides
  size_t stride_index = 0UL;
  for (; stride_index < tensor.attr.strides.size(); ++stride_index) {
    if (tensor.attr.strides[stride_index] != af::ops::Zero) {
      break;
    }
    GELOGD("Tensor stride %zu is zero, try to skip to next non-zero stride.", stride_index);
  }

  // 全为brc轴时，元素个数为1，其他情况下为repeats[index] * strides[index]
  af::Expression element_size = af::ops::One;
  if (stride_index < tensor.attr.repeats.size() && stride_index < tensor.attr.strides.size()) {
    element_size = af::sym::Mul(tensor.attr.repeats[stride_index], tensor.attr.strides[stride_index]).Simplify();
  }
  af::Expression need_malloc_size = af::sym::Mul(element_size, type_size).Simplify();
  GELOGD("Tensor element size: %s, need malloc size: %s", element_size.Str().get(), need_malloc_size.Str().get());
  return std::string(need_malloc_size.Str().get());
}

std::string TilingLib::PGOSearchTensorMallocDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for (auto &input : fused_schedule_result.input_nodes) {
    if (input->GetOutNodesPtr().empty()) {
      continue;
    }
    af::Node *out_node = input->GetOutNodesPtr()[0];
    af::AscNode *asc_out_node = static_cast<af::AscNode *>(out_node);
    ss << "  size_t input" << index << "_size = " << CalculateTensorMemorySizeStr(asc_out_node->outputs[0]) << ";"
       << std::endl;
    ss << "  ret = aclrtMalloc(&input" << index << ", input" << index << "_size, ACL_MEM_MALLOC_HUGE_FIRST);"
       << std::endl;
    ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
    ss << "    DLOGE(\"aclrtMalloc input" << index << " failed. ERROR: %d\", ret);" << std::endl;
    ss << "    return FAILED;" << std::endl;
    ss << "  }" << std::endl;
    index++;
  }
  index = 0;
  for (auto &output : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(output)) {
      ss << "  size_t output" << index << "_size = " << CalculateTensorMemorySizeStr(output->inputs[0]) << ";"
         << std::endl;
      ss << "  ret = aclrtMalloc(&output" << index << ", output" << index << "_size, ACL_MEM_MALLOC_HUGE_FIRST);"
         << std::endl;
      ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
      ss << "    DLOGE(\"aclrtMalloc output" << index << " failed. ERROR: %d\", ret);" << std::endl;
      ss << "    return FAILED;" << std::endl;
      ss << "  }" << std::endl;
      index++;
    }
  }
  return ss.str();
}

std::string TilingLib::PGOSearchTensorFreeDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for ([[maybe_unused]] auto &input : fused_schedule_result.input_nodes) {
    ss << "  if (input" << index << " != nullptr) {" << std::endl;
    ss << "    ret = aclrtFree(input" << index << ");" << std::endl;
    ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
    ss << "      DLOGW(\"aclrtFree input" << index << " failed. ERROR: %d\", ret);" << std::endl;
    ss << "    }" << std::endl;
    ss << "    input" << index << " = nullptr;" << std::endl;
    ss << "  }" << std::endl;
    index++;
  }
  index = 0;
  for (auto &output : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(output)) {
      ss << "  if (output" << index << " != nullptr) {" << std::endl;
      ss << "    ret = aclrtFree(output" << index << ");" << std::endl;
      ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
      ss << "      DLOGW(\"aclrtFree output" << index << " failed. ERROR: %d\", ret);" << std::endl;
      ss << "    }" << std::endl;
      ss << "    output" << index << " = nullptr;" << std::endl;
      ss << "  }" << std::endl;
      index++;
    }
  }
  return ss.str();
}

std::string TilingLib::InferShapeDef(const ascir::HintGraph &graph) const {
  (void)graph;
  std::stringstream ss;

  ss << "namespace ge {" << std::endl;
  ss << "static ge::graphStatus InferShape(gert::InferShapeContext* context)" << std::endl;
  ss << "{" << std::endl;
  ss << "    return GRAPH_SUCCESS;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenCheckStaticShapeFunc(bool is_static) const {
  std::stringstream ss;
  ss << "extern \"C\" bool AutofuseIsStaticShape() {" << std::endl;
  ss << "  return " << (is_static ? "true" : "false") << ";" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

// 生成tiling缓存需要的接口
std::string TilingLib::GenTilingCacheFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                          const std::map<std::string, std::string> &shape_info) const {
  std::stringstream ss;
  std::string extern_c = "extern \"C\"";
  ss << extern_c << " ge::graphStatus GetSymbolTilingCacheKey(gert::TilingSymbolEvalContext *context)" << std::endl;
  ss << "{" << std::endl;
  ss << "  auto kernel_context = reinterpret_cast<gert::KernelContext *>(context);" << std::endl;
  ss << "  auto symbol_src_vec = kernel_context->GetOutputPointer<gert::TypedContinuousVector<int64_t>>(0U);"
     << std::endl;
  ss << "  if (symbol_src_vec == nullptr) {" << std::endl;
  ss << "    return ge::GRAPH_FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << std::endl;

  uint32_t index = 0U;
  std::stringstream ss_tmp;

  for (const auto &vars : fused_schedule_result.origin_vars) {
    if (!(vars.IsConstExpr())) {
      std::string var_define = std::string(vars.Str().get());
      auto it = shape_info.find(var_define);
      if (it != shape_info.end()) {
        // shape dim 定义赋值和传参匹配
        ss_tmp << "  auto " << it->first << " = " << it->second << ";" << std::endl;
        ss_tmp << "  symbol_src_vec->MutableData()[" << std::to_string(index) << "] = " << it->first << ";"
               << std::endl;
        ss_tmp << std::endl;
        index++;
      }
    }
  }

  std::stringstream ss_size_chk;
  ss_size_chk << "  if (symbol_src_vec->GetCapacity() < " << std::to_string(index) << ") {" << std::endl;
  ss_size_chk << "    return ge::GRAPH_FAILED;" << std::endl;
  ss_size_chk << "  }" << std::endl;
  ss_size_chk << std::endl;
  ss << ((index != 0U) ? ss_size_chk.str() : "");

  ss << ss_tmp.str();
  ss << "  symbol_src_vec->SetSize(" << std::to_string(index) << ");" << std::endl;
  ss << "  return ge::GRAPH_SUCCESS;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenDfxInputSymbolInfo(const ascir::FusedScheduledResult &fused_schedule_result,
                                             const std::map<std::string, std::string> &shape_info) const {
  std::stringstream ss;
  ss << R"(extern "C" ge::graphStatus DfxInputSymbolInfo(gert::TilingSymbolEvalContext *context, char *out_symbol_info, size_t size)
{
  if (out_symbol_info == nullptr || size == 0) {
    return ge::GRAPH_SUCCESS;
  }
  std::string symbol_info;)"
     << std::endl;

  bool first_sym = true;
  for (const auto &vars : fused_schedule_result.origin_vars) {
    if (!(vars.IsConstExpr())) {
      std::string var_define = std::string(vars.Str().get());
      auto it = shape_info.find(var_define);
      if (it != shape_info.end()) {
        ss << "  auto " << it->first << " = " << it->second << ";" << std::endl;
        ss << "  symbol_info += (\"";
        if (first_sym) {
          first_sym = false;
        } else {
          ss << ", ";
        }
        ss << it->first << ": \" + std::to_string(" << it->first << "));" << std::endl;
        ss << std::endl;
      }
    }
  }
  ss << R"(
  if (symbol_info.empty()) {
    out_symbol_info[0] = '\0';
    return ge::GRAPH_SUCCESS;
  }
  symbol_info += ".";
  if (strncpy_s(out_symbol_info, size, symbol_info.c_str(), std::min(symbol_info.size(), size - 1)) != 0) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
})" << std::endl;
  return ss.str();
}

std::string TilingLib::GenFindBestTilingKeyFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                                const std::string &tiling_data_name) const {
  std::stringstream ss;
  ss << "extern \"C\" int64_t FindBestTilingKey(" << tiling_data_name << " &t)" << std::endl;
  ss << "{" << std::endl;
  if (ascgen_utils::IsSingleGroup(fused_schedule_result)) {
    auto schedule_graphs = fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
    for (uint32_t i = 0; i < schedule_graphs.size(); i++) {
      auto tiling_key = std::to_string(i);
      ss << (i == 0U ? "  if (" : "  } else if (") << ("t.tiling_key == " + tiling_key + ") {") << std::endl;
      ss << "    return " + tiling_key + ";" << std::endl;
    }
    ss << "  }" << std::endl;
  } else {
    GenMulGroupFindBestTilingKey(fused_schedule_result, ss);
  }
  ss << "  return -1;" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

std::string TilingLib::GenGetTilingKeyCount(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  size_t count = CalcTilingKeyCount(fused_schedule_result);
  ss << "extern \"C\" uint64_t GetTilingKeyCount()" << std::endl;
  ss << "{" << std::endl;
  ss << "  return " << std::to_string(count) << ";" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

std::string TilingLib::GenGetTilingKeyForStatic() const {
  std::stringstream ss;
  ss << "extern \"C\" int64_t GetTilingKeyForStatic()" << std::endl;
  ss << "{" << std::endl;
  ss << "  return FindBestTilingKey(TilingDataValue);" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

std::string TilingLib::GenGetTilingKeyKernelTypeForStatic(
    const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  ss << "std::string kernel_type;" << std::endl;
  ss << "extern \"C\" const char* GetTilingKeyKernelTypeForStatic()" << std::endl;
  ss << "{" << std::endl;
  ss << "  const std::map<int64_t, std::string> kernel_type_map = {" << std::endl;
  uint32_t tiling_key = 0U;
  for (const auto &scheduled_results : fused_schedule_result.node_idx_to_scheduled_results) {
    for (const auto &scheduled_result : scheduled_results) {
      auto schedule_groups = scheduled_result.schedule_groups;
      std::vector<std::vector<bool>> per_group_conditions;
      for (const auto &schedule_group : schedule_groups) {
        auto schedule_graphs = schedule_group.impl_graphs;
        std::vector<bool> conditions;
        for (const auto &schedule_graph : schedule_graphs) {
          bool has_workspace_node = HasWorkSpaceNode(schedule_graph);
          conditions.emplace_back(has_workspace_node);
        }
        per_group_conditions.emplace_back(std::move(conditions));
      }
      std::vector<bool> current;
      CodegenTilingKeyKerneType(ss, per_group_conditions, current, 0, tiling_key);
    }
  }
  ss << "  };" << std::endl;
  ss << R"(
  auto tiling_key = FindBestTilingKey(TilingDataValue);
  auto it = kernel_type_map.find(tiling_key);
  if (it != kernel_type_map.end()) {
    kernel_type = it->second;
  }
  return kernel_type.c_str();
})" << std::endl;
  return ss.str();
}

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
  ss << "  tiling_datas.clear();" << std::endl;
  ss << "  workspaces.clear();" << std::endl;
  ss << "  block_dims.clear();" << std::endl;
  ss << "  if (topn <= 0) {" << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"GenerateTopnSolutions failed: invalid topn.\");" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
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
