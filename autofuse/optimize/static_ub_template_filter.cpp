/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "static_ub_template_filter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "common/platform_context.h"
#include "common/ub_expr/asc_graph_ub_expr_builder.h"
#include "common/ub_expr/ub_expr_utils.h"
#include "rt_external_base.h"

namespace optimize {
namespace {
constexpr const char *kLogPrefix = "[StaticUbTemplateFilter]";
constexpr size_t kMaxExprLogLength = 512UL;
constexpr uint32_t kMaxRuntimeSpecValueLen = 16U;
constexpr const char *kAicoreSpecLabel = "AICoreSpec";
constexpr const char *kUbSizeSpecKey = "ub_size";

enum class EvalStatus {
  kKnown,
  kUnknown,
  kFailed,
};

struct EvalResult {
  EvalStatus status = EvalStatus::kUnknown;
  int64_t min_ub_usage = 0;
  af::Expression origin_expr;
  af::Expression min_expr;
};

struct TemplatePosition {
  size_t node_idx = 0UL;
  size_t result_idx = 0UL;
  size_t group_idx = 0UL;
  size_t impl_idx = 0UL;
};

struct FilterState {
  ascir::FusedScheduledResult &fused_scheduled_result;
  int64_t ub_size = 0;
};

struct UbLimitResult {
  bool has_ub_limit = false;
  int64_t ub_size = 0;
};

std::string ExprToString(const af::Expression &expr) {
  if (!expr.IsValid()) {
    return "<invalid>";
  }
  const auto expr_str_ptr = expr.Str();
  if (expr_str_ptr == nullptr || expr_str_ptr.get() == nullptr) {
    return "<null>";
  }
  std::string expr_str = expr_str_ptr.get();
  if (expr_str.size() <= kMaxExprLogLength) {
    return expr_str;
  }
  return expr_str.substr(0UL, kMaxExprLogLength) + "...";
}

const char *GetGraphName(const ge::AscendString &graph_name) {
  const auto *name = graph_name.GetString();
  return (name == nullptr || name[0] == '\0') ? "<null>" : name;
}

std::string BuildTemplateName(size_t node_idx, size_t result_idx, size_t group_idx, size_t impl_idx) {
  return "node" + std::to_string(node_idx) + "_result" + std::to_string(result_idx) + "_group" +
         std::to_string(group_idx) + "_impl" + std::to_string(impl_idx);
}

void AppendReplacement(const af::Expression &src, const af::Expression &dst,
                       std::vector<std::pair<af::Expression, af::Expression>> &replacements) {
  if (!src.IsValid() || !dst.IsValid()) {
    return;
  }
  const auto iter =
      std::find_if(replacements.cbegin(), replacements.cend(), [&src](const auto &item) { return item.first == src; });
  if (iter == replacements.cend()) {
    replacements.emplace_back(src, dst);
  }
}

void AppendVarOneReplacements(const std::vector<af::Expression> &vars,
                              std::vector<std::pair<af::Expression, af::Expression>> &replacements) {
  for (const auto &var : vars) {
    AppendReplacement(var, af::sym::kSymbolOne, replacements);
  }
}

void BuildReplacements(const ascir::UbExprContext &context,
                       std::vector<std::pair<af::Expression, af::Expression>> &container_replacements,
                       std::vector<std::pair<af::Expression, af::Expression>> &var_replacements) {
  for (const auto &item : context.container_expr) {
    AppendReplacement(item.first, item.second, container_replacements);
  }
  for (const auto &item : context.var_min_values) {
    AppendReplacement(item.first, item.second, var_replacements);
  }
  for (const auto &item : context.const_vars) {
    AppendReplacement(item.first, af::Symbol(item.second), var_replacements);
  }
  for (const auto &item : context.static_size_vars) {
    AppendReplacement(item.first, item.second, var_replacements);
  }
  AppendVarOneReplacements(context.dynamic_size_vars, var_replacements);
}

af::Expression ApplyReplacements(const af::Expression &expr,
                                 const std::vector<std::pair<af::Expression, af::Expression>> &complex_replacements,
                                 const std::vector<std::pair<af::Expression, af::Expression>> &symbol_replacements) {
  af::Expression result = expr;
  if (!complex_replacements.empty()) {
    result = result.Replace(complex_replacements);
  }
  if (!symbol_replacements.empty()) {
    result = result.Subs(symbol_replacements);
  }
  if (result.IsValid()) {
    result = result.Simplify();
  }
  return result;
}

bool TryGetConstInt64(const af::Expression &expr, int64_t &value) {
  if (!expr.IsValid() || !expr.IsConstExpr()) {
    return false;
  }
  if (expr.GetExprType() == af::ExprType::kExprConstantInteger) {
    return expr.GetConstValue(value);
  }
  if (expr.GetExprType() != af::ExprType::kExprConstantRation &&
      expr.GetExprType() != af::ExprType::kExprConstantRealDouble) {
    return false;
  }
  double double_value = 0.0;
  if (!expr.GetConstValue(double_value) || !std::isfinite(double_value)) {
    return false;
  }
  if (double_value < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
      double_value > static_cast<double>(std::numeric_limits<int64_t>::max())) {
    return false;
  }
  double integer_value = 0.0;
  const double fraction_value = std::modf(double_value, &integer_value);
  if (fraction_value > 0.0 || fraction_value < 0.0) {
    return false;
  }
  value = static_cast<int64_t>(integer_value);
  return true;
}

EvalResult EvalMinUbUsage(const ascir::UbExprContext &context) {
  EvalResult eval_result;
  const auto build_result = ascir::UbExprUtils::BuildUbExpr(context);
  if (!build_result.has_ub_expr) {
    eval_result.status = EvalStatus::kUnknown;
    return eval_result;
  }
  eval_result.origin_expr = build_result.ub_expr;

  std::vector<std::pair<af::Expression, af::Expression>> container_replacements;
  std::vector<std::pair<af::Expression, af::Expression>> var_replacements;
  BuildReplacements(context, container_replacements, var_replacements);
  eval_result.min_expr = ApplyReplacements(build_result.ub_expr, container_replacements, var_replacements);
  if (!eval_result.min_expr.IsValid()) {
    eval_result.status = EvalStatus::kFailed;
    return eval_result;
  }
  if (!eval_result.min_expr.FreeSymbols().empty()) {
    eval_result.status = EvalStatus::kUnknown;
    return eval_result;
  }
  if (!TryGetConstInt64(eval_result.min_expr, eval_result.min_ub_usage) || eval_result.min_ub_usage < 0) {
    eval_result.status = EvalStatus::kFailed;
    return eval_result;
  }
  eval_result.status = EvalStatus::kKnown;
  return eval_result;
}

EvalResult EvalGraphMinUbUsage(const af::AscGraph &graph) {
  ascir::UbExprContext context;
  try {
    const auto status = ascir::AscGraphUbExprBuilder().Build(graph, context);
    if (status != af::SUCCESS) {
      GELOGD("%s build UB expr failed, graph=%s", kLogPrefix, graph.GetName().c_str());
      EvalResult result;
      result.status = EvalStatus::kFailed;
      return result;
    }
    return EvalMinUbUsage(context);
  } catch (const std::exception &e) {
    GELOGD("%s eval UB expr failed, graph=%s, reason=%s", kLogPrefix, graph.GetName().c_str(), e.what());
  } catch (...) {
    GELOGD("%s eval UB expr failed, graph=%s, reason=unknown exception", kLogPrefix, graph.GetName().c_str());
  }
  EvalResult result;
  result.status = EvalStatus::kFailed;
  result.origin_expr = context.ub_expr;
  return result;
}

bool TryParseRuntimeUbSize(const char *ub_size_str, int64_t &ub_size) {
  try {
    size_t parsed_size = 0UL;
    const std::string str_value(ub_size_str);
    ub_size = static_cast<int64_t>(std::stoll(str_value, &parsed_size));
    if (parsed_size == str_value.size()) {
      return true;
    }
    GELOGD("%s parse runtime ub_size failed, value=%s", kLogPrefix, ub_size_str);
  } catch (const std::exception &e) {
    GELOGD("%s parse runtime ub_size failed, value=%s, reason=%s", kLogPrefix, ub_size_str, e.what());
  } catch (...) {
    GELOGD("%s parse runtime ub_size failed, value=%s, reason=unknown exception", kLogPrefix, ub_size_str);
  }
  ub_size = 0;
  return false;
}

bool TryGetRuntimeUbSize(int64_t &ub_size) {
  char ub_size_str[kMaxRuntimeSpecValueLen] = {};
  const auto ret = rtGetSocSpec(kAicoreSpecLabel, kUbSizeSpecKey, ub_size_str, kMaxRuntimeSpecValueLen);
  if (ret != RT_ERROR_NONE) {
    GELOGD("%s get runtime ub_size failed, ret=%d", kLogPrefix, ret);
    ub_size = 0;
    return false;
  }
  return TryParseRuntimeUbSize(ub_size_str, ub_size);
}

bool ShouldDropImplGraph(const af::AscGraph &impl_graph, int64_t ub_size, const TemplatePosition &position) {
  const auto eval_result = EvalGraphMinUbUsage(impl_graph);
  const auto template_name =
      BuildTemplateName(position.node_idx, position.result_idx, position.group_idx, position.impl_idx);
  if (eval_result.status == EvalStatus::kUnknown) {
    GELOGD(
        "%s keep template due to unknown UB expr, graph=%s, template=%s, tiling_case=%ld, ub_expr=%s, "
        "platform_ub_size=%ld",
        kLogPrefix, impl_graph.GetName().c_str(), template_name.c_str(), impl_graph.GetTilingKey(),
        ExprToString(eval_result.origin_expr).c_str(), ub_size);
    return false;
  }
  if (eval_result.status == EvalStatus::kFailed) {
    GELOGD(
        "%s keep template due to failed UB expr eval, graph=%s, template=%s, tiling_case=%ld, ub_expr=%s, min_expr=%s, "
        "platform_ub_size=%ld",
        kLogPrefix, impl_graph.GetName().c_str(), template_name.c_str(), impl_graph.GetTilingKey(),
        ExprToString(eval_result.origin_expr).c_str(), ExprToString(eval_result.min_expr).c_str(), ub_size);
    return false;
  }
  if (eval_result.min_ub_usage <= ub_size) {
    GELOGD(
        "%s keep template, graph=%s, template=%s, tiling_case=%ld, ub_expr=%s, min_expr=%s, min_ub_usage=%ld, "
        "platform_ub_size=%ld",
        kLogPrefix, impl_graph.GetName().c_str(), template_name.c_str(), impl_graph.GetTilingKey(),
        ExprToString(eval_result.origin_expr).c_str(), ExprToString(eval_result.min_expr).c_str(),
        eval_result.min_ub_usage, ub_size);
    return false;
  }
  GELOGD(
      "%s drop template, graph=%s, template=%s, tiling_case=%ld, ub_expr=%s, min_expr=%s, min_ub_usage=%ld, "
      "platform_ub_size=%ld, reason=min_ub_usage exceeds platform_ub_size",
      kLogPrefix, impl_graph.GetName().c_str(), template_name.c_str(), impl_graph.GetTilingKey(),
      ExprToString(eval_result.origin_expr).c_str(), ExprToString(eval_result.min_expr).c_str(),
      eval_result.min_ub_usage, ub_size);
  return true;
}

size_t CountImplGraphs(const std::vector<ascir::ScheduledResult> &scheduled_results) {
  size_t count = 0UL;
  for (const auto &scheduled_result : scheduled_results) {
    for (const auto &schedule_group : scheduled_result.schedule_groups) {
      count += schedule_group.impl_graphs.size();
    }
  }
  return count;
}

bool FilterScheduledResult(FilterState &state, ascir::ScheduledResult &scheduled_result, size_t node_idx,
                           size_t result_idx) {
  bool keep_scheduled_result = true;
  for (size_t group_idx = 0UL; group_idx < scheduled_result.schedule_groups.size(); ++group_idx) {
    auto &schedule_group = scheduled_result.schedule_groups[group_idx];
    const size_t before_group_size = schedule_group.impl_graphs.size();
    size_t impl_idx = 0UL;
    schedule_group.impl_graphs.erase(
        std::remove_if(
            schedule_group.impl_graphs.begin(), schedule_group.impl_graphs.end(),
            [&schedule_group, &impl_idx, &state, node_idx, result_idx, group_idx](const af::AscGraph &impl_graph) {
              const size_t current_impl_idx = impl_idx++;
              const TemplatePosition position = {node_idx, result_idx, group_idx, current_impl_idx};
              const bool drop = ShouldDropImplGraph(impl_graph, state.ub_size, position);
              if (drop) {
                schedule_group.graph_name_to_score_funcs.erase(impl_graph.GetName());
              }
              return drop;
            }),
        schedule_group.impl_graphs.end());
    if (before_group_size > 0UL && schedule_group.impl_graphs.empty()) {
      GELOGD(
          "%s drop scheduled result because all templates in group are filtered, graph=%s, node_idx=%zu, "
          "result_idx=%zu, group_idx=%zu",
          kLogPrefix, GetGraphName(state.fused_scheduled_result.fused_graph_name), node_idx, result_idx, group_idx);
      keep_scheduled_result = false;
    }
  }
  return keep_scheduled_result;
}

af::Status FilterNodeScheduledResults(ascir::FusedScheduledResult &fused_scheduled_result,
                                      std::vector<ascir::ScheduledResult> &scheduled_results, int64_t ub_size,
                                      size_t node_idx, size_t &total_dropped) {
  const size_t before = CountImplGraphs(scheduled_results);
  FilterState state = {fused_scheduled_result, ub_size};
  size_t result_idx = 0UL;
  scheduled_results.erase(std::remove_if(scheduled_results.begin(), scheduled_results.end(),
                                         [&state, node_idx, &result_idx](ascir::ScheduledResult &scheduled_result) {
                                           const size_t current_result_idx = result_idx++;
                                           return !FilterScheduledResult(state, scheduled_result, node_idx,
                                                                         current_result_idx);
                                         }),
                          scheduled_results.end());
  const size_t kept = CountImplGraphs(scheduled_results);
  const size_t dropped = before - kept;
  total_dropped += dropped;
  GELOGI("%s graph=%s, node_idx=%zu, templates_before=%zu, dropped=%zu, kept=%zu", kLogPrefix,
         GetGraphName(fused_scheduled_result.fused_graph_name), node_idx, before, dropped, kept);
  GE_ASSERT_TRUE(before == 0UL || !scheduled_results.empty(), "%s all templates are filtered, graph=%s, node_idx=%zu",
                 kLogPrefix, GetGraphName(fused_scheduled_result.fused_graph_name), node_idx);
  return af::SUCCESS;
}

UbLimitResult GetUbLimit() {
  UbLimitResult result;
  int64_t ub_size_override = 0;
  const bool has_ub_size_override = ge::PlatformContext::GetInstance().TryGetUbSizeOverride(ub_size_override);
  if (has_ub_size_override && ub_size_override > 0) {
    result.has_ub_limit = true;
    result.ub_size = ub_size_override;
    return result;
  }

  ge::PlatformInfo platform_info;
  const bool has_platform_info = ge::PlatformContext::GetInstance().TryGetInitializedPlatformInfo(platform_info);
  if (has_platform_info && platform_info.ub_size > 0) {
    result.has_ub_limit = true;
    result.ub_size = platform_info.ub_size;
    return result;
  }

  int64_t runtime_ub_size = 0;
  if (TryGetRuntimeUbSize(runtime_ub_size)) {
    result.has_ub_limit = true;
    result.ub_size = runtime_ub_size;
    return result;
  }

  if (has_ub_size_override) {
    result.has_ub_limit = true;
    result.ub_size = ub_size_override;
    return result;
  }
  if (has_platform_info) {
    result.has_ub_limit = true;
    result.ub_size = platform_info.ub_size;
  }
  return result;
}

}  // namespace

af::Status StaticUbTemplateFilter::Filter(ascir::FusedScheduledResult &fused_scheduled_result) const {
  const auto ub_limit = GetUbLimit();
  if (!ub_limit.has_ub_limit) {
    GELOGD("%s skip UB template filter because platform info is not initialized, graph=%s", kLogPrefix,
           GetGraphName(fused_scheduled_result.fused_graph_name));
    return af::SUCCESS;
  }
  const int64_t ub_size = ub_limit.ub_size;
  if (ub_size <= 0) {
    GELOGD("%s skip UB template filter because ub_size is invalid, graph=%s, ub_size=%ld", kLogPrefix,
           GetGraphName(fused_scheduled_result.fused_graph_name), ub_size);
    return af::SUCCESS;
  }

  size_t total_dropped = 0UL;
  for (size_t node_idx = 0UL; node_idx < fused_scheduled_result.node_idx_to_scheduled_results.size(); ++node_idx) {
    auto &scheduled_results = fused_scheduled_result.node_idx_to_scheduled_results[node_idx];
    GE_CHK_STATUS_RET(
        FilterNodeScheduledResults(fused_scheduled_result, scheduled_results, ub_size, node_idx, total_dropped));
  }
  GELOGI("%s graph=%s, total_dropped=%zu, platform_ub_size=%ld", kLogPrefix,
         GetGraphName(fused_scheduled_result.fused_graph_name), total_dropped, ub_size);
  return af::SUCCESS;
}

}  // namespace optimize
