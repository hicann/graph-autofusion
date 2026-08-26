/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "report.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace device_validation {
namespace {
double Percentile(std::vector<double> sorted, double percentile) {
  const double position = percentile * static_cast<double>(sorted.size() - 1);
  const auto lower = static_cast<size_t>(position);
  const auto upper = std::min(lower + 1, sorted.size() - 1);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - static_cast<double>(lower));
}
nlohmann::json SerializeMismatchValue(const MismatchValue &value) {
  switch (value.kind) {
    case MismatchValue::Kind::kNaN:
      return {{"kind", "nan"}};
    case MismatchValue::Kind::kInf:
      return {{"kind", "inf"}, {"sign", value.sign}};
    case MismatchValue::Kind::kInt64:
      return {{"kind", "int64"}, {"value", value.int_value}};
    case MismatchValue::Kind::kUInt64:
      return {{"kind", "uint64"}, {"value", value.uint_value}};
    case MismatchValue::Kind::kRawBits:
      return {{"kind", "raw_bits"}, {"raw_bits", value.raw_bits}};
    case MismatchValue::Kind::kFloat:
      if (!std::isfinite(value.float_value)) throw std::invalid_argument("non-finite float mismatch value");
      return {{"kind", "float"}, {"value", value.float_value}};
  }
  throw std::invalid_argument("unknown mismatch value kind");
}
}  // namespace

nlohmann::json SerializeCapabilities(const SupportDecision &support) {
  nlohmann::json capabilities = nlohmann::json::object();
  for (const auto &[capability, decision] : support.capabilities) {
    const auto name = capability == Capability::kCompile      ? "compile"
                      : capability == Capability::kFunctional ? "functional"
                      : capability == Capability::kPrecision  ? "precision"
                                                              : "performance";
    const auto result = decision.result == CapabilityResult::kPassed    ? "passed"
                        : decision.result == CapabilityResult::kFailed  ? "failed"
                        : decision.result == CapabilityResult::kSkipped ? "skipped"
                                                                        : "not_applicable";
    capabilities[name] = {{"result", result}, {"reason", decision.reason}};
  }
  return capabilities;
}

nlohmann::json SerializePrecision(const PrecisionReport &precision) {
  nlohmann::json first = nullptr;
  if (precision.first_mismatch) {
    first = {{"tensor_index", precision.first_mismatch->tensor_index},
             {"linear_index", precision.first_mismatch->linear_index},
             {"actual", SerializeMismatchValue(precision.first_mismatch->actual)},
             {"expected", SerializeMismatchValue(precision.first_mismatch->expected)}};
  }
  return {{"passed", precision.passed},
          {"tensor_count", precision.tensor_count},
          {"element_count", precision.element_count},
          {"mismatch_count", precision.mismatch_count},
          {"nan_mismatch_count", precision.nan_mismatch_count},
          {"inf_mismatch_count", precision.inf_mismatch_count},
          {"first_mismatch", first}};
}

nlohmann::json SerializePerformance(const PerformanceReport &performance) {
  const auto &declared = performance.declared;
  const auto &actual = performance.actual;
  const bool device_metric = actual.metric == "device_kernel_duration";
  nlohmann::json actual_values =
      device_metric ? nlohmann::json{{"min_us", actual.min_us}, {"mean_us", actual.mean_us}, {"p50_us", actual.p50_us},
                                     {"p90_us", actual.p90_us}, {"p99_us", actual.p99_us},   {"max_us", actual.max_us}}
                    : nlohmann::json{{"min_ms", actual.min_ms}, {"mean_ms", actual.mean_ms}, {"p50_ms", actual.p50_ms},
                                     {"p90_ms", actual.p90_ms}, {"p99_ms", actual.p99_ms},   {"max_ms", actual.max_ms}};
  actual_values["min"] = device_metric ? actual.min_us : actual.min_ms;
  actual_values["mean"] = device_metric ? actual.mean_us : actual.mean_ms;
  actual_values["p50"] = device_metric ? actual.p50_us : actual.p50_ms;
  actual_values["p90"] = device_metric ? actual.p90_us : actual.p90_ms;
  actual_values["p99"] = device_metric ? actual.p99_us : actual.p99_ms;
  actual_values["max"] = device_metric ? actual.max_us : actual.max_ms;
  nlohmann::json actual_report = {{"sample_count", actual.sample_count},
                                  {"samples", actual.samples},
                                  {"warmup_count", actual.warmup_count},
                                  {"kernel_count", actual.kernel_count},
                                  {"tiling_time_us", actual.tiling_time_us},
                                  {"workspace_size", actual.workspace_size},
                                  {"block_dimension", actual.block_dimension},
                                  {"profiler_requested", actual.profiler_requested},
                                  {"profiler_tool_available", actual.profiler_tool_available},
                                  {"tool_available", actual.profiler_tool_available},
                                  {"profiler_collected", actual.profiler_collected},
                                  {"collected", actual.profiler_collected},
                                  {"profiler_collect_dir", actual.profiler_collect_dir},
                                  {"metric", actual.metric},
                                  {"unit", actual.unit},
                                  {"timing_source", actual.timing_source}};
  actual_report.update(actual_values);
  return {{"declared",
           {{"required", declared.required},
            {"profiler", declared.profiler},
            {"metric", declared.metric},
            {"warmup_count", declared.warmup_count},
            {"kernel_count", declared.kernel_count},
            {"tiling_time_us", declared.tiling_time_us},
            {"workspace_size", declared.workspace_size},
            {"block_dimension", declared.block_dimension},
            {"profiler_requested", declared.profiler},
            {"declared_profiler", declared.profiler},
            {"profiler_tool_available", declared.profiler_available},
            {"tool_available", declared.profiler_available},
            {"profiler_collected", false},
            {"collected", false},
            {"timing_source", "runner_wall_clock"}}},
          {"actual", actual_report}};
}

PerformanceReport SummarizeSamples(const std::vector<double> &samples, const PerformanceSpec &spec) {
  if (spec.warmup_count >= samples.size()) throw std::invalid_argument("performance samples must not be empty");
  for (const double sample : samples) {
    if (!std::isfinite(sample)) throw std::invalid_argument("performance samples must be finite");
  }
  const auto measured =
      samples.size() > spec.warmup_count
          ? std::vector<double>(samples.begin() + static_cast<std::ptrdiff_t>(spec.warmup_count), samples.end())
          : std::vector<double>();
  if (measured.empty()) throw std::invalid_argument("performance samples must not be empty");
  std::vector<double> sorted = measured;
  std::sort(sorted.begin(), sorted.end());
  PerformanceReport report;
  report.declared = spec;
  report.actual.sample_count = measured.size();
  report.actual.min_ms = sorted.front();
  report.actual.max_ms = sorted.back();
  report.actual.mean_ms = 0.0;
  for (const double sample : measured) report.actual.mean_ms += sample / static_cast<double>(measured.size());
  report.actual.p50_ms = Percentile(sorted, 0.50);
  report.actual.p90_ms = Percentile(sorted, 0.90);
  report.actual.p99_ms = Percentile(sorted, 0.99);
  report.actual.min_us = report.actual.min_ms;
  report.actual.mean_us = report.actual.mean_ms;
  report.actual.p50_us = report.actual.p50_ms;
  report.actual.p90_us = report.actual.p90_ms;
  report.actual.p99_us = report.actual.p99_ms;
  report.actual.max_us = report.actual.max_ms;
  return report;
}

PerformanceReport SummarizeSamplesWithRuntimeState(const std::vector<double> &samples, const PerformanceSpec &spec,
                                                   bool profiler_requested, bool profiler_tool_available,
                                                   bool profiler_collected, const std::string &timing_source) {
  auto report = SummarizeSamples(samples, spec);
  report.actual.profiler_requested = profiler_requested;
  report.actual.profiler_tool_available = profiler_tool_available;
  report.actual.profiler_collected = profiler_collected;
  report.actual.timing_source = timing_source;
  if (timing_source == "host_clock_kernel_launch_us") {
    report.actual.min_us = report.actual.min_ms;
    report.actual.mean_us = report.actual.mean_ms;
    report.actual.p50_us = report.actual.p50_ms;
    report.actual.p90_us = report.actual.p90_ms;
    report.actual.p99_us = report.actual.p99_ms;
    report.actual.max_us = report.actual.max_ms;
    report.actual.min_ms = 0.0;
    report.actual.mean_ms = 0.0;
    report.actual.p50_ms = 0.0;
    report.actual.p90_ms = 0.0;
    report.actual.p99_ms = 0.0;
    report.actual.max_ms = 0.0;
  }
  return report;
}

nlohmann::json SerializeReport(const RunReport &report) {
  if (report.schema_version != 2) throw std::invalid_argument("unsupported report schema version");
  return {{"schema_version", report.schema_version},
          {"case", report.case_id},
          {"case_id", report.case_id},
          {"backend", report.backend},
          {"abi", report.abi},
          {"requested_abi", report.requested_abi},
          {"actual_abi", report.abi},
          {"abi_metadata", report.abi_metadata},
          {"variant", report.variant},
          {"soc_profile", report.soc_profile},
          {"soc", report.soc_profile},
          {"profile", report.profile},
          {"metric", report.metric},
          {"stage_status", report.stage_status},
          {"stage", report.stage},
          {"reason", report.reason},
          {"error_code", report.error_code},
          {"support_decisions", SerializeCapabilities(report.support)},
          {"artifact_paths", report.artifact_paths},
          {"artifact", report.artifact_paths},
          {"run_parameters", report.run_parameters},
          {"precision", SerializePrecision(report.precision)},
          {"performance", SerializePerformance(report.performance)}};
}
}  // namespace device_validation
