/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "case_contract.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace device_validation {
namespace {
using Json = nlohmann::json;
void ParseTolerance(const Json &verification, const char *name, double &target) {
  if (!verification.contains(name)) return;
  const auto &value = verification.at(name);
  if ((!value.is_number_float() && !value.is_number_integer()) || !std::isfinite(value.get<double>()) ||
      value.get<double>() < 0.0 || value.get<double>() > 1.0)
    throw std::invalid_argument("invalid verification tolerance");
  target = value.get<double>();
}
std::vector<int64_t> Shape(const Json &value) {
  if (!value.is_array() || value.empty()) throw std::invalid_argument("shape must be non-empty");
  std::vector<int64_t> result;
  uint64_t size = 1;
  for (const auto &dim : value) {
    if (!dim.is_number_integer()) throw std::invalid_argument("shape dimension must be integer");
    const auto n = dim.get<int64_t>();
    if (n <= 0 || size > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(n))
      throw std::invalid_argument("invalid or overflowing shape");
    size *= static_cast<uint64_t>(n);
    result.push_back(n);
  }
  return result;
}
std::vector<std::string> SplitTensors(const std::string &text) {
  std::vector<std::string> result;
  size_t start = 0;
  while (start <= text.size()) {
    const auto end = text.find(';', start);
    const auto item = text.substr(start, end == std::string::npos ? end : end - start);
    if (item.empty()) throw std::invalid_argument("empty legacy tensor");
    result.push_back(item);
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return result;
}
std::vector<int64_t> LegacyShape(const std::string &text) {
  std::vector<int64_t> result;
  uint64_t size = 1;
  size_t start = 0;
  while (start < text.size()) {
    const auto end = text.find(',', start);
    const auto item = text.substr(start, end == std::string::npos ? end : end - start);
    const auto n = std::stoll(item);
    if (n <= 0 || size > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(n))
      throw std::invalid_argument("invalid or overflowing legacy shape");
    size *= static_cast<uint64_t>(n);
    result.push_back(n);
    if (end == std::string::npos) break;
    start = end + 1;
  }
  if (result.empty()) throw std::invalid_argument("empty legacy shape");
  return result;
}
std::vector<std::vector<int64_t>> LegacyShapes(const std::string &text) {
  std::vector<std::vector<int64_t>> result;
  for (const auto &tensor : SplitTensors(text)) result.push_back(LegacyShape(tensor));
  return result;
}
SupportStatus Status(const std::string &value) {
  if (value == "required") return SupportStatus::kRequired;
  if (value == "optional") return SupportStatus::kOptional;
  if (value == "unsupported") return SupportStatus::kUnsupported;
  throw std::invalid_argument("unknown support status");
}
void ParsePerformance(const Json &j, PerformanceSpec &performance) {
  performance.required = j.value("required", false);
  performance.profiler = j.value("profiler", false);
  performance.metric = j.value("metric", performance.metric);
  performance.warmup_count = j.value("warmup_count", performance.warmup_count);
  performance.kernel_count = j.value("kernel_count", performance.kernel_count);
  performance.tiling_time_us = j.value("tiling_time_us", performance.tiling_time_us);
  performance.workspace_size = j.value("workspace_size", performance.workspace_size);
  performance.block_dimension = j.value("block_dimension", performance.block_dimension);
  performance.profiler_available = j.value("profiler_available", performance.profiler_available);
}
SupportEntry ParseSupportEntry(const Json &item, const std::string &default_case_id) {
  if (!item.contains("backend") || !item.contains("soc")) throw std::invalid_argument("support entry is incomplete");
  SupportEntry e{item.value("case_id", default_case_id), item.at("backend"), item.at("soc")};
  for (const auto cap :
       {Capability::kCompile, Capability::kFunctional, Capability::kPrecision, Capability::kPerformance}) {
    const char *name = cap == Capability::kCompile      ? "compile"
                       : cap == Capability::kFunctional ? "functional"
                       : cap == Capability::kPrecision  ? "precision"
                                                        : "performance";
    e.capabilities[cap] = Status(item.value(name, "unsupported"));
  }
  e.dtypes = item.value("dtypes", std::vector<std::string>{});
  e.input_dtypes = item.value("input_dtypes", std::vector<std::string>{});
  e.output_dtypes = item.value("output_dtypes", std::vector<std::string>{});
  e.allowed_abi = item.value("allowed_abi", "");
  e.variants = item.value("variants", std::vector<std::string>{});
  for (const auto &shape : item.value("shapes", Json::array())) e.shapes.push_back(Shape(shape));
  return e;
}
void ParseV1(const Json &j, CaseConfig &c) {
  if (!j.contains("schema_version") || j.at("schema_version") != 1)
    throw std::invalid_argument("unsupported schema version");
  c.schema_version = 1;
  c.case_id = j.value("case_id", "");
  c.graph_name = j.value("graph_name", c.case_id);
  if (j.contains("variants") && j.at("variants").is_object()) {
    for (const auto &[name, value] : j.at("variants").items()) {
      if (!value.is_object()) throw std::invalid_argument("variant is invalid");
      c.variants.push_back(name);
    }
  }
  for (const auto &item : j.value("inputs", Json::array())) {
    if (!item.contains("shape") || !item.contains("dtype")) throw std::invalid_argument("input spec is incomplete");
    c.inputs.push_back({Shape(item.at("shape")), item.at("dtype"), item.value("dynamic", false)});
  }
  for (const auto &item : j.value("outputs", Json::array())) {
    if (!item.contains("shape") || !item.contains("dtype")) throw std::invalid_argument("output spec is incomplete");
    c.outputs.push_back({Shape(item.at("shape")), item.at("dtype"), item.value("dynamic", false)});
  }
  if (j.contains("verification")) {
    c.verification.functional = j["verification"].value("functional", true);
    c.verification.precision = j["verification"].value("precision", false);
    ParseTolerance(j["verification"], "atol", c.verification.atol);
    ParseTolerance(j["verification"], "rtol", c.verification.rtol);
  }
  if (j.contains("performance")) ParsePerformance(j["performance"], c.performance);
  for (const auto &item : j.value("support_matrix", Json::array()))
    c.support_matrix.push_back(ParseSupportEntry(item, c.case_id));
}
}  // namespace

CaseConfig LoadCaseConfig(const std::filesystem::path &case_dir) {
  const auto path = case_dir / (std::filesystem::exists(case_dir / "case.json") ? "case.json" : "ascir.json");
  std::ifstream input(path);
  if (!input) throw std::invalid_argument("case config not found");
  Json j;
  input >> j;
  CaseConfig c;
  if (path.filename() == "case.json") {
    ParseV1(j, c);
    return c;
  }
  c.schema_version = 1;
  const auto &legacy = j.contains("kernel_config") ? j.at("kernel_config") : j;
  c.graph_name = legacy.at("graph_name");
  c.case_id = c.graph_name;
  const auto inputs = SplitTensors(legacy.at("input_data_types"));
  const auto outputs = SplitTensors(legacy.at("output_data_types"));
  const auto input_shapes = LegacyShapes(legacy.at("input_shapes"));
  const auto output_shapes = LegacyShapes(legacy.at("output_shapes"));
  if (static_cast<size_t>(std::stoul(legacy.at("input_num").get<std::string>())) != inputs.size() ||
      inputs.size() != input_shapes.size() ||
      static_cast<size_t>(std::stoul(legacy.at("output_num").get<std::string>())) != outputs.size() ||
      outputs.size() != output_shapes.size())
    throw std::invalid_argument("legacy count mismatch");
  for (size_t i = 0; i < inputs.size(); ++i) c.inputs.push_back({input_shapes[i], inputs[i]});
  for (size_t i = 0; i < outputs.size(); ++i) c.outputs.push_back({output_shapes[i], outputs[i]});
  return c;
}
}  // namespace device_validation
