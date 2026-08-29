/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "duration.h"
#include "graph_metadef/graph/debug/ge_util.h"

namespace att {
uint32_t kg_duration_level = 0U;
namespace {
DurationDef kg_duration_def[static_cast<uint32_t>(DurationType::DURATION_MAX)] = {{"GEN_MODEL_INFO", 0U}};
}

DurationManager &DurationManager::GetInstance() {
  static DurationManager ins;
  return ins;
}

DurationManager::DurationManager() {
  for (uint32_t index = 0U; index < static_cast<uint32_t>(DurationType::DURATION_MAX); index++) {
    AddDuration(index, kg_duration_def[index].name, kg_duration_def[index].level);
  }
}

void DurationManager::AddDuration(const uint32_t type, const std::string &name, const uint32_t level) {
  duration_infos_[type].level = level;
  duration_infos_[type].stat = std::unique_ptr<Duration>(new (std::nothrow) Duration(name));
  if (duration_infos_[type].stat == nullptr) {
    GELOGW("Create Duration failed.");
  }
}

void DurationManager::Begin(const DurationType type) {
  const auto &stat = duration_infos_[static_cast<int32_t>(type)].stat;
  if (stat == nullptr) {
    return;
  }
  stat->Begin();
}

void DurationManager::End(const DurationType type) {
  const auto &stat = duration_infos_[static_cast<int32_t>(type)].stat;
  if (stat == nullptr) {
    return;
  }
  stat->End();
}

DurationInitGuard::DurationInitGuard(const uint32_t level) {
  DurationInit(level);
}

DurationInitGuard::~DurationInitGuard() {
  DurationFinalize();
}

void DurationInit(const uint32_t level) {
  kg_duration_level =
      IsProfilingEnabled() ? static_cast<uint32_t>(TilingFuncDurationType::TILING_FUNC_DURATION_MAX) : level;
}

void DurationFinalize() {
  if (kg_duration_level > 0U) {
    DurationManager::GetInstance().Print();
    DurationManager::GetInstance().Clear();
  }
  kg_duration_level = 0U;
}

void DurationBegin(const DurationType type) {
  if (kg_duration_level > kg_duration_def[static_cast<int32_t>(type)].level) {
    DurationManager::GetInstance().Begin(type);
  }
}

void DurationEnd(const DurationType type) {
  if (kg_duration_level > kg_duration_def[static_cast<int32_t>(type)].level) {
    DurationManager::GetInstance().End(type);
  }
}

DurationDef kg_tiling_func_duration_def[static_cast<uint32_t>(TilingFuncDurationType::TILING_FUNC_DURATION_MAX)] = {
    {"TILING_FUNC_DURATION_TOTAL", 0U}, {"TILING_FUNC_DURATION_DOTILING", 1U}};

std::string DurationGenHeadCode() {
  if (kg_duration_level == 0U) {
    return "";
  }
  std::string code =
      "namespace duration_utils {\n"
      "enum DurationType {\n";
  int32_t duration_num = 0;
  for (uint32_t index = 0U; index < static_cast<uint32_t>(TilingFuncDurationType::TILING_FUNC_DURATION_MAX); index++) {
    if (kg_duration_level > kg_tiling_func_duration_def[index].level) {
      if (duration_num == 0) {
        code += ("  " + kg_tiling_func_duration_def[index].name + " = 0,\n");
      } else {
        code += ("  " + kg_tiling_func_duration_def[index].name + ",\n");
      }
      duration_num++;
    }
  }
  code +=
      "  TILING_FUNC_DURATION_MAX,\n"
      "};\n"
      "\n"
      "struct DurationDef {\n"
      "  std::string name;\n"
      "};\n"
      "\n";
  code += "extern DurationDef g_duration_def[TILING_FUNC_DURATION_MAX];\n";
  code +=
      "class Duration {\n"
      " public:\n"
      "  Duration(const std::string &name);\n"
      "  void Begin();\n"
      "  void End();\n"
      "  void Print();\n"
      "  void Clear();\n"
      "private:\n"
      "  uint64_t Now();\n"
      "  std::string name_;\n"
      "  uint64_t total_count_ = 0ULL;\n"
      "  uint64_t total_time_ = 0ULL;\n"
      "  uint64_t max_time_ = 0ULL;\n"
      "  uint64_t min_time_ = UINT64_MAX;\n"
      "  uint64_t call_start_ = 0ULL;\n"
      "};\n"
      "\n"
      "struct DurationInfo {\n"
      "  std::unique_ptr<Duration> stat;\n"
      "};\n"
      "\n"
      "class DurationManager {\n"
      "public:\n"
      "  static DurationManager &GetInstance();\n"
      "  void AddDuration(const uint32_t type, const std::string &name);\n"
      "  void Begin(const DurationType type);\n"
      "  void End(const DurationType type);\n"
      "  void Print();\n"
      "  void Clear();\n"
      "private:\n";
  code += "  DurationManager();\n";
  code += "  bool duration_open_now_ = true;\n";
  code +=
      "  DurationInfo duration_infos_[TILING_FUNC_DURATION_MAX];\n"
      "};\n";
  code += "static inline void DurationBegin(const DurationType type) {\n";
  code += "  DurationManager::GetInstance().Begin(type);\n";
  code += "}\n\n";
  code += "static inline void DurationEnd(const DurationType type) {\n";
  code += "  DurationManager::GetInstance().End(type);\n";
  code += "}\n\n";
  code +=
      "class DurationGuard {\n"
      "public:\n"
      "  DurationGuard(const DurationType type);\n"
      "  ~DurationGuard();\n"
      "private:\n"
      "  DurationType type_;\n"
      "};\n"
      "\n"
      "#define DURATION_GUARD(type) DurationGuard g_duration##__COUNTER__(type);\n"
      "} // namespace duration_utils\n";
  return code;
}

std::string DurationGenDefineCode() {
  if (kg_duration_level == 0U) {
    return "";
  }
  std::string code = "namespace duration_utils {\n";
  code += "DurationDef g_duration_def[TILING_FUNC_DURATION_MAX] = {\n";
  for (uint32_t index = 0U; index < static_cast<uint32_t>(TilingFuncDurationType::TILING_FUNC_DURATION_MAX); index++) {
    if (kg_duration_level > kg_tiling_func_duration_def[index].level) {
      code += ("  {\"" + kg_tiling_func_duration_def[index].name + "\"},\n");
    }
  }
  code +=
      "};\n"
      "\n";
  code += "Duration::Duration(const std::string &name) : name_(name) {}\n\n";
  code += "void Duration::Begin() {\n";
  code += "  call_start_ = Now();\n";
  code += "}\n\n";
  code += "void Duration::End() {\n";
  code += "  auto now = Now();\n";
  code += "  uint64_t duration = now - call_start_;\n";
  code += "  call_start_ = now;\n";
  code += "  total_count_++;\n";
  code += "  total_time_ += duration;\n";
  code += "  if (duration > max_time_) max_time_ = duration;\n";
  code += "  if (duration < min_time_) min_time_ = duration;\n";
  code += "}\n\n";
  code += "void Duration::Print() {\n";
  code += "  if (total_count_ == 0ULL) return;\n";
  code +=
      "  OP_EVENT(OP_NAME, \"Duration record: name[%s], total_count[%lu], total_time_ns[%lu], max_time_ns[%lu], "
      "min_time_ns[%lu], average_time_ns[%lu].\",\n";
  code += "    name_.c_str(), total_count_, total_time_, max_time_, min_time_,\n";
  code += "    static_cast<uint64_t>(total_time_ / total_count_));\n";
  code += "}\n\n";
  code += "void Duration::Clear() {\n";
  code += "  total_count_ = 0ULL;\n";
  code += "  total_time_ = 0ULL;\n";
  code += "  max_time_ = 0ULL;\n";
  code += "  min_time_ = UINT64_MAX;\n";
  code += "  call_start_ = 0ULL;\n";
  code += "}\n\n";
  code += "uint64_t Duration::Now() {\n";
  code += "  auto now = std::chrono::high_resolution_clock::now();\n";
  code += "  auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());\n";
  code += "  return static_cast<uint64_t>(nanoseconds.count());\n";
  code += "}\n\n";
  code += "// 3. DurationManager的成员函数实现\n";
  code += "DurationManager &DurationManager::GetInstance() {\n";
  code += "  static DurationManager ins;\n";
  code += "  return ins;\n";
  code += "}\n\n";
  code += "DurationManager::DurationManager() {\n";
  code += "  for (uint32_t index = 0U; index < static_cast<uint32_t>(TILING_FUNC_DURATION_MAX); index++) {\n";
  code += "    AddDuration(index, g_duration_def[index].name);\n";
  code += "  }\n";
  code += "}\n\n";
  code += "void DurationManager::AddDuration(const uint32_t type, const std::string &name) {\n";
  code += "  if (!duration_open_now_) {\n";
  code += "    return;\n";
  code += "  }\n";
  code += "  duration_infos_[type].stat = std::unique_ptr<Duration>(new(std::nothrow) Duration(name));\n";
  code += "  if (duration_infos_[type].stat == nullptr) {\n";
  code += "    OP_LOGW(OP_NAME, \"Create Duration failed.\");\n";
  code += "  }\n";
  code += "}\n\n";
  code += "void DurationManager::Begin(const DurationType type) {\n";
  code += "  if (!duration_open_now_) {\n";
  code += "    return;\n";
  code += "  }\n";
  code += "  const auto &stat = duration_infos_[type].stat;\n";
  code += "  if (stat == nullptr) {\n";
  code += "    return;\n";
  code += "  }\n";
  code += "  stat->Begin();\n";
  code += "}\n\n";
  code += "void DurationManager::End(const DurationType type) {\n";
  code += "  if (!duration_open_now_) {\n";
  code += "    return;\n";
  code += "  }\n";
  code += "  const auto &stat = duration_infos_[type].stat;\n";
  code += "  if (stat == nullptr) {\n";
  code += "    return;\n";
  code += "  }\n";
  code += "  stat->End();\n";
  code += "}\n\n";
  code += "void DurationManager::Print() {\n";
  code += "  if (!duration_open_now_) {\n";
  code += "    return;\n";
  code += "  }\n";
  code +=
      "  for (int32_t index = 0; index < static_cast<int32_t>(DurationType::TILING_FUNC_DURATION_MAX); index++) {\n";
  code += "    const auto &stat = duration_infos_[index].stat;\n";
  code += "    if (stat != nullptr) {\n";
  code += "      stat->Print();\n";
  code += "    }\n";
  code += "  }\n";
  code += "}\n\n";
  code += "void DurationManager::Clear() {\n";
  code += "  if (!duration_open_now_) {\n";
  code += "    return;\n";
  code += "  }\n";
  code +=
      "  for (int32_t index = 0; index < static_cast<int32_t>(DurationType::TILING_FUNC_DURATION_MAX); index++) {\n";
  code += "    const auto &stat = duration_infos_[index].stat;\n";
  code += "    if (stat != nullptr) {\n";
  code += "      stat->Clear();\n";
  code += "    }\n";
  code += "  }\n";
  code += "}\n\n";
  code += "DurationGuard::DurationGuard(const DurationType type) : type_(type) {\n";
  code += "  DurationBegin(type);\n";
  code += "}\n\n";
  code += "DurationGuard::~DurationGuard() {\n";
  code += "  DurationEnd(type_);\n";
  code += "}\n";
  code += "} // namespace duration_utils\n";
  return code;
}

std::string DurationPrintGenCode() {
  if (kg_duration_level == 0U) {
    return "";
  }
  std::string code = "duration_utils::DurationManager::GetInstance().Print();";
  return code;
}

std::string DurationClearGenCode() {
  if (kg_duration_level == 0U) {
    return "";
  }
  std::string code = "duration_utils::DurationManager::GetInstance().Clear();";
  return code;
}

std::string DurationBeginGenCode(const TilingFuncDurationType type) {
  if (kg_duration_level <= kg_tiling_func_duration_def[static_cast<int32_t>(type)].level) {
    return "";
  }
  return std::string("duration_utils::DurationBegin(duration_utils::") +
         kg_tiling_func_duration_def[static_cast<int32_t>(type)].name + ");";
}

std::string DurationEndGenCode(const TilingFuncDurationType type) {
  if (kg_duration_level <= kg_tiling_func_duration_def[static_cast<int32_t>(type)].level) {
    return "";
  }
  return std::string("duration_utils::DurationEnd(duration_utils::") +
         kg_tiling_func_duration_def[static_cast<int32_t>(type)].name + ");";
}

std::string DurationGuardGenCode(const TilingFuncDurationType type) {
  if (kg_duration_level <= kg_tiling_func_duration_def[static_cast<int32_t>(type)].level) {
    return "";
  }
  return std::string("DURATION_GUARD(") + kg_tiling_func_duration_def[static_cast<int32_t>(type)].name + ")";
}
}  // namespace att
