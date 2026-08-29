/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_DURATION_H_
#define ATT_DURATION_H_

#include <string>
#include <chrono>
#include <cstdint>
#include "common/checker.h"
#include "autofuse_config/auto_fuse_config.h"

namespace att {
extern uint32_t kg_duration_level;
inline bool IsProfilingEnabled() {
  bool env_status = false;
  if (AutoFuseConfig::MutableAttStrategyConfig().Init() == af::SUCCESS &&
      AutoFuseConfig::GetAttStrategyConfig().set_env_att_profiling) {
    env_status = AutoFuseConfig::GetAttStrategyConfig().att_profiling == "true";
  }
  return env_status || kg_duration_level > 0U;
}

// for compile
enum class DurationType {
  DURATION_GEN_MODEL_INFO = 0,
  DURATION_MAX,
};

struct DurationDef {
  std::string name;
  uint32_t level;
};

class Duration {
 public:
  explicit Duration(const std::string &name) : name_(name) {}

  void Begin() {
    call_start_ = Now();
  }

  void End() {
    auto now = Now();
    uint64_t duration = now - call_start_;
    total_count_++;
    total_time_ += duration;
    if (duration > max_time_) {
      max_time_ = duration;
    }
    if (duration < min_time_) {
      min_time_ = duration;
    }
  }

  void Print() {
    if (total_count_ == 0ULL) {
      return;
    }
    GEEVENT(
        "Duration record: name[%s], total_count[%lu], total_time_ns[%lu], max_time_ns[%lu], min_time_ns[%lu], "
        "average_time_ns[%lu].",
        name_.c_str(), total_count_, total_time_, max_time_, min_time_,
        static_cast<uint64_t>(total_time_ / total_count_));
  }

  void Clear() {
    total_count_ = 0ULL;
    total_time_ = 0ULL;
    max_time_ = 0ULL;
    min_time_ = UINT64_MAX;
    call_start_ = 0ULL;
  }

 private:
  uint64_t Now() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
    return static_cast<uint64_t>(nanoseconds.count());
  }

  std::string name_;
  uint64_t total_count_ = 0ULL;
  uint64_t total_time_ = 0ULL;
  uint64_t max_time_ = 0ULL;
  uint64_t min_time_ = UINT64_MAX;
  uint64_t call_start_ = 0ULL;
};

struct DurationInfo {
  uint32_t level;
  std::unique_ptr<Duration> stat;
};

class DurationManager {
 public:
  static DurationManager &GetInstance();

  DurationManager();

  void AddDuration(const uint32_t type, const std::string &name, const uint32_t level);

  void Begin(const DurationType type);

  void Print() {
    for (int32_t index = 0; index < static_cast<int32_t>(DurationType::DURATION_MAX); index++) {
      const auto &stat = duration_infos_[index].stat;
      if (stat != nullptr) {
        stat->Print();
      }
    }
  }

  void Clear() {
    for (int32_t index = 0; index < static_cast<int32_t>(DurationType::DURATION_MAX); index++) {
      const auto &stat = duration_infos_[index].stat;
      if (stat != nullptr) {
        stat->Clear();
      }
    }
  }

  void End(const DurationType type);

 private:
  DurationInfo duration_infos_[static_cast<uint32_t>(DurationType::DURATION_MAX)];
};

class DurationInitGuard {
 public:
  explicit DurationInitGuard(const uint32_t level);
  ~DurationInitGuard();
};

void DurationInit(const uint32_t level);
void DurationFinalize();

void DurationBegin(const DurationType type);

void DurationEnd(const DurationType type);

class DurationGuard {
 public:
  explicit DurationGuard(const DurationType type) : type_(type) {
    DurationBegin(type);
  }

  ~DurationGuard() {
    DurationEnd(type_);
  }

 private:
  DurationType type_;
};

#define DURATION_GUARD(type) DurationGuard g_duration##__COUNTER__(type)

// for generate tiling func
enum class TilingFuncDurationType {
  TILING_FUNC_DURATION_TOTAL = 0,
  TILING_FUNC_DURATION_DOTILING,
  TILING_FUNC_DURATION_MAX,
};

extern DurationDef kg_tiling_func_duration_def[static_cast<uint32_t>(TilingFuncDurationType::TILING_FUNC_DURATION_MAX)];

std::string DurationGenHeadCode();

std::string DurationGenDefineCode();

std::string DurationPrintGenCode();

std::string DurationClearGenCode();

std::string DurationBeginGenCode(const TilingFuncDurationType type);

std::string DurationEndGenCode(const TilingFuncDurationType type);

std::string DurationGuardGenCode(const TilingFuncDurationType type);
}  // namespace att

#endif  // ATT_DURATION_H_
