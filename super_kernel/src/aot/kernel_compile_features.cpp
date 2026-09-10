/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_compile_features.h"

#include <algorithm>
#include <array>
#include <limits>
#include <thread>
#include <utility>
#include "sk_log.h"

namespace sk {
namespace static_compile {
namespace detail {
namespace {
bool Invalid(const std::string &field, const char *reason) {
  SK_DLOGE("Static compile options: %s: %s", field.c_str(), reason);
  return false;
}

template <typename T>
bool CheckArray(const T *data, uint64_t count, const std::string &field) {
  if (count != 0 && data == nullptr) {
    return Invalid(field, "nonzero count requires a non-null array");
  }
  if (count > static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(T) ||
      count > std::vector<T>().max_size()) {
    return Invalid(field, "count exceeds addressable array capacity");
  }
  return true;
}

bool CopyStrings(const char *const *data, uint64_t count, const std::string &field, std::vector<std::string> &out) {
  if (!CheckArray(data, count, field)) {
    return false;
  }
  if (count > out.max_size() - out.size()) {
    return Invalid(field, "combined count exceeds string vector capacity");
  }
  for (uint64_t i = 0; i < count; ++i) {
    if (data[i] == nullptr || data[i][0] == '\0') {
      return Invalid(field + "[" + std::to_string(i) + "]", "string must not be null or empty");
    }
    out.emplace_back(data[i]);
  }
  return true;
}

using Arguments = std::vector<std::string>;
bool ConvertSwitch(uint32_t value, const char *argument, Arguments &output) {
  if (value > 1) {
    return false;
  }
  if (value == 1) {
    output.emplace_back(argument);
  }
  return true;
}
bool ConvertSync(const aclskOption &option, Arguments &output) {
  // Temporary mapping agreed for static-normal; kept separate from the existing SK parser.
  return ConvertSwitch(option.debugSync.debugSyncAll, "-D__DEBUG_SYNC_ALL__", output);
}
bool ConvertCoreCheck(const aclskOption &option, Arguments &output) {
  return ConvertSwitch(option.debugPerOpMaxCoreNum.enableDebugPerOpMaxCoreNum,
                       "-D__ENABLE_SUPER_KERNEL_INNER_CORE_SYNC_CHECK__", output);
}
bool ConvertDcci(const aclskOption &, Arguments &output) {
  // Presence in the original input enables this for the entire model, including an empty kernelNames list.
  output.emplace_back("--cce-no-dcache-flush");
  return true;
}
struct SkConverter {
  aclskOptionType type;
  const char *name;
  bool (*convert)(const aclskOption &, Arguments &);
};
// Single source of truth: nullptr means a known input not yet consumed by static-normal.
constexpr SkConverter SK_OPTIONS[] = {
    {aclskOptionType::PRELOAD_CODE, "PRELOAD_CODE", nullptr},
    {aclskOptionType::SPLIT_MODE, "SPLIT_MODE", nullptr},
    {aclskOptionType::STREAM_FUSION, "STREAM_FUSION", nullptr},
    {aclskOptionType::DCCI_DISABLE_ON_KERNEL, "DCCI_DISABLE_ON_KERNEL", ConvertDcci},
    {aclskOptionType::DEBUG_SYNC_ALL, "DEBUG_SYNC_ALL", ConvertSync},
    {aclskOptionType::KERNEL_MAP, "KERNEL_MAP", nullptr},
    {aclskOptionType::AUTO_OP_PARALLEL, "AUTO_OP_PARALLEL", nullptr},
    {aclskOptionType::DCCI_BEFORE_KERNEL_START, "DCCI_BEFORE_KERNEL_START", nullptr},
    {aclskOptionType::DEBUG_OP_EXEC_TRACE, "DEBUG_OP_EXEC_TRACE", nullptr},
    {aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK, "DEBUG_CROSS_CORE_SYNC_CHECK", nullptr},
    {aclskOptionType::OPT_EXTEND_OPTION, "OPT_EXTEND_OPTION", nullptr},
    {aclskOptionType::DEBUG_EXTEND_OPTION, "DEBUG_EXTEND_OPTION", nullptr},
    {aclskOptionType::DCCI_AFTER_KERNEL_END, "DCCI_AFTER_KERNEL_END", nullptr},
    {aclskOptionType::AGGRESSIVE_OPT_STRATEGIES, "AGGRESSIVE_OPT_STRATEGIES", nullptr},
    {aclskOptionType::UBUF_LOCK_IGNORE_KERNEL, "UBUF_LOCK_IGNORE_KERNEL", nullptr},
    {aclskOptionType::EARLY_START, "EARLY_START", nullptr},
    {aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM, "DEBUG_PER_OP_MAX_CORE_NUM", ConvertCoreCheck},
};

constexpr bool UniqueSkOptions() {
  for (size_t i = 0; i < std::size(SK_OPTIONS); ++i) {
    for (size_t j = i + 1; j < std::size(SK_OPTIONS); ++j) {
      if (SK_OPTIONS[i].type == SK_OPTIONS[j].type) {
        return false;
      }
    }
  }
  return true;
}
static_assert(UniqueSkOptions(), "Each SK option must have exactly one descriptor");
}  // namespace

bool ModelSettings::Init(const aclspecOptions *options) {
  jobs_ = options == nullptr ? 0 : options->jobs;
  if (jobs_ == 0) {
    jobs_ = std::max(1U, std::thread::hardware_concurrency());
  }
  return true;
}

bool KernelMatcher::Init(const std::vector<std::string> &patterns) {
  std::vector<std::regex> compiled;
  for (const auto &pattern : patterns) {
    if (pattern.empty()) {
      return Invalid("kernelPattern", "empty regular expression");
    }
    try {
      compiled.emplace_back(pattern, std::regex::ECMAScript | std::regex::optimize);
    } catch (const std::regex_error &error) {
      if (error.code() == std::regex_constants::error_space || error.code() == std::regex_constants::error_stack ||
          error.code() == std::regex_constants::error_complexity) {
        throw;
      }
      return Invalid("kernelPattern '" + pattern + "'", error.what());
    }
  }
  patterns_.swap(compiled);
  return true;
}

bool KernelMatcher::Matches(std::string_view entry) const {
  return std::any_of(patterns_.begin(), patterns_.end(),
                     [&](const std::regex &pattern) { return std::regex_match(entry.begin(), entry.end(), pattern); });
}

bool KernelFilter::Init(const aclspecOptions *options) {
  KernelFilter next;
  if (options != nullptr) {
    if (!CheckArray(options->filter, options->filterCount, "filter")) {
      return false;
    }
    bool hasDenyList = false;
    std::vector<std::string> allow;
    std::vector<std::string> deny;
    for (uint64_t i = 0; i < options->filterCount; ++i) {
      const auto field = "filter[" + std::to_string(i) + "]";
      const auto *filter = options->filter[i];
      if (filter == nullptr) {
        return Invalid(field, "filter must not be null");
      }
      if (filter->mode != ACL_KERNEL_SPEC_FILTER_ALLOW && filter->mode != ACL_KERNEL_SPEC_FILTER_DENY) {
        return Invalid(field + ".mode", "unknown filter mode");
      }
      const bool isAllow = filter->mode == ACL_KERNEL_SPEC_FILTER_ALLOW;
      next.hasAllowList_ |= isAllow;
      hasDenyList |= !isAllow;
      if (next.hasAllowList_ && hasDenyList) {
        return Invalid(field, "ALLOW and DENY cannot coexist");
      }
      if (!CopyStrings(filter->kernelEntries, filter->kernelEntryCount, field + ".kernelEntries",
                       isAllow ? allow : deny)) {
        return false;
      }
    }
    if (!next.allow_.Init(allow) || !next.deny_.Init(deny)) {
      return false;
    }
  }
  *this = std::move(next);
  return true;
}

bool KernelFilter::FilterSpecialize(std::string_view entry) const {
  return hasAllowList_ ? allow_.Matches(entry) : !deny_.Matches(entry);
}

bool CompileOptions::Init(const aclspecOptions *options) {
  CompileOptions next;
  if (options != nullptr && (!next.InitBasicOptions(options->specOptions, options->specOptionCount) ||
                             !next.InitSkOptions(options->skOptions))) {
    return false;
  }
  *this = std::move(next);
  return true;
}

bool CompileOptions::InitBasicOptions(const aclspecOption *options, uint64_t count) {
  if (!CheckArray(options, count, "specOptions") || count > basicRules_.max_size()) {
    return Invalid("specOptions", "invalid rule array");
  }
  for (uint64_t i = 0; i < count; ++i) {
    const auto field = "specOptions[" + std::to_string(i) + "]";
    const auto &option = options[i];
    if (option.kernelName == nullptr || option.kernelName[0] == '\0') {
      return Invalid(field + ".kernelName", "kernel pattern is required");
    }
    Rule rule;
    if (!rule.matcher.Init({option.kernelName})) {
      return Invalid(field + ".kernelName", "invalid regular expression");
    }
    if (!CopyStrings(option.compileOption, option.compileOptionCount, field + ".compileOption", rule.arguments)) {
      return false;
    }
    basicRules_.push_back(std::move(rule));
  }
  return true;
}

bool CompileOptions::InitSkOptions(const aclskOptions *options) {
  if (options == nullptr) {
    return true;
  }
  if (!CheckArray(options->options, options->numOptions, "skOptions.options")) {
    return false;
  }
  std::array<bool, std::size(SK_OPTIONS)> seen{};
  for (size_t i = 0; i < options->numOptions; ++i) {
    const auto &option = options->options[i];
    const auto field = "skOptions.options[" + std::to_string(i) + "]";
    const auto converter = std::find_if(std::begin(SK_OPTIONS), std::end(SK_OPTIONS),
                                        [&](const SkConverter &item) { return item.type == option.optionType; });
    if (converter == std::end(SK_OPTIONS)) {
      return Invalid(field + ".optionType", "unknown SK option enum");
    }
    const auto typeIndex = static_cast<size_t>(converter - std::begin(SK_OPTIONS));
    if (converter->convert == nullptr) {
      if (!seen[typeIndex]) {
        SK_DLOGI("Static compile options: ignoring unsupported SK option %s", converter->name);
      }
    } else {
      if (seen[typeIndex]) {
        return Invalid(field + "." + converter->name, "duplicate supported SK option type");
      }
      if (!converter->convert(option, globalSkArguments_)) {
        return Invalid(field + "." + converter->name, "SK switch must be 0 or 1");
      }
    }
    seen[typeIndex] = true;
  }
  if (!globalSkArguments_.empty()) {
    globalSkArguments_.insert(globalSkArguments_.begin(), "--enable-super-kernel");
  }
  return true;
}

std::vector<std::string> CompileOptions::Basic(std::string_view entry) const {
  std::vector<std::string> result;
  for (const auto &rule : basicRules_) {
    if (rule.matcher.Matches(entry)) {
      result.insert(result.end(), rule.arguments.begin(), rule.arguments.end());
    }
  }
  return result;
}

std::vector<std::string> CompileOptions::Sk(std::string_view) const {
  return globalSkArguments_;
}

}  // namespace detail
}  // namespace static_compile
}  // namespace sk
