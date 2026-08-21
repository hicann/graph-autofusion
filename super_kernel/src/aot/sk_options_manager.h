/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sk_options_manager.h
 * \brief
 */

#ifndef SK_OPTIONS_MANAGER_H
#define SK_OPTIONS_MANAGER_H

#include <iostream>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include <stdexcept>
#include <regex>
#include <string>
#include <map>
#include <nlohmann/json.hpp>
#include "super_kernel.h"

/*!
 * \brief Base class for optimization options, providing common interface for different option types.
 *
 * This abstract base class defines the interface for all optimization option types, including
 * name, type, and value accessors. Concrete classes (NumberOptOption, StringOptOption, etc.)
 * implement type-specific value storage and retrieval logic.
 */
class OptOptionBase {
 public:
  explicit OptOptionBase(std::string name, aclskOptionType optionTypeIn)
      : optionName(std::move(name)), optionType(optionTypeIn) {}

  virtual ~OptOptionBase() = default;

  virtual const std::string &GetName() const {
    return optionName;
  }

  virtual aclskOptionType GetType() const {
    return optionType;
  }

  virtual void SetValue(const uint32_t value) {}
  virtual void SetValue(const std::string &value) {}
  virtual void SetValue(const std::vector<std::string> &value) {}
  virtual void SetValue(const std::unordered_map<std::string, std::vector<std::string>> &value) {}
  virtual uint32_t GetIntValue() const {
    return 0;
  }

  virtual std::string GetStringValue() const {
    return "";
  }

  virtual std::unordered_map<std::string, std::vector<std::string>> GetMapValue() const {
    return {};
  }

  virtual std::vector<std::string> GetStringListValue() const {
    return {};
  }

 protected:
  std::string optionName;
  aclskOptionType optionType = aclskOptionType::SK_OPTION_MAX;
};

/*!
 * \brief Integer-type optimization option with value range validation.
 *
 * Stores numeric options with optional minimum and maximum bounds.
 * Values outside the specified range will be rejected during setting.
 */
class NumberOptOption : public OptOptionBase {
 public:
  NumberOptOption(std::string name, aclskOptionType optionTypeIn, uint32_t defaultValue = 0, uint32_t minValue = 0,
                  uint32_t maxValue = 0xffffffff)
      : OptOptionBase(std::move(name), optionTypeIn),
        optValue(defaultValue),
        optValueMin(minValue),
        optValueMax(maxValue) {}

  void SetValue(const uint32_t value) override;
  uint32_t GetIntValue() const override;

 private:
  uint32_t optValue = 0;
  uint32_t optValueMin = 0;
  uint32_t optValueMax = 0;
};

/*!
 * \brief String-type optimization option.
 *
 * Stores string-based options such as file paths, operation names, or configuration strings.
 */
class StringOptOption : public OptOptionBase {
 public:
  StringOptOption(std::string name, aclskOptionType optionTypeIn, std::string defaultValue = {})
      : OptOptionBase(std::move(name), optionTypeIn), optValue(defaultValue) {}

  void SetValue(const std::string &value) override;
  std::string GetStringValue() const override;

 private:
  std::string optValue;
};

/*!
 * \brief String list-type optimization option.
 *
 * Stores collections of string values, useful for multi-valued options
 * such as operation lists, include directories, or configuration sets.
 */
class StringListOptOption : public OptOptionBase {
 public:
  StringListOptOption(std::string name, aclskOptionType optionTypeIn, std::vector<std::string> defaultValue = {})
      : OptOptionBase(std::move(name), optionTypeIn), optValue(defaultValue) {}

  void SetValue(const std::vector<std::string> &value) override;
  std::vector<std::string> GetStringListValue() const override;

 private:
  std::vector<std::string> optValue;
};

/*!
 * \brief Map-type optimization option with string keys and string list values.
 *
 * Stores key-value pairs where each key maps to a list of strings.
 * Suitable for complex configuration options like operation attributes or named parameter lists.
 */
class MapOptOption : public OptOptionBase {
 public:
  MapOptOption(std::string name, aclskOptionType optionTypeIn,
               std::unordered_map<std::string, std::vector<std::string>> defaultValue = {})
      : OptOptionBase(std::move(name), optionTypeIn), optValue(defaultValue) {}

  void SetValue(const std::unordered_map<std::string, std::vector<std::string>> &value) override;
  std::unordered_map<std::string, std::vector<std::string>> GetMapValue() const override;

 private:
  std::unordered_map<std::string, std::vector<std::string>> optValue;
};

template <typename T>
class StructOptOption : public OptOptionBase {
 public:
  StructOptOption(std::string name, aclskOptionType optionTypeIn, T defaultValue = {})
      : OptOptionBase(std::move(name), optionTypeIn), optValue(defaultValue) {}

  void SetValue(const T &value) {
    optValue = value;
  }

  const T &GetValue() const {
    return optValue;
  }

 private:
  T optValue{};
};

using AggressiveOptStrategiesOption = StructOptOption<aclskAggressiveOptStrategies>;

/*!
 * \brief Manager class for super kernel optimization options.
 *
 * This class manages a collection of optimization options of various types (integer, string,
 * string list, map). It provides methods to add, retrieve, and configure options, as well as
 * utility functions for option validation and filtering.
 *
 * Key features:
 * - Type-safe option storage using polymorphic OptOptionBase hierarchy
 * - Dynamic option registration and retrieval by type
 * - DCCI (disable kernel) operation filtering based on configured patterns
 * - Debug mode control
 * - Automatic parsing and validation of aclskOptions structures
 */
class SuperKernelOptionsManager {
 public:
  SuperKernelOptionsManager() = default;
  ~SuperKernelOptionsManager() = default;

  SuperKernelOptionsManager(const SuperKernelOptionsManager &) = default;
  SuperKernelOptionsManager &operator=(const SuperKernelOptionsManager &) = default;

  /*!
   * \brief Add a new option to the manager
   * \param option Unique pointer to the option to add (ownership transferred)
   */
  void AddOption(std::unique_ptr<OptOptionBase> option);

  /*!
   * \brief Get an option by its type
   * \param optType The option type to retrieve
   * \return Pointer to the option, or nullptr if not found
   */
  OptOptionBase *GetOption(aclskOptionType optType);
  const OptOptionBase *GetOption(aclskOptionType optType) const;

  /*!
   * \brief Check if kernel name matches any pattern in the given list
   * \param kernelList List of kernel name patterns (supports regex wildcards . and *)
   * \param kernelName The kernel name to check against patterns
   * \return True if kernel name matches any pattern in the list, false otherwise
   */
  bool MatchKernelNameInList(const std::vector<std::string> &kernelList, const std::string &kernelName) const;

  /*!
   * \brief Judge whether a kernel should ignore MIX kernel split based on configured patterns
   * \param ignoredKernels List of operation patterns to ignore MIX kernel split
   * \param opName The operation name to check against patterns
   * \return True if the kernel should be treated as a normal kernel for MIX split, false otherwise
   */
  bool JudgeUbufLockIgnoreKernel(const std::vector<std::string> &ignoredKernels, const std::string &opName) const;

  /*!
   * \brief Simple regex-like pattern matching without std::regex
   * \param pattern The regex pattern (supports: . *)
   * \param opName The operator name to match against
   * \return True if opName matches the pattern, false otherwise
   */
  static bool MatchRegex(const std::string &pattern, const std::string &opName);

  /*!
   * \brief Check if debug mode is enabled
   * \return True if debug mode is enabled, false otherwise
   */
  bool EnableDebug() const;

  /*!
   * \brief Check whether an internal option is enabled
   * \param optionType Internal option type
   * \return True if the option is enabled
   */
  bool IsInnerOptionEnabled(SkInnerOptionType optionType) const;

  /*!
   * \brief Set option value from an aclskOption structure
   * \param option Pointer to the aclskOption structure containing value to set
   */
  void SetOptOptionValue(const aclskOption *option);

  /*!
   * \brief Parse and populate options from an aclskOptions structure
   * \param options Pointer to the aclskOptions structure containing all options
   */
  void ParseOptions(const aclskOptions *options);

  /*!
   * \brief Convert options to JSON format
   * \return JSON object containing all option values
   */
  nlohmann::ordered_json ToJson() const;

 private:
  void RegisterDefaultOptions();
  void RegisterDefaultSkOptions();
  void RegisterDefaultInnerOptions();
  void ApplyArchSpecificOptions();

  std::unordered_map<aclskOptionType, std::unique_ptr<OptOptionBase>> optionMap;
  std::unordered_map<SkInnerOptionType, std::unique_ptr<OptOptionBase>> innerOptionMap;
};

struct OptionDumpInfo {
  std::string name;
  int32_t type = -1;

  // 三种可能的值
  int64_t intValue = 0;
  std::vector<std::string> stringListValue;
  std::unordered_map<std::string, std::vector<std::string>> mapValue;

  // 标记当前是哪种类型
  enum class ValueType { INT, STRING_LIST, MAP, NONE } valueType = ValueType::NONE;
};

inline std::vector<OptionDumpInfo> CollectAllOptions(const SuperKernelOptionsManager &optsMgr) {
  std::vector<OptionDumpInfo> infos;

  for (int32_t i = 0; i < static_cast<int32_t>(aclskOptionType::SK_OPTION_MAX); ++i) {
    auto type = static_cast<aclskOptionType>(i);
    const OptOptionBase *opt = optsMgr.GetOption(type);
    if (!opt) continue;

    OptionDumpInfo info;
    info.name = opt->GetName();
    info.type = static_cast<int>(type);

    // 按类型填充纯数据，不做任何格式化
    switch (type) {
      case aclskOptionType::PRELOAD_CODE:
      case aclskOptionType::SPLIT_MODE:
      case aclskOptionType::DEBUG_SYNC_ALL:
      case aclskOptionType::STREAM_FUSION:
      case aclskOptionType::CONSTANT_CODEGEN:
      case aclskOptionType::AUTO_OP_PARALLEL:
      case aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK:
      case aclskOptionType::DEBUG_OP_EXEC_TRACE:
      case aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM:
        info.valueType = OptionDumpInfo::ValueType::INT;
        info.intValue = opt->GetIntValue();
        break;

      case aclskOptionType::DCCI_DISABLE_ON_KERNEL:
      case aclskOptionType::DCCI_BEFORE_KERNEL_START:
      case aclskOptionType::DCCI_AFTER_KERNEL_END:
      case aclskOptionType::UBUF_LOCK_IGNORE_KERNEL:
        info.valueType = OptionDumpInfo::ValueType::STRING_LIST;
        info.stringListValue = opt->GetStringListValue();
        break;

      case aclskOptionType::OPT_EXTEND_OPTION:
      case aclskOptionType::DEBUG_EXTEND_OPTION:
        info.valueType = OptionDumpInfo::ValueType::MAP;
        info.mapValue = opt->GetMapValue();
        break;

      default:
        break;
    }

    infos.push_back(std::move(info));
  }

  return infos;
}

#endif
