/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_OPTIMIZATION_OPTION_REGISTRY_H
#define INC_REGISTER_OPTIMIZATION_OPTION_REGISTRY_H

#include <memory>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include "graph/option/optimization_option_info.h"
#include "graph/ge_error_codes_af.h"

namespace af {
class OptionRegistry {
 public:
  static OptionRegistry &GetInstance();
  void Register(const OoInfo &option);
  const OoInfo *FindOptInfo(const std::string &opt_name) const;
  std::unordered_map<std::string, OoInfo> GetVisibleOptions(OoEntryPoint entry_point) const;

  const std::unordered_map<std::string, OoInfo> &GetRegisteredOptTable() const {
    return registered_opt_table_;
  }
  OptionRegistry(const OptionRegistry &) = delete;
  OptionRegistry &operator=(const OptionRegistry &) = delete;

 private:
  OptionRegistry() = default;
  ~OptionRegistry() = default;

 private:
  std::unordered_map<std::string, OoInfo> registered_opt_table_;
};

class PassOptionRegistry {
 public:
  using OptNameTable =
      std::unordered_map<std::string, std::array<std::string, static_cast<uint32_t>(OoHierarchy::kEnd)>>;
  static PassOptionRegistry &GetInstance();
  void Register(const std::string &pass_name, const std::map<OoHierarchy, std::string> &option_names);
  graphStatus FindOptionNamesByPassName(const std::string &pass_name, std::vector<std::string> &option_names) const;

  PassOptionRegistry(const PassOptionRegistry &) = delete;
  PassOptionRegistry &operator=(const PassOptionRegistry &) = delete;

 private:
  PassOptionRegistry() = default;
  ~PassOptionRegistry() = default;

 private:
  OptNameTable pass_names_to_options_;
};

class OptionRegister {
 public:
  explicit OptionRegister(std::string opt_name, OoHierarchy hierarchy = OoHierarchy::kH1)
      : opt_reg_data_(new(std::nothrow) OoInfo(std::move(opt_name), hierarchy)) {}
  OptionRegister(const OptionRegister &other);
  OptionRegister &SetDefaultValues(const std::map<OoLevel, std::string> &opt_values);
  OptionRegister &SetOptLevel(const std::vector<OoLevel> &levels);
  OptionRegister &SetVisibility(const std::vector<OoEntryPoint> &entry_points);
  OptionRegister &SetOptValueChecker(OoInfo::ValueChecker opt_checker);
  OptionRegister &SetHelpText(std::string opt_help);
  OptionRegister &SetShowName(OoEntryPoint entry_point, std::string show_name, OoCategory category = OoCategory::kEnd);

  OptionRegister &operator=(const OptionRegister &) = delete;
  OptionRegister &operator=(OptionRegister &&) = delete;
  OptionRegister(OptionRegister &&) = delete;

 private:
  std::unique_ptr<OoInfo> opt_reg_data_;
};

class PassOptionRegister {
 public:
  explicit PassOptionRegister(std::string pass_name)
      : pass_reg_data_(new(std::nothrow) PassOptRegData({std::move(pass_name), {}, {}})) {}
  PassOptionRegister(const PassOptionRegister &other);
  PassOptionRegister &SetOptLevel(const std::vector<OoLevel> &levels);
  PassOptionRegister &BindSwitchOption(const std::string &opt_name, OoHierarchy hierarchy = OoHierarchy::kH1);

  PassOptionRegister &operator=(const PassOptionRegister &) = delete;
  PassOptionRegister &operator=(PassOptionRegister &&) = delete;
  PassOptionRegister(PassOptionRegister &&) = delete;

 private:
  struct PassOptRegData {
    std::string pass_name;
    std::vector<OoLevel> levels;  // levels is used when options is empty
    std::map<OoHierarchy, std::string> options;
  };
  std::unique_ptr<PassOptRegData> pass_reg_data_;
};
}  // namespace af

#ifdef __GNUC__
#define ATTR_USED __attribute__((used))
#else
#define ATTR_USED
#endif

#define REG_OPTION(opt_name, ...) REG_UNIQUE_OPTION(__COUNTER__, opt_name, ##__VA_ARGS__)
#define REG_UNIQUE_OPTION(counter, opt_name, ...) REG_UNIQUE_OPTION_WRAPPER(counter, opt_name, ##__VA_ARGS__)
#define REG_UNIQUE_OPTION_WRAPPER(counter, opt_name, ...) \
  static ge::OptionRegister opt_register_##counter ATTR_USED = ge::OptionRegister((opt_name), ##__VA_ARGS__)

#define DEFAULT_VALUES(...) SetDefaultValues(std::map<ge::OoLevel, std::string>{__VA_ARGS__})
#define LEVELS(level1, ...) SetOptLevel(std::vector<ge::OoLevel>({(level1), ##__VA_ARGS__}))
#define VISIBILITY(...) SetVisibility(std::vector<ge::OoEntryPoint>({__VA_ARGS__}))
#define CHECKER(checker_func) SetOptValueChecker((checker_func))
#define HELP(help_text) SetHelpText((help_text))
#define SHOW_NAME(entry, show_name, ...) SetShowName((entry), (show_name), ##__VA_ARGS__)
#define SWITCH_OPT(opt_name, ...) BindSwitchOption((opt_name), ##__VA_ARGS__)

#define REG_PASS_OPTION(pass_name) REG_UNIQUE_PASS_OPTION(__COUNTER__, pass_name)
#define REG_UNIQUE_PASS_OPTION(counter, pass_name) REG_UNIQUE_PASS_OPTION_WRAPPER(counter, pass_name)
#define REG_UNIQUE_PASS_OPTION_WRAPPER(counter, pass_name) \
  static ge::PassOptionRegister pass_opt_register_##counter ATTR_USED = ge::PassOptionRegister((pass_name))

#endif  // INC_REGISTER_OPTIMIZATION_OPTION_REGISTRY_H
