/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_OPTION_OPTIMIZATION_OPTION_H_
#define INC_GRAPH_OPTION_OPTIMIZATION_OPTION_H_

#include <map>
#include <unordered_map>
#include <unordered_set>
#include "graph/ge_error_codes_af.h"
#include "optimization_option_info.h"

namespace af {
class OptimizationOption {
 public:
  OptimizationOption() = default;
  ~OptimizationOption() = default;

  graphStatus Initialize(const std::map<std::string, std::string> &ge_options,
                         const std::unordered_map<std::string, OoInfo> &registered_options);
  graphStatus Initialize(const std::map<std::string, std::string> &ge_options,
                         const std::unordered_map<std::string, OoInfo> &registered_options,
                         const std::unordered_set<std::string> &forbidden_option_set);
  graphStatus GetValue(const std::string &opt_name, std::string &opt_value) const;
  static graphStatus IsOoLevelValid(const std::string &oo_level);
  static graphStatus IsOptionValueValid(const std::string &opt_name, const std::string &opt_value,
                                        OoInfo::ValueChecker checker);
  graphStatus RefreshPassSwitch(const std::string &fusion_config_str);

 private:
  graphStatus InitWorkingOolevel(const std::map<std::string, std::string> &ge_options);
  void PrintAllWorkingOo();
  graphStatus SetPassSwitch(const std::string &pass_switch_str,
                            const std::unordered_set<std::string> &forbidden_option_set, bool force_update);
  graphStatus UpdatePassSwitchByOption(const std::map<std::string, std::string> &ge_options,
                                       const std::unordered_set<std::string> &forbidden_option_set);
  bool IsPassConfiguredWithOptimizationSwitch(const std::string &pass_name) const;

 private:
  OoLevel working_oo_level_{OoLevel::kEnd};
  std::unordered_map<std::string, std::string> working_opt_names_to_value_;
};
}  // namespace af
#endif  //  INC_GRAPH_OPTION_OPTIMIZATION_OPTION_H_
