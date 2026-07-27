/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TESTS_COMMON_ATT_COMMON_GEN_UTILS_H_
#define AUTOFUSE_TESTS_COMMON_ATT_COMMON_GEN_UTILS_H_

#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace att {
namespace test {

inline std::string RemoveAutoFuseTilingHeadGuards(const std::string &input) {
  std::istringstream iss(input);
  std::ostringstream oss;
  std::string line;
  const std::string guard_token = "__AUTOFUSE_TILING_FUNC_COMMON_H__";
  while (std::getline(iss, line)) {
    if (line.find(guard_token) == std::string::npos) {
      oss << line << "\n";
    }
  }
  return oss.str();
}

inline bool IsSplitHeaderKey(const std::string &key) {
  return key == "TilingStateHeader" || key == "TilingLogHeader" || key == "TilingPgoHeader" ||
         key == "TilingSolverHeader" || key == "TilingApiHeader" || key == "TilingBaseHeader" ||
         key == "TilingEntryHeader" || key == "TilingTailHeader";
}

inline std::string RemoveSplitIncludes(const std::string &value) {
  const std::set<std::string> split_includes = {
      "#include \"autofuse_tiling_func_common.h\"", "#include \"autofuse_tiling_func_base.h\"",
      "#include \"autofuse_tiling_func_state.h\"",  "#include \"autofuse_tiling_func_log.h\"",
      "#include \"autofuse_tiling_func_pgo.h\"",    "#include \"autofuse_tiling_func_api.h\"",
      "#include \"autofuse_tiling_func_solver.h\"", "#include \"autofuse_tiling_func_entry.h\"",
      "#include \"autofuse_tiling_func_tail.h\"",   "#include \"autofuse_tiling_data.h\""};
  std::istringstream input(value);
  std::ostringstream output;
  std::string line;
  bool in_include_prefix = true;
  while (std::getline(input, line)) {
    if (!in_include_prefix) {
      output << line << '\n';
      continue;
    }
    if (split_includes.count(line) != 0U) {
      continue;
    }
    in_include_prefix = line.empty() || line.rfind("#include ", 0U) == 0U;
    output << line << '\n';
  }
  return output.str();
}

inline void CombineTilings(const std::map<std::string, std::string> &tilings, std::string &result) {
  const std::string tiling_head = "TilingHead";
  const std::string tiling_data = "TilingData";
  result += RemoveAutoFuseTilingHeadGuards(tilings.at(tiling_head));
  for (const auto &[key, value] : tilings) {
    if (key == tiling_head || IsSplitHeaderKey(key) || key.find(tiling_data) != std::string::npos) {
      continue;
    }
    result += RemoveSplitIncludes(value);
    if (!result.empty() && result.back() != '\n') {
      result += '\n';
    }
  }
}

inline void AddHeaderGuardToFile(const std::string &file_name, const std::string &macro_name) {
  std::string content;
  std::ifstream in_file(file_name);
  if (in_file.is_open()) {
    std::string line;
    while (std::getline(in_file, line)) {
      content += line + "\n";
    }
    in_file.close();
  }
  std::ofstream out_file(file_name, std::ios::out);
  out_file << "#ifndef " << macro_name << "\n";
  out_file << "#define " << macro_name << "\n\n";
  out_file << content << "\n";
  out_file << "#endif // " << macro_name << "\n";
  out_file.close();
}

}  // namespace test
}  // namespace att

#endif  // AUTOFUSE_TESTS_COMMON_ATT_COMMON_GEN_UTILS_H_
