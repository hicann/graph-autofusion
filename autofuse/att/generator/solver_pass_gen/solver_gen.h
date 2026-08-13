/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_SOLVER_GEN_H_
#define ATT_SOLVER_GEN_H_
#include <string>
#include <cstdint>

namespace att {
constexpr char kSolverGenError[] = "Solver Gen Error";
inline std::string GetSmoothString(std::string str) {
  std::string ret;
  std::string target = "Ceiling";
  size_t pos = 0;
  while ((pos = str.find(target)) != std::string::npos) {
    ret += str.substr(0, pos);
    str.erase(0, pos + target.length());
  }
  ret += str;
  return ret;
}

class SolverGen {
 public:
  SolverGen(const std::string &tiling_case_id, const std::string &type_name)
      : tiling_case_id_(tiling_case_id), type_name_(type_name) {};
  virtual ~SolverGen() = default;

 protected:
  virtual std::string GenSolverClassImpl() = 0;
  virtual std::string GenSolverFuncImpl() = 0;
  virtual std::string GenSolverFuncInvoke() = 0;
  std::string tiling_case_id_;
  std::string type_name_;
};
}  // namespace att
#endif
