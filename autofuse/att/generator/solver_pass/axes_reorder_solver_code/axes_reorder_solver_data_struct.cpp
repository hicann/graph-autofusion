/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "axes_reorder_solver_code_common.h"

namespace att {

// ============================================================================
// Section 1: Core Data Structures
// ============================================================================

std::string GenConstraintType() {
  std::string codes;
  std::string strs;
  strs += "ConstraintType:约束类型\n";
  strs += "  LOCAL_BUFFER:仅与内存占用相关的约束, 如s1t * s2t < UB\n";
  strs += "  LB_MIXED:与内存占用相关的约束\n";
  strs += "  MC_MIXED:纯多核相关的约束\n";
  codes += AddAnotationBlock(strs);
  codes += "enum class ConstraintType {\n";
  codes += "  LOCAL_BUFFER = 0,\n";
  codes += "  LB_MIXED = 1,\n";
  codes += "  MC_MIXED = 2,\n";
  codes += "};\n\n";
  return codes;
}

std::string GenStructDef() {
  std::string codes;
  codes += "struct Variable;\n";
  codes += "struct TilingVariable;\n";
  codes += "struct Constraint;\n";
  codes +=
      "using ConsEvalFuncPtr = int64_t (*)(TilingVariable **rel_tiling_vars, Variable **rel_input_shapes, int64_t "
      "rel_hw_spec);\n";
  codes += "using GetUpperBoundFuncPtr = int64_t (*)(Variable **rel_ori_dims);\n";
  codes += "\n";
  return codes;
}

std::string GenVariable() {
  std::string codes;
  codes += "struct Variable {\n";
  codes += "  int64_t value = -1;\n";
  codes += "};\n";
  codes += "\n";
  return codes;
}

std::string GenConstraint() {
  std::string codes;
  codes += "struct Constraint {\n";
  codes += "  int64_t rel_hw_spec = 0;\n";
  codes += "  uint32_t rel_tiling_vars_size = 0u;\n";
  codes += "  uint32_t rel_in_shapes_size = 0u;\n";
  codes += "  TilingVariable **rel_tiling_vars = nullptr;\n";
  codes += "  Variable **rel_in_shapes = nullptr;\n";
  codes += "  ConsEvalFuncPtr eval = nullptr;\n";
  codes += "  ConstraintType type;\n";
  codes += "};\n";
  codes += "\n";
  return codes;
}

std::string GenTilingVariable() {
  std::string codes;
  codes += "struct TilingVariable : public Variable {\n";
  codes += "  int64_t align = 1;\n";
  codes += "  int64_t prompt_align = 1;\n";
  codes += "  int64_t data_type_size = 4;\n";
  codes += "  bool notail = false;\n";
  codes += "  bool mc_related = false;\n";
  codes += "  TilingVariable *notail_var = nullptr;\n";
  codes += "  uint32_t rel_cons_size = 0u;\n";
  codes += "  uint32_t upper_bound_vars_size = 0u;\n";
  codes += "  Variable **upper_bound_vars = nullptr;\n";
  codes += "  Constraint **rel_cons = nullptr;\n";
  codes += "  GetUpperBoundFuncPtr upper_bound = nullptr;\n";
  codes += "  size_t order{0UL};\n";
  codes += "  __attribute__((always_inline)) bool SetValue(int64_t val) noexcept{\n";
  codes += "    return (val > 0) ? (value = val, true) : false;\n";
  codes += "  }\n";
  codes += "};\n";
  codes += "\n";
  return codes;
}

std::string GenReorderSolverInputDebugString() {
  return R"(
std::string DebugString() const {
  std::stringstream ss;
  ss << "result_id: " << result_id << ", group_id: " << group_id << ", case_id: " << case_id << ", sub_case_id: " << sub_case_id;
  ss << "core_num: " << core_num;
  ss << ", ub_size: " << ub_size;
  if (input_vars != nullptr) {
    for (uint32_t i = 0; i < input_vars_size; i++) {
      if (input_vars[i] != nullptr) {
        ss << ", input_vars[" << i << "]: " << input_vars[i]->value;
      }
    }
  }
  if (tiling_vars != nullptr) {
    for (uint32_t i = 0; i < tiling_vars_size; i++) {
      if (tiling_vars[i] != nullptr) {
        ss << ", tiling_vars[" << i << "]: " << tiling_vars[i]->value;
      }
    }
  }
  if (pure_mc_vars != nullptr) {
    for (uint32_t i = 0; i < pure_mc_vars_size; i++) {
      if (pure_mc_vars[i] != nullptr) {
        ss << ", pure_mc_vars[" << i << "]: " << pure_mc_vars[i]->value;
      }
    }
  }
  if (local_buffer_vars != nullptr) {
    for (uint32_t i = 0; i < local_buffer_vars_size; i++) {
      if (local_buffer_vars[i] != nullptr) {
        ss << ", local_buffer_vars[" << i << "]: " << local_buffer_vars[i]->value;
      }
    }
  }
  ss << ", all_cons_size: " << all_cons_size;
  ss << ", ub_threshold: " << ub_threshold;
  ss << ", corenum_threshold: " << corenum_threshold;
  ss << ", perf_threshold: " << perf_threshold;
  return ss.str();
}
)";
}

std::string GenAxesReorderSolverInput() {
  std::string codes;
  codes += "struct AxesReorderSolverInput {\n";
  codes += "  uint32_t result_id = 0u;\n";
  codes += "  uint32_t group_id = 0u;\n";
  codes += "  uint32_t case_id = 0u;\n";
  codes += "  uint32_t sub_case_id = 0u;\n";
  codes += "  uint32_t core_num = 0u;\n";
  codes += "  uint32_t ub_size = 0u;\n";
  codes += "  uint32_t input_vars_size = 0u;\n";
  codes += "  uint32_t tiling_vars_size = 0u;\n";
  codes += "  uint32_t pure_mc_vars_size = 0u;\n";
  codes += "  uint32_t local_buffer_vars_size = 0u;\n";
  codes += "  uint32_t all_cons_size = 0u;\n";
  codes += "  double ub_threshold = 0.2f;\n";
  codes += "  double corenum_threshold = 0.4f;\n";
  codes += "  double perf_threshold = 0000.0f;\n";  // 经验值，理论上不使用
  codes += "  Variable **input_vars = nullptr;\n";
  codes += "  TilingVariable **tiling_vars = nullptr;\n";
  codes += "  TilingVariable **pure_mc_vars = nullptr;\n";
  codes += "  TilingVariable **local_buffer_vars = nullptr;\n";
  codes += "  TilingVariable **ordered_local_buffer_vars = nullptr;\n";
  codes += "  uint32_t *ordered_to_canonical = nullptr;\n";
  codes += "  Constraint **all_cons = nullptr;\n";
  codes += GenReorderSolverInputDebugString();
  codes += "};\n";
  codes += "\n";
  return codes;
}

}  // namespace att
