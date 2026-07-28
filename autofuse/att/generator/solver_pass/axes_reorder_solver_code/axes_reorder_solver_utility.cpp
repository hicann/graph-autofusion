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
// Section 8: Utility Functions
// ============================================================================

std::string GenIsSatisfyCons() {
  return R"(
bool AxesReorderSolver::IsSatisfyCons(TilingVariable *var, int64_t val, ConstraintType cons_type, double ratio) {
  auto old = var->value;
  var->SetValue(val);
  bool satisfied = SatisfyCons(var, cons_type, ratio);
  var->SetValue(old);
  return satisfied;
}
)";
}

// ============================================================================
// Helper functions for GenBinaryFindUpperBoundSatisfiedUBLimit
// ============================================================================

std::string GenBinaryFindUBLimit_MultiVarSetup() {
  return R"(
  (void)idxs;  // Suppress unused parameter warning

  if (count == 0) return false;

  const int64_t align = vars[0]->align;
  if (align <= 0) {
    OP_LOGW(OP_NAME, "MultiVarUBLimit: invalid align %ld", align);
    return false;
  }
  const int64_t raw_upper = vars[0]->value;
  const int64_t aligned_lower = CeilDiv(lower_bound, align) * align;
  const int64_t aligned_upper = (raw_upper / align) * align;
  if (aligned_lower > aligned_upper) {
    OP_LOGW(OP_NAME, "MultiVarUBLimit: aligned_lower > aligned_upper, lower:%ld, upper:%ld, align:%ld",
            aligned_lower, aligned_upper, align);
    return false;
  }

  // Save original values
  std::vector<int64_t> old_values(count);
  for (size_t i = 0; i < count; ++i) {
    old_values[i] = vars[i]->value;
  }

  // Check if aligned_upper satisfies constraint when all variables are set to it
  for (size_t i = 0; i < count; ++i) {
    vars[i]->SetValue(aligned_upper);
  }
  bool all_satisfied = SatisfyCons(ConstraintType::LOCAL_BUFFER);
  if (all_satisfied) {
    result = aligned_upper;
    return true;
  }
)";
}

std::string GenBinaryFindUBLimit_SearchLoop() {
  return R"(
  // Binary search on align-grid for maximum value satisfying UB constraint
  int64_t left_idx = aligned_lower / align;
  int64_t right_idx = aligned_upper / align;
  result = aligned_lower;

  while (left_idx <= right_idx) {
    int64_t mid_idx = left_idx + (right_idx - left_idx) / 2;
    int64_t candidate = mid_idx * align;

    // Set all variables to candidate and check constraint
    for (size_t i = 0; i < count; ++i) {
      vars[i]->SetValue(candidate);
    }
    bool mid_satisfied = SatisfyCons(ConstraintType::LOCAL_BUFFER);

    if (mid_satisfied) {
      result = candidate;
      left_idx = mid_idx + 1;
    } else {
      right_idx = mid_idx - 1;
    }
  }
)";
}

std::string GenBinaryFindUBLimit_RestoreValues() {
  return R"(
  // Restore original values
  for (size_t i = 0; i < count; ++i) {
    vars[i]->SetValue(old_values[i]);
  }
  return true;
)";
}

std::string GenBinaryFindUpperBoundSatisfiedUBLimit() {
  std::string codes;
  codes += "// Single-variable version delegates to multi-variable version\n";
  codes += "bool AxesReorderSolver::BinaryFindUpperBoundSatisfiedUBLimit(TilingVariable *var, int64_t lower_bound,\n";
  codes += "                                                             int64_t &result) {\n";
  codes += "  // Delegate to multi-variable version\n";
  codes += "  TilingVariable* vars[] = {var};\n";
  codes += "  uint32_t idxs[] = {0};\n";
  codes += "  return BinaryFindUpperBoundSatisfiedUBLimit(vars, idxs, 1, lower_bound, result);\n";
  codes += "}\n\n";

  codes += "// Multi-variable version of BinaryFindUpperBoundSatisfiedUBLimit\n";
  codes += "// Simultaneously sets all variables to the same value and checks UB constraint\n";
  codes += "bool AxesReorderSolver::BinaryFindUpperBoundSatisfiedUBLimit(\n";
  codes += "    TilingVariable **vars, uint32_t *idxs, size_t count,\n";
  codes += "    int64_t lower_bound, int64_t &result) {\n";
  codes += GenBinaryFindUBLimit_MultiVarSetup();
  codes += GenBinaryFindUBLimit_SearchLoop();
  codes += GenBinaryFindUBLimit_RestoreValues();
  codes += "}\n";

  return codes;
}

// ============================================================================
// Helper functions for BinaryFindLowerBoundSatisfiedUBThresholdCond
// ============================================================================

std::string GenBinaryFindUBThr_MultiVarSetup() {
  return R"(
  const int64_t align = vars[0]->align;
  if (align <= 0) {
    OP_LOGW(OP_NAME, "MultiVarUBThr: invalid align %ld", align);
    return false;
  }
  const int64_t aligned_lower = CeilDiv(lower_bound, align) * align;
  const int64_t aligned_upper = (upper_bound / align) * align;
  if (aligned_lower > aligned_upper) {
    OP_LOGW(OP_NAME, "MultiVarUBThr: aligned_lower > aligned_upper, lower:%ld, upper:%ld, align:%ld",
            aligned_lower, aligned_upper, align);
    return false;
  }

  // Save original values
  std::vector<int64_t> old_values(count);
  for (size_t i = 0; i < count; ++i) {
    old_values[i] = vars[i]->value;
  }
)";
}

std::string GenBinaryFindUBThr_CheckLowerBound() {
  return R"(
  // Check if aligned lower_bound satisfies conditions for all variables
  for (size_t i = 0; i < count; ++i) {
    vars[i]->SetValue(aligned_lower);
  }
  bool bound_ok = true;
  for (size_t i = 0; i < count; ++i) {
    if (!SatisfyUBSizeCacheLine(idxs[i])) {
      bound_ok = false;
      break;
    }
  }
  // Check if UB usage is ABOVE threshold (i.e., !SatisfyCons with threshold ratio)
  if (bound_ok && IsSatisfyCons(vars[0], vars[0]->value, ConstraintType::LOCAL_BUFFER, input_.ub_threshold)) {
    bound_ok = false;
  }
  result = aligned_lower;
  if (bound_ok) {
    return true;
  }
)";
}

std::string GenBinaryFindUBThr_SearchLoop() {
  return R"(
  // Binary search on align-grid for minimum value that satisfies conditions for all variables
  int64_t left_idx = aligned_lower / align;
  int64_t right_idx = aligned_upper / align;
  result = aligned_upper;
  while (left_idx <= right_idx) {
    int64_t mid_idx = left_idx + (right_idx - left_idx) / 2;
    int64_t candidate = mid_idx * align;
    // Set all variables to candidate and check conditions
    for (size_t i = 0; i < count; ++i) {
      vars[i]->SetValue(candidate);
    }
    bool mid_satisfied = true;
    for (size_t i = 0; i < count; ++i) {
      if (!SatisfyUBSizeCacheLine(idxs[i])) {
        mid_satisfied = false;
        break;
      }
    }
    // Check if UB usage is ABOVE threshold
    if (mid_satisfied && IsSatisfyCons(vars[0], candidate, ConstraintType::LOCAL_BUFFER, input_.ub_threshold)) {
      mid_satisfied = false;
    }
    if (mid_satisfied) {
      result = candidate;
      right_idx = mid_idx - 1;
    } else {
      left_idx = mid_idx + 1;
    }
  }
)";
}

std::string GenBinaryFindUBThr_RestoreValues() {
  return R"(
  // Restore original values
  for (size_t i = 0; i < count; ++i) {
    vars[i]->SetValue(old_values[i]);
  }
)";
}

std::string GenBinaryFindLowerBoundSatisfiedUBThresholdCond() {
  return R"(
// Single-variable version delegates to multi-variable version
bool AxesReorderSolver::BinaryFindLowerBoundSatisfiedUBThresholdCond(TilingVariable *var, uint32_t idx,
                                                                     int64_t lower_bound, int64_t &result) {
  // Delegate to multi-variable version
  OP_LOGD(OP_NAME, "Begin to search ub threshold limit bound, upper_bound:%ld, lower_bound:%ld, id:%u",
          var->value, lower_bound, idx);
  TilingVariable* vars[] = {var};
  uint32_t idxs[] = {idx};
  return BinaryFindLowerBoundSatisfiedUBThresholdCond(vars, idxs, 1, lower_bound, result);
}

// Multi-variable version of BinaryFindLowerBoundSatisfiedUBThresholdCond
// Simultaneously sets all variables to the same value and checks UB threshold
bool AxesReorderSolver::BinaryFindLowerBoundSatisfiedUBThresholdCond(TilingVariable **vars, uint32_t *idxs,
                                                                     size_t count, int64_t lower_bound,
                                                                     int64_t &result) {
  if (count == 0) return false;
  int64_t upper_bound = vars[0]->value;
  if (upper_bound < lower_bound) {
    OP_LOGW(OP_NAME, "MultiVarUBThr: upper_bound < lower_bound, return false");
    return false;
  }
)" + GenBinaryFindUBThr_MultiVarSetup() +
         GenBinaryFindUBThr_CheckLowerBound() + GenBinaryFindUBThr_SearchLoop() + GenBinaryFindUBThr_RestoreValues() +
         R"(
  return true;
}
)";
}

// ============================================================================
// Helper functions for GenBinaryFindLowerBoundSatisfiedCoreNum (basic version)
// ============================================================================

std::string GenBinaryFindCoreNum_BasicSetup() {
  return R"(
  if (count == 0) return {lower_bound, lower_bound};
  int64_t upper_bound = vars[0]->value;
  if (lower_bound >= upper_bound) {
    OP_LOGW(OP_NAME, "MultiVarCoreNum: lower_bound >= upper_bound, return lower_bound");
    return {lower_bound, lower_bound};
  }
  // Save original values
  std::vector<int64_t> old_values(count);
  for (size_t i = 0; i < count; ++i) {
    old_values[i] = vars[i]->value;
  }
  // Calculate max core number threshold
  int64_t max_core_num = static_cast<int64_t>(
      input_.corenum_threshold * static_cast<double>(input_.core_num));
  // Check if upper_bound already satisfies condition
  (void)MulticoreTilingCore(false);
  int64_t available_core_num = 0L;
  (void)CalRealUsedCoreNum(available_core_num);
  if (available_core_num >= max_core_num) {
    return {upper_bound, upper_bound};
  }
)";
}

std::string GenBinaryFindCoreNum_BasicSearch() {
  return R"(
  // Binary search for value that satisfies core number constraint
  int64_t left = lower_bound;
  int64_t right = upper_bound - vars[0]->align;
  while (left <= right) {
    int64_t mid = left + (right - left) / 2;
    // Set all variables to mid and check condition
    for (size_t i = 0; i < count; ++i) {
      vars[i]->SetValue(mid);
    }
    // Check if all variables satisfy cache line constraint
    bool all_cache_line_ok = true;
    for (size_t i = 0; i < count; ++i) {
      if (!SatisfyUBSizeCacheLine(idxs[i])) {
        all_cache_line_ok = false;
        break;
      }
    }
    if (!all_cache_line_ok ||
        (CalRealUsedCoreNum(available_core_num) && (available_core_num >= max_core_num))) {
      left = mid + vars[0]->align;
    } else {
      right = mid - vars[0]->align;
    }
    OP_LOGD(OP_NAME, "MultiVarCoreNum: left=%ld, right=%ld, mid=%ld, core_num=%ld, max=%ld",
            left, right, mid, available_core_num, max_core_num);
  }
)";
}

std::string GenBinaryFindCoreNum_BasicRestore() {
  return R"(
  // Restore original values
  for (size_t i = 0; i < count; ++i) {
    vars[i]->SetValue(old_values[i]);
  }
)";
}

std::string GenBinaryFindCoreNum_BasicReturn() {
  return R"(
  return {right, left};
)";
}

std::string GenBinaryFindLowerBoundSatisfiedCoreNum() {
  std::string codes;
  codes += "// Single-variable version delegates to multi-variable version\n";
  codes +=
      "inline std::pair<int64_t, int64_t> AxesReorderSolver::BinaryFindLowerBoundSatisfiedCoreNum(TilingVariable "
      "*var,\n";
  codes += "                                                                                           uint32_t idx,\n";
  codes +=
      "                                                                                           int64_t lower_bound) "
      "{\n";
  codes += "  // Delegate to multi-variable version\n";
  codes += "  TilingVariable* vars[] = {var};\n";
  codes += "  uint32_t idxs[] = {idx};\n";
  codes += "  return BinaryFindLowerBoundSatisfiedCoreNum(vars, idxs, 1, lower_bound);\n";
  codes += "}\n\n";

  codes += "// Multi-variable version of BinaryFindLowerBoundSatisfiedCoreNum\n";
  codes += "// Simultaneously sets all variables to the same value and checks core number constraint\n";
  codes += "std::pair<int64_t, int64_t> AxesReorderSolver::BinaryFindLowerBoundSatisfiedCoreNum(\n";
  codes += "    TilingVariable **vars, uint32_t *idxs, size_t count, int64_t lower_bound) {\n";
  codes += GenBinaryFindCoreNum_BasicSetup();
  codes += GenBinaryFindCoreNum_BasicSearch();
  codes += GenBinaryFindCoreNum_BasicRestore();
  codes += GenBinaryFindCoreNum_BasicReturn();
  codes += "}\n\n";

  codes += "// ============================================================================\n";
  codes += "// Helper functions for BinaryFindLowerBoundSatisfiedCoreNum (advanced version)\n";
  codes += "// ============================================================================\n\n";

  return codes;
}

std::string GenBinaryFindCoreNum_AdvancedSetup() {
  return R"(
  // Find the maximum upper bound for binary search range
  int64_t max_upper_bound = upper_bounds[0];
  for (size_t i = 1; i < count; ++i) {
    if (upper_bounds[i] > max_upper_bound) {
      max_upper_bound = upper_bounds[i];
    }
  }

  if (lower_bound >= max_upper_bound) {
    OP_LOGW(OP_NAME, "MultiVarCoreNumWithUpperBounds: lower_bound >= max_upper_bound, return lower_bound");
    return {lower_bound, lower_bound};
  }
)";
}

std::string GenBinaryFindCoreNum_AdvancedSaveValues() {
  return R"(
  // Save original values
  std::vector<int64_t> old_values(count);
  for (size_t i = 0; i < count; ++i) {
    old_values[i] = vars[i]->value;
  }
)";
}

std::string GenBinaryFindCoreNum_AdvancedCheckUpperBound() {
  return R"(
  // Calculate max core number threshold
  int64_t max_core_num = static_cast<int64_t>(
      input_.corenum_threshold * static_cast<double>(input_.core_num));

  // Check if max_upper_bound already satisfies condition
  for (size_t i = 0; i < count; ++i) {
    int64_t val = (max_upper_bound <= upper_bounds[i]) ? max_upper_bound : upper_bounds[i];
    vars[i]->SetValue(val);
  }
  (void)MulticoreTilingCore(false);
  int64_t available_core_num = 0L;
  (void)CalRealUsedCoreNum(available_core_num);

  if (available_core_num >= max_core_num) {
    // All variables at their respective bounds satisfy the condition
    std::pair<int64_t, int64_t> result = {upper_bounds[0], upper_bounds[0]};
    for (size_t i = 1; i < count; ++i) {
      result.first = std::min(result.first, upper_bounds[i]);
      result.second = std::max(result.second, upper_bounds[i]);
    }
    return result;
  }
)";
}

std::string GenBinaryFindCoreNum_AdvancedBinarySearch() {
  return R"(
  // Binary search for value that satisfies core number constraint
  int64_t left = lower_bound;
  int64_t right = max_upper_bound - vars[0]->align;

  while (left <= right) {
    int64_t mid = left + (right - left) / 2;

    // Set each variable to mid, but capped at its individual upper bound
    for (size_t i = 0; i < count; ++i) {
      int64_t val = (mid <= upper_bounds[i]) ? mid : upper_bounds[i];
      vars[i]->SetValue(val);
    }

    // Check if all variables satisfy cache line constraint
    bool all_cache_line_ok = true;
    for (size_t i = 0; i < count; ++i) {
      if (!SatisfyUBSizeCacheLine(idxs[i])) {
        all_cache_line_ok = false;
        break;
      }
    }

    // Check if UB usage is ABOVE threshold
    if (all_cache_line_ok && IsSatisfyCons(vars[0], mid, ConstraintType::LOCAL_BUFFER, input_.ub_threshold)) {
      all_cache_line_ok = false;
    }

    if (!all_cache_line_ok || (CalRealUsedCoreNum(available_core_num) && (available_core_num >= max_core_num))) {
      left = mid + vars[0]->align;
    } else {
      right = mid - vars[0]->align;
    }

    OP_LOGD(OP_NAME, "MultiVarCoreNumWithUpperBounds: left=%ld, right=%ld, mid=%ld, core_num=%ld, max=%ld",
            left, right, mid, available_core_num, max_core_num);
  }
)";
}

std::string GenBinaryFindCoreNum_AdvancedRestoreValues() {
  return R"(
  // Restore original values
  for (size_t i = 0; i < count; ++i) {
    vars[i]->SetValue(old_values[i]);
  }
)";
}

std::string GenBinaryFindCoreNum_AdvancedCalculateResult() {
  return R"(
  // Return values corresponding to each variable in order
  // For each variable: use left (converged value) clipped to its upper bound
  // This ensures return[i] corresponds to vars[i]
  //
  // For dual-axis non-symmetric case:
  // - vars[0] (a-axis): smaller upper bound (e.g., 32) → gets smaller value
  // - vars[1] (b-axis): larger upper bound (e.g., 128) → gets larger value
  // - Returns {value_for_vars[0], value_for_vars[1]}
  //
  // Example: if upper_bounds = {32, 128} and left = 64
  // - results[0] = min(64, 32) = 32
  // - results[1] = min(64, 128) = 64
  // - Returns {32, 64} where a-axis=32 (smaller), b-axis=64 (larger)
  std::vector<int64_t> results(count);
  for (size_t i = 0; i < count; ++i) {
    results[i] = std::min(left, upper_bounds[i]);
  }
  return {results[0], results[1]};
)";
}

// Multi-variable version of BinaryFindLowerBoundSatisfiedCoreNum with per-variable upper bounds
// This version allows different variables to have different upper bounds, enabling non-symmetric solutions
// When the binary search mid value exceeds a variable's upper bound, that variable stays at its upper bound
std::string GenBinaryFindLowerBoundSatisfiedCoreNum_Advanced() {
  std::string codes;
  codes += GenBinaryFindCoreNum_AdvancedSetup();
  codes += GenBinaryFindCoreNum_AdvancedSaveValues();
  codes += GenBinaryFindCoreNum_AdvancedCheckUpperBound();
  codes += GenBinaryFindCoreNum_AdvancedBinarySearch();
  codes += GenBinaryFindCoreNum_AdvancedRestoreValues();
  codes += GenBinaryFindCoreNum_AdvancedCalculateResult();

  return R"(
// Multi-variable version of BinaryFindLowerBoundSatisfiedCoreNum with per-variable upper bounds
// This version allows different variables to have different upper bounds, enabling non-symmetric solutions
// When the binary search mid value exceeds a variable's upper bound, that variable stays at its upper bound
std::pair<int64_t, int64_t> AxesReorderSolver::BinaryFindLowerBoundSatisfiedCoreNum(
    TilingVariable **vars, uint32_t *idxs, size_t count, int64_t lower_bound, const int64_t *upper_bounds) {
  if (count == 0) return {lower_bound, lower_bound};
)" + codes +
         R"(
}
)";
}

// ============================================================================
// Section 8.1: Math Utility Functions (for equal order tiling)
// ============================================================================

std::string GenGcd() {
  return R"(
// Custom gcd implementation for C++11 compatibility
template<typename T>
inline T gcd(T a, T b) {
  while (b != 0) {
    T temp = b;
    b = a % b;
    a = temp;
  }
  return a;
}
)";
}

std::string GenLcm() {
  return R"(
// Custom lcm implementation for C++11 compatibility
template<typename T>
inline T lcm(T a, T b) {
  if (a == 0 || b == 0) return 0;
  return (a / gcd(a, b)) * b;
}
)";
}

std::string GenAlignToUpperBound() {
  return R"(
// Helper function to align value to upper bound
inline int64_t AlignToUpperBound(int64_t value, int64_t align) {
  return CeilDiv(value, align) * align;
}
)";
}

std::string GenDualAxesInfo() {
  return R"(
// ProcessDualMCAxes helper functions
struct DualAxesInfo {
  TilingVariable *var_a{nullptr};
  TilingVariable *var_b{nullptr};
  uint32_t idx_a{0U};
  uint32_t idx_b{1U};
  int64_t upper_bound_a{0L};
  int64_t upper_bound_b{0L};
  int64_t low_bound_both{0L};
  int64_t upper_bound_both{0L};
};
)";
}

// ============================================================================
// Section 8.3: Three-Phase Algorithm Framework
// ============================================================================

std::string GenInitializeDualAxesInfo() {
  return R"(
// Initialize dual axes information
inline AxesReorderSolver::DualAxesInfo AxesReorderSolver::InitializeDualAxesInfo(
    TilingVariable **vars, uint32_t *axis_idx) const {
  int64_t upper_a = vars[0]->upper_bound(vars[0]->upper_bound_vars);
  int64_t upper_b = vars[1]->upper_bound(vars[1]->upper_bound_vars);
  bool a_is_first = (upper_a < upper_b);
  DualAxesInfo info{};
  info.var_a = a_is_first ? vars[0] : vars[1];
  info.var_b = a_is_first ? vars[1] : vars[0];
  info.idx_a = a_is_first ? axis_idx[0] : axis_idx[1];
  info.idx_b = a_is_first ? axis_idx[1] : axis_idx[0];
  info.upper_bound_a = info.var_a->upper_bound(info.var_a->upper_bound_vars);
  info.upper_bound_b = info.var_b->upper_bound(info.var_b->upper_bound_vars);
  info.upper_bound_both = std::min(info.upper_bound_a, info.upper_bound_b);
  info.low_bound_both = ::lcm(info.var_a->align, info.var_b->align);
  OP_LOGD(OP_NAME, "[DFX] Axes: a(ub=%ld, align=%ld), b(ub=%ld, align=%ld), low_bound_both=%ld",
          info.upper_bound_a, info.var_a->align, info.upper_bound_b, info.var_b->align, info.low_bound_both);
  return info;
}
)";
}

// ============================================================================
// Helper functions for GenExecuteStep1_FindUBLimit
// ============================================================================

std::string GenExecuteStep1_Initialize() {
  return R"(
  info.var_a->SetValue(info.upper_bound_a);
  info.var_b->SetValue(info.upper_bound_b);
  TilingVariable* both_vars[] = {info.var_a, info.var_b};
  uint32_t both_idxs[] = {info.idx_a, info.idx_b};
  int64_t low_bound_both = info.low_bound_both;
  bool step1_success = BinaryFindUpperBoundSatisfiedUBLimit(both_vars, both_idxs, 2, low_bound_both, low_bound_both_ub);
  if (!step1_success) {
    if (info.var_a->align == info.var_b->align) {
      OP_LOGW(OP_NAME, "[DFX] Step 1 failed, same align, return false");
      return false;
    }
    low_bound_both = std::min(info.var_a->align, info.var_b->align);
    step1_success = BinaryFindUpperBoundSatisfiedUBLimit(both_vars, both_idxs, 2, low_bound_both, low_bound_both_ub);
    if (!step1_success) {
      OP_LOGW(OP_NAME, "[DFX] Step 1 failed even with smaller lower_bound, return false");
      return false;
    }
  }
)";
}

std::string GenExecuteStep1_CalculateRanges() {
  return R"(
  OP_LOGD(OP_NAME, "[DFX] Step 1: low_bound_both_ub=%ld, upper_bound_both=%ld", low_bound_both_ub,
          info.upper_bound_both);
  low_bound_b_ub = info.upper_bound_b;
  if (low_bound_both_ub == info.upper_bound_both) {
    low_bound_b_ub = info.var_b->value;
    info.var_b->SetValue(info.upper_bound_b);
    TilingVariable* var_b_arr[] = {info.var_b};
    uint32_t idx_b_arr[] = {info.idx_b};
    BinaryFindUpperBoundSatisfiedUBLimit(var_b_arr, idx_b_arr, 1, info.upper_bound_a, low_bound_b_ub);
    OP_LOGD(OP_NAME, "[DFX] Step 1: UB sufficient, low_bound_b_ub=%ld", low_bound_b_ub);
  }
  range_a_high = low_bound_both_ub;
  range_b_high = (low_bound_both_ub == info.upper_bound_both) ? low_bound_b_ub : low_bound_both_ub;
  OP_LOGD(OP_NAME, "[DFX] Step 1 ranges: a_high=%ld, b_high=%ld", range_a_high, range_b_high);
)";
}

std::string GenExecuteStep1_Return() {
  return R"(
  return true;
)";
}

std::string GenExecuteStep1_FindUBLimit() {
  std::string codes;
  codes += "// Execute Step 1: Find UB limit\n";
  codes += "inline bool AxesReorderSolver::ExecuteStep1_FindUBLimit(const DualAxesInfo &info,\n";
  codes += "    int64_t &low_bound_both_ub, int64_t &low_bound_b_ub, int64_t &range_a_high, int64_t &range_b_high) {\n";
  codes += GenExecuteStep1_Initialize();
  codes += GenExecuteStep1_CalculateRanges();
  codes += GenExecuteStep1_Return();
  codes += "}\n";

  return codes;
}

// ============================================================================
// Helper functions for GenExecuteStep2_FindUBThreshold
// ============================================================================

std::string GenExecuteStep2_SetVariables() {
  return R"(
  info.var_a->SetValue(range_a_high);
  info.var_b->SetValue(range_a_high);
  TilingVariable* both_vars[] = {info.var_a, info.var_b};
  uint32_t both_idxs[] = {info.idx_a, info.idx_b};
  int64_t upper_bound_both_ub_thr = -1;
  bool step2_success =
      BinaryFindLowerBoundSatisfiedUBThresholdCond(both_vars, both_idxs, 2, range_a_low, upper_bound_both_ub_thr);
  if (!step2_success) {
    OP_LOGD(OP_NAME, "[DFX] Step 2 failed");
    return false;
  }
)";
}

std::string GenExecuteStep2_ReturnResult() {
  return R"(
  range_a_low_out = range_a_low;
  range_b_low_out = range_a_low;
  if (upper_bound_both_ub_thr > range_a_low) {
    range_a_low_out = range_b_low_out = upper_bound_both_ub_thr;
  }
  OP_LOGD(OP_NAME, "[DFX] Step 2: ub_thr=%ld, ranges: a_low=%ld, b_low=%ld",
          upper_bound_both_ub_thr, range_a_low_out, range_b_low_out);
  return true;
)";
}

std::string GenExecuteStep2_FindUBThreshold() {
  std::string codes;
  codes += "// Execute Step 2: Find UB threshold\n";
  codes += "inline bool AxesReorderSolver::ExecuteStep2_FindUBThreshold(const DualAxesInfo &info,\n";
  codes += "    int64_t range_a_low, int64_t range_a_high, int64_t &range_a_low_out, int64_t &range_b_low_out) {\n";
  codes += GenExecuteStep2_SetVariables();
  codes += GenExecuteStep2_ReturnResult();
  codes += "}\n";

  return codes;
}

// ============================================================================
// Helper functions for GenExecuteStep3_FindCoreNumTileSize
// ============================================================================

std::string GenExecuteStep3_SetupVariables() {
  return R"(
  info.var_a->SetValue(range_a_high);
  info.var_b->SetValue(range_a_high);
  TilingVariable* both_vars[] = {info.var_a, info.var_b};
  uint32_t both_idxs[] = {info.idx_a, info.idx_b};
  // Define separate upper bounds for each variable to enable non-symmetric solutions
  int64_t upper_bounds[2] = {range_a_high, range_b_high};
  OP_LOGD(OP_NAME, "[DFX] Step 3: Using per-variable upper bounds: a=%ld, b=%ld",
          upper_bounds[0], upper_bounds[1]);
)";
}

std::string GenExecuteStep3_BasicSearch() {
  return R"(
  auto lower_left = BinaryFindLowerBoundSatisfiedCoreNum(both_vars, both_idxs, 2, range_a_low, upper_bounds);
  auto lower_left_1 = lower_left.first;
  auto lower_left_2 = lower_left.second;
  OP_LOGD(OP_NAME, "[DFX] Step 3: lower_left_1=%ld, lower_left_2=%ld", lower_left_1, lower_left_2);
)";
}

std::string GenExecuteStep3_AdvancedSearch() {
  return R"(
  if (lower_left_1 < 0) {
    info.var_a->SetValue(range_a_high);
    TilingVariable *var_b_arr[] = {info.var_b};
    uint32_t idx_b_arr[] = {info.idx_b};
    auto lower_left = BinaryFindLowerBoundSatisfiedCoreNum(var_b_arr, idx_b_arr, 1, range_b_high);
    auto lower_left_b1 = lower_left.first;
    auto lower_left_b2 = lower_left.second;
    if (lower_left_b1 < 0) {
      OP_LOGW(OP_NAME, "[DFX] Step 3 failed for b axis, return false");
      return false;
    }
    info.var_b->SetValue((lower_left_b1 >= lower_left_b2) ? lower_left_b1 : lower_left_b2);
    OP_LOGD(OP_NAME, "[DFX] Step 3 fallback: a=%ld, b=%ld", range_a_high, info.var_b->value);
    vars_final[0] = info.var_a;
    vars_final[1] = info.var_b;
    return true;
  }
  info.var_a->SetValue(lower_left_1);
  info.var_b->SetValue(lower_left_2);
  OP_LOGD(OP_NAME, "[DFX] Step 3 success: a=%ld, b=%ld", lower_left_1, lower_left_2);
  vars_final[0] = info.var_a;
  vars_final[1] = info.var_b;
  return true;
)";
}

std::string GenExecuteStep3_FindCoreNumTileSize() {
  std::string codes;
  codes += "// Execute Step 3: Find tile size satisfying core number constraint\n";
  codes += "inline bool AxesReorderSolver::ExecuteStep3_FindCoreNumTileSize(const DualAxesInfo &info,\n";
  codes += "    int64_t range_a_low, int64_t range_a_high, int64_t range_b_high, TilingVariable **vars_final) {\n";
  codes += GenExecuteStep3_SetupVariables();
  codes += GenExecuteStep3_BasicSearch();
  codes += GenExecuteStep3_AdvancedSearch();
  codes += "}\n";

  return codes;
}

std::string GenProcessDualMCAxesOrchestration() {
  return R"(
// Orchestration function for dual multi-core axes processing
bool AxesReorderSolver::ProcessDualMCAxes(TilingVariable **vars, uint32_t *axis_idx, std::vector<bool> &solved_axes) {
  OP_LOGD(OP_NAME, "[DFX] ProcessDualMCAxes begin, input=%s", input_.DebugString().c_str());
  // Step 0: Initialize dual axes information
  DualAxesInfo info = InitializeDualAxesInfo(vars, axis_idx);
  // Step 1: Find UB limit
  int64_t low_bound_both_ub, low_bound_b_ub, range_a_high, range_b_high;
  if (!ExecuteStep1_FindUBLimit(info, low_bound_both_ub, low_bound_b_ub, range_a_high, range_b_high)) {
    return false;
  }
  // Step 2: Find UB threshold
  int64_t range_a_low = info.low_bound_both;
  int64_t range_b_low = info.low_bound_both;
  if (!ExecuteStep2_FindUBThreshold(info, range_a_low, range_a_high, range_a_low, range_b_low)) {
    info.var_a->SetValue(range_a_high);
    info.var_b->SetValue(range_a_high);
    return FinalizeEqualPriorityAxes(axis_idx, solved_axes);
  }
  // Step 3: Find tile size satisfying core number constraint
  TilingVariable* vars_final[2];
  if (!ExecuteStep3_FindCoreNumTileSize(info, range_a_low, range_a_high, range_b_high, vars_final)) {
    return false;
  }
  return FinalizeEqualPriorityAxes(axis_idx, solved_axes);
}
)";
}

// ============================================================================
// Section 8.4: NaiveTiling Helper Functions
// ============================================================================

// ============================================================================
// Helper functions for GenProcessMCAxisNaive
// ============================================================================

std::string GenProcessMCAxisNaive_FindUBThreshold() {
  return R"(
  int64_t upper_bound_satisfied_ub_threshold = var->value;
  int64_t lower_bound_satisfied_ub_threshold = var->align;
  // If no solution satisfies UB utilization, no further processing needed
  if (BinaryFindLowerBoundSatisfiedUBThresholdCond(var, var_idx, var->align, lower_bound_satisfied_ub_threshold)) {
    OP_LOGD(OP_NAME, "Found lower_bound_satisfied_ub_threshold:%ld, upper:%ld, lower:%ld, i:%u, input: %s",
            lower_bound_satisfied_ub_threshold, upper_bound_satisfied_ub_threshold, var->align, var_idx,
            input_.DebugString().c_str());
    var->SetValue(upper_bound_satisfied_ub_threshold);
    // 3.1) Binary search within [var->align, upper_bound_satisfied_ub_threshold] for UB utilization boundary
    auto satisfied_core_threshold = BinaryFindLowerBoundSatisfiedCoreNum(var, var_idx,
        lower_bound_satisfied_ub_threshold);
    auto satisfied_core_threshold_left = satisfied_core_threshold.first;
    auto satisfied_core_threshold_right = satisfied_core_threshold.second;
    OP_LOGD(OP_NAME, "Found lower bound satisfied core num:%ld, %ld, var upper:%ld, lower:%ld, i:%u, input: %s",
            satisfied_core_threshold_left, satisfied_core_threshold_right, upper_bound_satisfied_ub_threshold,
            var->align, var_idx, input_.DebugString().c_str());
)";
}

std::string GenProcessMCAxisNaive_TryLargerTile() {
  return R"(
    // 3.2) Try larger tile size first, update var if solution exists
    int64_t available_core_num_right = 0L;
    var->SetValue(satisfied_core_threshold_right);
    if (InitMulticoreVars() && MulticoreTilingCore(false) && CalRealUsedCoreNum(available_core_num_right)) {
      OP_LOGD(OP_NAME, "Found larger tile size:%ld, available_core_num:%ld, max_core_num:%ld, i:%u",
              satisfied_core_threshold_right, available_core_num_right, max_core_num, var_idx);
    }
)";
}

std::string GenProcessMCAxisNaive_SelectBestTile() {
  return R"(
    int64_t available_core_num_left = 0L;
    if ((satisfied_core_threshold_left != satisfied_core_threshold_right) && (satisfied_core_threshold_left > 0L)) {
      var->SetValue(satisfied_core_threshold_left);
      if (InitMulticoreVars() && MulticoreTilingCore(false) && CalRealUsedCoreNum(available_core_num_left) &&
          (available_core_num_left > available_core_num_right)) {
        OP_LOGD(OP_NAME, "Found smaller tile size:%ld, available_core_num:%ld, max_core_num:%ld, i:%u",
                satisfied_core_threshold_left, available_core_num_left, max_core_num, var_idx);
      } else {
        var->SetValue(satisfied_core_threshold_right);
      }
    }
  }
  return true;
)";
}

std::string GenProcessMCAxisNaive() {
  std::string codes;
  codes += "// Helper method to process multi-core related axis\n";
  codes +=
      "bool AxesReorderSolver::ProcessMCAxisNaive(TilingVariable *var, uint32_t var_idx, int64_t max_core_num) {\n";
  codes += GenProcessMCAxisNaive_FindUBThreshold();
  codes += GenProcessMCAxisNaive_TryLargerTile();
  codes += GenProcessMCAxisNaive_SelectBestTile();
  codes += "}\n";

  return codes;
}

// ============================================================================
// Helper functions for GenProcessSingleAxisNaive
// ============================================================================

std::string GenProcessSingleAxisNaive_Initialize() {
  return R"(
  auto upper_bound = var->upper_bound(var->upper_bound_vars);
  int64_t boundary = (upper_bound / var->align) * var->align;
  if (boundary < var->align) {
    OP_LOGW(OP_NAME, "Invalid aligned upper bound:%ld, raw upper:%ld, align:%ld, i:%u, input: %s.",
            boundary, upper_bound, var->align, var_idx, input_.DebugString().c_str());
    return false;
  }
  var->SetValue(boundary);

  int64_t upper_bound_satisfied_ub = -1L;
  if (!BinaryFindUpperBoundSatisfiedUBLimit(var, var->align, upper_bound_satisfied_ub)) {
    OP_LOGW(OP_NAME, "BinaryFindUpperBoundSatisfiedUBLimit failed, upper:%ld, lower:%ld, i:%u, input: %s.",
            upper_bound, var->align, var_idx, input_.DebugString().c_str());
    return false;
  }

  OP_LOGD(OP_NAME, "Found upper_bound_satisfied_ub:%ld, upper_bound:%ld, lower_bound:%ld, i:%u, input: %s",
          upper_bound_satisfied_ub, boundary, var->align, var_idx, input_.DebugString().c_str());
  var->SetValue(upper_bound_satisfied_ub);
)";
}

std::string GenProcessSingleAxisNaive_HandleMCAxis() {
  return R"(
  // Binary search for UB utilization values for multi-core related axis
  if (var->mc_related) {
    if (!ProcessMCAxisNaive(var, var_idx, max_core_num)) {
      return false;
    }
  }
)";
}

std::string GenProcessSingleAxisNaive_TuneAndSatisfy() {
  return R"(
  OP_LOGD(OP_NAME, "After local buffer tiling, input:%s", input_.DebugString().c_str());
  if (!TuneNotailVar(var)) {
    OP_LOGW(OP_NAME, "Tune notail var failed");
    return false;
  }

  while (!SatisfyCons(var, ConstraintType::LB_MIXED) && var->value != var->align) {
    var->value -= var->align;
    if (!TuneNotailVar(var)) {
      OP_LOGW(OP_NAME, "Tune notail var failed");
      return false;
    }
  }
)";
}

std::string GenProcessSingleAxisNaive_Finalize() {
  return R"(
  ApplyPromptAlign(var);
  return true;
)";
}

std::string GenProcessSingleAxisNaive() {
  std::string codes;
  codes += "// Helper method to process single axis\n";
  codes +=
      "bool AxesReorderSolver::ProcessSingleAxisNaive(TilingVariable *var, uint32_t var_idx, int64_t max_core_num) {\n";
  codes += GenProcessSingleAxisNaive_Initialize();
  codes += GenProcessSingleAxisNaive_HandleMCAxis();
  codes += GenProcessSingleAxisNaive_TuneAndSatisfy();
  codes += GenProcessSingleAxisNaive_Finalize();
  codes += "}\n";

  return codes;
}

}  // namespace att
