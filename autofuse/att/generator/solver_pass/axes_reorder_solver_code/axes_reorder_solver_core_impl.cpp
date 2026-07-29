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
// Section 3: Core Implementation
// ============================================================================

std::string GenInitAllVars() {
  std::string codes;
  codes += "bool AxesReorderSolver::InitLocalBufferVars() {\n";
  codes += "  auto *vars = input_.local_buffer_vars;\n";
  codes += "  const auto size = input_.local_buffer_vars_size;\n";
  codes += "  for (uint32_t i = 0u; i < size; ++i) {\n";
  codes += "    const uint32_t remain = std::min(4u, size - i);\n";
  codes += "    for (uint32_t k =0u; k < remain; ++k) {\n";
  codes += "      if (!vars[i+k]->SetValue(vars[i+k]->align)) {\n";
  codes += "        OP_LOGW(OP_NAME, \"Failed to init local buffer value.\");\n";
  codes += "        return false;\n";
  codes += "      }\n";
  codes += "    }\n";
  codes += "  }\n";
  codes += "  return true;\n";
  codes += "}\n\n";
  codes += "bool AxesReorderSolver::InitMulticoreVars() {\n";
  codes += "  uint32_t size = input_.pure_mc_vars_size;\n";
  codes += "  auto *vars = input_.pure_mc_vars;\n";
  codes += "  for (uint32_t i = 0u; i < size; i++) {\n";
  codes += "    auto &var = vars[i];\n";
  codes += "    auto upper_bound_val = var->upper_bound(var->upper_bound_vars);\n";
  codes += "    if (upper_bound_val == -1) {\n";
  codes += "      OP_LOGW(OP_NAME, \"Failed to init multicore value.\");\n";
  codes += "      return false;\n";
  codes += "    }\n";
  codes += "    upper_bound_val = CeilDiv(upper_bound_val, var->align) * var->align;\n";
  codes += "    if (!var->SetValue(upper_bound_val)) {\n";
  codes += "      OP_LOGW(OP_NAME, \"Failed to init multicore value.\");\n";
  codes += "      return false;\n";
  codes += "    }\n";
  codes += "  }\n";
  codes += "  return true;\n";
  codes += "}\n";
  return codes;
}

std::string GenCommonPartOfSatisfyCons() {
  return R"(    if (cons->type != cons_type) {
      continue;
    }
    if (cons->eval(cons->rel_tiling_vars, cons->rel_in_shapes, static_cast<double>(cons->rel_hw_spec)) > 0) {
      OP_LOGD(OP_NAME, "Check eval failed, rel_hw_spec:%ld, input:%s", cons->rel_hw_spec, input_.DebugString().c_str());
      return false;
    }
)";
}

std::string GenSatisfyConsByRatio() {
  return R"(    if (cons->type != cons_type) {
      continue;
    }
    if (cons->eval(cons->rel_tiling_vars, cons->rel_in_shapes,
        Ceiling(ratio * static_cast<double>(cons->rel_hw_spec))) > 0) {
      OP_LOGD(OP_NAME, "Check eval failed, rel_hw_spec:%ld, input:%s, ratio:%lf", cons->rel_hw_spec,
              input_.DebugString().c_str(), ratio);
      return false;
    }
)";
}

std::string GenSatisfyCons() {
  std::string codes;
  codes += "bool AxesReorderSolver::SatisfyCons(ConstraintType cons_type) {\n";
  codes += "  uint32_t size = input_.all_cons_size;\n";
  codes += "  auto *cons_list = input_.all_cons;\n";
  codes += "  for (uint32_t i = 0u; i < size; ++i) {\n";
  codes += "    const uint32_t remain = std::min(4u, size - i);\n";
  codes += "    for (uint32_t k =0u; k < remain; ++k) {\n";
  codes += "      auto &cons = cons_list[i+k];\n";
  codes += GenCommonPartOfSatisfyCons();
  codes += "    }\n";
  codes += "  }\n";
  codes += "  return true;\n";
  codes += "}\n";
  codes += "bool AxesReorderSolver::SatisfyCons(TilingVariable *var, ConstraintType cons_type, double ratio) {\n";
  codes += "  uint32_t size = var->rel_cons_size;\n";
  codes += "  auto *cons_list = var->rel_cons;\n";
  codes += "  for (uint32_t i = 0u; i < size; ++i) {\n";
  codes += "    const uint32_t remain = std::min(4u, size - i);\n";
  codes += "    for (uint32_t k =0u; k < remain; ++k) {\n";
  codes += "      auto &cons = cons_list[i+k];\n";
  codes += GenSatisfyConsByRatio();
  codes.append(R"(    }
  }
  return true;
}
bool AxesReorderSolver::SatisfyMCCons() {
  int64_t used_core_num = 0;
  CalRealUsedCoreNum(used_core_num);
  return used_core_num <= static_cast<int64_t>(input_.core_num);
}
)");
  return codes;
}

std::string GenTuneNoTailVar() {
  std::string codes = R"(
bool AxesReorderSolver::TuneNotailVar(TilingVariable* var) {
  if (!var->notail) {
    return true;
  }
  if (var->notail_var->value % var->value == 0) {
    return true;
  }
  for (; var->value > 0; var->value -= var->align) {
    if (var->notail_var->value % var->value != 0) {
      continue;
    }
    break;
  }
  return var->value != 0;
}
)";
  return codes;
}

std::string GenCopyVars() {
  return R"(
  for (uint32_t i = 0u; i < input_.pure_mc_vars_size; ++i) {
    optimal_mc_vars[i] = *vars[i];
  }
)";
}

std::string GenWorkLoadBalance() {
  std::string codes;
  codes += "  if (enable_workload_balance) {\n";
  codes += "  int64_t cur_corenum = 0;\n";
  codes += "  CalRealUsedCoreNum(cur_corenum);\n";
  codes += "  if (cur_corenum != 2) {\n";
  codes += "    return true;\n";
  codes += "  }\n";
  codes += "  const int32_t num_vars = input_.pure_mc_vars_size;\n";
  codes += "  auto *vars = input_.pure_mc_vars;\n";
  codes += "  double cur_corenum_fp = 0.0;\n";
  codes += "  CalUsedCoreNum(cur_corenum_fp);\n";
  codes += "  TilingVariable optimal_mc_vars[input_.pure_mc_vars_size];\n";
  codes += GenCopyVars();
  codes += "  double max_balance = std::fmod(cur_corenum_fp, 1.0);\n";
  codes += "  OP_LOGD(OP_NAME, \"max_balance initialized: %f, current corenum is %ld\", max_balance, cur_corenum);\n";
  codes += "  if (fabs(max_balance) < 0.00000001f) {\n";
  codes += "    OP_LOGI(OP_NAME, \"max_balance already satisfied\");\n";
  codes += "    return true;\n";
  codes += "  }\n";
  codes += "  double balance = max_balance;\n";
  codes += "  for (int32_t i=num_vars-1; i >= 0; i--) {\n";
  codes += "    auto &var = vars[i];\n";
  codes += "    int64_t corenum = 0;\n";
  codes += "    auto upper_bound_val = var->upper_bound(var->upper_bound_vars);\n";
  codes += "    upper_bound_val = CeilDiv(upper_bound_val, var->align) * var->align;\n";
  codes += "    while (SatisfyCons(ConstraintType::MC_MIXED) && (var->value < upper_bound_val)) {\n";
  codes += "      var->value += var->align;\n";
  codes += "      CalRealUsedCoreNum(corenum);\n";
  codes += "      if ((corenum < std::max(1L, cur_corenum - 1))) {\n";
  codes += "        break;\n";
  codes += "      }\n";
  codes += "      double corenum_fp;\n";
  codes += "      CalUsedCoreNum(corenum_fp);\n";
  codes += "      balance = std::fmod(corenum_fp, 1.0);\n";
  codes += "      if ((fabs(balance) < 0.00000001f) || (balance > max_balance)) {\n";
  codes += "        max_balance = balance;\n";
  codes += GenCopyVars();
  codes += "        OP_LOGD(OP_NAME, \"max_balance updated: %f, corenum updated: %ld\", max_balance, corenum);\n";
  codes += "      }\n";
  codes += "    }\n";
  codes += "  }\n";
  codes += "    for (int32_t i=0; i < num_vars; i++) {\n";
  codes += "      input_.pure_mc_vars[i]->value = optimal_mc_vars[i].value;\n";
  codes += "    }\n";
  codes += "  }\n";
  return codes;
}

std::string GenApplyPromptAlign() {
  std::string codes = R"(
  void AxesReorderSolver::ApplyPromptAlign(TilingVariable *var) {
    const auto original_val = var->value;
    auto aligned_val = original_val;
    bool found_prompt_aligned = false;
    while (aligned_val > 0) {
      if ((aligned_val % var->prompt_align == 0) && (aligned_val % var->align == 0)) {
        found_prompt_aligned = true;
        break;
      }
      aligned_val -= var->align;
    }
    bool is_applied = found_prompt_aligned && (aligned_val != original_val) && (aligned_val > 0);
    if (is_applied) {
      if (var->upper_bound == nullptr) {
        OP_LOGI(OP_NAME, "Var upper bound func is not set.");
        return;
      }
      const auto upper_bound_val = var->upper_bound(var->upper_bound_vars);
      const auto loop_size = upper_bound_val / original_val;
      const auto tail_size = upper_bound_val % original_val;
      const auto tile_data_size = original_val * var->data_type_size;
      // if tile data size is less than 512B, no need to update prompt align
      if ((loop_size == 1) && (tail_size == 0) && (tile_data_size <= 512)) {
        OP_LOGI(OP_NAME, "No need to update prompt align, as loop size is 1 and tail size is 0, tile data size is %ld",
                tile_data_size);
        return;
      }
      // 当block_len > 64B 对性能影响较大
      if ((original_val * var->data_type_size) <= 64) {
        OP_LOGI(OP_NAME, "No need to update prompt align, as block len is less than 64B");
        return;
      }
      OP_LOGI(OP_NAME, "Update prompt align from %ld to %ld", original_val, aligned_val);
      var->value = aligned_val;
    }
  }
)";
  return codes;
}

}  // namespace att
