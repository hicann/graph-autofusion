/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "autofuse_tiling_func_common.h"
namespace optiling {
inline int32_t CeilDiv(int32_t a, int32_t b) {
  int32_t res = a / b;
  return (res * b == a) ? res : (res + 1);
}
bool AxesReorderSolver::InitLocalBufferVars() {
  auto *vars = input_.local_buffer_vars;
  const auto size = input_.local_buffer_vars_size;
  for (uint32_t i = 0u; i < size; ++i) {
    const uint32_t remain = std::min(4u, size - i);
    for (uint32_t k = 0u; k < remain; ++k) {
      if (!vars[i + k]->SetValue(vars[i + k]->align)) {
        OP_LOGW(OP_NAME, "Failed to init local buffer value.");
        return false;
      }
    }
  }
  return true;
}

bool AxesReorderSolver::InitMulticoreVars() {
  uint32_t size = input_.pure_mc_vars_size;
  auto *vars = input_.pure_mc_vars;
  for (uint32_t i = 0u; i < size; i++) {
    auto &var = vars[i];
    auto upper_bound_val = var->upper_bound(var->upper_bound_vars);
    if (upper_bound_val == -1) {
      OP_LOGW(OP_NAME, "Failed to init multicore value.");
      return false;
    }
    upper_bound_val = CeilDiv(upper_bound_val, var->align) * var->align;
    if (!var->SetValue(upper_bound_val)) {
      OP_LOGW(OP_NAME, "Failed to init multicore value.");
      return false;
    }
  }
  return true;
}
bool AxesReorderSolver::GetMinMulticoreVars() {
  uint32_t size = input_.pure_mc_vars_size;
  auto *vars = input_.pure_mc_vars;
  for (uint32_t i = 0u; i < size; i++) {
    auto &var = vars[i];
    if (!var->SetValue(var->align)) {
      OP_LOGW(OP_NAME, "Failed to init multicore value.");
      return false;
    }
  }
  return true;
}
bool AxesReorderSolver::SatisfyCons(ConstraintType cons_type) {
  uint32_t size = input_.all_cons_size;
  auto *cons_list = input_.all_cons;
  for (uint32_t i = 0u; i < size; ++i) {
    const uint32_t remain = std::min(4u, size - i);
    for (uint32_t k = 0u; k < remain; ++k) {
      auto &cons = cons_list[i + k];
      if (cons->type != cons_type) {
        continue;
      }
      if (cons->eval(cons->rel_tiling_vars, cons->rel_in_shapes, cons->rel_hw_spec) > 0) {
        return false;
      }
    }
  }
  return true;
}
bool AxesReorderSolver::SatisfyCons(TilingVariable *var, ConstraintType cons_type) {
  uint32_t size = var->rel_cons_size;
  auto *cons_list = var->rel_cons;
  for (uint32_t i = 0u; i < size; ++i) {
    const uint32_t remain = std::min(4u, size - i);
    for (uint32_t k = 0u; k < remain; ++k) {
      auto &cons = cons_list[i + k];
      if (cons->type != cons_type) {
        continue;
      }
      if (cons->eval(cons->rel_tiling_vars, cons->rel_in_shapes, cons->rel_hw_spec) > 0) {
        return false;
      }
    }
  }
  return true;
}
bool AxesReorderSolver::SatisfyMCCons() {
  int32_t used_core_num = 0;
  CalRealUsedCoreNum(used_core_num);
  return used_core_num <= static_cast<int32_t>(input_.core_num);
}

bool AxesReorderSolver::TuneNotailVar(TilingVariable *var) {
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

bool AxesReorderSolver::MulticoreTiling(bool enable_workload_balance) {
  if (!InitMulticoreVars()) {
    OP_LOGW(OP_NAME, "multicore tiling failed");
    return false;
  }
  if (!SatisfyMCCons()) {
    OP_LOGW(OP_NAME, "Multicore Tiling Calculation failed in the first check, input: %s.",
            input_.DebugString().c_str());
    return false;
  }
  const int32_t num_vars = input_.pure_mc_vars_size;
  auto *vars = input_.pure_mc_vars;
  for (int32_t i = num_vars - 1; i >= 0; --i) {
    auto &var = vars[i];
    int32_t boundary = var->align;
    ;
    auto init_val = var->value;
    int32_t last_boundary = -1;
    int32_t last_val = -1;
    double pre_obj = GetPerf();
    while (!(last_boundary == boundary && last_val == var->value)) {
      last_boundary = boundary;
      last_val = var->value;
      var->value = CeilDiv((boundary + var->value) / 2, var->align);
      double cur_obj = GetPerf();
      var->value += 1;
      double next_obj = GetPerf();
      var->value -= 1;
      if (!SatisfyMCCons() || cur_obj > pre_obj || cur_obj > next_obj) {
        boundary = var->value;
        var->value = last_val;
      } else {
        pre_obj = cur_obj;
      }
    }
    if (!SatisfyMCCons()) {
      var->value = init_val;
    }
    while (!SatisfyCons(var, ConstraintType::MC_MIXED) && var->value != init_val) {
      var->value += var->align;
    }
  }
  if (!SatisfyCons(ConstraintType::MC_MIXED)) {
    OP_LOGW(OP_NAME, "Multicore Tiling Calculation failed in the final check, input: %s", input_.DebugString().c_str());
    return false;
  }
  if (enable_workload_balance) {
    int32_t cur_corenum = 0;
    CalRealUsedCoreNum(cur_corenum);
    if (cur_corenum != 2) {
      return true;
    }
    double cur_corenum_fp = 0.0;
    CalUsedCoreNum(cur_corenum_fp);
    TilingVariable optimal_mc_vars[input_.pure_mc_vars_size];

    for (uint32_t i = 0u; i < input_.pure_mc_vars_size; ++i) {
      optimal_mc_vars[i] = *vars[i];
    }
    double max_balance = std::fmod(cur_corenum_fp, 1.0);
    OP_LOGD(OP_NAME, "max_balance initialized: %f, current corenum is %d", max_balance, cur_corenum);
    if (fabs(max_balance) < 0.00000001f) {
      OP_LOGI(OP_NAME, "max_balance already satisfied");
      return true;
    }
    double balance = max_balance;
    for (int32_t i = num_vars - 1; i >= 0; i--) {
      auto &var = vars[i];
      int32_t corenum = 0;
      auto upper_bound_val = var->upper_bound(var->upper_bound_vars);
      upper_bound_val = CeilDiv(upper_bound_val, var->align) * var->align;
      while (SatisfyCons(ConstraintType::MC_MIXED) && (var->value < upper_bound_val)) {
        var->value += var->align;
        CalRealUsedCoreNum(corenum);
        if ((corenum < std::max(1, cur_corenum - 1))) {
          break;
        }
        double corenum_fp;
        CalUsedCoreNum(corenum_fp);
        balance = std::fmod(corenum_fp, 1.0);
        if ((fabs(balance) < 0.00000001f) || (balance > max_balance)) {
          max_balance = balance;

          for (uint32_t i = 0u; i < input_.pure_mc_vars_size; ++i) {
            optimal_mc_vars[i] = *vars[i];
          }
          OP_LOGD(OP_NAME, "max_balance updated: %f, corenum updated: %d", max_balance, corenum);
        }
      }
    }
    for (int32_t i = 0; i < num_vars; i++) {
      input_.pure_mc_vars[i]->value = optimal_mc_vars[i].value;
    }
  }
  return true;
}

void AxesReorderSolver::ApplyPromptAlign(TilingVariable *var) {
  auto aligned_val = var->value;
  while ((aligned_val >= var->prompt_align) && ((aligned_val) % var->prompt_align != 0)) {
    aligned_val -= var->align;
  }
  bool is_applied = (aligned_val != var->value) && (aligned_val > 0);
  if (is_applied) {
    if (var->upper_bound == nullptr) {
      OP_LOGI(OP_NAME, "Var upper bound func is not set.");
      return;
    }
    const auto upper_bound_val = var->upper_bound(var->upper_bound_vars);
    const auto loop_size = upper_bound_val / var->value;
    const auto tail_size = upper_bound_val % var->value;
    const auto tile_data_size = var->value * var->data_type_size;
    // if tile data size is less than 512B, no need to update prompt align
    if ((loop_size == 1) && (tail_size == 0) && (tile_data_size <= 512)) {
      OP_LOGI(OP_NAME, "No need to update prompt align, as loop size is 1 and tail size is 0, tile data size is %u",
              tile_data_size);
      return;
    }
    // 当block_len > 64B 对性能影响较大
    if ((var->value * var->data_type_size) <= 64) {
      OP_LOGI(OP_NAME, "No need to update prompt align, as block len is less than 64B");
      return;
    }
    OP_LOGI(OP_NAME, "Update prompt align from %u to %u", var->value, aligned_val);
    var->value = aligned_val;
  }
}
bool AxesReorderSolver::NaiveLocalBufTiling() {
  if (!InitLocalBufferVars()) {
    OP_LOGW(OP_NAME, "init local buffer failed");
    return false;
  }
  if (!InitMulticoreVars()) {
    OP_LOGW(OP_NAME, "multicore tiling failed");
    return false;
  }
  if (!SatisfyCons(ConstraintType::LOCAL_BUFFER)) {
    OP_LOGW(OP_NAME, "local buffer tiling failed in the initial check");
    return false;
  }

  uint32_t num_vars = input_.local_buffer_vars_size;
  auto *vars = input_.local_buffer_vars;
  for (uint32_t i = 0u; i < num_vars; ++i) {
    auto &var = vars[i];
    auto upper_bound = var->upper_bound(var->upper_bound_vars);
    int32_t boundary = CeilDiv(upper_bound, var->align) * var->align;
    for (int32_t val = boundary; val >= var->align; val -= var->align) {
      var->SetValue(val);
      if (SatisfyCons(var, ConstraintType::LOCAL_BUFFER)) {
        if (var->mc_related && SatisfyThresholdUBSize()) {
          if (!MulticoreTiling(true)) {
            OP_LOGW(OP_NAME, "Multicore Tiling Calculation failed in the first check, input: %s.",
                    input_.DebugString().c_str());
            return false;
          }
          int32_t max_corenum = 0;
          double threshold = input_.corenum_threshold;
          CalRealUsedCoreNum(max_corenum);
          if (max_corenum >= static_cast<int32_t>(threshold * static_cast<double>(input_.core_num))) {
            break;
          }
        } else {
          break;
        }
      }
    }
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
    ApplyPromptAlign(var);
  }
  if (!SatisfyCons(ConstraintType::LB_MIXED)) {
    OP_LOGW(OP_NAME, "Native local Tiling Calculation failed in the final check.");
    return false;
  }
  return true;
}
bool AxesReorderSolver::BinaryLocalBufTiling() {
  if (!InitLocalBufferVars()) {
    OP_LOGW(OP_NAME, "init local buffer failed");
    return false;
  }
  if (!InitMulticoreVars()) {
    OP_LOGW(OP_NAME, "multicore tiling failed");
    return false;
  }
  if (!SatisfyCons(ConstraintType::LOCAL_BUFFER)) {
    OP_LOGW(OP_NAME, "local buffer tiling failed in the initial check");
    return false;
  }
  uint32_t num_vars = input_.local_buffer_vars_size;
  auto *vars = input_.local_buffer_vars;
  for (uint32_t i = 0u; i < num_vars; ++i) {
    auto &var = vars[i];
    auto upper_bound = var->upper_bound(var->upper_bound_vars);
    int32_t boundary = CeilDiv(upper_bound, var->align) * var->align;
    int32_t init_val = var->value;
    var->SetValue(boundary);
    if (!SatisfyCons(var, ConstraintType::LOCAL_BUFFER)) {
      var->SetValue(init_val);
      int32_t last_boundary = -1;
      int32_t last_val = -1;
      while (!(last_boundary == boundary && last_val == var->value)) {
        last_boundary = boundary;
        last_val = var->value;
        var->value = CeilDiv((boundary + var->value) / 2, var->align) * var->align;
        if (!SatisfyCons(var, ConstraintType::LOCAL_BUFFER)) {
          boundary = var->value;
          var->value = last_val;
        }
      }
    }
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
    ApplyPromptAlign(var);
  }
  if (!SatisfyCons(ConstraintType::LB_MIXED)) {
    OP_LOGW(OP_NAME, "Binary local Tiling Calculation failed in the final check.");
    return false;
  }
  return true;
}
bool AxesReorderSolver::LocalBufTiling(const bool is_tuning) {
  if (is_tuning) {
    return NaiveLocalBufTiling();
  } else {
    return BinaryLocalBufTiling();
  }
}
bool AxesReorderSolver::PgoSolverGenerateAllTilingData() {
  uint32_t index = 0;
  std::vector<uint32_t> ans_item(input_.tiling_vars_size);
  std::vector<std::vector<uint32_t>> ans;
  OP_LOGI(OP_NAME, "Start PgoSolverGenerate AllTilingData");
  int32_t step_max = 16;
  auto step_value = pgo_step_max_;
  if ((step_value % 2) == 0) {
    step_max = step_value;
  }
  PgoSolverGenerateAllTilingDataInner(index, ans_item, ans, step_max);
  availiable_tiling_data_list_ = ans;
  OP_LOGI(OP_NAME, "Gen PgoSolverGenerate AllTilingData %d.", ans.size());
  return true;
}

void AxesReorderSolver::PgoSolverGenerateAllTilingDataInner(const uint32_t index, std::vector<uint32_t> &ans_item,
                                                            std::vector<std::vector<uint32_t>> &ans, int32_t step_max) {
  if (index >= input_.tiling_vars_size) {
    if (!SatisfyMCCons()) {
      return;
    }
    ans.push_back(ans_item);
    return;
  }
  TilingVariable *tilingDataVar;
  bool from_local_buffer_vars = true;
  if (index >= input_.local_buffer_vars_size) {
    tilingDataVar = input_.pure_mc_vars[index - input_.local_buffer_vars_size];
    from_local_buffer_vars = false;
  } else {
    tilingDataVar = input_.local_buffer_vars[index];
    from_local_buffer_vars = true;
  }
  auto min_ = tilingDataVar->align;
  auto max_ =
      CeilDiv(tilingDataVar->upper_bound(tilingDataVar->upper_bound_vars), tilingDataVar->align) * tilingDataVar->align;
  auto step = tilingDataVar->align;
  tilingDataVar->value = 0;
  auto var_value = tilingDataVar->value;
  OP_LOGI(OP_NAME, "PgoSolverGenerateAllTilingDataInner %zu.", ans.size());
  while (var_value < max_) {
    if (from_local_buffer_vars) {
      if (step <= step_max && tilingDataVar->value < step_max) {
        tilingDataVar->value = step;
      } else {
        tilingDataVar->value += step;
      }
      var_value = tilingDataVar->value;
      tilingDataVar->value = std::min(max_, tilingDataVar->value);
      step = std::min(step_max, step * 2);
    } else {
      tilingDataVar->value += step;
      var_value = tilingDataVar->value;
      if (tilingDataVar->value > max_) {
        continue;
      }
      step = 1;
    }
    // init 后面的变量
    auto tmp = index + 1;
    while (tmp < input_.local_buffer_vars_size + input_.pure_mc_vars_size) {
      if (tmp >= input_.local_buffer_vars_size) {
        auto tilingDataVarTmp = input_.pure_mc_vars[tmp - input_.local_buffer_vars_size];
        tilingDataVarTmp->value = tilingDataVarTmp->align;
      } else {
        auto tilingDataVarTmp = input_.local_buffer_vars[tmp];
        tilingDataVarTmp->value = tilingDataVarTmp->align;
      }
      tmp += 1;
    }

    if (!from_local_buffer_vars) {
      if (!SatisfyCons(tilingDataVar, ConstraintType::MC_MIXED)) {
        continue;
      }
    } else {
      if (!SatisfyCons(tilingDataVar, ConstraintType::LB_MIXED) ||
          !SatisfyCons(tilingDataVar, ConstraintType::LOCAL_BUFFER)) {
        continue;
      }
    }
    ans_item[index] = tilingDataVar->value;
    PgoSolverGenerateAllTilingDataInner(index + 1, ans_item, ans, step_max);
  }
}

inline bool AxesReorderSolver::GetTiling(const bool is_tuning, const bool enable_workload_balance) {
  if (!LocalBufTiling(is_tuning)) {
    OP_LOGW(OP_NAME, "local buffer tiling failed, input: %s", input_.DebugString().c_str());
    return false;
  }
  OP_LOGI(OP_NAME, "local buffer tiling success, input: %s", input_.DebugString().c_str());
  if (!MulticoreTiling(enable_workload_balance)) {
    OP_LOGW(OP_NAME, "multicore tiling failed, input: %s", input_.DebugString().c_str());
    return false;
  }
  OP_LOGI(OP_NAME, "multicore tiling success, input: %s", input_.DebugString().c_str());
  return true;
}

inline bool AxesReorderSolver::GetMaxBlockDimTiling(const uint32_t block_dim) {
  auto save_block_dim = input_.core_num;
  input_.core_num = block_dim;
  input_.ub_threshold = 0.0;
  input_.corenum_threshold = 1;
  // 按照约束规则求解不需要开启负载均衡
  auto ret = GetTiling(true, false);
  input_.core_num = save_block_dim;
  return ret;
}

inline bool AxesReorderSolver::FindNextUpperBlockDim(const uint32_t block_dim, uint32_t &next_lower_block_dim) const {
  // 当前核数已达到或超过上限，无更大档位
  if (block_dim >= input_.core_num) {
    return false;
  }
  // 确定当前区间的间隔
  uint32_t step;
  if (block_dim <= 4U) {
    step = 1U;
  } else if (block_dim < 16U) {
    step = 2U;
  } else {
    step = 4U;
  }
  // 计算理论上下一个档位
  uint32_t candidate = block_dim + step;
  // 若候选值超过上限，则直接取上限作为下一档位
  if (candidate > input_.core_num) {
    next_lower_block_dim = input_.core_num;
  } else {
    next_lower_block_dim = candidate;
  }
  return true;
}

inline bool AxesReorderSolver::FindNextLowerBlockDim(const uint32_t block_dim, uint32_t &next_upper_block_dim) const {
  // 核数必须大于1才可能有上一个档位（至少为1）
  if (block_dim <= 1U) {
    return false;
  }
  // 确定当前区间的间隔（与上限判断逻辑一致）
  uint32_t step;
  if (block_dim <= 4U) {
    step = 1;
  } else if (block_dim < 16U) {
    step = 2U;
  } else {
    step = 4U;
  }
  // 计算上一个档位（确保结果至少为1）
  uint32_t candidate = block_dim - step;
  next_upper_block_dim = std::max(candidate, 1u);  // 避免核数为0
  return true;
}

inline void AxesReorderSolver::SaveInputTilingVars(TilingVariable *tiling_vars, TilingVariable *pure_mc_vars,
                                                   TilingVariable *local_buffer_vars) const {
  for (uint32_t i = 0U; i < input_.tiling_vars_size; i++) {
    tiling_vars[i] = *input_.tiling_vars[i];
  }
  for (uint32_t i = 0U; i < input_.pure_mc_vars_size; i++) {
    pure_mc_vars[i] = *input_.pure_mc_vars[i];
  }
  for (uint32_t i = 0U; i < input_.local_buffer_vars_size; i++) {
    local_buffer_vars[i] = *input_.local_buffer_vars[i];
  }
}

inline void AxesReorderSolver::RestoreInputTilingVars(const TilingVariable *tiling_vars,
                                                      const TilingVariable *pure_mc_vars,
                                                      const TilingVariable *local_buffer_vars) const {
  for (uint32_t i = 0U; i < input_.tiling_vars_size; i++) {
    *input_.tiling_vars[i] = tiling_vars[i];
  }
  for (uint32_t i = 0U; i < input_.pure_mc_vars_size; i++) {
    *input_.pure_mc_vars[i] = pure_mc_vars[i];
  }
  for (uint32_t i = 0U; i < input_.local_buffer_vars_size; i++) {
    *input_.local_buffer_vars[i] = local_buffer_vars[i];
  }
}

inline void AxesReorderSolver::FindBetterSolutionByLowerBlockDim(double next_lower_perf,
                                                                 uint32_t next_lower_block_dim) {
  double current_perf;
  uint32_t current_block_dim;
  TilingVariable tiling_vars[input_.tiling_vars_size];
  TilingVariable pure_mc_vars[input_.pure_mc_vars_size];
  TilingVariable local_buffer_vars[input_.local_buffer_vars_size];
  do {
    // 更新当前的性能和block_dim
    current_perf = next_lower_perf;
    current_block_dim = next_lower_block_dim;
    if (!FindNextLowerBlockDim(current_block_dim, next_lower_block_dim)) {
      OP_LOGD(OP_NAME,
              "Found better solution by lower block dim, no lower block dim, current_perf: %f, "
              "current_block_dim: %u, input:%s",
              current_perf, current_block_dim, input_.DebugString().c_str());
      // 无更低档位，当前input_就是最优解，直接返回下档位的最优解
      OP_LOGD(OP_NAME, "current_perf: %f, next_lower_perf: %f, current_dim %u, next_lower_block_dim %u, input:%s",
              current_perf, next_lower_perf, current_block_dim, next_lower_block_dim, input_.DebugString().c_str());
      return;
    }
    // 由于GetMaxBlockDimTiling会修改input_，所以这里备份
    SaveInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
    // 根据不限制ub占用率，占满核数的策略更新tiling_data，并返回性能
    if (GetMaxBlockDimTiling(next_lower_block_dim)) {
      next_lower_perf = GetPerf();
    } else {
      // 继续求解失败，则跳出使用上一次成功求解的值
      break;
    }
  } while ((next_lower_perf - current_perf) < 0.0);  // 若当前性能差于下档位性能，继续下档位
  OP_LOGD(OP_NAME,
          "Found better solution by lower block dim, current_perf: %f, next_lower_perf: %f, "
          "use current_dim %u as best block dim[next_lower_block_dim %u]",
          current_perf, next_lower_perf, current_block_dim, next_lower_block_dim);
  // 下一档位性能差于当前档位，恢复当前档位为最优解返回(维测点：打印上档位最优解的perf,核数)
  RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
}

inline void AxesReorderSolver::FindBetterSolutionByUpperBlockDim(double next_upper_perf,
                                                                 uint32_t next_upper_block_dim) {
  double current_perf;
  uint32_t current_block_dim;
  // 备份input_
  TilingVariable tiling_vars[input_.tiling_vars_size];
  TilingVariable pure_mc_vars[input_.pure_mc_vars_size];
  TilingVariable local_buffer_vars[input_.local_buffer_vars_size];
  do {
    // 更新当前的性能和block_dim
    current_perf = next_upper_perf;
    current_block_dim = next_upper_block_dim;
    if (!FindNextUpperBlockDim(current_block_dim, next_upper_block_dim)) {
      OP_LOGD(OP_NAME,
              "Found better solution by upper block dim, no upper block dim, current_perf: %f, "
              "current_block_dim: %u, input:%s",
              current_perf, current_block_dim, input_.DebugString().c_str());
      // 无更高档位，当前input_就是最优解，直接返回上档位的最优解
      return;
    }
    // 由于GetMaxBlockDimTiling会修改input_，所以这里备份
    SaveInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
    // 根据不限制ub占用率，占满核数的策略更新tiling_data，并返回性能
    if (GetMaxBlockDimTiling(next_upper_block_dim)) {
      next_upper_perf = GetPerf();
    } else {
      // 继续求解失败，则跳出使用上一次成功求解的值
      break;
    }
    OP_LOGD(OP_NAME, "current_perf: %f, next_upper_perf: %f, current_dim %u, next_upper_block_dim %u, input:%s",
            current_perf, next_upper_perf, current_block_dim, next_upper_block_dim, input_.DebugString().c_str());
  } while ((next_upper_perf - current_perf) < 0.0);  // 若当前性能差于下档位性能，继续下档位
  // 返回上档位的最优解(维测点：打印上档位最优解的perf,核数)
  OP_LOGD(OP_NAME,
          "Found better solution by upper block dim, current_perf: %f, next_upper_perf: %f, "
          "use current_dim %u as best block dim[next_upper_block_dim %u], input:%s",
          current_perf, next_upper_perf, current_block_dim, next_upper_block_dim, input_.DebugString().c_str());
  // 上档位性能差于当前档位，恢复当前档位为最优解返回(维测点：打印上档位最优解的perf,核数)
  RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
}

bool AxesReorderSolver::AutoTuning(const bool is_trade_off) {
  OP_LOGI(OP_NAME, "Start auto tuning, input:%s", input_.DebugString().c_str());

  if (is_trade_off) {
    OP_LOGD(OP_NAME, "Do not auto tuning, as is_trade_off is %d.", is_trade_off);
    return true;
  }
  if (input_.corenum_threshold > 1.0f || input_.ub_threshold < 0.0f) {
    OP_LOGD(OP_NAME, "Do not auto tuning, as corenum_threshold is invalid:%f.", input_.corenum_threshold);
    return true;
  }
  auto current_perf = GetPerf();
  if (current_perf < input_.perf_threshold) {
    OP_LOGD(OP_NAME, "Do not auto tuning, as current_perf:%lf is less than threshold:%lf, input:%s.", current_perf,
            input_.perf_threshold, input_.DebugString().c_str());
    return true;
  }

  // 使能自动融合
  double got_block_dim = 0.0;
  (void)CalUsedCoreNum(got_block_dim);
  const auto block_dim = static_cast<uint32_t>(Ceiling(got_block_dim));
  uint32_t next_upper_block_dim = block_dim;
  uint32_t next_lower_block_dim = block_dim;
  bool got_upper_block_dim = FindNextUpperBlockDim(block_dim, next_upper_block_dim);
  bool got_lower_block_dim = FindNextLowerBlockDim(block_dim, next_lower_block_dim);
  // 1.上档位找不到，则向下档位寻找
  if (!got_upper_block_dim) {
    OP_LOGD(OP_NAME,
            "Ready to find better solution by lower block dim, no upper block dim, current_perf: %f, "
            "current_block_dim: %u, next_lower_block_dim: %u, input:%s",
            current_perf, block_dim, next_lower_block_dim, input_.DebugString().c_str());
    FindBetterSolutionByLowerBlockDim(current_perf, block_dim);
    return true;
  }
  // 2.下档位找不到，则向上档位寻找
  if (!got_lower_block_dim) {
    OP_LOGD(OP_NAME,
            "Ready to find better solution by upper block dim, no lower block dim, current_perf: %f, "
            "current_block_dim: %u, next_upper_block_dim: %u, input:%s",
            current_perf, block_dim, next_upper_block_dim, input_.DebugString().c_str());
    FindBetterSolutionByUpperBlockDim(current_perf, block_dim);
    return true;
  }
  TilingVariable tiling_vars[input_.tiling_vars_size];
  TilingVariable pure_mc_vars[input_.pure_mc_vars_size];
  TilingVariable local_buffer_vars[input_.local_buffer_vars_size];
  // 3.找下1个档位，比较当前档位与下档位性能
  SaveInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
  if (!GetMaxBlockDimTiling(next_lower_block_dim)) {
    RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
    return true;
  }

  double next_lower_perf = GetPerf();
  // 4.当前档位差于下档位，向下找更优解(考虑多核头开销对小Shape场景的影响和同地址冲突对多核的影响，当前更倾向于下档位)
  if (current_perf > next_lower_perf) {
    OP_LOGD(OP_NAME,
            "Find lower block dim, as next_lower_perf: %f(block_dim=%u) is better than"
            "current_perf: %f(block_dim=%u), input: %s",
            current_perf, block_dim, next_lower_perf, next_lower_block_dim, input_.DebugString().c_str());
    FindBetterSolutionByLowerBlockDim(next_lower_perf, next_lower_block_dim);
    return true;
  }
  // 因为档位档位性能好于下档位，所以还原当前档位的性能
  RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
  SaveInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
  if (!GetMaxBlockDimTiling(next_upper_block_dim)) {
    RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
    return true;
  }
  double next_upper_perf = GetPerf();
  // 5.当前档位差于上档位，向上找更优解
  if (current_perf > next_upper_perf) {
    OP_LOGD(OP_NAME,
            "Find upper block dim, as next_upper_perf: %f(block_dim=%u) is better than current_perf: %f(block_dim=%u).",
            current_perf, block_dim, next_upper_perf, next_upper_block_dim);
    FindBetterSolutionByUpperBlockDim(next_upper_perf, next_upper_block_dim);
    return true;
  }
  // 6.当前档位好于上档位也好于下档位，当前档位为最优解
  RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
  OP_LOGD(
      OP_NAME,
      "Current block dim %u(perf:%f) is best, next upper block dim: %u(perf:%f), next lower block dim: %u(perf:%f).",
      block_dim, current_perf, next_upper_block_dim, next_upper_perf, next_lower_block_dim, next_lower_perf);
  return true;
}

inline bool AxesReorderSolver::WorkloadBalance() {
  OP_LOGD(OP_NAME, "Begin to calculate core num for workload balance, input:%s.", input_.DebugString().c_str());
  int32_t used_corenum = 0;
  CalRealUsedCoreNum(used_corenum);
  if (used_corenum == 1) {
    OP_LOGI(OP_NAME, "used_corenum is 1, start to tune corenum_threshold");
    input_.ub_threshold = 0.0;
    input_.corenum_threshold = 0.05;
    if (!GetTiling(true, true)) {
      OP_LOGW(OP_NAME, "Get tiling failed.");
      return false;
    }
  }
  return true;
}

bool AxesReorderSolver::Run(const bool is_trade_off, const bool enable_auto_tune) {
  // 初始解默认会占满UB，核数使用较少
  if (!GetTiling(is_trade_off, false)) {
    OP_LOGW(OP_NAME, "Get default tiling failed");
    return false;
  }

  if (!enable_auto_tune) {
    // hi-perf level设置为0，不支持自动调优，直接返回默认值
    OP_LOGD(OP_NAME, "Do not need auto tuning, enable_auto_tune: %d, input: %s.", enable_auto_tune,
            input_.DebugString().c_str());
    return true;
  }
  // hi-perf level设置为1，支持自动调优
  auto save_core_num = input_.core_num;
  if (!AutoTuning(is_trade_off)) {
    OP_LOGD(OP_NAME, "Do not need auto tuning, is_trade_off: %d", is_trade_off);
    return false;
  }
  // 负载均衡的逻辑，当前仅考虑单核场景，后续优化
  input_.core_num = save_core_num;
  if (!WorkloadBalance()) {
    return false;
  }
  return true;
}

}  // namespace optiling
