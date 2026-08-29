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
// Section 7: PGO and Auto-tuning
// ============================================================================

std::string GenPgoSolverGenerateAllTilingDataHead() {
  std::string codes = R"(bool AxesReorderPgoSolver::PgoSolverGenerateAllTilingData() {
  uint32_t index = 0;
  std::vector<uint32_t> ans_item(input_.tiling_vars_size);
  std::vector<std::vector<uint32_t>> ans;
  OP_LOGI(OP_NAME, "Start PgoSolverGenerate AllTilingData");
  int64_t step_max = 16;
  auto step_value = pgo_step_max_;
  if ((step_value % 2) == 0) {
    step_max = step_value;
  }
  PgoSolverGenerateAllTilingDataInner(index, ans_item, ans, step_max);
  availiable_tiling_data_list_ = ans;
  OP_LOGI(OP_NAME, "Gen PgoSolverGenerate AllTilingData %ld.", ans.size());
  return true;
}
)";
  return codes;
}

std::string GenPgoSolverGenerateAllTilingDataBody() {
  std::string codes = R"(void AxesReorderPgoSolver::PgoSolverGenerateAllTilingDataInner(const uint32_t index,
    std::vector<uint32_t> &ans_item, std::vector<std::vector<uint32_t>> &ans, int64_t step_max) {
  if (index >= input_.tiling_vars_size) {
    if (!SatisfyMCCons()) {
      return;
    }
    ans.push_back(ans_item);
    return;
  }
  TilingVariable* tilingDataVar;
  bool from_local_buffer_vars = true;
  if (index >= input_.local_buffer_vars_size) {
    tilingDataVar = input_.pure_mc_vars[index - input_.local_buffer_vars_size];
    from_local_buffer_vars = false;
  } else {
    tilingDataVar = input_.ordered_local_buffer_vars[index];
    from_local_buffer_vars = true;
  }
  auto min_ = tilingDataVar->align;
  auto max_ = CeilDiv(tilingDataVar->upper_bound(tilingDataVar->upper_bound_vars), tilingDataVar->align) * tilingDataVar->align;
  auto step = tilingDataVar->align;
  tilingDataVar->value = 0;
  auto var_value = tilingDataVar->value;
  OP_LOGI(OP_NAME, "PgoSolverGenerateAllTilingDataInner %zu.", ans.size());
)";
  return codes;
}

std::string GenPgoSolverGenerateAllTilingDataTail() {
  std::string codes = R"(  while (var_value < max_) {
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
        auto tilingDataVarTmp = input_.ordered_local_buffer_vars[tmp];
        tilingDataVarTmp->value = tilingDataVarTmp->align;
      }
      tmp += 1;
    }

    if (!from_local_buffer_vars) {
      if (!SatisfyCons(tilingDataVar, ConstraintType::MC_MIXED)) {
        continue;
      }
    } else {
      if (!SatisfyCons(tilingDataVar, ConstraintType::LB_MIXED) || !SatisfyCons(tilingDataVar, ConstraintType::LOCAL_BUFFER)) {
        continue;
      }
    }
    if (from_local_buffer_vars) {
      ans_item[input_.ordered_to_canonical[index]] = tilingDataVar->value;
    } else {
      ans_item[index] = tilingDataVar->value;
    }
    PgoSolverGenerateAllTilingDataInner(index + 1, ans_item, ans, step_max);
  }
}
)";
  return codes;
}

std::string GenPgoSolverGenerateAllTilingData() {
  std::string TilingDataHead = GenPgoSolverGenerateAllTilingDataHead();
  std::string TilingDataBody = GenPgoSolverGenerateAllTilingDataBody();
  std::string TilingDataTail = GenPgoSolverGenerateAllTilingDataTail();
  return TilingDataHead + TilingDataBody + TilingDataTail;
}

std::string GenGetTiling(bool enable_equal_order_tiling) {
  std::string codes;
  codes.append(R"(
inline bool AxesReorderSolver::GetTiling(const bool is_tuning,
                                         const bool block_loop_auto_tune,
                                         const bool enable_workload_balance)");
  if (enable_equal_order_tiling) {
    codes.append(R"(,
                                         const bool enable_equal_order) {
  if (!LocalBufTiling(is_tuning, block_loop_auto_tune, enable_equal_order)) {
    OP_LOGW(OP_NAME, "local buffer tiling failed, is_tuning: %d, block_loop_auto_tune: %d, enable_equal_order: %d,"
    "input: %s", is_tuning, block_loop_auto_tune, enable_equal_order, input_.DebugString().c_str());
    return false;
  }
)");
  } else {
    codes.append(R"() {
  if (!LocalBufTiling(is_tuning, block_loop_auto_tune)) {
    OP_LOGW(OP_NAME, "local buffer tiling failed, is_tuning: %d, block_loop_auto_tune: %d,"
    "input: %s", is_tuning, block_loop_auto_tune, input_.DebugString().c_str());
    return false;
  }
)");
  }
  // 根据enable_equal_order_tiling条件生成不同的LocalBufTiling调用
  codes.append(R"(
  OP_LOGI(OP_NAME, "local buffer tiling success, input: %s", input_.DebugString().c_str());
  if (!MulticoreTiling(block_loop_auto_tune, enable_workload_balance)) {
    OP_LOGW(OP_NAME, "multicore tiling failed, input: %s", input_.DebugString().c_str());
    return false;
  }
  OP_LOGI(OP_NAME, "multicore tiling success, input: %s", input_.DebugString().c_str());
  return true;
  }
)");
  return codes;
}

std::string GenGetMaxBlockDimTiling(bool enable_equal_order_tiling) {
  std::string codes;

  if (enable_equal_order_tiling) {
    codes += R"(
inline bool AxesReorderSolver::GetMaxBlockDimTiling(const uint32_t block_dim, const bool enable_equal_order) {
  auto save_block_dim = input_.core_num;
  input_.core_num = block_dim;
  input_.ub_threshold = 0.0;
  input_.corenum_threshold = 1;
  // 按照约束规则求解不需要开启负载均衡
  auto ret = GetTiling(true, true, false, enable_equal_order);
  input_.core_num = save_block_dim;
  return ret;
}
)";
  } else {
    codes += R"(
inline bool AxesReorderSolver::GetMaxBlockDimTiling(const uint32_t block_dim) {
  auto save_block_dim = input_.core_num;
  input_.core_num = block_dim;
  input_.ub_threshold = 0.0;
  input_.corenum_threshold = 1;
  // 按照约束规则求解不需要开启负载均衡
  auto ret = GetTiling(true, true, false);
  input_.core_num = save_block_dim;
  return ret;
}
)";
  }
  return codes;
}

std::string GenFindNextUpperBlockDim() {
  return R"(
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
)";
}

std::string GenFindNextLowerBlockDim() {
  return R"(
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
  next_upper_block_dim = std::max(candidate, 1u); // 避免核数为0
  return true;
}
)";
}

std::string GenSaveInputTilingVars() {
  return R"(
inline void AxesReorderSolver::SaveInputTilingVars(TilingVariable *tiling_vars,
                                                   TilingVariable *pure_mc_vars,
                                                   TilingVariable *local_buffer_vars) const {
  if (tiling_vars != nullptr) {
    for (uint32_t i = 0U; i < input_.tiling_vars_size; i++) {
      tiling_vars[i] = *(input_.tiling_vars[i]);
    }
  }
  if (pure_mc_vars != nullptr) {
    for (uint32_t i = 0U; i < input_.pure_mc_vars_size; i++) {
      pure_mc_vars[i] = *(input_.pure_mc_vars[i]);
    }
  }
  if (local_buffer_vars != nullptr) {
    for (uint32_t i = 0U; i < input_.local_buffer_vars_size; i++) {
      local_buffer_vars[i] = *(input_.local_buffer_vars[i]);
    }
  }
}
)";
}

std::string GenRestoreInputTilingVars() {
  return R"(
inline void AxesReorderSolver::RestoreInputTilingVars(const TilingVariable *tiling_vars,
                                                      const TilingVariable *pure_mc_vars,
                                                      const TilingVariable *local_buffer_vars) const {
  if (tiling_vars != nullptr) {
    for (uint32_t i = 0U; i < input_.tiling_vars_size; i++) {
      *(input_.tiling_vars[i]) = tiling_vars[i];
    }
  }
  if (pure_mc_vars != nullptr) {
    for (uint32_t i = 0U; i < input_.pure_mc_vars_size; i++) {
      *(input_.pure_mc_vars[i]) = pure_mc_vars[i];
    }
  }
  if (local_buffer_vars != nullptr) {
    for (uint32_t i = 0U; i < input_.local_buffer_vars_size; i++) {
      *(input_.local_buffer_vars[i]) = local_buffer_vars[i];
    }
  }
}
)";
}

// Helper function: Generate function signature for FindBetterSolutionByLowerBlockDim
std::string GenFindBetterSolutionSignature() {
  return R"(
inline void AxesReorderSolver::FindBetterSolutionByLowerBlockDim(double next_lower_perf,
                                                                 uint32_t next_lower_block_dim
)";
}

// Helper function: Generate function signature parameter for equal order
std::string GenFindBetterSolutionSignatureParam(bool enable_equal_order_tiling) {
  if (enable_equal_order_tiling) {
    return ",\n                                                                 const bool enable_equal_order";
  }
  return "";
}

// Helper function: Generate variable declarations
std::string GenFindBetterSolutionVariables() {
  return R"() {
double current_perf;
uint32_t current_block_dim;
TilingVariable tiling_vars[input_.tiling_vars_size];
TilingVariable pure_mc_vars[input_.pure_mc_vars_size];
TilingVariable local_buffer_vars[input_.local_buffer_vars_size];
)";
}

// Helper function: Generate do-while loop code
std::string GenFindBetterSolutionLoop() {
  return R"(
do {
  // 更新当前的性能和block_dim
  current_perf = next_lower_perf;
  current_block_dim = next_lower_block_dim;
  if (!FindNextLowerBlockDim(current_block_dim, next_lower_block_dim)) {
  OP_LOGD(OP_NAME,
          "Found better solution by lower block dim, no lower block dim remains, current_perf: %f, "
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
)";
}

// Helper function: Generate conditional code for equal_order parameter
std::string GenFindBetterSolutionConditionals(bool enable_equal_order_tiling) {
  std::string codes = R"(  if (GetMaxBlockDimTiling(next_lower_block_dim)";
  if (enable_equal_order_tiling) {
    codes += ", enable_equal_order";
  }
  codes += R"()) {
      next_lower_perf = GetPerf();
    } else {
      // 继续求解失败，则跳出使用上一次成功求解的值
      break;
    }
  } while ((next_lower_perf - current_perf) < 0.0);  // 若当前性能差于下档位性能，继续下档位
)";
  return codes;
}

// Helper function: Generate final logging and restoration code
std::string GenFindBetterSolutionCleanup() {
  return R"(
  OP_LOGD(OP_NAME,
          "Found better solution by lower block dim, current_perf: %f, next_lower_perf: %f, "
          "use current_dim %u as best block dim[next_lower_block_dim %u]",
          current_perf, next_lower_perf, current_block_dim, next_lower_block_dim);
  // 下一档位性能差于当前档位，恢复当前档位为最优解返回(维测点：打印上档位最优解的perf,核数)
  RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
}
)";
}

std::string GenFindBetterSolutionByLowerBlockDim(bool enable_equal_order_tiling) {
  std::string codes;
  codes += GenFindBetterSolutionSignature();
  codes += GenFindBetterSolutionSignatureParam(enable_equal_order_tiling);
  codes += GenFindBetterSolutionVariables();
  codes += GenFindBetterSolutionLoop();
  codes += GenFindBetterSolutionConditionals(enable_equal_order_tiling);
  codes += GenFindBetterSolutionCleanup();
  return codes;
}

static std::string GenFindBetterSolutionByUpperBlockDimBodyPre() {
  return R"(
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
              "Found better solution by upper block dim, no upper block dim remains, current_perf: %f, "
              "current_block_dim: %u, input:%s",
              current_perf, current_block_dim, input_.DebugString().c_str());
      // 无更高档位，当前input_就是最优解，直接返回上档位的最优解
      return;
    }
    // 由于GetMaxBlockDimTiling会修改input_，所以这里备份
    SaveInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
    // 根据不限制ub占用率，占满核数的策略更新tiling_data，并返回性能
    if (GetMaxBlockDimTiling(next_upper_block_dim)";
}

// Helper function to generate common body for FindBetterSolutionByUpperBlockDim
static std::string GenFindBetterSolutionByUpperBlockDimBody(bool enable_equal_order_tiling) {
  std::string codes = GenFindBetterSolutionByUpperBlockDimBodyPre();
  if (enable_equal_order_tiling) {
    codes += ", enable_equal_order";
  }
  codes += R"()) {
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
)";
  return codes;
}

std::string GenFindBetterSolutionByUpperBlockDim(bool enable_equal_order_tiling) {
  std::string codes;
  if (enable_equal_order_tiling) {
    codes += R"(
inline void AxesReorderSolver::FindBetterSolutionByUpperBlockDim(double next_upper_perf,
                                                                 uint32_t next_upper_block_dim,
                                                                 const bool enable_equal_order) {
)";
  } else {
    codes += R"(
inline void AxesReorderSolver::FindBetterSolutionByUpperBlockDim(double next_upper_perf,
                                                                 uint32_t next_upper_block_dim) {
)";
  }
  codes += GenFindBetterSolutionByUpperBlockDimBody(enable_equal_order_tiling);
  return codes;
}

std::string GenAutoTuningInputCheck() {
  return R"(
  if (is_trade_off) {
    OP_LOGD(OP_NAME,"Do not auto tuning, as is_trade_off is %d.", is_trade_off);
    return true;
  }
  if (input_.corenum_threshold > 1.0f || input_.ub_threshold < 0.0f) {
    OP_LOGD(OP_NAME,"Do not auto tuning, as corenum_threshold is invalid:%f.", input_.corenum_threshold);
    return true;
  }
  auto current_perf = GetPerf();
  if (current_perf < input_.perf_threshold) {
    OP_LOGD(OP_NAME, "Do not auto tuning, as current_perf:%lf is less than threshold:%lf, input:%s.", current_perf,
            input_.perf_threshold, input_.DebugString().c_str());
    return true;
  }
)";
}

// ============================================================================
// Helper functions for GenAutoTuningFindUpperLowerBlockDimCommon
// ============================================================================

static std::string GenAutoTuningUpperLower_CalculateBlockDim() {
  return R"(
  // 使能自动融合
  double got_block_dim = 0.0;
  (void)CalUsedCoreNum(got_block_dim);
  const auto block_dim = static_cast<uint32_t>(Ceiling(got_block_dim));
  uint32_t next_upper_block_dim = block_dim;
  uint32_t next_lower_block_dim = block_dim;
  bool got_upper_block_dim = FindNextUpperBlockDim(block_dim, next_upper_block_dim);
  bool got_lower_block_dim = FindNextLowerBlockDim(block_dim, next_lower_block_dim);
)";
}

static std::string GenAutoTuningUpperLower_CheckNoUpper(bool enable_equal_order_tiling) {
  std::string codes = R"(
  // 1.上档位找不到，则向下档位寻找
  if (!got_upper_block_dim) {
    OP_LOGD(OP_NAME,
            "Ready to find better solution by lower block dim, no upper block dim, current_perf: %f, "
            "current_block_dim: %u, next_lower_block_dim: %u, input:%s",
            current_perf, block_dim, next_lower_block_dim, input_.DebugString().c_str());
)";
  if (enable_equal_order_tiling) {
    codes += "    FindBetterSolutionByLowerBlockDim(current_perf, block_dim, enable_equal_order);\n";
  } else {
    codes += "    FindBetterSolutionByLowerBlockDim(current_perf, block_dim);\n";
  }
  codes += R"(
    return true;
  }
)";
  return codes;
}

static std::string GenAutoTuningUpperLower_CheckNoLower(bool enable_equal_order_tiling) {
  std::string codes = R"(
  // 2.下档位找不到，则向上档位寻找
  if (!got_lower_block_dim) {
    OP_LOGD(OP_NAME,
            "Ready to find better solution by upper block dim, no lower block dim, current_perf: %f, "
            "current_block_dim: %u, next_upper_block_dim: %u, input:%s",
            current_perf, block_dim, next_upper_block_dim, input_.DebugString().c_str());
)";
  if (enable_equal_order_tiling) {
    codes += "    FindBetterSolutionByUpperBlockDim(current_perf, block_dim, enable_equal_order);\n";
  } else {
    codes += "    FindBetterSolutionByUpperBlockDim(current_perf, block_dim);\n";
  }
  codes += R"(
    return true;
  }
)";
  return codes;
}

static std::string GenAutoTuningUpperLower_PrepareLowerSearch(bool enable_equal_order_tiling) {
  std::string codes = R"(
  TilingVariable tiling_vars[input_.tiling_vars_size];
  TilingVariable pure_mc_vars[input_.pure_mc_vars_size];
  TilingVariable local_buffer_vars[input_.local_buffer_vars_size];
  // 3.找下1个档位，比较当前档位与下档位性能
  SaveInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
)";
  if (enable_equal_order_tiling) {
    codes += "  if (!GetMaxBlockDimTiling(next_lower_block_dim, enable_equal_order)) {\n";
  } else {
    codes += "  if (!GetMaxBlockDimTiling(next_lower_block_dim)) {\n";
  }
  codes += R"(
    RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
    return true;
  }
)";
  return codes;
}

// Helper function to generate common auto tuning find upper/lower block dim code
static std::string GenAutoTuningFindUpperLowerBlockDimCommon(bool enable_equal_order_tiling) {
  std::string codes;
  codes += GenAutoTuningUpperLower_CalculateBlockDim();
  codes += GenAutoTuningUpperLower_CheckNoUpper(enable_equal_order_tiling);
  codes += GenAutoTuningUpperLower_CheckNoLower(enable_equal_order_tiling);
  codes += GenAutoTuningUpperLower_PrepareLowerSearch(enable_equal_order_tiling);
  return codes;
}

std::string GenAutoTuningFindUpperLowerBlockDim() {
  return GenAutoTuningFindUpperLowerBlockDimCommon(true);
}

std::string GenAutoTuningFindUpperLowerBlockDimNoEqual() {
  return GenAutoTuningFindUpperLowerBlockDimCommon(false);
}

// ============================================================================
// Helper functions for GenAutoTuningFindBetteSolutionCommon
// ============================================================================

static std::string GenAutoTuningBetterSolution_CheckLower(bool enable_equal_order_tiling) {
  std::string codes = R"(
  double next_lower_perf = GetPerf();
  // 4.当前档位差于下档位，向下找更优解(考虑多核头开销对小Shape场景的影响和同地址冲突对多核的影响，当前更倾向于下档位)
  if (current_perf > next_lower_perf) {
    OP_LOGD(OP_NAME, "Find lower block dim, as next_lower_perf: %f(block_dim=%u) is better than "
            "current_perf: %f(block_dim=%u), input: %s", next_lower_perf, next_lower_block_dim, current_perf,
            block_dim, input_.DebugString().c_str());
)";
  if (enable_equal_order_tiling) {
    codes += "    FindBetterSolutionByLowerBlockDim(next_lower_perf, next_lower_block_dim, enable_equal_order);\n";
  } else {
    codes += "    FindBetterSolutionByLowerBlockDim(next_lower_perf, next_lower_block_dim);\n";
  }
  codes += R"(
    return true;
  }
)";
  return codes;
}

static std::string GenAutoTuningBetterSolution_CheckUpper(bool enable_equal_order_tiling) {
  std::string codes = R"(
  // 因为档位档位性能好于下档位，所以还原当前档位的性能
  RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
  SaveInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
)";
  if (enable_equal_order_tiling) {
    codes += "  if (!GetMaxBlockDimTiling(next_upper_block_dim, enable_equal_order)) {\n";
  } else {
    codes += "  if (!GetMaxBlockDimTiling(next_upper_block_dim)) {\n";
  }
  codes += R"(
    RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
    return true;
  }
  double next_upper_perf = GetPerf();
  // 5.当前档位差于上档位，向上找更优解
  if (current_perf > next_upper_perf) {
    OP_LOGD(OP_NAME,
        "Find upper block dim, as next_upper_perf: %f(block_dim=%u) is better than current_perf: %f(block_dim=%u).",
        next_upper_perf, next_upper_block_dim, current_perf, block_dim);
)";
  if (enable_equal_order_tiling) {
    codes += "    FindBetterSolutionByUpperBlockDim(next_upper_perf, next_upper_block_dim, enable_equal_order);\n";
  } else {
    codes += "    FindBetterSolutionByUpperBlockDim(next_upper_perf, next_upper_block_dim);\n";
  }
  codes += R"(
    return true;
  }
)";
  return codes;
}

static std::string GenAutoTuningBetterSolution_Finalize() {
  return R"(
  // 6.当前档位好于上档位也好于下档位，当前档位为最优解
  RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
  OP_LOGD(
      OP_NAME,
      "Current block dim %u(perf:%f) is best, next upper block dim: %u(perf:%f), next lower block dim: %u(perf:%f).",
      block_dim, current_perf, next_upper_block_dim, next_upper_perf, next_lower_block_dim, next_lower_perf);
  return true;
)";
}

// Helper function to generate common auto tuning find better solution code
static std::string GenAutoTuningFindBetteSolutionCommon(bool enable_equal_order_tiling) {
  std::string codes;
  codes += GenAutoTuningBetterSolution_CheckLower(enable_equal_order_tiling);
  codes += GenAutoTuningBetterSolution_CheckUpper(enable_equal_order_tiling);
  codes += GenAutoTuningBetterSolution_Finalize();
  return codes;
}

std::string GenAutoTuningFindBetteSolutionEqual() {
  return GenAutoTuningFindBetteSolutionCommon(true);
}

std::string GenAutoTuningFindBetteSolutionNoEqual() {
  return GenAutoTuningFindBetteSolutionCommon(false);
}

std::string GenAutoTuningFindBetteSolution(const bool enable_equal_order_tiling) {
  std::string codes;
  if (enable_equal_order_tiling) {
    codes.append(GenAutoTuningFindBetteSolutionEqual());
  } else {
    codes.append(GenAutoTuningFindBetteSolutionNoEqual());
  }
  return codes;
}

std::string GenAutoTuning(bool enable_equal_order_tiling) {
  std::string codes;
  if (enable_equal_order_tiling) {
    codes += R"(
bool AxesReorderSolver::AutoTuning(const bool is_trade_off, const bool enable_equal_order) {
  OP_LOGI(OP_NAME, "Start auto tuning, is_trade_off:%d, enable_equal_order:%d, input:%s", is_trade_off, enable_equal_order,
    input_.DebugString().c_str());
)";
  } else {
    codes += R"(
bool AxesReorderSolver::AutoTuning(const bool is_trade_off) {
  OP_LOGI(OP_NAME, "Start auto tuning, is_trade_off:%d, input:%s", is_trade_off, input_.DebugString().c_str());
)";
  }
  codes += GenAutoTuningInputCheck();
  if (enable_equal_order_tiling) {
    codes += GenAutoTuningFindUpperLowerBlockDim();
  } else {
    codes += GenAutoTuningFindUpperLowerBlockDimNoEqual();
  }
  codes += GenAutoTuningFindBetteSolution(enable_equal_order_tiling);
  codes += "}\n";
  return codes;
}

}  // namespace att
