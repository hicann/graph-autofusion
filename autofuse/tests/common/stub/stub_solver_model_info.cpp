/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stub_solver_model_info.h"

namespace {
struct SolverExprContext {
  att::Expr expr_m;
  att::Expr expr_stepm;
  att::Expr expr_n;
  att::Expr expr_stepn;
  att::Expr expr_k;
};

void InitDefaultExpr(const ge::ExprType expr_type, att::Expr &default_expr, bool &is_const) {
  is_const = true;
  if (expr_type == ge::ExprType::kExprConstantRation) {
    default_expr = ge::Symbol(8, "tmp") / ge::Symbol(3, "tmp");
    return;
  }
  if (expr_type == ge::ExprType::kExprConstantInteger) {
    default_expr = ge::Symbol(16, "tmp");
    return;
  }
  if (expr_type == ge::ExprType::kExprVariable) {
    is_const = false;
  }
}

void InitSymVar(att::SymVarInfoPtr &sym, const att::Expr &expr, uint32_t align_val = 0,
                const std::vector<att::HardwareDef> &scope = {}) {
  sym = std::make_shared<att::SymVarInfo>(expr);
  if (align_val > 0) {
    sym->align = ge::Symbol(align_val);
  }
  if (!scope.empty()) {
    sym->related_scope = scope;
  }
}

void SetAxisOrigin(att::AttAxisPtr &axis, const std::string &name, const att::SymVarInfoPtr &size) {
  axis = std::make_shared<att::AttAxis>();
  axis->name = name;
  axis->axis_pos = att::AxisPosition::ORIGIN;
  axis->bind_multicore = false;
  axis->is_last = false;
  axis->is_node_innerest_dim = false;
  axis->size = size;
}

void SetAxisInner(att::AttAxisPtr &axis, const std::string &name, const att::SymVarInfoPtr &size, bool bind_multicore,
                  bool is_last, att::AttAxis *orig, att::AttAxis *from) {
  axis = std::make_shared<att::AttAxis>();
  axis->name = name;
  axis->axis_pos = att::AxisPosition::INNER;
  axis->bind_multicore = bind_multicore;
  axis->is_last = is_last;
  axis->is_node_innerest_dim = true;
  axis->size = size;
  axis->orig_axis.push_back(orig);
  axis->from_axis = {from};
}

void BuildMArgList(att::ModelInfo &model_info, const bool is_const, const att::Expr &default_expr,
                   const uint32_t m_align, SolverExprContext &ctx) {
  ctx.expr_m = is_const ? default_expr : att::CreateExpr("m_size");
  ctx.expr_stepm = is_const ? default_expr : att::CreateExpr("stepm_size");

  att::SymVarInfoPtr sym_m;
  att::SymVarInfoPtr sym_stepm;
  InitSymVar(sym_m, ctx.expr_m);
  sym_m->value_range.first = 1;
  sym_m->value_range.second = 10000;
  sym_m->align = ge::Symbol(m_align);
  InitSymVar(sym_stepm, ctx.expr_stepm, 16, {att::HardwareDef::L1, att::HardwareDef::CORENUM});

  att::AttAxisPtr m;
  att::AttAxisPtr stepm;
  SetAxisOrigin(m, "m", sym_m);
  SetAxisInner(stepm, "stepm", sym_stepm, true, true, m.get(), m.get());

  model_info.arg_list.emplace_back(m);
  model_info.arg_list.emplace_back(stepm);
}

void BuildNArgList(att::ModelInfo &model_info, const bool is_const, const att::Expr &default_expr,
                   SolverExprContext &ctx) {
  ctx.expr_n = is_const ? default_expr : att::CreateExpr("n_size");
  ctx.expr_stepn = is_const ? default_expr : att::CreateExpr("stepn_size");

  att::SymVarInfoPtr sym_n;
  att::SymVarInfoPtr sym_stepn;
  InitSymVar(sym_n, ctx.expr_n);
  InitSymVar(sym_stepn, ctx.expr_stepn, 128, {att::HardwareDef::L1, att::HardwareDef::CORENUM});

  att::AttAxisPtr n;
  att::AttAxisPtr stepn;
  SetAxisOrigin(n, "n", sym_n);
  SetAxisInner(stepn, "stepn", sym_stepn, true, true, n.get(), n.get());

  model_info.arg_list.emplace_back(n);
  model_info.arg_list.emplace_back(stepn);
}

void BuildKArg(att::ModelInfo &model_info, SolverExprContext &ctx) {
  ctx.expr_k = att::CreateExpr("k_size");
  att::SymConstInfoPtr sym_k = std::make_shared<att::SymConstInfo>(ctx.expr_k);
  sym_k->const_value = 128u;
  att::AttAxisPtr k = std::make_shared<att::AttAxis>();
  k->name = "k";
  k->axis_pos = att::AxisPosition::ORIGIN;
  k->bind_multicore = false;
  k->is_last = false;
  k->is_node_innerest_dim = false;
  k->size = sym_k;
  model_info.arg_list.emplace_back(k);
}

void FillModelInfo(att::ModelInfo &model_info, const SolverExprContext &ctx) {
  att::Expr l1_occupy =
      (ctx.expr_k * ctx.expr_stepm * att::CreateExpr(4)) + (ctx.expr_k * ctx.expr_stepn * att::CreateExpr(4));
  att::Expr core_num = ((ctx.expr_m / ctx.expr_stepm) * (ctx.expr_n / ctx.expr_stepn));

  model_info.hardware_cons[att::HardwareDef::L1] = l1_occupy;
  model_info.hardware_cons[att::HardwareDef::UB] = ctx.expr_m * att::CreateExpr(10);
  model_info.hardware_cons[att::HardwareDef::CORENUM] = core_num;

  att::Expr mte =
      (((ctx.expr_stepm * ctx.expr_k) / att::CreateExpr(32)) + ((ctx.expr_stepn * ctx.expr_k) / att::CreateExpr(32)));
  model_info.objects[att::PipeType::AIC_MTE2] = mte;
  model_info.tiling_case_id = 0;
  model_info.leq_exprs[att::kFatherToChildLarger].push_back((ctx.expr_stepm - ctx.expr_m));
  model_info.leq_exprs[att::kFatherToChildLarger].push_back((ctx.expr_stepn - ctx.expr_n));
  model_info.container_exprs["Q1"] = (ctx.expr_m + ctx.expr_n);
  model_info.tensor_exprs["MATMUL_OUTPUT1"] = (ctx.expr_m + ctx.expr_n);
  model_info.output_size = 1;
}
}  // namespace

namespace att {
ModelInfo CreateModelInfo(const uint32_t m_align, const ge::ExprType expr_type) {
  ModelInfo model_info;
  Expr default_expr;
  bool is_const = true;
  InitDefaultExpr(expr_type, default_expr, is_const);

  SolverExprContext ctx;
  BuildMArgList(model_info, is_const, default_expr, m_align, ctx);

  Optional test_optional;
  test_optional.optional_name = "test";
  test_optional.data_type = "int32_t";
  test_optional.min_value = "1";
  test_optional.max_value = "100";
  InputTensor input_tensor;
  input_tensor.data_type = 1;

  BuildNArgList(model_info, is_const, default_expr, ctx);
  BuildKArg(model_info, ctx);
  FillModelInfo(model_info, ctx);
  return model_info;
}
}  // namespace att

namespace att {
namespace {
// Static storage so schedule tables survive the test lifetime
std::vector<std::unique_ptr<TestTilingScheduleConfigTable>> &GetTestScheduleTables() {
  static std::vector<std::unique_ptr<TestTilingScheduleConfigTable>> tables;
  return tables;
}
}  // namespace

const TilingScheduleConfigTable *RegisterTestScheduleTable(uint32_t cache_line_size, bool enable_cache_line_check) {
  auto &tables = GetTestScheduleTables();
  auto table = std::make_unique<TestTilingScheduleConfigTable>();
  table->cache_line_size_ = cache_line_size;
  table->enable_cache_line_check_ = enable_cache_line_check;
  auto *ptr = table.get();
  tables.push_back(std::move(table));
  return ptr;
}

ModelInfo CreateGroupParallelCacheLineModelInfo(size_t group_id, uint32_t tiling_case_id, const Expr &cache_line_expr,
                                                uint32_t cache_line_size, CacheLineDirection direction,
                                                const ge::ExprType expr_type) {
  ModelInfo model_info = CreateModelInfo(1U, expr_type);
  model_info.schedule_group_ident.asc_graph_id = 0;
  model_info.schedule_group_ident.impl_graph_id = 0;
  model_info.schedule_group_ident.group_id = group_id;
  model_info.tiling_case_id = tiling_case_id;
  model_info.enable_group_parallel = true;
  model_info.tiling_schedule_config_table = RegisterTestScheduleTable(cache_line_size);
  CacheLineConfig cfg;
  cfg.node_name = "test_cache_line_node";
  cfg.cache_line_expr = cache_line_expr;
  cfg.solver_cache_line_expr = cache_line_expr;
  cfg.cache_line_size = cache_line_size;
  cfg.direction = direction;
  model_info.cache_line_config = {cfg};
  return model_info;
}
}  // namespace att
