/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "stub_model_info.h"
#include "reuse_group_utils/reuse_group_utils.h"

namespace {
struct MatmulExprContext {
  att::Expr expr_m;
  att::Expr expr_stepm;
  att::Expr expr_n;
  att::Expr expr_stepn;
  att::Expr expr_k;
};

struct L2TileExprContext {
  att::Expr expr_corenum;
  att::Expr expr_m;
  att::Expr expr_n;
  att::Expr expr_k;
  att::Expr expr_stepka;
  att::Expr expr_stepkb;
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

void BuildCreateModelInfoMArgs(att::ModelInfo &model_info, const bool is_const, const att::Expr &default_expr,
                               MatmulExprContext &ctx) {
  ctx.expr_m = is_const ? default_expr : att::CreateExpr("m_size");
  ctx.expr_stepm = is_const ? default_expr : att::CreateExpr("stepm_size");

  att::SymVarInfoPtr sym_m;
  att::SymVarInfoPtr sym_stepm;
  InitSymVar(sym_m, ctx.expr_m);
  sym_m->value_range.first = 1;
  sym_m->value_range.second = 10000;
  InitSymVar(sym_stepm, ctx.expr_stepm, 16, {att::HardwareDef::L1, att::HardwareDef::CORENUM});

  att::AttAxisPtr m;
  att::AttAxisPtr stepm;
  SetAxisOrigin(m, "m", sym_m);
  SetAxisInner(stepm, "stepm", sym_stepm, true, true, m.get(), m.get());

  model_info.arg_list.emplace_back(m);
  model_info.arg_list.emplace_back(stepm);
}

void BuildCreateModelInfoNArgs(att::ModelInfo &model_info, MatmulExprContext &ctx) {
  ctx.expr_n = att::CreateExpr("n_size");
  ctx.expr_stepn = att::CreateExpr("stepn_size");

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

void BuildCreateModelInfoKArg(att::ModelInfo &model_info, MatmulExprContext &ctx) {
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

void FillCreateModelInfo(att::ModelInfo &model_info, const MatmulExprContext &ctx) {
  att::Expr l1_occupy =
      (ctx.expr_k * ctx.expr_stepm * att::CreateExpr(4)) + (ctx.expr_k * ctx.expr_stepn * att::CreateExpr(4));
  att::Expr core_num = (ctx.expr_m / ctx.expr_stepm) * (ctx.expr_n / ctx.expr_stepn);

  model_info.hardware_cons[att::HardwareDef::L1] = l1_occupy;
  model_info.hardware_cons[att::HardwareDef::CORENUM] = core_num;
  model_info.hardware_cons[att::HardwareDef::UB] = att::CreateExpr(0L);

  att::Expr mte =
      (((ctx.expr_stepm * ctx.expr_k) / att::CreateExpr(32)) + ((ctx.expr_stepn * ctx.expr_k) / att::CreateExpr(32)));
  model_info.objects[att::PipeType::AIC_MTE2] = mte;
  model_info.tiling_case_id = 0;
  model_info.leq_exprs[att::kFatherToChildLarger].push_back((ctx.expr_stepm - ctx.expr_m));
  model_info.leq_exprs[att::kFatherToChildLarger].push_back((ctx.expr_stepn - ctx.expr_n));
  model_info.output_size = 1;
}

void BuildCoreAxis(att::ModelInfo &model_info, L2TileExprContext &ctx) {
  ctx.expr_corenum = att::CreateExpr("block_dim");
  att::SymVarInfoPtr sym_corenum = std::make_shared<att::SymVarInfo>(ctx.expr_corenum);
  att::AttAxisPtr core = std::make_shared<att::AttAxis>();
  core->name = "corenum";
  core->axis_pos = att::AxisPosition::ORIGIN;
  core->bind_multicore = false;
  core->is_last = false;
  core->is_node_innerest_dim = false;
  core->size = sym_corenum;
  model_info.arg_list.emplace_back(core);
}

void BuildL2TileMArgs(att::ModelInfo &model_info, L2TileExprContext &ctx) {
  ctx.expr_m = att::CreateExpr("m_size");

  att::SymVarInfoPtr sym_m = std::make_shared<att::SymVarInfo>(ctx.expr_m);

  att::AttAxisPtr m = std::make_shared<att::AttAxis>();

  m->name = "m";
  m->axis_pos = att::AxisPosition::ORIGIN;
  m->bind_multicore = false;
  m->is_last = false;
  m->is_node_innerest_dim = false;
  m->size = sym_m;

  model_info.arg_list.emplace_back(m);
}

void BuildL2TileNArgs(att::ModelInfo &model_info, L2TileExprContext &ctx) {
  ctx.expr_n = att::CreateExpr("n_size");

  att::SymVarInfoPtr sym_n = std::make_shared<att::SymVarInfo>(ctx.expr_n);

  att::AttAxisPtr n = std::make_shared<att::AttAxis>();

  n->name = "n";
  n->axis_pos = att::AxisPosition::ORIGIN;
  n->bind_multicore = false;
  n->is_last = false;
  n->is_node_innerest_dim = false;
  n->size = sym_n;

  model_info.arg_list.emplace_back(n);
}

void BuildL2TileKArgs(att::ModelInfo &model_info, L2TileExprContext &ctx) {
  ctx.expr_k = att::CreateExpr("k_size");
  ctx.expr_stepka = att::CreateExpr("stepka_size");
  ctx.expr_stepkb = att::CreateExpr("stepkb_size");

  att::SymVarInfoPtr sym_k;
  att::SymVarInfoPtr sym_stepka;
  att::SymVarInfoPtr sym_stepkb;
  InitSymVar(sym_k, ctx.expr_k);
  InitSymVar(sym_stepka, ctx.expr_stepka, 256, {att::HardwareDef::L1});
  InitSymVar(sym_stepkb, ctx.expr_stepkb, 16, {att::HardwareDef::L1});

  att::AttAxisPtr k;
  att::AttAxisPtr stepka;
  att::AttAxisPtr stepkb;
  SetAxisOrigin(k, "k", sym_k);
  SetAxisInner(stepka, "stepka", sym_stepka, false, false, k.get(), k.get());
  SetAxisInner(stepkb, "stepkb", sym_stepkb, false, true, k.get(), stepka.get());

  model_info.arg_list.emplace_back(k);
  model_info.arg_list.emplace_back(stepka);
  model_info.arg_list.emplace_back(stepkb);
}

void FillL2TileHardwareCons(att::ModelInfo &model_info, const L2TileExprContext &ctx) {
  att::Expr l1_occupy = ctx.expr_stepka * ctx.expr_stepkb * att::CreateExpr(4);
  model_info.hardware_cons[att::HardwareDef::L1] = l1_occupy;
  model_info.hardware_cons[att::HardwareDef::UB] = att::CreateExpr(0L);
}

void FillL2TileModelInfo(att::ModelInfo &model_info, const L2TileExprContext &ctx) {
  FillL2TileHardwareCons(model_info, ctx);
  model_info.tiling_case_id = 1;
  model_info.eq_exprs[att::kFatherToChildNoTail].push_back(std::pair(ctx.expr_stepka, ctx.expr_stepkb));
  model_info.leq_exprs[att::kFatherToChildLarger].push_back((ctx.expr_stepka - ctx.expr_k));
  model_info.container_exprs["Q1"] = (ctx.expr_m + ctx.expr_n);
  model_info.tensor_exprs["MATMUL_OUTPUT1"] = (ctx.expr_m + ctx.expr_n);
  model_info.output_size = 1;
}
}  // namespace

namespace att {
ModelInfo CreateModelInfo(const ge::ExprType expr_type) {
  ModelInfo model_info;
  Expr default_expr;
  bool is_const = true;
  InitDefaultExpr(expr_type, default_expr, is_const);

  MatmulExprContext ctx;
  BuildCreateModelInfoMArgs(model_info, is_const, default_expr, ctx);

  Optional test_optional;
  test_optional.optional_name = "test";
  test_optional.data_type = "int32_t";
  test_optional.min_value = "1";
  test_optional.max_value = "100";

  BuildCreateModelInfoNArgs(model_info, ctx);
  BuildCreateModelInfoKArg(model_info, ctx);
  FillCreateModelInfo(model_info, ctx);
  return model_info;
}

ModelInfo GetMatmulL2TileInfo() {
  ModelInfo model_info;
  L2TileExprContext ctx;
  BuildCoreAxis(model_info, ctx);
  BuildL2TileMArgs(model_info, ctx);
  BuildL2TileNArgs(model_info, ctx);
  BuildL2TileKArgs(model_info, ctx);
  FillL2TileModelInfo(model_info, ctx);
  return model_info;
}

void CreateCeilingAxisS1(ModelInfo &model_info, const Expr &expr_s1, AttAxisPtr &s1) {
  SymVarInfoPtr sym_s1 = std::make_shared<SymVarInfo>(expr_s1);
  s1 = std::make_shared<AttAxis>();
  s1->name = "s1";
  s1->axis_pos = AxisPosition::ORIGIN;
  s1->bind_multicore = false;
  s1->is_last = false;
  s1->is_node_innerest_dim = false;
  s1->size = sym_s1;
  model_info.arg_list.emplace_back(s1);
}

void CreateCeilingAxisS2(ModelInfo &model_info, const Expr &expr_s2, const Expr &expr_s2t, AttAxisPtr &s2,
                         AttAxisPtr &s2t) {
  SymVarInfoPtr sym_s2 = std::make_shared<SymVarInfo>(expr_s2);
  s2 = std::make_shared<AttAxis>();
  s2->name = "s2";
  s2->axis_pos = AxisPosition::ORIGIN;
  s2->bind_multicore = false;
  s2->is_last = false;
  s2->is_node_innerest_dim = false;
  s2->size = sym_s2;

  SymVarInfoPtr sym_s2t = std::make_shared<SymVarInfo>(expr_s2t);
  s2t = std::make_shared<AttAxis>();
  s2t->name = "s2t";
  s2t->axis_pos = AxisPosition::INNER;
  s2t->bind_multicore = false;
  s2t->is_last = false;
  s2t->is_node_innerest_dim = false;
  s2t->size = sym_s2t;
  s2t->orig_axis.push_back(s2.get());
  s2t->from_axis = {s2.get()};
  model_info.arg_list.emplace_back(s2);
  model_info.arg_list.emplace_back(s2t);
}

void CreateCeilingAxisBlock(ModelInfo &model_info, const AttAxisPtr &s1, const AttAxisPtr &s2, const Expr &expr_s1s2Tb,
                            AttAxisPtr &s1s2Tb) {
  SymVarInfoPtr sym_s1s2Tb = std::make_shared<SymVarInfo>(expr_s1s2Tb);
  s1s2Tb = std::make_shared<AttAxis>();
  s1s2Tb->name = "s1s2Tb";
  s1s2Tb->axis_pos = AxisPosition::INNER;
  s1s2Tb->bind_multicore = false;
  s1s2Tb->is_last = false;
  s1s2Tb->is_node_innerest_dim = true;
  s1s2Tb->size = sym_s1s2Tb;
  s1s2Tb->orig_axis.push_back(s1.get());
  s1s2Tb->orig_axis.push_back(s2.get());
  s1s2Tb->from_axis = {s1.get()};
  model_info.arg_list.emplace_back(s1s2Tb);
}

ModelInfo CreateCeilingModel() {
  ModelInfo model_info;
  Expr expr_s1 = CreateExpr("s1_size");
  Expr expr_s2 = CreateExpr("s2_size");
  Expr expr_s2t = CreateExpr("s2t_size");
  Expr expr_s2T = af::sym::Ceiling(expr_s2 / af::sym::Max(af::sym::kSymbolOne, expr_s2t));
  Expr expr_s1s2T = (expr_s1 * expr_s2T);
  Expr expr_s1s2Tb = CreateExpr("s1s2Tb_size");

  AttAxisPtr s1;
  CreateCeilingAxisS1(model_info, expr_s1, s1);
  AttAxisPtr s2;
  AttAxisPtr s2t;
  CreateCeilingAxisS2(model_info, expr_s2, expr_s2t, s2, s2t);
  AttAxisPtr s1s2Tb;
  CreateCeilingAxisBlock(model_info, s1, s2, expr_s1s2Tb, s1s2Tb);

  Expr core_num = af::sym::Ceiling((expr_s1 * af::sym::Ceiling(expr_s2 / af::sym::Max(af::sym::kSymbolOne, expr_s2t))) /
                                   af::sym::Max(af::sym::kSymbolOne, expr_s1s2Tb));
  model_info.hardware_cons[HardwareDef::UB] = expr_s1 * CreateExpr(10);
  model_info.hardware_cons[HardwareDef::CORENUM] = core_num;

  model_info.output_size = 1;
  model_info.tiling_case_id = 0;
  return model_info;
}
}  // namespace att
