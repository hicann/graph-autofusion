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
#include "stub_matmul_modelinfo.h"

namespace {
struct MatmulExprContext {
  att::Expr expr_corenum;
  att::Expr expr_m;
  att::Expr expr_n;
  att::Expr expr_k;
  att::Expr expr_stepka;
  att::Expr expr_stepkb;
};

void BuildCoreAxis(att::ModelInfo &model_info, MatmulExprContext &ctx) {
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

void BuildMAxes(MatmulExprContext &ctx, att::AttAxisPtr &m) {
  ctx.expr_m = att::CreateExpr("m_size");

  att::SymVarInfoPtr sym_m = std::make_shared<att::SymVarInfo>(ctx.expr_m);

  m = std::make_shared<att::AttAxis>();
  m->name = "m";
  m->axis_pos = att::AxisPosition::ORIGIN;
  m->bind_multicore = false;
  m->is_last = false;
  m->is_node_innerest_dim = false;
  m->size = sym_m;
}

void BuildNAxes(MatmulExprContext &ctx, att::AttAxisPtr &n) {
  ctx.expr_n = att::CreateExpr("n_size");

  att::SymVarInfoPtr sym_n = std::make_shared<att::SymVarInfo>(ctx.expr_n);

  n = std::make_shared<att::AttAxis>();
  n->name = "n";
  n->axis_pos = att::AxisPosition::ORIGIN;
  n->bind_multicore = false;
  n->is_last = false;
  n->is_node_innerest_dim = false;
  n->size = sym_n;
}

void BuildKAxes(MatmulExprContext &ctx, att::AttAxisPtr &k, att::AttAxisPtr &stepka, att::AttAxisPtr &stepkb) {
  ctx.expr_k = att::CreateExpr("k_size");
  ctx.expr_stepka = att::CreateExpr("stepka_size");
  ctx.expr_stepkb = att::CreateExpr("stepkb_size");

  att::SymVarInfoPtr sym_k = std::make_shared<att::SymVarInfo>(ctx.expr_k);
  att::SymVarInfoPtr sym_stepka = std::make_shared<att::SymVarInfo>(ctx.expr_stepka);
  sym_stepka->align = ge::Symbol(256);
  sym_stepka->related_scope = {att::HardwareDef::L1};
  att::SymVarInfoPtr sym_stepkb = std::make_shared<att::SymVarInfo>(ctx.expr_stepkb);
  sym_stepkb->align = ge::Symbol(16);
  sym_stepkb->related_scope = {att::HardwareDef::L1};

  k = std::make_shared<att::AttAxis>();
  stepka = std::make_shared<att::AttAxis>();
  stepkb = std::make_shared<att::AttAxis>();

  k->name = "k";
  k->axis_pos = att::AxisPosition::ORIGIN;
  k->bind_multicore = false;
  k->is_last = false;
  k->is_node_innerest_dim = false;
  k->size = sym_k;

  stepka->name = "stepka";
  stepka->axis_pos = att::AxisPosition::INNER;
  stepka->bind_multicore = false;
  stepka->is_last = false;
  stepka->is_node_innerest_dim = true;
  stepka->size = sym_stepka;
  stepka->orig_axis.push_back(k.get());
  stepka->from_axis = {k.get()};

  stepkb->name = "stepkb";
  stepkb->axis_pos = att::AxisPosition::INNER;
  stepkb->bind_multicore = false;
  stepkb->is_last = true;
  stepkb->is_node_innerest_dim = true;
  stepkb->size = sym_stepkb;
  stepkb->orig_axis.push_back(k.get());
  stepkb->from_axis = {stepka.get()};
}

void AppendArgList(att::ModelInfo &model_info, const att::AttAxisPtr &m, const att::AttAxisPtr &n,
                   const att::AttAxisPtr &k, const att::AttAxisPtr &stepka, const att::AttAxisPtr &stepkb) {
  model_info.arg_list.emplace_back(m);
  model_info.arg_list.emplace_back(n);
  model_info.arg_list.emplace_back(k);
  model_info.arg_list.emplace_back(stepka);
  model_info.arg_list.emplace_back(stepkb);
}

void FillMatmulHardwareCons(att::ModelInfo &model_info, const MatmulExprContext &ctx) {
  model_info.hardware_cons[att::HardwareDef::L1] = (ctx.expr_stepka * ctx.expr_stepkb * att::CreateExpr(4));
  model_info.hardware_cons[att::HardwareDef::UB] = att::CreateExpr(0L);
}

void FillModelInfo(att::ModelInfo &model_info, const MatmulExprContext &ctx) {
  FillMatmulHardwareCons(model_info, ctx);

  model_info.tiling_case_id = 1;
  model_info.eq_exprs[att::kFatherToChildNoTail].push_back(std::pair(ctx.expr_stepka, ctx.expr_stepkb));
  model_info.leq_exprs[att::kFatherToChildLarger].push_back((ctx.expr_stepka - ctx.expr_k));
  model_info.container_exprs["Q1"] = (ctx.expr_m + ctx.expr_n);
  model_info.tensor_exprs["MATMUL_OUTPUT1"] = (ctx.expr_m + ctx.expr_n);
  model_info.output_size = 1;
}
}  // namespace

namespace att {
ModelInfo GenMatmulModelInfo() {
  ModelInfo model_info;
  MatmulExprContext ctx;
  BuildCoreAxis(model_info, ctx);

  AttAxisPtr m;
  BuildMAxes(ctx, m);

  AttAxisPtr n;
  BuildNAxes(ctx, n);

  AttAxisPtr k;
  AttAxisPtr stepka;
  AttAxisPtr stepkb;
  BuildKAxes(ctx, k, stepka, stepkb);

  AppendArgList(model_info, m, n, k, stepka, stepkb);
  FillModelInfo(model_info, ctx);
  return model_info;
}
}  // namespace att
