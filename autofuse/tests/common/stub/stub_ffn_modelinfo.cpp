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
#include "stub_ffn_modelinfo.h"

namespace {
att::Expr GetSafeDivisor(const att::Expr &expr) {
  return af::sym::Max(af::sym::kSymbolOne, expr);
}

att::Expr GetSafeOffsetDivisor(const att::Expr &expr) {
  return af::sym::Max(af::sym::kSymbolOne, att::CreateExpr(-1) + expr);
}

struct FfnExprContext {
  att::Expr expr_maxTokens;
  att::Expr expr_ubm;
  att::Expr expr_n1;
  att::Expr expr_basen1;
  att::Expr expr_k1;
  att::Expr expr_n2;
  att::Expr expr_basen2;
};

void BuildFfnMaxTokenAxes(att::ModelInfo &model_info, FfnExprContext &ctx, att::AttAxisPtr &maxTokens,
                          att::AttAxisPtr &ubm) {
  ctx.expr_maxTokens = att::CreateExpr("maxTokens");
  ctx.expr_ubm = att::CreateExpr("ub_m");

  att::SymVarInfoPtr sym_maxTokens = std::make_shared<att::SymVarInfo>(ctx.expr_maxTokens);
  att::SymVarInfoPtr sym_ubm = std::make_shared<att::SymVarInfo>(ctx.expr_ubm);
  sym_ubm->align = ge::Symbol(8);
  sym_ubm->related_scope = {att::HardwareDef::UB};

  maxTokens = std::make_shared<att::AttAxis>();
  ubm = std::make_shared<att::AttAxis>();

  maxTokens->name = "maxTokens";
  maxTokens->axis_pos = att::AxisPosition::ORIGIN;
  maxTokens->bind_multicore = false;
  maxTokens->is_last = false;
  maxTokens->is_node_innerest_dim = false;
  maxTokens->size = sym_maxTokens;

  ubm->name = "ub_m";
  ubm->axis_pos = att::AxisPosition::INNER;
  ubm->bind_multicore = false;
  ubm->is_last = true;
  ubm->is_node_innerest_dim = false;
  ubm->size = sym_ubm;
  ubm->orig_axis.push_back(maxTokens.get());
  ubm->from_axis = {maxTokens.get()};
}

void BuildFfnN1Axes(FfnExprContext &ctx, att::AttAxisPtr &n1, att::AttAxisPtr &basen1) {
  ctx.expr_n1 = att::CreateExpr("N1");
  ctx.expr_basen1 = att::CreateExpr("base_n1");

  att::SymVarInfoPtr sym_n1 = std::make_shared<att::SymVarInfo>(ctx.expr_n1);
  att::SymVarInfoPtr sym_basen1 = std::make_shared<att::SymVarInfo>(ctx.expr_basen1);
  sym_basen1->align = ge::Symbol(8);
  sym_basen1->related_scope = {att::HardwareDef::UB, att::HardwareDef::BTBUF};

  n1 = std::make_shared<att::AttAxis>();
  basen1 = std::make_shared<att::AttAxis>();

  n1->name = "N1";
  n1->axis_pos = att::AxisPosition::ORIGIN;
  n1->bind_multicore = false;
  n1->is_last = false;
  n1->is_node_innerest_dim = false;
  n1->size = sym_n1;

  basen1->name = "base_n1";
  basen1->axis_pos = att::AxisPosition::INNER;
  basen1->bind_multicore = false;
  basen1->is_last = true;
  basen1->is_node_innerest_dim = true;
  basen1->size = sym_basen1;
  basen1->orig_axis.push_back(n1.get());
  basen1->from_axis = {n1.get()};
}

void BuildFfnK1Axis(FfnExprContext &ctx, att::AttAxisPtr &k1) {
  ctx.expr_k1 = att::CreateExpr("K1");
  att::SymVarInfoPtr sym_k1 = std::make_shared<att::SymVarInfo>(ctx.expr_k1);
  k1 = std::make_shared<att::AttAxis>();
  k1->name = "K1";
  k1->axis_pos = att::AxisPosition::ORIGIN;
  k1->bind_multicore = false;
  k1->is_last = false;
  k1->is_node_innerest_dim = false;
  k1->size = sym_k1;
}

void BuildFfnN2Axes(FfnExprContext &ctx, att::AttAxisPtr &n2, att::AttAxisPtr &basen2) {
  ctx.expr_n2 = att::CreateExpr("N2");
  ctx.expr_basen2 = att::CreateExpr("base_n2");

  att::SymVarInfoPtr sym_n2 = std::make_shared<att::SymVarInfo>(ctx.expr_n2);
  att::SymVarInfoPtr sym_basen2 = std::make_shared<att::SymVarInfo>(ctx.expr_basen2);
  sym_basen2->align = ge::Symbol(8);
  sym_basen2->related_scope = {att::HardwareDef::BTBUF};

  n2 = std::make_shared<att::AttAxis>();
  basen2 = std::make_shared<att::AttAxis>();

  n2->name = "N2";
  n2->axis_pos = att::AxisPosition::ORIGIN;
  n2->bind_multicore = false;
  n2->is_last = false;
  n2->is_node_innerest_dim = false;
  n2->size = sym_n2;

  basen2->name = "base_n2";
  basen2->axis_pos = att::AxisPosition::INNER;
  basen2->bind_multicore = false;
  basen2->is_last = true;
  basen2->is_node_innerest_dim = true;
  basen2->size = sym_basen2;
  basen2->orig_axis.push_back(n2.get());
  basen2->from_axis = {n2.get()};
}

void FillFfnModelInfo(att::ModelInfo &model_info, const FfnExprContext &ctx) {
  att::Expr btbuf_occupy = af::sym::Max((att::CreateExpr(4) * ctx.expr_basen1), (att::CreateExpr(4) * ctx.expr_basen2));
  att::Expr ub_occupy = (att::CreateExpr(4) * ctx.expr_basen1 * ctx.expr_ubm);
  model_info.hardware_cons[att::HardwareDef::BTBUF] = btbuf_occupy;
  model_info.hardware_cons[att::HardwareDef::UB] = ub_occupy;

  att::Expr m_cnt = af::sym::Ceiling(ctx.expr_maxTokens / GetSafeDivisor(ctx.expr_ubm));
  att::Expr n1_cnt = af::sym::Ceiling(ctx.expr_n1 / GetSafeDivisor(ctx.expr_basen1));
  att::Expr n2_cnt = af::sym::Ceiling(ctx.expr_n2 / GetSafeDivisor(ctx.expr_basen2));

  att::Expr vec_ub = ((att::CreateExpr(4) * ctx.expr_basen1 * ctx.expr_ubm) / GetSafeOffsetDivisor(ctx.expr_basen1) +
                      att::CreateExpr(4));
  att::Expr vec = vec_ub * (m_cnt * n1_cnt);

  att::Expr mte3_ub = ((att::CreateExpr(0.01741f) * ctx.expr_basen1 * ctx.expr_ubm) + att::CreateExpr(0.22f));
  att::Expr v_mte3 = mte3_ub * (m_cnt * n1_cnt);

  att::Expr mte2_n1 =
      ((att::CreateExpr(5.01f) / (att::CreateExpr(27240.69f) + ctx.expr_basen1)) + att::CreateExpr(1051.66f)) *
      (ctx.expr_basen1 / att::CreateExpr(30421.24f));
  att::Expr mte2_n2 =
      ((att::CreateExpr(5.01f) / (att::CreateExpr(27240.69f) + ctx.expr_basen2)) + att::CreateExpr(1051.66f)) *
      (ctx.expr_basen2 / att::CreateExpr(30421.24f));
  att::Expr mte2_ub = (att::CreateExpr(0.007f) * ctx.expr_basen1 * ctx.expr_ubm) + att::CreateExpr(7.97f);
  att::Expr v_mte2 = mte2_n1 * n1_cnt + mte2_n2 * n2_cnt + mte2_ub * (m_cnt * n1_cnt);

  model_info.objects[att::PipeType::AIV_MTE2] = v_mte2;
  model_info.objects[att::PipeType::AIV_MTE3] = v_mte3;
  model_info.objects[att::PipeType::AIV_VEC] = vec;
  model_info.tiling_case_id = 0;
  model_info.output_size = 1;
}

void AppendFfnArgList(att::ModelInfo &model_info, const att::AttAxisPtr &maxTokens, const att::AttAxisPtr &basen1,
                      const att::AttAxisPtr &basen2, const att::AttAxisPtr &n1, const att::AttAxisPtr &k1,
                      const att::AttAxisPtr &n2, const att::AttAxisPtr &ubm) {
  model_info.arg_list.emplace_back(maxTokens);
  model_info.arg_list.emplace_back(basen1);
  model_info.arg_list.emplace_back(basen2);
  model_info.arg_list.emplace_back(n1);
  model_info.arg_list.emplace_back(k1);
  model_info.arg_list.emplace_back(n2);
  model_info.arg_list.emplace_back(ubm);
}
}  // namespace

namespace att {
ModelInfo GenFFNModelInfo() {
  ModelInfo model_info;
  FfnExprContext ctx;

  AttAxisPtr maxTokens;
  AttAxisPtr ubm;
  BuildFfnMaxTokenAxes(model_info, ctx, maxTokens, ubm);

  AttAxisPtr n1;
  AttAxisPtr basen1;
  BuildFfnN1Axes(ctx, n1, basen1);

  AttAxisPtr k1;
  BuildFfnK1Axis(ctx, k1);

  AttAxisPtr n2;
  AttAxisPtr basen2;
  BuildFfnN2Axes(ctx, n2, basen2);

  FillFfnModelInfo(model_info, ctx);
  AppendFfnArgList(model_info, maxTokens, basen1, basen2, n1, k1, n2, ubm);
  return model_info;
}
}  // namespace att
