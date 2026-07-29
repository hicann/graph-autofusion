/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "generator/preprocess/args_replace.h"

#include <string>
#include <array>

#include "common/checker.h"
#include "util/base_types_printer.h"
#include "base/att_const_values.h"
#include "generator/preprocess/preprocess_utils.h"

namespace att {
const std::string kAlignDelim = "_div_align";
const std::string kPowBase = "_base";
const int32_t kDefualtInitValue = 0;

inline bool IsPowerOfTwo(int32_t n) {
  if (n <= 0) {
    return false;
  }
  uint32_t val = static_cast<uint32_t>(n);
  return (val & (val - 1)) == 0U;
}

void ArgsReplacer::GetReplaceResult(ExprExprMap &replaced_vars, ExprExprMap &replacements,
                                    ExprExprMap &new_expr_replacements) {
  for (auto &new_to_ori_expr : new_expr_ori_expr_map_) {
    if (new_to_ori_expr.first != new_to_ori_expr.second) {
      replaced_vars.emplace(new_to_ori_expr.first, new_to_ori_expr.second);
      replacements.emplace(new_to_ori_expr.second, ori_expr_new_expr_map_[new_to_ori_expr.second]);
      new_expr_replacements.emplace(new_to_ori_expr.first, new_expr_replacements_[new_to_ori_expr.first]);
    }
  }
}

void ArgsReplacer::GetNewExprInitValue(ExprExprMap &new_expr_init_values) {
  new_expr_init_values = new_expr_init_values_;
}

bool ArgsReplacer::DoReplace(const std::map<std::string, std::vector<std::pair<Expr, Expr>>> &eq_exprs) {
  Reset();
  if (!InitWithEqCons(eq_exprs)) {
    return false;
  }
  if (!GetLeafExprs()) {
    GELOGW("Replace failed. Get leaf expr failed.");
    return false;
  }
  if (!ReplaceParentExprs()) {
    GELOGW("Replace failed. Replace parent expr failed.");
    return false;
  }
  ReplaceNaiveExpr();
  return true;
}

// 按对齐值来做变量替换，如果对齐值为1，那么不需要做变量替换，否则替换为x_div_align*align
Expr ArgsReplacer::ReplaceCommonExpr(const Expr &e, const Expr &align, ExprExprMap &new_expr_ori_expr_map,
                                     ExprExprMap &new_expr_replacements) {
  if (align.IsConstExpr() &&
      (af::SymbolicUtils::StaticCheckLe(align, af::Symbol(kMinDimLength)) == af::TriBool::kTrue)) {
    new_expr_ori_expr_map.emplace(e, e);
    new_expr_replacements.emplace(e, e);
    return e;
  }
  Expr new_epxr = CreateExpr((Str(e) + kAlignDelim).c_str());
  new_expr_ori_expr_map.emplace(new_epxr, e);
  auto replace_var = af::sym::Mul(align, new_epxr);
  new_expr_replacements.emplace(new_epxr, af::sym::Div(e, align));
  new_expr_init_values_.emplace(new_epxr, af::sym::kSymbolOne);
  return replace_var;
}

bool ArgsReplacer::IsAllFactorReplaced(const ExprExprMap &replaced_vars, std::vector<Expr> &factors) const {
  for (const auto &factor : factors) {
    if (replaced_vars.find(factor) == replaced_vars.end()) {
      return false;
    }
  }
  return true;
}

bool ArgsReplacer::InitWithEqCons(const std::map<std::string, std::vector<std::pair<Expr, Expr>>> &eq_exprs) {
  for (const auto &eq_cons : eq_exprs) {
    // 如果等式约束中包含非整除约束，返回失败
    GE_ASSERT_TRUE(eq_cons.first == kFatherToChildNoTail, "CreateExpr[%s] Replace doesn't support non notail eq exprs.",
                   eq_cons.first.c_str());
    for (const auto &expr : eq_cons.second) {
      expr_factors_map_[expr.first].emplace_back(expr.second);
      factor_expr_map_[expr.second] = expr.first;
    }
  }
  return true;
}

bool ArgsReplacer::GetLeafExprs() {
  for (auto &expr_factors : expr_factors_map_) {
    for (auto &factor : expr_factors.second) {
      if (expr_factors_map_.find(factor) == expr_factors_map_.end()) {
        // 要求当前叶子节点的对齐值为2的幂次
        if (!IsInExprInfo(vars_infos_, factor)) {
          return false;
        }
        Expr factor_align = vars_infos_.at(factor).align;
        if ((factor_align.IsConstExpr() &&
             (af::SymbolicUtils::StaticCheckLe(factor_align, af::Symbol(kMinDimLength)) == af::TriBool::kTrue)) ||
            (vars_infos_.at(factor).do_search == false)) {
          GetSelfReplacedVars(factor);
          continue;
        }
        if (factor_align.IsConstExpr()) {
          int32_t factor_align_const_value;
          factor_align.GetConstValue(factor_align_const_value);
          GE_ASSERT_TRUE(IsPowerOfTwo(factor_align_const_value),
                         "CreateExpr Replace doesn't support align is not power of 2.");
        }
        // 每个叶子节点只能有一个父节点，反过来一个父节点可以有多个子节点
        GE_ASSERT_TRUE(ori_expr_new_expr_map_.find(factor) == ori_expr_new_expr_map_.end(),
                       "CreateExpr Replace doesn't support multi-parent case.");
        // 对于存在变量整除关系的子节点，变量替换规则为align * 2^new_var
        Expr new_variable = CreateExpr((Str(factor) + kPowBase).c_str());
        Expr new_factor_expr = af::sym::Mul(factor_align, af::sym::Pow(CreateExpr(kBaseTwo), new_variable));
        new_expr_replacements_.emplace(new_variable,
                                       af::sym::Log(af::sym::Div(factor, factor_align), CreateExpr(kBaseTwo)));
        new_expr_init_values_.emplace(new_variable, af::sym::kSymbolZero);
        new_expr_ori_expr_map_.emplace(new_variable, factor);
        ori_expr_new_expr_map_.emplace(factor, new_factor_expr);
        replaced_expr_queue_.push(factor);
      }
    }
  }
  return true;
}

void ArgsReplacer::GetSelfReplacedVars(const Expr &expr) {
  new_expr_replacements_.emplace(expr, expr);
  new_expr_ori_expr_map_.emplace(expr, expr);
  ori_expr_new_expr_map_.emplace(expr, expr);
  replaced_expr_queue_.push(expr);
}

void ArgsReplacer::GetAlignReplacedVars(const Expr &expr) {
  Expr replaced_expr =
      ReplaceCommonExpr(expr, vars_infos_.at(expr).align, new_expr_ori_expr_map_, new_expr_replacements_);
  ori_expr_new_expr_map_.emplace(expr, replaced_expr);
  replaced_expr_queue_.push(expr);
}

void ArgsReplacer::GetFactorReplacedVars(const Expr &expr) {
  if (IsAllFactorReplaced(ori_expr_new_expr_map_, expr_factors_map_[expr])) {
    Expr new_align_expr = vars_infos_.at(expr).align;
    for (auto &factor : expr_factors_map_[expr]) {
      new_align_expr = af::sym::Max(new_align_expr, ori_expr_new_expr_map_[factor]);
    }
    // root节点，不要求为2的幂次
    Expr new_variable;
    Expr new_expr;
    if (factor_expr_map_.find(expr) == factor_expr_map_.end()) {
      new_variable = CreateExpr((Str(expr) + kAlignDelim).c_str());
      new_expr = af::sym::Mul(new_align_expr, new_variable);
      new_expr_replacements_.emplace(new_variable, af::sym::Div(expr, new_align_expr));
      new_expr_init_values_.emplace(new_variable, af::sym::kSymbolOne);
    } else {
      new_variable = CreateExpr((Str(expr) + kPowBase).c_str());
      new_expr = af::sym::Mul(new_align_expr, af::sym::Pow(CreateExpr(kBaseTwo), new_variable));
      new_expr_replacements_.emplace(new_variable,
                                     af::sym::Log(af::sym::Div(expr, new_align_expr), CreateExpr(kBaseTwo)));
      new_expr_init_values_.emplace(new_variable, af::sym::kSymbolZero);
    }
    new_expr_ori_expr_map_.emplace(new_variable, expr);
    ori_expr_new_expr_map_.emplace(expr, new_expr);
    replaced_expr_queue_.push(expr);
  }
}

bool ArgsReplacer::ReplaceParentExprs() {
  while (!replaced_expr_queue_.empty()) {
    Expr current_expr = replaced_expr_queue_.front();
    replaced_expr_queue_.pop();
    const std::vector<Expr> parent_exprs = vars_infos_.at(current_expr).from_axis_size;
    for (const auto &parent_expr : parent_exprs) {
      if (!IsInExprInfo(vars_infos_, parent_expr)) {
        continue;
      }
      if (!vars_infos_.at(parent_expr).do_search) {  // 只替换待搜索变量
        GetSelfReplacedVars(parent_expr);
        continue;
      }
      // 如果当前已经被替换了的变量的父节点与当前轴没有整除约束，
      // 那么只需要判断当前变量是否做关于对齐值的变量替换
      if (expr_factors_map_.find(parent_expr) == expr_factors_map_.end()) {
        GetAlignReplacedVars(parent_expr);
      } else {
        GetFactorReplacedVars(parent_expr);
      }
    }
  }
  return true;
}

void ArgsReplacer::ReplaceNaiveExpr() {
  for (const auto &expr_info : vars_infos_) {
    if (ori_expr_new_expr_map_.find(expr_info.first) == ori_expr_new_expr_map_.end()) {
      if (!expr_info.second.do_search) {
        ori_expr_new_expr_map_.emplace(expr_info.first, expr_info.first);
        continue;
      }
      Expr new_expr =
          ReplaceCommonExpr(expr_info.first, expr_info.second.align, new_expr_ori_expr_map_, new_expr_replacements_);
      ori_expr_new_expr_map_.emplace(expr_info.first, new_expr);
    }
  }
}

void ArgsReplacer::Reset() {
  ori_expr_new_expr_map_.clear();
  new_expr_ori_expr_map_.clear();
  new_expr_replacements_.clear();
  new_expr_init_values_.clear();
  factor_expr_map_.clear();
  std::queue<Expr> empty_queue;
  replaced_expr_queue_.swap(empty_queue);
  expr_factors_map_.clear();
}
}  // namespace att
