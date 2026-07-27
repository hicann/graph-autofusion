/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ternary_op.h"
#include <cmath>

#include "generator/preprocess/ast_optimizer.h"

namespace att {
namespace {
void AddUsedArgs(const Expr &expr, std::vector<Expr> &used_args) {
  for (const auto &arg : expr.FreeSymbols()) {
    used_args.emplace_back(arg);
  }
}
}  // namespace

std::string IfCase::GetStr() const {
  if (choice_b_ == nullptr) {
    return Str(expr_);
  } else {
    std::string cond_str;
    if (cond_type_ == CondType::K_EQ) {
      cond_str = "IsEqual(" + Str(cond_left_) + ", " + Str(cond_right_) + ")";
    } else if (cond_type_ == CondType::K_LT) {
      cond_str = Str(cond_left_) + " < " + Str(cond_right_);
    } else if (cond_type_ == CondType::K_GT) {
      cond_str = Str(cond_left_) + " > " + Str(cond_right_);
    } else if (cond_type_ == CondType::K_LE) {
      cond_str = Str(cond_left_) + " <= " + Str(cond_right_);
    } else if (cond_type_ == CondType::K_GE) {
      cond_str = Str(cond_left_) + " >= " + Str(cond_right_);
    }
    return "TernaryOp(" + cond_str + ", " + choice_a_->GetStr() + ", " + choice_b_->GetStr() + ")";
  }
}

void IfCase::Replace(const std::vector<std::pair<Expr, Expr>> &replace_vars) {
  if (choice_b_ != nullptr) {
    cond_left_ = cond_left_.Replace(replace_vars);
    cond_right_ = cond_right_.Replace(replace_vars);
    choice_a_->Replace(replace_vars);
    choice_b_->Replace(replace_vars);
  } else {
    expr_ = expr_.Replace(replace_vars);
  }
}

std::shared_ptr<IfCase> IfCase::DeepCopy() const {
  if (choice_b_ == nullptr) {
    return std::make_shared<IfCase>(expr_);
  }
  return std::make_shared<IfCase>(cond_type_, cond_left_, cond_right_, choice_a_->DeepCopy(), choice_b_->DeepCopy());
}

void IfCase::GetUsedArgs(std::vector<Expr> &used_args) const {
  if (choice_b_ != nullptr) {
    AddUsedArgs(cond_left_, used_args);
    AddUsedArgs(cond_right_, used_args);
    choice_a_->GetUsedArgs(used_args);
    choice_b_->GetUsedArgs(used_args);
  } else {
    AddUsedArgs(expr_, used_args);
  }
}

TernaryOp::TernaryOp(const Expr &expr) {
  ternary_op_ = std::make_shared<IfCase>(expr);
  AddUsedArgs(expr, related_vars_);
}

TernaryOp::TernaryOp(const CondType &cond_type, const Expr &cond_left, const Expr &cond_right, const Expr &choice_a,
                     const Expr &choice_b) {
  ternary_op_ = std::make_shared<IfCase>(cond_type, cond_left, cond_right, std::make_shared<IfCase>(choice_a),
                                         std::make_shared<IfCase>(choice_b));
  AddUsedArgs(cond_left, related_vars_);
  AddUsedArgs(cond_right, related_vars_);
  AddUsedArgs(choice_a, related_vars_);
  AddUsedArgs(choice_b, related_vars_);
}

TernaryOp::TernaryOp(const CondType &cond_type, const Expr &cond_left, const Expr &cond_right,
                     std::shared_ptr<IfCase> &&if_case_a, std::shared_ptr<IfCase> &&if_case_b) {
  ternary_op_ = std::make_shared<IfCase>(cond_type, cond_left, cond_right, std::move(if_case_a), std::move(if_case_b));
  ternary_op_->GetUsedArgs(related_vars_);
}

TernaryOp::TernaryOp(const Expr &var, std::shared_ptr<IfCase> &&op, const std::vector<Expr> &related) {
  variable_ = var;
  ternary_op_ = std::move(op);
  for (const auto &arg : related) {
    related_vars_.emplace_back(arg);
  }
}

TernaryOp::TernaryOp(const CondType &cond_type, const Expr &cond_left, const Expr &cond_right,
                     const TernaryOp &ternary_op_a, const TernaryOp &ternary_op_b) {
  ternary_op_ = std::make_shared<IfCase>(cond_type, cond_left, cond_right, ternary_op_a.DeepCopyIfCase(),
                                         ternary_op_b.DeepCopyIfCase());
  for (const auto &var : ternary_op_a.GetRelatedVars()) {
    related_vars_.emplace_back(var);
  }
  for (const auto &var : ternary_op_b.GetRelatedVars()) {
    related_vars_.emplace_back(var);
  }
  AddUsedArgs(cond_left, related_vars_);
  AddUsedArgs(cond_right, related_vars_);
}

void TernaryOp::SetVariable(const Expr &expr) {
  variable_ = expr;
}

void TernaryOp::SetDescription(const std::string &desc) {
  description_ = desc;
}

std::string TernaryOp::GetDescription() const {
  return description_;
}

Expr TernaryOp::GetVariable() const {
  return variable_;
}

std::string TernaryOp::GetTernaryOpStr() const {
  return ternary_op_->GetStr();
}

void TernaryOp::UpdateRelatedVars(const std::vector<std::pair<Expr, Expr>> &replace_vars) {
  ExprExprMap replace_ops;
  for (const auto &pair : replace_vars) {
    replace_ops[pair.first] = pair.second;
  }
  std::vector<Expr> new_related_vars;
  for (const auto &var : related_vars_) {
    auto iter = replace_ops.find(var);
    if (iter != replace_ops.end()) {
      new_related_vars.emplace_back(iter->second);
    } else {
      new_related_vars.emplace_back(var);
    }
  }
  related_vars_ = new_related_vars;
}

void TernaryOp::Replace(const std::vector<std::pair<Expr, Expr>> &replace_vars) {
  ternary_op_->Replace(replace_vars);
  ExprExprMap vars_map;
  std::vector<Expr> new_related_vars;
  for (const auto &pair : replace_vars) {
    vars_map[pair.first] = pair.second;
  }
  for (const auto &var : related_vars_) {
    auto iter = vars_map.find(var);
    if (iter != vars_map.end()) {
      for (const auto &arg : iter->second.FreeSymbols()) {
        new_related_vars.emplace_back(arg);
      }
    } else {
      new_related_vars.emplace_back(var);
    }
  }
  related_vars_ = new_related_vars;
}

std::vector<Expr> TernaryOp::GetRelatedVars() const {
  std::vector<Expr> res;
  for (const auto &var : related_vars_) {
    res.emplace_back(var);
  }
  return res;
}

std::vector<Expr> TernaryOp::GetUsedArgs() const {
  std::vector<Expr> used_args;
  ternary_op_->GetUsedArgs(used_args);
  return used_args;
}

TernaryOp TernaryOp::DeepCopy() const {
  TernaryOp copy(variable_, ternary_op_->DeepCopy(), related_vars_);
  copy.description_ = description_;  // 复制描述信息
  return copy;
}

std::shared_ptr<IfCase> TernaryOp::DeepCopyIfCase() const {
  return ternary_op_->DeepCopy();
}

namespace {
bool InTernaryOps(const TernaryOp &ternary_op, const std::map<Expr, TernaryOp, ExprCmp> &ternary_ops,
                  const ExprExprMap &res, std::stack<Expr> &replace_stack) {
  bool ret = false;
  for (const auto &args : ternary_op.GetRelatedVars()) {
    if (ternary_ops.find(args) != ternary_ops.end() && res.find(args) == res.end()) {
      ret = true;
      replace_stack.push(args);
    }
  }
  return ret;
}

void AddRelatedVars(const Expr &expr, const TernaryOp &ternary_op,
                    const std::map<Expr, TernaryOp, ExprCmp> &ternary_ops,
                    std::map<Expr, std::vector<Expr>, ExprCmp> &res) {
  std::vector<Expr> related_vars;
  for (const auto &arg : ternary_op.GetRelatedVars()) {
    if (const auto iter = ternary_ops.find(arg); iter != ternary_ops.end()) {
      if (res.find(arg) == res.end()) {
        AddRelatedVars(arg, iter->second, ternary_ops, res);
      }
      for (const auto &var : res.at(arg)) {
        related_vars.emplace_back(var);
      }
    } else {
      related_vars.emplace_back(arg);
    }
  }
  res[expr] = related_vars;
}
}  // namespace

std::vector<std::pair<Expr, Expr>> ConcursiveReplaceVars(const std::map<Expr, TernaryOp, ExprCmp> &ternary_ops) {
  Expr cur_var;
  Expr replace_var;
  ExprExprMap res;
  TernaryOp cur_ternary_op;
  std::stack<Expr> replace_stack;
  std::vector<std::pair<Expr, Expr>> replace_vars;
  for (const auto &pair : ternary_ops) {
    if (res.find(pair.first) != res.end()) {
      continue;
    }
    InTernaryOps(pair.second, ternary_ops, res, replace_stack);
    while (!replace_stack.empty()) {
      cur_var = replace_stack.top();
      cur_ternary_op = ternary_ops.at(cur_var).DeepCopy();
      if (!InTernaryOps(cur_ternary_op, ternary_ops, res, replace_stack)) {
        cur_ternary_op.Replace(replace_vars);
        replace_var = CreateExpr(cur_ternary_op.GetTernaryOpStr().c_str());
        res[cur_var] = replace_var;
        replace_vars.emplace_back(std::make_pair(cur_var, replace_var));
        GELOGD("Make concursive replace [%s] -> [%s].", Str(cur_var).c_str(), Str(replace_var).c_str());
        replace_stack.pop();
      }
    }
    cur_var = pair.first;
    cur_ternary_op = pair.second.DeepCopy();
    cur_ternary_op.Replace(replace_vars);
    replace_var = CreateExpr(cur_ternary_op.GetTernaryOpStr().c_str());
    res[cur_var] = replace_var;
    replace_vars.emplace_back(std::make_pair(cur_var, replace_var));
    GELOGD("Make concursive replace [%s] -> [%s].", Str(cur_var).c_str(), Str(replace_var).c_str());
  }
  return replace_vars;
}

std::map<Expr, std::vector<Expr>, ExprCmp> ConcursiveRelatedVars(
    const std::map<Expr, TernaryOp, ExprCmp> &ternary_ops) {
  std::vector<Expr> cur_related;
  std::map<Expr, std::vector<Expr>, ExprCmp> res;
  for (const auto &pair : ternary_ops) {
    if (res.find(pair.first) != res.end()) {
      continue;
    }
    AddRelatedVars(pair.first, pair.second, ternary_ops, res);
  }
  for (const auto &pair : res) {
    GELOGD("Make concursive vars [%s]:{%s}.", Str(pair.first).c_str(), GetVecString(pair.second).c_str());
  }
  return res;
}

void GetPerfVar(const std::string &prefix, Expr &res, const std::map<Expr, TernaryOp, ExprCmp> &ternary_ops) {
  uint32_t idx = 0;
  std::string perf_name = prefix;
  res = CreateExpr(perf_name.c_str());
  while (ternary_ops.find(res) != ternary_ops.end()) {
    perf_name = prefix + std::to_string(++idx);
    res = CreateExpr(perf_name.c_str());
  }
}

namespace {
// 叶子节点表达式超过此长度时提取为命名变量，避免内联过长
constexpr size_t kLeafExprMaxInlineLen = 60U;
// 重复子表达式达到此长度时提取为公共变量，避免二分树叶子重复展开
constexpr size_t kCommonExprMinLen = 60U;

struct DecomposeContext {
  DecomposeContext(const std::string &var_prefix, std::string &output) : prefix(var_prefix), preamble(output) {}

  std::string prefix;
  std::string &preamble;
  int node_counter = 0;
  int common_counter = 0;
  std::map<std::string, size_t> ast_ref_count;
  std::map<std::string, std::string> ast_hash_to_var;
  std::map<std::string, ASTPtr> expr_asts;
  AstCanonicalizer ast_canonicalizer;
};

struct BuiltAstExpr {
  std::string expr;
  bool replaced = false;
};

bool IsAstLeaf(const ASTNode &node) {
  return node.type == NodeType::VARIABLE || node.type == NodeType::NUMBER;
}

ASTPtr ParseCompleteAst(const std::string &expr_str, DecomposeContext &context) {
  auto ast_iter = context.expr_asts.find(expr_str);
  if (ast_iter != context.expr_asts.end()) {
    return ast_iter->second;
  }
  Parser parser(expr_str, false);
  ASTPtr ast = parser.Parse();
  if (!parser.IsFullyParsed()) {
    ast = nullptr;
  } else {
    context.ast_canonicalizer.Assign(ast.get());
  }
  context.expr_asts[expr_str] = ast;
  return ast;
}

void CountAstRefs(const ASTPtr &node, DecomposeContext &context) {
  if (node == nullptr) {
    return;
  }
  if (!IsAstLeaf(*node)) {
    ++context.ast_ref_count[node->hash];
  }
  for (const auto &child : node->children) {
    CountAstRefs(child, context);
  }
}

void CountExprAstRefs(const Expr &expr, DecomposeContext &context) {
  const std::string expr_str = Str(expr);
  if (expr_str.length() < kCommonExprMinLen) {
    return;
  }
  CountAstRefs(ParseCompleteAst(expr_str, context), context);
}

void CountIfCaseAstRefs(const IfCase &node, DecomposeContext &context) {
  if (node.IsLeaf()) {
    CountExprAstRefs(node.GetExpr(), context);
    return;
  }
  CountExprAstRefs(node.GetCondLeft(), context);
  CountExprAstRefs(node.GetCondRight(), context);
  CountIfCaseAstRefs(*node.GetChoiceA(), context);
  CountIfCaseAstRefs(*node.GetChoiceB(), context);
}

bool AppendAstExpr(const ASTPtr &node, DecomposeContext &context, std::string &output);

bool AppendAstNode(const ASTPtr &node, DecomposeContext &context, std::string &output) {
  bool replaced = false;
  if (node->type == NodeType::FUNCTION) {
    output += node->op + "(";
    for (size_t i = 0U; i < node->children.size(); ++i) {
      output += i == 0U ? "" : ", ";
      replaced = AppendAstExpr(node->children[i], context, output) || replaced;
    }
    output += ")";
    return replaced;
  }
  if (node->type == NodeType::OPERATOR && node->children.size() == 2U) {
    output += "(";
    replaced = AppendAstExpr(node->children[0], context, output) || replaced;
    output += " " + node->op + " ";
    replaced = AppendAstExpr(node->children[1], context, output) || replaced;
    output += ")";
    return replaced;
  }
  output += node->expr;
  return false;
}

bool AppendAstExpr(const ASTPtr &node, DecomposeContext &context, std::string &output) {
  if (node == nullptr) {
    return false;
  }
  if (IsAstLeaf(*node)) {
    output += node->expr;
    return false;
  }
  auto var_iter = context.ast_hash_to_var.find(node->hash);
  if (var_iter != context.ast_hash_to_var.end()) {
    output += var_iter->second;
    return true;
  }
  const size_t expr_start = output.size();
  const bool child_replaced = AppendAstNode(node, context, output);
  if (context.ast_ref_count[node->hash] <= 1U || output.size() - expr_start < kCommonExprMinLen) {
    return child_replaced;
  }
  const std::string expr = output.substr(expr_start);
  const std::string var_name = context.prefix + "_common" + std::to_string(context.common_counter++);
  context.preamble += "  double " + var_name + " = " + expr + ";\n";
  context.ast_hash_to_var[node->hash] = var_name;
  output.resize(expr_start);
  output += var_name;
  return true;
}

BuiltAstExpr BuildAstExpr(const ASTPtr &node, DecomposeContext &context) {
  BuiltAstExpr result;
  result.replaced = AppendAstExpr(node, context, result.expr);
  return result;
}

std::string BuildExpr(const Expr &expr, DecomposeContext &context) {
  const std::string expr_str = Str(expr);
  if (expr_str.length() < kCommonExprMinLen) {
    return expr_str;
  }
  ASTPtr ast = ParseCompleteAst(expr_str, context);
  if (ast == nullptr) {
    return expr_str;
  }
  BuiltAstExpr built_expr = BuildAstExpr(ast, context);
  return built_expr.replaced ? built_expr.expr : expr_str;
}

std::string CondToStr(CondType type, const std::string &left, const std::string &right) {
  if (type == CondType::K_EQ) {
    return "IsEqual(" + left + ", " + right + ")";
  } else if (type == CondType::K_LT) {
    return left + " < " + right;
  } else if (type == CondType::K_GT) {
    return left + " > " + right;
  } else if (type == CondType::K_LE) {
    return left + " <= " + right;
  } else {
    return left + " >= " + right;
  }
}

// 尝试将字符串解析为数值：处理 "False"/"True" 和纯数字字符串
bool TryParseConstStr(const std::string &s, double &val) {
  if (s == "False") {
    val = 0.0;
    return true;
  }
  if (s == "True") {
    val = 1.0;
    return true;
  }
  try {
    size_t pos = 0;
    val = std::stod(s, &pos);
    return pos == s.size();  // 整个字符串都被解析
  } catch (...) {
    return false;
  }
}

// 若 cond 两侧均为常量，尝试静态求值；返回 true 并设置 result，否则返回 false
bool TryEvalConstCond(CondType type, const Expr &left, const Expr &right, bool &result) {
  double lv = 0.0;
  double rv = 0.0;
  // 注意：IsConstExpr() 对 Integer/RealDouble/Rational 均返回 true，
  // 但 GetConstValue(double) 内部断言只允许 RealDouble/Rational，对 Integer 会 assert。
  // 因此直接走字符串解析，TryParseConstStr 可以处理所有常量（含 "0", "1", "False", "True"）。
  const bool lv_ok = TryParseConstStr(Str(left), lv);
  const bool rv_ok = TryParseConstStr(Str(right), rv);
  if (!lv_ok || !rv_ok) {
    return false;
  }
  switch (type) {
    case CondType::K_EQ:
      result = std::fabs(lv - rv) < 1e-9;
      break;
    case CondType::K_LT:
      result = (lv < rv);
      break;
    case CondType::K_GT:
      result = (lv > rv);
      break;
    case CondType::K_LE:
      result = (lv <= rv);
      break;
    default:
      result = (lv >= rv);
      break;
  }
  return true;
}

// 递归分解 IfCase 树，每个非叶节点都提取为独立的 double 变量。
// 叶子节点表达式超过 kLeafExprMaxInlineLen 时也提取为命名 double 变量。
// is_root=true 时，返回完整 TernaryOp 表达式供调用者赋值给外层变量（不额外包一层）。
// is_root=false 时，生成一个 double 子变量，返回该变量名（避免嵌套字面量）。
std::string DecomposeIfCase(const IfCase &node, DecomposeContext &context, bool is_root = false) {
  if (node.IsLeaf()) {
    std::string leaf_expr = BuildExpr(node.GetExpr(), context);
    if (leaf_expr.length() <= kLeafExprMaxInlineLen) {
      return leaf_expr;
    }
    // 叶子表达式过长：提取为命名变量 _caseN
    std::string case_var = context.prefix + "_case" + std::to_string(context.node_counter++);
    context.preamble += "  double " + case_var + " = " + leaf_expr + ";\n";
    return case_var;
  }
  // 常量条件折叠：若 cond 两侧均为常量，直接取对应分支，不生成 bool/TernaryOp
  if (bool const_result = false;
      TryEvalConstCond(node.GetCondType(), node.GetCondLeft(), node.GetCondRight(), const_result)) {
    const IfCase &taken = const_result ? *node.GetChoiceA() : *node.GetChoiceB();
    return DecomposeIfCase(taken, context, is_root);
  }
  const std::string cond_name = context.prefix + "_cond" + std::to_string(context.node_counter++);
  const std::string cond_left = BuildExpr(node.GetCondLeft(), context);
  const std::string cond_right = BuildExpr(node.GetCondRight(), context);
  context.preamble += "  bool " + cond_name + " = " + CondToStr(node.GetCondType(), cond_left, cond_right) + ";\n";
  const std::string true_str = DecomposeIfCase(*node.GetChoiceA(), context);
  const std::string false_str = DecomposeIfCase(*node.GetChoiceB(), context);
  const std::string tenary_str = "TernaryOp(" + cond_name + ", " + true_str + ", " + false_str + ")";
  if (is_root) {
    return tenary_str;
  }
  // 非根节点：提取为命名 double 变量，避免嵌套字面量
  std::string sub_var = context.prefix + "_branch" + std::to_string(context.node_counter++);
  context.preamble += "  double " + sub_var + " = " + tenary_str + ";\n";
  return sub_var;
}
}  // namespace

void TernaryOp::DecomposeNamedVars(const std::string &var_prefix, std::string &preamble,
                                   std::string &tenary_expr) const {
  DecomposeContext context(var_prefix, preamble);
  CountIfCaseAstRefs(*ternary_op_, context);
  tenary_expr = DecomposeIfCase(*ternary_op_, context, true);
}
}  // namespace att
