/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ast_optimizer.h"

namespace att {
bool AstCanonicalizer::NodeKey::operator<(const NodeKey &other) const {
  if (type != other.type) {
    return type < other.type;
  }
  if (value != other.value) {
    return value < other.value;
  }
  return child_ids < other.child_ids;
}

std::string AstCanonicalizer::AssignNode(ASTNode *node) {
  if (node == nullptr) {
    return "";
  }
  NodeKey key{
      node->type, node->type == NodeType::VARIABLE || node->type == NodeType::NUMBER ? node->expr : node->op, {}};
  key.child_ids.reserve(node->children.size());
  for (const auto &child : node->children) {
    key.child_ids.emplace_back(AssignNode(child.get()));
  }
  auto iter = node_ids_.find(key);
  if (iter == node_ids_.end()) {
    iter = node_ids_.emplace(std::move(key), std::to_string(next_id_++)).first;
  }
  node->hash = iter->second;
  return node->hash;
}

void AstCanonicalizer::Assign(ASTNode *root) {
  (void)AssignNode(root);
}

// 表达式涉及的函数
const std::vector<std::string> functions_set = {"Ceiling", "Min", "Max", "Rational", "Floor",
                                                "Log",     "Pow", "Mod", "Abs"};

// 判断字符是否是数字或相关符号
bool IsNumberChar(char c) {
  return isdigit(c) || c == '.' || c == '/';
}

// 处理负数
void HandleNegativeNumber(const std::string &s, size_t &i, std::vector<std::string> &tokens) {
  std::string num;
  num += s[i++];
  while (i < s.size() && IsNumberChar(s[i])) {
    num += s[i++];
  }
  tokens.push_back(num);
}

// 处理非负数字
void HandleNumber(const std::string &s, size_t &i, std::vector<std::string> &tokens) {
  std::string num;
  while (i < s.size() && IsNumberChar(s[i])) {
    num += s[i++];
  }
  tokens.push_back(num);
}

// 处理变量
void HandleIdentifier(const std::string &s, size_t &i, std::vector<std::string> &tokens) {
  std::string token;
  while (i < s.size() && (isalnum(s[i]) || s[i] == '_')) {
    token += s[i++];
  }
  tokens.push_back(token);
}

// 词法分析器
std::vector<std::string> Parser::Tokenize(const std::string &s) const {
  std::vector<std::string> tokens;
  for (size_t i = 0; i < s.size();) {
    if (isspace(s[i])) {
      ++i;
      continue;
    }
    // 处理负数
    if ((s[i] == '-') && ((i == 0u) || tokens.empty() || (tokens.back() == "(") || (tokens.back() == ",") ||
                          (std::string("+-*/(").find(tokens.back()[0]) != std::string::npos))) {
      HandleNegativeNumber(s, i, tokens);
    }
    // 处理非负数字
    else if (IsNumberChar(s[i]) && s[i] != '/') {
      HandleNumber(s, i, tokens);
    }
    // 处理变量
    else if (isalpha(s[i]) || s[i] == '_') {
      HandleIdentifier(s, i, tokens);
    } else {
      tokens.push_back(std::string(1, s[i++]));
    }
  }
  return tokens;
}

ASTPtr Parser::ParseFunction(const std::string &func) {
  if (Peek(1) != "(") {
    return nullptr;
  }
  // consume两次，第一次是函数名，第二次是(
  Consume();
  Consume();
  std::vector<ASTPtr> args;
  while (Peek() != ")") {
    if (Peek().empty()) {
      return nullptr;
    }
    const size_t previous_pos = pos_;
    ASTPtr arg = ParseExpr();
    if (arg == nullptr || pos_ == previous_pos) {
      return nullptr;
    }
    args.push_back(std::move(arg));
    if (Peek() == ",") {
      Consume();
    } else if (Peek() != ")") {
      return nullptr;
    }
  }
  Consume();
  return std::make_shared<ASTNode>(func, NodeType::FUNCTION, func, std::move(args));
}

ASTPtr Parser::ParsePrimary() {
  std::string token = Peek();
  if (token.empty()) {
    return nullptr;
  }
  if (token == "(") {
    Consume();
    auto node = ParseExpr();
    if (Peek() != ")") {
      if (enable_log_) {
        GELOGD("error: expected ')', got '%s'", Peek().c_str());
      }
      return nullptr;
    }
    Consume();
    return node;
  }
  if (std::find(functions_set.begin(), functions_set.end(), token) != functions_set.end()) {
    return ParseFunction(token);
  }
  if ((token[0] == '-' && token.size() > 1u && (isdigit(token[1]) || token[1] == '.')) || isdigit(token[0]) ||
      token.find('.') != std::string::npos || token.find('/') != std::string::npos) {
    Consume();
    return std::make_shared<ASTNode>(token, NodeType::NUMBER);
  }
  if (isdigit(token[0]) || token.find('.') != std::string::npos || token.find('/') != std::string::npos) {
    Consume();
    return std::make_shared<ASTNode>(token, NodeType::NUMBER);
  }
  if (isalpha(token[0]) || token[0] == '_') {
    Consume();
    return std::make_shared<ASTNode>(token, NodeType::VARIABLE);
  }
  if (enable_log_) {
    GELOGD("error: invalid expression: '%s'", token.c_str());
  }
  return nullptr;
}

ASTPtr CreateBinaryOpNode(ASTPtr &&lhs, const std::string &op, ASTPtr &&rhs) {
  std::vector<ASTPtr> children;
  children.push_back(std::move(lhs));
  children.push_back(std::move(rhs));
  return std::make_shared<ASTNode>("", NodeType::OPERATOR, op, std::move(children));
}

ASTPtr Parser::ParseExpr() {
  ASTPtr lhs = ParseTerm();
  if (!lhs) {
    return nullptr;
  }
  // 处理不带括号的连加连减
  while ((Peek() == "+") || (Peek() == "-")) {
    std::string op = Peek();
    Consume();
    ASTPtr rhs = ParseTerm();
    if (!rhs) {
      return nullptr;
    }
    lhs = CreateBinaryOpNode(std::move(lhs), op, std::move(rhs));
  }
  return lhs;
}

ASTPtr Parser::ParseTerm() {
  ASTPtr lhs = ParsePrimary();
  if (!lhs) {
    return nullptr;
  }
  // 处理不带括号的连乘连除
  while ((Peek() == "*") || (Peek() == "/")) {
    std::string op = Peek();
    Consume();
    ASTPtr rhs = ParsePrimary();
    if (!rhs) {
      return nullptr;
    }
    lhs = CreateBinaryOpNode(std::move(lhs), op, std::move(rhs));
  }
  return lhs;
}

ASTPtr Parser::Parse() {
  tokens_ = Tokenize(expr_);
  if (!enable_log_) {
    return ParseExpr();
  }
  GELOGD("tokenize success, tokens are: ");
  std::string buf;
  for (auto &t : tokens_) {
    if (constexpr int32_t kMaxLen = 800; buf.size() + t.size() > kMaxLen) {
      GELOGD("%s", buf.c_str());
      buf.clear();
      continue;
    }
    buf.append(" ").append(t);
  }
  if (!buf.empty()) {
    GELOGD("%s", buf.c_str());
  }
  return ParseExpr();
}

// 处理操作符或函数节点
void ProcessOperatorOrFunction(ASTNode *node, std::unordered_map<std::string, std::string> &expr_map_,
                               std::vector<ASTNode> &temp_order_, int32_t &temp_count_) {
  auto it = expr_map_.find(node->hash);
  if (it != expr_map_.end()) {
    node->temp_var = it->second;  // 复用已有变量名
  } else {
    // 分配新变量名并记录
    node->temp_var = "temp" + std::to_string(temp_count_++);
    expr_map_[node->hash] = node->temp_var;
    temp_order_.push_back(*node);
  }
}

void Optimizer::Traverse(ASTNode *node) {
  if (!node) {
    return;
  }

  // 先递归处理所有子节点
  for (auto &c : node->children) {
    Traverse(c.get());
  }

  // 处理操作符或函数节点
  if (node->type == NodeType::OPERATOR || node->type == NodeType::FUNCTION) {
    ProcessOperatorOrFunction(node, expr_map_, temp_order_, temp_count_);
  }
}

std::string RebuildFunctionCall(const ASTNode &node, int iter,
                                std::function<std::string(const ASTNode &, int)> rebuild_expr) {
  std::stringstream ss;
  ss << node.op << "(";
  for (size_t i = 0; i < node.children.size(); ++i) {
    if (i > 0u) {
      ss << ",";
    }
    ss << rebuild_expr(*node.children[i].get(), iter + 1);
  }
  ss << ")";
  return ss.str();
}

std::string RebuildBinaryOperation(const ASTNode &node, int iter,
                                   std::function<std::string(const ASTNode &, int)> rebuild_expr) {
  if (node.children.size() != 2u) {
    return node.expr;
  }
  return "(" + rebuild_expr(*node.children[0].get(), iter + 1) + " " + node.op + " " +
         rebuild_expr(*node.children[1].get(), iter + 1) + ")";
}

std::string Optimizer::RebuildExpr(const ASTNode &node, int iter) {
  // 复用已有变量名
  if (!node.temp_var.empty() && (iter != 0)) {
    return node.temp_var;
  }
  auto rebuild_expr = [this](const ASTNode &n, int i) { return this->RebuildExpr(n, i); };
  switch (node.type) {
    case NodeType::FUNCTION:
      return RebuildFunctionCall(node, iter, rebuild_expr);
    case NodeType::OPERATOR:
      return RebuildBinaryOperation(node, iter, rebuild_expr);
    default:
      return node.expr;
  }
}
std::string Optimizer::GenerateCode(const std::string &indent) {
  std::stringstream ss;
  if (temp_order_.empty()) {
    return "";
  }
  for (const auto &node : temp_order_) {
    // 跳过已生成的节点
    if (visited_.find(node.hash) != visited_.end()) {
      continue;
    }
    ss << indent << "auto " << node.temp_var << " = " << RebuildExpr(node, 0) << ";\n";
    visited_.insert(node.hash);
  }
  return ss.str();
}

void Optimizer::Optimize(ASTPtr &root) {
  if (!root) {
    return;
  }
  canonicalizer_.Assign(root.get());
  Traverse(root.get());
}
}  // namespace att
