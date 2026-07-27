/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_CODE_GEN_PREPROCESS_AST_OPTIMIZER_H_
#define ATT_CODE_GEN_PREPROCESS_AST_OPTIMIZER_H_
#include <iostream>
#include <map>
#include <vector>
#include <unordered_map>
#include <memory>
#include <sstream>
#include <cmath>
#include <cctype>
#include <fstream>
#include <set>
#include <functional>
#include "framework/common/debug/ge_log.h"

namespace att {
// AST节点类型
enum class NodeType : uint8_t { OPERATOR, FUNCTION, VARIABLE, NUMBER };

// AST节点数据结构
struct ASTNode {
  std::string expr;
  NodeType type;
  std::string op;
  std::vector<std::shared_ptr<ASTNode>> children;
  std::string hash;
  std::string temp_var;

  ASTNode(std::string e, NodeType t, std::string o = "", std::vector<std::shared_ptr<ASTNode>> &&c = {})
      : expr(std::move(e)), type(t), op(std::move(o)), children(std::move(c)) {}
};

using ASTPtr = std::shared_ptr<ASTNode>;

class AstCanonicalizer {
 public:
  void Assign(ASTNode *root);

 private:
  struct NodeKey {
    NodeType type;
    std::string value;
    std::vector<std::string> child_ids;

    bool operator<(const NodeKey &other) const;
  };

  std::string AssignNode(ASTNode *node);

  std::map<NodeKey, std::string> node_ids_;
  size_t next_id_ = 0U;
};

// AST解析模块，功能包含词法分析和语法分析，最终生成AST
class Parser {
 public:
  explicit Parser(const std::string &e, bool enable_log = true) : expr_(e), enable_log_(enable_log) {}
  ~Parser() = default;
  ASTPtr Parse();
  bool IsFullyParsed() const {
    return pos_ == tokens_.size();
  }

 private:
  std::string Peek(size_t offset = 0) {
    return (pos_ + offset) < tokens_.size() ? tokens_[pos_ + offset] : "";
  }
  void Consume() {
    ++pos_;
  }
  std::vector<std::string> Tokenize(const std::string &s) const;
  ASTPtr ParseFunction(const std::string &func);
  ASTPtr ParsePrimary();
  ASTPtr ParseTerm();
  ASTPtr ParseExpr();

  std::string expr_;
  std::vector<std::string> tokens_;
  size_t pos_ = 0;
  bool enable_log_ = true;
};

// AST优化器，功能为遍历语法树，提取公共子表达式，表达为临时变量，返回优化后的表达式
class Optimizer {
 public:
  Optimizer() = default;
  ~Optimizer() = default;
  std::string GenerateCode(const std::string &indent = "    ");
  void Optimize(ASTPtr &root);
  std::string RebuildExpr(const ASTNode &node, int iter);

 private:
  void Traverse(ASTNode *node);
  std::unordered_map<std::string, std::string> expr_map_;
  std::vector<ASTNode> temp_order_;
  std::set<std::string> visited_;
  int32_t temp_count_ = 0;
  AstCanonicalizer canonicalizer_;
};

// AST可视化模块
class ASTVisualizer {
 public:
  void InitDotFile(const std::string &filename) {
    dot_file_.open(filename + ".dot");
    dot_file_ << "digraph AST {\n";
    dot_file_ << "node [shape=box, fontname=\"Courier\"];\n";
  }
  void GenerateDotImage(const std::string &filename) {
    dot_file_ << "}\n";
    dot_file_.close();
    system(("dot -Tpng " + filename + ".dot -o " + filename + ".png").c_str());
  }

  void Visualize(ASTPtr &root, const std::string &filename = "ast") {
    if (!root) {
      return;
    }
    InitDotFile(filename);
    Traverse(root.get());
    GenerateDotImage(filename);
  }

 private:
  std::ofstream dot_file_;
  std::unordered_map<ASTNode *, std::string> node_ids_;
  uint32_t node_counter_ = 0u;

  std::string GenerateNodeId() {
    return "node_" + std::to_string(node_counter_++);
  }

  std::string GetNodeId(ASTNode *node) {
    if (!node) {
      return "null_node";
    }
    if (node_ids_.find(node) == node_ids_.end()) {
      node_ids_[node] = GenerateNodeId();
    }
    return node_ids_[node];
  }

  std::string GetNodeLabel(const ASTNode *node) const {
    std::string label;
    switch (node->type) {
      case NodeType::OPERATOR:
        label = node->op;
        break;
      case NodeType::FUNCTION:
        label = node->op + "()";
        break;
      case NodeType::VARIABLE:
      case NodeType::NUMBER:
        label = node->expr;
        break;
      default:
        label = "unknown";
    }
    if (!node->temp_var.empty()) {
      label = node->temp_var + " = " + label;
    }
    return label;
  }

  std::string GetNodeColor(const ASTNode *node) const {
    switch (node->type) {
      case NodeType::OPERATOR:
        return "lightblue";
      case NodeType::FUNCTION:
        return "orange";
      case NodeType::VARIABLE:
        return "green";
      case NodeType::NUMBER:
        return "yellow";
      default:
        return "gray";
    }
  }

  void AddNode(ASTNode *node) {
    if (!node) {
      return;
    }
    std::string node_id = GetNodeId(node);
    std::string label = GetNodeLabel(node);
    std::string color = GetNodeColor(node);
    dot_file_ << node_id << " [label=\"" << label << "\", style=filled, color=" << color << "];\n";
  }

  void AddNodeAndEdges(ASTNode *node) {
    if (!node) {
      return;
    }
    AddNode(node);
    for (auto &child : node->children) {
      if (child) {
        dot_file_ << GetNodeId(node) << " -> " << GetNodeId(child.get()) << ";\n";
      }
    }
  }

  void Traverse(ASTNode *node) {
    if (!node) {
      return;
    }
    AddNodeAndEdges(node);
    for (auto &child : node->children) {
      Traverse(child.get());
    }
  }
};
}  // namespace att
#endif
