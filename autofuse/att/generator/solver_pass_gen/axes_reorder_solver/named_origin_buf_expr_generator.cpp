/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "named_origin_buf_expr_generator.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <set>
#include <sstream>
#include <vector>

#include "common/checker.h"
#include "generator/preprocess/ast_optimizer.h"

namespace att {
namespace {
constexpr size_t kNamedExprInlineLengthLimit = 96U;

struct NamedExprCode {
  std::string preamble;
  std::string result_expr;
};

enum class NamedExprKind {
  kTensorSize,
  kQueSize,
  kTmpBufferSize,
  kCommonSize,
};

enum class MaterializeReason {
  kNone,
  kRoot,
  kQueueAlign,
  kTmpBufferMax,
  kRepeated,
  kLongExpr,
};

struct NamedExprContext {
  std::map<std::string, std::string> ast_hash_to_var;
  std::map<std::string, std::string> expr_string_to_var;
  std::map<std::string, std::string> source_var_to_named_var;
  std::map<std::string, std::string> source_var_to_base_name;
  std::map<std::string, std::string> ast_hash_to_base_name;
  std::map<std::string, std::string> ast_hash_to_preferred_var;
  std::map<std::string, std::string> var_to_base_name;
  std::map<std::string, NamedExprKind> var_kind;
  std::map<std::string, size_t> ast_hash_ref_count;
  std::set<std::string> declared_vars;
  std::set<std::string> used_names;
  std::map<std::string, size_t> prefix_indices;
  AstCanonicalizer ast_canonicalizer;
};

struct SemanticContainerInput {
  Expr arg;
  Expr container_expr;
  ASTPtr container_ast;
  std::string semantic_name;
  std::string preferred_var_name;
  bool has_container_name = false;
};

struct MaterializeInfo {
  NamedExprKind kind;
  MaterializeReason reason;
};

const char *KindToString(NamedExprKind kind) {
  switch (kind) {
    case NamedExprKind::kTensorSize:
      return "tensor_size";
    case NamedExprKind::kQueSize:
      return "que_size";
    case NamedExprKind::kTmpBufferSize:
      return "tmp_buffer_size";
    case NamedExprKind::kCommonSize:
    default:
      return "common_size";
  }
}

std::string KindPrefix(NamedExprKind kind) {
  return KindToString(kind);
}

const char *ReasonToString(MaterializeReason reason) {
  switch (reason) {
    case MaterializeReason::kRoot:
      return "root";
    case MaterializeReason::kQueueAlign:
      return "queue_align";
    case MaterializeReason::kTmpBufferMax:
      return "tmp_buffer_max";
    case MaterializeReason::kRepeated:
      return "repeated";
    case MaterializeReason::kLongExpr:
      return "long_expr";
    case MaterializeReason::kNone:
    default:
      return "none";
  }
}

bool ContainsToken(const std::string &name, const std::string &token) {
  return name.find(token) != std::string::npos;
}

std::string NormalizeName(const std::string &name);

bool IsSizeLikeName(const std::string &name) {
  return ContainsToken(name, "_size") || name == "size" || name.rfind("size_", 0U) == 0U;
}

std::string BuildSizeVarName(const std::string &base_name) {
  if (IsSizeLikeName(base_name)) {
    return base_name;
  }
  return base_name + "_size";
}

std::string BuildRootPreferredVarName(const std::string &semantic_name) {
  return BuildSizeVarName(NormalizeName(semantic_name));
}

std::string BuildQueuePreferredVarName(const std::string &base_name) {
  if (ContainsToken(base_name, "que_size") || ContainsToken(base_name, "queue_size")) {
    return base_name;
  }
  return base_name + "_que_size";
}

std::string BuildTmpBufferPreferredVarName(const std::string &base_name) {
  return BuildSizeVarName(base_name);
}

std::string BuildMaxPreferredVarName(const std::string &base_name) {
  constexpr const char *kMaxSuffix = "_max";
  const std::string suffix(kMaxSuffix);
  if (base_name.size() >= suffix.size() && base_name.rfind(suffix) == base_name.size() - suffix.size()) {
    return base_name;
  }
  return base_name + suffix;
}

std::string NormalizeName(const std::string &name) {
  std::string result;
  result.reserve(name.size());
  for (const char ch : name) {
    const auto c = static_cast<unsigned char>(ch);
    result.push_back((std::isalnum(c) || ch == '_') ? ch : '_');
  }
  if (result.empty() || (!std::isalpha(static_cast<unsigned char>(result[0])) && result[0] != '_')) {
    result = "expr_" + result;
  }
  return result;
}

std::string ToLower(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return name;
}

bool IsTmpBufferName(const std::string &name) {
  const std::string normalized = ToLower(name);
  if (normalized.size() >= 7U && normalized[0] == 'b' && normalized.rfind("_size") == normalized.size() - 5U) {
    return std::all_of(normalized.begin() + 1, normalized.end() - 5,
                       [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });
  }
  return ContainsToken(normalized, "tmp") || ContainsToken(normalized, "temp") || ContainsToken(normalized, "buffer");
}

bool IsPreferredTmpBufferName(const std::string &name) {
  const std::string normalized = ToLower(name);
  return ContainsToken(normalized, "tmp") || ContainsToken(normalized, "temp");
}

bool IsQueueSemanticName(const std::string &name) {
  std::string base_name = ToLower(name);
  constexpr size_t kSizeSuffixLength = 5U;
  if (base_name.size() > kSizeSuffixLength && base_name.rfind("_size") == base_name.size() - kSizeSuffixLength) {
    base_name = base_name.substr(0U, base_name.size() - kSizeSuffixLength);
  }
  return base_name.size() > 1U && base_name[0] == 'q' &&
         std::all_of(base_name.begin() + 1, base_name.end(),
                     [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });
}

bool ShouldUseSemanticBaseName(const std::string &name) {
  return IsQueueSemanticName(name) || IsTmpBufferName(name);
}

NamedExprKind ClassifySemanticName(const std::string &name, bool has_container_name) {
  const std::string normalized = ToLower(NormalizeName(name));
  if (IsTmpBufferName(normalized)) {
    return NamedExprKind::kTmpBufferSize;
  }
  if (has_container_name) {
    return NamedExprKind::kTensorSize;
  }
  if (ContainsToken(normalized, "que") || ContainsToken(normalized, "queue")) {
    return NamedExprKind::kQueSize;
  }
  if (ContainsToken(normalized, "tensor")) {
    return NamedExprKind::kTensorSize;
  }
  return NamedExprKind::kCommonSize;
}

std::string AllocateIndexedVarName(const std::string &prefix, NamedExprContext &ctx) {
  size_t &idx = ctx.prefix_indices[prefix];
  std::string name = prefix + "_" + std::to_string(idx++);
  while (ctx.used_names.find(name) != ctx.used_names.end()) {
    name = prefix + "_" + std::to_string(idx++);
  }
  ctx.used_names.insert(name);
  return name;
}

std::string AllocateVarName(NamedExprKind kind, const std::string &preferred_name, NamedExprContext &ctx) {
  if (preferred_name.empty()) {
    return AllocateIndexedVarName(KindPrefix(kind), ctx);
  }
  std::string name = NormalizeName(preferred_name);
  std::string candidate = name;
  size_t idx = 1U;
  while (ctx.used_names.find(candidate) != ctx.used_names.end()) {
    candidate = name + "_" + std::to_string(idx++);
  }
  ctx.used_names.insert(candidate);
  return candidate;
}

bool IsLeafNode(const ASTNode &node) {
  return node.type == NodeType::VARIABLE || node.type == NodeType::NUMBER;
}

bool IsCeilOrFloor(const ASTNode &node) {
  return node.type == NodeType::FUNCTION && (node.op == "Ceiling" || node.op == "Floor");
}

bool IsNumberLiteral(const ASTPtr &node, const std::string &value) {
  return node != nullptr && node->type == NodeType::NUMBER && node->expr == value;
}

bool IsAlignNumberLiteral(const ASTPtr &node) {
  return IsNumberLiteral(node, "32") || IsNumberLiteral(node, "64");
}

bool IsMaxNode(const ASTNode &node) {
  return node.type == NodeType::FUNCTION && node.op == "Max";
}

bool IsQueueSizeKind(NamedExprKind kind) {
  return kind == NamedExprKind::kTensorSize || kind == NamedExprKind::kQueSize || kind == NamedExprKind::kTmpBufferSize;
}

bool IsQueueAlignNode(const ASTNode &node) {
  if (node.type != NodeType::OPERATOR || node.op != "*" || node.children.size() != 2U) {
    return false;
  }
  const ASTPtr &left = node.children[0];
  const ASTPtr &right = node.children[1];
  return (IsAlignNumberLiteral(left) && right != nullptr && IsCeilOrFloor(*right)) ||
         (IsAlignNumberLiteral(right) && left != nullptr && IsCeilOrFloor(*left));
}

std::string JoinFunctionArgs(const ASTNode &node, const std::function<std::string(const ASTNode &)> &rebuild) {
  std::stringstream ss;
  ss << node.op << "(";
  for (size_t i = 0U; i < node.children.size(); ++i) {
    if (i > 0U) {
      ss << ",";
    }
    ss << rebuild(*node.children[i]);
  }
  ss << ")";
  return ss.str();
}

ASTPtr ParseExprToAst(const Expr &expr, const char *message) {
  const std::string expr_str = Str(expr);
  Parser parser(expr_str);
  ASTPtr ast = parser.Parse();
  GE_ASSERT_NOTNULL(ast, "%s: %s", message, expr_str.c_str());
  return ast;
}

void AddFreeSymbolsToUsedNames(const Expr &expr, NamedExprContext &ctx,
                               const std::set<std::string> &ignored_names = {}) {
  for (const auto &symbol : expr.FreeSymbols()) {
    const std::string symbol_name = Str(symbol);
    if (ignored_names.find(symbol_name) == ignored_names.end()) {
      ctx.used_names.insert(symbol_name);
    }
  }
}

void InitUsedNames(const Expr &expr, NamedExprContext &ctx, const std::set<std::string> &ignored_names = {}) {
  ctx.used_names.insert("ub_size");
  AddFreeSymbolsToUsedNames(expr, ctx, ignored_names);
}

void CountAstHashRefsImpl(const ASTPtr &node, NamedExprContext &ctx) {
  if (node == nullptr) {
    return;
  }
  if (!IsLeafNode(*node)) {
    ++ctx.ast_hash_ref_count[node->hash];
  }
  for (const auto &child : node->children) {
    CountAstHashRefsImpl(child, ctx);
  }
}

void CountAstHashRefs(const ASTPtr &node, NamedExprContext &ctx) {
  ctx.ast_canonicalizer.Assign(node.get());
  CountAstHashRefsImpl(node, ctx);
}

size_t GetAstRefCount(const ASTPtr &node, const NamedExprContext &ctx) {
  if (node == nullptr) {
    return 0U;
  }
  auto iter = ctx.ast_hash_ref_count.find(node->hash);
  return iter == ctx.ast_hash_ref_count.end() ? 0U : iter->second;
}

std::string GetVarBaseName(const std::string &var_name, const NamedExprContext &ctx) {
  auto iter = ctx.var_to_base_name.find(var_name);
  return iter == ctx.var_to_base_name.end() ? "" : iter->second;
}

class NamedExprPrinter {
 public:
  NamedExprCode Generate(const ASTPtr &root, NamedExprContext &ctx, const std::string &indent);
  NamedExprCode GenerateWithRootKind(const ASTPtr &root, NamedExprKind root_kind, NamedExprContext &ctx,
                                     const std::string &indent);

 private:
  struct BuildOptions {
    const NamedExprKind *root_kind = nullptr;
    bool force_materialize_root = false;
    bool is_root = false;
  };

  std::string BuildNode(const ASTPtr &node, NamedExprContext &ctx, const std::string &indent,
                        const BuildOptions &options, std::string &preamble);
  void BuildChildren(const ASTPtr &node, NamedExprContext &ctx, const std::string &indent, std::string &preamble);
  std::string BuildNodeExpr(const ASTNode &node, const NamedExprContext &ctx) const;
  std::string BuildInlineNodeExpr(const ASTNode &node, const NamedExprContext &ctx) const;
  std::string BuildLeafNode(const ASTPtr &node, NamedExprContext &ctx, const BuildOptions &options,
                            std::string &preamble, const std::string &indent);
  MaterializeReason GetMaterializeReason(const ASTPtr &node, const std::string &expr_string,
                                         const BuildOptions &options, const NamedExprContext &ctx) const;
  std::string GetNodeBaseName(const ASTPtr &node, const NamedExprContext &ctx) const;
  std::string GetTmpBufferBaseName(const std::vector<ASTPtr> &children, const NamedExprContext &ctx) const;
  std::string GetPreferredVarName(const ASTPtr &node, NamedExprKind kind, const NamedExprContext &ctx) const;
  bool HasTmpBufferChild(const ASTPtr &node, const NamedExprContext &ctx) const;
  std::string MaterializeNode(const ASTPtr &node, const MaterializeInfo &info, NamedExprContext &ctx,
                              const std::string &indent, std::string &preamble);
  NamedExprKind InferKind(const ASTNode &node, const NamedExprContext &ctx) const;
  NamedExprKind GetChildKind(const ASTPtr &child, const NamedExprContext &ctx) const;
};

NamedExprCode NamedExprPrinter::Generate(const ASTPtr &root, NamedExprContext &ctx, const std::string &indent) {
  NamedExprCode code;
  if (root == nullptr) {
    return code;
  }
  BuildOptions options;
  options.is_root = true;
  code.result_expr = BuildNode(root, ctx, indent, options, code.preamble);
  return code;
}

NamedExprCode NamedExprPrinter::GenerateWithRootKind(const ASTPtr &root, NamedExprKind root_kind, NamedExprContext &ctx,
                                                     const std::string &indent) {
  NamedExprCode code;
  if (root == nullptr) {
    return code;
  }
  BuildOptions options;
  options.root_kind = &root_kind;
  options.force_materialize_root = true;
  options.is_root = true;
  code.result_expr = BuildNode(root, ctx, indent, options, code.preamble);
  ctx.var_kind.emplace(code.result_expr, root_kind);
  return code;
}

std::string NamedExprPrinter::BuildLeafNode(const ASTPtr &node, NamedExprContext &ctx, const BuildOptions &options,
                                            std::string &preamble, const std::string &indent) {
  if (node->type == NodeType::VARIABLE) {
    auto iter = ctx.source_var_to_named_var.find(node->expr);
    if (iter != ctx.source_var_to_named_var.end()) {
      auto base_iter = ctx.source_var_to_base_name.find(node->expr);
      if (base_iter != ctx.source_var_to_base_name.end()) {
        ctx.var_to_base_name.emplace(iter->second, base_iter->second);
      }
      return iter->second;
    }
  }
  if (options.is_root && options.force_materialize_root && options.root_kind != nullptr) {
    return MaterializeNode(node, {*options.root_kind, MaterializeReason::kRoot}, ctx, indent, preamble);
  }
  return node->expr;
}

std::string NamedExprPrinter::BuildNode(const ASTPtr &node, NamedExprContext &ctx, const std::string &indent,
                                        const BuildOptions &options, std::string &preamble) {
  if (node == nullptr) {
    return "";
  }
  if (IsLeafNode(*node)) {
    return BuildLeafNode(node, ctx, options, preamble, indent);
  }
  BuildChildren(node, ctx, indent, preamble);
  auto hash_iter = ctx.ast_hash_to_var.find(node->hash);
  if (hash_iter != ctx.ast_hash_to_var.end()) {
    return hash_iter->second;
  }
  const std::string expr_string = BuildNodeExpr(*node, ctx);
  auto expr_iter = ctx.expr_string_to_var.find(expr_string);
  if (expr_iter != ctx.expr_string_to_var.end()) {
    ctx.ast_hash_to_var[node->hash] = expr_iter->second;
    const std::string base_name = GetVarBaseName(expr_iter->second, ctx);
    if (!base_name.empty()) {
      ctx.ast_hash_to_base_name[node->hash] = base_name;
    }
    return expr_iter->second;
  }
  const MaterializeReason reason = GetMaterializeReason(node, expr_string, options, ctx);
  if (reason == MaterializeReason::kNone) {
    return expr_string;
  }
  NamedExprKind kind = (options.is_root && options.root_kind != nullptr) ? *options.root_kind : InferKind(*node, ctx);
  if (IsCeilOrFloor(*node)) {
    for (const auto &child : node->children) {
      if (IsQueueSizeKind(GetChildKind(child, ctx))) {
        kind = NamedExprKind::kQueSize;
      }
    }
  }
  return MaterializeNode(node, {kind, reason}, ctx, indent, preamble);
}

void NamedExprPrinter::BuildChildren(const ASTPtr &node, NamedExprContext &ctx, const std::string &indent,
                                     std::string &preamble) {
  for (auto &child : node->children) {
    BuildOptions child_options;
    (void)BuildNode(child, ctx, indent, child_options, preamble);
  }
}

std::string NamedExprPrinter::BuildNodeExpr(const ASTNode &node, const NamedExprContext &ctx) const {
  if (node.type == NodeType::FUNCTION) {
    auto rebuild = [this, &ctx](const ASTNode &child) { return BuildInlineNodeExpr(child, ctx); };
    return JoinFunctionArgs(node, rebuild);
  }
  if (node.type == NodeType::OPERATOR && node.children.size() == 2U) {
    return "(" + BuildInlineNodeExpr(*node.children[0], ctx) + " " + node.op + " " +
           BuildInlineNodeExpr(*node.children[1], ctx) + ")";
  }
  return node.expr;
}

std::string NamedExprPrinter::BuildInlineNodeExpr(const ASTNode &node, const NamedExprContext &ctx) const {
  if (node.type == NodeType::VARIABLE) {
    auto source_iter = ctx.source_var_to_named_var.find(node.expr);
    if (source_iter != ctx.source_var_to_named_var.end()) {
      return source_iter->second;
    }
  }
  auto hash_iter = ctx.ast_hash_to_var.find(node.hash);
  if (hash_iter != ctx.ast_hash_to_var.end()) {
    return hash_iter->second;
  }
  return BuildNodeExpr(node, ctx);
}

MaterializeReason NamedExprPrinter::GetMaterializeReason(const ASTPtr &node, const std::string &expr_string,
                                                         const BuildOptions &options,
                                                         const NamedExprContext &ctx) const {
  if (options.is_root && options.force_materialize_root) {
    return MaterializeReason::kRoot;
  }
  const std::string base_name = GetNodeBaseName(node, ctx);
  if (IsQueueAlignNode(*node) && IsQueueSemanticName(base_name)) {
    return MaterializeReason::kQueueAlign;
  }
  if (IsMaxNode(*node) && HasTmpBufferChild(node, ctx)) {
    return MaterializeReason::kTmpBufferMax;
  }
  if (GetAstRefCount(node, ctx) > 1U) {
    return MaterializeReason::kRepeated;
  }
  return expr_string.size() > kNamedExprInlineLengthLimit ? MaterializeReason::kLongExpr : MaterializeReason::kNone;
}

std::string NamedExprPrinter::GetNodeBaseName(const ASTPtr &node, const NamedExprContext &ctx) const {
  if (node == nullptr) {
    return "";
  }
  if (node->type == NodeType::VARIABLE) {
    auto source_iter = ctx.source_var_to_base_name.find(node->expr);
    if (source_iter != ctx.source_var_to_base_name.end()) {
      return source_iter->second;
    }
  }
  auto hash_base_iter = ctx.ast_hash_to_base_name.find(node->hash);
  if (hash_base_iter != ctx.ast_hash_to_base_name.end()) {
    return hash_base_iter->second;
  }
  auto hash_var_iter = ctx.ast_hash_to_var.find(node->hash);
  if (hash_var_iter != ctx.ast_hash_to_var.end()) {
    return GetVarBaseName(hash_var_iter->second, ctx);
  }
  if (node->type != NodeType::FUNCTION && node->type != NodeType::OPERATOR) {
    return "";
  }
  std::set<std::string> child_base_names;
  for (const auto &child : node->children) {
    const std::string child_base_name = GetNodeBaseName(child, ctx);
    if (!child_base_name.empty()) {
      child_base_names.insert(child_base_name);
    }
  }
  if (child_base_names.size() == 1U) {
    return *child_base_names.begin();
  }
  return "";
}

bool NamedExprPrinter::HasTmpBufferChild(const ASTPtr &node, const NamedExprContext &ctx) const {
  if (node == nullptr) {
    return false;
  }
  for (const auto &child : node->children) {
    const std::string base_name = GetNodeBaseName(child, ctx);
    if (!base_name.empty() && IsTmpBufferName(base_name)) {
      return true;
    }
  }
  return false;
}

std::string NamedExprPrinter::GetTmpBufferBaseName(const std::vector<ASTPtr> &children,
                                                   const NamedExprContext &ctx) const {
  std::string fallback_base_name;
  for (const auto &child : children) {
    const std::string child_base_name = GetNodeBaseName(child, ctx);
    if (child_base_name.empty() || !IsTmpBufferName(child_base_name)) {
      continue;
    }
    if (IsPreferredTmpBufferName(child_base_name)) {
      return child_base_name;
    }
    if (fallback_base_name.empty()) {
      fallback_base_name = child_base_name;
    }
  }
  return fallback_base_name;
}

std::string NamedExprPrinter::GetPreferredVarName(const ASTPtr &node, NamedExprKind kind,
                                                  const NamedExprContext &ctx) const {
  auto preferred_iter = ctx.ast_hash_to_preferred_var.find(node->hash);
  if (preferred_iter != ctx.ast_hash_to_preferred_var.end()) {
    return preferred_iter->second;
  }
  std::string base_name = GetNodeBaseName(node, ctx);
  if (kind == NamedExprKind::kTmpBufferSize && IsMaxNode(*node)) {
    const std::string tmp_buffer_base_name = GetTmpBufferBaseName(node->children, ctx);
    if (!tmp_buffer_base_name.empty()) {
      base_name = tmp_buffer_base_name;
    }
  }
  if (base_name.empty()) {
    return "";
  }
  if (kind == NamedExprKind::kQueSize) {
    return BuildQueuePreferredVarName(base_name);
  }
  if (kind == NamedExprKind::kTmpBufferSize && IsMaxNode(*node)) {
    return BuildMaxPreferredVarName(BuildTmpBufferPreferredVarName(base_name));
  }
  if (kind == NamedExprKind::kTmpBufferSize) {
    return BuildTmpBufferPreferredVarName(base_name);
  }
  if (kind == NamedExprKind::kTensorSize) {
    return BuildSizeVarName(base_name);
  }
  return "";
}

std::string NamedExprPrinter::MaterializeNode(const ASTPtr &node, const MaterializeInfo &info, NamedExprContext &ctx,
                                              const std::string &indent, std::string &preamble) {
  const std::string expr_string = BuildNodeExpr(*node, ctx);
  auto expr_iter = ctx.expr_string_to_var.find(expr_string);
  if (expr_iter != ctx.expr_string_to_var.end()) {
    ctx.ast_hash_to_var[node->hash] = expr_iter->second;
    const std::string base_name = GetVarBaseName(expr_iter->second, ctx);
    if (!base_name.empty()) {
      ctx.ast_hash_to_base_name[node->hash] = base_name;
    }
    GELOGD("[DFX][GetUbSizeNamedExpr] reuse var[%s] kind[%s] base[%s] reason[%s] expr_len[%zu] ref_count[%zu]",
           expr_iter->second.c_str(), KindToString(info.kind), base_name.c_str(), ReasonToString(info.reason),
           expr_string.size(), GetAstRefCount(node, ctx));
    return expr_iter->second;
  }
  const std::string base_name = GetNodeBaseName(node, ctx);
  const std::string var_name = AllocateVarName(info.kind, GetPreferredVarName(node, info.kind, ctx), ctx);
  preamble += indent + "auto " + var_name + " = " + expr_string + ";\n";
  ctx.ast_hash_to_var[node->hash] = var_name;
  ctx.expr_string_to_var[expr_string] = var_name;
  ctx.declared_vars.insert(var_name);
  ctx.var_kind[var_name] = info.kind;
  if (!base_name.empty()) {
    ctx.ast_hash_to_base_name[node->hash] = base_name;
    ctx.var_to_base_name[var_name] = base_name;
  }
  GELOGD(
      "[DFX][GetUbSizeNamedExpr] materialize var[%s] kind[%s] base[%s] reason[%s] expr_len[%zu] "
      "ref_count[%zu] expr[%s]",
      var_name.c_str(), KindToString(info.kind), base_name.c_str(), ReasonToString(info.reason), expr_string.size(),
      GetAstRefCount(node, ctx), expr_string.c_str());
  return var_name;
}

NamedExprKind NamedExprPrinter::GetChildKind(const ASTPtr &child, const NamedExprContext &ctx) const {
  if (child == nullptr) {
    return NamedExprKind::kCommonSize;
  }
  if (child->type == NodeType::VARIABLE) {
    auto source_iter = ctx.source_var_to_named_var.find(child->expr);
    if (source_iter != ctx.source_var_to_named_var.end()) {
      auto kind_iter = ctx.var_kind.find(source_iter->second);
      return kind_iter == ctx.var_kind.end() ? NamedExprKind::kCommonSize : kind_iter->second;
    }
  }
  auto hash_iter = ctx.ast_hash_to_var.find(child->hash);
  if (hash_iter != ctx.ast_hash_to_var.end()) {
    auto kind_iter = ctx.var_kind.find(hash_iter->second);
    return kind_iter == ctx.var_kind.end() ? NamedExprKind::kCommonSize : kind_iter->second;
  }
  if (!IsLeafNode(*child)) {
    return InferKind(*child, ctx);
  }
  return NamedExprKind::kCommonSize;
}

NamedExprKind NamedExprPrinter::InferKind(const ASTNode &node, const NamedExprContext &ctx) const {
  std::set<NamedExprKind> child_kinds;
  for (const auto &child : node.children) {
    const NamedExprKind kind = GetChildKind(child, ctx);
    if (kind != NamedExprKind::kCommonSize) {
      child_kinds.insert(kind);
    }
  }
  if (IsQueueAlignNode(node)) {
    return NamedExprKind::kQueSize;
  }
  if (IsMaxNode(node) &&
      std::any_of(child_kinds.begin(), child_kinds.end(), [](NamedExprKind kind) { return IsQueueSizeKind(kind); })) {
    return NamedExprKind::kTmpBufferSize;
  }
  if (child_kinds.size() == 1U) {
    return *child_kinds.begin();
  }
  return NamedExprKind::kCommonSize;
}

void EmitSemanticContainerExpr(const SemanticContainerInput &input, NamedExprContext &ctx, NamedExprPrinter &printer,
                               std::string &preamble, const std::string &indent) {
  const NamedExprKind kind = ClassifySemanticName(input.semantic_name, input.has_container_name);
  if (!input.preferred_var_name.empty()) {
    ctx.ast_hash_to_preferred_var[input.container_ast->hash] = input.preferred_var_name;
  }
  NamedExprCode code = printer.GenerateWithRootKind(input.container_ast, kind, ctx, indent);
  preamble += code.preamble;
  const std::string source_var = Str(input.arg);
  const std::string normalized_name = NormalizeName(input.semantic_name);
  ctx.source_var_to_named_var[source_var] = code.result_expr;
  ctx.source_var_to_base_name[source_var] = normalized_name;
  ctx.ast_hash_to_var[input.container_ast->hash] = code.result_expr;
  ctx.ast_hash_to_base_name[input.container_ast->hash] = normalized_name;
  ctx.expr_string_to_var[Str(input.container_expr)] = code.result_expr;
  ctx.var_kind.emplace(code.result_expr, kind);
  ctx.var_to_base_name[code.result_expr] = normalized_name;
  GELOGD("[DFX][GetUbSizeNamedExpr] semantic source[%s] semantic[%s] preferred[%s] kind[%s] result[%s] expr[%s]",
         source_var.c_str(), input.semantic_name.c_str(), input.preferred_var_name.c_str(), KindToString(kind),
         code.result_expr.c_str(), Str(input.container_expr).c_str());
}
}  // namespace

std::pair<std::string, std::string> NamedOriginBufExprGenerator::Generate(const Expr &expr,
                                                                          const std::string &indent) const {
  NamedExprContext ctx;
  NamedExprPrinter printer;
  std::string preamble;
  std::vector<SemanticContainerInput> semantic_inputs;
  std::set<std::string> container_source_vars;
  const auto free_symbols = expr.FreeSymbols();
  GELOGD("[DFX][GetUbSizeNamedExpr] start free_symbols[%zu] expr[%s]", free_symbols.size(), Str(expr).c_str());
  for (const auto &arg : free_symbols) {
    if (container_expr_.find(arg) != container_expr_.end()) {
      container_source_vars.insert(Str(arg));
    }
  }
  InitUsedNames(expr, ctx, container_source_vars);

  for (const auto &arg : free_symbols) {
    auto iter = container_expr_.find(arg);
    if (iter == container_expr_.end()) {
      continue;
    }
    const auto name_iter = container_names_.find(arg);
    const bool has_name = name_iter != container_names_.end();
    const std::string semantic_name = has_name ? name_iter->second : Str(arg);
    AddFreeSymbolsToUsedNames(iter->second, ctx);
    ASTPtr container_ast = ParseExprToAst(iter->second, "Parse container expr failed");
    CountAstHashRefs(container_ast, ctx);
    const std::string normalized_name = NormalizeName(semantic_name);
    const std::string preferred_name =
        ShouldUseSemanticBaseName(normalized_name) ? BuildRootPreferredVarName(normalized_name) : "";
    semantic_inputs.push_back({arg, iter->second, container_ast, semantic_name, preferred_name, has_name});
  }
  GELOGD("[DFX][GetUbSizeNamedExpr] semantic_inputs[%zu] container_source_vars[%zu]", semantic_inputs.size(),
         container_source_vars.size());

  ASTPtr ast = ParseExprToAst(expr, "Parse UB expr failed");
  CountAstHashRefs(ast, ctx);

  for (const auto &input : semantic_inputs) {
    EmitSemanticContainerExpr(input, ctx, printer, preamble, indent);
  }

  NamedExprCode code = printer.Generate(ast, ctx, indent);
  preamble += code.preamble;
  GELOGD("[DFX][GetUbSizeNamedExpr] finish declared_vars[%zu] result[%s]", ctx.declared_vars.size(),
         code.result_expr.c_str());
  return std::make_pair(preamble, code.result_expr);
}
}  // namespace att
