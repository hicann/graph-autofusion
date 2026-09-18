/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_EXPRESSION_EXPRESSION_IMPL_H_
#define GRAPH_EXPRESSION_EXPRESSION_IMPL_H_
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <symengine/basic.h>
#include <symengine/integer.h>
#include "graph/symbolizer/symbolic.h"
#include "graph_metadef/graph/debug/ge_util.h"

#define IF_NULL_RETURN_NULL(x) \
  if ((x) == nullptr) return nullptr

namespace af {
constexpr int32_t kSizeTwo = 2;

class ExpressionImpl;
using ExpressionImplPtr = std::unique_ptr<ExpressionImpl>;
using RelationalFunc = std::function<SymEngine::RCP<const SymEngine::Basic>(
    const SymEngine::RCP<const SymEngine::Basic> &, const SymEngine::RCP<const SymEngine::Basic> &)>;

enum class OperationType : size_t {
  kOpAdd = 0,
  kOpMax,
  kOpMin,
  kOpMul,
  kOpAbs,
  kOpPow,
  kOpMod,
  kOpLog,
  kOpCeil,
  kOpFloor,
  kOpEq,
  kOpNe,
  kOpLt,
  kOpLe,
  kOpLogicalAnd,
  kOpLogicalOr,
  kOpNone = std::numeric_limits<size_t>::max()
};

using SymEngineExprPtr = SymEngine::RCP<const SymEngine::Basic>;

class ExpressionImpl {
 public:
  // 目前只支持int32_t,int64_t,uint32_t,uint64_t,const string&,const SymEngineExprPtr&
  template <typename T>
  static std::unique_ptr<ExpressionImpl> CreateExpressionImpl(T value, const std::string &name = "") {
    return ComGraphMakeUnique<ExpressionImpl>(value, name);
  }
  ExpressionImpl() = default;
  ExpressionImpl(int32_t const_value, const std::string &name);
  ExpressionImpl(int64_t const_value, const std::string &name);
  ExpressionImpl(uint32_t const_value, const std::string &name);
  ExpressionImpl(uint64_t const_value, const std::string &name);
  ExpressionImpl(double const_value, const std::string &name);
  ExpressionImpl(bool const_value, const std::string &name);
  explicit ExpressionImpl(const std::string &name);
  ExpressionImpl(const SymEngineExprPtr &sym_expr, const std::string &name);

  static ExpressionImplPtr CreateExpressionImpl(const std::string &name);
  ~ExpressionImpl();

  std::string Str(const StrType type = StrType::kStrCpp) const;
  static ExpressionImplPtr Parse(const std::string &expr_str);
  static ExpressionImplPtr Deserialize(const std::string &expr_str);
  ExprType GetExprType() const;
  bool IsConstExpr() const;
  bool IsVariableExpr() const;
  bool IsBooleanExpr() const;
  ExpressionImplPtr Replace(const std::map<ExpressionImpl *, ExpressionImpl *> &replace_vars) const;
  ExpressionImplPtr Subs(const std::map<ExpressionImpl *, ExpressionImpl *> &subs_vars) const;
  void AsNumerDenom(ExpressionImplPtr &numer, ExpressionImplPtr &denom) const;

  ExpressionImplPtr Simplify() const;
  bool ContainVar(const ExpressionImpl *e) const;
  bool operator==(const ExpressionImpl &e) const;
  std::vector<ExpressionImplPtr> FreeSymbols() const;
  OperationType GetOpType() const;
  std::string GetName() const;
  bool GetResult(double &result) const;
  uint64_t Hash() const;
  int64_t Compare(const ExpressionImpl &e) const;

  bool GetConstValue(uint64_t &value) const;
  bool GetConstValue(uint32_t &value) const;
  bool GetConstValue(int32_t &value) const;
  bool GetConstValue(int64_t &value) const;
  bool GetConstValue(bool &value) const;
  bool GetConstValue(double &value) const;
  bool GetConstValue(float &value) const;

  // 该方法不需要new一个ExpressionImpl对象(带来大量的指针校验)就能调用ExpressionImpl的方法
  // ***使用时需注意：1.返回的引用使用时，sym_expr对象必须存在；
  // ***使用时需注意：2.ExpressionImpl类只有一个SymEngineExprPtr类型的私有变量
  static const ExpressionImpl &SymExprToExpressionImplRef(const SymEngineExprPtr &sym_expr) {
    return *reinterpret_cast<const ExpressionImpl *>(&sym_expr);
  }

  /**
   * @brief bool表达式进行标准化处理，处理逻辑
          expra为表达式1，exprb为表达式2，OP是比较关系，支持四种表达式Eq、Ne、Lt、Le。Gt和Ge表达式由Lt、Le来进行替换
          1、原始表达式：expra OP exprb
          2、构造新的参数，右值：exprb - expra , 左值 = 0
          3、参数处理
          3.1、最大公约数化简，遍历右值的所有参数，计算最大公约数，如果gcd大于1则所有参数都除最大公约数
          3.1、区分正数和负数，遍历右值的所有参数，正数的集合相加作为新的右值，负数的集合相加作为新的左值
          4、通过新的左值和右值构造原始的布尔表达式类型
   */
  ExpressionImplPtr CanonicalizeBoolExpr();

  std::vector<ExpressionImplPtr> GetArgs();

 private:
  double GetIntegerResult(const SymEngine::Integer &integer_expr) const;

  friend ExpressionImplPtr Add(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Sub(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Mul(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Div(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Max(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Min(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Abs(const ExpressionImplPtr &a);
  friend ExpressionImplPtr Pow(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Mod(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Log(const ExpressionImplPtr &a);  // 默认以E为底
  friend ExpressionImplPtr Log(const ExpressionImplPtr &arg, const ExpressionImplPtr &base);
  friend ExpressionImplPtr Coeff(const ExpressionImplPtr &b, const ExpressionImplPtr &x, const ExpressionImplPtr &n);
  friend ExpressionImplPtr Ceiling(const ExpressionImplPtr &a);
  friend ExpressionImplPtr Floor(const ExpressionImplPtr &a);
  friend ExpressionImplPtr Rational(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Eq(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Ne(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Lt(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Le(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
  friend ExpressionImplPtr Not(const ExpressionImplPtr &a);
  friend ExpressionImplPtr Neg(const ExpressionImplPtr &a);
  friend ExpressionImplPtr LogicalAnd(const std::vector<ExpressionImplPtr> &s);
  friend ExpressionImplPtr LogicalOr(const std::vector<ExpressionImplPtr> &s);
  // friend std::string DefaultPowPrinter(const std::vector<ExpressionImplPtr> &args);
  friend class Parser;

 private:
  SymEngineExprPtr sym_expr_;  // 非空,symengine在内存不够时会抛异常
  mutable std::string name_;
};

ExpressionImplPtr Add(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Sub(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Mul(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Div(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Max(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Min(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Abs(const ExpressionImplPtr &a);
ExpressionImplPtr Pow(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Mod(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Log(const ExpressionImplPtr &a);  // 默认以E为底
ExpressionImplPtr Log(const ExpressionImplPtr &arg, const ExpressionImplPtr &base);
ExpressionImplPtr Coeff(const ExpressionImplPtr &b, const ExpressionImplPtr &x, const ExpressionImplPtr &n);
ExpressionImplPtr Ceiling(const ExpressionImplPtr &a);
ExpressionImplPtr Floor(const ExpressionImplPtr &a);
ExpressionImplPtr Rational(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
std::ostream &operator<<(std::ostream &os, const ExpressionImpl &e);
ExpressionImplPtr Eq(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Ne(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Lt(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Le(const ExpressionImplPtr &a, const ExpressionImplPtr &b);
ExpressionImplPtr Not(const ExpressionImplPtr &a);
ExpressionImplPtr Neg(const ExpressionImplPtr &a);
ExpressionImplPtr LogicalAnd(const std::vector<ExpressionImplPtr> &s);
ExpressionImplPtr LogicalOr(const std::vector<ExpressionImplPtr> &s);
}  // namespace af

#endif  // GRAPH_EXPRESSION_EXPRESSION_IMPL_H_
