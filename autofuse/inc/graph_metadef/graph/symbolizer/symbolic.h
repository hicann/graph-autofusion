/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 新的头文件，等air仓头文件切换后，搬移内容到这里
#ifndef AF_GRAPH_SYMBOLIZER_SYMBOLIC_H_
#define AF_GRAPH_SYMBOLIZER_SYMBOLIC_H_

#include <vector>
#include <limits>
#include "graph/ge_error_codes.h"
#include "graph/types_af.h"
#include "graph/type_utils.h"
#include "graph/symbolizer/symbol_operator.h"
#include "graph/symbolizer/symbol_checker.h"
// 不允许添加symbolic_utils.h，后续symbolic.h需要开源

namespace af {
class Expression;
class ExpressionImpl;
class ShapeEnvAttr;
using ExpressionImplPtr = std::unique_ptr<ExpressionImpl>;
std::ostream &operator<<(std::ostream &os, const Expression &e);

enum class ExprType : uint32_t {
  kExprConstantInteger = 0,
  kExprConstantRealDouble = 1,
  kExprConstantRation = 2,
  kExprConstantBoolean = 3,
  // add const definition here
  kExprVariable = 100,
  // add variable definition here
  kExprOperation = 200,
  kExprOperationBoolean,
  // add operation definition here
  kExprNone = std::numeric_limits<uint32_t>::max()
};

enum class StrType : size_t {
  kStrCpp = 0,
  kStrExpr = 1,  // kStrExpr和kStrCpp只有在处理除法的时候有区别，例如Div(a,b): kStrExpr返回a/b，kStrCpp返回Rational(a,b)
  kStrEnd = 2,
};

class Expression {
 public:
  Expression();
  ~Expression();
  Expression(const Expression &other);
  Expression(Expression &&other) noexcept;
  Expression &operator=(const Expression &other);
  Expression &operator=(Expression &&other) noexcept;
  /**
   * @brief 获取表达式转换成字符串
   */
  std::unique_ptr<char[]> Str(const StrType type = StrType::kStrCpp) const;
  /**
   * @brief 将字符串转换成表达式，与Str接口匹配
   */
  static Expression Parse(const char *str);
  /**
   * @brief 序列化，将表达式转换成字符串
   */
  std::unique_ptr<char[]> Serialize() const;

  /**
   * @brief
   * 反序列化，与Serialize接口匹配，将字符串转换成表达式，同时会校验字符串格式是否为序列化接口序列化出的字符串，如果不是则会报错
   */
  static Expression Deserialize(const char *str);

  /**
   * @brief 获取表达式的参数，例如表达式 2*s0 + Pow(s1, 2)，函数返回[2*s0, Pow(s1, 2)]
   */
  std::vector<Expression> GetArgs();
  /**
   * @brief 获取表达式的类型
   */
  ExprType GetExprType() const;
  /**
   * @brief 是否是ConstExpr类型
   */
  bool IsConstExpr() const;

  /**
   * @brief 是否是Symbol类型
   */
  bool IsVariableExpr() const;

  /**
   * @brief 是否是Bool类型
   */
  bool IsBooleanExpr() const;

  /**
   * @brief 对当前表达式中的表达式进行替换，例如 y= x+2. y.replace({x, 2*x}) -> y = 2*x + 2
   *        注意当前symengine对div sub的表达式替换能力有缺失，需要用户自己保证，例如x/y*z->Replace({{x/y, m}})会替换失败
   * @param pair<Expr first, Epxr second> first为被替换的表达式，second为替换的表达式
   */
  Expression Replace(const std::vector<std::pair<Expression, Expression>> &replace_vars) const;

  /**
   * @brief 对当前表达式中的符号进行替换，如对于表达式expr = x + y，expr.subs({x:2, y:z+1}) -> y + z +
   * 1。于repalce比较功能较单一， 只能替换单一符号，无法处理复杂的表达式
   * @param subs_vars 待替换的符号列表，pair中first为被替换的表达式，second为替换的表达式
   * @return 替换后表达式
   */
  Expression Subs(const std::vector<std::pair<Expression, Expression>> &subs_vars) const;

  /**
   * @brief 对当前表达式进行化简。例如2+x+4 -> 6+x
   */
  Expression Simplify() const;

  /**
   * @brief 判断当前表达式字符串中是否含有表达式e的子字符串，例如max((x+2), (4*y)) 含有 x和y
   */
  bool ContainVar(const Expression &e) const;

  /**
   * @brief 获取表达式的分子和分母，例如表达式Rational(2, 3) * s1，分子numer=2*s1，分母denom=3
   */
  void AsNumerDenom(Expression &numer, Expression &denom) const;

  /**
   * @brief 判断两个Expr是否相等
   */
  bool operator==(const Expression &e) const;
  /**
   * @brief 判断一个expr与常量是否相等
   */
  template <typename T>
  typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type operator==(
      const T &e) const;

  /**
   * @brief 判断两个Expr是否不相等
   */
  bool operator!=(const Expression &e) const;

  /**
   * @brief 判断一个expr与常量是否不相等
   */
  template <typename T>
  typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type operator!=(
      const T &e) const;

  /**
   * @brief 获取表达式最基础的元素。例如x - (y * z)，返回{x, y, z}, 注意该接口没有依据字符去重
   */
  std::vector<Expression> FreeSymbols() const;

  /**
   * @brief 获取表达式的值
   */
  graphStatus GetResult(const std::vector<std::pair<Expression, Expression>> &vars_value, double &result) const;

  /**
   * @brief 判断表达式是否合法，成员变量impl_为null则不合法
   */
  bool IsValid() const;

  /**
   * @brief 返回一个Expression类对象的hash值，主要目的是用于将Expression对象作为map的key时使用
   */
  uint64_t Hash() const;

  /**
   * @brief 分别返回 -1, 0, 1 当 `this < e, this == e, this > e`.
   */
  int64_t Compare(const Expression &e) const;

  /**
   * @brief 获取常量的值，只有GetExprType为EXPR_CONSTANT时有效
   * @param value 常量的值
   * @return 成功返回true，失败返回false，失败时value的值无效
   */
  template <typename T>
  typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type GetConstValue(
      T &value) const;

  /**
   * @brief 获取表达式hint值
   * @param hint 获取表达式的hint值
   * @return 成功返回true，失败返回false，失败时value的值无效
   */
  template <typename T>
  typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type GetHint(
      T &hint) const {
    return ComputeHint(hint);
  }

  Expression operator+(const Expression &other) const;
  Expression operator-(const Expression &other) const;
  Expression operator*(const Expression &other) const;
  Expression operator/(const Expression &other) const;

  friend Expression sym::Add(const Expression &a, const Expression &b);
  friend Expression sym::Sub(const Expression &a, const Expression &b);
  friend Expression sym::Mul(const Expression &a, const Expression &b);
  friend Expression sym::Div(const Expression &a, const Expression &b);
  friend Expression sym::Max(const Expression &a, const Expression &b);
  friend Expression sym::Min(const Expression &a, const Expression &b);
  friend Expression sym::Pow(const Expression &base, const Expression &exp);
  friend Expression sym::Mod(const Expression &base, const Expression &exp);
  friend Expression sym::Abs(const Expression &a);
  friend Expression sym::Log(const Expression &a);  // 默认以E为底
  friend Expression sym::Log(const Expression &arg, const Expression &base);
  friend Expression sym::Coeff(const Expression &b, const Expression &x, const Expression &n);
  friend Expression sym::Rational(int32_t num, int32_t den);  // 分数
  friend Expression sym::Ceiling(const Expression &a);
  friend Expression sym::Floor(const Expression &arg);
  friend Expression sym::Align(const Expression &arg, uint32_t alignment);
  friend Expression sym::AlignWithPositiveInteger(const Expression &arg, uint32_t alignment);
  friend std::ostream &operator<<(std::ostream &os, const Expression &e);
  friend Expression sym::Eq(const Expression &a, const Expression &b);  // ==
  friend Expression sym::Ne(const Expression &a, const Expression &b);  // !=
  friend Expression sym::Ge(const Expression &a, const Expression &b);  // >=
  friend Expression sym::Gt(const Expression &a, const Expression &b);  // >
  friend Expression sym::Le(const Expression &a, const Expression &b);  // <=
  friend Expression sym::Lt(const Expression &a, const Expression &b);  // <
  friend Expression sym::Not(const Expression &a);                      // !
  friend Expression sym::Neg(const Expression &a);                      // 负号
  friend Expression sym::LogicalAnd(const std::vector<Expression> &a);
  friend Expression sym::LogicalOr(const std::vector<Expression> &a);
  friend class ShapeEnvAttr;

 protected:
  explicit Expression(ExpressionImplPtr &&e);
  template <typename T>
  typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type ComputeHint(
      T &hint) const;
  ExpressionImplPtr impl_;

 private:
  Expression CanonicalizeBoolExpr() const;
};

class Symbol : public Expression {
 public:
  // 拷贝构造、赋值、移动构造、移动赋值默认使用基类，需要保证Symbol类大小与Expression类大小一致
  /**
   * @brief 创建常量
   * @param value 常量的值
   * @param name 常量的名称，默认为空，内部不持有该指针
   */
  explicit Symbol(int32_t value, const char *name = "");
  explicit Symbol(int64_t value, const char *name = "");
  explicit Symbol(uint32_t value, const char *name = "");
  explicit Symbol(uint64_t value, const char *name = "");
  explicit Symbol(double value, const char *name = "");

  /**
   * @brief 创建变量
   * @param name 变量的名称
   */
  explicit Symbol(const char *name = "");

  /**
   * @brief 获取symbol的name，返回值是一个unique_ptr，需要用户自己释放
   */
  std::unique_ptr<char[]> GetName() const;
  friend class ShapeEnvAttr;

 private:
  explicit Symbol(ExpressionImplPtr &&e);
};

template <typename T>
typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type
Expression::operator==(const T &e) const {
  Symbol symbol(e);
  return (*this == symbol);
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type
Expression::operator!=(const T &e) const {
  Symbol symbol(e);
  return !(*this == symbol);
}

// 为了保证ABI兼容性，禁用虚函数，Symbol的大小必须和Expression类大小一致
static_assert(sizeof(Symbol) == sizeof(Expression),
              "The size of the subclass Symbol must be equal to the size of the base class Expression.");

// 目的是构建以Expression作为key值的map、set、与unordered_map
// todoo：后续考虑挪到symbolic_utils.h或symbolic_dict.h中
struct ExpressionHash {
  //! Returns the hashed value.
  uint64_t operator()(const Expression &k) const {
    return k.Hash();
  }
};
struct ExpressionKeyEq {
  //! Comparison Operator `==`
  bool operator()(const Expression &x, const Expression &y) const {
    return x == y;
  }
};
struct ExpressionKeyLess {
  //! true if `x < y`, false otherwise
  bool operator()(const Expression &x, const Expression &y) const {
    int64_t xh = x.Hash();
    int64_t yh = y.Hash();
    if (xh != yh) return xh < yh;
    if (x == y) return false;
    return x.Compare(y) == -1;
  }
};
}  // namespace af

namespace ge {
using Expression = af::Expression;
using Symbol = af::Symbol;
using ExpressionImpl = af::ExpressionImpl;
using ExpressionImplPtr = af::ExpressionImplPtr;
using ExprType = af::ExprType;
using StrType = af::StrType;
using ExpressionHash = af::ExpressionHash;
using ExpressionKeyEq = af::ExpressionKeyEq;
using ExpressionKeyLess = af::ExpressionKeyLess;
template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY inline TypeId GetTypeId<Expression>() {
  return reinterpret_cast<TypeId>(1024);
}
}  // namespace ge

#endif
