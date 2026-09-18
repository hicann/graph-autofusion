/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <map>
#include <symengine/rational.h>
#include "graph/symbolizer/symbolic.h"
#include "attribute_group/attr_group_shape_env.h"
#include "expression_impl.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/math_util.h"
#include "common/checker.h"
#include "const_values.h"

namespace af {
namespace sym {
Expression Add(const Expression &a, const Expression &b) {
  return Expression(Add(a.impl_, b.impl_));
}

Expression Sub(const Expression &a, const Expression &b) {
  return Expression(Sub(a.impl_, b.impl_));
}

Expression Mul(const Expression &a, const Expression &b) {
  return Expression(Mul(a.impl_, b.impl_));
}

Expression Div(const Expression &a, const Expression &b) {
  return Expression(Div(a.impl_, b.impl_));
}

Expression Max(const Expression &a, const Expression &b) {
  return Expression(Max(a.impl_, b.impl_));
}

Expression Min(const Expression &a, const Expression &b) {
  return Expression(Min(a.impl_, b.impl_));
}

Expression Abs(const Expression &a) {
  return Expression(Abs(a.impl_));
}

Expression Pow(const Expression &base, const Expression &exp) {
  return Expression(Pow(base.impl_, exp.impl_));
}

Expression Mod(const Expression &base, const Expression &exp) {
  return Expression(Mod(base.impl_, exp.impl_));
}

Expression Log(const Expression &a) {
  return Expression(Log(a.impl_));
}

Expression Log(const Expression &arg, const Expression &base) {
  return Expression(Log(arg.impl_, base.impl_));
}

Expression Ceiling(const Expression &a) {
  return Expression(Ceiling(a.impl_));
}

Expression Floor(const Expression &arg) {
  return Expression(Floor(arg.impl_));
}

Expression Coeff(const Expression &b, const Expression &x, const Expression &n) {
  return Expression(Coeff(b.impl_, x.impl_, n.impl_));
}

Expression Rational(int32_t num, int32_t den) {
  auto left = ExpressionImpl::CreateExpressionImpl(num);
  auto right = ExpressionImpl::CreateExpressionImpl(den);
  return Expression(Rational(left, right));
}

Expression Align(const Expression &arg, uint32_t alignment) {
  if (alignment == 0U) {
    GELOGE(FAILED, "Alignment should be more than 0");
    return Expression(nullptr);
  }
  auto align = Symbol(alignment);
  return Mul(Ceiling(Div(arg, align)), align);
}

Expression AlignWithPositiveInteger(const Expression &arg, uint32_t alignment) {
  if (alignment == 0U) {
    GELOGE(FAILED, "Alignment should be more than 0");
    return Expression(nullptr);
  }
  auto align = Symbol(alignment);
  return Mul(Floor(Div(Add(arg, Sub(align, kSymbolOne)), align)), align);
}

Expression Eq(const Expression &a, const Expression &b) {
  return Expression(Eq(a.impl_, b.impl_));
}

Expression Ne(const Expression &a, const Expression &b) {
  return Expression(Ne(a.impl_, b.impl_));
}

Expression Ge(const Expression &a, const Expression &b) {
  return Expression(Le(b.impl_, a.impl_));
}

Expression Gt(const Expression &a, const Expression &b) {
  return Expression(Lt(b.impl_, a.impl_));
}

Expression Le(const Expression &a, const Expression &b) {
  return Expression(Le(a.impl_, b.impl_));
}

Expression Lt(const Expression &a, const Expression &b) {
  return Expression(Lt(a.impl_, b.impl_));
}

Expression Not(const Expression &a) {
  return Expression(Not(a.impl_));
}

Expression Neg(const Expression &a) {
  return Expression(Neg(a.impl_));
}

Expression LogicalAnd(const std::vector<Expression> &a) {
  std::vector<ExpressionImplPtr> impl_vec;
  for (auto s : a) {
    GE_ASSERT_NOTNULL(s.impl_);
    impl_vec.emplace_back(std::move(s.impl_));
  }
  return Expression(LogicalAnd(impl_vec));
}

Expression LogicalOr(const std::vector<Expression> &a) {
  std::vector<ExpressionImplPtr> impl_vec;
  for (auto s : a) {
    GE_ASSERT_NOTNULL(s.impl_);
    impl_vec.emplace_back(std::move(s.impl_));
  }
  return Expression(LogicalOr(impl_vec));
}
}  // namespace sym
}  // namespace af
