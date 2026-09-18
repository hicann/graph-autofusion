/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AF_GRAPH_SYMBOLIZER_SYMBOL_OPERATOR_H_
#define AF_GRAPH_SYMBOLIZER_SYMBOL_OPERATOR_H_

#include <string>
namespace af {
class Expression;
namespace sym {
Expression Add(const Expression &a, const Expression &b);
Expression Sub(const Expression &a, const Expression &b);
Expression Mul(const Expression &a, const Expression &b);
Expression Div(const Expression &a, const Expression &b);
Expression Max(const Expression &a, const Expression &b);
Expression Min(const Expression &a, const Expression &b);
Expression Pow(const Expression &base, const Expression &exp);
Expression Mod(const Expression &base, const Expression &exp);
Expression Abs(const Expression &a);
Expression Log(const Expression &a);  // 默认以E为底
Expression Log(const Expression &arg, const Expression &base);
Expression Coeff(const Expression &b, const Expression &x, const Expression &n);
Expression Rational(int32_t num, int32_t den);  // 分数
Expression Ceiling(const Expression &a);
Expression Align(const Expression &arg, uint32_t alignment);
Expression AlignWithPositiveInteger(const Expression &arg, uint32_t alignment);
Expression Floor(const Expression &arg);
Expression Eq(const Expression &a, const Expression &b);  // ==
Expression Ne(const Expression &a, const Expression &b);  // !=
Expression Ge(const Expression &a, const Expression &b);  // >=
Expression Gt(const Expression &a, const Expression &b);  // >
Expression Le(const Expression &a, const Expression &b);  // <=
Expression Lt(const Expression &a, const Expression &b);  // <
Expression Not(const Expression &a);                      // !
Expression Neg(const Expression &a);                      // 负号
Expression LogicalAnd(const std::vector<Expression> &a);
Expression LogicalOr(const std::vector<Expression> &a);
}  // namespace sym
}  // namespace af
#endif  // AF_GRAPH_SYMBOLIZER_SYMBOL_OPERATOR_H_
