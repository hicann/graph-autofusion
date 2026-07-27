/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "util/ternary_op.h"

namespace att {
namespace {
size_t CountOccurrences(const std::string &text, const std::string &pattern) {
  size_t count = 0U;
  size_t pos = text.find(pattern);
  while (pos != std::string::npos) {
    ++count;
    pos = text.find(pattern, pos + pattern.size());
  }
  return count;
}
}  // namespace

class TernaryOpsUtilsUnitTest : public testing::Test {
 public:
  void SetUp() override {}
  void TearDown() override {}

  void SetupCommonTernaryOps(std::map<Expr, TernaryOp, ExprCmp> &ternary_ops, Expr &res1, Expr &res2, Expr &res3,
                             Expr &expr1, Expr &expr2, Expr &expr3, Expr &expr4, Expr &expr5) const {
    res1 = CreateExpr("res1");
    res2 = CreateExpr("res2");
    res3 = CreateExpr("res3");
    expr1 = CreateExpr("expr1");
    expr2 = CreateExpr("expr2");
    expr3 = CreateExpr("expr3");
    expr4 = CreateExpr("expr4");
    expr5 = CreateExpr("expr5");
    TernaryOp ternary_op1 = TernaryOp(res3 + res2);
    ternary_op1.SetVariable(res1);
    ternary_ops[res1] = ternary_op1;
    TernaryOp ternary_op2 = TernaryOp(CondType::K_EQ, expr2, res3, expr4, expr3);
    ternary_op2.SetVariable(res2);
    ternary_ops[res2] = ternary_op2;
    TernaryOp ternary_op3 = TernaryOp(CondType::K_LE, expr4, expr5, expr1, expr3);
    ternary_op3.SetVariable(res3);
    ternary_ops[res3] = ternary_op3;
  }
};

TEST_F(TernaryOpsUtilsUnitTest, TestConcursiveReplaceVars) {
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  Expr res1 = CreateExpr("res1");
  Expr res2 = CreateExpr("res2");
  Expr res3 = CreateExpr("res3");
  Expr expr1 = CreateExpr("expr1");
  Expr expr2 = CreateExpr("expr2");
  Expr expr3 = CreateExpr("expr3");
  TernaryOp ternary_op1 = TernaryOp(expr1 + res2);
  ternary_op1.SetVariable(res1);
  ternary_ops[res1] = ternary_op1;
  TernaryOp ternary_op2 = TernaryOp(CondType::K_EQ, expr2, CreateExpr(2), res3, expr3);
  ternary_op2.SetVariable(res2);
  ternary_ops[res2] = ternary_op2;
  TernaryOp ternary_op3 = TernaryOp(CreateExpr(3));
  ternary_op3.SetVariable(res3);
  ternary_ops[res3] = ternary_op3;
  auto res = ConcursiveReplaceVars(ternary_ops);
  EXPECT_TRUE(!res.empty());
  EXPECT_EQ(Str(res1.Replace(res)), "(TernaryOp(IsEqual(expr2, 2), 3, expr3) + expr1)");
  EXPECT_EQ(Str(res2.Replace(res)), "TernaryOp(IsEqual(expr2, 2), 3, expr3)");
  EXPECT_EQ(Str(res3.Replace(res)), "3");
}

TEST_F(TernaryOpsUtilsUnitTest, TestConcursiveReplaceVars2) {
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  Expr res1, res2, res3, expr1, expr2, expr3, expr4, expr5;
  SetupCommonTernaryOps(ternary_ops, res1, res2, res3, expr1, expr2, expr3, expr4, expr5);
  Expr res4 = CreateExpr("res4");
  TernaryOp ternary_op4 = TernaryOp(CondType::K_GE, expr4, expr5, expr1, expr3);
  ternary_op4.SetVariable(res4);
  ternary_ops[res4] = ternary_op4;
  auto res = ConcursiveReplaceVars(ternary_ops);
  EXPECT_TRUE(!res.empty());
  EXPECT_EQ(Str(res1.Replace(res)),
            "(TernaryOp(IsEqual(expr2, TernaryOp(expr4 <= expr5, expr1, expr3)), expr4, expr3) + TernaryOp(expr4 <= "
            "expr5, expr1, expr3))");
  EXPECT_EQ(Str(res2.Replace(res)), "TernaryOp(IsEqual(expr2, TernaryOp(expr4 <= expr5, expr1, expr3)), expr4, expr3)");
  EXPECT_EQ(Str(res3.Replace(res)), "TernaryOp(expr4 <= expr5, expr1, expr3)");
  EXPECT_EQ(Str(res4.Replace(res)), "TernaryOp(expr4 >= expr5, expr1, expr3)");
}

TEST_F(TernaryOpsUtilsUnitTest, TestConcursiveRelatedVars) {
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  Expr res1, res2, res3, expr1, expr2, expr3, expr4, expr5;
  SetupCommonTernaryOps(ternary_ops, res1, res2, res3, expr1, expr2, expr3, expr4, expr5);
  auto res = ConcursiveRelatedVars(ternary_ops);
  EXPECT_TRUE(!res.empty());
  EXPECT_EQ(GetVecString(res[res1]), "expr4,expr5,expr1,expr3,expr2,expr4,expr5,expr1,expr3,expr4,expr3,");
  EXPECT_EQ(GetVecString(res[res2]), "expr2,expr4,expr5,expr1,expr3,expr4,expr3,");
  EXPECT_EQ(GetVecString(res[res3]), "expr4,expr5,expr1,expr3,");
}

TEST_F(TernaryOpsUtilsUnitTest, TestUpdateReplaceVars) {
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  Expr rec = CreateExpr("rec");
  Expr res1 = CreateExpr("res1");
  Expr expr1 = CreateExpr("expr1");
  TernaryOp ternary_op1 = TernaryOp(expr1 + res1);
  std::vector<std::pair<Expr, Expr>> expr_map;
  expr_map.emplace_back(std::make_pair(res1, rec));
  ternary_op1.SetVariable(res1);
  ternary_ops[res1] = ternary_op1;
  ternary_ops[res1].UpdateRelatedVars(expr_map);
  auto related_maps = ternary_ops[res1].GetRelatedVars();
  EXPECT_TRUE(find(related_maps.begin(), related_maps.end(), rec) != related_maps.end());
}

TEST_F(TernaryOpsUtilsUnitTest, TestDecomposeNamedVarsExtractsRepeatedLeafSubexpression) {
  const Expr dim_r = CreateExpr("dim_r");
  const Expr dim_a = CreateExpr("dim_a");
  const Expr vector_block = af::sym::Rational(1, 8);
  const Expr repeated_expr = af::sym::Ceiling(af::sym::Ceiling(vector_block * dim_a) * vector_block);
  const Expr small_case = repeated_expr * dim_r + repeated_expr + CreateExpr(40);
  const Expr large_case = repeated_expr * (dim_r + CreateExpr(1)) + repeated_expr + CreateExpr(63);
  const TernaryOp ternary_op(CondType::K_LT, dim_r, CreateExpr(2), small_case, large_case);

  std::string preamble;
  std::string ternary_expr;
  ternary_op.DecomposeNamedVars("reduce_tree_cost", preamble, ternary_expr);

  const std::string generated_code = preamble + ternary_expr;
  const size_t common_decl_pos = generated_code.find("double reduce_tree_cost_common0 =");
  const size_t first_case_pos = generated_code.find("double reduce_tree_cost_case");
  ASSERT_NE(common_decl_pos, std::string::npos) << generated_code;
  ASSERT_NE(first_case_pos, std::string::npos) << generated_code;
  EXPECT_LT(common_decl_pos, first_case_pos) << generated_code;
  EXPECT_EQ(CountOccurrences(generated_code, "Ceiling((Ceiling"), 1U) << generated_code;
}

TEST_F(TernaryOpsUtilsUnitTest, TestDecomposeNamedVarsPreservesNestedFunctionSyntax) {
  const Expr dim_a = CreateExpr("dim_a");
  const Expr long_var = CreateExpr("very_long_dynamic_shape_expression_used_by_abs_function");
  const Expr repeated_expr = (dim_a + long_var) * CreateExpr(2);
  const Expr nested_expr = af::sym::Mod(af::sym::Abs(dim_a), CreateExpr(256));
  const Expr small_case = nested_expr + repeated_expr;
  const Expr large_case = nested_expr + repeated_expr + CreateExpr(1);
  const TernaryOp ternary_op(CondType::K_LT, dim_a, CreateExpr(2), small_case, large_case);

  std::string preamble;
  std::string ternary_expr;
  ternary_op.DecomposeNamedVars("abs_cost", preamble, ternary_expr);

  const std::string generated_code = preamble + ternary_expr;
  EXPECT_NE(generated_code.find("abs_cost_common"), std::string::npos) << generated_code;
  EXPECT_EQ(generated_code.find("Mod(Abs,"), std::string::npos) << generated_code;
  EXPECT_NE(generated_code.find("Mod(Abs(dim_a), 256)"), std::string::npos) << generated_code;
  EXPECT_EQ(CountOccurrences(generated_code, "Abs("), 2U) << generated_code;
  EXPECT_NE(generated_code.find("very_long_dynamic_shape_expression_used_by_abs_function"), std::string::npos)
      << generated_code;
}

TEST_F(TernaryOpsUtilsUnitTest, TestDecomposeNamedVarsHandlesLongLeftDeepExpression) {
  Expr long_sum = CreateExpr("long_term_0");
  for (size_t i = 1U; i < 256U; ++i) {
    long_sum = long_sum + CreateExpr(("long_term_" + std::to_string(i)).c_str());
  }
  const Expr dim_r = CreateExpr("dim_r");
  const Expr repeated_expr = af::sym::Ceiling(long_sum * af::sym::Rational(1, 8));
  const Expr small_case = repeated_expr * dim_r + repeated_expr;
  const Expr large_case = repeated_expr * (dim_r + CreateExpr(1)) + repeated_expr;
  const TernaryOp ternary_op(CondType::K_LT, dim_r, CreateExpr(2), small_case, large_case);

  std::string preamble;
  std::string ternary_expr;
  ternary_op.DecomposeNamedVars("long_cost", preamble, ternary_expr);

  const std::string generated_code = preamble + ternary_expr;
  EXPECT_NE(generated_code.find("double long_cost_common"), std::string::npos) << generated_code;
  EXPECT_NE(generated_code.find("long_term_255"), std::string::npos) << generated_code;
  EXPECT_NE(ternary_expr.find("TernaryOp"), std::string::npos) << generated_code;
}
}  // namespace att
