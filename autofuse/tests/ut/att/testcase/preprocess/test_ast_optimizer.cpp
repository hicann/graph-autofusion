/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <fstream>
#include "base/base_types.h"
#include "base/model_info.h"
#include "stub_model_info.h"
#include "common/util/mem_utils.h"
#define private public
#define protected public
#include "generator/preprocess/ast_optimizer.h"

using namespace att;

namespace {
std::string BuildLongSum(size_t term_count) {
  std::string expr = "term_0";
  for (size_t i = 1U; i < term_count; ++i) {
    expr += "+term_" + std::to_string(i);
  }
  return expr;
}

void CountNodesAndHashBytes(const ASTPtr &node, size_t &node_count, size_t &hash_bytes) {
  if (node == nullptr) {
    return;
  }
  ++node_count;
  hash_bytes += node->hash.size();
  for (const auto &child : node->children) {
    CountNodesAndHashBytes(child, node_count, hash_bytes);
  }
}

const std::string kExpr3Str =
    "((32 * Ceiling((((104 * z0t_size) + -1) * Rational(1 , 4)))) + (32 * Ceiling((((104 * z0t_size) + -3) * "
    "Rational(1 , 4)))) + (32 * Ceiling((((104 * z0t_size) + -7) * Rational(1 , 4)))) + (32 * Ceiling((((112 * "
    "z0t_size) + -3) * Rational(1 , 4)))) + (32 * Ceiling((((112 * z0t_size) + -5) * Rational(1 , 4)))) + (32 * "
    "Ceiling((((120 * z0t_size) + -7) * Rational(1 , 4)))) + (32 * Ceiling((((128 * z0t_size) + -1) * Rational(1 , "
    "4)))) + (32 * Ceiling((((136 * z0t_size) + -5) * Rational(1 , 4)))) + (32 * Ceiling((((144 * z0t_size) + -5) * "
    "Rational(1 , 4)))) + (32 * Ceiling((((144 * z0t_size) + -7) * Rational(1 , 4)))) + (32 * Ceiling((((152 * "
    "z0t_size) + -1) * Rational(1 , 4)))) + (32 * Ceiling((((152 * z0t_size) + -3) * Rational(1 , 4)))) + (32 * "
    "Ceiling((((16 * z0t_size) + -3) * Rational(1 , 4)))) + (32 * Ceiling((((16 * z0t_size) + -5) * Rational(1 , "
    "4)))) + (32 * Ceiling((((160 * z0t_size) + -3) * Rational(1 , 4)))) + (32 * Ceiling((((168 * z0t_size) + -1) * "
    "Rational(1 , 4)))) + (32 * Ceiling((((168 * z0t_size) + -5) * Rational(1 , 4)))) + (32 * Ceiling((((176 * "
    "z0t_size) + -3) * Rational(1 , 4)))) + (32 * Ceiling((((184 * z0t_size) + -3) * Rational(1 , 4)))) + (32 * "
    "Ceiling((((184 * z0t_size) + -5) * Rational(1 , 4)))) + (32 * Ceiling((((192 * z0t_size) + -1) * Rational(1 , "
    "4)))) + (32 * Ceiling((((200 * z0t_size) + -1) * Rational(1 , 4)))) + (32 * Ceiling((((200 * z0t_size) + -3) * "
    "Rational(1 , 4)))) + (32 * Ceiling((((200 * z0t_size) + -7) * Rational(1 , 4)))) + (32 * Ceiling((((216 * "
    "z0t_size) + -5) * Rational(1 , 4)))) + (32 * Ceiling((((224 * z0t_size) + -1) * Rational(1 , 4)))) + (32 * "
    "Ceiling((((232 * z0t_size) + -3) * Rational(1 , 4)))) + (32 * Ceiling((((232 * z0t_size) + -5) * Rational(1 , "
    "4)))) + (32 * Ceiling((((24 * z0t_size) + -1) * Rational(1 , 4)))) + (32 * Ceiling((((24 * z0t_size) + -5) * "
    "Rational(1 , 4)))) + (32 * Ceiling((((24 * z0t_size) + -7) * Rational(1 , 4)))) + (32 * Ceiling((((240 * "
    "z0t_size) + -1) * Rational(1 , 4)))) + (32 * Ceiling((((240 * z0t_size) + -7) * Rational(1 , 4)))) + (32 * "
    "Ceiling((((248 * z0t_size) + -7) * Rational(1 , 4)))) + (32 * Ceiling((((256 * z0t_size) + -5) * Rational(1 , "
    "4)))) + (32 * Ceiling((((264 * z0t_size) + -1) * Rational(1 , 4)))) + (32 * Ceiling((((264 * z0t_size) + -7) * "
    "Rational(1 , 4)))) + (32 * Ceiling((((272 * z0t_size) + -1) * Rational(1 , 4)))) + (32 * Ceiling((((272 * "
    "z0t_size) + -3) * Rational(1 , 4)))) + (32 * Ceiling((((280 * z0t_size) + -3) * Rational(1 , 4)))) + (32 * "
    "Ceiling((((288 * z0t_size) + -7) * Rational(1 , 4)))) + (32 * Ceiling((((32 * z0t_size) + -1) * Rational(1 , "
    "4)))) + (32 * Ceiling((((32 * z0t_size) + -3) * Rational(1 , 4)))) + (32 * Ceiling((((40 * z0t_size) + -3) * "
    "Rational(1 , 4)))) + (32 * Ceiling((((48 * z0t_size) + -1) * Rational(1 , 4)))) + (32 * Ceiling((((48 * "
    "z0t_size) + -5) * Rational(1 , 4)))) + (32 * Ceiling((((48 * z0t_size) + -7) * Rational(1 , 4)))) + (32 * "
    "Ceiling((((56 * z0t_size) + -3) * Rational(1 , 4)))) + (32 * Ceiling((((64 * z0t_size) + -3) * Rational(1 , "
    "4)))) + (32 * Ceiling((((64 * z0t_size) + -5) * Rational(1 , 4)))) + (32 * Ceiling((((72 * z0t_size) + -1) * "
    "Rational(1 , 4)))) + (32 * Ceiling((((72 * z0t_size) + -5) * Rational(1 , 4)))) + (32 * Ceiling((((7704 * "
    "z0t_size) + -5) * Rational(1 , 2)))) + (32 * Ceiling((((8 * z0t_size) + -1) * Rational(1 , 4)))) + (32 * "
    "Ceiling((((8 * z0t_size) + -3) * Rational(1 , 4)))) + (32 * Ceiling((((8 * z0t_size) + -5) * Rational(1 , 4)))) "
    "+ (32 * Ceiling((((8 * z0t_size) + -6) * Rational(1 , 4)))) + (32 * Ceiling((((80 * z0t_size) + -1) * "
    "Rational(1 , 4)))) + (32 * Ceiling((((80 * z0t_size) + -7) * Rational(1 , 4)))) + (32 * Ceiling((((88 * "
    "z0t_size) + -5) * Rational(1 , 4)))) + (32 * Ceiling((((96 * z0t_size) + -7) * Rational(1 , 4)))) + (32 * "
    "Ceiling((Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max("
    "Max(Max(Max(Max((((24 * z0t_size) + -1) * 8), (((80 * z0t_size) + -7) * 8)), (((72 * z0t_size) + -1) * 8)), "
    "(((88 * z0t_size) + -5) * 8)), (((200 * z0t_size) + -3) * 8)), (((104 * z0t_size) + -1) * 8)), (((152 * "
    "z0t_size) + -1) * 8)), (((112 * z0t_size) + -3) * 8)), (((104 * z0t_size) + -7) * 8)), (((8 * z0t_size) + -3) * "
    "8)), (((24 * z0t_size) + -7) * 8)), (((16 * z0t_size) + -5) * 8)), (((8 * z0t_size) + -6) * 8)), (((216 * "
    "z0t_size) + -5) * 8)), (((136 * z0t_size) + -5) * 8)), (((144 * z0t_size) + -5) * 8)), (((56 * z0t_size) + -3) "
    "* 8)), (((64 * z0t_size) + -3) * 8)), (((40 * z0t_size) + -3) * 8)), (((48 * z0t_size) + -5) * 8)), (((256 * "
    "z0t_size) + -5) * 8)), (((232 * z0t_size) + -5) * 8)), (((288 * z0t_size) + -7) * 8)), (((240 * z0t_size) + -1) "
    "* 8)), (((264 * z0t_size) + -1) * 8)), (((272 * z0t_size) + -1) * 8)), (((184 * z0t_size) + -5) * 8)), (((176 * "
    "z0t_size) + -3) * 8)), (((192 * z0t_size) + -1) * 8)), (((168 * z0t_size) + -5) * 8)) * Rational(1 , 16)))) + "
    "(32 * "
    "Ceiling((Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max(Max("
    "Max(Max(Max(Max((((72 * z0t_size) + -5) * 8), (((80 * z0t_size) + -1) * 8)), (((8 * z0t_size) + -1) * 8)), "
    "(((24 * z0t_size) + -5) * 8)), (((32 * z0t_size) + -3) * 8)), (((96 * z0t_size) + -7) * 8)), (((48 * z0t_size) "
    "+ -1) * 8)), (((224 * z0t_size) + -1) * 8)), (((200 * z0t_size) + -7) * 8)), (((104 * z0t_size) + -3) * 8)), "
    "(((128 * z0t_size) + -1) * 8)), (((120 * z0t_size) + -7) * 8)), (((112 * z0t_size) + -5) * 8)), (((144 * "
    "z0t_size) + -7) * 8)), (((152 * z0t_size) + -3) * 8)), (((160 * z0t_size) + -3) * 8)), (((8 * z0t_size) + -5) * "
    "8)), (((32 * z0t_size) + -1) * 8)), (((16 * z0t_size) + -3) * 8)), (((200 * z0t_size) + -1) * 8)), (((48 * "
    "z0t_size) + -7) * 8)), (((64 * z0t_size) + -5) * 8)), (((240 * z0t_size) + -7) * 8)), (((248 * z0t_size) + -7) "
    "* 8)), (((232 * z0t_size) + -3) * 8)), (((280 * z0t_size) + -3) * 8)), (((272 * z0t_size) + -3) * 8)), (((264 * "
    "z0t_size) + -7) * 8)), (((184 * z0t_size) + -3) * 8)), (((168 * z0t_size) + -1) * 8)) * Rational(1 , 16)))) + "
    "39712)";
}  // namespace

class TestAstOptimizer : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }

  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(TestAstOptimizer, test_expr0) {
  std::string expr = "1+s2*8";
  Parser parser(expr);
  ASTPtr ast = parser.Parse();
  ASTVisualizer visualizer;
  visualizer.Visualize(ast, "original_ast_diagram");
  Optimizer optimizer;
  optimizer.Optimize(ast);
  visualizer.Visualize(ast, "optimized_ast_diagram");
  std::string optimized_expr = optimizer.GenerateCode();
  std::cout << optimized_expr << std::endl;
  EXPECT_NE(optimized_expr, "");
}

TEST_F(TestAstOptimizer, test_expr1) {
  std::string expr = "32*Ceiling((Min(32,s8)-Min(0,s8))/16)";
  Parser parser(expr);
  ASTPtr ast = parser.Parse();
  ASTVisualizer visualizer;
  visualizer.Visualize(ast, "original_ast_diagram");
  Optimizer optimizer;
  optimizer.Optimize(ast);
  visualizer.Visualize(ast, "optimized_ast_diagram");
  std::string optimized_expr = optimizer.GenerateCode();
  std::cout << optimized_expr << std::endl;
  EXPECT_NE(optimized_expr, "");
}

TEST_F(TestAstOptimizer, test_expr2) {
  std::string expr =
      "((32 * Ceiling((((((16 * Ceiling((Ceiling((Min(32, s8) - Min(0, s8))) * Rational(1 , 16))) * z0t_size) + "
      "Ceiling((Min(32, s8) - Min(0, s8))) - (16 * Ceiling((Ceiling((Min(32, s8) - Min(0, s8))) * Rational(1 , 16))))) "
      "* 2) + (((16 * Ceiling((Rational(1 , 16) * s1)) * z0t_size) + s1 - (16 * Ceiling((Rational(1 , 16) * s1)))) * "
      "2) + (((16 * Ceiling((Rational(1 , 16) * s11)) * z0t_size) + s11 - (16 * Ceiling((Rational(1 , 16) * s11)))) * "
      "2) + (((16 * Ceiling((Rational(1 , 16) * s14)) * z0t_size) + s14 - (16 * Ceiling((Rational(1 , 16) * s14)))) * "
      "2) + (((16 * Ceiling((Rational(1 , 16) * s17)) * z0t_size) + s17 - (16 * Ceiling((Rational(1 , 16) * s17)))) * "
      "2) + (((16 * Ceiling((Rational(1 , 16) * s20)) * z0t_size) + s20 - (16 * Ceiling((Rational(1 , 16) * s20)))) * "
      "2) + (((16 * Ceiling((Rational(1 , 16) * s23)) * z0t_size) + s23 - (16 * Ceiling((Rational(1 , 16) * s23)))) * "
      "2) + (((16 * Ceiling((Rational(1 , 16) * s3)) * z0t_size) + s3 - (16 * Ceiling((Rational(1 , 16) * s3)))) * 2)) "
      "* Rational(1 , 16)))) + (32 * Ceiling((((((16 * Ceiling((Rational(1 , 16) * s5)) * z0t_size) + s5 - (16 * "
      "Ceiling((Rational(1 , 16) * s5)))) * 2) + (((8 * Ceiling((Rational(1 , 8) * s5)) * z0t_size) + s5 - (8 * "
      "Ceiling((Rational(1 , 8) * s5)))) * 4)) * Rational(1 , 32)))) + (32 * Ceiling((((16 * "
      "Ceiling(((Ceiling((Min(32, s8) - Min(0, s8))) + s1 + s11 + s14 + s17 + s20 + s23 + s3 + s5) * Rational(1 , "
      "16))) * z0t_size) + Ceiling((Min(32, s8) - Min(0, s8))) + s1 + s11 + s14 + s17 + s20 + s23 + s3 + s5 - (16 * "
      "Ceiling(((Ceiling((Min(32, s8) - Min(0, s8))) + s1 + s11 + s14 + s17 + s20 + s23 + s3 + s5) * Rational(1 , "
      "16))))) * Rational(1 , 4)))) + (32 * Ceiling((((16 * Ceiling((Rational(1 , 16) * s5)) * z0t_size) + s5 - (16 * "
      "Ceiling((Rational(1 , 16) * s5)))) * Rational(1 , 4)))) + (32 * Ceiling((Rational(1 , 4) * s5))) + (32 * "
      "Ceiling((Rational(1 , 8) * s5))) + (64 * Ceiling((((8 * Ceiling((Rational(1 , 8) * s5)) * z0t_size) + s5 - (8 * "
      "Ceiling((Rational(1 , 8) * s5)))) * Rational(1 , 4)))) + 32 + Max(8192, Min(65536, Max(16384, (((16 * "
      "Ceiling((Max(Max(Max(Max(Max(Max(Max(Max(0, Ceiling((Min(32, s8) - Min(0, s8)))), s5), s3), s23), s11), s14), "
      "s17), s20) * Rational(1 , 16)))) + 30) * 64)))))";
  Parser parser(expr);
  ASTPtr ast = parser.Parse();
  ASTVisualizer visualizer;
  visualizer.Visualize(ast, "original_ast_diagram");
  Optimizer optimizer;
  optimizer.Optimize(ast);
  visualizer.Visualize(ast, "optimized_ast_diagram");
  std::string optimized_expr = optimizer.GenerateCode();
  std::cout << optimized_expr << std::endl;
  EXPECT_NE(optimized_expr, "");
}

TEST_F(TestAstOptimizer, test_expr3) {
  Parser parser(kExpr3Str);
  ASTPtr ast = parser.Parse();
  ASTVisualizer visualizer;
  visualizer.Visualize(ast, "original_ast_diagram");
  Optimizer optimizer;
  optimizer.Optimize(ast);
  visualizer.Visualize(ast, "optimized_ast_diagram");
  std::string optimized_expr = optimizer.GenerateCode();
  std::cout << optimized_expr << std::endl;
  EXPECT_NE(optimized_expr, "");
}

TEST_F(TestAstOptimizer, test_expr4) {
  std::string expr = "(#)";
  Parser parser(expr);
  ASTPtr ast = parser.Parse();
  EXPECT_EQ(ast, nullptr);
  Optimizer optimizer;
  optimizer.Optimize(ast);
  std::string optimized_expr = optimizer.GenerateCode();
  EXPECT_EQ(optimized_expr, "");
}

TEST_F(TestAstOptimizer, ParseFunctionAcceptsUnderscorePrefixedVariable) {
  Parser parser("Ceiling(_dynamic_shape)", false);

  ASTPtr ast = parser.Parse();

  ASSERT_NE(ast, nullptr);
  EXPECT_TRUE(parser.IsFullyParsed());
  ASSERT_EQ(ast->children.size(), 1U);
  EXPECT_EQ(ast->children[0]->expr, "_dynamic_shape");
}

TEST_F(TestAstOptimizer, ParseFunctionReturnsNullForInvalidArgument) {
  Parser parser("Max(a < b, c)", false);

  ASTPtr ast = parser.Parse();

  EXPECT_EQ(ast, nullptr);
}

TEST_F(TestAstOptimizer, AstHashStorageScalesLinearlyForLongExpression) {
  Parser parser(BuildLongSum(512U), false);
  ASTPtr ast = parser.Parse();
  ASSERT_NE(ast, nullptr);
  Optimizer optimizer;
  optimizer.Optimize(ast);

  size_t node_count = 0U;
  size_t hash_bytes = 0U;
  CountNodesAndHashBytes(ast, node_count, hash_bytes);

  EXPECT_LT(hash_bytes, node_count * 16U);
}
