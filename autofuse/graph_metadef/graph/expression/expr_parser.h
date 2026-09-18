/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_EXPRESSION_EXPR_PARSER_H_
#define GRAPH_EXPRESSION_EXPR_PARSER_H_
#include "scanner.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "expression_impl.h"

#include <symengine/integer.h>
#include <symengine/rational.h>
#include <memory>
namespace af {

class ExprParser {
 public:
  explicit ExprParser(Scanner scanner) : scanner_(scanner) {
    Init();
  }
  ExpressionImplPtr ParserExpression();

 private:
  graphStatus Init();
  graphStatus Eat(TokenType type);
  ExpressionImplPtr ParserFactor();
  ExpressionImplPtr ParserAddSubtract();
  ExpressionImplPtr ParserMulDivide();
  ExpressionImplPtr ParserMaxFunction();
  ExpressionImplPtr ParserMinFunction();
  ExpressionImplPtr ParserPowFunction();
  ExpressionImplPtr ParserModFunction();
  ExpressionImplPtr ParserLogFunction();
  ExpressionImplPtr ParserCeilFunction();
  ExpressionImplPtr ParserFloorFunction();
  ExpressionImplPtr ParserAbsFunction();
  ExpressionImplPtr ParserRationalFunction();
  ExpressionImplPtr ParserNumber();
  ExpressionImplPtr ParserIdentifier();
  ExpressionImplPtr ParserLParen();
  ExpressionImplPtr ParseConstBoolen();
  ExpressionImplPtr ParserMinus();
  ExpressionImplPtr ParserEqual();
  ExpressionImplPtr ParserUnequal();
  ExpressionImplPtr ParserLessEqual();
  ExpressionImplPtr ParserLessThan();
  ExpressionImplPtr ParserLogicalAnd();
  ExpressionImplPtr ParserLogicalOr();

  Scanner scanner_;
  Token currentToken_;
};

}  // namespace af
#endif  // GRAPH_EXPRESSION_EXPR_PARSER_H_
