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

#include "ascgraph_info_complete.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace optimize {
namespace {

std::vector<std::string> GetNames(const std::vector<af::Expression> &vars) {
  std::vector<std::string> names;
  names.reserve(vars.size());
  for (const auto &var : vars) {
    names.emplace_back(af::SymbolicUtils::ToString(var));
  }
  return names;
}

TEST(FrontendShapeVarsTest, NormalizeKsNamesUsesNaturalOrder) {
  std::vector<af::Expression> vars = {af::Symbol("ks10"), af::Symbol("ks2"), af::Symbol("ks0"), af::Symbol("ks1"),
                                      af::Symbol("ks2")};

  ASSERT_EQ(AscGraphInfoComplete::NormalizeFrontendShapeVars(vars), af::SUCCESS);
  EXPECT_EQ(GetNames(vars), (std::vector<std::string>{"ks0", "ks1", "ks2", "ks10"}));
}

TEST(FrontendShapeVarsTest, NormalizeKsNamesAllowsSparseIndices) {
  std::vector<af::Expression> vars = {af::Symbol("ks0"), af::Symbol("ks2")};

  ASSERT_EQ(AscGraphInfoComplete::NormalizeFrontendShapeVars(vars), af::SUCCESS);
  EXPECT_EQ(GetNames(vars), (std::vector<std::string>{"ks0", "ks2"}));
}

TEST(FrontendShapeVarsTest, CollectsOriginalAscGraphSymbolsBeforeOptimization) {
  af::AscGraph graph("frontend_shape_vars");
  graph.CreateSizeVar("ks10");
  graph.CreateSizeVar("ks2");
  graph.CreateSizeVar("ks0");
  graph.CreateSizeVar(16);

  std::vector<af::Expression> vars;
  ASSERT_EQ(AscGraphInfoComplete::CollectFrontendShapeVars(graph, vars), af::SUCCESS);
  ASSERT_EQ(AscGraphInfoComplete::NormalizeFrontendShapeVars(vars), af::SUCCESS);
  EXPECT_EQ(GetNames(vars), (std::vector<std::string>{"ks0", "ks2", "ks10"}));
}

TEST(FrontendShapeVarsTest, CollectsSymbolsEmbeddedInAxisExpressions) {
  af::AscGraph graph("axis_shape_vars");
  const auto shape = af::Symbol("s0");
  graph.CreateAxis("z0", shape);

  std::vector<af::Expression> vars;
  ASSERT_EQ(AscGraphInfoComplete::CollectFrontendShapeVars(graph, vars), af::SUCCESS);
  ASSERT_EQ(AscGraphInfoComplete::NormalizeFrontendShapeVars(vars), af::SUCCESS);
  EXPECT_EQ(GetNames(vars), (std::vector<std::string>{"s0"}));
}

}  // namespace
}  // namespace optimize
