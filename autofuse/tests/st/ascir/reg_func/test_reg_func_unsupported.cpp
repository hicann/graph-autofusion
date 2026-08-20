/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 */

#include <gtest/gtest.h>

#include "ascir_registry.h"

namespace af {
TEST(UnsupportedAscIrRegistration, HasExpectedDefinitionAndImplementations) {
  const auto &registry = ascir::AscirRegistry::GetInstance().GetAll();
  const auto iter = registry.find("Unsupported");
  ASSERT_NE(iter, registry.end());

  auto definition = iter->second;
  EXPECT_TRUE(definition.GetInputDefs().empty());
  ASSERT_EQ(definition.GetOutputDefs().size(), 1U);
  EXPECT_EQ(definition.GetOutputDefs()[0].first, "y");
  EXPECT_TRUE(definition.IsStartNode());
  ASSERT_EQ(definition.GetAttrDefs().size(), 1U);
  EXPECT_EQ(definition.GetAttrDefs()[0].name, "error_msg");
  EXPECT_EQ(definition.GetComputeType(), ComputeType::kComputeInvalid);

  auto codegen = definition.GetAscIrCodegenImpl("2201");
  ASSERT_NE(codegen, nullptr);
  EXPECT_TRUE(codegen->GetApiCallName().empty());
  EXPECT_EQ(codegen->GetApiName(), "Unsupported");

  auto att = definition.GetAscIrAttImpl("2201");
  ASSERT_NE(att, nullptr);
  EXPECT_EQ(att->GetApiPerf(), nullptr);
  EXPECT_EQ(att->GetAscendCApiPerfTable(), nullptr);
}
}  // namespace af
