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
#include "util/base_types_printer.h"

TEST(BaseTypesPrinterSt, SerializeValidExpression) {
  nlohmann::json json;
  const auto expression = af::Symbol("size");

  af::to_json(json, expression);

  ASSERT_TRUE(json.is_array());
  ASSERT_EQ(json.size(), 1U);
  ASSERT_TRUE(json[0].is_array());
  ASSERT_EQ(json[0].size(), 1U);
  EXPECT_EQ(json[0][0], "size");
}

TEST(BaseTypesPrinterSt, SerializeInvalidExpression) {
  nlohmann::json json;
  const af::Expression expression;

  af::to_json(json, expression);

  ASSERT_TRUE(json.is_array());
  ASSERT_EQ(json.size(), 1U);
  ASSERT_TRUE(json[0].is_array());
  ASSERT_EQ(json[0].size(), 1U);
  EXPECT_EQ(json[0][0], "");
}
