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
#include "codegen/ascendc_api_registry.h"

namespace codegen {
TEST(AscendcRegBaseApiRegisterReduceTest, RegistersCompleteReduceExtendSource) {
  const auto &source = AscendCApiRegistry::GetInstance().GetFileContent("reduce_extend_reg_base.h");

  ASSERT_FALSE(source.empty());
  EXPECT_NE(source.find("ReduceSumExtend"), std::string::npos);
  EXPECT_NE(source.find("ReduceMeanExtend"), std::string::npos);
  EXPECT_NE(source.find("ReduceMaxExtend"), std::string::npos);
  EXPECT_NE(source.find("ReduceMinExtend"), std::string::npos);
  EXPECT_NE(source.find("ReduceProdExtend"), std::string::npos);
  EXPECT_NE(source.find("ReduceAnyExtend"), std::string::npos);
  EXPECT_NE(source.find("ReduceAllExtend"), std::string::npos);
  EXPECT_NE(source.find("ReduceXorSumExtend"), std::string::npos);
  EXPECT_NE(source.find("MeanExtend"), std::string::npos);
  EXPECT_NE(source.find("SumExtend"), std::string::npos);
  EXPECT_EQ(source.find("#include \"reduce/"), std::string::npos);
}
}  // namespace codegen
