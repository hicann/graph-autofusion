/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "parser/tuning_space.h"

namespace {
using att::Container;

// 最小具体子类：覆盖纯虚 GetBufferNum，使 Container 可被实例化。
class TestContainer : public Container {
 public:
  explicit TestContainer(const std::string &name) : Container(name) {}
  int64_t GetBufferNum() const override {
    return 0;
  }
};

// 覆盖质量加固对 GetCoTensors 补充 const 后的执行行（增量行覆盖率门禁）。
TEST(QualityHardeningContainerTest, GetCoTensors) {
  TestContainer container("test");
  EXPECT_TRUE(container.GetCoTensors().empty());
}
}  // namespace
