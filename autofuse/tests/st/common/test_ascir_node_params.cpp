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

#include "ascir_node_param/ascir_node_param.h"
#include "ascir_ops.h"

TEST(AscirNodeParamsStTest, GetOrCreateAscirNodeParamsStoresCreatedParams) {
  af::AscGraph graph("test_graph");
  af::ascir_op::Add add_op("add");
  graph.AddNode(add_op);
  auto node = graph.FindNode("add");
  ASSERT_NE(node, nullptr);

  const auto params = ascir_param::GetOrCreateAscirNodeParams(node);
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(ascir_param::GetAscirNodeParams(node), params);
}
