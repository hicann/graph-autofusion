/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "st_fixture.h"

namespace {
aclskScopeVerifyNodeInfo Compute(int64_t id, aclskScopeVerifyKernelType type, uint32_t blocks, uint32_t cube,
                                 uint32_t vector, int32_t scheMode) {
    aclskScopeVerifyNodeInfo node{};
    node.taskId = id;
    node.scopeId = 1;
    node.taskType = ACLSK_SCOPE_VERIFY_NODE_COMPUTE;
    node.kernelType = type;
    node.numBlocks = blocks;
    node.taskRatio[0] = cube;
    node.taskRatio[1] = vector;
    node.scheMode = scheMode;
    return node;
}
}  // namespace
TEST_F(AotSystemTest, VerifyCompatibleComputeGraphNeedsNoSplit) {
    std::vector<aclskScopeVerifyNodeInfo> nodes{Compute(1, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 4, 1, 0, 0),
                                                Compute(2, ACLSK_SCOPE_VERIFY_KERNEL_VECTOR, 4, 0, 1, 0)};
    aclskScopeVerifyGraphInfo graph{nodes.data(), nodes.size(), 0, nullptr};
    size_t count = 123;
    ASSERT_EQ(aclskScopeVerify(&graph, 0, nullptr, &count), ACL_SUCCESS);
    EXPECT_EQ(count, 0U);
}
TEST_F(AotSystemTest, VerifyScheModeSplitReferencesOriginalInputNode) {
    std::vector<aclskScopeVerifyNodeInfo> nodes{Compute(1, ACLSK_SCOPE_VERIFY_KERNEL_MIX, 4, 1, 2, 1),
                                                Compute(2, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 2, 1, 0, 1)};
    aclskScopeVerifyGraphInfo graph{nodes.data(), nodes.size(), 0, nullptr};
    aclskScopeVerifySplitResult result{};
    size_t count = 0;
    ASSERT_EQ(aclskScopeVerify(&graph, 1, &result, &count), ACL_SUCCESS);
    ASSERT_EQ(count, 1U);
    EXPECT_EQ(result.splitNode, &nodes[1]);
    EXPECT_EQ(result.splitType, ACLSK_SCOPE_VERIFY_SPLIT_BEFORE_NODE);
    EXPECT_EQ(result.splitReason, ACLSK_SCOPE_VERIFY_SYNCALL_OP_DROP);
    EXPECT_EQ(result.extendType, 0);
    EXPECT_EQ(result.extendInfo, nullptr);
}
