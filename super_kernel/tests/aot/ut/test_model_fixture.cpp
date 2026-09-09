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
#include "model_fixture.h"

TEST(AotModelFixture, QueriesPreserveTaskChangesAndCheckCapacity) {
    sk::test::Model model;
    auto stream = model.AddStream();
    auto task = model.AddKernel(stream, "first");
    aclmdlRITaskParams params{};
    ASSERT_EQ(aclmdlRITaskGetParams(task, &params), ACL_SUCCESS);
    params.kernelTaskParams.numBlocks = 7;
    ASSERT_EQ(aclmdlRITaskSetParams(task, &params), ACL_SUCCESS);
    ASSERT_EQ(aclmdlRITaskDisable(task), ACL_SUCCESS);
    uint32_t count = 0;
    aclmdlRITask result = nullptr;
    EXPECT_EQ(aclmdlRIGetTasksByStream(stream, &result, &count), ACL_ERROR_INVALID_PARAM);
    count = 1;
    ASSERT_EQ(aclmdlRIGetTasksByStream(stream, &result, &count), ACL_SUCCESS);
    EXPECT_EQ(result, task);
    auto snapshot = model.Snapshot(task);
    EXPECT_EQ(snapshot.numBlocks, 7U);
    EXPECT_TRUE(snapshot.disabled);
    EXPECT_EQ(snapshot.setParamsCount, 1U);
}

TEST(AotModelFixture, ModelsAreIsolatedAndSnapshotsOwnArguments) {
    sk::test::Model first;
    sk::test::Model second;
    auto a = first.AddKernel(first.AddStream(), "a");
    auto b = second.AddKernel(second.AddStream(), "b");
    auto before = first.Snapshot(a);
    ASSERT_FALSE(before.args.empty());
    aclmdlRITaskParams params{};
    ASSERT_EQ(aclmdlRITaskGetParams(a, &params), ACL_SUCCESS);
    static_cast<unsigned char *>(params.kernelTaskParams.args)[0] = 42;
    EXPECT_EQ(before.args[0], 0);
    EXPECT_EQ(first.Snapshot(a).args[0], 42);
    EXPECT_EQ(second.Snapshot(b).args[0], 0);
    EXPECT_THROW(first.Snapshot(b), std::invalid_argument);
}

TEST(AotModelFixture, SetParamsCopiesHostArgumentsBeforeCallerReleasesThem) {
    sk::test::Model model;
    auto stream = model.AddStream();
    auto task = model.AddKernel(stream, "first");
    auto other = model.AddKernel(model.AddStream(), "second");
    EXPECT_NE(model.Snapshot(task).id, model.Snapshot(other).id);
    aclmdlRITaskParams params{};
    ASSERT_EQ(aclmdlRITaskGetParams(task, &params), ACL_SUCCESS);
    {
        std::vector<unsigned char> hostArgs{1, 2, 3, 4};
        aclrtLaunchKernelAttr attr{};
        attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
        attr.value.schemMode = 1;
        aclrtLaunchKernelCfg cfg{&attr, 1};
        params.kernelTaskParams.cfg = &cfg;
        params.kernelTaskParams.args = hostArgs.data();
        params.kernelTaskParams.argsSize = hostArgs.size();
        params.kernelTaskParams.isHostArgs = true;
        ASSERT_EQ(aclmdlRITaskSetParams(task, &params), ACL_SUCCESS);
        hostArgs.assign(hostArgs.size(), 9);
        attr.value.schemMode = 0;
    }
    EXPECT_EQ(model.Snapshot(task).args, (std::vector<unsigned char>{1, 2, 3, 4}));
    ASSERT_EQ(aclmdlRITaskGetParams(task, &params), ACL_SUCCESS);
    EXPECT_EQ(static_cast<unsigned char *>(params.kernelTaskParams.args)[0], 1);
    ASSERT_NE(params.kernelTaskParams.cfg, nullptr);
    ASSERT_EQ(params.kernelTaskParams.cfg->numAttrs, 1U);
    EXPECT_EQ(params.kernelTaskParams.cfg->attrs[0].value.schemMode, 1);
    auto snapshot = model.Snapshot(task);
    ASSERT_EQ(snapshot.attributes.size(), 1U);
    EXPECT_EQ(snapshot.attributes[0].value.schemMode, 1);
    params.kernelTaskParams.args = nullptr;
    EXPECT_EQ(aclmdlRITaskSetParams(task, &params), ACL_ERROR_INVALID_PARAM);
}

TEST(AotModelFixture, DeviceCopyDoesNotDereferenceLegacyOpaqueDeviceAddress) {
    void *host = nullptr;
    ASSERT_EQ(aclrtMallocHost(&host, sizeof(uint64_t)), ACL_SUCCESS);
    *static_cast<uint64_t *>(host) = 17;
    EXPECT_EQ(aclrtMemcpy(host, sizeof(uint64_t), reinterpret_cast<void *>(0x1000), sizeof(uint64_t),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_SUCCESS);
    EXPECT_EQ(*static_cast<uint64_t *>(host), 17U);
    EXPECT_EQ(aclrtFreeHost(host), ACL_SUCCESS);
}
