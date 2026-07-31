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
#include "mockcpp/mockcpp.hpp"
#include <memory>
#include <cstring>

#define private public
#define protected public
#include "sk_common.h"
#include "super_kernel.h"
#include "securec.h"

class SkScopeLaunchTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {
    GlobalMockObject::verify();
  }
};

TEST_F(SkScopeLaunchTest, ScopeBeginSuccess) {
  const char *scopeName = "test_scope";
  aclrtStream stream = nullptr;
  aclError ret = aclskScopeBegin(scopeName, stream);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkScopeLaunchTest, ScopeEndSuccess) {
  const char *scopeName = "test_scope";
  aclrtStream stream = nullptr;
  aclError ret = aclskScopeEnd(scopeName, stream);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkScopeLaunchTest, ScopeBeginWithNullScopeName) {
  aclrtStream stream = nullptr;
  aclError ret = aclskScopeBegin(nullptr, stream);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkScopeLaunchTest, ScopeEndWithNullScopeName) {
  aclrtStream stream = nullptr;
  aclError ret = aclskScopeEnd(nullptr, stream);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkScopeLaunchTest, ScopeBeginAndEndWithEmptyScopeName) {
  const char *scopeName = "";
  aclrtStream stream = nullptr;
  aclError ret = aclskScopeBegin(scopeName, stream);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclskScopeEnd(scopeName, stream);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SkScopeLaunchTest, ScopeBeginWithMaxLengthScopeName) {
  char scopeName[MAX_SCOPE_NAME_LEN];
  (void)memset_s(scopeName, sizeof(scopeName), 'a', MAX_SCOPE_NAME_LEN - 1);
  scopeName[MAX_SCOPE_NAME_LEN - 1] = '\0';
  aclrtStream stream = nullptr;
  aclError ret = aclskScopeBegin(scopeName, stream);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkScopeLaunchTest, ScopeBeginWithExceedMaxLengthScopeName) {
  char scopeName[MAX_SCOPE_NAME_LEN + 10];
  (void)memset_s(scopeName, sizeof(scopeName), 'a', MAX_SCOPE_NAME_LEN + 9);
  scopeName[MAX_SCOPE_NAME_LEN + 9] = '\0';
  aclrtStream stream = nullptr;
  aclError ret = aclskScopeBegin(scopeName, stream);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SkScopeLaunchTest, ScopeBeginAndEndReturnConsistentResult) {
  const char *scopeName = "consistency_test";
  aclrtStream stream = nullptr;

  aclError retBegin = aclskScopeBegin(scopeName, stream);
  aclError retEnd = aclskScopeEnd(scopeName, stream);

  EXPECT_EQ(retBegin, retEnd);
  EXPECT_EQ(retBegin, ACL_SUCCESS);
}
