/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/pgo/pgo_candidate_identity.h"

#include <string>

#include <gtest/gtest.h>

namespace codegen::pgo {
namespace {

TEST(PgoCandidateIdentity, BuildCandidateKeyReturnsLowercaseHexDigest) {
  const std::string key = BuildCandidateKey("graph0", "{repr:1}", 1024U, 8U);
  EXPECT_EQ(key.size(), 32U);
  EXPECT_TRUE(IsValidCandidateKey(key));
}

TEST(PgoCandidateIdentity, BuildCandidateKeyIsDeterministic) {
  const std::string first = BuildCandidateKey("graph0", "{repr:1}", 1024U, 8U);
  const std::string second = BuildCandidateKey("graph0", "{repr:1}", 1024U, 8U);
  EXPECT_EQ(first, second);
}

TEST(PgoCandidateIdentity, BuildCandidateKeyDiffersAcrossCandidateFields) {
  const std::string base = BuildCandidateKey("graph0", "{repr:1}", 1024U, 8U);
  EXPECT_NE(base, BuildCandidateKey("graph1", "{repr:1}", 1024U, 8U));
  EXPECT_NE(base, BuildCandidateKey("graph0", "{repr:2}", 1024U, 8U));
  EXPECT_NE(base, BuildCandidateKey("graph0", "{repr:1}", 2048U, 8U));
  EXPECT_NE(base, BuildCandidateKey("graph0", "{repr:1}", 1024U, 16U));
}

TEST(PgoCandidateIdentity, BuildCandidateKeySeparatesConcatenatedFields) {
  // "ab|c" and "a|b|c" must not collide: the key depends on field boundaries.
  EXPECT_NE(BuildCandidateKey("ab", "c", 1U, 1U), BuildCandidateKey("a", "b|c", 1U, 1U));
}

TEST(PgoCandidateIdentity, IsValidCandidateKeyRejectsMalformedKeys) {
  EXPECT_FALSE(IsValidCandidateKey(""));
  EXPECT_FALSE(IsValidCandidateKey("abc"));
  EXPECT_FALSE(IsValidCandidateKey(std::string(32U, 'g')));
  EXPECT_FALSE(IsValidCandidateKey(std::string(31U, 'a') + "A"));
  EXPECT_TRUE(IsValidCandidateKey("0123456789abcdef0123456789abcdef"));
}

}  // namespace
}  // namespace codegen::pgo
