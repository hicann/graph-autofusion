/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_executor.h"

#include <gtest/gtest.h>

#include <string>

namespace device_validation {
namespace {

TEST(AclnnDispatchTest, KnownOpNamesResolveToKinds) {
  EXPECT_EQ(AclnnOpKindFromName("IsInf"), AclnnOpKind::kIsInf);
  EXPECT_EQ(AclnnOpKindFromName("LogicalOr"), AclnnOpKind::kLogicalOr);
  EXPECT_EQ(AclnnOpKindFromName("MaskedFillScalar"), AclnnOpKind::kMaskedFillScalar);
  EXPECT_EQ(AclnnOpKindFromName("MaskedFillTensor"), AclnnOpKind::kMaskedFillTensor);
}

TEST(AclnnDispatchTest, UnknownOpNameIsUnknownKind) {
  EXPECT_EQ(AclnnOpKindFromName("NoSuchOp"), AclnnOpKind::kUnknown);
  EXPECT_EQ(AclnnOpKindFromName(""), AclnnOpKind::kUnknown);
}

TEST(AclnnDispatchTest, OpNameRoundTrip) {
  EXPECT_EQ(AclnnOpName(AclnnOpKindFromName("IsInf")), "IsInf");
  EXPECT_EQ(AclnnOpName(AclnnOpKind::kUnknown), "unknown");
}

TEST(AclnnDispatchTest, KnownArityIsAccepted) {
  EXPECT_TRUE(AclnnOpArityValid(AclnnOpKind::kIsInf, 1, 1));
  EXPECT_TRUE(AclnnOpArityValid(AclnnOpKind::kLogicalOr, 2, 1));
  EXPECT_TRUE(AclnnOpArityValid(AclnnOpKind::kMaskedFillScalar, 3, 1));
  EXPECT_TRUE(AclnnOpArityValid(AclnnOpKind::kMaskedFillTensor, 3, 1));
}

TEST(AclnnDispatchTest, WrongArityIsRejected) {
  EXPECT_FALSE(AclnnOpArityValid(AclnnOpKind::kIsInf, 2, 1));
  EXPECT_FALSE(AclnnOpArityValid(AclnnOpKind::kLogicalOr, 1, 1));
  EXPECT_FALSE(AclnnOpArityValid(AclnnOpKind::kMaskedFillScalar, 2, 1));
  EXPECT_FALSE(AclnnOpArityValid(AclnnOpKind::kMaskedFillTensor, 3, 2));
  EXPECT_FALSE(AclnnOpArityValid(AclnnOpKind::kUnknown, 1, 1));
}

}  // namespace
}  // namespace device_validation
