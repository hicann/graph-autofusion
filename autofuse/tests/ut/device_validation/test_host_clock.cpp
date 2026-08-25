/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_clock.h"

#include <gtest/gtest.h>

#include <cmath>

TEST(HostClockTest, ReturnsFiniteNonNegativeMilliseconds) {
  const double first = device_validation::MonotonicMilliseconds();
  const double second = device_validation::MonotonicMilliseconds();
  EXPECT_GE(first, 0.0);
  EXPECT_GE(second, first);
  EXPECT_TRUE(std::isfinite(first));
  EXPECT_TRUE(std::isfinite(second));
}
