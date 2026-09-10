/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef KERNEL_SPEC_OPTIONS_H
#define KERNEL_SPEC_OPTIONS_H

#include <cstdint>
#include "super_kernel.h"

// Internal input contract until the aclSpecOptimize public header is integrated.
enum aclKernelSpecFilterMode { ACL_KERNEL_SPEC_FILTER_ALLOW, ACL_KERNEL_SPEC_FILTER_DENY };

struct aclspecOption {
  const char *kernelName;  // ECMAScript regular expression matching the entire kernel entry.
  uint64_t compileOptionCount;
  const char **compileOption;  // Each element is one argv token; whitespace is not split.
};

struct aclKernelSpecFilter {
  aclKernelSpecFilterMode mode;
  uint64_t kernelEntryCount;
  const char **kernelEntries;
};

struct aclspecOptions {
  uint64_t jobs;  // Zero selects hardware concurrency, with a minimum of one.
  uint64_t specOptionCount;
  aclspecOption *specOptions;
  aclskOptions *skOptions;
  uint64_t filterCount;
  aclKernelSpecFilter **filter;  // ALLOW and DENY must not coexist, even with empty lists.
};

#endif
