/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLMODEL_SPECIALIZATION_H
#define ACLMODEL_SPECIALIZATION_H

#include <cstdint>
#include "super_kernel.h"

// Internal input contract until the public ACL header is integrated.
struct aclmdlRISpecCompileOption {
  const char *kernelName;
  uint64_t compileOptionCount;
  const char *const *compileOptions;
};

struct aclmdlRISpecSKFeature {
  const char *feature;
  uint64_t kernelNameCount;
  const char *const *kernelNames;
};

struct aclmdlRISpecOptions {
  uint64_t jobs;  // Zero defaults to one serial compilation job.
  uint64_t specCompileOptionCount;
  aclmdlRISpecCompileOption *specCompileOptions;
  bool enableSK;
  uint64_t specSKFeatureCount;
  aclmdlRISpecSKFeature *specSKFeatures;
  char reserved[128];
};

aclError aclmdlRISpecOptimize(aclmdlRI modelRI, aclmdlRISpecOptions *options);
aclError aclmdlRISpecScopeBegin(aclrtStream stream);
aclError aclmdlRISpecScopeEnd(aclrtStream stream);
aclError aclmdlRISpecOptimizeByFlag(aclmdlRI modelRI, aclmdlRISpecOptions *options);

#endif
