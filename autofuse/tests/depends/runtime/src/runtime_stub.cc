/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <queue>
#include <cstring>
#include "securec.h"
#include "runtime_stub.h"

namespace ge {
std::shared_ptr<RuntimeStub> RuntimeStub::instance_;
std::mutex RuntimeStub::mutex_;
thread_local RuntimeStub *RuntimeStub::fake_instance_;
RuntimeStub *RuntimeStub::GetInstance() {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (fake_instance_ != nullptr) {
    return fake_instance_;
  }
  if (instance_ == nullptr) {
    instance_ = std::make_shared<RuntimeStub>();
  }
  return instance_.get();
}

void RuntimeStub::Install(RuntimeStub *instance) {
  fake_instance_ = instance;
}

void RuntimeStub::UnInstall(RuntimeStub *) {
  fake_instance_ = nullptr;
}

rtError_t RuntimeStub::rtGetSocVersion(char *version, const uint32_t maxLen) {
  (void)strcpy_s(version, maxLen, "Ascend910B1");
  return RT_ERROR_NONE;
}

rtError_t RuntimeStub::rtGetSocSpec(const char *label, const char *key, char *val, const uint32_t maxLen) {
  (void)label;
  if (strcmp(key, "NpuArch") == 0) {
    (void)strcpy_s(val, maxLen, "2201");
    return RT_ERROR_NONE;
  }
  if (strcmp(key, "vector_core_cnt") == 0) {
    (void)strcpy_s(val, maxLen, "48");
    return RT_ERROR_NONE;
  }
  if (strcmp(key, "ub_size") == 0) {
    (void)strcpy_s(val, maxLen, "245760");
    return RT_ERROR_NONE;
  }
  if (strcmp(key, "l2_size") == 0) {
    (void)strcpy_s(val, maxLen, "134217728");
    return RT_ERROR_NONE;
  }
  // 返回 padding_size = 32 (兼容旧平台)
  if (strcmp(key, "padding_size") == 0) {
    (void)strcpy_s(val, maxLen, "32");
    return RT_ERROR_NONE;
  }
  (void)strcpy_s(val, maxLen, "2201");
  return RT_ERROR_NONE;
}

const char *RuntimeStub::aclrtGetSocName() {
  return "Ascend910B1";
}
}  // namespace ge

#ifdef __cplusplus
extern "C" {
#endif

rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) {
  return ge::RuntimeStub::GetInstance()->rtGetSocVersion(version, maxLen);
}

rtError_t rtGetSocSpec(const char *label, const char *key, char *val, const uint32_t maxLen) {
  return ge::RuntimeStub::GetInstance()->rtGetSocSpec(label, key, val, maxLen);
}

const char *aclrtGetSocName() {
  return ge::RuntimeStub::GetInstance()->aclrtGetSocName();
}
#ifdef __cplusplus
}
#endif
