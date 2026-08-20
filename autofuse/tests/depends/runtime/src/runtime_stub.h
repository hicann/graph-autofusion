/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INC_LLT_RUNTIME_STUB_H
#define __INC_LLT_RUNTIME_STUB_H

#include <cstdint>
#include <cstring>
#include <vector>
#include <memory>
#include <mutex>
#include "mmpa/mmpa_api.h"
#include "rt_error_codes.h"

using rtError_t = int32_t;
constexpr rtError_t RT_ERROR_NONE = 0;
namespace ge {
struct RuntimeSocSpecDefaults {
  const char *npu_arch = "3510";
  const char *fallback = "0";
};

struct RuntimeSocSpecValue {
  const char *key;
  const char *value;
};

inline rtError_t CopyRuntimeSocSpecValue(const char *label, const char *key, char *val, const uint32_t maxLen,
                                         const RuntimeSocSpecDefaults &defaults) {
  (void)label;
  const RuntimeSocSpecValue specs[] = {
      {"NpuArch", defaults.npu_arch},
      {"vector_core_cnt", "48"},
      {"ub_size", "245760"},
      {"l2_size", "134217728"},
  };
  for (const auto &spec : specs) {
    if (std::strcmp(key, spec.key) == 0) {
      (void)strcpy_s(val, maxLen, spec.value);
      return RT_ERROR_NONE;
    }
  }
  (void)strcpy_s(val, maxLen, defaults.fallback);
  return RT_ERROR_NONE;
}

class RuntimeStub {
 public:
  virtual ~RuntimeStub() = default;

  static RuntimeStub *GetInstance();

  static void SetInstance(const std::shared_ptr<RuntimeStub> &instance) {
    instance_ = instance;
  }

  static void Install(RuntimeStub *);
  static void UnInstall(RuntimeStub *);

  static void Reset() {
    instance_.reset();
  }

  virtual rtError_t rtGetSocVersion(char *version, const uint32_t maxLen);
  virtual rtError_t rtGetSocSpec(const char *label, const char *key, char *val, const uint32_t maxLen);
  virtual const char *aclrtGetSocName();

 private:
  static std::mutex mutex_;
  static std::shared_ptr<RuntimeStub> instance_;
  static thread_local RuntimeStub *fake_instance_;
};

class RuntimeStubV2Common : public RuntimeStub {
 public:
  rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) override {
    (void)strcpy_s(version, maxLen, "Ascend910_9591");
    return RT_ERROR_NONE;
  }

  rtError_t rtGetSocSpec(const char *label, const char *key, char *val, const uint32_t maxLen) override {
    return CopyRuntimeSocSpecValue(label, key, val, maxLen, RuntimeSocSpecDefaults{"3510", "0"});
  }
};
}  // namespace ge

#ifdef __cplusplus
extern "C" {
#endif

rtError_t rtGetSocVersion(char *version, const uint32_t maxLen);
rtError_t rtGetSocSpec(const char *label, const char *key, char *val, const uint32_t maxLen);
const char *aclrtGetSocName();

#ifdef __cplusplus
}
#endif

#endif  // __INC_LLT_RUNTIME_STUB_H
