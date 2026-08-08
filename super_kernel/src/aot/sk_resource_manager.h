/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SK_RESOURCE_MANAGER_H
#define SK_RESOURCE_MANAGER_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"

class SkResourceManager {
 public:
  static constexpr size_t kDefaultValueMemoryBytes = sizeof(uint64_t);

  enum class ResourceKind : uint8_t {
    kDeviceMemory = 0,
  };

  struct ResourceRecord {
    ResourceKind kind = ResourceKind::kDeviceMemory;
    void *addr = nullptr;
    size_t bytes = 0U;
  };

  struct ModelResourceContext {
    aclmdlRI model = nullptr;
    std::string modelId;
    std::vector<ResourceRecord> resources;
  };

  static SkResourceManager &GetInstance();

  static aclError GenerateModelId(aclmdlRI model, std::string &modelId);
  static void SetCurrentModel(aclmdlRI model);
  static aclError CallbackRegister(aclmdlRI model, const std::string &modelId);
  static aclError ValueMemory(void **addr, size_t bytes = kDefaultValueMemoryBytes);

  SkResourceManager(const SkResourceManager &) = delete;
  SkResourceManager &operator=(const SkResourceManager &) = delete;

 private:
  SkResourceManager() = default;
  ~SkResourceManager() = default;

  static std::mutex resourceMutex_;
  static std::unordered_map<aclmdlRI, ModelResourceContext> modelContexts_;
  static thread_local aclmdlRI currentModel_;

  aclError AllocForModel(aclmdlRI model, void **addr, size_t bytes);
  static aclError ReleaseRecord(const ResourceRecord &record);
  static bool ReleaseModelResources(const std::vector<ResourceRecord> &resources);
  static void OnModelDestroy(void *userData);
};

#endif
