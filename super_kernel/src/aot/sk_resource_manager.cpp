/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sk_resource_manager.h"
#include "sk_log.h"
#include "sk_model_context.h"

std::mutex SkResourceManager::resourceMutex_;
std::unordered_map<aclmdlRI, SkResourceManager::ModelResourceContext> SkResourceManager::modelContexts_;
thread_local aclmdlRI SkResourceManager::currentModel_ = nullptr;

SkResourceManager &SkResourceManager::GetInstance() {
  static SkResourceManager instance;
  return instance;
}

void SkResourceManager::SetCurrentModel(aclmdlRI model) {
  currentModel_ = model;
}

aclError SkResourceManager::ValueMemory(void **addr, size_t bytes) {
  return GetInstance().AllocForModel(currentModel_, addr, bytes);
}

aclError SkResourceManager::CallbackRegister(aclmdlRI model) {
  if (model == nullptr) {
    SK_LOGE("ensure destroy callback failed: current model is null");
    return ACL_ERROR_INVALID_PARAM;
  }

  const std::string modelId = GetCurrentModelId();
  const std::string modelLabel = GetCurrentModelLabel();
  if (modelId.empty() || modelLabel.empty()) {
    SK_LOGE("ensure destroy callback failed: no active model context, model=%p", model);
    return ACL_ERROR_FAILURE;
  }

  std::lock_guard<std::mutex> lock(resourceMutex_);
  if (modelContexts_.count(model) != 0U) {
    SK_LOGE(
        "model resource context already exists before model destroy callback: "
        "model=%p, modelLabel=%s, modelId=%s",
        model, modelLabel.c_str(), modelId.c_str());
    return ACL_ERROR_FAILURE;
  }

  aclError ret = aclmdlRIDestroyRegisterCallback(model, OnModelDestroy, model);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("register model destroy callback failed: modelLabel=%s, modelId=%s, ret=%d", modelLabel.c_str(),
            modelId.c_str(), ret);
    return ret;
  }

  modelContexts_.emplace(model, ModelResourceContext{model, modelId, modelLabel, {}});
  SK_LOGI("register model destroy callback success: modelLabel=%s, modelId=%s", modelLabel.c_str(), modelId.c_str());
  return ACL_SUCCESS;
}

aclError SkResourceManager::AllocForModel(aclmdlRI model, void **addr, size_t bytes) {
  if (addr == nullptr || bytes == 0U || model == nullptr) {
    SK_LOGE("resource alloc invalid param: model=%p, addr=%p, bytes=%zu", model, addr, bytes);
    return ACL_ERROR_INVALID_PARAM;
  }

  aclError ret = aclrtMalloc(addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("resource alloc by aclrtMalloc failed: model=%p, bytes=%zu, ret=%d", model, bytes, ret);
    return ret;
  }
  ret = aclrtMemset(*addr, bytes, 0, bytes);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("resource memset by aclrtMemset failed: model=%p, addr=%p, bytes=%zu, ret=%d", model, *addr, bytes, ret);
    aclrtFree(*addr);
    *addr = nullptr;
    return ret;
  }

  std::lock_guard<std::mutex> lock(resourceMutex_);
  auto it = modelContexts_.find(model);
  if (it == modelContexts_.end()) {
    SK_LOGE("resource alloc failed: model resource context is not registered, model=%p", model);
    aclrtFree(*addr);
    *addr = nullptr;
    return ACL_ERROR_FAILURE;
  }

  auto &context = it->second;
  context.resources.push_back(ResourceRecord{ResourceKind::kDeviceMemory, *addr, bytes});
  SK_LOGI("resource alloc success: modelLabel=%s, modelId=%s, addr=%p, bytes=%zu", context.modelLabel.c_str(),
          context.modelId.c_str(), *addr, bytes);
  return ACL_SUCCESS;
}

aclError SkResourceManager::ReleaseRecord(const ResourceRecord &record) {
  SK_LOGI("release resource record: addr=%p, bytes=%zu", record.addr, record.bytes);
  if (record.addr == nullptr) {
    return ACL_SUCCESS;
  }

  switch (record.kind) {
    case ResourceKind::kDeviceMemory: {
      aclError ret = aclrtFree(record.addr);
      if (ret != ACL_SUCCESS) {
        SK_LOGE("resource free failed: addr=%p, bytes=%zu, ret=%d", record.addr, record.bytes, ret);
      } else {
        SK_LOGI("resource free success: addr=%p, bytes=%zu", record.addr, record.bytes);
      }
      return ret;
    }
    default:
      SK_LOGE("unknown resource kind: addr=%p", record.addr);
      return ACL_ERROR_FAILURE;
  }
}

bool SkResourceManager::ReleaseModelResources(const std::vector<ResourceRecord> &resources) {
  bool releaseSuccess = true;
  for (const auto &record : resources) {
    aclError ret = ReleaseRecord(record);
    if (ret != ACL_SUCCESS) {
      releaseSuccess = false;
    }
  }
  return releaseSuccess;
}

void SkResourceManager::OnModelDestroy(void *userData) {
  aclmdlRI model = static_cast<aclmdlRI>(userData);
  if (model == nullptr) {
    SK_DLOGE("sk resource manager OnModelDestroy invalid userData=%p", userData);
    return;
  }

  ModelResourceContext context;
  {
    std::lock_guard<std::mutex> lock(resourceMutex_);
    auto it = modelContexts_.find(model);
    if (it == modelContexts_.end()) {
      return;
    }
    context = std::move(it->second);
    modelContexts_.erase(it);
  }

  sk::logger::LogContextGuard logContext(context.modelLabel);
  SK_LOGI("sk resource manager OnModelDestroy called: modelLabel=%s, modelId=%s", context.modelLabel.c_str(),
          context.modelId.c_str());

  bool releaseSuccess = ReleaseModelResources(context.resources);
  if (!releaseSuccess) {
    SK_LOGE("release some resources during model destroy failed: modelLabel=%s, modelId=%s", context.modelLabel.c_str(),
            context.modelId.c_str());
    return;
  }

  SK_LOGI("sk resource manager OnModelDestroy completed: modelLabel=%s, modelId=%s", context.modelLabel.c_str(),
          context.modelId.c_str());
}
