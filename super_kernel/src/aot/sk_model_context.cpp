/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sk_model_context.cpp
 * \brief Host-only per-model identity implementation.
 */

#include "sk_model_context.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <utility>

#include "sk_log.h"

namespace {

constexpr const char *MODEL_LABEL_PREFIX = "model_";
constexpr const char *DEFAULT_MODEL_ID = "unknown";

struct ModelIdentity {
  std::string modelId;
  std::string modelLabel;
};

class ModelContextState {
 public:
  // aclskOptimize treats every invocation as a distinct model instance. The RTS
  // model id is the stable prefix; this counter disambiguates repeated appearances
  // of the same RTS model id, regardless of whether the aclmdlRI address is reused.
  inline static std::map<uint32_t, uint64_t> modelCallCounter_{};
  inline static std::mutex modelCounterMutex_{};
  // Keep TLS storage trivial; model strings are owned by the stack RAII context.
  inline static thread_local const SkModelContext *currentContext_ = nullptr;
};

bool GetRtsModelId(aclmdlRI model, uint32_t &rtsModelId) {
  if (model == nullptr) {
    SK_DLOGE("Failed to get model id: model is nullptr");
    return false;
  }
  aclError ret = aclmdlRIGetId(model, &rtsModelId);
  if (ret != ACL_SUCCESS) {
    SK_DLOGE("Failed to get model id, ret=%d", ret);
    return false;
  }
  return true;
}

ModelIdentity BuildModelIdentity(aclmdlRI model, bool bumpCounter) {
  if (model == nullptr) {
    SK_DLOGE("Failed to make model identity: model is nullptr");
    return {DEFAULT_MODEL_ID, BuildModelLabel(DEFAULT_MODEL_ID)};
  }

  uint32_t rtsModelId = 0U;
  if (!GetRtsModelId(model, rtsModelId)) {
    return {DEFAULT_MODEL_ID, BuildModelLabel(DEFAULT_MODEL_ID)};
  }

  uint64_t callCount = 0U;
  if (bumpCounter) {
    std::lock_guard<std::mutex> lock(ModelContextState::modelCounterMutex_);
    callCount = ++ModelContextState::modelCallCounter_[rtsModelId];
  }

  std::string uniqueModelId = std::to_string(rtsModelId) + "_" + std::to_string(callCount);
  return {uniqueModelId, BuildModelLabel(uniqueModelId)};
}

}  // namespace

std::string BuildModelLabel(const std::string &modelId) {
  return std::string(MODEL_LABEL_PREFIX) + (modelId.empty() ? DEFAULT_MODEL_ID : modelId);
}

std::string GetCurrentModelId() {
  const auto *context = ModelContextState::currentContext_;
  if (context == nullptr) {
    SK_DLOGE("Failed to get current model id: no active SkModelContext");
    return DEFAULT_MODEL_ID;
  }
  return context->modelId_;
}

std::string GetCurrentModelLabel() {
  const auto *context = ModelContextState::currentContext_;
  if (context == nullptr) {
    SK_DLOGE("Failed to get current model label: no active SkModelContext");
    return BuildModelLabel(DEFAULT_MODEL_ID);
  }
  return context->modelLabel_;
}

SkModelContext::SkModelContext(aclmdlRI model) {
  if (ModelContextState::currentContext_ != nullptr) {
    SK_DLOGE("Nested SkModelContext is unsupported and will overwrite current model context");
  }
  auto identity = BuildModelIdentity(model, true);
  modelId_ = std::move(identity.modelId);
  modelLabel_ = std::move(identity.modelLabel);
  ModelContextState::currentContext_ = this;
}

SkModelContext::~SkModelContext() {
  if (ModelContextState::currentContext_ != this) {
    SK_DLOGE("SkModelContext destruction order mismatch, current model context has changed");
    return;
  }
  ModelContextState::currentContext_ = nullptr;
}
