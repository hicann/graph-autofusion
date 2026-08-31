/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tracing_recorder_manager.h"

namespace ge {

TracingRecorderManager &TracingRecorderManager::Instance() {
  static TracingRecorderManager ins;
  return ins;
}

void TracingRecorderManager::RecordDuration(const TracingModule module, const std::string &msg, uint64_t start,
                                            uint64_t duration) const {
  RecordDuration(module, std::vector<std::string>{msg}, start, duration);
}

void TracingRecorderManager::RecordDuration(TracingModule module, const std::vector<std::string> &msgs, uint64_t start,
                                            uint64_t duration) const {
  auto tracing_recorder = GetTracingRecorder(module);
  if (tracing_recorder == nullptr) {
    GELOGW("Cannot find tracing recorder of module[%u]", module);
    return;
  }
  tracing_recorder->RecordDuration(msgs, start, duration);
}

af::Status TracingRecorderManager::Report(const TracingModule module) const {
  auto tracing_recorder = GetTracingRecorder(module);
  GE_ASSERT_NOTNULL(tracing_recorder);
  tracing_recorder->Report();
  return af::SUCCESS;
}

void TracingRecorderManager::InitTracingRecorder() {
  for (int32_t i = 0; i < (static_cast<int32_t>(TracingModule::kTracingModuleEnd) - 1); ++i) {
    tracing_recorders_.emplace_back(std::move(std::make_unique<TracingRecorder>(static_cast<TracingModule>(i))));
  }
  GELOGI("Init tracing record success, size[%zu].", tracing_recorders_.size());
}

TracingRecorderManager::TracingRecorderManager() {
  InitTracingRecorder();
}

TracingRecorder *TracingRecorderManager::GetTracingRecorder(TracingModule module) const {
  GE_ASSERT_TRUE(static_cast<size_t>(module) <= tracing_recorders_.size(), "Module [%zu] should be less than %zu",
                 static_cast<size_t>(module), tracing_recorders_.size());
  return tracing_recorders_[static_cast<int32_t>(module)].get();
}
}  // namespace ge
