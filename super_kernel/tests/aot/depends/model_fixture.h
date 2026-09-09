/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl.h"

namespace sk::test {
struct TaskSnapshot {
    uint32_t id = 0;
    aclmdlRITaskType type = ACL_MODEL_RI_TASK_DEFAULT;
    bool disabled = false;
    size_t setParamsCount = 0;
    std::string function;
    uint32_t numBlocks = 0;
    std::vector<unsigned char> args;
    std::vector<aclrtLaunchKernelAttr> attributes;
    uintptr_t syncAddress = 0;
    uint64_t syncValue = 0;
};
struct LaunchRecord {
    aclrtStream stream;
    std::string function;
    std::string scope;
};

// Owns the external RI model, streams and task arguments. Function metadata
// remains stable in the process registry for production binary-cache lookups.
// Destroy() runs registered production cleanup before releasing fixture storage.
class Model {
   public:
    Model();
    ~Model();
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;
    aclmdlRI Handle() const;
    aclrtStream AddStream();
    aclmdlRITask AddKernel(aclrtStream stream, const std::string &name);
    aclmdlRITask AddEvent(aclrtStream stream, aclmdlRITaskType type, aclrtEvent event);
    TaskSnapshot Snapshot(aclmdlRITask task) const;
    std::vector<TaskSnapshot> Tasks(aclrtStream stream) const;
    // Stream creation order, then launch order within each stream.
    std::vector<LaunchRecord> Launches() const;
    size_t UpdateAttempts() const;
    size_t SuccessfulUpdates() const;
    void Destroy();

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

size_t OutstandingAllocations();
}  // namespace sk::test
