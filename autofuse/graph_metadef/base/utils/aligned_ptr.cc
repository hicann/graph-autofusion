/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_metadef/graph/aligned_ptr.h"
#include "common/util/mem_utils.h"
#include "common/ge_common/debug/ge_log.h"
#include "graph/ge_error_codes_af.h"
#include "graph/def_types.h"

namespace af {
AlignedPtr::AlignedPtr(const size_t buffer_size, const size_t alignment) {
  size_t alloc_size = buffer_size;
  if (alignment > 0U) {
    alloc_size = buffer_size + alignment - 1U;
  }
  if ((buffer_size == 0U) || (alloc_size < buffer_size)) {
    GELOGW("[Allocate][Buffer] Allocate empty buffer or overflow, size=%zu, alloc_size=%zu", buffer_size, alloc_size);
    return;
  }

  base_ =
      std::unique_ptr<uint8_t[], AlignedPtr::Deleter>(new (std::nothrow) uint8_t[alloc_size], [](const uint8_t *ptr) {
        delete[] ptr;
        ptr = nullptr;
      });
  if (base_ == nullptr) {
    GELOGW("[Allocate][Buffer] Allocate buffer failed, size=%zu", alloc_size);
    return;
  }

  if (alignment == 0U) {
    aligned_addr_ = base_.get();
  } else {
    const size_t offset = alignment - 1U;
    aligned_addr_ = ge::PtrToPtr<void, uint8_t>(
        ge::ValueToPtr((ge::PtrToValue(ge::PtrToPtr<uint8_t, void>(base_.get())) + offset) & ~offset));
  }
}

std::unique_ptr<uint8_t[], AlignedPtr::Deleter> AlignedPtr::Reset() {
  const auto deleter_func = base_.get_deleter();
  if (deleter_func == nullptr) {
    (void)base_.release();
    return std::unique_ptr<uint8_t[], AlignedPtr::Deleter>(aligned_addr_, nullptr);
  } else {
    const auto base_addr = base_.release();
    return std::unique_ptr<uint8_t[], AlignedPtr::Deleter>(
        aligned_addr_, [deleter_func, base_addr](const uint8_t *) { deleter_func(base_addr); });
  }
}

std::unique_ptr<uint8_t[], AlignedPtr::Deleter> AlignedPtr::Reset(uint8_t *const data,
                                                                  const AlignedPtr::Deleter &delete_func) {
  if ((data == nullptr) || (delete_func == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "data is nullptr or delete_func is nullptr");
    GELOGE(af::FAILED, "[Check][Param] data/delete_func is null");
    return nullptr;
  }
  auto ptr = Reset();
  base_.reset(data);
  base_.get_deleter() = delete_func;
  aligned_addr_ = base_.get();
  return ptr;
}

std::shared_ptr<AlignedPtr> AlignedPtr::BuildFromAllocFunc(const AlignedPtr::Allocator &alloc_func,
                                                           const AlignedPtr::Deleter &delete_func) {
  if ((alloc_func == nullptr) || (delete_func == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "alloc_func or delete_func is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] alloc_func/delete_func is null");
    return nullptr;
  }
  const auto aligned_ptr = MakeShared<AlignedPtr>();
  if (aligned_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create AlignedPtr failed.");
    GELOGE(af::INTERNAL_ERROR, "[Create][AlignedPtr] make shared for AlignedPtr failed");
    return nullptr;
  }
  aligned_ptr->base_.reset();
  alloc_func(aligned_ptr->base_);
  aligned_ptr->base_.get_deleter() = delete_func;
  if (aligned_ptr->base_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "allocate for AlignedPtr failed");
    GELOGE(FAILED, "[Call][AllocFunc] allocate for AlignedPtr failed");
    return nullptr;
  }
  aligned_ptr->aligned_addr_ = aligned_ptr->base_.get();
  return aligned_ptr;
}

std::shared_ptr<AlignedPtr> AlignedPtr::BuildFromData(uint8_t *const data, const AlignedPtr::Deleter &delete_func) {
  if ((data == nullptr) || (delete_func == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "data is nullptr or delete_func is nullptr");
    GELOGE(af::FAILED, "[Check][Param] data/delete_func is null");
    return nullptr;
  }
  const auto aligned_ptr = MakeShared<AlignedPtr>();
  if (aligned_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create AlignedPtr failed.");
    GELOGE(af::INTERNAL_ERROR, "[Create][AlignedPtr] make shared for AlignedPtr failed");
    return nullptr;
  }
  aligned_ptr->base_.reset(data);
  aligned_ptr->base_.get_deleter() = delete_func;
  aligned_ptr->aligned_addr_ = aligned_ptr->base_.get();
  return aligned_ptr;
}
}  // namespace af
