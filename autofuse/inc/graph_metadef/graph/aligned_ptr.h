/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_ALIGNED_PTR_H_
#define GE_ALIGNED_PTR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <functional>

namespace af {
class AlignedPtr {
 public:
  using Deleter = std::function<void(uint8_t *)>;
  using Allocator = std::function<void(std::unique_ptr<uint8_t[], Deleter> &base_addr)>;
  explicit AlignedPtr(const size_t buffer_size, const size_t alignment = 16U);
  AlignedPtr() = default;
  ~AlignedPtr() = default;
  AlignedPtr(const AlignedPtr &) = delete;
  AlignedPtr(AlignedPtr &&) = delete;
  AlignedPtr &operator=(const AlignedPtr &) = delete;
  AlignedPtr &operator=(AlignedPtr &&) = delete;

  const uint8_t *Get() const {
    return aligned_addr_;
  }
  uint8_t *MutableGet() {
    return aligned_addr_;
  }
  std::unique_ptr<uint8_t[], AlignedPtr::Deleter> Reset();
  std::unique_ptr<uint8_t[], AlignedPtr::Deleter> Reset(uint8_t *const data, const AlignedPtr::Deleter &delete_func);

  static std::shared_ptr<AlignedPtr> BuildFromAllocFunc(const AlignedPtr::Allocator &alloc_func,
                                                        const AlignedPtr::Deleter &delete_func);
  static std::shared_ptr<AlignedPtr> BuildFromData(uint8_t *const data, const AlignedPtr::Deleter &delete_func);

 private:
  std::unique_ptr<uint8_t[], AlignedPtr::Deleter> base_ = nullptr;
  uint8_t *aligned_addr_ = nullptr;
};
}  // namespace af
#endif  // GE_ALIGNED_PTR_H_
