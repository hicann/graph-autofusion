/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_COMMON_LARGE_BM_H_
#define INC_COMMON_LARGE_BM_H_

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>

/* LargeBitmap create a way to generate bitmaps larger than 64bit. */
namespace af {
class LargeBitmap {
 public:
  explicit LargeBitmap(const size_t &size);

  ~LargeBitmap() = default;

  bool operator==(const LargeBitmap &another_bm) const;

  bool operator!=(const LargeBitmap &another_bm) const;

  // set all vector to specific value
  void SetValues(const uint64_t &value);

  // Get the value on position index
  bool GetBit(const size_t &index) const;

  // Set the value on position index to 1
  void SetBit(const size_t &index);

  // Combine two bitmap with the following rule.
  // If one bit of either one of the two bitmaps is 1,
  // the result of final bitmap is 1.
  void Or(const LargeBitmap &another_bm);

  // Combine two bitmap with the following rule.
  // If one bit of either one of the two bitmaps is 0,
  // the result of final bitmap is 0.
  void And(const LargeBitmap &another_bm);

  void ClearBit(size_t bit_idx);

  void ResizeBits(size_t new_size);

 private:
  // Number of element in vector bits
  size_t size_;

  std::vector<uint64_t> bits_;
};
}  // namespace af
#endif  // INC_COMMON_LARGE_BM_H_
