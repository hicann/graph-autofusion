/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_BUFFER_ALLOCATE_TENSOR_MEM_DEFS_H_
#define OPTIMIZE_BUFFER_ALLOCATE_TENSOR_MEM_DEFS_H_
#include <vector>
#include <string>
#include "ascendc_ir_core/ascendc_ir.h"

namespace optimize {
constexpr auto kAttrNameNoReuseOutputIndices = "_no_reuse_output_indices";
constexpr auto kAttrNameNoReuseInputs = "_no_reuse_inputs";
constexpr int64_t kDbBufNum = 2;
// 将内存按照大小分为3个档位
enum class MemorySizeLevel : int32_t { kScalar = 0, kMedium, kLargest };

// tensor块大小信息
struct TensorInfo {
  int64_t group_id{-1};
  int64_t allocation_order{-1};
  af::AscTensorAttr *output_tensor_attr{nullptr};
  int64_t life_start{-1};
  int64_t life_end{-1};
  af::Position mem_position{af::Position::kPositionInvalid};
  MemorySizeLevel size_level{MemorySizeLevel::kLargest};
  bool is_reusable{true};          // 输出可以被复用
  bool is_can_reuse_others{true};  // 可以复用其他tensor
  std::set<int64_t> loop_axes;
  int64_t buf_num{kDbBufNum};

  std::string ToString() const {
    std::stringstream ss;
    ss << "TensorInfo{";
    ss << "group_id=" << group_id << ", ";
    ss << "life_start=" << life_start << ", ";
    ss << "life_end=" << life_end << ", ";
    ss << "buf_num=" << buf_num << ", ";
    ss << "is_reusable=" << std::boolalpha << is_reusable << ", ";
    ss << "is_can_reuse_others=" << std::boolalpha << is_can_reuse_others << ", ";
    ss << "mem_position=" << static_cast<int32_t>(mem_position) << ", ";
    ss << "size_level=" << static_cast<int32_t>(size_level) << ", ";
    ss << "loop_axes={";
    for (auto it = loop_axes.begin(); it != loop_axes.end(); ++it) {
      if (it != loop_axes.begin()) {
        ss << ", ";
      }
      ss << *it;
    }
    ss << "}";
    ss << "}";
    return ss.str();
  }
};

// 需要绑定进行内存分配的tensor链
struct TensorGroup {
  int64_t group_id{-1};
  int64_t allocation_order{-1};
  std::vector<TensorInfo *> grouped_tensors;  // group中的tensor，会绑定生命周期
  std::set<int64_t> merged_loop_axes;
  int64_t merged_life_start;
  int64_t merged_life_end;
  MemorySizeLevel max_size_level;  // 链中最大的大小档位
  af::Position mem_position;
  bool group_is_reusable;
  bool group_is_can_reuse_others;
};

enum class MemoryType { kCopyIn, kCopyOut, kCalc, kTmpBuff, kLoopTmpBuff };
struct MemoryBlock {
  int64_t id;
  MemoryType mem_type;
  MemorySizeLevel max_size_level;                  // 块中最大的大小档位
  std::vector<const TensorGroup *> tensor_groups;  // 块中包含的tensor链
};

using TensorInfoMap = std::map<af::AscTensorAttr *, TensorInfo>;
using TmpBuffInfoMap = std::map<af::TmpBuffer *, TensorInfo>;
}  // namespace optimize
#endif  // OPTIMIZE_BUFFER_ALLOCATE_TENSOR_MEM_DEFS_H_
