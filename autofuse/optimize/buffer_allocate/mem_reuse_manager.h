/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_BUFFER_ALLOCATE_MEM_REUSE_MANAGER_H_
#define OPTIMIZE_BUFFER_ALLOCATE_MEM_REUSE_MANAGER_H_

#include <vector>
#include "ascendc_ir_core/ascendc_ir.h"
#include "tensor_mem_defs.h"
namespace optimize {
class MemReuseManager {
 public:
  explicit MemReuseManager(TensorInfoMap &tensor_attr_to_tensor_info, TmpBuffInfoMap &tmp_buf_attr_to_tensor_info)
      : tensor_attr_to_tensor_info_(tensor_attr_to_tensor_info),
        tmp_buf_attr_to_tensor_info_(tmp_buf_attr_to_tensor_info) {}
  void AllocMemBlocks();
  void GetCopyInCopyOutQueNums(size_t &vecin_num, size_t &vecout_num);

 private:
  void MergeTensorByGroupId(std::vector<TensorGroup> &copy_in_groups, std::vector<TensorGroup> &copy_out_groups,
                            std::vector<TensorGroup> &calc_groups) const;
  void AllocForTQue(MemoryType mem_type, std::vector<TensorGroup> &que_groups);
  void AllocForCalc(std::vector<TensorGroup> &calc_groups);
  void CreateMemBlockByType(const TensorGroup *tensor, MemoryType mem_type);
  static bool IsLifetimeOverlap(int64_t start1, int64_t end1, int64_t start2, int64_t end2);
  static MemoryBlock *SelectBestMemoryBlock(const TensorGroup &tensor_group, int64_t last_mem_block_id,
                                            std::vector<MemoryBlock> &memory_blocks);
  void FindCandidateQueBlockWithSizeCheck(MemoryType mem_type, const TensorGroup &tensor_group,
                                          std::vector<MemoryBlock *> &candidate_blocks);
  static void SetQueBufIdToTensorAttr(const MemoryBlock &block);
  void AllocTmpBuff(std::map<af::TmpBuffer *, std::vector<TensorGroup>> &tmp_buff_to_groups);
  void AllocTmpBuffGroup(af::TmpBuffer *tmp_buff, const TensorGroup &group);
  void FindCandidateTmpBuffBlockWithSizeCheck(const TensorGroup &tensor_group,
                                              std::vector<MemoryBlock *> &candidate_blocks);
  // 按类型分组的内存块
  std::unordered_map<MemoryType, std::vector<MemoryBlock>> type_blocks_;

  TensorInfoMap &tensor_attr_to_tensor_info_;
  TmpBuffInfoMap &tmp_buf_attr_to_tensor_info_;
  int64_t buf_id_ = 0;
  int64_t que_id_ = 0;
};
}  // namespace optimize

#endif  // OPTIMIZE_BUFFER_ALLOCATE_MEM_REUSE_MANAGER_H_
