/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mem_reuse_manager.h"

namespace optimize {
size_t kDbReuseThreshold = 2UL;

static bool TensorGroupLifeLess(const TensorGroup &lhs, const TensorGroup &rhs) {
  if (lhs.merged_life_start != rhs.merged_life_start) {
    return lhs.merged_life_start < rhs.merged_life_start;
  }
  return std::make_pair(lhs.allocation_order, lhs.group_id) < std::make_pair(rhs.allocation_order, rhs.group_id);
}

bool MemReuseManager::IsLifetimeOverlap(int64_t start1, int64_t end1, int64_t start2, int64_t end2) {
  if (end1 == std::numeric_limits<int64_t>::max() || end2 == std::numeric_limits<int64_t>::max()) {
    return true;
  }
  return !(end1 < start2 || end2 < start1);
}

void MemReuseManager::MergeTensorByGroupId(std::vector<TensorGroup> &copy_in_groups,
                                           std::vector<TensorGroup> &copy_out_groups,
                                           std::vector<TensorGroup> &calc_groups) const {
  std::unordered_map<int64_t, TensorGroup> temp_groups;
  for (auto &info : tensor_attr_to_tensor_info_) {
    auto cur_tensor = &info.second;

    const auto key = cur_tensor->group_id;
    auto [it, is_new] = temp_groups.try_emplace(key);
    TensorGroup &group = it->second;
    if (is_new) {
      group.group_id = cur_tensor->group_id;
      group.allocation_order = cur_tensor->allocation_order;
      group.grouped_tensors = {cur_tensor};
      group.merged_life_start = cur_tensor->life_start;
      group.merged_life_end = cur_tensor->life_end;
      group.max_size_level = cur_tensor->size_level;
      group.mem_position = cur_tensor->mem_position;
      group.group_is_reusable = cur_tensor->is_reusable;
      group.group_is_can_reuse_others = cur_tensor->is_can_reuse_others;
      group.merged_loop_axes = cur_tensor->loop_axes;
    } else {
      // 已有组：更新属性
      group.grouped_tensors.push_back(cur_tensor);
      // 更新合并生命周期
      group.merged_life_start = std::min(group.merged_life_start, cur_tensor->life_start);
      group.merged_life_end = std::max(group.merged_life_end, cur_tensor->life_end);
      // 更新最大size_level
      group.max_size_level = std::max(group.max_size_level, cur_tensor->size_level);
      group.merged_loop_axes.insert(cur_tensor->loop_axes.begin(), cur_tensor->loop_axes.end());
      // 校验复用属性一致性（只要有一个tensor不一致，组属性就设为false）
      if (cur_tensor->is_reusable != group.group_is_reusable ||
          cur_tensor->is_can_reuse_others != group.group_is_can_reuse_others) {
        group.group_is_reusable = false;
        group.group_is_can_reuse_others = false;
      }
      // 前序vec复用后面的que
      if ((group.mem_position == af::Position::kPositionVecCalc) && (group.mem_position != cur_tensor->mem_position)) {
        group.mem_position = cur_tensor->mem_position;
      }
    }
    GELOGD("[MemReuse] Refresh group id [%ld] with life_time_range:[%ld, %ld] size_level:[%d].", cur_tensor->group_id,
           group.merged_life_start, group.merged_life_end, static_cast<int32_t>(group.max_size_level));
  }
  for (auto &[key, group] : temp_groups) {
    (void)key;
    switch (group.mem_position) {
      case af::Position::kPositionVecIn:
        copy_in_groups.push_back(std::move(group));
        break;
      case af::Position::kPositionVecOut:
        copy_out_groups.push_back(std::move(group));
        break;
      default:
        calc_groups.push_back(std::move(group));
        break;
    }
  }
}

void MemReuseManager::AllocMemBlocks() {
  std::vector<TensorGroup> copy_in_groups;
  std::vector<TensorGroup> copy_out_groups;
  std::vector<TensorGroup> calc_groups;
  std::map<af::TmpBuffer *, std::vector<TensorGroup>> tmp_buff_to_groups;
  MergeTensorByGroupId(copy_in_groups, copy_out_groups, calc_groups);
  AllocForTQue(MemoryType::kCopyIn, copy_in_groups);
  AllocForTQue(MemoryType::kCopyOut, copy_out_groups);
  AllocTmpBuff(tmp_buff_to_groups);
  AllocForCalc(calc_groups);
  // 根据复用关系刷新tque/tbuf
  for (const auto &iter : type_blocks_) {
    for (const auto &block : iter.second) {
      SetQueBufIdToTensorAttr(block);
    }
  }
}

void MemReuseManager::GetCopyInCopyOutQueNums(size_t &vecin_num, size_t &vecout_num) {
  vecin_num = type_blocks_[MemoryType::kCopyIn].size();
  vecout_num = type_blocks_[MemoryType::kCopyOut].size();
}

void MemReuseManager::CreateMemBlockByType(const TensorGroup *tensor, MemoryType mem_type) {
  MemoryBlock new_block;
  if (mem_type == MemoryType::kCopyOut || mem_type == MemoryType::kCopyIn) {
    new_block.id = que_id_++;
  } else {
    new_block.id = buf_id_++;
  }
  new_block.mem_type = mem_type;
  new_block.max_size_level = tensor->max_size_level;
  new_block.tensor_groups.push_back(tensor);
  type_blocks_[mem_type].push_back(new_block);
  GELOGD("[MemReuse] create new mem block with type[%d] id[%d] for group[%ld].", static_cast<int>(mem_type),
         new_block.id, tensor->group_id);
}

void MemReuseManager::FindCandidateQueBlockWithSizeCheck(MemoryType mem_type, const TensorGroup &tensor_group,
                                                         std::vector<MemoryBlock *> &candidate_blocks) {
  for (auto &block : type_blocks_[mem_type]) {
    bool is_reuse_ok =
        std::all_of(block.tensor_groups.begin(), block.tensor_groups.end(), [&tensor_group](const TensorGroup *info) {
          if (info->mem_position == af::Position::kPositionVecIn ||
              info->mem_position == af::Position::kPositionVecOut) {
            return (info->group_is_reusable) && (info->merged_life_end < tensor_group.merged_life_start) &&
                   (info->merged_loop_axes.size() == 1UL &&
                    info->merged_loop_axes == tensor_group.merged_loop_axes);  // 存在多层循环场景不支持融合
          }
          return (info->group_is_reusable) &&
                 !IsLifetimeOverlap(tensor_group.merged_life_start, tensor_group.merged_life_end,
                                    info->merged_life_start, info->merged_life_end);
        });
    bool is_size_ok = true;
    if (tensor_group.max_size_level > MemorySizeLevel::kScalar) {
      is_size_ok = std::any_of(block.tensor_groups.begin(), block.tensor_groups.end(), [](const TensorGroup *info) {
        return info->max_size_level == MemorySizeLevel::kLargest;
      });
    }
    if (is_reuse_ok && is_size_ok) {
      candidate_blocks.push_back(&block);
    }
  }
}

void MemReuseManager::FindCandidateTmpBuffBlockWithSizeCheck(const TensorGroup &tensor_group,
                                                             std::vector<MemoryBlock *> &candidate_blocks) {
  if (!tensor_group.group_is_can_reuse_others && !tensor_group.group_is_reusable) {
    return;
  }
  for (auto &block : type_blocks_[MemoryType::kTmpBuff]) {
    std::sort(block.tensor_groups.begin(), block.tensor_groups.end(),
              [](const TensorGroup *a, const TensorGroup *b) { return a->max_size_level < b->max_size_level; });
    bool is_reuse_ok =
        std::all_of(block.tensor_groups.begin(), block.tensor_groups.end(), [&tensor_group](const TensorGroup *info) {
          return (!IsLifetimeOverlap(tensor_group.merged_life_start, tensor_group.merged_life_end,
                                     info->merged_life_start, info->merged_life_end));
        });
    if (is_reuse_ok) {
      candidate_blocks.push_back(&block);
    }
  }
}

MemoryBlock *MemReuseManager::SelectBestMemoryBlock(const TensorGroup &tensor_group, int64_t last_mem_block_id,
                                                    std::vector<MemoryBlock> &memory_blocks) {
  std::vector<MemoryBlock *> candidate_blocks;
  for (auto &block : memory_blocks) {
    bool is_reuse_ok =
        std::all_of(block.tensor_groups.begin(), block.tensor_groups.end(), [&tensor_group](const TensorGroup *info) {
          return (info->group_is_reusable) &&
                 !IsLifetimeOverlap(tensor_group.merged_life_start, tensor_group.merged_life_end,
                                    info->merged_life_start, info->merged_life_end);
        });
    if (is_reuse_ok) {
      candidate_blocks.push_back(&block);
    }
  }

  if (candidate_blocks.empty()) {
    return nullptr;
  }
  // 生命周期相邻的tque采取间隔复用，防止流水冲突
  std::vector<MemoryBlock *> other_candidates;
  MemoryBlock *only_last_block = nullptr;
  for (auto *block : candidate_blocks) {
    if (block->id == last_mem_block_id) {
      only_last_block = block;
    } else {
      other_candidates.push_back(block);
    }
  }

  if (!other_candidates.empty()) {
    std::sort(
        other_candidates.begin(), other_candidates.end(), [&tensor_group](const MemoryBlock *a, const MemoryBlock *b) {
          int32_t diff_a = static_cast<int32_t>(a->max_size_level) - static_cast<int32_t>(tensor_group.max_size_level);
          int32_t diff_b = static_cast<int32_t>(b->max_size_level) - static_cast<int32_t>(tensor_group.max_size_level);
          return std::abs(diff_a) < std::abs(diff_b);
        });
    return other_candidates[0];
  }

  // 仅上一个块是候选块, 可用块在阈值内时才间隔复用
  if (only_last_block != nullptr && memory_blocks.size() > kDbReuseThreshold) {
    return only_last_block;
  }
  return nullptr;
}

void MemReuseManager::AllocForTQue(MemoryType mem_type, std::vector<TensorGroup> &que_groups) {
  std::sort(que_groups.begin(), que_groups.end(), TensorGroupLifeLess);
  int64_t last_block_id = -1;
  for (const auto &tensor : que_groups) {
    // 被标记为不能复用，需要直接创建
    if (!tensor.group_is_can_reuse_others) {
      CreateMemBlockByType(&tensor, mem_type);
      continue;
    }

    // 尝试找到合适的分组
    MemoryBlock *best_block = SelectBestMemoryBlock(tensor, last_block_id, type_blocks_[mem_type]);
    if (best_block != nullptr) {
      // 更新内存块信息
      best_block->tensor_groups.push_back(&tensor);
      best_block->max_size_level = std::max(best_block->max_size_level, tensor.max_size_level);
      last_block_id = best_block->id;
      GELOGD("[MemReuse] reuse mem block with type[%d] for group[%ld].", static_cast<int>(mem_type), tensor.group_id);
    } else {
      CreateMemBlockByType(&tensor, mem_type);
      last_block_id = type_blocks_[mem_type].back().id;
    }
  }
}

void MemReuseManager::AllocForCalc(std::vector<TensorGroup> &calc_groups) {
  std::sort(calc_groups.begin(), calc_groups.end(), TensorGroupLifeLess);
  for (const auto &tensor_group : calc_groups) {
    // 不能复用别人的tensor，直接创建新块
    if (!tensor_group.group_is_can_reuse_others) {
      CreateMemBlockByType(&tensor_group, MemoryType::kCalc);
      continue;
    }

    // 尝试找到合适的分组
    std::vector<MemoryBlock *> candidate_blocks;
    FindCandidateTmpBuffBlockWithSizeCheck(tensor_group, candidate_blocks);
    if (candidate_blocks.empty()) {
      FindCandidateQueBlockWithSizeCheck(MemoryType::kCopyIn, tensor_group, candidate_blocks);
      FindCandidateQueBlockWithSizeCheck(MemoryType::kCopyOut, tensor_group, candidate_blocks);
      // vecout 暂不支持
      for (auto &block : type_blocks_[MemoryType::kCalc]) {
        bool is_reuse_ok = std::all_of(
            block.tensor_groups.begin(), block.tensor_groups.end(), [&tensor_group](const TensorGroup *info) {
              return !IsLifetimeOverlap(tensor_group.merged_life_start, tensor_group.merged_life_end,
                                        info->merged_life_start, info->merged_life_end);
            });
        if (is_reuse_ok) {
          candidate_blocks.push_back(&block);
        }
      }
    }

    if (candidate_blocks.empty()) {
      CreateMemBlockByType(&tensor_group, MemoryType::kCalc);
    } else {
      // 按照大小接近进行排序
      std::sort(candidate_blocks.begin(), candidate_blocks.end(),
                [&tensor_group](const MemoryBlock *a, const MemoryBlock *b) {
                  int32_t diff_a =
                      static_cast<int32_t>(a->max_size_level) - static_cast<int32_t>(tensor_group.max_size_level);
                  int32_t diff_b =
                      static_cast<int32_t>(b->max_size_level) - static_cast<int32_t>(tensor_group.max_size_level);
                  return std::abs(diff_a) < std::abs(diff_b);
                });

      // 更新内存块信息
      candidate_blocks[0]->tensor_groups.push_back(&tensor_group);
      candidate_blocks[0]->max_size_level = std::max(candidate_blocks[0]->max_size_level, tensor_group.max_size_level);
      GELOGD("[MemReuse] reuse mem block with type[%d] id[%d] for group[%ld].",
             static_cast<int>(candidate_blocks[0]->mem_type), candidate_blocks[0]->id, tensor_group.group_id);
    }
  }
}

void MemReuseManager::SetQueBufIdToTensorAttr(const MemoryBlock &block) {
  for (auto tensor_group : block.tensor_groups) {
    for (auto tensor_info : tensor_group->grouped_tensors) {
      if (block.mem_type == MemoryType::kCopyIn || block.mem_type == MemoryType::kCopyOut) {
        tensor_info->output_tensor_attr->mem.alloc_type = af::AllocType::kAllocTypeQueue;
        tensor_info->output_tensor_attr->que.id = block.id;
        tensor_info->output_tensor_attr->que.depth = 2;  // 队列深度为2
        tensor_info->output_tensor_attr->que.buf_num = tensor_info->buf_num;
        continue;
      }
      if (tensor_info->output_tensor_attr != nullptr) {
        tensor_info->output_tensor_attr->mem.alloc_type = af::AllocType::kAllocTypeBuffer;
        tensor_info->output_tensor_attr->buf.id = block.id;
        tensor_info->output_tensor_attr->mem.reuse_id = af::kIdNone;
      }
    }
  }
}

void MemReuseManager::AllocTmpBuff(std::map<af::TmpBuffer *, std::vector<TensorGroup>> &tmp_buff_to_groups) {
  for (auto &info : tmp_buf_attr_to_tensor_info_) {
    auto cur_tensor = &info.second;
    TensorGroup group;
    group.group_id = cur_tensor->group_id;
    group.allocation_order = cur_tensor->allocation_order;
    group.grouped_tensors = {cur_tensor};
    group.merged_life_start = cur_tensor->life_start;
    group.merged_life_end = cur_tensor->life_end;
    group.max_size_level = cur_tensor->size_level;
    group.mem_position = cur_tensor->mem_position;
    group.group_is_can_reuse_others = cur_tensor->is_can_reuse_others;
    group.group_is_reusable = cur_tensor->is_reusable;
    tmp_buff_to_groups[info.first].push_back(std::move(group));
  }
  std::vector<std::pair<af::TmpBuffer *, const TensorGroup *>> ordered_groups;
  for (const auto &info : tmp_buff_to_groups) {
    for (const auto &group : info.second) {
      ordered_groups.emplace_back(info.first, &group);
    }
  }
  std::sort(ordered_groups.begin(), ordered_groups.end(),
            [](const auto &lhs, const auto &rhs) { return TensorGroupLifeLess(*lhs.second, *rhs.second); });
  for (const auto &[tmp_buff, group] : ordered_groups) {
    AllocTmpBuffGroup(tmp_buff, *group);
  }
}

void MemReuseManager::AllocTmpBuffGroup(af::TmpBuffer *tmp_buff, const TensorGroup &group) {
  tmp_buff->mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  if (group.group_id != -1) {
    tmp_buff->id = buf_id_;
    CreateMemBlockByType(&group, MemoryType::kLoopTmpBuff);
    return;
  }
  std::vector<MemoryBlock *> candidate_blocks;
  FindCandidateTmpBuffBlockWithSizeCheck(group, candidate_blocks);
  if (candidate_blocks.empty()) {
    tmp_buff->id = buf_id_;
    CreateMemBlockByType(&group, MemoryType::kTmpBuff);
    return;
  }
  tmp_buff->id = candidate_blocks[0]->id;
  candidate_blocks[0]->tensor_groups.push_back(&group);
  GELOGD("[MemReuse] reuse mem block with type[%d] id[%d] for group[%ld].",
         static_cast<int>(candidate_blocks[0]->mem_type), candidate_blocks[0]->id, group.group_id);
}
}  // namespace optimize
