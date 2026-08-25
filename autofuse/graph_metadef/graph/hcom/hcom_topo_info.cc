/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcom/hcom_topo_info.h"

#include "framework/common/debug/ge_log.h"
namespace af {
Status HcomTopoInfo::SetGroupTopoInfo(const char_t *group, const HcomTopoInfo::TopoInfo &info) {
  if (group == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Group key is nullptr,set failed.");
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] Group key is nullptr,set failed.");
    return ge::GRAPH_FAILED;
  }
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    rank_info_[group] = info;
  }
  GELOGI("Add group %s successfully.", group);
  return ge::GRAPH_SUCCESS;
}

Status HcomTopoInfo::GetGroupRankSize(const char_t *group, int64_t &rank_size) {
  if (group == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Group pointer is nullptr.");
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] group is nullptr.");
    return ge::GRAPH_FAILED;
  }
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto &iter_info = rank_info_.find(group);
    if (iter_info == rank_info_.end()) {
      REPORT_INNER_ERR_MSG("E18888", "Group key [%s] has not been added, get failed.", group);
      GELOGE(ge::GRAPH_FAILED, "[Check][Param] group key [%s] has not been added, get failed.", group);
      return ge::GRAPH_FAILED;
    }
    rank_size = iter_info->second.rank_size;
  }
  return ge::GRAPH_SUCCESS;
}

Status HcomTopoInfo::SetGroupOrderedStream(const int32_t device_id, const char_t *group, void *stream) {
  if (group == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Group is nullptr,set failed.");
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] group is nullptr,set failed.");
    return ge::GRAPH_FAILED;
  }
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    device_id_to_group_to_ordered_stream_[device_id][group] = stream;
  }
  GELOGI("Add device %d group %s stream %p successfully.", device_id, group, stream);
  return ge::GRAPH_SUCCESS;
}

Status HcomTopoInfo::GetGroupOrderedStream(const int32_t device_id, const char_t *group, void *&stream) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = device_id_to_group_to_ordered_stream_.find(device_id);
    if (iter == device_id_to_group_to_ordered_stream_.end()) {
      GELOGW("[Check][Param] device[%d] has not been added, get failed.", device_id);
      return ge::GRAPH_FAILED;
    }

    const auto &iter_inner = iter->second.find(group);
    if (iter_inner == iter->second.end()) {
      GELOGW("[Check][Param] device[%d] group [%s] has not been added, get failed.", device_id, group);
      return ge::GRAPH_FAILED;
    }
    stream = iter_inner->second;
  }

  return ge::GRAPH_SUCCESS;
}

void HcomTopoInfo::UnsetGroupOrderedStream(const int32_t device_id, const char_t *group) {
  const std::lock_guard<std::mutex> lock(mutex_);
  auto iter = device_id_to_group_to_ordered_stream_.find(device_id);
  if (iter != device_id_to_group_to_ordered_stream_.end()) {
    (void)iter->second.erase(group);
    if (iter->second.empty()) {
      (void)device_id_to_group_to_ordered_stream_.erase(iter);
    }
  }
};

HcomTopoInfo::TopoDescs *HcomTopoInfo::GetGroupTopoDesc(const char_t *group) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto &iter_info = rank_info_.find(group);
  if (iter_info == rank_info_.end()) {
    REPORT_INNER_ERR_MSG("E18888", "Group key [%s] has not been added, get failed.", group);
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] group key [%s] has not been added, get failed.", group);
    return nullptr;
  }
  return &iter_info->second.topo_level_descs;
}

Status HcomTopoInfo::GetGroupNotifyHandle(const char_t *group, void *&notify_handle) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto &iter_info = rank_info_.find(group);
    if (iter_info == rank_info_.end()) {
      REPORT_INNER_ERR_MSG("E18888", "Group key [%s] has not been added, get failed.", group);
      GELOGE(ge::GRAPH_FAILED, "[Check][Param] group key [%s] has not been added, get failed.", group);
      return ge::GRAPH_FAILED;
    }
    notify_handle = iter_info->second.notify_handle;
  }
  return ge::GRAPH_SUCCESS;
}

HcomTopoInfo &HcomTopoInfo::Instance() {
  static HcomTopoInfo hcom_topo_info;
  return hcom_topo_info;
}

bool HcomTopoInfo::TryGetGroupTopoInfo(const char_t *group, HcomTopoInfo::TopoInfo &info) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto &iter_info = rank_info_.find(group);
    if (iter_info == rank_info_.end()) {
      return false;
    }
    info = iter_info->second;
  }
  GELOGI("Get existed info of group %s successfully.", group);
  return true;
}

bool HcomTopoInfo::TopoInfoHasBeenSet(const char_t *group) {
  const std::lock_guard<std::mutex> lock(mutex_);
  return rank_info_.find(group) != rank_info_.end();
}

Status HcomTopoInfo::GetGroupLocalWindowSize(const char_t *group, uint64_t &local_window_size) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto &iter_info = rank_info_.find(group);
    if (iter_info == rank_info_.end()) {
      REPORT_INNER_ERR_MSG("E18888", "Group key [%s] has not been added, get failed.", group);
      GELOGE(ge::GRAPH_FAILED, "[Check][Param] group key [%s] has not been added, get failed.", group);
      return ge::GRAPH_FAILED;
    }
    local_window_size = iter_info->second.local_window_size;
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace af
