/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#include "pgo_stars_parser.h"

#include <securec.h>

#include "pgo_error.h"

namespace codegen::pgo {
namespace {
#pragma pack(push, 4)
struct StarsSocLog {
  uint16_t func_type : 6;
  uint16_t count : 4;
  uint16_t task_type : 6;
  uint16_t reserved0;
  uint16_t stream_id;
  uint16_t task_id;
  uint64_t timestamp;
  uint16_t reserved1;
  uint16_t acc_id : 6;
  uint16_t acsq_id : 10;
  uint32_t reserved2[11];
};

struct StarsSocLogV6 {
  uint16_t func_type : 6;
  uint16_t count : 4;
  uint16_t task_type : 6;
  uint16_t reserved0;
  uint32_t task_id;
  uint64_t timestamp;
  uint16_t reserved1;
  uint16_t acc_id : 6;
  uint16_t acsq_id : 10;
  uint32_t reserved2[3];
};
#pragma pack(pop)

static_assert(sizeof(StarsSocLog) == 64, "Unexpected 910B/310B STARS record size");
static_assert(sizeof(StarsSocLogV6) == 32, "Unexpected V6 STARS record size");

constexpr uint16_t kStreamJudgeBit12 = 0x1000;
constexpr uint16_t kStreamJudgeBit13 = 0x2000;
constexpr uint16_t kStreamJudgeBit15 = 0x8000;
constexpr uint16_t kTaskLow = 0x1fff;
constexpr uint16_t kStreamHigh = 0xe000;
constexpr uint16_t kCommonLow = 0x0fff;
constexpr uint16_t kCommonHigh = 0xf000;
constexpr uint16_t kExpandingLow = 0x7fff;
constexpr uint16_t kLegacyStreamModulus = 0x0800;

void RemapStreamTask(uint16_t raw_stream, uint16_t raw_task, bool expanded, uint32_t *stream, uint32_t *task) {
  uint16_t stream_id = raw_stream;
  uint16_t task_id = raw_task;
  if (expanded) {
    *stream = (stream_id & kStreamJudgeBit15) != 0 ? task_id & kExpandingLow : stream_id & kExpandingLow;
    *task =
        (stream_id & kStreamJudgeBit15) != 0 ? (task_id & kStreamJudgeBit15) | (stream_id & kExpandingLow) : task_id;
    return;
  }
  if ((stream_id & kStreamJudgeBit12) != 0) {
    task_id = (task_id & kTaskLow) | (stream_id & kStreamHigh);
    stream_id %= kLegacyStreamModulus;
  } else if ((stream_id & kStreamJudgeBit13) != 0) {
    const uint16_t raw_task = task_id;
    task_id = (stream_id & kCommonLow) | (task_id & kCommonHigh);
    stream_id = raw_task & kCommonLow;
  }
  *stream = stream_id;
  *task = task_id;
}

bool IsAclEvent(uint16_t type) {
  return type == static_cast<uint16_t>(PgoStarsFuncType::kBegin) || type == 1U;
}

}  // namespace

PgoStarsFormat ResolveStarsFormat(PgoPlatformType platform) {
  switch (platform) {
    case PgoPlatformType::kAscend910B:
    case PgoPlatformType::kAscend310B:
      return PgoStarsFormat::kLegacy64;
    case PgoPlatformType::kAscendV6:
      return PgoStarsFormat::kV6_32;
    case PgoPlatformType::kUnknown:
      return PgoStarsFormat::kUnsupported;
  }
  return PgoStarsFormat::kUnsupported;
}

size_t StarsRecordSize(PgoStarsFormat format) {
  switch (format) {
    case PgoStarsFormat::kLegacy64:
      return sizeof(StarsSocLog);
    case PgoStarsFormat::kV6_32:
      return sizeof(StarsSocLogV6);
    case PgoStarsFormat::kUnsupported:
    default:
      return 0U;
  }
}

bool ParseStarsRecord(const void *buffer, size_t length, uint32_t device_id, PgoStarsEvent *event, bool stream_expand) {
  if (buffer == nullptr || event == nullptr) {
    return false;
  }
  if (length == sizeof(StarsSocLog)) {
    StarsSocLog raw{};
    if (memcpy_s(&raw, sizeof(raw), buffer, sizeof(raw)) != EOK) return false;
    if (!IsAclEvent(raw.func_type)) {
      return false;
    }
    event->func_type = static_cast<PgoStarsFuncType>(raw.func_type);
    event->task_type = raw.task_type;
    event->device_id = device_id;
    RemapStreamTask(raw.stream_id, raw.task_id, stream_expand, &event->stream_id, &event->task_id);
    event->timestamp = raw.timestamp;
    return true;
  }
  if (length == sizeof(StarsSocLogV6)) {
    StarsSocLogV6 raw{};
    if (memcpy_s(&raw, sizeof(raw), buffer, sizeof(raw)) != EOK) return false;
    if (!IsAclEvent(raw.func_type)) {
      return false;
    }
    event->func_type = static_cast<PgoStarsFuncType>(raw.func_type);
    event->task_type = raw.task_type;
    event->device_id = device_id;
    event->stream_id = 0;
    event->task_id = raw.task_id;
    event->timestamp = raw.timestamp;
    return true;
  }
  return false;
}

bool ResolveStarsTaskIdentity(uint16_t raw_stream, uint32_t task_info, bool stream_expand, bool chip_v6,
                              uint32_t *stream_id, uint32_t *task_id) {
  if (stream_id == nullptr || task_id == nullptr) return false;
  if (chip_v6) {
    *stream_id = 0;
    *task_id = task_info;
    return true;
  }
  RemapStreamTask(raw_stream, static_cast<uint16_t>(task_info & UINT16_MAX), stream_expand, stream_id, task_id);
  return true;
}
}  // namespace codegen::pgo
