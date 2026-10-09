/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_CODEGEN_PGO_PGO_STARS_PARSER_H_
#define AUTOFUSE_CODEGEN_PGO_PGO_STARS_PARSER_H_

#include <cstddef>
#include <cstdint>

#include "pgo_capability.h"

namespace codegen::pgo {

enum class PgoStarsFuncType : uint16_t { kBegin = 0, kEnd = 1 };

struct PgoStarsEvent {
  PgoStarsFuncType func_type;
  uint16_t task_type = 0;
  uint32_t device_id = 0;
  uint32_t stream_id = 0;
  uint32_t task_id = 0;
  uint64_t timestamp = 0;
};

enum class PgoStarsFormat : uint8_t { kLegacy64 = 0, kV6_32 = 1, kUnsupported = 2 };

// The STARS record layout is selected by the resolved platform.  Unknown or
// unavailable platforms must not be guessed, because interpreting a 32-byte
// record as 64 bytes can pair data from adjacent records and produce invalid
// timings.
PgoStarsFormat ResolveStarsFormat(PgoPlatformType platform);
size_t StarsRecordSize(PgoStarsFormat format);

// Parse one ACSQ STARS record. FFTS records are intentionally rejected by
// their distinct record format; task type classification is left to the
// runtime-track/serial collector, matching MSPTI's dynamic kernel mapping.
// `stream_expand` selects the same stream/task remapping used by MSPTI's
// StarsCommon for expanded streams.  It defaults to the normal 910B/310B
// mapping; V6 records carry a globally unique task id and always use stream 0.
bool ParseStarsRecord(const void *buffer, size_t length, uint32_t device_id, PgoStarsEvent *event,
                      bool stream_expand = false);

bool ResolveStarsTaskIdentity(uint16_t raw_stream, uint32_t task_info, bool stream_expand, bool chip_v6,
                              uint32_t *stream_id, uint32_t *task_id);

}  // namespace codegen::pgo

#endif  // AUTOFUSE_CODEGEN_PGO_PGO_STARS_PARSER_H_
