/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_CODEGEN_PGO_PGO_TASK_RECORD_H_
#define AUTOFUSE_CODEGEN_PGO_PGO_TASK_RECORD_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "pgo_stars_parser.h"
#include "pgo_timestamp.h"

namespace codegen::pgo {

constexpr size_t kPgoMeasureSamples = 20U;

struct PgoTaskRecord {
  std::string candidate_key;
  uint32_t device_id = 0;
  uint32_t stream_id = 0;
  uint64_t launch_sequence = 0;
  uint32_t task_id = 0;
  uint64_t start_tick = 0;
  uint64_t end_tick = 0;
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  uint64_t duration_ns = 0;
};

struct PgoBatchCandidate {
  std::string candidate_key;
  std::vector<uint64_t> launch_sequences;
};

// Pair STARS begin/end records using the same (device, stream, task) identity
// used by MSPTI. Records are consumed in arrival order with FIFO queues per
// task key; malformed sequences (end without begin or an unmatched begin) are
// rejected while repeated begins are preserved for reused task IDs.
int PairPgoTaskEvents(const std::vector<PgoStarsEvent> &events, const std::vector<uint64_t> &launch_sequences,
                      const std::string &candidate_key, const PgoDeviceTimeInfo &time_info,
                      std::vector<PgoTaskRecord> *records);

int PairPgoTaskEventsBatch(const std::vector<PgoStarsEvent> &events, const std::vector<PgoBatchCandidate> &candidates,
                           const PgoDeviceTimeInfo &time_info, std::vector<std::vector<PgoTaskRecord>> *records);

int CalculatePgoDurationNs(const std::vector<PgoTaskRecord> &records, uint64_t *duration_ns);

// A serial fallback is safe only when all records belong to one stream.  With
// multiple streams and no runtime-track identity, candidate ownership cannot
// be reconstructed deterministically.
bool CanUseSerialPgoEvents(const std::vector<PgoStarsEvent> &events);

}  // namespace codegen::pgo

#endif  // AUTOFUSE_CODEGEN_PGO_PGO_TASK_RECORD_H_
