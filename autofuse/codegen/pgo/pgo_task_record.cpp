/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#include "pgo_task_record.h"

#include <algorithm>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <tuple>
#include <utility>

namespace codegen::pgo {
namespace {
using TaskKey = std::tuple<uint32_t, uint32_t, uint32_t>;

int PairRawPgoTaskEvents(const std::vector<PgoStarsEvent> &events, const PgoDeviceTimeInfo &time_info,
                         std::vector<PgoTaskRecord> *records) {
  if (records == nullptr) return -1;
  records->clear();
  std::map<TaskKey, std::deque<PgoStarsEvent>> pending;
  for (const auto &event : events) {
    const TaskKey key{event.device_id, event.stream_id, event.task_id};
    auto &queue = pending[key];
    if (event.func_type == PgoStarsFuncType::kBegin) {
      queue.push_back(event);
      continue;
    }
    if (event.func_type != PgoStarsFuncType::kEnd || queue.empty()) return -1;
    const PgoStarsEvent begin = queue.front();
    queue.pop_front();
    uint64_t duration_ns = 0;
    if (ConvertTaskDuration(begin.timestamp, event.timestamp, time_info, &duration_ns) != 0 || duration_ns == 0) {
      return -1;
    }
    PgoTaskRecord record;
    record.device_id = begin.device_id;
    record.stream_id = begin.stream_id;
    record.task_id = begin.task_id;
    record.start_tick = begin.timestamp;
    record.end_tick = event.timestamp;
    if (ConvertCounterToRealTime(begin.timestamp, time_info, &record.start_ns) != 0 ||
        ConvertCounterToRealTime(event.timestamp, time_info, &record.end_ns) != 0) {
      return -1;
    }
    record.duration_ns = duration_ns;
    records->push_back(std::move(record));
  }
  for (const auto &item : pending) {
    if (!item.second.empty()) return -1;
  }
  std::stable_sort(records->begin(), records->end(), [](const PgoTaskRecord &lhs, const PgoTaskRecord &rhs) {
    if (lhs.start_tick != rhs.start_tick) return lhs.start_tick < rhs.start_tick;
    if (lhs.end_tick != rhs.end_tick) return lhs.end_tick < rhs.end_tick;
    if (lhs.stream_id != rhs.stream_id) return lhs.stream_id < rhs.stream_id;
    return lhs.task_id < rhs.task_id;
  });
  return 0;
}
}  // namespace

bool CanUseSerialPgoEvents(const std::vector<PgoStarsEvent> &events) {
  if (events.empty()) return false;
  const uint32_t stream_id = events.front().stream_id;
  for (const auto &event : events) {
    if (event.stream_id != stream_id) return false;
  }
  return true;
}

int PairPgoTaskEvents(const std::vector<PgoStarsEvent> &events, const std::vector<uint64_t> &launch_sequences,
                      const std::string &candidate_key, const PgoDeviceTimeInfo &time_info,
                      std::vector<PgoTaskRecord> *records) {
  if (records == nullptr || launch_sequences.size() != kPgoMeasureSamples) return -1;
  if (PairRawPgoTaskEvents(events, time_info, records) != 0 || records->size() != kPgoMeasureSamples) return -1;
  for (size_t index = 0U; index < records->size(); ++index) {
    (*records)[index].candidate_key = candidate_key;
    (*records)[index].launch_sequence = launch_sequences[index];
  }
  return 0;
}

int PairPgoTaskEventsBatch(const std::vector<PgoStarsEvent> &events, const std::vector<PgoBatchCandidate> &candidates,
                           const PgoDeviceTimeInfo &time_info, std::vector<std::vector<PgoTaskRecord>> *records) {
  if (records == nullptr || candidates.empty()) return -1;
  // Candidate ownership is recovered from serial launch order because the
  // internal PGO protocol deliberately does not depend on correlation ids.
  // Interleaved streams cannot be assigned deterministically after filtering,
  // so fail closed instead of returning a misleading candidate duration.
  if (!CanUseSerialPgoEvents(events)) return -1;
  size_t expected_records = 0U;
  for (const auto &candidate : candidates) {
    if (candidate.launch_sequences.size() != kPgoMeasureSamples) return -1;
    expected_records += candidate.launch_sequences.size();
  }
  std::vector<PgoTaskRecord> paired_records;
  if (PairRawPgoTaskEvents(events, time_info, &paired_records) != 0 || paired_records.size() != expected_records)
    return -1;
  records->clear();
  records->resize(candidates.size());
  size_t offset = 0U;
  for (size_t candidate_index = 0U; candidate_index < candidates.size(); ++candidate_index) {
    auto &candidate_records = (*records)[candidate_index];
    candidate_records.insert(candidate_records.end(), paired_records.begin() + offset,
                             paired_records.begin() + offset + kPgoMeasureSamples);
    for (size_t sample = 0U; sample < kPgoMeasureSamples; ++sample) {
      candidate_records[sample].candidate_key = candidates[candidate_index].candidate_key;
      candidate_records[sample].launch_sequence = candidates[candidate_index].launch_sequences[sample];
    }
    offset += kPgoMeasureSamples;
  }
  return 0;
}

int CalculatePgoDurationNs(const std::vector<PgoTaskRecord> &records, uint64_t *duration_ns) {
  constexpr size_t kRetainedSamples = 5U;
  if (duration_ns == nullptr || records.size() <= kRetainedSamples) return -1;
  std::vector<uint64_t> durations;
  durations.reserve(records.size());
  for (const auto &record : records) durations.push_back(record.duration_ns);
  std::sort(durations.begin(), durations.end(), std::greater<uint64_t>());
  __uint128_t total = 0;
  for (size_t index = 1U; index <= kRetainedSamples; ++index) total += durations[index];
  const __uint128_t average = total / kRetainedSamples;
  if (average > std::numeric_limits<uint64_t>::max()) return -1;
  *duration_ns = static_cast<uint64_t>(average);
  return 0;
}
}  // namespace codegen::pgo
