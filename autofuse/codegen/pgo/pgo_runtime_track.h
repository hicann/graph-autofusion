/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_CODEGEN_PGO_PGO_RUNTIME_TRACK_H_
#define AUTOFUSE_CODEGEN_PGO_PGO_RUNTIME_TRACK_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pgo_capability.h"

namespace codegen::pgo {

constexpr size_t kRuntimeTrackQueueCapacity = 32768U;

struct PgoRuntimeTrack {
  uint32_t device_id = 0;
  uint32_t stream_id = 0;
  uint32_t task_info = 0;
  uint64_t task_type = 0;
  uint64_t kernel_name_hash = 0;
  bool aging = false;
};

using RuntimeTrackSink = int32_t (*)(const PgoRuntimeTrack &track);

int DecodeRuntimeTrack(uint32_t aging_flag, const void *data, uint32_t length, PgoRuntimeTrack *track);
int RegisterRuntimeTrackReporter(PgoReporterOwnership ownership, RuntimeTrackSink sink);
int UnregisterRuntimeTrackReporter();
int32_t DispatchRuntimeTrack(uint32_t aging_flag, const void *data, uint32_t length);
// Receive the same runtime type registrations used by MSPTI's
// CannHashCache.  The collector uses them to discard non-kernel task tracks
// without maintaining a chip-specific task-type allowlist.
int32_t DispatchRuntimeTypeInfo(uint16_t level, uint32_t type_id, const char *name, size_t length);
void SetRuntimeTypeInfoAvailable(bool available);
bool IsKernelTaskType(uint64_t task_type);
bool IsStreamExpandEnabled();
bool RuntimeTrackOverflowed();

// The bounded queue sink is intentionally exposed only to the collector. It
// keeps the profapi callback allocation-free and lets the session drain data
// after the device channel has been flushed.
int32_t QueueRuntimeTrack(const PgoRuntimeTrack &track);
void SetRuntimeTrackCaptureEnabled(bool enabled);
void ClearRuntimeTracks();
// Wait until compact callbacks already admitted by DispatchRuntimeTrack have
// finished.  The collector calls this after stopping a candidate and before
// draining the queue so a late callback cannot leak into the next candidate.
void WaitRuntimeTrackCallbacks();
void DrainRuntimeTracks(std::vector<PgoRuntimeTrack> *tracks);

}  // namespace codegen::pgo

#endif  // AUTOFUSE_CODEGEN_PGO_PGO_RUNTIME_TRACK_H_
