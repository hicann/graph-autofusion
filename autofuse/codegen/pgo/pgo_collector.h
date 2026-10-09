/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_CODEGEN_PGO_PGO_COLLECTOR_H_
#define AUTOFUSE_CODEGEN_PGO_PGO_COLLECTOR_H_

#include <cstdint>

namespace codegen::pgo {

class PgoCollector {
 public:
  virtual ~PgoCollector() = default;
  virtual int Initialize(uint32_t device_id, void *stream) = 0;
  virtual int BeginCandidate(const char *candidate_key) = 0;
  virtual int RecordLaunch(uint64_t launch_sequence) = 0;
  virtual int EndCandidate() = 0;
  // The legacy method name is retained for generated-runner ABI stability;
  // aborting invalidates the complete in-flight batch and its profiler session.
  virtual int AbortCandidate() = 0;
  virtual int FinalizeBatch() = 0;
  virtual int GetDurationNs(uint64_t *duration_ns) = 0;
  virtual int GetCandidateDurationNs(uint32_t candidate_index, uint64_t *duration_ns) = 0;
  virtual void Finalize() = 0;
};

// C ABI used by generated PGO runners. The runner links against libaihac_codegen,
// while the profiler/driver libraries are loaded lazily and are not link-time dependencies.
extern "C" int AutofusePgoCollectorCreate(uint32_t device_id, void *stream, void **collector);
// TensorFlow PGO keeps the original MSPTI graph-level candidate key semantics.
extern "C" int AutofusePgoCollectorBegin(void *collector, const char *candidate_key);
// Computes the schema-2 candidate key from the same graph/tiling/workspace tuple
// used by the manifest writer. This keeps Inductor runners independent of the
// identity digest implementation while preserving the tuple identity contract.
extern "C" int AutofusePgoCollectorBeginCandidate(void *collector, const char *graph_name, const char *tiling_repr,
                                                  uint64_t workspace_size, uint32_t block_dim);
extern "C" int AutofusePgoCollectorRecordLaunch(void *collector, uint64_t launch_sequence);
extern "C" int AutofusePgoCollectorEnd(void *collector);
// Compatibility name: this aborts the whole in-flight batch, not just one candidate.
extern "C" int AutofusePgoCollectorAbort(void *collector);
extern "C" int AutofusePgoCollectorFinalizeBatch(void *collector);
extern "C" int AutofusePgoCollectorGetDurationNs(void *collector, uint64_t *duration_ns);
extern "C" int AutofusePgoCollectorGetCandidateDurationNs(void *collector, uint32_t candidate_index,
                                                          uint64_t *duration_ns);
extern "C" void AutofusePgoCollectorDestroy(void *collector);

}  // namespace codegen::pgo

#endif  // AUTOFUSE_CODEGEN_PGO_PGO_COLLECTOR_H_
