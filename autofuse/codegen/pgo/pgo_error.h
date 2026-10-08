/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_CODEGEN_PGO_PGO_ERROR_H_
#define AUTOFUSE_CODEGEN_PGO_PGO_ERROR_H_

namespace codegen::pgo {

// Public status values used by the PGO task-track adapter.  Keep the values
// negative so a driver/profapi return code can still be returned unchanged.
enum PgoError {
  PGO_SUCCESS = 0,
  PGO_INVALID_ARGUMENT = -1,
  PGO_CAPABILITY_UNAVAILABLE = -2,
  PGO_PROFAPI_UNAVAILABLE = -3,
  PGO_CHANNEL_UNAVAILABLE = -4,
  PGO_PROFILING_START_FAILED = -5,
  PGO_PROFILING_STOP_FAILED = -6,
  PGO_DATA_UNAVAILABLE = -7,
  PGO_TIMESTAMP_UNAVAILABLE = -8,
  PGO_MANIFEST_INVALID = -9,
  PGO_CACHE_VERSION_MISMATCH = -10,
  PGO_IO_ERROR = -11,
  PGO_REPORTER_BUSY = -12,
  PGO_PROFILE_UNSUPPORTED = -13,
};

}  // namespace codegen::pgo

#endif  // AUTOFUSE_CODEGEN_PGO_PGO_ERROR_H_
