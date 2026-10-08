/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_CODEGEN_PGO_PGO_CANDIDATE_IDENTITY_H_
#define AUTOFUSE_CODEGEN_PGO_PGO_CANDIDATE_IDENTITY_H_

#include <cstdint>
#include <string>

namespace codegen::pgo {

struct PgoCandidateIdentity {
  std::string graph_name;
  std::string candidate_key;
  std::string profiler_name;
  std::string tiling_repr;
  std::string kernel_file;
  uint64_t workspace_size = 0;
  uint32_t block_dim = 0;
  uint64_t duration_ns = 0;
};

std::string BuildCandidateKey(const std::string &graph_name, const std::string &tiling_repr, uint64_t workspace_size,
                              uint32_t block_dim);
bool IsValidCandidateKey(const std::string &candidate_key);

}  // namespace codegen::pgo

#endif  // AUTOFUSE_CODEGEN_PGO_PGO_CANDIDATE_IDENTITY_H_
