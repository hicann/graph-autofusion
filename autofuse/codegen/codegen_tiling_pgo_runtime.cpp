/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_tiling.h"

namespace codegen {
void TilingLib::GenSharedPgoRuntimeLaunch(const ascir::FusedScheduledResult &fused_schedule_result,
                                          const std::string &pgo_dir, std::stringstream &ss, bool direct_link) const {
  GenPgoToolFunction(fused_schedule_result, pgo_dir, ss, direct_link);
  GenPgoWrapper(fused_schedule_result, ss, direct_link);
}

void TilingLib::GenSharedPgoRuntimeProfiling(const ascir::FusedScheduledResult &fused_schedule_result,
                                             std::stringstream &ss, bool direct_link) const {
  GenPgoMsptiProfiling(ss, direct_link);
  GenPgoBatchProcess(ss, direct_link);
  GenPgoGetProfilingBatch(fused_schedule_result, ss, direct_link);
  GenPgoGetProfiling(fused_schedule_result, ss, direct_link);
}

std::string TilingLib::GenerateForPgo(const ascir::FusedScheduledResult &fused_schedule_result,
                                      const std::string &pgo_dir) const {
  if (ShouldFallbackPgo(fused_schedule_result)) {
    return "int main() { return 0; }\n";
  }
  std::stringstream ss;
  GenPgoHeaders(ss, false);
  GenSharedPgoRuntimeLaunch(fused_schedule_result, pgo_dir, ss, false);
  GenSharedPgoRuntimeProfiling(fused_schedule_result, ss, false);
  GenPgoProfiling(fused_schedule_result, ss);
  GenPgoMain(fused_schedule_result, ss);
  return ss.str();
}

std::string TilingLib::GenInductorPgoRunner(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  GenPgoHeaders(ss, true);
  GenSharedPgoRuntimeLaunch(fused_schedule_result, "", ss, true);
  GenInductorPgoResultTypes(ss);
  GenInductorPgoHostLoader(ss);
  GenSharedPgoRuntimeProfiling(fused_schedule_result, ss, true);
  GenInductorPgoResultProtocol(ss);
  GenInductorPgoRuntime(fused_schedule_result, ss);
  GenInductorPgoMain(ss);
  return ss.str();
}

}  // namespace codegen
