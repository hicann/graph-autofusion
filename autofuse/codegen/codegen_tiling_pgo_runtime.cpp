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

std::string TilingLib::GenerateForPgo(const ascir::FusedScheduledResult &fused_schedule_result,
                                      const std::string &pgo_dir) const {
  // 生成PGO的头文件和函数定义
  std::stringstream ss;
  GenPgoHeaders(ss);
  // 生成PGO需要的工具函数
  GenPgoToolFunction(fused_schedule_result, pgo_dir, ss);
  // 生成PGO需要的wrapper函数
  GenPgoWrapper(fused_schedule_result, ss);
  // 生成PGO需要的求解代码
  GenPgoProfiling(fused_schedule_result, ss);
  // 生成PGO的main函数
  GenPgoMain(fused_schedule_result, ss);
  return ss.str();
}

}  // namespace codegen
