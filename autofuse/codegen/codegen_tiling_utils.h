/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CODEGEN_CODEGEN_TILING_UTILS_H_
#define AUTOFUSE_CODEGEN_CODEGEN_TILING_UTILS_H_

#include <sstream>

namespace codegen {

__attribute__((visibility("hidden"))) void AppendCvBaseAlignHelperDefs(std::stringstream &ss);
__attribute__((visibility("hidden"))) void AppendCvSafetyMixModeHelperDefs(std::stringstream &ss, bool is_batch);

}  // namespace codegen

#endif  // AUTOFUSE_CODEGEN_CODEGEN_TILING_UTILS_H_
