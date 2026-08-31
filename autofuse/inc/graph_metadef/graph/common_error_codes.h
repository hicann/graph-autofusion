/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_COMMON_ERROR_CODES_H_
#define INC_GRAPH_COMMON_ERROR_CODES_H_

#include "graph/ge_error_codes_af.h"

namespace af {
constexpr graphStatus NO_DEPENDENCE_FUNC = 50331647U;
constexpr graphStatus NO_OVERLAP_DIM = 50331646U;
constexpr graphStatus NOT_SUPPORT_SLICE = 50331645U;
}  // namespace af

#endif  // INC_GRAPH_COMMON_ERROR_CODES_H_
