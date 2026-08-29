/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AF_INC_BASE_UTILS_TYPE_UTILS_IMPL_H_
#define AF_INC_BASE_UTILS_TYPE_UTILS_IMPL_H_

#include "graph/ascend_string_af.h"
#include "graph/types_af.h"

namespace af {
class TypeUtilsImpl {
 public:
  static AscendString DataTypeToAscendString(const DataType data_type);
  static DataType AscendStringToDataType(const AscendString &str);
  static AscendString FormatToAscendString(const Format format);
  static Format AscendStringToFormat(const AscendString &str);
  static Format DataFormatToFormat(const AscendString &str);
  static bool GetDataTypeLength(const DataType data_type, uint32_t &length);
};
}  // namespace af
#endif  // AF_INC_BASE_UTILS_TYPE_UTILS_IMPL_H_
