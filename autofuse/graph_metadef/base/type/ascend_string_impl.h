/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_TYPE_ASCEND_STRING_IMPL_H_
#define BASE_TYPE_ASCEND_STRING_IMPL_H_

#include "graph/types_af.h"
#include "graph/ascend_string_af.h"

namespace af {
class AscendStringImpl {
 public:
  AscendStringImpl() = default;

  ~AscendStringImpl() = default;

  static const char_t *GetString(const AscendString &obj);

  static size_t GetLength(const AscendString &obj);

  static size_t Find(const AscendString &obj, const AscendString &ascend_string);

  static size_t Hash(const AscendString &obj);

  static bool Lt(const AscendString &obj, const AscendString &other);

  static bool Gt(const AscendString &obj, const AscendString &other);

  static bool Le(const AscendString &obj, const AscendString &other);

  static bool Ge(const AscendString &obj, const AscendString &other);

  static bool Eq(const AscendString &obj, const AscendString &other);

  static bool Ne(const AscendString &obj, const AscendString &other);

  static bool Eq(const AscendString &obj, const char_t *const other);

  static bool Ne(const AscendString &obj, const char_t *const other);

  static void Construct(AscendString &obj, const char_t *const name);

  static void Construct(AscendString &obj, const char_t *const name, size_t length);

  static void Destroy(const std::string *const ptr);
};
}  // namespace af
#endif  // BASE_TYPE_ASCEND_STRING_IMPL_H_
