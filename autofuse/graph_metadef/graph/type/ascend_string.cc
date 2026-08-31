/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ascend_string_af.h"
#include "framework/common/debug/ge_log.h"
#include "common/util/mem_utils.h"
#include "base/type/ascend_string_impl.h"

namespace af {
AscendString::AscendString(const char_t *const name) {
  AscendStringImpl::Construct(*this, name);
}

AscendString::AscendString(const char_t *const name, size_t length) {
  AscendStringImpl::Construct(*this, name, length);
}

const char_t *AscendString::GetString() const {
  return AscendStringImpl::GetString(*this);
}

size_t AscendString::GetLength() const {
  return AscendStringImpl::GetLength(*this);
}

size_t AscendString::Hash() const {
  return AscendStringImpl::Hash(*this);
}

bool AscendString::operator<(const AscendString &d) const {
  return AscendStringImpl::Lt(*this, d);
}

bool AscendString::operator>(const AscendString &d) const {
  return AscendStringImpl::Gt(*this, d);
}

bool AscendString::operator==(const AscendString &d) const {
  return AscendStringImpl::Eq(*this, d);
}

bool AscendString::operator<=(const AscendString &d) const {
  return AscendStringImpl::Le(*this, d);
}

bool AscendString::operator>=(const AscendString &d) const {
  return AscendStringImpl::Ge(*this, d);
}

bool AscendString::operator!=(const AscendString &d) const {
  return AscendStringImpl::Ne(*this, d);
}

bool AscendString::operator==(const char_t *const d) const {
  return AscendStringImpl::Eq(*this, d);
}

bool AscendString::operator!=(const char_t *const d) const {
  return AscendStringImpl::Ne(*this, d);
}

size_t AscendString::Find(const AscendString &ascend_string) const {
  return AscendStringImpl::Find(*this, ascend_string);
}
}  // namespace af
