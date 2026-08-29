/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base/type/ascend_string_impl.h"
#include "graph/ge_error_codes_af.h"
#include "graph/debug/ge_log.h"
#include <memory>

namespace af {
// 控制对象构造析构在同一个so实现
void AscendStringImpl::Destroy(const std::string *const ptr) {
  if (ptr != nullptr) {
    delete ptr;
  }
}

void AscendStringImpl::Construct(AscendString &obj, const char_t *const name) {
  if (name != nullptr) {
    obj.name_ = std::shared_ptr<std::string>(new (std::nothrow) std::string(name),
                                             [](const std::string *const ptr) { Destroy(ptr); });
    if (obj.name_ == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "new string failed.");
      GELOGE(FAILED, "[New][String]AscendStringImpl[%s] make shared failed.", name);
    }
  }
}

void AscendStringImpl::Construct(AscendString &obj, const char_t *const name, size_t length) {
  if (name != nullptr) {
    obj.name_ = std::shared_ptr<std::string>(new (std::nothrow) std::string(name, length),
                                             [](const std::string *const ptr) { Destroy(ptr); });
    if (obj.name_ == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "new string with length failed.");
      GELOGE(FAILED, "[New][String]AscendStringImpl make shared failed, length=%zu.", length);
    }
  }
}

const char_t *AscendStringImpl::GetString(const AscendString &obj) {
  if (obj.name_ == nullptr) {
    static const std::string kEmptyString = "";
    return kEmptyString.c_str();
  }
  return obj.name_->c_str();
}

size_t AscendStringImpl::GetLength(const AscendString &obj) {
  if (obj.name_ == nullptr) {
    return 0UL;
  }
  return (*(obj.name_)).length();
}

size_t AscendStringImpl::Hash(const AscendString &obj) {
  if (obj.name_ == nullptr) {
    static const size_t kEmptyStringHash = std::hash<std::string>()("");
    return kEmptyStringHash;
  }
  return std::hash<std::string>()(*(obj.name_));
}

bool AscendStringImpl::Lt(const AscendString &obj, const AscendString &other) {
  if ((obj.name_ == nullptr) && (other.name_ == nullptr)) {
    return false;
  } else if (obj.name_ == nullptr) {
    return true;
  } else if (other.name_ == nullptr) {
    return false;
  } else {
    return (*(obj.name_)) < (*(other.name_));
  }
}

bool AscendStringImpl::Gt(const AscendString &obj, const AscendString &other) {
  if ((obj.name_ == nullptr) && (other.name_ == nullptr)) {
    return false;
  } else if (obj.name_ == nullptr) {
    return false;
  } else if (other.name_ == nullptr) {
    return true;
  } else {
    return (*(obj.name_)) > (*(other.name_));
  }
}

bool AscendStringImpl::Eq(const AscendString &obj, const AscendString &other) {
  if ((obj.name_ == nullptr) && (other.name_ == nullptr)) {
    return true;
  } else if (obj.name_ == nullptr) {
    return false;
  } else if (other.name_ == nullptr) {
    return false;
  } else {
    return (*(obj.name_)) == (*(other.name_));
  }
}

bool AscendStringImpl::Le(const AscendString &obj, const AscendString &other) {
  if (obj.name_ == nullptr) {
    return true;
  } else if (other.name_ == nullptr) {
    return false;
  } else {
    return (*(obj.name_)) <= (*(other.name_));
  }
}

bool AscendStringImpl::Ge(const AscendString &obj, const AscendString &other) {
  if (other.name_ == nullptr) {
    return true;
  } else if (obj.name_ == nullptr) {
    return false;
  } else {
    return (*(obj.name_)) >= (*(other.name_));
  }
}

bool AscendStringImpl::Ne(const AscendString &obj, const AscendString &other) {
  if ((obj.name_ == nullptr) && (other.name_ == nullptr)) {
    return false;
  } else if (obj.name_ == nullptr) {
    return true;
  } else if (other.name_ == nullptr) {
    return true;
  } else {
    return (*(obj.name_)) != (*(other.name_));
  }
}

bool AscendStringImpl::Eq(const AscendString &obj, const char_t *const other) {
  if ((obj.name_ == nullptr) && (other == nullptr)) {
    return true;
  } else if ((obj.name_ == nullptr) || (other == nullptr)) {
    return false;
  } else {
    return (strcmp((*(obj.name_)).c_str(), other) == 0);
  }
}

bool AscendStringImpl::Ne(const AscendString &obj, const char_t *const other) {
  if ((obj.name_ == nullptr) && (other == nullptr)) {
    return false;
  } else if ((obj.name_ == nullptr) || (other == nullptr)) {
    return true;
  } else {
    return (strcmp((*(obj.name_)).c_str(), other) != 0);
  }
}

size_t AscendStringImpl::Find(const AscendString &obj, const AscendString &ascend_string) {
  if ((obj.name_ == nullptr) || (ascend_string.name_ == nullptr)) {
    return std::string::npos;
  }
  return obj.name_->find(*(ascend_string.name_));
}

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
