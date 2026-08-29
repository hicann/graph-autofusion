/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_GE_ATTR_VALUE_H_
#define INC_GRAPH_GE_ATTR_VALUE_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "graph/buffer.h"
#include "detail/attributes_holder.h"
#include "graph/ge_error_codes_af.h"
#include "graph/ge_tensor.h"
#include "graph/any_value.h"
#include "graph/attr_value_af.h"

using std::map;
using std::string;
using std::vector;

namespace af {
using ge::AnyValue;
using ge::AscendString;
using GeAttrValue = ge::AnyValue;

class GeTensor;
class Operator;

using GeTensorPtr = std::shared_ptr<GeTensor>;
using ConstGeTensorPtr = std::shared_ptr<const GeTensor>;

class ComputeGraph;
using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;
using ConstComputeGraphPtr = std::shared_ptr<const ComputeGraph>;

class GeTensorDesc;
}  // namespace af

namespace af {

template <typename T>
bool SetAttrValue(AttrStore &attrs, const std::string &name, T &&value) {
  return attrs.SetByName(name, std::forward<T>(value));
}

template <typename T>
bool GetAttrValue(const AttrStore &attrs, const std::string &name, T &value) {
  const auto p = attrs.GetByName<T>(name);
  if (p == nullptr) {
    return false;
  }
  value = *p;
  return true;
}

template <typename T>
const T *GetAttrValue(const AttrStore &attrs, const std::string &name) {
  return attrs.GetByName<T>(name);
}

template <typename T, typename RT = typename std::decay<T>::type>
RT *SetAndGetAttrValue(AttrStore &attrs, const std::string &name, T &&value) {
  if (!attrs.SetByName(name, std::forward<T>(value))) {
    return nullptr;
  }
  return attrs.MutableGetByName<RT>(name);
}

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NamedAttrs : public af::AttrHolder {
 public:
  NamedAttrs() = default;
  virtual ~NamedAttrs() = default;
  void SetName(const std::string &name);
  std::string GetName() const;
  AnyValue GetItem(const std::string &key) const;

 protected:
  ProtoAttrMap &MutableAttrMap() override;
  ConstProtoAttrMap &GetAttrMap() const override;

 private:
  AttrStore attrs_;
  std::string name_;

  friend class GeAttrValueImp;
};

class AttrValueImpl {
 public:
  AttrValueImpl() = default;
  ~AttrValueImpl() = default;

  static graphStatus GetValue(const af::AttrValue &obj, AscendString &val);
  static graphStatus GetValue(const af::AttrValue &obj, af::AttrValue::STR &val);
  static graphStatus GetValue(const af::AttrValue &obj, af::AttrValue::INT &val);
  static graphStatus GetValue(const af::AttrValue &obj, af::AttrValue::FLOAT &val);

  AnyValue &MutableAnyValue() {
    return geAttrValue_;
  };
  friend class af::AttrValue;
  friend class af::AttrHolder;
  friend class af::Operator;

 private:
  AnyValue geAttrValue_;
};
}  // namespace af
#endif  // INC_GRAPH_GE_ATTR_VALUE_H_
