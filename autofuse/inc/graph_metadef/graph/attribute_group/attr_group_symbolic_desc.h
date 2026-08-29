/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_ATTR_GROUP_SYMBOLIC_DESC_H
#define INC_GRAPH_ATTR_GROUP_SYMBOLIC_DESC_H

#include "graph/ge_error_codes_af.h"
#include "attribute_group/af_attr_group_base.h"
#include "graph/tensor_af.h"
#include "exe_graph/runtime/symbolic_tensor.h"

namespace af {

class SymbolicDescAttr : public AttrGroupsBase {
 public:
  SymbolicDescAttr() = default;

  ~SymbolicDescAttr() override = default;
  graphStatus Serialize(proto::AttrGroupDef &attr_group_def) override;
  graphStatus Deserialize(const proto::AttrGroupDef &attr_group_def, AttrHolder *attr_holder) override;
  std::unique_ptr<AttrGroupsBase> CloneAf() override;

  gert::SymbolTensor symbolic_tensor;
};
}  // namespace af
#endif  // INC_GRAPH_ATTR_GROUP_SYMBOLIC_DESC_H
