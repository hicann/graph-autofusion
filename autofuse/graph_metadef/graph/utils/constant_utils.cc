/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/constant_utils.h"
#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/tensor_adapter.h"

namespace af {
bool ConstantUtils::IsConstant(const NodePtr &node) {
  return IsConstant(node->GetOpDesc());
}

bool ConstantUtils::IsConstant(const OpDescPtr &op_desc) {
  if ((op_desc->GetType() == CONSTANT) || (op_desc->GetType() == CONSTANTOP)) {
    return true;
  }
  return IsPotentialConst(op_desc);
}

bool ConstantUtils::IsPotentialConst(const OpDescPtr &op_desc) {
  bool is_potential_const = false;
  const auto has_attr = AttrUtils::GetBool(op_desc, ATTR_NAME_POTENTIAL_CONST, is_potential_const);
  return (has_attr && is_potential_const);
}

bool ConstantUtils::IsRealConst(const OpDescPtr &op_desc) {
  return ((op_desc->GetType() == CONSTANT) || (op_desc->GetType() == CONSTANTOP));
}

bool ConstantUtils::GetWeight(const OpDescPtr &op_desc, const uint32_t index, ConstGeTensorPtr &weight) {
  if (AttrUtils::GetTensor(op_desc, ATTR_NAME_WEIGHTS, weight)) {
    return true;
  }
  if (!IsPotentialConst(op_desc)) {
    return false;
  }

  std::vector<uint32_t> weight_indices;
  std::vector<ConstGeTensorPtr> weights;
  if (!GetPotentialWeight(op_desc, weight_indices, weights)) {
    return false;
  }
  for (size_t i = 0U; i < weight_indices.size(); ++i) {
    if (weight_indices[i] == index) {
      weight = weights[i];
      return true;
    }
  }
  return false;
}

bool ConstantUtils::MutableWeight(const OpDescPtr &op_desc, const uint32_t index, GeTensorPtr &weight) {
  if (AttrUtils::MutableTensor(op_desc, ATTR_NAME_WEIGHTS, weight)) {
    return true;
  }
  if (!IsPotentialConst(op_desc)) {
    return false;
  }
  std::vector<uint32_t> weight_indices;
  std::vector<GeTensorPtr> weights;
  if (!MutablePotentialWeight(op_desc, weight_indices, weights)) {
    return false;
  }

  for (size_t i = 0U; i < weight_indices.size(); ++i) {
    if (weight_indices[i] == index) {
      weight = weights[i];
      return true;
    }
  }
  return false;
}
bool ConstantUtils::SetWeight(const OpDescPtr &op_desc, const uint32_t index, const GeTensorPtr weight) {
  if (IsRealConst(op_desc) && AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, weight)) {
    return true;
  }
  if (!IsPotentialConst(op_desc)) {
    return false;
  }
  std::vector<uint32_t> weight_indices;
  std::vector<GeTensorPtr> weights;
  if (!MutablePotentialWeight(op_desc, weight_indices, weights)) {
    return false;
  }

  for (size_t i = 0U; i < weight_indices.size(); ++i) {
    if (weight_indices[i] == index) {
      weights[i] = weight;
      return AttrUtils::SetListTensor(op_desc, ATTR_NAME_POTENTIAL_WEIGHT, weights);
    }
  }
  return false;
}
bool ConstantUtils::GetPotentialWeight(const OpDescPtr &op_desc, std::vector<uint32_t> &weight_indices,
                                       std::vector<ConstGeTensorPtr> &weights) {
  // check potential const attrs
  if (!AttrUtils::GetListInt(op_desc, ATTR_NAME_POTENTIAL_WEIGHT_INDICES, weight_indices)) {
    GELOGW("Missing ATTR_NAME_POTENTIAL_WEIGHT_INDICES attr on potential const %s.", op_desc->GetName().c_str());
    return false;
  }
  if (!AttrUtils::GetListTensor(op_desc, ATTR_NAME_POTENTIAL_WEIGHT, weights)) {
    GELOGW("Missing ATTR_NAME_POTENTIAL_WEIGHT attr on potential const %s.", op_desc->GetName().c_str());
    return false;
  }
  if (weight_indices.size() != weights.size()) {
    GELOGW("Weight indices do not match with weight size on potential const %s.", op_desc->GetName().c_str());
    return false;
  }
  return true;
}

bool ConstantUtils::MutablePotentialWeight(const OpDescPtr &op_desc, std::vector<uint32_t> &weight_indices,
                                           std::vector<GeTensorPtr> &weights) {
  // check potential const attrs
  if (!AttrUtils::GetListInt(op_desc, ATTR_NAME_POTENTIAL_WEIGHT_INDICES, weight_indices)) {
    GELOGW("Missing ATTR_NAME_POTENTIAL_WEIGHT_INDICES attr on potential const %s.", op_desc->GetName().c_str());
    return false;
  }
  if (!AttrUtils::MutableListTensor(op_desc, ATTR_NAME_POTENTIAL_WEIGHT, weights)) {
    GELOGW("Missing ATTR_NAME_POTENTIAL_WEIGHT attr on potential const %s.", op_desc->GetName().c_str());
    return false;
  }
  if (weight_indices.size() != weights.size()) {
    GELOGW("Weight indices do not match with weight size on potential const %s.", op_desc->GetName().c_str());
    return false;
  }
  return true;
}
bool ConstantUtils::MarkPotentialConst(const OpDescPtr &op_desc, const std::vector<int> indices,
                                       const std::vector<GeTensorPtr> weights) {
  if (indices.size() != weights.size()) {
    return false;
  }
  return (AttrUtils::SetBool(op_desc, ATTR_NAME_POTENTIAL_CONST, true) &&
          AttrUtils::SetListInt(op_desc, ATTR_NAME_POTENTIAL_WEIGHT_INDICES, indices) &&
          AttrUtils::SetListTensor(op_desc, ATTR_NAME_POTENTIAL_WEIGHT, weights));
}
bool ConstantUtils::UnMarkPotentialConst(const OpDescPtr &op_desc) {
  if (op_desc->HasAttr(ATTR_NAME_POTENTIAL_CONST) && op_desc->HasAttr(ATTR_NAME_POTENTIAL_WEIGHT_INDICES) &&
      op_desc->HasAttr(ATTR_NAME_POTENTIAL_WEIGHT)) {
    (void)op_desc->DelAttr(ATTR_NAME_POTENTIAL_CONST);
    (void)op_desc->DelAttr(ATTR_NAME_POTENTIAL_WEIGHT_INDICES);
    (void)op_desc->DelAttr(ATTR_NAME_POTENTIAL_WEIGHT);
    return true;
  }
  return false;
}

bool ConstantUtils::GetWeightFromFile(const OpDescPtr &op_desc, ConstGeTensorPtr &weight) {
  if (op_desc->GetType() != FILECONSTANT) {
    return false;
  }
  auto output_desc = op_desc->MutableOutputDesc(0U);
  GE_ASSERT_NOTNULL(output_desc);
  DataType out_type = ge::DT_UNDEFINED;
  (void)AttrUtils::GetDataType(op_desc, "dtype", out_type);
  output_desc->SetDataType(out_type);
  int64_t weight_size = 0;
  GE_ASSERT_SUCCESS(TensorUtils::GetTensorSizeInBytes(*output_desc, weight_size), "Failed to get weight size");
  std::string file_path;
  const std::string *location_ptr = AttrUtils::GetStr(op_desc, ATTR_NAME_LOCATION);
  if (location_ptr != nullptr) {
    file_path = *location_ptr;
  }
  int64_t attr_offset = 0;
  (void)AttrUtils::GetInt(op_desc, ATTR_NAME_OFFSET, attr_offset);
  const auto offset = static_cast<size_t>(attr_offset);
  int64_t attr_length = 0;
  (void)AttrUtils::GetInt(op_desc, ATTR_NAME_LENGTH, attr_length);
  const auto length = static_cast<size_t>(attr_length);
  if (file_path.empty()) {
    const std::string *path_ptr = AttrUtils::GetStr(op_desc, ATTR_NAME_FILE_PATH);
    if (path_ptr != nullptr) {
      file_path = *path_ptr;
    }
    if (file_path.empty()) {
      GELOGW("Failed to get file constant weight path, node:%s", op_desc->GetName().c_str());
      return false;
    }
  }
  const size_t file_length = (length == 0U ? static_cast<size_t>(weight_size) : length);
  const auto &bin_buff = GetBinFromFile(file_path, offset, file_length);
  GE_ASSERT_NOTNULL(bin_buff);
  const auto tensor =
      ComGraphMakeShared<GeTensor>(*output_desc, reinterpret_cast<uint8_t *>(bin_buff.get()), file_length);
  GE_ASSERT_NOTNULL(tensor);
  weight = tensor;
  return true;
}
}  // namespace af
