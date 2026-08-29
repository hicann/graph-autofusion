/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transformer_utils.h"

#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "expand_dimension.h"
#include "transfer_shape_according_to_format.h"

namespace af {
namespace {
bool OriginShapeInitialized(const GeTensorDescPtr &tensor_desc) {
  // The caller guarantees that the pointer is not null
  if (!tensor_desc->GetOriginShape().IsScalar()) {
    return true;
  }
  return tensor_desc->IsOriginShapeInitialized();
}
bool SameCurrentAndOrigin(const GeTensorDescPtr &tensor_desc) {
  // The caller guarantees that the pointer is not null
  if (tensor_desc->GetFormat() == tensor_desc->GetOriginFormat()) {
    if (tensor_desc->GetShape() == tensor_desc->GetOriginShape()) {
      return true;
    }
    return !OriginShapeInitialized(tensor_desc);
  }
  return false;
}
}  // namespace
bool NodeShapeTransUtils::Init() {
  if (op_desc_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "op_desc_ is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] input op_desc_ is nullptr!");
    return false;
  }
  in_num_ = op_desc_->MutableAllInputName().size();
  out_num_ = op_desc_->MutableAllOutputName().size();
  map_format_in_.resize(in_num_, FORMAT_RESERVED);
  map_ori_format_in_.resize(in_num_, FORMAT_RESERVED);
  map_dtype_in_.resize(in_num_, DT_UNDEFINED);
  map_format_out_.resize(out_num_, FORMAT_RESERVED);
  map_ori_format_out_.resize(out_num_, FORMAT_RESERVED);
  map_dtype_out_.resize(out_num_, DT_UNDEFINED);
  return true;
}
bool NodeShapeTransUtils::CatchFormatAndShape() {
  for (size_t i = 0UL; i < in_num_; i++) {
    const auto tensor_desc_input = op_desc_->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc_input == nullptr) {
      continue;
    }
    const auto format = tensor_desc_input->GetFormat();
    const auto ori_format = tensor_desc_input->GetOriginFormat();
    if ((format == ori_format) && (tensor_desc_input->GetShape() == tensor_desc_input->GetOriginShape())) {
      GELOGD(
          "Node is %s, input tensor idx is %zu. ori format: %s, format: %s, ori shape:%s, shape:%s is same! "
          "No need to catch format&shape!",
          op_desc_->GetName().c_str(), i, TypeUtils::FormatToSerialString(ori_format).c_str(),
          TypeUtils::FormatToSerialString(format).c_str(), tensor_desc_input->GetOriginShape().ToString().c_str(),
          tensor_desc_input->GetShape().ToString().c_str());
      continue;
    }
    map_format_in_[i] = format;
    map_ori_format_in_[i] = ori_format;
    map_dtype_in_[i] = tensor_desc_input->GetDataType();
    tensor_desc_input->SetFormat(ori_format);
    tensor_desc_input->SetShape(tensor_desc_input->GetOriginShape());
  }

  for (size_t i = 0UL; i < out_num_; i++) {
    const auto tensor_desc_output = op_desc_->MutableOutputDesc(static_cast<uint32_t>(i));
    if (tensor_desc_output == nullptr) {
      continue;
    }
    const auto format = tensor_desc_output->GetFormat();
    const auto ori_format = tensor_desc_output->GetOriginFormat();
    if (SameCurrentAndOrigin(tensor_desc_output)) {
      GELOGD(
          "Node is %s, output tensor idx is %zu. ori format: %s, format: %s, ori shape:%s, shape:%s is same! "
          "or output original not initialized. No need to catch format&shape!",
          op_desc_->GetName().c_str(), i, TypeUtils::FormatToSerialString(ori_format).c_str(),
          TypeUtils::FormatToSerialString(format).c_str(), tensor_desc_output->GetOriginShape().ToString().c_str(),
          tensor_desc_output->GetShape().ToString().c_str());
      continue;
    }
    map_format_out_[i] = format;
    map_ori_format_out_[i] = ori_format;
    map_dtype_out_[i] = tensor_desc_output->GetDataType();

    if (format == ori_format) {
      continue;
    }
    tensor_desc_output->SetFormat(ori_format);
  }

  return true;
}

bool NodeShapeTransUtils::UpdateFormatAndShape() {
  transformer::ShapeTransferAccordingToFormat shape_transfer;
  for (size_t i = 0UL; i < in_num_; i++) {
    const auto tensor_desc_input = op_desc_->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc_input == nullptr) {
      continue;
    }
    // if can not find saved info, it says format and origin format is same when caught
    if (map_format_in_[i] == FORMAT_RESERVED) {
      GELOGD("Node is [%s], input tensor idx [%zu] has not been caught. Skip update action for it!",
             op_desc_->GetName().c_str(), i);
      tensor_desc_input->SetOriginFormat(tensor_desc_input->GetFormat());
      tensor_desc_input->SetOriginShape(tensor_desc_input->MutableShape());
      continue;
    }
    const auto ori_format = tensor_desc_input->GetFormat();
    auto &ori_shape = tensor_desc_input->MutableShape();
    const auto curr_format = map_format_in_[i];
    if (curr_format == FORMAT_ND) {
      continue;
    }
    const ge::DataType dtype = map_dtype_in_[i];

    // FE set and Ge get for PadDimention
    std::string infer_reshape_type;
    const std::string *infer_reshape_type_ptr = AttrUtils::GetStr(*tensor_desc_input, ATTR_NAME_RESHAPE_INFER_TYPE);
    if (infer_reshape_type_ptr != nullptr) {
      infer_reshape_type = *infer_reshape_type_ptr;
    }
    const bool is_success =
        transformer::ExpandDimension(op_desc_->GetType(), ori_format, curr_format, i, infer_reshape_type, ori_shape);
    if (!is_success) {
      REPORT_INNER_ERR_MSG("E18888", "ExpandDimension failed, op type:%s", op_desc_->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Call][ExpandDimension] failed, op type:%s", op_desc_->GetType().c_str());
      return false;
    }
    transformer::ShapeAndFormat shape_and_format_info{ori_shape, ori_format, curr_format, dtype};
    (void)shape_transfer.GetShapeAccordingToFormat(op_desc_, shape_and_format_info);
    tensor_desc_input->SetFormat(curr_format);
  }

  for (size_t i = 0UL; i < out_num_; i++) {
    const auto tensor_desc_output = op_desc_->MutableOutputDesc(static_cast<uint32_t>(i));
    if (tensor_desc_output == nullptr) {
      continue;
    }
    // if can not find saved info, it says format and origin format is same when caught
    if (map_ori_format_out_[i] == FORMAT_RESERVED) {
      GELOGD("Node is [%s], output tensor idx [%zu] has not been caught. Skip update action for it!",
             op_desc_->GetName().c_str(), i);
      tensor_desc_output->SetOriginFormat(tensor_desc_output->GetFormat());
      tensor_desc_output->SetOriginShape(tensor_desc_output->MutableShape());
      continue;
    }
    auto &ori_shape = tensor_desc_output->MutableShape();
    const auto curr_format = tensor_desc_output->GetFormat();
    if (curr_format != map_ori_format_out_[i]) {
      REPORT_INNER_ERR_MSG("E18888",
                           "Node is %s, out tensor idx is %zu. format: %s, "
                           "recorded origin format: %s is not same",
                           op_desc_->GetName().c_str(), i, TypeUtils::FormatToSerialString(curr_format).c_str(),
                           TypeUtils::FormatToSerialString(map_ori_format_out_[i]).c_str());
      GELOGE(GRAPH_FAILED,
             "[Check][Param] Node is %s, out tensor idx is %zu. format: %s, "
             "recorded origin format: %s is not same",
             op_desc_->GetName().c_str(), i, TypeUtils::FormatToSerialString(curr_format).c_str(),
             TypeUtils::FormatToSerialString(map_ori_format_out_[i]).c_str());
      return false;
    }
    tensor_desc_output->SetOriginShape(ori_shape);
    const auto saved_format = map_format_out_[i];
    if (saved_format == FORMAT_ND) {
      GELOGD("Node is %s, out tensor idx is %zu. ori format: %s, recorded format: %s is same! No need to transfer",
             op_desc_->GetName().c_str(), i, TypeUtils::FormatToSerialString(curr_format).c_str(),
             TypeUtils::FormatToSerialString(saved_format).c_str());
      continue;
    }
    tensor_desc_output->SetFormat(saved_format);
    const ge::DataType dtype = tensor_desc_output->GetDataType();

    // FE set and Ge get for PadDimention
    std::string infer_reshape_type;
    const std::string *infer_reshape_type_ptr = AttrUtils::GetStr(*tensor_desc_output, ATTR_NAME_RESHAPE_INFER_TYPE);
    if (infer_reshape_type_ptr != nullptr) {
      infer_reshape_type = *infer_reshape_type_ptr;
    }
    const bool is_success =
        transformer::ExpandDimension(op_desc_->GetType(), curr_format, saved_format, i, infer_reshape_type, ori_shape);
    if (!is_success) {
      REPORT_INNER_ERR_MSG("E18888", "ExpandDimension failed, op type:%s.", op_desc_->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Call][ExpandDimension] failed, op type:%s.", op_desc_->GetType().c_str());
      return false;
    }
    transformer::ShapeAndFormat shape_and_format_info{ori_shape, curr_format, saved_format, dtype};
    (void)shape_transfer.GetShapeAccordingToFormat(op_desc_, shape_and_format_info);
    GELOGD("Node is %s, out tensor idx is %zu. Update format and shape success, ori format: %s, format: %s",
           op_desc_->GetName().c_str(), i, TypeUtils::FormatToSerialString(curr_format).c_str(),
           TypeUtils::FormatToSerialString(saved_format).c_str());
  }
  GELOGD("Node is %s. Update format and shape success", op_desc_->GetName().c_str());
  return true;
}
}  // namespace af
