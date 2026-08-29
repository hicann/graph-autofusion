/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "expand_dimension.h"
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <bitset>
#include "axis_constants.h"
#include "exe_graph/runtime/expand_dims_type.h"
#include "graph/types_af.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_error_codes_af.h"
#include "graph/utils/type_utils.h"

namespace transformer {
namespace {
const std::string RESHAPE_TYPE_FORBIDDEN = "FORBIDDEN";
const uint32_t kBitsOfByte = 8;
const uint32_t kBitSetDisplaySize = 8;
const uint32_t kMaxReshapeTypeSize = 56;

const std::set<af::Format> kSupportedTransFormat = {af::FORMAT_ND,
                                                    af::FORMAT_FRACTAL_NZ,
                                                    af::FORMAT_FRACTAL_NZ_C0_2,
                                                    af::FORMAT_FRACTAL_NZ_C0_4,
                                                    af::FORMAT_FRACTAL_NZ_C0_8,
                                                    af::FORMAT_FRACTAL_NZ_C0_16,
                                                    af::FORMAT_FRACTAL_NZ_C0_32,
                                                    af::FORMAT_ND_RNN_BIAS,
                                                    af::FORMAT_FRACTAL_ZN_RNN};

const std::map<af::Format, size_t> FULL_SIZE_OF_FORMAT{
    {af::FORMAT_NCHW, DIM_SIZE_FOUR},  {af::FORMAT_NHWC, DIM_SIZE_FOUR},  {af::FORMAT_HWCN, DIM_SIZE_FOUR},
    {af::FORMAT_CHWN, DIM_SIZE_FOUR},  {af::FORMAT_NDHWC, DIM_SIZE_FIVE}, {af::FORMAT_NCDHW, DIM_SIZE_FIVE},
    {af::FORMAT_DHWCN, DIM_SIZE_FIVE}, {af::FORMAT_DHWNC, DIM_SIZE_FIVE}, {af::FORMAT_ND, DIM_SIZE_FOUR}};

inline uint32_t GenerateFormatKey(af::Format format) {
  return ((static_cast<uint32_t>(format) & 0xff) << kBitsOfByte);
}

inline uint32_t GenerateReshapeTypeKey(af::Format format, size_t size) {
  return ((static_cast<uint32_t>(format) & 0xff) << kBitsOfByte) | (static_cast<uint32_t>(size) & 0xff);
}

inline uint32_t GenerateAxisIndexKey(af::Format format, char ch) {
  return ((static_cast<uint32_t>(format) & 0xff) << kBitsOfByte) | (static_cast<uint32_t>(ch) & 0xff);
}

const std::unordered_map<uint32_t, std::string> DEFAULT_RESHAPE_TYPE{
    {GenerateReshapeTypeKey(af::FORMAT_NCHW, 0), ""},      {GenerateReshapeTypeKey(af::FORMAT_NHWC, 0), ""},
    {GenerateReshapeTypeKey(af::FORMAT_HWCN, 0), ""},      {GenerateReshapeTypeKey(af::FORMAT_CHWN, 0), ""},
    {GenerateReshapeTypeKey(af::FORMAT_NDHWC, 0), ""},     {GenerateReshapeTypeKey(af::FORMAT_NCDHW, 0), ""},
    {GenerateReshapeTypeKey(af::FORMAT_DHWCN, 0), ""},     {GenerateReshapeTypeKey(af::FORMAT_DHWNC, 0), ""},

    {GenerateReshapeTypeKey(af::FORMAT_NCHW, 1), "C"},     {GenerateReshapeTypeKey(af::FORMAT_NHWC, 1), "C"},
    {GenerateReshapeTypeKey(af::FORMAT_HWCN, 1), "C"},     {GenerateReshapeTypeKey(af::FORMAT_CHWN, 1), "C"},
    {GenerateReshapeTypeKey(af::FORMAT_NDHWC, 1), "C"},    {GenerateReshapeTypeKey(af::FORMAT_NCDHW, 1), "C"},
    {GenerateReshapeTypeKey(af::FORMAT_DHWCN, 1), "C"},    {GenerateReshapeTypeKey(af::FORMAT_DHWNC, 1), "C"},

    {GenerateReshapeTypeKey(af::FORMAT_NCHW, 2), "CH"},    {GenerateReshapeTypeKey(af::FORMAT_NHWC, 2), "HW"},
    {GenerateReshapeTypeKey(af::FORMAT_HWCN, 2), "CN"},    {GenerateReshapeTypeKey(af::FORMAT_CHWN, 2), "WN"},
    {GenerateReshapeTypeKey(af::FORMAT_NDHWC, 2), "WC"},   {GenerateReshapeTypeKey(af::FORMAT_NCDHW, 2), "HW"},
    {GenerateReshapeTypeKey(af::FORMAT_DHWCN, 2), "CN"},   {GenerateReshapeTypeKey(af::FORMAT_DHWNC, 2), "NC"},

    {GenerateReshapeTypeKey(af::FORMAT_NCHW, 3), "CHW"},   {GenerateReshapeTypeKey(af::FORMAT_NHWC, 3), "HWC"},
    {GenerateReshapeTypeKey(af::FORMAT_HWCN, 3), "WCN"},   {GenerateReshapeTypeKey(af::FORMAT_CHWN, 3), "HWN"},
    {GenerateReshapeTypeKey(af::FORMAT_NDHWC, 3), "HWC"},  {GenerateReshapeTypeKey(af::FORMAT_NCDHW, 3), "DHW"},
    {GenerateReshapeTypeKey(af::FORMAT_DHWCN, 3), "WCN"},  {GenerateReshapeTypeKey(af::FORMAT_DHWNC, 3), "WNC"},

    {GenerateReshapeTypeKey(af::FORMAT_NDHWC, 4), "DHWC"}, {GenerateReshapeTypeKey(af::FORMAT_NCDHW, 4), "CDHW"},
    {GenerateReshapeTypeKey(af::FORMAT_DHWCN, 4), "HWCN"}, {GenerateReshapeTypeKey(af::FORMAT_DHWNC, 4), "HWNC"}};

const std::unordered_map<uint32_t, int32_t> AXIS_INDEX_OF_FORMAT{
    {GenerateAxisIndexKey(af::FORMAT_NCHW, 'N'), AXIS_NCHW_DIM_N},
    {GenerateAxisIndexKey(af::FORMAT_NCHW, 'C'), AXIS_NCHW_DIM_C},
    {GenerateAxisIndexKey(af::FORMAT_NCHW, 'H'), AXIS_NCHW_DIM_H},
    {GenerateAxisIndexKey(af::FORMAT_NCHW, 'W'), AXIS_NCHW_DIM_W},

    {GenerateAxisIndexKey(af::FORMAT_HWCN, 'N'), AXIS_HWCN_DIM_N},
    {GenerateAxisIndexKey(af::FORMAT_HWCN, 'C'), AXIS_HWCN_DIM_C},
    {GenerateAxisIndexKey(af::FORMAT_HWCN, 'H'), AXIS_HWCN_DIM_H},
    {GenerateAxisIndexKey(af::FORMAT_HWCN, 'W'), AXIS_HWCN_DIM_W},

    {GenerateAxisIndexKey(af::FORMAT_NHWC, 'N'), AXIS_NHWC_DIM_N},
    {GenerateAxisIndexKey(af::FORMAT_NHWC, 'C'), AXIS_NHWC_DIM_C},
    {GenerateAxisIndexKey(af::FORMAT_NHWC, 'H'), AXIS_NHWC_DIM_H},
    {GenerateAxisIndexKey(af::FORMAT_NHWC, 'W'), AXIS_NHWC_DIM_W},

    {GenerateAxisIndexKey(af::FORMAT_CHWN, 'N'), AXIS_CHWN_DIM_N},
    {GenerateAxisIndexKey(af::FORMAT_CHWN, 'C'), AXIS_CHWN_DIM_C},
    {GenerateAxisIndexKey(af::FORMAT_CHWN, 'H'), AXIS_CHWN_DIM_H},
    {GenerateAxisIndexKey(af::FORMAT_CHWN, 'W'), AXIS_CHWN_DIM_W},

    {GenerateAxisIndexKey(af::FORMAT_NDHWC, 'N'), NDHWC_DIM_N},
    {GenerateAxisIndexKey(af::FORMAT_NDHWC, 'C'), NDHWC_DIM_C},
    {GenerateAxisIndexKey(af::FORMAT_NDHWC, 'H'), NDHWC_DIM_H},
    {GenerateAxisIndexKey(af::FORMAT_NDHWC, 'W'), NDHWC_DIM_W},
    {GenerateAxisIndexKey(af::FORMAT_NDHWC, 'D'), NDHWC_DIM_D},

    {GenerateAxisIndexKey(af::FORMAT_NCDHW, 'N'), NCDHW_DIM_N},
    {GenerateAxisIndexKey(af::FORMAT_NCDHW, 'C'), NCDHW_DIM_C},
    {GenerateAxisIndexKey(af::FORMAT_NCDHW, 'H'), NCDHW_DIM_H},
    {GenerateAxisIndexKey(af::FORMAT_NCDHW, 'W'), NCDHW_DIM_W},
    {GenerateAxisIndexKey(af::FORMAT_NCDHW, 'D'), NCDHW_DIM_D},

    {GenerateAxisIndexKey(af::FORMAT_DHWCN, 'N'), DHWCN_DIM_N},
    {GenerateAxisIndexKey(af::FORMAT_DHWCN, 'C'), DHWCN_DIM_C},
    {GenerateAxisIndexKey(af::FORMAT_DHWCN, 'H'), DHWCN_DIM_H},
    {GenerateAxisIndexKey(af::FORMAT_DHWCN, 'W'), DHWCN_DIM_W},
    {GenerateAxisIndexKey(af::FORMAT_DHWCN, 'D'), DHWCN_DIM_D},

    {GenerateAxisIndexKey(af::FORMAT_DHWNC, 'N'), DHWNC_DIM_N},
    {GenerateAxisIndexKey(af::FORMAT_DHWNC, 'C'), DHWNC_DIM_C},
    {GenerateAxisIndexKey(af::FORMAT_DHWNC, 'H'), DHWNC_DIM_H},
    {GenerateAxisIndexKey(af::FORMAT_DHWNC, 'W'), DHWNC_DIM_W},
    {GenerateAxisIndexKey(af::FORMAT_DHWNC, 'D'), DHWNC_DIM_D}};

void GeShapeToRtShape(const af::GeShape &ge_shape, gert::Shape &rt_shape) {
  rt_shape.SetDimNum(0);
  for (size_t i = 0; i < ge_shape.GetDimNum(); ++i) {
    rt_shape.AppendDim(ge_shape.GetDim(i));
  }
}

void RtShapeToGeShape(const gert::Shape &rt_shape, af::GeShape &ge_shape) {
  ge_shape.SetDimNum(0);
  for (size_t i = 0; i < rt_shape.GetDimNum(); ++i) {
    ge_shape.AppendDim(rt_shape.GetDim(i));
  }
}
}  // namespace

bool GetDefaultReshapeType(const af::Format &original_format, const size_t &old_dims_size, std::string &reshape_type) {
  int32_t default_key = GenerateReshapeTypeKey(original_format, old_dims_size);
  auto iter = DEFAULT_RESHAPE_TYPE.find(default_key);
  if (iter == DEFAULT_RESHAPE_TYPE.end()) {
    GELOGW("dim size %zu is invalid.", old_dims_size);
    return false;
  }

  reshape_type = iter->second;
  return true;
}

bool IsExpandNecessary(const size_t &old_dims_size, const af::Format &original_format, const af::Format &final_format,
                       const std::string &reshape_type, size_t &full_size) {
  /* 1. Check whether the old dim size is full. Full size is not necessary for expand. */
  auto iter_full_size = FULL_SIZE_OF_FORMAT.find(original_format);
  if (iter_full_size == FULL_SIZE_OF_FORMAT.end()) {
    GELOGW("Original Format %u is invalid.", original_format);
    return false;
  } else {
    if (old_dims_size >= iter_full_size->second) {
      return false;
    }
  }
  /* 2. Check whether the final format does not need expanding dimension. */
  bool no_need_reshape_flag = reshape_type == RESHAPE_TYPE_FORBIDDEN || kFormatNZSet.count(final_format) > 0 ||
                              (original_format == af::FORMAT_ND && final_format == af::FORMAT_FRACTAL_Z);
  if (no_need_reshape_flag) {
    return false;
  }
  full_size = iter_full_size->second;
  return true;
}

bool IsReshapeTypeValid(const af::Format &original_format, const size_t &old_dims_size,
                        const std::string &reshape_type) {
  if (reshape_type.empty()) {
    return old_dims_size == 0;
  }
  int32_t pos = -1;
  uint32_t format_key = GenerateFormatKey(original_format);
  uint32_t axis_key = 0;
  for (const char &dim : reshape_type) {
    axis_key = format_key | (static_cast<uint32_t>(dim) & 0xff);
    auto iter = AXIS_INDEX_OF_FORMAT.find(axis_key);
    if (iter == AXIS_INDEX_OF_FORMAT.end()) {
      return false;
    }
    if (iter->second > pos) {
      pos = iter->second;
    } else {
      return false;
    }
  }

  return true;
}

void ExpandByReshapeType(af::GeShape &shape, const af::Format &original_format, const size_t &old_dims_size,
                         const size_t &full_size, const std::string &reshape_type) {
  GELOGD("Expand tensor through reshape of type %s.", reshape_type.c_str());
  /* Build a array with all 1 of full size. Then we will substitute some of the 1 with the original axis value. */
  for (size_t i = old_dims_size; i < full_size; i++) {
    shape.AppendDim(1);
  }
  if (reshape_type.empty() || old_dims_size == 0) {
    return;
  }

  uint32_t format_key = GenerateFormatKey(original_format);
  uint32_t axis_key = 0;
  for (int32_t i = static_cast<int32_t>(old_dims_size) - 1; i >= 0; i--) {
    axis_key = format_key | (static_cast<uint32_t>(reshape_type.at(i)) & 0xff);
    auto iter_axis_index = AXIS_INDEX_OF_FORMAT.find(axis_key);
    if (iter_axis_index == AXIS_INDEX_OF_FORMAT.end()) {
      continue;
    }
    if (iter_axis_index->second == i) {
      continue;
    }
    shape.SetDim(iter_axis_index->second, shape.GetDim(i));
    shape.SetDim(i, 1);
  }
}

bool ExpandDimension(const std::string &op_type, const af::Format &original_format, const af::Format &final_format,
                     const uint32_t &tensor_index, const std::string &reshape_type, af::GeShape &shape) {
  /* 1. Check expanding necessary. */
  size_t full_size = 0;
  size_t old_dims_size = shape.GetDimNum();
  auto primary_format = static_cast<af::Format>(ge::GetPrimaryFormat(final_format));
  if (!IsExpandNecessary(old_dims_size, original_format, primary_format, reshape_type, full_size)) {
    return true;
  }

  /* 2. Check whether the reshape type is consistent with the original format.
   * If not consistent, just return and report a warning. */
  std::string valid_reshape_type = reshape_type;
  if (!IsReshapeTypeValid(original_format, old_dims_size, reshape_type)) {
    if (!GetDefaultReshapeType(original_format, old_dims_size, valid_reshape_type)) {
      return true;
    }
  }

  /* 3. Check whether the dimension of original shape is less than or equal to
   * the length of reshape type. If the dimension of original shape if larger,
   * we cannot find a suitable position for all axes in the original shape, so we just return. */
  if (old_dims_size > valid_reshape_type.length()) {
    GELOGW("Dimension %zu of tensor %u in %s exceeds the length of the reshape type, which is %zu.", old_dims_size,
           tensor_index, op_type.c_str(), valid_reshape_type.length());
    return true;
  }

  /* 4. Expand dimension. */
  ExpandByReshapeType(shape, original_format, old_dims_size, full_size, valid_reshape_type);
  return true;
}

bool ExpandRangeDimension(const std::string &op_type, const af::Format &original_format, const af::Format &final_format,
                          const uint32_t &tensor_index, const std::string &reshape_type,
                          std::vector<std::pair<int64_t, int64_t>> &ranges) {
  std::vector<int64_t> range_upper;
  std::vector<int64_t> range_low;
  for (auto &i : ranges) {
    range_low.emplace_back(i.first);
    range_upper.emplace_back(i.second);
  }

  af::GeShape shape_low(range_low);
  af::GeShape shape_upper(range_upper);
  auto primary_format = static_cast<af::Format>(ge::GetPrimaryFormat(final_format));
  bool res = ExpandDimension(op_type, original_format, primary_format, tensor_index, reshape_type, shape_low) &&
             ExpandDimension(op_type, original_format, primary_format, tensor_index, reshape_type, shape_upper);
  if (!res || (shape_low.GetDimNum() != shape_upper.GetDimNum())) {
    return false;
  }
  ranges.clear();
  for (size_t idx = 0; idx < shape_low.GetDimNum(); ++idx) {
    ranges.emplace_back(std::pair<int64_t, int64_t>(shape_low.GetDim(idx), shape_upper.GetDim(idx)));
  }
  return res;
}

ExpandDimension::ExpandDimension() {}
ExpandDimension::~ExpandDimension() {}

int64_t ExpandDimension::GenerateReshapeType(const af::Format &origin_format, const af::Format &format,
                                             const size_t &origin_dim_size, const std::string &reshape_type) {
  auto primary_format = static_cast<af::Format>(ge::GetPrimaryFormat(format));
  GELOGD("Begin to generate integer reshape type, original format[%d], format[%d], dim size[%zu], reshape type[%s].",
         origin_format, primary_format, origin_dim_size, reshape_type.c_str());
  int64_t ret_reshape_type = 0;
  size_t full_size = 0;
  if (!GetFormatFullSize(origin_format, full_size)) {
    return ret_reshape_type;
  }
  if (!IsNeedExpand(origin_format, primary_format, origin_dim_size, full_size, reshape_type)) {
    return ret_reshape_type;
  }

  std::string valid_shape_type = reshape_type;
  if (!IsReshapeTypeValid(origin_format, origin_dim_size, reshape_type)) {
    if (!GetDefaultReshapeType(origin_format, origin_dim_size, valid_shape_type)) {
      return ret_reshape_type;
    }
    GELOGD("Invalid reshape type [%s], using default reshape type [%s]", reshape_type.c_str(),
           valid_shape_type.c_str());
  }

  if (origin_dim_size > valid_shape_type.length()) {
    GELOGW("The length of reshape type[%s] is shorter than dim size[%zu]. Can not generate integer reshape type.",
           valid_shape_type.c_str(), origin_dim_size);
    return ret_reshape_type;
  }

  uint32_t format_key = GenerateFormatKey(origin_format);
  std::unordered_set<int32_t> dim_pos_set;
  for (const char &dim : valid_shape_type.substr(0, origin_dim_size)) {
    uint32_t axis_key = format_key | (static_cast<uint32_t>(dim) & 0xff);
    auto iter_axis_index = AXIS_INDEX_OF_FORMAT.find(axis_key);
    if (iter_axis_index != AXIS_INDEX_OF_FORMAT.end()) {
      dim_pos_set.emplace(iter_axis_index->second);
    }
  }

  for (size_t i = 0; i < full_size; i++) {
    if (dim_pos_set.count(static_cast<int32_t>(i)) == 0) {
      ret_reshape_type = ret_reshape_type | (1 << i);
    }
  }

  ret_reshape_type = ret_reshape_type | (static_cast<uint64_t>(full_size) << kMaxReshapeTypeSize);
  GELOGD(
      "Integer reshape type [%s] has been generated for the original format [%d], with dim size [%zu] and reshape type "
      "[%s].",
      std::bitset<kBitSetDisplaySize>(ret_reshape_type).to_string().c_str(), origin_format, origin_dim_size,
      valid_shape_type.c_str());
  return ret_reshape_type;
}

bool ExpandDimension::GenerateReshapeType(const af::Format &origin_format, const af::Format &format,
                                          const size_t &origin_dim_size, const std::string &reshape_type,
                                          int64_t &reshape_type_mask) {
  auto primary_format = static_cast<af::Format>(ge::GetPrimaryFormat(format));
  GELOGD("Begin to generate integer reshape type, original format[%d], format[%d], dim size[%zu], reshape type[%s].",
         origin_format, primary_format, origin_dim_size, reshape_type.c_str());
  size_t full_size = 0;
  if (!GetFormatFullSize(origin_format, full_size)) {
    return true;
  }
  if (!IsNeedExpand(origin_format, primary_format, origin_dim_size, full_size, reshape_type)) {
    return true;
  }

  std::string valid_shape_type = reshape_type;
  if (!IsReshapeTypeValid(origin_format, origin_dim_size, reshape_type)) {
    if (!GetDefaultReshapeType(origin_format, origin_dim_size, valid_shape_type)) {
      return true;
    }
    GELOGD("Invalid reshape type [%s], using default reshape type [%s]", reshape_type.c_str(),
           valid_shape_type.c_str());
  }

  if (origin_dim_size > valid_shape_type.length()) {
    GELOGE(af::GRAPH_FAILED,
           "The length of reshape type[%s] is longer than dim size[%zu]. Can not generate integer reshape type.",
           valid_shape_type.c_str(), origin_dim_size);
    return false;
  }

  uint32_t format_key = GenerateFormatKey(origin_format);
  std::unordered_set<int32_t> dim_pos_set;
  for (const char &dim : valid_shape_type.substr(0, origin_dim_size)) {
    uint32_t axis_key = format_key | (static_cast<uint32_t>(dim) & 0xff);
    auto iter_axis_index = AXIS_INDEX_OF_FORMAT.find(axis_key);
    if (iter_axis_index != AXIS_INDEX_OF_FORMAT.end()) {
      dim_pos_set.emplace(iter_axis_index->second);
    }
  }

  for (size_t i = 0; i < full_size; i++) {
    if (dim_pos_set.count(static_cast<int32_t>(i)) == 0) {
      reshape_type_mask = reshape_type_mask | (1 << i);
    }
  }

  reshape_type_mask = reshape_type_mask | (static_cast<uint64_t>(full_size) << kMaxReshapeTypeSize);
  GELOGD(
      "Integer reshape type [%s] has been generated for the original format [%d], with dim size [%zu] and reshape type "
      "[%s].",
      std::bitset<kBitSetDisplaySize>(reshape_type_mask).to_string().c_str(), origin_format, origin_dim_size,
      valid_shape_type.c_str());
  return true;
}

bool ExpandDimension::GenerateReshapeTypeByMask(const af::Format &origin_format, const size_t &origin_dim_size,
                                                const int64_t &reshape_type_mask, std::string &reshape_type,
                                                std::string &failed_reason) {
  if (origin_format == af::FORMAT_ND) {
    if (reshape_type_mask == 0) {
      return true;
    } else {
      failed_reason = "Can not generate reshape type for ND format.";
      GELOGI("%s", failed_reason.c_str());
      return false;
    }
  }

  std::string origin_format_str = af::TypeUtils::FormatToSerialString(origin_format);
  size_t full_size = 0;
  if (!GetFormatFullSize(origin_format, full_size)) {
    failed_reason = origin_format_str + " is not supported for expanding dims.";
    GELOGI("%s", failed_reason.c_str());
    return false;
  }

  if (reshape_type_mask == 0 && origin_dim_size == full_size) {
    reshape_type = origin_format_str;
    return true;
  }

  size_t full_size_mask = static_cast<size_t>(reshape_type_mask >> kMaxReshapeTypeSize);
  if (full_size != full_size_mask) {
    failed_reason = "Full size[" + std::to_string(full_size_mask) + "] from reshape mask is not correct,";
    failed_reason += " it should be[" + std::to_string(full_size) + "].";
    GELOGI("%s", failed_reason.c_str());
    return false;
  }

  reshape_type.clear();
  size_t dim_count = 0;
  for (size_t i = 0; i < full_size; ++i) {
    if ((reshape_type_mask & (1 << i)) == 0) {
      reshape_type += origin_format_str.at(i);
      dim_count++;
    }
  }

  if (dim_count != origin_dim_size) {
    std::string bit_str = std::bitset<kBitSetDisplaySize>(reshape_type_mask).to_string();
    failed_reason = "[" + bit_str + "] is not correct when dim size is [" + std::to_string(origin_dim_size) + "].";
    GELOGI("%s", failed_reason.c_str());
    return false;
  }
  return true;
}

bool ExpandDimension::IsNeedExpand(const af::Format &origin_format, const af::Format &format,
                                   const size_t &origin_dim_size, const size_t &full_size,
                                   const std::string &reshape_type) {
  if (origin_dim_size >= full_size) {
    return false;
  }
  if (reshape_type == RESHAPE_TYPE_FORBIDDEN) {
    return false;
  }
  if (kSupportedTransFormat.count(format) != 0) {
    return false;
  }
  if (origin_format == af::FORMAT_ND && format == af::FORMAT_FRACTAL_Z) {
    return false;
  }
  return true;
}

bool ExpandDimension::IsReshapeTypeValid(const af::Format &origin_format, const size_t &origin_dim_size,
                                         const std::string &reshape_type) {
  if (reshape_type.empty()) {
    return origin_dim_size == 0;
  }
  int32_t pos = -1;
  uint32_t format_key = GenerateFormatKey(origin_format);
  uint32_t axis_key = 0;
  for (const char &dim : reshape_type) {
    axis_key = format_key | (static_cast<uint32_t>(dim) & 0xff);
    auto iter = AXIS_INDEX_OF_FORMAT.find(axis_key);
    if (iter == AXIS_INDEX_OF_FORMAT.end()) {
      return false;
    }
    if (iter->second > pos) {
      pos = iter->second;
    } else {
      return false;
    }
  }
  return true;
}

bool ExpandDimension::GetDefaultReshapeType(const af::Format &origin_format, const size_t &origin_dim_size,
                                            std::string &reshape_type) {
  int32_t default_key = GenerateReshapeTypeKey(origin_format, origin_dim_size);
  auto iter = DEFAULT_RESHAPE_TYPE.find(default_key);
  if (iter == DEFAULT_RESHAPE_TYPE.end()) {
    GELOGW("Dim size %zu is invalid, default reshape type not found.", origin_dim_size);
    return false;
  }

  reshape_type = iter->second;
  return true;
}

void ExpandDimension::ExpandDims(const int64_t &reshape_type, af::GeShape &shape) {
  GELOGD("Begin to expand dims, reshape type[%" PRId64 "], shape[%s].", reshape_type, shape.ToString().c_str());
  gert::Shape inner_shape;
  GeShapeToRtShape(shape, inner_shape);
  ExpandDims(reshape_type, inner_shape);
  RtShapeToGeShape(inner_shape, shape);
  GELOGD("After expanding dims, shape[%s].", shape.ToString().c_str());
}

void ExpandDimension::ExpandDims(const int64_t &reshape_type, const af::GeShape &origin_shape, af::GeShape &shape) {
  GELOGD("Begin to expand dims, reshape type[%" PRId64 "], origin shape[%s].", reshape_type,
         origin_shape.ToString().c_str());
  gert::Shape inner_ori_shape;
  GeShapeToRtShape(origin_shape, inner_ori_shape);
  gert::Shape inner_shape;
  GeShapeToRtShape(shape, inner_shape);
  ExpandDims(reshape_type, inner_ori_shape, inner_shape);
  RtShapeToGeShape(inner_shape, shape);
  GELOGD("After expanding dims, shape[%s].", shape.ToString().c_str());
}

void ExpandDimension::ExpandDims(const int64_t &reshape_type, gert::Shape &shape) {
  if (reshape_type == 0) {
    return;
  }
  gert::ExpandDimsType expand_dims_type(reshape_type);
  expand_dims_type.Expand(shape);
}

void ExpandDimension::ExpandDims(const int64_t &reshape_type, const gert::Shape &origin_shape, gert::Shape &shape) {
  if (reshape_type == 0) {
    return;
  }
  gert::ExpandDimsType expand_dims_type(reshape_type);
  expand_dims_type.Expand(origin_shape, shape);
}

bool ExpandDimension::GetFormatFullSize(const af::Format &format, size_t &full_size) {
  auto iter = FULL_SIZE_OF_FORMAT.find(format);
  if (iter == FULL_SIZE_OF_FORMAT.end()) {
    return false;
  }
  full_size = iter->second;
  return true;
}

int32_t ExpandDimension::GetAxisIndexByName(char ch, const af::Format &format) {
  uint32_t format_key = GenerateFormatKey(format);
  uint32_t axis_key = 0;
  axis_key = format_key | (static_cast<uint32_t>(ch) & 0xff);
  auto iter = AXIS_INDEX_OF_FORMAT.find(axis_key);
  if (iter == AXIS_INDEX_OF_FORMAT.end()) {
    return -1;
  }
  return iter->second;
}
int64_t ExpandDimension::GetReshapeAxicValue(const int64_t &reshape_type_mask, const af::GeShape &shape,
                                             int32_t axis_index) {
  GELOGD("axis_index = %d.", axis_index);
  if (axis_index == -1) {
    return -1;
  }
  gert::ExpandDimsType expand_dims_type(reshape_type_mask);
  if (!expand_dims_type.IsExpandIndex(axis_index)) {
    GELOGD("axis_index is %d.", axis_index);
  }
  return shape.GetDim(static_cast<size_t>(axis_index));
}
int64_t ExpandDimension::GetReshapeAxicValueByName(const int64_t &reshape_type_mask, char ch, const af::GeShape &shape,
                                                   const af::Format &format) {
  auto idx = GetAxisIndexByName(ch, format);
  return GetReshapeAxicValue(reshape_type_mask, shape, idx);
}
}  // namespace transformer
