/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/tensor_utils.h"

#include <cmath>

#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"
#include "mmpa/mmpa_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "base/err_msg.h"
#include "common/checker.h"

namespace af {
namespace {
// Unknown shape element num
const int64_t kElementCntUnknownShape = -1;

// Unknown shape mem size
const int64_t kUnknownShapeMemSize = -1;

// Nchw and nhwc dim size must be 4
const uint32_t kDimSize4d = 4U;

// C1HWNCoC0 dim size must be 6
const uint32_t kDimSizeC1hwncoc0 = 6U;

const int64_t kDataMemAlignSize = 32;
const int64_t kNum2 = 2;

const char_t *const kShapeRangeInvalid = "The format of the shape range is invalid";
const char_t *const kShapeRangeSample = "\"[1~20,3,3~6,-1]\"";
}  // namespace

static bool CheckMultiplyOverflowInt64(const int64_t &a, const int64_t &b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (std::numeric_limits<int64_t>::max() / b)) {
        return true;
      }
    } else {
      if (b < (std::numeric_limits<int64_t>::min() / a)) {
        return true;
      }
    }
  } else {
    if (b > 0) {
      if (a < (std::numeric_limits<int64_t>::min() / b)) {
        return true;
      }
    } else {
      if ((a != 0) && (b < (std::numeric_limits<int64_t>::max() / a))) {
        return true;
      }
    }
  }
  return false;
}

///
/// Calculate element num by dims directly.
/// @param dims dim info
/// @param element_cnt element count
/// @return GRAPH_SUCCESS:success
///         other:failed
///
static graphStatus CalcElementCntByDims(const std::vector<int64_t> &dims, int64_t &element_cnt) {
  element_cnt = 1;
  for (const int64_t dim : dims) {
    if (CheckMultiplyOverflowInt64(element_cnt, dim)) {
      REPORT_INNER_ERR_MSG("E18888", "result will overflow when multiplying %" PRId64 " and %" PRId64 ".", element_cnt,
                           dim);
      GELOGE(GRAPH_FAILED,
             "[Check][Overflow] CalcElementCntByDims failed, when multiplying %" PRId64 " and %" PRId64 ".",
             element_cnt, dim);
      return GRAPH_FAILED;
    }
    element_cnt *= dim;
  }
  return GRAPH_SUCCESS;
}

///
/// Calculate fixed dims element num.
/// @param dims dim info
/// @param fixed_dim_size fixed dim size
/// @param element_cnt element count
/// @return GRAPH_SUCCESS:success
///         other:failed
///
static graphStatus CalcElementCntOfFixedDims(const std::vector<int64_t> &dims, const Format format,
                                             const uint32_t fixed_dim_size, int64_t &element_cnt) {
  if (dims.size() != fixed_dim_size) {
    GELOGD("[Util][CalcElemCnt] Format %d(%s) need dim size=%u but %zu, calc as ND.", format,
           TypeUtils::FormatToSerialString(format).c_str(), fixed_dim_size, dims.size());
  }
  return CalcElementCntByDims(dims, element_cnt);
}

static graphStatus GetMaxShapeDimsFromNoTilingTensor(const GeTensorDesc &tensor_desc,
                                                     std::vector<int64_t> &output_dims) {
  const auto &shape = tensor_desc.GetShape();
  const std::vector<int64_t> &dims = shape.GetDims();
  std::vector<int64_t> max_shape_list;
  // use the max shape set by user
  const bool has_attr = AttrUtils::GetListInt(tensor_desc, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);
  if (has_attr) {
    if (max_shape_list.size() == dims.size()) {
      output_dims = std::move(max_shape_list);
      return GRAPH_SUCCESS;
    }
    REPORT_INNER_ERR_MSG("E18888", "invalid input shape range.");
    GELOGE(PARAM_INVALID, "[Check][Param]tensor invalid max_shape_list size[%zu], dim size[%zu].",
           max_shape_list.size(), dims.size());
    return PARAM_INVALID;
  }
  // if max shape attr not set, use shape range
  std::vector<std::pair<int64_t, int64_t>> range;
  const graphStatus graph_status = tensor_desc.GetShapeRange(range);
  if (graph_status != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "Get shape range failed.");
    GELOGE(PARAM_INVALID, "[Check][Param] GetShapeRange failed.");
    return graph_status;
  }
  if (dims.size() != range.size()) {
    REPORT_INNER_ERR_MSG("E18888", "Error shape range size.");
    GELOGE(PARAM_INVALID, "[Check][Param] size not matched dims_size[%zu] range_size[%zu].", dims.size(), range.size());
    return PARAM_INVALID;
  }
  for (size_t i = 0U; i < dims.size(); ++i) {
    const int64_t dim = (dims[i] < 0) ? range[i].second : dims[i];
    output_dims.push_back(dim);
  }
  return GRAPH_SUCCESS;
}

/// Calculate tensor element num.
/// @param dims dim info
/// @param format tensor format
/// @param data_type data type
/// @param element_cnt element count
/// @return GRAPH_SUCCESS:success
///         other:failed
///
static graphStatus CalcTensorElementCnt(const std::vector<int64_t> &dims, const Format format, const DataType data_type,
                                        int64_t &element_cnt) {
  const std::string format_str = TypeUtils::FormatToSerialString(format);
  // Check dims
  for (size_t i = 0U; i < dims.size(); ++i) {
    const int64_t dim = dims[i];
    if (dim < 0) {
      GELOGI("It's unknown shape, as dims[%zu]=%" PRId64 " negative, format=%d(%s).", i, dim, format,
             format_str.c_str());
      element_cnt = kElementCntUnknownShape;
      return GRAPH_SUCCESS;
    } else if (dim == 0) {
      GELOGI("No need calc element count, as dims[%zu]=%" PRId64 ", format=%d(%s).", i, dim, format,
             format_str.c_str());
      element_cnt = 0;
      return GRAPH_SUCCESS;
    } else {
      // else branch
    }
  }

  graphStatus graph_status;
  switch (GetPrimaryFormat(format)) {
    case FORMAT_ND:
    case FORMAT_MD:
      graph_status = CalcElementCntByDims(dims, element_cnt);
      break;
    case FORMAT_NCHW:
    case FORMAT_HWCN:
    case FORMAT_NHWC:
    case FORMAT_CHWN:
    case FORMAT_C1HWC0:
      graph_status = CalcElementCntOfFixedDims(dims, format, kDimSize4d, element_cnt);
      break;
    case FORMAT_C1HWNCoC0:
      graph_status = CalcElementCntOfFixedDims(dims, format, kDimSizeC1hwncoc0, element_cnt);
      break;
    case FORMAT_NC1HWC0:
    case FORMAT_FRACTAL_Z:
    case FORMAT_FILTER_HWCK:
    case FORMAT_FRACTAL_NZ:
    case FORMAT_FRACTAL_NZ_C0_16:
    case FORMAT_FRACTAL_NZ_C0_32:
    case FORMAT_FRACTAL_NZ_C0_2:
    case FORMAT_FRACTAL_NZ_C0_4:
    case FORMAT_FRACTAL_NZ_C0_8:
    case FORMAT_FRACTAL_ZZ:
    case FORMAT_NDHWC:
    case FORMAT_NCDHW:
    case FORMAT_DHWCN:
    case FORMAT_DHWNC:
    case FORMAT_FRACTAL_Z_3D:
    case FORMAT_FRACTAL_Z_3D_TRANSPOSE:
    case FORMAT_NDC1HWC0:
    case FORMAT_FRACTAL_Z_C04:
    case FORMAT_FRACTAL_ZN_LSTM:
    case FORMAT_NC1HWC0_C04:
    case FORMAT_ND_RNN_BIAS:
    case FORMAT_FRACTAL_ZN_RNN:
    case FORMAT_NYUV:
    case FORMAT_NYUV_A:
    case FORMAT_NCL:
    case FORMAT_FRACTAL_Z_WINO:
      graph_status = CalcElementCntByDims(dims, element_cnt);
      break;
    default:
      REPORT_INNER_ERR_MSG("E18888", "unsupported format, format=%d(%s).", format, format_str.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] unsupported format, format=%d(%s).", format, format_str.c_str());
      graph_status = GRAPH_FAILED;
      break;
  }

  const std::string type_str = TypeUtils::DataTypeToSerialString(data_type);
  if (graph_status == GRAPH_SUCCESS) {
    GELOGD("CalcTensorElementCnt end, format=%d(%s), data_type=%d(%s), element_cnt=%" PRId64 ".", format,
           format_str.c_str(), data_type, type_str.c_str(), element_cnt);
  } else {
    REPORT_INNER_ERR_MSG("E18888", "CalcTensorElementCnt failed, format=%d(%s), data_type=%d(%s).", format,
                         format_str.c_str(), data_type, type_str.c_str());
    GELOGE(GRAPH_FAILED, "[Calc][TensorElementCnt] failed, format=%d(%s), data_type=%d(%s).", format,
           format_str.c_str(), data_type, type_str.c_str());
  }
  return graph_status;
}

///
/// Calculate tensor mem size.
/// @param shape tensor shape
/// @param format tensor format
/// @param data_type tensor data type
/// @param mem_size -1 means unknown shape,other means mem size
/// @return GRAPH_SUCCESS:success, other:failed
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::CalcTensorMemSize(const GeShape &shape,
                                                                                          const Format format,
                                                                                          const DataType data_type,
                                                                                          int64_t &mem_size) {
  const std::string format_str = TypeUtils::FormatToSerialString(format);
  const std::string type_str = TypeUtils::DataTypeToSerialString(data_type);

  const std::vector<int64_t> dims = shape.GetDims();
  int64_t element_cnt = 0;
  const graphStatus status = CalcTensorElementCnt(dims, format, data_type, element_cnt);
  if (status != GRAPH_SUCCESS) {
    GELOGE(status, "[Calc][TensorElementCnt] failed, shape[%s], status=%u format=%d(%s) data_type=%d(%s).",
           shape.ToString().c_str(), status, format, format_str.c_str(), data_type, type_str.c_str());
    return status;
  }
  // Support unknown shape
  if (element_cnt < 0) {
    mem_size = kUnknownShapeMemSize;
    GELOGD("element_cnt is unknown. shape[%s], format=%d(%s), data_type=%d(%s), mem_size=%" PRId64,
           shape.ToString().c_str(), format, format_str.c_str(), data_type, type_str.c_str(), mem_size);
    return GRAPH_SUCCESS;
  }

  if ((data_type == DT_STRING) || (data_type == DT_STRING_REF)) {
    uint32_t type_size = 0U;
    const bool result = TypeUtils::GetDataTypeLength(data_type, type_size);
    if (!result) {
      REPORT_INNER_ERR_MSG("E18888", "GetDataTypeLength failed, data_type=%d(%s).", data_type, type_str.c_str());
      GELOGE(GRAPH_FAILED, "[Get][DataTypeLength] failed, data_type=%d(%s).", data_type, type_str.c_str());
      return GRAPH_FAILED;
    }
    const auto type_size_int64 = static_cast<int64_t>(type_size);
    GE_ASSERT_TRUE(!CheckMultiplyOverflowInt64(element_cnt, type_size_int64),
                   "[Check][Overflow] multiply %" PRId64 " and %u, shape[%s], format=%d(%s), data_type=%d(%s).",
                   element_cnt, type_size, shape.ToString().c_str(), format, format_str.c_str(), data_type,
                   type_str.c_str());
    mem_size = element_cnt * type_size_int64;
  } else {
    mem_size = ge::GetSizeInBytes(element_cnt, data_type);
  }

  GELOGD("shape[%s], format=%d(%s), data_type=%d(%s), mem_size=%" PRId64, shape.ToString().c_str(), format,
         format_str.c_str(), data_type, type_str.c_str(), mem_size);
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
TensorUtils::GetTensorMemorySizeInBytes(const GeTensorDesc &desc_temp, int64_t &size_temp) {
  const graphStatus graph_status = GetTensorSizeInBytes(desc_temp, size_temp);
  if (graph_status != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  // 64-byte alignment, if size is 0, align to 32 bytes
  if (size_temp > (std::numeric_limits<int64_t>::max() - (kNum2 * kDataMemAlignSize))) {
    GELOGW("[Util][CalcBytesSize] Mem size %" PRId64 " after alignment is bigger than INT64_MAX", size_temp);
  } else {
    size_temp = ((size_temp + (kNum2 * kDataMemAlignSize) - 1) / kDataMemAlignSize) * kDataMemAlignSize;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::CalcTensorMemSizeForNoTiling(
    const GeTensorDesc &tensor, const Format format, const DataType data_type, int64_t &mem_size) {
  if (tensor.GetShape().IsUnknownShape()) {
    std::vector<int64_t> dims;
    GE_CHK_STATUS_RET(GetMaxShapeDimsFromNoTilingTensor(tensor, dims),
                      "[Calc][GetMaxShapeDimsFromNoTilingTensor] failed.");
    return CalcTensorMemSize(GeShape(dims), format, data_type, mem_size);
  }
  return CalcTensorMemSize(tensor.GetShape(), format, data_type, mem_size);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
TensorUtils::GetTensorSizeInBytes(const GeTensorDesc &desc_temp, int64_t &size_temp) {
  const Format format = desc_temp.GetFormat();
  const DataType data_type = desc_temp.GetDataType();
  int64_t output_mem_size = 0;

  bool is_no_tiling = false;
  (void)AttrUtils::GetBool(desc_temp, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, is_no_tiling);
  graphStatus graph_status;
  if (is_no_tiling) {
    graph_status = CalcTensorMemSizeForNoTiling(desc_temp, format, data_type, output_mem_size);
  } else {
    graph_status = CalcTensorMemSize(desc_temp.GetShape(), format, data_type, output_mem_size);
  }
  if (graph_status != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Calc][TensorMemSize] failed! type:%s, is_no_tiling:%s",
           TypeUtils::DataTypeToSerialString(data_type).c_str(), is_no_tiling ? "true" : "false");
    return GRAPH_FAILED;
  }

  if (output_mem_size < 0) {
    REPORT_INNER_ERR_MSG("E18888",
                         "After calc concat tensor memory size, output_mem_size = %" PRId64
                         ","
                         " out of data range [0, %" PRId64 "]",
                         output_mem_size, std::numeric_limits<int64_t>::max());
    GELOGW(
        "[Check][Param] After calc concat tensor memory size, "
        "output_mem_size = %" PRId64 ", out of data range [0, %" PRId64 "]",
        output_mem_size, std::numeric_limits<int64_t>::max());
    return GRAPH_FAILED;
  }

  size_temp = output_mem_size;
  return GRAPH_SUCCESS;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
TensorUtils::CheckShapeByShapeRange(const GeShape &shape, const std::vector<std::pair<int64_t, int64_t>> &shape_range) {
  if ((shape.GetDimNum() == 0U) || shape_range.empty()) {
    GELOGD(" Shape or shape range is empty, no need to check.");
    return GRAPH_SUCCESS;
  }
  if (shape.GetDimNum() != shape_range.size()) {
    REPORT_PREDEFINED_ERR_MSG("E10049", std::vector<const char *>({"shape_range_size", "cur_dim_size"}),
                              std::vector<const char *>({std::to_string(shape_range.size()).c_str(),
                                                         std::to_string(shape.GetDimNum()).c_str()}));
    GELOGE(PARAM_INVALID,
           "[Check][Param] Given shape_range dim num [%zu] and current dim num [%zu] do not match. "
           "Please check",
           shape_range.size(), shape.GetDimNum());
    return PARAM_INVALID;
  }

  for (size_t idx = 0U; idx < shape.GetDimNum(); idx++) {
    const auto cur_dim = shape.GetDim(idx);
    if (cur_dim == UNKNOWN_DIM) {
      GELOGD("[Check][InputShape]cur shape dim [%" PRId64 "] is dynamic, no need to check.", cur_dim);
      continue;
    }
    const auto left_range = shape_range[idx].first;
    const auto right_range = shape_range[idx].second;
    if (left_range < 0) {
      const std::string error_range = std::to_string(left_range) + " ~ " + std::to_string(right_range);
      REPORT_PREDEFINED_ERR_MSG(
          "E10048", std::vector<const char *>({"shape_range", "reason", "sample"}),
          std::vector<const char *>({error_range.c_str(), kShapeRangeInvalid, kShapeRangeSample}));
      GELOGE(PARAM_INVALID, "[Check][Param] Given shape range[%s] is invalid, reason: %s, correct sample is %s.",
             error_range.c_str(), kShapeRangeInvalid, kShapeRangeSample);
      return PARAM_INVALID;
    }

    if (cur_dim < left_range) {
      REPORT_PREDEFINED_ERR_MSG(
          "E10050", std::vector<const char *>({"cur_dim", "shape_range_left", "shape_range_right"}),
          std::vector<const char *>({std::to_string(cur_dim).c_str(), std::to_string(left_range).c_str(),
                                     std::to_string(right_range).c_str()}));
      GELOGE(PARAM_INVALID,
             "[Check][Param] Current dim shape [%" PRId64
             "] is out of "
             "shape range [%" PRId64 "~%" PRId64 "]. Please check.",
             cur_dim, left_range, right_range);
      return PARAM_INVALID;
    }

    if (right_range < 0) {
      if (right_range != UNKNOWN_DIM) {
        const std::string error_range = std::to_string(left_range) + " ~ " + std::to_string(right_range);
        REPORT_PREDEFINED_ERR_MSG(
            "E10048", std::vector<const char *>({"shape_range", "reason", "sample"}),
            std::vector<const char *>({error_range.c_str(), kShapeRangeInvalid, kShapeRangeSample}));
        GELOGE(PARAM_INVALID, "[Check][Param] Given shape range[%s] is invalid, reason: %s, correct sample is %s.",
               error_range.c_str(), kShapeRangeInvalid, kShapeRangeSample);
        return PARAM_INVALID;
      }
    } else {
      if (cur_dim > right_range) {
        REPORT_PREDEFINED_ERR_MSG(
            "E10050", std::vector<const char *>({"cur_dim", "shape_range_left", "shape_range_right"}),
            std::vector<const char *>({std::to_string(cur_dim).c_str(), std::to_string(left_range).c_str(),
                                       std::to_string(right_range).c_str()}));
        GELOGE(PARAM_INVALID,
               "[Check][Param] Current dim shape [%" PRId64
               "] is out of "
               "shape range [%" PRId64 "~%" PRId64 "]. Please check.",
               cur_dim, left_range, right_range);
        return PARAM_INVALID;
      }
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace af
