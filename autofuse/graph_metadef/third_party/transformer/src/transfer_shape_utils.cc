/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transfer_shape_utils.h"
#include <mutex>
#include "axis_constants.h"
#include "common/ge_common/string_util.h"
#include "graph/ge_error_codes_af.h"
#include "platform/platform_info.h"
#include "graph/utils/type_utils.h"
#include "graph/types_af.h"
#include "expand_dimension.h"

namespace transformer {
std::array<uint32_t, static_cast<size_t>(ge::DataType::DT_MAX)> TransferShapeUtils::m0_list_{};
std::array<uint32_t, static_cast<size_t>(ge::DataType::DT_MAX)> TransferShapeUtils::k0_list_{};
std::array<uint32_t, static_cast<size_t>(ge::DataType::DT_MAX)> TransferShapeUtils::n0_list_{};
const std::map<TransferShapeType, GetAlignedShapeFunc> TransferShapeUtils::get_aligned_shape_func_map = {
    {TransferShapeType::ND_TO_ND, GetNdToNdAlignedShape},
    {TransferShapeType::ND_TO_NZ, GetNdToNzAlignedShape},
    {TransferShapeType::FULL_SIZE, GetFullSizeAlignedShape},
    {TransferShapeType::NOT_FULL_SIZE, GetNotFullSizeAlignedShape},
};
const std::map<TransferShapeType, TransferDimsFunc> TransferShapeUtils::transfer_dims_func_map = {
    {TransferShapeType::ND_TO_ND, GetNdToNdAxisIndexMapping},
    {TransferShapeType::ND_TO_NZ, GetNdToNzAxisIndexMapping},
    {TransferShapeType::FULL_SIZE, GetFullSizeAxisIndexMapping},
    {TransferShapeType::NOT_FULL_SIZE, GetNotFullSizeAxisIndexMapping},
};
namespace {
const int64_t SHAPE_NUMBER_32 = 32;
const int64_t SHAPE_NUMBER_16 = 16;
const int64_t SHAPE_NUMBER_8 = 8;
const int64_t SHAPE_NUMBER_4 = 4;
const int64_t SHAPE_NUMBER_2 = 2;
const int64_t NI = 16;
const int64_t LSTM_NI = 4;
const int64_t GROUPS_DEFAULT_VALUE = 1;
const int64_t UNKNOWN_SHAPE_VALUE = -1;
const int64_t RNN_STATE_SIZE_DEFAULT_VALUE = -1;
const size_t NUMBER_2 = 2;
const size_t MINUS_VALUE_ONE = 1;
const size_t MINUS_VALUE_TWO = 2;

const size_t DIM_INDEX_N = 0;
const size_t DIM_INDEX_C = 1;
const size_t DIM_INDEX_H = 2;
const size_t DIM_INDEX_W = 3;
const size_t DIM_INDEX_D = 4;
const size_t DIM_INDEX_ZERO = 0;
const size_t DIM_INDEX_ONE = 1;
const size_t DIM_INDEX_TWO = 2;
const size_t DIM_INDEX_THREE = 3;
const size_t DIM_INDEX_FOUR = 4;
const size_t kM0Index = 0;
const size_t kK0Index = 1;
const size_t kN0Index = 2;
const size_t kNzMinDimNum = 2;
constexpr size_t MOKOCO_CONFIG_SIZE = 3;
const std::string kPltDtypeMKN = "DtypeMKN";
const std::string kPltDefault = "Default";
const std::map<ge::Format, FormatIndex> kFormatIndexMap = {
    {ge::FORMAT_NCHW, {DIM_INDEX_ZERO, DIM_INDEX_ONE, DIM_INDEX_TWO, DIM_INDEX_THREE, DIM_INDEX_FOUR}},
    {ge::FORMAT_NHWC, {DIM_INDEX_ZERO, DIM_INDEX_THREE, DIM_INDEX_ONE, DIM_INDEX_TWO, DIM_INDEX_FOUR}},
    {ge::FORMAT_HWCN, {DIM_INDEX_THREE, DIM_INDEX_TWO, DIM_INDEX_ZERO, DIM_INDEX_ONE, DIM_INDEX_FOUR}},
    {ge::FORMAT_CHWN, {DIM_INDEX_THREE, DIM_INDEX_ZERO, DIM_INDEX_ONE, DIM_INDEX_TWO, DIM_INDEX_FOUR}},
    {ge::FORMAT_ND, {DIM_INDEX_ZERO, DIM_INDEX_ONE, DIM_INDEX_TWO, DIM_INDEX_THREE, DIM_INDEX_FOUR}},
    {ge::FORMAT_NCDHW, {DIM_INDEX_ZERO, DIM_INDEX_ONE, DIM_INDEX_THREE, DIM_INDEX_FOUR, DIM_INDEX_TWO}},
    {ge::FORMAT_NDHWC, {DIM_INDEX_ZERO, DIM_INDEX_FOUR, DIM_INDEX_TWO, DIM_INDEX_THREE, DIM_INDEX_ONE}},
    {ge::FORMAT_DHWCN, {DIM_INDEX_FOUR, DIM_INDEX_THREE, DIM_INDEX_ONE, DIM_INDEX_TWO, DIM_INDEX_ZERO}},
    {ge::FORMAT_DHWNC, {DIM_INDEX_THREE, DIM_INDEX_FOUR, DIM_INDEX_ONE, DIM_INDEX_TWO, DIM_INDEX_ZERO}}};

const std::set<ge::Format> kOriginFormatVec = {ge::FORMAT_NCHW,  ge::FORMAT_NHWC,  ge::FORMAT_HWCN,
                                               ge::FORMAT_CHWN,  ge::FORMAT_NDHWC, ge::FORMAT_NCDHW,
                                               ge::FORMAT_DHWCN, ge::FORMAT_DHWNC, ge::FORMAT_ND};

inline int64_t GetGreatestCommonDivisor(int64_t x, int64_t y) {
  if (y == 0) {
    return x;
  }
  int64_t z = y;
  while (x % y != 0) {
    z = x % y;
    x = y;
    y = z;
  }
  return z;
}

inline int64_t GetLeastCommonMultiple(int64_t x, int64_t y) {
  if (x == 0 || y == 0) {
    return 0;
  }
  return (x * y) / GetGreatestCommonDivisor(x, y);
}

inline int64_t GetAsisEnlargeValue(int64_t cin, int64_t cout, int64_t c0, int64_t group) {
  if (cin == 0 || cout == 0) {
    return 0;
  }

  return std::min(
      GetLeastCommonMultiple(c0 / GetGreatestCommonDivisor(cin, c0), NI / GetGreatestCommonDivisor(cout, NI)), group);
}
}  // namespace

bool TransferShapeUtils::InitM0K0CO(const fe::PlatFormInfos *platform_infos) {
  std::string default_mkn;
  fe::PlatFormInfos *temp_platform_infos = const_cast<fe::PlatFormInfos *>(platform_infos);
  if (temp_platform_infos->GetPlatformResWithLock(kPltDtypeMKN, kPltDefault, default_mkn)) {
    GELOGD("Default MKN value from platform is [%s].", default_mkn.c_str());
    std::vector<string> infos = af::StringUtils::Split(default_mkn, ',');
    if (infos.size() != MOKOCO_CONFIG_SIZE) {
      return false;
    }
    m0_list_.fill(static_cast<uint32_t>(std::atoi(infos[kM0Index].c_str())));
    k0_list_.fill(static_cast<uint32_t>(std::atoi(infos[kK0Index].c_str())));
    n0_list_.fill(static_cast<uint32_t>(std::atoi(infos[kN0Index].c_str())));
  }

  std::map<std::string, std::string> m0_k0_n0_info;
  temp_platform_infos->GetPlatformResWithLock(kPltDtypeMKN, m0_k0_n0_info);
  for (auto &item : m0_k0_n0_info) {
    if (item.first == kPltDefault) {
      continue;
    }
    ge::DataType dtype = ge::TypeUtils::SerialStringToDataType(item.first);
    if (dtype == ge::DT_UNDEFINED) {
      continue;
    }
    std::vector<string> infos = af::StringUtils::Split(item.second, ',');
    if (infos.size() != MOKOCO_CONFIG_SIZE) {
      continue;
    }
    m0_list_[static_cast<size_t>(dtype)] = static_cast<uint32_t>(std::atoi(infos[kM0Index].c_str()));
    k0_list_[static_cast<size_t>(dtype)] = static_cast<uint32_t>(std::atoi(infos[kK0Index].c_str()));
    n0_list_[static_cast<size_t>(dtype)] = static_cast<uint32_t>(std::atoi(infos[kN0Index].c_str()));
  }
  return true;
}

bool TransferShapeUtils::InitPlatformInfo(const fe::PlatFormInfos *platform_infos_ptr) {
  static std::once_flag flag;
  std::call_once(flag, [&platform_infos_ptr]() {
    m0_list_.fill(SHAPE_NUMBER_16);
    k0_list_.fill(SHAPE_NUMBER_16);
    n0_list_.fill(SHAPE_NUMBER_16);
    fe::PlatFormInfos platform_infos;
    fe::OptionalInfos optional_infos;
    if (platform_infos_ptr == nullptr) {
      GELOGI("Input platform is null; now retrieving m0k0c0 from platformmanager autonomously.");
      fe::PlatformInfoManager::GeInstance().InitializePlatformInfo();
      if (fe::PlatformInfoManager::GeInstance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos) != 0) {
        GELOGW("Failed to get platform info, using default MKN value.");
        return false;
      }
    }
    return platform_infos_ptr == nullptr ? TransferShapeUtils::InitM0K0CO(&platform_infos)
                                         : TransferShapeUtils::InitM0K0CO(platform_infos_ptr);
  });
  return true;
}

bool TransferShapeUtils::TransferShape(const ge::Format &origin_format, const ge::Format &format,
                                       const ge::DataType &data_type, const ExtAxisValue &ext_axis, gert::Shape &shape,
                                       const fe::PlatFormInfos *platform_infos_ptr) {
  if (!InitPlatformInfo(platform_infos_ptr)) {
    GELOGW("Init platform info failed");
  }
  ge::Format primary_format = static_cast<ge::Format>(GetPrimaryFormat(format));
  ge::Format origin_primary_format = static_cast<ge::Format>(GetPrimaryFormat(origin_format));
  GELOGD("Original format is %s, new format is %s",
         (ge::TypeUtils::FormatToSerialString(origin_primary_format)).c_str(),
         (ge::TypeUtils::FormatToSerialString(primary_format)).c_str());
  if (!IsNeedTransferShape(origin_primary_format, primary_format, shape)) {
    return true;
  }

  if (!CheckInputParam(origin_primary_format, primary_format, data_type)) {
    return false;
  }

  AxisValue axis_value;
  axis_value.fill(1);
  int64_t group = static_cast<int64_t>(ge::GetSubFormat(format));
  if (group > GROUPS_DEFAULT_VALUE) {
    axis_value[AXIS_G] = group;
  }

  axis_value[AXIS_C0] = GetC0Value(data_type, format);
  axis_value[AXIS_M0] = GetM0ByDtype(data_type);
  if (primary_format == ge::FORMAT_FRACTAL_ZN_RNN || primary_format == ge::FORMAT_ND_RNN_BIAS) {
    axis_value[AXIS_INPUT_SIZE] = ext_axis[EXT_INDEX_INPUT_SIZE];
    axis_value[AXIS_HIDDEN_SIZE] = ext_axis[EXT_INDEX_HIDDEN_SIZE];
    axis_value[AXIS_STATE_SIZE] = ext_axis[EXT_INDEX_STATE_SIZE];
  }

  if (!IsNeedAxisValue(primary_format, shape.GetDimNum())) {
    return TransferShapeByFormat(primary_format, axis_value, shape);
  }

  if (!AxisUtil::GetAxisValueByOriginFormat(origin_primary_format, shape, axis_value)) {
    return true;
  }

  return TransferShapeByAxisValue(primary_format, axis_value, shape);
}

bool TransferShapeUtils::TransferShape(const ge::Format &origin_format, const ge::Format &format,
                                       const ge::DataType &data_type, const ExtAxisValue &ext_axis,
                                       const gert::Shape &origin_shape, gert::Shape &shape,
                                       const fe::PlatFormInfos *platform_infos_ptr) {
  if (!InitPlatformInfo(platform_infos_ptr)) {
    GELOGW("Init platform info failed");
  }
  ge::Format primary_format = static_cast<ge::Format>(GetPrimaryFormat(format));
  ge::Format origin_primary_format = static_cast<ge::Format>(GetPrimaryFormat(origin_format));
  GELOGD("Transfer shape from original format[%s] to format [%s].",
         (ge::TypeUtils::FormatToSerialString(origin_primary_format)).c_str(),
         (ge::TypeUtils::FormatToSerialString(primary_format)).c_str());
  if (!IsNeedTransferShape(origin_primary_format, primary_format, origin_shape)) {
    return true;
  }

  if (!CheckInputParam(origin_primary_format, primary_format, data_type)) {
    return false;
  }

  int64_t c0 = GetC0Value(data_type, format);
  int64_t m0 = GetM0ByDtype(data_type);
  if (!IsNeedAxisValue(primary_format, origin_shape.GetDimNum())) {
    return TransferShapeByOriginShape(primary_format, c0, m0, ext_axis, origin_shape, shape);
  } else {
    return TransferShapeByFormatIndex(origin_primary_format, format, c0, origin_shape, shape);
  }
}

int64_t TransferShapeUtils::GetC0ByDtype(const ge::DataType &data_type) {
  if (static_cast<size_t>(data_type) < k0_list_.size()) {
    return static_cast<int64_t>(k0_list_[static_cast<size_t>(data_type)]);
  }
  return SHAPE_NUMBER_16;
}

int64_t TransferShapeUtils::GetM0ByDtype(const ge::DataType &data_type) {
  if (static_cast<size_t>(data_type) < m0_list_.size()) {
    return static_cast<int64_t>(m0_list_[static_cast<size_t>(data_type)]);
  }
  return SHAPE_NUMBER_16;
}

int64_t TransferShapeUtils::GetN0ByDtype(const ge::DataType &data_type) {
  if (static_cast<size_t>(data_type) < n0_list_.size()) {
    return static_cast<int64_t>(n0_list_[static_cast<size_t>(data_type)]);
  }
  return SHAPE_NUMBER_16;
}

bool TransferShapeUtils::IsNeedTransferShape(const ge::Format &origin_format, const ge::Format &format,
                                             const gert::Shape &shape) {
  if (origin_format == ge::FORMAT_ND && kOriginFormatVec.count(format) > 0) {
    GELOGD("No need for shape transformation from ND to the original format.");
    return false;
  }

  if (shape.IsScalar()) {
    GELOGD("Do not need to do shape transformation if the shape is scalar.");
    return false;
  }
  return true;
}

bool TransferShapeUtils::CheckInputParam(const ge::Format &origin_format, const ge::Format &primary_format,
                                         const ge::DataType &data_type) {
  bool invalid_format = (origin_format == ge::FORMAT_RESERVED || origin_format >= ge::FORMAT_END) ||
                        (primary_format == ge::FORMAT_RESERVED || primary_format >= ge::FORMAT_END);
  if (invalid_format) {
    GELOGE(ge::GRAPH_FAILED, "Origin format %u or new format %u is invalid.", origin_format, primary_format);
    return false;
  }

  if (data_type == ge::DT_UNDEFINED || data_type >= ge::DT_MAX) {
    GELOGE(ge::GRAPH_FAILED, "DataType %u is invalid.", origin_format);
    return false;
  }

  return true;
}

int64_t TransferShapeUtils::GetC0Value(const ge::DataType &data_type, const ge::Format &format) {
  // The value of C0 should be 4 while format is 5HD-4 or FRAZ-4
  ge::Format primary_format = static_cast<ge::Format>(GetPrimaryFormat(format));
  if (primary_format == ge::FORMAT_FRACTAL_NZ_C0_2) {
    return SHAPE_NUMBER_2;
  }
  if (primary_format == ge::FORMAT_NC1HWC0_C04 || primary_format == ge::FORMAT_FRACTAL_NZ_C0_4) {
    return SHAPE_NUMBER_4;
  }
  if (primary_format == ge::FORMAT_FRACTAL_NZ_C0_8) {
    return SHAPE_NUMBER_8;
  }
  if (primary_format == ge::FORMAT_FRACTAL_NZ_C0_16) {
    return SHAPE_NUMBER_16;
  }
  if (primary_format == ge::FORMAT_FRACTAL_NZ_C0_32) {
    return SHAPE_NUMBER_32;
  }

  if (ge::HasC0Format(format)) {
    return ge::GetC0Value(format);
  }

  return GetC0ByDtype(data_type);
}

bool TransferShapeUtils::IsNeedAxisValue(const ge::Format &format, const size_t &origin_dim_size) {
  if (kFormatNZSet.count(format) > 0 || format == ge::FORMAT_FRACTAL_ZN_RNN || format == ge::FORMAT_ND_RNN_BIAS ||
      format == ge::FORMAT_NYUV_A) {
    return false;
  }
  if (format == ge::FORMAT_FRACTAL_Z && origin_dim_size == DIM_SIZE_TWO) {
    return false;
  }
  return true;
}

bool TransferShapeUtils::TransferShapeByFormat(const ge::Format &primary_format, const AxisValue &axis_value,
                                               gert::Shape &shape) {
  switch (primary_format) {
    case ge::FORMAT_FRACTAL_Z:
      return GetFzShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_FRACTAL_NZ:
    case ge::FORMAT_FRACTAL_NZ_C0_2:
    case ge::FORMAT_FRACTAL_NZ_C0_4:
    case ge::FORMAT_FRACTAL_NZ_C0_8:
    case ge::FORMAT_FRACTAL_NZ_C0_16:
    case ge::FORMAT_FRACTAL_NZ_C0_32:
      return GetNzShapeByAxisValue(axis_value, shape);  // need c0
    case ge::FORMAT_FRACTAL_ZN_RNN:
      return GetFznRNNShapeByAxisValue(axis_value, shape);  // need c0, input, hidden, state
    case ge::FORMAT_ND_RNN_BIAS:
      return GetNDRNNShapeByAxisValue(axis_value, shape);  // need c0, input, hidden, state
    case ge::FORMAT_NYUV_A:
      return GetNYUVShape(shape);
    default:
      GELOGD("Cannot obtain new shape with format %d.", primary_format);
      return true;
  }
}

bool TransferShapeUtils::TransferShapeByAxisValue(const ge::Format &primary_format, const AxisValue &axis_value,
                                                  gert::Shape &shape) {
  switch (primary_format) {
    case ge::FORMAT_NCHW:
      return GetNCHWShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_NHWC:
      return GetNHWCShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_HWCN:
      return GetHWCNShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_CHWN:
      return GetCHWNShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_NDHWC:
      return GetNDHWCShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_NCDHW:
      return GetNCDHWShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_DHWCN:
      return GetDHWCNShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_DHWNC:
      return GetDHWNCShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_NC1HWC0:
    case ge::FORMAT_NC1HWC0_C04:
      return GetNC1HWC0ShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_C1HWC0:
      return GetC1HWC0ShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_NDC1HWC0:
      return GetNDC1HWC0ShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_C1HWNCoC0:
      return GetC1HWNCoC0ShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_FRACTAL_Z:
      return GetFzShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_FRACTAL_Z_WINO:
      return GetFzWinoShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_FRACTAL_Z_C04:
      return GetFzC04ShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_FRACTAL_Z_3D:
      return GetFz3DShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE:
      return GetFz3DTransposeShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_FRACTAL_ZN_LSTM:
      return GetFzLstmShapeByAxisValue(axis_value, shape);
    default:
      GELOGD("Cannot obtain new shape with format %d.", primary_format);
      return true;
  }
}

bool TransferShapeUtils::TransferShapeByOriginShape(const ge::Format &primary_format, const int64_t &c0,
                                                    const int64_t &m0, const ExtAxisValue &ext_axis,
                                                    const gert::Shape &origin_shape, gert::Shape &shape) {
  switch (primary_format) {
    case ge::FORMAT_FRACTAL_Z:
      return GetFractalZShape(c0, origin_shape, shape);
    case ge::FORMAT_FRACTAL_NZ:
    case ge::FORMAT_FRACTAL_NZ_C0_2:
    case ge::FORMAT_FRACTAL_NZ_C0_4:
    case ge::FORMAT_FRACTAL_NZ_C0_8:
    case ge::FORMAT_FRACTAL_NZ_C0_16:
    case ge::FORMAT_FRACTAL_NZ_C0_32:
      return GetFractalNzShape(c0, m0, origin_shape, shape);  // need c0
    case ge::FORMAT_FRACTAL_ZN_RNN:
      return GetFractalZnRnnShape(ext_axis, c0, origin_shape, shape);  // need c0, input, hidden, state
    case ge::FORMAT_ND_RNN_BIAS:
      return GetNdRnnBiasShape(ext_axis, c0, origin_shape, shape);  // need c0, input, hidden, state
    default:
      GELOGD("Cannot obtain new shape with format %d.", primary_format);
      return true;
  }
}

bool TransferShapeUtils::TransferShapeByFormatIndex(const ge::Format &origin_format, const ge::Format &format,
                                                    const int64_t &c0, const gert::Shape &origin_shape,
                                                    gert::Shape &shape) {
  std::map<ge::Format, FormatIndex>::const_iterator iter = kFormatIndexMap.find(origin_format);
  if (iter == kFormatIndexMap.end()) {
    return true;
  }
  ge::Format primary_format = static_cast<ge::Format>(GetPrimaryFormat(format));
  int64_t group = static_cast<int64_t>(ge::GetSubFormat(format));
  switch (primary_format) {
    case ge::FORMAT_NCHW:
      return GetNCHWShape(iter->second, origin_shape, shape);
    case ge::FORMAT_NHWC:
      return GetNHWCShape(iter->second, origin_shape, shape);
    case ge::FORMAT_HWCN:
      return GetHWCNShape(iter->second, origin_shape, shape);
    case ge::FORMAT_CHWN:
      return GetCHWNShape(iter->second, origin_shape, shape);
    case ge::FORMAT_NDHWC:
      return GetNDHWCShape(iter->second, origin_shape, shape);
    case ge::FORMAT_NCDHW:
      return GetNCDHWShape(iter->second, origin_shape, shape);
    case ge::FORMAT_DHWCN:
      return GetDHWCNShape(iter->second, origin_shape, shape);
    case ge::FORMAT_DHWNC:
      return GetDHWNCShape(iter->second, origin_shape, shape);
    case ge::FORMAT_NC1HWC0:
    case ge::FORMAT_NC1HWC0_C04:
      return GetNC1HWC0Shape(iter->second, c0, origin_shape, shape);
    case ge::FORMAT_C1HWC0:
      return GetC1HWC0Shape(iter->second, c0, origin_shape, shape);
    case ge::FORMAT_NDC1HWC0:
      return GetNDC1HWC0Shape(iter->second, c0, origin_shape, shape);
    case ge::FORMAT_C1HWNCoC0:
      return GetC1HWNCoC0Shape(iter->second, c0, origin_shape, shape);
    case ge::FORMAT_FRACTAL_Z:
      return GetFractalZShape(iter->second, c0, group, origin_shape, shape);
    case ge::FORMAT_FRACTAL_Z_3D:
      return GetFractalZ3DShape(iter->second, c0, group, origin_shape, shape);
    case ge::FORMAT_FRACTAL_Z_C04:
      return GetFractalZC04Shape(iter->second, c0, origin_shape, shape);
    case ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE:
      return GetFractalZ3DTransposeShape(iter->second, c0, origin_shape, shape);
    case ge::FORMAT_FRACTAL_ZN_LSTM:
      return GetFractalZLstmShape(iter->second, origin_shape, shape);
    case ge::FORMAT_FRACTAL_Z_WINO:
      return GetFractalZWinoShape(iter->second, c0, origin_shape, shape);
    default: {
      GELOGD("Cannot obtain new shape with format %d.", primary_format);
      return true;
    }
  }
}

bool TransferShapeUtils::GetNCHWShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_N]);
  shape.AppendDim(axis_value[AXIS_C]);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  return true;
}

bool TransferShapeUtils::GetNHWCShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_N]);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  shape.AppendDim(axis_value[AXIS_C]);
  return true;
}

bool TransferShapeUtils::GetHWCNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  shape.AppendDim(axis_value[AXIS_C]);
  shape.AppendDim(axis_value[AXIS_N]);
  return true;
}

bool TransferShapeUtils::GetCHWNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_C]);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  shape.AppendDim(axis_value[AXIS_N]);
  return true;
}

bool TransferShapeUtils::GetNDHWCShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_N]);
  shape.AppendDim(axis_value[AXIS_D]);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  shape.AppendDim(axis_value[AXIS_C]);
  return true;
}

bool TransferShapeUtils::GetNCDHWShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_N]);
  shape.AppendDim(axis_value[AXIS_C]);
  shape.AppendDim(axis_value[AXIS_D]);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  return true;
}

bool TransferShapeUtils::GetDHWCNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_D]);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  shape.AppendDim(axis_value[AXIS_C]);
  shape.AppendDim(axis_value[AXIS_N]);
  return true;
}

bool TransferShapeUtils::GetDHWNCShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_D]);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  shape.AppendDim(axis_value[AXIS_N]);
  shape.AppendDim(axis_value[AXIS_C]);
  return true;
}

bool TransferShapeUtils::GetNC1HWC0ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_N]);
  shape.AppendDim(axis_value[AXIS_C1]);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool TransferShapeUtils::GetC1HWC0ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(DivisionCeiling(axis_value[AXIS_N], axis_value[AXIS_C0]));
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool TransferShapeUtils::GetNDC1HWC0ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_N]);
  shape.AppendDim(axis_value[AXIS_D]);
  shape.AppendDim(axis_value[AXIS_C1]);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool TransferShapeUtils::GetC1HWNCoC0ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_C1]);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  shape.AppendDim(DivisionCeiling(axis_value[AXIS_N], NI));
  shape.AppendDim(axis_value[AXIS_Co]);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool TransferShapeUtils::GetFzWinoShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_C1]);
  shape.AppendDim(DivisionCeiling(axis_value[AXIS_N], SHAPE_NUMBER_16));
  shape.AppendDim(NUMBER_2);
  int64_t hw = axis_value[AXIS_H];
  MUL_OVERFLOW(hw, axis_value[AXIS_W], hw);
  shape.AppendDim(hw);
  shape.AppendDim(SHAPE_NUMBER_8);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool TransferShapeUtils::GetNzShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  CHECK(shape.IsScalar(), GELOGD("Origin shape is empty."), return true);
  size_t dim_size = shape.GetDimNum();
  if (dim_size < DIM_SIZE_TWO) {
    GELOGD("nd_value's dim num is less than 2.");
    shape.AppendDim(1);
    dim_size++;
  }
  /* dim_size - 1 mean the last value of original vec
   * dim_size - 2 mean the second last value of original vec */
  int64_t dim_back_two = shape.GetDim(dim_size - MINUS_VALUE_TWO);
  int64_t dim_back_one = shape.GetDim(dim_size - MINUS_VALUE_ONE);
  shape.SetDim((dim_size - MINUS_VALUE_ONE), DivisionCeiling(dim_back_two, axis_value[AXIS_M0]));
  shape.SetDim((dim_size - MINUS_VALUE_TWO), DivisionCeiling(dim_back_one, axis_value[AXIS_C0]));
  shape.AppendDim(axis_value[AXIS_M0]);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool TransferShapeUtils::GetFzShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  size_t size_of_original_vec = shape.GetDimNum();
  if (size_of_original_vec == DIM_SIZE_TWO) {
    /* size_of_original_vec - 1 mean the last value of original vec
     * size_of_original_vec - 2 mean the second last value of original vec */
    shape.SetDim((size_of_original_vec - MINUS_VALUE_ONE),
                 DivisionCeiling(shape.GetDim(size_of_original_vec - MINUS_VALUE_ONE), SHAPE_NUMBER_16));
    shape.SetDim((size_of_original_vec - MINUS_VALUE_TWO),
                 DivisionCeiling(shape.GetDim(size_of_original_vec - MINUS_VALUE_TWO), axis_value[AXIS_C0]));
    shape.AppendDim(SHAPE_NUMBER_16);
    shape.AppendDim(axis_value[AXIS_C0]);
    return true;
  }
  return GetFz3DShapeByAxisValue(axis_value, shape);
}

bool TransferShapeUtils::GetFz3DShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  bool has_unknown_shape = axis_value[AXIS_D] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_H] == UNKNOWN_SHAPE_VALUE ||
                           axis_value[AXIS_W] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_C] == UNKNOWN_SHAPE_VALUE;
  int64_t gdhwc1 = UNKNOWN_SHAPE_VALUE;
  int64_t axis_g_val = GROUPS_DEFAULT_VALUE;
  int64_t axis_n_val = axis_value[AXIS_N];
  int64_t axis_c_val = axis_value[AXIS_C];
  int64_t axis_c1_val = axis_value[AXIS_C1];
  if (!has_unknown_shape) {
    if (axis_value[AXIS_G] > GROUPS_DEFAULT_VALUE && axis_n_val >= axis_value[AXIS_G]) {
      axis_n_val = axis_n_val / axis_value[AXIS_G];
      int64_t enlarge_value = GetAsisEnlargeValue(axis_c_val, axis_n_val, axis_value[AXIS_C0], axis_value[AXIS_G]);
      axis_g_val = DivisionCeiling(axis_value[AXIS_G], enlarge_value);
      MUL_OVERFLOW(axis_c_val, enlarge_value, axis_c_val);
      MUL_OVERFLOW(axis_n_val, enlarge_value, axis_n_val);
      axis_c1_val = DivisionCeiling(axis_c_val, axis_value[AXIS_C0]);
    }
    MUL_OVERFLOW(axis_g_val, axis_c1_val, gdhwc1);
    MUL_OVERFLOW(gdhwc1, axis_value[AXIS_D], gdhwc1);
    MUL_OVERFLOW(gdhwc1, axis_value[AXIS_H], gdhwc1);
    MUL_OVERFLOW(gdhwc1, axis_value[AXIS_W], gdhwc1);
  }
  shape.SetDimNum(0);
  shape.AppendDim(gdhwc1);
  shape.AppendDim(DivisionCeiling(axis_n_val, NI));
  shape.AppendDim(NI);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool TransferShapeUtils::GetFz3DTransposeShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  int64_t dhwn1 = UNKNOWN_SHAPE_VALUE;
  if (axis_value[AXIS_N] != UNKNOWN_SHAPE_VALUE && axis_value[AXIS_H] != UNKNOWN_SHAPE_VALUE &&
      axis_value[AXIS_W] != UNKNOWN_SHAPE_VALUE && axis_value[AXIS_D] != UNKNOWN_SHAPE_VALUE) {
    dhwn1 = DivisionCeiling(axis_value[AXIS_N], NI);
    MUL_OVERFLOW(dhwn1, axis_value[AXIS_D], dhwn1);
    MUL_OVERFLOW(dhwn1, axis_value[AXIS_H], dhwn1);
    MUL_OVERFLOW(dhwn1, axis_value[AXIS_W], dhwn1);
  }

  shape.SetDimNum(0);
  shape.AppendDim(dhwn1);
  if (axis_value[AXIS_C] == UNKNOWN_SHAPE_VALUE) {
    shape.AppendDim(UNKNOWN_SHAPE_VALUE);
  } else {
    shape.AppendDim(axis_value[AXIS_C1]);
  }
  shape.AppendDim(NI);
  shape.AppendDim(axis_value[AXIS_C0]);

  return true;
}

bool TransferShapeUtils::GetFzLstmShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  int64_t h = axis_value[AXIS_N] >> NUMBER_2;
  int64_t i = axis_value[AXIS_C] - h;
  int64_t first_element_of_fz_lstm = DivisionCeiling(i, NI) + DivisionCeiling(h, NI);
  int64_t second_element_of_fz_lstm = DivisionCeiling(h, NI);
  MUL_OVERFLOW(second_element_of_fz_lstm, LSTM_NI, second_element_of_fz_lstm);
  if (axis_value[AXIS_N] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_C] == UNKNOWN_SHAPE_VALUE) {
    first_element_of_fz_lstm = UNKNOWN_SHAPE_VALUE;
    second_element_of_fz_lstm = UNKNOWN_SHAPE_VALUE;
  }
  shape.SetDimNum(0);
  shape.AppendDim(first_element_of_fz_lstm);
  shape.AppendDim(second_element_of_fz_lstm);
  shape.AppendDim(NI);
  shape.AppendDim(NI);
  return true;
}

bool TransferShapeUtils::GetFzC04ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  if (axis_value[AXIS_H] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_W] == UNKNOWN_SHAPE_VALUE) {
    shape.AppendDim(UNKNOWN_SHAPE_VALUE);
  } else {
    int64_t x = SHAPE_NUMBER_4;
    MUL_OVERFLOW(x, axis_value[AXIS_H], x);
    MUL_OVERFLOW(x, axis_value[AXIS_W], x);
    shape.AppendDim(DivisionCeiling(x, axis_value[AXIS_C0]));
  }
  shape.AppendDim(DivisionCeiling(axis_value[AXIS_N], NI));
  shape.AppendDim(NI);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool TransferShapeUtils::GetFznRNNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  size_t origin_shape_size = shape.GetDimNum();
  CHECK(origin_shape_size < DIM_SIZE_TWO, GELOGW("The number of dimensions in ndValue is less than 2!"), return true);
  /* Check ND shape value. */
  int64_t k_value = shape.GetDim(origin_shape_size - MINUS_VALUE_TWO);
  int64_t hidden_or_state_size = axis_value[AXIS_HIDDEN_SIZE];
  if (axis_value[AXIS_STATE_SIZE] != RNN_STATE_SIZE_DEFAULT_VALUE) {
    hidden_or_state_size = axis_value[AXIS_STATE_SIZE];
  }

  if (k_value == hidden_or_state_size + axis_value[AXIS_INPUT_SIZE]) {
    // use input size and hidden size
    shape.SetDim(origin_shape_size - MINUS_VALUE_TWO, DivisionCeiling(axis_value[AXIS_INPUT_SIZE], SHAPE_NUMBER_16) +
                                                          DivisionCeiling(hidden_or_state_size, SHAPE_NUMBER_16));
  } else if (k_value == hidden_or_state_size || k_value == axis_value[AXIS_INPUT_SIZE]) {
    // only use hidden size or input size
    shape.SetDim(origin_shape_size - MINUS_VALUE_TWO, DivisionCeiling(k_value, SHAPE_NUMBER_16));
  } else {
    return true;
  }

  int64_t n_value = shape.GetDim(origin_shape_size - MINUS_VALUE_ONE);
  INT64_ZEROCHECK(axis_value[AXIS_HIDDEN_SIZE]);
  int64_t n_num = n_value / axis_value[AXIS_HIDDEN_SIZE];
  MUL_OVERFLOW(n_num, DivisionCeiling(axis_value[AXIS_HIDDEN_SIZE], axis_value[AXIS_C0]), n_num);
  shape.SetDim(origin_shape_size - MINUS_VALUE_ONE, n_num);
  shape.AppendDim(SHAPE_NUMBER_16);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool TransferShapeUtils::GetNDRNNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  CHECK(axis_value[AXIS_HIDDEN_SIZE] == 0, GELOGD("hidden_size is zero"), return true);
  size_t size_of_original_vec = shape.GetDimNum();
  /* Check ND shape value. */
  int64_t n_num = shape.GetDim(size_of_original_vec - MINUS_VALUE_ONE) / axis_value[AXIS_HIDDEN_SIZE];
  MUL_OVERFLOW(n_num, DivisionCeiling(axis_value[AXIS_HIDDEN_SIZE], axis_value[AXIS_C0]), n_num);
  MUL_OVERFLOW(n_num, axis_value[AXIS_C0], n_num);
  shape.SetDim(size_of_original_vec - MINUS_VALUE_ONE, n_num);
  return true;
}

bool TransferShapeUtils::GetNCHWShape(const FormatIndex &format_index, const gert::Shape &origin_shape,
                                      gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_N]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_C]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  return true;
}

bool TransferShapeUtils::GetNHWCShape(const FormatIndex &format_index, const gert::Shape &origin_shape,
                                      gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_N]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_C]));
  return true;
}

bool TransferShapeUtils::GetHWCNShape(const FormatIndex &format_index, const gert::Shape &origin_shape,
                                      gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_C]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_N]));
  return true;
}

bool TransferShapeUtils::GetCHWNShape(const FormatIndex &format_index, const gert::Shape &origin_shape,
                                      gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_C]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_N]));
  return true;
}

bool TransferShapeUtils::GetNDHWCShape(const FormatIndex &format_index, const gert::Shape &origin_shape,
                                       gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FIVE, GELOGD("Dim size is less than 5."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_N]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_D]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_C]));
  return true;
}

bool TransferShapeUtils::GetNCDHWShape(const FormatIndex &format_index, const gert::Shape &origin_shape,
                                       gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FIVE, GELOGD("Dim size is less than 5."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_N]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_C]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_D]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  return true;
}

bool TransferShapeUtils::GetDHWCNShape(const FormatIndex &format_index, const gert::Shape &origin_shape,
                                       gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FIVE, GELOGD("Dim size is less than 5."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_D]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_C]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_N]));
  return true;
}

bool TransferShapeUtils::GetDHWNCShape(const FormatIndex &format_index, const gert::Shape &origin_shape,
                                       gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FIVE, GELOGD("Dim size is less than 5."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_D]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_N]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_C]));
  return true;
}

bool TransferShapeUtils::GetNC1HWC0Shape(const FormatIndex &format_index, const int64_t &c0,
                                         const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_N]));
  shape.AppendDim(DivisionCeiling(origin_shape.GetDim(format_index[DIM_INDEX_C]), c0));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetC1HWC0Shape(const FormatIndex &format_index, const int64_t &c0,
                                        const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(DivisionCeiling(origin_shape.GetDim(format_index[DIM_INDEX_N]), c0));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetNDC1HWC0Shape(const FormatIndex &format_index, const int64_t &c0,
                                          const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_N]));
  if (origin_shape.GetDimNum() == DIM_SIZE_FOUR) {
    shape.AppendDim(1);
  } else {
    shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_D]));
  }
  shape.AppendDim(DivisionCeiling(origin_shape.GetDim(format_index[DIM_INDEX_C]), c0));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetC1HWNCoC0Shape(const FormatIndex &format_index, const int64_t &c0,
                                           const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(DivisionCeiling(origin_shape.GetDim(format_index[DIM_INDEX_C]), c0));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  shape.AppendDim(DivisionCeiling(origin_shape.GetDim(format_index[DIM_INDEX_N]), NI));
  shape.AppendDim(c0);
  shape.AppendDim(c0);
  return true;
}
bool TransferShapeUtils::GetFractalNzShape(const int64_t &c0, const int64_t &m0, const gert::Shape &origin_shape,
                                           gert::Shape &shape) {
  size_t dim_size = origin_shape.GetDimNum();
  shape.SetDimNum(0);
  if (dim_size > DIM_SIZE_TWO) {
    for (size_t i = 0; i < dim_size - DIM_SIZE_TWO; i++) {
      shape.AppendDim(origin_shape.GetDim(i));
    }
  }

  /* dim_size - 1 mean the last value of original vec
   * dim_size - 2 mean the second last value of original vec */
  if (dim_size < DIM_SIZE_TWO) {
    shape.AppendDim(1);
    shape.AppendDim(DivisionCeiling(origin_shape.GetDim(dim_size - MINUS_VALUE_ONE), m0));
  } else {
    shape.AppendDim(DivisionCeiling(origin_shape.GetDim(dim_size - MINUS_VALUE_ONE), c0));
    shape.AppendDim(DivisionCeiling(origin_shape.GetDim(dim_size - MINUS_VALUE_TWO), m0));
  }
  shape.AppendDim(m0);
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetFractalZShape(const int64_t &c0, const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() != DIM_SIZE_TWO, GELOGD("Dim size is not 2."), return true);
  /* size_of_original_vec - 1 mean the last value of original vec
   * size_of_original_vec - 2 mean the second last value of original vec */
  shape.SetDimNum(0);
  shape.AppendDim(DivisionCeiling(origin_shape.GetDim(DIM_SIZE_TWO - MINUS_VALUE_TWO), c0));
  shape.AppendDim(DivisionCeiling(origin_shape.GetDim(DIM_SIZE_TWO - MINUS_VALUE_ONE), SHAPE_NUMBER_16));
  shape.AppendDim(SHAPE_NUMBER_16);
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetFractalZShape(const FormatIndex &format_index, const int64_t &c0, const int64_t &group,
                                          const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  int64_t axis_n_val = origin_shape.GetDim(format_index[DIM_INDEX_N]);
  int64_t axis_c_val = origin_shape.GetDim(format_index[DIM_INDEX_C]);
  int64_t axis_h_val = origin_shape.GetDim(format_index[DIM_INDEX_H]);
  int64_t axis_w_val = origin_shape.GetDim(format_index[DIM_INDEX_W]);
  int64_t ghwc1 = UNKNOWN_SHAPE_VALUE;
  bool is_unknown_shape =
      axis_c_val == UNKNOWN_SHAPE_VALUE || axis_h_val == UNKNOWN_SHAPE_VALUE || axis_w_val == UNKNOWN_SHAPE_VALUE;
  if (!is_unknown_shape) {
    int64_t axis_g_val = GROUPS_DEFAULT_VALUE;
    int64_t axis_c1_val = 0;
    if (group > GROUPS_DEFAULT_VALUE && axis_n_val >= group) {
      axis_n_val = axis_n_val / group;
      int64_t enlarge_value = GetAsisEnlargeValue(axis_c_val, axis_n_val, c0, group);
      axis_g_val = DivisionCeiling(group, enlarge_value);
      MUL_OVERFLOW(axis_c_val, enlarge_value, axis_c_val);
      MUL_OVERFLOW(axis_n_val, enlarge_value, axis_n_val);
      axis_c1_val = DivisionCeiling(axis_c_val, c0);
    } else {
      axis_c1_val = DivisionCeiling(axis_c_val, c0);
    }

    MUL_OVERFLOW(axis_g_val, axis_c1_val, ghwc1);
    MUL_OVERFLOW(ghwc1, origin_shape.GetDim(format_index[DIM_INDEX_H]), ghwc1);
    MUL_OVERFLOW(ghwc1, origin_shape.GetDim(format_index[DIM_INDEX_W]), ghwc1);
  }

  shape.SetDimNum(0);
  shape.AppendDim(ghwc1);
  shape.AppendDim(DivisionCeiling(axis_n_val, NI));
  shape.AppendDim(NI);
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetFractalZWinoShape(const FormatIndex &format_index, const int64_t &c0,
                                              const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  int64_t axis_n_val = origin_shape.GetDim(format_index[DIM_INDEX_N]);
  int64_t axis_c_val = origin_shape.GetDim(format_index[DIM_INDEX_C]);
  int64_t axis_h_val = origin_shape.GetDim(format_index[DIM_INDEX_H]);
  int64_t axis_w_val = origin_shape.GetDim(format_index[DIM_INDEX_W]);
  int64_t ghw = axis_h_val;
  MUL_OVERFLOW(ghw, axis_w_val, ghw);  // HW

  shape.SetDimNum(0);
  shape.AppendDim(DivisionCeiling(axis_c_val, c0));
  shape.AppendDim(DivisionCeiling(axis_n_val, NI));  // n1
  shape.AppendDim(NUMBER_2);
  shape.AppendDim(ghw);
  shape.AppendDim(SHAPE_NUMBER_8);
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetFractalZ3DShape(const FormatIndex &format_index, const int64_t &c0, const int64_t &group,
                                            const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FIVE, GELOGD("Dim size is less than 5."), return true);
  int64_t axis_n_val = origin_shape.GetDim(format_index[DIM_INDEX_N]);
  int64_t axis_c_val = origin_shape.GetDim(format_index[DIM_INDEX_C]);
  int64_t axis_h_val = origin_shape.GetDim(format_index[DIM_INDEX_H]);
  int64_t axis_w_val = origin_shape.GetDim(format_index[DIM_INDEX_W]);
  int64_t axis_d_val = origin_shape.GetDim(format_index[DIM_INDEX_D]);
  int64_t gdhwc1 = UNKNOWN_SHAPE_VALUE;
  bool is_unknown_shape = axis_c_val == UNKNOWN_SHAPE_VALUE || axis_d_val == UNKNOWN_SHAPE_VALUE ||
                          axis_h_val == UNKNOWN_SHAPE_VALUE || axis_w_val == UNKNOWN_SHAPE_VALUE;
  if (!is_unknown_shape) {
    int64_t axis_c1_val = 0;
    int64_t axis_g_val = GROUPS_DEFAULT_VALUE;
    if (group > GROUPS_DEFAULT_VALUE && axis_n_val >= group) {
      axis_n_val = axis_n_val / group;
      int64_t enlarge_value = GetAsisEnlargeValue(axis_c_val, axis_n_val, c0, group);
      axis_g_val = DivisionCeiling(group, enlarge_value);
      MUL_OVERFLOW(axis_c_val, enlarge_value, axis_c_val);
      MUL_OVERFLOW(axis_n_val, enlarge_value, axis_n_val);
      axis_c1_val = DivisionCeiling(axis_c_val, c0);
    } else {
      axis_c1_val = DivisionCeiling(axis_c_val, c0);
    }
    MUL_OVERFLOW(axis_g_val, axis_c1_val, gdhwc1);
    MUL_OVERFLOW(gdhwc1, origin_shape.GetDim(format_index[DIM_INDEX_D]), gdhwc1);
    MUL_OVERFLOW(gdhwc1, origin_shape.GetDim(format_index[DIM_INDEX_H]), gdhwc1);
    MUL_OVERFLOW(gdhwc1, origin_shape.GetDim(format_index[DIM_INDEX_W]), gdhwc1);
  }

  shape.SetDimNum(0);
  shape.AppendDim(gdhwc1);
  shape.AppendDim(DivisionCeiling(axis_n_val, NI));
  shape.AppendDim(NI);
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetFractalZ3DTransposeShape(const FormatIndex &format_index, const int64_t &c0,
                                                     const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FIVE, GELOGD("Dim size is less than 5."), return true);
  int64_t dhwn1 = UNKNOWN_SHAPE_VALUE;
  if (origin_shape.GetDim(format_index[DIM_INDEX_N]) != UNKNOWN_SHAPE_VALUE &&
      origin_shape.GetDim(format_index[DIM_INDEX_H]) != UNKNOWN_SHAPE_VALUE &&
      origin_shape.GetDim(format_index[DIM_INDEX_W]) != UNKNOWN_SHAPE_VALUE &&
      origin_shape.GetDim(format_index[DIM_INDEX_D]) != UNKNOWN_SHAPE_VALUE) {
    dhwn1 = DivisionCeiling(origin_shape.GetDim(format_index[DIM_INDEX_N]), NI);
    MUL_OVERFLOW(dhwn1, origin_shape.GetDim(format_index[DIM_INDEX_D]), dhwn1);
    MUL_OVERFLOW(dhwn1, origin_shape.GetDim(format_index[DIM_INDEX_H]), dhwn1);
    MUL_OVERFLOW(dhwn1, origin_shape.GetDim(format_index[DIM_INDEX_W]), dhwn1);
  }

  shape.SetDimNum(0);
  shape.AppendDim(dhwn1);
  if (origin_shape.GetDim(format_index[DIM_INDEX_C]) == UNKNOWN_SHAPE_VALUE) {
    shape.AppendDim(UNKNOWN_SHAPE_VALUE);
  } else {
    shape.AppendDim(DivisionCeiling(origin_shape.GetDim(format_index[DIM_INDEX_C]), c0));
  }
  shape.AppendDim(NI);
  shape.AppendDim(c0);

  return true;
}

bool TransferShapeUtils::GetFractalZLstmShape(const FormatIndex &format_index, const gert::Shape &origin_shape,
                                              gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  int64_t axis_n_val = origin_shape.GetDim(format_index[DIM_INDEX_N]);
  int64_t axis_c_val = origin_shape.GetDim(format_index[DIM_INDEX_C]);
  int64_t h = axis_n_val / LSTM_NI;
  int64_t i = axis_c_val - h;
  int64_t first_element_of_fz_lstm = DivisionCeiling(i, NI) + DivisionCeiling(h, NI);
  int64_t second_element_of_fz_lstm = DivisionCeiling(h, NI);
  MUL_OVERFLOW(second_element_of_fz_lstm, LSTM_NI, second_element_of_fz_lstm);
  if (axis_n_val == UNKNOWN_SHAPE_VALUE || axis_c_val == UNKNOWN_SHAPE_VALUE) {
    first_element_of_fz_lstm = UNKNOWN_SHAPE_VALUE;
    second_element_of_fz_lstm = UNKNOWN_SHAPE_VALUE;
  }
  shape.SetDimNum(0);
  shape.AppendDim(first_element_of_fz_lstm);
  shape.AppendDim(second_element_of_fz_lstm);
  shape.AppendDim(NI);
  shape.AppendDim(NI);
  return true;
}

bool TransferShapeUtils::GetFractalZC04Shape(const FormatIndex &format_index, const int64_t &c0,
                                             const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  int64_t axis_h_val = origin_shape.GetDim(format_index[DIM_INDEX_H]);
  int64_t axis_w_val = origin_shape.GetDim(format_index[DIM_INDEX_W]);
  shape.SetDimNum(0);
  if (axis_h_val == UNKNOWN_SHAPE_VALUE || axis_w_val == UNKNOWN_SHAPE_VALUE) {
    shape.AppendDim(UNKNOWN_SHAPE_VALUE);
  } else {
    int64_t x = SHAPE_NUMBER_4;
    MUL_OVERFLOW(x, origin_shape.GetDim(format_index[DIM_INDEX_H]), x);
    MUL_OVERFLOW(x, origin_shape.GetDim(format_index[DIM_INDEX_W]), x);
    shape.AppendDim(DivisionCeiling(x, c0));
  }
  shape.AppendDim(DivisionCeiling(origin_shape.GetDim(format_index[DIM_INDEX_N]), NI));
  shape.AppendDim(NI);
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetFractalZnRnnShape(const ExtAxisValue &ext_axis, const int64_t &c0,
                                              const gert::Shape &origin_shape, gert::Shape &shape) {
  size_t origin_shape_size = origin_shape.GetDimNum();
  CHECK(origin_shape_size < DIM_SIZE_TWO, GELOGD("Dim size is less than 2."), return true);
  shape.SetDimNum(0);
  for (size_t i = 0; i < origin_shape_size - DIM_SIZE_TWO; i++) {
    shape.AppendDim(origin_shape.GetDim(i));
  }
  /* Check ND shape value. */
  int64_t k_value = origin_shape.GetDim(origin_shape_size - MINUS_VALUE_TWO);
  int64_t hidden_or_state_size = ext_axis[EXT_INDEX_HIDDEN_SIZE];
  if (ext_axis[EXT_INDEX_STATE_SIZE] != RNN_STATE_SIZE_DEFAULT_VALUE) {
    hidden_or_state_size = ext_axis[EXT_INDEX_STATE_SIZE];
  }
  if (k_value == hidden_or_state_size + ext_axis[EXT_INDEX_INPUT_SIZE]) {
    // use input size and hidden size
    shape.AppendDim(DivisionCeiling(ext_axis[EXT_INDEX_INPUT_SIZE], SHAPE_NUMBER_16) +
                    DivisionCeiling(hidden_or_state_size, SHAPE_NUMBER_16));
  } else if (k_value == hidden_or_state_size || k_value == ext_axis[EXT_INDEX_INPUT_SIZE]) {
    // only use hidden size or input size
    shape.AppendDim(DivisionCeiling(k_value, SHAPE_NUMBER_16));
  } else {
    return true;
  }

  int64_t n_value = origin_shape.GetDim(origin_shape_size - MINUS_VALUE_ONE);
  INT64_ZEROCHECK(ext_axis[EXT_INDEX_HIDDEN_SIZE]);
  int64_t n_num = n_value / ext_axis[EXT_INDEX_HIDDEN_SIZE];
  MUL_OVERFLOW(n_num, DivisionCeiling(ext_axis[EXT_INDEX_HIDDEN_SIZE], c0), n_num);
  shape.AppendDim(n_num);
  shape.AppendDim(SHAPE_NUMBER_16);
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetNdRnnBiasShape(const ExtAxisValue &ext_axis, const int64_t &c0,
                                           const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(ext_axis[EXT_INDEX_HIDDEN_SIZE] == 0, GELOGD("hidden_size is zero"), return true);
  size_t size_of_original_vec = origin_shape.GetDimNum();
  shape.SetDimNum(0);
  for (size_t i = 0; i < size_of_original_vec - MINUS_VALUE_ONE; i++) {
    shape.AppendDim(origin_shape.GetDim(i));
  }
  /* Check ND shape value. */
  int64_t n_num = origin_shape.GetDim(size_of_original_vec - MINUS_VALUE_ONE) / ext_axis[EXT_INDEX_HIDDEN_SIZE];
  MUL_OVERFLOW(n_num, DivisionCeiling(ext_axis[EXT_INDEX_HIDDEN_SIZE], c0), n_num);
  MUL_OVERFLOW(n_num, c0, n_num);
  shape.AppendDim(n_num);
  return true;
}

bool TransferShapeUtils::GetNYUVShape(gert::Shape &shape) {
  const size_t kSize4 = 4U;
  const size_t kSize3 = 3U;
  size_t shape_size = shape.GetDimNum();
  CHECK(((shape_size != kSize3) && (shape_size != kSize4)), GELOGD("Dim size is not 3 or 4."), return false);
  const size_t kWdimOffset = 2U;
  const size_t kHdimOffset = 3U;
  const int64_t kAlaign64 = 64;
  const int64_t kAlaign16 = 16;
  auto width = shape.GetDim(shape_size - kWdimOffset);
  auto height = shape.GetDim(shape_size - kHdimOffset);
  width = (width + kAlaign64 - 1) / kAlaign64 * kAlaign64;
  height = (height + kAlaign16 - 1) / kAlaign16 * kAlaign16;
  shape.SetDim(shape_size - kWdimOffset, width);
  shape.SetDim(shape_size - kHdimOffset, height);
  return true;
}

TransferShapeType TransferShapeUtils::GetTransferShapeType(const ge::Format &src_format, const ge::Format &dst_format,
                                                           const gert::Shape &src_shape) {
  auto primary_src_format = static_cast<ge::Format>(ge::GetPrimaryFormat(src_format));
  auto primary_dst_format = static_cast<ge::Format>(ge::GetPrimaryFormat(dst_format));
  if (primary_src_format == primary_dst_format) {
    return TransferShapeType::ND_TO_ND;
  } else {
    if (primary_src_format == ge::FORMAT_ND || kFormatNZSet.count(primary_dst_format) > 0) {
      return TransferShapeType::ND_TO_NZ;
    }
    size_t src_format_full_size = 0;
    if (!ExpandDimension::GetFormatFullSize(primary_src_format, src_format_full_size)) {
      return TransferShapeType::INVALID;
    }
    if (src_shape.GetDimNum() == src_format_full_size) {
      return TransferShapeType::FULL_SIZE;
    }
    if (src_shape.GetDimNum() < src_format_full_size) {
      return TransferShapeType::NOT_FULL_SIZE;
    }
  }
  return TransferShapeType::INVALID;
}

bool TransferShapeUtils::GetAlignedShape(const AlignShapeInfo &align_shape_info, gert::Shape &aligned_shape) {
  auto func = get_aligned_shape_func_map.find(
      GetTransferShapeType(align_shape_info.src_format, align_shape_info.dst_format, align_shape_info.src_shape));
  if (func == get_aligned_shape_func_map.end()) {
    GELOGW("Do not support src_format %s to src_shape %s.",
           ge::TypeUtils::FormatToSerialString(align_shape_info.src_format).c_str(),
           ge::TypeUtils::FormatToSerialString(align_shape_info.dst_format).c_str());
    return false;
  }
  return func->second(align_shape_info, aligned_shape);
}

bool TransferShapeUtils::TransferDims(const TransferDimsInfo &transfer_dims_info,
                                      AxisIndexMapping &axis_index_mapping) {
  auto func = transfer_dims_func_map.find(
      GetTransferShapeType(transfer_dims_info.src_format, transfer_dims_info.dst_format, transfer_dims_info.src_shape));
  if (func == transfer_dims_func_map.end()) {
    GELOGW("Do not support src_format %s to src_shape %s.",
           ge::TypeUtils::FormatToSerialString(transfer_dims_info.src_format).c_str(),
           ge::TypeUtils::FormatToSerialString(transfer_dims_info.dst_format).c_str());
    return false;
  }
  return func->second(transfer_dims_info, axis_index_mapping);
}

/**
 * ND-to-ND
 * The aligned value for each axis is set to 1.
 **/
bool TransferShapeUtils::GetNdToNdAlignedShape(const AlignShapeInfo &align_shape_info, gert::Shape &aligned_shape) {
  GELOGD("There are no alignment requirements for the ND-to-ND scenario.");
  (void)aligned_shape.SetDimNum(align_shape_info.src_shape.GetDimNum());
  for (size_t i = 0; i < align_shape_info.src_shape.GetDimNum(); ++i) {
    (void)aligned_shape.SetDim(i, 1);
  }
  return true;
}

/**
 * ND-to-NZ
 * eg: ND(n, a, b) -> NZ(n, b//c0, a//m0, m0, c0)
 *           |
 *        target
 * The aligned value for the axis before target is set to 1.
 * The aligned value for the target axis is set to m0.
 * The aligned value for the axis behind target is set to c0.
 **/
bool TransferShapeUtils::GetNdToNzAlignedShape(const AlignShapeInfo &align_shape_info, gert::Shape &aligned_shape) {
  size_t src_shape_dim_num = align_shape_info.src_shape.GetDimNum();
  if (src_shape_dim_num < kNzMinDimNum) {
    GELOGW("The src_shape_dim_num %zu is invalid for the ND-to-NZ scenario.", src_shape_dim_num);
    return false;
  }
  ge::Format format = static_cast<ge::Format>(ge::GetPrimaryFormat(align_shape_info.dst_format));
  if (kFormatNZSet.count(format) == 0) {
    GELOGW("Unsupported dst_format %s for ND-to-NZ scenario.",
           ge::TypeUtils::FormatToSerialString(align_shape_info.dst_format).c_str());
    return false;
  }
  (void)aligned_shape.SetDimNum(src_shape_dim_num);
  size_t target_index = src_shape_dim_num - 2;
  for (size_t i = 0; i < target_index; ++i) {
    (void)aligned_shape.SetDim(i, 1);
  }
  int64_t m0 = GetM0ByDtype(align_shape_info.data_type);
  int64_t c0 = GetC0ByDtype(align_shape_info.data_type);
  (void)aligned_shape.SetDim(target_index, m0);
  (void)aligned_shape.SetDim(target_index + 1, c0);
  return true;
}

/**
 * full-size: no need to expand dims
 * NCHW/HWCN/NHWC/CHWN(a,b,c,d) -> FZ(c//c0*h*w, n//n0, n0, c0)
 * NCHW/HWCN/NHWC/CHWN(a,b,c,d) -> NCHW/HWCN/NHWC/CHWN/NC1HWC0
 * NDHWC/DHWCN/DHWNC(a,b,c,d,e) -> NDHWC/DHWCN/DHWNC/NDC1HWC0
 * Traverse each axis, if current axis is "C" and needs to be split, the aligned value is set to c0.
 * Otherwise, set it to 1.
 **/
bool TransferShapeUtils::GetFullSizeAlignedShape(const AlignShapeInfo &align_shape_info, gert::Shape &aligned_shape) {
  std::vector<std::string> src_axis_vec = AxisUtil::GetAxisVecByFormat(align_shape_info.src_format);
  if (src_axis_vec.empty()) {
    GELOGW("Does not support src_format %s for dst_format in full-size scene.",
           ge::TypeUtils::FormatToSerialString(align_shape_info.src_format).c_str(),
           ge::TypeUtils::FormatToSerialString(align_shape_info.dst_format).c_str());
    return false;
  }
  int64_t n0 = GetN0ByDtype(align_shape_info.data_type);
  int64_t c0 = GetC0ByDtype(align_shape_info.data_type);
  (void)aligned_shape.SetDimNum(src_axis_vec.size());
  for (size_t i = 0; i < src_axis_vec.size(); ++i) {
    if (AxisUtil::GetSplitOrConcatAxisByFormat(align_shape_info.dst_format, src_axis_vec.at(i)).size() > 1) {
      if (src_axis_vec.at(i) == "N") {
        (void)aligned_shape.SetDim(i, n0);
      } else if (src_axis_vec.at(i) == "C") {
        (void)aligned_shape.SetDim(i, c0);
      } else {
        (void)aligned_shape.SetDim(i, 1);
      }
    } else {
      (void)aligned_shape.SetDim(i, 1);
    }
  }
  return true;
}

/**
 * not-full-size: need to expand dims
 * NCHW/HWCN/NHWC/CHWN(a,b) -> FZ(c//c0*h*w, n//n0, n0, c0)
 * NCHW/HWCN/NHWC/CHWN(a,b) -> NC1HWC0
 * NDHWC/DHWCN/DHWNC(a,b,c) -> NDC1HWC0
 * Traverse each reshape type axis, if current axis is "C", the aligned value is set to c0.
 * Otherwise, set it to 1.
 **/
bool TransferShapeUtils::GetNotFullSizeAlignedShape(const AlignShapeInfo &align_shape_info,
                                                    gert::Shape &aligned_shape) {
  if (align_shape_info.reshape_type_mask <= 0) {
    GELOGW("The reshape type mask %lld is invalid in a non-full-size scenario.", align_shape_info.reshape_type_mask);
    return false;
  }
  std::vector<std::string> reshape_type_axis_vec =
      AxisUtil::GetReshapeTypeAxisVec(align_shape_info.src_format, align_shape_info.reshape_type_mask);
  if (reshape_type_axis_vec.empty()) {
    GELOGW("Does not support conversion from src_format %s to dst_format %s for non-full-size scenarios.",
           ge::TypeUtils::FormatToSerialString(align_shape_info.src_format).c_str(),
           ge::TypeUtils::FormatToSerialString(align_shape_info.dst_format).c_str());
    return false;
  }
  int64_t n0 = GetN0ByDtype(align_shape_info.data_type);
  int64_t c0 = GetC0ByDtype(align_shape_info.data_type);
  (void)aligned_shape.SetDimNum(reshape_type_axis_vec.size());
  for (size_t i = 0; i < reshape_type_axis_vec.size(); ++i) {
    if (AxisUtil::GetSplitOrConcatAxisByFormat(align_shape_info.dst_format, reshape_type_axis_vec.at(i)).size() > 1) {
      if (reshape_type_axis_vec.at(i) == "N") {
        (void)aligned_shape.SetDim(i, n0);
      } else if (reshape_type_axis_vec.at(i) == "C") {
        (void)aligned_shape.SetDim(i, c0);
      } else {
        (void)aligned_shape.SetDim(i, 1);
      }
    } else {
      (void)aligned_shape.SetDim(i, 1);
    }
  }
  return true;
}

/**
 * ND-to-ND
 * No change.
 **/
bool TransferShapeUtils::GetNdToNdAxisIndexMapping(const TransferDimsInfo &transfer_dims_info,
                                                   AxisIndexMapping &axis_index_mapping) {
  for (int32_t i = 0; i < static_cast<int32_t>(transfer_dims_info.src_shape.GetDimNum()); ++i) {
    std::vector<int32_t> cur_transfer_dims;
    cur_transfer_dims.emplace_back(i);
    axis_index_mapping.src_to_dst_transfer_dims.emplace_back(cur_transfer_dims);
  }
  axis_index_mapping.dst_to_src_transfer_dims = axis_index_mapping.src_to_dst_transfer_dims;
  return true;
}

/**
 * ND-to-NZ
 * eg: ND(n, a, b) -> NZ(n, b//c0, a//m0, m0, c0)
 *           |
 *        target
 * The aligned value for the axis before target is set to 1.
 * The aligned value for the target axis is set to m0.
 * The aligned value for the axis behind target is set to c0.
 **/
bool TransferShapeUtils::GetNdToNzAxisIndexMapping(const TransferDimsInfo &transfer_dims_info,
                                                   AxisIndexMapping &axis_index_mapping) {
  size_t src_shape_dim_num = transfer_dims_info.src_shape.GetDimNum();
  if (src_shape_dim_num < kNzMinDimNum) {
    GELOGW("The src_shape_dim_num %zu is invalid for the ND-to-NZ scenario.", src_shape_dim_num);
    return false;
  }
  ge::Format format = static_cast<ge::Format>(GetPrimaryFormat(transfer_dims_info.dst_format));
  if (kFormatNZSet.count(format) == 0) {
    GELOGW("Unsupported dst_format %s for ND-to-NZ scenario.",
           ge::TypeUtils::FormatToSerialString(transfer_dims_info.dst_format).c_str());
    return false;
  }
  int32_t target_index = static_cast<int32_t>(src_shape_dim_num - 2);
  for (int32_t i = 0; i < target_index; ++i) {
    std::vector<int32_t> cur_transfer_dims;
    cur_transfer_dims.emplace_back(i);
    axis_index_mapping.src_to_dst_transfer_dims.emplace_back(cur_transfer_dims);
  }
  std::vector<int32_t> cur_transfer_dims;
  cur_transfer_dims.emplace_back(target_index + 1);
  axis_index_mapping.src_to_dst_transfer_dims.emplace_back(cur_transfer_dims);
  cur_transfer_dims.clear();
  cur_transfer_dims.emplace_back(target_index);
  axis_index_mapping.src_to_dst_transfer_dims.emplace_back(cur_transfer_dims);

  axis_index_mapping.dst_to_src_transfer_dims.insert(axis_index_mapping.dst_to_src_transfer_dims.end(),
                                                     axis_index_mapping.src_to_dst_transfer_dims.begin(),
                                                     axis_index_mapping.src_to_dst_transfer_dims.end());
  cur_transfer_dims.clear();
  cur_transfer_dims.emplace_back(-1);
  axis_index_mapping.dst_to_src_transfer_dims.emplace_back(cur_transfer_dims);
  axis_index_mapping.dst_to_src_transfer_dims.emplace_back(cur_transfer_dims);
  return true;
}

/**
 * full-size: no need to expand dims
 * NCHW/HWCN/NHWC/CHWN(a,b,c,d) -> FZ(c//c0*h*w, n//n0, n0, c0)
 * NCHW/HWCN/NHWC/CHWN(a,b,c,d) -> NCHW/HWCN/NHWC/CHWN/NC1HWC0
 * NDHWC/DHWCN/DHWNC(a,b,c,d,e) -> NDHWC/DHWCN/DHWNC/NDC1HWC0
 * Traverse each axis in src_format, and the get its mapping-index in dst_format.
 * If current axis is "C" and needs to be split, get "C0" and "C1" mapping-index in dst_format.
 * Traverse each axis in dst_format, and the get its mapping-index in src_format.
 * If current axis is "C0" or "C1", get "C" mapping-index in src_format.
 **/
bool TransferShapeUtils::GetFullSizeAxisIndexMapping(const TransferDimsInfo &transfer_dims_info,
                                                     AxisIndexMapping &axis_index_mapping) {
  std::vector<std::string> src_axis_vec = AxisUtil::GetAxisVecByFormat(transfer_dims_info.src_format);
  if (src_axis_vec.empty()) {
    GELOGW("Does not support conversion from src_format %s to dst_format %s for full-size scenes.",
           ge::TypeUtils::FormatToSerialString(transfer_dims_info.src_format).c_str(),
           ge::TypeUtils::FormatToSerialString(transfer_dims_info.dst_format).c_str());
    return false;
  }
  for (auto &cur_axis : src_axis_vec) {
    std::vector<int32_t> cur_transfer_dims;
    std::vector<std::string> split_or_concat_axis =
        AxisUtil::GetSplitOrConcatAxisByFormat(transfer_dims_info.dst_format, cur_axis);
    if (!split_or_concat_axis.empty()) {
      for (auto &cur_split_or_concat_axis : split_or_concat_axis) {
        cur_transfer_dims.emplace_back(
            AxisUtil::GetAxisIndexByFormat(transfer_dims_info.dst_format, cur_split_or_concat_axis));
      }
    } else {
      cur_transfer_dims.emplace_back(AxisUtil::GetAxisIndexByFormat(transfer_dims_info.dst_format, cur_axis));
    }
    axis_index_mapping.src_to_dst_transfer_dims.emplace_back(cur_transfer_dims);
  }

  std::vector<std::string> dst_axis_vec = AxisUtil::GetAxisVecByFormat(transfer_dims_info.dst_format);
  if (dst_axis_vec.empty()) {
    GELOGW("Does not support conversion from src_format %s to dst_format %s for full-size scenes.",
           ge::TypeUtils::FormatToSerialString(transfer_dims_info.src_format).c_str(),
           ge::TypeUtils::FormatToSerialString(transfer_dims_info.dst_format).c_str());
    return false;
  }
  for (auto &cur_axis : dst_axis_vec) {
    std::vector<int32_t> cur_transfer_dims;
    std::vector<std::string> split_or_concat_axis =
        AxisUtil::GetSplitOrConcatAxisByFormat(transfer_dims_info.dst_format, cur_axis);
    if (!split_or_concat_axis.empty()) {
      for (auto cur_split_or_concat_axis : split_or_concat_axis) {
        cur_transfer_dims.emplace_back(
            AxisUtil::GetAxisIndexByFormat(transfer_dims_info.src_format, cur_split_or_concat_axis));
      }
    } else {
      cur_transfer_dims.emplace_back(AxisUtil::GetAxisIndexByFormat(transfer_dims_info.src_format, cur_axis));
    }
    axis_index_mapping.dst_to_src_transfer_dims.emplace_back(cur_transfer_dims);
  }
  return true;
}

/**
 * not-full-size: need to expand dims
 * NCHW/HWCN/NHWC/CHWN(a,b) -> NC1HWC0
 * NDHWC/DHWCN/DHWNC(a,b,c) -> NDC1HWC0
 * Traverse each axis in reshape_type, and the get its mapping-index in dst_format.
 * If current axis is "C" and needs to be split, get "C0" and "C1" mapping-index in dst_format.
 * Traverse each axis in dst_format, and the get its mapping-index in src_format.
 * If current axis is "C0" or "C1", get "C" mapping-index in src_format.
 **/
bool TransferShapeUtils::GetNotFullSizeAxisIndexMapping(const TransferDimsInfo &transfer_dims_info,
                                                        AxisIndexMapping &axis_index_mapping) {
  if (transfer_dims_info.reshape_type_mask <= 0) {
    GELOGW("Does not support conversion from src_format %s to dst_format %s for non-full-size scenarios.",
           ge::TypeUtils::FormatToSerialString(transfer_dims_info.src_format).c_str(),
           ge::TypeUtils::FormatToSerialString(transfer_dims_info.dst_format).c_str());
    return false;
  }
  std::vector<std::string> reshape_type_axis_vec =
      AxisUtil::GetReshapeTypeAxisVec(transfer_dims_info.src_format, transfer_dims_info.reshape_type_mask);
  if (reshape_type_axis_vec.empty()) {
    GELOGW("Does not support conversion from src_format %s to dst_format %s for non-full-size scenarios.",
           ge::TypeUtils::FormatToSerialString(transfer_dims_info.src_format).c_str(),
           ge::TypeUtils::FormatToSerialString(transfer_dims_info.dst_format).c_str());
    return false;
  }
  for (auto &cur_axis : reshape_type_axis_vec) {
    std::vector<int32_t> cur_transfer_dims;
    std::vector<std::string> split_or_concat_axis =
        AxisUtil::GetSplitOrConcatAxisByFormat(transfer_dims_info.dst_format, cur_axis);
    if (!split_or_concat_axis.empty()) {
      for (auto &cur_split_or_concat_axis : split_or_concat_axis) {
        cur_transfer_dims.emplace_back(
            AxisUtil::GetAxisIndexByFormat(transfer_dims_info.dst_format, cur_split_or_concat_axis));
      }
    } else {
      cur_transfer_dims.emplace_back(AxisUtil::GetAxisIndexByFormat(transfer_dims_info.dst_format, cur_axis));
    }
    axis_index_mapping.src_to_dst_transfer_dims.emplace_back(cur_transfer_dims);
  }

  std::vector<std::string> dst_axis_vec = AxisUtil::GetAxisVecByFormat(transfer_dims_info.dst_format);
  if (dst_axis_vec.empty()) {
    return false;
  }
  std::map<std::string, int32_t> valid_axis_map =
      AxisUtil::GetReshapeTypeAxisMap(transfer_dims_info.src_format, transfer_dims_info.reshape_type_mask);
  for (auto &cur_axis : dst_axis_vec) {
    std::vector<int32_t> cur_transfer_dims;
    std::vector<std::string> split_or_concat_axis =
        AxisUtil::GetSplitOrConcatAxisByFormat(transfer_dims_info.dst_format, cur_axis);
    if (!split_or_concat_axis.empty()) {
      for (auto &cur_split_or_concat_axis : split_or_concat_axis) {
        cur_transfer_dims.emplace_back(
            AxisUtil::GetAxisIndexByFormat(transfer_dims_info.src_format, cur_split_or_concat_axis, valid_axis_map));
      }
    } else {
      cur_transfer_dims.emplace_back(
          AxisUtil::GetAxisIndexByFormat(transfer_dims_info.src_format, cur_axis, valid_axis_map));
    }
    axis_index_mapping.dst_to_src_transfer_dims.emplace_back(cur_transfer_dims);
  }
  return true;
}
}  // namespace transformer
