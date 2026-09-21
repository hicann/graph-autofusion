/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base/utils/type_utils_impl_af.h"
#include "graph/utils/type_utils.h"
#include <algorithm>
#include <map>

#include "graph/debug/ge_util.h"
#include "graph/types_af.h"

namespace af {
namespace {
const std::map<Format, std::string> kFormatToStringMap = {
    {FORMAT_NCHW, "NCHW"},
    {FORMAT_NHWC, "NHWC"},
    {FORMAT_ND, "ND"},
    {FORMAT_NC1HWC0, "NC1HWC0"},
    {FORMAT_FRACTAL_Z, "FRACTAL_Z"},
    {FORMAT_NC1C0HWPAD, "NC1C0HWPAD"},
    {FORMAT_NHWC1C0, "NHWC1C0"},
    {FORMAT_FSR_NCHW, "FSR_NCHW"},
    {FORMAT_FRACTAL_DECONV, "FRACTAL_DECONV"},
    {FORMAT_C1HWNC0, "C1HWNC0"},
    {FORMAT_FRACTAL_DECONV_TRANSPOSE, "FRACTAL_DECONV_TRANSPOSE"},
    {FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS, "FRACTAL_DECONV_SP_STRIDE_TRANS"},
    {FORMAT_NC1HWC0_C04, "NC1HWC0_C04"},
    {FORMAT_FRACTAL_Z_C04, "FRACTAL_Z_C04"},
    {FORMAT_CHWN, "CHWN"},
    {FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS, "DECONV_SP_STRIDE8_TRANS"},
    {FORMAT_NC1KHKWHWC0, "NC1KHKWHWC0"},
    {FORMAT_BN_WEIGHT, "BN_WEIGHT"},
    {FORMAT_FILTER_HWCK, "FILTER_HWCK"},
    {FORMAT_HWCN, "HWCN"},
    {FORMAT_HASHTABLE_LOOKUP_LOOKUPS, "LOOKUP_LOOKUPS"},
    {FORMAT_HASHTABLE_LOOKUP_KEYS, "LOOKUP_KEYS"},
    {FORMAT_HASHTABLE_LOOKUP_VALUE, "LOOKUP_VALUE"},
    {FORMAT_HASHTABLE_LOOKUP_OUTPUT, "LOOKUP_OUTPUT"},
    {FORMAT_HASHTABLE_LOOKUP_HITS, "LOOKUP_HITS"},
    {FORMAT_MD, "MD"},
    {FORMAT_NDHWC, "NDHWC"},
    {FORMAT_NCDHW, "NCDHW"},
    {FORMAT_DHWCN, "DHWCN"},
    {FORMAT_DHWNC, "DHWNC"},
    {FORMAT_NDC1HWC0, "NDC1HWC0"},
    {FORMAT_FRACTAL_Z_3D, "FRACTAL_Z_3D"},
    {FORMAT_FRACTAL_Z_3D_TRANSPOSE, "FRACTAL_Z_3D_TRANSPOSE"},
    {FORMAT_C1HWNCoC0, "C1HWNCoC0"},
    {FORMAT_FRACTAL_NZ, "FRACTAL_NZ"},
    {FORMAT_FRACTAL_NZ_C0_16, "FRACTAL_NZ_C0_16"},
    {FORMAT_FRACTAL_NZ_C0_32, "FRACTAL_NZ_C0_32"},
    {FORMAT_FRACTAL_NZ_C0_2, "FRACTAL_NZ_C0_2"},
    {FORMAT_FRACTAL_NZ_C0_4, "FRACTAL_NZ_C0_4"},
    {FORMAT_FRACTAL_NZ_C0_8, "FRACTAL_NZ_C0_8"},
    {FORMAT_CN, "CN"},
    {FORMAT_NC, "NC"},
    {FORMAT_FRACTAL_ZN_LSTM, "FRACTAL_ZN_LSTM"},
    {FORMAT_FRACTAL_Z_G, "FRACTAL_Z_G"},
    {FORMAT_ND_RNN_BIAS, "ND_RNN_BIAS"},
    {FORMAT_FRACTAL_ZN_RNN, "FRACTAL_ZN_RNN"},
    {FORMAT_NYUV, "NYUV"},
    {FORMAT_NYUV_A, "NYUV_A"},
    {FORMAT_NCL, "NCL"},
    {FORMAT_FRACTAL_Z_WINO, "FRACTAL_Z_WINO"},
    {FORMAT_C1HWC0, "C1HWC0"},
    {FORMAT_RESERVED, "FORMAT_RESERVED"},
    {FORMAT_ALL, "ALL"},
    {FORMAT_NULL, "NULL"},
    {FORMAT_END, "END"},
    {FORMAT_MAX, "MAX"}};

const std::map<std::string, Format> kDataFormatMap = {
    {"NCHW", FORMAT_NCHW}, {"NHWC", FORMAT_NHWC}, {"NDHWC", FORMAT_NDHWC}, {"NCDHW", FORMAT_NCDHW}, {"ND", FORMAT_ND}};

const std::map<std::string, Format> kStringToFormatMap = {
    {"NCHW", FORMAT_NCHW},
    {"NHWC", FORMAT_NHWC},
    {"ND", FORMAT_ND},
    {"NC1HWC0", FORMAT_NC1HWC0},
    {"FRACTAL_Z", FORMAT_FRACTAL_Z},
    {"NC1C0HWPAD", FORMAT_NC1C0HWPAD},
    {"NHWC1C0", FORMAT_NHWC1C0},
    {"FSR_NCHW", FORMAT_FSR_NCHW},
    {"FRACTAL_DECONV", FORMAT_FRACTAL_DECONV},
    {"C1HWNC0", FORMAT_C1HWNC0},
    {"FRACTAL_DECONV_TRANSPOSE", FORMAT_FRACTAL_DECONV_TRANSPOSE},
    {"FRACTAL_DECONV_SP_STRIDE_TRANS", FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS},
    {"NC1HWC0_C04", FORMAT_NC1HWC0_C04},
    {"FRACTAL_Z_C04", FORMAT_FRACTAL_Z_C04},
    {"CHWN", FORMAT_CHWN},
    {"DECONV_SP_STRIDE8_TRANS", FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS},
    {"NC1KHKWHWC0", FORMAT_NC1KHKWHWC0},
    {"BN_WEIGHT", FORMAT_BN_WEIGHT},
    {"FILTER_HWCK", FORMAT_FILTER_HWCK},
    {"HWCN", FORMAT_HWCN},
    {"LOOKUP_LOOKUPS", FORMAT_HASHTABLE_LOOKUP_LOOKUPS},
    {"LOOKUP_KEYS", FORMAT_HASHTABLE_LOOKUP_KEYS},
    {"LOOKUP_VALUE", FORMAT_HASHTABLE_LOOKUP_VALUE},
    {"LOOKUP_OUTPUT", FORMAT_HASHTABLE_LOOKUP_OUTPUT},
    {"LOOKUP_HITS", FORMAT_HASHTABLE_LOOKUP_HITS},
    {"MD", FORMAT_MD},
    {"C1HWNCoC0", FORMAT_C1HWNCoC0},
    {"FRACTAL_NZ", FORMAT_FRACTAL_NZ},
    {"FRACTAL_NZ_C0_16", FORMAT_FRACTAL_NZ_C0_16},
    {"FRACTAL_NZ_C0_32", FORMAT_FRACTAL_NZ_C0_32},
    {"FRACTAL_NZ_C0_2", FORMAT_FRACTAL_NZ_C0_2},
    {"FRACTAL_NZ_C0_4", FORMAT_FRACTAL_NZ_C0_4},
    {"FRACTAL_NZ_C0_8", FORMAT_FRACTAL_NZ_C0_8},
    {"NDHWC", FORMAT_NDHWC},
    {"NCDHW", FORMAT_NCDHW},
    {"DHWCN", FORMAT_DHWCN},
    {"DHWNC", FORMAT_DHWNC},
    {"NDC1HWC0", FORMAT_NDC1HWC0},
    {"FRACTAL_Z_3D", FORMAT_FRACTAL_Z_3D},
    {"FRACTAL_Z_3D_TRANSPOSE", FORMAT_FRACTAL_Z_3D_TRANSPOSE},
    {"CN", FORMAT_CN},
    {"NC", FORMAT_NC},
    {"FRACTAL_ZN_LSTM", FORMAT_FRACTAL_ZN_LSTM},
    {"FRACTAL_Z_G", FORMAT_FRACTAL_Z_G},
    {"FORMAT_RESERVED", FORMAT_RESERVED},
    {"ALL", FORMAT_ALL},
    {"NULL", FORMAT_NULL},
    {"ND_RNN_BIAS", FORMAT_ND_RNN_BIAS},
    {"FRACTAL_ZN_RNN", FORMAT_FRACTAL_ZN_RNN},
    {"NYUV", FORMAT_NYUV},
    {"NYUV_A", FORMAT_NYUV_A},
    {"NCL", FORMAT_NCL},
    {"FRACTAL_Z_WINO", FORMAT_FRACTAL_Z_WINO},
    {"C1HWC0", FORMAT_C1HWC0},
    {"RESERVED", FORMAT_RESERVED},
    {"UNDEFINED", FORMAT_RESERVED}};

const std::map<DataType, std::string> kDataTypeToStringMap = {
    {DT_UNDEFINED, "DT_UNDEFINED"},
    {DT_FLOAT, "DT_FLOAT"},
    {DT_FLOAT16, "DT_FLOAT16"},
    {DT_INT8, "DT_INT8"},
    {DT_INT16, "DT_INT16"},
    {DT_UINT16, "DT_UINT16"},
    {DT_UINT8, "DT_UINT8"},
    {DT_INT32, "DT_INT32"},
    {DT_INT64, "DT_INT64"},
    {DT_UINT32, "DT_UINT32"},
    {DT_UINT64, "DT_UINT64"},
    {DT_BOOL, "DT_BOOL"},
    {DT_DOUBLE, "DT_DOUBLE"},
    {DT_DUAL, "DT_DUAL"},
    {DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},
    {DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"},
    {DT_COMPLEX32, "DT_COMPLEX32"},
    {DT_COMPLEX64, "DT_COMPLEX64"},
    {DT_COMPLEX128, "DT_COMPLEX128"},
    {DT_QINT8, "DT_QINT8"},
    {DT_QINT16, "DT_QINT16"},
    {DT_QINT32, "DT_QINT32"},
    {DT_QUINT8, "DT_QUINT8"},
    {DT_QUINT16, "DT_QUINT16"},
    {DT_RESOURCE, "DT_RESOURCE"},
    {DT_STRING_REF, "DT_STRING_REF"},
    {DT_STRING, "DT_STRING"},
    {DT_VARIANT, "DT_VARIANT"},
    {DT_BF16, "DT_BFLOAT16"},
    {DT_INT4, "DT_INT4"},
    {DT_UINT1, "DT_UINT1"},
    {DT_INT2, "DT_INT2"},
    {DT_UINT2, "DT_UINT2"},
    {DT_HIFLOAT8, "DT_HIFLOAT8"},
    {DT_FLOAT8_E5M2, "DT_FLOAT8_E5M2"},
    {DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},
    {DT_FLOAT8_E8M0, "DT_FLOAT8_E8M0"},
    {DT_FLOAT6_E3M2, "DT_FLOAT6_E3M2"},
    {DT_FLOAT6_E2M3, "DT_FLOAT6_E2M3"},
    {ge::DT_HIFLOAT4, "DT_HIFLOAT4"},
    {DT_FLOAT4_E2M1, "DT_FLOAT4_E2M1"},
    {DT_FLOAT4_E1M2, "DT_FLOAT4_E1M2"},
};

const std::map<std::string, DataType> kStringTodataTypeMap = {
    {"DT_UNDEFINED", DT_UNDEFINED},
    {"DT_FLOAT", DT_FLOAT},
    {"DT_FLOAT16", DT_FLOAT16},
    {"DT_INT8", DT_INT8},
    {"DT_INT16", DT_INT16},
    {"DT_UINT16", DT_UINT16},
    {"DT_UINT8", DT_UINT8},
    {"DT_INT32", DT_INT32},
    {"DT_INT64", DT_INT64},
    {"DT_UINT32", DT_UINT32},
    {"DT_UINT64", DT_UINT64},
    {"DT_BOOL", DT_BOOL},
    {"DT_DOUBLE", DT_DOUBLE},
    {"DT_DUAL", DT_DUAL},
    {"DT_DUAL_SUB_INT8", DT_DUAL_SUB_INT8},
    {"DT_DUAL_SUB_UINT8", DT_DUAL_SUB_UINT8},
    {"DT_COMPLEX32", DT_COMPLEX32},
    {"DT_COMPLEX64", DT_COMPLEX64},
    {"DT_COMPLEX128", DT_COMPLEX128},
    {"DT_QINT8", DT_QINT8},
    {"DT_QINT16", DT_QINT16},
    {"DT_QINT32", DT_QINT32},
    {"DT_QUINT8", DT_QUINT8},
    {"DT_QUINT16", DT_QUINT16},
    {"DT_RESOURCE", DT_RESOURCE},
    {"DT_STRING_REF", DT_STRING_REF},
    {"DT_STRING", DT_STRING},
    {"DT_FLOAT32", DT_FLOAT},
    {"DT_VARIANT", DT_VARIANT},
    {"DT_BFLOAT16", DT_BF16},
    {"DT_INT4", DT_INT4},
    {"DT_UINT1", DT_UINT1},
    {"DT_INT2", DT_INT2},
    {"DT_UINT2", DT_UINT2},
    {"DT_HIFLOAT8", DT_HIFLOAT8},
    {"DT_FLOAT8_E5M2", DT_FLOAT8_E5M2},
    {"DT_FLOAT8_E4M3FN", DT_FLOAT8_E4M3FN},
    {"DT_FLOAT8_E8M0", DT_FLOAT8_E8M0},
    {"DT_FLOAT6_E3M2", DT_FLOAT6_E3M2},
    {"DT_FLOAT6_E2M3", DT_FLOAT6_E2M3},
    {"DT_HIFLOAT4", ge::DT_HIFLOAT4},
    {"DT_FLOAT4_E2M1", DT_FLOAT4_E2M1},
    {"DT_FLOAT4_E1M2", DT_FLOAT4_E1M2},
    {"RESERVED", DT_UNDEFINED},
};

const std::map<ge::DataType, uint32_t> kDataTypeToLength = {
    {DT_STRING_REF, sizeof(uint64_t) * 2U},
    {DT_STRING, sizeof(uint64_t) * 2U},
};
}  // namespace

AscendString TypeUtilsImpl::DataTypeToAscendString(const DataType data_type) {
  const auto it = kDataTypeToStringMap.find(data_type);
  if (it != kDataTypeToStringMap.end()) {
    return it->second.c_str();
  } else {
    GELOGW("DataTypeToSerialString: datatype is not supported %d", data_type);
    return "UNDEFINED";
  }
}

DataType TypeUtilsImpl::AscendStringToDataType(const AscendString &str) {
  const auto it = kStringTodataTypeMap.find(str.GetString());
  if (it != kStringTodataTypeMap.end()) {
    return it->second;
  } else {
    GELOGW("[Check][Param] SerialStringToDataType: datatype is not supported %s", str.GetString());
    return DT_UNDEFINED;
  }
}

AscendString TypeUtilsImpl::FormatToAscendString(const Format format) {
  const auto it = kFormatToStringMap.find(static_cast<Format>(GetPrimaryFormat(format)));
  if (it != kFormatToStringMap.end()) {
    if (HasSubFormat(format)) {
      return std::string(it->second + ":" + std::to_string(GetSubFormat(format))).c_str();
    }
    return it->second.c_str();
  } else {
    GELOGW("[Check][Param] Format is not supported %d", format);
    return "RESERVED";
  }
}

graphStatus SplitFormatFromStr(const std::string &str, std::string &primary_format_str, int32_t &sub_format) {
  const size_t split_pos = str.find_first_of(':');
  if (split_pos != std::string::npos) {
    const std::string sub_format_str = str.substr(split_pos + 1U);
    try {
      primary_format_str = str.substr(0U, split_pos);
      if (std::any_of(sub_format_str.cbegin(), sub_format_str.cend(),
                      [](const char_t c) { return !(isdigit(static_cast<char_t>(c)) != 0); })) {
        REPORT_INNER_ERR_MSG("E18888", "sub_format: %s is not digital.", sub_format_str.c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] sub_format: %s is not digital.", sub_format_str.c_str());
        return GRAPH_FAILED;
      }
      sub_format = std::stoi(sub_format_str);
    } catch (std::invalid_argument &) {
      REPORT_INNER_ERR_MSG("E18888", "sub_format: %s is invalid.", sub_format_str.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] sub_format: %s is invalid.", sub_format_str.c_str());
      return GRAPH_FAILED;
    } catch (std::out_of_range &) {
      REPORT_INNER_ERR_MSG("E18888", "sub_format: %s is out of range.", sub_format_str.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] sub_format: %s is out of range.", sub_format_str.c_str());
      return GRAPH_FAILED;
    } catch (...) {
      REPORT_INNER_ERR_MSG("E18888", "sub_format: %s cannot change to int.", sub_format_str.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] sub_format: %s cannot change to int.", sub_format_str.c_str());
      return GRAPH_FAILED;
    }
    if (sub_format > 0xFFFF) {
      REPORT_INNER_ERR_MSG("E18888", "sub_format: %d is out of range [0, 0xffff].", sub_format);
      GELOGE(GRAPH_FAILED, "[Check][Param] sub_format: %d is out of range [0, 0xffff].", sub_format);
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

ge::AscendString TypeUtils::DataTypeToAscendString(const DataType &data_type) {
  return TypeUtilsImpl::DataTypeToAscendString(data_type);
}

DataType TypeUtils::AscendStringToDataType(const ge::AscendString &str) {
  return TypeUtilsImpl::AscendStringToDataType(str);
}

AscendString TypeUtils::FormatToAscendString(const Format &format) {
  return TypeUtilsImpl::FormatToAscendString(format);
}

Format TypeUtils::AscendStringToFormat(const AscendString &str) {
  return TypeUtilsImpl::AscendStringToFormat(str);
}

Format TypeUtils::DataFormatToFormat(const AscendString &str) {
  return TypeUtilsImpl::DataFormatToFormat(str);
}

Format TypeUtilsImpl::AscendStringToFormat(const AscendString &str) {
  std::string primary_format_str = str.GetString();
  int32_t sub_format = 0;
  if (SplitFormatFromStr(str.GetString(), primary_format_str, sub_format) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Split][Format] from %s failed", str.GetString());
    return FORMAT_RESERVED;
  }
  int32_t primary_format;
  const auto it = kStringToFormatMap.find(primary_format_str);
  if (it != kStringToFormatMap.end()) {
    primary_format = it->second;
  } else {
    GELOGW("[Check][Param] Format is not supported %s", str.GetString());
    return FORMAT_RESERVED;
  }
  return static_cast<Format>(GetFormatFromSub(primary_format, sub_format));
}

Format TypeUtilsImpl::DataFormatToFormat(const AscendString &str) {
  std::string primary_format_str = str.GetString();
  int32_t sub_format = 0;
  if (SplitFormatFromStr(str.GetString(), primary_format_str, sub_format) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Split][Format] from %s failed", str.GetString());
    return FORMAT_RESERVED;
  }
  int32_t primary_format;
  const auto it = kDataFormatMap.find(primary_format_str);
  if (it != kDataFormatMap.end()) {
    primary_format = it->second;
  } else {
    GELOGW("[Check][Param] Format is not supported %s", str.GetString());
    return FORMAT_RESERVED;
  }
  return static_cast<Format>(GetFormatFromSub(primary_format, sub_format));
}

bool TypeUtilsImpl::GetDataTypeLength(const ge::DataType data_type, uint32_t &length) {
  const auto it = kDataTypeToLength.find(data_type);
  if (it != kDataTypeToLength.end()) {
    length = it->second;
    return true;
  }

  const int32_t size = GetSizeByDataType(data_type);
  if (size > 0) {
    length = static_cast<uint32_t>(size);
    return true;
  } else {
    REPORT_INNER_ERR_MSG("E18888", "data_type is not supported [%s]", DataTypeToAscendString(data_type).GetString());
    GELOGE(GRAPH_FAILED, "[Check][Param] data_type is not supported [%s]",
           DataTypeToAscendString(data_type).GetString());
    return false;
  }
}

}  // namespace af
