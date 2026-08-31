/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/types_af.h"
#include <cinttypes>
#include <cmath>
#include <utility>
#include "framework/common/debug/ge_log.h"
#include "graph/ge_error_codes_af.h"
#include "graph/utils/type_utils.h"

namespace ge {

const char_t *GetFormatName(Format format) {
  static const char_t *names[FORMAT_END] = {
      "NCHW",
      "NHWC",
      "ND",
      "NC1HWC0",
      "FRACTAL_Z",
      "NC1C0HWPAD",  // 5
      "NHWC1C0",
      "FSR_NCHW",
      "FRACTAL_DECONV",
      "C1HWNC0",
      "FRACTAL_DECONV_TRANSPOSE",  // 10
      "FRACTAL_DECONV_SP_STRIDE_TRANS",
      "NC1HWC0_C04",
      "FRACTAL_Z_C04",
      "CHWN",
      "DECONV_SP_STRIDE8_TRANS",  // 15
      "HWCN",
      "NC1KHKWHWC0",
      "BN_WEIGHT",
      "FILTER_HWCK",
      "LOOKUP_LOOKUPS",  // 20
      "LOOKUP_KEYS",
      "LOOKUP_VALUE",
      "LOOKUP_OUTPUT",
      "LOOKUP_HITS",
      "C1HWNCoC0",  // 25
      "MD",
      "NDHWC",
      "UNKNOWN",  // FORMAT_FRACTAL_ZZ
      "FRACTAL_NZ",
      "NCDHW",  // 30
      "DHWCN",
      "NDC1HWC0",
      "FRACTAL_Z_3D",
      "CN",
      "NC",  // 35
      "DHWNC",
      "FRACTAL_Z_3D_TRANSPOSE",
      "FRACTAL_ZN_LSTM",
      "FRACTAL_Z_G",
      "UNKNOWN",  // 40, FORMAT_RESERVED
      "UNKNOWN",  // FORMAT_ALL
      "UNKNOWN",  // FORMAT_NULL
      "ND_RNN_BIAS",
      "FRACTAL_ZN_RNN",
      "NYUV",  // 45
      "NYUV_A",
      "NCL",
      "FRACTAL_Z_WINO",
      "C1HWC0",
      "FRACTAL_NZ_C0_16",
      "FRACTAL_NZ_C0_32",
      "FRACTAL_NZ_C0_2",
      "FRACTAL_NZ_C0_4",
      "FRACTAL_NZ_C0_8",
  };
  if (format >= FORMAT_END) {
    return "UNKNOWN";
  }
  return names[format];
}

static int64_t CeilDiv(const int64_t n1, const int64_t n2) {
  if (n1 == 0) {
    return 0;
  }
  return (n2 != 0) ? (((n1 - 1) / n2) + 1) : 0;
}

static af::Status CheckInt64MulOverflow(const int64_t a, const int64_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT64_MAX / b)) {
        return af::FAILED;
      }
    } else {
      if (b < (INT64_MIN / a)) {
        return af::FAILED;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT64_MIN / b)) {
        return af::FAILED;
      }
    } else {
      if ((a != 0) && (b < (INT64_MAX / a))) {
        return af::FAILED;
      }
    }
  }
  return af::SUCCESS;
}

int64_t GetSizeInBytes(int64_t element_count, DataType data_type) {
  if (element_count < 0) {
    GELOGW("[Check][param]GetSizeInBytes failed, element_count:%" PRId64 " less than 0.", element_count);
    return -1;
  }
  uint32_t type_size = 0U;
  if (!TypeUtils::GetDataTypeLength(data_type, type_size)) {
    GELOGW("[Check][DataType]GetSizeInBytes failed, data_type:%d not support.", data_type);
    return -1;
  } else if (type_size > kDataTypeSizeBitOffset) {
    const auto bit_size = type_size - kDataTypeSizeBitOffset;
    if (CheckInt64MulOverflow(element_count, static_cast<int64_t>(bit_size)) == af::FAILED) {
      GELOGW("[Check][overflow]GetSizeInBytes failed, when multiplying %" PRId64 " and %d.", element_count, bit_size);
      return -1;
    }
    return CeilDiv(element_count * bit_size, kBitNumOfOneByte);
  } else {
    if (CheckInt64MulOverflow(element_count, static_cast<int64_t>(type_size)) == af::FAILED) {
      GELOGW("[Check][overflow]GetSizeInBytes failed, when multiplying %" PRId64 " and %" PRId32 ".", element_count,
             type_size);
      return -1;
    }
    return element_count * type_size;
  }
}

std::vector<const char *> Promote::Syms() const {
  std::vector<const char *> result;
  if (data_ == nullptr) {
    return result;
  }
  auto &syms = *static_cast<std::vector<std::string> *>(data_.get());
  result.reserve(syms.size());
  for (const auto &sym : syms) {
    result.push_back(sym.c_str());
  }
  return result;
}

Promote::Promote(const std::initializer_list<const char *> &syms) {
  data_ = std::make_shared<std::vector<std::string>>();
  auto *vec = static_cast<std::vector<std::string> *>(data_.get());
  for (const auto &sym : syms) {
    vec->emplace_back((sym == nullptr) ? "" : sym);
  }
}

Promote::Promote(Promote &&other) noexcept {
  data_ = std::move(other.data_);
}

Promote &Promote::operator=(Promote &&other) noexcept {
  if (this != &other) {
    data_ = std::move(other.data_);
  }
  return *this;
}
}  // namespace ge
