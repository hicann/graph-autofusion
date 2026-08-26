/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "case_contract.h"

#include <cstdint>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace device_validation {

struct TensorView {
  TensorView() = default;
  TensorView(std::string dtype_in, std::vector<int64_t> shape_in, std::vector<double> values_in,
             std::vector<uint64_t> raw_bits_in)
      : dtype(std::move(dtype_in)),
        shape(std::move(shape_in)),
        values(std::move(values_in)),
        raw_bits(std::move(raw_bits_in)) {}
  TensorView(std::string dtype_in, std::vector<int64_t> shape_in, std::vector<double> values_in,
             std::vector<uint64_t> raw_bits_in, std::vector<int64_t> integer_values_in,
             std::vector<uint64_t> unsigned_values_in)
      : dtype(std::move(dtype_in)),
        shape(std::move(shape_in)),
        values(std::move(values_in)),
        raw_bits(std::move(raw_bits_in)),
        integer_values(std::move(integer_values_in)),
        unsigned_values(std::move(unsigned_values_in)) {}
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<double> values;
  std::vector<uint64_t> raw_bits;
  std::vector<int64_t> integer_values;
  std::vector<uint64_t> unsigned_values;
};

struct Mismatch {
  size_t tensor_index = 0;
  size_t linear_index = 0;
  struct MismatchValue {
    enum class Kind { kFloat, kNaN, kInf, kInt64, kUInt64, kRawBits };
    Kind kind = Kind::kFloat;
    double float_value = 0.0;
    int64_t int_value = 0;
    uint64_t uint_value = 0;
    int sign = 1;
    uint64_t raw_bits = 0;

    static MismatchValue Float(double value) {
      return {Kind::kFloat, value, 0, 0, 1, 0};
    }
    static MismatchValue NaN() {
      return {Kind::kNaN, 0, 0, 0, 1, 0};
    }
    static MismatchValue Inf(int value_sign) {
      return {Kind::kInf, 0, 0, 0, value_sign, 0};
    }
    static MismatchValue Int64(int64_t value) {
      return {Kind::kInt64, 0, value, 0, 1, 0};
    }
    static MismatchValue UInt64(uint64_t value) {
      return {Kind::kUInt64, 0, 0, value, 1, 0};
    }
    static MismatchValue RawBits(uint64_t value) {
      return {Kind::kRawBits, 0, 0, 0, 1, value};
    }
  };
  MismatchValue actual;
  MismatchValue expected;
};

using MismatchValue = Mismatch::MismatchValue;

struct PrecisionReport {
  bool passed = false;
  size_t tensor_count = 0;
  size_t element_count = 0;
  size_t mismatch_count = 0;
  size_t nan_mismatch_count = 0;
  size_t inf_mismatch_count = 0;
  std::optional<Mismatch> first_mismatch;
};

}  // namespace device_validation
