/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_UTILS_H__
#define __ASCENDC_API_UTILS_H__

namespace {

constexpr uint32_t MAX_REPEAT_TIME = 255;

constexpr inline __aicore__ unsigned int Ceiling(float num) {
  int ceil_num = static_cast<int>(num);
  float epsilon = 1e-6;
  if (num >= epsilon) {
    return (num - static_cast<float>(ceil_num) < epsilon) ? ceil_num : (ceil_num + 1);
  } else {
    return (static_cast<float>(ceil_num) - num < epsilon) ? ceil_num : (ceil_num + 1);
  }
}

constexpr inline __aicore__ unsigned int Ceiling(unsigned int num) {
  return num;
}

constexpr inline __aicore__ int64_t Ceiling(int64_t num) {
  return num;
}

constexpr inline __aicore__ unsigned int Ceiling(int num) {
  return num;
}

constexpr inline __aicore__ int64_t Floor(float num) {
  float epsilon = 1e-6;
  if (num >= epsilon) {
    return static_cast<int64_t>(num);
  } else {
    int64_t int_num = static_cast<int64_t>(num);
    return (static_cast<float>(int_num) - num < epsilon) ? int_num : int_num - 1;
  }
}

constexpr inline __aicore__ int64_t Floor(int64_t num) {
  return num;
}

constexpr inline __aicore__ int Floor(int num) {
  return num;
}

constexpr inline __aicore__ unsigned int Floor(unsigned int num) {
  return num;
}

constexpr inline __aicore__ float Rational(float a, float b) {
  return (a / b);
}

template <typename T1, typename T2>
inline __aicore__ T2 Mod(T1 a, T2 b) {
  if constexpr (std::is_same<T1, float>::value || std::is_same<T2, float>::value) {
    ASCENDC_ASSERT(((b > 1e-6) || (b < -1e-6)), { KERNEL_LOG(KERNEL_ERROR, "b can't be equal to 0, b is %f!", b); });
    int64_t trunc_num = static_cast<int>(a / b);
    return static_cast<T1>(a - trunc_num * b);
  } else if constexpr (std::is_same<T1, uint64_t>::value && std::is_same<T2, uint64_t>::value) {
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "b can't be equal to 0, b is %d!", b); });
    uint64_t a_tmp = static_cast<uint64_t>(a);
    uint64_t b_tmp = static_cast<uint64_t>(b);
    uint64_t mod_num = a_tmp % b_tmp;
    return static_cast<uint64_t>(mod_num);
  } else if constexpr (std::is_same<T1, uint64_t>::value || std::is_same<T2, uint64_t>::value) {
    ASCENDC_ASSERT(true,
                   { KERNEL_LOG(KERNEL_ERROR, "does not support mix type of uint64 because of possible overflow!"); });
    return 0;
  } else {
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "b can't be equal to 0, b is %d!", b); });
    int64_t a_tmp = static_cast<int64_t>(a);
    int64_t b_tmp = static_cast<int64_t>(b);
    int64_t mod_num = a_tmp % b_tmp;
    return static_cast<T1>(mod_num);
  }
}

template <typename TilingData>
inline __aicore__ bool MatchBlockDim(TilingData &t) {
  auto block_dim = GetBlockIdx();
  // reuse ub_size as block_offset
  auto rhs = t.ub_size + t.block_dim;
  if (rhs > GetBlockNum()) {
    return (block_dim >= t.ub_size) || (block_dim < (rhs - GetBlockNum()));
  } else {
    return (block_dim >= t.ub_size) && ((block_dim - t.ub_size) < t.block_dim);
  }
}

template <typename TilingData>
inline __aicore__ bool MatchTilingKeyAndBlockDim(TilingData &t, uint32_t tiling_key) {
  return (t.tiling_key == tiling_key) && MatchBlockDim(t);
}
}  // namespace
namespace KernelUtils {
static constexpr uint64_t CONST1 = 1;
static constexpr uint64_t CONST2 = 2;
static constexpr uint64_t CONST4 = 4;
static constexpr uint64_t CONST63 = 63;

constexpr inline __aicore__ int Ceil(float num) {
  int ceil_num = (int)num;
  return num == (float)ceil_num ? ceil_num : (ceil_num + 1);
}

template <typename T>
constexpr inline __aicore__ T Min(const T a) {
  return a;
}

template <typename T, typename... Ts>
constexpr inline __aicore__ T Min(const T a, const Ts... ts) {
  return a > Min(ts...) ? Min(ts...) : a;
}

template <typename T>
constexpr inline __aicore__ T Max(const T a) {
  return a;
}

template <typename T, typename... Ts>
constexpr inline __aicore__ T Max(const T a, const Ts... ts) {
  return a > Max(ts...) ? a : Max(ts...);
}

template <typename T, typename... Ts>
constexpr inline __aicore__ T Sum(const T a, const Ts... ts) {
  return (a + ... + ts);
}

template <typename DataType>
constexpr inline __aicore__ uint16_t BlkSize() {
  return ONE_BLK_SIZE / sizeof(DataType);
}

template <typename DataType>
constexpr inline __aicore__ uint32_t BlkNum(uint32_t size) {
  return size / BlkSize<DataType>();
}

template <typename DataType>
constexpr inline __aicore__ uint16_t RptSize() {
  return ONE_REPEAT_BYTE_SIZE / sizeof(DataType);
}

template <typename DataType>
constexpr inline __aicore__ uint32_t RptNum(uint32_t size) {
  return size / RptSize<DataType>();
}

template <typename DataType>
constexpr inline __aicore__ uint32_t MaxRptSize() {
  return MAX_REPEAT_TIME * RptSize<DataType>();
}

template <typename DataType>
constexpr inline __aicore__ uint32_t BlkAlign(uint32_t size) {
  return (size * sizeof(DataType) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE / sizeof(DataType);
}

template <typename DataType>
constexpr inline __aicore__ uint32_t BlkFloorAlign(uint32_t size) {
  if ((size * sizeof(DataType) + 1) > ONE_BLK_SIZE) {
    return ((size * sizeof(DataType) + 1) - ONE_BLK_SIZE) / ONE_BLK_SIZE * ONE_BLK_SIZE / sizeof(DataType);
  }
  return 0U;
}

constexpr inline __aicore__ uint32_t SizeAlign(uint32_t size, uint32_t align_size) {
  return (size + align_size - 1) / align_size * align_size;
}

template <typename T, typename T1>
inline __aicore__ LocalTensor<T> NewTensor(LocalTensor<T1> &tmp_buf, uint32_t offset, uint32_t cnt) {
  AscendC::LocalTensor<T> alloc_buf = tmp_buf[offset].template ReinterpretCast<T>();
  alloc_buf.SetSize(cnt);
  return alloc_buf;
}

__aicore__ inline uint64_t FindNearestPower2(const uint64_t value) {
  if (value == 0) {
    return 0;
  } else if (value <= CONST2) {
    return CONST1;
  } else if (value <= CONST4) {
    return CONST2;
  } else {
    const uint64_t num = value - CONST1;
    const uint64_t pow = CONST63 - clz(num);
    return (CONST1 << pow);
  }
}

__aicore__ inline uint64_t CalLog2(uint64_t value) {
  uint64_t res = 0;
  while (value > CONST1) {
    value = value >> CONST1;
    res++;
  }
  return res;
}

}  // namespace KernelUtils

template <typename T>
constexpr inline __aicore__ T Min(const T a) {
  return a;
}

template <typename T, typename... Ts>
constexpr inline __aicore__ T Min(const T a, const Ts... ts) {
  return KernelUtils::Min(a, ts...);
}

template <typename T>
constexpr inline __aicore__ T Max(const T a) {
  return a;
}

template <typename T, typename... Ts>
constexpr inline __aicore__ T Max(const T a, const Ts... ts) {
  return KernelUtils::Max(a, ts...);
}

inline __aicore__ uint64_t Log(uint64_t value) {
  return KernelUtils::CalLog2(value);
}

inline __aicore__ float Pow(const int base, int exponent) {
  float result = 1.0;
  if (exponent >= 0) {
    for (int i = 0; i < exponent; ++i) {
      result *= base;
    }
  } else {
    exponent = -exponent;
    for (int i = 0; i < exponent; ++i) {
      result *= base;
    }
    result = 1 / result;
  }

  return result;
}

// 定义inf
template <typename T, typename U>
__aicore__ inline U GetScalarValueByBitCode(T bit_code) {
  union ScalarBitcode {
    __aicore__ ScalarBitcode() {}
    T input;
    U output;
  } data;

  data.input = bit_code;
  return static_cast<U>(data.output);
}

template <typename T>
constexpr __aicore__ static inline T AfInfinity() {
  static_assert(SupportType<T, half, float>(), "current data type is not support inf");
  if constexpr (std::is_same_v<T, half>) {
    return GetScalarValueByBitCode<uint16_t, T>(0x7C00U);
  } else if constexpr (std::is_same_v<T, float>) {
    return GetScalarValueByBitCode<uint32_t, T>(0x7F800000U);
  }

  return T();
}

inline __aicore__ bool IsIntegerWithinUint32Magnitude(float value) {
  constexpr uint32_t kExponentMask = 0x7F800000U;
  constexpr uint32_t kFractionMask = 0x007FFFFFU;
  constexpr uint32_t kExponentBias = 127U;
  constexpr uint32_t kFractionBits = 23U;
  constexpr uint32_t kUint32Exponent = 32U;

  const uint32_t bits = GetScalarValueByBitCode<float, uint32_t>(value);
  const uint32_t exponent = (bits & kExponentMask) >> kFractionBits;
  const uint32_t fraction = bits & kFractionMask;

  // NaN and infinity are not integers.
  if (exponent == 0xFFU) {
    return false;
  }

  // Ignore the sign bit and limit the absolute value to the uint32 range.
  if (exponent >= kExponentBias + kUint32Exponent) {
    return false;
  }

  // Zero and non-zero values with an absolute value less than one.
  if (exponent < kExponentBias) {
    return exponent == 0U && fraction == 0U;
  }

  const uint32_t integer_bits = exponent - kExponentBias;
  if (integer_bits >= kFractionBits) {
    return true;
  }

  const uint32_t fractional_mask = (1U << (kFractionBits - integer_bits)) - 1U;
  return (fraction & fractional_mask) == 0U;
}

static constexpr float ROUND_TO_NEAREST_INT_BIAS = 0.5f;
template <typename T>
inline __aicore__ uint32_t ConvertToUint32(T value) {
  if constexpr (std::is_floating_point<T>::value) {
    // 默认tiling_data是uint32_t的，而ascendc不支持将uint32_t转为float
    // 因此参与float计算时，aicore函数中不能直接将float转换为unsigned int，需要转换为int，再转换为unsigned int
    return static_cast<uint32_t>(static_cast<int32_t>(value + ROUND_TO_NEAREST_INT_BIAS));
  } else {
    return static_cast<uint32_t>(value);
  }
}

#endif  // __ASCENDC_API_UTILS_H__
