/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_POW_H__
#define __ASCENDC_API_POW_H__

static constexpr AscendC::PowerConfig pow_config = {PowerAlgo::DOUBLE_FLOAT_TECH};

template <typename T>
inline __aicore__ void Pow(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src1,
                           const AscendC::LocalTensor<T> &src2, AscendC::LocalTensor<uint8_t> &tmp_buf,
                           const uint32_t calCount) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>) {
    Power<T, false, pow_config>(dst, src1, src2, tmp_buf, calCount);
  } else {
    Power(dst, src1, src2, tmp_buf, calCount);
  }
}

template <typename T>
inline __aicore__ void Pow(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src1, const T &src2,
                           AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t calCount) {
  if (static_cast<float>(src2) == 0.0f) {
    Duplicate(dst, static_cast<T>(1.0), calCount);
    return;
  } else if (static_cast<float>(src2) == 1.0f) {
    DataCopy(dst, src1, KernelUtils::BlkAlign<T>(calCount));
    return;
  } else if (static_cast<float>(src2) == -1.0f) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
      Reciprocal(dst, src1, calCount);
      return;
    }
  } else if (static_cast<float>(src2) == 2.0f) {
    if constexpr (!std::is_same_v<T, int8_t> && !std::is_same_v<T, uint8_t>) {
      Mul(dst, src1, src1, calCount);
      return;
    }
  } else if (static_cast<float>(src2) == 0.5f) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
      Sqrt(dst, src1, calCount);
      return;
    }
  } else if (static_cast<float>(src2) == -0.5f) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
      Rsqrt(dst, src1, calCount);
      return;
    }
  } else if (static_cast<float>(src2) == 3.0f) {
    if constexpr (!std::is_same_v<T, int8_t> && !std::is_same_v<T, uint8_t>) {
      Mul(dst, src1, src1, calCount);
      Mul(dst, dst, src1, calCount);
      return;
    }
  } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>) {
    Power<T, false, pow_config>(dst, src1, src2, tmp_buf, calCount);
    return;
  }
  Power(dst, src1, src2, tmp_buf, calCount);
}

template <typename T>
inline __aicore__ void Pow(const AscendC::LocalTensor<T> &dst, const T &src1, const AscendC::LocalTensor<T> &src2,
                           AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t calCount) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>) {
    Power<T, false, pow_config>(dst, src1, src2, tmp_buf, calCount);
  } else {
    Power(dst, src1, src2, tmp_buf, calCount);
  }
}

template <typename T>
inline __aicore__ void Pow(const AscendC::LocalTensor<T> &dst, const T &src1, const T &src2,
                           AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t calCount) {
  auto block_cnt = KernelUtils::BlkSize<T>();
  // 将src1扩充为一个blockSize的tensor
  LocalTensor<T> src1_buf = tmp_buf.template ReinterpretCast<T>();
  src1_buf.SetSize(block_cnt);
  LocalTensor<T> dst_buf = tmp_buf[ONE_BLK_SIZE].template ReinterpretCast<T>();
  dst_buf.SetSize(block_cnt);

  LocalTensor<uint8_t> left_tmp_buf = tmp_buf[2 * ONE_BLK_SIZE].template ReinterpretCast<uint8_t>();
  Duplicate(src1_buf, src1, block_cnt);
  // 调用Power基础API：tensor(block size) + scalar
  if (static_cast<float>(src2) == 0.0f) {
    Duplicate(dst, static_cast<T>(1.0), calCount);
    return;
  } else if (static_cast<float>(src2) == 1.0f) {
    Duplicate(dst, src1, calCount);
    return;
  } else if (static_cast<float>(src2) == -1.0f) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
      Reciprocal(dst_buf, src1_buf, block_cnt);
      Duplicate(dst, dst_buf.GetValue(0), calCount);
      return;
    }
  } else if (static_cast<float>(src2) == 2.0f) {
    if constexpr (!std::is_same_v<T, int8_t> && !std::is_same_v<T, uint8_t>) {
      Mul(dst_buf, src1_buf, src1_buf, block_cnt);
      Duplicate(dst, dst_buf.GetValue(0), calCount);
      return;
    }
  } else if (static_cast<float>(src2) == 0.5f) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
      Sqrt(dst_buf, src1_buf, block_cnt);
      Duplicate(dst, dst_buf.GetValue(0), calCount);
      return;
    }
  } else if (static_cast<float>(src2) == -0.5f) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
      Rsqrt(dst_buf, src1_buf, block_cnt);
      Duplicate(dst, dst_buf.GetValue(0), calCount);
      return;
    }
  } else if (static_cast<float>(src2) == 3.0f) {
    if constexpr (!std::is_same_v<T, int8_t> && !std::is_same_v<T, uint8_t>) {
      Mul(dst_buf, src1_buf, src1_buf, block_cnt);
      Mul(dst_buf, dst_buf, src1_buf, block_cnt);
      Duplicate(dst, dst_buf.GetValue(0), calCount);
      return;
    }
  } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>) {
    Power<T, false, pow_config>(dst_buf, src1_buf, src2, tmp_buf, block_cnt);
    Duplicate(dst, dst_buf.GetValue(0), calCount);
    return;
  }
  Power(dst_buf, src1_buf, src2, tmp_buf, block_cnt);
  // 取block tensor中的一个scalar元素，扩充为dst size长度的tensor
  Duplicate(dst, dst_buf.GetValue(0), calCount);
}
#endif  // __ASCENDC_API_POW_H__
