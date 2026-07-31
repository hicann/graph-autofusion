/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCEINIT_H
#define REDUCEINIT_H
using namespace AscendC;

#ifndef INFINITY
#define INFINITY (1.0f / 0.0f)
#endif

constexpr int32_t kReduceOpMin = 0;
constexpr int32_t kReduceOpMax = 1;
constexpr int32_t kReduceOpSum = 2;
constexpr int32_t kReduceOpProd = 3;
constexpr int32_t kReduceOpAny = 4;
constexpr int32_t kReduceOpAll = 5;
constexpr int32_t kReduceOpMean = 6;

template <typename T, int reduce_type>
inline __aicore__ T GetPaddingValue() {
  T paddingValue;
  if constexpr (reduce_type == kReduceOpMin) {
    if constexpr (AscendC::Std::is_same<T, half>::value || AscendC::Std::is_same<T, float>::value ||
                  AscendC::Std::is_same<T, bfloat16_t>::value) {
      paddingValue = INFINITY;
    } else if constexpr (static_cast<T>(-1) < static_cast<T>(0)) {
      paddingValue = static_cast<T>(~static_cast<T>(static_cast<T>(1) << (sizeof(T) * 8 - 1)));
    } else {
      paddingValue = static_cast<T>(~static_cast<T>(0));
    }
  } else if constexpr (reduce_type == kReduceOpMax) {
    if constexpr (AscendC::Std::is_same<T, half>::value || AscendC::Std::is_same<T, float>::value ||
                  AscendC::Std::is_same<T, bfloat16_t>::value) {
      paddingValue = -INFINITY;
    } else if constexpr (static_cast<T>(-1) < static_cast<T>(0)) {
      paddingValue = static_cast<T>(static_cast<T>(1) << (sizeof(T) * 8 - 1));
    } else {
      paddingValue = T(0);
    }
  } else if constexpr (reduce_type == kReduceOpSum) {
    paddingValue = T(0);
  } else if constexpr (reduce_type == kReduceOpProd) {
    paddingValue = T(1);
  } else if constexpr (reduce_type == kReduceOpAny) {
    paddingValue = T(0);
  } else if constexpr (reduce_type == kReduceOpAll) {
    paddingValue = T(1);
  } else if constexpr (reduce_type == kReduceOpMean) {
    paddingValue = T(0);
  }
  return paddingValue;
}

template <typename T>
__simd_vf__ inline void OptImpl(__ubuf__ T *dst, __ubuf__ uint32_t *maskBuf, uint32_t dim_a, uint32_t dim_r,
                                uint32_t dim_r_current, uint16_t outterLoopRepeat, uint16_t innerLoopRepeat,
                                uint16_t tailBlockCount, uint16_t dataBlockStride, uint16_t repeatStride,
                                uint16_t inner_r_upper, uint16_t inner_r_down, T padValue) {
  MicroAPI::RegTensor<T> mainPadReg;
  MicroAPI::RegTensor<T> tailPadReg;
  MicroAPI::MaskReg mainBlockPreg;
  MicroAPI::MaskReg tailBlockPreg;

  MicroAPI::LoadAlign(mainBlockPreg, maskBuf);
  MicroAPI::LoadAlign(tailBlockPreg, maskBuf + 8);
  MicroAPI::Duplicate<T>(mainPadReg, padValue, mainBlockPreg);
  MicroAPI::Duplicate<T>(tailPadReg, padValue, tailBlockPreg);

  for (uint16_t i = 0; i < outterLoopRepeat; i++) {
    for (uint16_t j = 0; j < innerLoopRepeat; j++) {
      AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
          dst + inner_r_down + i * inner_r_upper + 8 * j * dim_r, mainPadReg, uint32_t(dataBlockStride), mainBlockPreg);
    }
    if (tailBlockCount != 0) {
      AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
          dst + inner_r_down + i * inner_r_upper + 8 * innerLoopRepeat * dim_r, tailPadReg, dataBlockStride,
          tailBlockPreg);
    }
  }
}

__simd_vf__ inline void MaskDupVF(__ubuf__ uint32_t *maskBuf, uint32_t count, uint32_t value) {
  MicroAPI::RegTensor<uint32_t> dupVreg;
  MicroAPI::MaskReg preg;
  MicroAPI::MaskReg mask;
  mask = MicroAPI::UpdateMask<uint32_t>(count);
  preg = pset_b8(PAT_ALL);
  MicroAPI::Duplicate(dupVreg, value, mask);
  MicroAPI::DataCopy(maskBuf, dupVreg, preg);
}

template <typename T>
__aicore__ inline bool MaskPreprocess(__ubuf__ uint32_t *maskBuf, uint32_t blockCount, uint32_t inner_r,
                                      uint32_t inner_r_down) {
  uint16_t shift_length = (inner_r - inner_r_down) * sizeof(T);
  uint32_t full_mask_u32 = (shift_length == 0) ? 0x00000000 : 0xffffffff;
  if (full_mask_u32 == 0x00000000) {
    return false;
  }
  uint32_t pad_mask = full_mask_u32 << shift_length;
  uint32_t count_tail = blockCount % 8;
  uint32_t count_total = 8 + count_tail;
  MaskDupVF(maskBuf, count_total, pad_mask);
  return true;
}

template <typename T>
__simd_vf__ inline void ReduceInitAlignImpl(__local_mem__ T *dstUb, uint32_t repeatTimesAlign,
                                            uint32_t pad_length_total, uint32_t inner_r_offset, uint32_t stride_element,
                                            uint32_t tail_pad_count, uint32_t tail_ctrl, T padValue, uint32_t dim_a,
                                            uint32_t dim_r) {
  MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
  // align
  MicroAPI::MaskReg mask;
  MicroAPI::RegTensor<T> padReg;
  MicroAPI::Duplicate<T, AscendC::MicroAPI::MaskMergeMode::MERGING>(padReg, padValue, fullMask);

  for (uint32_t j = 0; j < dim_a; ++j) {
    for (uint32_t i = 0; i < repeatTimesAlign; ++i) {
      MicroAPI::DataCopy<T>(dstUb + inner_r_offset + i * stride_element + j * dim_r, padReg, fullMask);
    }
  }

  mask = AscendC::MicroAPI::UpdateMask<T>(tail_pad_count);
  for (uint32_t i = 0; i < tail_ctrl; ++i) {
    for (uint32_t j = 0; j < dim_a; ++j) {
      MicroAPI::DataCopy<T>(dstUb + inner_r_offset + repeatTimesAlign * stride_element + j * dim_r, padReg, mask);
    }
  }
}

template <typename T>
__simd_vf__ inline void ReduceInitUnalignImpl(__local_mem__ T *dstUb, uint32_t repeatTimesAlign,
                                              uint32_t pad_length_total, uint32_t inner_r_offset,
                                              uint32_t stride_element, uint32_t tail_pad_count, uint32_t tail_ctrl,
                                              T padValue, uint32_t dim_a, uint32_t dim_r, uint32_t sreg,
                                              uint32_t baseOffset, uint32_t inner_r_repeat_times, uint32_t inner_r_up) {
  MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::RegTensor<T> srcVreg;
  MicroAPI::MaskReg preg_pad;
  MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<T>(sreg);
  MicroAPI::MaskNot(preg_pad, preg, fullMask);
  uint32_t copy_out_num = inner_r_up - baseOffset;
  MicroAPI::MaskReg preg_out = AscendC::MicroAPI::UpdateMask<T>(copy_out_num);

  for (uint32_t j = 0; j < dim_a; ++j) {
    for (uint32_t i = 0; i < inner_r_repeat_times; ++i) {
      MicroAPI::DataCopy<T>(srcVreg, dstUb + baseOffset + i * inner_r_up + j * dim_r);
      MicroAPI::Duplicate<T, AscendC::MicroAPI::MaskMergeMode::MERGING>(srcVreg, padValue, preg_pad);
      MicroAPI::DataCopy<T>(dstUb + baseOffset + i * inner_r_up + j * dim_r, srcVreg, preg_out);
    }
  }
}

template <typename T>
__simd_vf__ inline void ReduceInitRowImpl(__ubuf__ T *dst, __ubuf__ uint32_t *maskBuf, uint32_t innerLoopRepeat,
                                          uint16_t tailBlockCount, uint32_t dataBlockStride, uint16_t inner_r_upper,
                                          uint16_t inner_r_down, T padValue) {
  MicroAPI::RegTensor<T> mainPadReg;
  MicroAPI::MaskReg mainBlockPreg;

  MicroAPI::LoadAlign(mainBlockPreg, maskBuf);
  MicroAPI::Duplicate<T>(mainPadReg, padValue, mainBlockPreg);

  for (uint32_t i = 0; i < innerLoopRepeat; ++i) {
    MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(dst + inner_r_down + 8 * i * inner_r_upper,
                                                                     mainPadReg, dataBlockStride, mainBlockPreg);
  }
  if (tailBlockCount != 0) {
    MicroAPI::RegTensor<T> tailPadReg;
    MicroAPI::MaskReg tailBlockPreg;
    MicroAPI::LoadAlign(tailBlockPreg, maskBuf + 8);
    MicroAPI::Duplicate<T>(tailPadReg, padValue, tailBlockPreg);
    MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
        dst + inner_r_down + 8 * innerLoopRepeat * inner_r_upper, tailPadReg, dataBlockStride, tailBlockPreg);
  }
}

template <typename T, int32_t ReduceType, bool isTailLast>
__aicore__ inline void ReduceInit(const LocalTensor<T> &dstTensor, const uint32_t dim_a, const uint32_t dim_r,
                                  const uint32_t dim_r_current, const uint32_t inner_r) {
  static_assert(SupportType<T, half, float, bfloat16_t, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t,
                            int64_t>());
  constexpr bool is_b64 = AscendC::Std::is_same<T, int64_t>::value || AscendC::Std::is_same<T, uint64_t>::value;
  uint16_t inner_r_upper;
  uint16_t inner_r_down;
  uint32_t un_dim_a, un_dim_r, un_dim_r_current;
  uint32_t alignCount = 32 / sizeof(T);
  inner_r_upper = CeilDivision(inner_r, alignCount) * alignCount;
  inner_r_down = (inner_r / alignCount) * alignCount;

  // Unfold to column major
  if ((dim_r % inner_r_upper == 0) && (dim_r == dim_r_current)) {
    un_dim_a = dim_a * dim_r / inner_r_upper;
    un_dim_r = inner_r_upper;
    un_dim_r_current = inner_r_upper;
  } else {
    un_dim_a = dim_a;
    un_dim_r = dim_r;
    un_dim_r_current = dim_r_current;
  }

  uint32_t blockCountOp0 = un_dim_a;
  uint32_t blockCountOp1 = un_dim_r_current / inner_r_upper;

  constexpr uint16_t BLOCK_COUNT_PER_REPEAT = 8;
  uint16_t outterLoopRepeat = static_cast<uint16_t>(un_dim_r_current / inner_r_upper);
  uint16_t innerLoopRepeat = static_cast<uint16_t>(un_dim_a / BLOCK_COUNT_PER_REPEAT);
  uint16_t tailBlockCount = un_dim_a % BLOCK_COUNT_PER_REPEAT;
  uint16_t dataBlockStride = un_dim_r * sizeof(T) / 32;
  uint16_t repeatStride = dataBlockStride * 8;
  uint32_t rowInnerLoopRepeat = blockCountOp1 / BLOCK_COUNT_PER_REPEAT;
  uint16_t rowTailBlockCount = blockCountOp1 % BLOCK_COUNT_PER_REPEAT;
  uint32_t rowDataBlockStride = inner_r_upper * sizeof(T) / 32;
  T padValue = GetPaddingValue<T, ReduceType>();

  // unalign
  uint32_t baseOffset = (inner_r * sizeof(T) / AscendC::ONE_BLK_SIZE) * AscendC::ONE_BLK_SIZE;
  baseOffset = baseOffset / sizeof(T);
  uint32_t sreg = inner_r - baseOffset;
  uint32_t inner_r_up =
      (inner_r * sizeof(T) + AscendC::ONE_BLK_SIZE - 1) / AscendC::ONE_BLK_SIZE * AscendC::ONE_BLK_SIZE / sizeof(T);
  uint32_t inner_r_repeat_times = un_dim_r_current / inner_r_up;

  // align
  uint32_t pad_length_tail = un_dim_r - un_dim_r_current;
  uint32_t pad_length_head = un_dim_r_current % inner_r_up;
  uint32_t pad_length_total = pad_length_tail + pad_length_head;
  uint32_t repeatTimesAlign = pad_length_total * sizeof(T) / VECTOR_REG_WIDTH;
  uint32_t inner_r_offset = un_dim_r_current / inner_r_up * inner_r_up;
  uint32_t stride_element = VECTOR_REG_WIDTH / sizeof(T);
  uint32_t tail_pad_count = pad_length_total % stride_element;
  uint32_t tail_ctrl = (tail_pad_count == 0) ? 0 : 1;

  bool is_inner_r_align = (inner_r * sizeof(T) % AscendC::ONE_BLK_SIZE == 0);

  __ubuf__ uint32_t *maskBuf = nullptr;
#if defined(AUTOFUSE_SIMT_RESERVED_UB_SIZE)
  maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint32_t>(TMP_UB_OFFSET - AUTOFUSE_SIMT_RESERVED_UB_SIZE, 64);
#else
  maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint32_t>(TMP_UB_OFFSET, 64);
#endif
  uint32_t maskBlockCount = (un_dim_a == 1) ? blockCountOp1 : blockCountOp0;
  bool need_exec = MaskPreprocess<T>(maskBuf, maskBlockCount, inner_r, inner_r_down);
  if constexpr (is_b64) {
    if (!is_inner_r_align) {
      ReduceInitUnalignImpl<T>((__ubuf__ T *)dstTensor.GetPhyAddr(), repeatTimesAlign, pad_length_total, inner_r_offset,
                               stride_element, tail_pad_count, tail_ctrl, padValue, un_dim_a, un_dim_r, sreg,
                               baseOffset, inner_r_repeat_times, inner_r_up);
    }
    ReduceInitAlignImpl<T>((__ubuf__ T *)dstTensor.GetPhyAddr(), repeatTimesAlign, pad_length_total, inner_r_offset,
                           stride_element, tail_pad_count, tail_ctrl, padValue, un_dim_a, un_dim_r);
  } else {
    if (un_dim_r > un_dim_r_current) {
      ReduceInitAlignImpl<T>((__ubuf__ T *)dstTensor.GetPhyAddr(), repeatTimesAlign, pad_length_total, inner_r_offset,
                             stride_element, tail_pad_count, tail_ctrl, padValue, un_dim_a, un_dim_r);
    }
    if (need_exec) {
      if (un_dim_a == 1) {
        ReduceInitRowImpl<T>((__ubuf__ T *)dstTensor.GetPhyAddr(), maskBuf, rowInnerLoopRepeat, rowTailBlockCount,
                             rowDataBlockStride, inner_r_upper, inner_r_down, padValue);
      } else {
        OptImpl<T>((__ubuf__ T *)dstTensor.GetPhyAddr(), maskBuf, un_dim_a, un_dim_r, un_dim_r_current,
                   outterLoopRepeat, innerLoopRepeat, tailBlockCount, dataBlockStride, repeatStride, inner_r_upper,
                   inner_r_down, padValue);
      }
    }
  }
  AscendC::AscendCUtils::FreeTemporaryBuffer<uint32_t>(maskBuf);
}
#endif
