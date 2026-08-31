/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REDUCE_H__
#define __ASCENDC_API_REDUCE_H__

template <typename T>
inline __aicore__ void WholeReduceSumAdapt(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                           const int32_t mask, const int32_t repeatTimes, const int32_t dstRepStride,
                                           const int32_t srcBlkStride, const int32_t srcRepStride, ReduceOrder order,
                                           const int32_t k) {
  WholeReduceSum(dstLocal, srcLocal, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride);
}

template <typename T>
inline __aicore__ void WholeReduceMaxAdapt(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                           const int32_t mask, const int32_t repeatTimes, const int32_t dstRepStride,
                                           const int32_t srcBlkStride, const int32_t srcRepStride, ReduceOrder order,
                                           const int32_t k) {
  WholeReduceMax(dstLocal, srcLocal, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, order);
}

template <typename T>
inline __aicore__ void WholeReduceMinAdapt(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                           const int32_t mask, const int32_t repeatTimes, const int32_t dstRepStride,
                                           const int32_t srcBlkStride, const int32_t srcRepStride, ReduceOrder order,
                                           const int32_t k) {
  WholeReduceMin(dstLocal, srcLocal, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, order);
}

template <typename T>
inline __aicore__ void WholeReduceMeanAdapt(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                            const int32_t mask, const int32_t repeatTimes, const int32_t dstRepStride,
                                            const int32_t srcBlkStride, const int32_t srcRepStride, ReduceOrder order,
                                            const int32_t k) {
  WholeReduceSum(dstLocal, srcLocal, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride);
  Muls(dstLocal, dstLocal, static_cast<T>(static_cast<float>(1) / static_cast<float>(k)), repeatTimes);
}

template <typename T,
          void (*WholeReduceFunc)(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, const int32_t mask,
                                  const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,
                                  const int32_t srcRepStride, ReduceOrder order, const int32_t k),
          void (*BinaryFunc)(const LocalTensor<T> &dstLocal, const LocalTensor<T> &src0Local,
                             const LocalTensor<T> &src1Local, const int32_t &calCount)>
inline __aicore__ void ReduceLast(const LocalTensor<T> &dst, const LocalTensor<T> &src, const int32_t m,
                                  const int32_t k, const int32_t stride, LocalTensor<uint8_t> &tmp_buf) {
  const uint16_t k_repeat_size = ONE_REPEAT_BYTE_SIZE / sizeof(T);
  const uint16_t m_repeat_size = MAX_REPEAT_TIMES * sizeof(T) / ONE_BLK_SIZE * ONE_BLK_SIZE / sizeof(T);
  int32_t srcRepStride = KernelUtils::BlkNum<T>(stride);
  if (m <= m_repeat_size && k <= k_repeat_size && (stride % KernelUtils::BlkSize<T>() == 0)) {
    WholeReduceFunc(dst, src, k, m, 1, 1, srcRepStride, ReduceOrder::ORDER_ONLY_VALUE, k);
  } else if (m > m_repeat_size && k <= k_repeat_size && (stride % KernelUtils::BlkSize<T>() == 0)) {
    for (int i = 0; i < m; i += m_repeat_size) {
      WholeReduceFunc(dst[i], src[i * k], k, KernelUtils::Min(m - i, m_repeat_size), 1, 1, srcRepStride,
                      ReduceOrder::ORDER_ONLY_VALUE, k);
    }
  } else if (m < m_repeat_size && k > k_repeat_size && (stride % KernelUtils::BlkSize<T>() == 0)) {
    LocalTensor<T> tmp_tensor = tmp_buf[0].template ReinterpretCast<T>();
    AscendC::PipeBarrier<PIPE_V>();
    WholeReduceFunc(dst, src, k_repeat_size, m, 1, 1, srcRepStride, ReduceOrder::ORDER_ONLY_VALUE, k);
    for (int i = k_repeat_size; i < k; i += k_repeat_size) {
      AscendC::PipeBarrier<PIPE_V>();
      WholeReduceFunc(tmp_tensor, src[i], i + k_repeat_size < k ? k_repeat_size : k - i, m, 1, 1, srcRepStride,
                      ReduceOrder::ORDER_ONLY_VALUE, k);
      AscendC::PipeBarrier<PIPE_V>();
      BinaryFunc(dst, dst, tmp_tensor, m);
    }
  } else {
    ASSERT(false && "Reduce k size not support.");
  }
}

__aicore__ inline int64_t GetCacheID(const int64_t idx) {
  return bcnt1(idx ^ (idx + 1)) - 1;
}

template <typename T,
          void (*operatorFunc)(const LocalTensor<T> &, const LocalTensor<T> &, const LocalTensor<T> &, const int32_t &)>
__aicore__ inline void CacheUpdate(const LocalTensor<T> &dst, const LocalTensor<T> &src, int64_t index, int64_t dim_a) {
  int64_t align_num = ONE_BLK_SIZE / sizeof(T);
  int64_t dim_a_align = (dim_a + align_num - 1) / align_num * align_num;
  int64_t cache_id = GetCacheID(index);
  int64_t cache_offset = cache_id * dim_a_align;
  for (uint64_t i = 0; i < cache_id; i++) {
    operatorFunc(src, dst[i * dim_a_align], src, dim_a);
    AscendC::PipeBarrier<PIPE_V>();
  }
  AscendC::PipeBarrier<PIPE_V>();
  DataCopy(dst[cache_offset], src, dim_a_align);
}

template <typename T>
__aicore__ inline void PreProcessReduceSumInt32ForAR(const LocalTensor<T> &src, const LocalTensor<T> &curr_buff,
                                                     uint32_t row, uint32_t last, uint32_t pad_last, uint32_t remain,
                                                     uint32_t &split_k) {
  BinaryRepeatParams default_param;
  UnaryRepeatParams default_unary_param;
  constexpr uint32_t one_blk_num = ONE_BLK_SIZE / sizeof(T);
  if (last >= one_blk_num && remain != 0) {
    SetVectorMask<T, MaskMode::COUNTER>(0, split_k);
    Adds<T, false>(curr_buff, src[row * pad_last], static_cast<T>(0), MASK_PLACEHOLDER, 1, default_unary_param);
    AscendC::PipeBarrier<PIPE_V>();

    SetVectorMask<T, MaskMode::COUNTER>(0, remain);
    Add<T, false>(curr_buff, curr_buff, src[row * pad_last + split_k], MASK_PLACEHOLDER, 1, default_param);
    AscendC::PipeBarrier<PIPE_V>();
  } else if (split_k > one_blk_num) {
    split_k >>= 1;
    SetVectorMask<T, MaskMode::COUNTER>(0, split_k);

    Add<T, false>(curr_buff, src[row * pad_last], src[row * pad_last + split_k], MASK_PLACEHOLDER, 1, default_param);
    AscendC::PipeBarrier<PIPE_V>();
  } else {
    SetVectorMask<T, MaskMode::COUNTER>(0, last);
    Adds<T, false>(curr_buff, src[row * pad_last], static_cast<T>(0), MASK_PLACEHOLDER, 1, default_unary_param);
    AscendC::PipeBarrier<PIPE_V>();
  }
}

template <typename T>
__aicore__ inline void ReduceSumByLastAxis(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                           const LocalTensor<T> &tmp, uint32_t first, uint32_t last,
                                           uint32_t pad_last) {
  constexpr uint32_t one_blk_num = ONE_BLK_SIZE / sizeof(T);
  constexpr uint32_t one_rpt_num = ONE_REPEAT_BYTE_SIZE / sizeof(T);

  BinaryRepeatParams default_param;
  UnaryRepeatParams default_unary_param;
  BrcbRepeatParams default_brcb_param;
  LocalTensor<T> res_before_gather = tmp[one_rpt_num];
  LocalTensor<T> final_res_stored = res_before_gather;

  uint32_t k = AscendC::Internal::FindClosestPowerOfTwo(last);
  uint32_t split_k = 1 << k;
  uint32_t remain = last - split_k;
  SetMaskCount();
  AscendC::CheckTmpBufferSize(tmp.GetSize(), 0, tmp.GetSize());
  for (uint32_t j = 0; j < first; j++) {
    uint32_t split_k_copy = split_k;
    LocalTensor<T> tmp_row_res = res_before_gather[j * one_blk_num];
    LocalTensor<T> tmp_dst = res_before_gather[j * one_blk_num];

    PreProcessReduceSumInt32ForAR(src, tmp_row_res, j, last, pad_last, remain, split_k_copy);

    while (split_k_copy > one_blk_num) {
      split_k_copy >>= 1;
      SetVectorMask<T, MaskMode::COUNTER>(0, split_k_copy);

      Add<T, false>(tmp_row_res, tmp_row_res, tmp_row_res[split_k_copy], MASK_PLACEHOLDER, 1, default_param);
      AscendC::PipeBarrier<PIPE_V>();
    }

    Brcb(tmp, tmp_row_res, 1, default_brcb_param);
    AscendC::PipeBarrier<PIPE_V>();
    uint32_t final_tail = last < one_blk_num ? split_k : one_blk_num;
    if (split_k != last && final_tail < one_blk_num) {
      SetVectorMask<T, MaskMode::COUNTER>(0, remain * one_blk_num);
      Add<T, false>(tmp, tmp, tmp[split_k * one_blk_num], MASK_PLACEHOLDER, 1, default_param);
      AscendC::PipeBarrier<PIPE_V>();
    }
    while (final_tail > 1) {
      final_tail >>= 1;
      SetVectorMask<T, MaskMode::COUNTER>(0, final_tail * one_blk_num);

      Add<T, false>(tmp, tmp, tmp[final_tail * one_blk_num], MASK_PLACEHOLDER, 1, default_param);
      AscendC::PipeBarrier<PIPE_V>();
    }
    SetVectorMask<T, MaskMode::COUNTER>(0, one_blk_num);
    Adds<T, false>(tmp_dst, tmp, static_cast<T>(0), MASK_PLACEHOLDER, 1, default_unary_param);
    AscendC::PipeBarrier<PIPE_V>();
  }
  LocalTensor<uint32_t> tmp_int = tmp.template ReinterpretCast<uint32_t>();
  Duplicate(tmp_int, 1u, one_rpt_num);
  AscendC::PipeBarrier<PIPE_V>();
  GatherMaskParams gather_mask_param = {1, static_cast<uint16_t>(first), 1, 0};
  uint64_t rsvd_cnt;
  GatherMask(dst, final_res_stored, tmp_int, true, one_blk_num, gather_mask_param, rsvd_cnt);
}

template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceSumInt32(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                      const LocalTensor<uint8_t> &tmp, const uint32_t src_shape[], bool src_inner_pad) {
  static_assert(SupportType<pattern, Pattern::Reduce::AR, Pattern::Reduce::RA>(),
                "failed to check the reduce pattern, it only supports AR/RA pattern");
  static_assert(SupportType<T, int32_t>(), "failed to check the data type, current api supports data type is int32_t");

  uint32_t last = src_shape[1];
  uint32_t first = src_shape[0];
  constexpr uint32_t one_blk_num = ONE_BLK_SIZE / sizeof(T);
  uint32_t pad_last = AlignUp(last, one_blk_num);
  LocalTensor<T> tmp_dst = tmp.ReinterpretCast<T>();
  if constexpr (IsSameType<pattern, Pattern::Reduce::AR>::value) {
    ASCENDC_ASSERT((dst.GetSize() >= first), {
      KERNEL_LOG(KERNEL_ERROR, "dstTensor must be greater than or equal to %u, current size is %u", first,
                 dst.GetSize());
    });
    ReduceSumByLastAxis(dst, src, tmp_dst, first, last, pad_last);
  } else {
    AscendC::Internal::BinaryReduceByFirstAxis<T, false, Add<T, false>>(dst, src, tmp_dst, first, last, pad_last);
  }
  SetMaskNorm();
  ResetMask();
}

#endif  // __ASCENDC_API_REDUCE_H__
