/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_CAST_H__
#define __ASCENDC_API_CAST_H__

template <typename InT, typename OutT>
inline __aicore__ AscendC::RoundMode GetRoundMode() {
  // 对标单算子和torch
  // 浮点转浮点：低转高：CAST_NONE，高转低：CAST_RINT
  // 浮点转整形：CAST_TRUNC
  // 整形转整形：CAST_NONE
  // 整形转浮点：CAST_NONE
  if constexpr (AscendC::IsSameType<InT, float>::value) {
    if constexpr (AscendC::SupportType<OutT, half, bfloat16_t>()) {
      return AscendC::RoundMode::CAST_RINT;
    }
    if constexpr (AscendC::SupportType<OutT, int64_t, int32_t, int16_t>()) {
      return AscendC::RoundMode::CAST_TRUNC;
    }
  }
  if constexpr (AscendC::IsSameType<InT, half>::value &&
                AscendC::SupportType<OutT, int32_t, int16_t, int8_t, uint8_t>()) {
    return AscendC::RoundMode::CAST_TRUNC;
  }
  if constexpr (AscendC::IsSameType<InT, int64_t>::value && AscendC::SupportType<OutT, float>()) {
    return AscendC::RoundMode::CAST_RINT;
  }
  return AscendC::RoundMode::CAST_NONE;
}

template <typename InT, typename OutT, typename InterT>
inline __aicore__ void CastWithOneTransfer(const AscendC::LocalTensor<OutT> &dst, const AscendC::LocalTensor<InT> &src,
                                           const uint32_t size, LocalTensor<uint8_t> &tmp_buf) {
  uint32_t buf_max_calc_size = tmp_buf.GetSize() * sizeof(uint8_t) / sizeof(InterT);
  uint32_t calc_loop = size / buf_max_calc_size;
  LocalTensor<InterT> inter_buf = tmp_buf.ReinterpretCast<InterT>();
  uint32_t offset = 0;
  for (uint32_t i = 0; i < calc_loop; i++) {
    AscendC::Cast(inter_buf, src[offset], GetRoundMode<InT, InterT>(), buf_max_calc_size);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(dst[offset], inter_buf, GetRoundMode<InterT, OutT>(), buf_max_calc_size);
    AscendC::PipeBarrier<PIPE_V>();
    offset += buf_max_calc_size;
  }
  uint32_t tail_calc_cnt = size - offset;
  if (tail_calc_cnt != 0) {
    AscendC::Cast(inter_buf, src[offset], GetRoundMode<InT, InterT>(), tail_calc_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(dst[offset], inter_buf, GetRoundMode<InterT, OutT>(), tail_calc_cnt);
  }
}

template <typename InT, typename OutT, typename FirstInterT, typename SecondInterT>
inline __aicore__ void CastWithTwoTransfer(const AscendC::LocalTensor<OutT> &dst, const AscendC::LocalTensor<InT> &src,
                                           const uint32_t size, LocalTensor<uint8_t> &tmp_buf) {
  uint32_t buf_max_calc_size = tmp_buf.GetSize() * sizeof(uint8_t) / sizeof(FirstInterT) / 2;
  uint32_t calc_loop = size / buf_max_calc_size;
  LocalTensor<FirstInterT> inter_buf_1 = tmp_buf.ReinterpretCast<FirstInterT>();
  LocalTensor<SecondInterT> inter_buf_2 = inter_buf_1.template ReinterpretCast<SecondInterT>();
  uint32_t offset = 0;
  for (uint32_t i = 0; i < calc_loop; i++) {
    AscendC::Cast(inter_buf_1, src[offset], GetRoundMode<InT, FirstInterT>(), buf_max_calc_size);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(inter_buf_2, inter_buf_1, GetRoundMode<FirstInterT, SecondInterT>(), buf_max_calc_size);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(dst[offset], inter_buf_2, GetRoundMode<SecondInterT, OutT>(), buf_max_calc_size);
    AscendC::PipeBarrier<PIPE_V>();
    offset += buf_max_calc_size;
  }
  uint32_t tail_calc_cnt = size - offset;
  if (tail_calc_cnt != 0) {
    AscendC::Cast(inter_buf_1, src[offset], GetRoundMode<InT, FirstInterT>(), tail_calc_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(inter_buf_2, inter_buf_1, GetRoundMode<FirstInterT, SecondInterT>(), tail_calc_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(dst[offset], inter_buf_2, GetRoundMode<SecondInterT, OutT>(), tail_calc_cnt);
  }
}

template <typename InT, typename OutT>
inline __aicore__ void CastWithOr(const AscendC::LocalTensor<OutT> &dst, const AscendC::LocalTensor<InT> &src,
                                  const uint32_t size, LocalTensor<uint8_t> &tmp_buf) {
  LocalTensor<int16_t> src_tmp = src.template ReinterpretCast<int16_t>();
  LocalTensor<int16_t> dst_tmp = dst.template ReinterpretCast<int16_t>();
  uint32_t or_count = size * sizeof(InT) / sizeof(int16_t);
  if constexpr (sizeof(InT) == sizeof(uint8_t) && sizeof(OutT) == sizeof(uint8_t)) {
    or_count = (size + 1U) / 2U;
  }
  AscendC::Or(dst_tmp, src_tmp, src_tmp, or_count);
}

template <typename InT, typename OutT>
inline __aicore__ void CastExtend(const AscendC::LocalTensor<OutT> &dst, const AscendC::LocalTensor<InT> &src,
                                  LocalTensor<uint8_t> &tmp_buf, const uint32_t size) {
  if constexpr ((AscendC::SupportType<InT, uint32_t, int32_t>() && AscendC::SupportType<OutT, uint32_t, int32_t>()) ||
                (AscendC::SupportType<InT, uint16_t, int16_t>() && AscendC::SupportType<OutT, uint16_t, int16_t>()) ||
                (AscendC::SupportType<InT, uint8_t, int8_t>() && AscendC::SupportType<OutT, uint8_t, int8_t>()) ||
                (AscendC::SupportType<InT, uint64_t, int64_t>() && AscendC::SupportType<OutT, uint64_t, int64_t>())) {
    CastWithOr(dst, src, size, tmp_buf);
  } else if constexpr (AscendC::IsSameType<InT, uint8_t>::value && !AscendC::IsSameType<OutT, half>::value) {
    // if input type is uint8_t, first cast it to half, then cast from half to dst type
    CastWithOneTransfer<InT, OutT, half>(dst, src, size, tmp_buf);
  } else if constexpr (AscendC::IsSameType<InT, int64_t>::value && !AscendC::SupportType<OutT, float, int32_t>()) {
    if constexpr (AscendC::IsSameType<OutT, uint8_t>::value) {
      CastWithTwoTransfer<InT, OutT, float, half>(dst, src, size, tmp_buf);
    } else if (AscendC::IsSameType<OutT, half>::value) {
      CastWithOneTransfer<InT, OutT, float>(dst, src, size, tmp_buf);
    }
  } else if constexpr (AscendC::IsSameType<InT, half>::value && AscendC::IsSameType<OutT, int64_t>::value) {
    // if input type is half, and output type is int64, first cast it to float, then cast from float to int64
    CastWithOneTransfer<InT, OutT, float>(dst, src, size, tmp_buf);
  } else {
    AscendC::Cast(dst, src, GetRoundMode<InT, OutT>(), size);
  }
}

template <typename InT, typename OutT>
inline __aicore__ void CastExtendWithMaskMode(const AscendC::LocalTensor<OutT> &dst,
                                              const AscendC::LocalTensor<InT> &src, const uint32_t repeat_times,
                                              const uint32_t input_last_dim_stride,
                                              const uint32_t output_last_dim_stride, const uint32_t last_dim) {
  uint32_t repeat_throw_for_extent = repeat_times / MAX_REPEAT_TIMES;
  uint32_t repeat_reminder = repeat_times - repeat_throw_for_extent * MAX_REPEAT_TIMES;
  uint16_t dst_block_stride = 1;
  uint16_t src_block_stride = 1;
  uint32_t dst_repeat_stride_in_blocks = output_last_dim_stride / (ONE_BLK_SIZE / sizeof(OutT));
  uint32_t src_repeat_stride_in_blocks = input_last_dim_stride / (ONE_BLK_SIZE / sizeof(InT));
  if (dst_repeat_stride_in_blocks > MAX_REPEAT_TIMES || src_repeat_stride_in_blocks > MAX_REPEAT_TIMES) {
    for (uint32_t i = 0; i < repeat_times; ++i) {
      AscendC::Cast(dst[i * output_last_dim_stride], src[i * input_last_dim_stride], GetRoundMode<InT, OutT>(),
                    last_dim);
    }
    return;
  }
  uint8_t dst_repeat_stride = static_cast<uint8_t>(dst_repeat_stride_in_blocks);
  uint8_t src_repeat_stride = static_cast<uint8_t>(src_repeat_stride_in_blocks);
  AscendC::SetMaskNorm();
  if constexpr (sizeof(InT) > sizeof(OutT)) {
    AscendC::SetVectorMask<InT, MaskMode::NORMAL>(last_dim);
  } else {
    AscendC::SetVectorMask<OutT, MaskMode::NORMAL>(last_dim);
  }
  auto dst_scalar = MAX_REPEAT_TIMES * output_last_dim_stride;
  auto src_scalar = MAX_REPEAT_TIMES * input_last_dim_stride;
  uint32_t dst_offset = 0;
  uint32_t src_offset = 0;
  for (uint32_t inner_for = 0; inner_for < repeat_throw_for_extent; inner_for++) {
    AscendC::Cast<OutT, InT, false>(dst[dst_offset], src[src_offset], GetRoundMode<InT, OutT>(), last_dim,
                                    MAX_REPEAT_TIMES,
                                    {dst_block_stride, src_block_stride, dst_repeat_stride, src_repeat_stride});
    dst_offset += dst_scalar;
    src_offset += src_scalar;
  }
  if (repeat_reminder != 0) {
    AscendC::Cast<OutT, InT, false>(dst[dst_offset], src[src_offset], GetRoundMode<InT, OutT>(), last_dim,
                                    repeat_reminder,
                                    {dst_block_stride, src_block_stride, dst_repeat_stride, src_repeat_stride});
  }
}

template <typename InT, typename OutT>
inline __aicore__ void CastExtendWithMaskMode(const AscendC::LocalTensor<OutT> &dst,
                                              const AscendC::LocalTensor<InT> &src, const uint32_t first_dim,
                                              const uint32_t last_dim, const uint32_t input_last_dim_stride,
                                              const uint32_t output_last_dim_stride, const uint32_t dtype_size,
                                              LocalTensor<uint8_t> &tmp_buf) {
  if (input_last_dim_stride == output_last_dim_stride) {
    AscendC::Cast(dst, src, GetRoundMode<InT, OutT>(), output_last_dim_stride * first_dim);
    return;
  }
  uint32_t elem_in_one_repeat = ONE_REPEAT_BYTE_SIZE / dtype_size;
  uint32_t repeat_times = first_dim;
  if (last_dim <= elem_in_one_repeat) {
    // 尾轴小于一个repeat的数据量，则将倒数第二根轴作为repeat, 其余轴作为for循环外抛
    CastExtendWithMaskMode<InT, OutT>(dst, src, repeat_times, input_last_dim_stride, output_last_dim_stride, last_dim);
  } else {
    uint32_t element_extent = last_dim / elem_in_one_repeat;
    uint32_t element_reminder = last_dim - element_extent * elem_in_one_repeat;
    if (element_extent <= repeat_times) {
      // 尾轴大于一个repeat的数据量，并且尾轴切分后的值小于等于repeat层，则将尾轴切分的系数外抛
      for (uint32_t outer_for = 0; outer_for < element_extent; outer_for++) {
        CastExtendWithMaskMode<InT, OutT>(dst[outer_for * elem_in_one_repeat], src[outer_for * elem_in_one_repeat],
                                          repeat_times, input_last_dim_stride, output_last_dim_stride,
                                          elem_in_one_repeat);
      }
      if (element_reminder != 0) {
        CastExtendWithMaskMode<InT, OutT>(dst[element_extent * elem_in_one_repeat],
                                          src[element_extent * elem_in_one_repeat], repeat_times, input_last_dim_stride,
                                          output_last_dim_stride, element_reminder);
      }
    } else {
      for (uint32_t outer_for = 0; outer_for < repeat_times; outer_for++) {
        AscendC::Cast(dst[outer_for * output_last_dim_stride], src[outer_for * input_last_dim_stride],
                      GetRoundMode<InT, OutT>(), last_dim);
      }
    }
  }
}

template <typename InT, typename OutT>
inline __aicore__ void CastExtendWithOneTransferWithMaskMode(const AscendC::LocalTensor<OutT> &dst,
                                                             const AscendC::LocalTensor<InT> &src,
                                                             const uint32_t first_dim, const uint32_t last_dim,
                                                             const uint32_t input_last_dim_stride,
                                                             const uint32_t output_last_dim_stride,
                                                             const uint32_t dtype_size, LocalTensor<uint8_t> &tmp_buf) {
  if constexpr (((AscendC::IsSameType<InT, uint8_t>::value) &&
                 (AscendC::SupportType<OutT, float, int32_t, int16_t, int8_t, int4b_t>()))) {  // u8 -> !(half)
    uint32_t max_dtype_size_between_src_and_mid = 2;
    uint32_t max_dtype_size_between_mid_and_dst = 0;
    auto elem_in_one_block =
        ConvertToUint32(Rational(32, 2));  // 一个block占的元素个数（中间转换类型为half，占2个字节）
    auto blocks_for_last_dim_elems = Ceiling(last_dim * Rational(2, 32));  // last_dim个元素所占的Block个数
    uint32_t mid_last_dim_stride = elem_in_one_block * blocks_for_last_dim_elems;
    if constexpr (AscendC::SupportType<OutT, float, int32_t>()) {
      max_dtype_size_between_mid_and_dst = 4;
    }
    if constexpr (AscendC::IsSameType<OutT, int16_t>::value) {
      max_dtype_size_between_mid_and_dst = 2;
    }
    if constexpr (AscendC::IsSameType<OutT, int8_t>::value) {
      max_dtype_size_between_mid_and_dst = 2;
    }
    if constexpr (AscendC::IsSameType<OutT, int4b_t>::value) {
      max_dtype_size_between_mid_and_dst = 2;
    }
    auto mid_ub = tmp_buf[0].template ReinterpretCast<half>();
    CastExtendWithMaskMode<InT, half>(mid_ub, src, first_dim, last_dim, input_last_dim_stride, mid_last_dim_stride,
                                      max_dtype_size_between_src_and_mid, tmp_buf);
    AscendC::PipeBarrier<PIPE_V>();
    CastExtendWithMaskMode<half, OutT>(dst, mid_ub, first_dim, last_dim, mid_last_dim_stride, output_last_dim_stride,
                                       max_dtype_size_between_mid_and_dst, tmp_buf);
  }
  if constexpr ((AscendC::IsSameType<InT, int64_t>::value && AscendC::IsSameType<OutT, half>::value) ||
                (AscendC::IsSameType<InT, half>::value && AscendC::IsSameType<OutT, int64_t>::value)) {
    uint32_t max_dtype_size_between_src_and_mid = 0;
    uint32_t max_dtype_size_between_mid_and_dst = 0;
    auto elem_in_one_block =
        ConvertToUint32(Rational(32, 4));  // 一个block占的元素个数（中间转换类型为float，占4个字节）
    auto blocks_for_last_dim_elems = Ceiling(last_dim * Rational(4, 32));  // last_dim个元素所占的Block个数
    uint32_t mid_last_dim_stride = elem_in_one_block * blocks_for_last_dim_elems;
    auto mid_ub = tmp_buf[0].template ReinterpretCast<float>();
    if constexpr (AscendC::IsSameType<InT, int64_t>::value) {
      uint32_t max_dtype_size_between_src_and_mid = 8;
      uint32_t max_dtype_size_between_mid_and_dst = 4;
    } else {
      uint32_t max_dtype_size_between_src_and_mid = 4;
      uint32_t max_dtype_size_between_mid_and_dst = 8;
    }
    CastExtendWithMaskMode<InT, float>(mid_ub, src, first_dim, last_dim, input_last_dim_stride, mid_last_dim_stride,
                                       max_dtype_size_between_src_and_mid, tmp_buf);
    AscendC::PipeBarrier<PIPE_V>();
    CastExtendWithMaskMode<float, OutT>(dst, mid_ub, first_dim, last_dim, mid_last_dim_stride, output_last_dim_stride,
                                        max_dtype_size_between_mid_and_dst, tmp_buf);
  }
}

template <typename InT, typename OutT>
inline __aicore__ void CastExtend(const AscendC::LocalTensor<OutT> &dst, const AscendC::LocalTensor<InT> &src,
                                  LocalTensor<uint8_t> &tmp_buf, const uint32_t first_dim, const uint32_t last_dim,
                                  const uint32_t input_last_dim_stride, const uint32_t output_last_dim_stride,
                                  const uint32_t dtype_size) {
  if constexpr (((AscendC::IsSameType<InT, uint8_t>::value) && (AscendC::SupportType<OutT, half>())) ||
                ((AscendC::IsSameType<InT, int64_t>::value) && (AscendC::SupportType<OutT, float, int32_t>())) ||
                ((AscendC::IsSameType<InT, half>::value) &&
                 (AscendC::SupportType<OutT, float, int32_t, int16_t, int8_t, uint8_t, int4b_t>())) ||
                ((AscendC::IsSameType<InT, float>::value) &&
                 (AscendC::SupportType<OutT, float, half, int64_t, int32_t, int16_t, bfloat16_t>())) ||
                ((AscendC::IsSameType<InT, int4b_t>::value) && (AscendC::SupportType<OutT, half>())) ||
                ((AscendC::IsSameType<InT, int16_t>::value) && (AscendC::SupportType<OutT, half, float>())) ||
                ((AscendC::IsSameType<InT, int32_t>::value) &&
                 (AscendC::SupportType<OutT, float, int64_t, int16_t, half>())) ||
                ((AscendC::IsSameType<InT, bfloat16_t>::value) && (AscendC::SupportType<OutT, float, int32_t>()))) {
    CastExtendWithMaskMode<InT, OutT>(dst, src, first_dim, last_dim, input_last_dim_stride, output_last_dim_stride,
                                      dtype_size, tmp_buf);  // 直接使用AscendC::Cast实现
  } else if constexpr (((AscendC::IsSameType<InT, uint8_t>::value) &&
                        (AscendC::SupportType<OutT, float, int32_t, int16_t, int8_t, int4b_t>())) ||
                       ((AscendC::IsSameType<InT, int64_t>::value) && (AscendC::SupportType<OutT, half>())) ||
                       ((AscendC::IsSameType<InT, half>::value) && (AscendC::SupportType<OutT, int64_t>()))) {
    CastExtendWithOneTransferWithMaskMode<InT, OutT>(dst, src, first_dim, last_dim, input_last_dim_stride,
                                                     output_last_dim_stride, dtype_size, tmp_buf);  // 需要一次中间转换
  } else {
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Current conversion does not support mask mode"); });
  }
}

#endif  // __ASCENDC_API_CAST_H__
