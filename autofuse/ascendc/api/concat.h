/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_CONCAT_H__
#define __ASCENDC_API_CONCAT_H__

template <typename T, int dim_size>
struct ConcatParams {
  uint32_t shape[dim_size];
  uint32_t stride[dim_size];
  LocalTensor<T> *tensor;
};

struct RowParam {
  uint32_t row_loop;
  uint32_t rows_cur_loop;
};

struct TransposeParams {
  struct RowParam row;
  uint32_t column_offset;  // 可以理解为按列拼接，当前处理到了哪一列
  uint32_t column_loop;
  uint32_t columns_cur_loop;
};

constexpr uint32_t kMergedDimNum = 2U;
constexpr uint32_t kDataBlockSize = 32U;
constexpr uint32_t kPerLoopRowSize = 16U;
constexpr uint32_t kOneByte = 1U;
constexpr uint32_t kTwoBytes = 2U;
constexpr uint32_t kFourBytes = 4U;
constexpr uint32_t kEightBytes = 8U;
constexpr uint32_t kAddrListSize = 16U;
constexpr uint32_t kFourBytesDataNumPerBlock = 8U;
constexpr uint32_t kScalarTow = 2U;

template <typename T, int dim_size>
inline __aicore__ void ConcatExtendColumnAligned(const ConcatParams<T, dim_size> &dst,
                                                 const ConcatParams<T, dim_size> &src,
                                                 const uint32_t &curr_column_cnt) {
  uint16_t row_cnt = src.shape[0];
  uint16_t column_cnt = src.stride[0];  // padding对齐之后的列数
  uint16_t align_size = (uint16_t)(kDataBlockSize / sizeof(T));
  DataCopyParams repeat_param = {row_cnt, (uint16_t)(column_cnt / align_size), 0,
                                 (uint16_t)((dst.stride[0] - column_cnt) / align_size)};
  DataCopy((*dst.tensor)[curr_column_cnt], (*src.tensor)[0], repeat_param);
  return;
}

template <typename T>
inline __aicore__ constexpr uint32_t CalcRepeatTimes(const uint32_t colums) {
  return (colums * sizeof(T) + kDataBlockSize - 1) / kDataBlockSize;
}

/* 获取第二次转置前的总列数（也是第一次转置后的总列数）,
 * 4字节和2字节都是16行转过来的，因此第一次转置之后的列宽是16,
 * 1字节虽然也是16行转过来的，但是只放了dst的低半部，高半部空闲，因此列宽是32；
 * 单位： sizeof(T)
 */
template <typename T>
inline __aicore__ constexpr uint32_t GetTotalColumns() {
  constexpr uint32_t kOneByteTotalColums = 32;
  constexpr uint32_t kTwoOrFourBytesTotalColums = 16;

  if (sizeof(T) == kOneByte) {
    return kOneByteTotalColums;
  } else {
    return kTwoOrFourBytesTotalColums;
  }
}

/* 第二次转置回来的列对齐数，
 * 4字节dst0和dst1是连续的，在同一行，是16个数对齐
 * 2字节dst0独占一行，是16个数对齐；
 * 1字节dst0分高半部和低半部独占一行，是32个数对齐
 * 单位： sizeof(T)
 */
template <typename T>
inline __aicore__ constexpr uint32_t GetColumnAlign() {
  constexpr uint32_t kOneByteAlign = 32U;
  constexpr uint32_t kTwoOrFourBytesAlign = 16U;

  if (sizeof(T) == kOneByte) {
    return kOneByteAlign;
  } else {
    return kTwoOrFourBytesAlign;
  }
}

template <typename T>
inline __aicore__ uint64_t CalcDstLocalList(LocalTensor<T> &tmp_buf, uint32_t dst_offset, const uint32_t dst_width,
                                            int index) {
  if (sizeof(T) == kFourBytes) {
    if (index % 2 == 0) {
      return (uint64_t)(tmp_buf[dst_offset + index / 2 * dst_width].GetPhyAddr());
    } else {
      return (uint64_t)(tmp_buf[dst_offset + index / 2 * dst_width + kFourBytesDataNumPerBlock].GetPhyAddr());
    }
  }
  return (uint64_t)(tmp_buf[dst_offset + index * dst_width].GetPhyAddr());
}

/* 4字节一次转16*8，2字节一次转16*16，因此都是16个datablock，
 * 1字节转一个半部跳一个半部，因此dst间隔32*32,因此是32个datablock，
 * 单位是datablock
 */
template <typename T>
inline __aicore__ constexpr uint32_t GetFirstTransDstRepStride() {
  constexpr uint32_t kOneByteRepStride = 32U;
  constexpr uint32_t kTwoOrFourBytesRepStride = 16U;

  if (sizeof(T) == kOneByte) {
    return kOneByteRepStride;
  } else {
    return kTwoOrFourBytesRepStride;
  }
}

template <typename T>
inline __aicore__ constexpr uint32_t GetSecondTransSrcRepStride() {
  constexpr uint32_t kOneByteRepStride = 32U;
  constexpr uint32_t kTwoBytesRepStride = 16U;
  constexpr uint32_t kFourBytesRepStride = 32U;

  if (sizeof(T) == kOneByte) {
    return kOneByteRepStride;
  } else if (sizeof(T) == kTwoBytes) {
    return kTwoBytesRepStride;
  } else {
    return kFourBytesRepStride;
  }
}

template <typename T>
inline __aicore__ constexpr uint32_t GetSecondTransDstRepStride() {
  constexpr uint32_t kOneByteRepStride = 1U;
  constexpr uint32_t kTwoBytesRepStride = 1U;
  constexpr uint32_t kFourBytesRepStride = 2U;

  if (sizeof(T) == kOneByte) {
    return kOneByteRepStride;
  } else if (sizeof(T) == kTwoBytes) {
    return kTwoBytesRepStride;
  } else {
    return kFourBytesRepStride;
  }
}

template <typename T>
inline __aicore__ void FirstTransposeMatrix(LocalTensor<T> &tmp_buf1,   // 用于拼接转置的临时buff
                                            const LocalTensor<T> &src,  // 拼接的输入
                                            const struct TransposeParams &trans_para,
                                            uint32_t cur_row_cnt,  // 临时buff拼接的起始列索引
                                            uint32_t stride) {  // padding对齐之后，一行的数据个数，单位sizeof(T)
  AscendC::TransDataTo5HDParams transDataParams;
  transDataParams.srcHighHalf = false;
  transDataParams.dstHighHalf = false;
  const uint32_t repeat_times = CalcRepeatTimes<T>(trans_para.columns_cur_loop);
  transDataParams.repeatTimes = repeat_times;

  if (repeat_times == 1U) {
    transDataParams.srcRepStride = 0U;
    transDataParams.dstRepStride = 0U;
  } else {
    transDataParams.srcRepStride = 1U;  // 单位是datablock
    transDataParams.dstRepStride = GetFirstTransDstRepStride<T>();
  }

  // 单位datablock
  uint64_t dst_local_list[kAddrListSize];
  uint64_t src_local_list[kAddrListSize];

  // 原始输入padding对齐之后，一行的数据个数，单位sizeof(T)
  const uint32_t src_width = stride;
  constexpr uint32_t dst_width = GetTotalColumns<T>();

  const uint32_t src_offset = trans_para.row.row_loop * (kPerLoopRowSize * src_width) + trans_para.column_offset;
  uint32_t dst_offset = cur_row_cnt * dst_width;

  for (int i = 0; i < kAddrListSize; i++) {
    const uint32_t src_id =
        (i < trans_para.row.rows_cur_loop) ? i : trans_para.row.rows_cur_loop - 1;  // first轴尾块场景
    src_local_list[i] = (uint64_t)(src[src_offset + src_id * src_width].GetPhyAddr());
    dst_local_list[i] = CalcDstLocalList(tmp_buf1, dst_offset, dst_width, i);
  }
  AscendC::TransDataTo5HD<T>(dst_local_list, src_local_list, transDataParams);

  // 单字节分两次转，第二次转src的高半部
  if (sizeof(T) == kOneByte) {
    transDataParams.srcHighHalf = true;
    dst_offset += kPerLoopRowSize * GetTotalColumns<T>();
    for (int i = 0; i < kAddrListSize; i++) {
      dst_local_list[i] = (uint64_t)(tmp_buf1[dst_offset + i * dst_width].GetPhyAddr());
    }
    AscendC::TransDataTo5HD<T>(dst_local_list, src_local_list, transDataParams);
  }
}

template <typename T>
inline __aicore__ void SecondTransposeMatrix(
    LocalTensor<T> &tmp_buf2,        // 用于存放二次转置结果的临时buff
    const LocalTensor<T> &tmp_buf1,  // 存放了两个输入第一次转置拼接结果的临时buff
    uint32_t total_columns,          // 矩阵的列数
    uint32_t total_rows, uint32_t src_offset, uint32_t dst_offset, uint32_t dst_width, bool is_second = false) {
  AscendC::TransDataTo5HDParams transDataParams;
  transDataParams.srcHighHalf = false;
  transDataParams.dstHighHalf = false;
  // 单字节场景，低半部和高半部交替着放
  if (sizeof(T) == kOneByte) {
    if (is_second) {
      transDataParams.dstHighHalf = true;
      transDataParams.repeatTimes = (total_rows + 15) >> 5;  // 计算2次转置次数的时候先减掉16行
    } else {
      transDataParams.repeatTimes = (total_rows + 31) >> 5;  // 1次转置repeat是行向32对齐之后，取32的倍数
    }
  } else {
    transDataParams.repeatTimes = (total_rows + 15) >> 4;  // 行向16对齐之后，取16的倍数
  }

  if (transDataParams.repeatTimes == 1) {
    transDataParams.srcRepStride = 0;
    transDataParams.dstRepStride = 0;
  } else {
    transDataParams.srcRepStride = GetSecondTransSrcRepStride<T>();
    transDataParams.dstRepStride = GetSecondTransDstRepStride<T>();
  }

  uint64_t dst_local_list[kAddrListSize];
  uint64_t src_local_list[kAddrListSize];
  constexpr uint32_t src_width = GetTotalColumns<T>();

  for (int i = 0; i < kAddrListSize; i++) {
    src_local_list[i] = (uint64_t)(tmp_buf1[src_offset + i * src_width].GetPhyAddr());
    dst_local_list[i] = CalcDstLocalList(tmp_buf2, dst_offset, dst_width, i);
  }

  AscendC::TransDataTo5HD<T>(dst_local_list, src_local_list, transDataParams);
}

// 和GetColumAlign相比，4字节存在差异，对4字节来说，转置回来的列数虽然是16对齐的，但只要按8对齐拷贝就可以了
template <typename T>
inline __aicore__ constexpr uint32_t GetColumnCopiedAlign() {
  constexpr uint32_t kOneByteAlign = 32U;
  constexpr uint32_t kTwoBytesAlign = 16U;
  constexpr uint32_t kFourBytesAlign = 8U;

  if (sizeof(T) == kOneByte) {
    return kOneByteAlign;
  } else if (sizeof(T) == kTwoBytes) {
    return kTwoBytesAlign;
  } else {
    return kFourBytesAlign;
  }
}

template <typename T>
inline __aicore__ void SecondTranspose(const LocalTensor<T> &tmp_buf1,  // 存放了两个输入第一次转置拼接结果的临时buff
                                       LocalTensor<T> &tmp_buf2, uint32_t total_rows, const uint32_t column_cnt) {
  // 2.第二次转置回来,转置策略为竖着取横着放，尽量增大repeat
  constexpr uint32_t align = GetColumnAlign<T>();
  uint32_t dst_width = (total_rows + align - 1) / align * align;
  SecondTransposeMatrix<T>(tmp_buf2, tmp_buf1, column_cnt, total_rows, 0, 0, dst_width);
  if (sizeof(T) == kFourBytes) {
    /* src_offset是8: 2次转置相比于1次转置，src向右移了一个datablock因此是8，
    dst_width是dst_width << 3：偏移是8*dst_width */
    SecondTransposeMatrix<T>(tmp_buf2, tmp_buf1, column_cnt, total_rows, 8, dst_width << 3, dst_width, true);
  } else if (sizeof(T) == kOneByte) {
    // src_offset是16*32: 2次转置从第2个16行开始，一行是32个数，因此偏移是16*32
    SecondTransposeMatrix<T>(tmp_buf2, tmp_buf1, column_cnt, total_rows, 16 * 32, 0, dst_width, true);
  }
}

template <typename T, int dim_size>
inline __aicore__ void DataCopyToDst(const ConcatParams<T, dim_size> &dst, const LocalTensor<T> &tmp_buf2,
                                     uint32_t row_cnt, const struct RowParam &row, uint32_t column_align_cnt) {
  constexpr uint32_t column_align = GetColumnAlign<T>();
  // 第二次转置之后的列数
  uint32_t final_column_cnt = ((row_cnt + column_align - 1) / column_align) * column_align;
  constexpr uint32_t copied_align = GetColumnCopiedAlign<T>();
  // 需要拷贝的列数
  uint32_t column_copied = ((row_cnt + copied_align - 1) / copied_align) * copied_align;
  // dst开始拷贝的位置
  uint32_t dst_start_pos = kPerLoopRowSize * row.row_loop * dst.stride[0] + column_align_cnt;
  uint16_t align_size = (uint16_t)(kDataBlockSize / sizeof(T));
  DataCopyParams repeat_param = {(uint16_t)row.rows_cur_loop, (uint16_t)(column_copied / align_size),
                                 (uint16_t)((final_column_cnt - column_copied) / align_size),
                                 (uint16_t)((dst.stride[0] - column_copied) / align_size)};

  DataCopy((*dst.tensor)[dst_start_pos], tmp_buf2[0], repeat_param);
}

template <typename T, int dim_size>
inline __aicore__ void Concat16MultipleColumns(const ConcatParams<T, dim_size> &dst,
                                               const ConcatParams<T, dim_size> &src, const uint32_t curr_column_cnt,
                                               LocalTensor<T> &tmp_buf1, const struct TransposeParams &trans_para,
                                               uint32_t max_columns) {
  // 1.第一次转置拼接
  uint32_t column_align_cnt = (curr_column_cnt * sizeof(T) / kDataBlockSize) * (kDataBlockSize / sizeof(T));
  uint32_t col_not_align_cnt = curr_column_cnt - column_align_cnt;

  // dst上非32B对齐的部分和后续的输入做转置拼接
  TransposeParams trans_param = trans_para;
  trans_param.column_offset = column_align_cnt;
  trans_param.columns_cur_loop = col_not_align_cnt;
  AscendC::PipeBarrier<PIPE_V>();
  FirstTransposeMatrix(tmp_buf1, *dst.tensor, trans_param, 0, dst.stride[0]);
  trans_param.column_offset = trans_para.column_loop * max_columns;
  trans_param.columns_cur_loop = trans_para.columns_cur_loop;
  FirstTransposeMatrix(tmp_buf1, *src.tensor, trans_param, col_not_align_cnt, src.stride[0]);
  // 2.第二次转置回来,转置策略为竖着取横着放，尽量增大repeat
  uint32_t row_cnt = col_not_align_cnt + trans_para.columns_cur_loop;  // 第二次转置前的总行数
  constexpr uint32_t column_cnt = GetTotalColumns<T>();                // 第二次转置前的总列数
  LocalTensor<T> tmp_buf2 = tmp_buf1[column_cnt * row_cnt];
  AscendC::PipeBarrier<PIPE_V>();
  SecondTranspose<T>(tmp_buf1, tmp_buf2, row_cnt, column_cnt);
  AscendC::PipeBarrier<PIPE_V>();

  // 3.拼接回dst
  struct RowParam row;
  row.row_loop = trans_para.row.row_loop;
  row.rows_cur_loop = trans_para.row.rows_cur_loop;
  DataCopyToDst<T, dim_size>(dst, tmp_buf2, row_cnt, row, column_align_cnt);
}

template <typename T, int dim_size>
inline __aicore__ void Concat16RowsNotAligned(const ConcatParams<T, dim_size> &dst,
                                              const ConcatParams<T, dim_size> &src, const uint32_t &curr_column_cnt,
                                              LocalTensor<T> &tmp_buf, const struct RowParam &row,
                                              uint32_t max_columns) {
  uint32_t curr_column_offset = curr_column_cnt;
  struct TransposeParams trans_para;

  uint32_t max_column_loop = src.shape[1] / max_columns;
  trans_para.row.row_loop = row.row_loop;
  trans_para.row.rows_cur_loop = row.rows_cur_loop;
  // 列太大时，tmpbuf放不下，需要拆分列
  for (uint32_t column_loop = 0; column_loop < max_column_loop; column_loop++) {
    trans_para.column_loop = column_loop;
    trans_para.columns_cur_loop = max_columns;
    Concat16MultipleColumns(dst, src, curr_column_offset, tmp_buf, trans_para, max_columns);
    curr_column_offset += max_columns;
  }

  uint32_t remainder = src.shape[1] % max_columns;
  if (remainder != 0) {
    trans_para.column_loop = max_column_loop;
    trans_para.columns_cur_loop = remainder;
    Concat16MultipleColumns(dst, src, curr_column_offset, tmp_buf, trans_para, max_columns);
    curr_column_offset += max_columns;
  }
}

template <typename T, int dim_size>
inline __aicore__ void ConcatExtendColumnNotAligned(const ConcatParams<T, dim_size> &dst,
                                                    const ConcatParams<T, dim_size> &src,
                                                    const uint32_t &curr_column_cnt, uint32_t max_column,
                                                    LocalTensor<T> &tmp_buf) {
  uint32_t row_cnt = src.shape[0];
  struct RowParam row;
  uint32_t max_row_loop = row_cnt / kPerLoopRowSize;
  // 一次处理16行
  for (int i = 0; i < max_row_loop; i++) {
    row.row_loop = i;
    row.rows_cur_loop = kPerLoopRowSize;
    Concat16RowsNotAligned(dst, src, curr_column_cnt, tmp_buf, row, max_column);
  }

  // 末尾不是16行对齐
  uint32_t remainder = row_cnt % kPerLoopRowSize;
  if (remainder != 0) {
    row.row_loop = max_row_loop;
    row.rows_cur_loop = remainder;
    Concat16RowsNotAligned(dst, src, curr_column_cnt, tmp_buf, row, max_column);
  }

  return;
}

template <typename T, int dim_size, int input_num>
inline __aicore__ void MultipleInputsConcat16Rows(const ConcatParams<T, dim_size> &dst,
                                                  const ConcatParams<T, kMergedDimNum> srcs[input_num],
                                                  const uint32_t &curr_column_cnt, uint32_t &sub_column_cnt,
                                                  uint32_t &sub_input_cnt, uint32_t max_column, uint32_t curr_input,
                                                  const struct RowParam &row, LocalTensor<T> &tmp_buf1) {
  // dst上非32B对齐的部分转置
  uint32_t column_align_cnt = (curr_column_cnt * sizeof(T) / kDataBlockSize) * (kDataBlockSize / sizeof(T));
  uint32_t col_not_align_cnt = curr_column_cnt - column_align_cnt;
  TransposeParams trans_param;
  trans_param.row.row_loop = row.row_loop;
  trans_param.row.rows_cur_loop = row.rows_cur_loop;
  trans_param.column_offset = column_align_cnt;
  trans_param.columns_cur_loop = col_not_align_cnt;
  AscendC::PipeBarrier<PIPE_V>();
  FirstTransposeMatrix(tmp_buf1, *dst.tensor, trans_param, 0, dst.stride[0]);

  auto total_column_cnt = col_not_align_cnt;
  for (uint32_t idx = curr_input; idx < input_num; idx++) {
    if (sub_column_cnt + srcs[idx].shape[1] > max_column) {
      break;
    }

    trans_param.column_offset = 0;
    trans_param.columns_cur_loop = srcs[idx].shape[1];
    FirstTransposeMatrix(tmp_buf1, *srcs[idx].tensor, trans_param, total_column_cnt, srcs[idx].stride[0]);

    sub_input_cnt++;
    sub_column_cnt += srcs[idx].shape[1];
    total_column_cnt += srcs[idx].shape[1];
  }
  AscendC::PipeBarrier<PIPE_V>();
  // 2.第二次转置回来,转置策略为竖着取横着放，尽量增大repeat
  constexpr uint32_t column_cnt = GetTotalColumns<T>();
  uint32_t row_cnt = total_column_cnt;
  LocalTensor<T> tmp_buf2 = tmp_buf1[column_cnt * row_cnt];
  SecondTranspose<T>(tmp_buf1, tmp_buf2, row_cnt, column_cnt);
  AscendC::PipeBarrier<PIPE_V>();

  // 3.拼接回dst
  DataCopyToDst<T, dim_size>(dst, tmp_buf2, row_cnt, row, column_align_cnt);
}

template <typename T, int dim_size>
inline __aicore__ void CopyUnAlignDstToTmp(const LocalTensor<T> &tmp_buf1, const ConcatParams<T, dim_size> &dst,
                                           const struct RowParam &row, const uint32_t column_align_cnt,
                                           const uint32_t tmp_stride0) {
  uint16_t align_size = GetColumnCopiedAlign<T>();
  // 处理DataCopyParams
  DataCopyParams repeat_param = {(uint16_t)row.rows_cur_loop, 1, (uint16_t)((dst.stride[0] - align_size) / align_size),
                                 (uint16_t)((tmp_stride0 - align_size) / align_size)};

  DataCopy(tmp_buf1[0], (*dst.tensor)[column_align_cnt + row.row_loop * kPerLoopRowSize * dst.stride[0]], repeat_param);
  return;
}

template <typename T, int dim_size>
inline __aicore__ void CopyPaddedSrcsToTmp(const LocalTensor<T> &tmp_buf1, const ConcatParams<T, dim_size> &src,
                                           const struct RowParam &row, const uint32_t dst_stride,
                                           const uint32_t curr_tmp_column) {
  uint16_t column_cnt = src.stride[0];  // padding对齐之后的列数
  uint16_t align_size = GetColumnCopiedAlign<T>();
  // 处理DataCopyParams
  DataCopyParams repeat_param = {(uint16_t)row.rows_cur_loop, (uint16_t)(column_cnt / align_size), 0,
                                 (uint16_t)((dst_stride - column_cnt) / align_size)};

  DataCopy(tmp_buf1[curr_tmp_column], (*src.tensor)[row.row_loop * kPerLoopRowSize * column_cnt], repeat_param);
  return;
}

template <typename T, int dim_size, int input_num>
inline __aicore__ void AlignTmpbuf(const LocalTensor<T> &tmp_buf1, const LocalTensor<T> &tmp_buf2,
                                   const ConcatParams<T, kMergedDimNum> srcs[input_num], uint32_t col_not_align_cnt,
                                   const uint32_t curr_input, const uint32_t input_cnt) {
  uint32_t row_not_align = 0;
  uint32_t dst_start_pos = 0;
  uint32_t column_cnt = GetTotalColumns<T>();

  // dst起始行数为dst上非对齐的部分
  uint32_t dst_row = col_not_align_cnt;
  uint32_t src_row = GetColumnCopiedAlign<T>();

  DataCopy(tmp_buf1[0], tmp_buf2[0], column_cnt * col_not_align_cnt);

  for (uint32_t i = curr_input; i < input_cnt; i++) {
    dst_start_pos = dst_row * column_cnt;
    DataCopy(tmp_buf1[dst_start_pos], tmp_buf2[src_row * column_cnt], column_cnt * srcs[i].shape[1]);
    // src行数加上当前src tensor的对齐stride
    src_row += srcs[i].stride[0];
    // dst行数加上当前src tensor的shape1
    dst_row += srcs[i].shape[1];
  }
  return;
}

template <typename T, int dim_size, int input_num>
inline __aicore__ void CopyThenTranspose16Rows(const ConcatParams<T, dim_size> &dst,
                                               const ConcatParams<T, kMergedDimNum> srcs[input_num],
                                               const uint32_t &curr_column_cnt, uint32_t &sub_column_cnt,
                                               uint32_t &sub_input_cnt, uint32_t max_column, uint32_t curr_input,
                                               const struct RowParam &row, LocalTensor<T> &tmp_buf1) {
  uint32_t curr_tmp_column = 0;
  uint32_t align = GetColumnCopiedAlign<T>();
  uint32_t tmp_stride0 = align;
  uint32_t max_input_loop = curr_input;
  for (uint32_t i = curr_input; i < input_num; i++) {
    if (tmp_stride0 + srcs[i].stride[0] > (max_column)) {
      break;
    }
    max_input_loop++;
    tmp_stride0 += srcs[i].stride[0];
  }

  uint32_t column_align_cnt = curr_column_cnt / align * align;
  uint32_t col_not_align_cnt = curr_column_cnt - column_align_cnt;

  // 由于该分支只会在dst上尾部column未对齐的时候进入，因此需要先处理dst上未对齐的部分
  // 将dst上未对齐的column拷贝到buf
  AscendC::PipeBarrier<PIPE_V>();
  CopyUnAlignDstToTmp<T, kMergedDimNum>(tmp_buf1, dst, row, column_align_cnt, tmp_stride0);
  AscendC::PipeBarrier<PIPE_V>();
  curr_tmp_column += align;

  // 将src tensor带着pad对齐拷贝到buf
  for (uint32_t idx = curr_input; idx < max_input_loop; idx++) {
    CopyPaddedSrcsToTmp<T, kMergedDimNum>(tmp_buf1, srcs[idx], row, tmp_stride0, curr_tmp_column);
    curr_tmp_column += srcs[idx].stride[0];

    sub_input_cnt++;
    sub_column_cnt += srcs[idx].shape[1];
  }

  LocalTensor<T> tmp_buf2 = tmp_buf1[row.rows_cur_loop * tmp_stride0];
  TransposeParams trans_param;
  trans_param.row.row_loop = 0;
  trans_param.row.rows_cur_loop = row.rows_cur_loop;
  trans_param.column_offset = 0;
  trans_param.columns_cur_loop = curr_tmp_column;

  // 一次性转置整块buf
  AscendC::PipeBarrier<PIPE_V>();
  FirstTransposeMatrix(tmp_buf2, tmp_buf1, trans_param, 0, curr_tmp_column);

  // 通过DataCopy将pad覆盖消除
  AscendC::PipeBarrier<PIPE_V>();
  AlignTmpbuf<T, kMergedDimNum, input_num>(tmp_buf1, tmp_buf2, srcs, col_not_align_cnt, curr_input,
                                           sub_input_cnt + curr_input);

  // 将处理好的buf转置回来
  uint32_t column_cnt = GetTotalColumns<T>();
  tmp_buf2 = tmp_buf1[(sub_column_cnt + col_not_align_cnt) * column_cnt];
  AscendC::PipeBarrier<PIPE_V>();
  SecondTranspose<T>(tmp_buf1, tmp_buf2, sub_column_cnt + col_not_align_cnt, column_cnt);

  // 拷贝到dst上
  AscendC::PipeBarrier<PIPE_V>();
  DataCopyToDst<T, dim_size>(dst, tmp_buf2, sub_column_cnt + col_not_align_cnt, row, column_align_cnt);
}

template <typename T, int dim_size, int input_num>
inline __aicore__ void MultipleInputsCopyThenTrans(const ConcatParams<T, dim_size> &dst,
                                                   const ConcatParams<T, kMergedDimNum> srcs[input_num],
                                                   const uint32_t &curr_column_cnt, LocalTensor<T> &tmp_buf1,
                                                   uint32_t max_column, uint32_t curr_input, uint32_t &sub_input_cnt,
                                                   uint32_t &sub_column_cnt) {
  // 一次处理16行
  struct RowParam row;
  uint32_t row_cnt = srcs[curr_input].shape[0];
  uint32_t max_row_loop = row_cnt / kPerLoopRowSize;

  for (uint32_t j = 0; j < max_row_loop; j++) {
    sub_input_cnt = 0;
    sub_column_cnt = 0;
    row.row_loop = j;
    row.rows_cur_loop = kPerLoopRowSize;
    CopyThenTranspose16Rows<T, dim_size, input_num>(dst, srcs, curr_column_cnt, sub_column_cnt, sub_input_cnt,
                                                    max_column, curr_input, row, tmp_buf1);
  }

  // 末尾不是16行对齐
  uint32_t remainder = row_cnt % kPerLoopRowSize;
  if (remainder != 0) {
    sub_input_cnt = 0;
    sub_column_cnt = 0;
    row.row_loop = max_row_loop;
    row.rows_cur_loop = remainder;
    CopyThenTranspose16Rows<T, dim_size, input_num>(dst, srcs, curr_column_cnt, sub_column_cnt, sub_input_cnt,
                                                    max_column, curr_input, row, tmp_buf1);
  }

  return;
}

template <typename T, int dim_size, int input_num>
inline __aicore__ void MultipleInputsMergeConcat(const ConcatParams<T, dim_size> &dst,
                                                 const ConcatParams<T, kMergedDimNum> srcs[input_num],
                                                 const uint32_t &curr_column_cnt, LocalTensor<T> &tmp_buf1,
                                                 uint32_t max_column, uint32_t curr_input, uint32_t &sub_input_cnt,
                                                 uint32_t &sub_column_cnt) {
  // 一次处理16行
  struct RowParam row;
  uint32_t row_cnt = srcs[curr_input].shape[0];
  uint32_t max_row_loop = row_cnt / kPerLoopRowSize;

  for (uint32_t j = 0; j < max_row_loop; j++) {
    sub_input_cnt = 0;
    sub_column_cnt = 0;
    row.row_loop = j;
    row.rows_cur_loop = kPerLoopRowSize;
    MultipleInputsConcat16Rows<T, dim_size, input_num>(dst, srcs, curr_column_cnt, sub_column_cnt, sub_input_cnt,
                                                       max_column, curr_input, row, tmp_buf1);
  }

  // 末尾不是16行对齐
  uint32_t remainder = row_cnt % kPerLoopRowSize;
  if (remainder != 0) {
    sub_input_cnt = 0;
    sub_column_cnt = 0;
    row.row_loop = max_row_loop;
    row.rows_cur_loop = remainder;
    MultipleInputsConcat16Rows<T, dim_size, input_num>(dst, srcs, curr_column_cnt, sub_column_cnt, sub_input_cnt,
                                                       max_column, curr_input, row, tmp_buf1);
  }

  return;
}

template <int dim_size>
inline __aicore__ uint32_t MultiplyShapeRange(const uint32_t (&shape)[dim_size], int start, int end) {
  uint32_t result = 1;
  for (int i = start; i < end; i++) {
    result *= shape[i];
  }
  return result;
}

// concat dim前边的所有轴合成一根轴，concat dim及后边的轴合成一根轴
template <typename T, int dim_size, int input_num>
inline __aicore__ void MergeAxis(const ConcatParams<T, dim_size> &dst, const ConcatParams<T, dim_size> srcs[input_num],
                                 const uint32_t concat_dim, ConcatParams<T, kMergedDimNum> &merge_dst,
                                 ConcatParams<T, kMergedDimNum> *merge_srcs) {
  merge_dst.stride[1] = 1;
  merge_dst.tensor = dst.tensor;
  merge_dst.shape[1] = MultiplyShapeRange<dim_size>(dst.shape, concat_dim, dim_size);
  merge_dst.stride[0] = KernelUtils::BlkAlign<T>(dst.shape[dim_size - 1]) *
                        MultiplyShapeRange<dim_size>(dst.shape, concat_dim, dim_size - 1);
  merge_dst.shape[0] = MultiplyShapeRange<dim_size>(dst.shape, 0, concat_dim);

  uint32_t tail_shape = 0;
  for (int i = 0; i < input_num; i++) {
    merge_srcs[i].tensor = srcs[i].tensor;
    merge_srcs[i].stride[1] = 1;
    if (concat_dim != dim_size - 1) {
      tail_shape = KernelUtils::BlkAlign<T>(srcs[i].shape[dim_size - 1]);
    } else {
      tail_shape = srcs[i].shape[dim_size - 1];
    }
    merge_srcs[i].shape[1] = tail_shape * MultiplyShapeRange<dim_size>(srcs[i].shape, concat_dim, dim_size - 1);
    merge_srcs[i].stride[0] = KernelUtils::BlkAlign<T>(srcs[i].shape[dim_size - 1]) *
                              MultiplyShapeRange<dim_size>(srcs[i].shape, concat_dim, dim_size - 1);
    merge_srcs[i].shape[0] = MultiplyShapeRange<dim_size>(srcs[i].shape, 0, concat_dim);
  }
}

template <typename T>
inline __aicore__ void ConcatParamsMapToFourBytes(ConcatParams<int32_t, kMergedDimNum> &dst,
                                                  const ConcatParams<T, kMergedDimNum> src,
                                                  LocalTensor<int32_t> &local_tensor) {
  dst.shape[0] = src.shape[0];
  dst.shape[1] = src.shape[1] * kScalarTow;
  dst.stride[0] = src.stride[0] * kScalarTow;
  dst.stride[1] = src.stride[1];
  dst.tensor = &local_tensor;
}

template <typename T>
inline __aicore__ uint32_t GetMaxColum(uint32_t size) {
  if (sizeof(T) == kOneByte) {
    return ((size / 48 - 31 - 31) / kDataBlockSize) * kDataBlockSize;
  } else if (sizeof(T) == kTwoBytes) {
    return (((size >> 5) - 15 - 15) / (kDataBlockSize / sizeof(T))) * (kDataBlockSize / sizeof(T));
  } else {
    return (((size >> 5) - 15 - 7) / (kDataBlockSize / sizeof(T))) * (kDataBlockSize / sizeof(T));
  }
  return 0;
}

template <typename T, int dimSize, int inputNum>
inline __aicore__ void ConcatExtendProcess(const ConcatParams<T, dimSize> &dst,
                                           const ConcatParams<T, kMergedDimNum> srcs[inputNum], uint32_t &currColumnCnt,
                                           LocalTensor<uint8_t> &tmp_buf) {
  // 将第一个输入拷贝到dst
  LocalTensor<T> tmp_buf1 = tmp_buf.ReinterpretCast<T>();
  ConcatExtendColumnAligned<T, kMergedDimNum>(dst, srcs[0], currColumnCnt);
  currColumnCnt += srcs[0].shape[1];
  uint32_t align = GetColumnCopiedAlign<T>();
  uint32_t max_column = GetMaxColum<T>(tmp_buf1.GetSize());
  uint32_t sub_column_cnt = 0;
  uint32_t sub_input_cnt = 0;

  for (uint32_t i = 1; i < inputNum; i += sub_input_cnt) {
    if (((currColumnCnt * sizeof(T)) % kDataBlockSize) == 0) {
      // 如果dst是32字节对齐的,直接DataCopy拼接，不需要转置
      ConcatExtendColumnAligned<T, kMergedDimNum>(dst, srcs[i], currColumnCnt);
      currColumnCnt += srcs[i].shape[1];
      sub_input_cnt = 1;
    } else {
      // 如果当前输入的列太大，走切列流程
      if ((srcs[i].shape[1] + align) > max_column) {
        ConcatExtendColumnNotAligned<T, kMergedDimNum>(dst, srcs[i], currColumnCnt, max_column, tmp_buf1);
        currColumnCnt += srcs[i].shape[1];
        sub_input_cnt = 1;
        continue;
      }

      // 当输入的列比较小时，走多输入合并转置流程
      if (sizeof(T) == kOneByte) {
        MultipleInputsMergeConcat<T, kMergedDimNum, inputNum>(dst, srcs, currColumnCnt, tmp_buf1, max_column, i,
                                                              sub_input_cnt, sub_column_cnt);
        currColumnCnt += sub_column_cnt;
      } else {
        MultipleInputsCopyThenTrans<T, kMergedDimNum, inputNum>(dst, srcs, currColumnCnt, tmp_buf1, max_column, i,
                                                                sub_input_cnt, sub_column_cnt);
        currColumnCnt += sub_column_cnt;
      }
    }
  }
}

template <typename T, int dimSize, int inputNum>
inline __aicore__ void ConcatExtend(const ConcatParams<T, dimSize> &dst, const ConcatParams<T, dimSize> srcs[inputNum],
                                    const uint32_t concatDim, LocalTensor<uint8_t> &tmpBuf) {
  if (inputNum == 0U) {
    ASSERT(false && "ConcatExtend input srcs is empty.");
    return;
  }
  ASSERT(((sizeof(T) == kOneByte) || (sizeof(T) == kTwoBytes) || (sizeof(T) == kFourBytes) ||
          (sizeof(T) == kEightBytes)) &&
         "ConcatExtend data type is not supported.");

  if (concatDim == 0U) {
    // 首轴concat不使用api方式实现
    ASSERT(false && "ConcatExtend axis 0 is not supported.");
  } else {
    ConcatParams<T, kMergedDimNum> mergedDst{};
    ConcatParams<T, kMergedDimNum> mergedSrcs[inputNum]{};

    // 合轴
    MergeAxis<T, dimSize, inputNum>(dst, srcs, concatDim, mergedDst, &mergedSrcs[0]);

    uint32_t currColumnCnt = 0;  // 当前拼接的总列数
    if (sizeof(T) == kEightBytes) {
      auto dst_tensor = dst.tensor->template ReinterpretCast<int32_t>();
      LocalTensor<int32_t> src_tensors[inputNum];
      ConcatParams<int32_t, kMergedDimNum> mergedDstFourBytes{};
      ConcatParams<int32_t, kMergedDimNum> mergedSrcsFourBytes[inputNum]{};
      ConcatParamsMapToFourBytes(mergedDstFourBytes, mergedDst, dst_tensor);
      for (int i = 0; i < inputNum; i++) {
        src_tensors[i] = mergedSrcs[i].tensor->template ReinterpretCast<int32_t>();
        ConcatParamsMapToFourBytes(mergedSrcsFourBytes[i], mergedSrcs[i], src_tensors[i]);
      }

      ConcatExtendProcess<int32_t, kMergedDimNum, inputNum>(mergedDstFourBytes, mergedSrcsFourBytes, currColumnCnt,
                                                            tmpBuf);
    } else {
      ConcatExtendProcess<T, kMergedDimNum, inputNum>(mergedDst, mergedSrcs, currColumnCnt, tmpBuf);
    }
  }
}

template <typename T>
inline __aicore__ void FillTransDataAddrList(uint64_t (&dst_addr_list)[kAddrListSize],
                                             uint64_t (&src_addr_list)[kAddrListSize], T *dst_addr, T *src_addr,
                                             uint64_t dst_stride, uint64_t src_stride) {
  for (uint32_t i = 0U; i < kAddrListSize; ++i) {
    dst_addr_list[i] = reinterpret_cast<uint64_t>(dst_addr);
    src_addr_list[i] = reinterpret_cast<uint64_t>(src_addr);
    dst_addr += dst_stride;
    src_addr += src_stride;
  }
}

template <typename T>
inline __aicore__ void FillTransDataAddrList(uint64_t (&dst_addr_list)[kAddrListSize],
                                             uint64_t (&src_addr_list)[kAddrListSize], const LocalTensor<T> &dst,
                                             const LocalTensor<T> &src, uint64_t dst_stride, uint64_t src_stride) {
  FillTransDataAddrList(dst_addr_list, src_addr_list, reinterpret_cast<T *>(dst.GetPhyAddr()),
                        reinterpret_cast<T *>(src.GetPhyAddr()), dst_stride, src_stride);
}

template <size_t INPUT_NUM>
struct ConcatTiling {
  uint32_t gcd;
  uint32_t tmp_buf_size;
  uint32_t dst_dim_size;
  uint32_t dst_row_num_unit;
  uint32_t max_repeat_times;
  uint32_t max_element_num;
  uint32_t max_orig_row_num;
  uint32_t per_repeat_size;
  uint16_t first_copy_repeat_times;  // for diff dim
  uint8_t last_trans_repeat_times;   // for diff dim
  uint32_t src_dim_sizes[INPUT_NUM];
  uint32_t src_strides[INPUT_NUM];  // src loop stride
  uint32_t src_buffer_offsets[INPUT_NUM];
  uint16_t gather_mask_repeat_strides[INPUT_NUM];  // for remove pad
  uint32_t gather_mask_dim_sizes[INPUT_NUM];       // for remove pad
};

template <size_t INPUT_NUM>
struct ConcatTilingAllAligned {
  uint32_t dst_col_size;
  uint32_t src_col_sizes[INPUT_NUM];
  uint32_t dst_offsets[INPUT_NUM];
};

template <typename T, int INPUT_NUM>
struct ConcatInputList {
  T *src_tensor_base_addrs[INPUT_NUM];
  const LocalTensor<T> *src_tensors[INPUT_NUM];
};

template <typename T, int INPUT_NUM>
struct ConcatContext {
  static constexpr int32_t kInputNum = INPUT_NUM;
  static constexpr uint32_t kDataTypeSize = sizeof(T);
  static constexpr bool kIsPadded = false;
  using DataType = T;

  uint32_t total_row_num;
  uint32_t loop_times;
  uint32_t tail_element_num;

  uint32_t num_elements;
  uint32_t repeat_times;
  uint32_t orig_row_num;

  LocalTensor<T> tmp_buf_low;
  LocalTensor<T> tmp_buf_high;
  LocalTensor<uint8_t> stack_buffer;

  ConcatInputList<T, INPUT_NUM> *input_list;
};

template <typename T, int INPUT_NUM, uint32_t SRC_DIM_SIZE>
struct ConcatContextSameDim : ConcatContext<T, INPUT_NUM> {
  static constexpr bool is_same_dim = true;
  static constexpr uint32_t kSrcDimSize = SRC_DIM_SIZE;
  static constexpr uint32_t kDstDimSize = SRC_DIM_SIZE * INPUT_NUM;
};

template <typename T, int INPUT_NUM>
struct ConcatContextDiffDim : ConcatContext<T, INPUT_NUM> {
  static constexpr bool is_same_dim = false;
};

template <typename T, int INPUT_NUM, uint32_t SRC_DIM_SIZE>
struct ConcatContextSameDimPadded : ConcatContextSameDim<T, INPUT_NUM, SRC_DIM_SIZE> {
  static constexpr bool kIsPadded = true;
};

template <typename T, int INPUT_NUM>
struct ConcatContextDiffDimPadded : ConcatContextDiffDim<T, INPUT_NUM> {
  static constexpr bool is_same_dim = false;
  static constexpr bool kIsPadded = true;
};

template <typename T>
inline __aicore__ constexpr bool IsSameDim() {
  return T::is_same_dim;
}

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template <typename ConcatContextType>
inline __aicore__ static void InitConcatContext(ConcatContextType &context,
                                                const ConcatTiling<ConcatContextType::kInputNum> &concat_tiling,
                                                LocalTensor<uint8_t> &tmp_buf) {
  using T = typename ConcatContextType::DataType;
  const auto tmp_buffer_size = concat_tiling.tmp_buf_size;
  context.tmp_buf_low = tmp_buf.ReinterpretCast<T>();
  context.tmp_buf_high = context.tmp_buf_low[tmp_buffer_size / 2 / sizeof(T)];

  uint32_t total_element_num = context.total_row_num * concat_tiling.dst_dim_size;
  context.loop_times = total_element_num / concat_tiling.max_element_num;
  context.tail_element_num = total_element_num % concat_tiling.max_element_num;
  if (context.tail_element_num != 0) {
    context.loop_times += 1;
  }

  context.repeat_times = concat_tiling.max_repeat_times;
  context.orig_row_num = concat_tiling.max_orig_row_num;
  context.num_elements = concat_tiling.max_element_num;
}

template <typename ConcatContextType>
inline __aicore__ static void UpdateConcatContext(ConcatContextType &context,
                                                  const ConcatTiling<ConcatContextType::kInputNum> &tiling,
                                                  uint32_t loop_index) {
  context.num_elements = context.tail_element_num;
  context.repeat_times = context.tail_element_num / tiling.per_repeat_size;
  if (context.repeat_times * tiling.per_repeat_size != context.tail_element_num) {
    context.repeat_times += 1;
  }
  context.orig_row_num = context.total_row_num - tiling.max_orig_row_num * loop_index;
}

template <typename ConcatContextType>
inline __aicore__ void RemoveInputPaddings(ConcatContextType &context,
                                           const ConcatTiling<ConcatContextType::kInputNum> &tiling,
                                           uint32_t loop_index, typename ConcatContextType::DataType **src_addrs,
                                           size_t num_addrs) {
  for (int index = 0; index < num_addrs; ++index) {
    if (tiling.gather_mask_repeat_strides[index] == 0) {
      src_addrs[index] = context.input_list->src_tensor_base_addrs[index] + loop_index * tiling.src_strides[index];
    } else {
      GatherMaskParams gather_mask_params{
          1,
          static_cast<uint16_t>(context.orig_row_num *
                                (tiling.src_dim_sizes[index] / tiling.gather_mask_dim_sizes[index])),
          tiling.gather_mask_repeat_strides[index], 0};
      uint64_t rsvd_cnt = 0;
      constexpr uint8_t kSrcPattern = 7;
      auto gather_dst = context.tmp_buf_high[tiling.src_buffer_offsets[index]];
      GatherMask(gather_dst, (*context.input_list->src_tensors[index])[loop_index * tiling.src_strides[index]],
                 kSrcPattern, true, tiling.gather_mask_dim_sizes[index], gather_mask_params, rsvd_cnt);
      src_addrs[index] = reinterpret_cast<typename ConcatContextType::DataType *>(gather_dst.GetPhyAddr());
    }
  }
}

template <typename ConcatContextType>
inline __aicore__ static void ConcatSameDimFirstTranspose(ConcatContextType &context,
                                                          typename ConcatContextType::DataType *src_addr,
                                                          uint32_t index,
                                                          const TransDataTo5HDParams &first_trans_params) {
  using T = typename ConcatContextType::DataType;
  constexpr uint32_t kScaleToB16 = sizeof(T) / sizeof(half);
  constexpr uint32_t kStride = 16U;
  constexpr uint32_t kEltNumPerRow = 16U / kScaleToB16;
  constexpr uint32_t kBufferStride = kStride * ConcatContextType::kSrcDimSize;
  uint64_t dst_addr_list[kAddrListSize];
  uint64_t src_addr_list[kAddrListSize];
  auto *dst_addr = reinterpret_cast<T *>(context.tmp_buf_low[index * kBufferStride].GetPhyAddr());
  if constexpr (sizeof(typename ConcatContextType::DataType) == sizeof(int32_t)) {
    constexpr auto kDstAddrStride = kEltNumPerRow * 2U * ConcatContextType::kDstDimSize;
    for (uint32_t i = 0U; i < kAddrListSize; i += 2) {
      dst_addr_list[i] = reinterpret_cast<uint64_t>(dst_addr);
      dst_addr_list[i + 1] = reinterpret_cast<uint64_t>(dst_addr + kEltNumPerRow);
      dst_addr += kDstAddrStride;
      src_addr_list[i] = reinterpret_cast<uint64_t>(src_addr);
      src_addr += kEltNumPerRow;
      src_addr_list[i + 1] = reinterpret_cast<uint64_t>(src_addr);
      src_addr += kEltNumPerRow;
    }
  } else {
    FillTransDataAddrList(dst_addr_list, src_addr_list, dst_addr, src_addr, kStride * ConcatContextType::kDstDimSize,
                          kStride);
  }
  TransDataTo5HD<uint16_t>(dst_addr_list, src_addr_list, first_trans_params);
}

template <typename ConcatContextType>
inline __aicore__ void ConcatSameDimFirstTranspose(ConcatContextType &context,
                                                   const ConcatTiling<ConcatContextType::kInputNum> &tiling,
                                                   uint32_t loop_index) {
  using T = typename ConcatContextType::DataType;
  TransDataTo5HDParams first_trans_params{
      false, false, static_cast<uint8_t>(context.repeat_times * ConcatContextType::kSrcDimSize),
      static_cast<uint16_t>(kAddrListSize * ConcatContextType::kInputNum), kAddrListSize};
  if (first_trans_params.repeatTimes == 1) {
    first_trans_params.srcRepStride = 0;
    first_trans_params.dstRepStride = 0;
  }

  if constexpr (ConcatContextType::kIsPadded) {
    // 存在需要RemovePad的
    T *src_addrs[ConcatContextType::kInputNum];
    RemoveInputPaddings(context, tiling, loop_index, src_addrs, ConcatContextType::kInputNum);
    AscendC::PipeBarrier<PIPE_V>();
    for (int index = 0; index < ConcatContextType::kInputNum; ++index) {
      auto src_addr = src_addrs[index];
      ConcatSameDimFirstTranspose(context, src_addr, index, first_trans_params);
    }
  } else {
    for (int index = 0; index < ConcatContextType::kInputNum; ++index) {
      auto src_addr = context.input_list->src_tensor_base_addrs[index] + loop_index * tiling.src_strides[index];
      ConcatSameDimFirstTranspose(context, src_addr, index, first_trans_params);
    }
  }
}

template <typename ConcatContextType>
inline __aicore__ void ConcatDiffDimFirstTranspose(ConcatContextType &context,
                                                   const ConcatTiling<ConcatContextType::kInputNum> &tiling,
                                                   uint32_t index, typename ConcatContextType::DataType *src_addr) {
  using T = typename ConcatContextType::DataType;
  constexpr auto kRowNumPerTrans = kAddrListSize;
  constexpr uint32_t kRowEltNum = kDataBlockSize / ConcatContextType::kDataTypeSize;
  uint64_t first_trans_src_list[kAddrListSize];
  uint64_t first_trans_dst_list[kAddrListSize];
  TransDataTo5HDParams first_trans_params{
      false, false, static_cast<uint8_t>(context.repeat_times * (tiling.src_dim_sizes[index] / tiling.gcd)),
      kRowNumPerTrans, 1};
  if (first_trans_params.repeatTimes == 1) {
    first_trans_params.srcRepStride = 0;
    first_trans_params.dstRepStride = 0;
  }
  FillTransDataAddrList(first_trans_dst_list, first_trans_src_list,
                        reinterpret_cast<T *>(context.tmp_buf_low[tiling.src_buffer_offsets[index]].GetPhyAddr()),
                        src_addr, kRowEltNum, kRowEltNum * first_trans_params.repeatTimes);
  TransDataTo5HD<uint16_t>(first_trans_dst_list, first_trans_src_list, first_trans_params);
}

template <typename ConcatContextType>
inline __aicore__ void ConcatDiffDimFirstTranspose(ConcatContextType &context,
                                                   const ConcatTiling<ConcatContextType::kInputNum> &tiling,
                                                   uint32_t loop_index) {
  using T = typename ConcatContextType::DataType;
  if constexpr (ConcatContextType::kIsPadded) {
    // 存在需要RemovePad的
    T *src_addrs[ConcatContextType::kInputNum];
    RemoveInputPaddings(context, tiling, loop_index, src_addrs, ConcatContextType::kInputNum);
    AscendC::PipeBarrier<PIPE_V>();
    for (int index = 0; index < ConcatContextType::kInputNum; ++index) {
      ConcatDiffDimFirstTranspose(context, tiling, index, src_addrs[index]);
    }
  } else {
    for (int index = 0; index < ConcatContextType::kInputNum; ++index) {
      auto src_addr = context.input_list->src_tensor_base_addrs[index] + loop_index * tiling.src_strides[index];
      ConcatDiffDimFirstTranspose(context, tiling, index, src_addr);
    }
  }
}

template <typename ConcatContextType, enable_if_t<!IsSameDim<ConcatContextType>()> * = nullptr>
inline __aicore__ void ConcatExtendV2(ConcatContextType &context,
                                      const ConcatTiling<ConcatContextType::kInputNum> &tiling,
                                      LocalTensor<typename ConcatContextType::DataType> &dst_tensor,
                                      LocalTensor<uint8_t> &tmp_buf) {
  using T = typename ConcatContextType::DataType;
  constexpr uint32_t kScaleToB16 = sizeof(T) / sizeof(half);
  constexpr uint32_t kRowEltNum = kDataBlockSize / sizeof(T);
  constexpr auto kRowNumPerTrans = kAddrListSize;

  InitConcatContext(context, tiling, tmp_buf);
  TransDataTo5HDParams last_trans_params{false, false, tiling.last_trans_repeat_times, kRowNumPerTrans,
                                         kRowNumPerTrans};
  DataCopyParams first_copy_params{tiling.first_copy_repeat_times, 0, 0, 0};
  DataCopyParams last_copy_params{last_trans_params.repeatTimes, 1, kRowNumPerTrans - 1, 0};

  auto tmp_buf_low = context.tmp_buf_low;
  auto tmp_buf_high = context.tmp_buf_high;
  uint64_t last_trans_src_list[kAddrListSize];
  uint64_t last_trans_dst_list[kAddrListSize];
  FillTransDataAddrList(last_trans_dst_list, last_trans_src_list, tmp_buf_low, tmp_buf_high, kRowEltNum, kRowEltNum);

  uint32_t dst_offset = 0;
  for (uint32_t k = 0; k < context.loop_times; ++k) {
    if ((context.tail_element_num != 0) && (k == context.loop_times - 1)) {
      // tail
      UpdateConcatContext(context, tiling, k);
      first_copy_params.blockCount = context.repeat_times * kRowNumPerTrans / kScaleToB16 / tiling.gcd;
      last_trans_params.repeatTimes = context.repeat_times * (tiling.dst_dim_size / tiling.gcd);
      last_copy_params.blockCount = last_trans_params.repeatTimes;
    }
    ConcatDiffDimFirstTranspose(context, tiling, k);
    AscendC::PipeBarrier<PIPE_V>();
    int32_t dim_start = 0;
#pragma unroll
    for (int index = 0; index < ConcatContextType::kInputNum; ++index) {
      first_copy_params.blockLen = tiling.src_dim_sizes[index] * kScaleToB16;
      first_copy_params.dstStride = tiling.dst_row_num_unit - first_copy_params.blockLen;
      DataCopy(tmp_buf_high[dim_start * kRowEltNum], tmp_buf_low[tiling.src_buffer_offsets[index]], first_copy_params);
      dim_start += first_copy_params.blockLen;
    }
    if (last_trans_params.repeatTimes == 1) {
      last_trans_params.srcRepStride = 0;
      last_trans_params.dstRepStride = 0;
    }
    AscendC::PipeBarrier<PIPE_V>();
    TransDataTo5HD<uint16_t>(last_trans_dst_list, last_trans_src_list, last_trans_params);

    uint32_t src_elt_offset = 0;
    uint32_t dst_elt_offset = 0;
    uint32_t dst_elt_stride = kRowEltNum * last_copy_params.blockCount;
    AscendC::PipeBarrier<PIPE_V>();
    if (context.num_elements == tiling.max_element_num) {
      // not tail block
      for (uint32_t i = 0; i < kRowNumPerTrans; ++i) {
        DataCopy(dst_tensor[dst_offset + dst_elt_offset], tmp_buf_low[src_elt_offset], last_copy_params);
        src_elt_offset += kRowEltNum;
        dst_elt_offset += dst_elt_stride;
      }
    } else {
      // tail block
      for (uint32_t i = 0; i < kRowNumPerTrans; ++i) {
        DataCopy(tmp_buf_high[dst_elt_offset], tmp_buf_low[src_elt_offset], last_copy_params);
        src_elt_offset += kRowEltNum;
        dst_elt_offset += dst_elt_stride;
      }
      AscendC::PipeBarrier<PIPE_V>();
      DataCopy(dst_tensor[dst_offset], tmp_buf_high, KernelUtils::BlkAlign<T>(context.num_elements));
    }
    dst_offset += tiling.max_element_num;
  }
}

template <typename ConcatContextType, enable_if_t<IsSameDim<ConcatContextType>()> * = nullptr>
inline __aicore__ void ConcatExtendV2(ConcatContextType &context,
                                      const ConcatTiling<ConcatContextType::kInputNum> &tiling,
                                      LocalTensor<typename ConcatContextType::DataType> &dst_tensor,
                                      LocalTensor<uint8_t> &tmp_buf) {
  using T = typename ConcatContextType::DataType;
  constexpr uint32_t kRowEltNum = kDataBlockSize / sizeof(T);
  constexpr auto kRowNumPerTrans = kAddrListSize;
  constexpr uint32_t kEltNumPerBlock = kRowNumPerTrans * kRowEltNum;
  InitConcatContext(context, tiling, tmp_buf);

  auto tmp_buf_low = context.tmp_buf_low;
  auto tmp_buf_high = context.tmp_buf_high;
  uint32_t dst_offset = 0;
  constexpr auto rep_stride = static_cast<uint16_t>(kRowNumPerTrans * ConcatContextType::kInputNum);
  constexpr auto dst_addr_stride = kRowEltNum * ConcatContextType::kInputNum;
  for (uint32_t k = 0U; k < context.loop_times; ++k) {
    if ((context.tail_element_num != 0) && (k == context.loop_times - 1)) {
      // tail
      UpdateConcatContext(context, tiling, k);
    }
    // first transpose
    ConcatSameDimFirstTranspose<ConcatContextType>(context, tiling, k);
    // transpose back
    TransDataTo5HDParams last_trans_params{false, false,
                                           static_cast<uint8_t>(context.repeat_times * ConcatContextType::kSrcDimSize),
                                           rep_stride, rep_stride};
    if (last_trans_params.repeatTimes == 1) {
      last_trans_params.srcRepStride = 0;
      last_trans_params.dstRepStride = 0;
    }
    uint32_t offset = 0;
    AscendC::PipeBarrier<PIPE_V>();
    for (int index = 0; index < ConcatContextType::kInputNum; ++index) {
      uint64_t first_trans_src_list[kAddrListSize];
      uint64_t first_trans_dst_list[kAddrListSize];
      FillTransDataAddrList(
          first_trans_dst_list, first_trans_src_list, reinterpret_cast<T *>(tmp_buf_high[offset].GetPhyAddr()),
          reinterpret_cast<T *>(tmp_buf_low[index * kEltNumPerBlock].GetPhyAddr()), dst_addr_stride, kRowEltNum);
      TransDataTo5HD<uint16_t>(first_trans_dst_list, first_trans_src_list, last_trans_params);
      offset += kRowEltNum;
    }
    // copy result
    AscendC::PipeBarrier<PIPE_V>();
    DataCopy(dst_tensor[dst_offset], tmp_buf_high, KernelUtils::BlkAlign<T>(context.num_elements));
    dst_offset += tiling.max_element_num;
  }
}

template <typename T, uint32_t INPUT_NUM>
inline __aicore__ void ConcatAllAligned(uint32_t num_rows, const ConcatTilingAllAligned<INPUT_NUM> &tiling,
                                        LocalTensor<T> &dst_tensor, LocalTensor<T> (&src_tensors)[INPUT_NUM]) {
  constexpr auto align_size = static_cast<uint16_t>(kDataBlockSize / sizeof(T));
#pragma unroll
  for (uint32_t i = 0U; i < INPUT_NUM; ++i) {
    const auto size = tiling.src_col_sizes[i];
    DataCopyParams copy_params{static_cast<uint16_t>(num_rows), static_cast<uint16_t>(size / align_size), 0,
                               static_cast<uint16_t>((tiling.dst_col_size - size) / align_size)};
    DataCopy(dst_tensor[tiling.dst_offsets[i]], src_tensors[i], copy_params);
  }
}

template <uint32_t INPUT_NUM>
struct ConcatShape {
  uint32_t dst_cols;
  uint32_t src_cols[INPUT_NUM];
  uint32_t src_row_strides[INPUT_NUM];
  uint32_t src_second_last_dim_strides[INPUT_NUM];
  uint32_t gather_mask_dim_sizes[INPUT_NUM];
};

template <typename ConcatContextType>
inline __aicore__ void ConcatExtendV2Dyn(ConcatContextType &concat_context,
                                         const ConcatShape<ConcatContextType::kInputNum> &concat_shape,
                                         LocalTensor<typename ConcatContextType::DataType> &dst_tensor,
                                         LocalTensor<uint8_t> &tmp_buf) {
  using T = typename ConcatContextType::DataType;
  constexpr uint32_t kScaleToB16 = sizeof(T) / sizeof(half);
  constexpr uint32_t kRowEltNum = kDataBlockSize / sizeof(T);
  constexpr auto kRowNumPerTrans = kAddrListSize;
  constexpr uint32_t kEltNumPerBlock = kRowNumPerTrans * kRowEltNum;
  auto tmp_buf_size = (tmp_buf.GetSize() / (16 * 1024) * (16 * 1024));
  auto max_repeat_times = (tmp_buf_size >> 10U) / concat_shape.dst_cols;
  auto max_element_num = max_repeat_times * concat_shape.dst_cols * kEltNumPerBlock;
  auto max_orig_row_num = max_element_num / concat_shape.dst_cols;
  ConcatTiling<ConcatContextType::kInputNum> concat_tiling{
      .gcd = 1,
      .tmp_buf_size = tmp_buf_size,
      .dst_dim_size = concat_shape.dst_cols,
      .dst_row_num_unit = concat_shape.dst_cols * kScaleToB16,
      .max_repeat_times = max_repeat_times,
      .max_element_num = max_element_num,
      .max_orig_row_num = max_orig_row_num,
      .per_repeat_size = concat_shape.dst_cols * kEltNumPerBlock,
      .first_copy_repeat_times = static_cast<uint16_t>(max_repeat_times * kAddrListSize / kScaleToB16),
      .last_trans_repeat_times = static_cast<uint8_t>(max_repeat_times * concat_shape.dst_cols),
      .gather_mask_repeat_strides = {},
  };

  uint32_t buffer_offset = 0;
  for (uint32_t i = 0U; i < ConcatContextType::kInputNum; ++i) {
    concat_tiling.src_dim_sizes[i] = concat_shape.src_cols[i];
    concat_tiling.src_buffer_offsets[i] = buffer_offset;
    buffer_offset += (max_repeat_times * concat_shape.src_cols[i]) * kEltNumPerBlock;
    if constexpr (ConcatContextType::kIsPadded) {
      concat_tiling.src_strides[i] = max_orig_row_num * concat_shape.src_row_strides[i];
      if (concat_shape.src_second_last_dim_strides[i] != concat_shape.gather_mask_dim_sizes[i]) {
        concat_tiling.gather_mask_repeat_strides[i] =
            static_cast<uint16_t>(concat_shape.src_second_last_dim_strides[i] * sizeof(T) / kDataBlockSize);
        concat_tiling.gather_mask_dim_sizes[i] = concat_shape.gather_mask_dim_sizes[i];
      }
    } else {
      concat_tiling.src_strides[i] = max_orig_row_num * concat_shape.src_cols[i];
    }
  }
  ConcatExtendV2(concat_context, concat_tiling, dst_tensor, tmp_buf);
}

#endif  // __ASCENDC_API_CONCAT_H__
