/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_UTILS_TENSOR_UTILS_H_
#define INC_GRAPH_UTILS_TENSOR_UTILS_H_

#include <vector>
#include "graph/attr_value_serializable.h"
#include "graph/def_types.h"
#include "graph/ge_error_codes_af.h"
#include "graph/ge_tensor.h"

namespace af {
class TensorUtils {
 public:
  static GeTensor CreateShareTensor(const GeTensor &other);
  static GeTensor CreateShareTensor(const GeTensorDesc &tensor_desc, std::shared_ptr<AlignedPtr> aligned_ptr,
                                    const size_t size);
  static void ShareTensor(const GeTensor &from, GeTensor &to);
  static TensorData CreateShareTensorData(const TensorData &other);
  static void ShareTensorData(const TensorData &from, TensorData &to);
  static void ShareAlignedPtr(std::shared_ptr<AlignedPtr> ptr, const size_t size, TensorData &to);
  static void ShareAlignedPtr(std::shared_ptr<AlignedPtr> ptr, const size_t size, GeTensor &to);
  static void CopyTensor(const GeTensor &from, GeTensor &to);
  static ge::graphStatus GetSize(const GeTensorDesc &tensor_desc, int64_t &size);
  static void SetSize(GeTensorDesc &tensor_desc, const int64_t size);
  static int64_t GetWeightSize(const ConstGeTensorPtr &tensor_ptr);
  static int64_t GetWeightSize(const GeTensor &tensor);
  static int64_t GetWeightSize(const GeTensorDesc &tensor_desc);
  static uint8_t *GetWeightAddr(const ConstGeTensorPtr &tensor_ptr, const uint8_t *const base);
  static uint8_t *GetWeightAddr(const GeTensor &tensor, const uint8_t *const base);
  static void SetWeightSize(GeTensorDesc &tensor_desc, const int64_t size);
  static ge::graphStatus GetReuseInput(const GeTensorDesc &tensor_desc, bool &flag);
  static void SetReuseInput(GeTensorDesc &tensor_desc, const bool flag);
  static ge::graphStatus GetOutputTensor(const GeTensorDesc &tensor_desc, bool &flag);
  static void SetOutputTensor(GeTensorDesc &tensor_desc, const bool flag);
  static graphStatus GetDeviceType(const GeTensorDesc &tensor_desc, ge::DeviceType &type);
  static void SetDeviceType(GeTensorDesc &tensor_desc, const ge::DeviceType type);
  static ge::graphStatus GetInputTensor(const GeTensorDesc &tensor_desc, bool &flag);
  static void SetInputTensor(GeTensorDesc &tensor_desc, const bool flag);
  static ge::graphStatus GetRealDimCnt(const GeTensorDesc &tensor_desc, uint32_t &cnt);
  static void SetRealDimCnt(GeTensorDesc &tensor_desc, const uint32_t cnt);
  static ge::graphStatus GetReuseInputIndex(const GeTensorDesc &tensor_desc, uint32_t &idx);
  static void SetReuseInputIndex(GeTensorDesc &tensor_desc, const uint32_t idx);
  static ge::graphStatus GetDataOffset(const GeTensorDesc &tensor_desc, int64_t &offset);
  static void SetDataOffset(GeTensorDesc &tensor_desc, const int64_t offset);
  static ge::graphStatus GetRC(const GeTensorDesc &tensor_desc, uint32_t &rc);
  static void SetRC(GeTensorDesc &tensor_desc, const uint32_t rc);
  static bool IsOriginShapeInited(const GeTensorDesc &tensor_desc);

  static ge::graphStatus CalcTensorMemSize(const GeShape &shape, const Format format, const DataType data_type,
                                           int64_t &mem_size);
  static ge::graphStatus CalcTensorMemSizeForNoTiling(const GeTensorDesc &tensor, const Format format,
                                                      const DataType data_type, int64_t &mem_size);
  // 待废弃接口，为保持兼容暂时保留。后续请使用GetTensorMemorySizeInBytesWithAutoPadding替代GetTensorMemorySizeInBytes.
  static ge::graphStatus GetTensorMemorySizeInBytes(const GeTensorDesc &desc_temp, int64_t &size_temp);

  /**
   * @brief Calculate tensor memory size with automatic alignment padding.
   *
   * This function calculates the tensor memory size with automatic alignment padding
   * added, ensuring the memory allocation meets hardware alignment requirements.
   * Compared to GetTensorMemorySizeInBytes, this function additionally considers
   * the actual SoC padding size.
   *
   * @param [in] desc_temp Tensor descriptor containing shape, data type, format, etc.
   * @param [out] size_temp Calculated memory size in bytes, including alignment padding.
   * @return ge::GRAPH_SUCCESS on success; ge::GRAPH_FAILED on failure.
   *
   * @note Recommended to use this function instead of GetTensorMemorySizeInBytes.
   */
  static ge::graphStatus GetTensorMemorySizeInBytesWithAutoPadding(const GeTensorDesc &desc_temp, int64_t &size_temp);
  static ge::graphStatus GetTensorSizeInBytes(const GeTensorDesc &desc_temp, int64_t &size_temp);
  static ge::graphStatus CheckShapeByShapeRange(const GeShape &shape,
                                                const std::vector<std::pair<int64_t, int64_t>> &shape_range);
  static bool IsShapeEqual(const GeShape &src, const GeShape &dst);
  static bool IsMemorySizeCalcTypeAlwaysEmpty(const GeTensorDesc &tensor_desc);

  /**
   * @brief Get the padding size for memory allocation.
   *
   * This function queries the SoC specification to get the actual padding size.
   * If the query fails, it returns the default padding size (32 bytes).
   * The result is cached for subsequent calls.
   *
   * @return The padding size in bytes.
   */
  static int64_t GetPaddingSize();
};
}  // namespace af
#endif  // INC_GRAPH_UTILS_TENSOR_UTILS_H_
