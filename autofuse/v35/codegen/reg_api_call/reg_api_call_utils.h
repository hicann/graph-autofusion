/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AUTOFUSE_REG_API_CALL_UTILS_H__
#define __AUTOFUSE_REG_API_CALL_UTILS_H__

#include "codegen_kernel.h"
#include "api_call/utils/api_call_utils.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
/*
for循环组织形式如下:
for (i, 0, loop2Size) {
for (j, 0, loop1Size) {
  dst[i * loop2DstStride + j * loop1DrcStride] = src[i * loop2SrcStride + j * loop1SrcStride]
}
}
*/

struct LoopModeParams {
  // loop1Size, loop2Size
  std::vector<std::string> loop_size = {"1", "1"};
  // loop1SrcStride, loop2SrcStride 单位: 数字个数，在拼接参数时，会转换成字节数
  std::vector<std::string> loop_src_stride = {"0", "0"};
  // loop1DrcStride, loop2DstStride单位: 数字个数，在拼接参数时，会转换成字节数
  std::vector<std::string> loop_dst_stride = {"0", "0"};
};

// LoopModeParams 表达式版本 - 使用 CombinedExpression 替代 std::string
struct LoopModeParamsExpr {
  // loop1Size, loop2Size - 使用 ActualSize 转换
  std::vector<CombinedExpression> loop_size;
  // loop1SrcStride, loop2SrcStride 单位: 数字个数，使用 Size 转换
  std::vector<CombinedExpression> loop_src_stride;
  // loop1DstStride, loop2DstStride 单位: 数字个数，使用 Size 转换
  std::vector<CombinedExpression> loop_dst_stride;

  // 默认构造：初始化为默认值，与 LoopModeParams 保持一致
  LoopModeParamsExpr() {
    loop_size.emplace_back(CombinedExprFactory::Constant(1));
    loop_size.emplace_back(CombinedExprFactory::Constant(1));
    loop_src_stride.emplace_back(CombinedExprFactory::Constant(0));
    loop_src_stride.emplace_back(CombinedExprFactory::Constant(0));
    loop_dst_stride.emplace_back(CombinedExprFactory::Constant(0));
    loop_dst_stride.emplace_back(CombinedExprFactory::Constant(0));
  }
};

struct NddmaParams {
  std::stringstream ss_output_dims;
  std::stringstream ss_output_stride;
  std::stringstream ss_input_stride;
};

void CreateEnhanceDmaCall(const TPipe &tpipe, const Tensor &input, const Tensor &output, const string &gm_offset,
                          const DataCopyParams &data_copy_param, const ascir::SizeExpr &offset, std::stringstream &ss,
                          bool copy_in);
void CreateNddmaCall(const TPipe &tpipe, const Tensor &input, const Tensor &output, const string &gm_offset,
                     const DataCopyParams &data_copy_param, const ascir::SizeExpr &offset, std::stringstream &ss);
void SetNddmaParams(const TPipe &tpipe, const DataCopyParams &data_copy_param, NddmaParams &nddma_param,
                    const int64_t &tensor_id, std::stringstream &ss);
void SetLoopModeParams(const TPipe &tpipe, const DataCopyParams &data_copy_param, LoopModeParams &loop_mode_param,
                       bool copy_in);
void SetLoopModeParamsExpr(const DataCopyParams &data_copy_param, LoopModeParamsExpr &loop_mode_param, bool copy_in);
std::string GetPaddingMode(const TPipe &tpipe, const Tensor &ub_tensor, const DataCopyParams &data_copy_param);
void BuildDataCopyApiParamInCVFusion(CodegenApiParam &api_param, DmaSpecificParams &dma_specific_params,
                                     const Tensor &gm, const Tensor &ub, std::string &dtype_name, bool copy_in);
Status BuildDataCopyApiParamInNormal(const TPipe &tpipe, CodegenApiParam &api_param,
                                     DmaSpecificParams &dma_specific_params, const Tensor &src, const Tensor &dst,
                                     std::string &gm_offset, bool copy_in);
Status GenDataCopyDimParam(const CodegenApiParam &api_param, const Tiler &tiler, std::string graph_name,
                           std::string node_name, std::stringstream &ss);

}  // namespace codegen

#endif
