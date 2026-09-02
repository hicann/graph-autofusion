/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDUCTOR_CV_FUSION
#include "autofuse_cube_tiling_data.h"
#endif

using namespace Atcos;
using namespace Atcos::Conv;

using aLayout = Atcos::Conv::layout::NCHW;
using bLayout = Atcos::Conv::layout::CI1KHKWCOCI0;
using cLayout = Atcos::Conv::layout::NCHW;
using biasLayout = Atcos::Conv::layout::NCHW;

template <int8_t FmapTiling, int8_t WeightTiling, int8_t L1PingPong, int8_t L0PingPong, int8_t OutputOrder,
          int8_t IterOrder, int8_t GroupType, int8_t EnableSmallChannel, int8_t WeightUbTrans, int8_t FmapCopyMode,
          int8_t InnerBatch, int8_t DisContinuous, int8_t BatchOne, int8_t NoPad, int8_t SmallWeight,
          int8_t SmallKernel>
__aicore__ void conv2d_v2(
#ifdef CV_UB_FUSION
    // scaleGM 对齐 ExtendConv2D fixpipe；无 scale0 时由 codegen 传入 nullptr。
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR scaleGM, GM_ADDR cGM, GM_ADDR workspaceGM,
    GM_ADDR tilingGM, AutoFusionVector::Params *params
#else
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR scaleGM, GM_ADDR cGM, GM_ADDR workspaceGM,
    GM_ADDR tilingGM
#endif
) {
  REGISTER_TILING_DEFAULT(Conv2DTilingData);
  GET_TILING_DATA_WITH_STRUCT(Conv2DTilingData, tilingData, tilingGM);
  // 新增 BatchOne/NoPad/SmallWeight/SmallKernel 模板维，与 ops-nn ActConvConfig 保持一致。
  using ConvConfig =
      Conv2DV2Advanced::ActConvConfig<FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, IterOrder,
                                      GroupType, EnableSmallChannel, WeightUbTrans, FmapCopyMode, InnerBatch,
                                      DisContinuous, BatchOne, NoPad, SmallWeight, SmallKernel>;
  Conv2DV2Advanced::ConvActKernel<tilingData, DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, DTYPE_SCALE0, aLayout, bLayout,
                                  cLayout, biasLayout, ConvConfig>(
#ifdef CV_UB_FUSION
      aGM, bGM, biasGM, offsetWGM, scaleGM, cGM, workspaceGM, params
#else
      aGM, bGM, biasGM, offsetWGM, scaleGM, cGM, workspaceGM
#endif
  );
}
