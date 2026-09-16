/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCIR_REG_FUNC_DEFAULT_REG_FUNC_H__
#define __ASCIR_REG_FUNC_DEFAULT_REG_FUNC_H__

#include "ascendc_ir.h"

namespace af {
namespace ascir {
std::vector<std::unique_ptr<TmpBufDesc>> GetTmpBuffer(const Expression &tmp_size);
std::vector<std::unique_ptr<TmpBufDesc>> CalcDefaultTmpSize(const AscNode &node);
Expression GetInputSize(AscNodeInputs &node_inputs);
std::vector<std::unique_ptr<TmpBufDesc>> GetInputDataSizeTmpBuffer(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcBinaryApiTmpSize(const AscNode &node);
uint32_t GetNonScalarAxisId(AscNodeInputs &node_inputs);
bool IsAllScalarOrUbScalar(AscNodeInputs &node_inputs);
bool HasScalarOrUbScalar(AscNodeInputs &node_inputs);

std::vector<std::unique_ptr<TmpBufDesc>> CalcBroadCastTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcConcatTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcConcatTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcPadTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcDivTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcEqTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcErfTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcGatherTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcGeTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcGtTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcIsnanTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcIsInfTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcLeTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcLogicalAndTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcLtTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcNeTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcPowTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcReduceTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcReduceMaxTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcRsqrtTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcRemainderTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcSelectTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcMaskedFillTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcSigmoidTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcSubTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcTrueDivTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcCastTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcClipByValueTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcIsFiniteTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcLogicalNotTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcLogicalOrTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcSignTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcTanhTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcWhereTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcAxpyTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcSplitTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcAbsTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcWelfordUpdateTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcWelfordFinalizeTmpSize(const AscNode &node);
}  // namespace ascir
}  // namespace af
#endif  // __ASCIR_REG_FUNC_DEFAULT_REG_FUNC_H__
