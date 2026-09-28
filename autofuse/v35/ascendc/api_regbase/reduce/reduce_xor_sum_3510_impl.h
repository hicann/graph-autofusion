/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_xor_sum_3510_impl.h
 * \brief
 */

#ifndef AUTOFUSE_REDUCE_EXTEND_REDUCE_XOR_SUM_C310_IMPL_H
#define AUTOFUSE_REDUCE_EXTEND_REDUCE_XOR_SUM_C310_IMPL_H
#include "kernel_tensor.h"
#include "kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
namespace ReduceExtendInternal {
namespace ReduceXorSumAPI {

constexpr Reg::CastTrait castTraitF16F32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                            RoundMode::UNKNOWN};
constexpr Reg::CastTrait castTraitF32F16 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING,
                                            RoundMode::CAST_RINT};

template <typename T>
__simd_callee__ inline void LoadSrcDataAndXor(Reg::RegTensor<float> &srcReg, __ubuf__ T *src0, __ubuf__ T *src1,
                                              uint16_t index, Reg::MaskReg &mask) {
  constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(float);
  Reg::RegTensor<T> src0TmpReg, src1TmpReg;
  Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(src0TmpReg, src0 + index * eleCountPerVL);
  Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(src1TmpReg, src1 + index * eleCountPerVL);
  Reg::Xor(src0TmpReg, src0TmpReg, src1TmpReg, mask);
  Reg::Cast<float, T, castTraitF16F32>(srcReg, src0TmpReg, mask);
}

template <typename T>
__simd_vf__ inline void ReduceXorSumCoreImpl(__ubuf__ T *dstUb, __ubuf__ T *src0Ub, __ubuf__ T *src1Ub,
                                             uint32_t calCount) {
  Reg::MaskReg mask;
  Reg::UnalignReg ureg;
  Reg::RegTensor<T> dstReg;
  Reg::RegTensor<float> srcTmpReg, dstTmpReg, tmpReg, zeroReg;
  Reg::MaskReg fullMask = Reg::CreateMask<float>();
  constexpr int32_t eleCountPerVL = GetVecLen() / sizeof(float);
  uint16_t repeatTimes = CeilDivision(calCount, eleCountPerVL);
  Reg::Duplicate(zeroReg, 0, fullMask);
  for (uint16_t i = 0; i < repeatTimes; i++) {
    mask = Reg::UpdateMask<float>(calCount);
    LoadSrcDataAndXor(srcTmpReg, src0Ub, src1Ub, i, mask);
    Reg::ReduceSum(dstTmpReg, srcTmpReg, mask);
    Reg::Add(zeroReg, zeroReg, dstTmpReg, mask);
  }
  Reg::Cast<T, float, castTraitF32F16>(dstReg, zeroReg, fullMask);
  Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t> &)dstReg,
                                                          (Reg::RegTensor<uint32_t> &)dstReg);
  Reg::StoreUnAlign(dstUb, dstReg, ureg, 1);
  Reg::StoreUnAlignPost(dstUb, ureg, 0);
}
}  // namespace ReduceXorSumAPI

template <typename T, bool isReuseSource = false>
__aicore__ inline void ReduceXorSumCheckParams(LocalTensor<T> &dstTensor, const LocalTensor<T> &src0Tensor,
                                               const LocalTensor<T> &src1Tensor, LocalTensor<uint8_t> &sharedTmpBuffer,
                                               const uint32_t calCount) {
  static_assert(std::is_same<T, int16_t>::value, "ReduceXorSum only support int16_t data type on current device!");
  CheckTensorPos<T>(dstTensor, Hardware::UB, "dstTensor", "VECIN/VECCALC/VECOUT", "ReduceXorSum");
  CheckTensorPos<T>(src0Tensor, Hardware::UB, "src0Tensor", "VECIN/VECCALC/VECOUT", "ReduceXorSum");
  CheckTensorPos<T>(src1Tensor, Hardware::UB, "src1Tensor", "VECIN/VECCALC/VECOUT", "ReduceXorSum");
  CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN/VECCALC/VECOUT", "ReduceXorSum");
  CheckCalCount(calCount, "calCount", src0Tensor, "src0Tensor", "ReduceXorSum");
  CheckCalCount(calCount, "calCount", src1Tensor, "src1Tensor", "ReduceXorSum");
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void ReduceXorSumCompute(LocalTensor<T> &dstTensor, const LocalTensor<T> &src0Tensor,
                                           const LocalTensor<T> &src1Tensor, LocalTensor<uint8_t> &sharedTmpBuffer,
                                           const uint32_t calCount) {
  // Only for AI Vector Core.
  if ASCEND_IS_AIC {
    return;
  }

  ReduceXorSumCheckParams<T, isReuseSource>(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
  __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
  __ubuf__ T *src0Ub = (__ubuf__ T *)src0Tensor.GetPhyAddr();
  __ubuf__ T *src1Ub = (__ubuf__ T *)src1Tensor.GetPhyAddr();
  ReduceXorSumAPI::ReduceXorSumCoreImpl<T>(dstUb, src0Ub, src1Ub, calCount);
}
}  // namespace ReduceExtendInternal
}  // namespace AscendC

#endif  // AUTOFUSE_REDUCE_EXTEND_REDUCE_XOR_SUM_C310_IMPL_H
