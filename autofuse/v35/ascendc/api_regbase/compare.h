/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_COMPARE_H__
#define __ASCENDC_API_REGBASE_COMPARE_H__

using namespace AscendC;
template <typename InT, CMPMODE mode, bool isScalar, const MicroAPI::RegTrait &regTraitNum = MicroAPI::RegTraitNumOne>
__simd_vf__ inline void CompareNormal2DVecImpl(__ubuf__ uint8_t *dst, __ubuf__ InT *src0, __ubuf__ InT *src1,
                                               const uint16_t dstStride, const uint16_t srcStride,
                                               const uint16_t repeatTime, const uint16_t counterFirst,
                                               uint32_t counterTail, InT scalar, uint16_t vlSize) {
  uint32_t mainBlockCount = GetVecLen() / sizeof(InT);
  if constexpr (sizeof(InT) == 8) {
    mainBlockCount = 2 * GetVecLen() / sizeof(InT);
  }
  MicroAPI::RegTensor<InT, regTraitNum> src0Reg, src1Reg;
  MicroAPI::MaskReg fullMask, tailMask, dstMask;
  MicroAPI::RegTensor<uint8_t> dstReg, oneAllReg, zeroAllReg;
  fullMask = MicroAPI::CreateMask<uint8_t>();
  MicroAPI::Duplicate(oneAllReg, 1);
  MicroAPI::Duplicate(zeroAllReg, 0);
  if constexpr (isScalar) {
    MicroAPI::Duplicate(src1Reg, scalar);
  }
  MicroAPI::MaskReg mainBlockMask = MicroAPI::UpdateMask<uint8_t>(mainBlockCount);
  MicroAPI::MaskReg tailBlockMask = MicroAPI::UpdateMask<uint8_t>(counterTail);
  for (uint16_t j = 0U; j < counterFirst; ++j) {
    // mainBlock
    for (uint16_t i = 0U; i < repeatTime; ++i) {
      MicroAPI::DataCopy(src0Reg, src0 + j * srcStride + i * vlSize);
      if constexpr (!isScalar) {
        MicroAPI::DataCopy(src1Reg, src1 + j * srcStride + i * vlSize);
      }
      MicroAPI::Compare<InT, mode>(dstMask, src0Reg, src1Reg, fullMask);
      if constexpr (sizeof(InT) == 2) {
        MicroAPI::MaskPack(dstMask, dstMask);
      } else if constexpr (sizeof(InT) == 4 || sizeof(InT) == 8) {
        MicroAPI::MaskPack(dstMask, dstMask);
        MicroAPI::MaskPack(dstMask, dstMask);
      }
      MicroAPI::Select(dstReg, oneAllReg, zeroAllReg, dstMask);
      MicroAPI::DataCopy(dst + j * dstStride + i * vlSize, dstReg, mainBlockMask);
    }
    // tailBlock
    MicroAPI::DataCopy(src0Reg, src0 + j * srcStride + repeatTime * vlSize);
    if constexpr (!isScalar) {
      MicroAPI::DataCopy(src1Reg, src1 + j * srcStride + repeatTime * vlSize);
    }
    MicroAPI::Compare<InT, mode>(dstMask, src0Reg, src1Reg, fullMask);
    if constexpr (sizeof(InT) == 2) {
      MicroAPI::MaskPack(dstMask, dstMask);
    } else if constexpr (sizeof(InT) == 4 || sizeof(InT) == 8) {
      MicroAPI::MaskPack(dstMask, dstMask);
      MicroAPI::MaskPack(dstMask, dstMask);
    }
    MicroAPI::Select(dstReg, oneAllReg, zeroAllReg, dstMask);
    MicroAPI::DataCopy(dst + j * dstStride + repeatTime * vlSize, dstReg, tailBlockMask);
  }
}

template <typename OutT>
__aicore__ inline LocalTensor<uint8_t> GetCompareUint8Output(const LocalTensor<OutT> &dst) {
  if constexpr (Std::is_same<OutT, uint8_t>::value) {
    return dst;
  } else {
    return dst.template ReinterpretCast<uint8_t>();
  }
}

template <typename InT, uint8_t dim, CMPMODE mode, typename OutT>
__aicore__ inline void CompareExtend(const LocalTensor<OutT> &dst, const LocalTensor<InT> &src0,
                                     const LocalTensor<InT> &src1, const uint16_t (&output_dims)[dim],
                                     const uint16_t (&output_stride)[dim], const uint16_t (&input_stride)[dim]) {
  static_assert((dim == 1) || (dim == 2), "CompareExtend only support dim=1 or dim=2");
  auto dstUint8 = GetCompareUint8Output(dst);
  bool src1IsScalar = false;
  InT scalar = 0;

  uint16_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(InT));
  if constexpr (sizeof(InT) == 8) {
    vlSize = static_cast<uint32_t>(2 * GetVecLen() / sizeof(InT));
  }
  if ((dim == 2) && (src1.GetSize() * sizeof(InT) == 32)) {
    src1IsScalar = true;
    scalar = src1.GetValue(0);
  }
  __ubuf__ uint8_t *dstLocal = (__ubuf__ uint8_t *)dstUint8.GetPhyAddr();
  __ubuf__ InT *src0Local = (__ubuf__ InT *)src0.GetPhyAddr();
  __ubuf__ InT *src1Local = (__ubuf__ InT *)src1.GetPhyAddr();
  const uint16_t dstStride = output_stride[0];
  const uint16_t srcStride = input_stride[0];
  uint16_t counterFirst = dim == 1 ? 1 : output_dims[0];
  uint16_t repeat = output_dims[dim - 1] / vlSize;
  uint32_t counterTail = output_dims[dim - 1] - repeat * vlSize;
  if ((counterTail == 0) && (repeat > 0)) {
    repeat--;
    counterTail += vlSize;
  }
  if (src1IsScalar) {
    if constexpr (sizeof(InT) == 8) {
      CompareNormal2DVecImpl<InT, mode, true, MicroAPI::RegTraitNumTwo>(
          dstLocal, src0Local, src1Local, dstStride, srcStride, repeat, counterFirst, counterTail, scalar, vlSize);
    } else {
      CompareNormal2DVecImpl<InT, mode, true>(dstLocal, src0Local, src1Local, dstStride, srcStride, repeat,
                                              counterFirst, counterTail, scalar, vlSize);
    }
  } else {
    if constexpr (sizeof(InT) == 8) {
      CompareNormal2DVecImpl<InT, mode, false, MicroAPI::RegTraitNumTwo>(
          dstLocal, src0Local, src1Local, dstStride, srcStride, repeat, counterFirst, counterTail, scalar, vlSize);
    } else {
      CompareNormal2DVecImpl<InT, mode, false>(dstLocal, src0Local, src1Local, dstStride, srcStride, repeat,
                                               counterFirst, counterTail, scalar, vlSize);
    }
  }
}

template <typename InT, uint8_t dim, CMPMODE mode, typename OutT>
__aicore__ inline void CompareScalarExtend(const LocalTensor<OutT> &dst, const LocalTensor<InT> &src0,
                                           const InT srcScalar, const uint16_t (&output_dims)[dim],
                                           const uint16_t (&output_stride)[dim], const uint16_t (&input_stride)[dim]) {
  static_assert((dim == 1) || (dim == 2), "CompareExtend only support dim=1 or dim=2");
  auto dstUint8 = GetCompareUint8Output(dst);
  uint16_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(InT));
  if constexpr (sizeof(InT) == 8) {
    vlSize = static_cast<uint32_t>(2 * GetVecLen() / sizeof(InT));
  }
  __ubuf__ uint8_t *dstLocal = (__ubuf__ uint8_t *)dstUint8.GetPhyAddr();
  __ubuf__ InT *src0Local = (__ubuf__ InT *)src0.GetPhyAddr();
  __ubuf__ InT *src1Local = (__ubuf__ InT *)src0.GetPhyAddr();
  const uint16_t dstStride = output_stride[0];
  const uint16_t srcStride = input_stride[0];
  uint16_t counterFirst = dim == 1 ? 1 : output_dims[0];
  uint16_t repeat = output_dims[dim - 1] / vlSize;
  uint32_t counterTail = output_dims[dim - 1] - repeat * vlSize;
  if ((counterTail == 0) && (repeat > 0)) {
    repeat--;
    counterTail += vlSize;
  }
  if constexpr (sizeof(InT) == 8) {
    CompareNormal2DVecImpl<InT, mode, true, MicroAPI::RegTraitNumTwo>(
        dstLocal, src0Local, src1Local, dstStride, srcStride, repeat, counterFirst, counterTail, srcScalar, vlSize);
  } else {
    CompareNormal2DVecImpl<InT, mode, true>(dstLocal, src0Local, src1Local, dstStride, srcStride, repeat, counterFirst,
                                            counterTail, srcScalar, vlSize);
  }
}
#endif  //__ASCENDC_API_REGBASE__COMPARE_H__
