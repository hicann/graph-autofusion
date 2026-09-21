/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_SILU_H__
#define __ASCENDC_API_REGBASE_SILU_H__

/**
 * @brief SiluExtend - compute the Sigmoid Linear Unit (Swish) activation: y = x * sigmoid(x)
 *
 * Mathematical formula:
 *   y = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Composed from built-in vector APIs:
 *   1. tmp = -x          (Muls)
 *   2. tmp = exp(tmp)    (Exp, in-place)
 *   3. tmp = 1 + tmp     (Adds, in-place)  => tmp = 1 + exp(-x)
 *   4. tmp = 1 / tmp     (Reciprocal, in-place)  => tmp = sigmoid(x)
 *   5. dst = x * tmp     (Mul)
 *
 * (Div instead of Reciprocal+Mul was tried but reverted: on the dav-3510
 * simulator the regular Div exposes the simulator's Exp inaccuracy for
 * |x| ~ 92 — Exp does not overflow to +Inf, so Div(x, finite) yields a tiny
 * non-zero where the reference is 0. Reciprocal's subnormal intermediate is
 * FTZ-flushed to 0, matching the reference. On real hardware Exp overflows
 * correctly and Div would be fine, but verification here is cleaner with
 * Reciprocal + Mul.)
 *
 * Precision note (half):
 *   exp() computed directly in half overflows for |x| > ~11.1 (half max is
 *   65504, exp(11.1) ~= 66000), which makes sigmoid saturate to 0 prematurely
 *   and diverges from the float32 reference (which keeps the small but
 *   representable value). To match the golden reference (computed in float32
 *   then rounded to half), the half path casts to float, performs the whole
 *   silu computation in float, then casts back to half. This mirrors what the
 *   rsqrt/half path does.
 *
 * Edge cases (handled by composed built-in APIs):
 *   - x == +Inf  -> +Inf
 *   - x == -Inf  -> NaN (the final Mul yields -Inf * 0 = NaN in IEEE 754)
 *   - x == NaN   -> NaN
 *
 * @tparam T data type, supports float, half
 * @param dst output tensor
 * @param src input tensor
 * @param tmp_buf temporary buffer:
 *                - float: at least size * sizeof(float) bytes
 *                - half:  at least AlignUp(size * sizeof(float), ONE_BLK_SIZE) +
 *                         size * sizeof(float) bytes (two float intermediates:
 *                         x and sigmoid, with the second 32-byte aligned)
 * @param size number of elements to compute
 */
template <typename T>
inline __aicore__ void SiluExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                  AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t size) {
  if constexpr (std::is_same<T, float>::value) {
    AscendC::LocalTensor<float> tmp = tmp_buf.template ReinterpretCast<float>();
    AscendC::Muls(tmp, src, -1.0f, size);
    AscendC::Exp(tmp, tmp, size);
    AscendC::Adds(tmp, tmp, 1.0f, size);
    AscendC::Reciprocal(tmp, tmp, size);
    AscendC::Mul(dst, src, tmp, size);
  } else {
    // half: cast to float, compute silu in float, cast back.
    uint32_t fbytes = AscendC::AlignUp(size * static_cast<uint32_t>(sizeof(float)), ONE_BLK_SIZE);
    AscendC::LocalTensor<float> xF = tmp_buf.template ReinterpretCast<float>();
    AscendC::LocalTensor<float> sig = tmp_buf[fbytes].template ReinterpretCast<float>();
    AscendC::Cast(xF, src, AscendC::RoundMode::CAST_NONE, size);
    AscendC::Muls(sig, xF, -1.0f, size);
    AscendC::Exp(sig, sig, size);
    AscendC::Adds(sig, sig, 1.0f, size);
    AscendC::Reciprocal(sig, sig, size);
    AscendC::Mul(xF, xF, sig, size);
    AscendC::Cast(dst, xF, AscendC::RoundMode::CAST_NONE, size);
  }
}

#endif  // __ASCENDC_API_REGBASE_SILU_H__
