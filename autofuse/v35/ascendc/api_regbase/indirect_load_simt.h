/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMT_H_
#define AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMT_H_

namespace AscendC {
namespace Internal {
__aicore__ inline uint64_t IndirectLoadGetUintDivMagic(uint64_t dividend, uint64_t divisor) {
  uint64_t quotient = 0U;
  uint64_t remainder = dividend;
  constexpr uint64_t kHighestBit = 1ULL << 63U;
  for (uint32_t bit = 0U; bit < 64U; ++bit) {
    quotient <<= 1U;
    const bool borrow = (remainder & kHighestBit) != 0U;
    remainder <<= 1U;
    if (borrow) {
      remainder = ~uint64_t{0U} - divisor + 1U + remainder;
      quotient |= 1U;
    } else if (remainder >= divisor) {
      remainder -= divisor;
      quotient |= 1U;
    }
  }
  return quotient + 1U;
}

template <typename T>
__aicore__ inline void IndirectLoadGetUintDivMagicAndShift(T &magic, T &shift, T divisor) {
  static_assert(std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value,
                "IndirectLoad SIMT division only supports uint32_t and uint64_t.");
  uint32_t position = 0U;
  for (T value = divisor; value != 0U; value >>= 1U) {
    ++position;
  }
  shift = static_cast<T>((divisor & (divisor - 1U)) == 0U ? position - 1U : position);
  if constexpr (std::is_same<T, uint32_t>::value) {
    magic = static_cast<T>((1ULL << 32U) * ((1ULL << shift) - divisor) / divisor + 1U);
  } else {
    const uint64_t dividend = shift < 64U ? (1ULL << shift) - divisor : ~uint64_t{0U} - divisor + 1U;
    magic = static_cast<T>(IndirectLoadGetUintDivMagic(dividend, divisor));
  }
}

template <uint64_t Value>
inline __aicore__ constexpr uint32_t IndirectLoadLog2() {
  static_assert(Value > 0U && (Value & (Value - 1U)) == 0U, "IndirectLoad SIMT span must be a power of two.");
  uint32_t shift = 0U;
  uint64_t value = Value;
  while (value > 1U) {
    value >>= 1U;
    ++shift;
  }
  return shift;
}

template <typename OffsetT>
struct IndirectLoadSimtAddress {
  OffsetT index_offset;
  OffsetT input_base;
};

template <int32_t Dim, typename OffsetT, typename AddressPolicy>
struct IndirectLoadSimtAddressDecoder {
  __simt_callee__ __aicore__ inline static void Call(OffsetT linear_index, const AddressPolicy &policy,
                                                     IndirectLoadSimtAddress<OffsetT> &address) {
    const OffsetT quotient = Simt::UintDiv(linear_index, policy.magic[Dim], policy.shift[Dim]);
    const OffsetT coordinate = linear_index - quotient * policy.shape[Dim];
    policy.template AddCoordinate<Dim>(coordinate, address);
    if constexpr (Dim > 0) {
      IndirectLoadSimtAddressDecoder<Dim - 1, OffsetT, AddressPolicy>::Call(quotient, policy, address);
    }
  }
};
}  // namespace Internal

template <typename OffsetT, uint64_t InnerSpan, uint64_t OutputAxisSpan, uint64_t InputAxisStride,
          uint64_t InputAxisSpan>
struct IndirectLoadSimtStaticPowerOfTwoPolicy {
  using OffsetType = OffsetT;
  static constexpr bool kStructured = true;
  static constexpr bool kUsesInputAxis = true;

  __simt_callee__ __aicore__ inline Internal::IndirectLoadSimtAddress<OffsetT> GetAddress(OffsetT output_index) const {
    constexpr uint32_t inner_shift = Internal::IndirectLoadLog2<InnerSpan>();
    constexpr uint32_t output_axis_shift = Internal::IndirectLoadLog2<OutputAxisSpan>();
    const OffsetT outer = output_index >> output_axis_shift;
    const OffsetT inner = output_index & static_cast<OffsetT>(InnerSpan - 1U);
    return {output_index, outer * static_cast<OffsetT>(InputAxisSpan) + inner};
  }

  static constexpr OffsetT input_axis_stride = static_cast<OffsetT>(InputAxisStride);
};

template <typename OffsetT, uint64_t InnerSpan, uint64_t InputAxisStride, uint64_t InputAxisSpan>
struct IndirectLoadSimtStaticInnerPolicy {
  using OffsetType = OffsetT;
  static constexpr bool kStructured = true;
  static constexpr bool kUsesInputAxis = true;

  __aicore__ explicit IndirectLoadSimtStaticInnerPolicy(OffsetT output_axis_span) {
    Internal::IndirectLoadGetUintDivMagicAndShift(output_axis_magic, output_axis_shift, output_axis_span);
  }

  __simt_callee__ __aicore__ inline Internal::IndirectLoadSimtAddress<OffsetT> GetAddress(OffsetT output_index) const {
    constexpr uint32_t inner_shift = Internal::IndirectLoadLog2<InnerSpan>();
    const OffsetT outer = Simt::UintDiv(output_index, output_axis_magic, output_axis_shift);
    const OffsetT inner = output_index & static_cast<OffsetT>(InnerSpan - 1U);
    return {output_index, outer * static_cast<OffsetT>(InputAxisSpan) + inner};
  }

  static constexpr OffsetT input_axis_stride = static_cast<OffsetT>(InputAxisStride);
  OffsetT output_axis_magic{0U};
  OffsetT output_axis_shift{0U};
};

template <typename OffsetT>
struct IndirectLoadSimtStructuredMagicPolicy {
  using OffsetType = OffsetT;
  static constexpr bool kStructured = true;
  static constexpr bool kUsesInputAxis = true;

  __aicore__ explicit IndirectLoadSimtStructuredMagicPolicy(OffsetT inner_span_arg, OffsetT output_axis_span_arg,
                                                            OffsetT input_axis_stride_arg, OffsetT input_axis_span_arg)
      : inner_span(inner_span_arg), input_axis_stride(input_axis_stride_arg), input_axis_span(input_axis_span_arg) {
    Internal::IndirectLoadGetUintDivMagicAndShift(inner_magic, inner_shift, inner_span);
    Internal::IndirectLoadGetUintDivMagicAndShift(output_axis_magic, output_axis_shift, output_axis_span_arg);
  }

  __simt_callee__ __aicore__ inline Internal::IndirectLoadSimtAddress<OffsetT> GetAddress(OffsetT output_index) const {
    const OffsetT outer = Simt::UintDiv(output_index, output_axis_magic, output_axis_shift);
    const OffsetT inner_quotient = Simt::UintDiv(output_index, inner_magic, inner_shift);
    const OffsetT inner = output_index - inner_quotient * inner_span;
    return {output_index, outer * input_axis_span + inner};
  }

  OffsetT inner_span;
  OffsetT input_axis_stride;
  OffsetT input_axis_span;
  OffsetT inner_magic{0U};
  OffsetT inner_shift{0U};
  OffsetT output_axis_magic{0U};
  OffsetT output_axis_shift{0U};
};

template <typename OffsetT, int32_t Rank, int32_t Axis>
struct IndirectLoadSimtRecursivePolicy {
  using OffsetType = OffsetT;
  static constexpr bool kStructured = false;
  static constexpr bool kUsesInputAxis = true;

  template <typename... ShapeArgs>
  __aicore__ explicit IndirectLoadSimtRecursivePolicy(ShapeArgs... shape_args)
      : shape{static_cast<OffsetT>(shape_args)...} {
    static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMT rank or axis is invalid.");
    static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(2 * Rank), "IndirectLoad SIMT shape is invalid.");
    for (int32_t dim = 0; dim < Rank; ++dim) {
      Internal::IndirectLoadGetUintDivMagicAndShift(magic[dim], shift[dim], shape[dim]);
    }
    input_axis_stride = shape[Rank + Axis];
  }

  __simt_callee__ __aicore__ inline Internal::IndirectLoadSimtAddress<OffsetT> GetAddress(OffsetT output_index) const {
    Internal::IndirectLoadSimtAddress<OffsetT> address{output_index, 0U};
    Internal::IndirectLoadSimtAddressDecoder<Rank - 1, OffsetT, IndirectLoadSimtRecursivePolicy>::Call(output_index,
                                                                                                       *this, address);
    return address;
  }

  template <int32_t Dim>
  __simt_callee__ __aicore__ inline void AddCoordinate(OffsetT coordinate,
                                                       Internal::IndirectLoadSimtAddress<OffsetT> &address) const {
    if constexpr (Dim != Axis) {
      address.input_base += coordinate * shape[Rank + Dim];
    }
  }

  OffsetT shape[2 * Rank];
  OffsetT magic[Rank];
  OffsetT shift[Rank];
  OffsetT input_axis_stride{0U};
};

template <typename OffsetT, int32_t Rank, int32_t Axis, uint64_t InputStrideMask, uint64_t IndexStrideMask>
struct IndirectLoadSimtStridedPolicy {
  using OffsetType = OffsetT;
  static constexpr bool kStructured = false;
  static constexpr bool kUsesInputAxis = (InputStrideMask & (1ULL << Axis)) != 0U;

  template <typename... ShapeArgs>
  __aicore__ explicit IndirectLoadSimtStridedPolicy(ShapeArgs... shape_args)
      : shape{static_cast<OffsetT>(shape_args)...} {
    static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMT rank or axis is invalid.");
    static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(3 * Rank), "IndirectLoad SIMT shape is invalid.");
    for (int32_t dim = 0; dim < Rank; ++dim) {
      Internal::IndirectLoadGetUintDivMagicAndShift(magic[dim], shift[dim], shape[dim]);
    }
    input_axis_stride = shape[Rank + Axis];
  }

  __simt_callee__ __aicore__ inline Internal::IndirectLoadSimtAddress<OffsetT> GetAddress(OffsetT output_index) const {
    Internal::IndirectLoadSimtAddress<OffsetT> address{0U, 0U};
    if constexpr (Rank == 3 && Axis == 1 && IndexStrideMask == 2U) {
      const OffsetT axis_and_inner = Simt::UintDiv(output_index, magic[2], shift[2]);
      const OffsetT inner = output_index - axis_and_inner * shape[2];
      const OffsetT outer = Simt::UintDiv(axis_and_inner, magic[1], shift[1]);
      const OffsetT axis = axis_and_inner - outer * shape[1];
      if constexpr ((InputStrideMask & 1U) != 0U) {
        address.input_base += outer * shape[3];
      }
      if constexpr ((InputStrideMask & 4U) != 0U) {
        address.input_base += inner * shape[5];
      }
      address.index_offset = axis * shape[7];
      return address;
    }
    if constexpr (Rank == 2 && Axis == 0 && IndexStrideMask == 1U) {
      const OffsetT row = Simt::UintDiv(output_index, magic[1], shift[1]);
      const OffsetT column = output_index - row * shape[1];
      address.index_offset = row * shape[2 * Rank];
      if constexpr ((InputStrideMask & 2U) != 0U) {
        address.input_base += column * shape[Rank + 1];
      }
      return address;
    }
    Internal::IndirectLoadSimtAddressDecoder<Rank - 1, OffsetT, IndirectLoadSimtStridedPolicy>::Call(output_index,
                                                                                                     *this, address);
    return address;
  }

  template <int32_t Dim>
  __simt_callee__ __aicore__ inline void AddCoordinate(OffsetT coordinate,
                                                       Internal::IndirectLoadSimtAddress<OffsetT> &address) const {
    if constexpr ((IndexStrideMask & (1ULL << Dim)) != 0U) {
      address.index_offset += coordinate * shape[2 * Rank + Dim];
    }
    if constexpr (Dim != Axis && (InputStrideMask & (1ULL << Dim)) != 0U) {
      address.input_base += coordinate * shape[Rank + Dim];
    }
  }

  OffsetT shape[3 * Rank];
  OffsetT magic[Rank];
  OffsetT shift[Rank];
  OffsetT input_axis_stride{0U};
};

namespace Internal {
template <typename X, typename Y, typename FusedBody, typename Context, typename AddressPolicy>
__simt_callee__ __aicore__ inline Y IndirectLoadSimtCompute(__gm__ X *x, Context context,
                                                            typename AddressPolicy::OffsetType output_index,
                                                            const AddressPolicy &address_policy) {
  using OffsetT = typename AddressPolicy::OffsetType;
  const IndirectLoadSimtAddress<OffsetT> address = address_policy.GetAddress(output_index);
  const OffsetT indirect_index = static_cast<OffsetT>(FusedBody::Index(address.index_offset, context));
  OffsetT input_offset = address.input_base;
  if constexpr (AddressPolicy::kUsesInputAxis) {
    input_offset += indirect_index * address_policy.input_axis_stride;
  }
  return FusedBody::Output(x[input_offset], output_index, context);
}

template <typename X, typename Y, typename FusedBody, typename Context, uint32_t ThreadNum, typename AddressPolicy>
__simt_vf__ __aicore__ LAUNCH_BOUND(ThreadNum) inline void IndirectLoadSimtKernel(
    __gm__ X *x, __gm__ Y *y, Context context, uint32_t actual_size, typename AddressPolicy::OffsetType output_offset,
    AddressPolicy address_policy) {
  using OffsetT = typename AddressPolicy::OffsetType;
  for (uint32_t i = threadIdx.x; i < actual_size; i += blockDim.x) {
    const OffsetT output_index = output_offset + static_cast<OffsetT>(i);
    y[output_index] = IndirectLoadSimtCompute<X, Y, FusedBody>(x, context, output_index, address_policy);
  }
}

template <typename X, typename Y, typename FusedBody, typename Context, uint32_t ThreadNum, typename AddressPolicy>
__simt_vf__ __aicore__ LAUNCH_BOUND(ThreadNum) inline void IndirectLoadSimtUbKernel(
    __gm__ X *x, __ubuf__ Y *y, Context context, uint32_t actual_size, typename AddressPolicy::OffsetType output_offset,
    AddressPolicy address_policy) {
  using OffsetT = typename AddressPolicy::OffsetType;
  for (uint32_t i = threadIdx.x; i < actual_size; i += blockDim.x) {
    const OffsetT output_index = output_offset + static_cast<OffsetT>(i);
    y[i] = IndirectLoadSimtCompute<X, Y, FusedBody>(x, context, output_index, address_policy);
  }
}

template <uint32_t ThreadNum, typename X, typename Y, typename FusedBody, typename Context, typename AddressPolicy>
__aicore__ inline void LaunchIndirectLoadSimt(__gm__ X *x, __gm__ Y *y, Context context, uint32_t actual_size,
                                              typename AddressPolicy::OffsetType output_offset,
                                              AddressPolicy address_policy) {
  Simt::VF_CALL<IndirectLoadSimtKernel<X, Y, FusedBody, Context, ThreadNum, AddressPolicy>>(
      Simt::Dim3(ThreadNum), x, y, context, actual_size, output_offset, address_policy);
}

template <uint32_t ThreadNum, typename X, typename Y, typename FusedBody, typename Context, typename AddressPolicy>
__aicore__ inline void LaunchIndirectLoadSimt(__gm__ X *x, LocalTensor<Y> y, Context context, uint32_t actual_size,
                                              typename AddressPolicy::OffsetType output_offset,
                                              AddressPolicy address_policy) {
  Simt::VF_CALL<IndirectLoadSimtUbKernel<X, Y, FusedBody, Context, ThreadNum, AddressPolicy>>(
      Simt::Dim3(ThreadNum), x, (__ubuf__ Y *)y.GetPhyAddr(), context, actual_size, output_offset, address_policy);
}

template <typename AddressPolicy>
inline __aicore__ constexpr bool IndirectLoadUse2048Threads() {
  return sizeof(typename AddressPolicy::OffsetType) == sizeof(uint32_t);
}

template <typename X, typename Y, typename FusedBody, typename Context, typename AddressPolicy, typename Output>
__aicore__ inline void DispatchIndirectLoadSimt(__gm__ X *x, Output y, Context context, uint32_t actual_size,
                                                typename AddressPolicy::OffsetType output_offset,
                                                AddressPolicy address_policy) {
  if (actual_size <= 128U) {
    LaunchIndirectLoadSimt<128U, X, Y, FusedBody>(x, y, context, actual_size, output_offset, address_policy);
    return;
  }
  if (actual_size <= 256U) {
    LaunchIndirectLoadSimt<256U, X, Y, FusedBody>(x, y, context, actual_size, output_offset, address_policy);
    return;
  }
  if (actual_size <= 512U) {
    LaunchIndirectLoadSimt<512U, X, Y, FusedBody>(x, y, context, actual_size, output_offset, address_policy);
    return;
  }
  if (actual_size <= 1024U) {
    LaunchIndirectLoadSimt<1024U, X, Y, FusedBody>(x, y, context, actual_size, output_offset, address_policy);
    return;
  }
  if constexpr (IndirectLoadUse2048Threads<AddressPolicy>()) {
    LaunchIndirectLoadSimt<2048U, X, Y, FusedBody>(x, y, context, actual_size, output_offset, address_policy);
  } else {
    LaunchIndirectLoadSimt<1024U, X, Y, FusedBody>(x, y, context, actual_size, output_offset, address_policy);
  }
}
}  // namespace Internal

template <typename X, typename Y, typename FusedBody, typename AddressPolicy, typename Context, typename... PolicyArgs>
__aicore__ inline void IndirectLoadSimt(__gm__ X *x, __gm__ Y *y, Context context, uint32_t actual_size,
                                        typename AddressPolicy::OffsetType output_offset, PolicyArgs... policy_args) {
  if (actual_size != 0U) {
    const AddressPolicy address_policy{static_cast<typename AddressPolicy::OffsetType>(policy_args)...};
    Internal::DispatchIndirectLoadSimt<X, Y, FusedBody>(x, y, context, actual_size, output_offset, address_policy);
  }
  const int32_t event_id = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
  SetFlag<HardEvent::V_MTE3>(event_id);
  WaitFlag<HardEvent::V_MTE3>(event_id);
}

template <typename X, typename Y, typename FusedBody, typename AddressPolicy, typename Context, typename... PolicyArgs>
__aicore__ inline void IndirectLoadSimt(__gm__ X *x, LocalTensor<Y> y, Context context, uint32_t actual_size,
                                        typename AddressPolicy::OffsetType output_offset, PolicyArgs... policy_args) {
  if (actual_size != 0U) {
    const AddressPolicy address_policy{static_cast<typename AddressPolicy::OffsetType>(policy_args)...};
    Internal::DispatchIndirectLoadSimt<X, Y, FusedBody>(x, y, context, actual_size, output_offset, address_policy);
  }
  PipeBarrier<PIPE_V>();
}

}  // namespace AscendC

#endif  // AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMT_H_
