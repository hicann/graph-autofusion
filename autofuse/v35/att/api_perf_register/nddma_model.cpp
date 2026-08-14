/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nddma_model.h"

#include <limits>
#include <unordered_set>
#include "base/att_const_values.h"
#include "common_utils.h"
#include "framework/common/debug/ge_log.h"

namespace att {
namespace {
constexpr size_t kMinNddmaRank = 1U;
constexpr size_t kMaxNddmaRank = 5U;
constexpr uint64_t kLowCoreUpperBound = 2U;
constexpr uint64_t kInputStrideUpperBound = 128U;

// 合并模型直接表示最终多项式，避免运行时重新组合 NG、NGM 和 rho。
struct NddmaLowCoreParams {
  double constant;
  double bytes;
  double stride;
  double bytes_stride;
};

struct NddmaHighCoreParams {
  double constant;
  double stride;
  double stride_square;
  double bytes;
  double bytes_stride;
  double bytes_stride_square;
};

struct Nddma1DParams {
  NddmaLowCoreParams low_os_one;
  NddmaLowCoreParams low_os_ge_two;
  NddmaHighCoreParams high_os_one;
  NddmaHighCoreParams high_os_ge_two;
};

const Nddma1DParams *GetNddma1DParams(uint64_t dtype_size) {
  static constexpr Nddma1DParams kB8 = {
      {194.421, 0.08501521772, 1.1117457, 0.0081160848},
      {53.81133, 6.998497418, 0.26072115, -0.0000079948},
      {373.274, 1.673508468, 0.008698207194, 0.1650886939, 0.01221712541, 0.00006349958196},
      {94.1599102, 0.5169452208, 0.000001100449223, 13.88854236, 0.00001331045708, -0.00000000003374437189}};
  static constexpr Nddma1DParams kB16 = {
      {204.604, 0.0388230407, 0.96094416, 0.0091341897},
      {61.53027, 3.498120341, 0.34357215, -0.0000114298},
      {399.909, 2.009597249, -0.007116959052, 0.07542046912, 0.01910209069, -0.00006764977277},
      {116.1609008, 0.6810487439, 0.0000008016156689, 6.936002307, -0.00001459675784, -0.00000000002666778076}};
  static constexpr Nddma1DParams kB32 = {
      {235.137, 0.01746346643, 0.3675047, 0.0058486918},
      {158.723147, 0.2427621164, 1.4596785, 0.0013874442},
      {453.859, 1.420515439, -0.008545279188, 0.03400250258, 0.02260694081, -0.0001359947351},
      {323.0812754, 0.7652737348, 0.03310192195, 0.4195876242, 0.007483747325, 0.00003146382551}};
  static constexpr Nddma1DParams kB64 = {
      {243.205, 0.01747194879, 0.90051511, 0.0030237371},
      {189.027128, 0.1244607288, 1.98330501, 0.0007092839},
      {468.971, 3.366949543, -0.02124182228, 0.03402448402, 0.0113054963, -0.00007132549515},
      {361.5012267, 4.200466405, -0.009748131943, 0.2462524565, 0.0008811088948, -0.000003486197538}};
  switch (dtype_size) {
    case 1U:
      return &kB8;
    case 2U:
      return &kB16;
    case 4U:
      return &kB32;
    case 8U:
      return &kB64;
    default:
      return nullptr;
  }
}

bool HasValidVectorLengths(const NddmaDescriptorInfo &descriptor) {
  const size_t rank = descriptor.output_dims.size();
  return rank == descriptor.input_strides.size() && rank == descriptor.output_strides.size();
}

bool HasValidAxisOrder(const NddmaDescriptorInfo &descriptor) {
  if (descriptor.vectorized_axis.size() != descriptor.output_dims.size()) {
    return false;
  }
  std::unordered_set<int64_t> unique_axes;
  for (const auto axis : descriptor.vectorized_axis) {
    if (!unique_axes.emplace(axis).second) {
      return false;
    }
  }
  return true;
}

bool IsStaticNonPositive(const Expr &expr) {
  return expr.IsConstExpr() && af::SymbolicUtils::StaticCheckLe(expr, af::sym::kSymbolZero) == af::TriBool::kTrue;
}

bool IsStaticNegative(const Expr &expr) {
  return expr.IsConstExpr() && af::SymbolicUtils::StaticCheckLt(expr, af::sym::kSymbolZero) == af::TriBool::kTrue;
}

bool HasInvalidStaticValue(const NddmaNormalizedDesc &descriptor) {
  for (const auto &dim : descriptor.output_dims) {
    if (!IsValid(dim) || IsStaticNonPositive(dim)) {
      return true;
    }
  }
  for (const auto &stride : descriptor.input_strides) {
    if (!IsValid(stride) || IsStaticNegative(stride)) {
      return true;
    }
  }
  for (const auto &stride : descriptor.output_strides) {
    if (!IsValid(stride) || IsStaticNonPositive(stride)) {
      return true;
    }
  }
  return false;
}

bool GetDtypeSize(const std::string &dtype, uint64_t &dtype_size) {
  const auto iter = kDataTypeSizeMap.find(dtype);
  return iter != kDataTypeSizeMap.end() && iter->second.GetConstValue(dtype_size);
}

bool HasStaticByteCountOverflow(const Expr &dim, uint64_t dtype_size) {
  if (dtype_size == 0U) {
    return false;
  }
  int64_t dim_value = 0;
  if (!dim.IsConstExpr() || !dim.GetConstValue(dim_value) || dim_value <= 0) {
    return false;
  }
  return static_cast<uint64_t>(dim_value) > std::numeric_limits<uint64_t>::max() / dtype_size;
}

Expr BuildLowCore(const NddmaLowCoreParams &params, const Expr &bytes, const Expr &stride) {
  return CreateExpr(params.constant) + CreateExpr(params.bytes) * bytes + CreateExpr(params.stride) * stride +
         CreateExpr(params.bytes_stride) * bytes * stride;
}

Expr BuildHighCore(const NddmaHighCoreParams &params, const Expr &bytes, const Expr &stride) {
  const Expr stride_square = stride * stride;
  return CreateExpr(params.constant) + CreateExpr(params.stride) * stride +
         CreateExpr(params.stride_square) * stride_square +
         bytes * (CreateExpr(params.bytes) + CreateExpr(params.bytes_stride) * stride +
                  CreateExpr(params.bytes_stride_square) * stride_square);
}

Expr SelectOutputStrideModel(const Expr &output_stride, const Expr &os_one, const Expr &os_ge_two) {
  if (output_stride.IsConstExpr()) {
    if (af::SymbolicUtils::StaticCheckEq(output_stride, af::sym::kSymbolOne) == af::TriBool::kTrue) {
      return os_one;
    }
    if (af::SymbolicUtils::StaticCheckGt(output_stride, af::sym::kSymbolOne) == af::TriBool::kTrue) {
      return os_ge_two;
    }
  }
  const Expr output_stride_gate =
      af::sym::Max(af::sym::kSymbolZero, af::sym::Min(af::sym::kSymbolOne, output_stride - af::sym::kSymbolOne));
  return os_one + output_stride_gate * (os_ge_two - os_one);
}

void BuildNddma1DBranches(const NddmaNormalizedDesc &descriptor, uint64_t dtype_size, const Nddma1DParams &params,
                          Expr &low_core, Expr &high_core) {
  const Expr bytes = descriptor.output_dims[0] * CreateExpr(dtype_size);
  const Expr input_stride = af::sym::Min(CreateExpr(kInputStrideUpperBound), descriptor.input_strides[0]);
  const Expr low_os_one = BuildLowCore(params.low_os_one, bytes, input_stride);
  const Expr low_os_ge_two = BuildLowCore(params.low_os_ge_two, bytes, input_stride);
  const Expr high_os_one = BuildHighCore(params.high_os_one, bytes, input_stride);
  const Expr high_os_ge_two = BuildHighCore(params.high_os_ge_two, bytes, input_stride);
  low_core = SelectOutputStrideModel(descriptor.output_strides[0], low_os_one, low_os_ge_two);
  high_core = SelectOutputStrideModel(descriptor.output_strides[0], high_os_one, high_os_ge_two);
}

af::Status SelectCoreBranch(const Expr &block_dim, const Expr &low_core, const Expr &high_core,
                            NddmaModelResult &result) {
  if (af::SymbolicUtils::StaticCheckLe(block_dim, CreateExpr(kLowCoreUpperBound)) == af::TriBool::kTrue) {
    result.cycles = low_core;
    return af::SUCCESS;
  }
  if (af::SymbolicUtils::StaticCheckGt(block_dim, CreateExpr(kLowCoreUpperBound)) == af::TriBool::kTrue) {
    result.cycles = high_core;
    return af::SUCCESS;
  }
  auto low_case = std::make_shared<IfCase>(low_core);
  auto high_case = std::make_shared<IfCase>(high_core);
  GE_ASSERT_NOTNULL(low_case);
  GE_ASSERT_NOTNULL(high_case);
  GetPerfVar("nddma_1d_multicore", result.cycles, result.ternary_ops);
  TernaryOp ternary(CondType::K_LE, block_dim, CreateExpr(kLowCoreUpperBound), std::move(low_case),
                    std::move(high_case));
  ternary.SetVariable(result.cycles);
  result.ternary_ops[result.cycles] = std::move(ternary);
  return af::SUCCESS;
}
}  // namespace

const char *NddmaFallbackReasonToString(NddmaFallbackReason reason) {
  static constexpr const char *kReasons[] = {
      "none",           "no_descriptor",    "rank_unsupported",   "schema_mismatch", "dtype_unsupported",
      "stride_invalid", "codegen_mismatch", "no_registered_model"};
  const auto index = static_cast<size_t>(reason);
  return index < (sizeof(kReasons) / sizeof(kReasons[0])) ? kReasons[index] : "unknown";
}

NddmaFallbackReason BuildNddmaDescriptor(const TensorShapeInfo &shape_info, const std::vector<int64_t> &vectorized_axis,
                                         NddmaDescriptorInfo &descriptor) {
  descriptor = NddmaDescriptorInfo{};
  if (shape_info.repeats.empty()) {
    return NddmaFallbackReason::kNoDescriptor;
  }
  if (shape_info.repeats.size() != shape_info.gm_strides.size() ||
      shape_info.repeats.size() != shape_info.strides.size()) {
    return NddmaFallbackReason::kSchemaMismatch;
  }
  descriptor.output_dims = shape_info.repeats;
  descriptor.input_strides = shape_info.gm_strides;
  descriptor.output_strides = shape_info.strides;
  if (vectorized_axis.empty()) {
    return NddmaFallbackReason::kCodegenMismatch;
  }
  descriptor.vectorized_axis = vectorized_axis;
  return NddmaFallbackReason::kNone;
}

NddmaFallbackReason NormalizeNddmaDescriptor(const NddmaDescriptorInfo &descriptor, NddmaNormalizedDesc &normalized) {
  normalized = NddmaNormalizedDesc{};
  normalized.raw_rank = descriptor.output_dims.size();
  normalized.effective_rank = normalized.raw_rank;
  if (normalized.raw_rank < kMinNddmaRank || normalized.raw_rank > kMaxNddmaRank) {
    return NddmaFallbackReason::kRankUnsupported;
  }
  if (!HasValidVectorLengths(descriptor)) {
    return NddmaFallbackReason::kSchemaMismatch;
  }
  if (!HasValidAxisOrder(descriptor)) {
    return NddmaFallbackReason::kCodegenMismatch;
  }
  normalized.output_dims = descriptor.output_dims;
  normalized.input_strides = descriptor.input_strides;
  normalized.output_strides = descriptor.output_strides;
  normalized.vectorized_axis = descriptor.vectorized_axis;
  return HasInvalidStaticValue(normalized) ? NddmaFallbackReason::kStrideInvalid : NddmaFallbackReason::kNone;
}

af::Status EvaluateNddmaModel(const NddmaDescriptorInfo &descriptor, const std::string &dtype, const Expr &block_dim,
                              NddmaModelResult &result) {
  result = NddmaModelResult{};
  NddmaNormalizedDesc normalized;
  result.fallback_reason = NormalizeNddmaDescriptor(descriptor, normalized);
  result.raw_rank = normalized.raw_rank;
  result.effective_rank = normalized.effective_rank;
  if (result.fallback_reason != NddmaFallbackReason::kNone) {
    return af::SUCCESS;
  }
  if (normalized.raw_rank != 1U || normalized.effective_rank != 1U) {
    result.fallback_reason = NddmaFallbackReason::kNoRegisteredModel;
    return af::SUCCESS;
  }
  uint64_t dtype_size = 0U;
  if (!GetDtypeSize(dtype, dtype_size)) {
    result.fallback_reason = NddmaFallbackReason::kDtypeUnsupported;
    return af::SUCCESS;
  }
  const auto *params = GetNddma1DParams(dtype_size);
  if (params == nullptr || IsStaticNonPositive(block_dim)) {
    result.fallback_reason =
        params == nullptr ? NddmaFallbackReason::kDtypeUnsupported : NddmaFallbackReason::kSchemaMismatch;
    return af::SUCCESS;
  }
  if (HasStaticByteCountOverflow(normalized.output_dims[0], dtype_size)) {
    result.fallback_reason = NddmaFallbackReason::kSchemaMismatch;
    return af::SUCCESS;
  }
  Expr low_core;
  Expr high_core;
  BuildNddma1DBranches(normalized, dtype_size, *params, low_core, high_core);
  GE_ASSERT_SUCCESS(SelectCoreBranch(block_dim, low_core, high_core, result));
  result.selected = true;
  result.fallback_reason = NddmaFallbackReason::kNone;
  return af::SUCCESS;
}

void LogNddmaFallback(const std::string &node_name, const std::string &dtype, const NddmaDescriptorInfo *descriptor,
                      const NddmaModelResult &result) {
  GELOGW(
      "[ATT NDDMA] fallback: node=%s, raw_rank=%zu, effective_rank=%zu, dtype=%s, candidate_model=%s, "
      "fallback_reason=%s",
      node_name.c_str(), result.raw_rank, result.effective_rank, dtype.c_str(), result.model_name.c_str(),
      NddmaFallbackReasonToString(result.fallback_reason));
  if (descriptor != nullptr) {
    GELOGD("[ATT NDDMA] fallback detail: output_dims=[%s], input_strides=[%s], output_strides=[%s]",
           GetVecString(descriptor->output_dims).c_str(), GetVecString(descriptor->input_strides).c_str(),
           GetVecString(descriptor->output_strides).c_str());
  }
}
}  // namespace att
