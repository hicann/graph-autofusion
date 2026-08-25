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

struct Nddma1DParams {
  double t1;
  double h1;
  double t2;
  double h2;
  double a1, a2, b1, b2, b3, b4, c1, c2, c3, c4;
};

const Nddma1DParams *GetNddma1DParams(uint64_t dtype_size) {
  static constexpr Nddma1DParams kB8 = {11.7626,      194.421,     6.05735,     373.274,      1.1117457,
                                        0.0081160848, -140.60967,  -0.85102455, 6.9134822,    -0.0081240796,
                                        1.5052979,    0.007823918, 0.47972982,  -0.0078196972};
  static constexpr Nddma1DParams kB16 = {25.7579,      204.604,     13.259,      399.909,     0.93077499,
                                         0.0041300563, -144.93927,  -0.65222426, 3.4593577,   -0.0041392279,
                                         1.1389125,    0.015061621, 0.84385983,  -0.015043028};
  static constexpr Nddma1DParams kB32 = {57.2624,      235.137,      29.4096,      453.859,     0.29118972,
                                         0.0021022688, -75.847618,   -0.098291434, 0.23017978,  -0.0021037148,
                                         1.5982043,    0.0048954057, 0.21802244,   0.0016884734};
  static constexpr Nddma1DParams kB64 = {57.2346,       243.205,     29.3906,      468.971,     0.32660604,
                                         0.00087486841, -62.346718,  -0.089369196, 0.10364992,  -0.00088051835,
                                         0.75774914,    0.021509891, 1.1274142,    -0.021494907};
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

NddmaFallbackReason ValidateNddmaDescriptorInput(const TensorShapeInfo &shape_info,
                                                 const std::vector<int64_t> &vectorized_axis,
                                                 const std::vector<bool> &tile_inner) {
  if (shape_info.repeats.empty()) {
    return NddmaFallbackReason::kNoDescriptor;
  }
  if (shape_info.repeats.size() != shape_info.gm_strides.size() ||
      shape_info.repeats.size() != shape_info.strides.size()) {
    return NddmaFallbackReason::kSchemaMismatch;
  }
  if (vectorized_axis.empty() || vectorized_axis.size() != shape_info.repeats.size()) {
    return NddmaFallbackReason::kCodegenMismatch;
  }
  if (!tile_inner.empty() && tile_inner.size() != shape_info.repeats.size()) {
    return NddmaFallbackReason::kSchemaMismatch;
  }
  return NddmaFallbackReason::kNone;
}

void BuildNddmaEffectiveView(const TensorShapeInfo &shape_info, const std::vector<bool> &tile_inner,
                             std::vector<Expr> &effective_repeats, std::vector<Expr> &effective_gm_strides,
                             std::vector<Expr> &effective_ub_strides, std::vector<int64_t> &effective_axes) {
  const auto &origin_repeats =
      shape_info.origin_repeats.size() == shape_info.repeats.size() ? shape_info.origin_repeats : shape_info.repeats;
  const size_t rank = origin_repeats.size();
  Expr prev_repeat = CreateExpr(1);
  Expr prev_actual_repeat = CreateExpr(1);
  Expr prev_gm_stride = CreateExpr(1);
  Expr prev_ub_stride = CreateExpr(1);
  bool has_non_zero_axis = false;
  for (size_t pos = rank; pos-- > 0U;) {
    const bool ignore_zero_axis =
        ascgen_utils::ShouldIgnoreDataCopyZeroAxis(has_non_zero_axis, pos, shape_info.strides);
    const bool zero_axis =
        af::SymbolicUtils::StaticCheckEq(shape_info.gm_strides[pos], CreateExpr(0)) == af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(shape_info.strides[pos], CreateExpr(0)) == af::TriBool::kTrue &&
        ignore_zero_axis;
    if (zero_axis) {
      continue;
    }
    has_non_zero_axis = true;
    const Expr cur_gm_stride = prev_gm_stride * prev_repeat;
    const Expr cur_ub_stride = prev_ub_stride * prev_repeat;
    const bool tile_boundary = !tile_inner.empty() && pos + 1U < rank && tile_inner[pos + 1U];
    if (!ascgen_utils::IsDataCopyAxisContinuous(cur_gm_stride, cur_ub_stride, shape_info.gm_strides[pos],
                                                shape_info.strides[pos]) ||
        effective_repeats.empty() || tile_boundary) {
      effective_repeats.emplace_back(shape_info.repeats[pos]);
      effective_gm_strides.emplace_back(shape_info.gm_strides[pos]);
      effective_ub_strides.emplace_back(shape_info.strides[pos]);
      effective_axes.emplace_back(static_cast<int64_t>(pos));
      prev_gm_stride = shape_info.gm_strides[pos];
      prev_ub_stride = shape_info.strides[pos];
      prev_repeat = origin_repeats[pos];
      prev_actual_repeat = shape_info.repeats[pos];
      continue;
    }
    const Expr product = origin_repeats[pos] * prev_repeat;
    effective_repeats.back() = shape_info.repeats[pos] * prev_actual_repeat;
    prev_repeat = product;
    prev_actual_repeat = shape_info.repeats[pos] * prev_actual_repeat;
  }
  std::reverse(effective_repeats.begin(), effective_repeats.end());
  std::reverse(effective_gm_strides.begin(), effective_gm_strides.end());
  std::reverse(effective_ub_strides.begin(), effective_ub_strides.end());
  std::reverse(effective_axes.begin(), effective_axes.end());
}

NddmaFallbackReason MapNddmaEffectiveAxes(std::vector<int64_t> &effective_axes,
                                          const std::vector<int64_t> &vectorized_axis) {
  for (auto &axis : effective_axes) {
    if (axis < 0 || static_cast<size_t>(axis) >= vectorized_axis.size()) {
      return NddmaFallbackReason::kCodegenMismatch;
    }
    axis = vectorized_axis[static_cast<size_t>(axis)];
  }
  return NddmaFallbackReason::kNone;
}

Expr AbsExpr(const Expr &value) {
  return af::sym::Max(value, af::sym::kSymbolZero - value);
}

Expr BuildNddma1DResidual(const Nddma1DParams &p, const Expr &bytes, const Expr &input_stride,
                          const Expr &output_stride, uint64_t dtype_size, bool high_core) {
  const Expr s = af::sym::Min(CreateExpr(kInputStrideUpperBound), input_stride * CreateExpr(dtype_size));
  const Expr g = af::sym::Min(af::sym::kSymbolOne, output_stride - af::sym::kSymbolOne);
  const Expr ng = (CreateExpr(p.a1) + CreateExpr(p.a2) * bytes) * s;
  const Expr ngu = (CreateExpr(p.b1) + CreateExpr(p.b2) * s + (CreateExpr(p.b3) + CreateExpr(p.b4) * s) * bytes) * g;
  const Expr rho = CreateExpr(p.c1) + CreateExpr(p.c2) * s + g * (CreateExpr(p.c3) + CreateExpr(p.c4) * s);
  return (ng + ngu) * (high_core ? rho : CreateExpr(1));
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

NddmaFallbackReason BuildNddmaCoreCycles(const NddmaNormalizedDesc &normalized, const Nddma1DParams &params,
                                         uint64_t dtype_size, Expr &low_core, Expr &high_core) {
  Expr total_elements = CreateExpr(1);
  for (const auto &dim : normalized.output_dims) {
    total_elements = total_elements * dim;
  }
  if (HasStaticByteCountOverflow(total_elements, dtype_size)) {
    return NddmaFallbackReason::kSchemaMismatch;
  }
  const Expr total_bytes = total_elements * CreateExpr(dtype_size);
  low_core = total_bytes / CreateExpr(params.t1) + CreateExpr(params.h1);
  high_core = total_bytes / CreateExpr(params.t2) + CreateExpr(params.h2);
  Expr input_prefix = CreateExpr(0);
  Expr output_prefix = CreateExpr(0);
  for (size_t i = 0; i < normalized.effective_rank; ++i) {
    if (i > 0U) {
      input_prefix = input_prefix + normalized.output_dims[i - 1U] * normalized.input_strides[i - 1U];
      output_prefix = output_prefix + normalized.output_dims[i - 1U] * normalized.output_strides[i - 1U];
    }
    Expr input_stride = normalized.input_strides[i];
    Expr output_stride = normalized.output_strides[i];
    if (i > 0U) {
      input_stride = AbsExpr(input_stride - input_prefix) + CreateExpr(1);
      output_stride = AbsExpr(output_stride - output_prefix) + CreateExpr(1);
    }
    Expr suffix_elements = CreateExpr(1);
    for (size_t suffix = i; suffix < normalized.effective_rank; ++suffix) {
      suffix_elements = suffix_elements * normalized.output_dims[suffix];
    }
    const Expr bytes = suffix_elements * CreateExpr(dtype_size);
    low_core = low_core + BuildNddma1DResidual(params, bytes, input_stride, output_stride, dtype_size, false);
    high_core = high_core + BuildNddma1DResidual(params, bytes, input_stride, output_stride, dtype_size, true);
  }
  return NddmaFallbackReason::kNone;
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
                                         NddmaDescriptorInfo &descriptor, const std::vector<bool> &tile_inner) {
  descriptor = NddmaDescriptorInfo{};
  const auto input_reason = ValidateNddmaDescriptorInput(shape_info, vectorized_axis, tile_inner);
  if (input_reason != NddmaFallbackReason::kNone) {
    return input_reason;
  }
  std::vector<Expr> effective_repeats;
  std::vector<Expr> effective_gm_strides;
  std::vector<Expr> effective_ub_strides;
  std::vector<int64_t> effective_axes;
  BuildNddmaEffectiveView(shape_info, tile_inner, effective_repeats, effective_gm_strides, effective_ub_strides,
                          effective_axes);
  const auto axis_reason = MapNddmaEffectiveAxes(effective_axes, vectorized_axis);
  if (axis_reason != NddmaFallbackReason::kNone) {
    return axis_reason;
  }
  descriptor.output_dims = std::move(effective_repeats);
  descriptor.input_strides = std::move(effective_gm_strides);
  descriptor.output_strides = std::move(effective_ub_strides);
  descriptor.vectorized_axis = std::move(effective_axes);
  GELOGD(
      "[ATT NDDMA] effective view: raw_rank=%zu, effective_rank=%zu, repeats=[%s], gm_strides=[%s], ub_strides=[%s], "
      "axes=[%s]",
      shape_info.repeats.size(), descriptor.output_dims.size(), GetVecString(descriptor.output_dims).c_str(),
      GetVecString(descriptor.input_strides).c_str(), GetVecString(descriptor.output_strides).c_str(),
      ascgen_utils::VectorToStr(descriptor.vectorized_axis).c_str());
  return NddmaFallbackReason::kNone;
}

NddmaFallbackReason NormalizeNddmaDescriptor(const NddmaDescriptorInfo &descriptor, NddmaNormalizedDesc &normalized) {
  normalized = NddmaNormalizedDesc{};
  normalized.effective_rank = descriptor.output_dims.size();
  if (normalized.effective_rank < kMinNddmaRank || normalized.effective_rank > kMaxNddmaRank) {
    return NddmaFallbackReason::kRankUnsupported;
  }
  if (!HasValidVectorLengths(descriptor)) {
    return NddmaFallbackReason::kSchemaMismatch;
  }
  if (!HasValidAxisOrder(descriptor)) {
    return NddmaFallbackReason::kCodegenMismatch;
  }
  normalized.output_dims.assign(descriptor.output_dims.rbegin(), descriptor.output_dims.rend());
  normalized.input_strides.assign(descriptor.input_strides.rbegin(), descriptor.input_strides.rend());
  normalized.output_strides.assign(descriptor.output_strides.rbegin(), descriptor.output_strides.rend());
  normalized.vectorized_axis.assign(descriptor.vectorized_axis.rbegin(), descriptor.vectorized_axis.rend());
  return HasInvalidStaticValue(normalized) ? NddmaFallbackReason::kStrideInvalid : NddmaFallbackReason::kNone;
}

af::Status EvaluateNddmaModel(const NddmaDescriptorInfo &descriptor, const std::string &dtype, const Expr &block_dim,
                              NddmaModelResult &result) {
  result = NddmaModelResult{};
  NddmaNormalizedDesc normalized;
  result.fallback_reason = NormalizeNddmaDescriptor(descriptor, normalized);
  result.effective_rank = normalized.effective_rank;
  if (result.fallback_reason != NddmaFallbackReason::kNone) {
    return af::SUCCESS;
  }
  if (normalized.effective_rank < kMinNddmaRank || normalized.effective_rank > kMaxNddmaRank) {
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
  Expr low_core;
  Expr high_core;
  result.fallback_reason = BuildNddmaCoreCycles(normalized, *params, dtype_size, low_core, high_core);
  if (result.fallback_reason != NddmaFallbackReason::kNone) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(SelectCoreBranch(block_dim, low_core, high_core, result));
  // Fitted residual terms can become non-positive outside their measured
  // domain (for example, tiny B with a large effective stride).  Do not emit
  // an invalid performance expression; use the existing legacy path instead.
  if (IsStaticNonPositive(result.cycles)) {
    result.selected = false;
    result.fallback_reason = NddmaFallbackReason::kSchemaMismatch;
    return af::SUCCESS;
  }
  result.selected = true;
  result.model_name = normalized.effective_rank == 1U ? "NDDMA_1D_MULTICORE_V2" : "NDDMA_ND_MULTICORE_V1";
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
