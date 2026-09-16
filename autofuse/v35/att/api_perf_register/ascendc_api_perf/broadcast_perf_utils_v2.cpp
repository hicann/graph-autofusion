/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "broadcast_perf_utils_v2.h"

#include "base/att_const_values.h"
#include "common/checker.h"
#include "../perf_param_v2.h"

#include <algorithm>

namespace att {
namespace ascendcapi_v2 {
namespace {
constexpr uint32_t kBlockBytes = 32U;
constexpr uint32_t kVectorBytes = 256U;

af::TriBool CheckCondition(CondType type, const Expr &lhs, const Expr &rhs) {
  switch (type) {
    case CondType::K_EQ:
      return af::SymbolicUtils::StaticCheckEq(lhs, rhs);
    case CondType::K_LT:
      return af::SymbolicUtils::StaticCheckLt(lhs, rhs);
    case CondType::K_GT:
      return af::SymbolicUtils::StaticCheckGt(lhs, rhs);
    case CondType::K_LE:
      return af::SymbolicUtils::StaticCheckLe(lhs, rhs);
    case CondType::K_GE:
      return af::SymbolicUtils::StaticCheckGe(lhs, rhs);
    default:
      return af::TriBool::kUnknown;
  }
}

void CollapseRank(std::vector<Expr> &src, std::vector<Expr> &dst) {
  std::vector<Expr> collapsed_src;
  std::vector<Expr> collapsed_dst;
  for (size_t i = 0U; i < src.size();) {
    const bool broadcast = src[i] == CreateExpr(1) && dst[i] != CreateExpr(1);
    Expr src_product = broadcast ? CreateExpr(1) : src[i];
    Expr dst_product = dst[i];
    ++i;
    while (i < src.size() &&
           ((broadcast && src[i] == CreateExpr(1) && dst[i] != CreateExpr(1)) || (!broadcast && src[i] == dst[i]))) {
      dst_product = dst_product * dst[i];
      if (!broadcast) {
        src_product = src_product * src[i];
      }
      ++i;
    }
    collapsed_src.push_back(src_product);
    collapsed_dst.push_back(dst_product);
  }
  src = std::move(collapsed_src);
  dst = std::move(collapsed_dst);
}

void BuildStrides(const std::vector<Expr> &src, const std::vector<Expr> &dst, std::vector<Expr> &src_stride,
                  std::vector<Expr> &dst_stride) {
  src_stride.assign(src.size(), CreateExpr(0));
  dst_stride.assign(dst.size(), CreateExpr(0));
  Expr src_step = CreateExpr(1);
  Expr dst_step = CreateExpr(1);
  for (size_t i = src.size(); i > 0U; --i) {
    const size_t index = i - 1U;
    dst_stride[index] = dst_step;
    dst_step = dst_step * dst[index];
    if (src[index] == CreateExpr(1) && dst[index] != CreateExpr(1)) {
      continue;
    }
    src_stride[index] = src_step;
    src_step = src_step * src[index];
  }
}

af::Status ApplyB64Tiling(const std::string &dtype, BroadcastTilingInfo &tiling) {
  if (dtype != kUInt64 && dtype != kInt64) {
    return af::SUCCESS;
  }
  if (tiling.src_size == tiling.dst_size) {
    return af::SUCCESS;
  }
  const size_t last = tiling.rank - 1U;
  if (tiling.src_shape[last] == CreateExpr(1) && tiling.dst_shape[last] != CreateExpr(1)) {
    if (tiling.rank < 9U) {
      tiling.src_shape.push_back(kSymTwo);
      tiling.dst_shape.push_back(kSymTwo);
      ++tiling.rank;
    } else {
      tiling.loop_num = tiling.dst_shape[0];
    }
  } else {
    tiling.src_shape[last] = tiling.src_shape[last] * kSymTwo;
    tiling.dst_shape[last] = tiling.dst_shape[last] * kSymTwo;
  }
  tiling.src_size = tiling.src_size * kSymTwo;
  tiling.dst_size = tiling.dst_size * kSymTwo;
  return af::SUCCESS;
}

bool HasParamExprInputs(const std::vector<ascir_param::ParamExprLeaf> &leaves, const ParamExprInputs &inputs) {
  return leaves.size() == inputs.semantic.size() && leaves.size() == inputs.size.size() &&
         leaves.size() == inputs.actual_size.size();
}
}  // namespace

af::Status GetDtypeByteSize(const std::string &dtype, Expr &byte_size) {
  const auto iter = kDataTypeSizeMap.find(dtype);
  GE_ASSERT_TRUE(iter != kDataTypeSizeMap.end(), "Unsupported broadcast dtype[%s].", dtype.c_str());
  byte_size = iter->second;
  return af::SUCCESS;
}

af::Status GetEffectiveHelperDtype(const std::string &dtype, std::string &helper_dtype) {
  Expr byte_size;
  GE_ASSERT_SUCCESS(GetDtypeByteSize(dtype, byte_size));
  helper_dtype = (dtype == kUInt64 || dtype == kInt64) ? kUInt32 : dtype;
  return af::SUCCESS;
}

af::Status GetBroadcastVectorElements(const std::string &dtype, Expr &vl, Expr &half_vl, Expr &block_elements) {
  Expr byte_size;
  GE_ASSERT_SUCCESS(GetDtypeByteSize(dtype, byte_size));
  if (af::SymbolicUtils::StaticCheckEq(byte_size, CreateExpr(0)) == af::TriBool::kTrue) {
    GELOGW("Broadcast dtype byte size cannot be zero.");
    return af::SUCCESS;
  }
  vl = CreateExpr(kVectorBytes) / byte_size;
  half_vl = vl / kSymTwo;
  block_elements = CreateExpr(kBlockBytes) / byte_size;
  return af::SUCCESS;
}

af::Status GetBroadcastTilingVectorElements(const std::string &dtype, Expr &vl, Expr &half_vl, Expr &block_elements) {
  std::string helper_dtype;
  GE_ASSERT_SUCCESS(GetEffectiveHelperDtype(dtype, helper_dtype));
  return GetBroadcastVectorElements(helper_dtype, vl, half_vl, block_elements);
}

af::Status CeilDiv(const Expr &value, const Expr &divisor, Expr &result) {
  if (af::SymbolicUtils::StaticCheckEq(divisor, CreateExpr(0)) == af::TriBool::kTrue) {
    GELOGW("Broadcast ceil divisor cannot be zero.");
    return af::SUCCESS;
  }
  result = af::sym::Ceiling(value / divisor);
  return af::SUCCESS;
}

Expr ResolveParamExprLeaf(const ascir_param::ParamExprLeaf &leaf, const Expr &semantic_expr, const Expr &size_expr,
                          const Expr &actual_size_expr) {
  if (leaf.role == ascir_param::ParamExprRole::kSize) {
    return size_expr;
  }
  if (leaf.role == ascir_param::ParamExprRole::kActualSize) {
    return actual_size_expr;
  }
  return semantic_expr;
}

Expr ShapeProduct(const std::vector<Expr> &shape) {
  Expr product = CreateExpr(1);
  for (const auto &dim : shape) {
    product = product * dim;
  }
  return product;
}

af::Status BuildBroadcastTiling(const std::vector<ascir_param::ParamExprLeaf> &src,
                                const std::vector<ascir_param::ParamExprLeaf> &dst, const std::string &dtype,
                                const ParamExprInputs &src_inputs, const ParamExprInputs &dst_inputs,
                                BroadcastTilingInfo &tiling) {
  GE_ASSERT_TRUE(!src.empty() && src.size() == dst.size(), "Broadcast shapes must have the same non-zero rank.");
  GE_ASSERT_TRUE(HasParamExprInputs(src, src_inputs) && HasParamExprInputs(dst, dst_inputs),
                 "Broadcast parameter expression inputs must match rank.");
  tiling.original_src_shape.clear();
  tiling.original_dst_shape.clear();
  for (size_t i = 0U; i < src.size(); ++i) {
    tiling.original_src_shape.push_back(
        ResolveParamExprLeaf(src[i], src_inputs.semantic[i], src_inputs.size[i], src_inputs.actual_size[i]));
    tiling.original_dst_shape.push_back(
        ResolveParamExprLeaf(dst[i], dst_inputs.semantic[i], dst_inputs.size[i], dst_inputs.actual_size[i]));
  }
  tiling.original_rank = src.size();
  tiling.src_shape = tiling.original_src_shape;
  tiling.dst_shape = tiling.original_dst_shape;
  tiling.src_size = ShapeProduct(tiling.original_src_shape);
  tiling.dst_size = ShapeProduct(tiling.original_dst_shape);
  if (tiling.src_shape.size() > 4U) {
    CollapseRank(tiling.src_shape, tiling.dst_shape);
  }
  tiling.folded_rank = tiling.dst_shape.size();
  if (tiling.original_rank == 4U && tiling.dst_shape[0] == CreateExpr(1) &&
      tiling.original_src_shape[0] == CreateExpr(1)) {
    tiling.src_shape.erase(tiling.src_shape.begin());
    tiling.dst_shape.erase(tiling.dst_shape.begin());
  }
  tiling.rank = tiling.dst_shape.size();
  GE_ASSERT_SUCCESS(ApplyB64Tiling(dtype, tiling));
  const bool loop_src_stride_zero =
      tiling.loop_num != CreateExpr(0) && tiling.src_shape[0] == CreateExpr(1) && tiling.dst_shape[0] != CreateExpr(1);
  if (tiling.loop_num != CreateExpr(0)) {
    tiling.src_shape.erase(tiling.src_shape.begin());
    tiling.dst_shape.erase(tiling.dst_shape.begin());
    tiling.src_shape.push_back(kSymTwo);
    tiling.dst_shape.push_back(kSymTwo);
  }
  BuildStrides(tiling.src_shape, tiling.dst_shape, tiling.src_stride, tiling.dst_stride);
  if (tiling.loop_num != CreateExpr(0)) {
    tiling.src_stride.push_back(loop_src_stride_zero ? CreateExpr(0) : ShapeProduct(tiling.src_shape));
  }
  return af::SUCCESS;
}

bool IsLastAxisBroadcast(const BroadcastTilingInfo &tiling) {
  return tiling.rank > 0U && tiling.src_stride.size() >= tiling.rank &&
         tiling.src_stride[tiling.rank - 1U] == CreateExpr(0);
}

af::Status AddBroadcastDataCopyPerf(const std::string &dtype, const Expr &repeat_time, VfCostAccumulator &acc) {
  return AddVfInstructPerf(kLoad, dtype, repeat_time, 1U, acc);
}

af::Status AddVfInstructPerf(const std::string &instruct, const std::string &dtype, const Expr &repeat_time,
                             uint32_t instruct_count, VfCostAccumulator &acc) {
  std::string helper_dtype;
  GE_ASSERT_SUCCESS(GetEffectiveHelperDtype(dtype, helper_dtype));
  if (instruct == kLoad) {
    acc.load_count += instruct_count;
  }
  static const PerfParamTableV2 perf_table;
  const auto &entries = perf_table.GetVfInstructPerfTable(instruct);
  for (uint32_t i = 0U; i < instruct_count; ++i) {
    for (const auto &entry : entries) {
      if (std::find(entry.support_data_types.begin(), entry.support_data_types.end(), helper_dtype) ==
          entry.support_data_types.end()) {
        continue;
      }
      acc.max_latency = af::sym::Max(acc.max_latency, CreateExpr(entry.latency));
      acc.throughput = acc.throughput + CreateExpr(entry.throughput) * repeat_time;
      break;
    }
  }
  return af::SUCCESS;
}

Expr GetVfCost(const VfCostAccumulator &acc, bool include_head) {
  Expr cost = acc.max_latency + acc.throughput;
  if (include_head) {
    static const PerfParamTableV2 perf_table;
    cost = cost + perf_table.GetVectorFunctionHeadCost();
  }
  cost.Simplify();
  return cost;
}

af::Status BuildBroadcastTernary(const std::string &name, CondType condition, const Expr &lhs, const Expr &rhs,
                                 const Expr &true_value, const Expr &false_value, TernaryOpMap &ternary_ops,
                                 Expr &result) {
  const auto check = CheckCondition(condition, lhs, rhs);
  if (check == af::TriBool::kTrue) {
    result = true_value;
    return af::SUCCESS;
  }
  if (check == af::TriBool::kFalse) {
    result = false_value;
    return af::SUCCESS;
  }
  GetPerfVar(name, result, ternary_ops);
  TernaryOp ternary(condition, lhs, rhs, true_value, false_value);
  ternary.SetVariable(result);
  ternary_ops[result] = ternary;
  return af::SUCCESS;
}

}  // namespace ascendcapi_v2
}  // namespace att
