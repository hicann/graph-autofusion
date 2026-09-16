/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef AUTOFUSE_ASCENDC_BROADCAST_PERF_UTILS_V2_H_
#define AUTOFUSE_ASCENDC_BROADCAST_PERF_UTILS_V2_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "api_perf_register/api_perf.h"
#include "ascir_node_param/ascir_node_param.h"
#include "gen_model_info/api_perf_register/utils/vf_perf_utils.h"

namespace att {
namespace ascendcapi_v2 {

struct BroadcastTilingInfo {
  std::vector<Expr> original_src_shape;
  std::vector<Expr> original_dst_shape;
  std::vector<Expr> src_shape;
  std::vector<Expr> dst_shape;
  std::vector<Expr> src_stride;
  std::vector<Expr> dst_stride;
  Expr src_size{CreateExpr(1)};
  Expr dst_size{CreateExpr(1)};
  Expr loop_num{CreateExpr(0)};
  size_t original_rank{0U};
  size_t folded_rank{0U};
  size_t rank{0U};
};

struct VfCostAccumulator {
  Expr max_latency{CreateExpr(0)};
  Expr throughput{CreateExpr(0)};
  uint32_t load_count{0U};
};

struct ParamExprInputs {
  std::vector<Expr> semantic;
  std::vector<Expr> size;
  std::vector<Expr> actual_size;
};

af::Status GetDtypeByteSize(const std::string &dtype, Expr &byte_size);
af::Status GetEffectiveHelperDtype(const std::string &dtype, std::string &helper_dtype);
af::Status GetBroadcastVectorElements(const std::string &dtype, Expr &vl, Expr &half_vl, Expr &block_elements);
af::Status GetBroadcastTilingVectorElements(const std::string &dtype, Expr &vl, Expr &half_vl, Expr &block_elements);
af::Status CeilDiv(const Expr &value, const Expr &divisor, Expr &result);
Expr ResolveParamExprLeaf(const ascir_param::ParamExprLeaf &leaf, const Expr &semantic_expr, const Expr &size_expr,
                          const Expr &actual_size_expr);
Expr ShapeProduct(const std::vector<Expr> &shape);
af::Status BuildBroadcastTiling(const std::vector<ascir_param::ParamExprLeaf> &src,
                                const std::vector<ascir_param::ParamExprLeaf> &dst, const std::string &dtype,
                                const ParamExprInputs &src_inputs, const ParamExprInputs &dst_inputs,
                                BroadcastTilingInfo &tiling);
bool IsLastAxisBroadcast(const BroadcastTilingInfo &tiling);

af::Status AddVfInstructPerf(const std::string &instruct, const std::string &dtype, const Expr &repeat_time,
                             uint32_t instruct_count, VfCostAccumulator &acc);
af::Status AddBroadcastDataCopyPerf(const std::string &dtype, const Expr &repeat_time, VfCostAccumulator &acc);
Expr GetVfCost(const VfCostAccumulator &acc, bool include_head = true);
af::Status BuildBroadcastTernary(const std::string &name, CondType condition, const Expr &lhs, const Expr &rhs,
                                 const Expr &true_value, const Expr &false_value, TernaryOpMap &ternary_ops,
                                 Expr &result);

}  // namespace ascendcapi_v2
}  // namespace att

#endif  // AUTOFUSE_ASCENDC_BROADCAST_PERF_UTILS_V2_H_
