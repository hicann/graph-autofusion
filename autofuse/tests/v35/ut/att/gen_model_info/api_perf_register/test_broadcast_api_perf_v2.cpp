/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms of
 * CANN Open Software License Agreement Version 2.0 of the License.
 */

#include <iostream>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "graph/ascendc_ir/ascir_registry.h"
#include "ascir_node_param/ascir_node_param.h"
#include "ascir_ops.h"
#include "att/api_perf_register/perf_param_v2.h"
#include "api_perf_register/api_perf_factory.h"
#include "api_perf_register/ascendc_api_perf.h"
#include "common/platform_context.h"
#include "common/checker.h"
#include "base/att_const_values.h"
#include "v35/att/api_perf_register/ascendc_regbase_perf.h"
#include "v35/att/api_perf_register/ascendc_api_perf/broadcast_last_axis_perf_v2.h"
#include "../../../../../ut/att/testcase/gen_model_info/api_perf_register/runtime_stub.h"
#include "tests/depends/slog/src/slog_stub.h"
#include "graph_construct_utils.h"
#include "parser/specific_params_builder.h"
#include "v35/att/api_perf_register/ascendc_api_perf/broadcast_api_perf_v2.h"
#include "v35/att/api_perf_register/ascendc_api_perf/broadcast_nlast_axis_perf_v2.h"

namespace att {
namespace {
using ascendcapi_v2::BroadcastTilingInfo;
using ascendcapi_v2::NlastAxisBranch;
using ascendcapi_v2::ParamExprInputs;
using ascendcapi_v2::VfCostAccumulator;

NodeDetail MakeNode(const std::string &dtype, const std::vector<int64_t> &src, const std::vector<int64_t> &dst) {
  NodeDetail node;
  node.broadcast_node_params.valid = true;
  node.broadcast_node_params.const_rank = static_cast<int32_t>(src.size());
  for (size_t i = 0U; i < src.size(); ++i) {
    node.broadcast_node_params.src_shape.push_back({CreateExpr(src[i]), ascir_param::ParamExprRole::kSemantic});
    node.broadcast_node_params.dst_shape.push_back({CreateExpr(dst[i]), ascir_param::ParamExprRole::kSemantic});
    node.input_dims.push_back(CreateExpr(src[i]));
    node.output_dims.push_back(CreateExpr(dst[i]));
    node.repeats.push_back(CreateExpr(src[i]));
  }
  node.input_dtype = {dtype};
  node.output_dtype = {dtype};
  return node;
}

NodeDetail MakeRoleNode(const std::string &dtype, ascir_param::ParamExprRole role,
                        const std::vector<int64_t> &input_dims, const std::vector<int64_t> &repeats,
                        const std::vector<int64_t> &dst) {
  auto node = MakeNode(dtype, {1, 1, 128}, dst);
  node.broadcast_node_params.src_shape[1].role = role;
  node.input_dims.clear();
  node.repeats.clear();
  for (const auto dim : input_dims) {
    node.input_dims.push_back(CreateExpr(dim));
  }
  for (const auto dim : repeats) {
    node.repeats.push_back(CreateExpr(dim));
  }
  return node;
}

BroadcastTilingInfo MakeTiling(const std::vector<int64_t> &dst) {
  BroadcastTilingInfo tiling;
  tiling.rank = dst.size();
  for (const auto dim : dst) {
    tiling.dst_shape.push_back(CreateExpr(dim));
  }
  tiling.dst_size = ascendcapi_v2::ShapeProduct(tiling.dst_shape);
  return tiling;
}

Expr ExpectedGatherBPerf(bool two) {
  ascendcapi_v2::VfCostAccumulator acc;
  const Expr count = two ? CreateExpr(2) : CreateExpr(1);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kLoad, kUInt32, CreateExpr(1), 1U, acc), af::SUCCESS);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kUInt8, acc.max_latency, acc.throughput, CreateExpr(1)),
            af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kUpdateMask, kUInt8, count, 1U, acc), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kStore, kUInt8, count, 1U, acc), af::SUCCESS);
  Expr result = ascendcapi_v2::GetVfCost(acc);
  result.Simplify();
  return result;
}

Expr ExpectedGatherWrapperPerf(const std::string &dtype, size_t rank, const Expr &arithmetic_count,
                               const Expr &gather_count) {
  const bool b8 = dtype == kUInt8 || dtype == kInt8;
  const std::string index_dtype = dtype == kUInt32 || dtype == kInt32 ? kInt32 : kInt16;
  const Expr index_calls = b8 ? CreateExpr(2) : CreateExpr(1);
  VfCostAccumulator acc;
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kDuplicate, index_dtype, CreateExpr(rank * 2), 1U, acc), af::SUCCESS);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, index_dtype, acc.max_latency, acc.throughput, index_calls),
            af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kDiv, index_dtype, CreateExpr(rank) * index_calls, 1U, acc), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kMul, index_dtype, CreateExpr(rank + 1) * index_calls, 1U, acc),
            af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kSub, index_dtype, CreateExpr(rank) * index_calls, 1U, acc), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kMulAddDst, index_dtype, CreateExpr(rank - 1) * index_calls, 1U, acc),
            af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kStore, index_dtype, index_calls, 1U, acc), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kDuplicate, index_dtype, CreateExpr(2), 1U, acc), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kLoad, index_dtype, CreateExpr(1), 1U, acc), af::SUCCESS);
  const std::string gather_dtype = dtype == kUInt32 || dtype == kInt32 ? kUInt32 : kUInt16;
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kMuls, gather_dtype, arithmetic_count, 1U, acc), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kAdd, gather_dtype, arithmetic_count, 1U, acc), af::SUCCESS);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, gather_dtype, acc.max_latency, acc.throughput, gather_count),
            af::SUCCESS);
  if (b8) {
    EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, gather_dtype, acc.max_latency, acc.throughput, gather_count),
              af::SUCCESS);
    EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kDeInterleave, dtype, gather_count, 1U, acc), af::SUCCESS);
  }
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kStore, dtype, gather_count, 1U, acc), af::SUCCESS);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, acc.max_latency, acc.throughput, CreateExpr(1)),
            af::SUCCESS);
  Expr result = ascendcapi_v2::GetVfCost(acc);
  result.Simplify();
  return result;
}

Expr ExpectedRank2UnalignedPerf(const std::string &dtype, const Expr &outer, const Expr &full_loads,
                                const Expr &tail_loads, bool update_mask) {
  VfCostAccumulator acc;
  if (update_mask) {
    EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kUpdateMask, dtype, CreateExpr(1), 1U, acc), af::SUCCESS);
  }
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kLoad, dtype, CreateExpr(1), 1U, acc), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kLoad, dtype, full_loads + tail_loads, 1U, acc), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kStore, dtype, full_loads + outer, 1U, acc), af::SUCCESS);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, acc.max_latency, acc.throughput, CreateExpr(1)),
            af::SUCCESS);
  Expr result = ascendcapi_v2::GetVfCost(acc);
  result.Simplify();
  return result;
}

Expr ExpectedRank2LessUnalignedPerf(const std::string &dtype, const Expr &outer) {
  VfCostAccumulator acc;
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kUpdateMask, dtype, CreateExpr(1), 1U, acc), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kLoad, dtype, CreateExpr(1), 1U, acc), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kStore, dtype, outer, 1U, acc), af::SUCCESS);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, acc.max_latency, acc.throughput, CreateExpr(1)),
            af::SUCCESS);
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedRank3UnalignedPerf(const std::string &dtype, const Expr &outer, const Expr &loads, const Expr &stores,
                                bool update_mask) {
  VfCostAccumulator acc;
  if (update_mask) {
    EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kUpdateMask, dtype, CreateExpr(1), 1U, acc), af::SUCCESS);
  }
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, acc.max_latency, acc.throughput, outer), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kLoad, dtype, loads, 1U, acc), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kStore, dtype, stores, 1U, acc), af::SUCCESS);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, acc.max_latency, acc.throughput, CreateExpr(1)),
            af::SUCCESS);
  Expr result = ascendcapi_v2::GetVfCost(acc);
  result.Simplify();
  return result;
}

void AddExpectedExpr(const std::string &op, const std::string &dtype, const Expr &count, VfCostAccumulator &acc) {
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(op, dtype, count, 1U, acc), af::SUCCESS);
}

Expr ExpectedB64MoreDimGatherPerf() {
  // Source ledger: uint64 helper runs at VL=64 after B64 doubling, index/gather registers use effective uint32.
  VfCostAccumulator acc;
  AddExpectedExpr(kDuplicate, kInt32, CreateExpr(16), acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kInt32, acc.max_latency, acc.throughput, CreateExpr(2)),
            af::SUCCESS);
  AddExpectedExpr(kDiv, kInt32, CreateExpr(8), acc);
  AddExpectedExpr(kMul, kInt32, CreateExpr(10), acc);
  AddExpectedExpr(kSub, kInt32, CreateExpr(8), acc);
  AddExpectedExpr(kMulAddDst, kInt32, CreateExpr(6), acc);
  AddExpectedExpr(kStore, kInt32, CreateExpr(2), acc);
  AddExpectedExpr(kDuplicate, kInt32, CreateExpr(4), acc);
  AddExpectedExpr(kLoad, kInt32, CreateExpr(2), acc);
  AddExpectedExpr(kMuls, kUInt32, CreateExpr(20), acc);
  AddExpectedExpr(kAdd, kUInt32, CreateExpr(20), acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kUInt32, acc.max_latency, acc.throughput, CreateExpr(12)),
            af::SUCCESS);
  AddExpectedExpr(kStore, kUInt64, CreateExpr(12), acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kUInt64, acc.max_latency, acc.throughput, CreateExpr(2)),
            af::SUCCESS);
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedAlignedLeaf(const std::string &dtype, const Expr &loads, const Expr &updates, const Expr &stores) {
  VfCostAccumulator acc;
  if (updates != CreateExpr(0)) {
    AddExpectedExpr(kUpdateMask, dtype, updates, acc);
  }
  AddExpectedExpr(kLoad, dtype, loads, acc);
  AddExpectedExpr(kStore, dtype, stores, acc);
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedDynamicLessLeaf(const std::string &dtype, const std::vector<int64_t> &shape, bool aligned) {
  const size_t rank = shape.size();
  Expr outer = CreateExpr(1);
  Expr helper_loops = CreateExpr(1);
  for (size_t i = 0U; i + 1U < rank; ++i) {
    outer = outer * CreateExpr(shape[i]);
  }
  for (size_t i = 0U; i + 4U < rank; ++i) {
    helper_loops = helper_loops * CreateExpr(shape[i]);
  }
  if (aligned) {
    return ExpectedAlignedLeaf(dtype, outer, helper_loops, outer);
  }
  VfCostAccumulator acc;
  AddExpectedExpr(kUpdateMask, dtype, helper_loops, acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, acc.max_latency, acc.throughput, outer), af::SUCCESS);
  AddExpectedExpr(kLoad, dtype, outer, acc);
  AddExpectedExpr(kStore, dtype, outer, acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, acc.max_latency, acc.throughput, helper_loops),
            af::SUCCESS);
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedDynamicLargerUnalignedLeaf(const std::string &dtype, const std::vector<int64_t> &shape, bool dynamic) {
  const size_t rank = shape.size();
  Expr outer = CreateExpr(1);
  Expr helper_loops = CreateExpr(1);
  for (size_t i = 0U; i + 1U < rank; ++i) {
    outer = outer * CreateExpr(shape[i]);
  }
  if (dynamic) {
    for (size_t i = 0U; i + 4U < rank; ++i) {
      helper_loops = helper_loops * CreateExpr(shape[i]);
    }
  }
  const Expr full = CreateExpr(shape.back() / 128);
  VfCostAccumulator acc;
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, acc.max_latency, acc.throughput, outer), af::SUCCESS);
  AddExpectedExpr(kLoad, dtype, outer * full, acc);
  AddExpectedExpr(kStore, dtype, outer * full, acc);
  AddExpectedExpr(kLoad, dtype, outer, acc);
  AddExpectedExpr(kStore, dtype, outer, acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, acc.max_latency, acc.throughput, helper_loops),
            af::SUCCESS);
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedRank2LargerAlignedLeaf(const std::string &dtype, int64_t outer, bool use_vl) {
  VfCostAccumulator acc;
  const Expr factor = CreateExpr(2);
  const Expr stores = CreateExpr(outer) * factor;
  if (!use_vl) {
    AddExpectedExpr(kUpdateMask, dtype, stores, acc);
  }
  AddExpectedExpr(kLoad, dtype, use_vl ? factor : stores, acc);
  AddExpectedExpr(kStore, dtype, stores, acc);
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedRank4LessAlignedLeaf(const std::string &dtype) {
  VfCostAccumulator acc;
  AddExpectedExpr(kUpdateMask, dtype, CreateExpr(1), acc);
  AddExpectedExpr(kLoad, dtype, CreateExpr(12), acc);
  AddExpectedExpr(kStore, dtype, CreateExpr(12), acc);
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedRank4LessUnalignedLeaf(const std::string &dtype) {
  VfCostAccumulator acc;
  AddExpectedExpr(kUpdateMask, dtype, CreateExpr(1), acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, acc.max_latency, acc.throughput, CreateExpr(12)),
            af::SUCCESS);
  AddExpectedExpr(kLoad, dtype, CreateExpr(12), acc);
  AddExpectedExpr(kStore, dtype, CreateExpr(12), acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, acc.max_latency, acc.throughput, CreateExpr(1)),
            af::SUCCESS);
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedRank4LargerAlignedLeaf(const std::string &dtype) {
  VfCostAccumulator acc;
  AddExpectedExpr(kUpdateMask, dtype, CreateExpr(6), acc);
  AddExpectedExpr(kLoad, dtype, CreateExpr(24), acc);
  AddExpectedExpr(kStore, dtype, CreateExpr(24), acc);
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedDynamicGatherLeaf(const std::string &dtype, int64_t multiplier) {
  // Reg sequence inside BroadcastExtend is a runtime helper; this is an independent outer-call ledger.
  const Expr calls = CreateExpr(multiplier);
  VfCostAccumulator acc;
  AddExpectedExpr(kDuplicate, kInt16, CreateExpr(8) * calls, acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kInt16, acc.max_latency, acc.throughput, calls), af::SUCCESS);
  AddExpectedExpr(kDiv, kInt16, CreateExpr(4) * calls, acc);
  AddExpectedExpr(kMul, kInt16, CreateExpr(5) * calls, acc);
  AddExpectedExpr(kSub, kInt16, CreateExpr(4) * calls, acc);
  AddExpectedExpr(kMulAddDst, kInt16, CreateExpr(3) * calls, acc);
  AddExpectedExpr(kStore, kInt16, calls, acc);
  AddExpectedExpr(kDuplicate, kInt16, CreateExpr(2) * calls, acc);
  AddExpectedExpr(kLoad, kInt16, calls, acc);
  AddExpectedExpr(kMuls, kUInt16, CreateExpr(7) * calls, acc);
  AddExpectedExpr(kAdd, kUInt16, CreateExpr(7) * calls, acc);
  EXPECT_EQ(
      VfPerfUtils::AddVfInstructPerf(kPlaceholder, kUInt16, acc.max_latency, acc.throughput, CreateExpr(4) * calls),
      af::SUCCESS);
  AddExpectedExpr(kStore, dtype, CreateExpr(4) * calls, acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, acc.max_latency, acc.throughput, calls), af::SUCCESS);
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedB64RightAlignedGatherLeaf() {
  // ApplyB64Tiling appends one dimension for a right-aligned rank-2 broadcast.
  VfCostAccumulator acc;
  AddExpectedExpr(kDuplicate, kInt32, CreateExpr(6), acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kInt32, acc.max_latency, acc.throughput, CreateExpr(1)),
            af::SUCCESS);
  AddExpectedExpr(kDiv, kInt32, CreateExpr(3), acc);
  AddExpectedExpr(kMul, kInt32, CreateExpr(4), acc);
  AddExpectedExpr(kSub, kInt32, CreateExpr(3), acc);
  AddExpectedExpr(kMulAddDst, kInt32, CreateExpr(2), acc);
  AddExpectedExpr(kStore, kInt32, CreateExpr(1), acc);
  AddExpectedExpr(kDuplicate, kInt32, CreateExpr(2), acc);
  AddExpectedExpr(kLoad, kInt32, CreateExpr(1), acc);
  AddExpectedExpr(kMuls, kUInt32, CreateExpr(2), acc);
  AddExpectedExpr(kAdd, kUInt32, CreateExpr(2), acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kUInt32, acc.max_latency, acc.throughput, CreateExpr(1)),
            af::SUCCESS);
  AddExpectedExpr(kStore, kUInt64, CreateExpr(1), acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kUInt64, acc.max_latency, acc.throughput, CreateExpr(1)),
            af::SUCCESS);
  return ascendcapi_v2::GetVfCost(acc);
}

NodeDetail MakeDynamicNlastNode(const std::string &dtype, std::vector<int64_t> dst) {
  std::vector<int64_t> src = dst;
  for (size_t i = 0U; i + 1U < src.size(); ++i) {
    if (i % 2U == 0U) {
      src[i] = 1;
    }
  }
  return MakeNode(dtype, src, dst);
}

Expr ExpectedRank2B8GatherPerf() {
  VfCostAccumulator acc;
  const Expr two = CreateExpr(2);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kDuplicate, kInt16, two, 1U, acc), af::SUCCESS);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kInt16, acc.max_latency, acc.throughput, two), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kDiv, kInt16, two, 1U, acc), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kMul, kInt16, two, 1U, acc), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kSub, kInt16, two, 1U, acc), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kStore, kInt16, two, 1U, acc), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kLoad, kUInt16, two, 1U, acc), af::SUCCESS);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kUInt8, acc.max_latency, acc.throughput, two), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kDeInterleave, kUInt8, CreateExpr(1), 1U, acc), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::AddVfInstructPerf(kStore, kUInt8, two, 1U, acc), af::SUCCESS);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kUInt8, acc.max_latency, acc.throughput, CreateExpr(1)),
            af::SUCCESS);
  Expr result = ascendcapi_v2::GetVfCost(acc);
  result.Simplify();
  return result;
}

Expr ExpectedRank2GatherLedger(const std::string &dtype, bool gather_b, bool two) {
  // This ledger is copied from the fixed rank-2 branches in broadcast_3510_extend_impl.h.
  // Gather helpers run inside BroadcastExtend at runtime; reg codegen only emits the outer call.
  // Keep fixed constants here and do not use the production classifier or BuildTiling as oracle.
  VfCostAccumulator acc;
  const Expr rounds = two ? CreateExpr(2) : CreateExpr(1);
  const bool b8 = dtype == kUInt8 || dtype == kInt8;
  const std::string reg_dtype = b8 ? kUInt8 : (dtype == kFloat32 ? kUInt32 : kUInt16);
  if (gather_b) {
    AddExpectedExpr(kLoad, kUInt32, CreateExpr(1), acc);
    EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, reg_dtype, acc.max_latency, acc.throughput, CreateExpr(1)),
              af::SUCCESS);
    AddExpectedExpr(kUpdateMask, dtype, rounds, acc);
  } else {
    AddExpectedExpr(kDuplicate, kInt16, CreateExpr(1), acc);
    EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kInt16, acc.max_latency, acc.throughput, CreateExpr(1)),
              af::SUCCESS);
    AddExpectedExpr(kDiv, kInt16, CreateExpr(1), acc);
    AddExpectedExpr(kMul, kInt16, CreateExpr(1), acc);
    AddExpectedExpr(kSub, kInt16, CreateExpr(1), acc);
    AddExpectedExpr(kStore, kInt16, CreateExpr(1), acc);
    AddExpectedExpr(kLoad, reg_dtype, b8 ? CreateExpr(2) : CreateExpr(1), acc);
    EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, reg_dtype, acc.max_latency, acc.throughput,
                                             b8 ? CreateExpr(2) : CreateExpr(1)),
              af::SUCCESS);
    if (b8) {
      AddExpectedExpr(kDeInterleave, dtype, CreateExpr(1), acc);
    }
    if (!two) {
      AddExpectedExpr(kUpdateMask, dtype, CreateExpr(1), acc);
    }
  }
  AddExpectedExpr(kStore, dtype, rounds, acc);
  if (two && !gather_b) {
    EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, acc.max_latency, acc.throughput, CreateExpr(1)),
              af::SUCCESS);
  }
  return ascendcapi_v2::GetVfCost(acc);
}

struct BranchCase {
  std::string dtype;
  std::vector<int64_t> src;
  std::vector<int64_t> dst;
  NlastAxisBranch implementation_branch;
};

struct FixedVectorSpec {
  int64_t vl;
  int64_t half_vl;
  int64_t block;
};

FixedVectorSpec GetFixedVectorSpec(const std::string &dtype) {
  if (dtype == kUInt8 || dtype == kInt8) {
    return {256, 128, 32};
  }
  if (dtype == kFloat32 || dtype == kUInt32 || dtype == kInt32) {
    return {64, 32, 8};
  }
  return {128, 64, 16};
}

int64_t FixedProduct(const std::vector<int64_t> &shape) {
  int64_t result = 1;
  for (const auto dim : shape) {
    result *= dim;
  }
  return result;
}

// Independent loop specification copied from the source helper conditions; it is not a production classifier oracle.
NlastAxisBranch FixedNlastAxisBranch(const std::string &dtype, const std::vector<int64_t> &src,
                                     const std::vector<int64_t> &dst, int32_t const_rank) {
  const auto spec = GetFixedVectorSpec(dtype);
  const int64_t last = dst.back();
  const bool b8 = dtype == kUInt8 || dtype == kInt8;
  const bool aligned = last % spec.block == 0;
  const int64_t dst_size = FixedProduct(dst);
  const size_t rank = dst.size();
  if (const_rank > 4 || (rank > 4U && const_rank == -1)) {
    if (last < spec.half_vl && !b8) {
      return NlastAxisBranch::kDynamicGather;
    }
    if (last <= spec.vl) {
      return aligned ? NlastAxisBranch::kLessThanVlAligned : NlastAxisBranch::kLessThanVlUnaligned;
    }
    return aligned ? NlastAxisBranch::kLargerThanVlAlignedWithBlock : NlastAxisBranch::kLargerThanVlUnaligned;
  }
  if (rank == 2U) {
    if (last < spec.half_vl) {
      const auto gather = dst_size < spec.vl ? NlastAxisBranch::kGatherOne : NlastAxisBranch::kGatherTwo;
      return aligned
                 ? (gather == NlastAxisBranch::kGatherOne ? NlastAxisBranch::kGatherBOne : NlastAxisBranch::kGatherBTwo)
                 : gather;
    }
    if (last <= spec.vl) {
      return NlastAxisBranch::kLessThanVlUnaligned;
    }
    if (!aligned) {
      return const_rank == -1 ? NlastAxisBranch::kFallback : NlastAxisBranch::kLargerThanVlUnaligned;
    }
    return last % spec.vl == 0 && dst[0] > 8 ? NlastAxisBranch::kLargerThanVlAlignedWithVl
                                             : NlastAxisBranch::kLargerThanVlAlignedWithBlock;
  }
  if (rank == 3U) {
    if (last < spec.half_vl) {
      return NlastAxisBranch::kGather;
    }
    if (last <= spec.vl) {
      return NlastAxisBranch::kLessThanVlUnaligned;
    }
    if (aligned) {
      return last % spec.vl == 0 ? NlastAxisBranch::kLargerThanVlAlignedWithVl
                                 : NlastAxisBranch::kLargerThanVlAlignedWithBlock;
    }
    return NlastAxisBranch::kLargerThanVlUnaligned;
  }
  if (last < spec.half_vl && !b8) {
    return NlastAxisBranch::kGather;
  }
  if (last <= spec.vl) {
    return NlastAxisBranch::kLessThanVlUnaligned;
  }
  if (!aligned) {
    return NlastAxisBranch::kLargerThanVlUnaligned;
  }
  return last % spec.vl == 0 ? NlastAxisBranch::kLargerThanVlAlignedWithVl
                             : NlastAxisBranch::kLargerThanVlAlignedWithBlock;
}

ascendcapi_v2::LastAxisBranch FixedLastAxisRouteBranch(const std::vector<int64_t> &dst) {
  return dst.back() <= 128 ? ascendcapi_v2::LastAxisBranch::kLessThanVlUnaligned
                           : ascendcapi_v2::LastAxisBranch::kLargerThanVlAligned;
}

TEST(BroadcastNlastAxisPerfV2, RankTwoMatchesGatherAndAlignedBoundaries) {
  const std::vector<BranchCase> cases = {
      {kUInt16, {1, 16}, {2, 16}, NlastAxisBranch::kGatherBOne},
      {kUInt16, {1, 16}, {9, 16}, NlastAxisBranch::kGatherBTwo},
      {kUInt16, {1, 15}, {2, 15}, NlastAxisBranch::kGatherOne},
      {kUInt16, {1, 15}, {9, 15}, NlastAxisBranch::kGatherTwo},
      {kUInt16, {1, 64}, {2, 64}, NlastAxisBranch::kLessThanVlUnaligned},
      {kUInt16, {1, 128}, {2, 128}, NlastAxisBranch::kLessThanVlUnaligned},
      {kUInt16, {1, 256}, {8, 256}, NlastAxisBranch::kLargerThanVlAlignedWithBlock},
      {kUInt16, {1, 256}, {9, 256}, NlastAxisBranch::kLargerThanVlAlignedWithVl},
      {kUInt16, {1, 129}, {2, 129}, NlastAxisBranch::kLargerThanVlUnaligned},
  };
  for (const auto &test_case : cases) {
    const auto tiling = MakeTiling(test_case.dst);
    ASSERT_EQ(ascendcapi_v2::GetNlastAxisBranch(tiling, test_case.dtype, 2), test_case.implementation_branch);
    EXPECT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(MakeNode(test_case.dtype, test_case.src, test_case.dst)),
              test_case.implementation_branch);
  }
}

TEST(BroadcastPerfV2, RankOneUsesFallbackForLastAndNlast) {
  const auto node = MakeNode(kFloat16, {1}, {8});
  BroadcastTilingInfo tiling;
  const ParamExprInputs inputs{{CreateExpr(1)}, {CreateExpr(1)}, {CreateExpr(1)}};
  const ParamExprInputs outputs{{CreateExpr(8)}, {CreateExpr(8)}, {CreateExpr(8)}};
  ASSERT_EQ(BuildBroadcastTiling(node.broadcast_node_params.src_shape, node.broadcast_node_params.dst_shape, kFloat16,
                                 inputs, outputs, tiling),
            af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::GetLastAxisBranch(tiling, kFloat16, 1), ascendcapi_v2::LastAxisBranch::kFallback);
  EXPECT_EQ(ascendcapi_v2::GetNlastAxisBranch(tiling, kFloat16, 1), NlastAxisBranch::kFallback);
  EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(node), ascendcapi_v2::LastAxisBranch::kFallback);
  EXPECT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kFallback);
}

TEST(BroadcastPerfV2, ScalarRankZeroUsesV2Wrapper) {
  TensorShapeInfo input;
  input.data_type = kFloat16;
  TensorShapeInfo output;
  output.data_type = kFloat16;
  NodeInfo node;
  node.broadcast_node_params.valid = true;
  node.broadcast_node_params.is_scalar = true;
  node.broadcast_node_params.duplicate_count = CreateExpr(8);

  const auto api = ApiPerfFactory::Instance().Create(kBroadcast + "V2");
  ASSERT_NE(api, nullptr);
  PerfOutputInfo perf;
  ASSERT_EQ(api->GetPerfFunc()({input}, {output}, node, perf), af::SUCCESS);
  EXPECT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
}

TEST(BroadcastPerfV2, RejectsMultipleInputs) {
  TensorShapeInfo input;
  input.data_type = kFloat16;
  TensorShapeInfo output = input;
  NodeInfo node;
  node.broadcast_node_params.valid = true;
  node.broadcast_node_params.is_scalar = true;
  node.broadcast_node_params.duplicate_count = CreateExpr(8);
  PerfOutputInfo perf;

  EXPECT_NE(ApiPerfFactory::Instance().Create(kBroadcast + "V2")->GetPerfFunc()({input, input}, {output}, node, perf),
            af::SUCCESS);
}

TEST(BroadcastPerfV2, RejectsMultipleOutputsForScalar) {
  TensorShapeInfo input;
  input.data_type = kFloat16;
  TensorShapeInfo output = input;
  NodeInfo node;
  node.broadcast_node_params.valid = true;
  node.broadcast_node_params.is_scalar = true;
  node.broadcast_node_params.duplicate_count = CreateExpr(8);
  PerfOutputInfo perf;

  EXPECT_NE(ApiPerfFactory::Instance().Create(kBroadcast + "V2")->GetPerfFunc()({input}, {output, output}, node, perf),
            af::SUCCESS);
}

TEST(BroadcastNlastAxisPerfV2, RankThreeAndFourMatchGatherAndB8Branches) {
  const std::vector<BranchCase> cases = {
      {kUInt16, {2, 1, 32}, {2, 3, 32}, NlastAxisBranch::kGather},
      {kUInt16, {2, 1, 128}, {2, 3, 128}, NlastAxisBranch::kLessThanVlUnaligned},
      {kUInt16, {2, 1, 256}, {2, 3, 256}, NlastAxisBranch::kLargerThanVlAlignedWithVl},
      {kUInt16, {2, 1, 2, 32}, {2, 3, 2, 32}, NlastAxisBranch::kGather},
      {kUInt8, {2, 1, 2, 64}, {2, 3, 2, 64}, NlastAxisBranch::kLessThanVlAligned},
      {kUInt32, {2, 1, 2, 65}, {2, 3, 2, 65}, NlastAxisBranch::kLargerThanVlUnaligned},
  };
  for (const auto &test_case : cases) {
    const auto rank = static_cast<int32_t>(test_case.dst.size());
    const auto tiling = MakeTiling(test_case.dst);
    ASSERT_EQ(ascendcapi_v2::GetNlastAxisBranch(tiling, test_case.dtype, rank), test_case.implementation_branch);
    EXPECT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(MakeNode(test_case.dtype, test_case.src, test_case.dst)),
              test_case.implementation_branch);
  }
}

TEST(BroadcastNlastAxisPerfV2, DynamicAndB64MoreDimMatchImplementationBranches) {
  const std::vector<BranchCase> cases = {
      {kUInt16, {2, 1, 2, 1, 32}, {2, 3, 2, 3, 32}, NlastAxisBranch::kGatherWrapperForFourDim},
      {kUInt8, {2, 1, 2, 1, 64}, {2, 3, 2, 3, 64}, NlastAxisBranch::kLessThanVlAligned},
      {kUInt16, {2, 1, 2, 1, 256}, {2, 3, 2, 3, 256}, NlastAxisBranch::kLargerThanVlAlignedWithBlock},
      {kUInt16, {2, 1, 2, 1, 129}, {2, 3, 2, 3, 129}, NlastAxisBranch::kLargerThanVlUnaligned},
      {kUInt64, {2, 1, 2, 1}, {2, 3, 2, 4}, NlastAxisBranch::kB64MoreDimGather},
  };
  for (const auto &test_case : cases) {
    auto node = MakeNode(test_case.dtype, test_case.src, test_case.dst);
    EXPECT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), test_case.implementation_branch);
  }
}

TEST(BroadcastPerfV2, BranchInputsUseInputDimsForSizeAndRepeatsForActualSize) {
  auto size_node = MakeRoleNode(kFloat16, ascir_param::ParamExprRole::kSize, {2, 4, 128}, {2, 1, 128}, {2, 32, 128});
  auto actual_node =
      MakeRoleNode(kFloat16, ascir_param::ParamExprRole::kActualSize, {2, 4, 128}, {2, 1, 128}, {2, 32, 128});
  ParamExprInputs expected_size{{CreateExpr(2), CreateExpr(4), CreateExpr(128)},
                                {CreateExpr(2), CreateExpr(4), CreateExpr(128)},
                                {CreateExpr(2), CreateExpr(1), CreateExpr(128)}};
  ParamExprInputs expected_actual{{CreateExpr(2), CreateExpr(4), CreateExpr(128)},
                                  {CreateExpr(2), CreateExpr(4), CreateExpr(128)},
                                  {CreateExpr(2), CreateExpr(1), CreateExpr(128)}};
  BroadcastTilingInfo size_tiling;
  BroadcastTilingInfo actual_tiling;
  ASSERT_EQ(BuildBroadcastTiling(size_node.broadcast_node_params.src_shape, size_node.broadcast_node_params.dst_shape,
                                 kFloat16, expected_size, expected_size, size_tiling),
            af::SUCCESS);
  ASSERT_EQ(
      BuildBroadcastTiling(actual_node.broadcast_node_params.src_shape, actual_node.broadcast_node_params.dst_shape,
                           kFloat16, expected_actual, expected_actual, actual_tiling),
      af::SUCCESS);
  EXPECT_EQ(size_tiling.src_shape[1], CreateExpr(4));
  EXPECT_EQ(actual_tiling.src_shape[1], CreateExpr(1));
  EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(size_node), ascendcapi_v2::LastAxisBranch::kLessThanVlAligned);
  EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(actual_node), ascendcapi_v2::LastAxisBranch::kLessThanVlAligned);
  EXPECT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(size_node), ascendcapi_v2::NlastAxisBranch::kLessThanVlUnaligned);
  EXPECT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(actual_node), ascendcapi_v2::NlastAxisBranch::kLessThanVlUnaligned);
}

TEST(BroadcastTilingV2, PreservesOriginalAndFoldedRankForWrapperSelection) {
  const std::vector<ascir_param::ParamExprLeaf> src = {{CreateExpr(1), ascir_param::ParamExprRole::kSemantic},
                                                       {CreateExpr(1), ascir_param::ParamExprRole::kSemantic},
                                                       {CreateExpr(2), ascir_param::ParamExprRole::kSemantic},
                                                       {CreateExpr(1), ascir_param::ParamExprRole::kSemantic}};
  const std::vector<ascir_param::ParamExprLeaf> dst = {{CreateExpr(1), ascir_param::ParamExprRole::kSemantic},
                                                       {CreateExpr(3), ascir_param::ParamExprRole::kSemantic},
                                                       {CreateExpr(2), ascir_param::ParamExprRole::kSemantic},
                                                       {CreateExpr(4), ascir_param::ParamExprRole::kSemantic}};
  const ParamExprInputs inputs{{CreateExpr(1), CreateExpr(1), CreateExpr(2), CreateExpr(1)},
                               {CreateExpr(1), CreateExpr(1), CreateExpr(2), CreateExpr(1)},
                               {CreateExpr(1), CreateExpr(1), CreateExpr(2), CreateExpr(1)}};
  const ParamExprInputs outputs{{CreateExpr(1), CreateExpr(3), CreateExpr(2), CreateExpr(4)},
                                {CreateExpr(1), CreateExpr(3), CreateExpr(2), CreateExpr(4)},
                                {CreateExpr(1), CreateExpr(3), CreateExpr(2), CreateExpr(4)}};
  BroadcastTilingInfo tiling;
  ASSERT_EQ(BuildBroadcastTiling(src, dst, kFloat16, inputs, outputs, tiling), af::SUCCESS);
  EXPECT_EQ(tiling.original_rank, 4U);
  EXPECT_EQ(tiling.folded_rank, 4U);
  EXPECT_EQ(tiling.rank, 3U);
  EXPECT_EQ(tiling.original_src_shape[0], CreateExpr(1));
}

TEST(BroadcastNlastAxisPerfV2, CollapsedDynamicRankBelowFourUsesDynamicWrapper) {
  auto node = MakeNode(kUInt16, {1, 1, 1, 1, 16}, {2, 3, 4, 5, 16});
  ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kGatherBTwo);
}

TEST(BroadcastNlastAxisPerfV2, GatherB8BranchesBuildWithoutSeparateDeinterleave) {
  const auto one = MakeNode(kUInt8, {1, 32}, {2, 32});
  const auto two = MakeNode(kUInt8, {1, 32}, {9, 32});
  ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(one), NlastAxisBranch::kGatherBOne);
  ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(two), NlastAxisBranch::kGatherBTwo);
  PerfOutputInfo one_perf;
  PerfOutputInfo two_perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(one, one_perf), af::SUCCESS);
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(two, two_perf), af::SUCCESS);
  EXPECT_EQ(one_perf.pipe_res[PipeType::AIV_VEC], ExpectedGatherBPerf(false));
  EXPECT_EQ(two_perf.pipe_res[PipeType::AIV_VEC], ExpectedGatherBPerf(true));
}

TEST(BroadcastNlastAxisPerfV2, RegularNonLastDoesNotUseLegacyFallback) {
  auto node = MakeNode(kUInt16, {1, 16}, {9, 16});
  ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kGatherBTwo);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
}

TEST(BroadcastNlastAxisPerfV2, RankTwoGatherTwoRoundsUseCeilDivAndKeepGatherSingle) {
  auto first = MakeNode(kFloat16, {1, 16}, {9, 16});
  auto second = MakeNode(kFloat16, {1, 16}, {17, 16});
  PerfOutputInfo first_perf;
  PerfOutputInfo second_perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(first, first_perf), af::SUCCESS);
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(second, second_perf), af::SUCCESS);
  EXPECT_NE(first_perf.pipe_res[PipeType::AIV_VEC], second_perf.pipe_res[PipeType::AIV_VEC]);
}

TEST(BroadcastNlastAxisPerfV2, RegularNlastUsesRealFloatDtypeForUnalignedTail) {
  auto float16_node = MakeNode(kFloat16, {1, 129}, {2, 129});
  auto float32_node = MakeNode(kFloat32, {1, 129}, {2, 129});
  ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(float16_node), NlastAxisBranch::kLargerThanVlUnaligned);
  ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(float32_node), NlastAxisBranch::kLargerThanVlUnaligned);
  PerfOutputInfo float16_perf;
  PerfOutputInfo float32_perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(float16_node, float16_perf), af::SUCCESS);
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(float32_node, float32_perf), af::SUCCESS);
  EXPECT_NE(float16_perf.pipe_res[PipeType::AIV_VEC], float32_perf.pipe_res[PipeType::AIV_VEC]);
}

TEST(BroadcastNlastAxisPerfV2, DynamicUnalignedPathBuildsTailCost) {
  auto node = MakeNode(kFloat16, {2, 1, 2, 1, 129}, {2, 3, 2, 3, 129});
  ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kLargerThanVlUnaligned);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
}

TEST(BroadcastNlastAxisPerfV2, FoldedDynamicRankUsesOriginalTailAndOuterLoops) {
  const auto node = MakeNode(kUInt16, {1, 1, 1, 1, 16}, {2, 3, 4, 5, 16});
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
}

TEST(BroadcastNlastAxisPerfV2, DynamicAlignedBlockUsesOriginalFourDimCounts) {
  const auto node = MakeNode(kUInt16, {2, 1, 2, 1, 256}, {2, 3, 2, 3, 256});
  PerfOutputInfo perf;
  EXPECT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
}

TEST(BroadcastNlastAxisPerfV2, DynamicB64MoreDimGatherBuildsPerf) {
  const auto node = MakeNode(kUInt64, {2, 1, 2, 1, 1}, {2, 3, 2, 3, 4});
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
}

TEST(BroadcastNlastAxisPerfV2, B64OriginalRankFourUsesMoreDimGather) {
  auto node = MakeNode(kUInt64, {2, 1, 2, 1}, {2, 3, 2, 4});
  node.broadcast_node_params.const_rank = 4;
  ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kB64MoreDimGather);
}

TEST(BroadcastNlastAxisPerfV2, B64OriginalRankFiveDoesNotUseMoreDimGather) {
  auto node = MakeNode(kUInt64, {2, 1, 2, 1, 1}, {2, 3, 2, 3, 4});
  node.broadcast_node_params.const_rank = 5;
  EXPECT_NE(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kB64MoreDimGather);
}

TEST(BroadcastPerfV2, MissingGeneratedParamsUseLegacyPerf) {
  TensorShapeInfo input;
  input.data_type = kFloat16;
  input.dims = {CreateExpr(1), CreateExpr(16)};
  TensorShapeInfo output;
  output.data_type = kFloat16;
  output.dims = {CreateExpr(2), CreateExpr(16)};
  NodeInfo node;
  PerfOutputInfo perf;

  const auto api = ApiPerfFactory::Instance().Create(kBroadcast + "V2");
  ASSERT_NE(api, nullptr);
  ASSERT_EQ(api->GetPerfFunc()({input}, {output}, node, perf), af::SUCCESS);
  EXPECT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
}

TEST(BroadcastPerfV2, FactoryDispatchesV2SymbolicToTernaryOrFallback) {
  TensorShapeInfo input;
  input.data_type = kFloat16;
  input.dims = {CreateExpr(2), CreateExpr(1)};
  input.repeats = input.dims;
  TensorShapeInfo output;
  output.data_type = kFloat16;
  output.dims = {CreateExpr(2), CreateExpr("last")};
  NodeInfo node;
  node.broadcast_node_params.valid = true;
  node.broadcast_node_params.const_rank = 2;
  node.broadcast_node_params.src_shape = {
      {CreateExpr(2), ascir_param::ParamExprRole::kSemantic},
      {CreateExpr(1), ascir_param::ParamExprRole::kSemantic},
  };
  node.broadcast_node_params.dst_shape = {
      {CreateExpr(2), ascir_param::ParamExprRole::kSemantic},
      {CreateExpr("last"), ascir_param::ParamExprRole::kSemantic},
  };
  PerfOutputInfo perf;
  const auto api = ApiPerfFactory::Instance().Create(kBroadcast + "V2");
  ASSERT_NE(api, nullptr);
  ASSERT_EQ(api->GetPerfFunc()({input}, {output}, node, perf), af::SUCCESS);
  EXPECT_FALSE(perf.pipe_res.empty());
  EXPECT_FALSE(perf.ternary_ops.empty());
}

TEST(BroadcastPerfV2, FactoryV2RoutesLastAndNonLastByIndependentShapeSpec) {
  const auto api = ApiPerfFactory::Instance().Create(kBroadcast + "V2");
  ASSERT_NE(api, nullptr);
  const std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>, bool>> cases = {
      {{2, 1}, {2, 64}, true},
      {{1, 16}, {9, 16}, false},
  };
  for (const auto &[src, dst, last_axis] : cases) {
    TensorShapeInfo input;
    input.data_type = kFloat16;
    for (const auto dim : src) {
      input.dims.push_back(CreateExpr(dim));
      input.repeats.push_back(CreateExpr(dim));
    }
    TensorShapeInfo output;
    output.data_type = kFloat16;
    for (const auto dim : dst) {
      output.dims.push_back(CreateExpr(dim));
    }
    NodeInfo node;
    node.broadcast_node_params.valid = true;
    node.broadcast_node_params.const_rank = static_cast<int32_t>(src.size());
    for (size_t i = 0U; i < src.size(); ++i) {
      node.broadcast_node_params.src_shape.push_back({CreateExpr(src[i]), ascir_param::ParamExprRole::kSemantic});
      node.broadcast_node_params.dst_shape.push_back({CreateExpr(dst[i]), ascir_param::ParamExprRole::kSemantic});
    }
    PerfOutputInfo perf;
    ASSERT_EQ(api->GetPerfFunc()({input}, {output}, node, perf), af::SUCCESS);
    ASSERT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
    EXPECT_TRUE(!perf.pipe_res.empty() || !perf.ternary_ops.empty());
    const auto detail = MakeNode(kFloat16, src, dst);
    if (last_axis) {
      EXPECT_EQ(FixedLastAxisRouteBranch(dst), ascendcapi_v2::LastAxisBranch::kLessThanVlUnaligned);
      EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(detail), FixedLastAxisRouteBranch(dst));
    } else {
      const auto expected = FixedNlastAxisBranch(kFloat16, src, dst, static_cast<int32_t>(src.size()));
      EXPECT_EQ(expected, NlastAxisBranch::kGatherBTwo);
      EXPECT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(detail), expected);
    }
  }

  for (const auto &[src, dst, last_axis] : cases) {
    TensorShapeInfo input;
    input.data_type = kFloat16;
    input.dims = {CreateExpr(src[0]), CreateExpr(src[1])};
    input.repeats = input.dims;
    TensorShapeInfo output;
    output.data_type = kFloat16;
    output.dims = {CreateExpr(dst[0]), CreateExpr("last")};
    NodeInfo node;
    node.broadcast_node_params.valid = true;
    node.broadcast_node_params.const_rank = 2;
    node.broadcast_node_params.src_shape = {{CreateExpr(src[0]), ascir_param::ParamExprRole::kSemantic},
                                            {CreateExpr(src[1]), ascir_param::ParamExprRole::kSemantic}};
    node.broadcast_node_params.dst_shape = {{CreateExpr(dst[0]), ascir_param::ParamExprRole::kSemantic},
                                            {CreateExpr("last"), ascir_param::ParamExprRole::kSemantic}};
    PerfOutputInfo perf;
    ASSERT_EQ(api->GetPerfFunc()({input}, {output}, node, perf), af::SUCCESS);
    ASSERT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
    EXPECT_FALSE(perf.ternary_ops.empty());
    const auto detail = MakeNode(kFloat16, src, dst);
    if (last_axis) {
      EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(detail), FixedLastAxisRouteBranch(dst));
    } else {
      EXPECT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(detail),
                FixedNlastAxisBranch(kFloat16, src, dst, static_cast<int32_t>(src.size())));
    }
  }
}

TEST(BroadcastPerfV2, FactoryClassifiedUnknownRankThreeUsesStaticUnalignedResult) {
  TensorShapeInfo input;
  input.data_type = kFloat16;
  input.dims = {CreateExpr(2), CreateExpr(3), CreateExpr(1)};
  input.repeats = input.dims;
  TensorShapeInfo output;
  output.data_type = kFloat16;
  output.dims = {CreateExpr(2), CreateExpr(3), CreateExpr(65)};
  NodeInfo node;
  node.broadcast_node_params.valid = true;
  node.broadcast_node_params.const_rank = -1;
  for (size_t i = 0U; i < input.dims.size(); ++i) {
    node.broadcast_node_params.src_shape.push_back({input.dims[i], ascir_param::ParamExprRole::kSemantic});
    node.broadcast_node_params.dst_shape.push_back({output.dims[i], ascir_param::ParamExprRole::kSemantic});
  }
  PerfOutputInfo perf;
  auto direct_node = MakeNode(kFloat16, {2, 3, 1}, {2, 3, 65});
  direct_node.broadcast_node_params.const_rank = -1;
  const auto api = ApiPerfFactory::Instance().Create(kBroadcast + "V2");
  ASSERT_NE(api, nullptr);
  ASSERT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(direct_node),
            ascendcapi_v2::LastAxisBranch::kDynamicLessThanVlUnaligned);
  PerfOutputInfo direct;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(direct_node, direct), af::SUCCESS);
  ASSERT_EQ(api->GetPerfFunc()({input}, {output}, node, perf), af::SUCCESS);
  EXPECT_EQ(perf.pipe_res, direct.pipe_res);
}

TEST(BroadcastPerfV2, FactoryUnknownDynamicNlastDoesNotUseLegacyFallback) {
  const Expr last = CreateExpr("last");
  TensorShapeInfo input;
  input.data_type = kFloat16;
  input.dims = {CreateExpr(2), CreateExpr(1), CreateExpr(2), CreateExpr(1), last};
  input.repeats = input.dims;
  TensorShapeInfo output;
  output.data_type = kFloat16;
  output.dims = {CreateExpr(2), CreateExpr(3), CreateExpr(2), CreateExpr(3), last};
  NodeInfo node;
  node.broadcast_node_params.valid = true;
  node.broadcast_node_params.const_rank = -1;
  for (size_t i = 0U; i < output.dims.size(); ++i) {
    node.broadcast_node_params.src_shape.push_back({input.dims[i], ascir_param::ParamExprRole::kSemantic});
    node.broadcast_node_params.dst_shape.push_back({output.dims[i], ascir_param::ParamExprRole::kSemantic});
  }

  auto detail = MakeNode(kFloat16, {2, 1, 2, 1, 1}, {2, 3, 2, 3, 64});
  detail.broadcast_node_params.const_rank = -1;
  detail.broadcast_node_params.src_shape.back().expr = last;
  detail.broadcast_node_params.dst_shape.back().expr = last;
  detail.input_dims.back() = last;
  detail.repeats.back() = last;
  detail.output_dims.back() = last;
  EXPECT_FALSE(ascendcperf_v2::IsBroadcastFallback(detail));

  const auto api = ApiPerfFactory::Instance().Create(kBroadcast + "V2");
  ASSERT_NE(api, nullptr);
  PerfOutputInfo perf;
  const auto status = api->GetPerfFunc()({input}, {output}, node, perf);
  EXPECT_TRUE(status != af::SUCCESS || !perf.ternary_ops.empty());
}

TEST(BroadcastPerfV2, FactoryClassifiedFallbackStillUsesLegacyPerf) {
  TensorShapeInfo input;
  input.data_type = kFloat16;
  input.dims = {CreateExpr(1)};
  input.repeats = input.dims;
  TensorShapeInfo output;
  output.data_type = kFloat16;
  output.dims = {CreateExpr(16)};
  NodeInfo node;
  node.broadcast_node_params.valid = true;
  node.broadcast_node_params.const_rank = 1;
  node.broadcast_node_params.src_shape = {{CreateExpr(1), ascir_param::ParamExprRole::kSemantic}};
  node.broadcast_node_params.dst_shape = {{CreateExpr(16), ascir_param::ParamExprRole::kSemantic}};

  const auto api = ApiPerfFactory::Instance().Create(kBroadcast + "V2");
  ASSERT_NE(api, nullptr);
  PerfOutputInfo perf;
  ASSERT_EQ(api->GetPerfFunc()({input}, {output}, node, perf), af::SUCCESS);
  EXPECT_TRUE(perf.ternary_ops.empty());
  EXPECT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
}

TEST(BroadcastLastAxisPerfV2, UnknownRankNonAlignedUsesDynamicLastHelper) {
  auto node = MakeNode(kFloat16, {2, 1}, {2, 129});
  node.broadcast_node_params.const_rank = -1;
  EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(node), ascendcapi_v2::LastAxisBranch::kDynamicLargerThanVlUnaligned);
  PerfOutputInfo perf;
  EXPECT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
}

TEST(BroadcastLastAxisPerfV2, UnknownRankTwoLessThanVlUsesStaticUnalignedLeaf) {
  auto node = MakeNode(kFloat16, {2, 1}, {2, 64});
  node.broadcast_node_params.const_rank = -1;
  EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(node), ascendcapi_v2::LastAxisBranch::kLessThanVlUnaligned);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
}

TEST(BroadcastNlastAxisPerfV2, UnknownRankNonAlignedUsesDynamicNlastHelper) {
  auto node = MakeNode(kFloat16, {2, 1, 1}, {2, 3, 129});
  node.broadcast_node_params.const_rank = -1;
  EXPECT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kDynamicLargerThanVlUnaligned);
  PerfOutputInfo perf;
  EXPECT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
}

TEST(BroadcastPerfV2, SymbolicRankMoreThanFourFoldedRankFourUsesDynamicLeaf) {
  auto node = MakeNode(kFloat16, {2, 1, 2, 1, 1}, {2, 3, 2, 3, 64});
  node.broadcast_node_params.const_rank = -1;
  const auto symbol = CreateExpr("last");
  node.broadcast_node_params.dst_shape.back().expr = symbol;
  node.output_dims.back() = symbol;
  ParamExprInputs inputs{{CreateExpr(2), CreateExpr(1), CreateExpr(2), CreateExpr(1), CreateExpr(1)},
                         {CreateExpr(2), CreateExpr(1), CreateExpr(2), CreateExpr(1), CreateExpr(1)},
                         {CreateExpr(2), CreateExpr(1), CreateExpr(2), CreateExpr(1), CreateExpr(1)}};
  ParamExprInputs outputs{{CreateExpr(2), CreateExpr(3), CreateExpr(2), CreateExpr(3), symbol},
                          {CreateExpr(2), CreateExpr(3), CreateExpr(2), CreateExpr(3), symbol},
                          {CreateExpr(2), CreateExpr(3), CreateExpr(2), CreateExpr(3), symbol}};
  BroadcastTilingInfo tiling;
  ASSERT_EQ(BuildBroadcastTiling(node.broadcast_node_params.src_shape, node.broadcast_node_params.dst_shape, kFloat16,
                                 inputs, outputs, tiling),
            af::SUCCESS);
  ASSERT_TRUE(ascendcapi_v2::IsLastAxisBroadcast(tiling));
  EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(node), ascendcapi_v2::LastAxisBranch::kDynamicLargerThanVlUnaligned);
  PerfOutputInfo direct;
  ASSERT_EQ(ascendcperf_v2::BuildLastAxisPerf(node, tiling, direct), af::SUCCESS);
  PerfOutputInfo perf;
  EXPECT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_TRUE(perf.ternary_ops.empty());
}

TEST(BroadcastNlastAxisPerfV2, SymbolicRankFourB8DoesNotSelectGatherWrapper) {
  auto node = MakeNode(kUInt8, {2, 1, 2, 1}, {2, 3, 2, 64});
  node.broadcast_node_params.const_rank = -1;
  const auto symbol = CreateExpr("last");
  node.broadcast_node_params.dst_shape.back().expr = symbol;
  node.output_dims.back() = symbol;
  PerfOutputInfo perf;
  EXPECT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_FALSE(perf.ternary_ops.empty());
}

TEST(BroadcastPerfV2, FactoryParameterErrorsAreNotSwallowedByFallback) {
  TensorShapeInfo input;
  input.data_type = kFloat16;
  input.dims = {CreateExpr(1), CreateExpr(16)};
  input.repeats = {CreateExpr(1)};
  TensorShapeInfo output;
  output.data_type = kFloat16;
  output.dims = {CreateExpr(2), CreateExpr(16)};
  NodeInfo node;
  node.broadcast_node_params.valid = true;
  node.broadcast_node_params.src_shape = {
      {CreateExpr(1), ascir_param::ParamExprRole::kSemantic},
      {CreateExpr(16), ascir_param::ParamExprRole::kSemantic},
  };
  node.broadcast_node_params.dst_shape = {
      {CreateExpr(2), ascir_param::ParamExprRole::kSemantic},
      {CreateExpr(16), ascir_param::ParamExprRole::kSemantic},
  };
  PerfOutputInfo perf;
  const auto api = ApiPerfFactory::Instance().Create(kBroadcast + "V2");
  ASSERT_NE(api, nullptr);
  EXPECT_NE(api->GetPerfFunc()({input}, {output}, node, perf), af::SUCCESS);
}

TEST(BroadcastNlastAxisPerfV2, GatherWrapperRankThreeUsesExactGroupedLoopCounts) {
  const auto node = MakeNode(kFloat16, {1, 1, 4}, {20, 3, 4});
  PerfOutputInfo actual;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, actual), af::SUCCESS);
  EXPECT_EQ(actual.pipe_res[PipeType::AIV_VEC], ExpectedGatherWrapperPerf(kFloat16, 3U, CreateExpr(4), CreateExpr(3)));
}

TEST(BroadcastNlastAxisPerfV2, GatherWrapperRankFourUsesExactGroupedLoopCounts) {
  const auto node = MakeNode(kFloat16, {1, 1, 4, 5}, {2, 3, 4, 5});
  PerfOutputInfo actual;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, actual), af::SUCCESS);
  EXPECT_EQ(actual.pipe_res[PipeType::AIV_VEC], ExpectedGatherWrapperPerf(kFloat16, 4U, CreateExpr(4), CreateExpr(2)));
}

TEST(BroadcastNlastAxisPerfV2, GatherWrapperB8UsesTwoArithmeticLanesAndInt16Indices) {
  const auto node = MakeNode(kUInt8, {1, 1, 4}, {42, 3, 4});
  PerfOutputInfo actual;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, actual), af::SUCCESS);
  EXPECT_EQ(actual.pipe_res[PipeType::AIV_VEC], ExpectedGatherWrapperPerf(kUInt8, 3U, CreateExpr(8), CreateExpr(3)));
}

TEST(BroadcastNlastAxisPerfV2, RankTwoB8IndexGenerationUsesInt16) {
  const auto node = MakeNode(kUInt8, {1, 15}, {20, 15});
  PerfOutputInfo actual;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, actual), af::SUCCESS);
  EXPECT_EQ(actual.pipe_res[PipeType::AIV_VEC], ExpectedRank2B8GatherPerf());
}

TEST(BroadcastNlastAxisPerfV2, GatherWrapperB64UsesEffectiveB32GeneratedIndices) {
  const auto node = MakeNode(kUInt64, {1, 1, 2, 4}, {2, 3, 2, 4});
  PerfOutputInfo actual;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, actual), af::SUCCESS);
  EXPECT_EQ(actual.pipe_res[PipeType::AIV_VEC], ExpectedGatherWrapperPerf(kUInt32, 4U, CreateExpr(5), CreateExpr(3)));
}

TEST(BroadcastNlastAxisPerfV2, RankTwoLessThanVlCountsOneUpdateMaskAndOneLoadAlign) {
  const auto node = MakeNode(kFloat16, {1, 64}, {8, 64});
  PerfOutputInfo actual;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, actual), af::SUCCESS);
  EXPECT_EQ(actual.pipe_res[PipeType::AIV_VEC],
            ExpectedRank2UnalignedPerf(kFloat16, CreateExpr(8), CreateExpr(0), CreateExpr(0), true));
}

TEST(BroadcastNlastAxisPerfV2, UnknownRankTwoKnownShapeUsesStaticLessLeafLedger) {
  auto node = MakeNode(kFloat16, {1, 64}, {8, 64});
  node.broadcast_node_params.const_rank = -1;
  PerfOutputInfo actual;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, actual), af::SUCCESS);
  EXPECT_TRUE(actual.ternary_ops.empty());
  EXPECT_EQ(actual.pipe_res[PipeType::AIV_VEC], ExpectedRank2LessUnalignedPerf(kFloat16, CreateExpr(8)));
}

TEST(BroadcastNlastAxisPerfV2, UnknownRankThreeAndFourKnownShapesUseDirectLeaves) {
  const std::vector<NodeDetail> nodes = {
      MakeNode(kFloat32, {2, 1, 65}, {2, 3, 65}),
      MakeNode(kFloat32, {2, 1, 2, 65}, {2, 3, 2, 65}),
  };
  for (auto node : nodes) {
    node.broadcast_node_params.const_rank = -1;
    PerfOutputInfo actual;
    ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, actual), af::SUCCESS);
    EXPECT_TRUE(actual.ternary_ops.empty());
    EXPECT_NE(actual.pipe_res.find(PipeType::AIV_VEC), actual.pipe_res.end());
  }
}

TEST(BroadcastNlastAxisPerfV2, RankTwoLargerUnalignedCountsLoadPreFullLoadsAndTailStores) {
  const auto node = MakeNode(kFloat16, {1, 129}, {9, 129});
  PerfOutputInfo actual;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, actual), af::SUCCESS);
  EXPECT_EQ(actual.pipe_res[PipeType::AIV_VEC],
            ExpectedRank2UnalignedPerf(kFloat16, CreateExpr(9), CreateExpr(9), CreateExpr(0), false));
}

TEST(BroadcastNlastAxisPerfV2, RankThreeLessThanVlCountsLoadPrePerOuterSlice) {
  const auto node = MakeNode(kFloat32, {2, 1, 33}, {2, 3, 33});
  PerfOutputInfo actual;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, actual), af::SUCCESS);
  EXPECT_EQ(actual.pipe_res[PipeType::AIV_VEC],
            ExpectedRank3UnalignedPerf(kFloat32, CreateExpr(6), CreateExpr(6), CreateExpr(6), true));
}

TEST(BroadcastNlastAxisPerfV2, RankThreeLargerUnalignedCountsLoadPreFullAndTailLoads) {
  const auto node = MakeNode(kFloat32, {2, 1, 65}, {2, 3, 65});
  PerfOutputInfo actual;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, actual), af::SUCCESS);
  EXPECT_EQ(actual.pipe_res[PipeType::AIV_VEC],
            ExpectedRank3UnalignedPerf(kFloat32, CreateExpr(6), CreateExpr(12), CreateExpr(12), false));
}

TEST(BroadcastNlastAxisPerfV2, RankFourLargerUnalignedCountsLoadPrePerOuterSlice) {
  const auto node = MakeNode(kFloat32, {2, 1, 2, 65}, {2, 3, 2, 65});
  PerfOutputInfo actual;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, actual), af::SUCCESS);
  EXPECT_EQ(actual.pipe_res[PipeType::AIV_VEC],
            ExpectedRank3UnalignedPerf(kFloat32, CreateExpr(12), CreateExpr(24), CreateExpr(24), false));
}

TEST(BroadcastNlastAxisPerfV2, RankThreeAlignedLeavesUseSourceLoopCounts) {
  const auto less = MakeNode(kFloat16, {2, 1, 64}, {2, 3, 64});
  const auto block = MakeNode(kFloat16, {2, 1, 144}, {2, 3, 144});
  const auto vl = MakeNode(kFloat16, {2, 1, 256}, {2, 3, 256});
  PerfOutputInfo less_perf;
  PerfOutputInfo block_perf;
  PerfOutputInfo vl_perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(less, less_perf), af::SUCCESS);
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(block, block_perf), af::SUCCESS);
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(vl, vl_perf), af::SUCCESS);
  EXPECT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(block), NlastAxisBranch::kLargerThanVlAlignedWithBlock);
  EXPECT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(vl), NlastAxisBranch::kLargerThanVlAlignedWithVl);
  EXPECT_EQ(less_perf.pipe_res[PipeType::AIV_VEC],
            ExpectedAlignedLeaf(kFloat16, CreateExpr(6), CreateExpr(1), CreateExpr(6)));
  EXPECT_EQ(block_perf.pipe_res[PipeType::AIV_VEC],
            ExpectedAlignedLeaf(kFloat16, CreateExpr(12), CreateExpr(4), CreateExpr(12)));
  EXPECT_EQ(vl_perf.pipe_res[PipeType::AIV_VEC],
            ExpectedAlignedLeaf(kFloat16, CreateExpr(12), CreateExpr(0), CreateExpr(12)));
}

TEST(BroadcastNlastAxisPerfV2, RankFourAlignedAndUnalignedLeavesAreIndependent) {
  const auto less_aligned = MakeNode(kFloat16, {2, 1, 2, 64}, {2, 3, 2, 64});
  const auto less_unaligned = MakeNode(kFloat16, {2, 1, 2, 65}, {2, 3, 2, 65});
  const auto larger_vl = MakeNode(kFloat16, {2, 1, 2, 256}, {2, 3, 2, 256});
  const auto larger_block = MakeNode(kFloat16, {2, 1, 2, 144}, {2, 3, 2, 144});
  const auto larger_unaligned = MakeNode(kFloat16, {2, 1, 2, 129}, {2, 3, 2, 129});
  const std::vector<NodeDetail> nodes = {less_aligned, less_unaligned, larger_vl, larger_block, larger_unaligned};
  for (const auto &node : nodes) {
    PerfOutputInfo perf;
    ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
    EXPECT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
  }
  PerfOutputInfo aligned_perf;
  PerfOutputInfo block_perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(less_aligned, aligned_perf), af::SUCCESS);
  PerfOutputInfo less_unaligned_perf;
  PerfOutputInfo larger_vl_perf;
  PerfOutputInfo larger_unaligned_perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(less_unaligned, less_unaligned_perf), af::SUCCESS);
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(larger_vl, larger_vl_perf), af::SUCCESS);
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(larger_block, block_perf), af::SUCCESS);
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(larger_unaligned, larger_unaligned_perf), af::SUCCESS);
  EXPECT_EQ(aligned_perf.pipe_res[PipeType::AIV_VEC],
            ExpectedAlignedLeaf(kFloat16, CreateExpr(12), CreateExpr(1), CreateExpr(12)));
  EXPECT_EQ(less_unaligned_perf.pipe_res[PipeType::AIV_VEC],
            ExpectedRank3UnalignedPerf(kFloat16, CreateExpr(12), CreateExpr(12), CreateExpr(12), true));
  EXPECT_EQ(larger_vl_perf.pipe_res[PipeType::AIV_VEC],
            ExpectedAlignedLeaf(kFloat16, CreateExpr(24), CreateExpr(0), CreateExpr(24)));
  EXPECT_EQ(block_perf.pipe_res[PipeType::AIV_VEC],
            ExpectedAlignedLeaf(kFloat16, CreateExpr(24), CreateExpr(6), CreateExpr(24)));
  EXPECT_EQ(larger_unaligned_perf.pipe_res[PipeType::AIV_VEC],
            ExpectedRank3UnalignedPerf(kFloat16, CreateExpr(12), CreateExpr(24), CreateExpr(24), false));
}

TEST(BroadcastNlastAxisPerfV2, DynamicRanksFiveThroughNineLessAlignedLeavesUseAlignedBranch) {
  for (size_t rank = 5U; rank <= 9U; ++rank) {
    std::vector<int64_t> dst(rank, 2);
    dst.back() = 64;
    const auto node = MakeDynamicNlastNode(kFloat16, dst);
    ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kLessThanVlAligned);
    PerfOutputInfo perf;
    ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
    EXPECT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
  }
}

TEST(BroadcastNlastAxisPerfV2, DynamicRanksFiveThroughNineUnalignedLeavesUseStaticFormula) {
  for (size_t rank = 5U; rank <= 9U; ++rank) {
    std::vector<int64_t> dst(rank, 2);
    dst.back() = 65;
    const auto node = MakeDynamicNlastNode(kFloat16, dst);
    const auto expected_branch = rank == 5U ? NlastAxisBranch::kDynamicLargerThanVlUnaligned
                                            : (rank % 2U == 0U ? NlastAxisBranch::kLessThanVlUnaligned
                                                               : NlastAxisBranch::kLargerThanVlUnaligned);
    ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), expected_branch);
    PerfOutputInfo perf;
    ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
    EXPECT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
  }
}

TEST(BroadcastNlastAxisPerfV2, DynamicRanksFiveThroughNineLargerBranchesKeepBlockAndUnalignedRoutes) {
  for (const int64_t last : {144, 129}) {
    for (size_t rank = 5U; rank <= 9U; ++rank) {
      std::vector<int64_t> dst(rank, 2);
      dst.back() = last;
      const auto node = MakeDynamicNlastNode(kFloat16, dst);
      const bool block = last == 144;
      const auto expected = block ? ExpectedAlignedLeaf(kFloat16, CreateExpr(32), CreateExpr(4), CreateExpr(32))
                                  : ExpectedDynamicLargerUnalignedLeaf(kFloat16, dst, true);
      const auto branch = block ? NlastAxisBranch::kLargerThanVlAlignedWithBlock
                                : (rank == 5U ? NlastAxisBranch::kDynamicLargerThanVlUnaligned
                                              : NlastAxisBranch::kLargerThanVlUnaligned);
      ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), branch);
      PerfOutputInfo perf;
      ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
      EXPECT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
    }
  }
}

TEST(BroadcastNlastAxisPerfV2, RankTwoGatherBTwoHasNoUnalignedPost) {
  const auto node = MakeNode(kUInt16, {1, 16}, {9, 16});
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_EQ(perf.pipe_res[PipeType::AIV_VEC], ExpectedGatherBPerf(true));
}

TEST(BroadcastNlastAxisPerfV2, RankTwoAllGatherLeavesMatchIndependentLedgers) {
  const std::vector<std::tuple<std::string, std::vector<int64_t>, std::vector<int64_t>, bool, bool>> cases = {
      {kFloat16, {1, 15}, {2, 15}, false, false},
      {kFloat16, {1, 15}, {9, 15}, false, true},
      {kUInt8, {1, 32}, {2, 32}, true, false},
      {kUInt8, {1, 32}, {9, 32}, true, true},
  };
  for (const auto &[dtype, src, dst, gather_b, two] : cases) {
    const auto node = MakeNode(dtype, src, dst);
    PerfOutputInfo perf;
    ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
    EXPECT_EQ(perf.pipe_res[PipeType::AIV_VEC], ExpectedRank2GatherLedger(dtype, gather_b, two));
  }
}

TEST(BroadcastNlastAxisPerfV2, RankFourB8SmallTailSkipsGatherWrapper) {
  const auto node = MakeNode(kUInt8, {2, 1, 2, 64}, {2, 3, 2, 64});
  ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kLessThanVlAligned);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_EQ(perf.pipe_res[PipeType::AIV_VEC],
            ExpectedAlignedLeaf(kUInt8, CreateExpr(12), CreateExpr(1), CreateExpr(12)));
}

TEST(BroadcastNlastAxisPerfV2, B64ConstRankFourMoreDimUsesOuterAndEffectiveDtype) {
  const auto node = MakeNode(kUInt64, {2, 1, 8, 1}, {2, 3, 8, 4});
  const auto tiling = [&node]() {
    const auto src_inputs = ParamExprInputs{node.input_dims, node.input_dims, node.repeats};
    const auto dst_inputs = ParamExprInputs{node.output_dims, node.output_dims, node.output_dims};
    BroadcastTilingInfo result;
    EXPECT_EQ(BuildBroadcastTiling(node.broadcast_node_params.src_shape, node.broadcast_node_params.dst_shape,
                                   node.input_dtype[0], src_inputs, dst_inputs, result),
              af::SUCCESS);
    return result;
  }();
  ASSERT_EQ(tiling.rank, 5U);
  EXPECT_EQ(tiling.dst_shape[0], CreateExpr(2));
  EXPECT_EQ(tiling.src_stride[4], CreateExpr(1));
  EXPECT_EQ(tiling.src_stride[3], CreateExpr(0));
  EXPECT_EQ(ascendcapi_v2::GetNlastAxisBranch(tiling, kUInt64, 4), NlastAxisBranch::kB64MoreDimGather);
  PerfOutputInfo actual;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, actual), af::SUCCESS);
  // B64 uses VL=64 for loop grouping and effective B32 index/gather registers; outer dstShape[0] is 2.
  EXPECT_EQ(actual.pipe_res[PipeType::AIV_VEC], ExpectedB64MoreDimGatherPerf());
}

TEST(BroadcastNlastAxisPerfV2, RankTwoLargerAlignedUsesIndependentOuterLedgers) {
  const auto block = MakeNode(kFloat16, {8, 1}, {8, 256});
  const auto vl = MakeNode(kFloat16, {9, 1}, {9, 256});
  PerfOutputInfo block_perf;
  PerfOutputInfo vl_perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(block, block_perf), af::SUCCESS);
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(vl, vl_perf), af::SUCCESS);
  EXPECT_EQ(block_perf.pipe_res[PipeType::AIV_VEC], ExpectedRank2LargerAlignedLeaf(kFloat16, 8, false));
  EXPECT_EQ(vl_perf.pipe_res[PipeType::AIV_VEC], ExpectedRank2LargerAlignedLeaf(kFloat16, 9, false));
}

TEST(BroadcastNlastAxisPerfV2, RankFourLeavesUseIndependentFixedFormulaLedgers) {
  const auto less = MakeNode(kFloat16, {2, 1, 2, 64}, {2, 3, 2, 64});
  const auto unaligned = MakeNode(kFloat16, {2, 1, 2, 65}, {2, 3, 2, 65});
  const auto aligned = MakeNode(kFloat16, {2, 1, 2, 144}, {2, 3, 2, 144});
  PerfOutputInfo less_perf;
  PerfOutputInfo unaligned_perf;
  PerfOutputInfo aligned_perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(less, less_perf), af::SUCCESS);
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(unaligned, unaligned_perf), af::SUCCESS);
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(aligned, aligned_perf), af::SUCCESS);
  EXPECT_EQ(less_perf.pipe_res[PipeType::AIV_VEC], ExpectedRank4LessAlignedLeaf(kFloat16));
  EXPECT_EQ(unaligned_perf.pipe_res[PipeType::AIV_VEC], ExpectedRank4LessUnalignedLeaf(kFloat16));
  EXPECT_EQ(aligned_perf.pipe_res[PipeType::AIV_VEC], ExpectedRank4LargerAlignedLeaf(kFloat16));
}

TEST(BroadcastNlastAxisPerfV2, DynamicGatherRanksFiveThroughNineUsePrefixMultipliers) {
  for (size_t rank = 5U; rank <= 9U; ++rank) {
    std::vector<int64_t> dst(rank, 2);
    dst.back() = 32;
    const auto node = MakeDynamicNlastNode(kFloat16, dst);
    PerfOutputInfo perf;
    if (rank % 2U == 0U) {
      ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kGatherWrapperForFourDim);
      ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
      int64_t multiplier = 1;
      for (size_t i = 0U; i + 4U < rank; ++i) {
        multiplier *= dst[i];
      }
      EXPECT_EQ(perf.pipe_res[PipeType::AIV_VEC], ExpectedDynamicGatherLeaf(kFloat16, multiplier));
    } else {
      ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kLessThanVlAligned);
      ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
      EXPECT_NE(perf.pipe_res.find(PipeType::AIV_VEC), perf.pipe_res.end());
    }
  }
}

TEST(BroadcastNlastAxisPerfV2, B64RightAlignedRankTwoAppendsOneDimensionWithExactLedger) {
  const auto node = MakeNode(kUInt64, {2, 1}, {2, 4});
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_EQ(perf.pipe_res[PipeType::AIV_VEC], ExpectedB64RightAlignedGatherLeaf());
}

TEST(BroadcastNlastAxisPerfV2, FixedTruthTableDoesNotUseClassifierAsOracle) {
  const std::vector<BranchCase> cases = {
      {kFloat16, {1, 15}, {2, 15}, NlastAxisBranch::kGatherOne},
      {kFloat16, {1, 64}, {2, 64}, NlastAxisBranch::kLessThanVlUnaligned},
      {kFloat16, {2, 1, 64}, {2, 3, 64}, NlastAxisBranch::kLessThanVlUnaligned},
      {kFloat16, {2, 1, 65}, {2, 3, 65}, NlastAxisBranch::kLessThanVlUnaligned},
      {kFloat16, {2, 1, 144}, {2, 3, 144}, NlastAxisBranch::kLargerThanVlAlignedWithBlock},
      {kFloat16, {2, 1, 129}, {2, 3, 129}, NlastAxisBranch::kLargerThanVlUnaligned},
      {kFloat16, {2, 1, 32}, {2, 3, 32}, NlastAxisBranch::kGather},
  };
  for (const auto &test_case : cases) {
    const auto node = MakeNode(test_case.dtype, test_case.src, test_case.dst);
    EXPECT_EQ(
        FixedNlastAxisBranch(test_case.dtype, test_case.src, test_case.dst, node.broadcast_node_params.const_rank),
        test_case.implementation_branch);
    EXPECT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), test_case.implementation_branch);
  }
}

TEST(BroadcastNlastAxisPerfV2, DynamicRankTwoLessThanVlUsesUnalignedLeafDirectly) {
  auto node = MakeNode(kFloat16, {1, 2}, {2, 64});
  node.broadcast_node_params.const_rank = -1;
  ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kLessThanVlUnaligned);

  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_EQ(perf.pipe_res[PipeType::AIV_VEC], ExpectedRank2LessUnalignedPerf(kFloat16, CreateExpr(2)));
}

TEST(BroadcastNlastAxisPerfV2, DynamicRankThreeLessThanVlUsesUnalignedLeaf) {
  auto node = MakeNode(kFloat16, {2, 1, 1}, {2, 3, 64});
  node.broadcast_node_params.const_rank = -1;
  EXPECT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kLessThanVlUnaligned);
}

TEST(BroadcastNlastAxisPerfV2, DynamicRankMoreThanFourAlignedLeavesFallback) {
  const auto less = MakeNode(kFloat16, {2, 1, 2, 1, 64}, {2, 3, 2, 3, 64});
  const auto larger = MakeNode(kFloat16, {2, 1, 2, 1, 256}, {2, 3, 2, 3, 256});
  EXPECT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(less), NlastAxisBranch::kLessThanVlAligned);
  EXPECT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(larger), NlastAxisBranch::kLargerThanVlAlignedWithBlock);
}

Expr ExpectedDynamicLastLeaf(const std::string &dtype, const Expr &outer, const Expr &stores,
                             const Expr &helper_calls) {
  VfCostAccumulator acc;
  AddExpectedExpr(kLoad, dtype, outer, acc);
  AddExpectedExpr(kStore, dtype, stores, acc);
  std::string helper_dtype;
  EXPECT_EQ(ascendcapi_v2::GetEffectiveHelperDtype(dtype, helper_dtype), af::SUCCESS);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, helper_dtype, acc.max_latency, acc.throughput, helper_calls),
            af::SUCCESS);
  return ascendcapi_v2::GetVfCost(acc);
}

TEST(BroadcastNlastAxisPerfV2, FoldedRankThreeWithBlockCountsUseFoldedShape) {
  const auto node = MakeNode(kFloat16, {2, 3, 1, 5, 64}, {2, 3, 7, 5, 64});
  ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kLargerThanVlAlignedWithBlock);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_EQ(perf.pipe_res[PipeType::AIV_VEC],
            ExpectedAlignedLeaf(kFloat16, CreateExpr(126), CreateExpr(18), CreateExpr(126)));
}

TEST(BroadcastNlastAxisPerfV2, FoldedRankThreeWithVlCountsUseFoldedShape) {
  const auto node = MakeNode(kFloat32, {2, 3, 1, 5, 64}, {2, 3, 7, 5, 64});
  ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kLargerThanVlAlignedWithVl);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_EQ(perf.pipe_res[PipeType::AIV_VEC],
            ExpectedAlignedLeaf(kFloat32, CreateExpr(210), CreateExpr(0), CreateExpr(210)));
}

TEST(BroadcastLastAxisPerfV2, FoldedRankFourLessAlignedCountsUseFoldedShape) {
  const auto node = MakeNode(kFloat16, {2, 3, 1, 5, 1}, {2, 3, 7, 5, 64});
  ASSERT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(node), ascendcapi_v2::LastAxisBranch::kLessThanVlAligned);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_EQ(perf.pipe_res[PipeType::AIV_VEC],
            ExpectedAlignedLeaf(kFloat16, CreateExpr(210), CreateExpr(1), CreateExpr(210)));
}

TEST(BroadcastLastAxisPerfV2, FoldedRankFourLargerUnalignedUsesDynamicLeafCounts) {
  const auto node = MakeNode(kFloat16, {2, 3, 1, 5, 1}, {2, 3, 7, 5, 130});
  ASSERT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(node), ascendcapi_v2::LastAxisBranch::kDynamicLargerThanVlUnaligned);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_EQ(perf.pipe_res[PipeType::AIV_VEC],
            ExpectedDynamicLastLeaf(kFloat16, CreateExpr(210), CreateExpr(420), CreateExpr(1)));
}

TEST(BroadcastNlastAxisPerfV2, B64RankTwoAppendGatherUsesEffectiveVectorLength) {
  const auto node = MakeNode(kUInt64, {4, 1}, {4, 16});
  ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kGather);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_EQ(perf.pipe_res[PipeType::AIV_VEC], ExpectedGatherWrapperPerf(kUInt32, 3U, CreateExpr(4), CreateExpr(3)));
}

TEST(BroadcastNlastAxisPerfV2, B64FoldedRankThreeWithVlUsesDoubledShape) {
  const auto node = MakeNode(kUInt64, {2, 3, 1, 5, 64}, {2, 3, 7, 5, 64});
  ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kLargerThanVlAlignedWithVl);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_EQ(perf.pipe_res[PipeType::AIV_VEC],
            ExpectedAlignedLeaf(kUInt64, CreateExpr(420), CreateExpr(0), CreateExpr(420)));
}

TEST(BroadcastNlastAxisPerfV2, B64FoldedRankThreeWithBlockUsesDoubledShape) {
  const auto node = MakeNode(kUInt64, {2, 3, 1, 5, 72}, {2, 3, 7, 5, 72});
  ASSERT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kLargerThanVlAlignedWithBlock);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  EXPECT_EQ(perf.pipe_res[PipeType::AIV_VEC],
            ExpectedAlignedLeaf(kUInt64, CreateExpr(504), CreateExpr(72), CreateExpr(504)));
}

}  // namespace
}  // namespace att
