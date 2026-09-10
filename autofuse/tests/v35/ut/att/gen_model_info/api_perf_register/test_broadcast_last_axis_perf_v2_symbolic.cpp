/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms of
 * CANN Open Software License Agreement Version 2.0 of the License.
 */

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "att/api_perf_register/perf_param_v2.h"
#include "ascir_node_param/ascir_node_param.h"
#include "base/att_const_values.h"
#include "v35/att/api_perf_register/ascendc_api_perf/broadcast_api_perf_v2.h"
#include "v35/att/api_perf_register/ascendc_api_perf/broadcast_last_axis_perf_v2.h"
#include "v35/att/api_perf_register/ascendc_api_perf/broadcast_nlast_axis_perf_v2.h"

namespace att {
namespace {
using ascendcapi_v2::NlastAxisBranch;

NodeDetail MakeSymbolicNode(const std::vector<Expr> &src, const std::vector<Expr> &dst) {
  NodeDetail node;
  node.broadcast_node_params.valid = true;
  node.broadcast_node_params.const_rank = static_cast<int32_t>(src.size());
  for (size_t i = 0U; i < src.size(); ++i) {
    node.broadcast_node_params.src_shape.push_back({src[i], ascir_param::ParamExprRole::kSemantic});
    node.broadcast_node_params.dst_shape.push_back({dst[i], ascir_param::ParamExprRole::kSemantic});
    node.input_dims.push_back(src[i]);
    node.output_dims.push_back(dst[i]);
    node.repeats.push_back(src[i]);
  }
  node.input_dtype = {kFloat16};
  node.output_dtype = {kFloat16};
  return node;
}

TEST(BroadcastLastAxisPerfV2Symbolic, RankTwoRegistersFinalDynamicTreeVariable) {
  const auto node = MakeSymbolicNode({CreateExpr(2), CreateExpr(1)}, {CreateExpr(2), CreateExpr("last")});
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  ASSERT_FALSE(perf.ternary_ops.empty());
  const auto replacements = ConcursiveReplaceVars(perf.ternary_ops);
  EXPECT_NE(Str(perf.pipe_res.at(PipeType::AIV_VEC).Replace(replacements)).find("TernaryOp"), std::string::npos);
}

TEST(BroadcastLastAxisPerfV2Symbolic, RankThreeE2BConditionsCascadeIntoFinalVariable) {
  const auto node = MakeSymbolicNode({CreateExpr(2), CreateExpr("middle"), CreateExpr(1)},
                                     {CreateExpr(2), CreateExpr("middle"), CreateExpr("last")});
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  ASSERT_FALSE(perf.ternary_ops.empty());
  const auto replacements = ConcursiveReplaceVars(perf.ternary_ops);
  const auto expression = Str(perf.pipe_res.at(PipeType::AIV_VEC).Replace(replacements));
  EXPECT_NE(expression.find("last"), std::string::npos);
  EXPECT_NE(expression.find("TernaryOp"), std::string::npos);
}

TEST(BroadcastLastAxisPerfV2Symbolic, RankGreaterThanFourUnknownLastUsesDynamicTernaryRoute) {
  auto node = MakeSymbolicNode({CreateExpr(2), CreateExpr(1), CreateExpr(2), CreateExpr(1), CreateExpr("last")},
                               {CreateExpr(2), CreateExpr(3), CreateExpr(2), CreateExpr(3), CreateExpr("last")});
  node.broadcast_node_params.const_rank = -1;
  const auto src_inputs = ascendcapi_v2::ParamExprInputs{node.input_dims, node.input_dims, node.repeats};
  const auto dst_inputs = ascendcapi_v2::ParamExprInputs{node.output_dims, node.output_dims, node.output_dims};
  ascendcapi_v2::BroadcastTilingInfo tiling;
  ASSERT_EQ(
      ascendcapi_v2::BuildBroadcastTiling(node.broadcast_node_params.src_shape, node.broadcast_node_params.dst_shape,
                                          kFloat16, src_inputs, dst_inputs, tiling),
      af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::GetLastAxisBranch(tiling, kFloat16, -1), ascendcapi_v2::LastAxisBranch::kFallback);
  EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(node), ascendcapi_v2::LastAxisBranch::kFallback);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  ASSERT_FALSE(perf.ternary_ops.empty());
  const auto replacements = ConcursiveReplaceVars(perf.ternary_ops);
  const auto expression = Str(perf.pipe_res.at(PipeType::AIV_VEC).Replace(replacements));
  EXPECT_NE(expression.find("last"), std::string::npos);
  EXPECT_NE(expression.find("TernaryOp"), std::string::npos);
}

TEST(BroadcastNlastAxisPerfV2Symbolic, RankTwoUnknownLastBuildsTernaryTree) {
  const auto node = MakeSymbolicNode({CreateExpr(1), CreateExpr("last")}, {CreateExpr(2), CreateExpr("last")});
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  ASSERT_FALSE(perf.ternary_ops.empty());
  const auto replacements = ConcursiveReplaceVars(perf.ternary_ops);
  const auto expression = Str(perf.pipe_res.at(PipeType::AIV_VEC).Replace(replacements));
  EXPECT_NE(expression.find("last"), std::string::npos);
  EXPECT_NE(expression.find("TernaryOp"), std::string::npos);
}

TEST(BroadcastNlastAxisPerfV2Symbolic, RankThreeAndFourUnknownLastBuildTernaryTrees) {
  for (const size_t rank : {3U, 4U}) {
    std::vector<Expr> src(rank, CreateExpr(1));
    std::vector<Expr> dst(rank, CreateExpr(2));
    src.back() = CreateExpr(32);
    dst.back() = CreateExpr("last");
    const auto node = MakeSymbolicNode(src, dst);
    PerfOutputInfo perf;
    ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
    EXPECT_FALSE(perf.ternary_ops.empty());
    const auto replacements = ConcursiveReplaceVars(perf.ternary_ops);
    EXPECT_NE(Str(perf.pipe_res.at(PipeType::AIV_VEC).Replace(replacements)).find("TernaryOp"), std::string::npos);
  }
}

TEST(BroadcastNlastAxisPerfV2Symbolic, RankGreaterThanFourUnknownLastBuildsDynamicTree) {
  const auto node = MakeSymbolicNode({CreateExpr(2), CreateExpr(1), CreateExpr(2), CreateExpr(1), CreateExpr("last")},
                                     {CreateExpr(2), CreateExpr(3), CreateExpr(2), CreateExpr(3), CreateExpr("last")});
  EXPECT_NE(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kFallback);
}

TEST(BroadcastNlastAxisPerfV2Symbolic, RankGreaterThanFourUnknownB64LastUsesDynamicTree) {
  auto node = MakeSymbolicNode({CreateExpr(2), CreateExpr(1), CreateExpr(2), CreateExpr(1), CreateExpr("last")},
                               {CreateExpr(2), CreateExpr(3), CreateExpr(2), CreateExpr(3), CreateExpr("last")});
  node.input_dtype = {kUInt64};
  node.output_dtype = {kUInt64};
  EXPECT_NE(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kFallback);
  EXPECT_FALSE(ascendcperf_v2::IsBroadcastFallback(node));
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  ASSERT_GE(perf.ternary_ops.size(), 4U);
  for (const auto &entry : perf.ternary_ops) {
    EXPECT_FALSE(entry.second.GetTernaryOpStr().empty());
  }
}

TEST(BroadcastNlastAxisPerfV2Symbolic, RankGreaterThanFourB8SkipsGatherLeaf) {
  auto node = MakeSymbolicNode({CreateExpr(2), CreateExpr(1), CreateExpr(2), CreateExpr(1), CreateExpr("last")},
                               {CreateExpr(2), CreateExpr(3), CreateExpr(2), CreateExpr(3), CreateExpr("last")});
  node.input_dtype = {kUInt8};
  node.output_dtype = {kUInt8};
  EXPECT_EQ(ascendcperf_v2::GetNlastAxisPerfBranch(node), NlastAxisBranch::kDynamicLessThanVlUnaligned);
  EXPECT_FALSE(ascendcperf_v2::IsBroadcastFallback(node));
}

}  // namespace
}  // namespace att
