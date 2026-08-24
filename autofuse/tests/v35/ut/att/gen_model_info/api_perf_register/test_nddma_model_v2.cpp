/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <limits>
#include <numeric>
#include "gtest/gtest.h"
#include "tests/depends/slog/src/slog_stub.h"
#include "common_utils.h"
#include "v35/att/api_perf_register/nddma_model.h"

namespace att {
namespace {
NddmaDescriptorInfo MakeDescriptor(const Expr &dim, const Expr &input_stride, const Expr &output_stride) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {dim};
  descriptor.input_strides = {input_stride};
  descriptor.output_strides = {output_stride};
  descriptor.vectorized_axis = {0};
  return descriptor;
}

double GetConstCycles(Expr cycles) {
  cycles.Simplify();
  double value = 0.0;
  EXPECT_TRUE(cycles.GetConstValue(value));
  return value;
}
}  // namespace

TEST(NddmaModelV2, EvaluatesStaticB8LowCoreFormula) {
  const auto descriptor = MakeDescriptor(CreateExpr(256), CreateExpr(1), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "uint8", CreateExpr(2), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.model_name, "NDDMA_1D_MULTICORE_V2");
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kNone);
  EXPECT_TRUE(result.ternary_ops.empty());
  EXPECT_NEAR(GetConstCycles(result.cycles), 219.37435914613698, 1e-6);
}

TEST(NddmaModelV2, UsesMergedLowCoreCoefficientsForStridedOutput) {
  const auto descriptor = MakeDescriptor(CreateExpr("B"), CreateExpr("s"), CreateExpr(2));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "uint8", CreateExpr(2), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_TRUE(result.cycles.IsValid());
}

TEST(NddmaModelV2, AllowsZeroInputStride) {
  const auto descriptor = MakeDescriptor(CreateExpr(256), CreateExpr(0), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "uint8", CreateExpr(2), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_NEAR(GetConstCycles(result.cycles), 216.18489573733697, 1e-6);
}

TEST(NddmaModelV2, EvaluatesStaticB16HighCoreFormula) {
  const auto descriptor = MakeDescriptor(CreateExpr(256), CreateExpr(4), CreateExpr(2));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(8), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_NEAR(GetConstCycles(result.cycles), 3667.5976503571769, 1e-6);
}

TEST(NddmaModelV2, EvaluatesStaticB32LowCoreFormula) {
  const auto descriptor = MakeDescriptor(CreateExpr(256), CreateExpr(4), CreateExpr(2));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(2), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_NEAR(GetConstCycles(result.cycles), 415.93874765450965, 1e-6);
}

TEST(NddmaModelV2, EvaluatesStaticB64HighCoreWithSaturatedInputStride) {
  const auto descriptor = MakeDescriptor(CreateExpr(256), CreateExpr(129), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int64", CreateExpr(8), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_NEAR(GetConstCycles(result.cycles), 1490.6546508742531, 1e-6);
}

TEST(NddmaModelV2, ReplaysAllCoefficientGroupsForFourDtypeSizes) {
  struct DtypeCase {
    const char *dtype;
    double low_core;
    double high_core;
  };
  const DtypeCase cases[] = {{"int8", 1846.461366862137, 3651.7082163688292},
                             {"float16", 1852.9241082038752, 3667.5976503571769},
                             {"float32", 415.93874765450965, 801.7379345551185},
                             {"int64", 436.1371738143462, 834.9812022377600}};
  const auto descriptor = MakeDescriptor(CreateExpr(256), CreateExpr(4), CreateExpr(2));
  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.dtype);
    NddmaModelResult low_result;
    NddmaModelResult high_result;
    ASSERT_EQ(EvaluateNddmaModel(descriptor, test_case.dtype, CreateExpr(2), low_result), af::SUCCESS);
    ASSERT_EQ(EvaluateNddmaModel(descriptor, test_case.dtype, CreateExpr(8), high_result), af::SUCCESS);
    EXPECT_NEAR(GetConstCycles(low_result.cycles), test_case.low_core, 1e-5);
    EXPECT_NEAR(GetConstCycles(high_result.cycles), test_case.high_core, 1e-5);
  }
}

TEST(NddmaModelV2, BuildsDynamicCoreTernary) {
  const auto descriptor = MakeDescriptor(CreateExpr("n"), CreateExpr("input_stride"), CreateExpr("output_stride"));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr("block_dim"), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  ASSERT_EQ(result.ternary_ops.size(), 1U);
  EXPECT_NE(result.ternary_ops.begin()->second.GetTernaryOpStr().find("block_dim"), std::string::npos);
  EXPECT_NE(Str(result.cycles).find("nddma_1d_multicore"), std::string::npos);
}

TEST(NddmaModelV2, RejectsMismatchedDescriptorSchema) {
  auto descriptor = MakeDescriptor(CreateExpr(16), CreateExpr(1), CreateExpr(1));
  descriptor.output_strides.clear();
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int32", CreateExpr(1), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kSchemaMismatch);
  EXPECT_STREQ(NddmaFallbackReasonToString(result.fallback_reason), "schema_mismatch");
}

TEST(NddmaModelV2, EvaluatesRegisteredRanks2To5) {
  for (size_t rank = 2U; rank <= 5U; ++rank) {
    SCOPED_TRACE(rank);
    NddmaDescriptorInfo descriptor;
    descriptor.output_dims.assign(rank, CreateExpr(8));
    descriptor.input_strides.assign(rank, CreateExpr(1));
    descriptor.output_strides.assign(rank, CreateExpr(1));
    for (size_t axis = 0U; axis < rank; ++axis) {
      descriptor.vectorized_axis.emplace_back(static_cast<int64_t>(axis));
    }
    NddmaModelResult result;

    ASSERT_EQ(EvaluateNddmaModel(descriptor, "int64", CreateExpr(1), result), af::SUCCESS);
    EXPECT_TRUE(result.selected);
    EXPECT_EQ(result.raw_rank, 0U);
    EXPECT_EQ(result.effective_rank, rank);
    EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kNone);
    EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
  }
}

TEST(NddmaModelV2, UsesInnerToOuterFormulaOrderForTwoDimensions) {
  NddmaDescriptorInfo descriptor;
  // Descriptor is effective outer -> inner; the formula must consume [inner, outer].
  descriptor.output_dims = {CreateExpr(3), CreateExpr(8)};
  descriptor.input_strides = {CreateExpr(8), CreateExpr(1)};
  descriptor.output_strides = {CreateExpr(8), CreateExpr(1)};
  descriptor.vectorized_axis = {1, 0};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(2), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.effective_rank, 2U);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
  EXPECT_TRUE(result.cycles.IsValid());
}

TEST(NddmaModelV2, RejectsNonPositiveStaticOutputStride) {
  const auto descriptor = MakeDescriptor(CreateExpr(16), CreateExpr(1), CreateExpr(0));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int8", CreateExpr(1), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kStrideInvalid);
}

TEST(NddmaModelV2, RejectsRankOutsideSupportedRange) {
  for (const size_t rank : {0U, 6U, 7U}) {
    SCOPED_TRACE(rank);
    NddmaDescriptorInfo descriptor;
    descriptor.output_dims.assign(rank, CreateExpr(8));
    descriptor.input_strides.assign(rank, CreateExpr(1));
    descriptor.output_strides.assign(rank, CreateExpr(1));
    descriptor.vectorized_axis.resize(rank);
    std::iota(descriptor.vectorized_axis.begin(), descriptor.vectorized_axis.end(), 0);
    NddmaModelResult result;

    ASSERT_EQ(EvaluateNddmaModel(descriptor, "int8", CreateExpr(1), result), af::SUCCESS);
    EXPECT_FALSE(result.selected);
    EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kRankUnsupported);
  }
}

TEST(NddmaModelV2, SharedDataCopyAxisRulesMatchCodegenConditions) {
  const std::vector<Expr> ub_strides = {CreateExpr(15), CreateExpr(5), CreateExpr(1)};
  EXPECT_TRUE(ascgen_utils::ShouldIgnoreDataCopyZeroAxis(false, 0U, ub_strides));
  EXPECT_TRUE(ascgen_utils::ShouldIgnoreDataCopyZeroAxis(true, 2U, ub_strides));
  EXPECT_FALSE(ascgen_utils::ShouldIgnoreDataCopyZeroAxis(false, 2U, {CreateExpr(15), CreateExpr(5), CreateExpr(3)}));

  EXPECT_TRUE(ascgen_utils::IsDataCopyAxisContinuous(CreateExpr(5), CreateExpr(1), CreateExpr(5), CreateExpr(1)));
  EXPECT_FALSE(ascgen_utils::IsDataCopyAxisContinuous(CreateExpr(4), CreateExpr(1), CreateExpr(5), CreateExpr(1)));
}

TEST(NddmaModelV2, RejectsUnsupportedDtype) {
  const auto descriptor = MakeDescriptor(CreateExpr(16), CreateExpr(1), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "unsupported_dtype", CreateExpr(1), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kDtypeUnsupported);
}

TEST(NddmaModelV2, RejectsZeroDtypeSizeFromUnsupportedMapping) {
  const auto descriptor = MakeDescriptor(CreateExpr(16), CreateExpr(1), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "", CreateExpr(1), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kDtypeUnsupported);
}

TEST(NddmaModelV2, RejectsNegativeStaticInputStride) {
  const auto descriptor = MakeDescriptor(CreateExpr(16), CreateExpr(-1), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int8", CreateExpr(1), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kStrideInvalid);
}

TEST(NddmaModelV2, RejectsNonPositiveStaticDimension) {
  const auto descriptor = MakeDescriptor(CreateExpr(0), CreateExpr(1), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int8", CreateExpr(1), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kStrideInvalid);
}

TEST(NddmaModelV2, RejectsNonPositiveStaticBlockDim) {
  const auto descriptor = MakeDescriptor(CreateExpr(16), CreateExpr(1), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int8", CreateExpr(0), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kSchemaMismatch);
}

TEST(NddmaModelV2, RejectsStaticByteCountOverflow) {
  const auto descriptor = MakeDescriptor(CreateExpr(std::numeric_limits<int64_t>::max()), CreateExpr(1), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int64", CreateExpr(1), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kSchemaMismatch);
}

TEST(NddmaModelV2, RejectsRepeatedVectorizedAxis) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(4), CreateExpr(8)};
  descriptor.input_strides = {CreateExpr(8), CreateExpr(1)};
  descriptor.output_strides = {CreateExpr(8), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 0};
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int8", CreateExpr(1), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kCodegenMismatch);
}

TEST(NddmaModelV2, BuildsDescriptorFromRawTensorShape) {
  TensorShapeInfo shape_info;
  shape_info.repeats = {CreateExpr(4), CreateExpr(64)};
  shape_info.gm_strides = {CreateExpr(128), CreateExpr(2)};
  shape_info.strides = {CreateExpr(64), CreateExpr(1)};
  NddmaDescriptorInfo descriptor;

  EXPECT_EQ(BuildNddmaDescriptor(shape_info, {3, 1}, descriptor), NddmaFallbackReason::kNone);
  EXPECT_EQ(descriptor.output_dims, (std::vector<Expr>{CreateExpr(256)}));
  EXPECT_EQ(descriptor.input_strides, (std::vector<Expr>{CreateExpr(2)}));
  EXPECT_EQ(descriptor.output_strides, (std::vector<Expr>{CreateExpr(1)}));
  EXPECT_EQ(descriptor.vectorized_axis, (std::vector<int64_t>{1}));
}

TEST(NddmaModelV2, RejectsMissingVectorizedAxisForCodegenParity) {
  TensorShapeInfo shape_info;
  shape_info.repeats = {CreateExpr(64)};
  shape_info.gm_strides = {CreateExpr(2)};
  shape_info.strides = {CreateExpr(1)};
  NddmaDescriptorInfo descriptor;

  EXPECT_EQ(BuildNddmaDescriptor(shape_info, {}, descriptor), NddmaFallbackReason::kCodegenMismatch);
}

TEST(NddmaModelV2, LogsOneStableFallbackSummary) {
  const auto descriptor = MakeDescriptor(CreateExpr(16), CreateExpr(1), CreateExpr(1));
  NddmaModelResult result;
  result.raw_rank = 2U;
  result.effective_rank = 2U;
  result.fallback_reason = NddmaFallbackReason::kNoRegisteredModel;
  auto *slog = ge::SlogStub::GetInstance();
  const int32_t old_level = slog->GetLevel();
  slog->SetLevel(DLOG_WARN);

  testing::internal::CaptureStdout();
  LogNddmaFallback("nddma", "float16", &descriptor, result);
  const std::string output = testing::internal::GetCapturedStdout();
  slog->SetLevel(old_level);

  const auto summary_pos = output.find("[ATT NDDMA] fallback:");
  ASSERT_NE(summary_pos, std::string::npos);
  EXPECT_EQ(output.find("[ATT NDDMA] fallback:", summary_pos + 1U), std::string::npos);
  EXPECT_NE(output.find("fallback_reason=no_registered_model"), std::string::npos);
  EXPECT_EQ(output.find("fallback detail"), std::string::npos);
}
}  // namespace att
