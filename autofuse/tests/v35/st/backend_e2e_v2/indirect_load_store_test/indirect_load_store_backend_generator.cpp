/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <exception>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>

#include "backend_common.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen.h"
#include "common/platform_context.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "indirect_load_utils.h"
#include "optimize.h"
#include "runtime_stub.h"
#include "share_graph.h"

namespace {
constexpr size_t kRank = IL_RANK;
constexpr int64_t kAxis = IL_AXIS;
constexpr auto kInputPreType = static_cast<ascir::IndirectLoadInputPreType>(IL_INPUT_PRE_TYPE);
constexpr bool kHasInputPre = kInputPreType != ascir::IndirectLoadInputPreType::kNone;
constexpr bool kUseExp2 = IL_USE_EXP2;
#ifdef IL_OUTPUT_POST_TYPE
constexpr auto kOutputPostType = static_cast<ascir::IndirectLoadOutputPostType>(IL_OUTPUT_POST_TYPE);
#else
constexpr auto kOutputPostType = ascir::IndirectLoadOutputPostType::kDefault;
#endif
#ifdef IL_EXPECT_SIMT
constexpr bool kExpectSimt = IL_EXPECT_SIMT;
#else
constexpr bool kExpectSimt = true;
#endif
#if defined(IL_DATA_BF16)
constexpr af::DataType kDataType = af::DT_BF16;
#elif defined(IL_DATA_UINT32)
constexpr af::DataType kDataType = af::DT_UINT32;
#elif defined(IL_DATA_FLOAT)
constexpr af::DataType kDataType = af::DT_FLOAT;
#else
constexpr af::DataType kDataType = af::DT_FLOAT16;
#endif
#ifdef IL_INDEX_INT64
constexpr af::DataType kIndexType = af::DT_INT64;
#else
constexpr af::DataType kIndexType = af::DT_INT32;
#endif
#ifdef IL_STATIC_SHAPE
constexpr bool kStaticShape = true;
#else
constexpr bool kStaticShape = false;
#endif
constexpr int64_t kInputNumel = IL_X_S0 * IL_X_S1 * IL_X_S2 * IL_X_S3;
constexpr int64_t kOutputNumel = IL_INDEX_S0 * IL_INDEX_S1 * IL_INDEX_S2 * IL_INDEX_S3;
constexpr bool kExpectInputPreMoved = kHasInputPre && (!kStaticShape || kOutputNumel <= kInputNumel);
#ifdef IL_MIXED_INDEX_PRE
constexpr bool kMixedIndexPre = true;
#else
constexpr bool kMixedIndexPre = false;
#endif

std::vector<int64_t> GetStaticShape(bool input) {
  if (!kStaticShape) {
    return {};
  }
  std::vector<int64_t> shape = input ? std::vector<int64_t>{IL_X_S0, IL_X_S1, IL_X_S2, IL_X_S3}
                                     : std::vector<int64_t>{IL_INDEX_S0, IL_INDEX_S1, IL_INDEX_S2, IL_INDEX_S3};
  shape.resize(kRank);
  return shape;
}

const af::Axis *FindDerivedAxis(const af::AscGraph &graph, af::Axis::Type type, af::AxisId from) {
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis != nullptr && axis->type == type && axis->from == std::vector<af::AxisId>{from}) {
      return axis.get();
    }
  }
  return nullptr;
}

void CollectOriginAxes(af::AscGraph &graph, af::AxisId axis_id, std::vector<af::AxisId> &origins) {
  const auto *axis = graph.FindAxis(axis_id);
  ASSERT_NE(axis, nullptr);
  if (axis->from.empty()) {
    origins.emplace_back(axis_id);
    return;
  }
  for (af::AxisId from : axis->from) {
    CollectOriginAxes(graph, from, origins);
  }
}

void ExpectAxisOrigins(af::AscGraph &graph, af::AxisId axis_id, const std::vector<af::AxisId> &expected) {
  std::vector<af::AxisId> origins;
  CollectOriginAxes(graph, axis_id, origins);
  EXPECT_EQ(origins, expected);
}

void ExpectMixedInputPre(const af::AscNodePtr &indirect_load) {
  if (!kMixedIndexPre) {
    return;
  }
  const af::AscNodePtr data_post = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  ASSERT_NE(data_post, nullptr);
  EXPECT_EQ(data_post->GetName(), "input_exp2");
  const af::AscNodePtr data_vf = ascgen_utils::indirect_load::GetInputProducer(data_post, 0UL);
  ASSERT_NE(data_vf, nullptr);
  EXPECT_TRUE(af::ops::IsOps<af::ascir_op::VectorFunc>(data_vf));

  const af::AscNodePtr index_post = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 1UL);
  ASSERT_NE(index_post, nullptr);
  EXPECT_EQ(index_post->GetName(), "index_floor_to_int");
  const af::AscNodePtr index_log2 = ascgen_utils::indirect_load::GetInputProducer(index_post, 0UL);
  ASSERT_NE(index_log2, nullptr);
  EXPECT_EQ(index_log2->GetName(), "index_log2");
  const af::AscNodePtr index_exp2 = ascgen_utils::indirect_load::GetInputProducer(index_log2, 0UL);
  ASSERT_NE(index_exp2, nullptr);
  EXPECT_EQ(index_exp2->GetName(), "index_exp2");
  const af::AscNodePtr index_vf = ascgen_utils::indirect_load::GetInputProducer(index_exp2, 0UL);
  ASSERT_NE(index_vf, nullptr);
  EXPECT_TRUE(af::ops::IsOps<af::ascir_op::VectorFunc>(index_vf));
}

void ExpectLoopFramework(af::AscGraph &graph, const af::AscNodePtr &indirect_load) {
  ascgen_utils::indirect_load::TemplateAxes axes;
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);

  const auto *outer = graph.FindAxis(axes.outer_axis);
  ASSERT_NE(outer, nullptr);
  const auto *tile_outer = FindDerivedAxis(graph, af::Axis::Type::kAxisTypeTileOuter, outer->id);
  const auto *tile_inner = FindDerivedAxis(graph, af::Axis::Type::kAxisTypeTileInner, outer->id);
  ASSERT_NE(tile_outer, nullptr);
  ASSERT_NE(tile_inner, nullptr);
  EXPECT_EQ(tile_outer->split_pair_other_id, tile_inner->id);
  EXPECT_EQ(tile_inner->split_pair_other_id, tile_outer->id);

  const auto *block_outer = FindDerivedAxis(graph, af::Axis::Type::kAxisTypeBlockOuter, tile_outer->id);
  const auto *block_inner = FindDerivedAxis(graph, af::Axis::Type::kAxisTypeBlockInner, tile_outer->id);
  ASSERT_NE(block_outer, nullptr);
  ASSERT_NE(block_inner, nullptr);
  EXPECT_EQ(block_outer->split_pair_other_id, block_inner->id);
  EXPECT_EQ(block_inner->split_pair_other_id, block_outer->id);

  const int64_t normalized_axis = kAxis < 0L ? kAxis + static_cast<int64_t>(kRank) : kAxis;
  const auto template_id = ascir::GetTemplateIdOrDefault(*indirect_load);
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    ExpectAxisOrigins(graph, axes.outer_axis, logical_view.output.axis_ids);
    EXPECT_EQ(axes.inner_axis, af::kIdNone);
    return;
  }

  ASSERT_EQ(template_id, ascir::TemplateId::kIndirectLoadSimd);
  const size_t split = static_cast<size_t>(normalized_axis);
  if (split == 0UL) {
    EXPECT_TRUE(outer->from.empty());
    EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(outer->size, af::ops::One), af::TriBool::kTrue);
  } else {
    ExpectAxisOrigins(
        graph, axes.outer_axis,
        std::vector<af::AxisId>(logical_view.output.axis_ids.begin(), logical_view.output.axis_ids.begin() + split));
  }
  ExpectAxisOrigins(
      graph, axes.inner_axis,
      std::vector<af::AxisId>(logical_view.output.axis_ids.begin() + split, logical_view.output.axis_ids.end()));
  if (kHasInputPre) {
    ExpectAxisOrigins(
        graph, axes.input_inner_axis,
        std::vector<af::AxisId>(logical_view.data.axis_ids.begin() + split, logical_view.data.axis_ids.end()));
  }

  const auto input_producer = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  ASSERT_NE(input_producer, nullptr);
  if (kExpectInputPreMoved) {
    EXPECT_EQ(input_producer->GetName(), "input_load");
  } else if (kHasInputPre) {
#ifdef IL_EXPECT_INPUT_PRE_VF
    EXPECT_EQ(af::ops::IsOps<af::ascir_op::VectorFunc>(input_producer), IL_EXPECT_INPUT_PRE_VF);
#endif
    if (af::ops::IsOps<af::ascir_op::VectorFunc>(input_producer)) {
      EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(input_producer),
                ascgen_utils::indirect_load::TemplateRole::kSimdInputPre);
    } else {
      EXPECT_EQ(input_producer->GetName(),
                kInputPreType == ascir::IndirectLoadInputPreType::kRelu ? "input_relu" : "input_exp2");
    }
  }
  const auto index_producer = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 1UL);
  ASSERT_NE(index_producer, nullptr);
  EXPECT_EQ(index_producer->GetName(), kMixedIndexPre ? "index_floor_to_int" : "index_abs");
  ExpectMixedInputPre(indirect_load);
#if IL_INPUT_PRE_TYPE == 5
  EXPECT_EQ(indirect_load->outputs()[0]->attr.dtype, af::DT_FLOAT16);
#endif
}

bool CheckScheduledLoopFramework(ascir::FusedScheduledResult &result) {
  size_t simd_count = 0UL;
  size_t simt_count = 0UL;
  for (auto &candidates : result.node_idx_to_scheduled_results) {
    for (auto &candidate : candidates) {
      for (auto &group : candidate.schedule_groups) {
        for (auto &graph : group.impl_graphs) {
          const af::AscNodePtr indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(graph);
          if (indirect_load == nullptr) {
            continue;
          }
          ExpectLoopFramework(graph, indirect_load);
          if (ascir::GetTemplateIdOrDefault(*indirect_load) == ascir::TemplateId::kIndirectLoadSimd) {
            ++simd_count;
          } else {
            ++simt_count;
          }
        }
      }
    }
  }
  EXPECT_GT(simd_count, 0UL);
  if (kExpectSimt) {
    EXPECT_GT(simt_count, 0UL);
    return simd_count > 0UL && simt_count > 0UL;
  } else {
    EXPECT_EQ(simt_count, 0UL);
    return simd_count > 0UL && simt_count == 0UL;
  }
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find("// IndirectLoad SIMD"), std::string::npos);
  EXPECT_NE(kernel.find("IndirectLoadSimd<"), std::string::npos);
  EXPECT_EQ(kernel.find("auto indirect_load_offset"), std::string::npos);
  if (kExpectSimt) {
    EXPECT_NE(kernel.find("// IndirectLoad SIMT"), std::string::npos);
    EXPECT_NE(kernel.find("IndirectLoadSimt<"), std::string::npos);
    EXPECT_EQ(kernel.find("IndirectLoadSimtKernel_"), std::string::npos);
  } else {
    EXPECT_EQ(kernel.find("// IndirectLoad SIMT"), std::string::npos);
    EXPECT_EQ(kernel.find("IndirectLoadSimt<"), std::string::npos);
  }
#ifdef IL_DATA_BF16
  EXPECT_NE(kernel.find("IndirectLoadSimt<bfloat16_t, int64_t, bfloat16_t"), std::string::npos);
#endif
#ifdef IL_DATA_UINT32
  EXPECT_NE(kernel.find("IndirectLoadSimd<uint32_t, int32_t"), std::string::npos);
#endif
}
}  // namespace

class TestBackendIndirectLoadStoreE2e : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    ge::RuntimeStub::SetInstance(std::make_shared<af::RuntimeStubV2>());
  }

  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
  }
};

TEST_F(TestBackendIndirectLoadStoreE2e, IndirectLoadStoreCodegen) {
  const std::string tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  auto graph = ascir::ShareGraph::IndirectLoadStoreFusedGraph(kRank, kAxis, kDataType, kIndexType, kInputPreType,
                                                              kUseExp2, kOutputPostType, GetStaticShape(true),
                                                              GetStaticShape(false), kMixedIndexPre);
  ASSERT_NE(graph, nullptr);
  std::map<std::string, std::string> shape_info;
  for (size_t i = 0UL; i < 2UL * kRank; ++i) {
    shape_info.emplace("s" + std::to_string(i), "stub_s" + std::to_string(i));
  }

  const std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  ASSERT_EQ(parts.size(), 3U);
  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{.graph_type = optimize::GraphType::kFusedAscBackend});
    codegen::Codegen codegen(codegen::CodegenOptions{});
    ascir::FusedScheduledResult fused_schedule_result;
    codegen::CodegenResult result;
    testing::internal::CaptureStdout();
    const auto optimize_status = optimizer.Optimize(graph, fused_schedule_result);
    const std::string optimize_logs = testing::internal::GetCapturedStdout();
    EXPECT_EQ(optimize_logs.find("[ERROR]"), std::string::npos) << optimize_logs;
    ASSERT_EQ(optimize_status, 0) << optimize_logs;
    ASSERT_TRUE(CheckScheduledLoopFramework(fused_schedule_result)) << optimize_logs;

    testing::internal::CaptureStdout();
    const auto codegen_status = codegen.Generate(shape_info, fused_schedule_result, result);
    const std::string codegen_logs = testing::internal::GetCapturedStdout();
    EXPECT_EQ(codegen_logs.find("[ERROR]"), std::string::npos) << codegen_logs;
    ASSERT_EQ(codegen_status, 0) << codegen_logs;
    CheckGeneratedKernel(result.kernel);

    std::fstream kernel_file(parts[0], std::ios::out);
    std::fstream tiling_file(parts[1], std::ios::out);
    std::fstream tiling_data_file(parts[2], std::ios::out);
    ASSERT_TRUE(kernel_file.is_open());
    ASSERT_TRUE(tiling_file.is_open());
    ASSERT_TRUE(tiling_data_file.is_open());
    kernel_file << tiling_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
    EXPECT_TRUE(kernel_file.good());
    EXPECT_TRUE(tiling_file.good());
    EXPECT_TRUE(tiling_data_file.good());
  } catch (const std::exception &e) {
    FAIL() << e.what();
  } catch (...) {
    FAIL() << "Unknown exception";
  }
}
