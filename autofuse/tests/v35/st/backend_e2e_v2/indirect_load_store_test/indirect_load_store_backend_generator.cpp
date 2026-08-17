/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <fstream>
#include <gtest/gtest.h>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

#include "backend_common.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen.h"
#include "common/platform_context.h"
#include "fusion/autofuse_attrs.h"
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
constexpr auto kSelectedTemplate = static_cast<ascir::TemplateId>(IL_SELECTED_TEMPLATE);
constexpr uint32_t kExpectedTemplates = IL_EXPECTED_TEMPLATES;
constexpr std::array<ascir::TemplateId, 3UL> kIndirectLoadTemplates = {
    ascir::TemplateId::kIndirectLoadSimd, ascir::TemplateId::kIndirectLoadSimt, ascir::TemplateId::kIndirectLoadSK};
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
#elif defined(IL_DATA_INT16)
constexpr af::DataType kDataType = af::DT_INT16;
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
#ifdef IL_DIRECT_INDEX
constexpr bool kDirectIndex = true;
#else
constexpr bool kDirectIndex = false;
#endif
#ifdef IL_SELECTED_IMPLEMENTATION
constexpr auto kSelectedImplementation =
    static_cast<ascgen_utils::indirect_load::Implementation>(IL_SELECTED_IMPLEMENTATION);
#else
// The default IndirectLoad SIMD implementation uses MicroAPI.
constexpr auto kSelectedImplementation = ascgen_utils::indirect_load::Implementation::kDefault;
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

bool IsMixedGatherShapeValid() {
  const std::vector<int64_t> input_shape = {IL_X_S0, IL_X_S1, IL_X_S2, IL_X_S3};
  const std::vector<int64_t> output_shape = {IL_INDEX_S0, IL_INDEX_S1, IL_INDEX_S2, IL_INDEX_S3};
  if (std::equal(input_shape.begin(), input_shape.begin() + static_cast<int64_t>(kRank), output_shape.begin())) {
    return false;
  }
  const int64_t axis = kAxis < 0L ? kAxis + static_cast<int64_t>(kRank) : kAxis;
  for (size_t i = 0UL; i < kRank; ++i) {
    if (static_cast<int64_t>(i) != axis && input_shape[i] < output_shape[i]) {
      return false;
    }
  }
  return true;
}

bool ContainsInOrder(const std::string &text, std::initializer_list<const char *> tokens) {
  size_t position = 0UL;
  for (const char *token : tokens) {
    position = text.find(token, position);
    if (position == std::string::npos) {
      return false;
    }
    position += std::char_traits<char>::length(token);
  }
  return true;
}

std::string GetFunctionContaining(const std::string &kernel, const char *marker) {
  const size_t marker_pos = kernel.find(marker);
  EXPECT_NE(marker_pos, std::string::npos) << marker;
  if (marker_pos == std::string::npos) {
    return {};
  }
  size_t begin = kernel.rfind("inline __aicore__", marker_pos);
  const size_t alternate_begin = kernel.rfind("__aicore__ inline", marker_pos);
  if (begin == std::string::npos || (alternate_begin != std::string::npos && alternate_begin > begin)) {
    begin = alternate_begin;
  }
  const size_t simd_begin = kernel.rfind("__simd_callee__ inline", marker_pos);
  if (begin == std::string::npos || (simd_begin != std::string::npos && simd_begin > begin)) {
    begin = simd_begin;
  }
  const size_t simt_begin = kernel.rfind("__simt_vf__ __aicore__", marker_pos);
  if (begin == std::string::npos || (simt_begin != std::string::npos && simt_begin > begin)) {
    begin = simt_begin;
  }
  EXPECT_NE(begin, std::string::npos) << marker;
  if (begin == std::string::npos) {
    return {};
  }
  const size_t body_begin = kernel.find('{', begin);
  EXPECT_NE(body_begin, std::string::npos) << marker;
  size_t depth = 0UL;
  for (size_t pos = body_begin; pos < kernel.size(); ++pos) {
    if (kernel[pos] == '{') {
      ++depth;
    } else if (kernel[pos] == '}' && --depth == 0UL) {
      return kernel.substr(begin, pos - begin + 1UL);
    }
  }
  ADD_FAILURE() << "Unclosed function containing " << marker;
  return {};
}

std::vector<std::string> GetCallArguments(const std::string &function, const char *api) {
  const size_t api_pos = function.find(api);
  const size_t begin = function.find(">(", api_pos);
  EXPECT_NE(api_pos, std::string::npos) << api;
  EXPECT_NE(begin, std::string::npos) << api;
  if (api_pos == std::string::npos || begin == std::string::npos) {
    return {};
  }
  std::vector<std::string> arguments;
  size_t argument_begin = begin + 2UL;
  int32_t depth = 0;
  for (size_t i = argument_begin; i < function.size(); ++i) {
    if (function[i] == '(') {
      ++depth;
    } else if (function[i] == ')' && depth > 0) {
      --depth;
    } else if ((function[i] == ',' || function[i] == ')') && depth == 0) {
      const size_t first = function.find_first_not_of(" \n\t", argument_begin);
      const size_t last = function.find_last_not_of(" \n\t", i - 1UL);
      arguments.emplace_back(function.substr(first, last - first + 1UL));
      if (function[i] == ')') {
        break;
      }
      argument_begin = i + 1UL;
    }
  }
  return arguments;
}

void ExpectBlockSplitFramework(const std::string &function, const char *outer_axes) {
  EXPECT_NE(function.find(outer_axes), std::string::npos);
  EXPECT_TRUE(ContainsInOrder(function, {"indirect_load_outerTb_axis_size = t->indirect_load_outerTb_size",
                                         "indirect_load_outerTB_axis_size = indirect_load_outerT_loop_size / "
                                         "indirect_load_outerTb_axis_size",
                                         "GetBlockIdx()", "indirect_load_outerTB = block_dim %",
                                         "block_dim_offset = indirect_load_outerTB * t->indirect_load_outerTb_size"}));
}

void ExpectSimdFramework(const std::string &kernel) {
  const std::string function = GetFunctionContaining(kernel, "// IndirectLoad SIMD");
  ExpectBlockSplitFramework(function, "indirect_load_outer_axis_size = z4_loop_size * z5_loop_size * 1");
  EXPECT_NE(function.find("indirect_load_outert_axis_size = 1"), std::string::npos);
  EXPECT_TRUE(
      ContainsInOrder(function, {"for (int indirect_load_outerTb", "for (int indirect_load_outert", "CopySignExtend(",
                                 "// IndirectLoad SIMD", "IndirectLoadSimd<", "CopySignExtend("}));
  EXPECT_TRUE(ContainsInOrder(function, {"for (int indirect_load_outerTb", "for (int indirect_load_outert", "VfNode_0",
                                         "// IndirectLoad SIMD"}));
  const std::vector<std::string> arguments = GetCallArguments(function, "IndirectLoadSimd<");
  ASSERT_GT(arguments.size(), 4UL);
  const std::string &actual_size = arguments[3UL];
  EXPECT_NE(
      function.find("const uint32_t " + actual_size + " = (z6_actual_size - 1) * t->s7 + (z7_actual_size - 1) + 1"),
      std::string::npos);
  EXPECT_TRUE(ContainsInOrder(arguments[4UL], {"indirect_load_outerTB", "t->s6 * t->s7", "indirect_load_outerTb",
                                               "t->s6 * t->s7", "indirect_load_outert", "t->s6 * t->s7"}));
}

void ExpectPostReduceSimtFramework(const std::string &kernel) {
  EXPECT_NE(kernel.find("__ubuf__ Y *y"), std::string::npos);
  const std::string ub_kernel = GetFunctionContaining(kernel, "inline void IndirectLoadSimtUbKernel(");
  EXPECT_EQ(ub_kernel.find("y[i] = static_cast<Y>(0)"), std::string::npos);
  EXPECT_NE(ub_kernel.find("y[i] = IndirectLoadSimtCompute"), std::string::npos);
  EXPECT_NE(kernel.find("(__ubuf__ Y *)y.GetPhyAddr()"), std::string::npos);
  const std::string function = GetFunctionContaining(kernel, "// IndirectLoad SIMT");
  ExpectBlockSplitFramework(function, "indirect_load_outer_axis_size = z4_loop_size * z5_loop_size * 1");
  EXPECT_NE(function.find("indirect_load_inner_axis_size = z6_loop_size * z7_loop_size * 1"), std::string::npos);
  const std::string vector_size = std::to_string(IL_INDEX_S2 * IL_INDEX_S3);
  const std::string actual_size = "static_cast<uint32_t>(" + vector_size + ")";
  const std::string offset_scale = "* " + vector_size;
  const std::vector<std::string> arguments = GetCallArguments(function, "IndirectLoadSimt<");
  ASSERT_GT(arguments.size(), 5UL);
  EXPECT_EQ(arguments[3UL], actual_size);
  EXPECT_TRUE(ContainsInOrder(arguments[4UL], {"block_dim_offset", "indirect_load_outerTb", offset_scale.c_str()}));
  EXPECT_TRUE(ContainsInOrder(
      function, {"for (int indirect_load_outerTb", "for (int indirect_load_outert", "// IndirectLoad SIMT",
                 "IndirectLoadSimt<", actual_size.c_str(), "PipeBarrier<PIPE_V>", "ReduceSum", "DataCopyPadExtend"}));
}

void ExpectNoReduceSimtFramework(const std::string &kernel) {
  const std::string function = GetFunctionContaining(kernel, "// IndirectLoad SIMT");
  ExpectBlockSplitFramework(
      function, "indirect_load_outer_axis_size = z4_loop_size * z5_loop_size * z6_loop_size * z7_loop_size * 1");
  const std::vector<std::string> arguments = GetCallArguments(function, "IndirectLoadSimt<");
  ASSERT_GT(arguments.size(), 5UL);
  EXPECT_EQ(arguments[3UL], "static_cast<uint32_t>(indirect_load_outerTb_loop_size)");
  EXPECT_NE(arguments[4UL].find("block_dim_offset"), std::string::npos);
  EXPECT_TRUE(
      ContainsInOrder(function, {"block_dim_offset", "// IndirectLoad SIMT", "IndirectLoadSimt<",
                                 "static_cast<uint32_t>(indirect_load_outerTb_loop_size)", "block_dim_offset"}));
  EXPECT_EQ(function.find("for (int indirect_load_outerTb"), std::string::npos);
  EXPECT_EQ(function.find("ReduceSum"), std::string::npos);
}

void ExpectSimtKernelStructure(const std::string &kernel) {
  const std::string compute = GetFunctionContaining(kernel, "inline Y IndirectLoadSimtCompute(");
  const std::string gm_kernel = GetFunctionContaining(kernel, "inline void IndirectLoadSimtKernel(");
  const std::string ub_kernel = GetFunctionContaining(kernel, "inline void IndirectLoadSimtUbKernel(");
  EXPECT_EQ(gm_kernel.find("indirect_index < 0"), std::string::npos);
  EXPECT_EQ(gm_kernel.find("indirect_index >="), std::string::npos);
  EXPECT_EQ(gm_kernel.find("static_cast<Y>(0)"), std::string::npos);
  EXPECT_EQ(ub_kernel.find("indirect_index < 0"), std::string::npos);
  EXPECT_EQ(ub_kernel.find("indirect_index >="), std::string::npos);
  EXPECT_EQ(ub_kernel.find("static_cast<Y>(0)"), std::string::npos);
  EXPECT_NE(compute.find("return FusedBody::Output(x[input_offset], output_index, context)"), std::string::npos);
  EXPECT_NE(gm_kernel.find("y[output_index] = IndirectLoadSimtCompute"), std::string::npos);
  EXPECT_NE(ub_kernel.find("y[i] = IndirectLoadSimtCompute"), std::string::npos);
  EXPECT_TRUE(ContainsInOrder(kernel, {"LaunchIndirectLoadSimt<128U", "LaunchIndirectLoadSimt<256U",
                                       "LaunchIndirectLoadSimt<512U", "LaunchIndirectLoadSimt<1024U"}));
  EXPECT_NE(kernel.find("constexpr bool IndirectLoadUse2048Threads()"), std::string::npos);
  EXPECT_TRUE(ContainsInOrder(kernel, {"IndirectLoadUse2048Threads<AddressPolicy>()", "LaunchIndirectLoadSimt<2048U"}));
}

#ifdef IL_EXPECT_SIMT_POLICY
#define IL_STRINGIFY_IMPL(value) #value
#define IL_STRINGIFY(value) IL_STRINGIFY_IMPL(value)
void ExpectSimtPolicyCall(const std::string &kernel) {
  const std::string function = GetFunctionContaining(kernel, "// IndirectLoad SIMT");
  const std::string policy = "AscendC::IndirectLoadSimt" IL_STRINGIFY(IL_EXPECT_SIMT_POLICY) "Policy<uint" +
                             std::to_string(IL_EXPECT_SIMT_OFFSET_BITS) + "_t";
  EXPECT_NE(function.find(policy), std::string::npos) << policy;
}
#endif

#ifdef IL_EXPECT_MICRO_SIMD
void ExpectMicroSimdCall(const std::string &kernel) {
  const size_t marker = kernel.find("// IndirectLoad SIMD");
  const size_t call = kernel.find("AscendC::", marker);
  const size_t micro = kernel.find("AscendC::IndirectLoadSimd<half, int32_t, 3, 1>", marker);
  ASSERT_NE(marker, std::string::npos);
  ASSERT_NE(micro, std::string::npos);
  EXPECT_EQ(call, micro);
}
#endif

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
  const af::AscNodePtr input_post =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
  ASSERT_NE(input_post, nullptr);
  EXPECT_EQ(input_post->GetName(), "input_exp2");
  const af::AscNodePtr input_vf = ascgen_utils::indirect_load::GetInputProducer(input_post, 0UL);
  ASSERT_NE(input_vf, nullptr);
  EXPECT_TRUE(af::ops::IsOps<af::ascir_op::VectorFunc>(input_vf));

  const af::AscNodePtr index_post =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
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

void ExpectFullPrefixPostReduceSimd(af::AscGraph &graph, const af::AscNodePtr &indirect_load) {
  const af::AscNodePtr input_exp2 =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
  ASSERT_NE(input_exp2, nullptr);
  EXPECT_EQ(input_exp2->GetName(), "input_exp2");
  const af::AscNodePtr input_pre = ascgen_utils::indirect_load::GetInputProducer(input_exp2, 0UL);
  ASSERT_NE(input_pre, nullptr);
  EXPECT_TRUE(input_pre->GetName() == "input_relu" || af::ops::IsOps<af::ascir_op::VectorFunc>(input_pre));
  const af::AscNodePtr input_load = ascgen_utils::indirect_load::GetInputProducer(input_pre, 0UL);
  ASSERT_NE(input_load, nullptr);
  EXPECT_EQ(input_load->GetName(), "input_load");

  const std::vector<const char *> index_chain = {"index_floor_to_int", "index_log2", "index_exp2"};
  af::AscNodePtr index_node =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
  for (const char *name : index_chain) {
    ASSERT_NE(index_node, nullptr);
    EXPECT_EQ(index_node->GetName(), name);
    index_node = ascgen_utils::indirect_load::GetInputProducer(index_node, 0UL);
  }
  ASSERT_NE(index_node, nullptr);
  EXPECT_TRUE(af::ops::IsOps<af::ascir_op::VectorFunc>(index_node));
  index_node = ascgen_utils::indirect_load::GetInputProducer(index_node, 0UL);
  ASSERT_NE(index_node, nullptr);
  EXPECT_EQ(index_node->GetName(), "index_load");
  EXPECT_NE(ascgen_utils::indirect_load::GetPostReduceConsumer(indirect_load), nullptr);
  EXPECT_NE(graph.FindNode("store"), nullptr);
}

void ExpectDirectInputPostReduceSimd(const af::AscNodePtr &indirect_load) {
  const af::AscNodePtr input_load =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
  ASSERT_NE(input_load, nullptr);
  EXPECT_EQ(input_load->GetName(), "input_load");

  const af::AscNodePtr index_producer =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
  ASSERT_NE(index_producer, nullptr);
  EXPECT_FALSE(kMixedIndexPre);
  if (kDirectIndex) {
    EXPECT_EQ(index_producer->GetName(), "index_load");
  } else {
    EXPECT_EQ(index_producer->GetName(), "index_abs");
    const af::AscNodePtr index_load = ascgen_utils::indirect_load::GetInputProducer(index_producer, 0UL);
    ASSERT_NE(index_load, nullptr);
    EXPECT_EQ(index_load->GetName(), "index_load");
  }

  const af::AscNodePtr reduce = ascgen_utils::indirect_load::GetPostReduceConsumer(indirect_load);
  ASSERT_NE(reduce, nullptr);
  std::vector<const char *> output_chain;
  if (kOutputPostType == ascir::IndirectLoadOutputPostType::kSum) {
    output_chain = {"output_sum", "store"};
  } else if (kOutputPostType == ascir::IndirectLoadOutputPostType::kExp2Sum) {
    output_chain = {"output_exp2", "output_sum", "store"};
  } else if (kOutputPostType == ascir::IndirectLoadOutputPostType::kAbsExp2Sum) {
    output_chain = {"output_abs", "output_exp2", "output_sum", "store"};
  } else {
    FAIL() << "Unexpected output post type for selected SIMD case.";
  }

  af::AscNodePtr consumer = indirect_load;
  for (const char *name : output_chain) {
    consumer = ascgen_utils::indirect_load::GetOnlyOutputConsumer(consumer);
    ASSERT_NE(consumer, nullptr) << name;
    EXPECT_EQ(consumer->GetName(), name);
  }
  EXPECT_EQ(reduce->GetName(), "output_sum");
  EXPECT_TRUE(af::ops::IsOps<af::ascir_op::Store>(consumer));
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
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    const auto input_boundary = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
    ASSERT_NE(input_boundary, nullptr);
    EXPECT_TRUE(af::ops::IsOps<af::ascir_op::Load>(input_boundary));
    EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(input_boundary),
              ascgen_utils::indirect_load::TemplateRole::kSkInputBoundary);
    EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(indirect_load),
              ascgen_utils::indirect_load::TemplateRole::kSkOp);
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
        std::vector<af::AxisId>(logical_view.input.axis_ids.begin() + split, logical_view.input.axis_ids.end()));
  }

  const auto input_producer =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
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
  const auto index_producer =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
  ASSERT_NE(index_producer, nullptr);
#ifdef IL_DIRECT_INDEX
  EXPECT_EQ(index_producer->GetName(), "index_load");
#else
#ifdef IL_MIXED_ELEMENTWISE
  EXPECT_EQ(index_producer->GetName(), "index_floor");
#else
  EXPECT_EQ(index_producer->GetName(), kMixedIndexPre ? "index_floor_to_int" : "index_abs");
  ExpectMixedInputPre(indirect_load);
#endif
#endif
#if IL_INPUT_PRE_TYPE == 5
  EXPECT_EQ(indirect_load->outputs()[0]->attr.dtype, af::DT_FLOAT16);
#endif
}

void CollectImplGraphs(ascir::ScheduleGroup &group, std::vector<af::AscGraph *> &graphs) {
  for (auto &graph : group.impl_graphs) {
    graphs.emplace_back(&graph);
  }
}

void CollectImplGraphs(ascir::ScheduledResult &candidate, std::vector<af::AscGraph *> &graphs) {
  for (auto &group : candidate.schedule_groups) {
    CollectImplGraphs(group, graphs);
  }
}

void CollectImplGraphs(std::vector<ascir::ScheduledResult> &candidates, std::vector<af::AscGraph *> &graphs) {
  for (auto &candidate : candidates) {
    CollectImplGraphs(candidate, graphs);
  }
}

std::vector<af::AscGraph *> CollectImplGraphs(ascir::FusedScheduledResult &result) {
  std::vector<af::AscGraph *> graphs;
  for (auto &candidates : result.node_idx_to_scheduled_results) {
    CollectImplGraphs(candidates, graphs);
  }
  return graphs;
}

#if defined(IL_EXPECT_ONLY_SIMT) || defined(IL_EXPECT_SIMT_ONLY)
void ExpectAraFallbackSimtSemantics(af::AscGraph &graph, const af::AscNodePtr &indirect_load) {
#ifdef IL_EXPECT_ONLY_SIMT
  static_assert(kRank == 4UL && kAxis == 1L);
  static_assert(kInputPreType == ascir::IndirectLoadInputPreType::kReluExp2);
  static_assert(kOutputPostType == ascir::IndirectLoadOutputPostType::kSumKeepTail);
  static_assert(kMixedIndexPre);

  ascgen_utils::indirect_load::TemplateAxes axes;
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  const std::vector<af::AxisId> outer_origins(logical_view.output.axis_ids.begin(),
                                              logical_view.output.axis_ids.begin() + 2L);
  const std::vector<af::AxisId> inner_origins(logical_view.output.axis_ids.begin() + 2L,
                                              logical_view.output.axis_ids.end());
  ExpectAxisOrigins(graph, axes.outer_axis, outer_origins);

  const af::AscNodePtr reduce = ascgen_utils::indirect_load::GetPostReduceConsumer(indirect_load);
  ASSERT_NE(reduce, nullptr);
  std::vector<af::AxisId> actual_outer_origins;
  CollectOriginAxes(graph, axes.outer_axis, actual_outer_origins);
  EXPECT_EQ(actual_outer_origins.size(), 2UL);
  ASSERT_NE(std::find(reduce->attr.sched.axis.begin(), reduce->attr.sched.axis.end(), axes.inner_axis),
            reduce->attr.sched.axis.end());
  ExpectAxisOrigins(graph, axes.inner_axis, inner_origins);

  const auto input_load = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  const auto index_load = graph.FindNode("index_load");
  const auto index_transform = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 1UL);
  ASSERT_TRUE(af::ops::IsOps<af::ascir_op::Load>(input_load));
  ASSERT_TRUE(af::ops::IsOps<af::ascir_op::Load>(index_load));
  EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(input_load),
            ascgen_utils::indirect_load::TemplateRole::kSimtInputBoundary);
  EXPECT_TRUE(ascgen_utils::indirect_load::GetTemplateBehavior(input_load).uses_direct_gm_pipeline);
  EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(index_load),
            ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary);
  EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(index_transform),
            ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform);
  for (const char *name : {"input_relu", "input_exp2", "index_abs", "index_abs2", "index_cast_float", "index_exp2",
                           "index_log2", "index_floor_to_int"}) {
    const auto transform = graph.FindNode(name);
    ASSERT_NE(transform, nullptr) << name;
    EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(transform),
              ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform)
        << name;
  }
  const af::AscNodePtr store = ascgen_utils::indirect_load::GetOnlyOutputConsumer(reduce);
  ASSERT_NE(store, nullptr);
  EXPECT_TRUE(af::ops::IsOps<af::ascir_op::Store>(store));
#else
  (void)graph;
  (void)indirect_load;
#endif
}

bool CheckOnlySimtAraSchedule(ascir::FusedScheduledResult &result) {
  size_t simd_count = 0UL;
  size_t simt_count = 0UL;
  size_t sk_count = 0UL;
  size_t unknown_template_count = 0UL;
  for (auto &candidates : result.node_idx_to_scheduled_results) {
    for (auto &candidate : candidates) {
      for (auto &group : candidate.schedule_groups) {
        for (auto &graph : group.impl_graphs) {
          const af::AscNodePtr indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(graph);
          if (indirect_load == nullptr) {
            continue;
          }
          const auto template_id = ascir::GetTemplateIdOrDefault(*indirect_load);
          if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
            ++simd_count;
          } else if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
            ++simt_count;
            ExpectAraFallbackSimtSemantics(graph, indirect_load);
          } else if (template_id == ascir::TemplateId::kIndirectLoadSK) {
            ++sk_count;
          } else {
            ++unknown_template_count;
          }
        }
      }
    }
  }
  EXPECT_EQ(simt_count, 1UL) << "The only SIMD/SIMT template across all impl graphs must be SIMT.";
  EXPECT_EQ(simd_count, 0UL) << "ARA fallback must not generate a SIMD candidate.";
  EXPECT_GT(sk_count, 0UL) << "The upstream SK candidate must remain available before selection.";
  EXPECT_EQ(unknown_template_count, 0UL) << "Unexpected IndirectLoad TemplateId must not be treated as SIMT.";
  return simt_count == 1UL && simd_count == 0UL && sk_count > 0UL && unknown_template_count == 0UL;
}
#endif

bool CheckScheduledLoopFramework(ascir::FusedScheduledResult &result) {
  size_t simd_count = 0UL;
  size_t simt_count = 0UL;
  size_t sk_count = 0UL;
  for (auto *graph : CollectImplGraphs(result)) {
    const af::AscNodePtr indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(*graph);
    if (indirect_load == nullptr) {
      continue;
    }
    ExpectLoopFramework(*graph, indirect_load);
    const auto template_id = ascir::GetTemplateIdOrDefault(*indirect_load);
    if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
      ++simd_count;
    } else if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
      ++simt_count;
    } else if (template_id == ascir::TemplateId::kIndirectLoadSK) {
      ++sk_count;
    }
  }
  EXPECT_GT(simd_count, 0UL);
  EXPECT_GT(sk_count, 0UL);
  if (kExpectSimt) {
    EXPECT_GT(simt_count, 0UL);
    return simd_count > 0UL && simt_count > 0UL && sk_count > 0UL;
  } else {
    EXPECT_EQ(simt_count, 0UL);
    return simd_count > 0UL && simt_count == 0UL && sk_count > 0UL;
  }
}

uint32_t TemplateMask(ascir::TemplateId template_id) {
  return 1U << static_cast<uint32_t>(template_id);
}

bool ContainsTemplate(const af::AscGraph &graph, ascir::TemplateId template_id) {
  const auto indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(graph);
  return indirect_load != nullptr && ascir::GetTemplateIdOrDefault(*indirect_load) == template_id;
}

bool ContainsTemplate(const ascir::ScheduleGroup &group, ascir::TemplateId template_id) {
  return std::any_of(group.impl_graphs.cbegin(), group.impl_graphs.cend(),
                     [template_id](const af::AscGraph &graph) { return ContainsTemplate(graph, template_id); });
}

bool ContainsTemplate(const ascir::ScheduledResult &candidate, ascir::TemplateId template_id) {
  return std::any_of(candidate.schedule_groups.cbegin(), candidate.schedule_groups.cend(),
                     [template_id](const ascir::ScheduleGroup &group) { return ContainsTemplate(group, template_id); });
}

bool ValidateAndSelectRuntimeTemplate(ascir::FusedScheduledResult &result) {
  constexpr char kSelectedScore[] = "int32_t CalcScore(AutofuseTilingData &tiling_data) { return 2; }";
  constexpr char kFallbackScore[] = "int32_t CalcScore(AutofuseTilingData &tiling_data) { return 0; }";
  std::map<ascir::TemplateId, size_t> template_counts;
  bool is_valid = (kExpectedTemplates & TemplateMask(kSelectedTemplate)) != 0U;
  EXPECT_TRUE(is_valid) << "Selected template must be included in the expected template set.";
  for (auto &candidates : result.node_idx_to_scheduled_results) {
    for (auto candidate = candidates.begin(); candidate != candidates.end();) {
      std::vector<ascir::TemplateId> matched_templates;
      for (const ascir::TemplateId template_id : kIndirectLoadTemplates) {
        if (ContainsTemplate(*candidate, template_id)) {
          matched_templates.emplace_back(template_id);
        }
      }
      if (matched_templates.empty()) {
        ++candidate;
        continue;
      }
      EXPECT_EQ(matched_templates.size(), 1UL) << "One candidate must contain exactly one IndirectLoad template.";
      if (matched_templates.size() != 1UL) {
        is_valid = false;
        ++candidate;
        continue;
      }
      const ascir::TemplateId template_id = matched_templates.front();
      if ((kExpectedTemplates & TemplateMask(template_id)) == 0U) {
        candidate = candidates.erase(candidate);
        continue;
      }
      ++template_counts[template_id];
      auto implementation = ascgen_utils::indirect_load::Implementation::kDefault;
      for (ascir::ScheduleGroup &group : candidate->schedule_groups) {
        for (af::AscGraph &graph : group.impl_graphs) {
          const af::AscNodePtr indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(graph);
          if (indirect_load == nullptr) {
            continue;
          }
          const af::Status status = ascgen_utils::indirect_load::GetImplementation(indirect_load, implementation);
          EXPECT_EQ(status, af::SUCCESS);
          is_valid = is_valid && status == af::SUCCESS;
        }
      }
      const bool is_selected = template_id == kSelectedTemplate && implementation == kSelectedImplementation;
      candidate->score_func = is_selected ? kSelectedScore : kFallbackScore;
      ++candidate;
    }
  }
  for (const ascir::TemplateId template_id : kIndirectLoadTemplates) {
    const size_t expected_count = (kExpectedTemplates & TemplateMask(template_id)) == 0U
                                      ? 0UL
                                      : (template_id == ascir::TemplateId::kIndirectLoadSimd ? 2UL : 1UL);
    EXPECT_EQ(template_counts[template_id], expected_count)
        << "Unexpected candidate count for template " << static_cast<int64_t>(template_id) << ".";
    is_valid = is_valid && template_counts[template_id] == expected_count;
  }
  return is_valid;
}

#ifdef IL_MIXED_ELEMENTWISE
bool CheckMixedElementwiseSchedule(ascir::FusedScheduledResult &result) {
  bool found = false;
  for (auto &candidates : result.node_idx_to_scheduled_results) {
    for (auto &candidate : candidates) {
      for (auto &group : candidate.schedule_groups) {
        for (auto &graph : group.impl_graphs) {
          const af::AscNodePtr indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(graph);
          if (indirect_load == nullptr ||
              ascir::GetTemplateIdOrDefault(*indirect_load) != ascir::TemplateId::kIndirectLoadSimd) {
            continue;
          }
          const af::AscNodePtr index_floor = graph.FindNode("index_floor");
          const af::AscNodePtr index_log2 = graph.FindNode("index_log2");
          const af::AscNodePtr index_exp2 = graph.FindNode("index_exp2");
          const af::AscNodePtr x_copy_sign = graph.FindNode("x_copy_sign");
          const af::AscNodePtr copy_sign = graph.FindNode("output_copy_sign");
          if (index_floor == nullptr || index_log2 == nullptr || index_exp2 == nullptr || x_copy_sign == nullptr ||
              copy_sign == nullptr) {
            ADD_FAILURE() << "Mixed elementwise nodes are missing from the SIMD candidate.";
            return false;
          }
          EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(indirect_load,
                                                                  ascgen_utils::indirect_load::kIndexTensorIndex),
                    index_floor);
          EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(indirect_load,
                                                                  ascgen_utils::indirect_load::kInputTensorIndex),
                    x_copy_sign);
          EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(index_floor, 0UL), index_log2);
          EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(index_log2, 0UL), index_exp2);
          EXPECT_TRUE(
              af::ops::IsOps<af::ascir_op::VectorFunc>(ascgen_utils::indirect_load::GetInputProducer(index_exp2, 0UL)));
          EXPECT_TRUE(af::ops::IsOps<af::ascir_op::VectorFunc>(
              ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load)));
          EXPECT_TRUE(
              af::ops::IsOps<af::ascir_op::VectorFunc>(ascgen_utils::indirect_load::GetInputProducer(copy_sign, 0UL)));
          EXPECT_TRUE(
              af::ops::IsOps<af::ascir_op::VectorFunc>(ascgen_utils::indirect_load::GetOnlyOutputConsumer(copy_sign)));
          ExpectLoopFramework(graph, indirect_load);
          found = true;
        }
      }
    }
  }
  EXPECT_TRUE(found);
  return found;
}

#if IL_EXPECT_SIMT
bool CheckMixedElementwiseSimtSchedule(ascir::FusedScheduledResult &result) {
  for (auto &candidates : result.node_idx_to_scheduled_results) {
    for (auto &candidate : candidates) {
      for (auto &group : candidate.schedule_groups) {
        for (auto &graph : group.impl_graphs) {
          const af::AscNodePtr indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(graph);
          if (indirect_load == nullptr ||
              ascir::GetTemplateIdOrDefault(*indirect_load) != ascir::TemplateId::kIndirectLoadSimt) {
            continue;
          }
          const af::AscNodePtr input = ascgen_utils::indirect_load::GetInputProducer(
              indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
          EXPECT_NE(input, nullptr);
          EXPECT_EQ(input == nullptr ? "" : input->GetName(), "x_load");
#ifdef IL_SIMT_ELEMENTWISE_COVERAGE
          for (const char *name : {"index_minimum", "index_maximum", "index_gt", "index_lt", "index_logical_or",
                                   "index_where", "output_sub", "output_div", "output_copy_sign"}) {
            EXPECT_NE(graph.FindNode(name), nullptr) << name;
          }
#else
          EXPECT_NE(graph.FindNode("index_add"), nullptr);
          EXPECT_NE(graph.FindNode("index_floor"), nullptr);
          EXPECT_NE(graph.FindNode("output_add"), nullptr);
          EXPECT_NE(graph.FindNode("output_copy_sign"), nullptr);
          EXPECT_NE(graph.FindNode("output_mul"), nullptr);
#endif
          ExpectLoopFramework(graph, indirect_load);
          return true;
        }
      }
    }
  }
  ADD_FAILURE() << "Mixed multi-input SIMT candidate was not generated.";
  return false;
}
#endif
#endif

void ExpectGeneratedTemplates(const std::string &kernel) {
  constexpr std::array<const char *, 3UL> kMarkers = {"// IndirectLoad SIMD", "// IndirectLoad SIMT",
                                                      "// IndirectLoad SK"};
  const std::array<const char *, 3UL> apis = {"IndirectLoadSimd<", "AscendC::IndirectLoadSimt<", "IndirectLoadSk<"};
  for (size_t i = 0UL; i < kIndirectLoadTemplates.size(); ++i) {
    const bool is_expected = (kExpectedTemplates & TemplateMask(kIndirectLoadTemplates[i])) != 0U;
    if (is_expected) {
      EXPECT_NE(kernel.find(kMarkers[i]), std::string::npos) << kMarkers[i];
      EXPECT_NE(kernel.find(apis[i]), std::string::npos) << apis[i];
      if (kIndirectLoadTemplates[i] == ascir::TemplateId::kIndirectLoadSimd) {
#ifdef IL_INPUT_OUTER_STRIDE
        EXPECT_EQ(kernel.find("IndirectLoadSimdGatherApi<"), std::string::npos);
#else
        EXPECT_NE(kernel.find("IndirectLoadSimdGatherApi<"), std::string::npos);
#endif
      }
    } else {
      EXPECT_EQ(kernel.find(kMarkers[i]), std::string::npos) << kMarkers[i];
      EXPECT_EQ(kernel.find(apis[i]), std::string::npos) << apis[i];
      if (kIndirectLoadTemplates[i] == ascir::TemplateId::kIndirectLoadSimd) {
        EXPECT_EQ(kernel.find("IndirectLoadSimdGatherApi<"), std::string::npos);
      }
    }
  }
}

void ExpectSimdKernelStructure(const std::string &kernel) {
#ifndef IL_INPUT_OUTER_STRIDE
  if ((kExpectedTemplates & TemplateMask(ascir::TemplateId::kIndirectLoadSimd)) != 0U) {
    const std::string gather_function = GetFunctionContaining(kernel, "IndirectLoadSimdGatherApi<");
    EXPECT_NE(gather_function.find("IndirectLoadSimdGatherApi<"), std::string::npos);
    EXPECT_EQ(gather_function.find("GetValue("), std::string::npos);
    EXPECT_EQ(gather_function.find("SetValue("), std::string::npos);
    const std::string gather_api = GetFunctionContaining(kernel, "__aicore__ inline void IndirectLoadSimdGatherApi(");
    EXPECT_TRUE(ContainsInOrder(gather_api, {"IndirectLoadSimdBuildOffsets", "PipeBarrier<PIPE_V>", "Gather(y"}));
    EXPECT_EQ(gather_api.find("HardEvent::V_MTE3"), std::string::npos);
    const std::string micro_dispatch = GetFunctionContaining(kernel, "IndirectLoadSimdRegGather(__ubuf__ X *x");
    EXPECT_NE(micro_dispatch.find("IndirectLoadSimdDispatch"), std::string::npos);
    EXPECT_EQ(micro_dispatch.find("__VEC_SCOPE__"), std::string::npos);
    const std::string gather_dispatch =
        GetFunctionContaining(kernel, "IndirectLoadSimdBuildOffsets(__ubuf__ Index *index");
    EXPECT_NE(gather_dispatch.find("IndirectLoadSimdDispatch"), std::string::npos);
    EXPECT_EQ(gather_dispatch.find("__VEC_SCOPE__"), std::string::npos);
    const std::string common_dispatch =
        GetFunctionContaining(kernel, "IndirectLoadSimdDispatch(typename DispatchPolicy::Args &args");
    EXPECT_TRUE(ContainsInOrder(common_dispatch, {"if constexpr (Axis + 1 == Rank)", "context.inner_layout_matches",
                                                  "IndirectLoadSimdIsPowerOfTwo(context.index_inner)",
                                                  "DispatchPolicy::RunReuse", "DispatchPolicy::template RunMode"}));
    EXPECT_EQ(common_dispatch.find("__VEC_SCOPE__"), std::string::npos);
    const std::string run_mode = GetFunctionContaining(kernel, "IndirectLoadSimdRunMode(");
    EXPECT_TRUE(
        ContainsInOrder(run_mode, {"full_repeats", "tail_count", "IndexPolicy::Init(state, args.index)",
                                   "IndirectLoadSimdRunPair", "CreateMask<uint32_t", "IndirectLoadSimdRunRepeat"}));
    const std::string run_reuse = GetFunctionContaining(kernel, "IndirectLoadSimdRunReuse(");
    EXPECT_TRUE(ContainsInOrder(
        run_reuse, {"full_repeats", "tail_count", "IndexPolicy::Init(state, args.index)",
                    "IndirectLoadSimdInitInnerOffset(inner_offset", "Action::template Commit<ValuePolicy>"}));
    EXPECT_NE(kernel.find("struct IndirectLoadSimdGatherAction"), std::string::npos);
    EXPECT_NE(kernel.find("struct IndirectLoadSimdOffsetAction"), std::string::npos);
    EXPECT_EQ(kernel.find("IndirectLoadSimdRunGatherReuse"), std::string::npos);
    EXPECT_EQ(kernel.find("IndirectLoadSimdRunBuildOffsetReuse"), std::string::npos);
  }
  EXPECT_NE(kernel.find("Mode != IndirectLoadSimdAddressMode::kDirect"), std::string::npos);
  EXPECT_NE(kernel.find("Mode == IndirectLoadSimdAddressMode::kDirect"), std::string::npos);
  EXPECT_EQ(kernel.find("IndirectLoadSimdSelectAddressMode"), std::string::npos);
  EXPECT_EQ(kernel.find("IndirectLoadSimdAddressMode::kCross"), std::string::npos);
  EXPECT_EQ(kernel.find("IndirectLoadSimdAddressMode::kGeneric"), std::string::npos);
  EXPECT_EQ(kernel.find("IndirectLoadSimdLog2"), std::string::npos);
  EXPECT_EQ(kernel.find("reuse_elements % context.index_inner"), std::string::npos);
#ifdef IL_EXPECT_MICRO_SIMD
  ExpectMicroSimdCall(kernel);
#endif
  EXPECT_EQ(kernel.find("MicroAPI::ShiftRights(outer, position"), std::string::npos);
  EXPECT_EQ(kernel.find("MicroAPI::RegTensor<uint32_t> &position, const int64_t *shape"), std::string::npos);
  EXPECT_NE(kernel.find("const IndirectLoadSimdAddressContext &context"), std::string::npos);
  EXPECT_EQ(kernel.find("__simd_vf__ inline static void Init(LoadState"), std::string::npos);
  EXPECT_EQ(kernel.find("__simd_vf__ inline void IndirectLoadSimd"), std::string::npos);
#endif
}

void CheckGeneratedKernel(const std::string &kernel) {
  ExpectGeneratedTemplates(kernel);
#ifdef IL_FUNCTIONAL_ONLY
  return;
#endif
  ExpectSimdKernelStructure(kernel);
  if ((kExpectedTemplates & TemplateMask(ascir::TemplateId::kIndirectLoadSimt)) != 0U) {
    ExpectSimtKernelStructure(kernel);
  }
#ifdef IL_EXPECT_SIMT_POLICY
  ExpectSimtPolicyCall(kernel);
#endif
#if IL_AXIS + 1 == IL_RANK
  EXPECT_EQ(kernel.find("if constexpr (Axis + 1 == Rank) {\n        MicroAPI::Muls(source_index"), std::string::npos);
#endif
#if defined(IL_DATA_FLOAT) && defined(IL_INDEX_INT64)
  EXPECT_NE(kernel.find("if (count0 != kElementsPerLoad || count1 != kElementsPerLoad)"), std::string::npos);
  EXPECT_NE(kernel.find("static constexpr uint32_t kElementsPerRepeat = VECTOR_REG_WIDTH / sizeof(uint32_t);"),
            std::string::npos);
#endif
#ifdef IL_POST_REDUCE
#if defined(IL_EXPECT_ONLY_SIMT) || defined(IL_EXPECT_SIMT_ONLY)
  EXPECT_EQ(kernel.find("// IndirectLoad SIMD"), std::string::npos);
  EXPECT_EQ(kernel.find("IndirectLoadSimd<"), std::string::npos);
  ExpectPostReduceSimtFramework(kernel);
  const size_t simt_body = kernel.find("struct IndirectLoadSimtBody");
  const size_t index_func = kernel.find(" Index(", simt_body);
  const size_t output_func = kernel.find(" Output(", index_func);
  ASSERT_NE(simt_body, std::string::npos);
  ASSERT_NE(index_func, std::string::npos);
  ASSERT_NE(output_func, std::string::npos);
  const std::string index_body = kernel.substr(index_func, output_func - index_func);
  EXPECT_TRUE(ContainsInOrder(index_body, {"Simt::Abs(", "Simt::Abs(", "static_cast<float>", "Simt::Exp2(",
                                           "Simt::Log2(", "RoundMode::CAST_FLOOR"}));
  const size_t simt_body_end = kernel.find("};", output_func);
  ASSERT_NE(simt_body_end, std::string::npos);
  const std::string output_body = kernel.substr(output_func, simt_body_end - output_func);
  EXPECT_TRUE(ContainsInOrder(output_body, {"Simt::Max(", "Simt::Exp2("}));
#elif defined(IL_POST_REDUCE_SIMD)
  EXPECT_NE(kernel.find("// IndirectLoad SIMD"), std::string::npos);
  EXPECT_NE(kernel.find("IndirectLoadSimd<"), std::string::npos);
  EXPECT_EQ(kernel.find("// IndirectLoad SIMT"), std::string::npos);
  EXPECT_TRUE(ContainsInOrder(kernel, {"IndirectLoadSimd<", "ModifiedBesselK0", "ReduceSum", "DataCopyPadExtend"}));
#else
  EXPECT_NE(kernel.find("// IndirectLoad SIMD"), std::string::npos);
  EXPECT_NE(kernel.find("// IndirectLoad SIMT"), std::string::npos);
  EXPECT_NE(kernel.find("IndirectLoadSimt<"), std::string::npos);
#ifdef IL_POST_REDUCE_EXP2
  EXPECT_NE(kernel.find("Simt::Exp2("), std::string::npos);
#endif
  EXPECT_NE(kernel.find("ReduceSum"), std::string::npos);
  EXPECT_NE(kernel.find("PipeBarrier<PIPE_V>"), std::string::npos);
#ifdef IL_POST_REDUCE_ABS
  ExpectPostReduceSimtFramework(kernel);
#endif
#endif
  return;
#endif
#ifdef IL_EXPECT_SK
  EXPECT_NE(kernel.find("// IndirectLoad SK"), std::string::npos);
  EXPECT_NE(kernel.find("IndirectLoadSk<"), std::string::npos);
  EXPECT_EQ(kernel.find("auto indirect_load_offset"), std::string::npos);
  EXPECT_EQ(kernel.find("int64_t inner = global_idx % index_inner;"), std::string::npos);
#endif
  if ((kExpectedTemplates & TemplateMask(ascir::TemplateId::kIndirectLoadSimt)) != 0U) {
    EXPECT_EQ(kernel.find("IndirectLoadSimtKernel_"), std::string::npos);
  }
#ifdef IL_DATA_BF16
  EXPECT_NE(kernel.find("IndirectLoadSimt<bfloat16_t, bfloat16_t"), std::string::npos);
#endif
#ifdef IL_DATA_UINT32
  EXPECT_NE(kernel.find("IndirectLoadSimd<uint32_t, int32_t"), std::string::npos);
#endif
#ifdef IL_MIXED_ELEMENTWISE
#if !IL_EXPECT_SIMT
  const size_t il_pos = kernel.find("// IndirectLoad SIMD");
  ASSERT_NE(il_pos, std::string::npos);
  const size_t input_copy_sign_pos = kernel.rfind("CopySignExtend(", il_pos);
  const size_t output_copy_sign_pos = kernel.find("CopySignExtend(", il_pos);
  EXPECT_NE(input_copy_sign_pos, std::string::npos);
  EXPECT_NE(output_copy_sign_pos, std::string::npos);
  EXPECT_LT(input_copy_sign_pos, il_pos);
  EXPECT_LT(il_pos, output_copy_sign_pos);
  EXPECT_NE(kernel.find("\n      Exp2("), std::string::npos);
  EXPECT_NE(kernel.find("\n      Log2("), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::RoundMode::CAST_FLOOR"), std::string::npos);
#if IL_RANK == 4 && IL_AXIS == 2
  ExpectSimdFramework(kernel);
#endif
#else
  EXPECT_NE(kernel.find("index0_data"), std::string::npos);
  EXPECT_NE(kernel.find("index1_data"), std::string::npos);
  EXPECT_NE(kernel.find("addend_data"), std::string::npos);
  EXPECT_NE(kernel.find("sign_data"), std::string::npos);
  EXPECT_NE(kernel.find("scale_data"), std::string::npos);
#if IL_RANK == 4 && IL_AXIS == 2
  ExpectNoReduceSimtFramework(kernel);
#endif
#endif
#endif
}

std::map<std::string, std::string> BuildShapeInfo() {
  std::map<std::string, std::string> shape_info;
  for (size_t i = 0UL; i < 2UL * kRank; ++i) {
    shape_info.emplace("s" + std::to_string(i), "stub_s" + std::to_string(i));
  }
  return shape_info;
}

void ApplyInputOuterStride(const af::ComputeGraphPtr &graph) {
#ifdef IL_INPUT_OUTER_STRIDE
  const auto backend = graph->FindNode("asc_backend");
  ASSERT_NE(backend, nullptr);
  const auto fuse_attrs = backend->GetOpDesc()->GetAttrsGroup<af::AutoFuseAttrs>();
  ASSERT_NE(fuse_attrs, nullptr);
  const auto &asc_graph = fuse_attrs->GetAscGraph();
  ASSERT_NE(asc_graph, nullptr);
  const auto input_load = asc_graph->FindNode("input_load");
  ASSERT_NE(input_load, nullptr);
  ASSERT_FALSE(input_load->outputs().empty());
  ASSERT_FALSE(input_load->outputs()[0]->attr.strides.empty());
  input_load->outputs()[0]->attr.strides[0] = af::Symbol(IL_INPUT_OUTER_STRIDE);
#else
  (void)graph;
#endif
}

void OptimizeGraph(const af::ComputeGraphPtr &graph, ascir::FusedScheduledResult &fused_schedule_result) {
  optimize::Optimizer optimizer(optimize::OptimizerOptions{.graph_type = optimize::GraphType::kFusedAscBackend});
  testing::internal::CaptureStdout();
  const auto optimize_status = optimizer.Optimize(graph, fused_schedule_result);
  const std::string optimize_logs = testing::internal::GetCapturedStdout();
  EXPECT_EQ(optimize_logs.find("[ERROR]"), std::string::npos) << optimize_logs;
  ASSERT_EQ(optimize_status, 0) << optimize_logs;
#ifdef IL_POST_REDUCE
#if defined(IL_EXPECT_ONLY_SIMT) || defined(IL_EXPECT_SIMT_ONLY)
  ASSERT_TRUE(CheckOnlySimtAraSchedule(fused_schedule_result)) << optimize_logs;
#else
  bool has_expected_template = false;
  for (auto &candidates : fused_schedule_result.node_idx_to_scheduled_results) {
    for (auto &candidate : candidates) {
      for (auto &group : candidate.schedule_groups) {
        for (auto &impl_graph : group.impl_graphs) {
          const auto indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(impl_graph);
          if (indirect_load != nullptr && ascir::GetTemplateIdOrDefault(*indirect_load) ==
#if defined(IL_POST_REDUCE_SIMD) || defined(IL_EXPECT_SIMD_SELECTED)
                                              ascir::TemplateId::kIndirectLoadSimd) {
            if (kInputPreType == ascir::IndirectLoadInputPreType::kNone) {
              ExpectDirectInputPostReduceSimd(indirect_load);
            } else {
              ExpectFullPrefixPostReduceSimd(impl_graph, indirect_load);
            }
#else
                                              ascir::TemplateId::kIndirectLoadSimt) {
#endif
            has_expected_template = true;
          }
        }
      }
    }
  }
  ASSERT_TRUE(has_expected_template) << optimize_logs;
#endif
#elif defined(IL_MIXED_ELEMENTWISE) && IL_EXPECT_SIMT
  ASSERT_TRUE(CheckMixedElementwiseSimtSchedule(fused_schedule_result)) << optimize_logs;
#elif defined(IL_MIXED_ELEMENTWISE)
  ASSERT_TRUE(CheckMixedElementwiseSchedule(fused_schedule_result)) << optimize_logs;
#else
  ASSERT_TRUE(CheckScheduledLoopFramework(fused_schedule_result)) << optimize_logs;
#endif
  ASSERT_TRUE(ValidateAndSelectRuntimeTemplate(fused_schedule_result));
}

void GenerateKernel(const std::map<std::string, std::string> &shape_info,
                    const ascir::FusedScheduledResult &fused_schedule_result, codegen::CodegenResult &result) {
  codegen::Codegen codegen(codegen::CodegenOptions{});
  testing::internal::CaptureStdout();
  const auto codegen_status = codegen.Generate(shape_info, fused_schedule_result, result);
  const std::string codegen_logs = testing::internal::GetCapturedStdout();
  EXPECT_EQ(codegen_logs.find("[ERROR]"), std::string::npos) << codegen_logs;
  ASSERT_EQ(codegen_status, 0) << codegen_logs;
  CheckGeneratedKernel(result.kernel);
}

void WriteGeneratedFiles(const codegen::CodegenResult &result) {
  constexpr char kTilingStub[] = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  const std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  ASSERT_EQ(parts.size(), 3U);
  std::fstream kernel_file(parts[0], std::ios::out);
  std::fstream tiling_file(parts[1], std::ios::out);
  std::fstream tiling_data_file(parts[2], std::ios::out);
  ASSERT_TRUE(kernel_file.is_open());
  ASSERT_TRUE(tiling_file.is_open());
  ASSERT_TRUE(tiling_data_file.is_open());
  kernel_file << kTilingStub << RemoveSubDirInclude(result.kernel);
  tiling_file << result.tiling;
  tiling_data_file << result.tiling_data;
  EXPECT_TRUE(kernel_file.good());
  EXPECT_TRUE(tiling_file.good());
  EXPECT_TRUE(tiling_data_file.good());
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
  try {
#ifdef IL_MIXED_ELEMENTWISE
    ASSERT_TRUE(IsMixedGatherShapeValid());
#ifdef IL_SIMT_ELEMENTWISE_COVERAGE
    auto graph = ascir::ShareGraph::IndirectLoadSimtElementwiseCoverageFusedGraph(kRank, kAxis, GetStaticShape(true),
                                                                                  GetStaticShape(false));
#else
    auto graph = ascir::ShareGraph::IndirectLoadMixedElementwiseFusedGraph(kRank, kAxis, kExpectSimt,
                                                                           GetStaticShape(true), GetStaticShape(false));
#endif
#else
    auto graph = ascir::ShareGraph::IndirectLoadStoreFusedGraph(kRank, kAxis, kDataType, kIndexType, kInputPreType,
                                                                kUseExp2, kOutputPostType, GetStaticShape(true),
                                                                GetStaticShape(false), kMixedIndexPre, kDirectIndex);
#endif
    ASSERT_NE(graph, nullptr);
    ASSERT_NO_FATAL_FAILURE(ApplyInputOuterStride(graph));
    const auto shape_info = BuildShapeInfo();
    ascir::FusedScheduledResult fused_schedule_result;
    ASSERT_NO_FATAL_FAILURE(OptimizeGraph(graph, fused_schedule_result));
    codegen::CodegenResult result;
    ASSERT_NO_FATAL_FAILURE(GenerateKernel(shape_info, fused_schedule_result, result));
    ASSERT_NO_FATAL_FAILURE(WriteGeneratedFiles(result));
  } catch (const std::exception &e) {
    FAIL() << e.what();
  } catch (...) {
    FAIL() << "Unknown exception";
  }
}
