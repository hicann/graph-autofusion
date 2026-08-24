/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_INDIRECT_LOAD_STORE_TEST_INDIRECT_LOAD_BACKEND_GENERATOR_COMMON_H_
#define AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_INDIRECT_LOAD_STORE_TEST_INDIRECT_LOAD_BACKEND_GENERATOR_COMMON_H_

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "backend_common.h"
#include "codegen.h"
#include "common/platform_context.h"
#include "fusion/autofuse_attrs.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "indirect_load_utils.h"
#include "optimize.h"
#include "optimize/pre_process/pre_process_config.h"
#include "runtime_stub.h"

namespace indirect_load_test {
constexpr char kTilingStub[] = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling) AutofuseTilingData t = *(AutofuseTilingData *)tiling;
)";

template <typename Op>
void SetView(Op &op, const std::vector<af::AxisId> &axes, const std::vector<af::Expression> &repeats,
             const std::vector<af::Expression> &strides, af::DataType dtype) {
  op.attr.sched.axis = axes;
  op.y.dtype = dtype;
  *op.y.axis = axes;
  *op.y.repeats = repeats;
  *op.y.strides = strides;
}

class BackendGraph {
 public:
  BackendGraph(const char *graph_name, const char *data_name, const char *index_name, af::DataType data_type,
               af::DataType index_type = af::DT_INT64)
      : fused_graph_(graph_name), data_name_(data_name), index_name_(index_name) {
    af::ascir_op::Data data(data_name, fused_graph_);
    data.ir_attr.SetIndex(0);
    af::ascir_op::Data index(index_name, fused_graph_);
    index.ir_attr.SetIndex(1);
    compute_graph_ = af::AscGraphUtils::GetComputeGraph(fused_graph_);
    if (compute_graph_ == nullptr) {
      return;
    }
    const auto data_desc = std::make_shared<af::GeTensorDesc>();
    data_desc->SetDataType(data_type);
    const auto index_desc = std::make_shared<af::GeTensorDesc>();
    index_desc->SetDataType(index_type);
    const auto backend_desc = std::make_shared<af::OpDesc>("asc_backend", "AscBackend");
    backend_desc->AddInputDesc(data_desc->Clone());
    backend_desc->AddInputDesc(index_desc->Clone());
    backend_desc->AddOutputDesc(data_desc->Clone());
    backend_ = compute_graph_->AddNode(backend_desc);
  }

  [[nodiscard]] bool IsValid() const {
    return compute_graph_ != nullptr && backend_ != nullptr;
  }

  [[nodiscard]] af::ComputeGraphPtr Finalize(const std::shared_ptr<af::AscGraph> &sub_graph,
                                             const char *output_name) const {
    if (!IsValid()) {
      return nullptr;
    }
    const auto fuse_attrs = backend_->GetOpDesc()->GetOrCreateAttrsGroup<af::AutoFuseAttrs>();
    if (fuse_attrs == nullptr) {
      return nullptr;
    }
    fuse_attrs->SetAscGraph(sub_graph);
    af::ascir_op::Output output(output_name);
    output.ir_attr.SetIndex(0);
    const auto output_node = compute_graph_->AddNode(af::OpDescUtils::GetOpDescFromOperator(output));
    const auto data_node = fused_graph_.FindNode(data_name_.c_str());
    const auto index_node = fused_graph_.FindNode(index_name_.c_str());
    if (data_node == nullptr || index_node == nullptr || output_node == nullptr) {
      return nullptr;
    }
    const bool edges_added =
        af::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), backend_->GetInDataAnchor(0)) == ge::GRAPH_SUCCESS &&
        af::GraphUtils::AddEdge(index_node->GetOutDataAnchor(0), backend_->GetInDataAnchor(1)) == ge::GRAPH_SUCCESS &&
        af::GraphUtils::AddEdge(backend_->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)) == ge::GRAPH_SUCCESS;
    return edges_added && compute_graph_->TopologicalSorting() == ge::GRAPH_SUCCESS ? compute_graph_ : nullptr;
  }

 private:
  af::AscGraph fused_graph_;
  std::string data_name_;
  std::string index_name_;
  af::ComputeGraphPtr compute_graph_;
  af::NodePtr backend_;
};

inline bool ContainsTemplate(const ascir::ScheduledResult &candidate, ascir::TemplateId template_id) {
  for (const auto &group : candidate.schedule_groups) {
    for (const auto &graph : group.impl_graphs) {
      const auto indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(graph);
      if (indirect_load != nullptr && ascir::GetTemplateIdOrDefault(*indirect_load) == template_id) {
        return true;
      }
    }
  }
  return false;
}

inline bool HasTemplate(const ascir::FusedScheduledResult &result, ascir::TemplateId template_id) {
  for (const auto &candidates : result.node_idx_to_scheduled_results) {
    for (const auto &candidate : candidates) {
      if (ContainsTemplate(candidate, template_id)) {
        return true;
      }
    }
  }
  return false;
}

inline void KeepOnlyTemplate(ascir::FusedScheduledResult &result, ascir::TemplateId template_id) {
  for (auto &candidates : result.node_idx_to_scheduled_results) {
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [template_id](const auto &candidate) { return !ContainsTemplate(candidate, template_id); }),
        candidates.end());
  }
}

inline ascir::TemplateId GetExpectedTemplate(bool expect_simt, bool expect_sk) {
  if (expect_sk) {
    return ascir::TemplateId::kIndirectLoadSK;
  }
  return expect_simt ? ascir::TemplateId::kIndirectLoadSimt : ascir::TemplateId::kIndirectLoadSimd;
}

inline const char *GetTemplateMarker(ascir::TemplateId template_id) {
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    return "// IndirectLoad SK";
  }
  return template_id == ascir::TemplateId::kIndirectLoadSimt ? "// IndirectLoad SIMT" : "// IndirectLoad SIMD";
}

inline void BuildOutputPath(const std::shared_ptr<af::AscGraph> &graph, af::ascir_op::IndirectLoad &indirect_load,
                            const std::vector<af::AxisId> &axes, const std::vector<af::Expression> &repeats,
                            const std::vector<af::Expression> &strides, bool with_relu) {
  indirect_load.ir_attr.SetAxis(2);
  SetView(indirect_load, axes, repeats, strides, af::DT_FLOAT16);
  af::ascir_op::Store store("store");
  graph->AddNode(store);
  if (with_relu) {
    af::ascir_op::Relu relu("output_relu");
    graph->AddNode(relu);
    relu.x = indirect_load.y;
    SetView(relu, axes, repeats, strides, af::DT_FLOAT16);
    store.x = relu.y;
  } else {
    store.x = indirect_load.y;
  }
  SetView(store, axes, repeats, strides, af::DT_FLOAT16);
  af::ascir_op::Output output("y");
  graph->AddNode(output);
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetView(output, axes, repeats, strides, af::DT_FLOAT16);
}

template <typename GraphView, typename InputBuilder, typename IndexBuilder, typename OutputBuilder>
std::shared_ptr<af::AscGraph> CreateSubGraph(GraphView view, InputBuilder build_input, IndexBuilder build_index,
                                             OutputBuilder build_output) {
  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  view.graph->AddNode(indirect_load);
  build_input(view, indirect_load);
  build_index(view, indirect_load);
  build_output(view, indirect_load);
  return view.graph;
}

inline bool SelectTemplate(const af::ComputeGraphPtr &graph, ascir::TemplateId expected_template,
                           ascir::FusedScheduledResult &scheduled_result) {
  optimize::Optimizer optimizer(optimize::OptimizerOptions{.graph_type = optimize::GraphType::kFusedAscBackend});
  if (optimizer.Optimize(graph, scheduled_result) != af::SUCCESS || !HasTemplate(scheduled_result, expected_template)) {
    return false;
  }
  KeepOnlyTemplate(scheduled_result, expected_template);
  return true;
}

inline void GenerateForTemplate(const af::ComputeGraphPtr &graph, const std::map<std::string, std::string> &shape_info,
                                ascir::TemplateId expected_template, codegen::CodegenResult &result) {
  ascir::FusedScheduledResult scheduled_result;
  ASSERT_TRUE(SelectTemplate(graph, expected_template, scheduled_result));
  codegen::Codegen codegen(codegen::CodegenOptions{});
  ASSERT_EQ(codegen.Generate(shape_info, scheduled_result, result), af::SUCCESS);
}

inline bool WriteGeneratedFile(const std::string &path, const std::string &content) {
  std::fstream file(path, std::ios::out);
  if (!file.is_open()) {
    return false;
  }
  file << content;
  return file.good();
}

inline void WriteGeneratedFiles(const codegen::CodegenResult &result) {
  const std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  ASSERT_EQ(parts.size(), 3U);
  const std::array<std::string, 3> contents = {std::string(kTilingStub) + RemoveSubDirInclude(result.kernel),
                                               result.tiling, result.tiling_data};
  for (size_t i = 0UL; i < parts.size(); ++i) {
    ASSERT_TRUE(WriteGeneratedFile(parts[i], contents[i])) << parts[i];
  }
}

inline void SetUpBackendRuntime() {
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  ge::PlatformContext::GetInstance().Reset();
  ge::RuntimeStub::SetInstance(std::make_shared<af::RuntimeStubV2>());
}

inline void TearDownBackendRuntime() {
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  ge::RuntimeStub::Reset();
}

class BackendE2e : public testing::Test {
 protected:
  void SetUp() override {
    SetUpBackendRuntime();
  }

  void TearDown() override {
    TearDownBackendRuntime();
  }
};

class PrecisionBackendE2e : public BackendE2e {
 protected:
  void SetUp() override {
    setenv("AUTOFUSE_FLAGS", "--autofuse_enhance_precision_blacklist=all", 1);
    af::pre_process::PreProcessConfig::Instance().Reset();
    BackendE2e::SetUp();
  }

  void TearDown() override {
    unsetenv("AUTOFUSE_FLAGS");
    af::pre_process::PreProcessConfig::Instance().Reset();
    BackendE2e::TearDown();
  }
};
}  // namespace indirect_load_test

#endif  // AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_INDIRECT_LOAD_STORE_TEST_INDIRECT_LOAD_BACKEND_GENERATOR_COMMON_H_

#if defined(IL_CASE_STORE) || defined(IL_CASE_MIXED)
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
  EXPECT_NE(compute.find("return FusedBody::Output(x[input_offset], output_index, address.index_offset, context)"),
            std::string::npos);
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
#if defined(IL_STATIC_SHAPE)
  const std::string policy = "AscendC::IndirectLoadSimt" IL_STRINGIFY(IL_EXPECT_SIMT_POLICY) "Policy<uint" +
                             std::to_string(IL_EXPECT_SIMT_OFFSET_BITS) + "_t";
#else
  const std::string policy =
      "AscendC::IndirectLoadSimtStridedPolicy<uint" + std::to_string(IL_EXPECT_SIMT_OFFSET_BITS) + "_t";
#endif
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
#if defined(IL_INPUT_OUTER_STRIDE) || !defined(IL_STATIC_SHAPE)
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
#if !defined(IL_INPUT_OUTER_STRIDE) && defined(IL_STATIC_SHAPE)
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
#if IL_RANK == 4 && IL_AXIS == 2 && defined(IL_STATIC_SHAPE)
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

#endif

#if defined(IL_CASE_BROADCAST)
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <sstream>

#include "common_utils.h"

#ifndef IL_CLEAR_BROADCAST_SOURCE_VIEW
#define IL_CLEAR_BROADCAST_SOURCE_VIEW 0
#endif
#ifndef IL_INPUT_BROADCAST
#define IL_INPUT_BROADCAST 0
#endif
#ifndef IL_INDEX_BROADCAST
#define IL_INDEX_BROADCAST 0
#endif
#ifndef IL_AIC_REPRO
#define IL_AIC_REPRO 0
#endif
#ifndef IL_COMPLEX_BROADCAST
#define IL_COMPLEX_BROADCAST 0
#endif
#ifndef IL_COMPLEX_SIMT
#define IL_COMPLEX_SIMT 0
#endif
#ifndef IL_COMPLEX_INPUT_BROADCAST
#define IL_COMPLEX_INPUT_BROADCAST 0
#endif
#ifndef IL_COMPLEX_INDEX_BROADCAST
#define IL_COMPLEX_INDEX_BROADCAST 0
#endif
#ifndef IL_INDEX_BINARY_SAME_VIEW
#define IL_INDEX_BINARY_SAME_VIEW 0
#endif
#ifndef IL_INDEX_ABS_DENSE_VIEW
#define IL_INDEX_ABS_DENSE_VIEW 0
#endif
#ifndef IL_BINARY_ELEMENT_KIND
#define IL_BINARY_ELEMENT_KIND 0
#endif
#ifndef IL_RETAIN_BROADCAST
#define IL_RETAIN_BROADCAST 0
#endif
#ifndef IL_DEGENERATE_BROADCAST
#define IL_DEGENERATE_BROADCAST 0
#endif
#ifndef IL_CONTINUOUS_BROADCAST
#define IL_CONTINUOUS_BROADCAST 0
#endif
#ifndef IL_CONTINUOUS_INDEX_BROADCAST
#define IL_CONTINUOUS_INDEX_BROADCAST 0
#endif
#ifndef IL_BROADCAST_POST_REDUCE
#define IL_BROADCAST_POST_REDUCE 0
#endif
#ifndef IL_INPUT_ABS_BEFORE_BROADCAST
#define IL_INPUT_ABS_BEFORE_BROADCAST 0
#endif
#ifndef IL_OUTPUT_S0
#define IL_OUTPUT_S0 4
#endif
#ifndef IL_OUTPUT_S1
#define IL_OUTPUT_S1 5
#endif
#ifndef IL_OUTPUT_S2
#define IL_OUTPUT_S2 4
#endif
#ifndef IL_OUTPUT_S3
#define IL_OUTPUT_S3 16
#endif
namespace {
constexpr int64_t kOutputS0 = IL_OUTPUT_S0;
constexpr int64_t kOutputS1 = IL_OUTPUT_S1;
constexpr int64_t kOutputS2 = IL_OUTPUT_S2;
constexpr int64_t kOutputS3 = IL_OUTPUT_S3;
constexpr std::array<int64_t, 4> kOutputShape = {kOutputS0, kOutputS1, kOutputS2, kOutputS3};
constexpr bool kComplexBroadcast = IL_COMPLEX_BROADCAST;
constexpr bool kComplexSimt = IL_COMPLEX_SIMT;
constexpr bool kComplexInputBroadcast = IL_COMPLEX_INPUT_BROADCAST;
constexpr bool kComplexIndexBroadcast = IL_COMPLEX_INDEX_BROADCAST;
constexpr bool kIndexBinarySameView = IL_INDEX_BINARY_SAME_VIEW;
constexpr bool kIndexAbsDenseView = IL_INDEX_ABS_DENSE_VIEW;
constexpr int32_t kBinaryElementKind = IL_BINARY_ELEMENT_KIND;
constexpr bool kRetainBroadcast = IL_RETAIN_BROADCAST;
constexpr bool kDegenerateBroadcast = IL_DEGENERATE_BROADCAST;
constexpr bool kContinuousBroadcast = IL_CONTINUOUS_BROADCAST;
constexpr bool kContinuousIndexBroadcast = IL_CONTINUOUS_INDEX_BROADCAST;
constexpr bool kInputAbsBeforeBroadcast = IL_INPUT_ABS_BEFORE_BROADCAST;
constexpr int32_t kInputElementCount = IL_HAS_INPUT_ELEMENT;
constexpr int32_t kIndexElementCount = IL_HAS_INDEX_ELEMENT;
constexpr bool kHasOutputRelu = IL_HAS_OUTPUT_RELU;
constexpr bool kInputBroadcast = IL_INPUT_BROADCAST && !kComplexBroadcast && !kComplexInputBroadcast;
constexpr bool kIndexBroadcast = IL_INDEX_BROADCAST;
constexpr uint32_t kBroadcastAxesMask = IL_BROADCAST_AXES_MASK;
constexpr bool kClearBroadcastSourceView = IL_CLEAR_BROADCAST_SOURCE_VIEW;
constexpr bool kExpectSimt = IL_EXPECT_SIMT;
constexpr bool kExpectSk = IL_EXPECT_SK;
constexpr bool kAicRepro = IL_AIC_REPRO;

using indirect_load_test::SetView;

struct TensorView {
  std::vector<af::AxisId> axes;
  std::vector<af::Expression> repeats;
  std::vector<af::Expression> strides;
  af::DataType dtype;
};

struct BroadcastGraphView {
  std::shared_ptr<af::AscGraph> graph;
  TensorView input;
  TensorView output;
  TensorView input_source;
  TensorView index_source;
  TensorView input_broadcast;
  TensorView input_intermediate_broadcast;
  TensorView index_broadcast;
  TensorView index_intermediate_broadcast;
};

template <typename Op>
void SetView(Op &op, const TensorView &view) {
  SetView(op, view.axes, view.repeats, view.strides, view.dtype);
}

template <typename Op>
void ClearView(Op &op) {
  op.y.axis->clear();
  op.y.repeats->clear();
  op.y.strides->clear();
}

std::vector<af::Expression> MakeDenseStrides(const std::vector<af::Expression> &repeats) {
  std::vector<af::Expression> strides(repeats.size(), af::ops::One);
  af::Expression stride = af::ops::One;
  for (size_t index = repeats.size(); index > 0UL; --index) {
    const size_t dim = index - 1UL;
    strides[dim] = stride;
    stride = stride * repeats[dim];
  }
  return strides;
}

TensorView MakeIntermediateBroadcastView(const TensorView &broadcast_view, const TensorView &logical_view) {
  constexpr size_t kFirstBroadcastAxis = 3UL;
  TensorView view = broadcast_view;
  view.repeats[kFirstBroadcastAxis] = logical_view.repeats[kFirstBroadcastAxis];
  view.strides = MakeDenseStrides(view.repeats);
  for (size_t dim = 0UL; dim < view.strides.size(); ++dim) {
    if ((kBroadcastAxesMask & (1U << dim)) != 0U &&
        af::SymbolicUtils::StaticCheckEq(view.repeats[dim], af::ops::One) == af::TriBool::kTrue) {
      view.strides[dim] = af::ops::Zero;
    }
  }
  return view;
}

std::array<int64_t, 4> MakeLogicalStrides(bool broadcast) {
  std::array<int64_t, 4> strides{};
  int64_t stride = 1;
  for (size_t index = kOutputShape.size(); index > 0UL; --index) {
    const size_t dim = index - 1UL;
    const bool is_broadcast_axis = broadcast && (kBroadcastAxesMask & (1U << dim)) != 0U;
    strides[dim] = is_broadcast_axis ? 0 : stride;
    if (!is_broadcast_axis) {
      stride *= kOutputShape[dim];
    }
  }
  return strides;
}

std::string MakeShapeArgs(bool is_sk) {
  const auto input_strides = MakeLogicalStrides(kInputBroadcast && !kDegenerateBroadcast);
  const auto index_strides = MakeLogicalStrides(kIndexBroadcast && !kDegenerateBroadcast);
  std::ostringstream stream;
  if (is_sk) {
    stream << ", 4";
  }
  for (const int64_t size : kOutputShape) {
    stream << ", " << size;
  }
  for (const int64_t stride : input_strides) {
    stream << ", " << stride;
  }
  for (const int64_t stride : index_strides) {
    stream << ", " << stride;
  }
  stream << ");";
  return stream.str();
}

BroadcastGraphView CreateGraphView() {
  BroadcastGraphView view;
  view.graph = std::make_shared<af::AscGraph>("indirect_load_broadcast_test");
  std::vector<af::AxisId> input_source_axis_candidates;
  std::vector<af::AxisId> index_source_axis_candidates;
  for (size_t dim = 0UL; dim < kOutputShape.size(); ++dim) {
    const auto input_size = view.graph->CreateSizeVar(kOutputShape[dim]);
    const auto output_size = view.graph->CreateSizeVar(kOutputShape[dim]);
    view.input.repeats.emplace_back(input_size);
    view.output.repeats.emplace_back(output_size);
    view.input.axes.emplace_back(view.graph->CreateAxis(("z" + std::to_string(dim)).c_str(), input_size).id);
    view.output.axes.emplace_back(view.graph->CreateAxis(("z" + std::to_string(dim + 4UL)).c_str(), output_size).id);
    input_source_axis_candidates.emplace_back(
        view.graph->CreateAxis(("z" + std::to_string(dim) + "_input").c_str(), af::ops::One).id);
    index_source_axis_candidates.emplace_back(
        view.graph->CreateAxis(("z" + std::to_string(dim + 4UL) + "_index").c_str(), af::ops::One).id);
  }
  view.input.dtype = af::DT_FLOAT16;
  view.output.dtype = af::DT_INT64;
  view.input_source = view.input;
  view.index_source = view.output;
  for (size_t dim = 0UL; dim < kOutputShape.size(); ++dim) {
    if ((kBroadcastAxesMask & (1U << dim)) != 0U) {
      view.input_source.axes[dim] = input_source_axis_candidates[dim];
      view.index_source.axes[dim] = index_source_axis_candidates[dim];
      view.input_source.repeats[dim] = af::ops::One;
      view.index_source.repeats[dim] = af::ops::One;
    }
  }
  if constexpr (kIndexAbsDenseView) {
    view.index_source.axes = view.output.axes;
  }
  view.input.strides = MakeDenseStrides(view.input.repeats);
  view.output.strides = MakeDenseStrides(view.output.repeats);
  view.input_source.strides = MakeDenseStrides(view.input_source.repeats);
  view.index_source.strides = MakeDenseStrides(view.index_source.repeats);
  view.input_broadcast = view.input;
  view.index_broadcast = view.output;
  for (size_t dim = 0UL; dim < kOutputShape.size(); ++dim) {
    view.input_broadcast.strides[dim] =
        (kBroadcastAxesMask & (1U << dim)) == 0U ? view.input_source.strides[dim] : af::ops::Zero;
    view.index_broadcast.strides[dim] =
        (kBroadcastAxesMask & (1U << dim)) == 0U ? view.index_source.strides[dim] : af::ops::Zero;
  }
  view.input_intermediate_broadcast = view.input_broadcast;
  if constexpr (kContinuousBroadcast) {
    view.input_intermediate_broadcast = MakeIntermediateBroadcastView(view.input_broadcast, view.input);
  }
  view.index_intermediate_broadcast = view.index_broadcast;
  if constexpr (kContinuousIndexBroadcast) {
    view.index_intermediate_broadcast = MakeIntermediateBroadcastView(view.index_broadcast, view.output);
  }
  return view;
}

template <typename Destination>
void ConnectAbsChain(const std::shared_ptr<af::AscGraph> &graph, const char *prefix, int32_t count,
                     const af::AscOpOutput &source, const TensorView &view, Destination &destination) {
  std::vector<std::unique_ptr<af::ascir_op::Abs>> elements;
  for (int32_t i = 0; i < count; ++i) {
    const auto name = std::string(prefix) + std::to_string(i);
    auto element = std::make_unique<af::ascir_op::Abs>(name.c_str());
    graph->AddNode(*element);
    element->attr.api.compute_type = af::ComputeType::kComputeElewise;
    element->x = elements.empty() ? source : elements.back()->y;
    SetView(*element, view);
    elements.emplace_back(std::move(element));
  }
  destination = elements.empty() ? source : elements.back()->y;
}

af::AscOpOutput ConnectBinaryElement(const std::shared_ptr<af::AscGraph> &graph, const char *name,
                                     const af::AscOpOutput &lhs, const af::AscOpOutput &rhs, const TensorView &view) {
  if constexpr (kBinaryElementKind == 1) {
    af::ascir_op::Mul element(name);
    graph->AddNode(element);
    element.x1 = lhs;
    element.x2 = rhs;
    element.attr.api.compute_type = af::ComputeType::kComputeElewise;
    SetView(element, view);
    return element.y;
  } else if constexpr (kBinaryElementKind == 2) {
    af::ascir_op::Sub element(name);
    graph->AddNode(element);
    element.x1 = lhs;
    element.x2 = rhs;
    element.attr.api.compute_type = af::ComputeType::kComputeElewise;
    SetView(element, view);
    return element.y;
  } else if constexpr (kBinaryElementKind == 3) {
    af::ascir_op::Maximum element(name);
    graph->AddNode(element);
    element.x1 = lhs;
    element.x2 = rhs;
    element.attr.api.compute_type = af::ComputeType::kComputeElewise;
    SetView(element, view);
    return element.y;
  } else {
    af::ascir_op::Add element(name);
    graph->AddNode(element);
    element.x1 = lhs;
    element.x2 = rhs;
    element.attr.api.compute_type = af::ComputeType::kComputeElewise;
    SetView(element, view);
    return element.y;
  }
}

void BuildInputPath(const BroadcastGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  if constexpr (kComplexSimt) {
    af::ascir_op::Data x("x");
    af::ascir_op::Load input_load("input_load");
    view.graph->AddNode(x);
    view.graph->AddNode(input_load);
    x.ir_attr.SetIndex(0);
    input_load.x = x.y;
    SetView(x, view.input);
    SetView(input_load, view.input);
    indirect_load.x1 = input_load.y;
    return;
  }
  constexpr bool use_broadcast = kInputBroadcast || kComplexBroadcast;
  af::ascir_op::Data x("x");
  view.graph->AddNode(x);
  x.ir_attr.SetIndex(0);
  SetView(x, use_broadcast ? view.input_source : view.input);
  if (kClearBroadcastSourceView && kInputBroadcast) {
    ClearView(x);
  }
  af::ascir_op::Load input_load("input_load");
  view.graph->AddNode(input_load);
  input_load.x = x.y;
  SetView(input_load, use_broadcast ? view.input_source : view.input);
  if (use_broadcast) {
    if constexpr (kContinuousBroadcast) {
      af::ascir_op::Broadcast first_broadcast("input_first_broadcast");
      af::ascir_op::Broadcast second_broadcast("input_second_broadcast");
      view.graph->AddNode(first_broadcast);
      view.graph->AddNode(second_broadcast);
      first_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
      second_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
      first_broadcast.x = input_load.y;
      second_broadcast.x = first_broadcast.y;
      SetView(first_broadcast, view.input_intermediate_broadcast);
      SetView(second_broadcast, view.input_broadcast);
      ConnectAbsChain(view.graph, "input_abs_", kInputElementCount, second_broadcast.y, view.input_broadcast,
                      indirect_load.x1);
      return;
    }
    af::ascir_op::Broadcast broadcast("input_broadcast");
    view.graph->AddNode(broadcast);
    broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
    if constexpr (kInputAbsBeforeBroadcast) {
      af::ascir_op::Abs input_abs_before_broadcast("input_abs_before_broadcast");
      view.graph->AddNode(input_abs_before_broadcast);
      input_abs_before_broadcast.attr.api.compute_type = af::ComputeType::kComputeElewise;
      input_abs_before_broadcast.x = input_load.y;
      SetView(input_abs_before_broadcast, view.input_source);
      broadcast.x = input_abs_before_broadcast.y;
    } else {
      broadcast.x = input_load.y;
    }
    SetView(broadcast, view.input_broadcast);
    if constexpr (kComplexBroadcast) {
      af::ascir_op::Add input_add("input_add");
      view.graph->AddNode(input_add);
      input_add.x1 = broadcast.y;
      input_add.x2 = input_load.y;
      SetView(input_add, view.input_broadcast);
      indirect_load.x1 = input_add.y;
    } else if constexpr (kInputAbsBeforeBroadcast) {
      indirect_load.x1 = broadcast.y;
    } else {
      ConnectAbsChain(view.graph, "input_abs_", kInputElementCount, broadcast.y, view.input_broadcast,
                      indirect_load.x1);
    }
  } else {
    indirect_load.x1 = input_load.y;
  }
}

void BuildIndexPath(const BroadcastGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Data index("index");
  view.graph->AddNode(index);
  index.ir_attr.SetIndex(1);
  SetView(index, kIndexBroadcast ? view.index_source : view.output);
  if (kClearBroadcastSourceView && kIndexBroadcast) {
    ClearView(index);
  }
  af::ascir_op::Load index_load("index_load");
  view.graph->AddNode(index_load);
  index_load.x = index.y;
  SetView(index_load, kIndexBroadcast ? view.index_source : view.output);
  if (kIndexBroadcast) {
    if constexpr (kContinuousIndexBroadcast) {
      af::ascir_op::Broadcast first_broadcast("index_first_broadcast");
      af::ascir_op::Broadcast second_broadcast("index_second_broadcast");
      view.graph->AddNode(first_broadcast);
      view.graph->AddNode(second_broadcast);
      first_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
      second_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
      first_broadcast.x = index_load.y;
      second_broadcast.x = first_broadcast.y;
      SetView(first_broadcast, view.index_intermediate_broadcast);
      SetView(second_broadcast, view.index_broadcast);
      ConnectAbsChain(view.graph, "index_abs_", kIndexElementCount, second_broadcast.y, view.index_broadcast,
                      indirect_load.x2);
      return;
    }
    af::ascir_op::Broadcast broadcast("index_broadcast");
    view.graph->AddNode(broadcast);
    broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
    broadcast.x = index_load.y;
    SetView(broadcast, kIndexAbsDenseView ? view.output : view.index_broadcast);
    const TensorView &index_element_view = kIndexAbsDenseView ? view.output : view.index_broadcast;
    ConnectAbsChain(view.graph, "index_abs_", kIndexElementCount, broadcast.y, index_element_view, indirect_load.x2);
  } else {
    ConnectAbsChain(view.graph, "index_abs_", kIndexElementCount, index_load.y, view.output, indirect_load.x2);
  }
}

void BuildOutputPath(const BroadcastGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
#if IL_BROADCAST_POST_REDUCE
  af::ascir_op::Sum sum("output_sum");
  af::ascir_op::Store store("store");
  af::ascir_op::Output output("y");
  view.graph->AddNode(sum);
  view.graph->AddNode(store);
  view.graph->AddNode(output);
  indirect_load.ir_attr.SetAxis(2);
  SetView(indirect_load, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.attr.sched.axis = view.output.axes;
  sum.x = indirect_load.y;
  auto reduce_repeats = view.output.repeats;
  auto reduce_strides = view.output.strides;
  reduce_repeats[2] = af::ops::One;
  reduce_repeats[3] = af::ops::One;
  reduce_strides[0] = view.output.repeats[1];
  reduce_strides[1] = af::ops::One;
  reduce_strides[2] = af::ops::Zero;
  reduce_strides[3] = af::ops::Zero;
  SetView(sum, view.output.axes, reduce_repeats, reduce_strides, af::DT_FLOAT16);
  store.x = sum.y;
  SetView(store, view.output.axes, reduce_repeats, reduce_strides, af::DT_FLOAT16);
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetView(output, view.output.axes, reduce_repeats, reduce_strides, af::DT_FLOAT16);
  return;
#endif
  if constexpr (kRetainBroadcast) {
    TensorView source_view = view.output;
    source_view.dtype = af::DT_FLOAT16;
    for (size_t dim = 0UL; dim < source_view.repeats.size(); ++dim) {
      if ((kBroadcastAxesMask & (1U << dim)) != 0U) {
        source_view.repeats[dim] = af::ops::One;
      }
    }
    source_view.strides = MakeDenseStrides(source_view.repeats);
    af::ascir_op::Scalar source("output_source", *view.graph);
    source.ir_attr.SetValue("1.5");
    source.y.dtype = af::DT_FLOAT16;
    af::ascir_op::Abs source_abs("output_source_abs");
    af::ascir_op::Broadcast broadcast("output_retained_broadcast");
    af::ascir_op::Add output_add("output_add");
    af::ascir_op::Store store("store");
    af::ascir_op::Output output("y");
    view.graph->AddNode(source_abs);
    view.graph->AddNode(broadcast);
    view.graph->AddNode(output_add);
    view.graph->AddNode(store);
    view.graph->AddNode(output);
    source_abs.x = source.y;
    broadcast.x = source_abs.y;
    broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
    output_add.x1 = indirect_load.y;
    output_add.x2 = broadcast.y;
    indirect_load.ir_attr.SetAxis(2);
    SetView(indirect_load, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    SetView(source_abs, source_view);
    SetView(broadcast, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    SetView(output_add, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    store.x = output_add.y;
    SetView(store, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    output.x = store.y;
    output.ir_attr.SetIndex(0);
    SetView(output, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    return;
  }
  if constexpr (kComplexBroadcast) {
    af::ascir_op::Scalar scalar0("output_scalar0", *view.graph);
    af::ascir_op::Scalar scalar1("output_scalar1", *view.graph);
    scalar0.ir_attr.SetValue("0.5");
    scalar1.ir_attr.SetValue("1.0");
    scalar0.y.dtype = af::DT_FLOAT16;
    scalar1.y.dtype = af::DT_FLOAT16;
    af::ascir_op::Broadcast broadcast0("output_broadcast0");
    af::ascir_op::Broadcast broadcast1("output_broadcast1");
    af::ascir_op::Add scalar_add("output_scalar_add");
    af::ascir_op::Add output_add("output_add");
    af::ascir_op::Store store("store");
    af::ascir_op::Output output("y");
    for (af::ascir_op::Broadcast *broadcast : {&broadcast0, &broadcast1}) {
      view.graph->AddNode(*broadcast);
      broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
      SetView(*broadcast, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    }
    view.graph->AddNode(scalar_add);
    view.graph->AddNode(output_add);
    view.graph->AddNode(store);
    view.graph->AddNode(output);
    broadcast0.x = scalar0.y;
    broadcast1.x = scalar1.y;
    scalar_add.x1 = broadcast0.y;
    scalar_add.x2 = broadcast1.y;
    output_add.x1 = indirect_load.y;
    output_add.x2 = scalar_add.y;
    indirect_load.ir_attr.SetAxis(2);
    SetView(indirect_load, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    SetView(scalar_add, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    SetView(output_add, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    store.x = output_add.y;
    SetView(store, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    output.x = store.y;
    output.ir_attr.SetIndex(0);
    SetView(output, view.output.axes, view.output.repeats, view.output.strides, af::DT_FLOAT16);
    return;
  }
  indirect_load_test::BuildOutputPath(view.graph, indirect_load, view.output.axes, view.output.repeats,
                                      view.output.strides, kHasOutputRelu);
}

void BuildComplexIndexPath(const BroadcastGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Data index("index");
  af::ascir_op::Load index_load("index_load");
  view.graph->AddNode(index);
  view.graph->AddNode(index_load);
  index.ir_attr.SetIndex(1);
  index_load.x = index.y;
  SetView(index, view.index_source);
  SetView(index_load, view.index_source);

  af::ascir_op::Scalar scalar0("index_scalar0", *view.graph);
  af::ascir_op::Scalar scalar1("index_scalar1", *view.graph);
  scalar0.ir_attr.SetValue("0");
  scalar1.ir_attr.SetValue("0");
  scalar0.y.dtype = af::DT_INT64;
  scalar1.y.dtype = af::DT_INT64;
  af::ascir_op::Broadcast broadcast0("index_broadcast0");
  af::ascir_op::Broadcast broadcast1("index_broadcast1");
  af::ascir_op::Add scalar_add("index_scalar_add");
  af::ascir_op::Broadcast final_broadcast("index_final_broadcast");
  for (af::ascir_op::Broadcast *broadcast : {&broadcast0, &broadcast1}) {
    view.graph->AddNode(*broadcast);
    broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
    SetView(*broadcast, view.index_source);
  }
  view.graph->AddNode(scalar_add);
  view.graph->AddNode(final_broadcast);
  final_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast0.x = scalar0.y;
  broadcast1.x = scalar1.y;
  scalar_add.x1 = broadcast0.y;
  scalar_add.x2 = broadcast1.y;
  const auto index_add = ConnectBinaryElement(view.graph, "index_add", index_load.y, scalar_add.y, view.index_source);
  final_broadcast.x = index_add;
  SetView(scalar_add, view.index_source);
  SetView(final_broadcast, view.index_broadcast);
  indirect_load.x2 = final_broadcast.y;
}

void BuildSameViewIndexBinaryPath(const BroadcastGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Data index("same_view_index");
  af::ascir_op::Load index_load("same_view_index_load");
  af::ascir_op::Abs index_abs("same_view_index_abs");
  af::ascir_op::Maximum index_maximum("same_view_index_maximum");
  view.graph->AddNode(index);
  view.graph->AddNode(index_load);
  view.graph->AddNode(index_abs);
  view.graph->AddNode(index_maximum);
  index.ir_attr.SetIndex(1);
  index_load.x = index.y;
  index_abs.x = index_load.y;
  index_maximum.x1 = index_load.y;
  index_maximum.x2 = index_abs.y;
  index_abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  index_maximum.attr.api.compute_type = af::ComputeType::kComputeElewise;
  SetView(index, view.output);
  SetView(index_load, view.output);
  SetView(index_abs, view.output);
  SetView(index_maximum, view.output);
  indirect_load.x2 = index_maximum.y;
}

void BuildComplexInputPath(const BroadcastGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Data x("x");
  af::ascir_op::Load input_load("input_load");
  view.graph->AddNode(x);
  view.graph->AddNode(input_load);
  x.ir_attr.SetIndex(0);
  input_load.x = x.y;
  SetView(x, view.input_source);
  SetView(input_load, view.input_source);

  af::ascir_op::Scalar scalar0("input_scalar0", *view.graph);
  af::ascir_op::Scalar scalar1("input_scalar1", *view.graph);
  scalar0.ir_attr.SetValue("0.0");
  scalar1.ir_attr.SetValue("0.0");
  scalar0.y.dtype = af::DT_FLOAT16;
  scalar1.y.dtype = af::DT_FLOAT16;
  af::ascir_op::Broadcast broadcast0("input_scalar_broadcast0");
  af::ascir_op::Broadcast broadcast1("input_scalar_broadcast1");
  af::ascir_op::Add scalar_add("input_scalar_add");
  af::ascir_op::Broadcast final_broadcast("input_final_broadcast");
  for (af::ascir_op::Broadcast *broadcast : {&broadcast0, &broadcast1}) {
    view.graph->AddNode(*broadcast);
    broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
    SetView(*broadcast, view.input_source);
  }
  view.graph->AddNode(scalar_add);
  view.graph->AddNode(final_broadcast);
  final_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast0.x = scalar0.y;
  broadcast1.x = scalar1.y;
  scalar_add.x1 = broadcast0.y;
  scalar_add.x2 = broadcast1.y;
  const auto input_add =
      ConnectBinaryElement(view.graph, "input_source_add", input_load.y, scalar_add.y, view.input_source);
  final_broadcast.x = input_add;
  SetView(scalar_add, view.input_source);
  SetView(final_broadcast, view.input_broadcast);
  indirect_load.x1 = final_broadcast.y;
}

af::ComputeGraphPtr CreateGraph() {
  indirect_load_test::BackendGraph backend("indirect_load_broadcast_test", "data0", "data1", af::DT_FLOAT16);
  const auto build_index = kComplexBroadcast || kComplexIndexBroadcast ? BuildComplexIndexPath
                           : kIndexBinarySameView                      ? BuildSameViewIndexBinaryPath
                                                                       : BuildIndexPath;
  const auto build_input = kComplexInputBroadcast ? BuildComplexInputPath : BuildInputPath;
  return backend.Finalize(
      indirect_load_test::CreateSubGraph(CreateGraphView(), build_input, build_index, BuildOutputPath), "output");
}

af::ComputeGraphPtr CreateAicReproGraph() {
  auto graph = std::make_shared<af::AscGraph>("indirect_load_aic_repro");
  const af::Expression input_s0 = graph->CreateSizeVar(100000);
  const af::Expression output_s0 = graph->CreateSizeVar(1024);
  const af::Expression s1 = graph->CreateSizeVar(1024);
  const af::Expression one = af::ops::One;
  const af::AxisId input_axis0 = graph->CreateAxis("x0", input_s0).id;
  const af::AxisId input_axis1 = graph->CreateAxis("x1", s1).id;
  const af::AxisId output_axis0 = graph->CreateAxis("y0", output_s0).id;
  const af::AxisId output_axis1 = graph->CreateAxis("y1", s1).id;

  af::ascir_op::Data input("input");
  af::ascir_op::Load input_load("input_load");
  graph->AddNode(input);
  graph->AddNode(input_load);
  input.ir_attr.SetIndex(0);
  input_load.x = input.y;
  SetView(input, {input_axis0, input_axis1}, {input_s0, s1}, {s1, one}, af::DT_FLOAT);
  SetView(input_load, {input_axis0, input_axis1}, {input_s0, s1}, {s1, one}, af::DT_FLOAT);

  af::ascir_op::Data index("index");
  af::ascir_op::Load index_load("index_load");
  af::ascir_op::Broadcast index_broadcast("index_broadcast");
  graph->AddNode(index);
  graph->AddNode(index_load);
  graph->AddNode(index_broadcast);
  index.ir_attr.SetIndex(1);
  index_load.x = index.y;
  index_broadcast.x = index_load.y;
  index_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  SetView(index, {output_axis0, output_axis1}, {output_s0, one}, {one, af::ops::Zero}, af::DT_INT64);
  SetView(index_load, {output_axis0, output_axis1}, {output_s0, one}, {one, af::ops::Zero}, af::DT_INT64);
  SetView(index_broadcast, {output_axis0, output_axis1}, {output_s0, s1}, {s1, one}, af::DT_INT64);

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  af::ascir_op::Store store("store");
  af::ascir_op::Output output("output");
  graph->AddNode(indirect_load);
  graph->AddNode(store);
  graph->AddNode(output);
  indirect_load.x1 = input_load.y;
  indirect_load.x2 = index_broadcast.y;
  indirect_load.ir_attr.SetAxis(0);
  SetView(indirect_load, {output_axis0, output_axis1}, {output_s0, s1}, {s1, one}, af::DT_FLOAT);
  store.x = indirect_load.y;
  SetView(store, {output_axis0, output_axis1}, {output_s0, s1}, {s1, one}, af::DT_FLOAT);
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetView(output, {output_axis0, output_axis1}, {output_s0, s1}, {s1, one}, af::DT_FLOAT);

  indirect_load_test::BackendGraph backend("indirect_load_aic_repro", "data0", "data1", af::DT_FLOAT);
  return backend.Finalize(graph, "output");
}

void CheckSkKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find("// IndirectLoad SK"), std::string::npos);
  EXPECT_EQ(kernel.find("// IndirectLoad SIMD"), std::string::npos);
  EXPECT_EQ(kernel.find("// IndirectLoad SIMT"), std::string::npos);
  EXPECT_EQ(kernel.find("BroadcastExtend<"), std::string::npos);
  EXPECT_NE(kernel.find(MakeShapeArgs(true)), std::string::npos);
}

void CheckSimtKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find("// IndirectLoad SIMT"), std::string::npos);
  EXPECT_NE(kernel.find("IndirectLoadSimt<"), std::string::npos);
  EXPECT_EQ(kernel.find("// IndirectLoad SIMD"), std::string::npos);
  if constexpr (kOutputS0 == 4 && kOutputS1 == 5 && kOutputS2 == 4 && kOutputS3 == 16) {
    EXPECT_NE(kernel.find(MakeShapeArgs(false)), std::string::npos);
  }
  EXPECT_EQ(kernel.find("AscendC::BroadcastExtend<"), std::string::npos);
}

void CheckSimdElements(const std::string &kernel) {
  const auto input_abs_pos = kernel.find("Abs(");
  const auto indirect_load_pos = kernel.find("// IndirectLoad SIMD");
  ASSERT_NE(indirect_load_pos, std::string::npos);
  if (kInputAbsBeforeBroadcast || kInputElementCount > 0 || kIndexElementCount > 0) {
    ASSERT_NE(input_abs_pos, std::string::npos);
    EXPECT_LT(input_abs_pos, indirect_load_pos);
  } else {
    EXPECT_EQ(input_abs_pos, std::string::npos);
  }
  const auto output_relu_pos = kernel.find("Relu(");
  EXPECT_EQ(output_relu_pos == std::string::npos, !kHasOutputRelu);
}

void CheckSimdKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find("// IndirectLoad SIMD"), std::string::npos);
  EXPECT_NE(kernel.find("IndirectLoadSimd<"), std::string::npos);
  if constexpr (kDegenerateBroadcast) {
    EXPECT_NE(kernel.find(", 20, 10, 10, 20, 20, 4000, 400, 20, 1);"), std::string::npos);
    EXPECT_NE(kernel.find("Duplicate(local_7[0], local_6.GetValue(0), local_7_actual_size);"), std::string::npos);
    EXPECT_NE(kernel.find("Duplicate(local_9[0], local_8.GetValue(0), local_9_actual_size);"), std::string::npos);
    EXPECT_NE(kernel.find("const uint32_t local_7_actual_size = (400 - 1) + 1;"), std::string::npos);
    EXPECT_NE(kernel.find("const uint32_t local_9_actual_size = (400 - 1) + 1;"), std::string::npos);
  } else {
#if IL_BROADCAST_POST_REDUCE
    EXPECT_NE(kernel.find("ReduceSum"), std::string::npos);
#else
    if constexpr (!(kIndexBroadcast && kBroadcastAxesMask == 0U) && !kIndexBinarySameView) {
      EXPECT_NE(kernel.find(MakeShapeArgs(false)), std::string::npos);
    }
#endif
  }
  if constexpr (!kDegenerateBroadcast && !IL_BROADCAST_POST_REDUCE) {
    EXPECT_NE(kernel.find("const int64_t indirect_load_outert_axis_size = 1;"), std::string::npos);
    EXPECT_NE(kernel.find("block_dim_offset = indirect_load_outerTB * t->indirect_load_outerTb_size"),
              std::string::npos);
  }
  if (!kIndexBroadcast && kBroadcastAxesMask == 2U) {
    EXPECT_NE(kernel.find("global_0[(int64_t)z4 * (int64_t)64 + 0 + 0]"), std::string::npos);
    EXPECT_EQ(kernel.find("global_0[(int64_t)z4 * (int64_t)64 + (int64_t)z5 * (int64_t)64"), std::string::npos);
  }
  EXPECT_EQ(kernel.find("AscendC::BroadcastExtend<"), std::string::npos);
  CheckSimdElements(kernel);
}

void CheckGeneratedKernel(const std::string &kernel, ascir::TemplateId template_id) {
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    CheckSkKernel(kernel);
  } else if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    CheckSimtKernel(kernel);
  } else {
    CheckSimdKernel(kernel);
  }
}

const af::Axis *FindDerivedAxis(const af::AscGraph &graph, af::Axis::Type type, af::AxisId from) {
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis->type == type && axis->from == std::vector<af::AxisId>{from}) {
      return axis.get();
    }
  }
  return nullptr;
}

void ExpectComplexNodeSchedule(const af::AscNodePtr &node, const std::vector<af::AxisId> &axes,
                               af::AxisId vectorized_axis) {
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->attr.sched.axis, axes) << node->GetName();
  ASSERT_GE(axes.size(), 2UL);
  EXPECT_EQ(node->attr.sched.loop_axis, axes[axes.size() - 2UL]) << node->GetName();
  ASSERT_FALSE(node->outputs().empty());
  EXPECT_EQ(node->outputs()[0]->attr.vectorized_axis, std::vector<af::AxisId>{vectorized_axis}) << node->GetName();
}

void ExpectComplexOuterAxes(af::AscGraph &graph, const ascgen_utils::indirect_load::TemplateAxes &axes,
                            const ascgen_utils::indirect_load::TemplateLogicalView &view,
                            std::vector<af::AxisId> &outer_loops) {
  const af::Axis *outer = graph.FindAxis(axes.outer_axis);
  const af::Axis *inner = graph.FindAxis(axes.inner_axis);
  const af::Axis *input_inner = graph.FindAxis(axes.input_inner_axis);
  const af::Axis *index_inner = graph.FindAxis(axes.index_inner_axis);
  ASSERT_NE(outer, nullptr);
  ASSERT_NE(inner, nullptr);
  ASSERT_NE(input_inner, nullptr);
  ASSERT_NE(index_inner, nullptr);
  EXPECT_EQ(outer->from, std::vector<af::AxisId>({view.output.axis_ids[0], view.output.axis_ids[1]}));
  EXPECT_EQ(inner->from, std::vector<af::AxisId>({view.output.axis_ids[2], view.output.axis_ids[3]}));
  EXPECT_EQ(input_inner->from, std::vector<af::AxisId>({view.input.axis_ids[2], view.input.axis_ids[3]}));
  EXPECT_EQ(index_inner->from, std::vector<af::AxisId>({view.output.axis_ids[2], view.output.axis_ids[3]}));
  const af::Axis *tile_outer = FindDerivedAxis(graph, af::Axis::Type::kAxisTypeTileOuter, outer->id);
  const af::Axis *tile_inner = FindDerivedAxis(graph, af::Axis::Type::kAxisTypeTileInner, outer->id);
  ASSERT_NE(tile_outer, nullptr);
  ASSERT_NE(tile_inner, nullptr);
  const af::Axis *block_outer = FindDerivedAxis(graph, af::Axis::Type::kAxisTypeBlockOuter, tile_outer->id);
  const af::Axis *block_inner = FindDerivedAxis(graph, af::Axis::Type::kAxisTypeBlockInner, tile_outer->id);
  ASSERT_NE(block_outer, nullptr);
  ASSERT_NE(block_inner, nullptr);
  outer_loops = {block_outer->id, block_inner->id, tile_inner->id};
}

void ExpectComplexLogicalView(const ascgen_utils::indirect_load::TemplateLogicalView &view) {
  ASSERT_EQ(view.output.axis_ids.size(), 4UL);
  ASSERT_EQ(view.input.axis_ids.size(), 4UL);
  EXPECT_EQ(view.input.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kZeroStrideCompact);
  EXPECT_EQ(view.index.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kZeroStrideCompact);
  ASSERT_EQ(view.input.physical_repeats.size(), 4UL);
  ASSERT_EQ(view.index.physical_repeats.size(), 4UL);
  EXPECT_TRUE(ascgen_utils::ExpressEq(view.input.physical_repeats[0], af::ops::One));
  EXPECT_TRUE(ascgen_utils::ExpressEq(view.input.physical_repeats[1], af::ops::One));
  EXPECT_TRUE(ascgen_utils::ExpressEq(view.index.physical_repeats[0], af::ops::One));
  EXPECT_TRUE(ascgen_utils::ExpressEq(view.index.physical_repeats[1], af::ops::One));
}

void ExpectComplexBrcRewrite(af::AscGraph &graph, const af::AscNodePtr &input_add, const af::AscNodePtr &index_vf,
                             const af::AscNodePtr &post_vf) {
  ASSERT_TRUE(af::ops::IsOps<af::ascir_op::Add>(input_add));
  ASSERT_TRUE(af::ops::IsOps<af::ascir_op::VectorFunc>(index_vf));
  ASSERT_TRUE(af::ops::IsOps<af::ascir_op::VectorFunc>(post_vf));
  EXPECT_EQ(index_vf->inputs.Size(), 3UL);
  EXPECT_EQ(post_vf->inputs.Size(), 3UL);
  EXPECT_EQ(graph.FindNode("input_broadcast"), nullptr);
  ASSERT_EQ(input_add->inputs.Size(), 2UL);
  for (size_t i = 0UL; i < input_add->inputs.Size(); ++i) {
    const af::AscNodePtr producer = ascgen_utils::indirect_load::GetInputProducer(input_add, i);
    ASSERT_NE(producer, nullptr);
    EXPECT_EQ(producer->GetName(), "input_load") << "input index=" << i;
  }
  for (const char *name : {"index_final_broadcast", "index_broadcast0", "index_broadcast1", "index_scalar_add",
                           "output_broadcast0", "output_broadcast1", "output_scalar_add"}) {
    EXPECT_EQ(graph.FindNode(name), nullptr) << name;
  }
}

void ExpectComplexSimdSchedule(af::AscGraph &graph, const af::AscNodePtr &indirect_load) {
  ascgen_utils::indirect_load::TemplateAxes axes;
  ascgen_utils::indirect_load::TemplateLogicalView view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, view), af::SUCCESS);
  ExpectComplexLogicalView(view);
  std::vector<af::AxisId> outer_loops;
  ExpectComplexOuterAxes(graph, axes, view, outer_loops);
  ASSERT_EQ(outer_loops.size(), 3UL);
  const std::vector<af::AxisId> index_axes = {outer_loops[0], outer_loops[1], outer_loops[2], axes.index_inner_axis};
  const std::vector<af::AxisId> input_axes = {outer_loops[0], outer_loops[1], outer_loops[2], axes.input_inner_axis};
  const std::vector<af::AxisId> output_axes = {outer_loops[0], outer_loops[1], outer_loops[2], axes.inner_axis};
  const af::AscNodePtr input_add = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  const af::AscNodePtr index_vf = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 1UL);
  const af::AscNodePtr post_vf = ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load);
  ExpectComplexBrcRewrite(graph, input_add, index_vf, post_vf);
  ExpectComplexNodeSchedule(graph.FindNode("index_load"), index_axes, axes.index_inner_axis);
  ExpectComplexNodeSchedule(index_vf, index_axes, axes.index_inner_axis);
  ExpectComplexNodeSchedule(graph.FindNode("input_load"), input_axes, axes.input_inner_axis);
  ExpectComplexNodeSchedule(input_add, input_axes, axes.input_inner_axis);
  for (const af::AscNodePtr &node : {indirect_load, post_vf, graph.FindNode("store")}) {
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->attr.sched.axis, output_axes) << node->GetName();
    EXPECT_EQ(node->attr.sched.loop_axis, outer_loops.back()) << node->GetName();
    ASSERT_FALSE(node->outputs().empty());
    EXPECT_EQ(node->outputs()[0]->attr.vectorized_axis, axes.vectorized_axes) << node->GetName();
  }
  std::vector<af::AscNodePtr> nodes;
  for (const af::AscNodePtr &node : graph.GetAllNodes()) {
    nodes.emplace_back(node);
  }
  const auto pos = [&](const af::AscNodePtr &node) { return std::find(nodes.begin(), nodes.end(), node); };
  EXPECT_LT(pos(graph.FindNode("index_load")), pos(index_vf));
  EXPECT_LT(pos(graph.FindNode("input_load")), pos(input_add));
  EXPECT_LT(pos(input_add), pos(indirect_load));
  EXPECT_LT(pos(indirect_load), pos(post_vf));
  EXPECT_LT(pos(post_vf), pos(graph.FindNode("store")));
}

}  // namespace

using TestBackendIndirectLoadBroadcastE2e = indirect_load_test::PrecisionBackendE2e;

TEST_F(TestBackendIndirectLoadBroadcastE2e, IndirectLoadBroadcastCodegen) {
  const auto graph = kAicRepro ? CreateAicReproGraph() : CreateGraph();
  ASSERT_NE(graph, nullptr);
  if (kAicRepro) {
    ascir::FusedScheduledResult scheduled_result;
    optimize::Optimizer optimizer(optimize::OptimizerOptions{.graph_type = optimize::GraphType::kFusedAscBackend});
    ASSERT_EQ(optimizer.Optimize(graph, scheduled_result), af::SUCCESS);
    ASSERT_TRUE(indirect_load_test::HasTemplate(scheduled_result, ascir::TemplateId::kIndirectLoadSimt));
    codegen::Codegen codegen(codegen::CodegenOptions{});
    codegen::CodegenResult result;
    ASSERT_EQ(codegen.Generate({}, scheduled_result, result), af::SUCCESS);
    EXPECT_NE(result.kernel.find("// IndirectLoad SIMT"), std::string::npos);
    EXPECT_NE(result.kernel.find("IndirectLoadSimtStridedPolicy<uint32_t, 2, 0, 3ULL, 1ULL>"), std::string::npos);
    EXPECT_NE(result.kernel.find(", 1024, 1024, 1024, 1, 1, 0);"), std::string::npos);
    indirect_load_test::WriteGeneratedFiles(result);
    return;
  }
  const std::map<std::string, std::string> shape_info = {{"s0", "stub_s0"}, {"s1", "stub_s1"}, {"s2", "stub_s2"},
                                                         {"s3", "stub_s3"}, {"s4", "stub_s4"}, {"s5", "stub_s5"},
                                                         {"s6", "stub_s6"}, {"s7", "stub_s7"}};
  const auto expected_template = indirect_load_test::GetExpectedTemplate(kExpectSimt, kExpectSk);
  if constexpr (kRetainBroadcast) {
    ascir::FusedScheduledResult scheduled_result;
    ASSERT_TRUE(indirect_load_test::SelectTemplate(graph, expected_template, scheduled_result));
    codegen::Codegen codegen(codegen::CodegenOptions{});
    codegen::CodegenResult result;
    ASSERT_EQ(codegen.Generate(shape_info, scheduled_result, result), af::SUCCESS);
    EXPECT_NE(result.kernel.find(kExpectSimt ? "IndirectLoadSimt<" : "IndirectLoadSimd<"), std::string::npos);
    if constexpr (kExpectSimt) {
      EXPECT_EQ(result.kernel.find("BroadcastExtend<"), std::string::npos);
    } else {
      EXPECT_NE(result.kernel.find("BroadcastExtend<"), std::string::npos);
    }
    indirect_load_test::WriteGeneratedFiles(result);
    return;
  }
  if constexpr (kComplexBroadcast) {
    ascir::FusedScheduledResult scheduled_result;
    ASSERT_TRUE(indirect_load_test::SelectTemplate(graph, expected_template, scheduled_result));
    size_t template_graph_count = 0UL;
    for (auto &candidates : scheduled_result.node_idx_to_scheduled_results) {
      for (auto &candidate : candidates) {
        for (auto &group : candidate.schedule_groups) {
          for (auto &impl_graph : group.impl_graphs) {
            const af::AscNodePtr indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(impl_graph);
            if (indirect_load != nullptr && ascir::GetTemplateIdOrDefault(*indirect_load) == expected_template) {
              ++template_graph_count;
              if (expected_template == ascir::TemplateId::kIndirectLoadSimd) {
                ExpectComplexSimdSchedule(impl_graph, indirect_load);
              }
            }
          }
        }
      }
    }
    EXPECT_GT(template_graph_count, 0UL);
    codegen::Codegen codegen(codegen::CodegenOptions{});
    codegen::CodegenResult result;
    ASSERT_EQ(codegen.Generate(shape_info, scheduled_result, result), af::SUCCESS);
    EXPECT_NE(result.kernel.find(expected_template == ascir::TemplateId::kIndirectLoadSimd ? "IndirectLoadSimd<"
                                                                                           : "IndirectLoadSimt<"),
              std::string::npos);
    const bool has_binary_element =
        result.kernel.find("Add(") != std::string::npos || result.kernel.find("Mul(") != std::string::npos ||
        result.kernel.find("Sub(") != std::string::npos || result.kernel.find("Maximum(") != std::string::npos;
    EXPECT_TRUE(has_binary_element);
    EXPECT_EQ(result.kernel.find("BroadcastExtend<"), std::string::npos);
    indirect_load_test::WriteGeneratedFiles(result);
    return;
  }
  if constexpr (kComplexInputBroadcast) {
    ascir::FusedScheduledResult scheduled_result;
    ASSERT_TRUE(indirect_load_test::SelectTemplate(graph, expected_template, scheduled_result));
    codegen::Codegen codegen(codegen::CodegenOptions{});
    codegen::CodegenResult result;
    ASSERT_EQ(codegen.Generate(shape_info, scheduled_result, result), af::SUCCESS);
    EXPECT_NE(result.kernel.find(indirect_load_test::GetTemplateMarker(expected_template)), std::string::npos);
    indirect_load_test::WriteGeneratedFiles(result);
    return;
  }
  codegen::CodegenResult result;
  indirect_load_test::GenerateForTemplate(graph, shape_info, expected_template, result);
  CheckGeneratedKernel(result.kernel, expected_template);
  indirect_load_test::WriteGeneratedFiles(result);
}

#endif

#if defined(IL_CASE_BROADCAST_WHERE)
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

namespace {
using indirect_load_test::SetView;

#ifndef IL_ADD_IL_REDUCE
#ifdef IL_EMBEDDING_REDUCE
constexpr int64_t kEmbRows = 2;
constexpr int64_t kEmbColumns = 2;
constexpr int64_t kEmbReduceSize = 2;
constexpr int64_t kEmbTableRows = 4;
constexpr char kEmbGraphName[] = "indirect_load_embedding_reduce_simt_test";

struct EmbReduceGraphView {
  std::shared_ptr<af::AscGraph> graph;
  af::AxisId a0;
  af::AxisId a1;
  af::AxisId a2;
  af::Expression rows;
  af::Expression columns;
  af::Expression reduce_size;
  af::Expression table_rows;
};

EmbReduceGraphView CreateEmbReduceGraphView() {
  EmbReduceGraphView view;
  view.graph = std::make_shared<af::AscGraph>(kEmbGraphName);
  view.rows = view.graph->CreateSizeVar(kEmbRows);
  view.columns = view.graph->CreateSizeVar(kEmbColumns);
  view.reduce_size = view.graph->CreateSizeVar(kEmbReduceSize);
  view.table_rows = view.graph->CreateSizeVar(kEmbTableRows);
  view.a0 = view.graph->CreateAxis("a0", view.rows).id;
  view.a1 = view.graph->CreateAxis("a1", view.columns).id;
  view.a2 = view.graph->CreateAxis("a2", view.reduce_size).id;
  return view;
}

std::shared_ptr<af::AscGraph> CreateEmbReduceSubGraph() {
  const EmbReduceGraphView view = CreateEmbReduceGraphView();
  const auto axes = std::vector<af::AxisId>{view.a0, view.a1, view.a2};
  const auto output_repeats = std::vector<af::Expression>{view.rows, view.columns, view.reduce_size};
  const auto output_strides =
      std::vector<af::Expression>{view.columns * view.reduce_size, view.reduce_size, af::ops::One};
  const auto index_repeats = output_repeats;
  const auto index_strides = std::vector<af::Expression>{view.reduce_size, af::ops::Zero, af::ops::One};
  const auto table_repeats = std::vector<af::Expression>{view.table_rows, view.columns, view.reduce_size};
  const auto table_strides = std::vector<af::Expression>{view.columns, af::ops::One, af::ops::Zero};

  af::ascir_op::Data index0("index0", *view.graph);
  index0.ir_attr.SetIndex(0);
  SetView(index0, axes, index_repeats, index_strides, af::DT_INT64);
  af::ascir_op::Load index0_load("index0_load");
  view.graph->AddNode(index0_load);
  index0_load.x = index0.y;
  SetView(index0_load, axes, index_repeats, index_strides, af::DT_INT64);

  // Keep the existing three-input backend wrapper stable; this input is intentionally unused by the minimal graph.
  af::ascir_op::Data unused_input("unused_input", *view.graph);
  unused_input.ir_attr.SetIndex(2);
  SetView(unused_input, axes, index_repeats, index_strides, af::DT_INT64);

  af::ascir_op::Data table("table", *view.graph);
  table.ir_attr.SetIndex(1);
  SetView(table, axes, table_repeats, table_strides, af::DT_FLOAT);
  af::ascir_op::Load table_load("table_load");
  view.graph->AddNode(table_load);
  table_load.x = table.y;
  SetView(table_load, axes, table_repeats, table_strides, af::DT_FLOAT);

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  view.graph->AddNode(indirect_load);
  indirect_load.x1 = table_load.y;
  indirect_load.x2 = index0_load.y;
  indirect_load.ir_attr.SetAxis(0);
  indirect_load.ir_attr.SetNegative_index_support(true);
  indirect_load.ir_attr.SetNeed_check_bound(true);
  indirect_load.ir_attr.SetMax(view.table_rows);
  SetView(indirect_load, axes, output_repeats, output_strides, af::DT_FLOAT);

  af::ascir_op::Scalar zero("zero", *view.graph);
  zero.ir_attr.SetValue("0");
  zero.y.dtype = af::DT_INT64;
  af::ascir_op::Ge ge("ge");
  view.graph->AddNode(ge);
  ge.x1 = index0_load.y;
  ge.x2 = zero.y;
  SetView(ge, axes, output_repeats, output_strides, af::DT_BOOL);

  af::ascir_op::Scalar one_f("one_f", *view.graph);
  one_f.ir_attr.SetValue("1");
  one_f.y.dtype = af::DT_FLOAT;
  af::ascir_op::Scalar zero_f("zero_f", *view.graph);
  zero_f.ir_attr.SetValue("0");
  zero_f.y.dtype = af::DT_FLOAT;
  af::ascir_op::Select select("select");
  view.graph->AddNode(select);
  select.x1 = ge.y;
  select.x2 = one_f.y;
  select.x3 = zero_f.y;
  SetView(select, axes, output_repeats, output_strides, af::DT_FLOAT);

  af::ascir_op::Mul mul("mul");
  view.graph->AddNode(mul);
  mul.x1 = indirect_load.y;
  mul.x2 = select.y;
  SetView(mul, axes, output_repeats, output_strides, af::DT_FLOAT);

  const auto reduce_repeats = std::vector<af::Expression>{view.rows, view.columns, af::ops::One};
  const auto reduce_strides = std::vector<af::Expression>{view.columns, af::ops::One, af::ops::Zero};
  af::ascir_op::Sum sum("sum");
  view.graph->AddNode(sum);
  sum.x = mul.y;
  SetView(sum, axes, reduce_repeats, reduce_strides, af::DT_FLOAT);
  af::ascir_op::Store store("store");
  view.graph->AddNode(store);
  store.x = sum.y;
  SetView(store, axes, reduce_repeats, reduce_strides, af::DT_FLOAT);
  af::ascir_op::Output output("output");
  view.graph->AddNode(output);
  output.ir_attr.SetIndex(0);
  output.x = store.y;
  SetView(output, axes, reduce_repeats, reduce_strides, af::DT_FLOAT);
  return view.graph;
}
#else
// 三输入 Where 链保持在 SIMT index 区（默认场景，indirect_load_broadcast_index_where_simt_test）。
constexpr int64_t kRows = 6400;
constexpr int64_t kColumns = 32;
constexpr int64_t kTableRows = 315511;
constexpr char kGraphName[] = "indirect_load_broadcast_index_where_simt_test";

struct WhereGraphView {
  std::shared_ptr<af::AscGraph> graph;
  af::AxisId rows_axis;
  af::AxisId columns_axis;
  af::Expression rows;
  af::Expression columns;
  af::Expression table_rows;
};

WhereGraphView CreateGraphView() {
  WhereGraphView view;
  view.graph = std::make_shared<af::AscGraph>(kGraphName);
  view.rows = view.graph->CreateSizeVar(kRows);
  view.columns = view.graph->CreateSizeVar(kColumns);
  view.table_rows = view.graph->CreateSizeVar(kTableRows);
  view.rows_axis = view.graph->CreateAxis("a0", view.rows).id;
  view.columns_axis = view.graph->CreateAxis("a1", view.columns).id;
  return view;
}

std::shared_ptr<af::AscGraph> CreateSubGraph() {
  const WhereGraphView view = CreateGraphView();
  const auto axes = std::vector<af::AxisId>{view.rows_axis, view.columns_axis};
  const auto output_repeats = std::vector<af::Expression>{view.rows, view.columns};
  const auto output_strides = std::vector<af::Expression>{view.columns, af::ops::One};
  const auto index_repeats = std::vector<af::Expression>{view.rows, af::ops::One};
  const auto index_strides = std::vector<af::Expression>{af::ops::One, af::ops::Zero};
  const auto table_repeats = std::vector<af::Expression>{view.table_rows, view.columns};
  const auto table_strides = std::vector<af::Expression>{view.columns, af::ops::One};

  af::ascir_op::Data index0("index0", *view.graph);
  index0.ir_attr.SetIndex(0);
  SetView(index0, axes, index_repeats, index_strides, af::DT_INT64);
  af::ascir_op::Load index0_load("index0_load");
  view.graph->AddNode(index0_load);
  index0_load.x = index0.y;
  SetView(index0_load, axes, index_repeats, index_strides, af::DT_INT64);

  af::ascir_op::Broadcast index0_broadcast("index0_broadcast");
  view.graph->AddNode(index0_broadcast);
  index0_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  index0_broadcast.x = index0_load.y;
  SetView(index0_broadcast, axes, output_repeats, output_strides, af::DT_INT64);

  af::ascir_op::Cast index0_cast("index0_cast");
  view.graph->AddNode(index0_cast);
  index0_cast.x = index0_broadcast.y;
  SetView(index0_cast, axes, output_repeats, output_strides, af::DT_FLOAT);

  af::ascir_op::Scalar minus_one("minus_one", *view.graph);
  minus_one.ir_attr.SetValue("-1");
  minus_one.y.dtype = af::DT_INT64;
  af::ascir_op::Broadcast minus_one_row("minus_one_row");
  view.graph->AddNode(minus_one_row);
  minus_one_row.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  minus_one_row.x = minus_one.y;
  SetView(minus_one_row, axes, index_repeats, index_strides, af::DT_INT64);
  af::ascir_op::Broadcast minus_one_full("minus_one_full");
  view.graph->AddNode(minus_one_full);
  minus_one_full.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  minus_one_full.x = minus_one_row.y;
  SetView(minus_one_full, axes, output_repeats, output_strides, af::DT_INT64);
  af::ascir_op::Cast minus_one_cast("minus_one_cast");
  view.graph->AddNode(minus_one_cast);
  minus_one_cast.x = minus_one_full.y;
  SetView(minus_one_cast, axes, output_repeats, output_strides, af::DT_FLOAT);

  af::ascir_op::Eq equal("equal");
  view.graph->AddNode(equal);
  equal.x1 = index0_cast.y;
  equal.x2 = minus_one_cast.y;
  SetView(equal, axes, output_repeats, output_strides, af::DT_BOOL);

  af::ascir_op::Data index2("index2", *view.graph);
  index2.ir_attr.SetIndex(2);
  SetView(index2, axes, index_repeats, index_strides, af::DT_INT64);
  af::ascir_op::Load index2_load("index2_load");
  view.graph->AddNode(index2_load);
  index2_load.x = index2.y;
  SetView(index2_load, axes, index_repeats, index_strides, af::DT_INT64);
  af::ascir_op::Broadcast index2_broadcast("index2_broadcast");
  view.graph->AddNode(index2_broadcast);
  index2_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  index2_broadcast.x = index2_load.y;
  SetView(index2_broadcast, axes, output_repeats, output_strides, af::DT_INT64);

  af::ascir_op::Where where("where");
  view.graph->AddNode(where);
  where.x1 = equal.y;
  where.x2 = index2_broadcast.y;
  where.x3 = index0_broadcast.y;
  SetView(where, axes, output_repeats, output_strides, af::DT_INT64);

  af::ascir_op::Data table("table", *view.graph);
  table.ir_attr.SetIndex(1);
  SetView(table, axes, table_repeats, table_strides, af::DT_FLOAT);
  af::ascir_op::Load table_load("table_load");
  view.graph->AddNode(table_load);
  table_load.x = table.y;
  SetView(table_load, axes, table_repeats, table_strides, af::DT_FLOAT);

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  view.graph->AddNode(indirect_load);
  indirect_load.x1 = table_load.y;
  indirect_load.x2 = where.y;
  indirect_load.ir_attr.SetAxis(0);
  indirect_load.ir_attr.SetNegative_index_support(true);
  indirect_load.ir_attr.SetNeed_check_bound(true);
  indirect_load.ir_attr.SetMax(view.table_rows);
  SetView(indirect_load, axes, output_repeats, output_strides, af::DT_FLOAT);

  af::ascir_op::Store store("store");
  view.graph->AddNode(store);
  store.x = indirect_load.y;
  SetView(store, axes, output_repeats, output_strides, af::DT_FLOAT);
  af::ascir_op::Output output("output");
  view.graph->AddNode(output);
  output.ir_attr.SetIndex(0);
  output.x = store.y;
  SetView(output, axes, output_repeats, output_strides, af::DT_FLOAT);
  return view.graph;
}
#endif  // IL_EMBEDDING_REDUCE
#else
// 双输入 Add 前置 + 后融合 ReduceSum 的 SIMT 最小场景（indirect_load_add_il_reduce_test）。
constexpr int64_t kAddIlReduceRows = 4;
constexpr int64_t kAddIlReduceColumns = 16;
constexpr int64_t kAddIlReduceTableRows = 8;
constexpr char kAddIlReduceGraphName[] = "indirect_load_add_il_reduce_test";

struct AddIlReduceGraphView {
  std::shared_ptr<af::AscGraph> graph;
  af::AxisId rows_axis;
  af::AxisId columns_axis;
  af::Expression rows;
  af::Expression columns;
  af::Expression table_rows;
};

AddIlReduceGraphView CreateAddIlReduceGraphView() {
  AddIlReduceGraphView view;
  view.graph = std::make_shared<af::AscGraph>(kAddIlReduceGraphName);
  view.rows = view.graph->CreateSizeVar(kAddIlReduceRows);
  view.columns = view.graph->CreateSizeVar(kAddIlReduceColumns);
  view.table_rows = view.graph->CreateSizeVar(kAddIlReduceTableRows);
  view.rows_axis = view.graph->CreateAxis("a0", view.rows).id;
  view.columns_axis = view.graph->CreateAxis("a1", view.columns).id;
  return view;
}

// index0 与 offset 各自 Load/Broadcast/Cast 后经双输入 Add 融合，再 Cast 回 INT64 作为 gather 索引。
void BuildAddIlReduceIndexChain(const AddIlReduceGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  const auto axes = std::vector<af::AxisId>{view.rows_axis, view.columns_axis};
  const auto output_repeats = std::vector<af::Expression>{view.rows, view.columns};
  const auto output_strides = std::vector<af::Expression>{view.columns, af::ops::One};
  const auto index_repeats = std::vector<af::Expression>{view.rows, af::ops::One};
  const auto index_strides = std::vector<af::Expression>{af::ops::One, af::ops::Zero};

  af::ascir_op::Data index0("index0", *view.graph);
  index0.ir_attr.SetIndex(0);
  SetView(index0, axes, index_repeats, index_strides, af::DT_INT64);
  af::ascir_op::Load index0_load("index0_load");
  view.graph->AddNode(index0_load);
  index0_load.x = index0.y;
  SetView(index0_load, axes, index_repeats, index_strides, af::DT_INT64);
  af::ascir_op::Broadcast index0_broadcast("index0_broadcast");
  view.graph->AddNode(index0_broadcast);
  index0_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  index0_broadcast.x = index0_load.y;
  SetView(index0_broadcast, axes, output_repeats, output_strides, af::DT_INT64);
  af::ascir_op::Cast index0_cast("index0_cast");
  view.graph->AddNode(index0_cast);
  index0_cast.x = index0_broadcast.y;
  SetView(index0_cast, axes, output_repeats, output_strides, af::DT_FLOAT);

  af::ascir_op::Data offset("offset", *view.graph);
  offset.ir_attr.SetIndex(2);
  SetView(offset, axes, index_repeats, index_strides, af::DT_INT64);
  af::ascir_op::Load offset_load("offset_load");
  view.graph->AddNode(offset_load);
  offset_load.x = offset.y;
  SetView(offset_load, axes, index_repeats, index_strides, af::DT_INT64);
  af::ascir_op::Broadcast offset_broadcast("offset_broadcast");
  view.graph->AddNode(offset_broadcast);
  offset_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  offset_broadcast.x = offset_load.y;
  SetView(offset_broadcast, axes, output_repeats, output_strides, af::DT_INT64);
  af::ascir_op::Cast offset_cast("offset_cast");
  view.graph->AddNode(offset_cast);
  offset_cast.x = offset_broadcast.y;
  SetView(offset_cast, axes, output_repeats, output_strides, af::DT_FLOAT);

  af::ascir_op::Add index_add("index_add");
  view.graph->AddNode(index_add);
  index_add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  index_add.x1 = index0_cast.y;
  index_add.x2 = offset_cast.y;
  SetView(index_add, axes, output_repeats, output_strides, af::DT_FLOAT);

  af::ascir_op::Cast index_add_cast("index_add_cast");
  view.graph->AddNode(index_add_cast);
  index_add_cast.x = index_add.y;
  SetView(index_add_cast, axes, output_repeats, output_strides, af::DT_INT64);
  indirect_load.x2 = index_add_cast.y;
}

std::shared_ptr<af::AscGraph> CreateAddIlReduceSubGraph() {
  const AddIlReduceGraphView view = CreateAddIlReduceGraphView();
  const auto axes = std::vector<af::AxisId>{view.rows_axis, view.columns_axis};
  const auto output_repeats = std::vector<af::Expression>{view.rows, view.columns};
  const auto output_strides = std::vector<af::Expression>{view.columns, af::ops::One};
  const auto table_repeats = std::vector<af::Expression>{view.table_rows, view.columns};
  const auto table_strides = std::vector<af::Expression>{view.columns, af::ops::One};
  const auto reduce_repeats = std::vector<af::Expression>{view.rows, af::ops::One};
  const auto reduce_strides = std::vector<af::Expression>{af::ops::One, af::ops::Zero};

  af::ascir_op::Data table("table", *view.graph);
  table.ir_attr.SetIndex(1);
  SetView(table, axes, table_repeats, table_strides, af::DT_FLOAT);
  af::ascir_op::Load table_load("table_load");
  view.graph->AddNode(table_load);
  table_load.x = table.y;
  SetView(table_load, axes, table_repeats, table_strides, af::DT_FLOAT);

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  view.graph->AddNode(indirect_load);
  indirect_load.x1 = table_load.y;
  indirect_load.ir_attr.SetAxis(0);
  SetView(indirect_load, axes, output_repeats, output_strides, af::DT_FLOAT);
  BuildAddIlReduceIndexChain(view, indirect_load);

  af::ascir_op::Sum sum("sum");
  view.graph->AddNode(sum);
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.attr.sched.axis = axes;
  sum.x = indirect_load.y;
  SetView(sum, axes, reduce_repeats, reduce_strides, af::DT_FLOAT);

  af::ascir_op::Store store("store");
  view.graph->AddNode(store);
  store.x = sum.y;
  SetView(store, axes, reduce_repeats, reduce_strides, af::DT_FLOAT);
  af::ascir_op::Output output("output");
  view.graph->AddNode(output);
  output.ir_attr.SetIndex(0);
  output.x = store.y;
  SetView(output, axes, reduce_repeats, reduce_strides, af::DT_FLOAT);
  return view.graph;
}
#endif  // IL_ADD_IL_REDUCE

class ThreeInputBackendGraph {
 public:
  explicit ThreeInputBackendGraph(const char *graph_name) : fused_graph_(graph_name) {
    af::ascir_op::Data index0("input0", fused_graph_);
    af::ascir_op::Data table("input1", fused_graph_);
    af::ascir_op::Data index2("input2", fused_graph_);
    index0.ir_attr.SetIndex(0);
    table.ir_attr.SetIndex(1);
    index2.ir_attr.SetIndex(2);
    compute_graph_ = af::AscGraphUtils::GetComputeGraph(fused_graph_);
    if (compute_graph_ == nullptr) {
      return;
    }
    const auto index_desc = std::make_shared<af::GeTensorDesc>();
    index_desc->SetDataType(af::DT_INT64);
    const auto table_desc = std::make_shared<af::GeTensorDesc>();
    table_desc->SetDataType(af::DT_FLOAT);
    const auto backend_desc = std::make_shared<af::OpDesc>("asc_backend", "AscBackend");
    backend_desc->AddInputDesc(index_desc->Clone());
    backend_desc->AddInputDesc(table_desc->Clone());
    backend_desc->AddInputDesc(index_desc->Clone());
    backend_desc->AddOutputDesc(table_desc->Clone());
    backend_ = compute_graph_->AddNode(backend_desc);
  }

  af::ComputeGraphPtr Finalize(const std::shared_ptr<af::AscGraph> &sub_graph) {
    if (compute_graph_ == nullptr || backend_ == nullptr) {
      return nullptr;
    }
    const auto attrs = backend_->GetOpDesc()->GetOrCreateAttrsGroup<af::AutoFuseAttrs>();
    if (attrs == nullptr) {
      return nullptr;
    }
    attrs->SetAscGraph(sub_graph);
    af::ascir_op::Output output("output");
    output.ir_attr.SetIndex(0);
    const auto output_node = compute_graph_->AddNode(af::OpDescUtils::GetOpDescFromOperator(output));
    const auto input0 = fused_graph_.FindNode("input0");
    const auto input1 = fused_graph_.FindNode("input1");
    const auto input2 = fused_graph_.FindNode("input2");
    if (output_node == nullptr || input0 == nullptr || input1 == nullptr || input2 == nullptr) {
      return nullptr;
    }
    const bool edges_added =
        af::GraphUtils::AddEdge(input0->GetOutDataAnchor(0), backend_->GetInDataAnchor(0)) == ge::GRAPH_SUCCESS &&
        af::GraphUtils::AddEdge(input1->GetOutDataAnchor(0), backend_->GetInDataAnchor(1)) == ge::GRAPH_SUCCESS &&
        af::GraphUtils::AddEdge(input2->GetOutDataAnchor(0), backend_->GetInDataAnchor(2)) == ge::GRAPH_SUCCESS &&
        af::GraphUtils::AddEdge(backend_->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)) == ge::GRAPH_SUCCESS;
    return edges_added && compute_graph_->TopologicalSorting() == ge::GRAPH_SUCCESS ? compute_graph_ : nullptr;
  }

 private:
  af::AscGraph fused_graph_;
  af::ComputeGraphPtr compute_graph_;
  af::NodePtr backend_;
};
}  // namespace

#ifndef IL_ADD_IL_REDUCE
#ifdef IL_EMBEDDING_REDUCE
using TestBackendIndirectLoadEmbReduceE2e = indirect_load_test::PrecisionBackendE2e;

TEST_F(TestBackendIndirectLoadEmbReduceE2e, GeneratesEmbeddingReduceSimtKernel) {
  ThreeInputBackendGraph backend(kEmbGraphName);
  const auto graph = backend.Finalize(CreateEmbReduceSubGraph());
  ASSERT_NE(graph, nullptr);
  codegen::CodegenResult result;
  indirect_load_test::GenerateForTemplate(graph, {}, ascir::TemplateId::kIndirectLoadSimt, result);
  EXPECT_NE(result.kernel.find("// IndirectLoad SIMT"), std::string::npos);
  EXPECT_NE(result.kernel.find("IndirectLoadSimt"), std::string::npos);
  EXPECT_NE(result.kernel.find("ReduceSum"), std::string::npos);
  EXPECT_NE(result.kernel.find("Output(float value, uint32_t output_index, uint32_t index_offset"), std::string::npos);
  EXPECT_NE(result.kernel.find("context.gm_5[output_index]"), std::string::npos);
  EXPECT_NE(result.kernel.find("context.gm_5[index_offset]"), std::string::npos);
  indirect_load_test::WriteGeneratedFiles(result);
}
#else
using TestBackendIndirectLoadBroadcastWhereE2e = indirect_load_test::PrecisionBackendE2e;

TEST_F(TestBackendIndirectLoadBroadcastWhereE2e, GeneratesWhereIndirectLoadSimtKernel) {
  ThreeInputBackendGraph backend(kGraphName);
  const auto graph = backend.Finalize(CreateSubGraph());
  ASSERT_NE(graph, nullptr);
  codegen::CodegenResult result;
  indirect_load_test::GenerateForTemplate(graph, {}, ascir::TemplateId::kIndirectLoadSimt, result);
  EXPECT_NE(result.kernel.find("// IndirectLoad SIMT"), std::string::npos);
  EXPECT_NE(result.kernel.find("IndirectLoadSimt"), std::string::npos);
  indirect_load_test::WriteGeneratedFiles(result);
}
#endif
#else
using TestBackendIndirectLoadAddIlReduceE2e = indirect_load_test::PrecisionBackendE2e;

TEST_F(TestBackendIndirectLoadAddIlReduceE2e, GeneratesAddIlReduceSimtKernel) {
  ThreeInputBackendGraph backend(kAddIlReduceGraphName);
  const auto graph = backend.Finalize(CreateAddIlReduceSubGraph());
  ASSERT_NE(graph, nullptr);
  codegen::CodegenResult result;
  indirect_load_test::GenerateForTemplate(graph, {}, ascir::TemplateId::kIndirectLoadSimt, result);
  EXPECT_NE(result.kernel.find("// IndirectLoad SIMT"), std::string::npos);
  EXPECT_NE(result.kernel.find("IndirectLoadSimt<"), std::string::npos);
  EXPECT_NE(result.kernel.find("ReduceSum"), std::string::npos);
  indirect_load_test::WriteGeneratedFiles(result);
}
#endif  // IL_ADD_IL_REDUCE

#endif

#if defined(IL_CASE_STRIDE_ZERO)
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

namespace {
constexpr uint32_t kInputZeroStrideMask = IL_INPUT_ZERO_STRIDE_MASK;
constexpr uint32_t kIndexZeroStrideMask = IL_INDEX_ZERO_STRIDE_MASK;
constexpr int32_t kInputElementCount = IL_HAS_INPUT_ELEMENT;
constexpr int32_t kIndexElementCount = IL_HAS_INDEX_ELEMENT;
constexpr bool kExpectSimt = IL_EXPECT_SIMT;
constexpr bool kExpectSk = IL_EXPECT_SK;

using indirect_load_test::SetView;

struct StrideZeroGraphView {
  std::shared_ptr<af::AscGraph> graph;
  std::vector<af::AxisId> axes;
  std::vector<af::Expression> repeats;
  std::vector<af::Expression> dense_strides;
  std::vector<af::Expression> input_strides;
  std::vector<af::Expression> index_strides;
};

std::vector<af::Expression> MakeStrides(const std::vector<af::Expression> &repeats, uint32_t zero_stride_mask) {
  std::vector<af::Expression> strides(repeats.size(), af::ops::Zero);
  af::Expression stride = af::ops::One;
  for (size_t index = repeats.size(); index > 0UL; --index) {
    const size_t dim = index - 1UL;
    if ((zero_stride_mask & (1U << dim)) == 0U) {
      strides[dim] = stride;
      stride = stride * repeats[dim];
    }
  }
  return strides;
}

StrideZeroGraphView CreateGraphView() {
  StrideZeroGraphView view;
  view.graph = std::make_shared<af::AscGraph>("indirect_load_stride_zero_test");
  const std::vector<int64_t> shape = {4, 5, 4, 16};
  for (size_t dim = 0; dim < shape.size(); ++dim) {
    const auto size = view.graph->CreateSizeVar(shape[dim]);
    view.repeats.emplace_back(size);
    view.axes.emplace_back(view.graph->CreateAxis(("z" + std::to_string(dim)).c_str(), size).id);
  }
  view.dense_strides = MakeStrides(view.repeats, 0U);
  view.input_strides = MakeStrides(view.repeats, kInputZeroStrideMask);
  view.index_strides = MakeStrides(view.repeats, kIndexZeroStrideMask);
  return view;
}

void BuildInputPath(const StrideZeroGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Data x("x");
  view.graph->AddNode(x);
  x.ir_attr.SetIndex(0);
  SetView(x, view.axes, view.repeats, view.input_strides, af::DT_FLOAT16);
  af::ascir_op::Load input_load("input_load");
  view.graph->AddNode(input_load);
  input_load.x = x.y;
  SetView(input_load, view.axes, view.repeats, view.input_strides, af::DT_FLOAT16);

  std::vector<std::unique_ptr<af::ascir_op::Abs>> input_elements;
  for (int32_t i = 0; i < kInputElementCount; ++i) {
    const auto name = "input_abs_" + std::to_string(i);
    auto input_abs = std::make_unique<af::ascir_op::Abs>(name.c_str());
    view.graph->AddNode(*input_abs);
    input_abs->attr.api.compute_type = af::ComputeType::kComputeElewise;
    input_abs->x = i == 0 ? input_load.y : input_elements.back()->y;
    SetView(*input_abs, view.axes, view.repeats, view.input_strides, af::DT_FLOAT16);
    input_elements.emplace_back(std::move(input_abs));
  }
  indirect_load.x1 = input_elements.empty() ? input_load.y : input_elements.back()->y;
}

void BuildIndexPath(const StrideZeroGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Data index("index");
  view.graph->AddNode(index);
  index.ir_attr.SetIndex(1);
  SetView(index, view.axes, view.repeats, view.index_strides, af::DT_INT64);
  af::ascir_op::Load index_load("index_load");
  view.graph->AddNode(index_load);
  index_load.x = index.y;
  SetView(index_load, view.axes, view.repeats, view.index_strides, af::DT_INT64);
  std::vector<std::unique_ptr<af::ascir_op::Abs>> index_elements;
  for (int32_t i = 0; i < kIndexElementCount; ++i) {
    const auto name = "index_abs_" + std::to_string(i);
    auto index_abs = std::make_unique<af::ascir_op::Abs>(name.c_str());
    view.graph->AddNode(*index_abs);
    index_abs->attr.api.compute_type = af::ComputeType::kComputeElewise;
    index_abs->x = i == 0 ? index_load.y : index_elements.back()->y;
    SetView(*index_abs, view.axes, view.repeats, view.index_strides, af::DT_INT64);
    index_elements.emplace_back(std::move(index_abs));
  }
  indirect_load.x2 = index_elements.empty() ? index_load.y : index_elements.back()->y;
}

void BuildOutputPath(const StrideZeroGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  indirect_load_test::BuildOutputPath(view.graph, indirect_load, view.axes, view.repeats, view.dense_strides, true);
}

af::ComputeGraphPtr CreateGraph() {
  indirect_load_test::BackendGraph backend("indirect_load_stride_zero_test", "data0", "data1", af::DT_FLOAT16);
  return backend.Finalize(
      indirect_load_test::CreateSubGraph(CreateGraphView(), BuildInputPath, BuildIndexPath, BuildOutputPath), "output");
}
}  // namespace

using TestBackendIndirectLoadStrideZeroE2e = indirect_load_test::PrecisionBackendE2e;

TEST_F(TestBackendIndirectLoadStrideZeroE2e, GeneratesSelectedTemplateWithoutBroadcastMaterialization) {
  const auto graph = CreateGraph();
  ASSERT_NE(graph, nullptr);
  const std::map<std::string, std::string> shape_info;
  const auto expected_template = indirect_load_test::GetExpectedTemplate(kExpectSimt, kExpectSk);
  codegen::CodegenResult result;
  indirect_load_test::GenerateForTemplate(graph, shape_info, expected_template, result);
  EXPECT_NE(result.kernel.find(indirect_load_test::GetTemplateMarker(expected_template)), std::string::npos);
  EXPECT_EQ(result.kernel.find("BroadcastExtend<"), std::string::npos);
  indirect_load_test::WriteGeneratedFiles(result);
}

#endif

#if defined(IL_CASE_TORCH_STRIDED)
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

namespace {
constexpr char kGraphName[] = "indirect_load_torch_gather_strided_test";
constexpr bool kExpectSimt = IL_EXPECT_SIMT;
constexpr bool kExpectSk = IL_EXPECT_SK;
#ifdef IL_INDEX_SELECT_CASE
constexpr int64_t kDim0 = 30;
constexpr int64_t kInputDim1 = 6;
constexpr int64_t kOutputDim1 = 3;
constexpr int64_t kDim2 = 23;
constexpr int64_t kEffectiveInputStride0 = 138;
constexpr int64_t kEffectiveInputStride1 = 23;
constexpr int64_t kEffectiveInputStride2 = 1;
#else
constexpr int64_t kDim0 = 8;
constexpr int64_t kInputDim1 = 32;
constexpr int64_t kOutputDim1 = 16;
constexpr int64_t kDim2 = 5;
#endif
constexpr int64_t kInputStride0 = IL_INPUT_STRIDE0;
constexpr int64_t kInputStride1 = IL_INPUT_STRIDE1;
constexpr int64_t kInputStride2 = IL_INPUT_STRIDE2;
constexpr int64_t kIndexStride0 = IL_INDEX_STRIDE0;
constexpr int64_t kIndexStride1 = IL_INDEX_STRIDE1;
constexpr int64_t kIndexStride2 = IL_INDEX_STRIDE2;
#ifndef IL_INDEX_SELECT_CASE
constexpr int64_t kEffectiveInputStride0 = kInputStride0;
constexpr int64_t kEffectiveInputStride1 = kInputStride1;
constexpr int64_t kEffectiveInputStride2 = kInputStride2;
#endif

using indirect_load_test::SetView;

struct TorchGatherGraphView {
  std::shared_ptr<af::AscGraph> graph;
  std::vector<af::AxisId> axes;
  std::vector<af::Expression> output_sizes;
  std::vector<af::Expression> output_strides;
  std::vector<af::Expression> index_strides;
  std::vector<af::Expression> input_sizes;
  std::vector<af::Expression> input_strides;
};

TorchGatherGraphView CreateGraphView() {
  TorchGatherGraphView view;
  view.graph = std::make_shared<af::AscGraph>(kGraphName);
  view.output_sizes = {af::Symbol(kDim0), af::Symbol(kOutputDim1), af::Symbol(kDim2)};
  view.output_strides = {af::Symbol(kOutputDim1 * kDim2), af::Symbol(kDim2), af::Symbol(1)};
  view.index_strides = {af::Symbol(kIndexStride0), af::Symbol(kIndexStride1), af::Symbol(kIndexStride2)};
  view.input_sizes = {af::Symbol(kDim0), af::Symbol(kInputDim1), af::Symbol(kDim2)};
  view.input_strides = {af::Symbol(kEffectiveInputStride0), af::Symbol(kEffectiveInputStride1),
                        af::Symbol(kEffectiveInputStride2)};
  const auto a0 = view.graph->CreateAxis("a0", view.output_sizes[0]);
  const auto a1 = view.graph->CreateAxis("a1", view.output_sizes[1]);
  const auto a2 = view.graph->CreateAxis("a2", view.output_sizes[2]);
  view.axes = {a0.id, a1.id, a2.id};
  return view;
}

std::shared_ptr<af::AscGraph> CreateSubGraph() {
  const auto view = CreateGraphView();
  af::ascir_op::Data index("graph_hint/data", *view.graph);
  index.ir_attr.SetIndex(1);
  index.y.dtype = af::DT_INT64;
  af::ascir_op::Load index_load("graph_hint/load");
  view.graph->AddNode(index_load);
  index_load.ir_attr.SetOffset(af::sym::kSymbolZero);
  index_load.x = index.y;
#ifdef IL_INDEX_SELECT_CASE
  const std::vector<af::Expression> index_source_sizes = {af::Symbol(1), af::Symbol(kOutputDim1), af::Symbol(1)};
  const std::vector<af::Expression> index_source_strides = {af::ops::Zero, af::ops::One, af::ops::Zero};
  SetView(index_load, view.axes, index_source_sizes, index_source_strides, af::DT_INT64);
  af::ascir_op::Broadcast index_first_broadcast("graph_hint/index_broadcast0");
  view.graph->AddNode(index_first_broadcast);
  index_first_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  index_first_broadcast.x = index_load.y;
  const std::vector<af::Expression> index_intermediate_sizes = {af::Symbol(kDim0), af::Symbol(kOutputDim1),
                                                                af::Symbol(1)};
  const std::vector<af::Expression> index_intermediate_strides = {af::Symbol(kOutputDim1), af::ops::One, af::ops::Zero};
  SetView(index_first_broadcast, view.axes, index_intermediate_sizes, index_intermediate_strides, af::DT_INT64);
  af::ascir_op::Broadcast index_final_broadcast("graph_hint/index_broadcast1");
  view.graph->AddNode(index_final_broadcast);
  index_final_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  index_final_broadcast.x = index_first_broadcast.y;
  SetView(index_final_broadcast, view.axes, view.output_sizes, view.output_strides, af::DT_INT64);
  const auto index_output = index_final_broadcast.y;
#else
  SetView(index_load, view.axes, view.output_sizes, view.index_strides, af::DT_INT64);
#endif

  af::ascir_op::Data data("graph_hint/data1", *view.graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = af::DT_FLOAT;
  af::ascir_op::Load data_load("graph_hint/load1");
  view.graph->AddNode(data_load);
  data_load.ir_attr.SetOffset(af::sym::kSymbolZero);
  data_load.x = data.y;
  SetView(data_load, view.axes, view.input_sizes, view.input_strides, af::DT_FLOAT);

  af::ascir_op::IndirectLoad indirect_load("graph_hint/indirectload");
  view.graph->AddNode(indirect_load);
  indirect_load.x1 = data_load.y;
#ifdef IL_INDEX_SELECT_CASE
  indirect_load.x2 = index_output;
#else
  indirect_load.x2 = index_load.y;
#endif
  indirect_load.ir_attr.SetAxis(1);
  indirect_load.ir_attr.SetNegative_index_support(true);
  indirect_load.ir_attr.SetNeed_check_bound(true);
  indirect_load.ir_attr.SetMax(view.graph->CreateSizeVar(32));
  SetView(indirect_load, view.axes, view.output_sizes, view.output_strides, af::DT_FLOAT);

  af::ascir_op::Store store("graph_hint/store");
  view.graph->AddNode(store);
  store.ir_attr.SetOffset(af::sym::kSymbolZero);
  store.x = indirect_load.y;
  SetView(store, view.axes, view.output_sizes, view.output_strides, af::DT_FLOAT);
  af::ascir_op::Output output("graph_hint/output");
  view.graph->AddNode(output);
  output.ir_attr.SetIndex(0);
  output.x = store.y;
  output.y.dtype = af::DT_FLOAT;
  return view.graph;
}

af::ComputeGraphPtr CreateGraph() {
  indirect_load_test::BackendGraph backend(kGraphName, "input0", "input1", af::DT_FLOAT);
  return backend.Finalize(CreateSubGraph(), "output0");
}
}  // namespace

using TestIndirectLoadTorchGatherStridedE2e = indirect_load_test::BackendE2e;

TEST_F(TestIndirectLoadTorchGatherStridedE2e, GeneratesKernelForInductorGraph) {
  const auto graph = CreateGraph();
  ASSERT_NE(graph, nullptr);
  const auto expected_template = indirect_load_test::GetExpectedTemplate(kExpectSimt, kExpectSk);
  codegen::CodegenResult result;
  indirect_load_test::GenerateForTemplate(graph, {}, expected_template, result);
  EXPECT_NE(result.kernel.find(indirect_load_test::GetTemplateMarker(expected_template)), std::string::npos);
  if (!kExpectSk) {
    EXPECT_EQ(result.kernel.find("// IndirectLoad SK"), std::string::npos);
  }
  if (!kExpectSimt) {
    EXPECT_EQ(result.kernel.find("// IndirectLoad SIMT"), std::string::npos);
  }
  if (kExpectSimt || kExpectSk) {
    EXPECT_EQ(result.kernel.find("// IndirectLoad SIMD"), std::string::npos);
  }
  if (kExpectSimt) {
#ifdef IL_INDEX_SELECT_CASE
    EXPECT_NE(result.kernel.find("IndirectLoadSimtStridedPolicy<uint32_t, 3, 1, 7ULL, 2ULL>"), std::string::npos);
#else
    EXPECT_NE(result.kernel.find("IndirectLoadSimtStridedPolicy<uint32_t, 3, 1, 7ULL, 7ULL>"), std::string::npos);
#endif
    EXPECT_EQ(result.kernel.find("x_axis_size"), std::string::npos);
    EXPECT_EQ(result.kernel.find("indirect_index < 0"), std::string::npos);
  }
  indirect_load_test::WriteGeneratedFiles(result);
}

#endif

#if defined(IL_CASE_EMBEDDING)
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

namespace {
constexpr char kGraphName[] = "indirect_load_embedding_test";
constexpr int64_t kInputRows = 64;
constexpr int64_t kEmbeddingSize = 32;
constexpr int64_t kIndexRows = 32;

using indirect_load_test::SetView;

struct EmbeddingGraphView {
  std::shared_ptr<af::AscGraph> graph;
  af::AxisId input_row_axis;
  af::AxisId input_inner_axis;
  af::AxisId output_row_axis;
  af::AxisId output_inner_axis;
  af::Expression input_rows;
  af::Expression embedding_size;
  af::Expression index_rows;
};

EmbeddingGraphView CreateGraphView() {
  EmbeddingGraphView view;
  view.graph = std::make_shared<af::AscGraph>(kGraphName);
  view.input_rows = view.graph->CreateSizeVar(kInputRows);
  view.embedding_size = view.graph->CreateSizeVar(kEmbeddingSize);
  view.index_rows = view.graph->CreateSizeVar(kIndexRows);
  view.input_row_axis = view.graph->CreateAxis("input_row", view.input_rows).id;
  view.input_inner_axis = view.graph->CreateAxis("embedding_inner", view.embedding_size).id;
  view.output_row_axis = view.graph->CreateAxis("output_row", view.index_rows).id;
  view.output_inner_axis = view.graph->CreateAxis("output_inner", view.embedding_size).id;
  return view;
}

std::shared_ptr<af::AscGraph> CreateSubGraph() {
  const EmbeddingGraphView view = CreateGraphView();
  const std::vector<af::AxisId> input_axes = {view.input_row_axis, view.input_inner_axis};
  const std::vector<af::Expression> input_repeats = {view.input_rows, view.embedding_size};
  const std::vector<af::Expression> input_strides = {view.embedding_size, af::ops::One};
  const std::vector<af::AxisId> output_axes = {view.output_row_axis, view.output_inner_axis};
  const std::vector<af::Expression> output_repeats = {view.index_rows, view.embedding_size};
  const std::vector<af::Expression> output_strides = {view.embedding_size, af::ops::One};
  const std::vector<af::Expression> index_repeats = {view.index_rows, af::ops::One};
  const std::vector<af::Expression> index_strides = {af::ops::One, af::ops::Zero};
  const std::vector<af::Expression> reduce_repeats = {view.index_rows, af::ops::One};
  const std::vector<af::Expression> reduce_strides = {af::ops::One, af::ops::Zero};

  af::ascir_op::Data input("input", *view.graph);
  input.ir_attr.SetIndex(0);
  af::ascir_op::Load input_load("input_load");
  view.graph->AddNode(input_load);
  input_load.x = input.y;
  SetView(input_load, input_axes, input_repeats, input_strides, af::DT_FLOAT);

  af::ascir_op::Data index("index", *view.graph);
  index.ir_attr.SetIndex(1);
  af::ascir_op::Load index_load("index_load");
  view.graph->AddNode(index_load);
  index_load.x = index.y;
  SetView(index_load, output_axes, index_repeats, index_strides, af::DT_INT32);

  af::ascir_op::Cast index_cast("index_cast");
  view.graph->AddNode(index_cast);
  index_cast.x = index_load.y;
  SetView(index_cast, output_axes, index_repeats, index_strides, af::DT_INT64);

  af::ascir_op::Broadcast index_broadcast("index_broadcast");
  view.graph->AddNode(index_broadcast);
  index_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  index_broadcast.x = index_cast.y;
  SetView(index_broadcast, output_axes, output_repeats, output_strides, af::DT_INT64);

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  view.graph->AddNode(indirect_load);
  indirect_load.x1 = input_load.y;
  indirect_load.x2 = index_broadcast.y;
  indirect_load.ir_attr.SetAxis(0);
  SetView(indirect_load, output_axes, output_repeats, output_strides, af::DT_FLOAT);

  af::ascir_op::Scalar add_scalar("add_scalar", *view.graph);
  add_scalar.ir_attr.SetValue("0.1");
  add_scalar.y.dtype = af::DT_FLOAT;
  af::ascir_op::Broadcast add_broadcast("add_broadcast");
  view.graph->AddNode(add_broadcast);
  add_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  add_broadcast.x = add_scalar.y;
  SetView(add_broadcast, output_axes, output_repeats, output_strides, af::DT_FLOAT);

  af::ascir_op::Add add("add");
  view.graph->AddNode(add);
  add.x1 = indirect_load.y;
  add.x2 = add_broadcast.y;
  SetView(add, output_axes, output_repeats, output_strides, af::DT_FLOAT);

  af::ascir_op::Scalar mul_scalar("mul_scalar", *view.graph);
  mul_scalar.ir_attr.SetValue("2.0");
  mul_scalar.y.dtype = af::DT_FLOAT;
  af::ascir_op::Broadcast mul_broadcast("mul_broadcast");
  view.graph->AddNode(mul_broadcast);
  mul_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  mul_broadcast.x = mul_scalar.y;
  SetView(mul_broadcast, output_axes, output_repeats, output_strides, af::DT_FLOAT);

  af::ascir_op::Mul mul("mul");
  view.graph->AddNode(mul);
  mul.x1 = add.y;
  mul.x2 = mul_broadcast.y;
  SetView(mul, output_axes, output_repeats, output_strides, af::DT_FLOAT);

  af::ascir_op::Sum sum("sum");
  view.graph->AddNode(sum);
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.attr.sched.axis = output_axes;
  sum.x = mul.y;
  SetView(sum, output_axes, reduce_repeats, reduce_strides, af::DT_FLOAT);

  af::ascir_op::Store store("store");
  view.graph->AddNode(store);
  store.x = sum.y;
  SetView(store, output_axes, reduce_repeats, reduce_strides, af::DT_FLOAT);

  af::ascir_op::Output output("output");
  view.graph->AddNode(output);
  output.ir_attr.SetIndex(0);
  output.x = store.y;
  SetView(output, output_axes, reduce_repeats, reduce_strides, af::DT_FLOAT);
  return view.graph;
}

af::ComputeGraphPtr CreateGraph() {
  indirect_load_test::BackendGraph backend(kGraphName, "input0", "input1", af::DT_FLOAT, af::DT_INT32);
  return backend.Finalize(CreateSubGraph(), "output0");
}
}  // namespace

using TestBackendIndirectLoadEmbeddingE2e = indirect_load_test::BackendE2e;

TEST_F(TestBackendIndirectLoadEmbeddingE2e, GeneratesEmbeddingIndirectLoadKernel) {
  const auto graph = CreateGraph();
  ASSERT_NE(graph, nullptr);
  codegen::CodegenResult result;
  indirect_load_test::GenerateForTemplate(graph, {}, ascir::TemplateId::kIndirectLoadSimd, result);
  EXPECT_NE(result.kernel.find("// IndirectLoad SIMD"), std::string::npos);
  EXPECT_NE(result.kernel.find("CastExtend"), std::string::npos);
  EXPECT_NE(result.kernel.find("IndirectLoadSimd<"), std::string::npos);
  indirect_load_test::WriteGeneratedFiles(result);
}

#endif
