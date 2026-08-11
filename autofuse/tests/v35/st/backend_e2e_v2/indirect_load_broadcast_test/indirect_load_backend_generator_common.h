/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_INDIRECT_LOAD_BROADCAST_TEST_INDIRECT_LOAD_BACKEND_GENERATOR_COMMON_H_
#define AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_INDIRECT_LOAD_BROADCAST_TEST_INDIRECT_LOAD_BACKEND_GENERATOR_COMMON_H_

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
  BackendGraph(const char *graph_name, const char *data_name, const char *index_name, af::DataType data_type)
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
    index_desc->SetDataType(af::DT_INT64);
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

#endif  // AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_INDIRECT_LOAD_BROADCAST_TEST_INDIRECT_LOAD_BACKEND_GENERATOR_COMMON_H_
