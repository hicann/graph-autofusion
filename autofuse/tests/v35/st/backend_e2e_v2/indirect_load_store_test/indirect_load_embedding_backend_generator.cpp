/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indirect_load_backend_generator_common.h"

namespace {
constexpr char kGraphName[] = "indirect_load_embedding_test";
constexpr int64_t kInputRows = 100;
constexpr int64_t kEmbeddingSize = 16;
constexpr int64_t kIndexRows = 92;

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
  SetView(index_load, output_axes, index_repeats, index_strides, af::DT_INT64);

  af::ascir_op::Broadcast index_broadcast("index_broadcast");
  view.graph->AddNode(index_broadcast);
  index_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  index_broadcast.x = index_load.y;
  SetView(index_broadcast, output_axes, output_repeats, output_strides, af::DT_INT64);

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  view.graph->AddNode(indirect_load);
  indirect_load.x1 = input_load.y;
  indirect_load.x2 = index_broadcast.y;
  indirect_load.ir_attr.SetAxis(0);
  SetView(indirect_load, output_axes, output_repeats, output_strides, af::DT_FLOAT);

  af::ascir_op::Store store("store");
  view.graph->AddNode(store);
  store.x = indirect_load.y;
  SetView(store, output_axes, output_repeats, output_strides, af::DT_FLOAT);

  af::ascir_op::Output output("output");
  view.graph->AddNode(output);
  output.ir_attr.SetIndex(0);
  output.x = store.y;
  SetView(output, output_axes, output_repeats, output_strides, af::DT_FLOAT);
  return view.graph;
}

af::ComputeGraphPtr CreateGraph() {
  indirect_load_test::BackendGraph backend(kGraphName, "input0", "input1", af::DT_FLOAT);
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
  EXPECT_NE(result.kernel.find("IndirectLoadSimd<"), std::string::npos);
  indirect_load_test::WriteGeneratedFiles(result);
}
