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
constexpr char kGraphName[] = "indirect_load_torch_gather_strided_test";
constexpr bool kExpectSimt = IL_EXPECT_SIMT;
constexpr bool kExpectSk = IL_EXPECT_SK;
constexpr int64_t kInputStride0 = IL_INPUT_STRIDE0;
constexpr int64_t kInputStride1 = IL_INPUT_STRIDE1;
constexpr int64_t kInputStride2 = IL_INPUT_STRIDE2;
constexpr int64_t kIndexStride0 = IL_INDEX_STRIDE0;
constexpr int64_t kIndexStride1 = IL_INDEX_STRIDE1;
constexpr int64_t kIndexStride2 = IL_INDEX_STRIDE2;

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
  view.output_sizes = {af::Symbol(8), af::Symbol(16), af::Symbol(5)};
  view.output_strides = {af::Symbol(80), af::Symbol(5), af::Symbol(1)};
  view.index_strides = {af::Symbol(kIndexStride0), af::Symbol(kIndexStride1), af::Symbol(kIndexStride2)};
  view.input_sizes = {af::Symbol(8), af::Symbol(32), af::Symbol(5)};
  view.input_strides = {af::Symbol(kInputStride0), af::Symbol(kInputStride1), af::Symbol(kInputStride2)};
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
  SetView(index_load, view.axes, view.output_sizes, view.index_strides, af::DT_INT64);

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
  indirect_load.x2 = index_load.y;
  indirect_load.ir_attr.SetAxis(1);
  indirect_load.ir_attr.SetNegative_index_support(true);
  indirect_load.ir_attr.SetNeed_check_bound(true);
  indirect_load.ir_attr.SetMax(32);
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
    EXPECT_NE(result.kernel.find("IndirectLoadSimtStridedPolicy<uint32_t, 3, 1, 7ULL, 7ULL>"), std::string::npos);
    EXPECT_EQ(result.kernel.find("x_axis_size"), std::string::npos);
    EXPECT_EQ(result.kernel.find("indirect_index < 0"), std::string::npos);
  }
  indirect_load_test::WriteGeneratedFiles(result);
}
