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
constexpr int64_t kRows = 6400;
constexpr int64_t kColumns = 32;
constexpr int64_t kTableRows = 315511;
constexpr char kGraphName[] = "indirect_load_broadcast_index_where_simt_test";

using indirect_load_test::SetView;

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
