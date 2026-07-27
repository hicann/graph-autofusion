/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>

#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_factory_af.h"
#include "graph/utils/op_desc_utils.h"
#include "ascir_ops.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#define private public
#include "fused_graph/fused_graph_unfolder.h"
#undef private
#include "ascgraph_info_complete.h"
#include "graph/debug/ge_op_types.h"
#include "ascgen_log.h"
#include "schedule_utils.h"
#include "platform_context.h"
#include "platform/v1/platformv1.h"

namespace optimize {
using namespace af;
class FusedGraphUnfolderTest : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};

namespace {
class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) {
    graph_ = std::make_shared<ComputeGraph>(name);
  }

  GraphBuilder(const std::string &name, const std::string &node_type) {
    graph_ = std::make_shared<ComputeGraph>(name);
    node_type_ = node_type;
  }

  NodePtr AddNode(const std::string &name, const std::string &type, const int in_cnt, const int out_cnt,
                  const std::vector<int64_t> shape = {1, 1, 1, 1}) {
    auto tensor_desc = std::make_shared<GeTensorDesc>();
    tensor_desc->SetShape(GeShape(std::move(shape)));
    tensor_desc->SetFormat(ge::FORMAT_NCHW);
    tensor_desc->SetDataType(ge::DT_FLOAT);
    tensor_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>();

    auto op_desc = std::make_shared<OpDesc>(name, (node_type_ == "") ? type : kAscGraphNodeType);
    for (std::int32_t i = 0; i < in_cnt; ++i) {
      op_desc->AddInputDesc(tensor_desc->Clone());
    }
    for (std::int32_t i = 0; i < out_cnt; ++i) {
      op_desc->AddOutputDesc(tensor_desc->Clone());
    }
    op_desc->AddInferFunc([](Operator &op) { return ge::GRAPH_SUCCESS; });
    return graph_->AddNode(op_desc);
  }

  void AddDataEdge(const NodePtr &src_node, const std::int32_t src_idx, const NodePtr &dst_node,
                   const std::int32_t dst_idx) {
    GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_idx), dst_node->GetInDataAnchor(dst_idx));
  }

  NodePtr AddNodeByIr(const std::string &op_name, const std::string &op_type) {
    auto op = OperatorFactory::CreateOperator(op_name.c_str(), op_type.c_str());
    if (op.IsEmpty()) {
      return nullptr;
    }
    OpDescPtr op_desc = af::OpDescUtils::GetOpDescFromOperator(op);
    return graph_->AddNode(op_desc);
  }

  void AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node) {
    GraphUtils::AddEdge(src_node->GetOutControlAnchor(), dst_node->GetInControlAnchor());
  }

  ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

  static void AddSubgraph(const ComputeGraphPtr &graph, const string &call_name, const ComputeGraphPtr &subgraph) {
    const auto &call_node = graph->FindNode(call_name);
    if (call_node == nullptr) {
      return;
    }
    call_node->GetOpDesc()->RegisterSubgraphIrName("f", SubgraphType::kStatic);

    size_t index = call_node->GetOpDesc()->GetSubgraphInstanceNames().size();
    call_node->GetOpDesc()->AddSubgraphName(subgraph->GetName());
    call_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph->GetName());

    subgraph->SetParentNode(call_node);
    subgraph->SetParentGraph(graph);
    GraphUtils::FindRootGraph(graph)->AddSubgraph(subgraph);
  }

 private:
  ComputeGraphPtr graph_;
  std::string node_type_;
};

/**
 *          NetOutput
 *            |
 *          AscBc4
 *            |
 *          AscBc3
 *        /       / \
 *      AscBc1    AscBc2
 *    /   \         /   \.
 * data0  data1   data2 data3
 */
ComputeGraphPtr BuildFusedGraph(const std::string node_type = "") {
  auto builder = GraphBuilder("test", node_type);
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDesc(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", 2);
  auto data3 = builder.AddNode("data3", "Data", 0, 1);
  af::AttrUtils::SetInt(data3->GetOpDesc(), "_parent_node_index", 3);

  auto ascbc1 = builder.AddNode("ascbc1", kAscGraphNodeType, 2, 1);
  auto ascbc2 = builder.AddNode("ascbc2", kAscGraphNodeType, 2, 2);
  auto ascbc3 = builder.AddNode("ascbc3", kAscGraphNodeType, 3, 1);
  auto ascbc4 = builder.AddNode("ascbc4", kAscGraphNodeType, 1, 1);

  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 2, 0);

  builder.AddDataEdge(data0, 0, ascbc1, 0);
  builder.AddDataEdge(data1, 0, ascbc1, 1);
  builder.AddDataEdge(data2, 0, ascbc2, 0);
  builder.AddDataEdge(data3, 0, ascbc2, 1);

  builder.AddDataEdge(ascbc1, 0, ascbc3, 0);
  builder.AddDataEdge(ascbc2, 0, ascbc3, 1);
  builder.AddDataEdge(ascbc2, 1, ascbc3, 2);

  builder.AddDataEdge(ascbc3, 0, ascbc4, 0);
  builder.AddDataEdge(ascbc4, 0, netoutput1, 0);

  return builder.GetGraph();
}

/**
 *         NetOutput
 *            |
 *          AscBc3
 *        /     /\
 *    AscBc1   AscBc2
 *    /   \   /    \
 * data0  data1   data2
 */
ComputeGraphPtr BuildFusedGraphWithSharedData(const std::string node_type = "") {
  auto builder = GraphBuilder("BuildFusedGraphWithSharedData", node_type);
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDesc(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 2);
  af::AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", 2);

  auto ascbc1 = builder.AddNode("ascbc1", kAscGraphNodeType, 2, 1);
  auto ascbc2 = builder.AddNode("ascbc2", kAscGraphNodeType, 2, 2);
  auto ascbc3 = builder.AddNode("ascbc3", kAscGraphNodeType, 3, 1);

  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc1, 0);
  builder.AddDataEdge(data1, 0, ascbc1, 1);
  builder.AddDataEdge(data1, 1, ascbc2, 0);
  builder.AddDataEdge(data2, 0, ascbc2, 1);

  builder.AddDataEdge(ascbc1, 0, ascbc3, 0);
  builder.AddDataEdge(ascbc2, 0, ascbc3, 1);
  builder.AddDataEdge(ascbc2, 1, ascbc3, 2);

  builder.AddDataEdge(ascbc3, 0, netoutput1, 0);

  return builder.GetGraph();
}

/**
 *         NetOutput
 *            |
 *          AscBc3
 *        /     /\
 *    AscBc1   AscBc2
 *    /   \   / /
 * data0  data1
 */
ComputeGraphPtr BuildFusedGraphWithSharedDataWithTwoData(const std::string node_type = "") {
  auto builder = GraphBuilder("BuildFusedGraphWithSharedData", node_type);
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDesc(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 3);
  af::AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", 1);

  auto ascbc1 = builder.AddNode("ascbc1", kAscGraphNodeType, 2, 1);
  auto ascbc2 = builder.AddNode("ascbc2", kAscGraphNodeType, 2, 2);
  auto ascbc3 = builder.AddNode("ascbc3", kAscGraphNodeType, 3, 1);

  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc1, 0);
  builder.AddDataEdge(data1, 0, ascbc1, 1);
  builder.AddDataEdge(data1, 1, ascbc2, 0);
  builder.AddDataEdge(data1, 2, ascbc2, 1);

  builder.AddDataEdge(ascbc1, 0, ascbc3, 0);
  builder.AddDataEdge(ascbc2, 0, ascbc3, 1);
  builder.AddDataEdge(ascbc2, 1, ascbc3, 2);

  builder.AddDataEdge(ascbc3, 0, netoutput1, 0);

  return builder.GetGraph();
}

/**
 *         NetOutput
 *        /    /\
 *    AscBc1  AscBc2
 *    /   \   /    \
 * data0  data1   data2
 */
ComputeGraphPtr BuildFusedGraphWithMultiOutput(const std::string node_type = "") {
  auto builder = GraphBuilder("BuildFusedGraphWithMultiOutput", node_type);
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDesc(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", 2);

  auto ascbc1 = builder.AddNode("ascbc1", kAscGraphNodeType, 2, 1);
  auto ascbc2 = builder.AddNode("ascbc2", kAscGraphNodeType, 2, 2);

  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 3, 0);

  builder.AddDataEdge(data0, 0, ascbc1, 0);
  builder.AddDataEdge(data1, 0, ascbc1, 1);
  builder.AddDataEdge(data1, 0, ascbc2, 0);
  builder.AddDataEdge(data2, 0, ascbc2, 1);

  builder.AddDataEdge(ascbc1, 0, netoutput1, 0);
  builder.AddDataEdge(ascbc2, 0, netoutput1, 1);
  builder.AddDataEdge(ascbc2, 1, netoutput1, 2);
  return builder.GetGraph();
}

/**
 *      NetOutput
 *       |     |
 *       |  AscBc3
 *       | /   /\
 *    AscBc1  AscBc2
 *    /   \   /    \
 * data0  data1   data2
 */
ComputeGraphPtr BuildFusedGraphWithReuseOutput(const std::string node_type = "") {
  auto builder = GraphBuilder("BuildFusedGraphWithReuseOutput", node_type);
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDesc(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", 2);

  auto ascbc1 = builder.AddNode("ascbc1", kAscGraphNodeType, 2, 1);
  auto ascbc2 = builder.AddNode("ascbc2", kAscGraphNodeType, 2, 2);
  auto ascbc3 = builder.AddNode("ascbc3", kAscGraphNodeType, 3, 1);

  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 2, 0);

  builder.AddDataEdge(data0, 0, ascbc1, 0);
  builder.AddDataEdge(data1, 0, ascbc1, 1);
  builder.AddDataEdge(data1, 0, ascbc2, 0);
  builder.AddDataEdge(data2, 0, ascbc2, 1);

  builder.AddDataEdge(ascbc1, 0, netoutput1, 0);
  builder.AddDataEdge(ascbc1, 0, ascbc3, 0);
  builder.AddDataEdge(ascbc2, 0, ascbc3, 1);
  builder.AddDataEdge(ascbc2, 1, ascbc3, 2);
  builder.AddDataEdge(ascbc3, 0, netoutput1, 1);
  return builder.GetGraph();
}

void CreateAddAscGraph(af::AscGraph &graph) {
  auto ONE = af::Symbol(1);
  const af::Expression s0 = af::Symbol("s0");
  const af::Expression s1 = af::Symbol("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x1("sub1_data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.attr.sched.axis = {z0.id, z1.id};
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, ONE};
  x1.y.dtype = ge::DT_INT8;

  af::ascir_op::Load x1Local("sub1_load0");
  x1Local.ir_attr.SetOffset(af::Symbol(0));
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {z0.id, z1.id};
  *x1Local.y.axis = {z0.id, z1.id};
  *x1Local.y.repeats = {s0, s1};
  *x1Local.y.strides = {s1, ONE};

  af::ascir_op::Data x2("sub1_data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.attr.sched.axis = {z0.id, z1.id};
  *x2.y.axis = {z0.id, z1.id};
  *x2.y.repeats = {s0, s1};
  *x2.y.strides = {s1, ONE};

  af::ascir_op::Load x2Local("sub1_load1");
  x2Local.ir_attr.SetOffset(af::Symbol(0));
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {z0.id, z1.id};
  *x2Local.y.axis = {z0.id, z1.id};
  *x2Local.y.repeats = {s0, s1};
  *x2Local.y.strides = {s1, ONE};

  af::ascir_op::Add add("sub1_add0");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, ONE};

  af::ascir_op::Store x_out("sub1_store0");
  x_out.x = add.y;
  x_out.attr.sched.axis = {z0.id, z1.id};
  *x_out.y.axis = {z0.id, z1.id};
  *x_out.y.repeats = {s0, s1};
  *x_out.y.strides = {s1, ONE};

  af::ascir_op::Output y("sub1_out0");
  y.x = x_out.y;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);
}

void CreateAddAscGraphOneDim(af::AscGraph &graph) {
  auto ONE = af::Symbol(1);
  const af::Expression s0 = graph.CreateSizeVar("s0");
  const af::Expression s1 = graph.CreateSizeVar("s1");
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x1("sub1_data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.attr.sched.axis = {z1.id};
  *x1.y.axis = {z1.id};
  *x1.y.repeats = {s1};
  *x1.y.strides = {ONE};

  af::ascir_op::Load x1Local("sub1_load0");
  x1Local.ir_attr.SetOffset(af::Symbol(0));
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {z1.id};
  *x1Local.y.axis = {z1.id};
  *x1Local.y.repeats = {s1};
  *x1Local.y.strides = {ONE};

  af::ascir_op::Data x2("sub1_data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.attr.sched.axis = {z1.id};
  *x2.y.axis = {z1.id};
  *x2.y.repeats = {s1};
  *x2.y.strides = {ONE};

  af::ascir_op::Load x2Local("sub1_load1");
  x2Local.ir_attr.SetOffset(af::Symbol(0));
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {z1.id};
  *x2Local.y.axis = {z1.id};
  *x2Local.y.repeats = {s1};
  *x2Local.y.strides = {ONE};

  af::ascir_op::Add add("sub1_add0");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.attr.sched.axis = {z1.id};
  *add.y.axis = {z1.id};
  *add.y.repeats = {s1};
  *add.y.strides = {ONE};

  af::ascir_op::Store x_out("sub1_store0");
  x_out.x = add.y;
  x_out.attr.sched.axis = {z1.id};
  *x_out.y.axis = {z1.id};
  *x_out.y.repeats = {s1};
  *x_out.y.strides = {ONE};

  af::ascir_op::Output y("sub1_out0");
  y.x = x_out.y;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);
}

void CreateAddAscGraph2(af::AscGraph &graph, const int64_t load1_offset = 0) {
  auto ONE = af::Symbol(1);
  const af::Expression s0 = af::Symbol("s0");
  const af::Expression s2 = af::Symbol("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s2);

  af::ascir_op::Data x1("sub2_data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.attr.sched.axis = {z0.id, z1.id};
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s2};
  *x1.y.strides = {s2, ONE};

  af::ascir_op::Load x1Local("sub2_load0");
  x1Local.ir_attr.SetOffset(af::Symbol(load1_offset));
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {z0.id, z1.id};
  *x1Local.y.axis = {z0.id, z1.id};
  *x1Local.y.repeats = {s0, s2};
  *x1Local.y.strides = {s2, ONE};

  af::ascir_op::Data x2("sub2_data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.attr.sched.axis = {z0.id, z1.id};
  *x2.y.axis = {z0.id, z1.id};
  *x2.y.repeats = {s0, s2};
  *x2.y.strides = {s2, ONE};

  af::ascir_op::Load x2Local("sub2_load1");
  x2Local.ir_attr.SetOffset(af::Symbol(0));
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {z0.id, z1.id};
  *x2Local.y.axis = {z0.id, z1.id};
  *x2Local.y.repeats = {s0, s2};
  *x2Local.y.strides = {s2, ONE};

  af::ascir_op::Add add("sub2_add0");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s2};
  *add.y.strides = {s2, ONE};

  af::ascir_op::Store x_out("sub2_store0");
  x_out.x = add.y;
  x_out.attr.sched.axis = {z0.id, z1.id};
  *x_out.y.axis = {z0.id, z1.id};
  *x_out.y.repeats = {s0, s2};
  *x_out.y.strides = {s2, ONE};

  af::ascir_op::Output y("sub2_out0");
  y.x = x_out.y;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);

  af::ascir_op::Store store2("sub2_store1");
  store2.x = add.y;
  store2.attr.sched.axis = {z0.id, z1.id};
  *store2.y.axis = {z0.id, z1.id};
  *store2.y.repeats = {s0, s2};
  *store2.y.strides = {s2, ONE};

  af::ascir_op::Output y2("sub2_out1");
  y2.x = store2.y;
  y2.y.dtype = ge::DT_FLOAT16;
  y2.ir_attr.SetIndex(1);
}

void CreateAddAscGraph3(af::AscGraph &graph, const int64_t load1_offset = 0) {
  auto ONE = af::Symbol(1);
  const af::Expression s0 = af::Symbol("s0");
  const af::Expression s2 = af::Symbol("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s2);

  af::ascir_op::Data x1("sub2_data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.attr.sched.axis = {z0.id, z1.id};
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s2};
  *x1.y.strides = {s2, ONE};

  af::ascir_op::Load x1Local("sub2_load0");
  x1Local.ir_attr.SetOffset(af::Symbol(load1_offset));
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {z0.id, z1.id};
  *x1Local.y.axis = {z0.id, z1.id};
  *x1Local.y.repeats = {s0, s2};
  *x1Local.y.strides = {s2, ONE};

  af::ascir_op::Data x2("sub2_data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.attr.sched.axis = {z0.id, z1.id};
  *x2.y.axis = {z0.id, z1.id};
  *x2.y.repeats = {s0, s2};
  *x2.y.strides = {s2, ONE};

  af::ascir_op::Load x2Local("sub2_load1");
  x2Local.ir_attr.SetOffset(af::Symbol(0));
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {z0.id, z1.id};
  *x2Local.y.axis = {z0.id, z1.id};
  *x2Local.y.repeats = {s0, s2};
  *x2Local.y.strides = {s2, ONE};

  af::ascir_op::Concat concat("concat");
  concat.x = {x1Local.y, x2Local.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s2 + s2};
  *concat.y.strides = {s2 + s2, ONE};
  concat.attr.api.compute_type = af::ComputeType::kComputeConcat;

  af::ascir_op::Store x_out("sub2_store0");
  x_out.x = concat.y;
  x_out.attr.sched.axis = {z0.id, z1.id};
  *x_out.y.axis = {z0.id, z1.id};
  *x_out.y.repeats = {s0, s2 + s2};
  *x_out.y.strides = {s2 + s2, ONE};

  af::ascir_op::Output y("sub2_out0");
  y.x = x_out.y;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);

  af::ascir_op::Store store2("sub2_store1");
  store2.x = concat.y;
  store2.attr.sched.axis = {z0.id, z1.id};
  *store2.y.axis = {z0.id, z1.id};
  *store2.y.repeats = {s0, s2 + s2};
  *store2.y.strides = {s2 + s2, ONE};

  af::ascir_op::Output y2("sub2_out1");
  y2.x = store2.y;
  y2.y.dtype = ge::DT_FLOAT16;
  y2.ir_attr.SetIndex(1);
}

void CreatePackFirstDimAscGraph(af::AscGraph &graph, const int64_t load1_offset = 0) {
  auto ONE = af::Symbol(1);
  const af::Expression s2 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", af::Symbol(2));
  auto z1 = graph.CreateAxis("z1", s2);

  af::ascir_op::Data x1("sub2_data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.attr.sched.axis = {z0.id, z1.id};
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {af::sym::kSymbolOne, s2};
  *x1.y.strides = {s2, ONE};

  af::ascir_op::Load x1Local("sub2_load0");
  x1Local.ir_attr.SetOffset(af::Symbol(load1_offset));
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {z0.id, z1.id};
  *x1Local.y.axis = {z0.id, z1.id};
  *x1Local.y.repeats = {af::sym::kSymbolOne, s2};
  *x1Local.y.strides = {s2, ONE};

  af::ascir_op::Data x2("sub2_data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.attr.sched.axis = {z0.id, z1.id};
  *x2.y.axis = {z0.id, z1.id};
  *x2.y.repeats = {af::sym::kSymbolOne, s2};
  *x2.y.strides = {s2, ONE};

  af::ascir_op::Load x2Local("sub2_load1");
  x2Local.ir_attr.SetOffset(af::Symbol(0));
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {z0.id, z1.id};
  *x2Local.y.axis = {z0.id, z1.id};
  *x2Local.y.repeats = {af::sym::kSymbolOne, s2};
  *x2Local.y.strides = {s2, ONE};

  af::ascir_op::Concat concat("concat");
  concat.x = {x1Local.y, x2Local.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {af::Symbol(2), s2};
  *concat.y.strides = {s2, ONE};
  concat.attr.api.compute_type = af::ComputeType::kComputeConcat;

  af::ascir_op::Store x_out("sub2_store0");
  x_out.x = concat.y;
  x_out.attr.sched.axis = {z0.id, z1.id};
  *x_out.y.axis = {z0.id, z1.id};
  *x_out.y.repeats = {af::Symbol(2), s2};
  *x_out.y.strides = {s2, ONE};

  af::ascir_op::Output y("sub2_out0");
  y.x = x_out.y;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);

  af::ascir_op::Store store2("sub2_store1");
  store2.x = concat.y;
  store2.attr.sched.axis = {z0.id, z1.id};
  *store2.y.axis = {z0.id, z1.id};
  *store2.y.repeats = {af::Symbol(2), s2};
  *store2.y.strides = {s2, ONE};

  af::ascir_op::Output y2("sub2_out1");
  y2.x = store2.y;
  y2.y.dtype = ge::DT_FLOAT16;
  y2.ir_attr.SetIndex(1);
}

void CreateAddAscGraph3SameData(af::AscGraph &graph, const int64_t load1_offset = 0) {
  auto ONE = af::Symbol(1);
  const af::Expression s0 = af::Symbol("s0");
  const af::Expression s2 = af::Symbol("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s2);

  af::ascir_op::Data x1("sub2_data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.attr.sched.axis = {z0.id, z1.id};
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s2};
  *x1.y.strides = {s2, ONE};

  af::ascir_op::Load x1Local("sub2_load0");
  x1Local.ir_attr.SetOffset(af::Symbol(load1_offset));
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {z0.id, z1.id};
  *x1Local.y.axis = {z0.id, z1.id};
  *x1Local.y.repeats = {s0, s2};
  *x1Local.y.strides = {s2, ONE};

  af::ascir_op::Data x2("sub2_data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.attr.sched.axis = {z0.id, z1.id};
  *x2.y.axis = {z0.id, z1.id};
  *x2.y.repeats = {s0, s2};
  *x2.y.strides = {s2, ONE};

  af::ascir_op::Load x2Local("sub2_load1");
  x2Local.ir_attr.SetOffset(af::Symbol(0));
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {z0.id, z1.id};
  *x2Local.y.axis = {z0.id, z1.id};
  *x2Local.y.repeats = {s0, s2};
  *x2Local.y.strides = {s2, ONE};

  af::ascir_op::Add add("sub2_add0");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s2};
  *add.y.strides = {s2, ONE};

  af::ascir_op::Store x_out("sub2_store0");
  x_out.x = add.y;
  x_out.attr.sched.axis = {z0.id, z1.id};
  *x_out.y.axis = {z0.id, z1.id};
  *x_out.y.repeats = {s0, s2};
  *x_out.y.strides = {s2, ONE};

  af::ascir_op::Output y("sub2_out0");
  y.x = x_out.y;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);

  af::ascir_op::Store store2("sub2_store1");
  store2.x = add.y;
  store2.attr.sched.axis = {z0.id, z1.id};
  *store2.y.axis = {z0.id, z1.id};
  *store2.y.repeats = {s0, s2};
  *store2.y.strides = {s2, ONE};

  af::ascir_op::Output y2("sub2_out1");
  y2.x = store2.y;
  y2.y.dtype = ge::DT_FLOAT16;
  y2.ir_attr.SetIndex(1);
}

void CreateConcatInputChain(af::ascir_op::Data &data, af::ascir_op::Load &load, const af::Axis &z0, const af::Axis &z1,
                            const af::Expression &s0, const af::Expression &input_dim, const af::Expression &stride) {
  auto ONE = af::Symbol(1);
  data.attr.sched.axis = {z0.id, z1.id};
  *data.y.axis = {z0.id, z1.id};
  *data.y.repeats = {s0, input_dim};
  *data.y.strides = {stride, ONE};
  load.ir_attr.SetOffset(af::Symbol(0));
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, input_dim};
  *load.y.strides = {stride, ONE};
}

void CreateConcatOutputChain(af::ascir_op::Concat &concat, af::ascir_op::Store &store, af::ascir_op::Output &output,
                             const af::Axis &z0, const af::Axis &z1, const af::Expression &s0,
                             const af::Expression &total_dim) {
  auto ONE = af::Symbol(1);
  concat.attr.sched.axis = {z0.id, z1.id};
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, total_dim};
  *concat.y.strides = {total_dim, ONE};
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, total_dim};
  *store.y.strides = {total_dim, ONE};
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;
  output.ir_attr.SetIndex(0);
}

void CreateConcatAscGraph(af::AscGraph &graph, const bool same_data = false) {
  const af::Expression s0 = af::Symbol("s0");
  const af::Expression s1 = af::Symbol("s1");
  const af::Expression s2 = af::Symbol("s2");
  const auto input_dim = same_data ? s1 : s2;
  const auto total_dim = s1 + input_dim + input_dim;
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", total_dim);
  af::ascir_op::Data x1("concat_data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Load x1Local("concat_load0");
  CreateConcatInputChain(x1, x1Local, z0, z1, s0, s1, s1);
  af::ascir_op::Data x2("concat_data1", graph);
  x2.ir_attr.SetIndex(1);
  af::ascir_op::Load x2Local("concat_load1");
  CreateConcatInputChain(x2, x2Local, z0, z1, s0, input_dim, input_dim);
  af::ascir_op::Data concat_data2("concat_data2", graph);
  concat_data2.ir_attr.SetIndex(2);
  af::ascir_op::Load concat_load2("concat_load2");
  CreateConcatInputChain(concat_data2, concat_load2, z0, z1, s0, input_dim, input_dim);
  af::ascir_op::Concat concat("concat");
  concat.x = {x1Local.y, x2Local.y, concat_load2.y};
  af::ascir_op::Store x_out("concat_store");
  af::ascir_op::Output y("concat_out");
  CreateConcatOutputChain(concat, x_out, y, z0, z1, s0, total_dim);
  AscGraphInfoComplete::CompleteApiInfo(graph);
}

void CreatePostAscGraph(af::AscGraph &graph) {
  auto ONE = af::Symbol(1);
  const af::Expression s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Data x1("post_data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.attr.sched.axis = {z0.id};
  *x1.y.axis = {z0.id};
  *x1.y.repeats = {s0};
  *x1.y.strides = {ONE};

  af::ascir_op::Load load0("post_load0");
  load0.x = x1.y;
  load0.attr.sched.axis = {z0.id};
  *load0.y.axis = {z0.id};
  *load0.y.repeats = {s0};
  *load0.y.strides = {ONE};

  af::ascir_op::Abs abs("post_abs");
  abs.x = load0.y;
  abs.attr.sched.axis = {z0.id};
  *abs.y.axis = {z0.id};
  *abs.y.repeats = {s0};
  *abs.y.strides = {ONE};

  af::ascir_op::Store x_out("post_store");
  x_out.x = abs.y;
  x_out.attr.sched.axis = {z0.id};
  *x_out.y.axis = {z0.id};
  *x_out.y.repeats = {s0};
  *x_out.y.strides = {ONE};

  af::ascir_op::Output y("post_out");
  y.x = x_out.y;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);

  AscGraphInfoComplete::CompleteApiInfo(graph);
}

}  // namespace

/**
 *          NetOutput
 *            |
 *          AscBc4
 *            |
 *          AscBc3
 *        /       / \
 *      AscBc1    AscBc2
 *    /   \         /   \.
 * data0  data1   data2 data3
 */
TEST_F(FusedGraphUnfolderTest, AscBcNodeUnfolder_With_2_Add_Node) {
  ComputeGraphPtr compute_graph = BuildFusedGraph();
  ASSERT_NE(compute_graph, nullptr);
  std::map<Node *, af::AscGraph> asc_backend_to_asc_graph;

  auto ascbc1 = compute_graph->FindNode("ascbc1");
  ASSERT_NE(ascbc1, nullptr);
  auto ascbc2 = compute_graph->FindNode("ascbc2");
  ASSERT_NE(ascbc2, nullptr);
  auto ascbc3 = compute_graph->FindNode("ascbc3");
  ASSERT_NE(ascbc3, nullptr);
  auto ascbc4 = compute_graph->FindNode("ascbc4");
  ASSERT_NE(ascbc4, nullptr);

  af::AscGraph add_sub_graph1("add1");
  af::AscGraph add_sub_graph2("add2");
  af::AscGraph concat_sub_graph("concat");
  af::AscGraph concat_post_sub_graph("concat_post");

  CreateAddAscGraph(add_sub_graph1);
  CreateAddAscGraph2(add_sub_graph2);
  CreateConcatAscGraph(concat_sub_graph);
  CreatePostAscGraph(concat_post_sub_graph);

  asc_backend_to_asc_graph.emplace(ascbc1.get(), add_sub_graph1);
  asc_backend_to_asc_graph.emplace(ascbc2.get(), add_sub_graph2);
  asc_backend_to_asc_graph.emplace(ascbc3.get(), concat_sub_graph);
  asc_backend_to_asc_graph.emplace(ascbc4.get(), concat_post_sub_graph);

  af::AscGraph unfolded_asc_graph("unfolded_asc_graph");
  Status ret = FusedGraphUnfolder::UnfoldFusedGraph(compute_graph, asc_backend_to_asc_graph, unfolded_asc_graph);
  ASSERT_EQ(ret, 0);
  ::ascir::utils::DumpGraph(unfolded_asc_graph, "after_unfold");

  auto axis = unfolded_asc_graph.GetAllAxis();
  ASSERT_EQ(axis.size(), 2);
  EXPECT_EQ(axis[0]->size, concat_sub_graph.GetAllAxis()[0]->size);
  EXPECT_EQ(axis[1]->size, concat_sub_graph.GetAllAxis()[1]->size);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 14UL);
}

/**
 *         NetOutput
 *            |
 *          AscBc3
 *        /    /\
 *    AscBc1  AscBc2
 *    /   \   /    \
 * data0  data1   data2
 */
TEST_F(FusedGraphUnfolderTest, AscBcNodeUnfolder_With_Same_Data_Diff_Repeats) {
  ComputeGraphPtr compute_graph = BuildFusedGraphWithSharedData();
  ASSERT_NE(compute_graph, nullptr);
  std::map<Node *, af::AscGraph> asc_backend_to_asc_graph;

  auto ascbc1 = compute_graph->FindNode("ascbc1");
  ASSERT_NE(ascbc1, nullptr);
  auto ascbc2 = compute_graph->FindNode("ascbc2");
  ASSERT_NE(ascbc2, nullptr);
  auto ascbc3 = compute_graph->FindNode("ascbc3");
  ASSERT_NE(ascbc3, nullptr);

  af::AscGraph add_sub_graph1("sub1_add");
  af::AscGraph add_sub_graph2("sub2_add");
  af::AscGraph concat_sub_graph("sub3_concat");

  CreateAddAscGraph(add_sub_graph1);
  CreateAddAscGraph2(add_sub_graph2);
  CreateConcatAscGraph(concat_sub_graph);

  asc_backend_to_asc_graph.emplace(ascbc1.get(), add_sub_graph1);
  asc_backend_to_asc_graph.emplace(ascbc2.get(), add_sub_graph2);
  asc_backend_to_asc_graph.emplace(ascbc3.get(), concat_sub_graph);

  af::AscGraph unfolded_asc_graph("unfolded_asc_graph");
  Status ret = FusedGraphUnfolder::UnfoldFusedGraph(compute_graph, asc_backend_to_asc_graph, unfolded_asc_graph);
  ASSERT_EQ(ret, af::SUCCESS);

  auto axis = unfolded_asc_graph.GetAllAxis();
  ASSERT_EQ(axis.size(), 2);
  EXPECT_EQ(axis[0]->size, concat_sub_graph.GetAllAxis()[0]->size);
  EXPECT_EQ(axis[1]->size, concat_sub_graph.GetAllAxis()[1]->size);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 12UL);
}

/**
 *         NetOutput
 *            |
 *          AscBc3
 *        /    /\
 *    AscBc1  AscBc2
 *    /   \   /    \
 * data0  data1   data2
 */
TEST_F(FusedGraphUnfolderTest, AscBcNodeUnfolder_With_Same_Data_Diff_Offset) {
  ComputeGraphPtr compute_graph = BuildFusedGraphWithSharedData();
  ASSERT_NE(compute_graph, nullptr);
  std::map<Node *, af::AscGraph> asc_backend_to_asc_graph;

  auto ascbc1 = compute_graph->FindNode("ascbc1");
  ASSERT_NE(ascbc1, nullptr);
  auto ascbc2 = compute_graph->FindNode("ascbc2");
  ASSERT_NE(ascbc2, nullptr);
  auto ascbc3 = compute_graph->FindNode("ascbc3");
  ASSERT_NE(ascbc3, nullptr);

  af::AscGraph add_sub_graph1("sub1_add");
  af::AscGraph add_sub_graph2("sub2_add");
  af::AscGraph concat_sub_graph("sub3_concat");

  CreateAddAscGraph(add_sub_graph1);
  CreateAddAscGraph2(add_sub_graph2, 1);
  CreateConcatAscGraph(concat_sub_graph);

  asc_backend_to_asc_graph.emplace(ascbc1.get(), add_sub_graph1);
  asc_backend_to_asc_graph.emplace(ascbc2.get(), add_sub_graph2);
  asc_backend_to_asc_graph.emplace(ascbc3.get(), concat_sub_graph);

  af::AscGraph unfolded_asc_graph("unfolded_asc_graph");
  Status ret = FusedGraphUnfolder::UnfoldFusedGraph(compute_graph, asc_backend_to_asc_graph, unfolded_asc_graph);
  ASSERT_EQ(ret, af::SUCCESS);

  auto axis = unfolded_asc_graph.GetAllAxis();
  ASSERT_EQ(axis.size(), 2);
  EXPECT_EQ(axis[0]->size, concat_sub_graph.GetAllAxis()[0]->size);
  EXPECT_EQ(axis[1]->size, concat_sub_graph.GetAllAxis()[1]->size);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 12UL);
}

/**
 *         NetOutput
 *            |
 *          AscBc3
 *        /    /\
 *    AscBc1  AscBc2
 *    /   \   //
 * data0  data1
 */
TEST_F(FusedGraphUnfolderTest, AscBcNodeUnfolder_With_Same_Data_Same_Load) {
  ComputeGraphPtr compute_graph = BuildFusedGraphWithSharedDataWithTwoData();
  ASSERT_NE(compute_graph, nullptr);
  std::map<Node *, af::AscGraph> asc_backend_to_asc_graph;

  auto ascbc1 = compute_graph->FindNode("ascbc1");
  ASSERT_NE(ascbc1, nullptr);
  auto ascbc2 = compute_graph->FindNode("ascbc2");
  ASSERT_NE(ascbc2, nullptr);
  auto ascbc3 = compute_graph->FindNode("ascbc3");
  ASSERT_NE(ascbc3, nullptr);

  af::AscGraph add_sub_graph1("sub1_add");
  af::AscGraph add_sub_graph2("sub2_add");
  af::AscGraph concat_sub_graph("sub3_concat");

  CreateAddAscGraph(add_sub_graph1);
  CreateAddAscGraph3SameData(add_sub_graph2);
  CreateConcatAscGraph(concat_sub_graph, true);

  asc_backend_to_asc_graph.emplace(ascbc1.get(), add_sub_graph1);
  asc_backend_to_asc_graph.emplace(ascbc2.get(), add_sub_graph2);
  asc_backend_to_asc_graph.emplace(ascbc3.get(), concat_sub_graph);

  af::AscGraph unfolded_asc_graph("unfolded_asc_graph");
  Status ret = FusedGraphUnfolder::UnfoldFusedGraph(compute_graph, asc_backend_to_asc_graph, unfolded_asc_graph);
  ASSERT_EQ(ret, af::SUCCESS);

  auto axis = unfolded_asc_graph.GetAllAxis();
  ASSERT_EQ(axis.size(), 2);
  EXPECT_EQ(axis[0]->size, concat_sub_graph.GetAllAxis()[0]->size);
  EXPECT_EQ(axis[1]->size, concat_sub_graph.GetAllAxis()[1]->size);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 11UL);
}

/**
 *         NetOutput
 *        /    /\
 *    AscBc1  AscBc2
 *    /   \   /    \
 * data0  data1   data2
 */
TEST_F(FusedGraphUnfolderTest, AscBcNodeUnfolder_With_Multi_Output) {
  ComputeGraphPtr compute_graph = BuildFusedGraphWithMultiOutput();
  ASSERT_NE(compute_graph, nullptr);
  std::map<Node *, af::AscGraph> asc_backend_to_asc_graph;

  auto ascbc1 = compute_graph->FindNode("ascbc1");
  ASSERT_NE(ascbc1, nullptr);
  auto ascbc2 = compute_graph->FindNode("ascbc2");
  ASSERT_NE(ascbc2, nullptr);

  af::AscGraph add_sub_graph1("sub1_add");
  af::AscGraph add_sub_graph2("sub2_add");

  CreateAddAscGraph(add_sub_graph1);
  CreateAddAscGraph3(add_sub_graph2);

  asc_backend_to_asc_graph.emplace(ascbc1.get(), add_sub_graph1);
  asc_backend_to_asc_graph.emplace(ascbc2.get(), add_sub_graph2);

  af::AscGraph unfolded_asc_graph("unfolded_asc_graph");
  Status ret = FusedGraphUnfolder::UnfoldFusedGraph(compute_graph, asc_backend_to_asc_graph, unfolded_asc_graph);
  ASSERT_EQ(ret, af::SUCCESS);

  auto axis = unfolded_asc_graph.GetAllAxis();
  ASSERT_EQ(axis.size(), 2);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 14UL);
}

/**
 *         NetOutput
 *        /    /\
 *    AscBc1  AscBc2
 *    /   \   /    \
 * data0  data1   data2
 */
TEST_F(FusedGraphUnfolderTest, AscBcNodeUnfolder_With_DifferentAxis) {
  ComputeGraphPtr compute_graph = BuildFusedGraphWithMultiOutput();
  ASSERT_NE(compute_graph, nullptr);
  std::map<Node *, af::AscGraph> asc_backend_to_asc_graph;

  auto ascbc1 = compute_graph->FindNode("ascbc1");
  ASSERT_NE(ascbc1, nullptr);
  auto ascbc2 = compute_graph->FindNode("ascbc2");
  ASSERT_NE(ascbc2, nullptr);

  af::AscGraph add_sub_graph1("sub1_add");
  af::AscGraph add_sub_graph2("sub2_add");

  CreateAddAscGraphOneDim(add_sub_graph1);
  CreatePackFirstDimAscGraph(add_sub_graph2);
  optimize::AscGraphInfoComplete::CompleteApiInfo(add_sub_graph1);
  optimize::AscGraphInfoComplete::CompleteApiInfo(add_sub_graph2);

  asc_backend_to_asc_graph.emplace(ascbc1.get(), add_sub_graph1);
  asc_backend_to_asc_graph.emplace(ascbc2.get(), add_sub_graph2);

  af::AscGraph unfolded_asc_graph("unfolded_asc_graph");
  Status ret = FusedGraphUnfolder::UnfoldFusedGraph(compute_graph, asc_backend_to_asc_graph, unfolded_asc_graph);
  ASSERT_EQ(ret, af::SUCCESS);
  auto axis = unfolded_asc_graph.GetAllAxis();

  ASSERT_EQ(axis.size(), 2);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 15UL);
  std::vector<int64_t> golden_axis_concat = {axis[0]->id, axis[1]->id};

  for (auto node : unfolded_asc_graph.GetAllNodes()) {
    if (ScheduleUtils::IsLoad(node)) {
      EXPECT_EQ(node->attr.sched.axis, golden_axis_concat);
    }
    if (ScheduleUtils::IsConcat(node)) {
      EXPECT_EQ(node->attr.sched.axis, golden_axis_concat);
    }
  }
}

/**
 *      NetOutput
 *       |     |
 *       |  AscBc3
 *       | /   /\
 *    AscBc1  AscBc2
 *    /   \   /    \
 * data0  data1   data2
 */
TEST_F(FusedGraphUnfolderTest, AscBcNodeUnfolder_With_Reuse_Output) {
  ComputeGraphPtr compute_graph = BuildFusedGraphWithReuseOutput();
  ASSERT_NE(compute_graph, nullptr);
  std::map<Node *, af::AscGraph> asc_backend_to_asc_graph;

  auto ascbc1 = compute_graph->FindNode("ascbc1");
  ASSERT_NE(ascbc1, nullptr);
  auto ascbc2 = compute_graph->FindNode("ascbc2");
  ASSERT_NE(ascbc2, nullptr);
  auto ascbc3 = compute_graph->FindNode("ascbc3");
  ASSERT_NE(ascbc3, nullptr);

  af::AscGraph add_sub_graph1("sub1_add");
  af::AscGraph add_sub_graph2("sub2_add");
  af::AscGraph concat_sub_graph("sub3_concat");

  CreateAddAscGraph(add_sub_graph1);
  CreateAddAscGraph2(add_sub_graph2);
  CreateConcatAscGraph(concat_sub_graph);

  asc_backend_to_asc_graph.emplace(ascbc1.get(), add_sub_graph1);
  asc_backend_to_asc_graph.emplace(ascbc2.get(), add_sub_graph2);
  asc_backend_to_asc_graph.emplace(ascbc3.get(), concat_sub_graph);

  af::AscGraph unfolded_asc_graph("unfolded_asc_graph");
  Status ret = FusedGraphUnfolder::UnfoldFusedGraph(compute_graph, asc_backend_to_asc_graph, unfolded_asc_graph);
  ASSERT_EQ(ret, af::SUCCESS);

  auto axis = unfolded_asc_graph.GetAllAxis();
  ASSERT_EQ(axis.size(), 2);
  EXPECT_EQ(axis[0]->size, concat_sub_graph.GetAllAxis()[0]->size);
  EXPECT_EQ(axis[1]->size, concat_sub_graph.GetAllAxis()[1]->size);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 14UL);
  auto data0 = unfolded_asc_graph.FindNode("data0");
  ASSERT_NE(data0, nullptr);
  EXPECT_EQ(data0->outputs[0].attr.dtype, ge::DT_INT8);
  int64_t idx = -1;
  data0->attr.ir_attr->GetAttrValue("index", idx);
  EXPECT_EQ(idx, 0);
}

TEST_F(FusedGraphUnfolderTest, BuildLocalAxisMapping_MultipleUnitAxesHasUniqueMapping) {
  AscTensorAttr source;
  source.axis = {101, 102};
  source.repeats = {af::Symbol(55), af::Symbol(3)};
  source.strides = {af::Symbol(3), af::Symbol(1)};
  AscTensorAttr target;
  target.axis = {201, 202, 203, 204};
  target.repeats = {af::Symbol(55), af::Symbol(1), af::Symbol(1), af::Symbol(3)};
  target.strides = {af::Symbol(3), af::Symbol(0), af::Symbol(0), af::Symbol(1)};
  FusedGraphUnfolder::AxisMappingResult result;

  EXPECT_EQ(FusedGraphUnfolder::BuildLocalAxisMapping(source, target, result), af::SUCCESS);
  EXPECT_EQ(result.old_to_global, (std::vector<size_t>{0UL, 3UL}));
  EXPECT_EQ(result.inserted_axes, (std::vector<bool>{false, true, true, false}));
}

TEST_F(FusedGraphUnfolderTest, BuildLocalAxisMapping_MultipleEmbeddingsAreAmbiguous) {
  AscTensorAttr source;
  source.axis = {101, 102, 103};
  source.repeats = {af::Symbol(55), af::Symbol(1), af::Symbol(3)};
  source.strides = {af::Symbol(3), af::Symbol(0), af::Symbol(1)};
  AscTensorAttr target;
  target.axis = {201, 202, 203, 204};
  target.repeats = {af::Symbol(55), af::Symbol(1), af::Symbol(1), af::Symbol(3)};
  target.strides = {af::Symbol(3), af::Symbol(0), af::Symbol(0), af::Symbol(1)};
  FusedGraphUnfolder::AxisMappingResult result;

  EXPECT_EQ(FusedGraphUnfolder::BuildLocalAxisMapping(source, target, result), af::FAILED);
  EXPECT_EQ(result.status, FusedGraphUnfolder::AxisMappingStatus::kAmbiguous);
  EXPECT_EQ(result.reason, FusedGraphUnfolder::AxisMappingFailureReason::kMultipleMappings);
}

TEST_F(FusedGraphUnfolderTest, BuildLocalAxisMapping_NonUnitInsertedAxisIsUnsupported) {
  AscTensorAttr source;
  source.axis = {101, 102};
  source.repeats = {af::Symbol(55), af::Symbol(3)};
  source.strides = {af::Symbol(3), af::Symbol(1)};
  AscTensorAttr target;
  target.axis = {201, 202, 203};
  target.repeats = {af::Symbol(55), af::Symbol(2), af::Symbol(3)};
  target.strides = {af::Symbol(3), af::Symbol(0), af::Symbol(1)};
  FusedGraphUnfolder::AxisMappingResult result;

  EXPECT_EQ(FusedGraphUnfolder::BuildLocalAxisMapping(source, target, result), af::FAILED);
  EXPECT_EQ(result.status, FusedGraphUnfolder::AxisMappingStatus::kUnsupported);
  EXPECT_EQ(result.reason, FusedGraphUnfolder::AxisMappingFailureReason::kNonUnitInsertedAxis);
}

TEST_F(FusedGraphUnfolderTest, BuildLocalAxisMapping_UnknownRepeatEqualityIsUnsupported) {
  AscTensorAttr source;
  source.axis = {101};
  source.repeats = {af::Symbol("source_repeat")};
  source.strides = {af::Symbol(1)};
  AscTensorAttr target;
  target.axis = {201};
  target.repeats = {af::Symbol("target_repeat")};
  target.strides = {af::Symbol(1)};
  FusedGraphUnfolder::AxisMappingResult result;

  EXPECT_EQ(FusedGraphUnfolder::BuildLocalAxisMapping(source, target, result), af::FAILED);
  EXPECT_EQ(result.status, FusedGraphUnfolder::AxisMappingStatus::kUnsupported);
}

TEST_F(FusedGraphUnfolderTest, BuildGraphAxisMapping_ComposesIntermediateMapping) {
  AscGraph source_graph("source_graph");
  auto source_axis0 = source_graph.CreateAxis("source_axis0", af::Symbol(55));
  auto source_axis1 = source_graph.CreateAxis("source_axis1", af::Symbol(3));
  AscGraph target_graph("target_graph");
  auto target_axis0 = target_graph.CreateAxis("target_axis0", af::Symbol(55));
  auto target_axis1 = target_graph.CreateAxis("target_axis1", af::Symbol(1));
  auto target_axis2 = target_graph.CreateAxis("target_axis2", af::Symbol(3));
  AscTensorAttr source;
  source.axis = {source_axis0.id, source_axis1.id};
  source.repeats = {af::Symbol(55), af::Symbol(3)};
  source.strides = {af::Symbol(3), af::Symbol(1)};
  AscTensorAttr target;
  target.axis = {target_axis0.id, target_axis1.id, target_axis2.id};
  target.repeats = {af::Symbol(55), af::Symbol(1), af::Symbol(3)};
  target.strides = {af::Symbol(3), af::Symbol(0), af::Symbol(1)};
  std::vector<size_t> source_to_global;

  ASSERT_EQ(source_graph.GetAllAxis().size(), 2UL);
  ASSERT_EQ(target_graph.GetAllAxis().size(), 3UL);
  EXPECT_TRUE(FusedGraphUnfolder::BuildGraphAxisMapping(source_graph, source, target_graph, target, {0UL, 1UL, 3UL},
                                                        source_to_global));
  EXPECT_EQ(source_to_global, (std::vector<size_t>{0UL, 3UL}));
}

TEST_F(FusedGraphUnfolderTest, BuildGraphAxisMapping_RejectsDuplicateGlobalAxis) {
  AscGraph source_graph("source_graph");
  auto source_axis0 = source_graph.CreateAxis("source_axis0", af::Symbol(55));
  auto source_axis1 = source_graph.CreateAxis("source_axis1", af::Symbol(3));
  AscGraph target_graph("target_graph");
  auto target_axis0 = target_graph.CreateAxis("target_axis0", af::Symbol(55));
  auto target_axis1 = target_graph.CreateAxis("target_axis1", af::Symbol(3));
  AscTensorAttr source;
  source.axis = {source_axis0.id, source_axis1.id};
  source.repeats = {af::Symbol(55), af::Symbol(3)};
  source.strides = {af::Symbol(3), af::Symbol(1)};
  AscTensorAttr target;
  target.axis = {target_axis0.id, target_axis1.id};
  target.repeats = {af::Symbol(55), af::Symbol(3)};
  target.strides = {af::Symbol(3), af::Symbol(1)};
  std::vector<size_t> source_to_global;

  EXPECT_FALSE(FusedGraphUnfolder::BuildGraphAxisMapping(source_graph, source, target_graph, target, {0UL, 0UL},
                                                         source_to_global));
}

TEST_F(FusedGraphUnfolderTest, ApplyMappedLoopAxis_PreservesBufferAttributes) {
  AscGraph graph("buffer_graph");
  const auto s0 = af::Symbol(2);
  const auto s1 = af::Symbol(3);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  AscGraph global_graph("global_graph");
  auto global_z0 = global_graph.CreateAxis("z0", s0);
  auto global_z1 = global_graph.CreateAxis("z1", s1);
  auto global_z2 = global_graph.CreateAxis("z2", af::Symbol(5));
  af::ascir_op::Data buffer("buffer", graph);
  buffer.attr.api.type = af::ApiType::kAPITypeBuffer;
  buffer.attr.sched.axis = {z0.id, z1.id};
  *buffer.y.axis = {z0.id, z1.id};
  *buffer.y.repeats = {s0, s1};
  *buffer.y.strides = {s1, af::Symbol(1)};

  ASSERT_EQ(FusedGraphUnfolder::ApplyMappedLoopAxis(graph, global_graph.GetAllAxis(),
                                                    {global_z0.id, global_z1.id, global_z2.id}, {0UL, 2UL}),
            af::SUCCESS);
  const auto buffer_node = graph.FindNode("buffer");
  ASSERT_NE(buffer_node, nullptr);
  EXPECT_EQ(buffer_node->attr.sched.axis, (std::vector<af::AxisId>{z0.id, z1.id}));
  EXPECT_EQ(buffer_node->outputs[0].attr.axis, (std::vector<af::AxisId>{z0.id, z1.id}));
  EXPECT_EQ(buffer_node->outputs[0].attr.repeats, (std::vector<af::Expression>{s0, s1}));
  EXPECT_EQ(buffer_node->outputs[0].attr.strides, (std::vector<af::Expression>{s1, af::Symbol(1)}));
}

TEST_F(FusedGraphUnfolderTest, TestIsSameLoad) {
  af::AscGraph graph("test");
  auto ONE = af::Symbol(1);
  const af::Expression s0 = graph.CreateSizeVar("s0");
  const af::Expression s1 = graph.CreateSizeVar("s1");
  const af::Expression s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_INT8;

  af::ascir_op::Load load0("load0");
  load0.ir_attr.SetOffset(af::Symbol(0));
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, ONE};

  af::ascir_op::Load load1("load1");
  load1.ir_attr.SetOffset(af::Symbol(888));
  load1.x = data0.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, ONE};

  af::ascir_op::Load load2("load2");
  load2.ir_attr.SetOffset(af::Symbol(0));
  load2.x = data0.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, ONE};

  af::ascir_op::Load load3("load3");
  load3.ir_attr.SetOffset(af::Symbol(0));
  load3.x = data0.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s2};
  *load3.y.strides = {s1, ONE};

  af::ascir_op::Load load4("load4");
  load4.ir_attr.SetOffset(af::Symbol(0));
  load4.x = data0.y;
  load4.attr.sched.axis = {z0.id, z1.id};
  *load4.y.axis = {z0.id, z1.id};
  *load4.y.repeats = {s0, s1};
  *load4.y.strides = {s1 * s2, s2};

  af::ascir_op::Load load5("load5");
  load5.ir_attr.SetOffset(af::Symbol(0));
  load5.x = data0.y;
  load5.attr.sched.axis = {z0.id, z1.id};
  *load5.y.axis = {z0.id, z1.id};
  *load5.y.repeats = {s0, s1};
  *load5.y.strides = {s1, ONE};

  auto data0_node = graph.FindNode("data0");
  auto load0_node = graph.FindNode("load0");
  auto load1_node = graph.FindNode("load1");
  auto load2_node = graph.FindNode("load2");
  auto load3_node = graph.FindNode("load3");
  auto load4_node = graph.FindNode("load4");
  auto load5_node = graph.FindNode("load5");

  EXPECT_FALSE(FusedGraphUnfolder::IsSameLoadNode(load0_node, data0_node));
  EXPECT_FALSE(FusedGraphUnfolder::IsSameLoadNode(load0_node, load1_node));
  EXPECT_FALSE(FusedGraphUnfolder::IsSameLoadNode(load0_node, load2_node));
  EXPECT_FALSE(FusedGraphUnfolder::IsSameLoadNode(load0_node, load3_node));
  EXPECT_FALSE(FusedGraphUnfolder::IsSameLoadNode(load0_node, load4_node));
  EXPECT_TRUE(FusedGraphUnfolder::IsSameLoadNode(load0_node, load5_node));
}
}  // namespace optimize
