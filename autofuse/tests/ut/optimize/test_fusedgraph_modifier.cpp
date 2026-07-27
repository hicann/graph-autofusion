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

#include "ascendc_ir.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_factory_af.h"
#include "graph/utils/op_desc_utils.h"
#include "ascir_ops.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "fused_graph/fused_graph_modifier.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph_utils.h"
#include "ascir_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "ascgen_log.h"
#include "schedule_utils.h"
#include "fusion/autofuse_attrs.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/expression/const_values.h"
#include "platform_context.h"
#include "platform/v1/platformv1.h"

namespace optimize {
using namespace af;
class FusedGraphModifierTest : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};

static void CreateAscBackendGraph(std::shared_ptr<af::AscGraph> &graph, const std::string &prefix,
                                  int64_t axis_num = 2) {
  auto ONE = af::Symbol(1);
  std::vector<int64_t> axis_ids;
  std::vector<af::Expression> repeats;
  for (int64_t i = 0; i < axis_num; ++i) {
    const af::Expression exp = graph->CreateSizeVar("s" + std::to_string(i));
    auto axis = graph->CreateAxis("z" + std::to_string(i), exp);
    axis_ids.push_back(i);
    repeats.push_back(exp);
  }

  std::vector<af::Expression> strides(repeats.size(), af::sym::kSymbolOne);
  if (axis_num > 1) {
    for (int64_t i = axis_num - 2; i >= 0; --i) {
      strides[i] = repeats[i + 1] * strides[i + 1];
    }
  }

  af::ascir_op::Data data(std::string(prefix + "_data").c_str(), *graph);
  data.attr.sched.axis = axis_ids;
  *data.y.axis = axis_ids;
  *data.y.repeats = repeats;
  *data.y.strides = strides;
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_INT8;

  af::ascir_op::Load load(std::string(prefix + "_load").c_str());
  load.x = data.y;
  load.attr.sched.axis = axis_ids;
  *load.y.axis = axis_ids;
  *load.y.repeats = repeats;
  *load.y.strides = strides;

  af::ascir_op::Abs abs(std::string(prefix + "_abs").c_str());
  abs.x = load.y;
  abs.attr.sched.axis = axis_ids;
  *abs.y.axis = axis_ids;
  *abs.y.repeats = repeats;
  *abs.y.strides = strides;

  af::ascir_op::Store store(std::string(prefix + "_store").c_str());
  store.x = abs.y;
  store.attr.sched.axis = axis_ids;
  *store.y.axis = axis_ids;
  *store.y.repeats = repeats;
  *store.y.strides = strides;

  af::ascir_op::Output y(std::string(prefix + "_out").c_str());
  y.x = store.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = ge::DT_FLOAT16;
}

static NodePtr CreateAscbcToAscGraph(const std::string &name, ComputeGraphPtr &compute_graph, int64_t in_num = 1,
                                     int64_t out_num = 1) {
  OpDescBuilder op_desc_builder(name, "AscBackend");
  op_desc_builder.AddDynamicInput("x", in_num);
  op_desc_builder.AddDynamicOutput("y", out_num);
  const auto &op_desc = op_desc_builder.Build();
  auto node = compute_graph->AddNode(op_desc);
  node->SetOwnerComputeGraph(compute_graph);
  return node;
}

TEST_F(FusedGraphModifierTest, test_workspace_reuse) {
  std::shared_ptr<af::AscGraph> g0 = std::make_shared<af::AscGraph>("g0");
  CreateAscBackendGraph(g0, "g0", 2);
  std::shared_ptr<af::AscGraph> g1 = std::make_shared<af::AscGraph>("g1");
  CreateAscBackendGraph(g1, "g1", 1);
  std::shared_ptr<af::AscGraph> g2 = std::make_shared<af::AscGraph>("g2");
  CreateAscBackendGraph(g2, "g2", 2);
  std::shared_ptr<af::AscGraph> g3 = std::make_shared<af::AscGraph>("g3");
  CreateAscBackendGraph(g3, "g3", 1);

  af::AscGraph fused_asc_graph("fused_graph");

  af::ascir_op::Data data0("data0", fused_asc_graph);
  auto ir_attr = data0.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  ir_attr->SetIndex(0);

  auto fused_graph = af::AscGraphUtils::GetComputeGraph(fused_asc_graph);
  auto data_node = fused_asc_graph.FindNode("data0");

  auto ascbc0 = CreateAscbcToAscGraph("ascbc0", fused_graph);
  auto ascbc1 = CreateAscbcToAscGraph("ascbc1", fused_graph);
  auto ascbc2 = CreateAscbcToAscGraph("ascbc2", fused_graph);
  auto ascbc3 = CreateAscbcToAscGraph("ascbc3", fused_graph);
  af::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), ascbc0->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc0->GetOutDataAnchor(0), ascbc1->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc1->GetOutDataAnchor(0), ascbc2->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc2->GetOutDataAnchor(0), ascbc3->GetInDataAnchor(0));

  af::ascir_op::Output output("output");
  auto out_ir_attr = output.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  out_ir_attr->SetIndex(0);
  auto out_desc = OpDescUtils::GetOpDescFromOperator(output);
  auto output_node = fused_graph->AddNode(out_desc);
  af::GraphUtils::AddEdge(ascbc3->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));

  FusedGraphModifier modifier;
  std::map<af::Node *, af::AscGraph> asc_backend_to_ascgraph;
  asc_backend_to_ascgraph.emplace(ascbc0.get(), *g0);
  asc_backend_to_ascgraph.emplace(ascbc1.get(), *g1);
  asc_backend_to_ascgraph.emplace(ascbc2.get(), *g2);
  asc_backend_to_ascgraph.emplace(ascbc3.get(), *g3);
  EXPECT_EQ(modifier.SubgraphConnectionsToWorkspace(fused_graph, asc_backend_to_ascgraph), af::SUCCESS);

  auto ws0_g0 = g0->FindNode("fused_workspace0");
  EXPECT_NE(ws0_g0, nullptr);
  auto ws0_g1 = g1->FindNode("fused_workspace0");
  EXPECT_NE(ws0_g1, nullptr);
  auto ws1_g1 = g1->FindNode("fused_workspace1");
  EXPECT_NE(ws1_g1, nullptr);
  auto ws1_g2 = g2->FindNode("fused_workspace1");
  EXPECT_NE(ws1_g2, nullptr);
  auto ws0_g2 = g2->FindNode("fused_workspace0");  // reuse
  EXPECT_NE(ws0_g2, nullptr);
  auto ws0_g3 = g3->FindNode("fused_workspace0");
  EXPECT_NE(ws0_g3, nullptr);
}
}  // namespace optimize
