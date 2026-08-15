#include <gtest/gtest.h>

#include "ascir_ops.h"
#include "reg_broadcast_api_call.h"

namespace {

static void BuildBroadcastRegApiCallPrePosNotFoundGraph(af::AscGraph &graph, af::Expression s0, af::Expression s1,
                                                        af::Expression s2, af::Axis z0, af::Axis z1, af::Axis z2) {
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Load load("load");
  af::ascir_op::Broadcast broadcast("broadcast");
  graph.AddNode(load);
  graph.AddNode(broadcast);
  load.x = x.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {af::ops::One, af::ops::One, s2};
  *load.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};
  broadcast.x = load.y;
  *broadcast.y.axis = {z0.id, z1.id, z2.id};
  *broadcast.y.repeats = {s0, s1, s2};
  *broadcast.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  auto load_node = graph.FindNode("load");
  load_node->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  load_node->outputs[0].attr.vectorized_strides = {af::ops::Zero, af::ops::Zero, af::ops::One};
  load_node->outputs[0].attr.dtype = af::DT_FLOAT;
  load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load_node->outputs[0].attr.mem.tensor_id = 0;
  load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load_node->outputs[0].attr.que.id = 0;
  load_node->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast_node = graph.FindNode("broadcast");
  broadcast_node->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  broadcast_node->outputs[0].attr.vectorized_strides = {af::ops::Zero, af::ops::Zero, af::ops::One};
  broadcast_node->outputs[0].attr.dtype = af::DT_FLOAT;
  broadcast_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast_node->outputs[0].attr.mem.tensor_id = 1;
  broadcast_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast_node->outputs[0].attr.que.id = 1;
  broadcast_node->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

TEST(CodegenSt, BroadcastRegApiCallPrePosNotFound) {
  af::AscGraph graph("broadcast_pre_pos_not_found");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  BuildBroadcastRegApiCallPrePosNotFoundGraph(graph, s0, s1, s2, z0, z1, z2);

  auto load_node = graph.FindNode("load");
  auto broadcast_node = graph.FindNode("broadcast");

  codegen::Tiler tiler;
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load_node->outputs[0]);
  tpipe.AddTensor(broadcast_node->outputs[0]);

  codegen::ApiTensor input;
  input.id = load_node->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastRegApiCall call("BroadcastExtend");
  ASSERT_EQ(call.Init(broadcast_node), af::SUCCESS);
  call.inputs.push_back(&input);

  std::string result;
  ASSERT_EQ(call.Generate(tpipe, {}, result), af::SUCCESS);
  EXPECT_NE(result.find("static_cast<uint32_t>(t->s2)"), std::string::npos);
}

}  // namespace
