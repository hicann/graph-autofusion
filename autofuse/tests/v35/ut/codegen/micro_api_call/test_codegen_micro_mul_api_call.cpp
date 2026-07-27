/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "gtest/gtest.h"
#include "node_utils_ex.h"
#include "graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "micro_api_call_factory.h"
#include "micro_api_call.h"
#include "micro_binary_scalar_api_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

namespace {
void InitTensorAttr(ascir::TensorAttr &tensor, ascir::TensorId tensor_id, ge::DataType dtype) {
  tensor.attr.dtype = dtype;
  tensor.attr.mem.tensor_id = tensor_id;
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensor.attr.buf.id = tensor_id;
  tensor.attr.opt.merge_scope = af::kIdNone;
}

TensorManager CreateTensorManager(const ascir::TensorAttr &input, const ascir::TensorAttr &output) {
  std::string input_dtype = "float";
  std::string output_dtype = "float";
  TensorManager tensor_mng;
  tensor_mng.AddTensor(MicroApiTensor(input, input_dtype));
  tensor_mng.AddTensor(MicroApiTensor(output, output_dtype));
  return tensor_mng;
}
}  // namespace

// Test Mul with BF16 type
TEST(CodegenKernel, MicroMulApiCall_Load_Mul_BF16_Store) {
  af::AscGraph graph("test_mul_bf16_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);

  auto s0 = af::Symbol(16);
  auto s1 = af::Symbol(8);
  auto s2 = af::Symbol(4);
  auto s3 = af::Symbol(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data x_op("x", graph);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Mul mul_op("mul");
  af::ascir_op::Store store_op("store");

  graph.AddNode(load_op);
  graph.AddNode(mul_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, s3};
  *load_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  mul_op.x1 = load_op.y;
  mul_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *mul_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *mul_op.y.repeats = {s0, s1, s2, s3};
  *mul_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  store_op.x = mul_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store_op.y.repeats = {s0, s1, s2, s3};
  *store_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  load->outputs[0].attr.dtype = af::DT_BF16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto mul_node = graph.FindNode("mul");
  mul_node->attr.api.compute_type = af::ComputeType::kComputeLoad;
  mul_node->attr.api.type = af::ApiType::kAPITypeCompute;
  mul_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  mul_node->attr.sched.loop_axis = z0.id;
  mul_node->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  mul_node->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  mul_node->outputs[0].attr.dtype = af::DT_BF16;
  mul_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  mul_node->outputs[0].attr.mem.tensor_id = 1;
  mul_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  mul_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  mul_node->outputs[0].attr.que.id = 2;
  mul_node->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeElewise;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  store->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  store->outputs[0].attr.dtype = af::DT_BF16;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 1;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddSizeVar(af::SizeVar(s3));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = store->outputs[0].attr.mem.tensor_id;
  codegen::CallParam cp = {"p_reg", ""};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(load->outputs[0], tensor_load);
  auto tensor_store = store->GetName() + "_" + store->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor2(store->outputs[0], tensor_store);
  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor1);
  tensor_mng.AddTensor(tensor2);
  codegen::MicroApiCall call("Mul");
  EXPECT_EQ(call.Init(mul_node), 0);
  call.AddInput(x1.id);
  call.AddOutput(y1.id);

  std::string result;
  call.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::Mul(vreg_1, vreg_0, p_reg);\n"});
}

// Test Mul with INT64 type
TEST(CodegenKernel, MicroMulApiCall_Load_Mul_INT64_Store) {
  af::AscGraph graph("test_mul_int64_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);

  auto s0 = af::Symbol(16);
  auto s1 = af::Symbol(8);
  auto s2 = af::Symbol(4);
  auto s3 = af::Symbol(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data x_op("x", graph);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Mul mul_op("mul");
  af::ascir_op::Store store_op("store");

  graph.AddNode(load_op);
  graph.AddNode(mul_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, s3};
  *load_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  mul_op.x1 = load_op.y;
  mul_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *mul_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *mul_op.y.repeats = {s0, s1, s2, s3};
  *mul_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  store_op.x = mul_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store_op.y.repeats = {s0, s1, s2, s3};
  *store_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  load->outputs[0].attr.dtype = af::DT_INT64;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto mul_node = graph.FindNode("mul");
  mul_node->attr.api.compute_type = af::ComputeType::kComputeLoad;
  mul_node->attr.api.type = af::ApiType::kAPITypeCompute;
  mul_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  mul_node->attr.sched.loop_axis = z0.id;
  mul_node->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  mul_node->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  mul_node->outputs[0].attr.dtype = af::DT_INT64;
  mul_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  mul_node->outputs[0].attr.mem.tensor_id = 1;
  mul_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  mul_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  mul_node->outputs[0].attr.que.id = 2;
  mul_node->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeElewise;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  store->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  store->outputs[0].attr.dtype = af::DT_INT64;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 1;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddSizeVar(af::SizeVar(s3));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = store->outputs[0].attr.mem.tensor_id;
  codegen::CallParam cp = {"p_reg", ""};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(load->outputs[0], tensor_load);
  auto tensor_store = store->GetName() + "_" + store->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor2(store->outputs[0], tensor_store);
  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor1);
  tensor_mng.AddTensor(tensor2);
  codegen::MicroApiCall call("Mul");
  EXPECT_EQ(call.Init(mul_node), 0);
  call.AddInput(x1.id);
  call.AddOutput(y1.id);

  std::string result;
  call.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::Mul(vreg_1, vreg_0, p_reg);\n"});
}

TEST(CodegenKernel, MicroBinaryScalarApiCall_BothScalar_ReturnFailed) {
  af::AscGraph graph("test_binary_scalar_graph");
  af::ascir_op::Scalar scalar0("scalar0", graph);
  af::ascir_op::Scalar scalar1("scalar1", graph);
  af::ascir_op::Mul mul_op("mul");
  graph.AddNode(mul_op);

  scalar0.ir_attr.SetValue("1.0");
  scalar1.ir_attr.SetValue("2.0");
  mul_op.x1 = scalar0.y;
  mul_op.x2 = scalar1.y;

  codegen::MicroBinaryScalarApiCall call("Mul");
  EXPECT_NE(call.Init(graph.FindNode("mul")), af::SUCCESS);
}

TEST(CodegenKernel, MicroBinaryScalarApiCall_FirstScalar_GenerateExchangeInput) {
  af::AscGraph graph("test_first_scalar_graph");
  af::ascir_op::Scalar scalar_op("scalar", graph);
  af::ascir_op::Data x_op("x", graph);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Mul mul_op("mul");
  graph.AddNode(load_op);
  graph.AddNode(mul_op);

  scalar_op.ir_attr.SetValue("2.0");
  scalar_op.y.dtype = ge::DT_FLOAT;
  load_op.x = x_op.y;
  mul_op.x1 = scalar_op.y;
  mul_op.x2 = load_op.y;
  auto load = graph.FindNode("load");
  auto scalar = graph.FindNode("scalar");
  auto mul = graph.FindNode("mul");
  InitTensorAttr(load->outputs[0], 0, ge::DT_FLOAT);
  InitTensorAttr(scalar->outputs[0], 2, ge::DT_FLOAT);
  InitTensorAttr(mul->outputs[0], 1, ge::DT_FLOAT);
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  EXPECT_EQ(tpipe.AddTensor("2.0", scalar->outputs[0], ""), af::SUCCESS);
  auto tensor_mng = CreateTensorManager(load->outputs[0], mul->outputs[0]);

  codegen::MicroBinaryScalarApiCall call("Mul");
  EXPECT_EQ(call.Init(mul), af::SUCCESS);
  call.AddInput(2, codegen::TensorType::UB_TENSOR);
  call.AddInput(0);
  call.AddOutput(1);
  codegen::CallParam cp = {"p_reg", ""};
  std::string result;
  EXPECT_EQ(call.Generate(tensor_mng, tpipe, cp, result), af::SUCCESS);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::Muls(vreg_1, vreg_0, scalar_2, p_reg);\n"});
}

TEST(CodegenKernel, MicroBinaryScalarApiCall_SecondScalarData_GenerateScalarApi) {
  af::AscGraph graph("test_second_scalar_data_graph");
  af::ascir_op::ScalarData scalar_data_op("scalar_data", graph);
  af::ascir_op::Data x_op("x", graph);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Mul mul_op("mul");
  graph.AddNode(load_op);
  graph.AddNode(mul_op);

  scalar_data_op.ir_attr.SetIndex(0);
  scalar_data_op.y.dtype = ge::DT_FLOAT;
  load_op.x = x_op.y;
  mul_op.x1 = load_op.y;
  mul_op.x2 = scalar_data_op.y;
  auto load = graph.FindNode("load");
  auto scalar_data = graph.FindNode("scalar_data");
  auto mul = graph.FindNode("mul");
  InitTensorAttr(load->outputs[0], 0, ge::DT_FLOAT);
  InitTensorAttr(scalar_data->outputs[0], 2, ge::DT_FLOAT);
  InitTensorAttr(mul->outputs[0], 1, ge::DT_FLOAT);
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  EXPECT_EQ(tpipe.AddTensor(scalar_data->outputs[0]), af::SUCCESS);
  auto tensor_mng = CreateTensorManager(load->outputs[0], mul->outputs[0]);

  codegen::MicroBinaryScalarApiCall call("Mul");
  EXPECT_EQ(call.Init(mul), af::SUCCESS);
  call.AddInput(0);
  call.AddInput(2, codegen::TensorType::UB_TENSOR);
  call.AddOutput(1);
  codegen::CallParam cp = {"p_reg", ""};
  std::string result;
  EXPECT_EQ(call.Generate(tensor_mng, tpipe, cp, result), af::SUCCESS);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::Muls(vreg_1, vreg_0, scalar_data, p_reg);\n"});
}
