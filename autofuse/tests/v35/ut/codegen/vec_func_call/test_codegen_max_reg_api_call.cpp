/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "ascir_ops.h"
#include "codegen_kernel.h"
#include "utils/api_call_factory.h"
#include "../reg_api_call/reg_reduce_api_call.h"

#include "runtime_stub.h"
#include "platform_context.h"

using namespace af::ops;
using namespace codegen;
using namespace af::ascir_op;
using namespace testing;

class RegReduceMaxApicallTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
    ge::RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

// Test Max with INT8 type
TEST_F(RegReduceMaxApicallTest, Max_INT8) {
  std::string api_name = "ReduceMax";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeTileInner, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);
  af::ascir_op::Max reduce("reduce");  // 创建Reduce类型的节点
  graph.AddNode(reduce);               // 将Reduce节点添加到图中

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  af::AscTensor tensory = nodey->outputs[0];
  auto reduce_node = graph.FindNode("reduce");  // 获取Reduce节点

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  tensorx.attr.dtype = af::DT_INT8;
  vector<af::Expression> vectorized_stride_x{One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats = {z0.size, One, One};
  tensory.attr.strides = {One, Zero, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = af::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = af::kIdNone;
  tensory.attr.buf.id = 4;
  tensory.attr.dtype = af::DT_INT8;
  vector<af::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
  // Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::RegReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Max";
  call.node = reduce_node;  // 设置Reduce节点，BuildApiParam需要有效的node
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
  EXPECT_NE(result.find("ReduceMaxExtend<int8_t"), std::string::npos) << result;
}

// Test Max with INT16 type
TEST_F(RegReduceMaxApicallTest, Max_INT16) {
  std::string api_name = "ReduceMax";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeTileInner, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);
  af::ascir_op::Max reduce("reduce");  // 创建Reduce类型的节点
  graph.AddNode(reduce);               // 将Reduce节点添加到图中

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  af::AscTensor tensory = nodey->outputs[0];
  auto reduce_node = graph.FindNode("reduce");  // 获取Reduce节点

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  tensorx.attr.dtype = af::DT_INT16;
  vector<af::Expression> vectorized_stride_x{One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats = {z0.size, One, One};
  tensory.attr.strides = {One, Zero, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = af::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = af::kIdNone;
  tensory.attr.buf.id = 4;
  tensory.attr.dtype = af::DT_INT16;
  vector<af::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
  // Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::RegReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Max";
  call.node = reduce_node;  // 设置Reduce节点，BuildApiParam需要有效的node
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
}

// Test Max with BF16 type
TEST_F(RegReduceMaxApicallTest, Max_BF16) {
  std::string api_name = "ReduceMax";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeTileInner, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);
  af::ascir_op::Max reduce("reduce");  // 创建Reduce类型的节点
  graph.AddNode(reduce);               // 将Reduce节点添加到图中

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  af::AscTensor tensory = nodey->outputs[0];
  auto reduce_node = graph.FindNode("reduce");  // 获取Reduce节点

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  tensorx.attr.dtype = af::DT_BF16;
  vector<af::Expression> vectorized_stride_x{One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats = {z0.size, One, One};
  tensory.attr.strides = {One, Zero, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = af::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = af::kIdNone;
  tensory.attr.buf.id = 4;
  tensory.attr.dtype = af::DT_BF16;
  vector<af::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
  // Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::RegReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Max";
  call.node = reduce_node;  // 设置Reduce节点，BuildApiParam需要有效的node
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
}
