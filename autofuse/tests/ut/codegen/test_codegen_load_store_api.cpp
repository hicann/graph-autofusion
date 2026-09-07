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
#include "utils/api_call_utils.h"

using namespace std;
using namespace af::ops;
using namespace codegen;
using namespace af::ascir_op;

/*
 * Axis:              z0     z1  z2
 * repeat:            s0     s1  s2
 * stride:            s1*s2  s2  one
 * vectorized stride: s2 one
 */
TEST(CodegenLoadStore, CalculateDmaParams_UbGMContinus) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size};
  tensor.attr.strides = {z1.size * z2.size, z2.size, One};

  vector<af::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = z2.size;
  tensor.attr.vectorized_strides = vectorized_strides;
  std::string dtype_name;
  Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  DataCopyParams param;
  Tensor tensor1(tensor, dtype_name);
  Tensor tensor2(tensor, dtype_name);
  tensor1.vectorized_axis_pos = {1, 2};
  tensor2.vectorized_axis_pos = {1, 2};
  EXPECT_EQ(CalculateDmaParams(tpipe, tensor1, tensor2, param), true);
  EXPECT_EQ(param.repeats.size(), 1);
  EXPECT_EQ(param.repeats[0], af::Expression::Parse("s1*s2"));
  EXPECT_EQ(param.gm_strides.size(), 1);
  EXPECT_EQ(param.gm_strides[0], af::Expression::Parse("1"));
  EXPECT_EQ(param.ub_strides.size(), 1);
  EXPECT_EQ(param.ub_strides[0], af::Expression::Parse("1"));
}

/*
 * Axis:              z0     z1  z2
 * repeat:            s0     s1  s2
 * stride:            s1*s2  s2  one
 * vectorized stride: align(s2) one
 */
TEST(CodegenLoadStore, CalculateDmaParams_DiscreteUbContinusGM) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size};
  tensor.attr.strides = {z1.size * z2.size, z2.size, One};

  vector<af::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = af::sym::Align(z2.size, 32);
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  DataCopyParams param;
  Tensor tensor1(tensor, dtype_name);
  Tensor tensor2(tensor, dtype_name);
  tensor1.vectorized_axis_pos = {1, 2};
  tensor2.vectorized_axis_pos = {1, 2};
  EXPECT_EQ(CalculateDmaParams(tpipe, tensor1, tensor2, param), true);
  EXPECT_EQ(param.repeats.size(), 2);
  EXPECT_EQ(param.repeats[0], af::Expression::Parse("s1"));
  EXPECT_EQ(param.repeats[1], af::Expression::Parse("s2"));
  EXPECT_EQ(param.gm_strides.size(), 2);
  EXPECT_EQ(param.gm_strides[0], af::Expression::Parse("s2"));
  EXPECT_EQ(param.gm_strides[1], af::Expression::Parse("1"));
  EXPECT_EQ(param.ub_strides.size(), 2);
  EXPECT_EQ(param.ub_strides[0], af::sym::Align(z2.size, 32));
  EXPECT_EQ(param.ub_strides[1], af::Expression::Parse("1"));
}

/*
 * Axis:              z0     z1  z2
 * repeat:            s0     s1  s2
 * stride:            2*s1*s2  2*s2  one
 * vectorized stride: s2 one
 */
TEST(CodegenLoadStore, CalculateDmaParams_ContinusUbDiscreteGM) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size};
  tensor.attr.strides = {z1.size * (z2.size + z2.size), z2.size + z2.size, One};

  vector<af::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = z2.size;
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  DataCopyParams param;
  Tensor tensor1(tensor, dtype_name);
  Tensor tensor2(tensor, dtype_name);
  tensor1.vectorized_axis_pos = {1, 2};
  tensor2.vectorized_axis_pos = {1, 2};
  EXPECT_EQ(CalculateDmaParams(tpipe, tensor1, tensor2, param), true);
  EXPECT_EQ(param.repeats.size(), 2);
  EXPECT_EQ(param.repeats[0], af::Expression::Parse("s1"));
  EXPECT_EQ(param.repeats[1], af::Expression::Parse("s2"));
  EXPECT_EQ(param.gm_strides.size(), 2);
  EXPECT_EQ(param.gm_strides[0], af::Expression::Parse("2 * s2"));
  EXPECT_EQ(param.gm_strides[1], af::Expression::Parse("1"));
  EXPECT_EQ(param.ub_strides.size(), 2);
  EXPECT_EQ(param.ub_strides[0], af::Expression::Parse("s2"));
  EXPECT_EQ(param.ub_strides[1], af::Expression::Parse("1"));
}

/*
 * Axis:              z0                     z1                   z2       z3    z4
 * repeat:            s0                     s1                   s2       s3    s4
 * stride:            2*s1*s2*(s3*s4+s3*s4)  (s2+s2)*(s3*s4+s3*s4)   2*s3*s4  2*s4  one
 * vectorized stride: s2*s3*s4 s3*s4 s4 one
 */
TEST(CodegenLoadStore, CalculateDmaParams_4DContinusUbDiscreteGM) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));
  af::SizeVar s3(af::Symbol("s3"));
  af::SizeVar s4(af::Symbol("s4"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};
  af::Axis z3{.id = 3, .name = "z3", .type = af::Axis::Type::kAxisTypeOriginal, .size = s3.expr};
  af::Axis z4{.id = 4, .name = "z4", .type = af::Axis::Type::kAxisTypeOriginal, .size = s4.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddSizeVar(af::SizeVar(s3));
  tiler.AddSizeVar(af::SizeVar(s4));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddAxis(z4);

  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id, z3.id, z4.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size, z3.size, z4.size};
  tensor.attr.strides = {z1.size * (z2.size + z2.size) * (z3.size * (z4.size + z4.size)),
                         (z2.size + z2.size) * (z3.size * (z4.size + z4.size)), z3.size * (z4.size + z4.size),
                         z4.size + z4.size, One};

  vector<af::Expression> vectorized_strides{z2.size * z3.size * z4.size, z3.size * z4.size, z4.size, One};
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  DataCopyParams param;
  Tensor tensor1(tensor, dtype_name);
  Tensor tensor2(tensor, dtype_name);
  tensor1.vectorized_axis_pos = {1, 2, 3, 4};
  tensor2.vectorized_axis_pos = {1, 2, 3, 4};

  EXPECT_EQ(CalculateDmaParams(tpipe, tensor1, tensor2, param), true);
  EXPECT_EQ(param.repeats.size(), 3);
  EXPECT_EQ(param.repeats[0], af::Expression::Parse("s1"));
  EXPECT_EQ(param.repeats[1], af::Expression::Parse("s2*s3"));
  EXPECT_EQ(param.repeats[2], af::Expression::Parse("s4"));
  EXPECT_EQ(param.gm_strides.size(), 3);
  EXPECT_EQ(param.gm_strides[0], af::Expression::Parse("4*s2*s3*s4"));
  EXPECT_EQ(param.gm_strides[1], af::Expression::Parse("2*s4"));
  EXPECT_EQ(param.gm_strides[2], af::Expression::Parse("1"));
  EXPECT_EQ(param.ub_strides.size(), 3);
  EXPECT_EQ(param.ub_strides[0], af::Expression::Parse("s2*s3*s4"));
  EXPECT_EQ(param.ub_strides[1], af::Expression::Parse("s4"));
  EXPECT_EQ(param.ub_strides[2], af::Expression::Parse("1"));
}

/*
 * Axis:              z0                     z1                   z2       z3    z4
 * repeat:            s0                     s1                   s2       s3    s4
 * stride:            2*s1*s2*(s3*s4+s3*s4)  (s2+s2)*(s3*s4+s3*s4)   2*s3*s4  2*s4  one
 * vectorized stride: s2*align(s3*s4) align(s3*s4) s4 one
 */
TEST(CodegenLoadStore, CalculateDmaParams_4DDiscreteUbDiscreteGM) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));
  af::SizeVar s3(af::Symbol("s3"));
  af::SizeVar s4(af::Symbol("s4"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};
  af::Axis z3{.id = 3, .name = "z3", .type = af::Axis::Type::kAxisTypeOriginal, .size = s3.expr};
  af::Axis z4{.id = 4, .name = "z4", .type = af::Axis::Type::kAxisTypeOriginal, .size = s4.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddSizeVar(af::SizeVar(s3));
  tiler.AddSizeVar(af::SizeVar(s4));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddAxis(z4);

  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id, z3.id, z4.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size, z3.size, z4.size};
  tensor.attr.strides = {z1.size * (z2.size + z2.size) * (z3.size * (z4.size + z4.size)),
                         (z2.size + z2.size) * (z3.size * (z4.size + z4.size)), z3.size * (z4.size + z4.size),
                         z4.size + z4.size, One};

  vector<af::Expression> vectorized_strides{z2.size * af::sym::Align(z3.size * z4.size, 32),
                                            af::sym::Align(z3.size * z4.size, 32), z4.size, One};
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  DataCopyParams param;
  Tensor tensor1(tensor, dtype_name);
  Tensor tensor2(tensor, dtype_name);
  tensor1.vectorized_axis_pos = {1, 2, 3, 4};
  tensor2.vectorized_axis_pos = {1, 2, 3, 4};
  EXPECT_EQ(CalculateDmaParams(tpipe, tensor1, tensor2, param), true);
  EXPECT_EQ(param.repeats.size(), 4);
  EXPECT_EQ(param.gm_strides.size(), 4);
  EXPECT_EQ(param.gm_strides[0], af::Expression::Parse("4*s2*s3*s4"));
  EXPECT_EQ(param.gm_strides[1], af::Expression::Parse("2*s3*s4"));
  EXPECT_EQ(param.ub_strides.size(), 4);
  EXPECT_EQ(param.ub_strides[0], z2.size * af::sym::Align(z3.size * z4.size, 32));
  EXPECT_EQ(param.ub_strides[1], af::sym::Align(z3.size * z4.size, 32));
}

/*
 * Axis:              z0                     z1                      z2       z3    z4
 * repeat:            s0                     s1                      s2       s3    s4
 * stride:            s1*s2*(s3*s4+s3*s4)    s2*(s3*s4+s3*s4)        2*s3*s4  2*s4  one
 * vectorized axis:   z2        z1   z3 z4
 * vectorized stride: s1*s3*s4 s3*s4 s4 one
 */
TEST(CodegenLoadStore, CalculateDmaParams_ReorderContinusUbDiscreteGM) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));
  af::SizeVar s3(af::Symbol("s3"));
  af::SizeVar s4(af::Symbol("s4"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};
  af::Axis z3{.id = 3, .name = "z3", .type = af::Axis::Type::kAxisTypeOriginal, .size = s3.expr};
  af::Axis z4{.id = 4, .name = "z4", .type = af::Axis::Type::kAxisTypeOriginal, .size = s4.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddSizeVar(af::SizeVar(s3));
  tiler.AddSizeVar(af::SizeVar(s4));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddAxis(z4);

  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  tensor.attr.vectorized_axis = {z2.id, z1.id, z3.id, z4.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size, z3.size, z4.size};
  tensor.attr.strides = {z1.size * z2.size * (z3.size * (z4.size + z4.size)),
                         (z2.size) * (z3.size * (z4.size + z4.size)), z3.size * (z4.size + z4.size), z4.size + z4.size,
                         One};

  vector<af::Expression> vectorized_strides{z1.size * z3.size * z4.size, z3.size * z4.size, z4.size, One};
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  DataCopyParams param;
  Tensor tensor1(tensor, dtype_name);
  Tensor tensor2(tensor, dtype_name);
  tensor1.vectorized_axis_pos = {2, 1, 3, 4};
  tensor2.vectorized_axis_pos = {2, 1, 3, 4};
  EXPECT_EQ(CalculateDmaParams(tpipe, tensor1, tensor2, param), true);
  EXPECT_EQ(param.repeats.size(), 4);
  EXPECT_EQ(param.repeats[0], af::Expression::Parse("s2"));
  EXPECT_EQ(param.repeats[1], af::Expression::Parse("s1"));
  EXPECT_EQ(param.gm_strides.size(), 4);
  EXPECT_EQ(param.gm_strides[0], af::Expression::Parse("2*s3*s4"));
  EXPECT_EQ(param.gm_strides[1], af::Expression::Parse("2*s2*s3*s4"));
  EXPECT_EQ(param.ub_strides.size(), 4);
  EXPECT_EQ(param.ub_strides[0], af::Expression::Parse("s1*s3*s4"));
  EXPECT_EQ(param.ub_strides[1], af::Expression::Parse("s3*s4"));
}

/*
 * Axis:              z0     z1  z2
 * repeat:            s0     s1  s2
 * stride:            s1*s2  s2  Zero
 * vectorized stride: s2 one
 */
TEST(CodegenLoadStore, CalculateDmaParams_scalar) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};
  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.repeats = {One, One, One};
  tensor.attr.strides = {Zero, Zero, Zero};

  vector<af::Expression> vectorized_strides{One, One};
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  DataCopyParams param;
  Tensor tensor1(tensor, dtype_name);
  Tensor tensor2(tensor, dtype_name);
  tensor1.vectorized_axis_pos = {1, 2};
  tensor2.vectorized_axis_pos = {1, 2};
  EXPECT_EQ(CalculateDmaParams(tpipe, tensor1, tensor2, param), true);
  EXPECT_EQ(param.repeats.size(), 1);
  EXPECT_EQ(param.gm_strides.size(), 1);
  EXPECT_EQ(param.ub_strides.size(), 1);
}

TEST(CodegenLoadStore, CalculateDmaParams_LastAxisDisContinuous) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};
  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size};
  tensor.attr.strides = {z1.size * z2.size * af::Symbol(2), z2.size * af::Symbol(2), af::Symbol(2)};

  vector<af::Expression> vectorized_strides{z2.size * af::Symbol(2), af::Symbol(2)};
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  DataCopyParams param;
  Tensor tensor1(tensor, dtype_name);
  Tensor tensor2(tensor, dtype_name);
  tensor1.vectorized_axis_pos = {1, 2};
  tensor2.vectorized_axis_pos = {1, 2};
  EXPECT_EQ(CalculateDmaParams(tpipe, tensor1, tensor2, param), true);
  EXPECT_EQ(param.repeats.size(), 2);
  EXPECT_EQ(param.gm_strides.size(), 2);
  EXPECT_EQ(param.ub_strides.size(), 2);
}

TEST(CodegenLoadStore, CalculateDmaParams_IgnoresStaticSingletonAxes) {
  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeOriginal, .size = af::Symbol(8)};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeOriginal, .size = One};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = One};
  codegen::Tiler tiler;
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("static_singleton_dma");
  af::ascir_op::Data x("x", graph);
  auto tensor = graph.FindNode("x")->outputs[0];
  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.vectorized_axis = tensor.attr.axis;
  tensor.attr.repeats = {z0.size, One, One};
  tensor.attr.strides = {af::Symbol(6), af::Symbol(3), One};
  tensor.attr.vectorized_strides = {One, Zero, Zero};

  std::string dtype_name;
  Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  Tensor gm_tensor(tensor, dtype_name);
  Tensor ub_tensor(tensor, dtype_name);
  gm_tensor.vectorized_axis_pos = {0, 1, 2};
  ub_tensor.vectorized_axis_pos = {0, 1, 2};
  DataCopyParams param;
  ASSERT_TRUE(CalculateDmaParams(tpipe, gm_tensor, ub_tensor, param, true));
  ASSERT_EQ(param.repeats.size(), 1UL);
  EXPECT_EQ(param.repeats[0], af::Symbol(8));
}

TEST(CodegenLoadStore, CalculateDmaParams_KeepsDynamicAxes) {
  af::SizeVar dynamic_size(af::Symbol("dynamic_size"));
  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeOriginal, .size = af::Symbol(8)};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeOriginal, .size = dynamic_size.expr};
  codegen::Tiler tiler;
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(dynamic_size);
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("dynamic_dma");
  af::ascir_op::Data x("x", graph);
  auto tensor = graph.FindNode("x")->outputs[0];
  tensor.attr.axis = {z0.id, z1.id};
  tensor.attr.vectorized_axis = tensor.attr.axis;
  tensor.attr.repeats = {z0.size, z1.size};
  tensor.attr.strides = {z1.size, One};
  tensor.attr.vectorized_strides = {z1.size, One};

  std::string dtype_name;
  Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  Tensor gm_tensor(tensor, dtype_name);
  Tensor ub_tensor(tensor, dtype_name);
  DataCopyParams param;
  ASSERT_TRUE(CalculateDmaParams(tpipe, gm_tensor, ub_tensor, param, true));
  ASSERT_EQ(param.repeats.size(), 1UL);
  EXPECT_EQ(param.repeats[0], af::Symbol(8) * dynamic_size.expr);
}
