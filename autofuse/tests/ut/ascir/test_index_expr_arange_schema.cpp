/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascir.h"
#include "ascir_ops.h"

TEST(IndexExprSchema, StoresExpressionAttribute) {
  af::AscGraph graph("index_expr_schema");
  const auto expr = af::Expression::Parse("s0 + 2");

  (void)af::ascir::cg::IndexExpr("index", graph, af::DT_INT64, {}, {}, {}, expr);

  auto index_node = graph.FindNode("index");
  ASSERT_NE(index_node, nullptr);
  const auto &index_attr =
      dynamic_cast<const af::ascir_op::IndexExpr::AscIndexExprIrAttrDef &>(*index_node->attr.ir_attr);
  af::Expression stored_expr;
  ASSERT_EQ(index_attr.GetExpr(stored_expr), af::GRAPH_SUCCESS);
  EXPECT_EQ(stored_expr, expr);
  EXPECT_EQ(index_node->outputs[0].attr.dtype, af::DT_INT64);
  EXPECT_TRUE(index_node->outputs[0].attr.axis.empty());
}

TEST(IndexExprSchema, SupportsOnlyV2Int32AndInt64) {
  for (const auto dtype : {af::DT_INT32, af::DT_INT64}) {
    for (const auto *npu_arch : {"3510", "5102"}) {
      std::vector<af::DataType> output_dtypes{dtype};
      EXPECT_EQ(af::ascir_op::IndexExpr::InferDataType({}, output_dtypes, npu_arch), af::SUCCESS);
    }
  }

  std::vector<af::DataType> float_output{af::DT_FLOAT};
  EXPECT_NE(af::ascir_op::IndexExpr::InferDataType({}, float_output, "3510"), af::SUCCESS);
}

TEST(IndexExprSchema, MissingExpressionAttributeFails) {
  af::AscGraph graph("missing_index_expr");
  af::ascir_op::IndexExpr index("index", graph);
  af::Expression expr;

  EXPECT_NE(index.ir_attr.GetExpr(expr), af::GRAPH_SUCCESS);
}

TEST(ArangeSchema, StoresAttributesAndOneDimensionalLaneView) {
  af::AscGraph graph("arange_schema");
  const auto lane_count = graph.CreateSizeVar("lane_count");
  const auto lane = graph.CreateAxis("lane", lane_count);
  const auto base = af::Expression::Parse("s0 + 1");
  const auto step = af::Expression::Parse("2");

  (void)af::ascir::cg::Arange("arange", graph, af::DT_INT32, {lane.id}, std::vector<af::Expression>{lane_count},
                              std::vector<af::Expression>{af::Expression::Parse("1")}, base, step);

  auto arange_node = graph.FindNode("arange");
  ASSERT_NE(arange_node, nullptr);
  const auto &arange_attr = dynamic_cast<const af::ascir_op::Arange::AscArangeIrAttrDef &>(*arange_node->attr.ir_attr);
  af::Expression stored_base;
  af::Expression stored_step;
  ASSERT_EQ(arange_attr.GetBase(stored_base), af::GRAPH_SUCCESS);
  ASSERT_EQ(arange_attr.GetStep(stored_step), af::GRAPH_SUCCESS);
  EXPECT_EQ(stored_base, base);
  EXPECT_EQ(stored_step, step);
  EXPECT_EQ(arange_node->outputs[0].attr.dtype, af::DT_INT32);
  EXPECT_EQ(arange_node->outputs[0].attr.axis, std::vector<af::AxisId>{lane.id});
  EXPECT_EQ(arange_node->outputs[0].attr.repeats, std::vector<af::Expression>{lane_count});
  EXPECT_EQ(arange_node->outputs[0].attr.strides, std::vector<af::Expression>{af::Expression::Parse("1")});
}

TEST(ArangeSchema, SupportsOnlyV2Int32AndInt64) {
  for (const auto dtype : {af::DT_INT32, af::DT_INT64}) {
    for (const auto *npu_arch : {"3510", "5102"}) {
      std::vector<af::DataType> output_dtypes{dtype};
      EXPECT_EQ(af::ascir_op::Arange::InferDataType({}, output_dtypes, npu_arch), af::SUCCESS);
    }
  }

  std::vector<af::DataType> float_output{af::DT_FLOAT};
  const auto invalid_dtype_status = af::ascir_op::Arange::InferDataType({}, float_output, "3510");
  EXPECT_NE(invalid_dtype_status, af::SUCCESS);

  std::vector<af::DataType> v1_output{af::DT_INT32};
  EXPECT_NE(af::ascir_op::Arange::InferDataType({}, v1_output, "2201"), af::SUCCESS);

  std::vector<af::DataType> unknown_platform_output{af::DT_INT64};
  EXPECT_NE(af::ascir_op::Arange::InferDataType({}, unknown_platform_output, "unknown"), af::SUCCESS);
}

TEST(ArangeSchema, MissingBaseOrStepAttributeFails) {
  af::AscGraph graph("missing_arange_attrs");
  af::ascir_op::Arange arange("arange", graph);
  af::Expression base;
  af::Expression step;

  EXPECT_NE(arange.ir_attr.GetBase(base), af::GRAPH_SUCCESS);
  EXPECT_NE(arange.ir_attr.GetStep(step), af::GRAPH_SUCCESS);
}
