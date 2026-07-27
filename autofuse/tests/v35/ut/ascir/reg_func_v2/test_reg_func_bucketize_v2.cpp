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

#include "graph/operator_reg_af.h"
#include "graph_utils_ex.h"
#include "node_utils.h"
#include "op_desc_utils.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "default_reg_func_v2.h"

namespace af {
namespace ascir {

using namespace testing;
using namespace af::ascir_op;

extern std::vector<std::unique_ptr<af::TmpBufDesc>> CalcBucketizeTmpSizeV2(const af::AscNode &node);

class CalcBucketizeTmpSizeV2Test : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  // Sets up a 2D Data+Load pair with the given data type and axis/repeat configuration.
  static void SetupDataAndLoad(af::ascir_op::Data &data, af::ascir_op::Load &load, af::DataType dtype,
                               af::Expression s0, af::Expression s1, int64_t z0_id, int64_t z1_id) {
    data.attr.sched.axis = {z0_id, z1_id};
    data.y.dtype = dtype;
    *data.y.axis = {z0_id, z1_id};
    *data.y.repeats = {s0, s1};
    *data.y.strides = {s1, Symbol(1)};

    load.x = data.y;
    load.attr.sched.axis = {z0_id, z1_id};
    load.y.dtype = dtype;
    *load.y.axis = {z0_id, z1_id};
    *load.y.repeats = {s0, s1};
    *load.y.strides = {s1, Symbol(1)};
    *load.y.vectorized_axis = {z0_id, z1_id};
  }

  // Builds a b8-type bucketize graph with 2D src (repeats={s0,s1}) and 1D boundaries (repeats={s2}).
  static void BuildB8Graph(af::AscGraph &graph, af::DataType data_type, af::Expression &out_s0, af::Expression &out_s1,
                           af::Expression &out_s2, std::shared_ptr<af::AscNode> &out_node) {
    out_s0 = graph.CreateSizeVar("s0");
    out_s1 = graph.CreateSizeVar("s1");
    out_s2 = graph.CreateSizeVar("s2");

    auto z0 = graph.CreateAxis("z0", out_s0);
    auto z1 = graph.CreateAxis("z1", out_s1);
    auto z2 = graph.CreateAxis("z2", out_s2);

    af::ascir_op::Data x1("x1", graph);
    af::ascir_op::Data x2("x2", graph);
    af::ascir_op::Load load1("load1");
    af::ascir_op::Load load2("load2");
    af::ascir_op::Add bucketize("bucketize");
    af::ascir_op::Store store("store");
    af::ascir_op::Output y("y");

    // ---- src input (2D) ----
    SetupDataAndLoad(x1, load1, data_type, out_s0, out_s1, z0.id, z1.id);

    // ---- boundaries input (1D) ----
    x2.attr.sched.axis = {z2.id};
    x2.y.dtype = data_type;
    *x2.y.axis = {z2.id};
    *x2.y.repeats = {out_s2};
    *x2.y.strides = {Symbol(1)};

    load2.x = x2.y;
    load2.attr.sched.axis = {z2.id};
    load2.y.dtype = data_type;
    *load2.y.axis = {z2.id};
    *load2.y.repeats = {out_s2};
    *load2.y.strides = {Symbol(1)};
    *load2.y.vectorized_axis = {z2.id};

    // ---- bucketize / store / output ----
    bucketize.x1 = load1.y;
    bucketize.x2 = load2.y;
    bucketize.attr.sched.axis = {z0.id, z1.id};
    bucketize.y.dtype = af::DT_INT32;
    *bucketize.y.axis = {z0.id, z1.id};
    *bucketize.y.repeats = {out_s0, out_s1};
    *bucketize.y.strides = {out_s1, Symbol(1)};

    store.x = bucketize.y;
    store.attr.sched.axis = {z0.id, z1.id};
    store.y.dtype = af::DT_INT32;
    *store.y.axis = {z0.id, z1.id};
    *store.y.repeats = {out_s0, out_s1};
    *store.y.strides = {out_s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, z1.id};
    y.y.dtype = af::DT_INT32;
    *y.y.axis = {z0.id, z1.id};
    *y.y.repeats = {out_s0, out_s1};
    *y.y.strides = {out_s1, Symbol(1)};

    out_node = graph.FindNode("bucketize");
  }

  // Builds a single-input (or pseudo-2-input) bucketize graph for the MIN_TMP_SIZE path.
  // When set_boundaries=true, x2 reuses the loaded src so the node has 2 inputs (verifies
  // non-b8 types still fall through to 32). When false, x2 is left unset (verifies b8 types
  // with only 1 input also fall through).
  static void BuildMinSizeGraph(af::AscGraph &graph, af::DataType data_type, bool set_boundaries,
                                std::shared_ptr<af::AscNode> &out_node) {
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);

    af::ascir_op::Data x1("x1", graph);
    af::ascir_op::Load load1("load1");
    af::ascir_op::Add bucketize("bucketize");
    af::ascir_op::Store store("store");
    af::ascir_op::Output y("y");

    SetupDataAndLoad(x1, load1, data_type, s0, s1, z0.id, z1.id);

    bucketize.x1 = load1.y;
    if (set_boundaries) {
      bucketize.x2 = load1.y;
    }
    bucketize.attr.sched.axis = {z0.id, z1.id};
    bucketize.y.dtype = af::DT_INT32;
    *bucketize.y.axis = {z0.id, z1.id};
    *bucketize.y.repeats = {s0, s1};
    *bucketize.y.strides = {s1, Symbol(1)};

    store.x = bucketize.y;
    store.attr.sched.axis = {z0.id, z1.id};
    store.y.dtype = af::DT_INT32;
    *store.y.axis = {z0.id, z1.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, z1.id};
    y.y.dtype = af::DT_INT32;
    *y.y.axis = {z0.id, z1.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    out_node = graph.FindNode("bucketize");
  }
};

// Non-b8 types: tmp size is always 32 (BUCKETIZE_MIN_TMP_SIZE).
TEST_F(CalcBucketizeTmpSizeV2Test, CalcBucketizeTmpSize_ShouldReturnMinSize_WhenSrcIsFloat) {
  af::AscGraph graph("test");
  std::shared_ptr<af::AscNode> node;
  BuildMinSizeGraph(graph, af::DT_FLOAT, true, node);

  auto result = CalcBucketizeTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

// b8 type (INT8): PromoteB8 writes CeilDiv(count, vecLen) * GetVecLen() * 2 bytes.
// Where vecLen = GetVecLen() / sizeof(T) = 256 elements for b8.
// Total tmp = AlignUp(calCount, 256) * 2 + AlignUp(boundCount, 256) * 2.
// With src repeats={s0,s1} and bound repeats={s2}, the expected expression is:
//   (AlignUp(s0*s1, 256) + AlignUp(s2, 256)) * 2
TEST_F(CalcBucketizeTmpSizeV2Test, CalcBucketizeTmpSize_ShouldReturnPromotedSize_WhenSrcIsInt8) {
  af::AscGraph graph("test");
  af::Expression s0, s1, s2;
  std::shared_ptr<af::AscNode> node;
  BuildB8Graph(graph, af::DT_INT8, s0, s1, s2, node);

  ASSERT_EQ(node->inputs.Size(), static_cast<size_t>(2));
  auto result = CalcBucketizeTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->life_time_axis_id, -1);

  auto expected_size =
      sym::Mul(sym::Add(sym::Align(sym::Mul(s0, s1), 256), sym::Align(s2, 256)), af::Symbol(2)).Simplify();
  ASSERT_TRUE(sym::AssertSymbolEq(result[0]->size, expected_size, __FILE__, __LINE__));
}

// b8 type (UINT8): same AlignUp formula as INT8.
TEST_F(CalcBucketizeTmpSizeV2Test, CalcBucketizeTmpSize_ShouldReturnPromotedSize_WhenSrcIsUint8) {
  af::AscGraph graph("test");
  af::Expression s0, s1, s2;
  std::shared_ptr<af::AscNode> node;
  BuildB8Graph(graph, af::DT_UINT8, s0, s1, s2, node);

  ASSERT_EQ(node->inputs.Size(), static_cast<size_t>(2));
  auto result = CalcBucketizeTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->life_time_axis_id, -1);

  auto expected_size =
      sym::Mul(sym::Add(sym::Align(sym::Mul(s0, s1), 256), sym::Align(s2, 256)), af::Symbol(2)).Simplify();
  ASSERT_TRUE(sym::AssertSymbolEq(result[0]->size, expected_size, __FILE__, __LINE__));
}

// MIN_TMP_SIZE path: when node has only 1 input (no boundaries), falls back to 32.
TEST_F(CalcBucketizeTmpSizeV2Test, CalcBucketizeTmpSize_ShouldReturnMinSize_WhenSingleInput) {
  af::AscGraph graph("test");
  std::shared_ptr<af::AscNode> node;
  BuildMinSizeGraph(graph, af::DT_INT8, false, node);

  auto result = CalcBucketizeTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

}  // namespace ascir
}  // namespace af
