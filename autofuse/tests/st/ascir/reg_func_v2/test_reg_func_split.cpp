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

#include "graph/operator_reg_af.h"
#include "graph_utils_ex.h"
#include "node_utils.h"
#include "op_desc_utils.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "../../compiler/graph/optimize/autofuse/ascir/reg_func/default_reg_func.h"

namespace af {
namespace ascir {

using namespace testing;
using namespace af::ascir_op;

class CalcSplitTmpSizeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};
template <ge::DataType T>
void CreateStaticGraphSplit(af::AscGraph &graph, int32_t dim0, int32_t dim1, int32_t dim2) {
  af::Expression One = af::Symbol(1);
  auto s0 = graph.CreateSizeVar(dim0);
  auto s1 = graph.CreateSizeVar(dim1);
  auto s2 = graph.CreateSizeVar(dim2);

  auto z0 = graph.CreateAxis("z0", s0);
  auto zo = graph.CreateAxis("zo", s1 + s2);
  auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, af::kIdNone);
  auto zo_s_1 = graph.CreateAxis("zo_s_1", Axis::Type::kAxisTypeOriginal, s2, {zo.id}, af::kIdNone);

  Data x1("x1", graph);
  Load load1("load");
  af::ascir_op::Split split("split");
  Store store1("store1");
  Store store2("store2");
  Output y0("y0");
  Output y1("y1");

  x1.attr.sched.axis = {z0.id, zo.id};
  x1.y.dtype = T;
  *x1.y.axis = {z0.id, zo.id};
  *x1.y.repeats = {s0, s1 + s2};
  *x1.y.strides = {s1 + s2, One};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, zo.id};
  load1.y.dtype = T;
  *load1.y.axis = {z0.id, zo.id};
  *load1.y.repeats = {s0, s1 + s2};
  *load1.y.strides = {s1 + s2, One};

  split.InstanceOutputy(2U);
  split.x = load1.y;
  split.attr.sched.axis = {z0.id, zo.id};
  split.y[0].dtype = T;
  *split.y[0].axis = {z0.id, zo_s_0.id};
  *split.y[0].repeats = {s0, s1};
  *split.y[0].strides = {s1, One};

  split.y[1].dtype = T;
  *split.y[1].axis = {z0.id, zo_s_1.id};
  *split.y[1].repeats = {s0, s2};
  *split.y[1].strides = {s2, One};

  store1.x = split.y[0];
  store1.attr.sched.axis = {z0.id, zo_s_0.id};
  store1.y.dtype = T;
  *store1.y.axis = {z0.id, zo_s_0.id};
  *store1.y.repeats = {s0, s1};
  *store1.y.strides = {s1, One};

  store2.x = split.y[0];
  store2.attr.sched.axis = {z0.id, zo.id};
  store2.y.dtype = T;
  *store2.y.axis = {z0.id, zo_s_1.id};
  *store2.y.repeats = {s0, s2};
  *store2.y.strides = {s2, One};

  y0.x = store1.y;
  y0.attr.sched.axis = {z0.id, zo_s_0.id};
  y0.y.dtype = T;
  *y0.y.axis = {z0.id, zo_s_0.id};
  *y0.y.repeats = {s0, s1};
  *y0.y.strides = {s1, One};

  y1.x = store2.y;
  y1.attr.sched.axis = {z0.id, zo_s_1.id};
  y1.y.dtype = T;
  *y1.y.axis = {z0.id, zo_s_1.id};
  *y1.y.repeats = {s0, s2};
  *y1.y.strides = {s2, One};
}

TEST_F(CalcSplitTmpSizeTest, CalcSplitTmpSizeV2_AllAligned) {
  af::AscGraph graph("test");
  CreateStaticGraphSplit<ge::DT_FLOAT>(graph, 100, 8, 16);
  std::shared_ptr<af::AscNode> node = graph.FindNode("split");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcSplitTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 0);
}

TEST_F(CalcSplitTmpSizeTest, CalcSplitTmpSizeV2_NotAllAligned) {
  af::AscGraph graph("test");
  CreateStaticGraphSplit<ge::DT_FLOAT>(graph, 100, 7, 15);
  std::shared_ptr<af::AscNode> node = graph.FindNode("split");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcSplitTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, 1024);
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

}  // namespace ascir
}  // namespace af
