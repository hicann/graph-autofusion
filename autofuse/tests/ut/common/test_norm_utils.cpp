/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#include "gtest/gtest.h"

#include <string>
#include <vector>

#include "ascir_ops.h"
#include "norm_utils.h"

namespace {
af::AscNodePtr BuildSingleLoadNode() {
  af::AscGraph &graph = *new af::AscGraph("norm_utils_ut_graph");
  const af::Expression s0 = graph.CreateSizeVar(2);
  const af::Expression s1 = graph.CreateSizeVar(3);
  const auto y0 = graph.CreateAxis("y0", s0);
  const auto y1 = graph.CreateAxis("y1", s1);
  const std::vector<af::AxisId> axes = {y0.id, y1.id};
  const std::vector<af::Expression> repeats = {s0, s1};
  const std::vector<af::Expression> strides = {s1, af::sym::kSymbolOne};

  af::ascir_op::Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = af::DT_FLOAT16;
  data.attr.sched.axis = axes;
  *data.y.axis = axes;
  *data.y.repeats = repeats;
  *data.y.strides = strides;
  af::ascir_op::Load load("load");
  load.x = data.y;
  load.y.dtype = af::DT_FLOAT16;
  load.attr.sched.axis = axes;
  *load.y.axis = axes;
  *load.y.repeats = repeats;
  *load.y.strides = strides;
  return graph.FindNode("load");
}
}  // namespace

TEST(NormUtilsTest, NormInfoDefaultsAreEmpty) {
  ascgen_utils::norm::NormInfo info;
  EXPECT_EQ(info.kind, ascgen_utils::norm::NormInfo::Kind::kNone);
  EXPECT_TRUE(info.entry_node_name.empty());
  EXPECT_TRUE(info.exit_node_name.empty());
  EXPECT_TRUE(info.region_node_names.empty());
  EXPECT_TRUE(info.stages.empty());
  EXPECT_TRUE(info.entry_axes.empty());
  EXPECT_TRUE(info.preserved_axes.empty());
  EXPECT_EQ(info.softmax_reduce_axis, af::kIdNone);
}

TEST(NormUtilsTest, HasNormInfoReflectsSetState) {
  const auto node = BuildSingleLoadNode();
  ASSERT_NE(node, nullptr);
  EXPECT_FALSE(ascgen_utils::norm::HasNormInfo(node));

  ascgen_utils::norm::NormInfo info;
  info.kind = ascgen_utils::norm::NormInfo::Kind::kGenericComposite;
  info.entry_node_name = "entry";
  ASSERT_EQ(ascgen_utils::norm::SetNormInfo(node, info), af::SUCCESS);
  EXPECT_TRUE(ascgen_utils::norm::HasNormInfo(node));
}

TEST(NormUtilsTest, SetAndTryGetNormInfoRoundTrip) {
  const auto node = BuildSingleLoadNode();
  ASSERT_NE(node, nullptr);

  ascgen_utils::norm::NormInfo info;
  info.kind = ascgen_utils::norm::NormInfo::Kind::kSoftmaxDedicated;
  info.entry_node_name = "softmax_entry";
  info.exit_node_name = "softmax_entry";
  info.region_node_names = {"softmax_entry"};
  info.stages = {{"softmax_entry", "", {5L}}};
  info.entry_axes = {1L, 2L};
  info.preserved_axes = {1L};
  info.softmax_reduce_axis = 2L;
  ASSERT_EQ(ascgen_utils::norm::SetNormInfo(node, info), af::SUCCESS);

  ascgen_utils::norm::NormInfo loaded;
  ASSERT_EQ(ascgen_utils::norm::TryGetNormInfo(node, loaded), af::SUCCESS);
  EXPECT_EQ(loaded.kind, ascgen_utils::norm::NormInfo::Kind::kSoftmaxDedicated);
  EXPECT_EQ(loaded.entry_node_name, "softmax_entry");
  EXPECT_EQ(loaded.exit_node_name, "softmax_entry");
  EXPECT_EQ(loaded.region_node_names, std::vector<std::string>{"softmax_entry"});
  ASSERT_EQ(loaded.stages.size(), 1UL);
  EXPECT_EQ(loaded.stages[0].reduce_node_name, "softmax_entry");
  EXPECT_EQ(loaded.stages[0].broadcast_node_name, "");
  EXPECT_EQ(loaded.stages[0].reduced_axes, std::vector<af::AxisId>{5L});
  EXPECT_EQ(loaded.entry_axes, std::vector<af::AxisId>({1L, 2L}));
  EXPECT_EQ(loaded.preserved_axes, std::vector<af::AxisId>{1L});
  EXPECT_EQ(loaded.softmax_reduce_axis, 2L);
}

TEST(NormUtilsTest, SetNormInfoOverwritesPreviousValue) {
  const auto node = BuildSingleLoadNode();
  ASSERT_NE(node, nullptr);

  ascgen_utils::norm::NormInfo first;
  first.kind = ascgen_utils::norm::NormInfo::Kind::kGenericComposite;
  first.entry_node_name = "first_entry";
  ASSERT_EQ(ascgen_utils::norm::SetNormInfo(node, first), af::SUCCESS);

  ascgen_utils::norm::NormInfo second;
  second.kind = ascgen_utils::norm::NormInfo::Kind::kSoftmaxDedicated;
  second.entry_node_name = "second_entry";
  ASSERT_EQ(ascgen_utils::norm::SetNormInfo(node, second), af::SUCCESS);

  ascgen_utils::norm::NormInfo loaded;
  ASSERT_EQ(ascgen_utils::norm::TryGetNormInfo(node, loaded), af::SUCCESS);
  EXPECT_EQ(loaded.kind, ascgen_utils::norm::NormInfo::Kind::kSoftmaxDedicated);
  EXPECT_EQ(loaded.entry_node_name, "second_entry");
}

TEST(NormUtilsTest, TryGetNormInfoResetsToDefaultWithoutSet) {
  const auto node = BuildSingleLoadNode();
  ASSERT_NE(node, nullptr);
  ascgen_utils::norm::NormInfo loaded;
  loaded.kind = ascgen_utils::norm::NormInfo::Kind::kGenericComposite;
  loaded.entry_node_name = "stale";
  // 未设置时 TryGetNormInfo 成功返回并把出参重置为默认值。
  ASSERT_EQ(ascgen_utils::norm::TryGetNormInfo(node, loaded), af::SUCCESS);
  EXPECT_EQ(loaded.kind, ascgen_utils::norm::NormInfo::Kind::kNone);
  EXPECT_TRUE(loaded.entry_node_name.empty());
}
