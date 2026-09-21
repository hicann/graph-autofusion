/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for the specific language governing permissions and limitations under the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_TESTS_FRAMEWORK_BROADCAST_BACKWARD_TEST_UTILS_H_
#define AUTOFUSE_TESTS_FRAMEWORK_BROADCAST_BACKWARD_TEST_UTILS_H_

#include "gtest/gtest.h"

#include <string>

#include "asc_graph_builder.h"
#include "ascgraph_info_complete.h"
#include "optimize/graph_pass/broadcast_backward_pass.h"
#include "platform_context.h"

namespace broadcast_backward_test {

inline af::testing::AscGraphBuilder BuildCommonAxisGraphPrefix(const std::string &name) {
  const auto s0 = af::testing::Sym("s0");
  const auto s1 = af::testing::Sym("s1");
  const auto s2 = af::testing::Sym("s2");
  af::testing::AscGraphBuilder builder(name);
  builder.Loops({s0, s1, s2})
      .Data("data0", 0)
      .Data("data1", 1)
      .Load("load0", "data0", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2},
            {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne})
      .Load("load1", "data1", {s0, af::sym::kSymbolOne, af::sym::kSymbolOne},
            {af::sym::kSymbolOne, af::sym::kSymbolZero, af::sym::kSymbolZero})
      .Broadcast("broadcast0", "load0", {0, 1})
      .Broadcast("broadcast1", "load1", {1, 2});
  return builder;
}

inline af::AscGraph BuildCommonAxisGraph(const std::string &name) {
  return BuildCommonAxisGraphPrefix(name)
      .Add("merge", "broadcast0", "broadcast1")
      .Store("store", "merge")
      .Output("output", "store")
      .Build();
}

inline af::AscGraph BuildDtypeAwareCommonAxisGraph(const std::string &name) {
  return BuildCommonAxisGraphPrefix(name)
      .Abs("abs", "broadcast0")
      .Cast("cast0", "abs", af::DT_FLOAT16)
      .Cast("cast1", "broadcast1", af::DT_FLOAT16)
      .Relu("relu", "cast1")
      .Add("merge", "cast0", "relu")
      .Store("store", "merge")
      .Output("output", "store")
      .Build();
}

inline af::AscNodePtr FindNode(af::AscGraph &graph, const std::string &name) {
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetName() == name) {
      return std::dynamic_pointer_cast<af::AscNode>(node);
    }
  }
  return nullptr;
}

inline std::string GetInputNodeName(const af::AscNodePtr &node, size_t input_index = 0U) {
  if (node == nullptr || input_index >= node->GetInDataNodesSize()) {
    return {};
  }
  const auto input_node = node->GetInDataNodes().at(input_index);
  return input_node == nullptr ? std::string() : input_node->GetName();
}

inline void CompleteApiInfo(af::AscGraph &graph) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  ASSERT_EQ(optimize::AscGraphInfoComplete::CompleteApiInfo(graph), af::SUCCESS);
}

inline void SetNodeDtype(af::AscGraph &graph, const std::string &name, af::DataType dtype) {
  const auto node = FindNode(graph, name);
  ASSERT_NE(node, nullptr);
  const auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  for (size_t input_index = 0U; input_index < node->GetAllInDataAnchorsSize(); ++input_index) {
    const auto input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(input_index));
    ASSERT_NE(input_desc, nullptr);
    input_desc->SetDataType(dtype);
  }
  node->outputs[0].attr.dtype = dtype;
}

}  // namespace broadcast_backward_test

#endif  // AUTOFUSE_TESTS_FRAMEWORK_BROADCAST_BACKWARD_TEST_UTILS_H_
