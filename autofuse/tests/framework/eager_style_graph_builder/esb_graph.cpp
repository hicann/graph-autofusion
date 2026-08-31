/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "esb_graph.h"
#include "graph/utils/op_type_utils.h"
#include "graph/operator_factory_af.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "compliant_op_desc_builder.h"

using namespace af;

namespace {
OpDescPtr CreateCompliantData(const char_t *name, const char_t *type, int64_t index) {
  return CompliantOpDescBuilder()
      .OpType(type)
      .Name(name)
      .IrDefInputs({{"x", kIrInputRequired, "x"}})
      .IrDefOutputs({{"y", kIrOutputRequired, "y"}})
      .IrDefAttrs({{"index", kAttrOptional, "Int", AnyValue::CreateFrom(index)}})
      .Build();
}
OpDescPtr CreateCompliantNetOutput(int32_t output_num) {
  return CompliantOpDescBuilder()
      .OpType("NetOutput")
      .Name("NetOutput")
      .IrDefInputs({{"x", kIrInputDynamic, ""}})
      .IrDefOutputs({{"y", kIrOutputDynamic, ""}})
      .InstanceDynamicInputNum("x", output_num)
      .InstanceDynamicOutputNum("y", output_num)
      .Build();
}
}  // namespace
EsbTensor *EsbGraph::GetEsbTensorFromNode(NodePtr node, int32_t output_index) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT(!OpTypeUtils::IsDataNode(node->GetType()), "Call `EsbGraph::AppendGraphInput` to add graph input");
  return GetEsbTensorFromNodeInner(std::move(node), output_index);
}
EsbTensor *EsbGraph::GetEsbTensorFromNodeInner(NodePtr node, int32_t output_index) {
  auto tensor = ComGraphMakeUnique<EsbTensor>(*this, std::move(node), output_index);
  GE_ASSERT_NOTNULL(tensor);
  tensors_holder_.emplace_back(std::move(tensor));
  return tensors_holder_.back().get();
}
EsbTensor *EsbGraph::AddGraphInput(int32_t index, const char_t *name, const char_t *type) {
  GE_ASSERT_TRUE(index >= 0, "Invalid input index %d, must be non-negative", index);
  GE_ASSERT_TRUE(graph_input_indexes_.count(index) == 0, "Duplicated graph input index %d", index);

  if (type == nullptr) {
    type = "Data";
  } else {
    GE_ASSERT_TRUE(OpTypeUtils::IsDataNode(type));
  }

  OpDescPtr op_desc;
  if (name == nullptr) {
    std::string cpp_name = "input_" + std::to_string(index);
    op_desc = CreateCompliantData(cpp_name.c_str(), type, index);
  } else {
    op_desc = CreateCompliantData(name, type, index);
  }
  GE_ASSERT_NOTNULL(op_desc);
  auto node = graph_->AddNode(op_desc);
  GE_ASSERT_NOTNULL(node);
  graph_input_indexes_.insert(index);
  return GetEsbTensorFromNodeInner(std::move(node), 0);
}
Status EsbGraph::SetGraphOutput(EsbTensor *tensor, int32_t output_index) {
  GE_ASSERT_NOTNULL(tensor);
  GE_ASSERT_TRUE(output_index >= 0, "Invalid output index %d, must be non-negative", output_index);
  GE_ASSERT_TRUE(output_indexes_to_tensor_.count(output_index) == 0, "Duplicated output index %d", output_index);
  output_indexes_to_tensor_[output_index] = tensor;
  return SUCCESS;
}
bool EsbGraph::IsGraphValid() const {
  if (!graph_input_indexes_.empty()) {
    GE_ASSERT_TRUE(*graph_input_indexes_.begin() == 0, "Invalid graph, graph input index must start with 0");
    if (static_cast<size_t>(*graph_input_indexes_.rbegin()) + 1U != graph_input_indexes_.size()) {
      std::stringstream ss;
      ss << "Invalid graph, graph input indexes are not continuous: ";
      bool first = true;
      for (auto index : graph_input_indexes_) {
        if (!first) {
          ss << ", ";
        }
        ss << index;
        first = false;
      }
      GELOGE(GRAPH_FAILED, "%s", ss.str().c_str());
      return false;
    }
  }

  if (!output_indexes_to_tensor_.empty()) {
    GE_ASSERT_TRUE(output_indexes_to_tensor_.begin()->first == 0, "Invalid graph, output index must start with 0");
    if (static_cast<size_t>(output_indexes_to_tensor_.rbegin()->first) + 1U != output_indexes_to_tensor_.size()) {
      std::stringstream ss;
      ss << "Invalid graph, output indexes are not continuous: ";
      bool first = true;
      for (auto index : output_indexes_to_tensor_) {
        if (!first) {
          ss << ", ";
        }
        ss << index.first;
        first = false;
      }
      GELOGE(GRAPH_FAILED, "%s", ss.str().c_str());
      return false;
    }
  }

  return true;
}
ComputeGraphPtr EsbGraph::BuildComputeGraph() {
  GE_ASSERT_TRUE(IsGraphValid());

  auto desc = CreateCompliantNetOutput(static_cast<int32_t>(output_indexes_to_tensor_.size()));
  GE_ASSERT_NOTNULL(desc);

  auto node = graph_->AddNode(desc);
  GE_ASSERT_NOTNULL(node);

  for (const auto &output : output_indexes_to_tensor_) {
    auto tensor = output.second;
    GE_ASSERT_NOTNULL(tensor);
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(tensor->GetAnchor(), node->GetInDataAnchor(output.first)));
  }

  return std::move(graph_);
}
std::unique_ptr<Graph> EsbGraph::BuildGraph() {
  return GraphUtilsEx::CreateGraphUniquePtrFromComputeGraph(BuildComputeGraph());
}
