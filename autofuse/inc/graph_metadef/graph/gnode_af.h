/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GRAPH_GNODE_H_
#define INC_EXTERNAL_GRAPH_GNODE_H_

#include <vector>
#include <cstdint>

#include "graph/ge_error_codes_af.h"
#include "graph/types_af.h"
#include "graph/tensor_af.h"
#include "graph/ascend_string_af.h"

namespace af {
class AttrValue;
class GNode;
class OpDesc;
class Graph;
class ComputeGraph;
using GNodePtr = std::shared_ptr<GNode>;
using GraphPtr = std::shared_ptr<Graph>;
using OpBytes = std::vector<uint8_t>;
using OpDescPtr = std::shared_ptr<OpDesc>;
using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;

class NodeImpl;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GNode {
 public:
  GNode();

  ~GNode() = default;

  graphStatus GetType(AscendString &type) const;

  graphStatus GetName(AscendString &name) const;

  std::pair<GNodePtr, int32_t> GetInDataNodesAndPortIndexs(const int32_t index) const;

  std::vector<GNodePtr> GetInControlNodes() const;

  std::vector<std::pair<GNodePtr, int32_t>> GetOutDataNodesAndPortIndexs(const int32_t index) const;

  std::vector<GNodePtr> GetOutControlNodes() const;

  graphStatus GetInputConstData(const int32_t index, Tensor &data) const;

  graphStatus GetInputIndexByName(const AscendString &name, int32_t &index);

  graphStatus GetOutputIndexByName(const AscendString &name, int32_t &index);

  graphStatus GetDynamicInputIndexesByName(const AscendString &name, std::vector<int32_t> &indexes);

  graphStatus GetDynamicOutputIndexesByName(const AscendString &name, std::vector<int32_t> &indexes);

  size_t GetInputsSize() const;

  size_t GetOutputsSize() const;

  graphStatus GetInputDesc(const int32_t index, TensorDesc &tensor_desc) const;

  graphStatus UpdateInputDesc(const int32_t index, const TensorDesc &tensor_desc);

  graphStatus GetOutputDesc(const int32_t index, TensorDesc &tensor_desc) const;

  graphStatus UpdateOutputDesc(const int32_t index, const TensorDesc &tensor_desc);

  graphStatus GetAttr(const AscendString &name, int64_t &attr_value) const;
  graphStatus GetAttr(const AscendString &name, int32_t &attr_value) const;
  graphStatus GetAttr(const AscendString &name, uint32_t &attr_value) const;
  graphStatus GetAttr(const AscendString &name, float32_t &attr_value) const;
  graphStatus GetAttr(const AscendString &name, AscendString &attr_value) const;
  graphStatus GetAttr(const AscendString &name, bool &attr_value) const;
  graphStatus GetAttr(const AscendString &name, Tensor &attr_value) const;
  graphStatus GetAttr(const AscendString &name, std::vector<int64_t> &attr_value) const;
  graphStatus GetAttr(const AscendString &name, std::vector<int32_t> &attr_value) const;
  graphStatus GetAttr(const AscendString &name, std::vector<uint32_t> &attr_value) const;
  graphStatus GetAttr(const AscendString &name, std::vector<float32_t> &attr_value) const;
  graphStatus GetAttr(const AscendString &name, std::vector<AscendString> &attr_values) const;
  graphStatus GetAttr(const AscendString &name, std::vector<bool> &attr_value) const;
  graphStatus GetAttr(const AscendString &name, std::vector<Tensor> &attr_value) const;
  graphStatus GetAttr(const AscendString &name, OpBytes &attr_value) const;
  graphStatus GetAttr(const AscendString &name, std::vector<std::vector<int64_t>> &attr_value) const;
  graphStatus GetAttr(const AscendString &name, std::vector<ge::DataType> &attr_value) const;
  graphStatus GetAttr(const AscendString &name, ge::DataType &attr_value) const;
  graphStatus GetAttr(const AscendString &name, AttrValue &attr_value) const;

  graphStatus SetAttr(const AscendString &name, int64_t &attr_value) const;
  graphStatus SetAttr(const AscendString &name, int32_t &attr_value) const;
  graphStatus SetAttr(const AscendString &name, uint32_t &attr_value) const;
  graphStatus SetAttr(const AscendString &name, float32_t &attr_value) const;
  graphStatus SetAttr(const AscendString &name, AscendString &attr_value) const;
  graphStatus SetAttr(const AscendString &name, bool &attr_value) const;
  graphStatus SetAttr(const AscendString &name, Tensor &attr_value) const;
  graphStatus SetAttr(const AscendString &name, std::vector<int64_t> &attr_value) const;
  graphStatus SetAttr(const AscendString &name, std::vector<int32_t> &attr_value) const;
  graphStatus SetAttr(const AscendString &name, std::vector<uint32_t> &attr_value) const;
  graphStatus SetAttr(const AscendString &name, std::vector<float32_t> &attr_value) const;
  graphStatus SetAttr(const AscendString &name, std::vector<AscendString> &attr_values) const;
  graphStatus SetAttr(const AscendString &name, std::vector<bool> &attr_value) const;
  graphStatus SetAttr(const AscendString &name, std::vector<Tensor> &attr_value) const;
  graphStatus SetAttr(const AscendString &name, OpBytes &attr_value) const;
  graphStatus SetAttr(const AscendString &name, std::vector<std::vector<int64_t>> &attr_value) const;
  graphStatus SetAttr(const AscendString &name, std::vector<ge::DataType> &attr_value) const;
  graphStatus SetAttr(const AscendString &name, ge::DataType &attr_value) const;
  graphStatus SetAttr(const AscendString &name, AttrValue &attr_value) const;

  // 添加AttrValue类型的输入输出属性支持
  graphStatus GetOutputAttr(const AscendString &name, uint32_t output_index, AttrValue &attr_value) const;
  graphStatus SetOutputAttr(const AscendString &name, uint32_t output_index, const AttrValue &attr_value);
  graphStatus GetInputAttr(const AscendString &name, uint32_t input_index, AttrValue &attr_value) const;
  graphStatus SetInputAttr(const AscendString &name, uint32_t input_index, const AttrValue &attr_value);

  bool HasAttr(const AscendString &name);

  graphStatus GetSubgraph(uint32_t index, GraphPtr &graph) const;

  graphStatus GetALLSubgraphs(std::vector<GraphPtr> &graph_list) const;

  /**
   * @brief Add the subgraph to the node
   * @param subgraph_ir_name IR subgraph name
   * @param subgraph the subgraph to be added
   * @return GRAPH_SUCCESS: success, others: failed
   */
  graphStatus SetSubgraph(const AscendString &subgraph_ir_name, const Graph &subgraph);

  /**
   * @brief Add subgraphs to the node
   * @param subgraph_ir_name Dynamic IR subgraphs name
   * @param subgraphs subgraphs to be added
   * @return GRAPH_SUCCESS: success, others: failed
   */
  graphStatus SetSubgraphs(const AscendString &subgraph_ir_name, const std::vector<Graph> &subgraphs);

 private:
  std::shared_ptr<NodeImpl> impl_;
  friend class NodeAdapter;
};
}  // namespace af

#endif  // INC_EXTERNAL_GRAPH_GNODE_H_
