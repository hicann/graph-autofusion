/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base_alignment_strategy.h"
#include "common_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "platform/platform_factory.h"

namespace optimize {

bool IsTailAxisTranspose(const af::AscTensorAttr &attr) {
  const size_t axis_size = attr.vectorized_axis.size();
  if (axis_size <= 1UL) {
    return false;
  }
  // check tail_axis
  auto tail_axis_iter = std::find(attr.axis.begin(), attr.axis.end(), attr.vectorized_axis[axis_size - 1UL]);
  GE_ASSERT_TRUE(tail_axis_iter != attr.axis.end(), "Cannot find vectorized axis [%ld], axis attr may be invalid.",
                 attr.vectorized_axis[axis_size - 1UL]);
  const size_t tail_index = std::distance(attr.axis.begin(), tail_axis_iter);
  if (af::SymbolicUtils::StaticCheckEq(attr.strides[tail_index], af::sym::kSymbolZero) != af::TriBool::kTrue) {
    return false;
  }

  for (size_t i = static_cast<size_t>(axis_size - 1UL); i > 0UL; --i) {
    auto iter = std::find(attr.axis.begin(), attr.axis.end(), attr.vectorized_axis[i]);
    GE_ASSERT_TRUE(tail_axis_iter != attr.axis.end(), "Cannot find vectorized axis [%ld], axis attr may be invalid.",
                   attr.vectorized_axis[i]);
    const size_t index = std::distance(attr.axis.begin(), iter);
    if (af::SymbolicUtils::StaticCheckEq(attr.strides[index], af::sym::kSymbolZero) != af::TriBool::kTrue) {
      return true;
    }
  }

  // 非连续搬运
  for (auto id = attr.vectorized_axis.rbegin(); id != attr.vectorized_axis.rend(); ++id) {
    auto iter = std::find(attr.axis.begin(), attr.axis.end(), *id);
    GE_ASSERT_TRUE(iter != attr.axis.end(), "Cannot find vectorized axis [%ld], axis attr may be invalid.", *id);
    const size_t index = std::distance(attr.axis.begin(), iter);
    // 考虑到通用模板要兼顾reduce的限制, 因此,尾轴为1的非连续load,不会当成DisContinuous处理
    if (af::SymbolicUtils::StaticCheckEq(attr.strides[index], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    return af::SymbolicUtils::StaticCheckNe(attr.strides[index], af::sym::kSymbolOne) == af::TriBool::kTrue;
  }
  return false;
}

bool IsHasReduceNode(const af::AscNodePtr &node) {
  std::set<af::Node *> visited_nodes;
  std::queue<af::Node *> node_queue;
  node_queue.emplace(node.get());
  visited_nodes.emplace(node.get());
  while (!node_queue.empty()) {
    const auto top_node = node_queue.front();
    node_queue.pop();
    for (auto &out_node : top_node->GetOutNodes()) {
      auto asc_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
      if ((asc_node == nullptr) || optimize::ScheduleUtils::IsBuffer(asc_node)) {
        continue;
      }
      if (optimize::ScheduleUtils::IsReduce(asc_node)) {
        return true;
      }
      if (visited_nodes.insert(asc_node.get()).second) {
        node_queue.push(asc_node.get());
      }
    }
  }
  return false;
}

bool IsLoadNeedAlignForReduce(const af::AscNodePtr &node) {
  bool is_tail_axis_transpose = IsTailAxisTranspose(node->outputs[0].attr);
  if (!is_tail_axis_transpose) {
    return false;
  }

  return IsHasReduceNode(node);
}

/**
 * 功能：判断尾轴是否做了transpose
 * 判断逻辑：tensor的最后一根有效轴和向量化的最后一根有效轴是否不同
 */
bool IsTailAxisTransposeV2(const af::AscNodePtr &node_load) {
  auto attr = node_load->outputs[0].attr;
  GE_ASSERT_TRUE(attr.vectorized_axis.size() > 0, "Node %s with unexpected vectorized axis.", node_load->GetNamePtr());

  auto last_valid_axis_index = attr.axis.size();
  for (auto id = attr.vectorized_axis.rbegin(); id != attr.vectorized_axis.rend(); ++id) {
    size_t reverse_pos = std::distance(attr.vectorized_axis.rbegin(), id);
    const size_t index = attr.vectorized_axis.size() - reverse_pos - 1;
    if (af::SymbolicUtils::StaticCheckEq(attr.vectorized_strides[index], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    auto iter = std::find(attr.axis.begin(), attr.axis.end(), *id);
    GE_ASSERT_TRUE(iter != attr.axis.end(), "Cannot find vectorized axis [%ld], axis attr may be invalid.", *id);
    last_valid_axis_index = std::distance(attr.axis.begin(), iter);
    break;
  }

  if (last_valid_axis_index >= attr.axis.size()) {
    GELOGD("Node %s with unexpected vectorized strides.", node_load->GetNamePtr());
    return false;
  }

  for (auto id = attr.axis.begin() + last_valid_axis_index + 1; id != attr.axis.end(); ++id) {
    size_t index = std::distance(attr.axis.begin(), id);
    if (af::SymbolicUtils::StaticCheckNe(attr.strides[index], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      return true;
    }
  }
  return false;
}

/**
 * 功能：判断load节点是否需要做对齐
 * 判断逻辑：IsTailAxisTransposeV2为A5判断尾轴转置的逻辑；IsVectorizedAxisContinuousInGM和IsLoadNeedAlignForReduce是继承的A3判断是否需要针对reduce做对齐的逻辑
 */

bool IsLoadNeedAlign(const af::AscNodePtr &node_load) {
  if (IsHasReduceNode(node_load) &&
      (IsTailAxisTransposeV2(node_load) || IsTailAxisTranspose(node_load->outputs[0].attr))) {
    return true;
  }
  return false;
}

af::Status BaseAlignmentStrategy::DefaultAlignmentInferFunc(const af::AscNodePtr &node) {
  if (af::ops::IsOps<af::ascir_op::RemovePad>(node)) {
    tensor_to_align_type_[&node->outputs[0].attr] = {AlignmentType::kFixedNotAligned};
    return af::SUCCESS;
  }

  bool has_aligned_input = false;
  bool has_fix_unaligned = false;
  auto out_type = AlignmentType::kNotAligned;
  for (const auto &input : node->inputs()) {
    auto alignment_iter = tensor_to_align_type_.find(&input->attr);
    if (alignment_iter == tensor_to_align_type_.end()) {
      continue;
    }

    const AlignmentType input_alignment = alignment_iter->second.align_type;
    if (input_alignment == AlignmentType::kAligned || input_alignment == AlignmentType::kDiscontinuous) {
      has_aligned_input = true;
      out_type = std::max(out_type, input_alignment);
    } else if (input_alignment == AlignmentType::kFixedNotAligned) {
      has_fix_unaligned = true;
    }
  }

  if (has_aligned_input) {
    for (const auto &output : node->outputs()) {
      tensor_to_align_type_[&output->attr] = {out_type};
    }
    return BackPropagateAlignment(node, out_type);
  }

  if (has_fix_unaligned) {
    // 反响传递fix
    out_type = AlignmentType::kFixedNotAligned;
    GE_ASSERT_SUCCESS(BackPropagateFixUnAlignType(node));
  }

  for (const auto &output : node->outputs()) {
    tensor_to_align_type_[&output->attr] = {out_type};
  }
  return af::SUCCESS;
}

af::Status BaseAlignmentStrategy::BroadcastAlignmentInferFunc(const af::AscNodePtr &node) {
  return DefaultAlignmentInferFunc(node);
}

af::Status BaseAlignmentStrategy::ConcatAlignmentInferFunc(const af::AscNodePtr &node) {
  return DefaultAlignmentInferFunc(node);
}

af::Status BaseAlignmentStrategy::EleWiseAlignmentInferFunc(const af::AscNodePtr &node) {
  return DefaultAlignmentInferFunc(node);
}

af::Status BaseAlignmentStrategy::LoadAlignmentInferFunc(const af::AscNodePtr &node) {
  return DefaultAlignmentInferFunc(node);
}

af::Status BaseAlignmentStrategy::StoreAlignmentInferFunc(const af::AscNodePtr &node) {
  return DefaultAlignmentInferFunc(node);
}

af::Status BaseAlignmentStrategy::SplitAlignmentInferFunc(const af::AscNodePtr &node) {
  return DefaultAlignmentInferFunc(node);
}

void BaseAlignmentStrategy::InitAlignmentInferFunc() {
  compute_type_to_infer_func_[af::ComputeType::kComputeElewise] = [this](const std::shared_ptr<af::AscNode> &node) {
    return this->EleWiseAlignmentInferFunc(node);
  };
  compute_type_to_infer_func_[af::ComputeType::kComputeBroadcast] = [this](const std::shared_ptr<af::AscNode> &node) {
    return this->BroadcastAlignmentInferFunc(node);
  };
  compute_type_to_infer_func_[af::ComputeType::kComputeConcat] = [this](const std::shared_ptr<af::AscNode> &node) {
    return this->ConcatAlignmentInferFunc(node);
  };
  compute_type_to_infer_func_[af::ComputeType::kComputeLoad] = [this](const std::shared_ptr<af::AscNode> &node) {
    return this->LoadAlignmentInferFunc(node);
  };
  compute_type_to_infer_func_[af::ComputeType::kComputeStore] = [this](const std::shared_ptr<af::AscNode> &node) {
    return this->StoreAlignmentInferFunc(node);
  };
  compute_type_to_infer_func_[af::ComputeType::kComputeReduce] = [this](const std::shared_ptr<af::AscNode> &node) {
    return this->ReduceAlignmentInferFunc(node);
  };
  compute_type_to_infer_func_[af::ComputeType::kComputeSplit] = [this](const std::shared_ptr<af::AscNode> &node) {
    return this->SplitAlignmentInferFunc(node);
  };
}

af::Status BaseAlignmentStrategy::ReduceAlignmentInferFunc(const af::AscNodePtr &node) {
  for (const auto &output : node->outputs()) {
    auto &output_attr = output->attr;
    GE_ASSERT_TRUE(!output_attr.vectorized_axis.empty());
    const auto axis = output_attr.vectorized_axis.back();
    auto axis_tensor_iter = std::find(output_attr.axis.begin(), output_attr.axis.end(), axis);
    GE_ASSERT_TRUE(axis_tensor_iter != output_attr.axis.end(),
                   "Cannot find vectorized axis [%ld] in [%s]'s output tensor.", axis, node->GetNamePtr());
    const int64_t axis_index = std::distance(output_attr.axis.begin(), axis_tensor_iter);
    const auto &stride = output_attr.strides[axis_index];
    if (af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolZero) == af::TriBool::kTrue) {
      tensor_to_align_type_[&output->attr] = {AlignmentType::kNotAligned};
    } else {
      tensor_to_align_type_[&output->attr] = {AlignmentType::kAligned};
    }
  }
  GE_ASSERT_SUCCESS(BackPropagateAlignment(node));
  return af::SUCCESS;
}

af::Status BaseAlignmentStrategy::AddRemovePadForTailAxisDiscontinuousLoad(ascir::ImplGraph &impl_graph) {
  const auto &config = PlatformFactory::GetInstance().GetPlatform()->GetPlatformConfig();
  if (config.is_support_compat_mode) {
    return af::SUCCESS;
  }
  bool inserted = false;
  for (const auto &node : impl_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (!ScheduleUtils::IsLoad(node) || !ScheduleUtils::IsNeedDiscontinuousAligned(node->outputs[0].attr)) {
      continue;
    }
    af::AscNodePtr remove_pad_node = nullptr;
    if (ScheduleUtils::AddRemovePadAfter(impl_graph, node, remove_pad_node) != af::SUCCESS) {
      continue;
    }
    inserted = true;
  }
  if (inserted) {
    GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(impl_graph));
  }

  return af::SUCCESS;
}

af::Status BaseAlignmentStrategy::CheckIsNoNeedPad(const af::AscNodePtr &node, af::AscTensorAttr &out_attr,
                                                   bool &is_no_need_pad) const {
  size_t valid_axis_num = 0UL;
  bool tail_axis_aligned = false;
  for (auto axis_it = out_attr.vectorized_axis.rbegin(); axis_it != out_attr.vectorized_axis.rend(); ++axis_it) {
    auto it = std::find(out_attr.axis.begin(), out_attr.axis.end(), *axis_it);
    GE_ASSERT_TRUE(it != out_attr.axis.end());
    const size_t distance = std::distance(out_attr.axis.begin(), it);
    if (axis_it == out_attr.vectorized_axis.rbegin()) {
      const auto dtype_size = af::GetSizeByDataType(out_attr.dtype);
      GE_ASSERT_TRUE(dtype_size > 0, "Node [%s]'s data type size:[%d] is invalid.", node->GetNamePtr(), dtype_size);
      auto repeat = out_attr.repeats[distance];
      auto aligned_repeat = af::sym::Align(repeat, align_width_ / dtype_size);
      if (af::SymbolicUtils::StaticCheckEq(repeat, aligned_repeat) == af::TriBool::kTrue) {
        tail_axis_aligned = true;
        GELOGD("Tail repeat [%s] is aligned, no need to add pad for node:[%s].",
               af::SymbolicUtils::ToString(repeat).c_str(), node->GetNamePtr());
        break;
      }
    } else if (af::SymbolicUtils::StaticCheckNe(out_attr.strides[distance], af::sym::kSymbolZero) ==
               af::TriBool::kTrue) {
      valid_axis_num++;
    }
  }
  is_no_need_pad = tail_axis_aligned || valid_axis_num == 0UL;
  return af::SUCCESS;
}

af::Status BaseAlignmentStrategy::AddPadForAlignmentConflictNode(ascir::ImplGraph &impl_graph) {
  bool inserted = false;
  for (const auto &node : impl_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }

    for (size_t i = 0UL; i < node->GetAllOutDataAnchorsSize(); ++i) {
      auto &out_attr = node->outputs[i].attr;
      if (!tensor_to_align_type_[&out_attr].conflict_with_output) {
        continue;
      }

      bool is_no_need_pad{false};
      GE_ASSERT_SUCCESS(CheckIsNoNeedPad(node, out_attr, is_no_need_pad));
      if (is_no_need_pad) {
        tensor_to_align_type_[&out_attr] = {AlignmentType::kAligned};
        continue;
      }

      const auto &dtype = node->outputs[0].attr.dtype;
      std::vector<af::DataType> exp_dtypes{dtype};

      auto ret = ScheduleUtils::CallAscirInferDataType<af::ascir_op::Pad>({dtype}, exp_dtypes);
      if (ret != af::SUCCESS) {
        GELOGW("Pad is unsupported for dtype [%s] in graph [%s], skip this template.",
               af::TypeUtils::DataTypeToSerialString(dtype).c_str(), impl_graph.GetName().c_str());
        return af::UNSUPPORTED;
      }
      inserted = true;
      // Add pad node
      const std::string node_name = node->GetName() + "_" + std::to_string(i) + "_pad";
      af::ascir_op::Pad pad_op(node_name.c_str());
      auto pad_node = impl_graph.AddNode(pad_op);
      GE_ASSERT_NOTNULL(pad_node);
      pad_node->attr = node->attr;
      pad_node->outputs[0].attr = out_attr;
      pad_node->attr.api.compute_type = af::ComputeType::kComputeElewise;
      pad_node->attr.api.type = af::ApiType::kAPITypeCompute;
      pad_node->attr.api.unit = af::ComputeUnit::kUnitVector;
      tensor_to_align_type_[&pad_node->outputs[0].attr] = {AlignmentType::kAligned};

      auto out_anchor = node->GetOutDataAnchor(static_cast<int32_t>(i));
      GE_ASSERT_NOTNULL(out_anchor);
      for (auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
        GE_ASSERT_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(out_anchor, in_anchor, pad_node->GetOutDataAnchor(0)));
      }
      GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(out_anchor, pad_node->GetInDataAnchor(0)));
    }
  }

  if (inserted) {
    GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(impl_graph));
  }

  return af::SUCCESS;
}

af::Status BaseAlignmentStrategy::SetAlignWidth(const ascir::ImplGraph &impl_graph) {
  // 依据数据类型判断对齐到32B还是64B
  align_width_ = 32U;
  GELOGD("[%s]'s align width is [%d].", impl_graph.GetName().c_str(), align_width_);
  return af::SUCCESS;
}

af::Status BaseAlignmentStrategy::AlignVectorizedStrides(ascir::ImplGraph &impl_graph) {
  GE_ASSERT_SUCCESS(SetAlignWidth(impl_graph), "Failed to set align width for [%s].", impl_graph.GetName().c_str());
  if (compute_type_to_infer_func_.empty()) {
    InitAlignmentInferFunc();
  }
  // Add RemovePad
  GE_ASSERT_SUCCESS(AddRemovePadForTailAxisDiscontinuousLoad(impl_graph), "Failed to add removepad for [%s].",
                    impl_graph.GetName().c_str());
  for (const auto &node : impl_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    GE_ASSERT_SUCCESS(InferAlignmentForOneNode(node));
  }
  // Add pad nodes for alignment conflict
  auto ret = AddPadForAlignmentConflictNode(impl_graph);
  if (ret != af::SUCCESS) {
    return ret;
  }

  for (const auto &node : impl_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    GE_ASSERT_SUCCESS(SetVectorizedStridesForOneNode(node));
  }
  return af::SUCCESS;
}

af::Status BaseAlignmentStrategy::InferAlignmentForOneNode(const af::AscNodePtr &node) {
  if (ScheduleUtils::IsBuffer(node)) {
    return af::SUCCESS;
  }
  GE_ASSERT_TRUE(!node->inputs().empty(), "The inputs of %s(%s) is empty.", node->GetTypePtr(), node->GetNamePtr());
  GE_ASSERT_TRUE(!node->outputs().empty(), "The output of %s(%s) is empty.", node->GetTypePtr(), node->GetNamePtr());
  af::ComputeType compute_type = node->attr.api.compute_type;
  auto it = compute_type_to_infer_func_.find(compute_type);
  if (it != compute_type_to_infer_func_.end()) {
    GE_ASSERT_SUCCESS(it->second(node), "Failed to infer alignment for node: [%s], type: %d", node->GetNamePtr(),
                      static_cast<int32_t>(compute_type));
  } else {
    // 默认情况使用尾轴对齐
    GE_ASSERT_SUCCESS(DefaultAlignmentInferFunc(node),
                      "Failed to infer alignment for node: [%s] by default align func, type: %d", node->GetNamePtr(),
                      static_cast<int32_t>(compute_type));
  }

  return af::SUCCESS;
}

void BaseAlignmentStrategy::SetAlignInfoForNodeInputs(AlignmentType aligned_type, af::AscNode *node,
                                                      std::set<af::Node *> &visited_nodes,
                                                      std::queue<af::Node *> &node_queue) {
  for (const auto &input : node->inputs()) {
    auto asc_node = std::dynamic_pointer_cast<af::AscNode>(input->anchor.GetOwnerNode());
    if ((asc_node == nullptr) || ScheduleUtils::IsBuffer(asc_node)) {
      continue;
    }

    auto &align_info = tensor_to_align_type_[&input->attr];
    if (align_info.align_type == aligned_type) {
      continue;
    }

    if (align_info.align_type == AlignmentType::kFixedNotAligned) {
      align_info.conflict_with_output = true;
    } else if (visited_nodes.insert(asc_node.get()).second) {
      node_queue.push(asc_node.get());
    }
  }
}

bool BaseAlignmentStrategy::SetAlignInfoForNodeOutputs(AlignmentType aligned_type, af::AscNode *node,
                                                       std::set<af::Node *> &visited_nodes,
                                                       std::queue<af::Node *> &node_queue) {
  bool alignment_changed = false;
  for (const auto &output : node->outputs()) {
    auto &align_info = tensor_to_align_type_[&output->attr];
    if (align_info.align_type == aligned_type) {
      continue;
    }

    if (align_info.align_type == AlignmentType::kFixedNotAligned) {
      align_info.conflict_with_output = true;
      continue;
    }

    GELOGD("Node [%s]'s align type need to be changed.", node->GetNamePtr());
    align_info.align_type = aligned_type;
    alignment_changed = true;

    // 处理输出连接的节点
    for (const auto &peer_in : output->anchor.GetPeerInDataAnchorsPtr()) {
      if (peer_in == nullptr) {
        continue;
      }
      auto asc_node = std::dynamic_pointer_cast<af::AscNode>(peer_in->GetOwnerNode());
      GE_ASSERT_NOTNULL(asc_node);
      if (ScheduleUtils::IsBuffer(asc_node)) {
        continue;
      }
      if (ScheduleUtils::IsReduce(asc_node)) {
        // Reduce changes the tensor layout, so input alignment must not propagate to its output.
        GELOGI("Stop alignment propagation from node [%s] to Reduce node [%s].", node->GetNamePtr(),
               asc_node->GetNamePtr());
        continue;
      }
      if (visited_nodes.insert(asc_node.get()).second) {
        node_queue.push(asc_node.get());
      }
    }
  }

  return alignment_changed;
}

af::Status BaseAlignmentStrategy::BackPropagateAlignment(const af::AscNodePtr &node, AlignmentType aligned_type) {
  std::set<af::Node *> visited_nodes;
  std::queue<af::Node *> node_queue;
  visited_nodes.emplace(node.get());
  SetAlignInfoForNodeInputs(aligned_type, node.get(), visited_nodes, node_queue);
  while (!node_queue.empty()) {
    const auto curr_node = dynamic_cast<af::AscNode *>(node_queue.front());
    node_queue.pop();
    GE_ASSERT_NOTNULL(curr_node);

    bool alignment_changed = SetAlignInfoForNodeOutputs(aligned_type, curr_node, visited_nodes, node_queue);
    if (alignment_changed) {
      SetAlignInfoForNodeInputs(aligned_type, curr_node, visited_nodes, node_queue);
    }
  }
  return af::SUCCESS;
}

af::Status BaseAlignmentStrategy::BackPropagateFixUnAlignType(const af::AscNodePtr &node) {
  std::queue<af::Node *> node_queue;
  node_queue.emplace(node.get());
  while (!node_queue.empty()) {
    const auto curr_node = dynamic_cast<af::AscNode *>(node_queue.front());
    node_queue.pop();
    GE_ASSERT_NOTNULL(curr_node);

    for (const auto &input : node->inputs()) {
      auto asc_node = std::dynamic_pointer_cast<af::AscNode>(input->anchor.GetOwnerNode());
      if ((asc_node == nullptr) || ScheduleUtils::IsBuffer(asc_node)) {
        continue;
      }

      auto &align_info = tensor_to_align_type_[&input->attr];
      if (align_info.align_type == AlignmentType::kFixedNotAligned) {
        continue;
      }
      GE_ASSERT_TRUE(align_info.align_type == AlignmentType::kNotAligned);
      align_info.align_type = AlignmentType::kFixedNotAligned;
      node_queue.push(asc_node.get());
    }
  }

  return af::SUCCESS;
}

af::Status BaseAlignmentStrategy::SetVectorizedStridesForOneNode(const af::AscNodePtr &node) {
  if (ScheduleUtils::IsStore(node)) {
    node->outputs[0].attr.vectorized_strides = node->inputs[0].attr.vectorized_strides;
    return af::SUCCESS;
  }
  for (const auto &output : node->outputs()) {
    auto &output_attr = output->attr;
    auto type_iter = tensor_to_align_type_.find(&output_attr);
    GE_ASSERT_TRUE(type_iter != tensor_to_align_type_.end(), "Node [%s] has not visited tensor.", node->GetNamePtr());
    GE_ASSERT_SUCCESS(SetVectorizedStridesForTensor(node, output_attr, type_iter->second.align_type));
  }

  return af::SUCCESS;
}

af::Status BaseAlignmentStrategy::SetVectorizedStridesForTensor(const af::NodePtr &node, af::AscTensorAttr &output_attr,
                                                                const AlignmentType align_type) {
  const auto &output_vec_axis = output_attr.vectorized_axis;
  GE_ASSERT_TRUE(!output_vec_axis.empty(), "Node [%s] output tensor has empty vectorized_axis.", node->GetNamePtr());
  GE_ASSERT_TRUE(output_attr.axis.size() == output_attr.repeats.size(),
                 "Node [%s] output tensor axis and repeats size mismatch.", node->GetNamePtr());
  GE_ASSERT_TRUE(output_attr.axis.size() == output_attr.strides.size(),
                 "Node [%s] output tensor axis and strides size mismatch.", node->GetNamePtr());

  const auto dtype_size = af::GetSizeByDataType(output_attr.dtype);
  GE_ASSERT_TRUE(dtype_size > 0, "Node [%s] output tensor dtype is invalid.", node->GetNamePtr());
  const uint32_t align_factor = align_width_ / static_cast<uint32_t>(dtype_size);

  ascir::SizeExpr size_product = af::sym::kSymbolOne;
  std::vector<ascir::SizeExpr> vectorized_strides;
  vectorized_strides.reserve(output_vec_axis.size());
  for (auto axis_it = output_vec_axis.rbegin(); axis_it != output_vec_axis.rend(); ++axis_it) {
    const auto axis = *axis_it;
    auto axis_tensor_iter = std::find(output_attr.axis.begin(), output_attr.axis.end(), axis);
    GE_ASSERT_TRUE(axis_tensor_iter != output_attr.axis.end(),
                   "Cannot find vectorized axis [%ld] in [%s]'s output tensor.", axis, node->GetNamePtr());

    const int64_t axis_index = std::distance(output_attr.axis.begin(), axis_tensor_iter);
    const auto &stride = output_attr.strides[axis_index];
    const auto &repeat = output_attr.repeats[axis_index];

    ascir::SizeExpr current_stride = af::sym::kSymbolZero;
    const bool is_last_axis = (axis_it == output_vec_axis.rbegin());
    const bool is_zero_stride =
        ascgen_utils::ExpressEq(stride, af::sym::kSymbolZero) || ascgen_utils::ExpressEq(repeat, af::sym::kSymbolOne);
    if (is_last_axis && align_type == AlignmentType::kDiscontinuous) {
      // 尾轴非连续
      auto aligned_stride = af::sym::Align(af::sym::kSymbolOne, align_factor);
      if (!is_zero_stride) {
        current_stride = aligned_stride;
      }
      size_product = aligned_stride * repeat;
    } else {
      if (!is_zero_stride) {
        current_stride = size_product;
      }
      if (is_last_axis && align_type == AlignmentType::kAligned) {
        size_product = size_product * af::sym::Align(repeat, align_factor);
      } else if (!is_zero_stride) {
        size_product = size_product * repeat;
      }
    }
    vectorized_strides.push_back(current_stride);
  }

  std::reverse(vectorized_strides.begin(), vectorized_strides.end());
  output_attr.vectorized_strides = vectorized_strides;
  return af::SUCCESS;
}

}  // namespace optimize
