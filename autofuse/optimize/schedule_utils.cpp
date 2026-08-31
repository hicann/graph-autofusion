/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "schedule_utils.h"
#include <unordered_map>
#include <stack>
#include <queue>
#include <optional>
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/compute_graph.h"
#include "util/mem_utils.h"
#include "graph_utils.h"
#include "node_utils.h"
#include "ascir_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "ascir_ops_utils.h"
#include "common_utils.h"

namespace optimize {
namespace {
bool IsMulConsumerStruct(const af::NodePtr &node) {
  std::unordered_set<af::NodePtr> visited;
  std::stack<af::NodePtr> stack;
  stack.push(node);

  while (!stack.empty()) {
    auto current = stack.top();
    stack.pop();
    for (const auto &current_parent : current->GetInDataNodes()) {
      if (visited.find(current_parent) == visited.end()) {
        visited.insert(current_parent);
        stack.push(current_parent);
        if (current_parent->GetOutDataNodesSize() > 1UL) {
          return true;
        }
      }
    }
  }
  return false;
}

Status FindNodeSequence(af::Node *start_node, std::unordered_set<af::Node *> &reduce_sequences) {
  GE_ASSERT_NOTNULL(start_node);
  if (reduce_sequences.count(start_node) > 0UL) {
    return af::SUCCESS;
  }
  std::queue<af::Node *> node_queue;
  node_queue.emplace(start_node);
  reduce_sequences.emplace(start_node);
  while (!node_queue.empty()) {
    auto node = node_queue.front();
    node_queue.pop();
    for (auto &out_node : node->GetOutDataNodes()) {
      GE_ASSERT_NOTNULL(out_node);
      if (reduce_sequences.count(out_node.get()) == 0UL) {
        reduce_sequences.emplace(out_node.get());
        node_queue.emplace(out_node.get());
      }
    }
  }

  return af::SUCCESS;
}

bool IsNeedFixTopo(const af::AscGraph &graph, bool use_rdfs_v2) {
  if (use_rdfs_v2) {
    std::unordered_set<int64_t> unique_loop_axes;
    for (const auto &node : graph.GetAllNodes()) {
      if (node->attr.sched.loop_axis != af::kIdNone) {
        unique_loop_axes.insert(node->attr.sched.loop_axis);
      }
    }
    if (unique_loop_axes.size() <= 1U) {
      GELOGD("Skip fix topo: only %zu unique loop_axis in graph[%s].", unique_loop_axes.size(),
             graph.GetName().c_str());
      return false;
    }
  }
  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsReduce(node) && IsMulConsumerStruct(node)) {
      GELOGD("Need fix topo: reduce node[%s] with multi-consumer found in graph[%s].", node->GetNamePtr(),
             graph.GetName().c_str());
      return true;
    }
  }
  GELOGD("Skip fix topo: no reduce multi-consumer found in graph[%s].", graph.GetName().c_str());
  return false;
}
}  // namespace

std::vector<af::AxisId> ScheduleUtils::CalcReduceAxes(const std::vector<af::Expression> &src_strides,
                                                      const std::vector<af::Expression> &dst_strides,
                                                      const std::vector<ascir::AxisId> &axes) {
  GE_ASSERT_TRUE(src_strides.size() == dst_strides.size(),
                 "The output dim cnt [%zu] of reduce mismatch with input dim cnt [%zu].", dst_strides.size(),
                 src_strides.size());
  GE_ASSERT_TRUE(src_strides.size() == axes.size(),
                 "The input dim cnt [%zu] of reduce mismatch with axis dim cnt [%zu].", src_strides.size(),
                 axes.size());
  std::vector<af::AxisId> reduce_axes;
  for (size_t i = 0UL; i < src_strides.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(src_strides[i], dst_strides[i]) != af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(dst_strides[i], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      reduce_axes.emplace_back(axes[i]);
    }
  }
  return reduce_axes;
}

bool ScheduleUtils::IsNextNodeRemovePad(const ascir::NodeView &node) {
  // 如果当前节点是单输出多引用，则后继节点中只会有1个RemovePad，不会每个引用都单独去Pad。
  const auto &out_nodes = node->GetOutDataNodes();
  return out_nodes.size() == 1UL && IsRemovePad(out_nodes.at(0));
}

bool ScheduleUtils::IsContinuesBroadcast(const std::vector<af::Expression> &in_repeats,
                                         const std::vector<af::Expression> &out_repeats) {
  // 输入和输出size不同，认为不是连续的广播轴
  if (in_repeats.size() != out_repeats.size()) {
    return false;
  }

  std::optional<size_t> last_one_index;
  for (size_t i = 0UL; i < in_repeats.size(); ++i) {
    // 连续广播轴是只有一处repeat不同，其余全部相同，比如：(1,1,1,3) -> (1,4,2,3)，
    // 则是将中间的两个轴(1,1)->(4,2)，则是连续的。
    if (af::SymbolicUtils::StaticCheckEq(in_repeats[i], out_repeats[i]) != af::TriBool::kTrue) {
      if (last_one_index.has_value() && i != *last_one_index + 1) {
        return false;
      }
      last_one_index = i;
    }
  }
  return true;
}

bool ScheduleUtils::IsContinuesStrides(const std::vector<af::Expression> &repeats,
                                       const std::vector<af::Expression> &strides) {
  GE_ASSERT_EQ(repeats.size(), strides.size());
  af::Expression size_product = af::sym::kSymbolOne;
  bool is_first_non_zero = true;
  for (int64_t i = static_cast<int64_t>(strides.size()) - 1; i >= 0; i--) {
    // stride为0，则直接跳过
    if (af::SymbolicUtils::StaticCheckEq(strides[i], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    // inductor场景，右侧第一个不是0的stride，如果也不是1，则一定不连续。因为连续场景时，右侧第1个非0的stride一定是1开始。
    if (is_first_non_zero && af::SymbolicUtils::StaticCheckEq(strides[i], af::sym::kSymbolOne) != af::TriBool::kTrue) {
      return false;
    }
    is_first_non_zero = false;
    // 非0时，应该等于累积
    if (af::SymbolicUtils::StaticCheckEq(strides[i], size_product) != af::TriBool::kTrue) {
      return false;
    }
    size_product = size_product * repeats[i];
  }
  return true;
}

bool ScheduleUtils::IsContinuesVecStrides(const ascir::NodeView &node) {
  std::vector<af::Expression> vec_repeats;
  GE_WARN_ASSERT(GetNodeOutVectorRepeats(node, vec_repeats) == af::SUCCESS);
  return IsContinuesStrides(vec_repeats, node->outputs[0].attr.vectorized_strides);
}

bool ScheduleUtils::IsVectorizedAxisContinuousInGM(const af::AscTensorAttr &output_tensor) {
  auto &axis = output_tensor.axis;
  auto &repeats = output_tensor.repeats;
  auto &strides = output_tensor.strides;
  GE_ASSERT_TRUE(axis.size() == repeats.size(), "axis size:[%zu] mismatch with repeat size:[%zu].", axis.size(),
                 repeats.size());
  GE_ASSERT_TRUE(axis.size() == strides.size(), "axis size:[%zu] mismatch with repeat size:[%zu].", axis.size(),
                 strides.size());
  std::map<int64_t, af::Expression> id_2_repeat_map;
  std::map<int64_t, af::Expression> id_2_stride_map;
  for (size_t i = 0UL; i < axis.size(); ++i) {
    id_2_repeat_map[axis[i]] = repeats[i];
    id_2_stride_map[axis[i]] = strides[i];
  }

  std::vector<af::Expression> vectorized_axis_repeats;
  std::vector<af::Expression> vectorized_axis_strides;
  for (const auto &axis_id : output_tensor.vectorized_axis) {
    GE_ASSERT_TRUE(id_2_repeat_map.find(axis_id) != id_2_repeat_map.end(), "Not found axis=%ld", axis_id);
    vectorized_axis_repeats.push_back(id_2_repeat_map[axis_id]);
    vectorized_axis_strides.push_back(id_2_stride_map[axis_id]);
  }
  return IsContinuesStrides(vectorized_axis_repeats, vectorized_axis_strides);
}

bool ScheduleUtils::IsLastAxisSliceLoad(const af::AscNodePtr &node) {
  if (!af::ops::IsOps<af::ascir_op::Load>(node)) {
    return false;
  }

  auto strides = node->outputs[0U].attr.strides;
  for (int64_t i = static_cast<int64_t>(strides.size() - 1); i >= 0; --i) {
    if (af::SymbolicUtils::StaticCheckEq(strides[i], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    return af::SymbolicUtils::StaticCheckNe(strides[i], af::sym::kSymbolOne) == af::TriBool::kTrue;
  }
  return false;
}

/**
 * 当前只支持对Elementwise、Broadcast、等效成Load的节点做不对齐，其中Data、Output节点没有VectorStride
 */
bool ScheduleUtils::NotNeedAlignVectorStride(const af::AscGraph &graph) {
  using func_type = std::function<bool(af::AscNodePtr)>;
  static std::vector<func_type> support_list{IsElewise, IsBroadcast, IsLoad, IsStore, IsConcat};
  bool exist_concat_node = false;
  for (const auto &node : graph.GetAllNodes()) {
    if (IsIOBuffer(node)) {
      continue;
    }
    if (!std::any_of(support_list.begin(), support_list.end(), [&node](const auto &func) { return func(node); })) {
      GELOGD("Graph[%s], %s[%s] does not support unaligned vector stride.", graph.GetName().c_str(), node->GetTypePtr(),
             node->GetNamePtr());
      return false;
    }
    GE_WARN_ASSERT(!node->outputs().empty(), "Node %s[%s] output is empty.", node->GetTypePtr(), node->GetNamePtr());
    GE_WARN_ASSERT(!node->inputs().empty(), "Node %s[%s] input is empty.", node->GetTypePtr(), node->GetNamePtr());
    // 若原图有Concat节点，会在GenerateTask阶段将Concat转成Store节点，因此要根据Store是否是连续的判断是否由Concat节点转换而来。
    // 存在Concat节点时以Concat的判断为准
    if (IsStore(node) && (!exist_concat_node)) {
      if (!IsContinuesStrides(node->outputs[0].attr.repeats, node->outputs[0].attr.strides)) {
        GELOGD("Graph[%s], %s[%s] is not continuous Store, skip it.", graph.GetName().c_str(), node->GetTypePtr(),
               node->GetNamePtr());
        return false;
      }
    } else if (IsConcat(node)) {
      bool output_need_align = false;
      bool need_align = (!ascir::utils::IsConcatAllInputsAligned(*node)) &&
                        (!ascir::utils::UseSmallTailConcatApi(*node, &output_need_align));
      GE_CHK_BOOL_RET_SPECIAL_STATUS((need_align || output_need_align), false, "Node %s[%s] need align vector stride",
                                     node->GetTypePtr(), node->GetNamePtr());
      exist_concat_node = true;
    } else {
      // do nothing
    }
  }
  return true;
}

/**
 * 判断两个级联的broadcast节点，是否是ABAB或BABA这种场景
 */
bool ScheduleUtils::IsIntervalBroadcast(const std::vector<af::Expression> &in_repeats,
                                        const std::vector<af::Expression> &out_repeats) {
  if (in_repeats.size() != out_repeats.size()) {
    return false;
  }
  constexpr int64_t api_support_brc_axes_cnt = 2L;  // 目前api只支持同时广播两根轴
  constexpr int64_t api_support_vec_axes_cnt = 4L;  // 目前api只支持ABAB、BABA、BAB 三种场景，最多支持4根向量化轴
  int64_t brc_cnt = 0L;
  int64_t pre_brc_index = -1L;

  for (size_t i = 0UL; i < in_repeats.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(in_repeats[i], out_repeats[i]) == af::TriBool::kTrue) {
      continue;
    }
    brc_cnt++;
    if (brc_cnt == 1L) {
      pre_brc_index = static_cast<int64_t>(i);
    } else if (brc_cnt == api_support_brc_axes_cnt) {
      if (static_cast<int64_t>(i) - pre_brc_index != api_support_brc_axes_cnt) {
        return false;
      }
    } else {
      return false;
    }
  }

  return brc_cnt == api_support_brc_axes_cnt && in_repeats.size() <= api_support_vec_axes_cnt;
}

/**
 * 判断节点是否是静态Shape，要求其输出repeats不为空，因为不适合判断Scalar、Output等特殊节点
 */
bool ScheduleUtils::IsStaticShape(const ascir::NodeView &node) {
  GE_WARN_ASSERT(node != nullptr);
  GE_WARN_ASSERT(!node->outputs().empty());
  for (const auto &node_out : node->outputs()) {
    GE_WARN_ASSERT(!node_out->attr.repeats.empty());
    for (const auto &repeat : node_out->attr.repeats) {
      GE_WARN_ASSERT(repeat.IsConstExpr());
    }
  }
  return true;
}

bool ScheduleUtils::IsStaticGraph(const af::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (!af::ops::IsOps<af::ascir_op::Load>(node)) {
      continue;
    }
    if (!IsStaticShape(node)) {
      return false;
    }
  }
  GELOGD("Graph[%s] is static", graph.GetName().c_str());
  return true;
}

Status ScheduleUtils::GetNonBrcInputTensor(const ascir::NodeView &node, const size_t index,
                                           std::unique_ptr<af::AscTensor> &tensor) {
  GE_WARN_ASSERT(node != nullptr);
  GE_ASSERT_TRUE(index < node->inputs().size());
  GE_WARN_ASSERT(node->GetInDataNodes().at(index) != nullptr);
  const auto &in_node = std::dynamic_pointer_cast<af::AscNode>(node->GetInDataNodes().at(index));
  GE_WARN_ASSERT(in_node != nullptr);
  const auto &input = af::ops::IsOps<af::ascir_op::Broadcast>(in_node) ? in_node->inputs[0] : node->inputs[index];
  tensor = af::ComGraphMakeUnique<af::AscTensor>(input);
  return af::SUCCESS;
}

bool ScheduleUtils::GetTailAxisDataSize(const af::AscNodePtr &node, uint32_t &size) {
  GE_WARN_ASSERT(node != nullptr);
  GE_WARN_ASSERT(!node->outputs().empty());
  GE_WARN_ASSERT(!node->outputs[0].attr.repeats.empty());
  const auto &tail_axis_size = node->outputs[0].attr.repeats.back();
  if (!tail_axis_size.IsConstExpr()) {
    return false;
  }
  uint32_t last_dim = 0;
  GE_WARN_ASSERT(tail_axis_size.GetConstValue(last_dim));
  const auto dsize = static_cast<uint32_t>(ge::GetSizeByDataType(node->outputs[0].attr.dtype));
  size = last_dim * dsize;
  return true;
}

bool ScheduleUtils::IsTailAxisLessThan(const af::AscNodePtr &node, const uint32_t value) {
  uint32_t size = 0;
  return GetTailAxisDataSize(node, size) && size < value;
}

bool ScheduleUtils::IsTailAxisAlignedBy(const af::AscNodePtr &node, const uint32_t align_bytes) {
  GE_ASSERT_TRUE(align_bytes > 0U, "Align bytes should not be 0.");
  uint32_t size = 0;
  return GetTailAxisDataSize(node, size) && size % align_bytes == 0;
}

Status ScheduleUtils::TopologicalSorting(af::AscGraph &graph, bool use_rdfs_v2) {
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  const auto mode = use_rdfs_v2 ? af::TopoSortingMode::kRDFSV2 : af::TopoSortingMode::kRDFS;
  GE_ASSERT_GRAPH_SUCCESS(compute_graph->TopologicalSorting(mode), "TopologicalSorting failed, graph:[%s].",
                          compute_graph->GetName().c_str());

  if (!IsNeedFixTopo(graph, use_rdfs_v2)) {
    return af::SUCCESS;
  }

  GELOGI("Graph [%s] will be sorted with a specific rule.", graph.GetName().c_str());
  std::unordered_set<af::Node *> reduce_sequences;
  for (const auto &node : graph.GetAllNodes()) {
    if (IsReduce(node)) {
      GE_ASSERT_SUCCESS(FindNodeSequence(node.get(), reduce_sequences));
    }
  }
  const auto func = [&reduce_sequences](const af::NodePtr &node1, const af::NodePtr &node2) -> bool {
    bool is_node1_in_reduce_seq = reduce_sequences.find(node1.get()) != reduce_sequences.end();
    bool is_node2_in_reduce_seq = reduce_sequences.find(node2.get()) != reduce_sequences.end();
    if (is_node1_in_reduce_seq && !is_node2_in_reduce_seq) {
      return false;
    } else if (!is_node1_in_reduce_seq && is_node2_in_reduce_seq) {
      return true;
    } else {
      return node1->GetOpDescBarePtr()->GetId() < node2->GetOpDescBarePtr()->GetId();
    }
  };

  compute_graph->TopologicalSorting(func);

  return af::SUCCESS;
}

Status ScheduleUtils::RemoveUnusedAxes(af::AscGraph &graph) {
  GELOGD("RemoveUnusedAxes start, graph = %s", graph.GetName().c_str());
  const auto graph_attr = af::AscGraphUtils::GetComputeGraph(graph)->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  auto src_axes = graph_attr->axis;  // copy
  std::map<af::AxisId, af::AxisId> prev_id_to_new_id;
  for (const auto &node : graph.GetAllNodes()) {
    for (auto &axis_id : node->attr.sched.axis) {
      if (prev_id_to_new_id.find(axis_id) == prev_id_to_new_id.cend()) {
        prev_id_to_new_id[axis_id] = static_cast<int64_t>(prev_id_to_new_id.size());
      }
      axis_id = prev_id_to_new_id[axis_id];
    }
    for (uint32_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
      for (auto &axis_id : node->outputs[i].attr.axis) {
        if (prev_id_to_new_id.find(axis_id) == prev_id_to_new_id.cend()) {
          prev_id_to_new_id[axis_id] = static_cast<int64_t>(prev_id_to_new_id.size());
        }
        axis_id = prev_id_to_new_id[axis_id];
      }
    }
  }
  std::vector<af::AxisPtr> new_axes(prev_id_to_new_id.size());
  for (const auto &prev_id_and_new_id : prev_id_to_new_id) {
    const auto &src_axis = src_axes[prev_id_and_new_id.first];
    std::shared_ptr<af::Axis> axis = af::MakeShared<af::Axis>();
    GE_CHECK_NOTNULL(axis, "create axis failed");
    axis->id = prev_id_and_new_id.second;
    axis->name = src_axis->name;
    axis->type = src_axis->type;
    axis->size = src_axis->size;
    new_axes[prev_id_and_new_id.second] = std::move(axis);
  }

  GELOGD("before: axes = %s", AxesToString(graph_attr->axis).c_str());
  graph_attr->axis = std::move(new_axes);
  GELOGD("after: axes = %s", AxesToString(graph_attr->axis).c_str());

  GELOGD("RemoveUnusedAxes success, graph = %s", graph.GetName().c_str());
  return af::SUCCESS;
}

static void ReplaceAxisId(const std::unordered_map<int64_t, int64_t> &old_id_to_new_id,
                          std::vector<int64_t> &axis_ids) {
  for (int64_t &axis_id : axis_ids) {
    auto it = old_id_to_new_id.find(axis_id);
    if (it != old_id_to_new_id.cend()) {
      axis_id = it->second;
    }
  }
}

Status ScheduleUtils::GetVectorRepeats(const std::vector<af::Expression> &repeats, const std::vector<int64_t> &axis,
                                       const std::vector<int64_t> &vector_axis,
                                       std::vector<af::Expression> &vector_repeats) {
  GE_WARN_ASSERT(repeats.size() == axis.size(), "Repeats size(%zu) != axis size(%zu)", repeats.size(), axis.size());
  GE_WARN_ASSERT(vector_axis.size() <= axis.size(), "Vector axis size(%zu) >= axis size(%zu)", vector_axis.size(),
                 axis.size());
  if (vector_axis.empty()) {
    return af::SUCCESS;
  }

  std::map<int64_t, af::Expression> id_2_repeat_map;
  for (size_t i = 0UL; i < repeats.size(); ++i) {
    id_2_repeat_map[axis[i]] = repeats[i];
  }
  vector_repeats.clear();
  for (const auto &v_axis : vector_axis) {
    GE_ASSERT_TRUE(id_2_repeat_map.find(v_axis) != id_2_repeat_map.end(), "Not found axis=%ld", v_axis);
    vector_repeats.push_back(id_2_repeat_map.at(v_axis));
  }
  return af::SUCCESS;
}

Status ScheduleUtils::GetNodeInputVectorRepeats(const ascir::NodeView &node,
                                                std::vector<af::Expression> &vector_repeats) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(!node->inputs().empty());
  const auto &attr = node->inputs[0].attr;
  return GetVectorRepeats(attr.repeats, attr.axis, attr.vectorized_axis, vector_repeats);
}

Status ScheduleUtils::GetNodeOutVectorRepeats(const ascir::NodeView &node, std::vector<af::Expression> &vec_repeats) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(!node->outputs().empty());
  const auto &attr = node->outputs[0].attr;
  return GetVectorRepeats(attr.repeats, attr.axis, attr.vectorized_axis, vec_repeats);
}

Status ScheduleUtils::GetConcatDim(const af::AscNodePtr &node, size_t &concat_dim) {
  const std::vector<ascir::SizeExpr> &input_repeats = node->inputs[0].attr.repeats;
  const std::vector<ascir::SizeExpr> &output_repeats = node->outputs[0].attr.repeats;
  GE_ASSERT_TRUE((input_repeats.size() == output_repeats.size()),
                 "The output dim cnt [%zu] of concat mismatch with input dim cnt [%zu].", output_repeats.size(),
                 input_repeats.size());
  for (size_t i = 0UL; i < input_repeats.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(input_repeats[i], output_repeats[i]) != af::TriBool::kTrue) {
      concat_dim = i;
      break;
    }
  }
  for (size_t i = concat_dim + 1UL; i < input_repeats.size(); ++i) {
    GE_ASSERT_TRUE(af::SymbolicUtils::StaticCheckEq(input_repeats[i], output_repeats[i]) == af::TriBool::kTrue,
                   "The [%zu]th sizes of the non-concat_dim do not match.", i);
  }
  return af::SUCCESS;
}

void ScheduleUtils::NormalizeAxisIds(const af::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  std::unordered_map<int64_t, int64_t> origin_id_to_new_id;
  for (int64_t i = 0; i < static_cast<int64_t>(all_axis.size()); ++i) {
    const auto &cur_axis = all_axis[static_cast<size_t>(i)];
    if (cur_axis->id != i) {
      origin_id_to_new_id[cur_axis->id] = i;
      cur_axis->id = i;
    }
  }
  if (origin_id_to_new_id.empty()) {
    return;
  }
  for (const auto &node : graph.GetAllNodes()) {
    ReplaceAxisId(origin_id_to_new_id, node->attr.sched.axis);
    for (auto &output : node->outputs()) {
      ReplaceAxisId(origin_id_to_new_id, output->attr.axis);
    }
  }
}

std::string ScheduleUtils::AxesToString(const std::vector<af::AxisPtr> &axes) {
  std::vector<std::string> axes_strs;
  axes_strs.reserve(axes.size());
  for (const auto &axis : axes) {
    axes_strs.emplace_back(axis == nullptr ? "nullptr" : axis->name);
  }
  return af::ToString(axes_strs);
}

std::vector<af::AscNodePtr> GetParentNodes(const af::AscNodePtr &node) {
  std::vector<af::AscNodePtr> parentNodes;
  const auto &inNodes = node->GetInNodes();
  for (const auto &parentNode : inNodes) {
    af::AscNodePtr ascParentNode = std::dynamic_pointer_cast<af::AscNode>(parentNode);
    if (ascParentNode != nullptr) {
      parentNodes.push_back(ascParentNode);
    }
  }
  return parentNodes;
}

std::vector<af::AscNodePtr> GetChildNodes(const af::AscNodePtr &node) {
  std::vector<af::AscNodePtr> childNodes;
  const auto &outNodes = node->GetOutNodes();
  for (const auto &childNode : outNodes) {
    af::AscNodePtr ascChildNode = std::dynamic_pointer_cast<af::AscNode>(childNode);
    if (ascChildNode != nullptr) {
      childNodes.push_back(ascChildNode);
    }
  }
  return childNodes;
}

static bool DfsReduceNodeBetweenBA(const af::AscNodePtr &current, const af::AscNodePtr &target, bool hasReduce) {
  if (ScheduleUtils::IsReduce(current)) {
    hasReduce = true;
  }

  if (current == target) {
    return hasReduce;
  }

  const auto parents = GetParentNodes(current);
  for (const auto &parent : parents) {
    if (DfsReduceNodeBetweenBA(parent, target, hasReduce)) {
      return true;
    }
  }

  return false;
}

bool HasReduceNodeOnPath(const af::AscNodePtr &b, const af::AscNodePtr &a) {
  return DfsReduceNodeBetweenBA(b, a, false);
}

bool ScheduleUtils::IsLastAxisReduce(const ascir::ImplGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (ScheduleUtils::IsReduce(node)) {
      if (node->GetType() == "Softmax") {
        return true;
      }
      std::vector<ascir::SizeExpr> src_strides;
      ScheduleUtils::GetReduceInputStrides(*node, src_strides);
      const std::vector<ascir::SizeExpr> &dst_strides = node->outputs[0].attr.strides;
      auto last_index = src_strides.size() - 1;
      return (af::SymbolicUtils::StaticCheckEq(src_strides[last_index], dst_strides[last_index]) !=
              af::TriBool::kTrue) &&
             (af::SymbolicUtils::StaticCheckEq(dst_strides[last_index], af::sym::kSymbolZero) == af::TriBool::kTrue);
    }
  }
  return false;
}

bool ScheduleUtils::IsReduceOnTailAxis(const af::AscNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  const auto &input_repeats = node->inputs[0].attr.repeats;
  const auto &output_repeats = node->outputs[0].attr.repeats;
  if (input_repeats.empty() || input_repeats.size() != output_repeats.size()) {
    return false;
  }
  const auto tail_index = input_repeats.size() - 1UL;
  for (size_t i = 0UL; i < tail_index; ++i) {
    if (af::SymbolicUtils::StaticCheckEq(input_repeats[i], output_repeats[i]) != af::TriBool::kTrue) {
      return false;
    }
  }
  return af::SymbolicUtils::StaticCheckEq(input_repeats[tail_index], output_repeats[tail_index]) != af::TriBool::kTrue;
}

bool ScheduleUtils::IsNormStruct(const ascir::ImplGraph &implGraph) {
  for (const auto &node : implGraph.GetAllNodes()) {
    auto parents = GetParentNodes(node);
    if (parents.size() <= 1) {
      continue;
    }
    std::unordered_set<af::AscNodePtr> allAncestors;
    std::vector<std::unordered_set<af::AscNodePtr>> parentAncestors(parents.size());
    std::unordered_map<af::AscNodePtr, int> ancestorDistances;
    for (size_t i = 0; i < parents.size(); ++i) {
      auto &ancestors = parentAncestors[i];
      std::stack<std::pair<af::AscNodePtr, int>> stack;
      stack.push({parents[i], 1});

      while (!stack.empty()) {
        auto [current, distance] = stack.top();
        stack.pop();

        if (ancestors.count(current) != 0) {
          continue;
        }
        ancestors.insert(current);

        if ((ancestorDistances.count(current) == 0) || distance < ancestorDistances[current]) {
          ancestorDistances[current] = distance;
        }

        const auto currentParents = GetParentNodes(current);
        for (const auto &currentParent : currentParents) {
          stack.push({currentParent, distance + 1});
        }
      }
      allAncestors.insert(ancestors.begin(), ancestors.end());
    }

    af::AscNodePtr nearestCommonAncestor = nullptr;
    int32_t minDistance = std::numeric_limits<int>::max();
    for (const auto &potentialAncestor : allAncestors) {
      bool isCommon = true;
      for (const auto &ancestors : parentAncestors) {
        if (ancestors.count(potentialAncestor) == 0) {
          isCommon = false;
          break;
        }
      }

      if (isCommon) {
        int distance = ancestorDistances[potentialAncestor];
        if (distance < minDistance) {
          minDistance = distance;
          nearestCommonAncestor = potentialAncestor;
        }
      }
    }

    if (nearestCommonAncestor != nullptr) {
      if (IsCompute(nearestCommonAncestor) && !IsStore(nearestCommonAncestor)) {
        if (HasReduceNodeOnPath(node, nearestCommonAncestor)) {
          GELOGD("The node %s is norm struct.", node->GetName().c_str());
          return true;
        }
      }
    }
  }

  return false;
}

bool HasBroadcastDescendantNode(const af::AscNodePtr &node) {
  const auto &outNodes = node->GetOutNodes();
  for (const auto &childNode : outNodes) {
    af::AscNodePtr ascChildNode = std::dynamic_pointer_cast<af::AscNode>(childNode);
    std::stack<af::AscNodePtr> stack;
    stack.push(ascChildNode);
    while (!stack.empty()) {
      auto current = stack.top();
      stack.pop();
      if (ScheduleUtils::IsBroadcast(current)) {
        return true;
      }

      const auto currentChilds = GetChildNodes(current);
      for (const auto &currentChild : currentChilds) {
        stack.push(currentChild);
      }
    }
  }
  return false;
}

bool ScheduleUtils::IsReduceArFullLoad(const ascir::ImplGraph &implGraph) {
  for (const auto &node : implGraph.GetAllNodes()) {
    if (!ScheduleUtils::IsReduce(node)) {
      continue;
    }

    if (node->GetType() == "Softmax") {
      GELOGD("Reduce node %s is Softmax, force all load.", node->GetName().c_str());
      return true;
    }

    if (HasBroadcastDescendantNode(node)) {
      GELOGD("There is a broadcast node behind the reduced node %s.", node->GetName().c_str());
      return true;
    }

    auto parents = GetParentNodes(node);
    std::unordered_set<af::AscNodePtr> allAncestors;
    std::vector<std::unordered_set<af::AscNodePtr>> parentAncestors(parents.size());
    for (size_t i = 0; i < parents.size(); ++i) {
      auto &ancestors = parentAncestors[i];
      std::stack<std::pair<af::AscNodePtr, int>> stack;
      stack.push({parents[i], 1});

      while (!stack.empty()) {
        auto [current, distance] = stack.top();
        stack.pop();

        if (ancestors.count(current) != 0) {
          continue;
        }
        ancestors.insert(current);

        const auto currentParents = GetParentNodes(current);
        for (const auto &currentParent : currentParents) {
          stack.push({currentParent, distance + 1});
        }
      }
      allAncestors.insert(ancestors.begin(), ancestors.end());
    }

    for (const auto &potentialAncestor : allAncestors) {
      if (GetChildNodes(potentialAncestor).size() > 1) {
        GELOGD("The reduce node %s is multiref struct.", node->GetName().c_str());
        return true;
      }
    }
  }
  return false;
}

bool ScheduleUtils::HasComputeType(const ascir::ImplGraph &impl_graph, const af::ComputeType compute_type) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (node->attr.api.compute_type == compute_type) {
      return true;
    }
  }
  return false;
}

// 该接口校验了brc单输出，并非针对sclar直连brc的通用接口
bool ScheduleUtils::IsScalarBroadcastNode(const ascir::NodeView &node) {
  GELOGD("%s[%s] output_size=%u, input_size=%u", node->GetTypePtr(), node->GetNamePtr(), node->GetOutDataNodesSize(),
         node->GetInDataNodesSize());
  if (!IsBroadcast(node)) {
    return false;
  }
  // 当前只支持输入输入输出节点数是1的Scalar广播场景
  if (node->GetOutDataNodesSize() != 1UL || node->GetInDataNodesSize() != 1UL) {
    return false;
  }
  return ascgen_utils::IsScalarInput(node->inputs[0].attr.repeats);
}

bool ScheduleUtils::IsScalarBrc(const af::AscNodePtr &node) {
  if (!IsBroadcast(node)) {
    return false;
  }
  const auto &repeats = node->inputs[0].attr.repeats;
  return std::all_of(repeats.begin(), repeats.end(),
                     [](const af::Expression &repeat) { return ascgen_utils::ExpressEq(repeat, af::sym::kSymbolOne); });
}

bool ScheduleUtils::HasSameInput(const af::AscNodePtr &node) {
  std::set<af::NodePtr> inputs;
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    GE_ASSERT_NOTNULL(in_anchor);
    GE_ASSERT_NOTNULL(in_anchor->GetPeerOutAnchor());
    GE_ASSERT_NOTNULL(in_anchor->GetPeerOutAnchor()->GetOwnerNode());
    const af::NodePtr in_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();
    if (inputs.count(in_node) > 0) {
      return true;
    }
    inputs.emplace(in_node);
  }
  return false;
}

Status ScheduleUtils::SwapInputIndex(const ascir::NodeView &node, const int32_t idx1, const int32_t idx2) {
  GELOGD("Swap input %d and %d for node %s[%s].", idx1, idx2, node->GetTypePtr(), node->GetNamePtr());
  GE_ASSERT_TRUE(static_cast<uint32_t>(std::max(idx1, idx2)) < node->GetAllInDataAnchorsSize());
  const auto &first_in_anchor = node->GetInDataAnchor(idx1);
  const auto &second_in_anchor = node->GetInDataAnchor(idx2);
  GE_ASSERT_NOTNULL(first_in_anchor);
  GE_ASSERT_NOTNULL(second_in_anchor);
  const auto &first_out_anchor = first_in_anchor->GetPeerOutAnchor();
  const auto &second_out_anchor = second_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(first_out_anchor);
  GE_ASSERT_NOTNULL(second_out_anchor);

  GE_ASSERT_SUCCESS(af::GraphUtils::RemoveEdge(first_out_anchor, first_in_anchor));
  GE_ASSERT_SUCCESS(af::GraphUtils::RemoveEdge(second_out_anchor, second_in_anchor));
  GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(first_out_anchor, second_in_anchor));
  GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(second_out_anchor, first_in_anchor));
  return af::SUCCESS;
}

Status ScheduleUtils::GetInputForTranspose(af::AscNode &node, std::vector<ascir::AxisId> &input_axis) {
  const auto begin_node = std::dynamic_pointer_cast<af::AscNode>(node.shared_from_this());
  GE_ASSERT_NOTNULL(begin_node);
  auto parent_nodes = GetParentNodes(begin_node);
  GE_ASSERT_TRUE(!parent_nodes.empty(), "The node %s has no parent node.", node.GetNamePtr());
  input_axis = parent_nodes[0]->outputs[0].attr.axis;
  GELOGD("Found transpose input from %s, the axis is %s.", parent_nodes[0]->GetNamePtr(),
         af::ViewMemberToString(input_axis).c_str());
  return af::SUCCESS;
}

bool ScheduleUtils::IsNeedDiscontinuousAligned(const af::AscTensorAttr &attr) {
  for (auto id = attr.vectorized_axis.rbegin(); id != attr.vectorized_axis.rend(); ++id) {
    auto iter = std::find(attr.axis.begin(), attr.axis.end(), *id);
    GE_ASSERT_TRUE(iter != attr.axis.end(), "Cannot find vectorized axis [%ld], axis attr may be invalid.", *id);
    const size_t index = std::distance(attr.axis.begin(), iter);
    // 考虑到通用模板要兼顾reduce的限制, 因此,尾轴为1的非连续load,不会当成DisContinuous处理
    if ((index == attr.repeats.size() - 1UL) &&
        (af::SymbolicUtils::StaticCheckEq(attr.repeats[index], af::sym::kSymbolOne) == af::TriBool::kTrue)) {
      return false;
    }
    if (af::SymbolicUtils::StaticCheckEq(attr.strides[index], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    return af::SymbolicUtils::StaticCheckNe(attr.strides[index], af::sym::kSymbolOne) == af::TriBool::kTrue;
  }
  return false;
}

Status ScheduleUtils::RemoveNode(const ascir::ImplGraph &impl_graph, const af::AscNodePtr &node,
                                 const af::OutDataAnchorPtr &pre_out_anchor) {
  for (auto &out_anchor : node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_anchor);
    for (auto &next_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(next_in_anchor);
      GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(out_anchor, next_in_anchor, pre_out_anchor));
    }
  }
  af::NodeUtils::UnlinkAll(*node);
  GE_CHECK_NOTNULL(af::AscGraphUtils::GetComputeGraph(impl_graph));
  af::GraphUtils::RemoveNodeWithoutRelink(af::AscGraphUtils::GetComputeGraph(impl_graph), node);
  return af::SUCCESS;
}

bool ScheduleUtils::FindContinuesBroadcastNode(const ascir::NodeView &node,
                                               std::vector<af::AscNodePtr> &continues_brc_nodes) {
  auto brc_node = node;
  continues_brc_nodes.push_back(node);
  while (brc_node != nullptr) {
    GE_ASSERT_TRUE(brc_node->GetInDataNodesSize() == 1UL, "Brc[%s] input size != 1", brc_node->GetNamePtr());
    GE_ASSERT_TRUE(brc_node->GetOutDataNodesSize() > 0UL, "Brc[%s] has not output.", brc_node->GetNamePtr());
    GE_ASSERT_NOTNULL(brc_node->GetOutDataNodes().at(0UL));
    af::AscNodePtr next_brc_node = std::dynamic_pointer_cast<af::AscNode>(brc_node->GetOutDataNodes().at(0UL));
    GE_ASSERT_NOTNULL(next_brc_node);
    // 如果下一个节点是brc，则本节点输出引用只能是1个；否则，本节点可以接多个引用。
    if (!af::ops::IsOps<af::ascir_op::Broadcast>(next_brc_node) || brc_node->GetOutDataNodesSize() != 1UL) {
      GELOGD("Next node of Broadcast is %s[%s], stop find.", next_brc_node->GetTypePtr(), next_brc_node->GetNamePtr());
      break;
    }

    continues_brc_nodes.push_back(next_brc_node);
    brc_node = next_brc_node;
  }
  return true;
}

Status ScheduleUtils::AddRemovePadAfter(af::AscGraph &graph, const af::AscNodePtr &node,
                                        af::AscNodePtr &remove_pad_node, const int32_t idx) {
  GE_ASSERT_NOTNULL(node);
  const auto &dtype = node->outputs[idx].attr.dtype;
  GE_WARN_ASSERT(ScheduleUtils::IsNodeSupportDataType<af::ascir_op::RemovePad>(dtype));

  const std::string node_name = node->GetName() + "_remove_pad_" + std::to_string(idx);
  af::ascir_op::RemovePad remove_pad_op(node_name.c_str());
  remove_pad_node = graph.AddNode(remove_pad_op);
  GE_ASSERT_NOTNULL(remove_pad_node);
  remove_pad_node->attr = node->attr;
  remove_pad_node->outputs[0].attr = node->outputs[0].attr;
  remove_pad_node->attr.api.compute_type = af::ComputeType::kComputeElewise;
  remove_pad_node->attr.api.type = af::ApiType::kAPITypeCompute;
  remove_pad_node->attr.api.unit = af::ComputeUnit::kUnitVector;

  const auto out_anchor = node->GetOutDataAnchor(idx);
  GE_ASSERT_NOTNULL(out_anchor);
  for (auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
    GE_ASSERT_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(out_anchor, in_anchor, remove_pad_node->GetOutDataAnchor(0)));
  }
  GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(out_anchor, remove_pad_node->GetInDataAnchor(0)));
  return af::SUCCESS;
}

Status ScheduleUtils::RemoveNodeDst(const ascir::ImplGraph &impl_graph, const af::AscNodePtr &node,
                                    const af::InDataAnchorPtr &next_in_anchor) {
  for (auto &in_anchor : node->GetAllInDataAnchors()) {
    GE_CHECK_NOTNULL(in_anchor);
    auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeDst(peer_out_anchor, in_anchor, next_in_anchor));
  }
  af::NodeUtils::UnlinkAll(*node);
  GE_CHECK_NOTNULL(af::AscGraphUtils::GetComputeGraph(impl_graph));
  af::GraphUtils::RemoveNodeWithoutRelink(af::AscGraphUtils::GetComputeGraph(impl_graph), node);
  return af::SUCCESS;
}

bool ScheduleUtils::IsOutNodeWithMultiInputs(const af::AscNodePtr &node) {
  GE_CHECK_NOTNULL(node);
  for (const auto &out_node : node->GetOutAllNodes()) {
    GE_CHECK_NOTNULL(out_node);
    if (out_node->GetInDataNodes().size() != 1UL) {
      GELOGD("Node %s out node %s has multiple input nodes.", node->GetNamePtr(), out_node->GetNamePtr());
      return true;
    }
  }
  return false;
}

Status ScheduleUtils::ResolveDiffDim(const af::AscNodePtr &node, size_t &diff_dim, bool &is_first_dim) {
  const auto &input_repeats = node->inputs[0].attr.repeats;
  const auto &output_repeats = node->outputs[0].attr.repeats;
  GE_ASSERT_TRUE(input_repeats.size() == output_repeats.size(),
                 "input_repeats.size() = %zu, mismatches output_repeats.size() = %zu", input_repeats.size(),
                 output_repeats.size());
  diff_dim = 0UL;
  size_t non_one_count = 0U;
  for (size_t i = 0U; i < input_repeats.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(input_repeats[i], output_repeats[i]) != af::TriBool::kTrue) {
      diff_dim = i;
      is_first_dim = (non_one_count == 0);
      break;
    }
    if (af::SymbolicUtils::StaticCheckEq(input_repeats[i], af::ops::One) != af::TriBool::kTrue) {
      ++non_one_count;
    }
  }
  is_first_dim = (is_first_dim || (diff_dim == 0UL));  // 单输入时，当成首轴转store处理
  GELOGI("node:%s input_shape = %s, output_shape = %s, is_first_dim = %d, diff_dim = %zu", node->GetName().c_str(),
         af::ToString(input_repeats).c_str(), af::ToString(output_repeats).c_str(), is_first_dim, diff_dim);
  return af::SUCCESS;
}

Status ScheduleUtils::RecalculateStridesFromRepeats(const std::vector<af::Expression> &repeats,
                                                    std::vector<af::Expression> &strides) {
  GE_ASSERT_TRUE(!repeats.empty(), "The repeats is empty.");
  strides.resize(repeats.size());
  af::Expression current_stride = af::sym::kSymbolOne;
  for (size_t i = repeats.size(); i > 0; --i) {
    size_t idx = i - 1;
    if (af::SymbolicUtils::StaticCheckEq(repeats[i - 1], af::sym::kSymbolOne) == af::TriBool::kTrue) {
      strides[idx] = af::sym::kSymbolZero;
    } else {
      strides[idx] = current_stride;
      current_stride = current_stride * repeats[idx];
    }
  }
  return af::SUCCESS;
}

Status ScheduleUtils::ClearAllSizeVar(const af::AscGraph &graph) {
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  const auto graph_attr = af::AscGraphUtils::GetComputeGraph(graph)->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  graph_attr->size_vars.clear();
  return af::SUCCESS;
}

// 判断节点的Micro API是否支持Scalar输入，用于scalar_broadcast优化
// 支持的算子：比较类(Ge/Eq/Ne/Le/Lt/Gt)和二元计算类(Add/Minimum/Maximum)
bool ScheduleUtils::IsMicroApiSupportsScalarInput(const af::AscNodePtr &node) {
  return af::ops::IsOps<af::ascir_op::Ge>(node) || af::ops::IsOps<af::ascir_op::Eq>(node) ||
         af::ops::IsOps<af::ascir_op::Ne>(node) || af::ops::IsOps<af::ascir_op::Le>(node) ||
         af::ops::IsOps<af::ascir_op::Lt>(node) || af::ops::IsOps<af::ascir_op::Gt>(node) ||
         af::ops::IsOps<af::ascir_op::Add>(node) || af::ops::IsOps<af::ascir_op::Minimum>(node) ||
         af::ops::IsOps<af::ascir_op::Maximum>(node);
}

void ScheduleUtils::GenerateStrides(const std::vector<ge::Expression> &repeats, std::vector<ge::Expression> &strides) {
  ge::Expression stride = af::sym::kSymbolOne;
  strides.resize(repeats.size());
  for (auto i = static_cast<int32_t>(repeats.size() - 1); i >= 0; --i) {
    if (ascgen_utils::ExpressEq(repeats[i], af::sym::kSymbolOne)) {
      strides[i] = af::sym::kSymbolZero;
    } else {
      strides[i] = stride;
    }
    stride = (stride * repeats[i]);
  }
}
}  // namespace optimize
