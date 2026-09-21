/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "brc_inline_template_v2.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <string>
#include <queue>
#include "platform/common/base_alignment_strategy.h"

namespace optimize {

bool BrcInlineTemplateV2::IsNodeAligned(const af::NodePtr &node) const {
  return aligned_nodes_.find(node) != aligned_nodes_.end();
}

void BrcInlineTemplateV2::MarkNodeAligned(const af::NodePtr &node) {
  aligned_nodes_.insert(node);
}

GenerationMode BrcInlineTemplateV2::GetGenerationMode() {
  return GenerationMode::kAppendCase;
}

af::Status BrcInlineTemplateV2::AlignTensor(const af::NodePtr &node, const af::AscTensor *tensor) {
  return BaseAlignmentStrategy::SetVectorizedStridesForTensor(node, tensor->attr, AlignmentType::kAligned);
}

bool BrcInlineTemplateV2::IsNodeSupportBrcInline(const af::NodePtr &node) {
  for (const auto &brc_out_node : node->GetOutDataNodes()) {
    const auto &out_node = std::dynamic_pointer_cast<af::AscNode>(brc_out_node);
    GE_ASSERT_NOTNULL(out_node);
    if (!ascgen_utils::IsNodeSupportsBrcInline(out_node)) {
      GELOGD("[%s][%s] is not in supported list v2.", brc_out_node->GetTypePtr(), brc_out_node->GetNamePtr());
      return false;
    }
    // RemovePad不支持的数据类型，broadcast inline也不支持
    const auto &dtype = std::dynamic_pointer_cast<af::AscNode>(node)->outputs[0].attr.dtype;
    if (!ScheduleUtils::IsNodeSupportDataType<af::ascir_op::RemovePad>(dtype)) {
      GELOGD("Broadcast inline does not support dtype=%s, node=%s",
             ge::TypeUtils::DataTypeToSerialString(dtype).c_str(), node->GetNamePtr());
      return false;
    }
    std::unique_ptr<af::AscTensor> input0;
    std::unique_ptr<af::AscTensor> input1;
    GE_WARN_ASSERT(ScheduleUtils::GetNonBrcInputTensor(out_node, 0UL, input0) == af::SUCCESS);
    GE_WARN_ASSERT(ScheduleUtils::GetNonBrcInputTensor(out_node, 1UL, input1) == af::SUCCESS);
    ascgen_utils::MergeBrcAxisParams in0(input0->attr.repeats, input0->attr.strides);
    ascgen_utils::MergeBrcAxisParams in1(input1->attr.repeats, input1->attr.strides);
    ascgen_utils::MergeBrcAxisRepeats(in0, in1);
    if (in0.merge_repeats.size() > 4UL) {
      GELOGD("V2 broadcast inline [%s] does not support merged axes count > 4", node->GetNamePtr());
      return false;
    }
    // 暂时不支持尾轴广播inline，因为涉及VF内节点根据loop_axis重新做topo排序，逻辑非常复杂
    if (in0.merge_repeats.back() != in1.merge_repeats.back()) {
      GELOGD("V2 broadcast inline [%s] does not support broadcast axis is last.", node->GetNamePtr());
      return false;
    }
  }
  return true;
}

/**
 * 对Load->...->Broadcast 全部节点进行尾轴32Byte对齐
 * （1）若是静态Shape且尾轴已经32Byte对齐，则不需要处理，直接执行第2步；
 * （2）否则，若Broadcast的输入是RemovePad，则说明Load->...->RemovePad之前，已经对齐了，只需要把RemovePad节点删掉即可；
 * （3）需要做对齐，则调用 BaseAlignmentStrategy::SetVectorizedStridesForTensor 对齐
 */
af::Status BrcInlineTemplateV2::AlignAssociateNodes(const af::AscGraph &graph, const af::AscNodePtr &brc_node) {
  // 把所有 inline 节点加入队列开始，自下而上递归对齐
  std::queue<af::AscNodePtr> need_aligned_nodes_queue;
  need_aligned_nodes_queue.push(brc_node);

  const auto &add_nodes_to_queue = [&need_aligned_nodes_queue, this](const af::Node::Vistor<af::NodePtr> &nodes) {
    for (const auto &node : nodes) {
      if (node->GetInDataNodesSize() == 0UL || node->GetOutDataNodesSize() == 0U || IsNodeAligned(node)) {
        continue;
      }
      need_aligned_nodes_queue.push(std::dynamic_pointer_cast<af::AscNode>(node));
    }
    return af::SUCCESS;
  };

  while (!need_aligned_nodes_queue.empty()) {
    const auto &cur_node = need_aligned_nodes_queue.front();
    GE_CHECK_NOTNULL(cur_node);
    need_aligned_nodes_queue.pop();
    // 1. 该节点已经处理过，则意味着其输入节点和输出节点都已经对齐，直接跳过；
    if (IsNodeAligned(cur_node)) {
      continue;
    }
    MarkNodeAligned(cur_node);

    // 2. 若是静态 Shape 且已经 32Byte 对齐，则节点本身和其输入不需要对齐，直接跳过；
    if (ScheduleUtils::IsTailAxisAlignedBy(cur_node)) {
      GELOGD("Graph[%s] %s[%s] is aligned %uB, no need to align.", graph.GetName().c_str(), cur_node->GetTypePtr(),
             cur_node->GetNamePtr(), kAlignWidth);
      continue;
    }

    // 3. 若是RemovePad，则说明Load->...->RemovePad之前已经对齐了，只要把RemovePad节点删掉，继续对齐输出节点即可；
    GE_ASSERT_TRUE(cur_node->GetInDataNodesSize() > 0UL);
    if (ScheduleUtils::IsRemovePad(cur_node)) {
      GELOGD("Graph[%s] %s[%s] is RemovePad, no need to align after delete RemovePad.", graph.GetName().c_str(),
             cur_node->GetTypePtr(), cur_node->GetNamePtr());
      const auto in_data_anchor = cur_node->GetInDataAnchor(0);
      GE_CHECK_NOTNULL(in_data_anchor);
      GE_ASSERT_SUCCESS(add_nodes_to_queue(cur_node->GetOutDataNodes()));
      GE_ASSERT_SUCCESS(ScheduleUtils::RemoveNode(graph, cur_node, in_data_anchor->GetPeerOutAnchor()));
      continue;
    }

    // 4. 其余场景，都需要对齐
    GELOGD("Graph[%s] %s[%s] is not aligned %uB, need to align.", graph.GetName().c_str(), cur_node->GetTypePtr(),
           cur_node->GetNamePtr(), kAlignWidth);
    for (const auto &out_tensor : cur_node->outputs()) {
      GE_ASSERT_SUCCESS(AlignTensor(cur_node, out_tensor));
    }

    GE_ASSERT_SUCCESS(add_nodes_to_queue(cur_node->GetInDataNodes()));
    GE_ASSERT_SUCCESS(add_nodes_to_queue(cur_node->GetOutDataNodes()));
  }
  return af::SUCCESS;
}

/**
 * 在optimized graph基础上，查找满足brc inline的节点并优化，具体逻辑如下：
 * 1. 遍历所有节点，如果遇到brc节点，则逐个检查brc与其输出节点是否满足brc inline
 * 2. 若满足brc inline，则将本brc节点删掉，计数+1。继续查找其他brc节点
 * 3. 最后，若计数>0，返回成功；否则返回失败。
 */
af::Status BrcInlineTemplateV2::Generate([[maybe_unused]] const af::AscGraph &origin_graph,
                                         [[maybe_unused]] const af::AscGraph &based_case, af::AscGraph &new_case) {
  int32_t brc_inlined_count = 0;
  for (const auto &node : new_case.GetAllNodes()) {
    GE_WARN_ASSERT(!ScheduleUtils::IsReduce(node), "Brc inline does not support Reduce[%s] now.", node->GetNamePtr());
    if (!af::ops::IsOps<af::ascir_op::Broadcast>(node) || ScheduleUtils::IsScalarBroadcastNode(node)) {
      continue;
    }
    GE_ASSERT_TRUE(node->GetOutDataNodesSize() > 0U);
    if (!IsNodeSupportBrcInline(node)) {
      GELOGD("Graph[%s] Broadcast[%s] does not support brc inline", new_case.GetName().c_str(), node->GetNamePtr());
      continue;
    }
    // 若支持inline，要做的是：
    // 1. Load -> ... -> Broadcast, 尾轴32Byte对齐
    GE_ASSERT_SUCCESS(AlignAssociateNodes(new_case, node));
    // 2. 删除Broadcast节点
    const auto in_data_anchor = node->GetInDataAnchor(0);
    GE_CHECK_NOTNULL(in_data_anchor);
    GE_ASSERT_SUCCESS(ScheduleUtils::RemoveNode(new_case, node, in_data_anchor->GetPeerOutAnchor()));
    brc_inlined_count++;
  }
  return brc_inlined_count == 0 ? af::FAILED : af::SUCCESS;
}

bool BrcInlineTemplateV2::NeedDropBasedCase([[maybe_unused]] const af::AscGraph &origin_graph,
                                            [[maybe_unused]] const af::AscGraph &based_case,
                                            [[maybe_unused]] const af::AscGraph &new_case) {
  return false;
}

}  // namespace optimize
