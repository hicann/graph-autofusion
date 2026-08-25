/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */
#ifndef __AUTOFUSE_REDUCE_REUSE_UTILS_H__
#define __AUTOFUSE_REDUCE_REUSE_UTILS_H__

#include <string>
#include <unordered_set>
#include <vector>

#include "ascir.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace af {
namespace ascir {

const std::string kNodeBroadcast = "Broadcast";

inline AscNodePtr GetReduceInputNode(const AscNode &node) {
  AscNodePtr input_node = nullptr;
  auto node_in_anchor = node.GetInDataAnchor(0);
  auto peer_out_anchor = node_in_anchor == nullptr ? nullptr : node_in_anchor->GetPeerOutAnchor();
  input_node =
      peer_out_anchor == nullptr ? nullptr : std::dynamic_pointer_cast<AscNode>(peer_out_anchor->GetOwnerNode());
  return input_node;
}

inline AscNodePtr GetReduceInputNode(const AscNodePtr &node) {
  return node == nullptr ? nullptr : GetReduceInputNode(*node);
}

inline bool HasSingleOutNode(const AscNodePtr &node) {
  return node == nullptr ? false : node->GetOutAllNodes().size() == 1UL;
}

inline bool IsBroadcastReduceAxis(const AscNode &broadcast, const AscNode &reduce) {
  AscNodeInputs broadcast_inputs = broadcast.inputs;
  AscNodeOutputs broadcast_outputs = broadcast.outputs;
  AscNodeInputs reduce_inputs = reduce.inputs;
  AscNodeOutputs reduce_outputs = reduce.outputs;
  const auto &broadcast_input = broadcast_inputs[0].attr;
  const auto &broadcast_output = broadcast_outputs[0].attr;
  const auto &reduce_input = reduce_inputs[0].attr;
  const auto &reduce_output = reduce_outputs[0].attr;
  for (size_t broadcast_pos = 0UL; broadcast_pos < broadcast_output.vectorized_axis.size(); ++broadcast_pos) {
    if (broadcast_pos >= broadcast_input.vectorized_strides.size() ||
        broadcast_pos >= broadcast_output.vectorized_strides.size()) {
      continue;
    }
    const bool is_broadcast_axis = SymbolicUtils::StaticCheckEq(broadcast_input.vectorized_strides[broadcast_pos],
                                                                sym::kSymbolZero) == TriBool::kTrue &&
                                   SymbolicUtils::StaticCheckEq(broadcast_output.vectorized_strides[broadcast_pos],
                                                                sym::kSymbolZero) != TriBool::kTrue;
    if (!is_broadcast_axis) {
      continue;
    }
    for (size_t reduce_pos = 0UL; reduce_pos < reduce_input.vectorized_axis.size(); ++reduce_pos) {
      if (reduce_pos >= reduce_input.vectorized_strides.size() ||
          reduce_pos >= reduce_output.vectorized_strides.size() ||
          broadcast_output.vectorized_axis[broadcast_pos] != reduce_input.vectorized_axis[reduce_pos]) {
        continue;
      }
      const bool is_reduce_axis = SymbolicUtils::StaticCheckEq(reduce_input.vectorized_strides[reduce_pos],
                                                               sym::kSymbolZero) != TriBool::kTrue &&
                                  SymbolicUtils::StaticCheckEq(reduce_output.vectorized_strides[reduce_pos],
                                                               sym::kSymbolZero) == TriBool::kTrue;
      if (is_reduce_axis) {
        return true;
      }
    }
  }
  return false;
}

// 用于检查 Reduce 上游是否存在在 Reduce 轴上的 Broadcast, 避免复用输入source被改写导致精度失败。
inline bool HasUpstreamBroadcastOnReduceAxis(const AscNode &reduce) {
  std::vector<AscNodePtr> pending;
  std::unordered_set<const AscNode *> visited;
  for (const auto &in_node : reduce.GetInDataNodes()) {
    auto asc_in_node = std::dynamic_pointer_cast<AscNode>(in_node);
    if (asc_in_node != nullptr) {
      pending.emplace_back(asc_in_node);
    }
  }
  while (!pending.empty()) {
    auto current = pending.back();
    pending.pop_back();
    if (current == nullptr || !visited.insert(current.get()).second) {
      continue;
    }
    if (current->GetType() == kNodeBroadcast && IsBroadcastReduceAxis(*current, reduce)) {
      return true;
    }
    for (const auto &in_node : current->GetInDataNodes()) {
      auto asc_in_node = std::dynamic_pointer_cast<AscNode>(in_node);
      if (asc_in_node != nullptr) {
        pending.emplace_back(asc_in_node);
      }
    }
  }
  return false;
}

}  // namespace ascir
}  // namespace af

#endif  // __AUTOFUSE_REDUCE_REUSE_UTILS_H__
