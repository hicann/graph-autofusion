/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_time_pass.h"
#include "base/att_const_values.h"
#include "common/common_utils.h"
#include "ascir_ops.h"
#include "optimize/platform/platform_factory.h"

namespace att {
namespace {
[[maybe_unused]] void AddUsedArgs(const Expr &expr, std::vector<Expr> &used_args) {
  for (const auto &arg : expr.FreeSymbols()) {
    used_args.emplace_back(arg);
  }
}

void CheckSplit(const std::map<std::string, std::set<std::string>> &axis_list, const std::string &node_name,
                const std::string &orig_name, bool &is_split) {
  const auto iter = axis_list.find(node_name);
  if (iter != axis_list.end()) {
    if (iter->second.size() > 0) {
      if (iter->second.find(orig_name) != iter->second.end()) {
        is_split = true;
      }
    }
  }
}

void InsertAxis(const SubAxis *cur_dim, const NodeInfo &node_info,
                std::map<std::string, std::set<std::string>> &axis_list) {
  for (const auto &dim : cur_dim->orig_axis) {
    for (const auto &node_name : node_info.from_data) {
      axis_list[node_name].insert(dim->name);
    }
  }
}

bool AxisStableLess(const SubAxis *lhs, const SubAxis *rhs) {
  if (lhs->id != rhs->id) {
    return lhs->id < rhs->id;
  }
  return lhs->name < rhs->name;
}

bool IsBindMultiCoreInner(const SubAxis *axis) {
  return axis != nullptr && axis->is_bind_multi_core && axis->axis_type == AxisPosition::INNER;
}

bool SharesParent(const SubAxis *lhs, const SubAxis *rhs) {
  for (const auto *parent : lhs->parent_axis) {
    if (parent == nullptr) {
      continue;
    }
    for (const auto *other_parent : rhs->parent_axis) {
      if (other_parent != nullptr &&
          (other_parent == parent || (!parent->name.empty() && parent->name == other_parent->name))) {
        return true;
      }
    }
  }
  return false;
}
}  // namespace

bool ExeTimePassManager::TryBuildCacheGuardInfo(const NodeInfo &node, CacheGuardInfo &info) const {
  info = CacheGuardInfo{};
  if (node.is_cv_ub_fusion || (node.exec_condition != af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis &&
                               node.exec_condition != af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis)) {
    return false;
  }

  std::vector<const SubAxis *> block_inner_axes;
  for (const auto *axis : node.loop_axes) {
    if (IsBindMultiCoreInner(axis)) {
      block_inner_axes.emplace_back(axis);
    }
  }
  std::sort(block_inner_axes.begin(), block_inner_axes.end(), AxisStableLess);
  // NDDMA is issued once per block-inner iteration; trailing vector axes are
  // consumed inside that operation and must not disable its cache guard.
  const bool nddma_with_trailing_vector_axis = node.node_type == "Nddma";
  if (block_inner_axes.size() != 1U || node.loop_axes.empty() ||
      (!nddma_with_trailing_vector_axis && node.loop_axes.back() != block_inner_axes.front())) {
    return false;
  }
  const auto *block_inner = block_inner_axes.front();
  if (!block_inner->repeat.IsValid()) {
    return false;
  }

  info.guard_axis = af::Symbol(block_inner->name.c_str());
  info.loop_extent = block_inner->repeat;
  info.address_invariant = true;
  // Parser axis repeats are constrained to positive tile sizes by the
  // scheduler.  Record that provenance so symbolic modulo/division is only
  // emitted for metadata produced by this adapter, never for arbitrary
  // caller-provided expressions.
  info.positive_range_proven = true;
  if (node.exec_condition == af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis) {
    info.kind = CacheGuardKind::kOriginBroadcast;
    return ValidateCacheGuardInfo(info);
  }

  std::vector<const SubAxis *> outer_parents;
  for (const auto *parent : block_inner->parent_axis) {
    if (parent != nullptr && parent->axis_type == AxisPosition::OUTER && !parent->is_bind_multi_core) {
      outer_parents.emplace_back(parent);
    }
  }
  if (tuning_space_ != nullptr) {
    for (const auto &axis_holder : tuning_space_->sub_axes) {
      const auto *axis = axis_holder.get();
      if (axis == nullptr || axis->axis_type != AxisPosition::OUTER || axis->is_bind_multi_core) {
        continue;
      }
      const bool direct_parent =
          std::find(axis->parent_axis.begin(), axis->parent_axis.end(), block_inner) != axis->parent_axis.end();
      if ((direct_parent || SharesParent(axis, block_inner)) &&
          std::find(outer_parents.begin(), outer_parents.end(), axis) == outer_parents.end()) {
        outer_parents.emplace_back(axis);
      }
    }
  }
  std::sort(outer_parents.begin(), outer_parents.end(), AxisStableLess);
  if (outer_parents.size() != 1U || !outer_parents.front()->repeat.IsValid()) {
    return false;
  }
  info.kind = CacheGuardKind::kFusedBroadcast;
  info.period = outer_parents.front()->repeat;
  info.block_inner_extent = block_inner->repeat;
  return ValidateCacheGuardInfo(info);
}

void ExeTimePassManager::AddBAxis(const std::string &dim_name, const Expr &repeat, const Expr &stride,
                                  const NodeInfo &node_info) {
  TensorPtr output_tensor = node_info.outputs[0];
  for (size_t i = 0; i < output_tensor->dim_info.size(); i++) {
    auto &cur_dim = output_tensor->dim_info[i];
    if ((cur_dim->name == dim_name)) {
      if ((output_tensor->repeat[i] != af::sym::kSymbolOne && repeat == af::sym::kSymbolOne) ||
          (output_tensor->stride[i] != af::sym::kSymbolZero && stride == af::sym::kSymbolZero)) {
        InsertAxis(cur_dim, node_info, broadcast_axis_);
      }
      break;
    }
  }
}

void ExeTimePassManager::AddRAxis(const std::string &dim_name, const Expr &repeat, const Expr &stride,
                                  const NodeInfo &node_info) {
  TensorPtr output_tensor = node_info.outputs[0];
  for (size_t i = 0; i < output_tensor->dim_info.size(); i++) {
    auto &cur_dim = output_tensor->dim_info[i];
    if ((cur_dim->name == dim_name)) {
      if ((repeat != af::sym::kSymbolOne && output_tensor->repeat[i] == af::sym::kSymbolOne) ||
          (stride != af::sym::kSymbolZero && output_tensor->stride[i] == af::sym::kSymbolZero)) {
        InsertAxis(cur_dim, node_info, reduce_axis_);
      } else {
        InsertAxis(cur_dim, node_info, non_reduce_axis_);
      }
      break;
    }
  }
}

void ExeTimePassManager::CheckBroadcast(const NodeInfo &node) {
  const std::string kNodeBroadcast = "Broadcast";
  if (node.node_type == kNodeBroadcast ||
      (node.node_ptr != nullptr && ascgen_utils::IsGeneralizeBrcInlineScene(node.node_ptr))) {
    auto &tensor = node.inputs[0];
    for (size_t i = 0; i < tensor->dim_info.size(); i++) {
      AddBAxis(tensor->dim_info[i]->name, tensor->repeat[i], tensor->stride[i], node);
    }
  }
}

af::Status CheckExecConditionBroadcast(const TuningSpacePtr tuning_space, const NodeInfo &node, SubAxis *axis,
                                       bool &is_split, Expr **fused_axis) {
  GE_ASSERT_NOTNULL(tuning_space);
  GE_ASSERT_NOTNULL(axis);
  GE_ASSERT_NOTNULL(fused_axis);
  if (node.exec_condition != af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis &&
      node.exec_condition != af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis) {
    GELOGD("exec_condition not match");
    return af::SUCCESS;
  }
  is_split = true;
  if (node.exec_condition == af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis) {
    for (const auto &axes : tuning_space->sub_axes) {
      GE_ASSERT_NOTNULL(axes);
      if (!axes->is_split || axes->axis_type != AxisPosition::OUTER) {
        continue;
      }
      for (const auto parent_axis : axes->parent_axis) {
        GE_ASSERT_NOTNULL(parent_axis);
        if (parent_axis->name == axis->name) {
          GELOGD("fused_axis %s", Str(axes->repeat).c_str());
          *fused_axis = &axes->repeat;
          break;
        }
      }
    }
  }
  return af::SUCCESS;
}

void ExeTimePassManager::CheckReduce(const NodeInfo &node) {
  const std::vector<std::string> kNodeReduce = {"Sum", "Max", "Min"};
  if (std::find(kNodeReduce.begin(), kNodeReduce.end(), node.node_type) != kNodeReduce.end()) {
    auto &tensor = node.inputs[0];
    for (size_t i = 0; i < tensor->dim_info.size(); i++) {
      AddRAxis(tensor->dim_info[i]->name, tensor->repeat[i], tensor->stride[i], node);
    }
  }
}

void ExeTimePassManager::UpdateBufNode(const std::vector<NodeInfo> &nodes) {
  std::string log;
  bool broadcast_node = false;
  for (int32_t i = nodes.size() - 1; i >= 0; --i) {
    auto &node = nodes[i];
    if (node.exec_condition == af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis ||
        node.exec_condition == af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis) {
      GELOGD("insert brc inline node by exec_condition %s", node.name.c_str());
      brc_buf_node_.insert(node.name);
    }
    if (broadcast_node) {
      brc_buf_node_.insert(node.name);
    }
    if (node.node_type == kBroadcast) {
      brc_buf_node_.insert(node.name);
      broadcast_node = true;
    } else if (node.node_type == kLoad) {
      broadcast_node = false;
      const auto platform = optimize::PlatformFactory::GetInstance().GetPlatform();
      if (platform != nullptr && ascgen_utils::IsLinkToBrdcst(node.node_ptr, platform->BroadcastTypes())) {
        GELOGD("insert brc inline node %s", node.name.c_str());
        brc_buf_node_.insert(node.name);
      }
    }
  }
  for (const auto &node_name : brc_buf_node_) {
    log += node_name + ", ";
  }
  GELOGD("Brc related node is {%s}.", log.c_str());
}

void ExeTimePassManager::GenLog(const std::string &type_name,
                                const std::map<std::string, std::set<std::string>> &axis_list) const {
  std::string log;
  for (const auto &pair : axis_list) {
    log.clear();
    for (const auto &node_name : pair.second) {
      log += node_name + ", ";
    }
    GELOGD("%s axis for node[%s] is {%s}.", type_name.c_str(), pair.first.c_str(), log.c_str());
  }
}

bool ExeTimePassManager::GetRLoop(const NodeInfo &node, Expr &r_loop) const {
  for (const auto &axis : node.loop_axes) {
    GE_ASSERT_NOTNULL(axis, "Get axis failed.");
    if (axis->axis_type != AxisPosition::OUTER && axis->axis_type != AxisPosition::INNER) {
      continue;
    }
    for (const auto &node_name : node.from_data) {
      bool r_split = false;
      for (const auto &orig_axis : axis->orig_axis) {
        CheckSplit(reduce_axis_, node_name, orig_axis->name, r_split);
      }
      if (r_split) {
        r_loop = axis->repeat;
        GELOGD("r_loop of [%s] is [%s].", node.name.c_str(), Str(r_loop).c_str());
        return true;
      }
    }
  }
  return false;
}

Expr ExeTimePassManager::GetBlockDimExpr() const {
  if (tuning_space_ == nullptr || tuning_space_->block_dims.empty()) {
    return Expr();
  }
  Expr block_dim;
  bool has_block_dim = false;
  for (const auto &block_axes : tuning_space_->block_dims) {
    Expr group_dim = CreateExpr(1U);
    for (const auto *axis : block_axes) {
      if (axis == nullptr) {
        return Expr();
      }
      Expr axis_dim = axis->repeat;
      if (!axis_dim.IsValid()) {
        return Expr();
      }
      group_dim = af::sym::Mul(group_dim, axis_dim);
    }
    block_dim = has_block_dim ? af::sym::Max(block_dim, group_dim) : group_dim;
    has_block_dim = true;
  }
  return block_dim.Simplify();
}

/*
brc缓存执行逻辑：
1.递归计算节点的输入信息，将每一个节点与一个或多个Data节点绑定
2.扫描Broadcast节点与Reduce节点，根据绑定的Data节点确定每个Data节点的B轴，A轴与R轴
3.扫描Load至Broadcast间的所有节点
4.判断这些节点的循环轴间是否存在绑定的Data节点的B切分轴，若存在，则使能缓存：
1）若切分轴同时为R轴或非A轴，则使用性能公式除以切分轴的循环次数
2）若切分轴同时是A轴，且当前loop axis中与R轴相关的循环轴的循环轮次为1，则使用性能公式除以B切分轴的循环次数，否则不做压缩
*/
bool ExeTimePassManager::CheckAxisSplit(const NodeInfo &node, const SubAxis *axis, Expr *fused_axis) const {
  bool b_split = false;
  bool a_split = false;
  bool r_split = false;
  for (const auto &node_name : node.from_data) {
    for (const auto &orig_axis : axis->orig_axis) {
      CheckSplit(broadcast_axis_, node_name, orig_axis->name, b_split);
      CheckSplit(reduce_axis_, node_name, orig_axis->name, r_split);
      CheckSplit(non_reduce_axis_, node_name, orig_axis->name, a_split);
      CheckExecConditionBroadcast(tuning_space_, node, orig_axis, b_split, &fused_axis);
    }
    if (b_split) {
      return true;
    }
  }
  return b_split;
}

TernaryOp ExeTimePassManager::HandleBroadcastSplit(const NodeInfo &node, const Expr &exe_time,
                                                   const Expr &fused_exe_time) const {
  GELOGD("[DFX] fused broadcast updates [%s] exe time : [%s] -> [%s]", node.name.c_str(), Str(exe_time).c_str(),
         Str(fused_exe_time).c_str());
  return TernaryOp(fused_exe_time);
}

TernaryOp ExeTimePassManager::HandleReduceOrNormalSplit(const NodeInfo &node, const Expr &exe_time, const SubAxis *axis,
                                                        bool r_split, bool a_split) const {
  Expr cur_exe_time = af::sym::Div(exe_time, axis->repeat);
  if (r_split || !a_split) {
    GELOGD("Axis [%s] r_split and not asplit.", axis->name.c_str());
    GELOGD("Brc buf module updates [%s] exe time : [%s] -> [%s]", node.name.c_str(), Str(exe_time).c_str(),
           Str(cur_exe_time).c_str());
    return TernaryOp(cur_exe_time);
  }
  GELOGD("Axis [%s] a_split.", axis->name.c_str());
  Expr r_loop;
  if (GetRLoop(node, r_loop)) {
    GELOGD("Brc buf module updates [%s] exe time : [%s] -> [%s == 1 ? %s : %s]", node.name.c_str(),
           Str(exe_time).c_str(), Str(r_loop).c_str(), Str(cur_exe_time).c_str(), Str(exe_time).c_str());
    return TernaryOp(CondType::K_EQ, r_loop, CreateExpr(1.0f), af::sym::Div(cur_exe_time, r_loop), exe_time);
  }
  return TernaryOp(exe_time);
}

TernaryOp ExeTimePassManager::UpdateNodeExeTime(const NodeInfo &node, const Expr &exe_time) const {
  const bool is_cache_guard = node.exec_condition == af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis ||
                              node.exec_condition == af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  CacheGuardInfo guard_info;
  if (TryBuildCacheGuardInfo(node, guard_info) && guard_info.address_invariant) {
    const auto block_dim = GetBlockDimExpr();
    if (IsBlockCountUniform(guard_info, block_dim)) {
      return TernaryOp(CountGuardHits(guard_info, CreateExpr(0U)));
    }
    if (is_cache_guard) {
      GELOGD("Node[%s] cache guard exe-time fallback: block hit count is not uniform.", node.name.c_str());
    }
  } else if (is_cache_guard) {
    GELOGD("Node[%s] cache guard exe-time fallback: metadata unavailable or address is variant.", node.name.c_str());
  }
  if (brc_buf_node_.find(node.name) == brc_buf_node_.end()) {
    return TernaryOp(exe_time);
  }
  Expr *fused_axis = nullptr;
  bool b_split = false;
  bool a_split = false;
  bool r_split = false;

  for (const auto &axis : node.loop_axes) {
    GE_ASSERT_NOTNULL(axis, "Get axis failed.");
    if (axis->axis_type != AxisPosition::OUTER && axis->axis_type != AxisPosition::INNER) {
      continue;
    }
    // Reset flags for each axis
    b_split = false;
    a_split = false;
    r_split = false;

    for (const auto &node_name : node.from_data) {
      for (const auto &orig_axis : axis->orig_axis) {
        CheckSplit(broadcast_axis_, node_name, orig_axis->name, b_split);
        CheckSplit(reduce_axis_, node_name, orig_axis->name, r_split);
        CheckSplit(non_reduce_axis_, node_name, orig_axis->name, a_split);
        CheckExecConditionBroadcast(tuning_space_, node, orig_axis, b_split, &fused_axis);
      }
      if (b_split) {
        break;
      }
    }

    if (fused_axis != nullptr) {
      Expr fused_exe_time = af::sym::Max(af::sym::kSymbolOne, af::sym::Div(exe_time, *fused_axis));
      return HandleBroadcastSplit(node, exe_time, fused_exe_time);
    }

    if (b_split) {
      return HandleReduceOrNormalSplit(node, exe_time, axis, r_split, a_split);
    }
  }
  return TernaryOp(exe_time);
}
}  // namespace att
