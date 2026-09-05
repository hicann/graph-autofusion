/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPR_GEN_EXE_TIME_PASS_MANAGER_H_
#define EXPR_GEN_EXE_TIME_PASS_MANAGER_H_

#include <set>
#include "util/ternary_op.h"
#include "parser/tuning_space.h"
#include "expr_gen/cache_guard_count.h"

namespace att {
class ExeTimePassManager {
 public:
  // 解析B轴，a轴，r轴
  explicit ExeTimePassManager(const TuningSpacePtr &tuning_space) {
    broadcast_axis_.clear();
    reduce_axis_.clear();
    brc_buf_node_.clear();
    tuning_space_ = tuning_space;
    std::vector<NodeInfo> nodes = tuning_space->node_infos;
    for (const auto &node : nodes) {
      CheckBroadcast(node);
      CheckReduce(node);
    }
    GenLog("B", broadcast_axis_);
    GenLog("R", reduce_axis_);
    GenLog("A", non_reduce_axis_);
    UpdateBufNode(nodes);
  }
  ~ExeTimePassManager() = default;

  // 获取需要处理的节点
  TernaryOp UpdateNodeExeTime(const NodeInfo &node, const Expr &exe_time) const;

  // 从 parser 已有的轴关系中重建 cache guard 元数据；无法唯一确定时返回 false。
  bool TryBuildCacheGuardInfo(const NodeInfo &node, CacheGuardInfo &info) const;

 private:
  void CheckReduce(const NodeInfo &node);
  void CheckBroadcast(const NodeInfo &node);
  void AddRAxis(const std::string &dim_name, const Expr &repeat, const Expr &stride, const NodeInfo &node_info);
  void AddBAxis(const std::string &dim_name, const Expr &repeat, const Expr &stride, const NodeInfo &node_info);
  void UpdateBufNode(const std::vector<NodeInfo> &nodes);
  void GenLog(const std::string &type_name, const std::map<std::string, std::set<std::string>> &axis_list) const;
  bool GetRLoop(const NodeInfo &node, Expr &r_loop) const;
  Expr GetBlockDimExpr() const;
  bool CheckAxisSplit(const NodeInfo &node, const SubAxis *axis, Expr *fused_axis) const;
  TernaryOp HandleBroadcastSplit(const NodeInfo &node, const Expr &exe_time, const Expr &fused_exe_time) const;
  TernaryOp HandleReduceOrNormalSplit(const NodeInfo &node, const Expr &exe_time, const SubAxis *axis, bool r_split,
                                      bool a_split) const;
  TuningSpacePtr tuning_space_;
  std::map<std::string, std::set<std::string>> broadcast_axis_;
  std::map<std::string, std::set<std::string>> reduce_axis_;
  std::map<std::string, std::set<std::string>> non_reduce_axis_;
  std::set<std::string> brc_buf_node_;
};
}  // namespace att

#endif
