/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_BASE_COMMON_SCHEDULE_RESULT_H_
#define ASCGEN_DEV_BASE_COMMON_SCHEDULE_RESULT_H_

#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "common/checker.h"

namespace {
constexpr char kTemplateIdAttr[] = "af.internal.template.id";
constexpr char kTemplateRoleAttr[] = "af.internal.indirect_load.role";
constexpr char kDcacheSizeAttr[] = "af.internal.template.dcache_size";
}  // namespace

namespace ascir {
struct ScheduleGroup {
  std::vector<af::AscGraph> impl_graphs;
  std::map<std::string, std::string> graph_name_to_score_funcs;
  bool double_buffer{false};
};

enum class CubeTemplateType : int32_t {
  kDefault = -1,  // no cube
  kFixpip,        // fixpip模板
  kCommon,        // 兜底模板
  kUBFuse,        // ub复用模板
  kL2Fuse,        // L2复用模板
};

struct ScheduledResult {
  std::vector<ScheduleGroup> schedule_groups;
  // dst -> src, <dst_groupid, <src_groupid, <dst_var_name, src_var>>>;
  std::map<size_t, std::map<size_t, std::map<std::string, af::Expression>>> var_relations;
  ge::AscendString score_func;
  bool is_reduce_mem_reuse{false};
  bool enable_group_parallel{false};
  CubeTemplateType cube_type{CubeTemplateType::kDefault};
};

struct FusedScheduledResult {
  ge::AscendString fused_graph_name;
  std::vector<af::AscNodePtr> input_nodes;
  std::vector<af::AscNodePtr> output_nodes;
  std::vector<af::AscNodePtr> workspace_nodes;
  std::vector<af::Expression> origin_vars;
  std::vector<std::vector<ScheduledResult>> node_idx_to_scheduled_results;
};

enum class TemplateId : int64_t {
  kDefault = -1,
  kIndirectLoadSimd = 0,
  kIndirectLoadSimt = 1,
  kIndirectLoadSK = 2,
};

inline af::Status SetTemplateId(const af::AscNodePtr &node, TemplateId template_id) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateIdAttr, static_cast<int64_t>(template_id)),
                 "Set internal template id failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

inline af::Status SetTemplateRole(const af::AscNodePtr &node, int64_t role) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateRoleAttr, role), "Set internal template role failed, node = %s",
                 node->GetNamePtr());
  return af::SUCCESS;
}

inline int64_t GetTemplateRoleOrDefault(const af::AscNode &node, int64_t default_role = -1) {
  if (node.GetOpDesc() == nullptr) {
    return default_role;
  }
  return node.GetOpDesc()->TryGetExtAttr(kTemplateRoleAttr, default_role);
}

inline TemplateId GetTemplateIdOrDefault(const af::AscNode &node, TemplateId default_id = TemplateId::kDefault) {
  if (node.GetOpDesc() == nullptr) {
    return default_id;
  }
  return static_cast<TemplateId>(node.GetOpDesc()->TryGetExtAttr(kTemplateIdAttr, static_cast<int64_t>(default_id)));
}

inline af::Status SetDcacheSize(const af::AscNodePtr &node, int64_t dcache_size) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kDcacheSizeAttr, dcache_size),
                 "Set internal template dcache size failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

inline int64_t GetDcacheSize(const af::AscNode &node) {
  if (node.GetOpDesc() == nullptr) {
    return 0;
  }
  return node.GetOpDesc()->TryGetExtAttr(kDcacheSizeAttr, int64_t{0});
}
}  // namespace ascir

#endif  // ASCGEN_DEV_BASE_COMMON_SCHEDULE_RESULT_H_
