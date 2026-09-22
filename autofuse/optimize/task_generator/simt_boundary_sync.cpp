/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.hiascend.com/software/licensedistributionexception
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#include "task_generator/simt_boundary_sync.h"
#include "common/common_utils.h"

#include <algorithm>
#include <set>

#include "utils/axis_utils.h"

namespace optimize::task_generator {
namespace {

constexpr int32_t kMaxAxisAncestorDepth = 8;

// 判断节点是否为需要 view 对齐的 SIMT 模板角色（跳过主调度 tiling 的角色）
bool IsSimtViewSyncRole(const ascgen_utils::indirect_load::TemplateRole role) {
  return role == ascgen_utils::indirect_load::TemplateRole::kSimtInputBoundary ||
         role == ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary ||
         role == ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform ||
         role == ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch ||
         role == ascgen_utils::indirect_load::TemplateRole::kSkInputBoundary;
}

// 判断 axis_id 是否为 target_id 的祖先轴：沿 from 链向上追溯（含自身），
// 既覆盖 split 链（from.size()==1）也覆盖 merge 关系（target 是 axis 的 from 成员）
bool IsAxisAncestorOf(af::AscGraph &graph, const ascir::AxisId axis_id, const ascir::AxisId target_id) {
  if (axis_id == target_id) {
    return true;
  }
  ascir::AxisId cursor = axis_id;
  for (int32_t depth = 0; depth < kMaxAxisAncestorDepth && cursor != af::kIdNone; ++depth) {
    const auto axis = graph.FindAxis(cursor);
    if (axis == nullptr) {
      break;
    }
    // merge 关系：target 是当前轴的合并源成员（如旧轴 1 被 merge 进 from=[0,1] 的轴 2）
    if (std::find(axis->from.begin(), axis->from.end(), target_id) != axis->from.end()) {
      return true;
    }
    if (axis->from.size() != 1UL) {
      break;
    }
    cursor = axis->from.front();
  }
  return false;
}

// 对单个 output 的 tensor view 执行 split 同步（不改 sched 轴）
void SplitOutputView(const af::AscTensor &output, const af::AxisPtr &outer, const af::AxisPtr &inner) {
  const ascir::AxisId split_original = outer->from[0];
  if (std::find(output.attr.axis.begin(), output.attr.axis.end(), split_original) == output.attr.axis.end()) {
    return;
  }
  const auto view = af::AxisUtils::SplitView({output.attr.axis, output.attr.repeats, output.attr.strides}, inner->size,
                                             outer->id, inner->id, split_original);
  output.attr.axis = view.axis_ids;
  output.attr.repeats = view.repeats;
  output.attr.strides = view.strides;
}

// vectorized_axis 重映射：旧轴被模板 merge 合并、合并轴又被 tiling split 时，
// 支持两级链（旧轴 -> 合并轴 -> inner 轴）映射为 view 中实际存在的新轴
void RemapOutputVectorizedAxes(af::AscGraph &graph, const af::AscNodePtr &node, const af::AscTensor &output,
                               const std::set<ascir::AxisId> &merged_axis_ids,
                               const std::vector<std::pair<af::AxisPtr, af::AxisPtr>> &tiled_axes_list) {
  std::vector<ascir::AxisId> remapped;
  remapped.reserve(output.attr.vectorized_axis.size());
  for (const auto vec_axis_id : output.attr.vectorized_axis) {
    if (std::find(output.attr.axis.begin(), output.attr.axis.end(), vec_axis_id) != output.attr.axis.end()) {
      remapped.push_back(vec_axis_id);
      continue;
    }
    // 旧轴不在 view 中：遍历 split 对，若 inner 轴在 view 中且 inner 的祖先链命中
    // 旧轴或旧轴所属的合并轴，则映射到 inner 轴
    bool remapped_ok = false;
    for (const auto &tiled_axes : tiled_axes_list) {
      if (tiled_axes.first == nullptr || tiled_axes.second == nullptr || tiled_axes.first->from.size() != 1UL) {
        continue;
      }
      const ascir::AxisId inner_id = tiled_axes.second->id;
      if (std::find(output.attr.axis.begin(), output.attr.axis.end(), inner_id) == output.attr.axis.end()) {
        continue;
      }
      const bool direct_ancestor = IsAxisAncestorOf(graph, tiled_axes.first->from[0], vec_axis_id);
      bool via_merged = false;
      if (!direct_ancestor) {
        for (const auto merged_axis_id : merged_axis_ids) {
          if (IsAxisAncestorOf(graph, tiled_axes.first->from[0], merged_axis_id) &&
              IsAxisAncestorOf(graph, merged_axis_id, vec_axis_id)) {
            via_merged = true;
            break;
          }
        }
      }
      if (direct_ancestor || via_merged) {
        GELOGD("[IndirectLoad] SIMT role node[%s] remap vectorized axis[%ld] to inner axis[%ld].", node->GetNamePtr(),
               vec_axis_id, inner_id);
        remapped.push_back(inner_id);
        remapped_ok = true;
        break;
      }
    }
    if (!remapped_ok) {
      // 无法映射时保留原值，等待后续校验暴露
      GELOGD("[IndirectLoad] SIMT role node[%s] keep unmapped vectorized axis[%ld].", node->GetNamePtr(), vec_axis_id);
      remapped.push_back(vec_axis_id);
    }
  }
  output.attr.vectorized_axis = remapped;
}

}  // namespace

af::Status SyncSimtBoundaryViews(af::AscGraph &graph,
                                 const std::vector<std::pair<af::AxisPtr, af::AxisPtr>> &tiled_axes_list) {
  if (tiled_axes_list.empty()) {
    return af::SUCCESS;
  }
  const auto indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(graph);
  if (indirect_load == nullptr) {
    return af::SUCCESS;
  }
  // 模板合并轴集合（outer/inner），用于 vectorized_axis 两级重映射
  std::set<ascir::AxisId> merged_axis_ids;
  ascgen_utils::indirect_load::TemplateAxes template_axes;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, template_axes));
  for (const auto axis_id : {template_axes.outer_axis, template_axes.inner_axis}) {
    if (axis_id != af::kIdNone) {
      merged_axis_ids.insert(axis_id);
    }
  }

  for (const auto &node : graph.GetAllNodes()) {
    if (node == nullptr || !IsSimtViewSyncRole(ascgen_utils::indirect_load::GetTemplateRole(node))) {
      continue;
    }
    for (auto &output : node->outputs()) {
      if (output == nullptr) {
        continue;
      }
      // 1) tensor view split 同步：按调度实际执行的 (outer, inner) 逐层 split，
      //    使 view 状态与被完整调度的普通节点等价
      for (const auto &tiled_axes : tiled_axes_list) {
        if (tiled_axes.first == nullptr || tiled_axes.second == nullptr || tiled_axes.first->from.size() != 1UL) {
          continue;
        }
        GELOGD("[IndirectLoad] SIMT role node[%s] sync tensor view split axis[%ld] to [outer:%ld, inner:%ld].",
               node->GetNamePtr(), tiled_axes.first->from[0], tiled_axes.first->id, tiled_axes.second->id);
        SplitOutputView(*output, tiled_axes.first, tiled_axes.second);
      }
    }
    // 2) vectorized_axis 重映射：旧轴 ->（合并轴 ->）inner 轴
    for (auto &output : node->outputs()) {
      if (output == nullptr) {
        continue;
      }
      RemapOutputVectorizedAxes(graph, node, *output, merged_axis_ids, tiled_axes_list);
    }
  }
  return af::SUCCESS;
}

// post-Reduce SIMT 输出行数维收口（统一语义）：直接调度路径通过 prepend 把求解的
// TileInner 加入模板向量化轴集合，但多阶段展开（Reduce FirstStage）等路径重建的
// Phase 图仍按预建固定 tile 语义构造，其节点的向量化视图只剩尾轴；而 buffer 分配
// 与数据搬运按求解的多行 tile 生成。此处统一把视图前置的 TileInner 轴补入
// post-Reduce 链上各节点（含 IndirectLoad 输出及其消费者）的向量化视图前部，
// 使 Softmax/Reduce 等消费者按通用 A/R 逻辑自然得到正确行数；vectorized_strides
// 由随后的对齐阶段按补全后的轴重建，无需手工同步。
}  // namespace optimize::task_generator
