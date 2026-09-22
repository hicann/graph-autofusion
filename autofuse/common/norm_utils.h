/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.huawei.com
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NORM_UTILS_H__
#define __NORM_UTILS_H__

#include <cstdint>
#include <vector>
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace ascgen_utils::norm {

// Norm 元数据属性键，附着在候选图的 IndirectLoad 节点上随图传递。
constexpr char kNormInfoAttr[] = "af.internal.gather_norm.norm_info";

// 单个 Reduce → Broadcast 组合的对应关系。
// reduce_node 与 broadcast_node 通过节点名配对（候选图内名字唯一），
// reduced_axes 保存归约轴在区域入口逻辑 View 中的位置，用于轴保持证明。
struct NormStage {
  std::string reduce_node_name;
  std::string broadcast_node_name;
  std::vector<af::AxisId> reduced_axes;
};

// 轴保持复合区域的完整描述。区域入口为 Gather 输出（或其后置
// Elementwise 链的起点），最终输出轴与入口轴保持一致。
// kind 为 kSoftmaxDedicated 时走现有专用 Softmax API；
// kGenericComposite 时保留原始节点按通用模板处理。
struct NormInfo {
  enum class Kind : int64_t {
    kNone = 0,
    kSoftmaxDedicated = 1,  // Pattern 命中且尾轴 R 约束满足
    kGenericComposite = 2,  // 通用轴保持复合区域（含 Softmax 尾轴不满足兜底）
  };

  Kind kind = Kind::kNone;
  // 区域入口节点名（Gather 输出的直接后置链起点）。
  std::string entry_node_name;
  // 最终 Store 前的输出节点名。
  std::string exit_node_name;
  // 区域内全部节点名，用于分组完整性检查。
  std::vector<std::string> region_node_names;
  // 各 Reduce → Broadcast 组合。
  std::vector<NormStage> stages;
  // 入口逻辑 View 的轴序列与保持轴集合。
  std::vector<af::AxisId> entry_axes;
  std::vector<af::AxisId> preserved_axes;
  // Softmax 专用路径：归约轴（尾轴约束已验证）。
  af::AxisId softmax_reduce_axis = af::kIdNone;
};

af::Status SetNormInfo(const af::AscNodePtr &node, const NormInfo &info);
af::Status TryGetNormInfo(const af::AscNodePtr &node, NormInfo &info);
bool HasNormInfo(const af::AscNodePtr &node);

}  // namespace ascgen_utils::norm

#endif  // __NORM_UTILS_H__
