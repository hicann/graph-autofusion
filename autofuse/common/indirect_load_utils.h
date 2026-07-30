/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INDIRECT_LOAD_UTILS_H__
#define __INDIRECT_LOAD_UTILS_H__

#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace ascgen_utils::indirect_load {
enum class TemplateRole : int64_t {
  kNone,
  kSimdInputPre,
  kSimtInputBoundary,
  kSimtDirectGmBoundary,
  kSimtInlineTransform,
  kSimtOp,
};

struct TemplateBehavior {
  bool excludes_tiling_group = false;
  bool skips_main_schedule_tiling = false;
  bool skips_api_emit = false;
  bool uses_direct_gm_pipeline = false;
  bool skips_ub_lifecycle = false;
  bool preserves_vectorized_axis = false;
};

struct TemplateAxes {
  af::AxisId outer_axis = af::kIdNone;
  af::AxisId inner_axis = af::kIdNone;
  af::AxisId input_inner_axis = af::kIdNone;
};

struct LogicalTensorView {
  std::vector<af::AxisId> axis_ids;
  std::vector<af::Expression> strides;
};

struct TemplateLogicalView {
  LogicalTensorView data;
  LogicalTensorView index;
  LogicalTensorView output;
};

TemplateBehavior GetTemplateBehavior(const af::AscNodePtr &node);
TemplateRole GetTemplateRole(const af::AscNodePtr &node);
af::Status InheritTemplateRoleIfIL(af::AscGraph &graph, const std::string &vf_node_name, const af::AscNodePtr &src);
af::Status SetTemplateRole(const af::AscNodePtr &node, TemplateRole role);
af::Status SetTemplateAxes(const af::AscNodePtr &node, const TemplateAxes &axes);
af::Status GetTemplateAxes(const af::AscNodePtr &node, TemplateAxes &axes);
af::Status SetTemplateLogicalView(const af::AscNodePtr &node, const TemplateLogicalView &view);
af::Status GetTemplateLogicalView(const af::AscNodePtr &node, TemplateLogicalView &view);
bool ShouldSkipMainScheduleTiling(const af::AscNodePtr &node);
bool ShouldPreserveVectorizedAxis(const af::AscNodePtr &node);
bool ShouldApplyInputInnerVectorization(const af::AscNodePtr &node);
bool ShouldDisableRegularVectorFunc(const af::AscNodePtr &node);
af::AscNodePtr GetInputProducer(const af::AscNodePtr &node, size_t input_index);
af::AscNodePtr GetOnlyOutputConsumer(const af::AscNodePtr &node);
af::AscNodePtr FindIndirectLoadNode(const af::AscGraph &graph);
af::Status ValidateSingleIndirectLoadNode(const af::AscGraph &graph, af::AscNodePtr &node);
af::Status GetPrebuiltYTilingCase(const af::AscGraph &graph, bool &has_case, af::AxisId &tile_id,
                                  std::pair<af::AxisPtr, af::AxisPtr> &tiling);
}  // namespace ascgen_utils::indirect_load

#endif  // __INDIRECT_LOAD_UTILS_H__
