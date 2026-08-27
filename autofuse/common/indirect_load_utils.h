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
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace ascgen_utils::indirect_load {
constexpr size_t kInputTensorIndex = 0UL;
constexpr size_t kIndexTensorIndex = 1UL;
constexpr char kTemplateLogicalViewAttr[] = "af.internal.indirect_load.logical_view";

enum class TemplateRole : int64_t {
  kNone,
  kSimdInputPre,
  kSimdInputPreStridedUbPath,
  kSimdIndexPre,
  kSimtInputBoundary,
  kSimtDirectGmBoundary,
  kSimtInlineTransform,
  kSimtFanoutBranch,
  kSimtOp,
  kSkInputBoundary,
  kStridedUbPath,
};

enum class Implementation : int64_t {
  // The default SIMD implementation uses MicroAPI.
  kDefault,
  kGatherApi,
};

struct TemplateBehavior {
  bool excludes_tiling_group = false;
  bool skips_main_schedule_tiling = false;
  bool skips_api_emit = false;
  bool uses_direct_gm_pipeline = false;
  bool skips_ub_lifecycle = false;
  bool skips_input_lifecycle = false;
  bool preserves_vectorized_axis = false;
};

struct TemplateAxes {
  af::AxisId outer_axis = af::kIdNone;
  af::AxisId inner_axis = af::kIdNone;
  af::AxisId input_inner_axis = af::kIdNone;
  af::AxisId index_inner_axis = af::kIdNone;
  af::AxisId tile_outer_axis = af::kIdNone;
  af::AxisId tile_inner_axis = af::kIdNone;
  std::vector<af::AxisId> vectorized_axes;
  bool synthetic_outer = false;
};

struct LogicalTensorView {
  std::vector<af::AxisId> axis_ids;
  std::vector<af::Expression> sizes;
  std::vector<af::Expression> strides;
};

enum class IndirectLoadLayoutKind : int64_t {
  kDense = 0,
  kZeroStrideCompact = 1,
  kStrided = 2,
  kUnsupported = 3,
};

struct IndirectLoadTensorLayout : LogicalTensorView {
  IndirectLoadLayoutKind kind = IndirectLoadLayoutKind::kUnsupported;
  std::vector<af::Expression> physical_repeats;
};

struct TemplateLogicalView {
  IndirectLoadTensorLayout input;
  IndirectLoadTensorLayout index;
  LogicalTensorView output;
};

TemplateBehavior GetTemplateBehavior(const af::AscNodePtr &node);
TemplateRole GetTemplateRole(const af::AscNodePtr &node);
af::AscNodePtr GetPostReduceConsumer(const af::AscNodePtr &node);
af::AscNodePtr GetPostReduceInputProducer(const af::AscNodePtr &node);
bool ShouldSkipTpipeTensorCollection(const af::AscNodePtr &node);
af::Status InheritTemplateRoleIfIL(af::AscGraph &graph, const std::string &vf_node_name, const af::AscNodePtr &src);
af::Status SetTemplateRole(const af::AscNodePtr &node, TemplateRole role);
af::Status SetTemplateAxes(const af::AscNodePtr &node, const TemplateAxes &axes);
af::Status GetTemplateAxes(const af::AscNodePtr &node, TemplateAxes &axes);
af::Status SetTemplateLogicalView(const af::AscNodePtr &node, const TemplateLogicalView &view);
af::Status GetTemplateLogicalView(const af::AscNodePtr &node, TemplateLogicalView &view);
af::Status SetImplementation(const af::AscNodePtr &node, Implementation implementation);
af::Status GetImplementation(const af::AscNodePtr &node, Implementation &implementation);
af::Status ClassifyIndirectLoadLayout(const LogicalTensorView &logical, IndirectLoadTensorLayout &layout,
                                      bool allow_non_overlapping_zero_stride = false);
af::Status ValidateIndirectLoadOutputLayout(const LogicalTensorView &output);
bool ShouldApplyInputInnerVectorization(const af::AscNodePtr &node);
af::AscNodePtr GetInputProducer(const af::AscNodePtr &node, size_t input_index);
af::AscNodePtr GetOnlyOutputConsumer(const af::AscNodePtr &node);
af::AscNodePtr FindIndirectLoadNode(const af::AscGraph &graph);
af::Status ValidateSingleIndirectLoadNode(const af::AscGraph &graph, af::AscNodePtr &node);
bool GetPrebuiltYTilingCase(const af::AscGraph &graph, af::AxisId &tile_id,
                            std::pair<af::AxisPtr, af::AxisPtr> &tiling);
}  // namespace ascgen_utils::indirect_load

#endif  // __INDIRECT_LOAD_UTILS_H__
