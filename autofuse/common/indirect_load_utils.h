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
#include "ascir.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace ascgen_utils::indirect_load {
constexpr size_t kInputTensorIndex = 0UL;
constexpr size_t kIndexTensorIndex = 1UL;
constexpr char kTemplateLogicalViewAttr[] = "af.internal.indirect_load.logical_view";
constexpr char kAccessInfoAttr[] = "af.internal.indirect_load.access_info";
constexpr char kLoweringMetadataAttr[] = "af.internal.indirect_load.lowering_metadata";

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

struct IndirectLoadAccessInfo {
  enum class Kind : int64_t { kGeneric, kEmbeddingLike };

  Kind kind = Kind::kGeneric;
  int64_t axis = -1L;
  af::Expression input_slice_bytes;
  af::Expression index_varying_extent;
  bool can_use_simt_structured = false;
  // SIMD uses a contiguous payload suffix and a zero-stride index suffix.
  // This is stricter than the generalized SIMT structured-layout condition.
  bool can_use_simd_embedding = false;
};

enum class SimdFallback : int64_t {
  kRegisterGather = 0,
  kGatherApi = 1,
  kStrided = 2,
};

struct SimdLoweringMetadata {
  SimdFallback fallback = SimdFallback::kRegisterGather;
  bool try_embedding = false;
};

enum class SimtAddressPolicy : int64_t {
  kStaticPowerOfTwo = 0,
  kStaticInner = 1,
  kStructuredMagic = 2,
  kEmbedding = 3,
  kRecursive = 4,
  kStrided = 5,
};

enum class SimtOffsetWidth : int64_t {
  kUint32,
  kUint64,
};

enum class SimtLoadAddressSource : int64_t {
  kZeroOffset,
  kOutputOffset,
  kIndexOffset,
};

struct SimtPolicyMetadata {
  SimtAddressPolicy policy = SimtAddressPolicy::kRecursive;
  SimtOffsetWidth offset_width = SimtOffsetWidth::kUint64;
  uint64_t inner_span = 0U;
  uint64_t output_axis_span = 0U;
  uint64_t input_axis_stride = 0U;
  uint64_t input_axis_span = 0U;
  uint64_t input_stride_mask = 0U;
  uint64_t index_stride_mask = 0U;
  std::vector<af::Expression> runtime_params;
};

struct SimtGmTensorMetadata {
  ascir::TensorId value_tensor_id = af::kIdNone;
  ascir::TensorId gm_tensor_id = af::kIdNone;
  af::DataType dtype = af::DT_UNDEFINED;
  bool is_scalar = false;
};

struct SimtLoadMetadata {
  std::string node_name;
  SimtLoadAddressSource address_source = SimtLoadAddressSource::kOutputOffset;
  bool use_logical_offset = false;
  LogicalTensorView physical_view;
};

struct SimtOutputChainMetadata {
  std::vector<std::string> node_names;
  ascir::TensorId result_tensor_id = af::kIdNone;
  ascir::TensorId target_tensor_id = af::kIdNone;
  af::DataType dtype = af::DT_UNDEFINED;
  bool local_target = false;
};

struct SimtLoweringMetadata {
  bool has_post_reduce = false;
  std::vector<std::string> index_node_names;
  std::vector<std::string> output_node_names;
  std::vector<SimtLoadMetadata> index_loads;
  std::vector<SimtLoadMetadata> output_loads;
  std::vector<SimtOutputChainMetadata> output_chains;
  std::vector<SimtGmTensorMetadata> gm_tensors;
  SimtPolicyMetadata policy;
  ascir::TensorId index_result_tensor_id = af::kIdNone;
  ascir::TensorId value_tensor_id = af::kIdNone;
  ascir::TensorId output_result_tensor_id = af::kIdNone;
  af::DataType index_dtype = af::DT_UNDEFINED;
  af::DataType output_dtype = af::DT_UNDEFINED;
};

struct IndirectLoadLoweringMetadata {
  uint32_t version = 1U;
  int64_t axis = -1L;
  af::AxisId outer_axis = af::kIdNone;
  TemplateLogicalView logical_view;
  IndirectLoadAccessInfo access_info;
  SimdLoweringMetadata simd;
  SimtLoweringMetadata simt;
};

TemplateBehavior GetTemplateBehavior(const af::AscNodePtr &node);
TemplateRole GetTemplateRole(const af::AscNodePtr &node);
af::AscNodePtr GetPostReduceConsumer(const af::AscNodePtr &node);
af::AscNodePtr GetPostReduceInputProducer(const af::AscNodePtr &node);
// Returns the terminal tensor ID of a chain whose API emission is skipped.
ascir::TensorId FindSkippedChainResultTensor(const af::AscNodePtr &root);
bool ShouldSkipTpipeTensorCollection(const af::AscNodePtr &node);
af::Status InheritTemplateRoleIfIL(af::AscGraph &graph, const std::string &vf_node_name, const af::AscNodePtr &src);
af::Status SetTemplateRole(const af::AscNodePtr &node, TemplateRole role);
af::Status SetTemplateAxes(const af::AscNodePtr &node, const TemplateAxes &axes);
af::Status GetTemplateAxes(const af::AscNodePtr &node, TemplateAxes &axes);
af::Status SetTemplateLogicalView(const af::AscNodePtr &node, const TemplateLogicalView &view);
af::Status GetTemplateLogicalView(const af::AscNodePtr &node, TemplateLogicalView &view);
af::Status SetIndirectLoadAccessInfo(const af::AscNodePtr &node, const IndirectLoadAccessInfo &info);
af::Status GetIndirectLoadAccessInfo(const af::AscNodePtr &node, IndirectLoadAccessInfo &info);
af::Status SetLoweringMetadata(const af::AscNodePtr &node, const IndirectLoadLoweringMetadata &metadata);
af::Status GetLoweringMetadata(const af::AscNodePtr &node, IndirectLoadLoweringMetadata &metadata);
bool HasLoweringMetadata(const af::AscNodePtr &node);
af::Status FinalizeLoweringMetadata(const af::AscNodePtr &node, bool &is_supported);
af::Status AnalyzeIndirectLoadAccess(const af::AscNodePtr &node, const TemplateLogicalView &logical_view,
                                     IndirectLoadAccessInfo &info);
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
