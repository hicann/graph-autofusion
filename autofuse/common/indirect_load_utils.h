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
  // [SK窗口可行性] index 源头是否物理收敛（沿生产链回溯到 Load，其原始视图在
  // gather 轴 payload 维上零贡献/退化）。源头收敛时 gather 的引用按行/列成组，
  // SK 行窗口只需装载被引用分片；源头不收敛（逐元素独立引用，如
  // index=[13,10,20000] 全轴 stride 非零）时 SK 窗口必须覆盖 axis 维全部，
  // 退化为全量。kind 为 generic 但源头收敛的形态（如尾轴 gather 无 payload
  // 后缀）不受 kind 标签影响。
  bool index_source_converged = false;
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
  // [行级广播 side-input] 尾轴 stride==0 且 size==1 的广播 side-input（每行读一个
  // 标量，如生产 gather+softmax 图的 load3 [8,2048,1]/strides=[2048,1,0]）：调度期
  // SIMT 边界的轴 split/merge 会把其视图改写为 rank 不匹配且尾段尺寸符号化的形态，
  // codegen 的兜底（稠密尾段取模）要求编译期常量而失效。此处保留改写前的原始
  // 视图，兜底时按原始语义重建坐标（行号定位，与 split 无关）。
  LogicalTensorView original_view;
  bool is_row_broadcast = false;
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
// [SK窗口可行性] 现场判定 index 源头收敛性（不依赖候选流程构建的 attr，跨图副本安全）。
bool IsIndexSourceConverged(const af::AscNodePtr &indirect_load);
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
// 判断 load 的物理视图是否与输出逻辑视图覆盖同一 dense 连续区域（语义等价）：
// 1) load 视图各有效轴（size!=1 且 stride!=0）的 stride 满足后缀乘积连续性；
// 2) 所有有效轴 sizes 的乘积与输出视图 sizes 乘积符号相等。
// 满足时 load 的线性偏移与 output_index 相同，可直接使用线性偏移，无需坐标重建。
// 调度会把普通节点视图 merge/split 到模板轴空间（如 outer 被拆为
// [s3*s4*s5/Tb, Tb]），符号级全等比较会漏判这类等价视图。
bool IsDenseEquivalentView(const LogicalTensorView &load_view, const LogicalTensorView &output_view);
bool ShouldApplyInputInnerVectorization(const af::AscNodePtr &node);
af::AscNodePtr GetInputProducer(const af::AscNodePtr &node, size_t input_index);
af::AscNodePtr GetOnlyOutputConsumer(const af::AscNodePtr &node);
af::AscNodePtr FindIndirectLoadNode(const af::AscGraph &graph);
af::Status ValidateSingleIndirectLoadNode(const af::AscGraph &graph, af::AscNodePtr &node);
bool GetPrebuiltYTilingCase(const af::AscGraph &graph, af::AxisId &tile_id,
                            std::pair<af::AxisPtr, af::AxisPtr> &tiling);
}  // namespace ascgen_utils::indirect_load

#endif  // __INDIRECT_LOAD_UTILS_H__
