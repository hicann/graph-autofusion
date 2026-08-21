/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reg_indirect_load_api_call.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_set>
#include "api_call/utils/api_call_factory.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "common/checker.h"
#include "common_utils.h"
#include "indirect_load_utils.h"
#include "utils/extern_math_util.h"
#include "v35/ascir/ascir_codegen_v2.h"

namespace codegen {
namespace {
constexpr size_t kIndirectLoadInputCount = 2UL;
constexpr size_t kIndirectLoadOutputCount = 1UL;
constexpr char kSimtContextNamePrefix[] = "IndirectLoadSimtContext_";
constexpr char kSimtBodyNamePrefix[] = "IndirectLoadSimtBody_";
constexpr char kGlobalTensorNamePrefix[] = "global_";
constexpr char kSimtGmFieldNamePrefix[] = "gm_";
constexpr char kSimtValueNamePrefix[] = "v_";

struct LogicalTensorInfo {
  LogicalTensorInfo() = default;
  explicit LogicalTensorInfo(const ascgen_utils::indirect_load::LogicalTensorView &view)
      : sizes(view.sizes), strides(view.strides) {}
  LogicalTensorInfo(const std::vector<ascir::SizeExpr> &sizes_in, const std::vector<ascir::SizeExpr> &strides_in)
      : sizes(sizes_in), strides(strides_in) {}

  std::vector<ascir::SizeExpr> sizes;
  std::vector<ascir::SizeExpr> strides;
};

enum class SimtAddressPolicy {
  kStaticPowerOfTwo,
  kStaticInner,
  kStructuredMagic,
  kRecursive,
  kStrided,
};

struct SimtCodegenPlan {
  SimtAddressPolicy policy = SimtAddressPolicy::kRecursive;
  std::string offset_type = "uint64_t";
  af::Expression inner_span = af::ops::One;
  af::Expression output_axis_span = af::ops::One;
  af::Expression input_axis_stride = af::ops::One;
  af::Expression input_axis_span = af::ops::One;
  uint64_t inner_span_value = 0U;
  uint64_t output_axis_span_value = 0U;
  uint64_t input_axis_stride_value = 0U;
  uint64_t input_axis_span_value = 0U;
  uint64_t input_stride_mask = 0U;
  uint64_t index_stride_mask = 0U;
};

af::Status EmitSimtScalarExpr(const ascir::NodeView &node, const std::vector<std::string> &inputs, std::string &expr) {
  GE_ASSERT_NOTNULL(node, "SIMT scalar node is null.");
  GE_ASSERT_TRUE(!inputs.empty() && inputs.size() == node->inputs.Size(),
                 "SIMT scalar node %s[%s] expects %zu non-empty inputs, but got %zu.", node->GetTypePtr(),
                 node->GetNamePtr(), node->inputs.Size(), inputs.size());
  const auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
  GE_ASSERT_NOTNULL(impl, "SIMT scalar codegen is not registered for node %s[%s].", node->GetTypePtr(),
                    node->GetNamePtr());
  const auto *v2_impl = dynamic_cast<af::ascir::AscIrCodegenV2 *>(impl.get());
  GE_ASSERT_NOTNULL(v2_impl, "SIMT scalar codegen for node %s[%s] is not a V2 implementation.", node->GetTypePtr(),
                    node->GetNamePtr());
  GE_ASSERT_TRUE(v2_impl->IsSimtScalarSupported(*node), "SIMT scalar codegen is not supported for node %s[%s].",
                 node->GetTypePtr(), node->GetNamePtr());
  return v2_impl->GenerateSimtScalarExpr(*node, inputs, expr);
}

bool IsAxisDerivedFrom(const TPipe &tpipe, ascir::AxisId axis_id, ascir::AxisId ancestor_axis_id) {
  if (axis_id == ancestor_axis_id) {
    return true;
  }
  for (ascir::AxisId from_axis_id : tpipe.tiler.GetAxis(axis_id).from) {
    if (IsAxisDerivedFrom(tpipe, from_axis_id, ancestor_axis_id)) {
      return true;
    }
  }
  return false;
}

af::Status BuildTensorWindowInfo(const ascgen_utils::indirect_load::IndirectLoadTensorLayout &layout,
                                 const Tensor &tensor, size_t axis_pos, LogicalTensorInfo &info) {
  GE_ASSERT_TRUE(layout.axis_ids.size() == layout.sizes.size() && layout.sizes.size() == layout.strides.size(),
                 "IndirectLoad tensor window layout rank mismatch.");
  GE_ASSERT_TRUE(axis_pos < layout.sizes.size(), "IndirectLoad tensor window axis is out of range.");
  GE_ASSERT_TRUE(tensor.vectorized_axis.size() == tensor.vectorized_strides.size(),
                 "IndirectLoad tensor vectorized axis/stride rank mismatch.");
  info = LogicalTensorInfo(layout);
  // Dense layouts keep their logical row-major strides after the local window is built. For a zero-stride-compact
  // view, preserve zero-stride axes but derive non-zero window strides from the physical vectorized tensor view;
  // bitwidth-changing producers can introduce padding between logical elements.
  if (layout.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense) {
    return af::SUCCESS;
  }
  af::Expression compact_stride = af::sym::kSymbolOne;
  for (size_t index = info.sizes.size(); index > axis_pos; --index) {
    const size_t dim = index - 1UL;
    if (af::SymbolicUtils::StaticCheckEq(layout.strides[dim], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      info.strides[dim] = af::sym::kSymbolZero;
      continue;
    }
    const auto vectorized_axis =
        std::find(tensor.vectorized_axis.begin(), tensor.vectorized_axis.end(), layout.axis_ids[dim]);
    if (vectorized_axis != tensor.vectorized_axis.end()) {
      info.strides[dim] =
          tensor
              .vectorized_strides[static_cast<size_t>(std::distance(tensor.vectorized_axis.begin(), vectorized_axis))];
    } else {
      info.strides[dim] = compact_stride;
    }
    if (af::SymbolicUtils::StaticCheckEq(info.strides[dim], af::sym::kSymbolZero) != af::TriBool::kTrue) {
      compact_stride = af::sym::Mul(info.strides[dim], info.sizes[dim]);
    }
  }
  return af::SUCCESS;
}

bool TryBuildStaticSpan(const af::Expression &size_expr, uint64_t stride, uint64_t &span) {
  int64_t size = 0L;
  if (!size_expr.GetConstValue(size) || size < 0L) {
    return false;
  }
  return !ge::MulOverflow(static_cast<uint64_t>(size), stride, span);
}

bool TryAccumulateStaticOffset(const af::Expression &size_expr, const af::Expression &stride_expr, uint64_t &offset) {
  int64_t size = 0L;
  int64_t stride = 0L;
  if (!size_expr.GetConstValue(size) || size <= 0L || !stride_expr.GetConstValue(stride) || stride < 0L) {
    return false;
  }
  uint64_t dim_offset = 0U;
  return !ge::MulOverflow(static_cast<uint64_t>(size - 1L), static_cast<uint64_t>(stride), dim_offset) &&
         !ge::AddOverflow(offset, dim_offset, offset);
}

bool TryGetStaticSpans(const LogicalTensorInfo &input, const LogicalTensorInfo &output, size_t axis,
                       SimtCodegenPlan &plan) {
  uint64_t inner = 1U;
  for (size_t i = axis + 1U; i < output.sizes.size(); ++i) {
    if (!TryBuildStaticSpan(output.sizes[i], inner, inner)) {
      return false;
    }
  }
  int64_t input_stride = 0L;
  if (!input.strides[axis].GetConstValue(input_stride) || input_stride < 0L ||
      !TryBuildStaticSpan(output.sizes[axis], inner, plan.output_axis_span_value) ||
      !TryBuildStaticSpan(input.sizes[axis], static_cast<uint64_t>(input_stride), plan.input_axis_span_value)) {
    return false;
  }
  plan.inner_span_value = inner;
  plan.input_axis_stride_value = static_cast<uint64_t>(input_stride);
  return true;
}

bool TryGetMaxElementOffset(const LogicalTensorInfo &tensor, uint64_t &max_offset) {
  max_offset = 0U;
  for (size_t i = 0U; i < tensor.sizes.size(); ++i) {
    if (!TryAccumulateStaticOffset(tensor.sizes[i], tensor.strides[i], max_offset)) {
      return false;
    }
  }
  return true;
}

bool IsDense(const LogicalTensorInfo &tensor) {
  af::Expression expected_stride = af::ops::One;
  for (size_t i = tensor.sizes.size(); i > 0U; --i) {
    const size_t dim = i - 1U;
    if (af::SymbolicUtils::StaticCheckEq(tensor.strides[dim], expected_stride) != af::TriBool::kTrue) {
      return false;
    }
    expected_stride = af::sym::Mul(expected_stride, tensor.sizes[dim]);
  }
  return true;
}

bool IsStructuredSimt(const LogicalTensorInfo &input, const LogicalTensorInfo &index, const LogicalTensorInfo &output,
                      size_t axis) {
  if (!IsDense(input) || !IsDense(index) || !IsDense(output) || index.sizes.size() != output.sizes.size() ||
      input.sizes.size() != output.sizes.size()) {
    return false;
  }
  for (size_t i = 0U; i < output.sizes.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(index.sizes[i], output.sizes[i]) != af::TriBool::kTrue ||
        (i != axis && af::SymbolicUtils::StaticCheckEq(input.sizes[i], output.sizes[i]) != af::TriBool::kTrue)) {
      return false;
    }
  }
  return true;
}

bool IsPowerOfTwo(uint64_t value) {
  return value != 0U && (value & (value - 1U)) == 0U;
}

bool CanUseUint32Offsets(const LogicalTensorInfo &input, const LogicalTensorInfo &index,
                         const LogicalTensorInfo &output) {
  uint64_t input_max = 0U;
  uint64_t index_max = 0U;
  uint64_t output_max = 0U;
  const uint64_t limit = std::numeric_limits<uint32_t>::max();
  return TryGetMaxElementOffset(input, input_max) && input_max <= limit && TryGetMaxElementOffset(index, index_max) &&
         index_max <= limit && TryGetMaxElementOffset(output, output_max) && output_max <= limit;
}

bool CanUseUint32Divisors(const LogicalTensorInfo &index) {
  for (const af::Expression &size_expr : index.sizes) {
    int64_t size = 0L;
    if (!size_expr.GetConstValue(size) || size < 0L || size > static_cast<int64_t>(INT32_MAX)) {
      return false;
    }
  }
  return true;
}

uint64_t BuildNonZeroStrideMask(const std::vector<ascir::SizeExpr> &strides) {
  uint64_t mask = 0U;
  for (size_t dim = 0U; dim < strides.size(); ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(strides[dim], af::sym::kSymbolZero) != af::TriBool::kTrue) {
      mask |= 1ULL << dim;
    }
  }
  return mask;
}

bool ApplySimtIndexPhysicalStrides(const std::vector<af::AscNodePtr> &index_nodes, LogicalTensorInfo &index) {
  for (const af::AscNodePtr &node : index_nodes) {
    const std::vector<af::Expression> *physical_strides = nullptr;
    if (af::ops::IsOps<af::ascir_op::Broadcast>(node) && !node->inputs().empty() &&
        node->inputs()[0]->attr.strides.size() == index.strides.size()) {
      physical_strides = &node->inputs()[0]->attr.strides;
    } else if (!node->outputs().empty() && node->outputs()[0]->attr.strides.size() == index.strides.size()) {
      physical_strides = &node->outputs()[0]->attr.strides;
    }
    if (physical_strides == nullptr) {
      continue;
    }
    const bool has_zero_stride =
        std::any_of(physical_strides->begin(), physical_strides->end(), [](const af::Expression &stride) {
          return af::SymbolicUtils::StaticCheckEq(stride, af::ops::Zero) == af::TriBool::kTrue;
        });
    if (has_zero_stride) {
      index.strides = *physical_strides;
      return true;
    }
  }
  return false;
}

SimtCodegenPlan BuildSimtCodegenPlan(const LogicalTensorInfo &input, const LogicalTensorInfo &index,
                                     const LogicalTensorInfo &output, size_t axis, bool strided) {
  SimtCodegenPlan plan;
  for (size_t i = axis + 1U; i < output.sizes.size(); ++i) {
    plan.inner_span = af::sym::Mul(plan.inner_span, output.sizes[i]);
  }
  plan.output_axis_span = af::sym::Mul(output.sizes[axis], plan.inner_span);
  plan.input_axis_stride = input.strides[axis];
  plan.input_axis_span = af::sym::Mul(input.sizes[axis], plan.input_axis_stride);
  const bool structured = IsStructuredSimt(input, index, output, axis);
  const bool static_spans = TryGetStaticSpans(input, output, axis, plan);
  if (strided) {
    plan.policy = SimtAddressPolicy::kStrided;
    plan.input_stride_mask = BuildNonZeroStrideMask(input.strides);
    plan.index_stride_mask = BuildNonZeroStrideMask(index.strides);
  } else if (structured) {
    const bool static_power_of_two =
        static_spans && IsPowerOfTwo(plan.inner_span_value) && IsPowerOfTwo(plan.output_axis_span_value);
    const bool static_inner = static_spans && IsPowerOfTwo(plan.inner_span_value);
    const bool magic_divisors_supported =
        !static_spans || (plan.inner_span_value <= static_cast<uint64_t>(INT64_MAX) &&
                          plan.output_axis_span_value <= static_cast<uint64_t>(INT64_MAX));
    plan.policy = static_power_of_two        ? SimtAddressPolicy::kStaticPowerOfTwo
                  : static_inner             ? SimtAddressPolicy::kStaticInner
                  : magic_divisors_supported ? SimtAddressPolicy::kStructuredMagic
                                             : SimtAddressPolicy::kRecursive;
  }
  const bool structured_magic_supported = plan.policy != SimtAddressPolicy::kStructuredMagic ||
                                          (static_spans && plan.inner_span_value <= static_cast<uint64_t>(INT32_MAX) &&
                                           plan.output_axis_span_value <= static_cast<uint64_t>(INT32_MAX));
  const bool static_inner_magic_supported =
      plan.policy != SimtAddressPolicy::kStaticInner || plan.output_axis_span_value <= static_cast<uint64_t>(INT32_MAX);
  const bool general_magic_supported =
      (plan.policy != SimtAddressPolicy::kRecursive && plan.policy != SimtAddressPolicy::kStrided) ||
      CanUseUint32Divisors(index);
  if (CanUseUint32Offsets(input, index, output) && structured_magic_supported && static_inner_magic_supported &&
      general_magic_supported) {
    plan.offset_type = "uint32_t";
  }
  return plan;
}

std::string PromoteSizeExpr(const std::string &expr, const std::string &type) {
  if (type == "uint32_t") {
    return expr;
  }
  const size_t begin = expr.find("t->");
  if (begin == std::string::npos) {
    return "static_cast<" + type + ">(" + expr + ")";
  }
  size_t end = begin + 3U;
  while (end < expr.size() && (std::isalnum(static_cast<unsigned char>(expr[end])) != 0 || expr[end] == '_')) {
    ++end;
  }
  return expr.substr(0U, begin) + "static_cast<" + type + ">(" + expr.substr(begin, end - begin) + ")" +
         expr.substr(end);
}

std::string JoinSizeExprs(const std::vector<ascir::SizeExpr> &exprs, const TPipe &tpipe,
                          const std::string &type = "uint32_t") {
  std::stringstream ss;
  for (size_t i = 0; i < exprs.size(); ++i) {
    if (i > 0) {
      ss << ", ";
    }
    ss << PromoteSizeExpr(tpipe.tiler.Size(exprs[i]), type);
  }
  return ss.str();
}

std::string GetSimtPolicyType(const SimtCodegenPlan &plan, size_t rank, int64_t axis) {
  std::stringstream ss;
  if (plan.policy == SimtAddressPolicy::kStaticPowerOfTwo) {
    ss << "AscendC::IndirectLoadSimtStaticPowerOfTwoPolicy<" << plan.offset_type << ", " << plan.inner_span_value
       << "ULL, " << plan.output_axis_span_value << "ULL, " << plan.input_axis_stride_value << "ULL, "
       << plan.input_axis_span_value << "ULL>";
  } else if (plan.policy == SimtAddressPolicy::kStaticInner) {
    ss << "AscendC::IndirectLoadSimtStaticInnerPolicy<" << plan.offset_type << ", " << plan.inner_span_value << "ULL, "
       << plan.input_axis_stride_value << "ULL, " << plan.input_axis_span_value << "ULL>";
  } else if (plan.policy == SimtAddressPolicy::kStructuredMagic) {
    ss << "AscendC::IndirectLoadSimtStructuredMagicPolicy<" << plan.offset_type << ">";
  } else if (plan.policy == SimtAddressPolicy::kStrided) {
    ss << "AscendC::IndirectLoadSimtStridedPolicy<" << plan.offset_type << ", " << rank << ", " << axis << ", "
       << plan.input_stride_mask << "ULL, " << plan.index_stride_mask << "ULL>";
  } else {
    ss << "AscendC::IndirectLoadSimtRecursivePolicy<" << plan.offset_type << ", " << rank << ", " << axis << ">";
  }
  return ss.str();
}

std::string GetSimtPolicyArgs(const SimtCodegenPlan &plan, const LogicalTensorInfo &input,
                              const LogicalTensorInfo &index, const TPipe &tpipe) {
  if (plan.policy == SimtAddressPolicy::kStaticPowerOfTwo) {
    return "";
  }
  if (plan.policy == SimtAddressPolicy::kStaticInner) {
    return PromoteSizeExpr(tpipe.tiler.Size(plan.output_axis_span), plan.offset_type);
  }
  if (plan.policy == SimtAddressPolicy::kStructuredMagic) {
    return PromoteSizeExpr(tpipe.tiler.Size(plan.inner_span), plan.offset_type) + ", " +
           PromoteSizeExpr(tpipe.tiler.Size(plan.output_axis_span), plan.offset_type) + ", " +
           PromoteSizeExpr(tpipe.tiler.Size(plan.input_axis_stride), plan.offset_type) + ", " +
           PromoteSizeExpr(tpipe.tiler.Size(plan.input_axis_span), plan.offset_type);
  }
  std::string args = JoinSizeExprs(index.sizes, tpipe, plan.offset_type) + ", " +
                     JoinSizeExprs(input.strides, tpipe, plan.offset_type);
  if (plan.policy == SimtAddressPolicy::kStrided) {
    args += ", " + JoinSizeExprs(index.strides, tpipe, plan.offset_type);
  }
  return args;
}

bool FindCurrentAxisVar(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, Axis::Type axis_type,
                        ascir::AxisId ancestor_axis_id, std::string &axis_var) {
  for (ascir::AxisId axis_id : current_axis) {
    const Axis &axis = tpipe.tiler.GetAxis(axis_id);
    if (axis.type == axis_type && IsAxisDerivedFrom(tpipe, axis_id, ancestor_axis_id)) {
      axis_var = axis.Str();
      return true;
    }
  }
  return false;
}

af::Status CheckDenseStrides(const LogicalTensorInfo &tensor, const char *tensor_name) {
  af::Expression expected_stride = af::ops::One;
  for (int64_t i = static_cast<int64_t>(tensor.sizes.size()) - 1; i >= 0; --i) {
    if (af::SymbolicUtils::StaticCheckEq(tensor.sizes[i], af::ops::One) == af::TriBool::kTrue) {
      continue;
    }
    GE_ASSERT_TRUE(af::SymbolicUtils::StaticCheckEq(tensor.strides[i], expected_stride) == af::TriBool::kTrue,
                   "IndirectLoad %s must be dense contiguous.", tensor_name);
    expected_stride = af::sym::Mul(expected_stride, tensor.sizes[i]);
  }
  return af::SUCCESS;
}

af::Status CheckIndirectLoadShape(const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                                  const Tensor &output_tensor) {
  const auto &input = logical_view.input;
  const auto &index = logical_view.index;
  const auto &output = logical_view.output;
  GE_ASSERT_TRUE(input.sizes.size() == index.sizes.size(), "Invalid IndirectLoad logical rank, input:%zu, index:%zu.",
                 input.sizes.size(), index.sizes.size());
  GE_ASSERT_TRUE(index.sizes.size() == output.sizes.size(), "Invalid IndirectLoad logical rank, index:%zu, output:%zu.",
                 index.sizes.size(), output.sizes.size());
  GE_ASSERT_TRUE(input.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported &&
                     index.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported,
                 "IndirectLoad input or index layout is unsupported.");
  GE_ASSERT_TRUE(input.sizes.size() == input.strides.size() && index.sizes.size() == index.strides.size(),
                 "IndirectLoad logical sizes/strides rank mismatch.");
  GE_ASSERT_SUCCESS(CheckDenseStrides(LogicalTensorInfo(output.sizes, output.strides), "output logical view"));
  GE_ASSERT_TRUE(output_tensor.axis_size.size() == output_tensor.axis_strides.size(),
                 "IndirectLoad output tensor sizes/strides rank mismatch.");
  GE_ASSERT_SUCCESS(
      CheckDenseStrides(LogicalTensorInfo(output_tensor.axis_size, output_tensor.axis_strides), "output tensor"));
  return af::SUCCESS;
}

using SimtNodeSet = std::unordered_set<const af::AscNode *>;

af::Status CollectSimtBackwardNodes(const af::AscNodePtr &root, const ascir::NodeView &indirect_load,
                                    SimtNodeSet &nodes) {
  std::vector<af::AscNodePtr> pending = {root};
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr current = pending[cursor];
    if (current == nullptr || current == indirect_load || !nodes.emplace(current.get()).second) {
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Load>(current) || af::ops::IsOps<af::ascir_op::Scalar>(current)) {
      continue;
    }
    GE_ASSERT_TRUE(current->inputs.Size() > 0UL, "IndirectLoad SIMT node[%s] has no input.", current->GetNamePtr());
    for (size_t i = 0UL; i < current->inputs.Size(); ++i) {
      const af::AscNodePtr producer = ascgen_utils::indirect_load::GetInputProducer(current, i);
      GE_ASSERT_NOTNULL(producer, "IndirectLoad SIMT node[%s] input[%zu] has no producer.", current->GetNamePtr(), i);
      pending.emplace_back(producer);
    }
  }
  return af::SUCCESS;
}

af::AscNodePtr FindSimtOutputStore(const ascir::NodeView &indirect_load) {
  for (af::AscNodePtr current = ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load); current != nullptr;
       current = ascgen_utils::indirect_load::GetOnlyOutputConsumer(current)) {
    if (af::ops::IsOps<af::ascir_op::Store>(current)) {
      return current;
    }
  }
  return nullptr;
}

af::Status ValidateSimtRegionNode(const af::AscNodePtr &node) {
  if (af::ops::IsOps<af::ascir_op::Load>(node) || af::ops::IsOps<af::ascir_op::Scalar>(node) ||
      af::ops::IsOps<af::ascir_op::Store>(node)) {
    return af::SUCCESS;
  }
  GE_ASSERT_TRUE(ascgen_utils::indirect_load::GetTemplateRole(node) ==
                     ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform,
                 "IndirectLoad SIMT node[%s] has no inline-transform role.", node->GetNamePtr());
  GE_ASSERT_TRUE(!af::ops::IsOps<af::ascir_op::VectorFunc>(node),
                 "IndirectLoad SIMT transform must use scalar emission, node:%s", node->GetNamePtr());
  return af::SUCCESS;
}

void AppendSimtGmTensor(const af::AscNodePtr &node, std::vector<SimtGmTensor> &gm_tensors) {
  const auto output = node->outputs()[0];
  const auto found = std::find_if(gm_tensors.begin(), gm_tensors.end(), [output](const SimtGmTensor &tensor) {
    return tensor.value_tensor_id == output->attr.mem.tensor_id;
  });
  if (found == gm_tensors.end()) {
    gm_tensors.push_back({output->attr.mem.tensor_id, node->inputs()[0]->attr.mem.tensor_id, output->attr.dtype});
  }
}

af::Status CollectSimtRegionMetadata(const ascir::NodeView &indirect_load, std::vector<af::AscNodePtr> &index_nodes,
                                     std::vector<af::AscNodePtr> &output_nodes, std::vector<SimtGmTensor> &gm_tensors,
                                     af::AscNodePtr &store) {
  const af::AscNodePtr index_root =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
  GE_ASSERT_NOTNULL(index_root, "IndirectLoad SIMT index input has no producer.");
  store = FindSimtOutputStore(indirect_load);
  GE_ASSERT_NOTNULL(store, "IndirectLoad SIMT output chain must end with a Store node.");
  SimtNodeSet index_set;
  SimtNodeSet output_set;
  GE_ASSERT_SUCCESS(CollectSimtBackwardNodes(index_root, indirect_load, index_set));
  GE_ASSERT_SUCCESS(CollectSimtBackwardNodes(store, indirect_load, output_set));
  const auto owner_graph = indirect_load->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph, "IndirectLoad SIMT node has no owner graph.");
  for (const auto &graph_node : owner_graph->GetDirectNode()) {
    const af::AscNodePtr node = std::dynamic_pointer_cast<af::AscNode>(graph_node);
    GE_ASSERT_NOTNULL(node, "IndirectLoad SIMT graph contains invalid node.");
    if (index_set.count(node.get()) != 0UL) {
      GE_ASSERT_SUCCESS(ValidateSimtRegionNode(node));
      index_nodes.emplace_back(node);
    }
    if (output_set.count(node.get()) != 0UL) {
      GE_ASSERT_SUCCESS(ValidateSimtRegionNode(node));
      output_nodes.emplace_back(node);
    }
    if (af::ops::IsOps<af::ascir_op::Load>(node) &&
        (index_set.count(node.get()) != 0UL || output_set.count(node.get()) != 0UL)) {
      AppendSimtGmTensor(node, gm_tensors);
    }
  }
  return af::SUCCESS;
}

af::Status CollectSimtMetadata(const ascir::NodeView &indirect_load, const af::AscNodePtr &root,
                               std::vector<af::AscNodePtr> &nodes, std::vector<SimtGmTensor> &gm_tensors) {
  GE_ASSERT_NOTNULL(root, "IndirectLoad SIMT region root is missing.");
  SimtNodeSet node_set;
  GE_ASSERT_SUCCESS(CollectSimtBackwardNodes(root, indirect_load, node_set));
  const auto owner_graph = indirect_load->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph, "IndirectLoad SIMT node has no owner graph.");
  for (const auto &graph_node : owner_graph->GetDirectNode()) {
    const af::AscNodePtr node = std::dynamic_pointer_cast<af::AscNode>(graph_node);
    if (node == nullptr || node_set.count(node.get()) == 0UL) {
      continue;
    }
    GE_ASSERT_SUCCESS(ValidateSimtRegionNode(node));
    nodes.emplace_back(node);
    if (af::ops::IsOps<af::ascir_op::Load>(node)) {
      AppendSimtGmTensor(node, gm_tensors);
    }
  }
  return af::SUCCESS;
}

af::Status EmitSimtScalarInput(const af::AscNodePtr &node, std::map<ascir::TensorId, std::string> &values,
                               std::stringstream &ss) {
  std::string value;
  GE_ASSERT_NOTNULL(node->attr.ir_attr, "IndirectLoad SIMT Scalar node[%s] has no IR attr.", node->GetNamePtr());
  GE_ASSERT_GRAPH_SUCCESS(node->attr.ir_attr->GetAttrValue("value", value));
  const auto output = node->outputs()[0];
  std::string dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(output->attr.dtype, dtype));
  std::string processed_value;
  GE_ASSERT_SUCCESS(ascgen_utils::ScalarValuePreProcess(value, dtype, processed_value));
  const std::string variable = kSimtValueNamePrefix + std::to_string(output->attr.mem.tensor_id);
  ss << "    " << dtype << " " << variable << " = static_cast<" << dtype << ">(" << processed_value << ");"
     << std::endl;
  values[output->attr.mem.tensor_id] = variable;
  return af::SUCCESS;
}

af::Status EmitSimtTransform(const af::AscNodePtr &node, std::map<ascir::TensorId, std::string> &values,
                             std::stringstream &ss) {
  std::vector<std::string> inputs;
  for (size_t i = 0UL; i < node->inputs.Size(); ++i) {
    const auto found = values.find(node->inputs()[i]->attr.mem.tensor_id);
    GE_ASSERT_TRUE(found != values.end(), "SIMT node[%s] input[%zu] has no scalar value.", node->GetNamePtr(), i);
    inputs.emplace_back(found->second);
  }
  std::string expr;
  GE_ASSERT_SUCCESS(EmitSimtScalarExpr(node, inputs, expr));
  const auto output = node->outputs()[0];
  std::string output_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(output->attr.dtype, output_dtype));
  const std::string variable = kSimtValueNamePrefix + std::to_string(output->attr.mem.tensor_id);
  ss << "    " << output_dtype << " " << variable << " = " << expr << ";" << std::endl;
  values[output->attr.mem.tensor_id] = variable;
  return af::SUCCESS;
}

af::Status GenerateSimtEvaluatorBody(const std::vector<af::AscNodePtr> &nodes,
                                     std::map<ascir::TensorId, std::string> &values, ascir::TensorId result_tensor_id,
                                     std::stringstream &ss, const SimtNodeSet *index_nodes = nullptr) {
  for (const af::AscNodePtr &node : nodes) {
    if (af::ops::IsOps<af::ascir_op::Load>(node)) {
      const auto output = node->outputs()[0];
      const bool is_index_node = index_nodes != nullptr && index_nodes->count(node.get()) != 0UL;
      const char *offset = is_index_node ? "index_offset" : "output_index";
      values[output->attr.mem.tensor_id] = "context." + std::string(kSimtGmFieldNamePrefix) +
                                           std::to_string(output->attr.mem.tensor_id) + "[" + offset + "]";
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Scalar>(node)) {
      GE_ASSERT_SUCCESS(EmitSimtScalarInput(node, values, ss));
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Store>(node)) {
      continue;
    }
    GE_ASSERT_SUCCESS(EmitSimtTransform(node, values, ss));
  }
  const auto result = values.find(result_tensor_id);
  GE_ASSERT_TRUE(result != values.end(), "SIMT evaluator result tensor[%ld] has no scalar value.", result_tensor_id);
  ss << "    return " << result->second << ";" << std::endl;
  ss << "  }" << std::endl;
  return af::SUCCESS;
}

af::Status GenSimtIndexEvaluator(const std::string &index_dtype, const std::string &offset_type,
                                 ascir::TensorId result_tensor_id, const std::vector<af::AscNodePtr> &nodes,
                                 std::stringstream &ss) {
  std::map<ascir::TensorId, std::string> values;
  ss << "  __simt_callee__ __aicore__ inline static " << index_dtype << " Index(" << offset_type
     << " output_index, const Context &context) {" << std::endl;
  return GenerateSimtEvaluatorBody(nodes, values, result_tensor_id, ss);
}

af::Status GenSimtOutputEvaluator(const std::string &output_dtype, const std::string &input_dtype,
                                  const std::string &offset_type, ascir::TensorId value_tensor_id,
                                  ascir::TensorId result_tensor_id, const std::vector<af::AscNodePtr> &nodes,
                                  const SimtNodeSet &index_nodes, std::stringstream &ss) {
  std::map<ascir::TensorId, std::string> values{{value_tensor_id, "value"}};
  ss << "  __simt_callee__ __aicore__ inline static " << output_dtype << " Output(" << input_dtype << " value, "
     << offset_type << " output_index, " << offset_type << " index_offset, const Context &context) {" << std::endl;
  return GenerateSimtEvaluatorBody(nodes, values, result_tensor_id, ss, &index_nodes);
}

af::Status CalcVectorizedElementCount(const Tensor &tensor, af::Expression &element_count) {
  element_count = af::ops::One;
  for (uint32_t axis_pos : tensor.vectorized_axis_pos) {
    GE_ASSERT_TRUE(axis_pos < tensor.axis_size.size(), "IndirectLoad SIMT output axis is invalid.");
    element_count = af::sym::Mul(element_count, tensor.axis_size[axis_pos]);
  }
  return af::SUCCESS;
}

af::Status GenerateSimtContextInitializer(const std::string &context_name, const std::vector<SimtGmTensor> &gm_tensors,
                                          std::stringstream &ss) {
  ss << "  " << context_name << " context{";
  for (size_t i = 0UL; i < gm_tensors.size(); ++i) {
    const SimtGmTensor &gm_tensor = gm_tensors[i];
    std::string dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(gm_tensor.dtype, dtype));
    ss << (i == 0UL ? "" : ", ") << "(__gm__ " << dtype << " *)" << kGlobalTensorNamePrefix << gm_tensor.gm_tensor_id
       << ".GetPhyAddr()";
  }
  ss << "};" << std::endl;
  return af::SUCCESS;
}

}  // namespace

Status IndirectLoadRegApiCall::ParseAttr(const ascir::NodeView &node) {
  int64_t axis = 0;
  GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("axis", axis),
                          "Failed to get IndirectLoad axis attr, node = %s", node->GetNamePtr());
  ascgen_utils::indirect_load::TemplateAxes template_axes;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateAxes(node, template_axes));
  outer_axis_ = template_axes.outer_axis;
  const af::AscNodePtr post_reduce = ascgen_utils::indirect_load::GetPostReduceConsumer(node);
  has_post_reduce_ = post_reduce != nullptr;
  template_id_ = ::ascir::GetTemplateIdOrDefault(*node);
  GE_ASSERT_TRUE(
      template_id_ == ascir::TemplateId::kIndirectLoadSK || template_id_ == ascir::TemplateId::kIndirectLoadSimd ||
          template_id_ == ascir::TemplateId::kIndirectLoadSimt,
      "IndirectLoad node[%s] has invalid template id[%d].", node->GetNamePtr(), static_cast<int32_t>(template_id_));
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateLogicalView(node, logical_view_));
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetImplementation(node, implementation_));
  const int64_t rank = static_cast<int64_t>(logical_view_.input.sizes.size());
  GE_ASSERT_TRUE(axis >= -rank && axis < rank, "IndirectLoad axis is out of range.");
  axis_ = axis < 0L ? axis + rank : axis;
  if (template_id_ == ascir::TemplateId::kIndirectLoadSimt) {
    GE_ASSERT_SUCCESS(ParseSimtAttr(node));
  }
  GELOGI("[IndirectLoad] Parse codegen attrs for node[%s], axis[%ld], template_id[%d].", node->GetNamePtr(), axis_,
         static_cast<int32_t>(template_id_));
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::ParseSimtAttr(const ascir::NodeView &node) {
  GE_ASSERT_TRUE(node->inputs.Size() == kIndirectLoadInputCount, "Invalid IndirectLoad SIMT input number:%zu.",
                 node->inputs.Size());
  GE_ASSERT_TRUE(node->outputs().size() == kIndirectLoadOutputCount, "Invalid IndirectLoad SIMT output number:%zu.",
                 node->outputs().size());
  GE_ASSERT_TRUE(logical_view_.input.sizes.size() < 64UL, "IndirectLoad SIMT rank must be smaller than 64.");
  const auto node_inputs = node->inputs();
  index_result_tensor_id_ = node_inputs[ascgen_utils::indirect_load::kIndexTensorIndex]->attr.mem.tensor_id;
  index_dtype_ = node_inputs[ascgen_utils::indirect_load::kIndexTensorIndex]->attr.dtype;
  simt_value_tensor_id_ = node->outputs()[0]->attr.mem.tensor_id;
  if (has_post_reduce_) {
    const af::AscNodePtr output_root = ascgen_utils::indirect_load::GetPostReduceInputProducer(node);
    GE_ASSERT_NOTNULL(output_root, "IndirectLoad post Reduce input has no producer.");
    GE_ASSERT_TRUE(!output_root->outputs().empty(), "IndirectLoad post Reduce input producer has no output.");
    output_result_tensor_id_ = output_root->outputs()[0]->attr.mem.tensor_id;
    output_dtype_ = output_root->outputs()[0]->attr.dtype;
    const af::AscNodePtr index_root =
        ascgen_utils::indirect_load::GetInputProducer(node, ascgen_utils::indirect_load::kIndexTensorIndex);
    GE_ASSERT_SUCCESS(CollectSimtMetadata(node, index_root, index_nodes_, simt_gm_tensors_));
    GE_ASSERT_SUCCESS(CollectSimtMetadata(node, output_root, output_nodes_, simt_gm_tensors_));
    GE_ASSERT_TRUE(outputs.size() == kIndirectLoadOutputCount, "Invalid IndirectLoad SIMT output number:%zu.",
                   outputs.size());
    outputs[0].id = output_result_tensor_id_;
    return af::SUCCESS;
  }
  af::AscNodePtr store;
  GE_ASSERT_SUCCESS(CollectSimtRegionMetadata(node, index_nodes_, output_nodes_, simt_gm_tensors_, store));
  GE_ASSERT_TRUE(store->inputs.Size() == 1UL, "Invalid IndirectLoad SIMT Store input number:%zu.",
                 store->inputs.Size());
  GE_ASSERT_TRUE(store->outputs().size() == 1UL, "Invalid IndirectLoad SIMT Store output number:%zu.",
                 store->outputs().size());
  output_result_tensor_id_ = store->inputs()[0]->attr.mem.tensor_id;
  output_gm_tensor_ = kGlobalTensorNamePrefix + std::to_string(store->outputs()[0]->attr.mem.tensor_id);
  output_dtype_ = store->outputs()[0]->attr.dtype;
  GE_ASSERT_TRUE(store->inputs()[0]->attr.dtype == output_dtype_,
                 "IndirectLoad SIMT output transform dtype[%d] does not match Store dtype[%d].",
                 static_cast<int32_t>(store->inputs()[0]->attr.dtype), static_cast<int32_t>(output_dtype_));
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::GenerateFuncDefinition(const TPipe &tpipe, const Tiler &tiler,
                                                      std::stringstream &ss) const {
  (void)tiler;
  if (template_id_ != ascir::TemplateId::kIndirectLoadSimt) {
    return af::SUCCESS;
  }

  GE_ASSERT_TRUE(inputs.size() == kIndirectLoadInputCount, "Invalid IndirectLoad SIMT input number:%zu.",
                 inputs.size());
  const Tensor *input_tensor = tpipe.GetTensor(inputs[ascgen_utils::indirect_load::kInputTensorIndex]->id);
  GE_ASSERT_NOTNULL(input_tensor, "IndirectLoad SIMT input tensor is missing.");
  const LogicalTensorInfo input_info(logical_view_.input);
  LogicalTensorInfo index_info(logical_view_.index);
  const LogicalTensorInfo output_info(logical_view_.output);
  const bool index_broadcast_strided = ApplySimtIndexPhysicalStrides(index_nodes_, index_info);
  const bool strided = logical_view_.input.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense ||
                       logical_view_.index.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense ||
                       index_broadcast_strided;
  const SimtCodegenPlan plan =
      BuildSimtCodegenPlan(input_info, index_info, output_info, static_cast<size_t>(axis_), strided);
  std::string input_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(input_tensor->dtype, input_dtype));
  const std::string valid_node_name = ascgen_utils::GenValidName(node_name);
  const std::string context_name = kSimtContextNamePrefix + valid_node_name;
  const std::string body_name = kSimtBodyNamePrefix + valid_node_name;
  std::string index_dtype;
  std::string output_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(index_dtype_, index_dtype));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(output_dtype_, output_dtype));
  GELOGI(
      "[IndirectLoad] Generate SIMT body for node[%s], rank[%zu], axis[%ld], index_nodes[%zu], output_nodes[%zu], "
      "gm_inputs[%zu].",
      node_name.c_str(), input_info.sizes.size(), axis_, index_nodes_.size(), output_nodes_.size(),
      simt_gm_tensors_.size());

  ss << "struct " << context_name << " {" << std::endl;
  for (const SimtGmTensor &tensor : simt_gm_tensors_) {
    std::string dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(tensor.dtype, dtype));
    ss << "  __gm__ " << dtype << " *" << kSimtGmFieldNamePrefix << tensor.value_tensor_id << ";" << std::endl;
  }
  ss << "};" << std::endl;
  ss << "struct " << body_name << " {" << std::endl;
  ss << "  using Context = " << context_name << ";" << std::endl;
  SimtNodeSet index_node_set;
  for (const af::AscNodePtr &node : index_nodes_) {
    index_node_set.insert(node.get());
  }
  GE_ASSERT_SUCCESS(GenSimtIndexEvaluator(index_dtype, plan.offset_type, index_result_tensor_id_, index_nodes_, ss));
  GE_ASSERT_SUCCESS(GenSimtOutputEvaluator(output_dtype, input_dtype, plan.offset_type, simt_value_tensor_id_,
                                           output_result_tensor_id_, output_nodes_, index_node_set, ss));
  ss << "};" << std::endl;
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                        const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                        const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                        std::string &result) const {
  GE_ASSERT_TRUE(inputs.size() == kIndirectLoadInputCount, "Invalid IndirectLoad input number:%zu.", inputs.size());
  GE_ASSERT_TRUE(outputs.size() == kIndirectLoadOutputCount, "Invalid IndirectLoad output number:%zu.", outputs.size());
  GE_ASSERT_TRUE(
      template_id_ == ascir::TemplateId::kIndirectLoadSK || template_id_ == ascir::TemplateId::kIndirectLoadSimd,
      "IndirectLoad tensor-based Generate only supports SK and SIMD.");
  GE_ASSERT_SUCCESS(CheckIndirectLoadShape(logical_view_, outputs[0].get()));
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs);
  if (template_id_ == ascir::TemplateId::kIndirectLoadSK) {
    GELOGI("[IndirectLoad] Generate SK API body for node[%s].", node_name.c_str());
    return GenerateSk(tpipe, current_axis, inputs, outputs, result);
  } else if (template_id_ == ascir::TemplateId::kIndirectLoadSimd) {
    GELOGI("[IndirectLoad] Generate SIMD API body for node[%s].", node_name.c_str());
    return GenerateSimd(tpipe, current_axis, inputs, outputs, result);
  } else {
    GE_ASSERT_TRUE(false, "IndirectLoad tensor-based Generate only supports SK and SIMD.");
  }
}

Status IndirectLoadRegApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                        std::string &result) const {
  if (template_id_ != ascir::TemplateId::kIndirectLoadSimt) {
    return ApiCall::Generate(tpipe, current_axis, result);
  }
  GE_ASSERT_TRUE(inputs.size() == kIndirectLoadInputCount, "Invalid IndirectLoad SIMT input number:%zu.",
                 inputs.size());
  const Tensor *input_tensor = tpipe.GetTensor(inputs[ascgen_utils::indirect_load::kInputTensorIndex]->id);
  GE_ASSERT_NOTNULL(input_tensor, "IndirectLoad SIMT input tensor is missing.");
  return GenerateSimt(tpipe, current_axis, *input_tensor, result);
}

Status IndirectLoadRegApiCall::GenerateSk(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                          const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                          const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                          std::string &result) const {
  const Tensor &x = inputs[0].get();
  const Tensor &index = inputs[1].get();
  const Tensor &y = outputs[0].get();
  const auto tmp_iter = tmp_buf_id.find(-1L);
  GE_ASSERT_TRUE(tmp_iter != tmp_buf_id.end(), "IndirectLoad SK requires an API-level tmp buffer.");

  const size_t axis_pos = static_cast<size_t>(axis_);
  const LogicalTensorInfo x_info(logical_view_.input);
  LogicalTensorInfo index_info;
  GE_ASSERT_SUCCESS(BuildTensorWindowInfo(logical_view_.index, index, axis_pos, index_info));
  std::string x_dtype_name;
  std::string index_dtype_name;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(x.dtype, x_dtype_name));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(index.dtype, index_dtype_name));
  GE_ASSERT_TRUE(x.dtype == y.dtype, "IndirectLoad SK input/output dtype must match.");
  GELOGD("[IndirectLoad] Generate SK body for node[%s], rank[%zu], axis[%zu].", node_name.c_str(), x_info.sizes.size(),
         axis_pos);

  std::stringstream ss;
  ss << "// IndirectLoad SK" << std::endl;
  ss << "{" << std::endl;
  ss << "  AscendC::IndirectLoadSk<" << x_dtype_name << ", " << index_dtype_name << ", " << x_info.sizes.size() << ", "
     << axis_ << ">(" << std::endl;
  ss << "      " << x << ", " << index << ", " << y << ", " << tpipe.tmp_buf.name << "_" << tmp_iter->second << ", "
     << y.actual_size << ", " << tpipe.tiler.Offset(current_axis, y.axis, y.axis_strides) << ", "
     << tpipe.tiler.Size(x_info.sizes[axis_pos]) << ", " << JoinSizeExprs(index_info.sizes, tpipe) << ", "
     << JoinSizeExprs(x_info.strides, tpipe) << ", " << JoinSizeExprs(index_info.strides, tpipe) << ");" << std::endl;
  ss << "}" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::GenerateSimd(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                            const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                            const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                            std::string &result) const {
  const Tensor &input = inputs[ascgen_utils::indirect_load::kInputTensorIndex].get();
  const Tensor &index = inputs[ascgen_utils::indirect_load::kIndexTensorIndex].get();
  const Tensor &output = outputs[0].get();
  const size_t axis_pos = static_cast<size_t>(axis_);
  LogicalTensorInfo input_info;
  LogicalTensorInfo index_info;
  GE_ASSERT_SUCCESS(BuildTensorWindowInfo(logical_view_.input, input, axis_pos, input_info));
  GE_ASSERT_SUCCESS(BuildTensorWindowInfo(logical_view_.index, index, axis_pos, index_info));
  const bool requires_strided_api =
      logical_view_.input.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense ||
      logical_view_.index.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense;
  const auto tmp_iter = tmp_buf_id.find(-1L);
  if (requires_strided_api) {
    GE_ASSERT_TRUE(tmp_iter != tmp_buf_id.end(), "IndirectLoad SIMD requires an API-level tmp buffer.");
  }

  std::string input_dtype;
  std::string index_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(input.dtype, input_dtype));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(index.dtype, index_dtype));
  GE_ASSERT_TRUE(input.dtype == output.dtype,
                 "IndirectLoad SIMD input/output dtype must match after input preprocess.");
  GELOGD("[IndirectLoad] Generate SIMD body for node[%s], rank[%zu], axis[%zu].", node_name.c_str(),
         input_info.sizes.size(), axis_pos);

  std::stringstream ss;
  ss << "// IndirectLoad SIMD" << std::endl;
  ss << "{" << std::endl;
  const char *api = !requires_strided_api && implementation_ == ascgen_utils::indirect_load::Implementation::kGatherApi
                        ? "IndirectLoadSimdGatherApi"
                        : "IndirectLoadSimd";
  ss << "  AscendC::" << api << "<" << input_dtype << ", " << index_dtype << ", " << input_info.sizes.size() << ", "
     << axis_ << ">(" << std::endl;
  ss << "      " << input << ", " << index << ", " << output << ", ";
  if (requires_strided_api) {
    ss << tpipe.tmp_buf.name << "_" << tmp_iter->second << ", " << output.actual_size << ", "
       << tpipe.tiler.Offset(current_axis, output.axis, output.axis_strides) << ", "
       << JoinSizeExprs(index_info.sizes, tpipe) << ", " << JoinSizeExprs(input_info.strides, tpipe) << ", "
       << JoinSizeExprs(index_info.strides, tpipe);
  } else {
    ss << output.actual_size << ", " << tpipe.tiler.Offset(current_axis, output.axis, output.axis_strides) << ", "
       << input.actual_size << ", " << tpipe.tiler.Size(input_info.sizes[axis_pos]) << ", "
       << JoinSizeExprs(index_info.sizes, tpipe) << ", " << JoinSizeExprs(input_info.strides, tpipe);
  }
  ss << ");" << std::endl;
  ss << "}" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::GenerateSimtInvocation(const TPipe &tpipe, const std::string &input_dtype,
                                                      const std::string &output_dtype, const std::string &outer_tb_var,
                                                      std::stringstream &ss) const {
  const LogicalTensorInfo input_info(logical_view_.input);
  LogicalTensorInfo index_info(logical_view_.index);
  const LogicalTensorInfo output_info(logical_view_.output);
  const bool index_broadcast_strided = ApplySimtIndexPhysicalStrides(index_nodes_, index_info);
  const bool strided = logical_view_.input.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense ||
                       logical_view_.index.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense ||
                       index_broadcast_strided;
  const SimtCodegenPlan plan =
      BuildSimtCodegenPlan(input_info, index_info, output_info, static_cast<size_t>(axis_), strided);
  const std::string body_name = kSimtBodyNamePrefix + ascgen_utils::GenValidName(node_name);
  const Tensor *output_tensor = has_post_reduce_ ? tpipe.GetTensor(output_result_tensor_id_) : nullptr;
  af::Expression output_element_count = af::ops::One;
  if (has_post_reduce_) {
    GE_ASSERT_NOTNULL(output_tensor, "IndirectLoad SIMT UB output tensor is missing.");
    GE_ASSERT_SUCCESS(CalcVectorizedElementCount(*output_tensor, output_element_count));
  }
  ss << "  AscendC::IndirectLoadSimt<" << input_dtype << ", " << output_dtype << ", ";
  ss << body_name << ", " << GetSimtPolicyType(plan, input_info.sizes.size(), axis_);
  ss << ">(" << std::endl;
  if (has_post_reduce_) {
    const std::string output_elements = tpipe.tiler.Size(output_element_count);
    ss << "      input_ptr, " << *output_tensor << ", context, static_cast<uint32_t>(" << output_elements << "), ";
    ss << "(static_cast<" << plan.offset_type << ">(block_dim_offset) + static_cast<" << plan.offset_type << ">(";
    ss << outer_tb_var << ")) * " << PromoteSizeExpr(output_elements, plan.offset_type);
  } else {
    ss << "      input_ptr, y_ptr, context, static_cast<uint32_t>(" << outer_tb_var << "_loop_size), ";
    ss << "static_cast<" << plan.offset_type << ">(block_dim_offset)";
  }
  const std::string policy_args = GetSimtPolicyArgs(plan, input_info, index_info, tpipe);
  if (!policy_args.empty()) {
    ss << ", " << policy_args;
  }
  ss << ");" << std::endl;
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::GenerateSimt(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                            const Tensor &input, std::string &result) const {
  GE_ASSERT_TRUE(logical_view_.input.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported &&
                     logical_view_.index.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported,
                 "IndirectLoad SIMT input or index layout is unsupported.");
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ValidateIndirectLoadOutputLayout(logical_view_.output));
  GELOGD("[IndirectLoad] Generate SIMT body for node[%s], output_gm[%s].", node_name.c_str(),
         output_gm_tensor_.c_str());

  std::string outer_tb_var;
  const bool has_outer_tb =
      FindCurrentAxisVar(tpipe, current_axis, Axis::Type::kAxisTypeBlockInner, outer_axis_, outer_tb_var);
  GE_ASSERT_TRUE(has_outer_tb, "IndirectLoad SIMT current axes must contain the output block-inner axis.");
  std::string input_dtype, output_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(input.dtype, input_dtype));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(output_dtype_, output_dtype));
  const std::string valid_node_name = ascgen_utils::GenValidName(node_name);
  const std::string context_name = kSimtContextNamePrefix + valid_node_name;
  std::stringstream ss;
  ss << "// IndirectLoad SIMT" << std::endl;
  ss << "{" << std::endl;
  ss << "  __gm__ " << input_dtype << " *input_ptr = (__gm__ " << input_dtype << " *)" << input << ".GetPhyAddr();"
     << std::endl;
  if (!has_post_reduce_) {
    ss << "  __gm__ " << output_dtype << " *y_ptr = (__gm__ " << output_dtype << " *)" << output_gm_tensor_
       << ".GetPhyAddr();" << std::endl;
  }
  GE_ASSERT_SUCCESS(GenerateSimtContextInitializer(context_name, simt_gm_tensors_, ss));
  GE_ASSERT_SUCCESS(GenerateSimtInvocation(tpipe, input_dtype, output_dtype, outer_tb_var, ss));
  ss << "}" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<IndirectLoadRegApiCall> register_indirect_load_reg_api_call("IndirectLoadRegApiCall");
}  // namespace codegen
