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
#include <unordered_map>
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

Status GenerateSimtContextDefinition(const std::string &context_name, const std::vector<SimtGmTensor> &gm_tensors,
                                     std::stringstream &ss) {
  ss << "struct " << context_name << " {" << std::endl;
  for (const SimtGmTensor &tensor : gm_tensors) {
    std::string dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(tensor.dtype, dtype));
    if (tensor.is_scalar) {
      ss << "  " << dtype << " " << kSimtValueNamePrefix << tensor.value_tensor_id << ";" << std::endl;
    } else {
      ss << "  __gm__ " << dtype << " *" << kSimtGmFieldNamePrefix << tensor.value_tensor_id << ";" << std::endl;
    }
  }
  ss << "};" << std::endl;
  return af::SUCCESS;
}

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

SimtCodegenPlan BuildSimtCodegenPlan(const LogicalTensorInfo &input, const LogicalTensorInfo &index,
                                     const LogicalTensorInfo &output, size_t axis, bool strided,
                                     bool embedding_structured) {
  SimtCodegenPlan plan;
  for (size_t i = axis + 1U; i < output.sizes.size(); ++i) {
    plan.inner_span = af::sym::Mul(plan.inner_span, output.sizes[i]);
  }
  plan.output_axis_span = af::sym::Mul(output.sizes[axis], plan.inner_span);
  plan.input_axis_stride = input.strides[axis];
  plan.input_axis_span = af::sym::Mul(input.sizes[axis], plan.input_axis_stride);
  const bool structured = IsStructuredSimt(input, index, output, axis);
  const bool static_spans = TryGetStaticSpans(input, output, axis, plan);
  if (embedding_structured) {
    plan.policy = SimtAddressPolicy::kEmbedding;
    plan.input_stride_mask = BuildNonZeroStrideMask(input.strides);
    plan.index_stride_mask = BuildNonZeroStrideMask(index.strides);
  } else if (strided) {
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
      (plan.policy != SimtAddressPolicy::kRecursive && plan.policy != SimtAddressPolicy::kStrided &&
       plan.policy != SimtAddressPolicy::kEmbedding) ||
      CanUseUint32Divisors(index);
  if (CanUseUint32Offsets(input, index, output) && structured_magic_supported && static_inner_magic_supported &&
      general_magic_supported) {
    plan.offset_type = "uint32_t";
  }
  return plan;
}

// The scheduler's logical index view may be dense along a dimension whose physical source is a
// broadcast (zero stride), for example where(…, x, broadcast(load)) index chains.  The address
// policy and the Index() evaluator both consume physical strides, so recover them from the index
// producer chain before building the plan.
bool SimtPhysicalStridesEqual(const std::vector<af::Expression> &lhs, const std::vector<af::Expression> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  return std::equal(lhs.begin(), lhs.end(), rhs.begin(), [](const af::Expression &left, const af::Expression &right) {
    return af::SymbolicUtils::StaticCheckEq(left, right) == af::TriBool::kTrue;
  });
}

bool SimtHasZeroStride(const std::vector<af::Expression> &strides) {
  return std::any_of(strides.begin(), strides.end(), [](const af::Expression &stride) {
    return af::SymbolicUtils::StaticCheckEq(stride, af::ops::Zero) == af::TriBool::kTrue;
  });
}

bool ApplySimtIndexPhysicalStrides(const std::vector<af::AscNodePtr> &index_nodes, LogicalTensorInfo &index,
                                   bool &mixed_views) {
  mixed_views = false;
  const std::vector<af::Expression> *canonical_strides = nullptr;
  bool has_zero_stride = false;

  // Load outputs are the physical address views consumed by the evaluator.  Comparing all
  // reachable Loads catches a mixed Where (broadcast branch + dense branch) without treating
  // the Broadcast node's logical output as a second physical view of the same load.
  for (const af::AscNodePtr &node : index_nodes) {
    if (!af::ops::IsOps<af::ascir_op::Load>(node) || node->outputs().empty() ||
        node->outputs()[0]->attr.strides.size() != index.strides.size()) {
      continue;
    }
    const auto &strides = node->outputs()[0]->attr.strides;
    if (canonical_strides == nullptr) {
      canonical_strides = &strides;
    } else if (!SimtPhysicalStridesEqual(*canonical_strides, strides)) {
      mixed_views = true;
    }
    has_zero_stride = has_zero_stride || SimtHasZeroStride(strides);
  }

  if (canonical_strides == nullptr) {
    // Preserve the old producer-chain fallback for graphs whose index region has no Load
    // boundary (for example a synthetic Scalar/Broadcast-only test graph).
    for (const af::AscNodePtr &node : index_nodes) {
      const std::vector<af::Expression> *physical_strides = nullptr;
      if (af::ops::IsOps<af::ascir_op::Broadcast>(node) && !node->inputs().empty() &&
          node->inputs()[0]->attr.strides.size() == index.strides.size()) {
        physical_strides = &node->inputs()[0]->attr.strides;
      } else if (!node->outputs().empty() && node->outputs()[0]->attr.strides.size() == index.strides.size()) {
        physical_strides = &node->outputs()[0]->attr.strides;
      }
      if (physical_strides != nullptr && SimtHasZeroStride(*physical_strides)) {
        canonical_strides = physical_strides;
        has_zero_stride = true;
        break;
      }
    }
  }

  if (canonical_strides != nullptr && has_zero_stride && !mixed_views) {
    index.strides = *canonical_strides;
  }
  return has_zero_stride || mixed_views;
}

SimtCodegenPlan BuildSimtCodegenPlanForNode(const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                                            const std::vector<af::AscNodePtr> &index_nodes, size_t axis,
                                            LogicalTensorInfo &input_info, LogicalTensorInfo &index_info,
                                            bool embedding_structured) {
  input_info = LogicalTensorInfo(logical_view.input);
  index_info = LogicalTensorInfo(logical_view.index);
  const LogicalTensorInfo output_info(logical_view.output);
  bool mixed_index_views = false;
  const bool index_broadcast_strided = ApplySimtIndexPhysicalStrides(index_nodes, index_info, mixed_index_views);
  const bool strided = logical_view.input.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense ||
                       logical_view.index.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense ||
                       index_broadcast_strided;
  SimtCodegenPlan plan = BuildSimtCodegenPlan(input_info, index_info, output_info, axis, strided, embedding_structured);
  plan.use_per_load_index_offsets = mixed_index_views;
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

af::Status BuildSimtPerLoadIndexOffsetExpressions(const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                                                  const std::vector<af::AscNodePtr> &index_nodes, const TPipe &tpipe,
                                                  const std::string &offset_type,
                                                  SimtLoadIndexOffsetExpressions &expressions) {
  expressions.clear();
  const size_t rank = logical_view.output.sizes.size();
  GE_ASSERT_TRUE(logical_view.output.strides.size() == rank, "SIMT output view rank mismatch for index offsets.");
  for (const af::AscNodePtr &node : index_nodes) {
    if (!af::ops::IsOps<af::ascir_op::Load>(node)) {
      continue;
    }
    GE_ASSERT_TRUE(!node->outputs().empty(), "SIMT index Load[%s] has no output.", node->GetNamePtr());
    const auto &attr = node->outputs()[0]->attr;
    GE_ASSERT_TRUE(attr.repeats.size() == rank && attr.strides.size() == rank,
                   "SIMT index Load[%s] physical view rank mismatch.", node->GetNamePtr());

    std::string offset = "0";
    for (size_t dim = 0UL; dim < rank; ++dim) {
      if (af::SymbolicUtils::StaticCheckEq(attr.strides[dim], af::ops::Zero) == af::TriBool::kTrue ||
          af::SymbolicUtils::StaticCheckEq(attr.repeats[dim], af::ops::One) == af::TriBool::kTrue) {
        continue;
      }
      const std::string output_stride =
          PromoteSizeExpr(tpipe.tiler.Size(logical_view.output.strides[dim]), offset_type);
      const std::string output_size = PromoteSizeExpr(tpipe.tiler.Size(logical_view.output.sizes[dim]), offset_type);
      const std::string load_stride = PromoteSizeExpr(tpipe.tiler.Size(attr.strides[dim]), offset_type);
      const std::string coordinate = "((output_index / " + output_stride + ") % " + output_size + ")";
      offset = "(" + offset + " + static_cast<" + offset_type + ">(" + coordinate + ") * " + load_stride + ")";
    }
    expressions[node.get()] = offset;
  }
  return af::SUCCESS;
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
  } else if (plan.policy == SimtAddressPolicy::kEmbedding) {
    ss << "AscendC::IndirectLoadSimtEmbeddingPolicy<" << plan.offset_type;
    if (rank != 2U || axis != 0L) {
      ss << ", " << rank << ", " << axis << ", " << plan.input_stride_mask << "ULL, " << plan.index_stride_mask
         << "ULL";
    }
    ss << ">";
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
  if (plan.policy == SimtAddressPolicy::kEmbedding) {
    return JoinSizeExprs(index.sizes, tpipe, plan.offset_type) + ", " +
           JoinSizeExprs(input.strides, tpipe, plan.offset_type) + ", " +
           JoinSizeExprs(index.strides, tpipe, plan.offset_type);
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

bool SimtLoadUsesZeroOffset(const af::AscNodePtr &node);

bool SimtLoadViewsMatch(const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) {
  if (lhs == nullptr || rhs == nullptr || lhs->outputs().empty() || rhs->outputs().empty()) {
    return false;
  }
  const auto &lhs_attr = lhs->outputs()[0]->attr;
  const auto &rhs_attr = rhs->outputs()[0]->attr;
  if (lhs_attr.repeats.empty() || lhs_attr.strides.empty() || lhs_attr.repeats.size() != rhs_attr.repeats.size() ||
      lhs_attr.strides.size() != rhs_attr.strides.size()) {
    return false;
  }
  for (size_t dim = 0UL; dim < lhs_attr.repeats.size(); ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(lhs_attr.repeats[dim], rhs_attr.repeats[dim]) != af::TriBool::kTrue ||
        af::SymbolicUtils::StaticCheckEq(lhs_attr.strides[dim], rhs_attr.strides[dim]) != af::TriBool::kTrue) {
      return false;
    }
  }
  return true;
}

void BuildSimtLoadAddressSources(const std::vector<af::AscNodePtr> &index_nodes,
                                 const std::vector<SimtOutputChain> &output_chains, SimtLoadAddressSources &sources) {
  // Resolve Load address bases once with the graph metadata; evaluator emission only consumes this classification.
  sources.clear();
  std::vector<af::AscNodePtr> index_loads;
  for (const af::AscNodePtr &node : index_nodes) {
    if (!af::ops::IsOps<af::ascir_op::Load>(node)) {
      continue;
    }
    index_loads.emplace_back(node);
    sources[node.get()] =
        SimtLoadUsesZeroOffset(node) ? SimtLoadAddressSource::kZeroOffset : SimtLoadAddressSource::kIndexOffset;
  }
  for (const SimtOutputChain &chain : output_chains) {
    for (const af::AscNodePtr &node : chain.nodes) {
      if (!af::ops::IsOps<af::ascir_op::Load>(node) || sources.count(node.get()) != 0UL) {
        continue;
      }
      SimtLoadAddressSource source = SimtLoadAddressSource::kOutputOffset;
      if (SimtLoadUsesZeroOffset(node)) {
        source = SimtLoadAddressSource::kZeroOffset;
      } else if (std::any_of(index_loads.begin(), index_loads.end(), [&node](const af::AscNodePtr &index_load) {
                   return SimtLoadViewsMatch(node, index_load);
                 })) {
        source = SimtLoadAddressSource::kIndexOffset;
      }
      sources[node.get()] = source;
    }
  }
}

struct SimtOutputChainInfo {
  SimtOutputChain chain;
  size_t node_set_index = 0UL;
};

using SimtChainIndexMap = std::unordered_map<const af::AscNode *, std::vector<size_t>>;

size_t GetProducerOutputIndex(const af::AscNodePtr &consumer) {
  const auto input_anchor = consumer == nullptr ? nullptr : consumer->GetInDataAnchor(0UL);
  const auto peer_anchor = input_anchor == nullptr ? nullptr : input_anchor->GetPeerOutAnchor();
  return peer_anchor == nullptr ? 0UL : static_cast<size_t>(peer_anchor->GetIdx());
}

af::Status CollectSimtBackwardNodes(const af::AscNodePtr &root, const ascir::NodeView &indirect_load,
                                    SimtNodeSet &nodes) {
  std::vector<af::AscNodePtr> pending = {root};
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr current = pending[cursor];
    if (current == nullptr || current == indirect_load || !nodes.emplace(current.get()).second) {
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Load>(current) || af::ops::IsOps<af::ascir_op::Scalar>(current) ||
        af::ops::IsOps<af::ascir_op::ScalarData>(current)) {
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

af::Status CollectSimtOutputChainInfos(const ascir::NodeView &indirect_load,
                                       std::vector<SimtOutputChainInfo> &chain_infos,
                                       std::vector<SimtNodeSet> &node_sets) {
  std::vector<af::AscNodePtr> pending;
  SimtNodeSet visited;
  for (const auto &out_node : indirect_load->GetOutDataNodes()) {
    const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
    GE_ASSERT_NOTNULL(consumer, "IndirectLoad SIMT output successor is invalid.");
    if (visited.emplace(consumer.get()).second) {
      pending.emplace_back(consumer);
    }
  }
  std::unordered_map<const af::AscNode *, size_t> cached_node_set_indices;
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const auto &current = pending[cursor];
    if (af::ops::IsOps<af::ascir_op::Store>(current)) {
      const auto producer = ascgen_utils::indirect_load::GetInputProducer(current, 0UL);
      GE_ASSERT_NOTNULL(producer, "IndirectLoad SIMT output terminal[%s] has no producer.", current->GetNamePtr());
      const size_t producer_output_index = GetProducerOutputIndex(current);
      GE_ASSERT_TRUE(producer_output_index < producer->outputs().size(),
                     "IndirectLoad SIMT terminal[%s] producer output index[%zu] is invalid.", current->GetNamePtr(),
                     producer_output_index);
      size_t node_set_index = 0UL;
      const auto cached = cached_node_set_indices.find(producer.get());
      if (cached != cached_node_set_indices.end()) {
        node_set_index = cached->second;
      } else {
        node_set_index = node_sets.size();
        node_sets.emplace_back();
        GE_ASSERT_SUCCESS(CollectSimtBackwardNodes(producer, indirect_load, node_sets.back()));
        cached_node_set_indices.emplace(producer.get(), node_set_index);
      }
      SimtOutputChainInfo chain_info;
      chain_info.node_set_index = node_set_index;
      chain_info.chain.result_tensor_id = producer->outputs()[producer_output_index]->attr.mem.tensor_id;
      chain_info.chain.target_tensor_id = current->outputs()[0]->attr.mem.tensor_id;
      chain_info.chain.dtype = producer->outputs()[producer_output_index]->attr.dtype;
      GELOGD("[IndirectLoad] SIMT chain terminal[%s] producer[%s] result_tensor[%ld] target_tensor[%ld].",
             current->GetNamePtr(), producer->GetNamePtr(), chain_info.chain.result_tensor_id,
             chain_info.chain.target_tensor_id);
      chain_infos.emplace_back(std::move(chain_info));
      continue;
    }
    for (const auto &out_node : current->GetOutDataNodes()) {
      const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
      GE_ASSERT_NOTNULL(consumer, "IndirectLoad SIMT output successor is invalid.");
      if (visited.emplace(consumer.get()).second) {
        pending.emplace_back(consumer);
      }
    }
  }
  return af::SUCCESS;
}

af::Status ValidateSimtRegionNode(const af::AscNodePtr &node) {
  if (af::ops::IsOps<af::ascir_op::Load>(node) || af::ops::IsOps<af::ascir_op::Scalar>(node) ||
      af::ops::IsOps<af::ascir_op::ScalarData>(node) || af::ops::IsOps<af::ascir_op::Store>(node)) {
    return af::SUCCESS;
  }
  const auto role = ascgen_utils::indirect_load::GetTemplateRole(node);
  GE_ASSERT_TRUE(role == ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform ||
                     role == ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch,
                 "IndirectLoad SIMT node[%s] has no scalar evaluator role.", node->GetNamePtr());
  GE_ASSERT_TRUE(!af::ops::IsOps<af::ascir_op::VectorFunc>(node),
                 "IndirectLoad SIMT transform must use scalar emission, node:%s", node->GetNamePtr());
  return af::SUCCESS;
}

void AppendSimtGmTensor(const af::AscNodePtr &node, std::vector<SimtGmTensor> &gm_tensors,
                        std::unordered_set<ascir::TensorId> &seen_tensor_ids) {
  const auto output = node->outputs()[0];
  if (seen_tensor_ids.emplace(output->attr.mem.tensor_id).second) {
    gm_tensors.push_back({output->attr.mem.tensor_id, node->inputs()[0]->attr.mem.tensor_id, output->attr.dtype});
  }
}

void AppendSimtScalarData(const af::AscNodePtr &node, std::vector<SimtGmTensor> &gm_tensors,
                          std::unordered_set<ascir::TensorId> &seen_tensor_ids) {
  const auto output = node->outputs()[0];
  if (seen_tensor_ids.emplace(output->attr.mem.tensor_id).second) {
    gm_tensors.push_back({output->attr.mem.tensor_id, output->attr.mem.tensor_id, output->attr.dtype, true});
  }
}

af::Status CollectSimtGraphMetadata(const ascir::NodeView &indirect_load, const SimtNodeSet &index_set,
                                    const SimtNodeSet &output_set, const SimtChainIndexMap &chain_indices,
                                    std::vector<SimtOutputChainInfo> &chain_infos,
                                    std::vector<af::AscNodePtr> &index_nodes, std::vector<af::AscNodePtr> &output_nodes,
                                    std::vector<SimtGmTensor> &gm_tensors,
                                    std::unordered_set<ascir::TensorId> &seen_tensor_ids) {
  const auto owner_graph = indirect_load->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph, "IndirectLoad SIMT node has no owner graph.");
  for (const auto &graph_node : owner_graph->GetDirectNode()) {
    const af::AscNodePtr node = std::dynamic_pointer_cast<af::AscNode>(graph_node);
    GE_ASSERT_NOTNULL(node, "IndirectLoad SIMT graph contains invalid node.");
    const bool in_index = index_set.count(node.get()) != 0UL;
    const auto chain_indices_it = chain_indices.find(node.get());
    const bool in_output_region = output_set.count(node.get()) != 0UL;
    const bool in_output = in_output_region || chain_indices_it != chain_indices.end();
    if (!in_index && !in_output) {
      continue;
    }
    GE_ASSERT_SUCCESS(ValidateSimtRegionNode(node));
    if (in_index) {
      index_nodes.emplace_back(node);
    }
    if (in_output_region) {
      output_nodes.emplace_back(node);
    }
    if (chain_indices_it != chain_indices.end() && !af::ops::IsOps<af::ascir_op::Store>(node)) {
      for (const size_t chain_index : chain_indices_it->second) {
        GE_ASSERT_TRUE(chain_index < chain_infos.size(), "IndirectLoad SIMT output chain index is invalid.");
        chain_infos[chain_index].chain.nodes.emplace_back(node);
      }
    }
    if (af::ops::IsOps<af::ascir_op::Load>(node)) {
      AppendSimtGmTensor(node, gm_tensors, seen_tensor_ids);
    } else if (af::ops::IsOps<af::ascir_op::ScalarData>(node)) {
      AppendSimtScalarData(node, gm_tensors, seen_tensor_ids);
    }
  }
  return af::SUCCESS;
}

// Shared skeleton for both SIMT metadata variants.  With a null output_root (plain region)
// every output chain is kept and no output region nodes are collected; otherwise (post-reduce)
// the output region is collected and chains whose backward region contains the Reduce node are
// dropped because the reduced chain is emitted separately.
af::Status CollectSimtMetadataImpl(const ascir::NodeView &indirect_load, const af::AscNodePtr &index_root,
                                   const af::AscNodePtr &output_root, std::vector<af::AscNodePtr> &index_nodes,
                                   std::vector<af::AscNodePtr> &output_nodes, std::vector<SimtGmTensor> &gm_tensors,
                                   std::unordered_set<ascir::TensorId> &seen_tensor_ids,
                                   std::vector<SimtOutputChain> &output_chains) {
  GE_ASSERT_NOTNULL(index_root, "IndirectLoad SIMT index region root is missing.");
  SimtNodeSet index_set;
  GE_ASSERT_SUCCESS(CollectSimtBackwardNodes(index_root, indirect_load, index_set));
  SimtNodeSet output_set;
  if (output_root != nullptr) {
    GE_ASSERT_SUCCESS(CollectSimtBackwardNodes(output_root, indirect_load, output_set));
  }
  std::vector<SimtOutputChainInfo> chain_infos;
  std::vector<SimtNodeSet> node_sets;
  GE_ASSERT_SUCCESS(CollectSimtOutputChainInfos(indirect_load, chain_infos, node_sets));
  GE_ASSERT_TRUE(output_root != nullptr || !chain_infos.empty(), "IndirectLoad SIMT output chains are empty.");
  SimtChainIndexMap chain_indices;
  std::vector<bool> keep_chain(chain_infos.size(), true);
  for (size_t chain_index = 0UL; chain_index < chain_infos.size(); ++chain_index) {
    const bool contains_reduce =
        output_root != nullptr &&
        std::any_of(node_sets[chain_infos[chain_index].node_set_index].begin(),
                    node_sets[chain_infos[chain_index].node_set_index].end(), [](const af::AscNode *node) {
                      return node != nullptr && node->attr.api.compute_type == af::ComputeType::kComputeReduce;
                    });
    if (contains_reduce) {
      keep_chain[chain_index] = false;
      continue;
    }
    for (const af::AscNode *node : node_sets[chain_infos[chain_index].node_set_index]) {
      chain_indices[node].emplace_back(chain_index);
    }
  }
  GE_ASSERT_SUCCESS(CollectSimtGraphMetadata(indirect_load, index_set, output_set, chain_indices, chain_infos,
                                             index_nodes, output_nodes, gm_tensors, seen_tensor_ids));
  for (size_t chain_index = 0UL; chain_index < chain_infos.size(); ++chain_index) {
    if (keep_chain[chain_index]) {
      output_chains.emplace_back(std::move(chain_infos[chain_index].chain));
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

bool SimtLoadUsesZeroOffset(const af::AscNodePtr &node) {
  if (node == nullptr || node->outputs().empty()) {
    return false;
  }
  const auto &attr = node->outputs()[0]->attr;
  if (attr.strides.empty() && attr.repeats.empty()) {
    return false;
  }
  const bool all_zero_strides =
      !attr.strides.empty() && std::all_of(attr.strides.begin(), attr.strides.end(), [](const af::Expression &stride) {
        return af::SymbolicUtils::StaticCheckEq(stride, af::ops::Zero) == af::TriBool::kTrue;
      });
  const bool single_element_shape =
      !attr.repeats.empty() && std::all_of(attr.repeats.begin(), attr.repeats.end(), [](const af::Expression &size) {
        return af::SymbolicUtils::StaticCheckEq(size, af::ops::One) == af::TriBool::kTrue;
      });
  return all_zero_strides || single_element_shape;
}

af::Status EmitSimtEvaluatorNodes(const std::vector<af::AscNodePtr> &nodes,
                                  const SimtLoadAddressSources *load_address_sources,
                                  const SimtLoadIndexOffsetExpressions *index_offset_expressions,
                                  std::map<ascir::TensorId, std::string> &values, std::stringstream &ss) {
  for (const af::AscNodePtr &node : nodes) {
    if (af::ops::IsOps<af::ascir_op::Load>(node)) {
      const auto output = node->outputs()[0];
      const char *offset = nullptr;
      std::string custom_offset;
      if (index_offset_expressions != nullptr) {
        const auto expression = index_offset_expressions->find(node.get());
        if (expression != index_offset_expressions->end()) {
          custom_offset = expression->second;
          offset = custom_offset.c_str();
        }
      }
      if (load_address_sources != nullptr) {
        const auto source = load_address_sources->find(node.get());
        GE_ASSERT_TRUE(source != load_address_sources->end(), "SIMT Load[%s] has no address source.",
                       node->GetNamePtr());
        if (offset == nullptr) {
          switch (source->second) {
            case SimtLoadAddressSource::kZeroOffset:
              offset = "0";
              break;
            case SimtLoadAddressSource::kIndexOffset:
              offset = "index_offset";
              break;
            case SimtLoadAddressSource::kOutputOffset:
              offset = "output_index";
              break;
          }
        }
      } else if (offset == nullptr) {
        offset = SimtLoadUsesZeroOffset(node) ? "0" : "output_index";
      }
      values[output->attr.mem.tensor_id] = "context." + std::string(kSimtGmFieldNamePrefix) +
                                           std::to_string(output->attr.mem.tensor_id) + "[" + offset + "]";
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Scalar>(node)) {
      GE_ASSERT_SUCCESS(EmitSimtScalarInput(node, values, ss));
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::ScalarData>(node)) {
      const auto output = node->outputs()[0];
      const std::string variable =
          "context." + std::string(kSimtValueNamePrefix) + std::to_string(output->attr.mem.tensor_id);
      values[output->attr.mem.tensor_id] = variable;
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Store>(node)) {
      continue;
    }
    GE_ASSERT_SUCCESS(EmitSimtTransform(node, values, ss));
  }
  return af::SUCCESS;
}

af::Status GenerateSimtEvaluatorBody(const std::vector<af::AscNodePtr> &nodes,
                                     std::map<ascir::TensorId, std::string> &values, ascir::TensorId result_tensor_id,
                                     std::stringstream &ss,
                                     const SimtLoadAddressSources *load_address_sources = nullptr,
                                     const SimtLoadIndexOffsetExpressions *index_offset_expressions = nullptr) {
  GE_ASSERT_SUCCESS(EmitSimtEvaluatorNodes(nodes, load_address_sources, index_offset_expressions, values, ss));
  const auto result = values.find(result_tensor_id);
  GE_ASSERT_TRUE(result != values.end(), "SIMT evaluator result tensor[%ld] has no scalar value.", result_tensor_id);
  ss << "    return " << result->second << ";" << std::endl;
  ss << "  }" << std::endl;
  return af::SUCCESS;
}

af::Status GenerateSimtMultiOutputEvaluator(const std::string &input_dtype, const std::string &offset_type,
                                            ascir::TensorId value_tensor_id, const std::vector<SimtOutputChain> &chains,
                                            const SimtLoadAddressSources &load_address_sources, std::stringstream &ss) {
  ss << "  struct OutputPack {" << std::endl;
  for (size_t i = 0UL; i < chains.size(); ++i) {
    std::string dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(chains[i].dtype, dtype));
    ss << "    " << dtype << " output" << i << ";" << std::endl;
  }
  ss << "  };" << std::endl;
  ss << "  struct OutputTargets {" << std::endl;
  for (size_t i = 0UL; i < chains.size(); ++i) {
    std::string dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(chains[i].dtype, dtype));
    ss << "    " << (chains[i].local_target ? "__ubuf__ " : "__gm__ ") << dtype << " *output" << i << ";" << std::endl;
  }
  ss << "  };" << std::endl;
  std::map<ascir::TensorId, std::string> values{{value_tensor_id, "value"}};
  ss << "  __simt_callee__ __aicore__ inline static OutputPack Outputs(" << input_dtype << " value, " << offset_type
     << " output_index, " << offset_type << " index_offset, const Context &context) {" << std::endl;
  std::vector<af::AscNodePtr> nodes;
  SimtNodeSet seen;
  for (const auto &chain : chains) {
    for (const auto &node : chain.nodes) {
      if (seen.emplace(node.get()).second) {
        nodes.emplace_back(node);
      }
    }
  }
  GE_ASSERT_SUCCESS(EmitSimtEvaluatorNodes(nodes, &load_address_sources, nullptr, values, ss));
  ss << "    OutputPack outputs;" << std::endl;
  for (size_t i = 0UL; i < chains.size(); ++i) {
    const auto found = values.find(chains[i].result_tensor_id);
    GE_ASSERT_TRUE(found != values.end(), "SIMT output chain[%zu] result tensor[%ld] has no scalar value.", i,
                   chains[i].result_tensor_id);
    ss << "    outputs.output" << i << " = " << found->second << ";" << std::endl;
  }
  ss << "    return outputs;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  __simt_callee__ __aicore__ inline static void Store(const OutputTargets &targets, " << offset_type
     << " output_index, " << offset_type << " local_index, const OutputPack &outputs) {" << std::endl;
  for (size_t i = 0UL; i < chains.size(); ++i) {
    ss << "    targets.output" << i << "[" << (chains[i].local_target ? "local_index" : "output_index")
       << "] = outputs.output" << i << ";" << std::endl;
  }
  ss << "  }" << std::endl;
  return af::SUCCESS;
}

std::string GetSimdLogicalOutputSize(const TPipe &tpipe, const Tensor &tensor) {
  std::stringstream ss;
  ss << "1";
  for (size_t i = 0; i < tensor.vectorized_axis.size(); ++i) {
    const uint32_t axis_pos = tensor.vectorized_axis_pos[i];
    if (axis_pos >= tensor.axis_size.size()) {
      continue;
    }
    const auto &axis = tpipe.tiler.GetAxis(tensor.vectorized_axis[i]);
    const auto &axis_size = tensor.axis_size[axis_pos];
    const bool use_actual = axis.type == Axis::Type::kAxisTypeTileInner ||
                            af::SymbolicUtils::StaticCheckEq(axis_size, axis.size_expr) == af::TriBool::kTrue;
    ss << " * " << (use_actual ? "(" + axis.actual_size.Str() + ")" : "(" + tpipe.tiler.Size(axis_size) + ")");
  }
  return ss.str();
}

af::Status GenSimtIndexEvaluator(const std::string &index_dtype, const std::string &offset_type,
                                 ascir::TensorId result_tensor_id, const std::vector<af::AscNodePtr> &nodes,
                                 const SimtLoadIndexOffsetExpressions *index_offset_expressions,
                                 std::stringstream &ss) {
  std::map<ascir::TensorId, std::string> values;
  ss << "  __simt_callee__ __aicore__ inline static " << index_dtype << " Index(" << offset_type
     << " output_index, const Context &context) {" << std::endl;
  return GenerateSimtEvaluatorBody(nodes, values, result_tensor_id, ss, nullptr, index_offset_expressions);
}

af::Status GenSimtOutputEvaluator(const std::string &output_dtype, const std::string &input_dtype,
                                  const std::string &offset_type, ascir::TensorId value_tensor_id,
                                  ascir::TensorId result_tensor_id, const std::vector<af::AscNodePtr> &nodes,
                                  const SimtLoadAddressSources &load_address_sources, std::stringstream &ss) {
  std::map<ascir::TensorId, std::string> values{{value_tensor_id, "value"}};
  ss << "  __simt_callee__ __aicore__ inline static " << output_dtype << " Output(" << input_dtype << " value, "
     << offset_type << " output_index, " << offset_type << " index_offset, const Context &context) {" << std::endl;
  return GenerateSimtEvaluatorBody(nodes, values, result_tensor_id, ss, &load_address_sources);
}

af::Status CalcVectorizedElementCount(const Tensor &tensor, af::Expression &element_count) {
  element_count = af::ops::One;
  for (uint32_t axis_pos : tensor.vectorized_axis_pos) {
    GE_ASSERT_TRUE(axis_pos < tensor.axis_size.size(), "IndirectLoad SIMT output axis is invalid.");
    element_count = af::sym::Mul(element_count, tensor.axis_size[axis_pos]);
  }
  return af::SUCCESS;
}

std::vector<ascir::SizeExpr> GetSimdOutputStrides(const Tensor &tensor,
                                                  const ascgen_utils::indirect_load::LogicalTensorView &layout) {
  std::vector<ascir::SizeExpr> strides(layout.axis_ids.size(), af::sym::kSymbolOne);
  for (size_t dim = 0; dim < layout.axis_ids.size(); ++dim) {
    const auto it = std::find(tensor.vectorized_axis.begin(), tensor.vectorized_axis.end(), layout.axis_ids[dim]);
    if (it != tensor.vectorized_axis.end()) {
      strides[dim] = tensor.vectorized_strides[static_cast<size_t>(std::distance(tensor.vectorized_axis.begin(), it))];
      continue;
    }
    const auto axis_it = std::find(tensor.axis.begin(), tensor.axis.end(), layout.axis_ids[dim]);
    if (axis_it != tensor.axis.end()) {
      strides[dim] = tensor.axis_strides[static_cast<size_t>(std::distance(tensor.axis.begin(), axis_it))];
    }
  }
  return strides;
}

// Logical eligibility of the input view is already proven by access_info_.can_use_simd_embedding
// at the call site; this only checks the physical output window: the output suffix must be dense
// and the output axis stride must cover the payload span.
bool IsSimdEmbeddingPhysicalWindow(const Tensor &output,
                                   const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                                   const LogicalTensorInfo &input_info, size_t axis) {
  const auto output_strides = GetSimdOutputStrides(output, logical_view.output);
  const size_t rank = input_info.sizes.size();
  if (output_strides.size() != rank || axis + 1UL >= rank) {
    return false;
  }
  ascir::SizeExpr payload_span = af::sym::kSymbolOne;
  for (size_t dim = rank; dim > axis + 1UL; --dim) {
    const size_t current = dim - 1UL;
    if (af::SymbolicUtils::StaticCheckEq(output_strides[current], payload_span) != af::TriBool::kTrue) {
      return false;
    }
    payload_span = af::sym::Mul(payload_span, input_info.sizes[current]);
  }
  if (af::SymbolicUtils::StaticCheckLt(output_strides[axis], payload_span) == af::TriBool::kTrue) {
    return false;
  }
  return af::SymbolicUtils::StaticCheckGt(output_strides[axis], af::sym::kSymbolZero) == af::TriBool::kTrue;
}

enum class SimdApiKind { kDense, kGather, kStrided };

SimdApiKind SelectSimdApi(const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                          ascgen_utils::indirect_load::Implementation implementation) {
  const bool strided = logical_view.input.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense ||
                       logical_view.index.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense;
  if (strided) {
    return SimdApiKind::kStrided;
  }
  return implementation == ascgen_utils::indirect_load::Implementation::kGatherApi ? SimdApiKind::kGather
                                                                                   : SimdApiKind::kDense;
}

const char *GetSimdApiName(SimdApiKind api_kind) {
  switch (api_kind) {
    case SimdApiKind::kStrided:
      return "IndirectLoadSimdStrided";
    case SimdApiKind::kGather:
      return "IndirectLoadSimdGatherApi";
    case SimdApiKind::kDense:
      return "IndirectLoadSimd";
  }
  return "IndirectLoadSimd";
}

void EmitSimdStridedParams(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, const Tensor &output,
                           const LogicalTensorInfo &input_info, const LogicalTensorInfo &index_info,
                           const ascgen_utils::indirect_load::LogicalTensorView &output_layout, size_t rank,
                           std::stringstream &ss) {
  const std::string logical_output_size = GetSimdLogicalOutputSize(tpipe, output);
  const auto output_strides = GetSimdOutputStrides(output, output_layout);
  ss << "  AscendC::IndirectLoadSimdStridedParams<" << rank << "> indirect_load_simd_params{static_cast<uint32_t>("
     << logical_output_size << "), static_cast<uint32_t>(" << output.actual_size << "), "
     << tpipe.tiler.Offset(current_axis, output.axis, output.axis_strides) << ", {"
     << JoinSizeExprs(index_info.sizes, tpipe) << "}, {" << JoinSizeExprs(input_info.strides, tpipe) << "}, {"
     << JoinSizeExprs(index_info.strides, tpipe) << "}, {" << JoinSizeExprs(output_strides, tpipe) << "}};"
     << std::endl;
}

void EmitSimdInvocation(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, const Tensor &input,
                        const Tensor &index, const Tensor &output, const LogicalTensorInfo &input_info,
                        const LogicalTensorInfo &index_info, int64_t axis, size_t axis_pos, SimdApiKind api_kind,
                        const std::string &input_dtype, const std::string &index_dtype, const std::string &tmp_name,
                        std::stringstream &ss) {
  ss << "  AscendC::" << GetSimdApiName(api_kind) << "<" << input_dtype << ", " << index_dtype << ", "
     << input_info.sizes.size() << ", " << axis << ">(\n";
  ss << "      " << input << ", " << index << ", " << output << ", ";
  if (api_kind == SimdApiKind::kStrided) {
    ss << tmp_name << ", indirect_load_simd_params";
  } else {
    ss << output.actual_size << ", " << tpipe.tiler.Offset(current_axis, output.axis, output.axis_strides) << ", "
       << input.actual_size << ", " << tpipe.tiler.Size(input_info.sizes[axis_pos]) << ", "
       << JoinSizeExprs(index_info.sizes, tpipe) << ", " << JoinSizeExprs(input_info.strides, tpipe);
  }
  ss << ");" << std::endl;
}

void EmitSimtInvocation(const TPipe &tpipe, const std::string &input_dtype, const std::string &output_dtype,
                        const std::string &outer_tb_var, const std::string &body_name,
                        const LogicalTensorInfo &input_info, const LogicalTensorInfo &index_info,
                        const SimtCodegenPlan &plan, int64_t axis, bool has_post_reduce, const Tensor *output_tensor,
                        const af::Expression &output_element_count, std::stringstream &ss) {
  ss << "  AscendC::IndirectLoadSimt<" << input_dtype << ", " << output_dtype << ", " << body_name << ", ";
  ss << GetSimtPolicyType(plan, input_info.sizes.size(), axis) << ">(" << std::endl;
  if (has_post_reduce) {
    const std::string output_elements = tpipe.tiler.Size(output_element_count);
    ss << "      input_ptr, " << *output_tensor << ", context, static_cast<uint32_t>(" << output_elements << "), ";
    ss << "(static_cast<" << plan.offset_type << ">(block_dim_offset) + static_cast<" << plan.offset_type << ">("
       << outer_tb_var << ")) * " << PromoteSizeExpr(output_elements, plan.offset_type);
  } else {
    ss << "      input_ptr, y_ptr, context, static_cast<uint32_t>(" << outer_tb_var << "_loop_size), ";
    ss << "static_cast<" << plan.offset_type << ">(block_dim_offset)";
  }
  const std::string policy_args = GetSimtPolicyArgs(plan, input_info, index_info, tpipe);
  if (!policy_args.empty()) {
    ss << ", " << policy_args;
  }
  ss << ");" << std::endl;
}

af::Status GenerateSimtContextInitializer(const std::string &context_name, const std::vector<SimtGmTensor> &gm_tensors,
                                          const TPipe &tpipe, std::stringstream &ss) {
  ss << "  " << context_name << " context{";
  for (size_t i = 0UL; i < gm_tensors.size(); ++i) {
    const SimtGmTensor &gm_tensor = gm_tensors[i];
    std::string dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(gm_tensor.dtype, dtype));
    if (gm_tensor.is_scalar) {
      const Tensor *scalar = tpipe.GetTensor(gm_tensor.value_tensor_id);
      GE_ASSERT_NOTNULL(scalar, "IndirectLoad SIMT ScalarData tensor is missing.");
      ss << (i == 0UL ? "" : ", ") << scalar->name;
    } else {
      ss << (i == 0UL ? "" : ", ") << "(__gm__ " << dtype << " *)" << kGlobalTensorNamePrefix << gm_tensor.gm_tensor_id
         << ".GetPhyAddr()";
    }
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
  has_post_reduce_ =
      ascgen_utils::indirect_load::GetPostReduceConsumer(std::dynamic_pointer_cast<af::AscNode>(node)) != nullptr;
  template_id_ = ::ascir::GetTemplateIdOrDefault(*node);
  GE_ASSERT_TRUE(
      template_id_ == ascir::TemplateId::kIndirectLoadSK || template_id_ == ascir::TemplateId::kIndirectLoadSimd ||
          template_id_ == ascir::TemplateId::kIndirectLoadSimt,
      "IndirectLoad node[%s] has invalid template id[%d].", node->GetNamePtr(), static_cast<int32_t>(template_id_));
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateLogicalView(node, logical_view_));
  // Recompute the layout capabilities from the persisted logical view.  The scheduler
  // stores access info before graph rewrites (input-pre moves, broadcast deletion), while
  // the index-invariance proof walks the rewritten graph, so recomputing here keeps the
  // classification consistent with the graph codegen actually emits.
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::AnalyzeIndirectLoadAccess(node, logical_view_, access_info_));
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
  simt_gm_tensors_.clear();
  simt_gm_tensor_ids_.clear();
  index_nodes_.clear();
  output_nodes_.clear();
  output_chains_.clear();
  simt_load_address_sources_.clear();
  simt_input_info_ = {};
  simt_index_info_ = {};
  simt_plan_ = {};
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
    const af::AscNodePtr output_root =
        ascgen_utils::indirect_load::GetPostReduceInputProducer(std::dynamic_pointer_cast<af::AscNode>(node));
    GE_ASSERT_NOTNULL(output_root, "IndirectLoad post Reduce input has no producer.");
    GE_ASSERT_TRUE(!output_root->outputs().empty(), "IndirectLoad post Reduce input producer has no output.");
    output_result_tensor_id_ = output_root->outputs()[0]->attr.mem.tensor_id;
    output_dtype_ = output_root->outputs()[0]->attr.dtype;
    const af::AscNodePtr index_root = ascgen_utils::indirect_load::GetInputProducer(
        std::dynamic_pointer_cast<af::AscNode>(node), ascgen_utils::indirect_load::kIndexTensorIndex);
    GE_ASSERT_SUCCESS(CollectSimtMetadataImpl(node, index_root, output_root, index_nodes_, output_nodes_,
                                              simt_gm_tensors_, simt_gm_tensor_ids_, output_chains_));
    SimtOutputChain reduce_chain;
    reduce_chain.nodes = output_nodes_;
    reduce_chain.result_tensor_id = output_result_tensor_id_;
    reduce_chain.target_tensor_id = output_result_tensor_id_;
    reduce_chain.dtype = output_dtype_;
    reduce_chain.local_target = true;
    output_chains_.emplace_back(std::move(reduce_chain));
    BuildSimtLoadAddressSources(index_nodes_, output_chains_, simt_load_address_sources_);
    simt_plan_ = BuildSimtCodegenPlanForNode(logical_view_, index_nodes_, static_cast<size_t>(axis_), simt_input_info_,
                                             simt_index_info_, access_info_.can_use_simt_structured);
    GE_ASSERT_TRUE(outputs.size() == kIndirectLoadOutputCount, "Invalid IndirectLoad SIMT output number:%zu.",
                   outputs.size());
    outputs[0].id = output_result_tensor_id_;
    return af::SUCCESS;
  }
  const af::AscNodePtr index_root = ascgen_utils::indirect_load::GetInputProducer(
      std::dynamic_pointer_cast<af::AscNode>(node), ascgen_utils::indirect_load::kIndexTensorIndex);
  GE_ASSERT_NOTNULL(index_root, "IndirectLoad SIMT index input has no producer.");
  GE_ASSERT_SUCCESS(CollectSimtMetadataImpl(node, index_root, nullptr, index_nodes_, output_nodes_, simt_gm_tensors_,
                                            simt_gm_tensor_ids_, output_chains_));
  GE_ASSERT_TRUE(!output_chains_.empty(), "IndirectLoad SIMT output chains are empty.");
  if (output_chains_.size() == 1UL) {
    output_result_tensor_id_ = output_chains_[0].result_tensor_id;
    output_nodes_ = output_chains_[0].nodes;
  }
  output_dtype_ = output_chains_.front().dtype;
  BuildSimtLoadAddressSources(index_nodes_, output_chains_, simt_load_address_sources_);
  simt_plan_ = BuildSimtCodegenPlanForNode(logical_view_, index_nodes_, static_cast<size_t>(axis_), simt_input_info_,
                                           simt_index_info_, access_info_.can_use_simt_structured);
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
  const auto &input_info = simt_input_info_;
  const auto &plan = simt_plan_;
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
      node_name.c_str(), input_info.sizes.size(), axis_, index_nodes_.size(),
      has_post_reduce_ ? output_nodes_.size() : output_chains_.size(), simt_gm_tensors_.size());

  GE_ASSERT_SUCCESS(GenerateSimtContextDefinition(context_name, simt_gm_tensors_, ss));
  ss << "struct " << body_name << " {" << std::endl;
  ss << "  using Context = " << context_name << ";" << std::endl;
  SimtLoadIndexOffsetExpressions index_offset_expressions;
  if (plan.use_per_load_index_offsets) {
    GE_ASSERT_SUCCESS(BuildSimtPerLoadIndexOffsetExpressions(logical_view_, index_nodes_, tpipe, plan.offset_type,
                                                             index_offset_expressions));
  }
  GE_ASSERT_SUCCESS(GenSimtIndexEvaluator(index_dtype, plan.offset_type, index_result_tensor_id_, index_nodes_,
                                          plan.use_per_load_index_offsets ? &index_offset_expressions : nullptr, ss));
  if (output_chains_.size() <= 1UL) {
    GE_ASSERT_SUCCESS(GenSimtOutputEvaluator(output_dtype, input_dtype, plan.offset_type, simt_value_tensor_id_,
                                             output_result_tensor_id_, output_nodes_, simt_load_address_sources_, ss));
  } else {
    GE_ASSERT_SUCCESS(GenerateSimtMultiOutputEvaluator(input_dtype, plan.offset_type, simt_value_tensor_id_,
                                                       output_chains_, simt_load_address_sources_, ss));
  }
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
  const SimdApiKind api_kind = SelectSimdApi(logical_view_, implementation_);
  const auto tmp_iter = tmp_buf_id.find(-1L);
  if (api_kind == SimdApiKind::kStrided) {
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
  if (api_kind == SimdApiKind::kStrided) {
    EmitSimdStridedParams(tpipe, current_axis, output, input_info, index_info, logical_view_.output,
                          input_info.sizes.size(), ss);
  }
  const std::string tmp_name =
      api_kind == SimdApiKind::kStrided ? tpipe.tmp_buf.name + "_" + std::to_string(tmp_iter->second) : "";
  const bool embedding_fast_path = implementation_ == ascgen_utils::indirect_load::Implementation::kDefault &&
                                   access_info_.can_use_simd_embedding &&
                                   IsSimdEmbeddingPhysicalWindow(output, logical_view_, input_info, axis_pos);
  if (embedding_fast_path) {
    const std::string logical_output_size = GetSimdLogicalOutputSize(tpipe, output);
    const auto output_strides = GetSimdOutputStrides(output, logical_view_.output);
    ss << "  if (!AscendC::Internal::TryIndirectLoadSimdEmbedding<" << input_dtype << ", " << index_dtype << ", "
       << input_info.sizes.size() << ", " << axis_ << ">(\n";
    ss << "      " << input << ", " << index << ", " << output << ", static_cast<uint32_t>(" << logical_output_size
       << "), " << tpipe.tiler.Offset(current_axis, output.axis, output.axis_strides) << ", "
       << JoinSizeExprs(index_info.sizes, tpipe) << ", " << JoinSizeExprs(input_info.strides, tpipe) << ", "
       << JoinSizeExprs(index_info.strides, tpipe) << ", " << JoinSizeExprs(output_strides, tpipe) << ")) {\n";
    EmitSimdInvocation(tpipe, current_axis, input, index, output, input_info, index_info, axis_, axis_pos, api_kind,
                       input_dtype, index_dtype, tmp_name, ss);
    ss << "  }\n";
  } else {
    EmitSimdInvocation(tpipe, current_axis, input, index, output, input_info, index_info, axis_, axis_pos, api_kind,
                       input_dtype, index_dtype, tmp_name, ss);
  }
  ss << "}" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::GenerateSimtInvocation(const TPipe &tpipe, const std::string &input_dtype,
                                                      const std::string &outer_tb_var, std::stringstream &ss) const {
  std::string output_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(output_dtype_, output_dtype));
  const auto &input_info = simt_input_info_;
  const auto &index_info = simt_index_info_;
  const auto &plan = simt_plan_;
  const std::string body_name = kSimtBodyNamePrefix + ascgen_utils::GenValidName(node_name);
  std::string actual_size_expr;
  std::string output_offset_expr;
  if (has_post_reduce_) {
    const Tensor *output_tensor = tpipe.GetTensor(output_result_tensor_id_);
    GE_ASSERT_NOTNULL(output_tensor, "IndirectLoad SIMT post Reduce output tensor is missing.");
    af::Expression output_element_count;
    GE_ASSERT_SUCCESS(CalcVectorizedElementCount(*output_tensor, output_element_count));
    actual_size_expr = tpipe.tiler.Size(output_element_count);
    output_offset_expr = "(static_cast<" + plan.offset_type + ">(block_dim_offset) + static_cast<" + plan.offset_type +
                         ">( " + outer_tb_var + ")) * " + PromoteSizeExpr(actual_size_expr, plan.offset_type);
  } else {
    actual_size_expr = outer_tb_var + "_loop_size";
    output_offset_expr = "static_cast<" + plan.offset_type + ">(block_dim_offset)";
  }
  if (output_chains_.size() > 1UL) {
    ss << "  AscendC::IndirectLoadSimtMulti<" << input_dtype << ", " << body_name << ", ";
    ss << GetSimtPolicyType(plan, input_info.sizes.size(), axis_) << ">(\n";
    ss << "      input_ptr, " << body_name << "::OutputTargets{";
    for (size_t i = 0UL; i < output_chains_.size(); ++i) {
      const auto &chain = output_chains_[i];
      std::string dtype;
      GE_ASSERT_SUCCESS(Tensor::DtypeName(chain.dtype, dtype));
      if (i != 0UL) {
        ss << ", ";
      }
      if (chain.local_target) {
        const Tensor *output_tensor = tpipe.GetTensor(chain.target_tensor_id);
        GE_ASSERT_NOTNULL(output_tensor, "IndirectLoad SIMT local output tensor[%ld] is missing.",
                          chain.target_tensor_id);
        ss << "(__ubuf__ " << dtype << " *)" << output_tensor->name << ".GetPhyAddr()";
      } else {
        ss << "(__gm__ " << dtype << " *)" << kGlobalTensorNamePrefix << chain.target_tensor_id << ".GetPhyAddr()";
      }
    }
    ss << "}, context, static_cast<uint32_t>(" << actual_size_expr << "), ";
    ss << output_offset_expr;
    const std::string policy_args = GetSimtPolicyArgs(plan, input_info, index_info, tpipe);
    if (!policy_args.empty()) {
      ss << ", " << policy_args;
    }
    ss << ");" << std::endl;
    return af::SUCCESS;
  }
  const Tensor *output_tensor = nullptr;
  af::Expression output_element_count = af::ops::One;
  if (has_post_reduce_) {
    output_tensor = tpipe.GetTensor(output_result_tensor_id_);
    GE_ASSERT_NOTNULL(output_tensor, "IndirectLoad SIMT post Reduce output tensor is missing.");
    GE_ASSERT_SUCCESS(CalcVectorizedElementCount(*output_tensor, output_element_count));
  }
  EmitSimtInvocation(tpipe, input_dtype, output_dtype, outer_tb_var, body_name, input_info, index_info, plan, axis_,
                     has_post_reduce_, output_tensor, output_element_count, ss);
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::GenerateSimt(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                            const Tensor &input, std::string &result) const {
  GE_ASSERT_TRUE(logical_view_.input.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported &&
                     logical_view_.index.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported,
                 "IndirectLoad SIMT input or index layout is unsupported.");
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ValidateIndirectLoadOutputLayout(logical_view_.output));
  GELOGD("[IndirectLoad] Generate SIMT body for node[%s], output_chains[%zu].", node_name.c_str(),
         output_chains_.size());

  std::string outer_tb_var;
  const bool has_outer_tb =
      FindCurrentAxisVar(tpipe, current_axis, Axis::Type::kAxisTypeBlockInner, outer_axis_, outer_tb_var);
  GE_ASSERT_TRUE(has_outer_tb, "IndirectLoad SIMT current axes must contain the output block-inner axis.");
  std::string input_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(input.dtype, input_dtype));
  const std::string valid_node_name = ascgen_utils::GenValidName(node_name);
  const std::string context_name = kSimtContextNamePrefix + valid_node_name;
  std::stringstream ss;
  ss << "// IndirectLoad SIMT" << std::endl;
  ss << "{" << std::endl;
  ss << "  __gm__ " << input_dtype << " *input_ptr = (__gm__ " << input_dtype << " *)" << input << ".GetPhyAddr();"
     << std::endl;
  if (!has_post_reduce_ && output_chains_.size() == 1UL) {
    const auto &output_chain = output_chains_.front();
    std::string output_dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(output_chain.dtype, output_dtype));
    ss << "  __gm__ " << output_dtype << " *y_ptr = (__gm__ " << output_dtype << " *)" << kGlobalTensorNamePrefix
       << output_chain.target_tensor_id << ".GetPhyAddr();" << std::endl;
  }
  GE_ASSERT_SUCCESS(GenerateSimtContextInitializer(context_name, simt_gm_tensors_, tpipe, ss));
  GE_ASSERT_SUCCESS(GenerateSimtInvocation(tpipe, input_dtype, outer_tb_var, ss));
  ss << "}" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<IndirectLoadRegApiCall> register_indirect_load_reg_api_call("IndirectLoadRegApiCall");
}  // namespace codegen
