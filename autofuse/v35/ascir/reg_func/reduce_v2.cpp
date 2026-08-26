/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascendc_ir.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "reduce_reuse_utils.h"

namespace af {
namespace ascir {

constexpr int32_t kFloatAlignSize = 8;
constexpr int32_t kPerformanceOptimization = 256;
constexpr int32_t kBlockSize = 32;

static AscGraphAttr *GetOrCreateGraphAttrsGroup(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr;);
  auto attr = graph->GetOrCreateAttrsGroup<AscGraphAttr>();
  GE_CHECK_NOTNULL_EXEC(attr, return nullptr;);
  return attr;
}

static Expression GetAlignSize(Expression in) {
  return sym::Align(in, kFloatAlignSize);
}

static Expression GetByteSize(Expression in) {
  return sym::Mul(in, Symbol(sizeof(float)));
}

static bool IsNeedAccumulation(const AscNode &node) {
  if (node.GetType() == "Sum" || node.GetType() == "Mean" || node.GetType() == "Prod") {
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcReduceTmpSizeV2(const AscNode &node) {
  std::vector<std::unique_ptr<TmpBufDesc>> tmp_buf_desc;
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;
  if (node_inputs.Size() <= 0) {
    return tmp_buf_desc;
  }

  if (node_outputs[0].attr.vectorized_strides.size() <= 0) {
    return tmp_buf_desc;
  }

  bool is_reuse_source = false;
  const auto &in_node = GetReduceInputNode(node);
  if (HasSingleOutNode(in_node) && !HasUpstreamBroadcastOnReduceAxis(node)) {
    is_reuse_source = true;
  }

  bool isAr =
      SymbolicUtils::StaticCheckEq(node_outputs[0].attr.vectorized_strides.back(), sym::kSymbolZero) == TriBool::kTrue;
  auto attr = GetOrCreateGraphAttrsGroup(node.GetOwnerComputeGraph());

  Expression r_in_ub_exp = Symbol(1);
  Expression a_in_ub_exp = Symbol(1);
  for (size_t i = 0; i < node_outputs[0].attr.vectorized_strides.size(); i++) {
    uint64_t vectorized_axis_id = node_outputs[0].attr.vectorized_axis[i];
    Expression tmp_exp = attr->axis[vectorized_axis_id]->size;
    if (i == node_outputs[0].attr.vectorized_strides.size() - 1) {
      tmp_exp = GetAlignSize(tmp_exp);
    }

    if (SymbolicUtils::StaticCheckEq(node_outputs[0].attr.vectorized_strides[i], sym::kSymbolZero) == TriBool::kTrue &&
        SymbolicUtils::StaticCheckEq(node_inputs[0].attr.vectorized_strides[i], sym::kSymbolZero) != TriBool::kTrue) {
      r_in_ub_exp = sym::Mul(r_in_ub_exp, tmp_exp);
    } else {
      a_in_ub_exp = sym::Mul(a_in_ub_exp, tmp_exp);
    }
  }

  Expression rFusedExpression = attr->axis[node.attr.sched.loop_axis]->size;
  if (IsNeedAccumulation(node)) {
    // 高阶API使用  a.UB * r.UB， ar场景需要加一个block，生命周期为-1
    Expression api_size = GetByteSize(sym::Mul(a_in_ub_exp, r_in_ub_exp));
    if (node.GetType() == "Prod") {
      api_size = sym::Add(api_size, Symbol(kPerformanceOptimization));
    }
    if (isAr) {
      api_size = sym::Add(api_size, Symbol(kBlockSize));
    }
    if (!is_reuse_source) {
      TmpBufDesc desc2 = {api_size, -1};
      tmp_buf_desc.emplace_back(std::make_unique<TmpBufDesc>(desc2));
    }

    // UB 间
    if (isAr) {
      a_in_ub_exp = GetAlignSize(a_in_ub_exp);
    }
    Expression a_size = GetByteSize(a_in_ub_exp);
    TmpBufDesc desc3 = {a_size, 0};
    tmp_buf_desc.emplace_back(std::make_unique<TmpBufDesc>(desc3));
  } else {
    // 高阶api部分 先按照最大的申请
    if (!is_reuse_source) {
      Expression api_size = GetByteSize(sym::Mul(a_in_ub_exp, r_in_ub_exp));
      TmpBufDesc desc1 = {api_size, -1};
      tmp_buf_desc.emplace_back(std::make_unique<TmpBufDesc>(desc1));
    }

    // UB 间
    if (isAr) {
      a_in_ub_exp = GetAlignSize(a_in_ub_exp);
    }
    Expression ub_size = GetByteSize(a_in_ub_exp);
    TmpBufDesc desc2 = {ub_size, 0};
    tmp_buf_desc.emplace_back(std::make_unique<TmpBufDesc>(desc2));
  }

  return tmp_buf_desc;
}
}  // namespace ascir
}  // namespace af
