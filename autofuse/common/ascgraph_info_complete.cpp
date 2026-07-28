/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascgraph_info_complete.h"
#include <map>
#include <queue>
#include "ascir_ops.h"
#include "ascendc_ir_def.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/attribute_group/attr_group_shape_env.h"
#include "ascir_ops_utils.h"
#include "schedule_result.h"

using namespace af::ascir_op;

namespace optimize {
namespace {
constexpr int64_t kGatherSimtDcacheSize = 32 * 1024;

static Status GetNodeIrAttrOffset(const af::NodePtr &node, af::Expression &offset) {
  auto asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
  GE_ASSERT_NOTNULL(asc_node);
  GE_ASSERT_NOTNULL(asc_node->attr.ir_attr);
  return asc_node->attr.ir_attr->GetAttrValue("offset", offset);
}

void InsertFreeSymbolsIntoVarSet(const af::Expression &exp, SizeVarSet &size_vars) {
  std::vector<af::Expression> free_symbols = exp.FreeSymbols();
  size_vars.insert(free_symbols.begin(), free_symbols.end());
}

void CompleteDataApiInfo(af::AscNodePtr &node) {
  node->attr.api.type = af::ApiType::kAPITypeBuffer;
  node->attr.api.unit = af::ComputeUnit::kUnitNone;
}

void CompleteLoadApiInfo(af::AscNodePtr &node) {
  node->attr.api.type = af::ApiType::kAPITypeCompute;
  node->attr.api.unit = af::ComputeUnit::kUnitMTE2;
}

void CompleteStoreApiInfo(af::AscNodePtr &node) {
  node->attr.api.type = af::ApiType::kAPITypeCompute;
  node->attr.api.unit = af::ComputeUnit::kUnitMTE2;
}

void CompleteElewiseApiInfo(af::AscNodePtr &node) {
  node->attr.api.type = af::ApiType::kAPITypeCompute;
  node->attr.api.unit = af::ComputeUnit::kUnitVector;
}

void CompleteBroadcastApiInfo(af::AscNodePtr &node) {
  node->attr.api.type = af::ApiType::kAPITypeCompute;
  node->attr.api.unit = af::ComputeUnit::kUnitVector;
}

void CompleteReduceApiInfo(af::AscNodePtr &node) {
  node->attr.api.type = af::ApiType::kAPITypeCompute;
  node->attr.api.unit = af::ComputeUnit::kUnitVector;
}

void CompleteConcatApiInfo(af::AscNodePtr &node) {
  node->attr.api.type = af::ApiType::kAPITypeCompute;
  node->attr.api.unit = af::ComputeUnit::kUnitVector;
}

void CompleteSplitApiInfo(af::AscNodePtr &node) {
  node->attr.api.type = af::ApiType::kAPITypeCompute;
  node->attr.api.unit = af::ComputeUnit::kUnitVector;
}

void CompleteGatherApiInfo(af::AscNodePtr &node) {
  node->attr.api.type = af::ApiType::kAPITypeCompute;
  node->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  (void)::ascir::SetDcacheSize(node, kGatherSimtDcacheSize);
}

void CompleteCubeApiInfo(af::AscNodePtr &node) {
  node->attr.api.type = af::ApiType::kAPITypeCompute;
  node->attr.api.unit = af::ComputeUnit::kUnitCube;
}
}  // namespace

using CompleteApiInfoFunc = std::function<void(af::AscNodePtr &)>;
struct Completer {
  CompleteApiInfoFunc complete_api_info;
};

void CompleteTransposeApiInfo(af::AscNodePtr &node) {
  node->attr.api.type = af::ApiType::kAPITypeCompute;
  node->attr.api.unit = af::ComputeUnit::kUnitVector;
}

static const std::map<std::string, af::ComputeType> kOpTypeToComputeType = {
    {Workspace::Type, af::ComputeType::kComputeInvalid},
    {Data::Type, af::ComputeType::kComputeInvalid},
    {Scalar::Type, af::ComputeType::kComputeInvalid},
    {Output::Type, af::ComputeType::kComputeInvalid},
    {IndexExpr::Type, af::ComputeType::kComputeInvalid},

    {Load::Type, af::ComputeType::kComputeLoad},
    {Store::Type, af::ComputeType::kComputeStore},

    {Sum::Type, af::ComputeType::kComputeReduce},
    {Max::Type, af::ComputeType::kComputeReduce},
    {ArgMax::Type, af::ComputeType::kComputeReduce},
    {Mean::Type, af::ComputeType::kComputeReduce},
    {Min::Type, af::ComputeType::kComputeReduce},
    {Prod::Type, af::ComputeType::kComputeReduce},
    {All::Type, af::ComputeType::kComputeReduce},
    {Any::Type, af::ComputeType::kComputeReduce},

    {Broadcast::Type, af::ComputeType::kComputeBroadcast},
    {RemovePad::Type, af::ComputeType::kComputeElewise},
    {Pad::Type, af::ComputeType::kComputeElewise},
    {Round::Type, af::ComputeType::kComputeElewise},

    {Cast::Type, af::ComputeType::kComputeElewise},
    {Abs::Type, af::ComputeType::kComputeElewise},
    {Neg::Type, af::ComputeType::kComputeElewise},
    {Exp::Type, af::ComputeType::kComputeElewise},
    {Sqrt::Type, af::ComputeType::kComputeElewise},
    {Rsqrt::Type, af::ComputeType::kComputeElewise},
    {Relu::Type, af::ComputeType::kComputeElewise},
    {Reciprocal::Type, af::ComputeType::kComputeElewise},
    {Erf::Type, af::ComputeType::kComputeElewise},
    {Sign::Type, af::ComputeType::kComputeElewise},
    {Tanh::Type, af::ComputeType::kComputeElewise},
    {Isnan::Type, af::ComputeType::kComputeElewise},
    {IsFinite::Type, af::ComputeType::kComputeElewise},
    {Ln::Type, af::ComputeType::kComputeElewise},
    {LogicalNot::Type, af::ComputeType::kComputeElewise},

    {Add::Type, af::ComputeType::kComputeElewise},
    {Sub::Type, af::ComputeType::kComputeElewise},
    {Mul::Type, af::ComputeType::kComputeElewise},
    {Div::Type, af::ComputeType::kComputeElewise},
    {TrueDiv::Type, af::ComputeType::kComputeElewise},
    {Minimum::Type, af::ComputeType::kComputeElewise},
    {Maximum::Type, af::ComputeType::kComputeElewise},
    {LogicalOr::Type, af::ComputeType::kComputeElewise},
    {LogicalAnd::Type, af::ComputeType::kComputeElewise},

    {Ge::Type, af::ComputeType::kComputeElewise},
    {Eq::Type, af::ComputeType::kComputeElewise},
    {Ne::Type, af::ComputeType::kComputeElewise},
    {Gt::Type, af::ComputeType::kComputeElewise},
    {Le::Type, af::ComputeType::kComputeElewise},
    {Lt::Type, af::ComputeType::kComputeElewise},
    {Broadcast::Type, af::ComputeType::kComputeElewise},
    {Sigmoid::Type, af::ComputeType::kComputeElewise},
    {Concat::Type, af::ComputeType::kComputeConcat},
    {Gather::Type, af::ComputeType::kComputeGather},
    {IndirectLoad::Type, af::ComputeType::kComputeLoad},

    {Where::Type, af::ComputeType::kComputeElewise},
    {Select::Type, af::ComputeType::kComputeElewise},
    {ClipByValue::Type, af::ComputeType::kComputeElewise},
    {Pow::Type, af::ComputeType::kComputeElewise},
    {Transpose::Type, af::ComputeType::kComputeTranspose},
    {BitwiseAnd::Type, af::ComputeType::kComputeElewise},
    {LeakyRelu::Type, af::ComputeType::kComputeElewise},
    {FloorDiv::Type, af::ComputeType::kComputeElewise},
    {Gelu::Type, af::ComputeType::kComputeElewise},
    {Axpy::Type, af::ComputeType::kComputeElewise},
    {Split::Type, af::ComputeType::kComputeSplit},
    {MatMul::Type, af::ComputeType::kComputeCube},
    {MatMulBias::Type, af::ComputeType::kComputeCube},
    {MatMulOffset::Type, af::ComputeType::kComputeCube},
    {MatMulOffsetBias::Type, af::ComputeType::kComputeCube},
    {BatchMatMul::Type, af::ComputeType::kComputeCube},
    {BatchMatMulBias::Type, af::ComputeType::kComputeCube},
    {BatchMatMulOffset::Type, af::ComputeType::kComputeCube},
    {BatchMatMulOffsetBias::Type, af::ComputeType::kComputeCube},
    {Conv2D::Type, af::ComputeType::kComputeCube},
    {Conv2DBias::Type, af::ComputeType::kComputeCube},
    {Conv2DOffset::Type, af::ComputeType::kComputeCube},
    {Conv2DOffsetBias::Type, af::ComputeType::kComputeCube},
};

static const std::map<af::ComputeType, Completer> kComputeTypeToCompleter = {
    {af::ComputeType::kComputeInvalid, {&CompleteDataApiInfo}},
    {af::ComputeType::kComputeLoad, {&CompleteLoadApiInfo}},
    {af::ComputeType::kComputeStore, {&CompleteStoreApiInfo}},
    {af::ComputeType::kComputeReduce, {&CompleteReduceApiInfo}},
    {af::ComputeType::kComputeBroadcast, {&CompleteBroadcastApiInfo}},
    {af::ComputeType::kComputeElewise, {&CompleteElewiseApiInfo}},
    {af::ComputeType::kComputeConcat, {&CompleteConcatApiInfo}},
    {af::ComputeType::kComputeGather, {&CompleteGatherApiInfo}},
    {af::ComputeType::kComputeTranspose, {&CompleteTransposeApiInfo}},
    {af::ComputeType::kComputeSplit, {&CompleteSplitApiInfo}},
    {af::ComputeType::kComputeCube, {&CompleteCubeApiInfo}},
};

Status AscGraphInfoComplete::CompleteApiInfo(const af::AscGraph &optimize_graph) {
  for (auto node : optimize_graph.GetAllNodes()) {
    auto node_compute_type = &node->attr.api.compute_type;
    if (*node_compute_type >= af::ComputeType::kComputeInvalid) {
      auto item = kOpTypeToComputeType.find(node->GetType());
      if (item != kOpTypeToComputeType.end()) {
        *node_compute_type = item->second;
      }
    }
    auto it = kComputeTypeToCompleter.find(*node_compute_type);
    GE_ASSERT_TRUE((it != kComputeTypeToCompleter.end()), "CompleteApiInfo unsupported node name:[%s], type: [%s].",
                   node->GetNamePtr(), node->GetTypePtr());
    it->second.complete_api_info(node);
  }
  return af::SUCCESS;
}

void AscGraphInfoComplete::AppendOriginalSizeVar(const af::AscGraph &graph, SizeVarSet &size_vars) {
  auto axes = graph.GetAllAxis();
  for (const auto &axis : axes) {
    InsertFreeSymbolsIntoVarSet(axis->size, size_vars);
  }
  auto all_nodes = graph.GetAllNodes();
  for (const auto &node : all_nodes) {
    if (!af::ops::IsOps<Nddma>(node) && !af::ops::IsOps<Store>(node) && !af::ops::IsOps<Load>(node) &&
        !af::ops::IsOps<Gather>(node)) {
      continue;
    }

    af::Expression cur_load_offset;
    if (GetNodeIrAttrOffset(node, cur_load_offset) == af::SUCCESS) {
      InsertFreeSymbolsIntoVarSet(cur_load_offset, size_vars);
    }

    if (af::ops::IsOps<Gather>(node)) {
      for (const auto &exp : node->inputs[0].attr.repeats) {
        InsertFreeSymbolsIntoVarSet(exp, size_vars);
      }
      for (const auto &exp : node->inputs[1].attr.repeats) {
        InsertFreeSymbolsIntoVarSet(exp, size_vars);
      }
    }

    for (const auto &exp : node->outputs[0].attr.repeats) {
      InsertFreeSymbolsIntoVarSet(exp, size_vars);
    }
    for (const auto &exp : node->outputs[0].attr.strides) {
      InsertFreeSymbolsIntoVarSet(exp, size_vars);
    }
  }
}
}  // namespace optimize
