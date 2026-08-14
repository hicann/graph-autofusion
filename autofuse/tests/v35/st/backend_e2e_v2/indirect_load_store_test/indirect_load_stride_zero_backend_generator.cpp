/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indirect_load_backend_generator_common.h"

namespace {
constexpr uint32_t kInputZeroStrideMask = IL_INPUT_ZERO_STRIDE_MASK;
constexpr uint32_t kIndexZeroStrideMask = IL_INDEX_ZERO_STRIDE_MASK;
constexpr int32_t kInputElementCount = IL_HAS_INPUT_ELEMENT;
constexpr int32_t kIndexElementCount = IL_HAS_INDEX_ELEMENT;
constexpr bool kExpectSimt = IL_EXPECT_SIMT;
constexpr bool kExpectSk = IL_EXPECT_SK;

using indirect_load_test::SetView;

struct StrideZeroGraphView {
  std::shared_ptr<af::AscGraph> graph;
  std::vector<af::AxisId> axes;
  std::vector<af::Expression> repeats;
  std::vector<af::Expression> dense_strides;
  std::vector<af::Expression> input_strides;
  std::vector<af::Expression> index_strides;
};

std::vector<af::Expression> MakeStrides(const std::vector<af::Expression> &repeats, uint32_t zero_stride_mask) {
  std::vector<af::Expression> strides(repeats.size(), af::ops::Zero);
  af::Expression stride = af::ops::One;
  for (size_t index = repeats.size(); index > 0UL; --index) {
    const size_t dim = index - 1UL;
    if ((zero_stride_mask & (1U << dim)) == 0U) {
      strides[dim] = stride;
      stride = stride * repeats[dim];
    }
  }
  return strides;
}

StrideZeroGraphView CreateGraphView() {
  StrideZeroGraphView view;
  view.graph = std::make_shared<af::AscGraph>("indirect_load_stride_zero_test");
  const std::vector<int64_t> shape = {4, 5, 4, 16};
  for (size_t dim = 0; dim < shape.size(); ++dim) {
    const auto size = view.graph->CreateSizeVar(shape[dim]);
    view.repeats.emplace_back(size);
    view.axes.emplace_back(view.graph->CreateAxis(("z" + std::to_string(dim)).c_str(), size).id);
  }
  view.dense_strides = MakeStrides(view.repeats, 0U);
  view.input_strides = MakeStrides(view.repeats, kInputZeroStrideMask);
  view.index_strides = MakeStrides(view.repeats, kIndexZeroStrideMask);
  return view;
}

void BuildInputPath(const StrideZeroGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Data x("x");
  view.graph->AddNode(x);
  x.ir_attr.SetIndex(0);
  SetView(x, view.axes, view.repeats, view.input_strides, af::DT_FLOAT16);
  af::ascir_op::Load input_load("input_load");
  view.graph->AddNode(input_load);
  input_load.x = x.y;
  SetView(input_load, view.axes, view.repeats, view.input_strides, af::DT_FLOAT16);

  std::vector<std::unique_ptr<af::ascir_op::Abs>> input_elements;
  for (int32_t i = 0; i < kInputElementCount; ++i) {
    const auto name = "input_abs_" + std::to_string(i);
    auto input_abs = std::make_unique<af::ascir_op::Abs>(name.c_str());
    view.graph->AddNode(*input_abs);
    input_abs->attr.api.compute_type = af::ComputeType::kComputeElewise;
    input_abs->x = i == 0 ? input_load.y : input_elements.back()->y;
    SetView(*input_abs, view.axes, view.repeats, view.input_strides, af::DT_FLOAT16);
    input_elements.emplace_back(std::move(input_abs));
  }
  indirect_load.x1 = input_elements.empty() ? input_load.y : input_elements.back()->y;
}

void BuildIndexPath(const StrideZeroGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Data index("index");
  view.graph->AddNode(index);
  index.ir_attr.SetIndex(1);
  SetView(index, view.axes, view.repeats, view.index_strides, af::DT_INT64);
  af::ascir_op::Load index_load("index_load");
  view.graph->AddNode(index_load);
  index_load.x = index.y;
  SetView(index_load, view.axes, view.repeats, view.index_strides, af::DT_INT64);
  std::vector<std::unique_ptr<af::ascir_op::Abs>> index_elements;
  for (int32_t i = 0; i < kIndexElementCount; ++i) {
    const auto name = "index_abs_" + std::to_string(i);
    auto index_abs = std::make_unique<af::ascir_op::Abs>(name.c_str());
    view.graph->AddNode(*index_abs);
    index_abs->attr.api.compute_type = af::ComputeType::kComputeElewise;
    index_abs->x = i == 0 ? index_load.y : index_elements.back()->y;
    SetView(*index_abs, view.axes, view.repeats, view.index_strides, af::DT_INT64);
    index_elements.emplace_back(std::move(index_abs));
  }
  indirect_load.x2 = index_elements.empty() ? index_load.y : index_elements.back()->y;
}

void BuildOutputPath(const StrideZeroGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  indirect_load_test::BuildOutputPath(view.graph, indirect_load, view.axes, view.repeats, view.dense_strides, true);
}

af::ComputeGraphPtr CreateGraph() {
  indirect_load_test::BackendGraph backend("indirect_load_stride_zero_test", "data0", "data1", af::DT_FLOAT16);
  return backend.Finalize(
      indirect_load_test::CreateSubGraph(CreateGraphView(), BuildInputPath, BuildIndexPath, BuildOutputPath), "output");
}
}  // namespace

using TestBackendIndirectLoadStrideZeroE2e = indirect_load_test::PrecisionBackendE2e;

TEST_F(TestBackendIndirectLoadStrideZeroE2e, GeneratesSelectedTemplateWithoutBroadcastMaterialization) {
  const auto graph = CreateGraph();
  ASSERT_NE(graph, nullptr);
  const std::map<std::string, std::string> shape_info;
  const auto expected_template = indirect_load_test::GetExpectedTemplate(kExpectSimt, kExpectSk);
  codegen::CodegenResult result;
  indirect_load_test::GenerateForTemplate(graph, shape_info, expected_template, result);
  EXPECT_NE(result.kernel.find(indirect_load_test::GetTemplateMarker(expected_template)), std::string::npos);
  EXPECT_EQ(result.kernel.find("BroadcastExtend<"), std::string::npos);
  indirect_load_test::WriteGeneratedFiles(result);
}
