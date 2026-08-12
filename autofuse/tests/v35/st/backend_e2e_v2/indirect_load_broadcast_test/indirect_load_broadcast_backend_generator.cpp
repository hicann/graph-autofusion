/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <sstream>

#include "indirect_load_backend_generator_common.h"

#ifndef IL_CLEAR_BROADCAST_SOURCE_VIEW
#define IL_CLEAR_BROADCAST_SOURCE_VIEW 0
#endif
#ifndef IL_INPUT_BROADCAST
#define IL_INPUT_BROADCAST 0
#endif
#ifndef IL_INDEX_BROADCAST
#define IL_INDEX_BROADCAST 0
#endif

namespace {
constexpr int64_t kOutputS0 = 4;
constexpr int64_t kOutputS1 = 5;
constexpr int64_t kOutputS2 = 4;
constexpr int64_t kOutputS3 = 16;
constexpr std::array<int64_t, 4> kOutputShape = {kOutputS0, kOutputS1, kOutputS2, kOutputS3};
constexpr int32_t kInputElementCount = IL_HAS_INPUT_ELEMENT;
constexpr int32_t kIndexElementCount = IL_HAS_INDEX_ELEMENT;
constexpr bool kHasOutputRelu = IL_HAS_OUTPUT_RELU;
constexpr bool kInputBroadcast = IL_INPUT_BROADCAST;
constexpr bool kIndexBroadcast = IL_INDEX_BROADCAST;
constexpr uint32_t kBroadcastAxesMask = IL_BROADCAST_AXES_MASK;
constexpr bool kClearBroadcastSourceView = IL_CLEAR_BROADCAST_SOURCE_VIEW;
constexpr bool kExpectSimt = IL_EXPECT_SIMT;
constexpr bool kExpectSk = IL_EXPECT_SK;

using indirect_load_test::SetView;

struct TensorView {
  std::vector<af::AxisId> axes;
  std::vector<af::Expression> repeats;
  std::vector<af::Expression> strides;
  af::DataType dtype;
};

struct BroadcastGraphView {
  std::shared_ptr<af::AscGraph> graph;
  TensorView input;
  TensorView output;
  TensorView input_source;
  TensorView index_source;
  TensorView input_broadcast;
  TensorView index_broadcast;
};

template <typename Op>
void SetView(Op &op, const TensorView &view) {
  SetView(op, view.axes, view.repeats, view.strides, view.dtype);
}

template <typename Op>
void ClearView(Op &op) {
  op.y.axis->clear();
  op.y.repeats->clear();
  op.y.strides->clear();
}

std::vector<af::Expression> MakeDenseStrides(const std::vector<af::Expression> &repeats) {
  std::vector<af::Expression> strides(repeats.size(), af::ops::One);
  af::Expression stride = af::ops::One;
  for (size_t index = repeats.size(); index > 0UL; --index) {
    const size_t dim = index - 1UL;
    strides[dim] = stride;
    stride = stride * repeats[dim];
  }
  return strides;
}

std::array<int64_t, 4> MakeLogicalStrides(bool broadcast) {
  std::array<int64_t, 4> strides{};
  int64_t stride = 1;
  for (size_t index = kOutputShape.size(); index > 0UL; --index) {
    const size_t dim = index - 1UL;
    const bool is_broadcast_axis = broadcast && (kBroadcastAxesMask & (1U << dim)) != 0U;
    strides[dim] = is_broadcast_axis ? 0 : stride;
    if (!is_broadcast_axis) {
      stride *= kOutputShape[dim];
    }
  }
  return strides;
}

std::string MakeShapeArgs(bool is_sk) {
  const auto input_strides = MakeLogicalStrides(kInputBroadcast);
  const auto index_strides = MakeLogicalStrides(kIndexBroadcast);
  std::ostringstream stream;
  if (is_sk) {
    stream << ", 4";
  }
  for (const int64_t size : kOutputShape) {
    stream << ", " << size;
  }
  for (const int64_t stride : input_strides) {
    stream << ", " << stride;
  }
  for (const int64_t stride : index_strides) {
    stream << ", " << stride;
  }
  stream << ");";
  return stream.str();
}

BroadcastGraphView CreateGraphView() {
  BroadcastGraphView view;
  view.graph = std::make_shared<af::AscGraph>("indirect_load_broadcast_test");
  std::vector<af::AxisId> input_source_axis_candidates;
  std::vector<af::AxisId> index_source_axis_candidates;
  for (size_t dim = 0UL; dim < kOutputShape.size(); ++dim) {
    const auto input_size = view.graph->CreateSizeVar(kOutputShape[dim]);
    const auto output_size = view.graph->CreateSizeVar(kOutputShape[dim]);
    view.input.repeats.emplace_back(input_size);
    view.output.repeats.emplace_back(output_size);
    view.input.axes.emplace_back(view.graph->CreateAxis(("z" + std::to_string(dim)).c_str(), input_size).id);
    view.output.axes.emplace_back(view.graph->CreateAxis(("z" + std::to_string(dim + 4UL)).c_str(), output_size).id);
    input_source_axis_candidates.emplace_back(
        view.graph->CreateAxis(("z" + std::to_string(dim) + "_input").c_str(), af::ops::One).id);
    index_source_axis_candidates.emplace_back(
        view.graph->CreateAxis(("z" + std::to_string(dim + 4UL) + "_index").c_str(), af::ops::One).id);
  }
  view.input.dtype = af::DT_FLOAT16;
  view.output.dtype = af::DT_INT64;
  view.input_source = view.input;
  view.index_source = view.output;
  for (size_t dim = 0UL; dim < kOutputShape.size(); ++dim) {
    if ((kBroadcastAxesMask & (1U << dim)) != 0U) {
      view.input_source.axes[dim] = input_source_axis_candidates[dim];
      view.index_source.axes[dim] = index_source_axis_candidates[dim];
      view.input_source.repeats[dim] = af::ops::One;
      view.index_source.repeats[dim] = af::ops::One;
    }
  }
  view.input.strides = MakeDenseStrides(view.input.repeats);
  view.output.strides = MakeDenseStrides(view.output.repeats);
  view.input_source.strides = MakeDenseStrides(view.input_source.repeats);
  view.index_source.strides = MakeDenseStrides(view.index_source.repeats);
  view.input_broadcast = view.input;
  view.index_broadcast = view.output;
  for (size_t dim = 0UL; dim < kOutputShape.size(); ++dim) {
    view.input_broadcast.strides[dim] =
        (kBroadcastAxesMask & (1U << dim)) == 0U ? view.input_source.strides[dim] : af::ops::Zero;
    view.index_broadcast.strides[dim] =
        (kBroadcastAxesMask & (1U << dim)) == 0U ? view.index_source.strides[dim] : af::ops::Zero;
  }
  return view;
}

template <typename Destination>
void ConnectAbsChain(const std::shared_ptr<af::AscGraph> &graph, const char *prefix, int32_t count,
                     const af::AscOpOutput &source, const TensorView &view, Destination &destination) {
  std::vector<std::unique_ptr<af::ascir_op::Abs>> elements;
  for (int32_t i = 0; i < count; ++i) {
    const auto name = std::string(prefix) + std::to_string(i);
    auto element = std::make_unique<af::ascir_op::Abs>(name.c_str());
    graph->AddNode(*element);
    element->attr.api.compute_type = af::ComputeType::kComputeElewise;
    element->x = elements.empty() ? source : elements.back()->y;
    SetView(*element, view);
    elements.emplace_back(std::move(element));
  }
  destination = elements.empty() ? source : elements.back()->y;
}

void BuildInputPath(const BroadcastGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Data x("x");
  view.graph->AddNode(x);
  x.ir_attr.SetIndex(0);
  SetView(x, kInputBroadcast ? view.input_source : view.input);
  if (kClearBroadcastSourceView && kInputBroadcast) {
    ClearView(x);
  }
  af::ascir_op::Load input_load("input_load");
  view.graph->AddNode(input_load);
  input_load.x = x.y;
  SetView(input_load, kInputBroadcast ? view.input_source : view.input);
  if (kInputBroadcast) {
    af::ascir_op::Broadcast broadcast("input_broadcast");
    view.graph->AddNode(broadcast);
    broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
    broadcast.x = input_load.y;
    SetView(broadcast, view.input_broadcast);
    ConnectAbsChain(view.graph, "input_abs_", kInputElementCount, broadcast.y, view.input_broadcast, indirect_load.x1);
  } else {
    indirect_load.x1 = input_load.y;
  }
}

void BuildIndexPath(const BroadcastGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Data index("index");
  view.graph->AddNode(index);
  index.ir_attr.SetIndex(1);
  SetView(index, kIndexBroadcast ? view.index_source : view.output);
  if (kClearBroadcastSourceView && kIndexBroadcast) {
    ClearView(index);
  }
  af::ascir_op::Load index_load("index_load");
  view.graph->AddNode(index_load);
  index_load.x = index.y;
  SetView(index_load, kIndexBroadcast ? view.index_source : view.output);
  if (kIndexBroadcast) {
    af::ascir_op::Broadcast broadcast("index_broadcast");
    view.graph->AddNode(broadcast);
    broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
    broadcast.x = index_load.y;
    SetView(broadcast, view.index_broadcast);
    ConnectAbsChain(view.graph, "index_abs_", kIndexElementCount, broadcast.y, view.index_broadcast, indirect_load.x2);
  } else {
    ConnectAbsChain(view.graph, "index_abs_", kIndexElementCount, index_load.y, view.output, indirect_load.x2);
  }
}

void BuildOutputPath(const BroadcastGraphView &view, af::ascir_op::IndirectLoad &indirect_load) {
  indirect_load_test::BuildOutputPath(view.graph, indirect_load, view.output.axes, view.output.repeats,
                                      view.output.strides, kHasOutputRelu);
}

af::ComputeGraphPtr CreateGraph() {
  indirect_load_test::BackendGraph backend("indirect_load_broadcast_test", "data0", "data1", af::DT_FLOAT16);
  return backend.Finalize(
      indirect_load_test::CreateSubGraph(CreateGraphView(), BuildInputPath, BuildIndexPath, BuildOutputPath), "output");
}

void CheckSkKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find("// IndirectLoad SK"), std::string::npos);
  EXPECT_EQ(kernel.find("// IndirectLoad SIMD"), std::string::npos);
  EXPECT_EQ(kernel.find("// IndirectLoad SIMT"), std::string::npos);
  EXPECT_EQ(kernel.find("BroadcastExtend<"), std::string::npos);
  EXPECT_NE(kernel.find(MakeShapeArgs(true)), std::string::npos);
}

void CheckSimtKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find("// IndirectLoad SIMT"), std::string::npos);
  EXPECT_NE(kernel.find("IndirectLoadSimt<"), std::string::npos);
  EXPECT_EQ(kernel.find("// IndirectLoad SIMD"), std::string::npos);
  EXPECT_NE(kernel.find(MakeShapeArgs(false)), std::string::npos);
  EXPECT_EQ(kernel.find("AscendC::BroadcastExtend<"), std::string::npos);
}

void CheckSimdElements(const std::string &kernel) {
  const auto input_abs_pos = kernel.find("Abs(");
  const auto indirect_load_pos = kernel.find("// IndirectLoad SIMD");
  ASSERT_NE(indirect_load_pos, std::string::npos);
  if (kInputElementCount > 0 || kIndexElementCount > 0) {
    ASSERT_NE(input_abs_pos, std::string::npos);
    EXPECT_LT(input_abs_pos, indirect_load_pos);
  } else {
    EXPECT_EQ(input_abs_pos, std::string::npos);
  }
  const auto output_relu_pos = kernel.find("Relu(");
  EXPECT_EQ(output_relu_pos == std::string::npos, !kHasOutputRelu);
}

void CheckSimdKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find("// IndirectLoad SIMD"), std::string::npos);
  EXPECT_NE(kernel.find("IndirectLoadSimd<"), std::string::npos);
  EXPECT_NE(kernel.find(MakeShapeArgs(false)), std::string::npos);
  if (!kIndexBroadcast && kBroadcastAxesMask == 2U) {
    EXPECT_NE(kernel.find("global_0[(int64_t)z4 * (int64_t)64 + 0 + 0]"), std::string::npos);
    EXPECT_EQ(kernel.find("global_0[(int64_t)z4 * (int64_t)64 + (int64_t)z5 * (int64_t)64"), std::string::npos);
  }
  EXPECT_EQ(kernel.find("AscendC::BroadcastExtend<"), std::string::npos);
  CheckSimdElements(kernel);
}

void CheckGeneratedKernel(const std::string &kernel, ascir::TemplateId template_id) {
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    CheckSkKernel(kernel);
  } else if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    CheckSimtKernel(kernel);
  } else {
    CheckSimdKernel(kernel);
  }
}
}  // namespace

using TestBackendIndirectLoadBroadcastE2e = indirect_load_test::PrecisionBackendE2e;

TEST_F(TestBackendIndirectLoadBroadcastE2e, IndirectLoadBroadcastCodegen) {
  const auto graph = CreateGraph();
  ASSERT_NE(graph, nullptr);
  const std::map<std::string, std::string> shape_info = {{"s0", "stub_s0"}, {"s1", "stub_s1"}, {"s2", "stub_s2"},
                                                         {"s3", "stub_s3"}, {"s4", "stub_s4"}, {"s5", "stub_s5"},
                                                         {"s6", "stub_s6"}, {"s7", "stub_s7"}};
  const auto expected_template = indirect_load_test::GetExpectedTemplate(kExpectSimt, kExpectSk);
  codegen::CodegenResult result;
  indirect_load_test::GenerateForTemplate(graph, shape_info, expected_template, result);
  CheckGeneratedKernel(result.kernel, expected_template);
  indirect_load_test::WriteGeneratedFiles(result);
}
