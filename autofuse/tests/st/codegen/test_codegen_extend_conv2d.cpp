/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "codegen_kernel.h"
#include "common_utils.h"
#include "schedule_result.h"
#include "common/platform_context.h"
#include "pyascir_types.h"

#include <Python.h>
#include <sstream>

using namespace af;
using namespace af::ascir_op;
using namespace codegen;
using namespace ascgen_utils;

namespace {
template <typename ConvOp>
void FillExtendConv2DIrAttr(ConvOp &conv, bool enable_relu0 = false) {
  conv.ir_attr.SetStrides({1, 1, 1, 1});
  conv.ir_attr.SetPads({0, 0, 0, 0});
  conv.ir_attr.SetDilations({1, 1, 1, 1});
  conv.ir_attr.SetGroups(1);
  conv.ir_attr.SetPad_mode("SPECIFIC");
  conv.ir_attr.SetData_format("NCHW");
  conv.ir_attr.SetOffset_x(0);
  conv.ir_attr.SetEnable_hf32(false);
  conv.ir_attr.SetFixed_shift_value(0);
  conv.ir_attr.SetRound_mode("rint");
  conv.ir_attr.SetEnable_relu0(enable_relu0);
  conv.attr.api.compute_type = af::ComputeType::kComputeCube;
}

void PrepareConvKernelInputs(Kernel &kernel, size_t input_num) {
  for (size_t i = 0; i < input_num; ++i) {
    kernel.inputs.emplace_back(GM_ADDR("input_" + std::to_string(i)));
  }
  kernel.outputs.emplace_back(GM_ADDR("output_0"));
}

::ascir::FusedScheduledResult MakeCubeScheduledResult() {
  ::ascir::FusedScheduledResult fused;
  fused.node_idx_to_scheduled_results.resize(1);
  fused.node_idx_to_scheduled_results[0].resize(1);
  fused.node_idx_to_scheduled_results[0][0].cube_type = ::ascir::CubeTemplateType::kUBFuse;
  return fused;
}

bool InitFusedScheduledResultType(PyTypeObject &type) {
  if (!Py_IsInitialized()) {
    Py_Initialize();
  }
  if (type.tp_name == nullptr) {
    type.tp_name = "FusedScheduledResult";
    type.tp_basicsize = sizeof(pyascir::FusedScheduledResult::Object);
    type.tp_itemsize = 0;
    type.tp_dealloc = pyascir::FusedScheduledResult::Dealloc;
    type.tp_flags = Py_TPFLAGS_DEFAULT;
    type.tp_new = pyascir::FusedScheduledResult::New;
    type.tp_init = pyascir::FusedScheduledResult::Init;
  }
  return PyType_Ready(&type) == 0;
}

af::AscGraph BuildExtendConv2DAttrGraph() {
  af::AscGraph graph("extend_conv2d_pyattr");
  Data data0("x", graph);
  Data data1("filter", graph);
  Load load0("load0");
  Load load1("load1");
  ExtendConv2D conv("extend_conv2d");
  graph.AddNode(load0);
  graph.AddNode(load1);
  graph.AddNode(conv);
  load0.x = data0.y;
  load1.x = data1.y;
  conv.x = load0.y;
  conv.filter = load1.y;
  conv.y.dtype = ge::DT_FLOAT16;
  FillExtendConv2DIrAttr(conv);
  return graph;
}

void AttachCubeImplGraph(pyascir::FusedScheduledResult::Object *self, const af::AscGraph &graph) {
  self->fused_schedule_result.node_idx_to_scheduled_results.resize(1);
  self->fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);
  auto &scheduled = self->fused_schedule_result.node_idx_to_scheduled_results[0][0];
  scheduled.cube_type = ::ascir::CubeTemplateType::kUBFuse;
  ::ascir::ScheduleGroup group;
  group.impl_graphs.push_back(graph);
  scheduled.schedule_groups.push_back(group);
}

void ExpectExtendConv2DCubeAttrKeys(PyObject *attrs) {
  PyObject *cube_attrs = PyDict_GetItemString(attrs, "cube_attributes");
  ASSERT_NE(cube_attrs, nullptr);
  PyObject *is_extend = PyDict_GetItemString(cube_attrs, "is_extend_conv2d");
  ASSERT_NE(is_extend, nullptr);
  EXPECT_EQ(is_extend, Py_True);
  PyObject *has_scale0 = PyDict_GetItemString(cube_attrs, "has_scale0");
  ASSERT_NE(has_scale0, nullptr);
  EXPECT_EQ(has_scale0, Py_False);
  PyObject *round_mode = PyDict_GetItemString(cube_attrs, "round_mode");
  ASSERT_NE(round_mode, nullptr);
  EXPECT_STREQ(PyUnicode_AsUTF8(round_mode), "rint");
}
}  // namespace

TEST(CodegenExtendConv2D, KernelFuncDeclareContainsNewConv2DTemplateArgs) {
  const auto fused = MakeCubeScheduledResult();
  const std::string decl = Kernel::KernelFuncDeclare("ExtendConv2DGraph", fused, false, false, true);
  EXPECT_NE(decl.find("SmallKernel"), std::string::npos);
  EXPECT_NE(decl.find("BatchOne"), std::string::npos);
}

TEST(CodegenExtendConv2D, GenCubeTilingFuncCallPlainExtendConv2D) {
  af::AscGraph graph("extend_conv2d");
  ExtendConv2D conv("extend_conv2d");
  graph.AddNode(conv);
  FillExtendConv2DIrAttr(conv);

  Kernel kernel("extend_conv2d_kernel");
  PrepareConvKernelInputs(kernel, 2U);

  const std::string call = kernel.GenCubeTilingFuncCall(graph);
  EXPECT_NE(call.find("conv2d_v2"), std::string::npos);
  EXPECT_NE(call.find("SmallKernel"), std::string::npos);
  EXPECT_NE(call.find("input_0"), std::string::npos);
  EXPECT_NE(call.find("nullptr"), std::string::npos);

  const std::string common_call = kernel.GenCubeCommonTilingSingleFuncCall(graph);
  EXPECT_NE(common_call.find("conv2d_v2"), std::string::npos);
  EXPECT_NE(common_call.find("input_0"), std::string::npos);
}

TEST(CodegenExtendConv2D, GetCubeAttributesExportsExtendConv2DFields) {
  auto &type = pyascir::FusedScheduledResult::type;
  ASSERT_TRUE(InitFusedScheduledResultType(type));
  ge::PlatformContext::GetInstance().SetPlatform("2201");

  const auto graph = BuildExtendConv2DAttrGraph();
  PyObject *obj = pyascir::FusedScheduledResult::New(&type, nullptr, nullptr);
  ASSERT_NE(obj, nullptr);
  auto *self = reinterpret_cast<pyascir::FusedScheduledResult::Object *>(obj);
  AttachCubeImplGraph(self, graph);

  PyObject *attrs = pyascir::FusedScheduledResult::GetCubeAttributes(obj);
  ASSERT_NE(attrs, nullptr);
  ASSERT_EQ(PyErr_Occurred(), nullptr);
  ExpectExtendConv2DCubeAttrKeys(attrs);

  Py_DECREF(attrs);
  pyascir::FusedScheduledResult::Dealloc(obj);
}

TEST(CodegenExtendConv2D, GenCubeTilingFuncCallBiasScaleAndDynamic) {
  af::AscGraph graph("extend_conv2d_bias_scale");
  ExtendConv2DBiasScale conv("extend_conv2d_bias_scale");
  graph.AddNode(conv);
  FillExtendConv2DIrAttr(conv, true);

  EXPECT_TRUE(IsConv2DGraphType(graph));
  EXPECT_TRUE(IsConv2DTypeWithBias(graph));
  EXPECT_TRUE(IsConv2DTypeWithScale0(graph));

  Kernel kernel("extend_conv2d_bias_scale_kernel");
  PrepareConvKernelInputs(kernel, 4U);

  const std::string call = kernel.GenCubeTilingFuncCall(graph, true);
  EXPECT_NE(call.find("conv2d_v2"), std::string::npos);
  EXPECT_NE(call.find("input_2"), std::string::npos);
  EXPECT_NE(call.find("input_3"), std::string::npos);
  EXPECT_NE(call.find("CV_FUSION_ADDR"), std::string::npos);

  kernel.outputs.clear();
  const std::string common_call = kernel.GenCubeCommonTilingSingleFuncCall(graph, "output_override");
  EXPECT_NE(common_call.find("input_2"), std::string::npos);
  EXPECT_NE(common_call.find("output_override"), std::string::npos);
}

TEST(CodegenExtendConv2D, GenerateVecFuncOfCVFusionUsesNL0) {
  ge::PlatformContext::GetInstance().SetPlatform("2201");

  af::AscGraph graph("cv_ub_fuse");
  Data data0("x", graph);
  Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.y.dtype = ge::DT_FLOAT16;

  auto load_node = graph.FindNode("load0");
  ASSERT_NE(load_node, nullptr);
  load_node->outputs[0].attr.mem.tensor_id = 1;
  load_node->outputs[0].attr.dtype = ge::DT_FLOAT16;

  Kernel kernel("cv_ub_fuse_kernel");
  kernel.tpipe.cube_output_tensor_id = 1;
  ASSERT_EQ(kernel.tpipe.AddTensor(load_node->outputs[0]), af::SUCCESS);

  std::stringstream ss;
  ASSERT_EQ(kernel.GenerateVecFuncOfCVFusion(ss, true, true), af::SUCCESS);
  EXPECT_NE(ss.str().find("tmpTilingData.nL0"), std::string::npos);
}
