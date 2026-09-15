/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"

#include "ascir_utils.h"
#include "ascendc_graph_txt_dumper.h"
#include "ascir.h"
#include "ascir_ops.h"

namespace af {
namespace ascir {

class AscirMetaUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test: identifier rendering produces a stable, non-empty textual form
TEST_F(AscirMetaUtilsTest, IdentifierToStr_ShouldRenderNonEmpty) {
  EXPECT_FALSE(::ascir::utils::IdentifierToStr(0).empty());
  EXPECT_FALSE(::ascir::utils::IdentifierToStr(42).empty());
  EXPECT_EQ(::ascir::utils::IdentifierToStr(42), ::ascir::utils::IdentifierToStr(42));
}

// Test: dump prefix is empty while codegen-compile-debug is disabled, and stays deterministic
TEST_F(AscirMetaUtilsTest, GetDumpFilePrefix_ShouldBeEmptyWhenDumpDisabled) {
  const auto prefix = ::ascir::utils::GetDumpFilePrefix();
  EXPECT_EQ(prefix, ::ascir::utils::GetDumpFilePrefix());
}

// Test: dtype info lookup succeeds for common dtypes and fails gracefully for unknown ones
TEST_F(AscirMetaUtilsTest, GetDtypeInfo_ShouldResolveCommonDtypes) {
  const std::vector<ge::DataType> common_dtypes = {
      ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_INT8,   ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16,
      ge::DT_INT32, ge::DT_UINT32,  ge::DT_INT64, ge::DT_UINT64, ge::DT_BOOL,  ge::DT_DOUBLE};
  for (const auto dtype : common_dtypes) {
    EXPECT_NE(::ascir::dumper::GetDtypeInfo(dtype), nullptr);
  }
}

// Test: axis type priority/suffix are consistent lookups for the same type
TEST_F(AscirMetaUtilsTest, AxisTypeHelpers_ShouldBeConsistent) {
  const auto type = af::Axis::Type::kAxisTypeOriginal;
  EXPECT_GE(::ascir::dumper::GetAxisTypePriority(type), 0);
  EXPECT_FALSE(::ascir::dumper::GetAxisTypeSuffix(type).empty());
  EXPECT_EQ(::ascir::dumper::GetAxisTypePriority(type), ::ascir::dumper::GetAxisTypePriority(type));
}

// Test: tensor type parsing extracts the dtype fragment
TEST_F(AscirMetaUtilsTest, ExtractDtypeFromTensorType_ShouldExtractDtypeFragment) {
  const std::string tensor_type = "Tensor<fp16>";
  const auto dtype = ::ascir::dumper::ExtractDtypeFromTensorType(tensor_type);
  EXPECT_FALSE(dtype.empty());
  EXPECT_EQ(dtype, ::ascir::dumper::ExtractDtypeFromTensorType(tensor_type));
}

// Test: tensor type parsing extracts the axis list fragment
TEST_F(AscirMetaUtilsTest, ExtractAxisListFromTensorType_ShouldExtractAxisFragment) {
  const auto axes = ::ascir::dumper::ExtractAxisListFromTensorType("Tensor<fp16, ax0, ax1>");
  EXPECT_FALSE(axes.empty());
}

// Test: axis id maps are built from graph axes with stable names/types
TEST_F(AscirMetaUtilsTest, BuildAxisIdMaps_ShouldMapGraphAxes) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  const std::vector<af::AxisPtr> axes = graph.GetAllAxis();
  const auto id_to_name = ::ascir::dumper::BuildAxisIdToNameMap(axes);
  ASSERT_GE(id_to_name.size(), 1U);
  EXPECT_FALSE(id_to_name.at(z0.id).empty());
  const auto id_to_type = ::ascir::dumper::BuildAxisIdToTypeMap(axes);
  ASSERT_GE(id_to_type.size(), 1U);
}

// Test: DebugImplGraphStr renders node attributes end-to-end: exec condition, compute unit,
// queue/buffer memory info with hardware and position rendering
TEST_F(AscirMetaUtilsTest, DebugImplGraphStr_ShouldRenderNodeAndMemAttributes) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Abs abs("abs");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, af::Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  abs.x = load1.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  // non-default exec condition so that DebugStr renders it via ExecConditionToStr
  abs.attr.sched.exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, af::Symbol(1)};
  *abs.y.vectorized_axis = {z0.id, z1.id};
  // queue-typed output memory rendered through OutputQueueMemStr / MemHardwareToStr / PositionToStr
  abs.y.mem->alloc_type = af::AllocType::kAllocTypeQueue;
  abs.y.mem->hardware = af::MemHardware::kMemHardwareUB;
  abs.y.mem->position = af::Position::kPositionInvalid;
  abs.y.mem->reuse_id = 1;
  abs.y.que->id = 5;
  abs.y.que->depth = 2;

  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, af::Symbol(1)};
  // buffer-typed output memory rendered through OutputBufferMemStr
  store.y.mem->alloc_type = af::AllocType::kAllocTypeBuffer;
  store.y.mem->hardware = af::MemHardware::kMemHardwareUB;
  store.y.buf->id = 3;

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, af::Symbol(1)};

  const auto debug_str = ::ascir::utils::DebugImplGraphStr(graph);
  ASSERT_FALSE(debug_str.empty());
  EXPECT_NE(debug_str.find(".api.unit"), std::string::npos);
  EXPECT_NE(debug_str.find(".exec_condition"), std::string::npos);
  EXPECT_NE(debug_str.find("que_id=5"), std::string::npos);
  EXPECT_NE(debug_str.find("buf_id=3"), std::string::npos);
}

// Test: subgraph-mode dump walks the loop-execution chain (loop axes grouping, node loops)
TEST_F(AscirMetaUtilsTest, DumpLoopExecutionView_ShouldRenderSubgraphLoopChain) {
  // the "_VfSubgraph_" fragment switches the dumper into subgraph mode
  af::AscGraph graph("test_VfSubgraph_0");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Abs abs("abs");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, af::Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  abs.x = load1.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.attr.sched.loop_axis = {z0.id};
  abs.attr.sched.exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, af::Symbol(1)};
  *abs.y.vectorized_axis = {z0.id, z1.id};

  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, af::Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, af::Symbol(1)};

  const auto ctx = ::ascir::dumper::BuildDumpContext(graph);
  const auto text = ::ascir::dumper::DumpLoopExecutionView(graph, ctx);
  EXPECT_FALSE(text.empty());
}

// Test: regular-mode dump with a scalar node and a non-default exec condition renders node execution
TEST_F(AscirMetaUtilsTest, DumpLoopExecutionView_ShouldRenderScalarAndExecCondition) {
  af::AscGraph graph("regular_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Scalar sc("sc", graph);
  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Mul mul("mul");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  sc.attr.sched.axis = {z0.id, z1.id};
  sc.y.dtype = af::DT_FLOAT;
  *sc.y.axis = {};
  *sc.y.repeats = {};
  *sc.y.strides = {};

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, af::Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  mul.x1 = load1.y;
  mul.x2 = sc.y;
  mul.attr.sched.axis = {z0.id, z1.id};
  mul.attr.sched.exec_condition = af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
  mul.y.dtype = af::DT_FLOAT;
  *mul.y.axis = {z0.id, z1.id};
  *mul.y.repeats = {s0, s1};
  *mul.y.strides = {s1, af::Symbol(1)};
  *mul.y.vectorized_axis = {z0.id, z1.id};

  store.x = mul.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, af::Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, af::Symbol(1)};

  const auto ctx = ::ascir::dumper::BuildDumpContext(graph);
  const auto text = ::ascir::dumper::DumpLoopExecutionView(graph, ctx);
  EXPECT_FALSE(text.empty());
}

// Test: dtype variants flow through DtypeToStr and the normal-mem branch of OutputMemStr
TEST_F(AscirMetaUtilsTest, DebugImplGraphStr_ShouldRenderDtypeVariants) {
  const std::vector<af::DataType> dtypes = {af::DT_INT32, af::DT_BF16, af::DT_INT8, af::DT_BOOL};
  for (const auto dtype : dtypes) {
    af::AscGraph graph("t_dtype");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    af::ascir_op::Data x1("x1", graph);
    af::ascir_op::Load load1("load1");
    af::ascir_op::Abs abs("abs");
    af::ascir_op::Store store("store");
    af::ascir_op::Output y("y");
    x1.attr.sched.axis = {z0.id, z1.id};
    x1.y.dtype = dtype;
    *x1.y.axis = {z0.id, z1.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, af::Symbol(1)};
    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, z1.id};
    load1.y.dtype = dtype;
    *load1.y.axis = {z0.id, z1.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, af::Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, z1.id};
    abs.x = load1.y;
    abs.attr.sched.axis = {z0.id, z1.id};
    abs.y.dtype = dtype;
    *abs.y.axis = {z0.id, z1.id};
    *abs.y.repeats = {s0, s1};
    *abs.y.strides = {s1, af::Symbol(1)};
    *abs.y.vectorized_axis = {z0.id, z1.id};
    // normal-mem output with tensor id and reuse id to cover those rendering branches
    abs.y.mem->alloc_type = af::AllocType::kAllocTypeGlobal;
    abs.y.mem->hardware = af::MemHardware::kMemHardwareGM;
    abs.y.mem->position = af::Position::kPositionVecIn;
    abs.y.mem->tensor_id = 7;
    abs.y.mem->reuse_id = 2;
    store.x = abs.y;
    store.attr.sched.axis = {z0.id, z1.id};
    store.y.dtype = dtype;
    *store.y.axis = {z0.id, z1.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, af::Symbol(1)};
    y.x = store.y;
    y.attr.sched.axis = {z0.id, z1.id};
    y.y.dtype = dtype;
    *y.y.axis = {z0.id, z1.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, af::Symbol(1)};
    const auto text = ::ascir::utils::DebugImplGraphStr(graph);
    EXPECT_FALSE(text.empty());
    EXPECT_NE(text.find(".api.unit"), std::string::npos);
  }
}

// Test: a broadcast node in the dumped graph exercises the broadcast-related helper branches
TEST_F(AscirMetaUtilsTest, DumpLoopExecutionView_ShouldRenderBroadcastNode) {
  af::AscGraph graph("bcast_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Broadcast brc("brc");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, af::Symbol(1)};
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};
  brc.x = load1.y;
  brc.attr.sched.axis = {z0.id, z1.id};
  brc.y.dtype = af::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id};
  *brc.y.repeats = {s0, s1};
  *brc.y.strides = {s1, af::Symbol(1)};
  *brc.y.vectorized_axis = {z0.id, z1.id};
  store.x = brc.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, af::Symbol(1)};
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, af::Symbol(1)};
  const auto ctx = ::ascir::dumper::BuildDumpContext(graph);
  const auto text = ::ascir::dumper::DumpLoopExecutionView(graph, ctx);
  EXPECT_FALSE(text.empty());
}

// Test: enabling codegen-compile-debug via AUTOFUSE_DFX_FLAGS produces a real dump prefix
TEST_F(AscirMetaUtilsTest, GetDumpFilePrefix_ShouldBuildPrefixWhenDebugEnabled) {
  const std::string debug_root = "/tmp/ascir_ut_dump_" + std::to_string(getpid());
  setenv("AUTOFUSE_DFX_FLAGS", ("--codegen_compile_debug=true;--debug_dir=" + debug_root).c_str(), 1);
  const auto prefix = ::ascir::utils::GetDumpFilePrefix();
  unsetenv("AUTOFUSE_DFX_FLAGS");
  EXPECT_FALSE(prefix.empty());
  EXPECT_EQ(prefix.find(debug_root), 0U);
}

// Test: exec-condition variants on a scalar node cover every rendering branch
TEST_F(AscirMetaUtilsTest, DumpLoopExecutionView_ShouldRenderExecConditionVariants) {
  const std::vector<af::ExecuteCondition> conditions = {
      af::ExecuteCondition::kNoCache, af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis,
      af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis, af::ExecuteCondition::kConditionInvalid};
  for (const auto cond : conditions) {
    af::AscGraph graph("t_cond");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    af::ascir_op::Scalar sc("sc", graph);
    af::ascir_op::Data x1("x1", graph);
    af::ascir_op::Load load1("load1");
    af::ascir_op::Mul mul("mul");
    af::ascir_op::Store store("store");
    af::ascir_op::Output y("y");
    sc.attr.sched.axis = {z0.id, z1.id};
    sc.attr.sched.exec_condition = cond;
    sc.y.dtype = af::DT_FLOAT;
    *sc.y.axis = {};
    *sc.y.repeats = {};
    *sc.y.strides = {};
    x1.attr.sched.axis = {z0.id, z1.id};
    x1.y.dtype = af::DT_FLOAT;
    *x1.y.axis = {z0.id, z1.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, af::Symbol(1)};
    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, z1.id};
    load1.y.dtype = af::DT_FLOAT;
    *load1.y.axis = {z0.id, z1.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, af::Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, z1.id};
    mul.x1 = load1.y;
    mul.x2 = sc.y;
    mul.attr.sched.axis = {z0.id, z1.id};
    mul.y.dtype = af::DT_FLOAT;
    *mul.y.axis = {z0.id, z1.id};
    *mul.y.repeats = {s0, s1};
    *mul.y.strides = {s1, af::Symbol(1)};
    *mul.y.vectorized_axis = {z0.id, z1.id};
    // unset tensor id and negative reuse id cover the "omitted field" rendering branches
    mul.y.mem->alloc_type = af::AllocType::kAllocTypeQueue;
    mul.y.mem->hardware = af::MemHardware::kMemHardwareUB;
    mul.y.mem->position = af::Position::kPositionVecOut;
    mul.y.mem->tensor_id = af::kIdNone;
    mul.y.mem->reuse_id = -1;
    mul.y.que->id = 2;
    mul.y.que->depth = 1;
    store.x = mul.y;
    store.attr.sched.axis = {z0.id, z1.id};
    store.y.dtype = af::DT_FLOAT;
    *store.y.axis = {z0.id, z1.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, af::Symbol(1)};
    y.x = store.y;
    y.attr.sched.axis = {z0.id, z1.id};
    y.y.dtype = af::DT_FLOAT;
    *y.y.axis = {z0.id, z1.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, af::Symbol(1)};
    const auto ctx = ::ascir::dumper::BuildDumpContext(graph);
    const auto text = ::ascir::dumper::DumpLoopExecutionView(graph, ctx);
    EXPECT_FALSE(text.empty());
  }
}

// Test: a node with tmp buffer descriptors renders the tmp_buf block in verbose mode
TEST_F(AscirMetaUtilsTest, DebugImplGraphStr_ShouldRenderTmpBuffers) {
  af::AscGraph graph("t_tmpbuf");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Abs abs("abs");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, af::Symbol(1)};
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};
  abs.x = load1.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, af::Symbol(1)};
  *abs.y.vectorized_axis = {z0.id, z1.id};
  const auto node = graph.FindNode("abs");
  ASSERT_NE(node, nullptr);
  af::TmpBuffer tmp_buf;
  tmp_buf.id = 1;
  tmp_buf.buf_desc.size = af::Symbol(8192);
  node->attr.tmp_buffers = {tmp_buf};
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, af::Symbol(1)};
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, af::Symbol(1)};
  const auto text = ::ascir::utils::DebugImplGraphStr(graph);
  EXPECT_NE(text.find("tmp_buf"), std::string::npos);
}

// Final sweep: constant-size axes, workspace node, broadcast axis with zero stride and a
// buffer-typed output inside a subgraph-named graph tick the remaining rendering branches.
TEST_F(AscirMetaUtilsTest, DumpLoopExecutionView_FinalSweepRegular) {
  af::AscGraph graph("sweep_regular_graph");
  // constant axes: stride/repeat const-value branches and axis-size rendering
  auto c0 = graph.CreateSizeVar(4);
  auto c1 = graph.CreateSizeVar(8);
  auto z0 = graph.CreateAxis("z0", c0);
  auto z1 = graph.CreateAxis("z1", c1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Workspace ws("ws");
  af::ascir_op::Load load1("load1");
  af::ascir_op::Broadcast brc("brc");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  ws.attr.sched.axis = {z0.id, z1.id};
  ws.y.dtype = af::DT_FLOAT;
  *ws.y.axis = {z0.id, z1.id};
  *ws.y.repeats = {c0, c1};
  *ws.y.strides = {c1, af::Symbol(1)};

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {c0, c1};
  *x1.y.strides = {c1, af::Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {c0, c1};
  *load1.y.strides = {c1, af::Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  brc.x = load1.y;
  brc.attr.sched.axis = {z0.id, z1.id};
  brc.y.dtype = af::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id};
  *brc.y.repeats = {c0, c1};
  // zero stride on the broadcast axis marks the broadcast rendering path
  *brc.y.strides = {af::Symbol(0), af::Symbol(1)};
  *brc.y.vectorized_axis = {z0.id, z1.id};

  store.x = brc.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {c0, c1};
  *store.y.strides = {c1, af::Symbol(1)};
  store.y.mem->alloc_type = af::AllocType::kAllocTypeBuffer;
  store.y.mem->hardware = af::MemHardware::kMemHardwareUB;
  store.y.mem->position = af::Position::kPositionVecCalc;
  store.y.buf->id = 4;

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {c0, c1};
  *y.y.strides = {c1, af::Symbol(1)};

  const auto ctx = ::ascir::dumper::BuildDumpContext(graph);
  const auto text = ::ascir::dumper::DumpLoopExecutionView(graph, ctx);
  EXPECT_FALSE(text.empty());
  EXPECT_NE(text.find("store"), std::string::npos);
}
TEST_F(AscirMetaUtilsTest, DumpLoopExecutionView_FinalSweep) {
  af::AscGraph graph("sweep_VfSubgraph_0");
  // constant axes: stride/repeat const-value branches and axis-size rendering
  auto c0 = graph.CreateSizeVar(4);
  auto c1 = graph.CreateSizeVar(8);
  auto z0 = graph.CreateAxis("z0", c0);
  auto z1 = graph.CreateAxis("z1", c1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Workspace ws("ws");
  af::ascir_op::Load load1("load1");
  af::ascir_op::Broadcast brc("brc");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  ws.attr.sched.axis = {z0.id, z1.id};
  ws.y.dtype = af::DT_FLOAT;
  *ws.y.axis = {z0.id, z1.id};
  *ws.y.repeats = {c0, c1};
  *ws.y.strides = {c1, af::Symbol(1)};

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {c0, c1};
  *x1.y.strides = {c1, af::Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {c0, c1};
  *load1.y.strides = {c1, af::Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  brc.x = load1.y;
  brc.attr.sched.axis = {z0.id, z1.id};
  brc.y.dtype = af::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id};
  *brc.y.repeats = {c0, c1};
  // zero stride on the broadcast axis marks the broadcast rendering path
  *brc.y.strides = {af::Symbol(0), af::Symbol(1)};
  *brc.y.vectorized_axis = {z0.id, z1.id};

  store.x = brc.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {c0, c1};
  *store.y.strides = {c1, af::Symbol(1)};
  store.y.mem->alloc_type = af::AllocType::kAllocTypeBuffer;
  store.y.mem->hardware = af::MemHardware::kMemHardwareUB;
  store.y.mem->position = af::Position::kPositionVecCalc;
  store.y.buf->id = 4;

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {c0, c1};
  *y.y.strides = {c1, af::Symbol(1)};

  const auto ctx = ::ascir::dumper::BuildDumpContext(graph);
  const auto text = ::ascir::dumper::DumpLoopExecutionView(graph, ctx);
  EXPECT_FALSE(text.empty());
  EXPECT_NE(text.find("store"), std::string::npos);
}

// Direct probes of axis-type helpers across every enum value (including out-of-range casts)
TEST_F(AscirMetaUtilsTest, AxisTypeHelpers_AllValues) {
  const std::vector<af::Axis::Type> types = {af::Axis::Type::kAxisTypeOriginal, af::Axis::Type::kAxisTypeBlockInner,
                                             af::Axis::Type::kAxisTypeBlockOuter, static_cast<af::Axis::Type>(99)};
  for (const auto t : types) {
    (void)::ascir::dumper::GetAxisTypePriority(t);
    (void)::ascir::dumper::GetAxisTypeSuffix(t);
  }
}

// CollectInputNames on a node whose peers include an unconnected data node and a multi-output peer
TEST_F(AscirMetaUtilsTest, CollectInputNames_ShouldHandleUnconnectedAndIndexedPeers) {
  af::AscGraph graph("t_inputs");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  // an orphan data node stays unconnected: its consumer renders "nil"
  af::ascir_op::Data orphan("orphan", graph);
  orphan.attr.sched.axis = {z0.id, z1.id};
  orphan.y.dtype = af::DT_FLOAT;
  *orphan.y.axis = {z0.id, z1.id};
  *orphan.y.repeats = {s0, s1};
  *orphan.y.strides = {s1, af::Symbol(1)};

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::ArgMaxMultiRPhase1 p1("p1");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, af::Symbol(1)};
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};
  p1.x = load1.y;
  p1.attr.sched.axis = {z0.id, z1.id};
  p1.value.dtype = af::DT_FLOAT;
  *p1.value.axis = {z0.id, z1.id};
  *p1.value.repeats = {s0, af::Symbol(1)};
  *p1.value.strides = {af::Symbol(1), af::Symbol(0)};
  store.x = p1.value;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, af::Symbol(1)};
  *store.y.strides = {af::Symbol(1), af::Symbol(0)};
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, af::Symbol(1)};
  *y.y.strides = {af::Symbol(1), af::Symbol(0)};

  const auto node = graph.FindNode("p1");
  ASSERT_NE(node, nullptr);
  const auto names = ::ascir::dumper::CollectInputNames(graph, node);
  EXPECT_FALSE(names.empty());
}

// meta-95 push: dtype family expansion covers DtypeToStr int16/uint16/uint32/uint64 branches
TEST_F(AscirMetaUtilsTest, DebugImplGraphStr_ShouldRenderFullDtypeFamily) {
  const std::vector<af::DataType> dtypes = {af::DT_INT16, af::DT_UINT16, af::DT_UINT32, af::DT_UINT64, af::DT_UINT8};
  for (const auto dtype : dtypes) {
    af::AscGraph graph("t_dtypes");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    af::ascir_op::Data x1("x1", graph);
    af::ascir_op::Load load1("load1");
    af::ascir_op::Abs abs("abs");
    af::ascir_op::Store store("store");
    af::ascir_op::Output y("y");
    x1.attr.sched.axis = {z0.id, z1.id};
    x1.y.dtype = dtype;
    *x1.y.axis = {z0.id, z1.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, af::Symbol(1)};
    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, z1.id};
    load1.y.dtype = dtype;
    *load1.y.axis = {z0.id, z1.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, af::Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, z1.id};
    abs.x = load1.y;
    abs.attr.sched.axis = {z0.id, z1.id};
    abs.y.dtype = dtype;
    *abs.y.axis = {z0.id, z1.id};
    *abs.y.repeats = {s0, s1};
    *abs.y.strides = {s1, af::Symbol(1)};
    *abs.y.vectorized_axis = {z0.id, z1.id};
    store.x = abs.y;
    store.attr.sched.axis = {z0.id, z1.id};
    store.y.dtype = dtype;
    *store.y.axis = {z0.id, z1.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, af::Symbol(1)};
    y.x = store.y;
    y.attr.sched.axis = {z0.id, z1.id};
    y.y.dtype = dtype;
    *y.y.axis = {z0.id, z1.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, af::Symbol(1)};
    const auto debug_text = ::ascir::utils::DebugImplGraphStr(graph);
    EXPECT_NE(debug_text.find(".api.unit"), std::string::npos);
    const auto ctx = ::ascir::dumper::BuildDumpContext(graph);
    const auto dump_text = ::ascir::dumper::DumpLoopExecutionView(graph, ctx);
    EXPECT_FALSE(dump_text.empty());
  }
}

// meta-95 push: env parse variants (quoted/spaced dir), prefix caching and graph file dump
TEST_F(AscirMetaUtilsTest, DumpPrefixAndGraphFile_EnvVariants) {
  const std::string debug_root = "/tmp/ascir_ut_meta95_" + std::to_string(getpid());
  setenv("AUTOFUSE_DFX_FLAGS", ("--codegen_compile_debug=true;--debug_dir= \"" + debug_root + "/ \"").c_str(), 1);
  const auto prefix1 = ::ascir::utils::GetDumpFilePrefix();
  EXPECT_FALSE(prefix1.empty());
  const auto prefix2 = ::ascir::utils::GetDumpFilePrefix();
  EXPECT_EQ(prefix1, prefix2);

  af::AscGraph graph("t_file_dump");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Abs abs("abs");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, af::Symbol(1)};
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};
  abs.x = load1.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, af::Symbol(1)};
  *abs.y.vectorized_axis = {z0.id, z1.id};
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, af::Symbol(1)};
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, af::Symbol(1)};
  ::ascir::utils::DumpGraph(graph, "meta95", 0U, true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// meta-95 push: constant size vars render the CONST branch in verbose graph dump
TEST_F(AscirMetaUtilsTest, DebugImplGraphStr_ShouldRenderConstantSizeVars) {
  af::AscGraph graph("t_const_size");
  auto c0 = graph.CreateSizeVar(16);
  auto c1 = graph.CreateSizeVar(32);
  auto z0 = graph.CreateAxis("z0", c0);
  auto z1 = graph.CreateAxis("z1", c1);
  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Abs abs("abs");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {c0, c1};
  *x1.y.strides = {c1, af::Symbol(1)};
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {c0, c1};
  *load1.y.strides = {c1, af::Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};
  abs.x = load1.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {c0, c1};
  *abs.y.strides = {c1, af::Symbol(1)};
  *abs.y.vectorized_axis = {z0.id, z1.id};
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {c0, c1};
  *store.y.strides = {c1, af::Symbol(1)};
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {c0, c1};
  *y.y.strides = {c1, af::Symbol(1)};
  const auto text = ::ascir::utils::DebugImplGraphStr(graph);
  EXPECT_NE(text.find("CONST"), std::string::npos);
}

// meta-95 push: an unconnected second input renders "nil"; consuming a secondary output of a
// multi-output peer renders the indexed form
TEST_F(AscirMetaUtilsTest, NodeInputRendering_NilAndIndexedPeer) {
  af::AscGraph graph("t_nil");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Mul mul("mul");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, af::Symbol(1)};
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};
  mul.x1 = load1.y;
  mul.attr.sched.axis = {z0.id, z1.id};
  mul.y.dtype = af::DT_FLOAT;
  *mul.y.axis = {z0.id, z1.id};
  *mul.y.repeats = {s0, s1};
  *mul.y.strides = {s1, af::Symbol(1)};
  *mul.y.vectorized_axis = {z0.id, z1.id};
  store.x = mul.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, af::Symbol(1)};
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, af::Symbol(1)};
  const auto node = graph.FindNode("mul");
  ASSERT_NE(node, nullptr);
  const auto names = ::ascir::dumper::CollectInputNames(graph, node);
  EXPECT_FALSE(names.empty());
  const auto ctx = ::ascir::dumper::BuildDumpContext(graph);
  const auto text = ::ascir::dumper::DumpLoopExecutionView(graph, ctx);
  EXPECT_FALSE(text.empty());
}

// meta-95 push: subgraph mode with a valued scalar, a workspace node and exec conditions
TEST_F(AscirMetaUtilsTest, DumpLoopExecutionView_SubgraphScalarWorkspace) {
  af::AscGraph graph("meta95_VfSubgraph_0");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  af::ascir_op::Scalar sc("sc");
  sc.ir_attr.SetValue("2.5");
  af::ascir_op::Workspace ws("ws");
  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Mul mul("mul");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");
  sc.attr.sched.axis = {z0.id, z1.id};
  sc.attr.sched.exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  sc.y.dtype = af::DT_FLOAT;
  *sc.y.axis = {};
  *sc.y.repeats = {};
  *sc.y.strides = {};
  ws.attr.sched.axis = {z0.id, z1.id};
  ws.y.dtype = af::DT_FLOAT;
  *ws.y.axis = {z0.id, z1.id};
  *ws.y.repeats = {s0, s1};
  *ws.y.strides = {s1, af::Symbol(1)};
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, af::Symbol(1)};
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};
  mul.x1 = load1.y;
  mul.x2 = sc.y;
  mul.attr.sched.axis = {z0.id, z1.id};
  mul.y.dtype = af::DT_FLOAT;
  *mul.y.axis = {z0.id, z1.id};
  *mul.y.repeats = {s0, s1};
  *mul.y.strides = {s1, af::Symbol(1)};
  *mul.y.vectorized_axis = {z0.id, z1.id};
  store.x = mul.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, af::Symbol(1)};
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, af::Symbol(1)};
  const auto ctx = ::ascir::dumper::BuildDumpContext(graph);
  const auto text = ::ascir::dumper::DumpLoopExecutionView(graph, ctx);
  EXPECT_FALSE(text.empty());
}

}  // namespace ascir
}  // namespace af
