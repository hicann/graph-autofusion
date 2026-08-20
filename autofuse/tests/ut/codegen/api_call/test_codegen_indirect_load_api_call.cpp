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
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "common_utils.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "indirect_load_utils.h"
#include "platform_context.h"
#include "runtime_stub.h"
#include "utils/api_call_factory.h"
#include "v35/codegen/reg_api_call/reg_indirect_load_api_call.h"

using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;
using namespace ascgen_utils::indirect_load;

namespace {

struct ILTestGraph {
  af::AscGraph graph;
  af::Expression s0;
  af::Expression s1;
  af::Expression s2;
  af::Expression s3;
  af::Axis z0;
  af::Axis z1;
  af::Axis z2;
  af::Axis z3;

  explicit ILTestGraph(const char *name)
      : graph(name),
        s0(graph.CreateSizeVar("s0")),
        s1(graph.CreateSizeVar("s1")),
        s2(graph.CreateSizeVar("s2")),
        s3(graph.CreateSizeVar("s3")),
        z0(graph.CreateAxis("z0", s0)),
        z1(graph.CreateAxis("z1", s1)),
        z2(graph.CreateAxis("z2", s2)),
        z3(graph.CreateAxis("z3", s3)) {}

  ILTestGraph(const char *name, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
      : graph(name),
        s0(af::Symbol(size0)),
        s1(af::Symbol(size1)),
        s2(af::Symbol(size2)),
        s3(af::Symbol(size3)),
        z0(graph.CreateAxis("z0", s0)),
        z1(graph.CreateAxis("z1", s1)),
        z2(graph.CreateAxis("z2", s2)),
        z3(graph.CreateAxis("z3", s3)) {}
};

// ======================== SIMD graph ========================
// z0, z1: input axes  |  z2, z3: output/index axes
// IndirectLoad axis=1, rank=2

void BuildSimdGraph(ILTestGraph &g, af::DataType input_dtype = ge::DT_FLOAT16) {
  const af::Expression One = af::sym::kSymbolOne;

  Data x_data("x", g.graph);
  x_data.ir_attr.SetIndex(0);
  x_data.y.dtype = input_dtype;
  x_data.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_data.y.axis = {g.z0.id, g.z1.id};
  *x_data.y.repeats = {g.s0, g.s1};
  *x_data.y.strides = {g.s1, One};

  Data idx_data("idx", g.graph);
  idx_data.ir_attr.SetIndex(1);
  idx_data.y.dtype = ge::DT_INT32;
  idx_data.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_data.y.axis = {g.z2.id, g.z3.id};
  *idx_data.y.repeats = {g.s2, g.s3};
  *idx_data.y.strides = {g.s3, One};

  Load x_load("x_load");
  g.graph.AddNode(x_load);
  x_load.x = x_data.y;
  x_load.y.dtype = input_dtype;
  x_load.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_load.y.axis = {g.z0.id, g.z1.id};
  *x_load.y.repeats = {g.s0, g.s1};
  *x_load.y.strides = {g.s1, One};

  Load idx_load("idx_load");
  g.graph.AddNode(idx_load);
  idx_load.x = idx_data.y;
  idx_load.y.dtype = ge::DT_INT32;
  idx_load.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_load.y.axis = {g.z2.id, g.z3.id};
  *idx_load.y.repeats = {g.s2, g.s3};
  *idx_load.y.strides = {g.s3, One};

  IndirectLoad il("indirect_load");
  g.graph.AddNode(il);
  il.x1 = x_load.y;
  il.x2 = idx_load.y;
  il.ir_attr.SetAxis(1);
  il.attr.sched.axis = {g.z2.id, g.z3.id};
  *il.y.axis = {g.z2.id, g.z3.id};
  *il.y.repeats = {g.s2, g.s3};
  *il.y.strides = {g.s3, One};

  Store store("store");
  g.graph.AddNode(store);
  store.x = il.y;
  store.y.dtype = input_dtype;
  store.attr.sched.axis = {g.z2.id, g.z3.id};
  *store.y.axis = {g.z2.id, g.z3.id};
  *store.y.repeats = {g.s2, g.s3};
  *store.y.strides = {g.s3, One};

  Output y_out("y");
  g.graph.AddNode(y_out);
  y_out.ir_attr.SetIndex(0);
  y_out.x = store.y;
  y_out.y.dtype = input_dtype;

  // ----- API attrs -----
  for (const char *name : {"x", "idx"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeInvalid;
    n->attr.api.type = af::ApiType::kAPITypeBuffer;
    n->attr.api.unit = af::ComputeUnit::kUnitNone;
  }
  for (const char *name : {"x_load", "idx_load"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeLoad;
    n->attr.api.type = af::ApiType::kAPITypeCompute;
    n->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  }
  auto il_node = g.graph.FindNode("indirect_load");
  il_node->attr.api.compute_type = af::ComputeType::kComputeLoad;
  il_node->attr.api.type = af::ApiType::kAPITypeCompute;
  il_node->attr.api.unit = af::ComputeUnit::kUnitVector;

  auto store_node = g.graph.FindNode("store");
  store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
  store_node->attr.api.type = af::ApiType::kAPITypeCompute;
  store_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto y_node = g.graph.FindNode("y");
  y_node->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y_node->attr.api.type = af::ApiType::kAPITypeBuffer;
  y_node->attr.api.unit = af::ComputeUnit::kUnitNone;

  // ----- Axis splitting -----
  auto axes = g.graph.GetAllAxis();
  auto z2_id = axes[2]->id;
  auto z3_id = axes[3]->id;

  auto z2z3 = g.graph.MergeAxis({z2_id, z3_id});
  auto [z2z3T, z2z3t] = g.graph.TileSplit(z2z3->id);
  auto [z2z3TB, z2z3Tb] = g.graph.BlockSplit(z2z3T->id);

  for (auto n : g.graph.GetAllNodes()) {
    if (IsOps<Data>(n) || IsOps<Output>(n) || n->GetName() == "x_load") continue;
    g.graph.ApplyMerge(n, z2z3->id);
    g.graph.ApplySplit(n, z2z3T->id, z2z3t->id);
    g.graph.ApplySplit(n, z2z3TB->id, z2z3Tb->id);
    g.graph.ApplyReorder(n, {z2z3TB->id, z2z3Tb->id, z2z3t->id});
  }

  il_node->attr.sched.loop_axis = z2z3Tb->id;
  il_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  il_node->outputs[0].attr.vectorized_strides = {One};

  store_node->attr.sched.loop_axis = z2z3Tb->id;
  store_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  store_node->outputs[0].attr.vectorized_strides = {One};

  il_node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  // ----- Memory allocation -----
  auto x = g.graph.FindNode("x");
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x->outputs[0].attr.que.id = af::kIdNone;

  auto idx = g.graph.FindNode("idx");
  idx->outputs[0].attr.mem.tensor_id = 3;
  idx->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  idx->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  idx->outputs[0].attr.mem.position = af::Position::kPositionGM;
  idx->outputs[0].attr.que.id = af::kIdNone;

  auto x_load_node = g.graph.FindNode("x_load");
  x_load_node->outputs[0].attr.mem.tensor_id = 4;
  x_load_node->outputs[0].attr.mem.reuse_id = 4;
  x_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  x_load_node->outputs[0].attr.que.id = 1;
  x_load_node->outputs[0].attr.que.depth = 2;
  x_load_node->outputs[0].attr.que.buf_num = 2;

  auto idx_load_node = g.graph.FindNode("idx_load");
  idx_load_node->outputs[0].attr.mem.tensor_id = 5;
  idx_load_node->outputs[0].attr.mem.reuse_id = 5;
  idx_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  idx_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  idx_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  idx_load_node->outputs[0].attr.que.id = 2;
  idx_load_node->outputs[0].attr.que.depth = 2;
  idx_load_node->outputs[0].attr.que.buf_num = 2;

  il_node->outputs[0].attr.dtype = input_dtype;
  il_node->outputs[0].attr.mem.tensor_id = 1;
  il_node->outputs[0].attr.mem.reuse_id = 1;
  il_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  il_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  il_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  il_node->outputs[0].attr.que.id = 0;
  il_node->outputs[0].attr.que.depth = 2;
  il_node->outputs[0].attr.que.buf_num = 2;

  store_node->outputs[0].attr.mem.tensor_id = 2;
  store_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store_node->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store_node->outputs[0].attr.que.id = af::kIdNone;
}

// ======================== SIMT graph ========================
// Same output axes z2, z3 as SIMD but annotated as SIMT.
// IndirectLoad output chain ends at Store (for FindSimtOutputStore).

void BuildSimtGraph(ILTestGraph &g) {
  const af::Expression One = af::sym::kSymbolOne;

  Data x_data("x", g.graph);
  x_data.ir_attr.SetIndex(0);
  x_data.y.dtype = ge::DT_FLOAT16;
  x_data.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_data.y.axis = {g.z0.id, g.z1.id};
  *x_data.y.repeats = {g.s0, g.s1};
  *x_data.y.strides = {g.s1, One};

  Data idx_data("idx", g.graph);
  idx_data.ir_attr.SetIndex(1);
  idx_data.y.dtype = ge::DT_INT32;
  idx_data.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_data.y.axis = {g.z2.id, g.z3.id};
  *idx_data.y.repeats = {g.s2, g.s3};
  *idx_data.y.strides = {g.s3, One};

  Load x_load("x_load");
  g.graph.AddNode(x_load);
  x_load.x = x_data.y;
  x_load.y.dtype = ge::DT_FLOAT16;
  x_load.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_load.y.axis = {g.z0.id, g.z1.id};
  *x_load.y.repeats = {g.s0, g.s1};
  *x_load.y.strides = {g.s1, One};

  Load idx_load("idx_load");
  g.graph.AddNode(idx_load);
  idx_load.x = idx_data.y;
  idx_load.y.dtype = ge::DT_INT32;
  idx_load.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_load.y.axis = {g.z2.id, g.z3.id};
  *idx_load.y.repeats = {g.s2, g.s3};
  *idx_load.y.strides = {g.s3, One};

  IndirectLoad il("indirect_load");
  g.graph.AddNode(il);
  il.x1 = x_load.y;
  il.x2 = idx_load.y;
  il.ir_attr.SetAxis(1);
  il.attr.sched.axis = {g.z2.id, g.z3.id};
  *il.y.axis = {g.z2.id, g.z3.id};
  *il.y.repeats = {g.s2, g.s3};
  *il.y.strides = {g.s3, One};

  Store store("store");
  g.graph.AddNode(store);
  store.x = il.y;
  store.y.dtype = ge::DT_FLOAT16;
  store.attr.sched.axis = {g.z2.id, g.z3.id};
  *store.y.axis = {g.z2.id, g.z3.id};
  *store.y.repeats = {g.s2, g.s3};
  *store.y.strides = {g.s3, One};

  Output y_out("y");
  g.graph.AddNode(y_out);
  y_out.ir_attr.SetIndex(0);
  y_out.x = store.y;
  y_out.y.dtype = ge::DT_FLOAT16;

  // ----- API attrs -----
  for (const char *name : {"x", "idx"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeInvalid;
    n->attr.api.type = af::ApiType::kAPITypeBuffer;
    n->attr.api.unit = af::ComputeUnit::kUnitNone;
  }
  for (const char *name : {"x_load", "idx_load"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeLoad;
    n->attr.api.type = af::ApiType::kAPITypeCompute;
    n->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  }
  auto il_node = g.graph.FindNode("indirect_load");
  il_node->attr.api.compute_type = af::ComputeType::kComputeLoad;
  il_node->attr.api.type = af::ApiType::kAPITypeCompute;
  il_node->attr.api.unit = af::ComputeUnit::kUnitVector;

  auto store_node = g.graph.FindNode("store");
  store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
  store_node->attr.api.type = af::ApiType::kAPITypeCompute;
  store_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto y_node = g.graph.FindNode("y");
  y_node->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y_node->attr.api.type = af::ApiType::kAPITypeBuffer;
  y_node->attr.api.unit = af::ComputeUnit::kUnitNone;

  // ----- Axis splitting: merge output axes → z2z3 -----
  auto axes = g.graph.GetAllAxis();
  auto z2_id = axes[2]->id;
  auto z3_id = axes[3]->id;

  auto z2z3 = g.graph.MergeAxis({z2_id, z3_id});
  auto [z2z3T, z2z3t] = g.graph.TileSplit(z2z3->id);
  auto [z2z3TB, z2z3Tb] = g.graph.BlockSplit(z2z3T->id);

  for (auto n : g.graph.GetAllNodes()) {
    if (IsOps<Data>(n) || IsOps<Output>(n) || n->GetName() == "x_load") continue;
    g.graph.ApplyMerge(n, z2z3->id);
    g.graph.ApplySplit(n, z2z3T->id, z2z3t->id);
    g.graph.ApplySplit(n, z2z3TB->id, z2z3Tb->id);
    g.graph.ApplyReorder(n, {z2z3TB->id, z2z3Tb->id, z2z3t->id});
  }

  il_node->attr.sched.loop_axis = z2z3Tb->id;
  il_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  il_node->outputs[0].attr.vectorized_strides = {One};

  store_node->attr.sched.loop_axis = z2z3Tb->id;
  store_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  store_node->outputs[0].attr.vectorized_strides = {One};

  // ----- Memory allocation -----
  auto x = g.graph.FindNode("x");
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x->outputs[0].attr.que.id = af::kIdNone;

  auto idx = g.graph.FindNode("idx");
  idx->outputs[0].attr.mem.tensor_id = 3;
  idx->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  idx->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  idx->outputs[0].attr.mem.position = af::Position::kPositionGM;
  idx->outputs[0].attr.que.id = af::kIdNone;

  auto x_load_node = g.graph.FindNode("x_load");
  x_load_node->outputs[0].attr.mem.tensor_id = 4;
  x_load_node->outputs[0].attr.mem.reuse_id = 4;
  x_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  x_load_node->outputs[0].attr.que.id = 1;
  x_load_node->outputs[0].attr.que.depth = 2;
  x_load_node->outputs[0].attr.que.buf_num = 2;

  auto idx_load_node = g.graph.FindNode("idx_load");
  idx_load_node->outputs[0].attr.mem.tensor_id = 5;
  idx_load_node->outputs[0].attr.mem.reuse_id = 5;
  idx_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  idx_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  idx_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  idx_load_node->outputs[0].attr.que.id = 2;
  idx_load_node->outputs[0].attr.que.depth = 2;
  idx_load_node->outputs[0].attr.que.buf_num = 2;

  il_node->outputs[0].attr.dtype = ge::DT_FLOAT16;  // needed for SIMT dtype check vs Store
  il_node->outputs[0].attr.mem.tensor_id = 1;
  il_node->outputs[0].attr.mem.reuse_id = 1;
  il_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  il_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  il_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  il_node->outputs[0].attr.que.id = 0;
  il_node->outputs[0].attr.que.depth = 2;
  il_node->outputs[0].attr.que.buf_num = 2;

  store_node->outputs[0].attr.mem.tensor_id = 2;
  store_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store_node->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store_node->outputs[0].attr.que.id = af::kIdNone;
}

// ======================== SIMT post-reduce graph ========================
// IndirectLoad + addend -> inline Add -> fake Reduce (compute_type=kComputeReduce) -> Store

void BuildSimtPostReduceGraph(ILTestGraph &g) {
  const af::Expression One = af::sym::kSymbolOne;

  Data x_data("x", g.graph);
  x_data.ir_attr.SetIndex(0);
  x_data.y.dtype = ge::DT_FLOAT16;
  x_data.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_data.y.axis = {g.z0.id, g.z1.id};
  *x_data.y.repeats = {g.s0, g.s1};
  *x_data.y.strides = {g.s1, One};

  Data idx_data("idx", g.graph);
  idx_data.ir_attr.SetIndex(1);
  idx_data.y.dtype = ge::DT_INT32;
  idx_data.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_data.y.axis = {g.z2.id, g.z3.id};
  *idx_data.y.repeats = {g.s2, g.s3};
  *idx_data.y.strides = {g.s3, One};

  Data addend_data("addend", g.graph);
  addend_data.ir_attr.SetIndex(2);
  addend_data.y.dtype = ge::DT_FLOAT16;
  addend_data.attr.sched.axis = {g.z2.id, g.z3.id};
  *addend_data.y.axis = {g.z2.id, g.z3.id};
  *addend_data.y.repeats = {g.s2, g.s3};
  *addend_data.y.strides = {g.s3, One};

  Load x_load("x_load");
  g.graph.AddNode(x_load);
  x_load.x = x_data.y;
  x_load.y.dtype = ge::DT_FLOAT16;
  x_load.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_load.y.axis = {g.z0.id, g.z1.id};
  *x_load.y.repeats = {g.s0, g.s1};
  *x_load.y.strides = {g.s1, One};

  Load idx_load("idx_load");
  g.graph.AddNode(idx_load);
  idx_load.x = idx_data.y;
  idx_load.y.dtype = ge::DT_INT32;
  idx_load.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_load.y.axis = {g.z2.id, g.z3.id};
  *idx_load.y.repeats = {g.s2, g.s3};
  *idx_load.y.strides = {g.s3, One};

  Load addend_load("addend_load");
  g.graph.AddNode(addend_load);
  addend_load.x = addend_data.y;
  addend_load.y.dtype = ge::DT_FLOAT16;
  addend_load.attr.sched.axis = {g.z2.id, g.z3.id};
  *addend_load.y.axis = {g.z2.id, g.z3.id};
  *addend_load.y.repeats = {g.s2, g.s3};
  *addend_load.y.strides = {g.s3, One};

  IndirectLoad il("indirect_load");
  g.graph.AddNode(il);
  il.x1 = x_load.y;
  il.x2 = idx_load.y;
  il.ir_attr.SetAxis(1);
  il.attr.sched.axis = {g.z2.id, g.z3.id};
  *il.y.axis = {g.z2.id, g.z3.id};
  *il.y.repeats = {g.s2, g.s3};
  *il.y.strides = {g.s3, One};

  Add output_transform("output_transform");
  g.graph.AddNode(output_transform);
  output_transform.x1 = il.y;
  output_transform.x2 = addend_load.y;
  output_transform.attr.sched.axis = {g.z2.id, g.z3.id};
  *output_transform.y.axis = {g.z2.id, g.z3.id};
  *output_transform.y.repeats = {g.s2, g.s3};
  *output_transform.y.strides = {g.s3, One};

  // Fake Reduce: Abs with compute_type=kComputeReduce
  Abs reduce("reduce");
  g.graph.AddNode(reduce);
  reduce.x = output_transform.y;
  reduce.attr.sched.axis = {g.z2.id, g.z3.id};
  *reduce.y.axis = {g.z2.id, g.z3.id};
  *reduce.y.repeats = {g.s2, g.s3};
  *reduce.y.strides = {g.s3, One};

  Store store("store");
  g.graph.AddNode(store);
  store.x = reduce.y;
  store.y.dtype = ge::DT_FLOAT16;
  store.attr.sched.axis = {g.z2.id, g.z3.id};
  *store.y.axis = {g.z2.id, g.z3.id};
  *store.y.repeats = {g.s2, g.s3};
  *store.y.strides = {g.s3, One};

  Output y_out("y");
  g.graph.AddNode(y_out);
  y_out.ir_attr.SetIndex(0);
  y_out.x = store.y;
  y_out.y.dtype = ge::DT_FLOAT16;

  // ----- API attrs -----
  for (const char *name : {"x", "idx", "addend"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeInvalid;
    n->attr.api.type = af::ApiType::kAPITypeBuffer;
    n->attr.api.unit = af::ComputeUnit::kUnitNone;
  }
  for (const char *name : {"x_load", "idx_load", "addend_load"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeLoad;
    n->attr.api.type = af::ApiType::kAPITypeCompute;
    n->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  }
  auto il_node = g.graph.FindNode("indirect_load");
  il_node->attr.api.compute_type = af::ComputeType::kComputeLoad;
  il_node->attr.api.type = af::ApiType::kAPITypeCompute;
  il_node->attr.api.unit = af::ComputeUnit::kUnitVector;

  auto output_transform_node = g.graph.FindNode("output_transform");
  output_transform_node->attr.api.compute_type = af::ComputeType::kComputeElewise;
  output_transform_node->attr.api.type = af::ApiType::kAPITypeCompute;
  output_transform_node->attr.api.unit = af::ComputeUnit::kUnitVector;

  auto reduce_node = g.graph.FindNode("reduce");
  reduce_node->attr.api.compute_type = af::ComputeType::kComputeReduce;
  reduce_node->attr.api.type = af::ApiType::kAPITypeCompute;
  reduce_node->attr.api.unit = af::ComputeUnit::kUnitVector;

  auto store_node = g.graph.FindNode("store");
  store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
  store_node->attr.api.type = af::ApiType::kAPITypeCompute;
  store_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto y_node = g.graph.FindNode("y");
  y_node->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y_node->attr.api.type = af::ApiType::kAPITypeBuffer;
  y_node->attr.api.unit = af::ComputeUnit::kUnitNone;

  // ----- Axis splitting -----
  auto axes = g.graph.GetAllAxis();
  auto z2_id = axes[2]->id;
  auto z3_id = axes[3]->id;

  auto z2z3 = g.graph.MergeAxis({z2_id, z3_id});
  auto [z2z3T, z2z3t] = g.graph.TileSplit(z2z3->id);
  auto [z2z3TB, z2z3Tb] = g.graph.BlockSplit(z2z3T->id);

  for (auto n : g.graph.GetAllNodes()) {
    if (IsOps<Data>(n) || IsOps<Output>(n) || n->GetName() == "x_load") continue;
    g.graph.ApplyMerge(n, z2z3->id);
    g.graph.ApplySplit(n, z2z3T->id, z2z3t->id);
    g.graph.ApplySplit(n, z2z3TB->id, z2z3Tb->id);
    g.graph.ApplyReorder(n, {z2z3TB->id, z2z3Tb->id, z2z3t->id});
  }

  il_node->attr.sched.loop_axis = z2z3Tb->id;
  il_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  il_node->outputs[0].attr.vectorized_strides = {One};

  output_transform_node->attr.sched.loop_axis = z2z3Tb->id;
  output_transform_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  output_transform_node->outputs[0].attr.vectorized_strides = {One};

  reduce_node->attr.sched.loop_axis = z2z3Tb->id;
  reduce_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  reduce_node->outputs[0].attr.vectorized_strides = {One};

  store_node->attr.sched.loop_axis = z2z3Tb->id;
  store_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  store_node->outputs[0].attr.vectorized_strides = {One};

  // ----- Memory allocation -----
  auto x = g.graph.FindNode("x");
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x->outputs[0].attr.que.id = af::kIdNone;

  auto idx = g.graph.FindNode("idx");
  idx->outputs[0].attr.mem.tensor_id = 3;
  idx->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  idx->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  idx->outputs[0].attr.mem.position = af::Position::kPositionGM;
  idx->outputs[0].attr.que.id = af::kIdNone;

  auto addend = g.graph.FindNode("addend");
  addend->outputs[0].attr.mem.tensor_id = 6;
  addend->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  addend->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  addend->outputs[0].attr.mem.position = af::Position::kPositionGM;
  addend->outputs[0].attr.que.id = af::kIdNone;

  auto x_load_node = g.graph.FindNode("x_load");
  x_load_node->outputs[0].attr.mem.tensor_id = 7;
  x_load_node->outputs[0].attr.mem.reuse_id = 7;
  x_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  x_load_node->outputs[0].attr.que.id = 3;
  x_load_node->outputs[0].attr.que.depth = 2;
  x_load_node->outputs[0].attr.que.buf_num = 2;

  auto idx_load_node = g.graph.FindNode("idx_load");
  idx_load_node->outputs[0].attr.mem.tensor_id = 8;
  idx_load_node->outputs[0].attr.mem.reuse_id = 8;
  idx_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  idx_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  idx_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  idx_load_node->outputs[0].attr.que.id = 4;
  idx_load_node->outputs[0].attr.que.depth = 2;
  idx_load_node->outputs[0].attr.que.buf_num = 2;

  auto addend_load_node = g.graph.FindNode("addend_load");
  addend_load_node->outputs[0].attr.mem.tensor_id = 9;
  addend_load_node->outputs[0].attr.mem.reuse_id = 9;
  addend_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  addend_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  addend_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  addend_load_node->outputs[0].attr.que.id = 5;
  addend_load_node->outputs[0].attr.que.depth = 2;
  addend_load_node->outputs[0].attr.que.buf_num = 2;

  il_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  il_node->outputs[0].attr.mem.tensor_id = 1;
  il_node->outputs[0].attr.mem.reuse_id = 1;
  il_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  il_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  il_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  il_node->outputs[0].attr.que.id = 0;
  il_node->outputs[0].attr.que.depth = 2;
  il_node->outputs[0].attr.que.buf_num = 2;

  output_transform_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  output_transform_node->outputs[0].attr.mem.tensor_id = 5;
  output_transform_node->outputs[0].attr.mem.reuse_id = 5;
  output_transform_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  output_transform_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  output_transform_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  output_transform_node->outputs[0].attr.que.id = 2;
  output_transform_node->outputs[0].attr.que.depth = 2;
  output_transform_node->outputs[0].attr.que.buf_num = 2;

  reduce_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  reduce_node->outputs[0].attr.mem.tensor_id = 4;
  reduce_node->outputs[0].attr.mem.reuse_id = 4;
  reduce_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  reduce_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  reduce_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  reduce_node->outputs[0].attr.que.id = 1;
  reduce_node->outputs[0].attr.que.depth = 2;
  reduce_node->outputs[0].attr.que.buf_num = 2;

  store_node->outputs[0].attr.mem.tensor_id = 2;
  store_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store_node->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store_node->outputs[0].attr.que.id = af::kIdNone;
}

// ======================== Template annotation ========================

void AnnotateSimdTemplate(af::AscGraph &graph) {
  auto il = graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  ASSERT_EQ(::ascir::SetTemplateId(il, ::ascir::TemplateId::kIndirectLoadSimd), af::SUCCESS);
  ASSERT_EQ(SetImplementation(il, Implementation::kDefault), af::SUCCESS);

  auto axes = graph.GetAllAxis();
  af::AxisId outer_id = af::kIdNone, inner_id = af::kIdNone, input_inner_id = af::kIdNone;
  for (const auto &axis : axes) {
    if (axis == nullptr) continue;
    if (axis->name == "z2z3TB") outer_id = axis->id;
    if (axis->name == "z2z3t") inner_id = axis->id;
    if (axis->name == "z2z3t") input_inner_id = axis->id;
  }
  ASSERT_NE(outer_id, af::kIdNone);
  ASSERT_NE(inner_id, af::kIdNone);

  TemplateAxes t{outer_id, inner_id, input_inner_id};
  ASSERT_EQ(SetTemplateAxes(il, t), af::SUCCESS);

  const af::Expression One = af::sym::kSymbolOne;
  TemplateLogicalView view;
  view.input.axis_ids = {axes[0]->id, axes[1]->id};
  view.input.sizes = {graph.FindAxis(axes[0]->id)->size, graph.FindAxis(axes[1]->id)->size};
  view.input.strides = {graph.FindAxis(axes[1]->id)->size, One};
  view.index.axis_ids = {axes[2]->id, axes[3]->id};
  view.index.sizes = {graph.FindAxis(axes[2]->id)->size, graph.FindAxis(axes[3]->id)->size};
  view.index.strides = {graph.FindAxis(axes[3]->id)->size, One};
  view.output.axis_ids = {axes[2]->id, axes[3]->id};
  view.output.sizes = {graph.FindAxis(axes[2]->id)->size, graph.FindAxis(axes[3]->id)->size};
  view.output.strides = {graph.FindAxis(axes[3]->id)->size, One};
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.input, view.input), af::SUCCESS);
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.index, view.index), af::SUCCESS);
  ASSERT_EQ(SetTemplateLogicalView(il, view), af::SUCCESS);
}

void AnnotateSimtTemplate(af::AscGraph &graph) {
  auto il = graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  ASSERT_EQ(::ascir::SetTemplateId(il, ::ascir::TemplateId::kIndirectLoadSimt), af::SUCCESS);
  ASSERT_EQ(SetImplementation(il, Implementation::kDefault), af::SUCCESS);
  const auto input_load = graph.FindNode("x_load");
  const auto index_load = graph.FindNode("idx_load");
  const auto store = graph.FindNode("store");
  ASSERT_NE(input_load, nullptr);
  ASSERT_NE(index_load, nullptr);
  ASSERT_NE(store, nullptr);
  ASSERT_EQ(SetTemplateRole(input_load, TemplateRole::kSimtInputBoundary), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(index_load, TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(store, TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  const auto addend_load = graph.FindNode("addend_load");
  if (addend_load != nullptr) {
    ASSERT_EQ(SetTemplateRole(addend_load, TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  }

  // SIMT outer_axis derives from all output axes (z2, z3).
  // Use the merged axis z2z3 which derives from both.
  auto axes = graph.GetAllAxis();
  af::AxisId outer_id = af::kIdNone;
  for (const auto &axis : axes) {
    if (axis == nullptr) continue;
    if (axis->name == "z2z3") outer_id = axis->id;
  }
  ASSERT_NE(outer_id, af::kIdNone);

  TemplateAxes t{outer_id, af::kIdNone, af::kIdNone};
  ASSERT_EQ(SetTemplateAxes(il, t), af::SUCCESS);

  const af::Expression One = af::sym::kSymbolOne;
  TemplateLogicalView view;
  view.input.axis_ids = {axes[0]->id, axes[1]->id};
  view.input.sizes = {graph.FindAxis(axes[0]->id)->size, graph.FindAxis(axes[1]->id)->size};
  view.input.strides = {graph.FindAxis(axes[1]->id)->size, One};
  view.index.axis_ids = {axes[2]->id, axes[3]->id};
  view.index.sizes = {graph.FindAxis(axes[2]->id)->size, graph.FindAxis(axes[3]->id)->size};
  view.index.strides = {graph.FindAxis(axes[3]->id)->size, One};
  view.output.axis_ids = {axes[2]->id, axes[3]->id};
  view.output.sizes = {graph.FindAxis(axes[2]->id)->size, graph.FindAxis(axes[3]->id)->size};
  view.output.strides = {graph.FindAxis(axes[3]->id)->size, One};
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.input, view.input), af::SUCCESS);
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.index, view.index), af::SUCCESS);
  ASSERT_EQ(SetTemplateLogicalView(il, view), af::SUCCESS);
}

ApiCall *FindCallByNodeName(const Loop &loop, const std::string &name) {
  for (const auto &body : loop.bodys) {
    if (body.type == LoopType::CALL && body.call != nullptr && body.call->node->GetName() == name) {
      return body.call;
    }
    if (body.type == LoopType::LOOP && body.loop != nullptr) {
      ApiCall *call = FindCallByNodeName(*body.loop, name);
      if (call != nullptr) {
        return call;
      }
    }
  }
  return nullptr;
}

class IndirectLoadConstructFromNodesTest : public testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    ge::RuntimeStub::SetInstance(std::make_shared<ge::RuntimeStubV2Common>());
  }

  void TearDown() override {
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

void SetupTiler(const af::AscGraph &graph, Tiler &tiler) {
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis != nullptr) {
      tiler.AddAxis(*axis);
      tiler.AddSizeVar(af::SizeVar(axis->size));
    }
  }
}

Status ParseGraphAndGenerateLoop(ILTestGraph &g, const std::vector<const char *> &input_names, Kernel &kernel,
                                 std::string &generated) {
  ascir::FusedScheduledResult result;
  for (const char *name : input_names) {
    const auto input = g.graph.FindNode(name);
    GE_ASSERT_NOTNULL(input, "Input node[%s] is missing.", name);
    result.input_nodes.emplace_back(input);
  }
  const auto output = g.graph.FindNode("y");
  GE_ASSERT_NOTNULL(output, "Output node[y] is missing.");
  result.output_nodes = {output};
  GE_ASSERT_SUCCESS(Kernel::ParseGraph(g.graph, result, kernel));
  return kernel.root_loop.Generate(kernel.tiler, kernel.tpipe, generated);
}

}  // namespace

// ==================== Factory ====================

TEST(IndirectLoadApiCallTest, FactoryCreatesIndirectLoadRegApiCall) {
  auto *handler = ApiCallFactory::Instance().Create("IndirectLoadRegApiCall", "IndirectLoadRegApiCall");
  ASSERT_NE(handler, nullptr);
  auto *il_handler = dynamic_cast<IndirectLoadRegApiCall *>(handler);
  ASSERT_NE(il_handler, nullptr);
  EXPECT_EQ(il_handler->api_name_, "IndirectLoadRegApiCall");
  delete handler;
}

// ==================== SIMD Init + Generate ====================

void GenerateSimdCall(af::DataType input_dtype, std::string &result) {
  ILTestGraph g("simd_gen");
  BuildSimdGraph(g, input_dtype);
  AnnotateSimdTemplate(g.graph);

  Tiler tiler;
  SetupTiler(g.graph, tiler);
  TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(g.graph);

  auto il_node = g.graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);
  tpipe.AddTensor(il_node->inputs[0]);
  tpipe.AddTensor(il_node->inputs[1]);
  tpipe.AddTensor(*il_node->outputs()[0]);

  IndirectLoadRegApiCall call("IndirectLoadRegApiCall");
  ASSERT_EQ(call.Init(il_node), af::SUCCESS);

  // Set up ApiTensors for inputs (outputs already populated by Init)
  ApiTensor x1, x2;
  x1.id = il_node->inputs[0].attr.mem.tensor_id;
  x2.id = il_node->inputs[1].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::vector<std::reference_wrapper<const Tensor>> input_refs, output_refs;
  for (auto *in : call.inputs) {
    auto *t = tpipe.GetTensor(in->id);
    ASSERT_NE(t, nullptr);
    input_refs.emplace_back(*t);
  }
  for (auto &out : call.outputs) {
    auto *t = tpipe.GetTensor(out.id);
    ASSERT_NE(t, nullptr);
    output_refs.emplace_back(*t);
  }

  std::vector<af::AxisId> current_axis;
  for (const auto &axis : g.graph.GetAllAxis()) {
    if (axis != nullptr) current_axis.emplace_back(axis->id);
  }

  auto status = call.Generate(tpipe, current_axis, input_refs, output_refs, result);
  ASSERT_EQ(status, af::SUCCESS);
}

TEST(IndirectLoadApiCallTest, GenerateSimdProducesIndirectLoadSimdCall) {
  std::string result;
  GenerateSimdCall(ge::DT_FLOAT16, result);
  EXPECT_NE(result.find("// IndirectLoad SIMD"), std::string::npos);
  EXPECT_NE(result.find("IndirectLoadSimd<half, int32_t, 2, 1>"), std::string::npos);
  // Symbolic sizes are dynamic shapes, which use the strided path and require a temporary UB buffer.
  EXPECT_NE(result.find("tmp_buf_0"), std::string::npos);
}

TEST(IndirectLoadApiCallTest, GenerateSimdUint32InputProducesTypedCall) {
  std::string result;
  GenerateSimdCall(ge::DT_UINT32, result);
  EXPECT_NE(result.find("IndirectLoadSimd<uint32_t, int32_t, 2, 1>"), std::string::npos);
}

// ==================== SIMT Init + GenerateFuncDefinition ====================

void GenerateSimtFuncDefinition(ILTestGraph &g, std::string &definition) {
  BuildSimtGraph(g);
  AnnotateSimtTemplate(g.graph);

  Tiler tiler;
  SetupTiler(g.graph, tiler);
  TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(g.graph);

  auto il_node = g.graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);

  IndirectLoadRegApiCall call("IndirectLoadRegApiCall");
  ASSERT_EQ(call.Init(il_node), af::SUCCESS);

  // Set up input tensors needed by GenerateFuncDefinition
  ApiTensor x1, x2;
  x1.id = il_node->inputs[0].attr.mem.tensor_id;
  x2.id = il_node->inputs[1].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  tpipe.AddTensor(il_node->inputs[0]);
  tpipe.AddTensor(il_node->inputs[1]);

  std::stringstream ss;
  auto status = call.GenerateFuncDefinition(tpipe, tiler, ss);
  ASSERT_EQ(status, af::SUCCESS);
  definition = ss.str();
}

TEST(IndirectLoadApiCallTest, GenerateFuncDefinitionSimtProducesBodyStruct) {
  ILTestGraph g("simt_gen");
  std::string def;
  GenerateSimtFuncDefinition(g, def);

  EXPECT_NE(def.find("struct IndirectLoadSimtContext_"), std::string::npos);
  EXPECT_NE(def.find("struct IndirectLoadSimtBody_"), std::string::npos);
  EXPECT_NE(def.find("using Context = IndirectLoadSimtContext_"), std::string::npos);
  EXPECT_NE(def.find("__simt_callee__ __aicore__ inline static"), std::string::npos);
  EXPECT_NE(def.find(" Index(uint64_t output_index, const Context &context)"), std::string::npos);
  EXPECT_NE(def.find(" Output("), std::string::npos);
}

TEST(IndirectLoadApiCallTest, GenerateFuncDefinitionUsesUint32OffsetsForStaticShapes) {
  ILTestGraph g("simt_static_power_of_two", 2, 8, 2, 8);
  std::string def;
  GenerateSimtFuncDefinition(g, def);

  EXPECT_NE(def.find("uint32_t output_index"), std::string::npos);
}

// ==================== SIMT post-reduce Init ====================

TEST(IndirectLoadApiCallTest, InitSimtPostReduceSucceeds) {
  ILTestGraph g("simt_post");
  BuildSimtPostReduceGraph(g);
  AnnotateSimtTemplate(g.graph);

  auto il_node = g.graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);
  auto output_transform = g.graph.FindNode("output_transform");
  ASSERT_NE(output_transform, nullptr);
  ASSERT_EQ(SetTemplateRole(output_transform, TemplateRole::kSimtInlineTransform), af::SUCCESS);

  IndirectLoadRegApiCall call("IndirectLoadRegApiCall");
  auto status = call.Init(il_node);
  EXPECT_EQ(status, af::SUCCESS);
}

TEST_F(IndirectLoadConstructFromNodesTest, ConnectsRedirectedSimtOutputToReduce) {
  ILTestGraph g("simt_post_construct");
  BuildSimtPostReduceGraph(g);
  AnnotateSimtTemplate(g.graph);

  auto x = g.graph.FindNode("x");
  auto idx = g.graph.FindNode("idx");
  auto addend = g.graph.FindNode("addend");
  auto y = g.graph.FindNode("y");
  auto il = g.graph.FindNode("indirect_load");
  auto output_transform = g.graph.FindNode("output_transform");
  ASSERT_NE(x, nullptr);
  ASSERT_NE(idx, nullptr);
  ASSERT_NE(addend, nullptr);
  ASSERT_NE(y, nullptr);
  ASSERT_NE(il, nullptr);
  ASSERT_NE(output_transform, nullptr);
  ASSERT_EQ(SetTemplateRole(il, TemplateRole::kSimtOp), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(output_transform, TemplateRole::kSimtInlineTransform), af::SUCCESS);

  ascir::FusedScheduledResult result;
  result.input_nodes = {x, idx, addend};
  result.output_nodes = {y};
  Kernel kernel("simt_post_construct");
  ASSERT_EQ(Kernel::ParseGraph(g.graph, result, kernel), af::SUCCESS);

  ApiCall *il_call = FindCallByNodeName(kernel.root_loop, "indirect_load");
  ApiCall *transform_call = FindCallByNodeName(kernel.root_loop, "output_transform");
  ApiCall *reduce_call = FindCallByNodeName(kernel.root_loop, "reduce");
  ASSERT_NE(il_call, nullptr);
  ASSERT_NE(transform_call, nullptr);
  ASSERT_NE(reduce_call, nullptr);
  ASSERT_EQ(il_call->outputs.size(), 1UL);
  ASSERT_EQ(reduce_call->inputs.size(), 1UL);
  const ascir::TensorId redirected_id = output_transform->outputs[0].attr.mem.tensor_id;
  const Tensor *redirected_tensor = kernel.tpipe.GetTensor(redirected_id);
  ASSERT_NE(redirected_tensor, nullptr);
  EXPECT_TRUE(transform_call->skip_api_emit);
  EXPECT_EQ(il_call->outputs[0].id, redirected_id);
  EXPECT_EQ(il_call->outputs[0].reuse_id, redirected_tensor->reuse_id);
  EXPECT_EQ(reduce_call->inputs[0], &il_call->outputs[0]);
}

TEST_F(IndirectLoadConstructFromNodesTest, GeneratesSimdLoopWithUbLifecycleAndNoLoadStoreSync) {
  ILTestGraph g("simd_tile_inner");
  BuildSimdGraph(g);
  AnnotateSimdTemplate(g.graph);

  Kernel kernel("simd_tile_inner");
  std::string generated;
  ASSERT_EQ(ParseGraphAndGenerateLoop(g, {"x", "idx"}, kernel, generated), af::SUCCESS);
  const size_t actual_size = generated.find("z2z3t_actual_size = ");
  const size_t indirect_load = generated.find("// IndirectLoad SIMD");
  ASSERT_NE(actual_size, std::string::npos);
  ASSERT_NE(indirect_load, std::string::npos);
  EXPECT_LT(actual_size, indirect_load);
  EXPECT_NE(generated.find("AllocTensor"), std::string::npos);
  EXPECT_NE(generated.find("EnQue"), std::string::npos);
  EXPECT_NE(generated.find("DeQue"), std::string::npos);
  EXPECT_EQ(generated.find("MTE2_MTE3"), std::string::npos);
}

TEST_F(IndirectLoadConstructFromNodesTest, GeneratesSimtDirectGmCallOutsideBlockInnerLoop) {
  ILTestGraph g("simt_block_inner");
  BuildSimtGraph(g);
  AnnotateSimtTemplate(g.graph);
  auto il = g.graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  ASSERT_EQ(SetTemplateRole(il, TemplateRole::kSimtOp), af::SUCCESS);

  Kernel kernel("simt_block_inner");
  std::string generated;
  ASSERT_EQ(ParseGraphAndGenerateLoop(g, {"x", "idx"}, kernel, generated), af::SUCCESS);
  const size_t block_offset = generated.find("block_dim_offset");
  const size_t indirect_load = generated.find("// IndirectLoad SIMT");
  ASSERT_NE(block_offset, std::string::npos);
  ASSERT_NE(indirect_load, std::string::npos);
  EXPECT_LT(block_offset, indirect_load);
  EXPECT_EQ(generated.find("for (int z2z3Tb"), std::string::npos);
  EXPECT_EQ(generated.find("AllocTensor"), std::string::npos);
  EXPECT_EQ(generated.find("EnQue"), std::string::npos);
  EXPECT_EQ(generated.find("DeQue"), std::string::npos);
}

TEST_F(IndirectLoadConstructFromNodesTest, GeneratesStaticPowerOfTwoPolicyWithUint32Offsets) {
  ILTestGraph g("simt_static_power_of_two", 2, 8, 2, 8);
  BuildSimtGraph(g);
  AnnotateSimtTemplate(g.graph);
  auto il = g.graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  ASSERT_EQ(SetTemplateRole(il, TemplateRole::kSimtOp), af::SUCCESS);

  Kernel kernel("simt_static_power_of_two");
  std::string generated;
  ASSERT_EQ(ParseGraphAndGenerateLoop(g, {"x", "idx"}, kernel, generated), af::SUCCESS);
  EXPECT_NE(generated.find("IndirectLoadSimtStaticPowerOfTwoPolicy<uint32_t, 1ULL, 8ULL, 1ULL, 8ULL>"),
            std::string::npos);
}

TEST_F(IndirectLoadConstructFromNodesTest, GeneratesStaticInnerPolicyForLastAxis) {
  ILTestGraph g("simt_static_inner", 2, 7, 2, 7);
  BuildSimtGraph(g);
  AnnotateSimtTemplate(g.graph);
  auto il = g.graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  ASSERT_EQ(SetTemplateRole(il, TemplateRole::kSimtOp), af::SUCCESS);

  Kernel kernel("simt_static_inner");
  std::string generated;
  ASSERT_EQ(ParseGraphAndGenerateLoop(g, {"x", "idx"}, kernel, generated), af::SUCCESS);
  EXPECT_NE(generated.find("IndirectLoadSimtStaticInnerPolicy<uint32_t, 1ULL, 1ULL, 7ULL>"), std::string::npos);
  EXPECT_NE(generated.find(", 7);"), std::string::npos);
}

TEST_F(IndirectLoadConstructFromNodesTest, GeneratesSimtPostReduceCallWithUbOutput) {
  ILTestGraph g("simt_post_reduce");
  BuildSimtPostReduceGraph(g);
  AnnotateSimtTemplate(g.graph);
  auto il = g.graph.FindNode("indirect_load");
  auto output_transform = g.graph.FindNode("output_transform");
  ASSERT_NE(il, nullptr);
  ASSERT_NE(output_transform, nullptr);
  ASSERT_EQ(SetTemplateRole(il, TemplateRole::kSimtOp), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(output_transform, TemplateRole::kSimtInlineTransform), af::SUCCESS);

  Kernel kernel("simt_post_reduce");
  std::string generated;
  ASSERT_EQ(ParseGraphAndGenerateLoop(g, {"x", "idx", "addend"}, kernel, generated), af::SUCCESS);
  EXPECT_NE(generated.find("// IndirectLoad SIMT"), std::string::npos);
  EXPECT_NE(generated.find("AllocTensor"), std::string::npos);
  EXPECT_NE(generated.find("static_cast<uint32_t>"), std::string::npos);
  EXPECT_EQ(generated.find("y_ptr"), std::string::npos);
}
