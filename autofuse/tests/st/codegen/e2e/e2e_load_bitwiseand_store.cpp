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
#include "ascir_ops.h"
#include "ascir_ops_utils.h"

using namespace ge;
using namespace af;
using namespace af::ops;
using namespace af::ascir_op;

void LoadBitwiseAndStore_BeforeAutofuse(af::AscGraph &graph, ge::DataType data_type) {
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  Data x1("x1");
  graph.AddNode(x1);
  x1.y.dtype = data_type;

  Data x2("x2");
  graph.AddNode(x2);
  x2.y.dtype = data_type;

  Load load1("load1");
  graph.AddNode(load1);
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id};
  *load1.y.axis = {z0.id};
  *load1.y.repeats = {s0};
  *load1.y.strides = {One};

  Load load2("load2");
  graph.AddNode(load2);
  load2.x = x2.y;
  load2.attr.sched.axis = {z0.id};
  *load2.y.axis = {z0.id};
  *load2.y.repeats = {s0};
  *load2.y.strides = {One};

  af::ascir_op::BitwiseAnd bitwiseAnd("bitwiseAnd");
  graph.AddNode(bitwiseAnd);
  bitwiseAnd.x1 = load1.y;
  bitwiseAnd.x2 = load2.y;
  bitwiseAnd.attr.sched.axis = {z0.id};
  *bitwiseAnd.y.axis = {z0.id};
  *bitwiseAnd.y.repeats = {s0};
  *bitwiseAnd.y.strides = {One};
  bitwiseAnd.attr.tmp_buffers = {{{af::Symbol(8192), -1}, MemAttr(), 0}};

  Store store("store");
  graph.AddNode(store);
  store.x = bitwiseAnd.y;
  store.attr.sched.axis = {z0.id};
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.y.dtype = data_type;
}

void LoadBitwiseAndStore_AfterAutofuse(af::AscGraph &graph, ge::DataType data_type) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid;
  x1->attr.api.type = ApiType::kAPITypeBuffer;
  x1->attr.api.unit = ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid;
  x2->attr.api.type = ApiType::kAPITypeBuffer;
  x2->attr.api.unit = ComputeUnit::kUnitNone;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.dtype = data_type;
  load1->attr.api.compute_type = ComputeType::kComputeLoad;
  load1->attr.api.type = ApiType::kAPITypeCompute;
  load1->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto load2 = graph.FindNode("load2");
  load2->outputs[0].attr.dtype = data_type;
  load2->attr.api.compute_type = ComputeType::kComputeLoad;
  load2->attr.api.type = ApiType::kAPITypeCompute;
  load2->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto bitwiseAnd = graph.FindNode("bitwiseAnd");
  bitwiseAnd->outputs[0].attr.dtype = data_type;
  bitwiseAnd->attr.api.compute_type = ComputeType::kComputeElewise;
  bitwiseAnd->attr.api.type = ApiType::kAPITypeCompute;
  bitwiseAnd->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = data_type;
  store->attr.api.compute_type = ComputeType::kComputeStore;
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;

  // Scheduler
  auto z0 = load1->attr.sched.axis[0];

  auto [z0T, z0t] = graph.TileSplit(z0);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }

    graph.ApplySplit(node, z0T->id, z0t->id);
    graph.ApplySplit(node, z0TB->id, z0Tb->id);
  }

  vector<af::AxisId> vectorized_axis = {
      z0t->id,
  };
  vector<af::Expression> vectorized_strides{
      One,
  };
  // Vectorized/Loop axis
  load1->attr.sched.loop_axis = z0Tb->id;
  load1->outputs[0].attr.vectorized_axis = vectorized_axis;
  load1->outputs[0].attr.vectorized_strides = vectorized_strides;

  load2->attr.sched.loop_axis = z0Tb->id;
  load2->outputs[0].attr.vectorized_axis = vectorized_axis;
  load2->outputs[0].attr.vectorized_strides = vectorized_strides;

  bitwiseAnd->attr.sched.loop_axis = z0Tb->id;
  bitwiseAnd->outputs[0].attr.vectorized_axis = vectorized_axis;
  bitwiseAnd->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z0Tb->id;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = Position::kPositionGM;
  x1->outputs[0].attr.buf.id = af::kIdNone;
  x1->outputs[0].attr.que.id = af::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = Position::kPositionGM;
  x2->outputs[0].attr.buf.id = af::kIdNone;
  x2->outputs[0].attr.que.id = af::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load1->outputs[0].attr.mem.tensor_id = 2;
  load1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load1->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load1->outputs[0].attr.buf.id = af::kIdNone;
  load1->outputs[0].attr.que.id = 0;
  load1->outputs[0].attr.mem.reuse_id = 0;
  load1->outputs[0].attr.que.depth = 2;
  load1->outputs[0].attr.que.buf_num = 2;
  load1->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load2->outputs[0].attr.mem.tensor_id = 3;
  load2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load2->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load2->outputs[0].attr.buf.id = af::kIdNone;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.mem.reuse_id = 1;
  load2->outputs[0].attr.que.depth = 2;
  load2->outputs[0].attr.que.buf_num = 2;
  load2->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  bitwiseAnd->outputs[0].attr.mem.tensor_id = 4;
  bitwiseAnd->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeBuffer;
  bitwiseAnd->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  bitwiseAnd->outputs[0].attr.mem.position = Position::kPositionVecCalc;
  bitwiseAnd->outputs[0].attr.buf.id = 1;
  bitwiseAnd->outputs[0].attr.que.id = af::kIdNone;
  bitwiseAnd->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  bitwiseAnd->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.tensor_id = 5;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = af::kIdNone;
  store->outputs[0].attr.que.id = af::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;
}
