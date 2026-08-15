#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "ascir_ops_utils.h"

using namespace std;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

void LoadScalarBrcStore_BeforeAutofuse2D(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x("x");
  graph.AddNode(x);
  x.attr.sched.axis = {z0.id, z1.id};
  x.y.dtype = af::DT_FLOAT;
  *x.y.axis = {z0.id, z1.id};

  Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id};
  load.y.dtype = af::DT_FLOAT;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, One};
  *load.y.strides = {One, Zero};

  Broadcast broadcast("broadcast");
  graph.AddNode(broadcast);
  broadcast.attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  broadcast.x = load.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  broadcast.y.dtype = af::DT_FLOAT;
  *broadcast.y.axis = {z0.id, z1.id};
  *broadcast.y.repeats = {s0, s1};
  *broadcast.y.strides = {s1, One};

  Store store("store");
  graph.AddNode(store);
  store.x = broadcast.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
}

static void ConfigureLoadScalarBrcStoreInputAndLoad(af::AscGraph &graph, af::AxisId z0, af::AxisId z1) {
  auto x = graph.FindNode("x");
  x->attr.api.compute_type = ComputeType::kComputeInvalid;
  x->attr.api.type = ApiType::kAPITypeBuffer;
  x->attr.api.unit = ComputeUnit::kUnitNone;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x->outputs[0].attr.mem.position = Position::kPositionGM;
  x->outputs[0].attr.buf.id = af::kIdNone;
  x->outputs[0].attr.que.id = af::kIdNone;
  x->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = ComputeType::kComputeLoad;
  load->attr.api.type = ApiType::kAPITypeCompute;
  load->attr.api.unit = ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0;
  load->outputs[0].attr.vectorized_axis = {z1};
  load->outputs[0].attr.vectorized_strides = {Zero};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load->outputs[0].attr.buf.id = af::kIdNone;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

static void ConfigureLoadScalarBrcStoreBroadcastAndStore(af::AscGraph &graph, af::AxisId z0, af::AxisId z1,
                                                         const vector<AxisId> &axes) {
  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = ComputeType::kComputeElewise;
  broadcast->attr.api.type = ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0;
  broadcast->outputs[0].attr.vectorized_axis = axes;
  broadcast->outputs[0].attr.vectorized_strides = {Zero, One};
  broadcast->outputs[0].attr.dtype = af::DT_FLOAT;
  broadcast->outputs[0].attr.mem.tensor_id = 2;
  broadcast->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  broadcast->outputs[0].attr.mem.position = Position::kPositionVecOut;
  broadcast->outputs[0].attr.buf.id = af::kIdNone;
  broadcast->outputs[0].attr.que.id = 1;
  broadcast->outputs[0].attr.mem.reuse_id = 1;
  broadcast->outputs[0].attr.que.depth = 2;
  broadcast->outputs[0].attr.que.buf_num = 2;
  broadcast->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ComputeType::kComputeStore;
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;
  store->attr.sched.loop_axis = z0;
  store->outputs[0].attr.vectorized_axis = {z1};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = af::kIdNone;
  store->outputs[0].attr.que.id = af::kIdNone;
  store->outputs[0].attr.mem.reuse_id = af::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

void LoadScalarBrcStore_AfterAutofuse2D(af::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  auto z0 = all_axis[0]->id;
  auto z1 = all_axis[1]->id;
  vector<AxisId> axes{z0, z1};

  ConfigureLoadScalarBrcStoreInputAndLoad(graph, z0, z1);
  ConfigureLoadScalarBrcStoreBroadcastAndStore(graph, z0, z1, axes);
}
