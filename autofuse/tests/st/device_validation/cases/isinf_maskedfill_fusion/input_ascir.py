# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import argparse
import json
from pathlib import Path

from autofuse.pyautofuse import ascir, Autofuser, AutofuserOptions


def _create_graph_and_axes(name, shape):
    rows, cols = shape
    graph = ascir.HintGraph(name)
    a0 = ascir.SizeExpr(rows)
    a1 = ascir.SizeExpr(cols)
    z0 = graph.create_axis("z0", a0)
    z1 = graph.create_axis("z1", a1)
    return graph, a0, a1, z0, z1


def _create_data_node(graph, name, index, dtype):
    node = ascir.ops.Data(name, graph)
    node.attr.ir_attr.index = index
    node.y.dtype = dtype
    return node


def _create_data_nodes(graph):
    return (
        _create_data_node(graph, "data_x", 0, ascir.dtypes.float16),
        _create_data_node(graph, "data_mask", 1, ascir.dtypes.uint8),
        _create_data_node(graph, "data_value", 2, ascir.dtypes.float16),
    )


def _create_loads(graph, meta, x, mask, value):
    a0, a1, z0, z1 = meta
    loaded = []
    for name, source, dtype, _ in (
        ("load_x", x, ascir.dtypes.float16, 0),
        ("load_mask", mask, ascir.dtypes.uint8, 1),
        ("load_value", value, ascir.dtypes.float16, 2),
    ):
        node = ascir.ops.Load(name, graph)
        node.attr.ir_attr.offset = ascir.SizeExpr(0)
        node.attr.sched.axis = [z0, z1]
        node.x = source.y
        node.y.axis = [z0, z1]
        node.y.strides = [a1, ascir.SizeExpr(1)]
        node.y.size = [a0, a1]
        node.y.dtype = dtype
        loaded.append(node)
    return loaded


def _create_compute_and_output(graph, meta, loaded):
    a0, a1, z0, z1 = meta
    is_inf = ascir.ops.IsInf("is_inf", graph)
    is_inf.attr.sched.axis = [z0, z1]
    is_inf.x = loaded[0].y
    is_inf.y.axis = [z0, z1]
    is_inf.y.strides = [a1, ascir.SizeExpr(1)]
    is_inf.y.size = [a0, a1]
    is_inf.y.dtype = ascir.dtypes.uint8
    logical_or = ascir.ops.LogicalOr("logical_or", graph)
    logical_or.attr.sched.axis = [z0, z1]
    logical_or.x1 = is_inf.y
    logical_or.x2 = loaded[1].y
    logical_or.y.axis = [z0, z1]
    logical_or.y.strides = [a1, ascir.SizeExpr(1)]
    logical_or.y.size = [a0, a1]
    logical_or.y.dtype = ascir.dtypes.uint8
    filled = ascir.ops.MaskedFill("masked_fill", graph)
    filled.attr.sched.axis = [z0, z1]
    filled.x = loaded[0].y
    filled.mask = logical_or.y
    filled.value = loaded[2].y
    filled.y.axis = [z0, z1]
    filled.y.strides = [a1, ascir.SizeExpr(1)]
    filled.y.size = [a0, a1]
    filled.y.dtype = ascir.dtypes.float16
    store = ascir.ops.Store("store", graph)
    store.attr.ir_attr.offset = ascir.SizeExpr(0)
    store.attr.sched.axis = [z0, z1]
    store.x = filled.y
    store.y.axis = [z0, z1]
    store.y.strides = [a1, ascir.SizeExpr(1)]
    store.y.size = [a0, a1]
    store.y.dtype = ascir.dtypes.float16
    output = ascir.ops.Output("output", graph)
    output.attr.ir_attr.index = 0
    output.x = store.y
    output.y.dtype = ascir.dtypes.float16


def build_graph(shape):
    graph, a0, a1, z0, z1 = _create_graph_and_axes("isinf_maskedfill_fusion", shape)
    meta = (a0, a1, z0, z1)
    x, mask, value = _create_data_nodes(graph)
    loaded = _create_loads(graph, meta, x, mask, value)
    _create_compute_and_output(graph, meta, loaded)
    return graph


def generate_codegen(shape, output_dir, profile):
    platform = json.loads(Path(profile).read_text(encoding="utf-8"))["ascir"]
    ascir.utils.set_platform(
        platform["platform"], platform["core_type"], platform["ub_size"]
    )
    fuser = Autofuser(AutofuserOptions())
    fused = fuser.schedule(build_graph(shape))
    tiling_def, host_impl, device_impl = fuser.codegen(fused)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tiling.h").write_text(tiling_def)
    (output / "host_impl.cpp").write_text(host_impl)
    (output / "device_impl.cpp").write_text(device_impl)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", required=True)
    options = parser.parse_args()
    generate_codegen((options.rows, options.cols), options.output_dir, options.profile)
