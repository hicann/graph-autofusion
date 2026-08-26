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
"""Build and generate one real ASCIR graph for an unfused step."""

import json
import argparse
from dataclasses import dataclass
from pathlib import Path

from autofuse.pyautofuse import ascir, Autofuser, AutofuserOptions


STEP_DTYPES = {"float16": ascir.dtypes.float16, "uint8": ascir.dtypes.uint8}


@dataclass
class StepCodegenConfig:
    name: str
    op_name: str
    input_dtypes: tuple
    output_dtype: str
    shape: tuple
    output_dir: str
    platform: dict


@dataclass
class GraphContext:
    graph: ascir.HintGraph
    z0: ascir.Axis
    z1: ascir.Axis
    s0: ascir.SizeExpr
    s1: ascir.SizeExpr
    buf_z0: ascir.Axis
    buf_z1: ascir.Axis


def _create_graph_context(name):
    graph = ascir.HintGraph(name)
    s0 = graph.create_size("s0")
    s1 = graph.create_size("s1")
    z0 = graph.create_axis("z0", s0)
    z1 = graph.create_axis("z1", s1)
    buf_z0 = graph.create_axis("buf_z0", s0)
    buf_z1 = graph.create_axis("buf_z1", s1)
    return GraphContext(graph, z0, z1, s0, s1, buf_z0, buf_z1)


def _set_node_output(node, dtype, ctx):
    node.y.dtype = dtype
    node.y.axis = [ctx.z0, ctx.z1]
    node.y.size = [ctx.s0, ctx.s1]
    node.y.strides = [ctx.s1, ascir.SizeExpr(1)]


def _set_node_sched(node, ctx):
    node.attr.sched.axis = [ctx.z0, ctx.z1]


def _create_data_node(graph, ctx, index, dtype):
    node = ascir.ops.Data(f"data_{index}", graph)
    node.attr.ir_attr.index = index
    _set_node_sched(node, ctx)
    dtype_obj = STEP_DTYPES.get(dtype)
    if dtype_obj is None:
        raise ValueError(f"unsupported step dtype: {dtype}")
    _set_node_output(node, dtype_obj, ctx)
    return node


def _build_loaded(graph, ctx, input_dtypes):
    loaded = []
    for index, dtype in enumerate(input_dtypes):
        source = _create_data_node(graph, ctx, index, dtype)
        load = ascir.ops.Load(f"load_{index}")
        load.attr.ir_attr.offset = ascir.SizeExpr(0)
        load.x = source
        _set_node_sched(load, ctx)
        _set_node_output(load, STEP_DTYPES[dtype], ctx)
        loaded.append(load)
    return loaded


def _build_step_op(graph, ctx, op_name, loaded, output_dtype):
    op = getattr(ascir.ops, op_name)(op_name, graph)
    if op_name == "IsInf":
        op.x = loaded[0]
    elif op_name == "LogicalOr":
        op.x1 = loaded[0]
        op.x2 = loaded[1]
    else:
        op.x = loaded[0]
        op.mask = loaded[1]
        op.value = loaded[2]
    _set_node_sched(op, ctx)
    _set_node_output(op, STEP_DTYPES[output_dtype], ctx)
    return op


def _build_step_graph(config):
    ctx = _create_graph_context(config.name)
    loaded = _build_loaded(ctx.graph, ctx, config.input_dtypes)
    op = _build_step_op(ctx.graph, ctx, config.op_name, loaded, config.output_dtype)

    store = ascir.ops.Store("store")
    store.attr.ir_attr.offset = ascir.SizeExpr(0)
    store.x = op
    _set_node_sched(store, ctx)
    _set_node_output(store, STEP_DTYPES[config.output_dtype], ctx)
    output = ascir.ops.Output("output", ctx.graph)
    output.attr.ir_attr.index = 0
    output.x = store
    _set_node_sched(output, ctx)
    _set_node_output(output, STEP_DTYPES[config.output_dtype], ctx)
    ctx.graph.set_axis_map({ctx.z0: [ctx.buf_z0], ctx.z1: [ctx.buf_z1]})
    return ctx.graph


def _write_artifacts(tiling, host, device, config):
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "tiling.h").write_text(tiling)
    (output_path / "host_impl.cpp").write_text(host)
    (output_path / "device_impl.cpp").write_text(device)
    launch_abi = "AutofuseLaunchV2" if "AutofuseLaunchV2" in host else "AutofuseLaunch"
    (output_path / "abi_metadata.json").write_text(
        json.dumps(
            {
                "launch_abi": launch_abi,
                "input_count": len(config.input_dtypes),
                "output_count": 1,
                "input_dtypes": list(config.input_dtypes),
                "output_dtypes": [config.output_dtype],
            },
            indent=2,
        )
    )
    return output_path


def generate_step_codegen(config):
    ascir.utils.set_platform(
        config.platform.get("platform"),
        config.platform.get("core_type"),
        config.platform.get("ub_size"),
    )
    graph = _build_step_graph(config)
    fuser = Autofuser(AutofuserOptions())
    tiling, host, device = fuser.codegen(fuser.schedule(graph))
    return _write_artifacts(tiling, host, device, config)


def run_step_codegen(graph_name, op_name, input_dtypes, output_dtype):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", required=True)
    options = parser.parse_args()
    config = StepCodegenConfig(
        name=graph_name,
        op_name=op_name,
        input_dtypes=input_dtypes,
        output_dtype=output_dtype,
        shape=(options.rows, options.cols),
        output_dir=options.output_dir,
        platform=load_platform(options.profile),
    )
    return generate_step_codegen(config)


def load_platform(profile):
    data = json.loads(Path(profile).read_text(encoding="utf-8"))
    platform = data.get("ascir")
    if not isinstance(platform, dict) or not isinstance(platform.get("platform"), str):
        raise ValueError("profile is missing ascir platform")
    return platform
