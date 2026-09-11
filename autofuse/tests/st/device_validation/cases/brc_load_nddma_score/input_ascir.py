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

from autofuse.pyautofuse import Autofuser, AutofuserOptions, ascir


def build_graph(shape):
    rows, cols = shape
    graph = ascir.HintGraph("brc_load_nddma_score")
    a0 = ascir.SizeExpr(rows)
    a1 = ascir.SizeExpr(cols)
    z0 = graph.create_axis("z0", a0)
    z1 = graph.create_axis("z1", a1)

    data = ascir.ops.Data("data", graph)
    data.attr.ir_attr.index = 0
    data.y.dtype = ascir.dtypes.float16

    load = ascir.ops.Load("load", graph)
    load.attr.sched.axis = [z0, z1]
    load.attr.ir_attr.offset = ascir.SizeExpr(0)
    load.x = data.y
    load.y.axis = [z0, z1]
    load.y.size = [ascir.SizeExpr(1), a1]
    load.y.strides = [ascir.SizeExpr(0), ascir.SizeExpr(1)]
    load.y.dtype = ascir.dtypes.float16

    broadcast = ascir.ops.Broadcast("broadcast", graph)
    broadcast.attr.sched.axis = [z0, z1]
    broadcast.x = load.y
    broadcast.y.axis = [z0, z1]
    broadcast.y.size = [a0, a1]
    broadcast.y.strides = [a1, ascir.SizeExpr(1)]
    broadcast.y.dtype = ascir.dtypes.float16

    exp = ascir.ops.Exp("exp", graph)
    exp.attr.sched.axis = [z0, z1]
    exp.x = broadcast.y
    exp.y.axis = [z0, z1]
    exp.y.size = [a0, a1]
    exp.y.strides = [a1, ascir.SizeExpr(1)]
    exp.y.dtype = ascir.dtypes.float16

    absolute = ascir.ops.Abs("abs", graph)
    absolute.attr.sched.axis = [z0, z1]
    absolute.x = exp.y
    absolute.y.axis = [z0, z1]
    absolute.y.size = [a0, a1]
    absolute.y.strides = [a1, ascir.SizeExpr(1)]
    absolute.y.dtype = ascir.dtypes.float16

    multiply = ascir.ops.Mul("mul", graph)
    multiply.attr.sched.axis = [z0, z1]
    multiply.x1 = exp.y
    multiply.x2 = absolute.y
    multiply.y.axis = [z0, z1]
    multiply.y.size = [a0, a1]
    multiply.y.strides = [a1, ascir.SizeExpr(1)]
    multiply.y.dtype = ascir.dtypes.float16

    store = ascir.ops.Store("store", graph)
    store.attr.sched.axis = [z0, z1]
    store.attr.ir_attr.offset = ascir.SizeExpr(0)
    store.x = multiply.y
    store.y.axis = [z0, z1]
    store.y.size = [a0, a1]
    store.y.strides = [a1, ascir.SizeExpr(1)]
    store.y.dtype = ascir.dtypes.float16

    output = ascir.ops.Output("output", graph)
    output.attr.ir_attr.index = 0
    output.x = store.y
    output.y.dtype = ascir.dtypes.float16
    graph.infer_dtypes()
    return graph


def generate_codegen(shape, output_dir, profile):
    platform = json.loads(Path(profile).read_text(encoding="utf-8"))["ascir"]
    ascir.utils.set_platform(
        platform["platform"], platform["core_type"], platform["ub_size"]
    )
    fuser = Autofuser(AutofuserOptions(graph_type=1))
    scheduled = fuser.schedule(build_graph(shape))
    tiling_def, host_impl, device_impl = fuser.codegen(scheduled)
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
