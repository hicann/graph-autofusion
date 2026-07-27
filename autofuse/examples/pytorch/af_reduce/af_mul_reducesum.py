#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------
# 导包
import torch
import torch_npu
import torch.nn as nn

# ===== 1. 昇腾 NPU 配置 =====
DEVICE = "npu:0"  # 假设使用0卡
torch.npu.set_device(DEVICE)


# ===== 2. 构造简单模型 =====
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        result = torch.sum(torch.mul(x, y))
        return result

# ===== 3. inductor + 昇腾NPU自动融合后端 =====
model = MyModel().to(DEVICE)
model = torch.compile(
    model,
    dynamic=False,
    fullgraph=True,
    options={"npu_backend": "ascendc"},
)

# ===== 4. 创建输入 =====
x = torch.randn(256, 100, device=DEVICE)
y = torch.randn(256, 100, device=DEVICE)

# ===== 5. 执行 =====
model.eval()

# 开启 profiling
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[torch_npu.profiler.ExportType.Text],
    profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    l2_cache=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None
)

with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./profiling"),
    record_shapes=True,
    profile_memory=False,
    with_stack=False,
    with_modules=False,
    with_flops=False,
    experimental_config=experimental_config) as prof:
    # 跑 100 step
    for _ in range(100):
        result = model(x, y)
