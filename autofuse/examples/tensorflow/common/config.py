#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TensorFlow AutoFuse 示例的公共配置。"""

import os

# Profiling 数据输出目录。保持与原有用例一致，默认输出到当前工作目录下的 profiling 目录。
PROFILING_DIR = os.path.abspath("./profiling")

# 每个示例执行的推理步数。增加步数可获得更稳定的 Profiling 数据，但会增加运行时间。
RUN_STEPS = 100

# Session 允许 TensorFlow 在目标设备无法执行某个算子时进行设备回退。
ALLOW_SOFT_PLACEMENT = True

# 是否在日志中打印每个算子的实际设备放置信息。默认关闭，避免产生大量日志。
LOG_DEVICE_PLACEMENT = False

# NpuOptimizer 的离线编译开关。True 表示使用离线编译方式生成并执行 NPU 图。
USE_OFF_LINE = True

# NpuOptimizer 图运行模式。0 表示推理场景。
GRAPH_RUN_MODE = 0

# 是否开启 NPU Profiling。开启后会在 PROFILING_DIR 下生成性能采集数据。
PROFILING_MODE = True

# NPU Profiling 采集配置：
# - output：Profiling 数据输出目录。
# - training_trace：采集迭代轨迹信息。
# - task_time：采集 Task 执行时间。
# - hccl：采集 HCCL 通信信息。
# - aicpu：采集 AI CPU 算子信息。
# - aic_metrics：采集 AI Core 指标，此处使用 PipeUtilization。
# - msproftx：是否采集 msproftx 标记信息，本示例关闭。
PROFILING_OPTIONS = (
    '{"output":"%s","training_trace":"on","task_time":"on",'
    '"hccl":"on","aicpu":"on","aic_metrics":"PipeUtilization","msproftx":"off"}'
) % PROFILING_DIR
