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
"""TensorFlow AutoFuse 示例的公共运行框架。"""

import argparse

import tensorflow as tf

from .config import (
    ALLOW_SOFT_PLACEMENT,
    GRAPH_RUN_MODE,
    LOG_DEVICE_PLACEMENT,
    PROFILING_MODE,
    PROFILING_OPTIONS,
    RUN_STEPS,
    USE_OFF_LINE,
)
from .profiling_utils import export_new_profiling, get_profile_dirs


def configure_npu(sess_config):
    """为 TensorFlow Session 配置 NpuOptimizer 和 Profiling 参数。"""
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"

    # 使用离线编译模式执行 NPU 图。
    custom_op.parameter_map["use_off_line"].b = USE_OFF_LINE

    # graph_run_mode=0 表示推理模式。
    custom_op.parameter_map["graph_run_mode"].i = GRAPH_RUN_MODE

    # 开启 Profiling，并将公共 Profiling 配置传递给 NpuOptimizer。
    custom_op.parameter_map["profiling_mode"].b = PROFILING_MODE
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
        PROFILING_OPTIONS
    )
    return sess_config


def run_model(build_model, placeholder_fn, configproto_fn):
    """构建用例模型，并使用统一的 Session/NPU 配置执行推理。"""
    profile_dirs_before = get_profile_dirs()

    # 各用例只负责定义模型和输入数据，并返回待执行 Tensor 与 feed_dict。
    output_tensor, feed_dict = build_model(placeholder_fn)

    sess_config = configproto_fn(
        allow_soft_placement=ALLOW_SOFT_PLACEMENT,
        log_device_placement=LOG_DEVICE_PLACEMENT,
    )
    configure_npu(sess_config)

    with tf.compat.v1.Session(config=sess_config) as sess:
        for _ in range(RUN_STEPS):
            sess.run(output_tensor, feed_dict=feed_dict)

    export_new_profiling(profile_dirs_before)


def run_tf1(build_model):
    """使用 TF1 + npu_bridge 方式运行用例。"""
    import npu_bridge

    _ = npu_bridge  # 通过 import 副作用注册 NPU 算子。
    run_model(build_model, tf.placeholder, tf.ConfigProto)


def run_tf2_compat(build_model):
    """使用 TF2 的 v1 兼容模式 + npu_device 运行用例。"""
    import npu_device
    import npu_device.compat

    _ = npu_device  # 保留 npu_device 导入，确保 NPU 相关能力完成注册。
    npu_device.compat.enable_v1()
    run_model(build_model, tf.compat.v1.placeholder, tf.compat.v1.ConfigProto)


def run_example(build_model, description):
    """解析运行模式，并调用对应 TensorFlow 运行入口。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=["tf1", "tf2-compat"], required=True)
    mode = parser.parse_args().mode

    if mode == "tf1":
        run_tf1(build_model)
    else:
        run_tf2_compat(build_model)
