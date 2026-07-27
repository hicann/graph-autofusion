#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# TensorFlow 场景 autofuse 示例（abs -> relu -> exp 逐元素融合）。
# 支持 TF1（npu_bridge）和 TF2 兼容模式（npu_device.compat），通过 --mode 选择。
#
# 用法：
#   TF1 环境：  python3 test_abs_relu_exp.py --mode tf1
#   TF2 环境：  python3 test_abs_relu_exp.py --mode tf2-compat
#

import argparse

import numpy as np
import tensorflow as tf

_PROFILES_JSON = (
    '{"output":"./profiling","training_trace":"on","task_time":"on",'
    '"hccl":"on","aicpu":"on","aic_metrics":"PipeUtilization","msproftx":"off"}'
)


def configure_npu(sess_config):
    """在已有 sess_config 上配置 NpuOptimizer（推理模式 + profiling）。"""
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["graph_run_mode"].i = 0
    custom_op.parameter_map["profiling_mode"].b = True
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(_PROFILES_JSON)
    return sess_config


def run_model(placeholder_fn, configproto_fn):
    """构建 abs->relu->exp 模型并在 NPU 上执行 100 步。"""
    data1 = placeholder_fn(tf.float16, shape=[128, 192])
    input_data = np.random.rand(128, 192).astype(np.float16)
    abs_0 = tf.abs(data1)
    relu_0 = tf.nn.relu(abs_0)
    exp_0 = tf.exp(relu_0)
    sess_config = configproto_fn(allow_soft_placement=True, log_device_placement=False)
    configure_npu(sess_config)
    feed_dict = {data1: input_data}
    step = 100
    with tf.compat.v1.Session(config=sess_config) as sess:
        for _ in range(step):
            sess.run(exp_0, feed_dict=feed_dict)


def run_tf1():
    """TF1 场景：通过 npu_bridge 注册 NPU 算子。"""
    import npu_bridge

    _ = npu_bridge  # 通过import副作用注册NPU算子
    run_model(tf.placeholder, tf.ConfigProto)


def run_tf2_compat():
    """TF2 兼容场景：通过 npu_device.compat 切到 v1 行为。"""
    import npu_device
    import npu_device.compat

    npu_device.compat.enable_v1()
    run_model(tf.compat.v1.placeholder, tf.compat.v1.ConfigProto)


_MODES = {"tf1": run_tf1, "tf2-compat": run_tf2_compat}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="autofuse abs-relu-exp 示例")
    parser.add_argument("--mode", choices=list(_MODES), required=True)
    _MODES[parser.parse_args().mode]()
