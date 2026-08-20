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
#
# TensorFlow 场景 AutoFuse 示例（abs -> add(broadcast) -> relu）。
# 支持 TF1（npu_bridge）和 TF2 兼容模式（npu_device.compat），通过 --mode 选择。
#
# 用法：
#   TF1 环境：  python3 test_abs_add_relu.py --mode tf1
#   TF2 环境：  python3 test_abs_add_relu.py --mode tf2-compat
#

import os
import sys

import numpy as np
import tensorflow as tf

# 将 tensorflow 示例根目录加入模块搜索路径，便于直接运行当前脚本时复用 common 中的公共能力。
_TENSORFLOW_EXAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TENSORFLOW_EXAMPLE_DIR)

from common.tf_runner import run_example  # noqa: E402


def build_model(placeholder_fn):
    """构建 abs -> add(broadcast) -> relu 计算图及对应输入数据。"""
    data1 = placeholder_fn(tf.float16, shape=[128, 192])
    data2 = placeholder_fn(tf.float16, shape=[192])

    input_data1 = np.random.uniform(-1.0, 1.0, size=(128, 192)).astype(np.float16)
    input_data2 = np.random.uniform(-1.0, 1.0, size=(192,)).astype(np.float16)

    abs_0 = tf.abs(data1)
    add_0 = tf.add(abs_0, data2)
    relu_0 = tf.nn.relu(add_0)

    return relu_0, {data1: input_data1, data2: input_data2}


if __name__ == "__main__":
    run_example(
        build_model,
        description="AutoFuse abs-add(broadcast)-relu 示例",
    )
