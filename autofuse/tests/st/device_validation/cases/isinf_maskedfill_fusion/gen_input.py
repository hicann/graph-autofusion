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
import numpy as np


def generate_inputs(shape, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-4, 4, size=shape).astype(np.float16)
    mask = np.zeros(shape, dtype=np.uint8)
    value = np.full(shape, 0.5, dtype=np.float16)
    flat_x = x.reshape(-1)
    flat_mask = mask.reshape(-1)
    flat_x[0] = np.inf
    flat_x[1] = -np.inf
    flat_mask[2] = 1
    flat_x[3] = np.inf
    flat_mask[3] = 1
    return x, mask, value
