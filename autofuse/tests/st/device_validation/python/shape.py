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
"""Checked tensor shape and byte-size calculations."""

import struct

import numpy as np

INT64_MAX = (1 << 63) - 1
SIZE_MAX = (1 << (8 * struct.calcsize("P"))) - 1

DTYPE_SIZES = {
    "float16": 2,
    "bfloat16": 2,
    "uint8": 1,
    "float32": 4,
    "int8": 1,
    "int32": 4,
    "int64": 8,
    "uint32": 4,
    "uint64": 8,
    "bool": 1,
}
DTYPE_NUMPY = {name: np.dtype(name) for name in DTYPE_SIZES if name != "bfloat16"}
DTYPE_NUMPY["bfloat16"] = np.dtype("uint16")


def checked_shape_size(shape, dtype_size=1):
    """Return (shape, element_count, byte_count) after checked arithmetic."""
    if not isinstance(shape, (list, tuple)) or not shape:
        raise ValueError("shape must contain positive integers")
    if type(dtype_size) is not int or dtype_size <= 0:
        raise ValueError("dtype size must be a positive integer")
    elements = 1
    checked_shape = []
    for dimension in shape:
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("shape must contain positive integers")
        if elements > INT64_MAX // dimension or elements > SIZE_MAX // dimension:
            raise ValueError("shape element count overflows")
        elements *= dimension
        checked_shape.append(dimension)
    if elements > INT64_MAX // dtype_size or elements > SIZE_MAX // dtype_size:
        raise ValueError("shape byte size overflows")
    return tuple(checked_shape), elements, elements * dtype_size
