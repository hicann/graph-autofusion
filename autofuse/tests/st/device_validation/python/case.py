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
"""Typed case contract loading. Policy lives here, not in the C++ runner."""

from dataclasses import dataclass
import json
from pathlib import Path

try:
    from .shape import DTYPE_SIZES, checked_shape_size
except ImportError:
    from shape import DTYPE_SIZES, checked_shape_size


SUPPORTED_DTYPES = frozenset(
    (
        "float16",
        "bfloat16",
        "uint8",
        "float32",
        "int8",
        "int32",
        "int64",
        "uint32",
        "uint64",
        "bool",
    )
)


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    inputs: tuple[dict, ...]
    outputs: tuple[dict, ...]
    support_matrix: tuple[dict, ...]
    shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    output_dtypes: tuple[str, ...]
    variants: tuple[str, ...]
    raw: dict
    case_dir: str = ""

    def variant_config(self, variant):
        raw_variants = self.raw.get("variants", {})
        if not isinstance(raw_variants, dict):
            raw_variants = {name: {} for name in raw_variants}
        config = dict(raw_variants.get(variant, {}))
        if variant == "fused" and not config:
            config = {"codegen_entry": "input_ascir.py"}
        return config

    def steps(self, variant):
        config = self.variant_config(variant)
        return tuple(dict(step) for step in config.get("steps", ()))


def _validate_shape(shape):
    return checked_shape_size(shape)[0]


def load_typed_case(case_dir):
    case_dir = Path(case_dir)
    path = case_dir / "case.json"
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported or missing case schema version")
    inputs = tuple(dict(item) for item in raw.get("inputs", ()))
    outputs = tuple(dict(item) for item in raw.get("outputs", ()))
    if not inputs or not outputs:
        raise ValueError("case must declare inputs and outputs")
    if any(item.get("dtype") not in SUPPORTED_DTYPES for item in (*inputs, *outputs)):
        raise ValueError("unsupported dtype")
    for item in (*inputs, *outputs):
        if "shape" in item:
            checked_shape_size(item["shape"], dtype_size=DTYPE_SIZES[item["dtype"]])
    matrix = tuple(dict(item) for item in raw.get("support_matrix", ()))
    if not matrix:
        raise ValueError("case must declare support_matrix")
    shapes = tuple(
        _validate_shape(shape) for entry in matrix for shape in entry.get("shapes", ())
    )
    raw_variants = raw.get("variants", {})
    if isinstance(raw_variants, dict):
        variant_names = tuple(raw_variants) or ("fused",)
    else:
        variant_names = tuple(raw_variants) or ("fused",)
    if any(not isinstance(name, str) or not name for name in variant_names):
        raise ValueError("invalid variant")
    return CaseConfig(
        raw.get("case_id", ""),
        inputs,
        outputs,
        matrix,
        shapes,
        tuple(item.get("dtype") for item in inputs),
        tuple(item.get("dtype") for item in outputs),
        variant_names,
        raw,
        str(case_dir),
    )
