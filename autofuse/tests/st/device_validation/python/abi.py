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
"""ABI metadata parsing and validation for generated kernel modules."""

import json
import re
from pathlib import Path

LEGACY_TENSOR_ARITIES = ((1, 1), (2, 1), (3, 1))
ABI_KINDS = ("AutofuseLaunch", "AutofuseLaunchV2")


def parse_legacy_launch_signature(source):
    match = re.search(
        r'extern\s+"C"\s+int64_t\s+AutofuseLaunch\s*\((.*?)\)\s*(?:;|\{)', source, re.S
    )
    if not match:
        raise ValueError("AutofuseLaunch declaration not found")
    parameters = [item.strip() for item in match.group(1).split(",") if item.strip()]
    expected_prefix = (r"uint32_t\s+blockDim", r"void\s*\*\s*stream")
    if len(parameters) < 5 or any(
        not re.fullmatch(pattern, parameters[index])
        for index, pattern in enumerate(expected_prefix)
    ):
        raise ValueError("invalid AutofuseLaunch declaration")
    tensor_parameters = parameters[2:-2]
    if len(tensor_parameters) < 2:
        raise ValueError("AutofuseLaunch tensor parameters are missing")
    if not all(
        re.fullmatch(r"void\s*\*\s*(?:input\d+|output\d+)", item)
        for item in tensor_parameters
    ):
        raise ValueError("invalid AutofuseLaunch tensor parameter")
    names = [
        re.search(r"([A-Za-z_]\w*)\s*$", item).group(1) for item in tensor_parameters
    ]
    input_names = [name for name in names if name.startswith("input")]
    output_names = [name for name in names if name.startswith("output")]
    if names != input_names + output_names:
        raise ValueError("invalid AutofuseLaunch tensor parameter order")
    if input_names != [
        f"input{index}" for index in range(len(input_names))
    ] or output_names != [f"output{index}" for index in range(len(output_names))]:
        raise ValueError("invalid AutofuseLaunch tensor parameter numbering")
    input_count = len(input_names)
    output_count = len(output_names)
    if not input_count or not output_count:
        raise ValueError("AutofuseLaunch input/output names are missing")
    if (input_count, output_count) not in LEGACY_TENSOR_ARITIES:
        raise ValueError("unsupported_abi_arity")
    if not re.fullmatch(r"void\s*\*\s*workspace", parameters[-2]) or not re.fullmatch(
        r"[A-Za-z_]\w*\s*\*\s*tiling(?:_data)?", parameters[-1]
    ):
        raise ValueError("invalid AutofuseLaunch trailing parameter")
    return {
        "launch_abi": "AutofuseLaunch",
        "input_count": input_count,
        "output_count": output_count,
        "tensor_parameter_count": len(tensor_parameters),
        "parameter_count": len(parameters),
    }


def parse_v2_launch_signature(source):
    match = re.search(
        r'extern\s+"C"\s+(\w+)\s+AutofuseLaunchV2\s*\((.*?)\)\s*(?:;|\{)', source, re.S
    )
    if not match:
        raise ValueError("AutofuseLaunchV2 declaration not found")
    parameters = [item.strip() for item in match.group(2).split(",") if item.strip()]
    expected = [
        r"uint32_t\s+blockDim",
        r"void\s*\*\s*stream",
        r"void\s*\*\*\s*inputs",
        r"int32_t\s+input_count",
        r"void\s*\*\*\s*outputs",
        r"int32_t\s+output_count",
        r"void\s*\*\s*workspace",
        r"void\s*\*\s*tiling(?:_data)?",
    ]
    if (
        match.group(1) != "uint32_t"
        or len(parameters) != len(expected)
        or any(
            not re.fullmatch(pattern, parameter)
            for parameter, pattern in zip(parameters, expected)
        )
    ):
        raise ValueError("invalid AutofuseLaunchV2 declaration")
    return {
        "launch_abi": "AutofuseLaunchV2",
        "input_count": None,
        "output_count": None,
        "parameter_count": len(parameters),
        "return_type": match.group(1),
    }


def parse_launch_signature(source):
    v2 = re.search(r'extern\s+"C"\s+\w+\s+AutofuseLaunchV2\s*\(', source)
    legacy = re.search(r'extern\s+"C"\s+int64_t\s+AutofuseLaunch\s*\(', source)
    if v2:
        return parse_v2_launch_signature(source)
    if legacy:
        return parse_legacy_launch_signature(source)
    return None


def validate_abi_metadata(metadata, input_count=None, output_count=None):
    if not isinstance(metadata, dict) or metadata.get("launch_abi") not in ABI_KINDS:
        raise ValueError("abi_metadata")
    if metadata.get("launch_abi") == "AutofuseLaunch":
        counts = (metadata.get("input_count"), metadata.get("output_count"))
        if counts not in LEGACY_TENSOR_ARITIES:
            raise ValueError("abi_metadata")
    v2_counts_missing = (
        "input_count" not in metadata
        or "output_count" not in metadata
        or not isinstance(metadata.get("input_count"), int)
        or not isinstance(metadata.get("output_count"), int)
    )
    if metadata.get("launch_abi") == "AutofuseLaunchV2" and v2_counts_missing:
        raise ValueError("abi_metadata")
    declared_counts = "input_count" in metadata or "output_count" in metadata
    count_mismatch = (
        metadata.get("input_count") != input_count
        or metadata.get("output_count") != output_count
    )
    if input_count is not None and declared_counts and count_mismatch:
        raise ValueError("abi_metadata")
    return metadata


def read_abi_metadata(
    artifact, input_count, output_count, input_dtypes=None, output_dtypes=None
):
    path = artifact / "abi_metadata.json"
    if not path.exists():
        raise ValueError("abi_metadata is required")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    validate_abi_metadata(metadata, input_count, output_count)
    if input_dtypes is not None or output_dtypes is not None:
        if metadata.get("input_dtypes") != list(input_dtypes or []) or metadata.get(
            "output_dtypes"
        ) != list(output_dtypes or []):
            raise ValueError("abi_metadata: declared dtype mismatch")
    request_metadata = dict(metadata)
    request_metadata.setdefault("input_count", input_count)
    request_metadata.setdefault("output_count", output_count)
    return request_metadata


def validate_codegen_abi(artifact, metadata, input_dtypes, output_dtypes):
    validate_abi_metadata(metadata, len(input_dtypes), len(output_dtypes))
    if metadata.get("input_dtypes") != list(input_dtypes) or metadata.get(
        "output_dtypes"
    ) != list(output_dtypes):
        raise ValueError("abi_metadata: declared dtype mismatch")
    sources = []
    for name in ("host_impl.cpp", "device_impl.cpp"):
        path = Path(artifact) / name
        if path.exists():
            sources.append(path.read_text(encoding="utf-8"))
    signatures = [parse_launch_signature(source) for source in sources]
    signatures = [signature for signature in signatures if signature is not None]
    if not signatures:
        raise ValueError("abi_metadata: generated launch signature not found")
    normalized = [
        {
            key: signature[key]
            for key in ("launch_abi", "input_count", "output_count", "parameter_count")
        }
        for signature in signatures
    ]
    if any(signature != normalized[0] for signature in normalized[1:]):
        raise ValueError("abi_metadata: generated launch signatures disagree")
    actual = signatures[0]
    if any(
        metadata.get(key) != actual[key]
        for key in ("launch_abi", "input_count", "output_count")
        if actual[key] is not None
    ):
        raise ValueError("abi_metadata: generated launch signature mismatch")
    if (
        metadata.get("launch_abi") == "AutofuseLaunchV2"
        and metadata.get("parameter_count", 8) != actual["parameter_count"]
    ):
        raise ValueError("abi_metadata: generated launch parameter count mismatch")
    return actual
