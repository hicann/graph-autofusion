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
"""Shared construction helpers for device validation tests.

These helpers only build fixtures; assertions and case-specific semantics
stay in the tests that use them.
"""

import json
import math
import subprocess
from pathlib import Path

_DTYPE_BYTE_WIDTHS = {"float16": 2, "bfloat16": 2, "uint8": 1, "float32": 4, "int32": 4}


def cli_args(case_dir, mode, shape=None, output_dir=None, **overrides):
    """Build a CLI argv list.

    ``shape`` as an (h, w) tuple appends two ``--shape`` values; keys in
    ``overrides`` have ``_`` mapped to ``-`` as flags, ``True`` adds the flag
    without a value, ``False`` skips the override, everything else appends as
    a value.
    """
    argv = ["--case", str(case_dir), "--mode", mode]
    if shape is not None:
        argv += ["--shape", str(shape[0]), str(shape[1])]
    if output_dir is not None:
        argv += ["--output-dir", str(output_dir)]
    for flag, value in overrides.items():
        argv.append(f"--{flag.replace('_', '-')}")
        if value is not True and value is not False:
            argv.append(str(value))
    return argv


def passed_payload(shape, dtype="float16"):
    """Build a passed runner payload.

    ``shape`` is used as a JSON-serializable list in ``run_parameters``; the
    payload carries schema_version 2, a passed stage status and precision, a
    performance section with samples/repeat/warmup, and complete
    run_parameters/artifact_paths.
    """
    samples = [1.5, 2.0, 2.5]
    repeat = len(samples)
    return {
        "schema_version": 2,
        "stage_status": "passed",
        "precision": {"passed": True},
        "performance": {
            "declared": {},
            "actual": {
                "samples": samples,
                "sample_count": repeat,
                "warmup": 2,
                "repeat": repeat,
                "metric": "runner_wall_clock",
                "unit": "ms",
            },
            "status": "passed",
        },
        "run_parameters": {"shape": list(shape), "dtype": dtype},
        "artifact_paths": [],
    }


def mocked_abi_metadata(
    artifact, input_count, output_count, input_dtypes=None, output_dtypes=None
):
    """Stub for run_device_validation.read_abi_metadata in test monkeypatches."""
    return {
        "launch_abi": "AutofuseLaunchV2",
        "input_count": input_count,
        "output_count": output_count,
        "input_dtypes": list(input_dtypes or []),
        "output_dtypes": list(output_dtypes or []),
    }


def write_tensor_file(path, dtype, shape, value=0):
    """Write a flat binary tensor file readable by the runner and verifier.

    ``dtype`` maps to a byte width (float16/uint8/...); ``shape`` may be an
    integer or a tuple/list of dimensions.
    """
    byte_width = _DTYPE_BYTE_WIDTHS[dtype]
    size = shape if isinstance(shape, int) else math.prod(shape)
    Path(path).write_bytes(bytes([value]) * (byte_width * size))


def flat_request_runner(runner, tmp_path, request, name="request.json"):
    """Write a request file and run the flat validation runner against it.

    Returns ``(result, payload)`` where the payload is decoded from the last
    stdout line of the runner result.
    """
    request_path = tmp_path / name
    request_path.write_text(json.dumps(request), encoding="utf-8")
    result = subprocess.run(
        [str(runner), "--request", str(request_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return result, payload


def flat_case_runner(runner, case_argv, shape=None):
    """Run the flat validation runner with positional case arguments.

    ``case_argv`` holds the positional case arguments (case dir, profile,
    module, input files). ``shape`` as an (rows, cols) pair appends
    ``--shape`` values. Returns ``(result, payload)`` where the payload is
    decoded from the last stdout line of the runner result.
    """
    command = [str(runner), *[str(arg) for arg in case_argv]]
    if shape is not None:
        command += ["--shape", str(shape[0]), str(shape[1])]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return result, payload
