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
"""Backend subprocess orchestration with a flat JSON request."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import tempfile


@dataclass(frozen=True)
class ExecutorRequest:
    module: str
    abi: str
    device: int
    tensor_files: tuple[str, ...]
    tensor_specs: tuple[dict, ...]
    warmup: int
    repeat: int
    variant: str = "fused"
    artifact_dir: str = ""
    case_dir: str = ""
    profile: str = ""
    shape: tuple[int, ...] = ()
    selected_shape: tuple[int, ...] = ()
    soc_profile: str = ""
    profiler: bool = False
    profile_dir: str = ""
    output_specs: tuple[dict, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    launch_abi: str = "AutofuseLaunchV2"
    input_count: int = 0
    output_count: int = 0
    abi_metadata: dict = None
    case_id: str = ""
    step: int = -1
    contract_schema: dict = None
    aclnn_op: str = ""


def request_payload(request):
    """Serialize an ExecutorRequest, omitting empty optional fields.

    Module-internal orchestration helper kept public for test access.
    ``aclnn_op`` is only emitted when the step runs via a CANN aclnn op, so
    the legacy flat request contract stays byte-identical for codegen steps.
    """
    payload = asdict(request)
    if not payload.get("aclnn_op"):
        payload.pop("aclnn_op", None)
    return payload


@dataclass(frozen=True)
class ExecutorResult:
    payload: dict
    stdout: str
    stderr: str
    returncode: int
    request_path: str


def _failure_payload(error_code, reason, returncode=None):
    payload = {
        "schema_version": 2,
        "stage": "execution",
        "stage_status": "failed",
        "error_code": error_code,
        "reason": reason,
    }
    if returncode is not None:
        payload["returncode"] = returncode
    return payload


def _as_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def execute_command(
    command, timeout, artifact_dir, output_names, subprocess_module=subprocess
):
    """Run a legacy CLI subprocess and retain its structured stdout/artifacts."""
    completed = subprocess_module.run(
        command, text=True, capture_output=True, timeout=timeout
    )
    root = Path(artifact_dir)
    (root / output_names[0]).write_text(completed.stdout, encoding="utf-8")
    (root / output_names[1]).write_text(completed.stderr, encoding="utf-8")
    return completed


def _result(request_path, payload, stdout, stderr, returncode):
    return ExecutorResult(payload, stdout, stderr, returncode, str(request_path))


def _try_decode_last(lines):
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _decode_payload(lines, returncode, stderr):
    if returncode != 0 and lines:
        payload = _try_decode_last(lines)
        if payload is not None:
            return payload
    if returncode != 0:
        return _failure_payload(
            "backend_returncode",
            stderr.strip() or "backend returned non-zero",
            returncode,
        )
    if not lines:
        return _failure_payload(
            "empty_stdout", stderr.strip() or "executor emitted no structured payload"
        )
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        payload = _failure_payload("invalid_json", str(error))
    if not isinstance(payload, dict):
        payload = _failure_payload(
            "invalid_json_payload", "executor payload must be a JSON object"
        )
    return payload


def run_subprocess(command, request, timeout, artifact_dir=None, append_request=True):
    root = (
        Path(artifact_dir)
        if artifact_dir
        else Path(tempfile.mkdtemp(prefix="device-validation-"))
    )
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "executor_request.json"
    request_path.write_text(json.dumps(request_payload(request)), encoding="utf-8")
    stdout_path = root / "backend.stdout"
    stderr_path = root / "backend.stderr"
    actual_command = [*command, str(request_path)] if append_request else list(command)
    try:
        completed = subprocess.run(
            actual_command, text=True, capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        stdout = _as_text(error.stdout)
        stderr = _as_text(error.stderr)
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        return _result(
            request_path,
            _failure_payload("timeout", stderr or "backend timed out"),
            stdout,
            stderr,
            -1,
        )
    lines = completed.stdout.strip().splitlines()
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    payload = _decode_payload(lines, completed.returncode, completed.stderr)
    return _result(
        request_path,
        payload,
        completed.stdout,
        completed.stderr,
        completed.returncode,
    )
