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
"""Host orchestrator for real-device validation cases."""

import argparse
import ctypes
import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import shlex
from pathlib import Path

import numpy as np

try:
    from device_validation.python.abi import (
        parse_launch_signature,
        read_abi_metadata,
        validate_codegen_abi,
    )
    from device_validation.python.case import load_typed_case
    from device_validation.python.executor import (
        ExecutorRequest,
        ExecutorResult,
        _failure_payload,
        request_payload,
        execute_command,
        run_subprocess,
    )
    from device_validation.python.kernel_activity import KernelActivityError
    from device_validation.python.profiler import (
        ProfilerExportError,
        ProfilerUnavailableError,
    )
    from device_validation.python.report import build_report
    from device_validation.python.shape import DTYPE_SIZES, checked_shape_size
    from device_validation.python.support_matrix import (
        SupportConfig,
        resolve_support,
    )
    from device_validation.python.verification import (
        EXIT_CODES,
        classify_backend_payload,
        classify_failure,
        decode_output_payload,
        report_precision,
        verify_outputs,
    )
except ImportError:  # Keep direct execution compatible with the old PYTHONPATH.
    from abi import parse_launch_signature, read_abi_metadata, validate_codegen_abi
    from case import load_typed_case
    from executor import (
        ExecutorRequest,
        ExecutorResult,
        _failure_payload,
        request_payload,
        execute_command,
        run_subprocess,
    )
    from kernel_activity import KernelActivityError
    from profiler import ProfilerExportError, ProfilerUnavailableError
    from report import build_report
    from device_validation.python.verification import (
        EXIT_CODES,
        classify_backend_payload,
        classify_failure,
        decode_output_payload,
        report_precision,
        verify_outputs,
    )


REPORT_SCHEMA_VERSION = 2
MAX_TOLERANCE = 1.0


@dataclasses.dataclass
class UnfusedContext:
    case: dict
    case_dir: Path
    shape: tuple
    args: argparse.Namespace
    run_parameters: dict
    artifact: Path
    runner_command: list[str]
    msprof_path: str | None
    metric: str
    previous: str | None = None
    previous_spec: dict | None = None
    step_samples: list = dataclasses.field(default_factory=list)
    collected_step_reports: list = dataclasses.field(default_factory=list)
    final_outputs: list = dataclasses.field(default_factory=list)
    step_actual: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class CliReportOptions:
    case: dict
    backend: str
    soc: str
    status: str
    reason: str
    artifact: Path
    variant: str
    precision: dict | None = None
    performance: dict | None = None
    support: dict | None = None
    run_parameters: dict | None = None
    stage: str | None = None
    error_code: str = ""
    step: int = -1
    actual_abi: str = ""
    requested_abi: str = ""
    abi_metadata: dict | None = None
    steps: list | None = None


@dataclasses.dataclass
class ReportDocument:
    case_id: str
    backend: str
    soc: str
    status: str
    reason: str
    artifact: Path
    stage: str | None = None
    support: dict | None = None
    precision: dict | None = None
    performance: dict | None = None
    run_parameters: dict | None = None
    error_code: str | None = None
    variant: str = ""
    step: int = -1
    actual_abi: str = ""
    requested_abi: str = ""
    abi_metadata: dict | None = None
    steps: list | None = None


@dataclasses.dataclass(frozen=True)
class CodegenOptions:
    case_dir: Path
    artifact: Path
    shape: tuple
    variant: str = "fused"
    codegen_entry: str | None = None
    profile: str | None = None
    input_dtypes: list | None = None
    output_dtypes: list | None = None


@dataclasses.dataclass
class ProfiledRun:
    runner_command: list
    request: ExecutorRequest
    artifact_dir: Path
    msprof_path: str | None
    kernel_names: list
    timeout: int = 600


@dataclasses.dataclass(frozen=True)
class FusedContext:
    case: dict
    case_dir: Path
    shape: tuple
    args: argparse.Namespace
    support: dict
    artifact: Path
    run_parameters: dict
    goldens: list
    metric: str
    msprof_path: str | None


def load_case(case_dir):
    return load_typed_case(case_dir).raw


def select_support(case, backend, soc_profile, variant="fused", shape=None):
    if isinstance(variant, (tuple, list)) and shape is None:
        shape = tuple(variant)
        variant = "fused"
    typed_case = load_typed_case_from_value(case)
    decision = resolve_support(
        typed_case, backend, soc_profile, shape, SupportConfig(variant)
    )
    if decision.status == "supported" and decision.entry is not None:
        return decision.entry
    return decision


def load_typed_case_from_value(case):
    if hasattr(case, "support_matrix"):
        return case
    # Typed validation is performed by the policy module; this adapter retains
    # the legacy dict API used by existing callers and tests.
    return (
        load_typed_case(case.get("_case_dir", "."))
        if isinstance(case, dict) and "_case_dir" in case
        else _typed_from_dict(case)
    )


def _typed_from_dict(case):
    try:
        from device_validation.python.case import CaseConfig
    except ImportError:
        from python.case import CaseConfig
    entries = tuple(dict(entry) for entry in case.get("support_matrix", ()))
    shapes = tuple(
        tuple(shape) for entry in entries for shape in entry.get("shapes", ())
    )
    return CaseConfig(
        case_id=case.get("case_id", ""),
        case_dir=case.get("_case_dir", ""),
        inputs=tuple(case.get("inputs", ())),
        outputs=tuple(case.get("outputs", ())),
        support_matrix=entries,
        shapes=shapes,
        input_dtypes=tuple(item.get("dtype") for item in case.get("inputs", ())),
        output_dtypes=tuple(item.get("dtype") for item in case.get("outputs", ())),
        variants=tuple(case.get("variants", {})) or ("fused",),
        raw=case,
    )


def _validate_case_shapes(case, selected_shape):
    checked_shape_size(selected_shape)
    for tensor in (*case["inputs"], *case["outputs"]):
        dtype = tensor.get("dtype")
        if dtype not in DTYPE_SIZES:
            raise ValueError("unsupported dtype")
        shape = selected_shape if tensor.get("dynamic", False) else tensor.get("shape")
        if shape is not None:
            checked_shape_size(shape, DTYPE_SIZES[dtype])


def validate_case(case, shape, backend, soc_profile, variant="fused"):
    if (
        case.get("schema_version") != 1
        or not case.get("inputs")
        or not case.get("outputs")
    ):
        raise ValueError("invalid case schema or tensor count")
    _validate_case_shapes(case, shape)
    entry = select_support(case, backend, soc_profile, variant, shape)
    if not isinstance(entry, dict):
        raise ValueError(entry.reason)
    if list(shape) not in entry.get("shapes", []):
        raise ValueError(f"unsupported shape profile: {list(shape)}")
    input_dtypes = [item.get("dtype") for item in case["inputs"]]
    output_dtypes = [item.get("dtype") for item in case["outputs"]]
    if input_dtypes != entry.get("input_dtypes"):
        raise ValueError("case/support mixed input dtype signature mismatch")
    if not set(output_dtypes).issubset(entry.get("output_dtypes", [])):
        raise ValueError("case/support mixed output dtype signature mismatch")
    verification = case.get("verification", {})
    atol = verification.get("atol")
    rtol = verification.get("rtol")
    tolerance_bad = (
        not isinstance(atol, (int, float))
        or not isinstance(rtol, (int, float))
        or not np.isfinite(atol)
        or not np.isfinite(rtol)
        or atol < 0
        or rtol < 0
        or atol > MAX_TOLERANCE
        or rtol > MAX_TOLERANCE
    )
    if tolerance_bad:
        raise ValueError("invalid verification tolerance")
    return entry


def create_artifact_dir(output_dir, case_id):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{case_id}-", dir=root))


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inputs(case_dir, artifact, shape):
    generator = _load_module(
        Path(case_dir) / "gen_input.py", "device_validation_gen_input"
    )
    reference = _load_module(
        Path(case_dir) / "reference.py", "device_validation_reference"
    )
    inputs = generator.generate_inputs(shape)
    for index, value in enumerate(inputs):
        np.asarray(value).tofile(artifact / f"input_{index}.bin")
    golden = reference.compute_reference(*inputs)
    goldens = list(golden) if isinstance(golden, (list, tuple)) else [golden]
    for index, value in enumerate(goldens):
        np.asarray(value).tofile(artifact / f"output_{index}.bin")
    return inputs, goldens


def validate_step_outputs(request, payload):
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(request.outputs):
        raise ValueError("output file count mismatch")
    for path, spec, output in zip(request.outputs, request.output_specs, outputs):
        output_path = Path(path)
        if not output_path.is_file():
            raise ValueError("output file is unavailable")
        dtype = spec.get("dtype")
        shape = spec.get("shape", ())
        if dtype not in DTYPE_SIZES:
            raise ValueError("invalid output specification")
        shape, _, expected_size = checked_shape_size(shape, DTYPE_SIZES[dtype])
        if output.get("dtype") != dtype or tuple(output.get("shape", ())) != shape:
            raise ValueError("output file dtype or shape mismatch")
        if output_path.stat().st_size != expected_size:
            raise ValueError("output file size mismatch")


def _jit_graph_name(case_dir):
    graph_name = "autofuse"
    case_contract = Path(case_dir) / "case.json"
    if case_contract.is_file():
        contract = json.loads(case_contract.read_text(encoding="utf-8"))
        graph_name = contract.get("graph_name", "autofuse")
        if not isinstance(graph_name, str) or not graph_name:
            graph_name = "autofuse"
    return graph_name


def _jit_module(artifact, shape, profile, case_dir):
    jit = os.environ.get("AUTOFUSE_DEVICE_JIT")
    if not jit:
        raise FileNotFoundError(
            "AUTOFUSE_DEVICE_JIT is not set; codegen completed but JIT is unavailable"
        )
    jit_path = Path(jit)
    if not jit_path.is_file() or not os.access(jit_path, os.X_OK):
        raise FileNotFoundError("AUTOFUSE_DEVICE_JIT must be a regular executable file")
    destination = artifact / "kernel_module.so"
    graph_name = _jit_graph_name(case_dir)
    result = execute_command(
        [
            str(jit_path),
            "--codegen-dir",
            str(artifact),
            "--output",
            str(destination),
            "--rows",
            str(shape[0]),
            "--cols",
            str(shape[1]),
            "--profile",
            str(profile),
            "--graph-name",
            graph_name,
        ],
        300,
        artifact,
        ("jit.stdout", "jit.stderr"),
        subprocess,
    )
    if result.returncode != 0 or not destination.is_file():
        raise RuntimeError("JIT failed to produce kernel module")
    try:
        module = ctypes.CDLL(str(destination))
        get_size = module.GetTilingDataSize
        get_size.restype = ctypes.c_size_t
        if get_size() == 0:
            raise RuntimeError("kernel module returned an empty tiling contract")
    except (AttributeError, OSError, RuntimeError) as error:
        raise RuntimeError("kernel module minimal invocation failed") from error
    return destination


def _ensure_abi_metadata(artifact, input_dtypes, output_dtypes):
    required_generated = (
        "tiling.h",
        "host_impl.cpp",
        "device_impl.cpp",
        "kernel_module.so",
    )
    missing = [name for name in required_generated if not (artifact / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "real_codegen missing generated artifacts: " + ", ".join(missing)
        )
    metadata_path = artifact / "abi_metadata.json"
    if not metadata_path.is_file():
        device_source = (artifact / "device_impl.cpp").read_text(encoding="utf-8")
        signature = parse_launch_signature(device_source)
        if signature is None:
            raise ValueError("abi_metadata: generated launch signature not found")
        if signature["launch_abi"] == "AutofuseLaunchV2":
            if not input_dtypes:
                raise ValueError(
                    "abi_metadata: v2 input dtypes are required when metadata is missing"
                )
            signature["input_count"] = len(input_dtypes)
            signature["output_count"] = len(output_dtypes or [])
        metadata = dict(signature)
        metadata.setdefault("input_dtypes", list(input_dtypes or []))
        metadata.setdefault("output_dtypes", list(output_dtypes or []))
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if input_dtypes is not None or output_dtypes is not None:
        validate_codegen_abi(
            artifact,
            metadata,
            input_dtypes or metadata.get("input_dtypes", []),
            output_dtypes or metadata.get("output_dtypes", []),
        )
    return metadata


def _write_codegen(options):
    script = Path(options.case_dir) / (options.codegen_entry or "input_ascir.py")
    if not script.exists():
        raise RuntimeError("case graph script not found")
    profile = Path(options.profile or os.environ.get("DEVICE_VALIDATION_PROFILE", ""))
    if not profile.is_file():
        raise FileNotFoundError("selected device profile is not a regular file")
    result = execute_command(
        [
            sys.executable,
            str(script),
            "--rows",
            str(options.shape[0]),
            "--cols",
            str(options.shape[1]),
            "--profile",
            str(profile),
            "--output-dir",
            str(options.artifact),
        ],
        120,
        options.artifact,
        ("codegen.stdout", "codegen.stderr"),
        subprocess,
    )
    if result.returncode != 0:
        raise RuntimeError(f"codegen failed with exit code {result.returncode}")
    destination = _jit_module(
        options.artifact, options.shape, profile, options.case_dir
    )
    _ensure_abi_metadata(options.artifact, options.input_dtypes, options.output_dtypes)
    return destination


def _default_performance(case, performance=None):
    declared = case.get("performance", {})
    performance = performance or {
        "declared": {
            "required": declared.get("required", False),
            "profiler": declared.get("profiler", False),
            "metric": "runner_wall_clock",
            "warmup_count": declared.get("warmup_count", 0),
            "kernel_count": declared.get("kernel_count", 1),
            "profiler_requested": declared.get("profiler", False),
            "profiler_tool_available": declared.get("profiler_available", False),
            "profiler_collected": False,
            "timing_source": "runner_wall_clock",
        },
        "actual": {},
        "status": "not_applicable",
    }
    declared_profile = declared.get("profiler", False)
    performance.setdefault("declared", {})
    declared_values = performance["declared"]
    declared_values.setdefault("required", declared.get("required", False))
    declared_values.setdefault("profiler", declared_profile)
    declared_values.setdefault("metric", declared.get("metric", "latency_ms"))
    declared_values.setdefault("warmup_count", declared.get("warmup_count", 0))
    declared_values.setdefault("kernel_count", declared.get("kernel_count", 1))
    declared_values.setdefault(
        "profiler_tool_available", declared.get("profiler_available", False)
    )
    declared_values.setdefault("profiler_collected", False)
    declared_values.setdefault("timing_source", "runner_wall_clock")
    actual = performance.setdefault("actual", {})
    actual.setdefault("profiler_requested", False)
    actual.setdefault("profiler_tool_available", False)
    actual.setdefault("profiler_collected", False)
    actual.setdefault("timing_source", "runner_wall_clock")
    performance["declared"].setdefault("declared_profiler", declared_profile)
    performance["declared"].setdefault("profiler_requested", declared_profile)
    performance["declared"].setdefault(
        "tool_available", declared.get("profiler_available", False)
    )
    performance["declared"].setdefault("collected", False)
    performance["actual"].setdefault(
        "tool_available", actual["profiler_tool_available"]
    )
    performance["actual"].setdefault("collected", actual["profiler_collected"])
    return performance


def _attach_local_abi_metadata(report, artifact):
    abi_path = artifact / "abi_metadata.json"
    if abi_path.exists():
        local_metadata = json.loads(abi_path.read_text(encoding="utf-8"))
        report.setdefault("local_abi_metadata", local_metadata)
        if not report["abi_metadata"]:
            report["abi_metadata"] = local_metadata
    return report


def _finalize_cli_report(report, artifact, run_parameters):
    report["metric"] = (
        report.get("performance", {})
        .get("actual", {})
        .get("metric", "runner_wall_clock")
    )
    report["profile"] = run_parameters.get("profile", "") if run_parameters else ""

    return report


def _default_support_decisions(status, reason):
    return {
        "compile": {
            "result": status if status in ("passed", "failed") else "not_applicable",
            "reason": reason,
        },
        "functional": {
            "result": status if status in ("passed", "failed") else "not_applicable",
            "reason": reason,
        },
        "precision": {
            "result": status if status in ("passed", "failed") else "not_applicable",
            "reason": reason,
        },
        "performance": {"result": "not_applicable", "reason": "profiler optional"},
    }


def _report_document(doc):
    return build_report(
        doc.case_id,
        doc.backend,
        doc.soc,
        doc.status,
        doc.reason,
        stage=doc.stage
        or ("verification" if doc.status == "precision_failed" else "execution"),
        variant=doc.variant,
        step=doc.step,
        actual_abi=doc.actual_abi,
        requested_abi=doc.requested_abi,
        abi_metadata=doc.abi_metadata,
        steps=doc.steps,
        support_decisions=doc.support
        or _default_support_decisions(doc.status, doc.reason),
        precision=doc.precision,
        performance=doc.performance,
        artifact_paths=sorted(str(path.name) for path in doc.artifact.iterdir()),
        run_parameters=doc.run_parameters,
        error_code=doc.error_code
        or ("precision_mismatch" if doc.status == "precision_failed" else ""),
    )


def cli_report(options):
    performance = _default_performance(options.case, options.performance)
    doc = ReportDocument(
        options.case["case_id"],
        options.backend,
        options.soc,
        options.status,
        options.reason,
        options.artifact,
        options.stage,
        options.support,
        options.precision,
        performance,
        options.run_parameters,
        options.error_code,
        options.variant,
        options.step,
        options.actual_abi,
        options.requested_abi,
        options.abi_metadata,
        options.steps,
    )
    report = _report_document(doc)
    report = _finalize_cli_report(report, options.artifact, options.run_parameters)
    return _attach_local_abi_metadata(report, options.artifact)


def _enrich_performance_result(result, precision_passed, run_parameters, actual):
    result["precision_passed"] = bool(precision_passed)
    for key in ("shape", "dtype", "soc", "warmup", "repeat", "reference"):
        if run_parameters and key in run_parameters:
            result[key] = run_parameters[key]
    for key in (
        "timing_source",
        "profiler_requested",
        "profiler_tool_available",
        "profiler_collected",
        "profiler_collect_dir",
        "profiler_output_dir",
        "raw_kernel_activity",
        "profiler_export_status",
        "profiler_export_failed",
        "profiler_export_unavailable",
    ):
        if key in actual:
            result[key] = actual[key]
    return result


def _actual_performance(
    payload,
    *,
    variant,
    step_count=1,
    precision_passed=False,
    run_parameters=None,
    metric="runner_wall_clock",
):
    from device_validation.python.benchmark import summarize_samples

    actual = (payload.get("performance") or {}).get("actual", {})
    samples = actual.get("samples", [])
    if not samples:
        samples = payload.get("samples", [])
    if samples:
        result = summarize_samples(
            samples,
            variant=variant,
            step_count=step_count,
            metric=metric,
            runtime_state={
                key: actual[key]
                for key in ("tiling_time_us", "workspace_size", "block_dimension")
                if key in actual
            },
        )
    else:
        result = {}
        for key in (
            "p50",
            "mean",
            "p90",
            "p99",
            "kernel_count",
            "unit",
            "min_us",
            "mean_us",
            "p50_us",
            "p90_us",
            "p99_us",
            "max_us",
            "min_ms",
            "mean_ms",
            "p50_ms",
            "p90_ms",
            "p99_ms",
            "max_ms",
        ):
            if key in actual:
                result[key] = actual[key]
        result["samples"] = []
        result.setdefault("metric", metric)
        result.setdefault("unit", "us" if metric == "device_kernel_duration" else "ms")
    if "metric" in actual:
        result["metric"] = actual["metric"]
    if "unit" in actual:
        result["unit"] = actual["unit"]
    return _enrich_performance_result(result, precision_passed, run_parameters, actual)


def kernel_names(case, variant, step_index=None):
    # Module-internal orchestration helper kept public for test access.
    config = _variant_config(case, variant)
    declared = _explicit_kernel_names(config)
    if declared is not None:
        return declared
    name = _step_kernel_name(config, variant, step_index)
    return [name] if name else []


def _variant_config(case, variant):
    variants = case.get("variants", {})
    if isinstance(variants, dict):
        return variants.get(variant, {})
    return {}


def _explicit_kernel_names(config):
    names = config.get("kernel_names")
    if (
        isinstance(names, list)
        and names
        and all(isinstance(name, str) and name for name in names)
    ):
        return names
    return None


def _step_kernel_name(config, variant, step_index):
    steps = config.get("steps", ()) if isinstance(config, dict) else ()
    if variant != "fused" and step_index is not None and steps:
        if 0 <= step_index < len(steps):
            step = steps[step_index]
            if isinstance(step, dict):
                return step.get("aclnn") or step.get("graph") or step.get("name")
        return None
    if isinstance(config, dict):
        return config.get("graph")
    return None


def _last_json_payload(stdout):
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _export_and_measure_device_samples(collect_dir, run_context):
    from device_validation.python.kernel_activity import (
        extract_measured_records,
        filter_kernel_records,
        parse_profiler_output,
    )
    from device_validation.python.profiler import ExportOptions, export_profiling_data

    export_dir = run_context.artifact_dir / "profiler_export"
    summary_dir = export_profiling_data(
        str(collect_dir),
        str(export_dir),
        timeout=run_context.timeout,
        options=ExportOptions(
            profile=run_context.request.profile,
            env=os.environ,
        ),
    )
    records = parse_profiler_output(summary_dir)
    filtered = filter_kernel_records(records, run_context.kernel_names)
    measured = extract_measured_records(
        filtered, run_context.request.repeat, run_context.request.warmup
    )
    kernel_activity_path = run_context.artifact_dir / "kernel_activity.json"
    kernel_activity_path.write_text(
        json.dumps(
            [
                {
                    "kernel_name": record.kernel_name,
                    "duration_us": record.duration_us,
                    "start_us": record.start_us,
                    "end_us": record.end_us,
                }
                for record in filtered
            ]
        ),
        encoding="utf-8",
    )
    return measured, summary_dir, kernel_activity_path


def _record_exported_payload(
    payload, collect_dir, summary_dir, kernel_activity_path, measured
):
    actual = payload.setdefault("performance", {}).setdefault("actual", {})
    actual["samples"] = measured
    actual["metric"] = "device_kernel_duration"
    actual["unit"] = "us"
    actual["timing_source"] = "msprof"
    actual["profiler_collect_dir"] = str(collect_dir)
    actual["profiler_output_dir"] = str(summary_dir)
    actual["raw_kernel_activity"] = str(kernel_activity_path)


def _record_unavailable_payload(payload, collect_dir, export_status):
    # Host launch samples are not device timings.  Do not retain them when
    # the profiler export cannot provide the requested device samples.
    actual = payload.setdefault("performance", {}).setdefault("actual", {})
    actual["samples"] = []
    actual["sample_count"] = 0
    actual["metric"] = "runner_wall_clock"
    actual["unit"] = "ms"
    actual["timing_source"] = "profiler_export_unavailable"
    actual["profiler_collected"] = False
    actual["profiler_export_status"] = export_status
    actual["profiler_export_failed"] = True
    actual["profiler_export_unavailable"] = True
    actual.setdefault("profiler_collect_dir", str(collect_dir))


def _apply_msprof_export(payload, collect_dir, run_context):
    export_status = "exported"
    try:
        measured, summary_dir, kernel_activity_path = (
            _export_and_measure_device_samples(collect_dir, run_context)
        )
    except (ProfilerExportError, KernelActivityError, ValueError) as error:
        export_status = f"profiler_export_failed: {error}"
    if export_status == "exported":
        _record_exported_payload(
            payload, collect_dir, summary_dir, kernel_activity_path, measured
        )
    else:
        _record_unavailable_payload(payload, collect_dir, export_status)


def _run_profiled(run_context):
    from device_validation.python.profiler import (
        run_application_with_inprocess_profile,
    )

    root = Path(run_context.artifact_dir)
    request_path = root / "executor_request.json"
    request_path.write_text(
        json.dumps(request_payload(run_context.request)), encoding="utf-8"
    )
    collect_dir = root / "profiler_collect"
    app = run_application_with_inprocess_profile(
        list(run_context.runner_command),
        ["--request", str(request_path)],
        str(collect_dir),
        timeout=run_context.timeout,
    )
    if not app.available:
        raise ProfilerUnavailableError(app.reason or "profiler_unavailable")
    payload = _last_json_payload(app.stdout)
    if app.returncode != 0:
        payload = payload or _failure_payload(
            "backend_returncode",
            app.stderr.strip() or "backend returned non-zero",
            app.returncode,
        )
    elif payload is None:
        payload = _failure_payload(
            "empty_stdout",
            app.stderr.strip() or "executor emitted no structured payload",
            app.returncode,
        )
    if payload.get("stage_status") == "passed":
        if run_context.msprof_path is None:
            actual = payload.setdefault("performance", {}).setdefault("actual", {})
            actual["metric"] = "runner_wall_clock"
            actual["unit"] = "ms"
            actual["timing_source"] = "runner_wall_clock"
            actual["profiler_export_status"] = "profiler_unavailable"
        else:
            _apply_msprof_export(payload, collect_dir, run_context)
    return ExecutorResult(
        payload, app.stdout, app.stderr, app.returncode, str(request_path)
    )


def _step_spec(spec, selected_shape, path):
    item = dict(spec)
    # Step codegen is generated for the selected request shape, not the case
    # schema's placeholder shape ([1]).
    item["shape"] = list(selected_shape)
    item["file"] = path
    if item.get("dtype") in DTYPE_SIZES:
        checked_shape_size(item.get("shape", ()), DTYPE_SIZES[item["dtype"]])
    return item


def _step_specs(ctx, step, index):
    step_artifact = ctx.artifact / f"step_{index}_{step.get('name', index)}"
    step_artifact.mkdir(parents=True, exist_ok=True)
    input_paths = []
    input_specs = []
    for spec in step.get("inputs", []):
        if spec == "$previous":
            if ctx.previous is None:
                raise ValueError("first unfused step cannot use $previous")
            if not Path(ctx.previous).is_file():
                raise ValueError("previous output is unavailable")
            input_paths.append(ctx.previous)
            input_specs.append(_step_spec(ctx.previous_spec, ctx.shape, ctx.previous))
        else:
            item = spec if isinstance(spec, dict) else {"file": spec}
            input_paths.append(str(ctx.artifact / item["file"]))
            input_specs.append(_step_spec(item, ctx.shape, input_paths[-1]))
    output_paths = [
        str(step_artifact / item["file"]) for item in step.get("outputs", [])
    ]
    output_specs = []
    for item, path in zip(step.get("outputs", []), output_paths):
        output_specs.append(_step_spec(item, ctx.shape, path))

    return {
        "input_paths": input_paths,
        "input_specs": input_specs,
        "output_paths": output_paths,
        "output_specs": output_specs,
        "step_artifact": step_artifact,
        "previous": ctx.previous,
        "previous_spec": ctx.previous_spec,
    }


def _step_metadata(specs):
    return read_abi_metadata(
        specs["step_artifact"],
        len(specs["input_paths"]),
        len(specs["output_paths"]),
        [spec["dtype"] for spec in specs["input_specs"]],
        [spec["dtype"] for spec in specs["output_specs"]],
    )


def build_step_request(ctx, step, index):
    # Module-internal orchestration helper kept public for test access.
    specs = _step_specs(ctx, step, index)
    aclnn_op = step.get("aclnn") if isinstance(step, dict) else ""
    if aclnn_op:
        return _build_step_bundle(ctx, index, specs, "", aclnn_op)
    module = _write_codegen(
        CodegenOptions(
            ctx.case_dir,
            specs["step_artifact"],
            ctx.shape,
            "unfused",
            step.get("script"),
            ctx.args.profile,
            [spec["dtype"] for spec in specs["input_specs"]],
            [spec["dtype"] for spec in specs["output_specs"]],
        )
    )
    return _build_step_bundle(ctx, index, specs, module)


def _build_step_bundle(ctx, index, specs, module, aclnn_op=""):
    metadata = {} if aclnn_op else _step_metadata(specs)
    launch_abi = metadata.get("launch_abi", "")
    request = ExecutorRequest(
        module=str(module),
        abi=launch_abi,
        device=ctx.args.device,
        tensor_files=tuple(specs["input_paths"]),
        tensor_specs=tuple(specs["input_specs"]),
        warmup=ctx.args.warmup,
        repeat=ctx.args.repeat,
        variant="unfused",
        artifact_dir=str(specs["step_artifact"]),
        case_dir=str(ctx.case_dir),
        profile=str(Path(ctx.args.profile)),
        shape=ctx.shape,
        selected_shape=ctx.shape,
        soc_profile=ctx.run_parameters.get("soc", ""),
        profiler=ctx.args.profiler or ctx.metric == "device_kernel_duration",
        output_specs=tuple(specs["output_specs"]),
        inputs=tuple(specs["input_paths"]),
        outputs=tuple(specs["output_paths"]),
        launch_abi=launch_abi,
        input_count=len(specs["input_paths"]),
        output_count=len(specs["output_paths"]),
        abi_metadata=metadata,
        case_id=ctx.case["case_id"],
        step=index,
        contract_schema={
            "schema_version": 1,
            "case_id": ctx.case["case_id"],
            "variant": "unfused",
            "step": index,
            "inputs": specs["input_specs"],
            "outputs": specs["output_specs"],
        },
        aclnn_op=aclnn_op,
    )
    return {
        "index": index,
        "request": request,
        "step_artifact": specs["step_artifact"],
        "previous": specs["previous"],
        "previous_spec": specs["previous_spec"],
        "output_paths": specs["output_paths"],
        "output_specs": specs["output_specs"],
        "metadata": metadata,
        "input_specs": specs["input_specs"],
    }


def _record_step_result(ctx, request, result, step_request, samples):
    from device_validation.python.benchmark import save_samples

    validate_step_outputs(request, result.payload)
    step_artifact = step_request["step_artifact"]
    metadata = step_request["metadata"]
    launch_abi = metadata.get("launch_abi", "")
    index = step_request["index"]
    save_samples(step_artifact, samples, variant="unfused", step=index)
    ctx.step_samples.append(samples)
    ctx.previous = step_request["output_paths"][-1]
    ctx.previous_spec = step_request["output_specs"][-1]
    ctx.final_outputs = result.payload.get("outputs", [])
    ctx.step_actual = (result.payload.get("performance") or {}).get("actual", {})
    ctx.collected_step_reports.append(
        {
            "step": index,
            "actual_abi": result.payload.get("actual_abi", launch_abi),
            "requested_abi": result.payload.get("requested_abi", launch_abi),
            "input_specs": list(step_request["input_specs"]),
            "output_specs": list(step_request["output_specs"]),
            "samples": list(samples),
            "sample_count": len(samples),
            "kernel_count": ctx.step_actual.get("kernel_count", 1),
            "metric": ctx.step_actual.get("metric", "runner_wall_clock"),
            "timing_source": ctx.step_actual.get("timing_source", "runner_wall_clock"),
            "artifact_path": str(step_artifact),
            "report_path": str(step_artifact / "report.json"),
        }
    )
    (step_artifact / "report.json").write_text(
        json.dumps(result.payload, indent=2), encoding="utf-8"
    )

    return samples


def _run_step(ctx, step, index):
    step_req = build_step_request(ctx, step, index)
    request = step_req["request"]
    step_artifact = step_req["step_artifact"]
    if ctx.metric == "device_kernel_duration":
        result = _run_profiled(
            ProfiledRun(
                ctx.runner_command,
                request,
                step_artifact,
                ctx.msprof_path,
                kernel_names(ctx.case, ctx.args.variant, index),
            )
        )
    else:
        result = run_subprocess(
            [*ctx.runner_command, "--request"], request, 600, step_artifact
        )
    if result.returncode != 0 or result.payload.get("stage_status") != "passed":
        raise RuntimeError(result.payload.get("reason", "unfused step failed"))
    actual = (result.payload.get("performance") or {}).get("actual", {})
    samples = actual.get("samples", result.payload.get("samples", []))
    if not samples:
        raise ValueError("runner did not return benchmark samples")
    if len(samples) != ctx.args.repeat:
        raise ValueError("sample count does not match repeat")
    return _record_step_result(ctx, request, result, step_req, samples)


def _carried_activity(step_actual):
    carried = {}
    for key in (
        "timing_source",
        "profiler_collected",
        "profiler_collect_dir",
        "profiler_output_dir",
        "raw_kernel_activity",
    ):
        if key in step_actual:
            carried[key] = step_actual[key]

    return carried


def _unfused_summary(ctx, steps, goldens):
    from device_validation.python.benchmark import save_samples, sum_step_samples

    final_actual = [decode_output_payload(item) for item in ctx.final_outputs]
    precision = verify_outputs(
        final_actual,
        goldens,
        ctx.case["verification"]["atol"],
        ctx.case["verification"]["rtol"],
    )
    samples = sum_step_samples(ctx.step_samples)
    save_samples(ctx.artifact, samples, variant="unfused")
    carried = _carried_activity(ctx.step_actual)
    payload = {
        "performance": {"actual": {"samples": samples, **carried}},
        "outputs": ctx.final_outputs,
    }
    performance = {
        "actual": _actual_performance(
            payload,
            variant="unfused",
            step_count=len(steps),
            precision_passed=precision["passed"],
            run_parameters=ctx.run_parameters,
            metric=ctx.metric,
        ),
        "declared": {},
    }
    if ctx.metric == "device_kernel_duration":
        performance["actual"]["steps"] = _steps_decomposition(
            ctx.step_samples, steps, ctx.metric
        )
    return precision, performance


def unfused_steps(case_dir, args):
    # Module-internal orchestration helper kept public for test access.
    metric = getattr(args, "metric", "runner_wall_clock")
    steps = load_typed_case(case_dir).steps(args.variant)
    if not steps:
        raise ValueError("unfused variant must declare steps")
    runner = os.environ.get("DEVICE_VALIDATION_RUNNER")
    jit = os.environ.get("AUTOFUSE_DEVICE_JIT")
    if not runner:
        raise FileNotFoundError("DEVICE_VALIDATION_RUNNER is required")
    all_aclnn = all(
        isinstance(step, dict) and bool(step.get("aclnn")) for step in steps
    )
    if not all_aclnn and not jit:
        raise FileNotFoundError(
            "DEVICE_VALIDATION_RUNNER and AUTOFUSE_DEVICE_JIT are required"
        )
    runner_command = shlex.split(runner)
    return {
        "runner_command": runner_command,
        "steps": steps,
        "metric": metric,
    }


def _run_unfused(ctx, goldens, step_reports=None):
    unfused = unfused_steps(ctx.case_dir, ctx.args)
    steps = unfused["steps"]
    metric = unfused["metric"]
    ctx.runner_command = unfused["runner_command"]
    ctx.metric = metric
    for index, step in enumerate(steps):
        _run_step(ctx, step, index)
        if not isinstance(ctx.final_outputs, list):
            ctx.final_outputs = []
    if step_reports is not None:
        step_reports.extend(ctx.collected_step_reports)
    precision, performance = _unfused_summary(ctx, steps, goldens)
    return precision, performance


def _steps_decomposition(step_samples, steps, metric):
    from device_validation.python.benchmark import summarize_samples

    rows = []
    for index, samples in enumerate(step_samples):
        summary = summarize_samples(
            samples, variant="fused", step_count=1, metric=metric
        )
        step = steps[index] if index < len(steps) else {}
        rows.append(
            {
                "step": index,
                "name": step.get("name", index) if isinstance(step, dict) else index,
                "kernel_count": 1,
                "sample_count": summary["sample_count"],
                "p50": summary["p50"],
                "mean": summary["mean"],
                "p90": summary["p90"],
                "p99": summary["p99"],
                "unit": summary["unit"],
                "metric": metric,
            }
        )
    return rows


def _write_run_manifest(artifact, case_dir, shape, args, run_parameters):
    manifest = {
        "case": str(case_dir),
        "shape": list(shape),
        "selected_shape": list(shape),
        "mode": args.mode,
        "backend": args.backend,
        "soc_profile": args.soc_profile,
        "device": args.device,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "profile": str(Path(args.profile).resolve()),
        "variant": args.variant,
        "run_parameters": run_parameters,
    }
    (artifact / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _build_run_parameters(args, shape, profile):
    metric = getattr(args, "metric", "runner_wall_clock")
    run_parameters = {
        "selected_shape": list(shape),
        "profile": profile,
        "device": args.device,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "mode": args.mode,
        "profiler_requested": args.profiler,
        "variant": args.variant,
        "metric": metric,
    }
    return run_parameters


def _prepare_case(args):
    case_dir = Path(args.case).resolve()
    case = load_case(case_dir)
    artifact = create_artifact_dir(args.output_dir, case.get("case_id", "invalid-case"))
    args.last_report_path = artifact / "report.json"
    if args.device < 0:
        raise ValueError("invalid device_id: must be non-negative")
    if (
        case.get("schema_version") != 1
        or not case.get("inputs")
        or not case.get("outputs")
    ):
        raise ValueError("invalid case schema or tensor count")
    return case_dir, case, artifact


def _write_not_applicable_report(case, args, shape, reason, artifact):
    report = cli_report(
        CliReportOptions(
            case,
            args.backend,
            args.soc_profile,
            "not_applicable",
            reason,
            artifact,
            variant=args.variant,
            support={
                name: {"result": "not_applicable", "reason": reason}
                for name in ("compile", "functional", "precision", "performance")
            },
            run_parameters={"selected_shape": list(shape), "mode": args.mode},
            stage="preflight",
        )
    )
    (artifact / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return 0


def _resolve_support_decision(args, shape, case, artifact, profile):
    support_decision = resolve_support(
        load_typed_case(Path(args.case).resolve()),
        args.backend,
        args.soc_profile,
        shape,
        SupportConfig(args.variant, profile),
    )
    if support_decision.status != "supported" or support_decision.entry is None:
        return _write_not_applicable_report(
            case, args, shape, support_decision.reason, artifact
        )
    return support_decision


def _resolve_msprof(profile, args):
    metric = getattr(args, "metric", "runner_wall_clock")
    if metric != "device_kernel_duration":
        return metric, None
    from device_validation.python.profiler import resolve_msprof

    return metric, resolve_msprof(profile, os.environ)


def _prepare_run(args, shape):
    case_dir, case, artifact = _prepare_case(args)
    from device_validation.python.support_matrix import resolve_case_profile

    args.soc_profile, resolved_profile, profile = resolve_case_profile(
        case_dir, args.backend, args.soc_profile, args.profile
    )
    args.profile = str(resolved_profile.resolve())
    support_decision = _resolve_support_decision(args, shape, case, artifact, profile)
    if support_decision == 0:
        return 0
    support = support_decision.entry
    if list(shape) not in support.get("shapes", []):
        return _write_not_applicable_report(
            case, args, shape, f"unsupported shape profile: {list(shape)}", artifact
        )
    validate_case(case, shape, args.backend, args.soc_profile, variant=args.variant)
    metric, msprof_path = _resolve_msprof(profile, args)
    run_profile = args.profile
    run_parameters = _build_run_parameters(args, shape, run_profile)
    _write_run_manifest(artifact, case_dir, shape, args, run_parameters)
    return {
        "case_dir": case_dir,
        "case": case,
        "artifact": artifact,
        "profile": run_profile,
        "metric": metric,
        "msprof_path": msprof_path,
        "run_parameters": run_parameters,
        "support": support,
    }


def _input_specs(inputs, shape):
    return tuple(
        {**spec, "shape": list(shape)} if spec.get("dynamic") else spec
        for spec in inputs
    )


def _output_specs(outputs, shape):
    return tuple(
        {**spec, "shape": list(shape)} if spec.get("dynamic") else spec
        for spec in outputs
    )


def _fused_contract_schema(ctx, request):
    return {
        "schema_version": 1,
        "case_id": ctx.case["case_id"],
        "variant": ctx.args.variant,
        "step": -1,
        "inputs": list(request.tensor_specs),
        "outputs": list(request.output_specs),
    }


def _build_fused_request(ctx, module):
    input_dtypes = [item["dtype"] for item in ctx.case["inputs"]]
    output_dtypes = [item["dtype"] for item in ctx.case["outputs"]]
    metadata = read_abi_metadata(
        ctx.artifact,
        len(input_dtypes),
        len(output_dtypes),
        input_dtypes=input_dtypes,
        output_dtypes=output_dtypes,
    )
    request = _fused_executor_request(ctx, module, metadata)
    return dataclasses.replace(
        request, contract_schema=_fused_contract_schema(ctx, request)
    )


def _fused_executor_request(ctx, module, metadata):
    perf_config = ctx.case.get("performance", {})
    performance_required = ctx.support.get(
        "performance"
    ) == "required" and perf_config.get("profiler", False)
    return ExecutorRequest(
        module=str(module),
        abi=metadata["launch_abi"],
        device=ctx.args.device,
        tensor_files=tuple(
            str(ctx.artifact / f"input_{i}.bin") for i in range(len(ctx.case["inputs"]))
        ),
        tensor_specs=_input_specs(ctx.case["inputs"], ctx.shape),
        warmup=ctx.args.warmup,
        repeat=ctx.args.repeat,
        variant=ctx.args.variant,
        artifact_dir=str(ctx.artifact),
        case_dir=str(ctx.case_dir),
        profile=str(Path(ctx.args.profile)),
        shape=ctx.shape,
        selected_shape=ctx.shape,
        soc_profile=ctx.args.soc_profile,
        profiler=ctx.args.profiler
        or (ctx.args.mode == "performance" and performance_required)
        or ctx.metric == "device_kernel_duration",
        output_specs=_output_specs(ctx.case["outputs"], ctx.shape),
        inputs=tuple(
            str(ctx.artifact / f"input_{i}.bin") for i in range(len(ctx.case["inputs"]))
        ),
        outputs=tuple(
            str(ctx.artifact / f"output_raw_{i}.bin")
            for i in range(len(ctx.case["outputs"]))
        ),
        launch_abi=metadata["launch_abi"],
        input_count=metadata["input_count"],
        output_count=metadata["output_count"],
        abi_metadata=metadata,
        case_id=ctx.case["case_id"],
        step=-1,
    )


def _invoke_fused(ctx, module):
    runner = os.environ.get("DEVICE_VALIDATION_RUNNER")
    if not runner:
        raise FileNotFoundError("DEVICE_VALIDATION_RUNNER is not set")
    runner_command = shlex.split(runner)
    if (
        not runner_command
        or not Path(
            runner_command[-1]
            if runner_command[0] == sys.executable
            else runner_command[0]
        ).is_file()
    ):
        raise FileNotFoundError("DEVICE_VALIDATION_RUNNER is not a regular file")
    request = _build_fused_request(ctx, module)
    if ctx.metric == "device_kernel_duration":
        result = _run_profiled(
            ProfiledRun(
                runner_command,
                request,
                ctx.artifact,
                ctx.msprof_path,
                kernel_names(ctx.case, ctx.args.variant),
            )
        )
    else:
        result = run_subprocess(
            [*runner_command, "--request"], request, 600, ctx.artifact
        )
    payload = result.payload
    return payload, result.returncode


def _fused_performance(ctx, payload, precision):
    performance = payload.get("performance") or {}
    performance["actual"] = _actual_performance(
        payload,
        variant=ctx.args.variant,
        precision_passed=precision["passed"],
        run_parameters={
            **ctx.run_parameters,
            "shape": list(ctx.shape),
            "dtype": ctx.case["inputs"][0]["dtype"],
            "soc": ctx.args.soc_profile,
            "warmup": ctx.args.warmup,
            "repeat": ctx.args.repeat,
            "reference": str(ctx.case_dir / "reference.py"),
        },
        metric=ctx.metric,
    )


def _fail_fused_report(ctx, payload, status, exit_code, *, reason="", error_code=""):
    report = cli_report(
        CliReportOptions(
            ctx.case,
            ctx.args.backend,
            ctx.args.soc_profile,
            status,
            payload.get("reason", reason),
            ctx.artifact,
            variant=ctx.args.variant,
            precision=payload.get("precision"),
            performance=payload.get("performance"),
            support=payload.get("support_decisions"),
            run_parameters=ctx.run_parameters,
            stage=payload.get("stage", "preflight"),
            error_code=payload.get("error_code", error_code),
            requested_abi=payload.get("requested_abi", ""),
            actual_abi=payload.get("actual_abi", ""),
            abi_metadata=payload.get("abi_metadata"),
        )
    )
    (ctx.artifact / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return exit_code, None


def _fused_precision(ctx, payload):
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(ctx.case["outputs"]):
        raise ValueError("backend output count mismatch")
    actual_outputs = []
    for index, output in enumerate(outputs):
        actual = decode_output_payload(output)
        expected_spec = ctx.case["outputs"][index]
        expected_shape = (
            tuple(ctx.shape)
            if expected_spec.get("dynamic", False)
            else tuple(expected_spec["shape"])
        )
        if output.get("dtype") != expected_spec["dtype"] or tuple(
            actual.shape
        ) != tuple(expected_shape):
            raise RuntimeError(f"output {index} dtype or shape mismatch")
        actual_outputs.append(actual)
    return verify_outputs(
        actual_outputs,
        ctx.goldens,
        ctx.case["verification"]["atol"],
        ctx.case["verification"]["rtol"],
    )


def _finalize_fused_report(ctx, payload, returncode, precision):
    payload_status, payload_exit = classify_backend_payload(payload, returncode)
    if payload_exit != 0:
        report = cli_report(
            CliReportOptions(
                ctx.case,
                ctx.args.backend,
                ctx.args.soc_profile,
                payload_status,
                payload.get("reason", "backend failed"),
                ctx.artifact,
                variant=ctx.args.variant,
                precision=payload.get("precision"),
                performance=payload.get("performance"),
                support=payload.get("support_decisions"),
                run_parameters=ctx.run_parameters,
                stage=payload.get("stage", "execution"),
                error_code=payload.get("error_code", "backend_failure"),
                requested_abi=payload.get("requested_abi", ""),
                actual_abi=payload.get("actual_abi", ""),
                abi_metadata=payload.get("abi_metadata"),
            )
        )
        (ctx.artifact / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return payload_exit, None
    report = cli_report(
        CliReportOptions(
            ctx.case,
            ctx.args.backend,
            ctx.args.soc_profile,
            payload.get("stage_status", "failed"),
            payload.get("reason", ""),
            ctx.artifact,
            variant=ctx.args.variant,
            precision=precision,
            performance=payload.get("performance"),
            support=payload.get("support_decisions"),
            run_parameters=ctx.run_parameters,
            stage=payload.get("stage", "execution"),
            error_code=payload.get("error_code", ""),
            requested_abi=payload.get("requested_abi", ""),
            actual_abi=payload.get("actual_abi", ""),
            abi_metadata=payload.get("abi_metadata"),
        )
    )
    return report


def _fused_precision_report(ctx, payload, precision):
    report = cli_report(
        CliReportOptions(
            ctx.case,
            ctx.args.backend,
            ctx.args.soc_profile,
            "precision_failed",
            "precision verification failed",
            ctx.artifact,
            variant=ctx.args.variant,
            precision=precision,
            performance=payload.get("performance"),
            support=payload.get("support_decisions"),
            run_parameters=ctx.run_parameters,
            stage="verification",
            requested_abi=payload.get("requested_abi", ""),
            actual_abi=payload.get("actual_abi", ""),
            abi_metadata=payload.get("abi_metadata"),
        )
    )
    report["precision"] = report_precision(precision, ctx.artifact)
    (ctx.artifact / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return EXIT_CODES["precision_failed"], None


def _handle_fused_payload(ctx, payload, returncode):
    preflight_status, preflight_exit = classify_backend_payload(payload, returncode)
    if preflight_status in (
        "required_capability_failed",
        "skipped",
        "not_applicable",
    ):
        return _fail_fused_report(
            ctx,
            payload,
            preflight_status,
            preflight_exit,
            reason="capability is not applicable",
        )
    if returncode != 0:
        return _fail_fused_report(
            ctx,
            payload,
            preflight_status,
            EXIT_CODES[preflight_status],
            reason="backend failed",
            error_code="backend_failure",
        )
    precision = _fused_precision(ctx, payload)
    _fused_performance(ctx, payload, precision)
    if not precision["passed"]:
        return _fused_precision_report(ctx, payload, precision)
    report = _finalize_fused_report(ctx, payload, returncode, precision)
    return None, report


def _run_fused(ctx):
    module = _write_codegen(
        CodegenOptions(
            ctx.case_dir,
            ctx.artifact,
            ctx.shape,
            ctx.args.variant,
            ctx.support.get("codegen_entry"),
            ctx.args.profile,
            [item["dtype"] for item in ctx.case["inputs"]],
            [item["dtype"] for item in ctx.case["outputs"]],
        )
    )
    payload, returncode = _invoke_fused(ctx, module)

    exit_code, report = _handle_fused_payload(ctx, payload, returncode)
    return exit_code, report


def _run_unfused_mode(ctx, goldens, run_parameters):
    step_reports = []
    ctx.run_parameters = _unfused_run_parameters(ctx, run_parameters)
    precision, performance = _run_unfused(ctx, goldens, step_reports)
    report = cli_report(
        CliReportOptions(
            ctx.case,
            ctx.args.backend,
            ctx.args.soc_profile,
            "passed" if precision["passed"] else "precision_failed",
            "" if precision["passed"] else "precision verification failed",
            ctx.artifact,
            variant=ctx.args.variant,
            precision=precision,
            performance=performance,
            run_parameters=run_parameters,
        )
    )
    report["steps"] = step_reports
    report["precision"] = report_precision(precision, ctx.artifact)
    return report


def _unfused_run_parameters(ctx, run_parameters):
    return {
        **run_parameters,
        "shape": list(ctx.shape),
        "dtype": ctx.case["inputs"][0]["dtype"],
        "soc": ctx.args.soc_profile,
        "warmup": ctx.args.warmup,
        "repeat": ctx.args.repeat,
        "reference": str(Path(ctx.case_dir) / "reference.py"),
    }


def _run_error_report(error, case, args, artifact, run_parameters):
    status, _ = classify_failure(str(error), "required")
    if isinstance(error, KernelActivityError):
        error_code = error.error_code
    elif isinstance(error, ProfilerExportError):
        error_code = error.error_code
    elif isinstance(error, ProfilerUnavailableError):
        error_code = "profiler_unavailable"
    elif "AutofuseLaunch" in str(error):
        (artifact / "abi_metadata.json").write_text(
            json.dumps(
                {
                    "launch_abi": "AutofuseLaunch",
                    "status": "failed",
                    "reason": str(error),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        error_code = ""
    else:
        error_code = ""
    return cli_report(
        CliReportOptions(
            case,
            args.backend,
            args.soc_profile,
            status,
            str(error),
            artifact,
            variant=args.variant,
            run_parameters=run_parameters,
            error_code=error_code,
        )
    )


def _write_case_inputs(case_dir, artifact, shape):
    _, goldens = _write_inputs(case_dir, artifact, shape)
    if not isinstance(goldens, (list, tuple)):
        goldens = [goldens]
    return goldens


def _prepare_only_report(case, args, artifact, run_parameters):
    return cli_report(
        CliReportOptions(
            case,
            args.backend,
            args.soc_profile,
            "not_applicable",
            "prepare_only",
            artifact,
            variant=args.variant,
            run_parameters=run_parameters,
        )
    )


def _run_variant(args, shape, prepared):
    try:
        return run_prepared_variant(args, shape, prepared)
    except (
        RuntimeError,
        ValueError,
        OSError,
    ) as error:
        return _run_error_report(
            error,
            prepared["case"],
            args,
            prepared["artifact"],
            prepared["run_parameters"],
        )


def run_prepared_variant(args, shape, prepared):
    # Module-internal orchestration helper kept public for test access.
    run_parameters = prepared["run_parameters"]
    goldens = _write_case_inputs(prepared["case_dir"], prepared["artifact"], shape)
    if args.mode == "prepare":
        return _prepare_only_report(
            prepared["case"], args, prepared["artifact"], run_parameters
        )
    variants = prepared["case"].get("variants", {})
    declared_steps = (
        variants.get(args.variant, {}).get("steps")
        if isinstance(variants, dict) and isinstance(args.variant, str)
        else None
    )
    if declared_steps:
        context = UnfusedContext(
            prepared["case"],
            Path(prepared["case_dir"]),
            tuple(shape),
            args,
            dict(run_parameters),
            Path(prepared["artifact"]),
            [],
            prepared["msprof_path"],
            prepared["metric"],
        )
        return _run_unfused_mode(context, goldens, run_parameters)
    fused_ctx = FusedContext(
        prepared["case"],
        Path(prepared["case_dir"]),
        tuple(shape),
        args,
        prepared["support"],
        Path(prepared["artifact"]),
        run_parameters,
        goldens,
        prepared["metric"],
        prepared["msprof_path"],
    )
    exit_code, report = _run_fused(fused_ctx)
    if exit_code is not None:
        return int(exit_code)
    return report


def _run(args, shape):
    prepared = _prepare_run(args, shape)
    if isinstance(prepared, int):
        return prepared
    report = _run_variant(args, shape, prepared)
    if isinstance(report, int):
        return report
    (prepared["artifact"] / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return EXIT_CODES.get(report["stage_status"], 4)


def _matrix_cell(args, matrix_case, request, artifact):
    cell_args = argparse.Namespace(**vars(args))
    cell_args.case = request["case_dir"]
    cell_args.variant = request["variant"]
    cell_args.shape = list(request["shape"])
    cell_args.output_dir = str(artifact)
    exit_code = _run(cell_args, tuple(request["shape"]))
    report_path = getattr(cell_args, "last_report_path", artifact / "report.json")
    if not Path(report_path).is_file():
        return {"returncode": exit_code, "report_path": str(report_path)}
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    payload.setdefault("schema_version", REPORT_SCHEMA_VERSION)
    payload.setdefault("case", matrix_case.case_id)
    payload.setdefault("backend", args.backend)
    payload.setdefault("soc_profile", args.soc_profile)
    payload.setdefault("variant", request["variant"])
    payload.setdefault("stage_status", "failed" if exit_code else "passed")
    payload.setdefault("stage", "execution")
    payload.setdefault("reason", "")
    payload.setdefault("error_code", "")
    payload.setdefault("precision", {"passed": payload["stage_status"] == "passed"})
    payload.setdefault("performance", {})
    payload.setdefault("run_parameters", {"shape": list(request["shape"])})
    payload.setdefault("artifact_paths", [])
    payload.setdefault("support_decisions", {})
    for name in ("compile", "functional", "precision", "performance"):
        payload["support_decisions"][name] = {
            "result": "passed",
            "reason": "validated by matrix preflight",
        }
    payload["returncode"] = exit_code
    payload["report_path"] = str(report_path)
    payload["artifact_root"] = str(Path(report_path).parent)
    return payload


def _matrix_case(args, case, typed_case, shapes):
    matrix_case = (
        typed_case if hasattr(typed_case, "support_matrix") else _typed_from_dict(case)
    )
    if getattr(matrix_case, "support_matrix", ()):
        return matrix_case
    return _typed_from_dict(
        {
            **case,
            "support_matrix": [
                {
                    "backend": args.backend,
                    "soc": args.soc_profile,
                    "shapes": [list(shape) for shape in shapes],
                    "input_dtypes": [
                        item.get("dtype") for item in case.get("inputs", ())
                    ],
                    "output_dtypes": [
                        item.get("dtype") for item in case.get("outputs", ())
                    ],
                    "compile": "required",
                    "functional": "required",
                    "precision": "required",
                    "performance": "required",
                }
            ],
        }
    )


def _matrix_shapes(args, typed_case, case):
    shapes = (
        [tuple(args.shape)]
        if args.shape
        else [tuple(shape) for shape in typed_case.shapes]
    )
    for shape in shapes:
        _validate_case_shapes(case, shape)
    return [checked_shape_size(shape)[0] for shape in shapes]


def _require_all_variants(typed_case, args):
    if args.variant != "all":
        return False
    if not {"fused", "unfused"}.issubset(set(typed_case.variants)):
        raise ValueError("all requires fused and unfused variants")
    return True


def _execute_matrix_run(args, matrix_case, shapes, root):
    from device_validation.python.matrix import MatrixSpec, execute_matrix

    def execute_cli_cell(request, artifact):
        return _matrix_cell(args, matrix_case, request, artifact)

    return execute_matrix(
        matrix_case,
        args.backend,
        MatrixSpec(
            socs=(args.soc_profile,),
            shapes=tuple(shapes),
            variants=("fused", "unfused"),
        ),
        root,
        profile=args.profile,
        device=args.device,
        mode=args.mode,
        warmup=args.warmup,
        repeat=args.repeat,
        profiler=args.profiler,
        executor=execute_cli_cell,
        comparison_root=root,
        metric=args.metric,
    )


def _matrix_aggregate_report(result):
    aggregate = dict(result)
    aggregate["stage"] = "comparison"
    if result["exit_code"] != 0:
        aggregate["stage_status"] = "failed"
    elif result["comparisons"] and all(
        item["stage_status"] == "not_applicable" for item in result["comparisons"]
    ):
        aggregate["stage_status"] = "not_applicable"
    else:
        aggregate["stage_status"] = "passed"
    aggregate["shapes"] = [
        {
            "shape": item["shape"],
            "stage_status": item["stage_status"],
            "comparison": item["performance_comparison"],
            "variants": item.get("variants", {}),
        }
        for item in result["comparisons"]
    ]
    return aggregate


def _run_matrix_mode(args, case, typed_case):
    if not _require_all_variants(typed_case, args):
        if args.variant not in typed_case.variants:
            raise ValueError(f"unsupported variant: {args.variant}")
        return None
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    shapes = _matrix_shapes(args, typed_case, case)
    matrix_case = _matrix_case(args, case, typed_case, shapes)
    result = _execute_matrix_run(args, matrix_case, shapes, root)
    (root / "report.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    aggregate = _matrix_aggregate_report(result)
    (root / "report.json").write_text(
        json.dumps(aggregate, indent=2, default=str), encoding="utf-8"
    )
    return result["exit_code"]


def _resolve_case_selection(args):
    from device_validation.python.support_matrix import resolve_case_profile

    soc, profile_path, _ = resolve_case_profile(
        Path(args.case).resolve(), args.backend, args.soc_profile, args.profile
    )
    args.soc_profile = soc
    args.profile = str(profile_path.resolve())


def _resolve_cli_shapes(args):
    if args.shape:
        return [tuple(args.shape)]
    typed_case = load_typed_case(Path(args.case).resolve())
    shapes = []
    for entry in typed_case.support_matrix:
        if entry.get("backend") != args.backend or entry.get("soc") != args.soc_profile:
            continue
        for shape in entry.get("shapes", ()):
            shapes.append(tuple(shape))
    if not shapes:
        shapes = list(typed_case.shapes)
    if not shapes:
        raise ValueError("support preflight failed: case declares no shapes")
    return shapes


def run(args):
    try:
        case = load_case(Path(args.case).resolve())
        typed_case = load_typed_case(Path(args.case).resolve())
        _resolve_case_selection(args)
        if args.variant == "all":
            return _run_matrix_mode(args, case, typed_case)
        shapes = _resolve_cli_shapes(args)
        return max((_run(args, shape) for shape in shapes), default=0)
    except (
        RuntimeError,
        ValueError,
        OSError,
    ) as error:
        artifact = create_artifact_dir(args.output_dir, "invalid-case")
        error_code = (
            "codegen_not_executed"
            if any(token in str(error).lower() for token in ("jit", "codegen"))
            else "invalid_case"
        )
        profile_value = str(Path(args.profile).resolve()) if args.profile else ""
        report = build_report(
            "unknown",
            args.backend,
            args.soc_profile or "",
            "failed",
            str(error),
            stage="preflight",
            variant=args.variant,
            artifact_paths=["report.json"],
            run_parameters={"profile": profile_value},
            error_code=error_code
            if error_code != "invalid_case"
            else ("invalid_device_id" if "device_id" in str(error) else "invalid_case"),
        )
        report["profile"] = profile_value
        (artifact / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return EXIT_CODES["failed"]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--backend", default="ascendc_real_device")
    parser.add_argument("--soc-profile")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--mode", choices=("functional", "performance", "run", "prepare"), default="run"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--profiler",
        action="store_true",
        help="request profiler collection through the backend wrapper",
    )
    parser.add_argument(
        "--metric",
        choices=("runner_wall_clock", "device_kernel_duration"),
        default="runner_wall_clock",
        help="performance metric: runner wall clock (ms) or pure AI Core kernel "
        "duration collected by msprof (us)",
    )
    parser.add_argument(
        "--variant", default="fused", help="case variant selected by the Python policy"
    )
    parser.add_argument("--output-dir", default="device_validation_artifacts")
    parser.add_argument("--profile")
    parser.add_argument("--shape", nargs=2, type=int)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
