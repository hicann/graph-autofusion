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
"""Real device-validation case/SoC/shape/variant matrix orchestration."""

import json
import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from .case import CaseConfig
from .comparison import compare_variants
from .report import build_report
from .support_matrix import (
    SupportConfig,
    SupportDecision,
    load_profile,
    resolve_support,
)


def _artifact(root, case_id, soc, shape, variant):
    return Path(
        tempfile.mkdtemp(
            prefix=f"{case_id}-{soc}-{shape[0]}x{shape[1]}-{variant}-", dir=root
        )
    )


@dataclass(frozen=True)
class MatrixValidationContext:
    case: CaseConfig
    backend: str
    soc: str
    shape: tuple[int, ...]
    variant: str
    decision: SupportDecision
    request: dict
    artifact: Path
    artifact_root: Path


@dataclass(frozen=True)
class MatrixSpec:
    socs: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    variants: tuple[str, ...]


@dataclass(frozen=True)
class MatrixRunConfig:
    case: CaseConfig
    backend: str
    socs: list
    shapes: list
    variants: list
    root: Path
    loaded_profile: dict | None
    profile: object
    device: int
    mode: str
    warmup: int
    repeat: int
    profiler: bool
    metric: str
    executor: object | None


def _make_context(config, soc, shape, variant):
    decision = resolve_support(
        config.case,
        config.backend,
        soc,
        shape,
        SupportConfig(variant, config.loaded_profile),
    )
    ctx = MatrixValidationContext(
        case=config.case,
        backend=config.backend,
        soc=soc,
        shape=shape,
        variant=variant,
        decision=decision,
        request=None,
        artifact=_artifact(config.root, config.case.case_id, soc, shape, variant),
        artifact_root=config.root,
    )
    return replace(ctx, request=_request(config, ctx))


def _support_decisions(decision, reason):
    return {
        name: {
            "result": decision.capabilities.get(name, "not_applicable"),
            "reason": reason,
        }
        for name in ("compile", "functional", "precision", "performance")
    }


def _required_capability_failed(decision, decisions):
    if decision.entry is None or decision.status != "supported":
        return False
    return any(
        decision.capabilities.get(name) == "required"
        and decisions.get(name, {}).get("result") != "passed"
        for name in ("compile", "functional", "precision", "performance")
    )


def _failed(ctx, reason, code):
    return _schema_error(ctx, reason, code)


def _validate_precision_tensors(precision):
    outputs = precision.get("outputs")
    if outputs is None:
        return None
    if not isinstance(outputs, list) or len(outputs) != precision["tensor_count"]:
        return "precision tensor evidence is invalid"
    if (
        sum(item.get("element_count", -1) for item in outputs if isinstance(item, dict))
        != precision["element_count"]
    ):
        return "precision element evidence is invalid"
    if (
        sum(
            item.get("mismatch_count", -1) for item in outputs if isinstance(item, dict)
        )
        != precision["mismatch_count"]
    ):
        return "precision mismatch evidence is invalid"
    for item in outputs:
        if not _valid_precision_item(item):
            return "precision tensor evidence is invalid"
    return None


def _valid_precision_item(item):
    if not isinstance(item, dict):
        return False
    element_count = item.get("element_count")
    mismatch_count = item.get("mismatch_count")
    bad_element = (
        isinstance(element_count, bool)
        or not isinstance(element_count, int)
        or element_count < 0
    )
    bad_mismatch = (
        isinstance(mismatch_count, bool)
        or not isinstance(mismatch_count, int)
        or mismatch_count < 0
        or mismatch_count > element_count
    )
    return not bad_element and not bad_mismatch


def _validate_precision(report, required):
    precision = report.get("precision")
    if not isinstance(precision, dict):
        return "precision evidence is missing"
    if required and precision.get("passed") is not True:
        return "required precision did not pass"
    for key in ("tensor_count", "element_count", "mismatch_count"):
        if (
            isinstance(precision.get(key), bool)
            or not isinstance(precision.get(key), int)
            or precision[key] < 0
        ):
            return "precision evidence has invalid counts"
    tensor_error = _validate_precision_tensors(precision)
    if tensor_error is not None:
        return tensor_error
    if required and precision["mismatch_count"] != 0:
        return "required precision has mismatches"
    return ""


def _validate_performance(report, request, case, variant, required):
    actual = (report.get("performance") or {}).get("actual")
    if not isinstance(actual, dict):
        return "performance evidence is missing"
    samples = actual.get("samples")
    repeat = request.get("repeat")
    samples_bad = (
        not isinstance(samples, list)
        or not isinstance(repeat, int)
        or repeat < 1
        or len(samples) != actual.get("sample_count")
        or len(samples) != repeat
        or any(not _valid_number(value) for value in samples)
    )
    if required and samples_bad:
        return "performance samples do not match repeat"
    expected_kernels = len(case.steps("unfused")) if variant == "unfused" else 1
    kernel_bad = (
        actual.get("kernel_count") != expected_kernels
        or not isinstance(actual.get("metric"), str)
        or not actual["metric"]
        or not isinstance(actual.get("unit"), str)
        or not actual["unit"]
    )
    if required and kernel_bad:
        return "performance kernel or timing evidence is invalid"
    return ""


def _valid_number(value):
    return not (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    )


def _validate_abi(report, request, artifact_root):
    requested = report.get("requested_abi")
    actual = report.get("actual_abi")
    metadata = report.get("abi_metadata")
    abi_fields_ok = (
        isinstance(requested, str)
        and requested
        and isinstance(actual, str)
        and actual
        and requested == actual
    )
    metadata_ok = isinstance(metadata, dict) and metadata.get("launch_abi") == actual
    if not abi_fields_ok or not metadata_ok:
        return "requested and actual ABI metadata differ"
    return ""


def _step_abi_evidence_ok(step, report_path, step_report):
    if not (step.get("requested_abi") or step.get("actual_abi")):
        return True
    try:
        metadata_path = report_path.parent / "abi_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return step_report.get("abi_metadata") == metadata and metadata.get(
        "launch_abi"
    ) == step.get("actual_abi")


def _valid_step_item(step, index, artifact_root=None):
    if not isinstance(step, dict):
        return False
    if step.get("step") != index:
        return False
    lists_ok = (
        isinstance(step.get("input_specs"), list)
        and isinstance(step.get("output_specs"), list)
        and isinstance(step.get("samples"), list)
    )
    if not lists_ok:
        return False
    abi_ok = isinstance(step.get("requested_abi"), str) and step.get(
        "requested_abi"
    ) == step.get("actual_abi")
    if not abi_ok:
        return False
    report_path = Path(step.get("report_path", ""))
    if not report_path.is_file():
        return False
    if artifact_root is not None and not _contained(report_path, artifact_root):
        return False
    try:
        step_report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if step_report.get("requested_abi") != step.get("requested_abi") or step_report.get(
        "actual_abi"
    ) != step.get("actual_abi"):
        return False
    if not _step_abi_evidence_ok(step, report_path, step_report):
        return False
    if step.get("sample_count") != len(step["samples"]):
        return False
    if any(not _valid_number(value) for value in step["samples"]):
        return False
    if step.get("kernel_count") != 1:
        return False
    metric_ok = (
        isinstance(step.get("metric"), str)
        and step["metric"]
        and isinstance(step.get("timing_source"), str)
        and step["timing_source"]
    )
    return metric_ok


def _validate_steps(report, case, variant, artifact_root=None):
    if variant != "unfused":
        return ""
    steps = report.get("steps")
    expected = case.steps("unfused")
    if not isinstance(steps, list) or len(steps) != len(expected):
        return "unfused step evidence is incomplete"
    for index, step in enumerate(steps):
        if not _valid_step_item(step, index, artifact_root):
            return "unfused step evidence is incomplete"
    return ""


def _write_report(artifact, report):
    (artifact / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )


def _contained(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _resolve_artifact_path(ctx, item):
    path = Path(item)
    roots = (Path(ctx.artifact), Path(ctx.artifact_root))
    candidates = (path,) if path.is_absolute() else tuple(root / path for root in roots)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and any(_contained(resolved, root) for root in roots):
            return resolved
    return None


def _schema_error(ctx, reason, error_code="invalid_report_schema"):
    report = build_report(
        ctx.case.case_id,
        ctx.backend,
        ctx.soc,
        "failed",
        reason,
        stage="execution",
        variant=ctx.variant,
        error_code=error_code,
        support_decisions=_support_decisions(ctx.decision, reason),
        run_parameters=ctx.request,
        artifact_paths=[],
    )
    report["shape"] = list(ctx.shape)
    return report


def _check_report_identity(ctx, payload):
    identity = {
        "case": ctx.case.case_id,
        "backend": ctx.backend,
        "soc_profile": ctx.soc,
        "variant": ctx.variant,
    }
    if any(payload.get(key) != value for key, value in identity.items()):
        return _schema_error(
            ctx, "executor report identity mismatch", "invalid_report_identity"
        )
    reported_shape = payload.get(
        "shape",
        payload["run_parameters"].get(
            "shape", payload["run_parameters"].get("selected_shape")
        ),
    )
    if reported_shape is None:
        return _schema_error(ctx, "executor report shape is missing")
    if reported_shape != list(ctx.shape):
        return _schema_error(
            ctx, "executor report shape mismatch", "invalid_report_identity"
        )
    return None


def _check_artifact_paths(ctx, payload):
    for item in payload["artifact_paths"]:
        if not isinstance(item, str):
            return _schema_error(
                ctx, "executor artifact path has invalid type", "invalid_artifact_path"
            )
        if _resolve_artifact_path(ctx, item) is None:
            return _schema_error(
                ctx,
                "executor artifact path is outside or missing",
                "invalid_artifact_path",
            )

    return None


def _schema_structure_error(ctx, payload):
    required = {
        "schema_version",
        "case",
        "backend",
        "soc_profile",
        "variant",
        "stage_status",
        "precision",
        "performance",
        "run_parameters",
        "artifact_paths",
    }
    types = {
        "schema_version": int,
        "case": str,
        "backend": str,
        "soc_profile": str,
        "variant": str,
        "stage_status": str,
        "precision": dict,
        "performance": dict,
        "run_parameters": dict,
        "artifact_paths": list,
    }
    if (
        payload.get("schema_version") != 2
        or not required.issubset(payload)
        or any(
            isinstance(payload.get(key), bool)
            or not isinstance(payload.get(key), expected)
            for key, expected in types.items()
        )
    ):
        return _schema_error(ctx, "executor report has invalid schema")

    return None


def _validate_schema(ctx, payload, exit_code):
    if not isinstance(payload, dict):
        return _schema_error(ctx, "executor report is not an object")
    structure_error = _schema_structure_error(ctx, payload)
    if structure_error is not None:
        return structure_error
    identity_error = _check_report_identity(ctx, payload)
    if identity_error is not None:
        return identity_error
    path_error = _check_artifact_paths(ctx, payload)
    if path_error is not None:
        return path_error
    return None


def _normalize_report(ctx, report, payload, exit_code):
    report = dict(payload)

    if payload.get("backend") == "host_fake":
        report.update(
            {
                "host_fake": True,
                "stage_status": "failed",
                "error_code": "host_fake_not_hardware",
                "reason": "host_fake execution is contract-only and cannot pass hardware validation",
            }
        )
    report["artifact_paths"] = [
        str(_resolve_artifact_path(ctx, item)) for item in payload["artifact_paths"]
    ]
    report["shape"] = list(ctx.shape)
    report["artifact"] = str(ctx.artifact)
    report["execution_exit_code"] = exit_code
    report.setdefault("error_code", "")
    report.setdefault("stage", "execution")
    report.setdefault("reason", "")
    if ctx.backend == "ascendc_real_device":
        capability_outcome = report["stage"] == "preflight" and report[
            "stage_status"
        ] in ("skipped", "not_applicable")
        abi_error = (
            _validate_abi(report, ctx.request, ctx.artifact_root)
            if ctx.variant == "fused"
            else ""
        )
        if abi_error and not capability_outcome:
            return None, _failed(ctx, abi_error, "abi_mismatch")
    if exit_code != 0:
        report["stage_status"] = "failed"
        report["error_code"] = report["error_code"] or "execution_exit_code"

    return report, None


def _assemble_decisions(backend, report, payload_decisions, decision, required_names):
    if backend == "ascendc_real_device":
        if report["stage"] == "preflight" and report["stage_status"] in (
            "skipped",
            "not_applicable",
        ):
            decisions = payload_decisions
        else:
            decisions = {
                name: {
                    "result": (
                        "passed"
                        if payload_decisions[name]["result"] == "passed"
                        and report["stage_status"] == "passed"
                        else "failed"
                    ),
                    "reason": "matrix policy plus executor evidence",
                }
                for name in required_names
            }
    else:
        decisions = {}
        if payload_decisions:
            decisions = payload_decisions
        else:
            for name in required_names:
                value = decision.capabilities.get(name, "unsupported")
                decisions[name] = {
                    "result": (
                        "failed"
                        if value == "required" and report["stage_status"] == "skipped"
                        else "skipped"
                        if value == "optional"
                        else "passed"
                    ),
                    "reason": "legacy test executor",
                }

    return decisions


def _build_decisions(ctx, report, payload_decisions):
    required_names = {"compile", "functional", "precision", "performance"}
    if ctx.backend == "ascendc_real_device" and (
        set(payload_decisions) != required_names
        or any(
            not isinstance(payload_decisions[name], dict)
            or payload_decisions[name].get("result")
            not in {"passed", "failed", "skipped", "not_applicable"}
            for name in required_names
        )
    ):
        return None, _schema_error(
            ctx,
            "executor support decisions are incomplete",
            "invalid_support_decisions",
        )
    decisions = _assemble_decisions(
        ctx.backend, report, payload_decisions, ctx.decision, required_names
    )
    return decisions, None


def _optional_skip(report, exit_code, decision, required_failed, decisions):
    if report["stage"] != "preflight" or report["stage_status"] != "skipped":
        return False
    if exit_code != 0 or not decision.capabilities or required_failed:
        return False
    return any(
        value == "optional" and decisions.get(name, {}).get("result") == "skipped"
        for name, value in decision.capabilities.items()
    )


def _validate_evidence(ctx, report, strict_evidence, capability_outcome):
    if not strict_evidence or capability_outcome:
        return None
    precision_error = _validate_precision(
        report, ctx.decision.capabilities.get("precision") == "required"
    )
    if precision_error and ctx.decision.capabilities.get("precision") == "required":
        return _failed(ctx, precision_error, "precision_invalid")
    performance_error = _validate_performance(
        report,
        ctx.request,
        ctx.case,
        ctx.variant,
        ctx.decision.capabilities.get("performance") == "required",
    )
    if performance_error and ctx.decision.capabilities.get("performance") == "required":
        return _failed(ctx, performance_error, "performance_invalid")
    steps_error = _validate_steps(report, ctx.case, ctx.variant, ctx.artifact_root)
    if steps_error:
        return _failed(ctx, steps_error, "step_evidence_invalid")
    return None


def _apply_outcomes(ctx, report, decisions, payload_decisions, exit_code):
    required_failed = _required_capability_failed(ctx.decision, decisions)
    optional_skip = _optional_skip(
        report, exit_code, ctx.decision, required_failed, decisions
    )
    if required_failed:
        report["stage_status"] = "failed"
        report["error_code"] = report["error_code"] or "required_capability_failed"
    strict_evidence = ctx.backend == "ascendc_real_device"
    capability_outcome = report["stage"] == "preflight" and report["stage_status"] in (
        "skipped",
        "not_applicable",
    )
    evidence_error = _validate_evidence(
        ctx, report, strict_evidence, capability_outcome
    )
    if evidence_error is not None:
        return evidence_error
    preflight_gate = (
        report["stage"] == "preflight"
        and report["stage_status"] in ("skipped", "not_applicable")
        and not optional_skip
        and not required_failed
    )
    if preflight_gate:
        report["stage_status"] = "failed"
        report["error_code"] = report["error_code"] or "required_capability_failed"
    if report["stage_status"] == "failed" and optional_skip:
        report["stage_status"] = "skipped"
    return report


def _validate_payload(ctx, payload, exit_code):
    schema_error = _validate_schema(ctx, payload, exit_code)
    if schema_error is not None:
        return schema_error
    report = dict(payload)
    report, error = _normalize_report(ctx, report, payload, exit_code)
    if error is not None:
        return error
    payload_decisions = payload.get("support_decisions", {})
    decisions, error = _build_decisions(ctx, report, payload_decisions)
    if error is not None:
        return error
    return _apply_outcomes(ctx, report, decisions, payload_decisions, exit_code)


def _not_applicable(ctx):
    report = build_report(
        ctx.case.case_id,
        ctx.backend,
        ctx.soc,
        "not_applicable",
        ctx.decision.reason,
        stage="preflight",
        variant=ctx.variant,
        support_decisions=_support_decisions(ctx.decision, ctx.decision.reason),
        run_parameters=ctx.request,
        artifact_paths=["report.json"],
    )
    report["shape"] = list(ctx.shape)
    _write_report(ctx.artifact, report)
    return report | {"artifact": str(ctx.artifact), "execution_exit_code": 0}


def _request_specs(case, shape, artifact):
    specs = [
        {
            "dtype": item["dtype"],
            "shape": list(shape) if item.get("dynamic") else item.get("shape", shape),
            "file": str(artifact / f"input_{index}.bin"),
        }
        for index, item in enumerate(case.inputs)
    ]
    output_specs = [
        {
            "dtype": item["dtype"],
            "shape": list(shape) if item.get("dynamic") else item.get("shape", shape),
            "file": str(artifact / f"output_{index}.bin"),
        }
        for index, item in enumerate(case.outputs)
    ]

    return specs, output_specs


def _request(config, ctx):
    case = ctx.case
    soc = ctx.soc
    shape = ctx.shape
    variant = ctx.variant
    artifact = ctx.artifact
    decision = ctx.decision
    specs, output_specs = _request_specs(case, shape, artifact)
    variant_config = case.variant_config(variant)
    return {
        "case": case.case_dir,
        "case_dir": case.case_dir,
        "case_id": case.case_id,
        "backend": ctx.backend,
        "soc": soc,
        "soc_profile": soc,
        "profile": str(config.profile),
        "device": config.device,
        "mode": config.mode,
        "variant": variant,
        "shape": list(shape),
        "warmup": config.warmup,
        "repeat": config.repeat,
        "profiler": config.profiler,
        "metric": config.metric,
        "input_specs": specs,
        "output_specs": output_specs,
        "reference": str(Path(case.case_dir) / "reference.py"),
        "artifact": str(artifact),
        "artifact_dir": str(artifact),
        "support": decision.capabilities,
        "codegen_entry": variant_config.get("codegen_entry", "input_ascir.py"),
        "inputs": [],
        "outputs": [],
    }


def _real_executor(request, artifact):
    from device_validation.tools import run_device_validation

    args = run_device_validation.build_parser().parse_args(
        [
            "--case",
            request["case_dir"],
            "--backend",
            request["backend"],
            "--soc-profile",
            request["soc_profile"],
            "--profile",
            request["profile"],
            "--device",
            str(request["device"]),
            "--mode",
            request["mode"],
            "--variant",
            request["variant"],
            "--warmup",
            str(request["warmup"]),
            "--repeat",
            str(request["repeat"]),
            "--output-dir",
            str(request["artifact"]),
            "--shape",
            *map(str, request["shape"]),
        ]
        + (["--profiler"] if request["profiler"] else [])
    )
    exit_code = run_device_validation.run(args)
    report_path = getattr(args, "last_report_path", None)
    if report_path is None:
        report_path = getattr(args, "_last_report_path", None)
    if report_path is None or not Path(report_path).is_file():
        return {
            "stage_status": "failed",
            "error_code": "missing_report",
            "reason": "executor report is missing",
            "returncode": exit_code,
        }
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    payload["returncode"] = exit_code
    payload["report_path"] = str(report_path)
    payload["artifact_root"] = str(Path(report_path).parent)
    return payload


def _cell_report(ctx, result, exit_code):
    return _validate_payload(ctx, result, exit_code)


def _cell_result(ctx, executor):
    result = (executor or _real_executor)(ctx.request, ctx.artifact)
    exit_code = int(result.get("returncode", result.get("execution_exit_code", 0)))
    report_path = result.get("report_path")
    cell_artifact_root = ctx.artifact
    ctx = replace(ctx, artifact_root=result.get("artifact_root", ctx.artifact))
    invalid_location = (
        report_path is not None and (not _contained(report_path, cell_artifact_root))
    ) or not _contained(ctx.artifact_root, cell_artifact_root)
    if invalid_location:
        report = _schema_error(
            ctx,
            "executor report or artifact is outside cell artifact",
            "invalid_artifact_path",
        )
        report["execution_exit_code"] = exit_code
    else:
        report = _cell_report(ctx, result, exit_code)

    return report, report_path, ctx.artifact_root, result, invalid_location


def _run_cell(ctx, executor):
    if (
        ctx.decision.status != "supported"
        or "unsupported" in ctx.decision.capabilities.values()
    ):
        return _not_applicable(ctx)
    report, report_path, artifact_root, result, invalid_location = _cell_result(
        ctx, executor
    )
    (ctx.artifact / "raw_report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    report["artifact"] = str(ctx.artifact)
    report["artifact_root"] = str(Path(artifact_root).resolve())
    report["report_path"] = (
        str(Path(report_path).resolve())
        if report_path is not None and not invalid_location
        else str(ctx.artifact / "report.json")
    )
    _write_report(ctx.artifact, report)
    return report


def _gate_failed(records):
    return any(
        item.get("stage_status") not in ("passed", "skipped", "not_applicable")
        or item.get("execution_exit_code", 0) != 0
        for item in records.values()
    )


def _variant_actuals(records):
    actual = [
        records[name].get("performance", {}).get("actual", {})
        for name in ("fused", "unfused")
    ]
    if any(
        not isinstance(item.get("samples"), list) or not item.get("samples")
        for item in actual
    ):
        return None
    return actual


def _variants_ready(records, gate_failed):
    return (
        set(records) >= {"fused", "unfused"}
        and not gate_failed
        and all(item.get("stage_status") == "passed" for item in records.values())
    )


def _variant_thresholds(case):
    performance_config = case.raw.get("performance", {})
    thresholds = {}
    for key in ("max_regression_ratio", "min_speedup", "kernel_reduction_required"):
        if key in performance_config:
            thresholds[key] = performance_config[key]
    return thresholds


def _cell_comparison(case, records, gate_failed):
    gate_failed = _gate_failed(records)
    actual = (
        _variant_actuals(records) if _variants_ready(records, gate_failed) else None
    )
    if actual is None:
        return {
            "status": "failed" if gate_failed else "not_applicable",
            "reason": "variant execution gate failed"
            if gate_failed
            else "variant is optional or unsupported",
            "threshold_status": "not_configured",
        }
    return compare_variants(*actual, _variant_thresholds(case))


def _compare_cells(config, soc, shape, runs):
    pair = {}
    for item in runs:
        if item["soc_profile"] == soc and item.get("run_parameters", {}).get(
            "shape", item.get("shape")
        ) == list(shape):
            pair[item["variant"]] = item
    records = {}
    for variant in config.variants:
        records[variant] = pair.get(
            variant,
            {
                "stage_status": "failed",
                "error_code": "missing_report",
                "reason": f"{variant} report is missing",
            },
        )
    gate_failed = _gate_failed(records)
    comparison = _cell_comparison(config.case, records, gate_failed)
    item = {
        "schema_version": 2,
        "case": config.case.case_id,
        "case_id": config.case.case_id,
        "backend": config.backend,
        "soc": soc,
        "soc_profile": soc,
        "shape": list(shape),
        "variant": "all",
        "stage_status": comparison["status"],
        "performance_comparison": comparison,
        "reports": [records[variant] for variant in config.variants],
        "variants": records,
    }
    return item


def _collect_runs(config):
    runs = []
    for soc in config.socs:
        for shape in config.shapes:
            for variant in config.variants:
                ctx = _make_context(config, soc, tuple(shape), variant)
                runs.append(_run_cell(ctx, config.executor))

    return runs


def execute_matrix(
    case,
    backend,
    spec,
    output_dir,
    executor=None,
    *,
    profile=None,
    device=0,
    mode="run",
    warmup=3,
    repeat=2,
    profiler=False,
    comparison_root=None,
    metric="runner_wall_clock",
):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if backend == "ascendc_real_device" and profile is None:
        raise ValueError("device profile is required")
    loaded_profile = load_profile(profile) if profile is not None else None
    config = MatrixRunConfig(
        case=case,
        backend=backend,
        socs=spec.socs,
        shapes=spec.shapes,
        variants=spec.variants,
        root=root,
        loaded_profile=loaded_profile,
        profile=profile,
        device=device,
        mode=mode,
        warmup=warmup,
        repeat=repeat,
        profiler=profiler,
        metric=metric,
        executor=executor,
    )
    runs = _collect_runs(config)
    comparisons = []
    for soc in spec.socs:
        for shape in spec.shapes:
            item = _compare_cells(config, soc, tuple(shape), runs)
            comparison_dir = (
                _artifact(root, case.case_id, soc, tuple(shape), "comparison")
                if comparison_root is None
                else Path(comparison_root) / f"{shape[0]}x{shape[1]}" / "comparison"
            )
            comparison_dir.mkdir(parents=True, exist_ok=True)
            (comparison_dir / "comparison_report.json").write_text(
                json.dumps(item, indent=2), encoding="utf-8"
            )
            comparisons.append(item)
    neutral = {"passed", "skipped", "not_applicable"}
    return {
        "runs": runs,
        "comparisons": comparisons,
        "host_policy": "fake_matrix_host_only",
        "exit_code": 0
        if all(item["stage_status"] in neutral for item in comparisons)
        else 4,
    }
