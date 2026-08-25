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
import json
import os
import shutil
from subprocess import run as real_run
from pathlib import Path
from device_validation.tools import run_device_validation
from device_validation.tools.run_device_validation import (
    build_parser as cli_build_parser,
    classify_backend_payload,
    classify_failure,
    decode_output_payload,
    run,
    validate_case,
    verify_outputs,
    report_precision,
)

import numpy as np
import pytest

from device_validation.cases.isinf_maskedfill_fusion.gen_input import generate_inputs
from device_validation.cases.isinf_maskedfill_fusion.reference import compute_reference
from device_validation.python.test_helpers import (
    cli_args,
    flat_case_runner,
    flat_request_runner,
    mocked_abi_metadata,
    write_tensor_file,
)
from device_validation.tools.run_device_validation import (
    load_case,
    select_support,
    create_artifact_dir,
)


CASE_DIR = Path(__file__).parent.parent / "cases" / "isinf_maskedfill_fusion"


def build_parser():
    parser = cli_build_parser()
    parser.set_defaults(
        soc_profile="ascend950",
        profile=str(CASE_DIR.parents[1] / "profiles" / "ascend950.json"),
    )
    return parser


def runner_path():
    return Path(
        os.environ.get(
            "DEVICE_VALIDATION_RUNNER",
            "build/autofuse/tests/st/device_validation/device_validation_runner",
        )
    )


def require_runner():
    runner = runner_path()
    if not runner.is_file():
        pytest.skip("device_validation_runner is not built")
    return runner


def runner_request(case_dir, profile, module, **overrides):
    request = {
        "case_dir": str(case_dir),
        "case_id": "isinf_maskedfill_fusion",
        "step": 0,
        "profile": str(profile),
        "module": str(module),
        "abi": "AutofuseLaunchV2",
        "tensor_files": ["a", "b", "c"],
        "launch_abi": "AutofuseLaunchV2",
        "input_count": 3,
        "output_count": 1,
        "abi_metadata": {
            "launch_abi": "AutofuseLaunchV2",
            "input_count": 3,
            "output_count": 1,
            "input_dtypes": ["float16", "uint8", "float16"],
            "output_dtypes": ["float16"],
        },
        "inputs": ["a", "b", "c"],
        "outputs": ["out"],
        "tensor_specs": [
            {"dtype": "float16", "shape": [1]},
            {"dtype": "uint8", "shape": [1]},
            {"dtype": "float16", "shape": [1]},
        ],
        "output_specs": [{"dtype": "float16", "shape": [1]}],
        "artifact_dir": str(case_dir),
        "device": 0,
        "warmup": 0,
        "repeat": 1,
        "variant": "fused",
    }
    request.update(overrides)
    return request


def flat_profile():
    return CASE_DIR.parent.parent / "profiles" / "ascend950.json"


def preflight_case_dir(tmp_path, case):
    """Write a runner case dir restricted to one supported shape.

    Returns ``(case_dir, profile)``.
    """
    case["support_matrix"][0]["shapes"] = [[128, 128]]
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "case.json").write_text(json.dumps(case), encoding="utf-8")
    return case_dir, flat_profile()


def fake_runner(tmp_path, monkeypatch):
    """Create a stub runner file and wire it as the device runner."""
    runner = tmp_path / "runner"
    runner.write_text("runner", encoding="utf-8")
    monkeypatch.setenv("DEVICE_VALIDATION_RUNNER", str(runner))
    monkeypatch.setattr(run_device_validation, "_write_codegen", lambda *args: runner)
    monkeypatch.setattr(
        run_device_validation,
        "_write_inputs",
        lambda *args: ([], np.zeros((127, 129), dtype=np.float16)),
    )


def fake_subprocess_payload(monkeypatch, payload, calls=None):
    """Stub subprocess.run to return a Completed echo of ``payload``.

    When ``calls`` is given, each command is appended to it.
    """

    def fake_run(command, **kwargs):
        if calls is not None:
            calls.append(command)
        return type(
            "Completed",
            (),
            {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""},
        )()

    monkeypatch.setattr(run_device_validation.subprocess, "run", fake_run)


def stub_passed_execution(tmp_path, monkeypatch, actual, calls=None):
    """Stub the runner with a passed execution payload for ``actual``."""
    fake_runner(tmp_path, monkeypatch)
    payload = {
        "stage_status": "passed",
        "stage": "execution",
        "outputs": [
            {"dtype": "float16", "shape": [127, 129], "data": [0.0] * (127 * 129)}
        ],
        "performance": {"declared": {}, "actual": actual},
    }
    fake_subprocess_payload(monkeypatch, payload, calls)


def preflight_report_cases():
    """(payload, expected_status, expected_exit) triples for artifact storage."""
    support = {
        "compile": {"result": "failed", "reason": "required compile unavailable"}
    }
    return (
        (
            {
                "stage_status": "failed",
                "stage": "preflight",
                "support_decisions": support,
                "reason": "required compile unavailable",
            },
            "required_capability_failed",
            3,
        ),
        (
            {
                "stage_status": "skipped",
                "stage": "preflight",
                "support_decisions": {
                    "performance": {
                        "result": "skipped",
                        "reason": "profiler unavailable",
                    }
                },
                "reason": "optional profiler unavailable",
            },
            "skipped",
            0,
        ),
        (
            {
                "stage_status": "not_applicable",
                "stage": "preflight",
                "support_decisions": {
                    "compile": {
                        "result": "not_applicable",
                        "reason": "unsupported case",
                    }
                },
                "reason": "unsupported case",
            },
            "not_applicable",
            0,
        ),
    )


def test_manifest_declares_required_profiles_and_dtypes():
    case = load_case(CASE_DIR)
    assert case["schema_version"] == 1
    assert [item["shape"] for item in case["inputs"]] == [[1], [1], [1]]
    assert [item["dtype"] for item in case["inputs"]] == ["float16", "uint8", "float16"]
    assert case["outputs"][0]["dtype"] == "float16"
    entry = select_support(case, "ascendc_real_device", "ascend950", shape=(128, 128))
    assert entry["compile"] == "required"
    assert entry["functional"] == "required"
    assert entry["precision"] == "required"
    assert entry["performance"] == "optional"
    assert entry["shapes"] == [[128, 128], [128, 130], [127, 129], [512, 512]]


def test_manifest_declares_ascend910_with_explicit_profile_and_preserves_ascend950():
    case = load_case(CASE_DIR)
    entry_910 = select_support(
        case, "ascendc_real_device", "ascend910_9362", shape=(128, 130)
    )
    assert entry_910["soc"] == "ascend910_9362"
    assert entry_910["input_dtypes"] == ["float16", "uint8", "float16"]
    assert entry_910["output_dtypes"] == ["float16", "uint8"]
    assert validate_case(case, (128, 130), "ascendc_real_device", "ascend910_9362")

    profile = CASE_DIR.parents[1] / "profiles" / "ascend910_9362.json"
    args = build_parser().parse_args(
        cli_args(
            CASE_DIR,
            "prepare",
            shape=(128, 130),
            soc_profile="ascend910_9362",
            profile=profile,
        )
    )
    assert args.soc_profile == "ascend910_9362"
    assert Path(args.profile) == profile

    entry_950 = select_support(
        case, "ascendc_real_device", "ascend950", shape=(128, 130)
    )
    assert entry_950["soc"] == "ascend950"
    assert validate_case(case, (128, 130), "ascendc_real_device", "ascend950")


def test_reference_handles_finite_inf_mask_and_overlap():
    x = np.array([[1.0, np.inf, -np.inf, 4.0]], dtype=np.float16)
    mask = np.array([[0, 0, 0, 1]], dtype=np.uint8)
    value = np.array([[9.0, 9.0, 9.0, 9.0]], dtype=np.float16)
    np.testing.assert_array_equal(
        compute_reference(x, mask, value), np.array([[1, 9, 9, 9]], dtype=np.float16)
    )


def test_input_generation_is_deterministic_and_artifacts_are_isolated(tmp_path):
    first = generate_inputs((127, 129), seed=7)
    second = generate_inputs((127, 129), seed=7)
    for lhs, rhs in zip(first, second):
        np.testing.assert_array_equal(lhs, rhs)
    artifact = create_artifact_dir(tmp_path, "isinf_maskedfill_fusion")
    assert artifact.parent == tmp_path
    assert not artifact.is_relative_to(CASE_DIR)


def test_prepare_mode_writes_v2_manifest_and_report_only_to_artifact(tmp_path):
    args = build_parser().parse_args(
        cli_args(
            CASE_DIR,
            "prepare",
            output_dir=tmp_path,
            backend="ascendc_real_device",
            soc_profile="ascend950",
        )
    )
    assert run(args) == 0
    artifacts = list(tmp_path.iterdir())
    assert len(artifacts) == 4
    report = json.loads((artifacts[0] / "report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == 2
    assert report["stage_status"] == "not_applicable"
    assert (artifacts[0] / "manifest.json").exists()
    assert not list(CASE_DIR.glob("*.bin"))


def test_prepare_mode_supports_all_declared_shapes(tmp_path):
    for shape in ((128, 128), (128, 130), (127, 129), (512, 512)):
        args = build_parser().parse_args(
            cli_args(CASE_DIR, "prepare", shape=shape, output_dir=tmp_path)
        )
        assert run(args) == 0
    manifests = [
        json.loads(path.read_text()) for path in tmp_path.glob("*/manifest.json")
    ]
    assert sorted(tuple(item["shape"]) for item in manifests) == [
        (127, 129),
        (128, 128),
        (128, 130),
        (512, 512),
    ]


def test_float16_payload_is_decoded_and_empty_values_are_rejected():
    values = decode_output_payload(
        {"dtype": "float16", "shape": [2], "data": [1.0, -2.0]}
    )
    np.testing.assert_array_equal(values, np.array([1.0, -2.0], dtype=np.float16))
    with np.testing.assert_raises(ValueError):
        decode_output_payload({"dtype": "float16", "shape": [2], "data": []})


def test_bfloat16_payload_is_decoded_and_validated_as_two_bytes():
    values = decode_output_payload({"dtype": "bfloat16", "shape": [2], "data": [1, 2]})
    assert values.dtype == np.dtype("uint16")
    np.testing.assert_array_equal(values, np.array([1, 2], dtype=np.uint16))


def test_bfloat16_case_dtype_signature_is_supported():
    case = load_case(CASE_DIR)
    case["inputs"][0]["dtype"] = "bfloat16"
    case["outputs"][0]["dtype"] = "bfloat16"
    case["support_matrix"][0]["input_dtypes"][0] = "bfloat16"
    case["support_matrix"][0]["output_dtypes"][0] = "bfloat16"
    validate_case(case, (127, 129), "ascendc_real_device", "ascend950")


def test_validation_checks_all_mixed_dtype_signatures():
    case = load_case(CASE_DIR)
    validate_case(case, (127, 129), "ascendc_real_device", "ascend950")
    bad = json.loads(json.dumps(case))
    bad["inputs"][1]["dtype"] = "float16"
    with np.testing.assert_raises(ValueError):
        validate_case(bad, (128, 128), "ascendc_real_device", "ascend950")


def test_failure_classification_is_stable():
    assert classify_failure("optional profiler unavailable", "optional") == (
        "skipped",
        0,
    )
    assert classify_failure("unsupported shape", "unsupported") == ("not_applicable", 0)
    assert classify_failure("precision verification failed", "required") == (
        "precision_failed",
        5,
    )
    assert classify_failure("backend execution failed", "required") == ("failed", 4)


def test_backend_payload_requires_every_output_dtype_and_shape():
    payload = {"dtype": "float16", "shape": [2], "data": [1.0, 2.0]}
    np.testing.assert_array_equal(
        decode_output_payload(payload), np.array([1, 2], dtype=np.float16)
    )
    with np.testing.assert_raises(ValueError):
        decode_output_payload({"dtype": "uint8", "shape": [2], "data": [1]})


def test_decode_rejects_non_positive_and_overflowing_shapes():
    for shape in ([0], [-1], [2, 0], [2, -3], [2**63, 2**63], [2**63]):
        with np.testing.assert_raises(ValueError):
            decode_output_payload({"dtype": "float16", "shape": shape, "data": []})


def test_decode_rejects_bool_shape_dimensions():
    for dimension in (True, False):
        with np.testing.assert_raises(ValueError):
            decode_output_payload(
                {"dtype": "float16", "shape": [dimension], "data": []}
            )


def test_case_validation_rejects_bool_and_overflow_shapes():
    case = load_case(CASE_DIR)
    for shape in ((True,), (2**63, 2**63)):
        with np.testing.assert_raises(ValueError):
            validate_case(case, shape, "ascendc_real_device", "ascend950")


def test_unfused_step_output_validation_rejects_bool_and_overflow_shapes(tmp_path):
    from types import SimpleNamespace
    from device_validation.tools import run_device_validation as cli

    for shape in ([True], [2**63, 2**63]):
        output = tmp_path / "output.bin"
        output.write_bytes(b"")
        request = SimpleNamespace(
            outputs=(str(output),), output_specs=({"dtype": "float16", "shape": shape},)
        )
        payload = {"outputs": [{"dtype": "float16", "shape": shape}]}
        with np.testing.assert_raises(ValueError):
            cli.validate_step_outputs(request, payload)


def test_structured_backend_status_preserves_exit_semantics():
    assert classify_backend_payload(
        {
            "stage_status": "failed",
            "support_decisions": {"compile": {"result": "failed"}},
        }
    ) == ("required_capability_failed", 3)
    assert classify_backend_payload(
        {
            "stage_status": "failed",
            "support_decisions": {"compile": {"result": "passed"}},
        }
    ) == ("failed", 4)
    assert classify_backend_payload(
        {
            "stage_status": "failed",
            "stage": "verification",
            "precision": {"passed": False},
            "support_decisions": {"precision": {"result": "passed"}},
        }
    ) == ("precision_failed", 5)
    assert classify_backend_payload(
        {
            "stage_status": "skipped",
            "support_decisions": {"performance": {"result": "skipped"}},
        }
    ) == ("skipped", 0)
    assert classify_backend_payload(
        {
            "stage_status": "not_applicable",
            "support_decisions": {"compile": {"result": "not_applicable"}},
        }
    ) == ("not_applicable", 0)


def test_nonzero_backend_returncode_overrides_passed_payload():
    assert classify_backend_payload(
        {"stage": "execution", "stage_status": "passed"}, returncode=7
    ) == ("failed", 4)


def test_required_capability_preflight_beats_precision_failure_classification():
    for capability in ("compile", "functional", "precision", "performance"):
        payload = {
            "stage_status": "failed",
            "stage": "preflight",
            "support_decisions": {
                capability: {"result": "failed", "reason": "required unavailable"}
            },
            "precision": {"passed": False},
            "reason": "required capability failed",
        }
        assert classify_backend_payload(payload) == ("required_capability_failed", 3)

    assert classify_backend_payload(
        {
            "stage_status": "skipped",
            "stage": "preflight",
            "support_decisions": {"performance": {"result": "skipped"}},
        }
    ) == ("skipped", 0)
    assert classify_backend_payload(
        {
            "stage_status": "not_applicable",
            "stage": "preflight",
            "support_decisions": {"compile": {"result": "not_applicable"}},
        }
    ) == ("not_applicable", 0)
    assert classify_backend_payload(
        {
            "stage_status": "failed",
            "stage": "execution",
            "support_decisions": {"compile": {"result": "passed"}},
            "reason": "launch failed",
        }
    ) == ("failed", 4)
    assert classify_backend_payload(
        {
            "stage_status": "failed",
            "stage": "verification",
            "support_decisions": {"precision": {"result": "passed"}},
            "precision": {"passed": False},
            "reason": "precision verification failed",
        }
    ) == ("precision_failed", 5)


def test_preflight_artifact_reports_preserve_stage_and_support_reasons(tmp_path):
    case = load_case(CASE_DIR)
    for payload, expected_status, expected_exit in preflight_report_cases():
        artifact = create_artifact_dir(tmp_path, "preflight")
        report = run_device_validation.cli_report(
            run_device_validation.CliReportOptions(
                case,
                "ascendc_real_device",
                "ascend950",
                expected_status,
                payload["reason"],
                artifact,
                variant="fused",
                support=payload["support_decisions"],
                stage=payload["stage"],
            )
        )
        (artifact / "report.json").write_text(json.dumps(report), encoding="utf-8")
        stored = json.loads((artifact / "report.json").read_text())
        assert stored["stage"] == "preflight"
        assert stored["reason"] == payload["reason"]
        assert stored["support_decisions"] == payload["support_decisions"]
        assert {"required_capability_failed": 3, "skipped": 0, "not_applicable": 0}[
            expected_status
        ] == expected_exit


def test_cli_consumes_preflight_payload_stage_for_artifact_report(
    tmp_path, monkeypatch
):
    runner = tmp_path / "runner"
    runner.write_text("trusted runner")
    monkeypatch.setenv("DEVICE_VALIDATION_RUNNER", str(runner))
    monkeypatch.setattr(
        run_device_validation, "_write_codegen", lambda *args: tmp_path / "module.so"
    )
    (tmp_path / "module.so").write_text("module")
    payload = {
        "schema_version": 2,
        "stage_status": "skipped",
        "stage": "preflight",
        "reason": "optional profiler unavailable",
        "support_decisions": {
            "performance": {"result": "skipped", "reason": "profiler unavailable"}
        },
        "outputs": [],
    }

    def fake_run(*args, **kwargs):
        return type(
            "Completed",
            (),
            {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""},
        )()

    monkeypatch.setattr(run_device_validation.subprocess, "run", fake_run)
    args = build_parser().parse_args(cli_args(CASE_DIR, "run", output_dir=tmp_path))
    assert run(args) == 0
    reports = list(tmp_path.glob("*/report.json"))
    stored = json.loads(reports[-1].read_text())
    assert stored["stage"] == "preflight"
    assert (
        stored["support_decisions"]["performance"]["reason"] == "profiler unavailable"
    )


def test_real_runner_preflight_payload_classifies_required_failure(tmp_path):
    runner = require_runner()
    case = load_case(CASE_DIR)
    case_dir, profile = preflight_case_dir(tmp_path, case)
    inputs = []
    for index in range(len(case["inputs"])):
        path = tmp_path / f"input_{index}.bin"
        write_tensor_file(path, case["inputs"][index]["dtype"], [1])
        inputs.append(str(path))
    result, payload = flat_case_runner(
        runner, [case_dir, profile, tmp_path / "missing.so", *inputs], (127, 129)
    )
    assert result.returncode == 3
    assert payload["stage_status"] == "failed"
    assert payload["support_decisions"]["compile"]["result"] == "failed"
    assert classify_backend_payload(payload) == ("required_capability_failed", 3)


def test_real_runner_rejects_flat_input_dtype_and_shape_mismatch(tmp_path):
    runner = require_runner()
    profile = flat_profile()
    for specs in (
        [
            {"dtype": "float32", "shape": [1]},
            {"dtype": "uint8", "shape": [1]},
            {"dtype": "float16", "shape": [1]},
        ],
        [
            {"dtype": "float16", "shape": [2]},
            {"dtype": "uint8", "shape": [1]},
            {"dtype": "float16", "shape": [1]},
        ],
    ):
        request = runner_request(
            CASE_DIR, profile, tmp_path / "missing.so", tensor_specs=specs
        )
        result, payload = flat_request_runner(runner, tmp_path, request)
        assert result.returncode in (2, 3)
        assert payload["error_code"] != "spec_mismatch"


def test_real_runner_rejects_flat_file_and_spec_schema_mismatch(tmp_path):
    runner = require_runner()
    profile = flat_profile()
    cases = (
        {"tensor_files": ["different", "b", "c"]},
        {"inputs": ["a", "b"]},
        {"outputs": []},
        {"tensor_specs": "not-an-array"},
        {"output_specs": ["not-a-spec"]},
    )
    for overrides in cases:
        request = runner_request(
            CASE_DIR, profile, tmp_path / "missing.so", **overrides
        )
        result, payload = flat_request_runner(runner, tmp_path, request)
        assert result.returncode == 2
        assert payload["stage_status"] == "failed"


def test_real_runner_reports_missing_flat_output_fields_with_specific_codes(tmp_path):
    runner = require_runner()
    profile = flat_profile()
    for field, error_code in (
        ("outputs", "output_count"),
        ("output_specs", "output_specs"),
    ):
        request = runner_request(CASE_DIR, profile, tmp_path / "missing.so")
        del request[field]
        result, payload = flat_request_runner(
            runner, tmp_path, request, f"missing_{field}.json"
        )
        assert result.returncode == 2
        assert payload["error_code"] == error_code


def test_real_runner_reports_flat_output_field_type_with_specific_codes(tmp_path):
    runner = require_runner()
    profile = flat_profile()
    for field, value, error_code in (
        ("outputs", "out", "output_count"),
        ("output_specs", {"dtype": "float16"}, "output_specs"),
    ):
        request = runner_request(
            CASE_DIR, profile, tmp_path / "missing.so", **{field: value}
        )
        result, payload = flat_request_runner(
            runner, tmp_path, request, f"invalid_{field}.json"
        )
        assert result.returncode == 2
        assert payload["error_code"] == error_code


def test_real_runner_flat_output_count_and_positional_arity_errors(tmp_path):
    runner = require_runner()
    profile = flat_profile()
    request = runner_request(CASE_DIR, profile, tmp_path / "missing.so", outputs=[])
    result, payload = flat_request_runner(runner, tmp_path, request)
    assert result.returncode == 2
    assert payload["error_code"] == "output_count"

    inputs = [str(tmp_path / name) for name in ("a", "b", "c")]
    for path in inputs:
        write_tensor_file(Path(path), "float16", [1])
    correct, _ = flat_case_runner(
        runner, [CASE_DIR, profile, tmp_path / "missing.so", *inputs]
    )
    missing, _ = flat_case_runner(
        runner, [CASE_DIR, profile, tmp_path / "missing.so", *inputs[:2]]
    )
    extra, _ = flat_case_runner(
        runner,
        [CASE_DIR, profile, tmp_path / "missing.so", *inputs, str(tmp_path / "extra")],
    )
    assert correct.returncode == 3
    assert missing.returncode == 2
    assert extra.returncode == 2


def test_real_runner_rejects_non_string_flat_output_path(tmp_path):
    runner = require_runner()
    profile = flat_profile()
    request = runner_request(CASE_DIR, profile, tmp_path / "missing.so", outputs=[1])
    result, payload = flat_request_runner(runner, tmp_path, request)
    assert result.returncode == 2
    assert payload["error_code"] == "output_count"


def test_real_runner_preserves_backend_error_before_flat_output_write(tmp_path):
    runner = require_runner()
    profile = flat_profile()
    inputs = []
    for index in range(3):
        path = tmp_path / f"input_{index}.bin"
        write_tensor_file(path, "float16", [1])
        inputs.append(str(path))
    output_path = tmp_path / "missing_parent" / "output.bin"
    request = runner_request(
        CASE_DIR,
        profile,
        tmp_path / "missing.so",
        inputs=inputs,
        tensor_files=inputs,
        outputs=[str(output_path)],
        shape=[127, 129],
    )
    result, payload = flat_request_runner(runner, tmp_path, request)

    assert result.returncode != 0
    assert payload["stage"]
    assert payload["error_code"]
    assert payload["error_code"] != "output_write"
    assert payload["reason"]
    assert payload["outputs"] == []
    assert not output_path.exists()


def test_real_runner_preflight_preserves_stage_and_support_reason(tmp_path):
    runner = require_runner()
    case = load_case(CASE_DIR)
    case_dir, profile = preflight_case_dir(tmp_path, case)
    inputs = []
    for index in range(len(case["inputs"])):
        path = tmp_path / f"input_{index}.bin"
        path.write_bytes(b"\0\0" if index != 1 else b"\0")
        inputs.append(str(path))
    _, payload = flat_case_runner(
        runner, [case_dir, profile, tmp_path / "missing.so", *inputs], (127, 129)
    )
    assert payload["stage"] == "preflight"
    assert payload["support_decisions"]["compile"]["reason"]
    assert payload["reason"]
    assert classify_backend_payload(payload)[1] == 3


def test_manifest_records_runtime_parameters(tmp_path):
    args = build_parser().parse_args(
        cli_args(
            CASE_DIR,
            "prepare",
            (127, 129),
            output_dir=tmp_path,
            warmup="7",
            repeat="3",
        )
    )
    assert run(args) == 0
    artifact = next(tmp_path.iterdir())
    manifest = json.loads((artifact / "manifest.json").read_text())
    report = json.loads((artifact / "report.json").read_text())
    for payload in (manifest, report["run_parameters"]):
        assert payload["selected_shape"] == [127, 129]
        assert payload["warmup"] == 7
        assert payload["repeat"] == 3
        assert payload["profile"]


@pytest.mark.real_codegen
def test_codegen_entry_receives_selected_shape_and_writes_jit_module(
    tmp_path, monkeypatch
):
    from device_validation.tools.run_device_validation import (
        _write_codegen,
        CodegenOptions,
    )

    jit = tmp_path / "jit"
    jit.write_text(
        "#!/bin/sh\n"
        "output=''\n"
        "while [ $# -gt 0 ]; do\n"
        '  if [ "$1" = "--output" ]; then output=$2; shift 2; else shift; fi\n'
        "done\n"
        "cat > \"${output}.c\" <<'EOF'\n"
        "#include <stddef.h>\n"
        "size_t GetTilingDataSize(void) { return 1; }\n"
        "EOF\n"
        'cc -shared -fPIC "${output}.c" -o "$output"\n',
        encoding="utf-8",
    )
    jit.chmod(0o755)
    monkeypatch.setenv("AUTOFUSE_DEVICE_JIT", str(jit))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "--output-dir" in command:
            Path(
                command[command.index("--output-dir") + 1], "device_impl.cpp"
            ).write_text(
                'extern "C" uint32_t AutofuseLaunchV2(uint32_t blockDim, void *stream, void **inputs, '
                "int32_t input_count, void **outputs, int32_t output_count, void *workspace, "
                "void *tiling_data) { return 0; }"
            )
            output_dir = Path(command[command.index("--output-dir") + 1])
            (output_dir / "host_impl.cpp").write_text(
                (output_dir / "device_impl.cpp").read_text()
            )
            (output_dir / "tiling.h").write_text("#pragma once\n")
            (output_dir / "abi_metadata.json").write_text(
                json.dumps(
                    {
                        "launch_abi": "AutofuseLaunchV2",
                        "input_count": 3,
                        "output_count": 1,
                        "input_dtypes": ["float16", "uint8", "float16"],
                        "output_dtypes": ["float16"],
                        "parameter_count": 8,
                    }
                )
            )
        else:
            destination = command[command.index("--output") + 1]
            source = Path(destination).with_suffix(".c")
            source.write_text(
                "#include <stddef.h>\nsize_t GetTilingDataSize(void) { return 1; }\n"
            )
            real_run(
                [
                    shutil.which("cc") or "cc",
                    "-shared",
                    "-fPIC",
                    str(source),
                    "-o",
                    destination,
                ],
                check=True,
            )
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(run_device_validation.subprocess, "run", fake_run)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_codegen(
        CodegenOptions(
            CASE_DIR,
            artifact,
            (127, 129),
            profile=CASE_DIR.parent.parent / "profiles" / "ascend950.json",
            input_dtypes=["float16", "uint8", "float16"],
            output_dtypes=["float16"],
        )
    )
    assert "--rows" in calls[0] and calls[0][calls[0].index("--rows") + 1] == "127"
    assert "--cols" in calls[1] and calls[1][calls[1].index("--cols") + 1] == "129"


def test_explicit_functional_and_performance_modes_are_cli_aliases():
    functional = build_parser().parse_args(cli_args(CASE_DIR, "functional"))
    performance = build_parser().parse_args(cli_args(CASE_DIR, "performance"))

    assert functional.mode == "functional"
    assert performance.mode == "performance"
    assert functional.output_dir == "device_validation_artifacts"


def test_cli_rejects_unknown_mode():
    exited = False
    try:
        build_parser().parse_args(cli_args(CASE_DIR, "hardware"))
    except BaseException:
        exited = True
    assert exited


def test_performance_without_msprof_is_reported_as_runner_timing(tmp_path, monkeypatch):
    args = build_parser().parse_args(
        cli_args(CASE_DIR, "performance", (127, 129), output_dir=tmp_path)
    )
    # Performance remains an explicit runner timing report when no profiler
    # wrapper is present; the fake runner represents that host-only contract.
    stub_passed_execution(
        tmp_path,
        monkeypatch,
        {
            "timing_source": "runner_wall_clock",
            "profiler_collected": False,
            "profiler_tool_available": False,
        },
    )

    assert run(args) == 0
    report = json.loads(
        next(tmp_path.glob("*/report.json")).read_text(encoding="utf-8")
    )
    assert report["performance"]["actual"]["timing_source"] == "runner_wall_clock"
    assert report["performance"]["actual"]["profiler_collected"] is False
    assert report["performance"]["actual"]["profiler_tool_available"] is False
    assert report["precision"]["passed"] is True
    assert report["precision"]["tensor_count"] == 1
    assert report["precision"]["element_count"] == 127 * 129
    assert report["precision"]["mismatch_count"] == 0


def test_performance_mode_requests_profiler_and_required_unavailable_stops_preflight(
    tmp_path, monkeypatch
):
    case = load_case(CASE_DIR)
    case["support_matrix"][0]["performance"] = "required"
    required_case = tmp_path / "required_profiler_case"
    required_case.mkdir()
    (required_case / "case.json").write_text(json.dumps(case), encoding="utf-8")
    calls = []
    fake_runner(tmp_path, monkeypatch)

    payload = {
        "schema_version": 2,
        "stage_status": "failed",
        "stage": "preflight",
        "reason": "profiler is unavailable",
        "support_decisions": {
            "performance": {"result": "failed", "reason": "profiler is unavailable"}
        },
        "performance": {
            "actual": {
                "profiler_requested": True,
                "profiler_tool_available": False,
                "profiler_collected": False,
            }
        },
        "outputs": [],
    }

    fake_subprocess_payload(monkeypatch, payload, calls)
    args = build_parser().parse_args(
        cli_args(
            required_case,
            "performance",
            (127, 129),
            output_dir=tmp_path / "required-artifacts",
        )
    )

    assert run(args) == 3
    request = json.loads(Path(calls[0][-1]).read_text())
    assert request["profiler"] is True
    report = json.loads(
        next((tmp_path / "required-artifacts").glob("*/report.json")).read_text()
    )
    assert report["stage_status"] == "required_capability_failed"
    assert report["performance"]["actual"]["profiler_requested"] is True


def test_wall_clock_mode_does_not_request_profiler(tmp_path, monkeypatch):
    calls = []
    stub_passed_execution(
        tmp_path,
        monkeypatch,
        {
            "profiler_requested": False,
            "profiler_tool_available": False,
            "profiler_collected": False,
            "timing_source": "runner_wall_clock",
        },
        calls,
    )
    args = build_parser().parse_args(
        cli_args(
            CASE_DIR,
            "functional",
            (127, 129),
            output_dir=tmp_path / "wall-clock-artifacts",
        )
    )

    assert run(args) == 0
    assert "--profiler" not in calls[0]


def test_explicit_profiler_flag_is_forwarded(tmp_path, monkeypatch):
    calls = []
    fake_runner(tmp_path, monkeypatch)
    payload = {
        "stage_status": "failed",
        "stage": "preflight",
        "reason": "profiler is unavailable",
        "support_decisions": {"performance": {"result": "failed"}},
        "outputs": [],
    }
    fake_subprocess_payload(monkeypatch, payload, calls)
    args = build_parser().parse_args(
        cli_args(
            CASE_DIR,
            "functional",
            (127, 129),
            output_dir=tmp_path / "artifacts",
            profiler=True,
        )
    )

    assert run(args) == 3
    request = json.loads(Path(calls[0][-1]).read_text())
    assert request["profiler"] is True


def test_prepare_without_shape_creates_one_artifact_for_every_supported_shape(tmp_path):
    args = build_parser().parse_args(cli_args(CASE_DIR, "prepare", output_dir=tmp_path))

    assert run(args) == 0
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in tmp_path.glob("*/manifest.json")
    ]
    assert sorted(tuple(manifest["selected_shape"]) for manifest in manifests) == [
        (127, 129),
        (128, 128),
        (128, 130),
        (512, 512),
    ]


def test_prepare_with_shape_creates_only_the_selected_artifact(tmp_path):
    args = build_parser().parse_args(
        cli_args(CASE_DIR, "prepare", (127, 129), output_dir=tmp_path)
    )

    assert run(args) == 0
    manifests = list(tmp_path.glob("*/manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text(encoding="utf-8"))["selected_shape"] == [
        127,
        129,
    ]


def test_undeclared_backend_and_unsupported_shape_are_not_applicable(tmp_path):
    for overrides, label in (
        ({"backend": "unknown_backend"}, "unknown_backend"),
        ({"shape": (1, 2)}, "2"),
    ):
        output_dir = tmp_path / label
        args = build_parser().parse_args(
            cli_args(CASE_DIR, "prepare", output_dir=output_dir, **overrides)
        )
        assert run(args) == 0
        report = json.loads(
            next(output_dir.glob("*/report.json")).read_text(encoding="utf-8")
        )
        assert report["stage_status"] == "not_applicable"
        assert report["stage"] == "preflight"
        assert report["support_decisions"]


def test_invalid_schema_remains_a_failed_invalid_case(tmp_path):
    case_dir = tmp_path / "invalid"
    case_dir.mkdir()
    (case_dir / "case.json").write_text(
        json.dumps({"schema_version": 99}), encoding="utf-8"
    )
    args = build_parser().parse_args(cli_args(case_dir, "prepare", output_dir=tmp_path))

    assert run(args) == 4
    report = json.loads(
        next(tmp_path.glob("invalid-case-*/report.json")).read_text(encoding="utf-8")
    )
    assert report["stage_status"] == "failed"
    assert report["case_id"] == "unknown"


def test_python_verifier_reports_each_output_mismatch_details(tmp_path):
    actual = [
        np.array([1.0, np.nan], dtype=np.float32),
        np.array([-1, 3], dtype=np.int32),
    ]
    expected = [
        np.array([2.0, 4.0], dtype=np.float32),
        np.array([-1, 2], dtype=np.int32),
    ]
    precision = verify_outputs(actual, expected, atol=0.0, rtol=0.0)
    report = report_precision(precision, tmp_path)
    assert report["mismatch_count"] == 3
    assert report["nan_mismatch_count"] == 1
    assert report["max_abs_error"] == 1.0
    assert report["max_rel_error"] == 0.5
    assert report["first_mismatch"]["linear_index"] == 0
    assert report["first_mismatch"]["actual"] == 1.0
    assert report["first_mismatch"]["expected"] == 2.0
    assert report["outputs"][1]["mismatch_count"] == 1
    assert report["artifact"] == str(tmp_path)


def test_integer_outputs_are_exact_even_with_nonzero_tolerance():
    actual = [np.array([1, 3], dtype=np.int32)]
    expected = [np.array([2, 3], dtype=np.int32)]
    assert not verify_outputs(actual, expected, atol=1.0, rtol=1.0)["passed"]


def test_fixed_shape_one_is_not_a_dynamic_shape_sentinel():
    case = load_case(CASE_DIR)
    assert validate_case(case, (128, 128), "ascendc_real_device", "ascend950")
    assert case["outputs"][0]["shape"] == [1]


def test_unknown_backend_status_is_classified_as_failed():
    assert classify_backend_payload(
        {"stage_status": "unknown", "support_decisions": {}}
    ) == ("failed", 4)


def test_backend_error_code_is_preserved_in_report(tmp_path):
    case = load_case(CASE_DIR)
    artifact = create_artifact_dir(tmp_path, "error")
    report = run_device_validation.cli_report(
        run_device_validation.CliReportOptions(
            case,
            "backend",
            "soc",
            "failed",
            "launch failed",
            artifact,
            variant="fused",
            error_code="launch_failed",
            stage="launch",
        )
    )
    assert report["schema_version"] == 2
    assert report["error_code"] == "launch_failed"


def test_output_verification_uses_independent_output_shapes():
    actual = [np.zeros((2, 3), dtype=np.float32), np.zeros((4,), dtype=np.float32)]
    expected = [np.zeros((2, 3), dtype=np.float32), np.zeros((4,), dtype=np.float32)]
    assert verify_outputs(actual, expected, atol=0.0, rtol=0.0)["passed"]


def test_cli_rejects_negative_device_id_before_preflight(tmp_path):
    args = build_parser().parse_args(
        cli_args(CASE_DIR, "prepare", output_dir=tmp_path, device="-1")
    )
    assert run(args) == 4
    report = json.loads(next(tmp_path.glob("invalid-case-*/report.json")).read_text())
    assert report["stage"] == "preflight"
    assert report["error_code"] == "invalid_device_id"


def _unfused_case_with_aclnn(step_index, op_name):
    case = load_case(CASE_DIR)
    steps = [dict(step) for step in case["variants"]["unfused"]["steps"]]
    steps[step_index]["aclnn"] = op_name
    case["variants"]["unfused"]["steps"] = steps
    return case


def _unfused_ctx(tmp_path, case):
    args = build_parser().parse_args(
        cli_args(CASE_DIR, "run", (128, 128), output_dir=tmp_path / "out")
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir(parents=True, exist_ok=True)
    return run_device_validation.UnfusedContext(
        case=case,
        case_dir=CASE_DIR,
        shape=(128, 128),
        args=args,
        run_parameters={"soc": "ascend950"},
        artifact=artifact,
        runner_command=["runner"],
        msprof_path=None,
        metric="runner_wall_clock",
    )


def test_unfused_steps_parse_aclnn_field(tmp_path):
    from device_validation.python.case import load_typed_case

    plain = load_typed_case(CASE_DIR).steps("unfused")
    assert all("aclnn" not in step for step in plain)
    case = _unfused_case_with_aclnn(1, "LogicalOr")
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "case.json").write_text(json.dumps(case), encoding="utf-8")
    steps = load_typed_case(case_dir).steps("unfused")
    assert [step.get("aclnn") for step in steps] == [None, "LogicalOr", None]
    assert steps[0] == dict(plain[0])
    assert steps[1]["inputs"] == dict(plain[1])["inputs"]


def _unfused_case_with_all_aclnn():
    case = load_case(CASE_DIR)
    steps = [dict(step) for step in case["variants"]["unfused"]["steps"]]
    for step, op_name in zip(steps, ("IsInf", "LogicalOr", "MaskedFillTensor")):
        step["aclnn"] = op_name
    case["variants"]["unfused"]["steps"] = steps
    return case


def _write_case(tmp_path, case):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "case.json").write_text(json.dumps(case), encoding="utf-8")
    return case_dir


def _unfused_args(tmp_path):
    args = build_parser().parse_args(
        cli_args(
            CASE_DIR,
            "run",
            (128, 128),
            output_dir=tmp_path / "out",
            variant="unfused",
        )
    )
    return args


def test_unfused_steps_all_aclnn_does_not_require_jit_env(tmp_path, monkeypatch):
    case = _unfused_case_with_all_aclnn()
    case_dir = _write_case(tmp_path, case)
    args = _unfused_args(tmp_path)
    monkeypatch.setenv("DEVICE_VALIDATION_RUNNER", "runner")
    monkeypatch.delenv("AUTOFUSE_DEVICE_JIT", raising=False)
    unfused = run_device_validation.unfused_steps(case_dir, args)
    assert unfused["runner_command"] == ["runner"]
    assert len(unfused["steps"]) == 3


def test_unfused_steps_all_aclnn_still_requires_runner_env(tmp_path, monkeypatch):
    case = _unfused_case_with_all_aclnn()
    case_dir = _write_case(tmp_path, case)
    args = _unfused_args(tmp_path)
    monkeypatch.delenv("DEVICE_VALIDATION_RUNNER", raising=False)
    with pytest.raises(FileNotFoundError):
        run_device_validation.unfused_steps(case_dir, args)


def test_unfused_steps_mixed_steps_require_jit_env(tmp_path, monkeypatch):
    case = _unfused_case_with_aclnn(0, "IsInf")
    case_dir = _write_case(tmp_path, case)
    args = _unfused_args(tmp_path)
    monkeypatch.setenv("DEVICE_VALIDATION_RUNNER", "runner")
    monkeypatch.delenv("AUTOFUSE_DEVICE_JIT", raising=False)
    with pytest.raises(FileNotFoundError):
        run_device_validation.unfused_steps(case_dir, args)


def test_unfused_steps_without_aclnn_keeps_jit_requirement(tmp_path, monkeypatch):
    case_dir = _write_case(tmp_path, load_case(CASE_DIR))
    args = _unfused_args(tmp_path)
    monkeypatch.setenv("DEVICE_VALIDATION_RUNNER", "runner")
    monkeypatch.delenv("AUTOFUSE_DEVICE_JIT", raising=False)
    with pytest.raises(FileNotFoundError):
        run_device_validation.unfused_steps(case_dir, args)


def test_unfused_case_declares_aclnn_steps():
    from device_validation.python.case import load_typed_case

    steps = load_typed_case(CASE_DIR).steps("unfused_aclnn")
    assert len(steps) == 3
    assert [step.get("aclnn") for step in steps] == [
        "IsInf",
        "LogicalOr",
        "MaskedFillScalar",
    ]
    assert [step.get("graph") for step in steps] == [None, None, None]


def test_unfused_steps_selects_variant_from_args(tmp_path, monkeypatch):
    args = build_parser().parse_args(
        cli_args(
            CASE_DIR,
            "run",
            (128, 128),
            output_dir=tmp_path / "out",
            variant="unfused_aclnn",
        )
    )
    monkeypatch.setenv("DEVICE_VALIDATION_RUNNER", "runner")
    monkeypatch.delenv("AUTOFUSE_DEVICE_JIT", raising=False)
    unfused = run_device_validation.unfused_steps(CASE_DIR, args)
    assert [step.get("aclnn") for step in unfused["steps"]] == [
        "IsInf",
        "LogicalOr",
        "MaskedFillScalar",
    ]
    monkeypatch.setenv("AUTOFUSE_DEVICE_JIT", "jit")
    args.variant = "unfused"
    unfused = run_device_validation.unfused_steps(CASE_DIR, args)
    assert all("aclnn" not in step for step in unfused["steps"])


def test_prepared_variant_dispatches_on_declared_steps(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact"
    artifact.mkdir(parents=True, exist_ok=True)
    prepared = {
        "case_dir": str(CASE_DIR),
        "case": load_case(CASE_DIR),
        "artifact": artifact,
        "profile": str(flat_profile()),
        "metric": "runner_wall_clock",
        "msprof_path": None,
        "run_parameters": {"soc": "ascend950"},
        "support": {},
    }
    unfused_variants = []

    def capture_unfused(context, goldens, run_parameters):
        unfused_variants.append(context.args.variant)
        return {"stage_status": "passed"}

    monkeypatch.setattr(run_device_validation, "_run_unfused_mode", capture_unfused)
    monkeypatch.setattr(
        run_device_validation,
        "_run_fused",
        lambda ctx: (None, "fused-report"),
    )
    args = build_parser().parse_args(
        cli_args(
            CASE_DIR,
            "functional",
            (128, 128),
            output_dir=tmp_path / "out",
            variant="unfused_aclnn",
        )
    )
    reported = run_device_validation.run_prepared_variant(args, (128, 128), prepared)
    assert unfused_variants == ["unfused_aclnn"]
    assert reported == {"stage_status": "passed"}
    args.variant = "fused"
    reported = run_device_validation.run_prepared_variant(args, (128, 128), prepared)
    assert unfused_variants == ["unfused_aclnn"]
    assert reported == "fused-report"


def test_build_step_request_aclnn_skips_codegen_and_metadata(tmp_path, monkeypatch):
    case = _unfused_case_with_aclnn(0, "IsInf")
    ctx = _unfused_ctx(tmp_path, case)

    def fail_codegen(*args, **kwargs):
        raise AssertionError("aclnn steps must not run codegen or JIT")

    monkeypatch.setattr(run_device_validation, "_write_codegen", fail_codegen)
    bundle = run_device_validation.build_step_request(
        ctx, case["variants"]["unfused"]["steps"][0], 0
    )
    request = bundle["request"]
    assert request.aclnn_op == "IsInf"
    assert request.module == ""
    assert request.launch_abi == ""
    assert request.abi == ""
    assert request.abi_metadata == {}
    assert request.input_count == len(request.inputs) == 1
    assert request.output_count == len(request.outputs) == 1
    assert list(request.tensor_specs) == bundle["input_specs"]
    assert request.contract_schema["inputs"] == list(request.tensor_specs)
    assert request.contract_schema["outputs"] == list(request.output_specs)
    assert bundle["metadata"] == {}
    assert run_device_validation.request_payload(request)["aclnn_op"] == "IsInf"


def test_build_step_request_without_aclnn_keeps_legacy_contract(tmp_path, monkeypatch):
    case = load_case(CASE_DIR)
    ctx = _unfused_ctx(tmp_path, case)
    module = tmp_path / "kernel_module.so"
    monkeypatch.setattr(
        run_device_validation, "_write_codegen", lambda *args, **kwargs: module
    )
    monkeypatch.setattr(run_device_validation, "read_abi_metadata", mocked_abi_metadata)
    bundle = run_device_validation.build_step_request(
        ctx, case["variants"]["unfused"]["steps"][0], 0
    )
    request = bundle["request"]
    assert request.aclnn_op == ""
    assert request.module == str(module)
    assert request.launch_abi == "AutofuseLaunchV2"
    assert request.abi == "AutofuseLaunchV2"
    assert request.abi_metadata["input_count"] == request.input_count == 1
    assert request.abi_metadata["output_count"] == request.output_count == 1
    assert bundle["metadata"] == request.abi_metadata
    payload = run_device_validation.request_payload(request)
    assert "aclnn_op" not in payload


def test_aclnn_step_with_previous_input_resolves_previous_file(tmp_path, monkeypatch):
    case = _unfused_case_with_aclnn(1, "LogicalOr")
    ctx = _unfused_ctx(tmp_path, case)
    previous = tmp_path / "prev.bin"
    previous.write_bytes(b"\0" * (128 * 128 * 2))
    ctx.previous = str(previous)
    ctx.previous_spec = {"dtype": "uint8", "shape": [128, 128]}
    bundle = run_device_validation.build_step_request(
        ctx, case["variants"]["unfused"]["steps"][1], 1
    )
    request = bundle["request"]
    assert request.aclnn_op == "LogicalOr"
    assert request.module == ""
    assert request.inputs[0] == str(previous)
    assert request.tensor_files[0] == str(previous)
    assert request.input_count == 2
    assert request.contract_schema["inputs"] == list(request.tensor_specs)
    assert bundle["previous"] == str(previous)


def test_kernel_names_use_aclnn_op_name():
    case = load_case(CASE_DIR)
    assert run_device_validation.kernel_names(case, "unfused", 0) == ["isinf_graph"]
    assert run_device_validation.kernel_names(case, "unfused", 1) == [
        "logical_or_graph"
    ]
    aclnn_case = _unfused_case_with_aclnn(0, "IsInf")
    assert run_device_validation.kernel_names(aclnn_case, "unfused", 0) == ["IsInf"]
    assert run_device_validation.kernel_names(aclnn_case, "unfused", 1) == [
        "logical_or_graph"
    ]
    assert run_device_validation.kernel_names(case, "fused") == [
        "isinf_maskedfill_fusion"
    ]


def test_kernel_names_use_aclnn_op_name_for_unfused_aclnn_variant():
    case = load_case(CASE_DIR)
    assert run_device_validation.kernel_names(case, "unfused_aclnn", 0) == ["IsInf"]
    assert run_device_validation.kernel_names(case, "unfused_aclnn", 1) == ["LogicalOr"]
    assert run_device_validation.kernel_names(case, "unfused_aclnn", 2) == [
        "MaskedFillScalar"
    ]
