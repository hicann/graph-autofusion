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
"""Host fixtures for msprof wrapping and kernel activity parsing (no hardware)."""

import csv
import ctypes
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from device_validation.python.benchmark import summarize_samples
from device_validation.python.comparison import compare_variants
from device_validation.python.kernel_activity import (
    KernelActivityError,
    align_unfused_iterations,
    extract_measured_records,
    filter_kernel_records,
    parse_profiler_output,
)
from device_validation.python.profiler import (
    ExportOptions,
    ProfilerExportError,
    build_msprof_command,
    export_profiling_data,
    _sqlite_library_paths,
    resolve_analysis_root,
    resolve_msprof,
    run_application_with_inprocess_profile,
    run_msprof,
)
from device_validation.python.matrix import (
    MatrixSpec,
    MatrixValidationContext,
    _cell_result,
    _normalize_report,
    _real_executor,
    _validate_schema,
    _validate_steps,
    execute_matrix,
)
from device_validation.python.test_helpers import (
    cli_args,
    mocked_abi_metadata,
    passed_payload,
)
from device_validation.tools import run_device_validation


def _abi_matrix_context(tmp_path, variant):
    from types import SimpleNamespace

    artifact = tmp_path / variant
    artifact.mkdir()
    case = SimpleNamespace(
        case_id="case",
        steps=lambda selected: ({"name": "step"},) if selected == "unfused" else (),
    )
    return MatrixValidationContext(
        case=case,
        backend="ascendc_real_device",
        soc="soc",
        shape=(128, 128),
        variant=variant,
        decision=SimpleNamespace(status="supported", capabilities={}),
        request={"shape": [128, 128]},
        artifact=artifact,
        artifact_root=tmp_path,
    )


def _abi_step(tmp_path, requested="AutofuseLaunch", actual="AutofuseLaunch"):
    report_path = tmp_path / "step" / "report.json"
    report_path.parent.mkdir(exist_ok=True)
    metadata = {
        "launch_abi": actual,
        "input_count": 1,
        "output_count": 1,
        "input_dtypes": ["float16"],
        "output_dtypes": ["float16"],
    }
    report_path.write_text(
        json.dumps(
            {"requested_abi": requested, "actual_abi": actual, "abi_metadata": metadata}
        ),
        encoding="utf-8",
    )
    (report_path.parent / "abi_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return {
        "step": 0,
        "requested_abi": requested,
        "actual_abi": actual,
        "abi_metadata": metadata,
        "input_specs": [],
        "output_specs": [],
        "samples": [1.0],
        "sample_count": 1,
        "kernel_count": 1,
        "metric": "runner_wall_clock",
        "timing_source": "runner_wall_clock",
        "report_path": str(report_path),
    }


def _unfused_step(report_path):
    return {
        "step": 0,
        "requested_abi": "",
        "actual_abi": "",
        "input_specs": [],
        "output_specs": [],
        "samples": [1.0],
        "sample_count": 1,
        "kernel_count": 1,
        "metric": "runner_wall_clock",
        "timing_source": "runner_wall_clock",
        "report_path": str(report_path),
    }


def _matrix_context(cell, variant, artifact_root=None):
    from types import SimpleNamespace

    return MatrixValidationContext(
        case=SimpleNamespace(case_id="case", steps=lambda variant: ()),
        backend="host_fake",
        soc="soc",
        shape=(128, 128),
        variant=variant,
        decision=SimpleNamespace(status="supported", capabilities={}),
        request={"shape": [128, 128]},
        artifact=cell,
        artifact_root=cell if artifact_root is None else artifact_root,
    )


def _unfused_matrix_payload(artifact_paths):
    return {
        "schema_version": 2,
        "case": "case",
        "backend": "host_fake",
        "soc_profile": "soc",
        "variant": "unfused",
        "stage_status": "passed",
        "precision": {},
        "performance": {},
        "run_parameters": {"shape": [128, 128]},
        "artifact_paths": artifact_paths,
    }


def test_matrix_unfused_allows_empty_top_level_abi_when_steps_are_valid(tmp_path):
    context = _abi_matrix_context(tmp_path, "unfused")
    report = {
        "requested_abi": "",
        "actual_abi": "",
        "abi_metadata": {},
        "steps": [_abi_step(tmp_path)],
        "artifact_paths": [],
        "stage": "execution",
        "stage_status": "passed",
    }

    normalized, error = _normalize_report(context, {}, report, 0)

    assert error is None
    assert normalized["requested_abi"] == ""
    assert _validate_steps(report, context.case, "unfused") == ""


def test_matrix_fused_rejects_top_level_abi_mismatch(tmp_path):
    context = _abi_matrix_context(tmp_path, "fused")
    report = {
        "requested_abi": "AutofuseLaunch",
        "actual_abi": "OtherLaunch",
        "abi_metadata": {"launch_abi": "OtherLaunch"},
        "artifact_paths": [],
        "stage": "execution",
        "stage_status": "passed",
    }

    normalized, error = _normalize_report(context, {}, report, 0)

    assert normalized is None
    assert error["error_code"] == "abi_mismatch"


def test_matrix_unfused_rejects_step_abi_mismatch_or_missing_report(tmp_path):
    context = _abi_matrix_context(tmp_path, "unfused")
    mismatch = _abi_step(tmp_path, actual="OtherLaunch")
    assert (
        _validate_steps({"steps": [mismatch]}, context.case, "unfused")
        == "unfused step evidence is incomplete"
    )

    missing = _abi_step(tmp_path)
    missing["report_path"] = str(tmp_path / "missing-report.json")
    assert (
        _validate_steps({"steps": [missing]}, context.case, "unfused")
        == "unfused step evidence is incomplete"
    )


def test_matrix_unfused_accepts_aclnn_step_without_abi_metadata(tmp_path):
    context = _abi_matrix_context(tmp_path, "unfused")
    report_path = tmp_path / "step" / "report.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "requested_abi": "",
                "actual_abi": "",
                "abi_metadata": {},
            }
        ),
        encoding="utf-8",
    )
    step = _unfused_step(report_path)
    assert _validate_steps({"steps": [step]}, context.case, "unfused") == ""


def test_matrix_unfused_aclnn_step_still_requires_report(tmp_path):
    context = _abi_matrix_context(tmp_path, "unfused")
    step = _unfused_step(tmp_path / "missing-report.json")
    assert (
        _validate_steps({"steps": [step]}, context.case, "unfused")
        == "unfused step evidence is incomplete"
    )


OP_SUMMARY_HEADER = [
    "Task Type",
    "Op Name",
    "Task Start Time (us)",
    "Task End Time (us)",
    "Task Duration (us)",
]
KERNEL_DETAILS_HEADER = [
    "Kernel Name",
    "Task Duration (us)",
    "Task Start Time (us)",
    "Task End Time (us)",
]


def _write_csv(directory, filename, header, rows):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return root / filename


def _op_summary(
    output_dir, rows, filename="op_summary.csv", layout="mindstudio_profiler_output"
):
    return _write_csv(Path(output_dir) / layout, filename, OP_SUMMARY_HEADER, rows)


def _kernel_details(output_dir, rows, filename="kernel_details.csv"):
    return _write_csv(Path(output_dir), filename, KERNEL_DETAILS_HEADER, rows)


def _kernel_rows(name, starts, durations):
    return [
        ("AI Core", name, start, start + duration, duration)
        for start, duration in zip(starts, durations)
    ]


def _records_from_output(tmp_path, rows):
    _op_summary(tmp_path, rows)
    return parse_profiler_output(tmp_path)


def _executable_script(tmp_path, name, source):
    path = tmp_path / name
    path.write_text("#!/usr/bin/env python3\n" + source, encoding="utf-8")
    path.chmod(0o755)
    return path


def _fixture_apps(tmp_path):
    runner = _executable_script(tmp_path, "fixture_app", "print('fixture app')")
    msprof = _executable_script(tmp_path, "fixture_msprof", "print('fixture msprof')")
    return runner, msprof


def _timeout_subprocess():
    class FakeSubprocess:
        TimeoutExpired = subprocess.TimeoutExpired

        @staticmethod
        def run(*args, **kwargs):
            error = subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 600))
            error.stdout = "partial"
            error.stderr = "was interrupted"
            raise error

    return FakeSubprocess


def _assert_timed_out(result, output_dir):
    assert result.available is True
    assert result.returncode == -1
    assert "timed out" in result.reason
    assert (output_dir / "app_stdout.txt").read_text(encoding="utf-8") == "partial"


class TestBuildMsprofCommand:
    @staticmethod
    def test_wraps_runner_and_args_into_application():
        command = build_msprof_command(
            "/tool/msprof",
            "runner",
            ["--request", "/tmp/req.json"],
            "/tmp/prof",
        )
        assert command[:1] == ["/tool/msprof"]
        assert command[1] == "--output=/tmp/prof"
        assert command[2:] == ["runner", "--request", "/tmp/req.json"]

    @staticmethod
    def test_accepts_runner_as_command_list():
        command = build_msprof_command(
            "msprof", ["python3", "runner.py"], ["--request", "r.json"], "out"
        )
        assert command == [
            "msprof",
            "--output=out",
            "python3",
            "runner.py",
            "--request",
            "r.json",
        ]


class TestRunApplicationWithInProcessProfile:
    @staticmethod
    def test_missing_runner_is_unavailable_without_running(tmp_path):
        recorded = []

        class Probe:
            @staticmethod
            def run(*args, **kwargs):
                recorded.append(args)
                raise AssertionError(
                    "subprocess must not run when the runner is missing"
                )

        result = run_application_with_inprocess_profile(
            ["definitely_not_a_real_runner"], [], str(tmp_path), subprocess_module=Probe
        )
        assert result.available is False
        assert "profiler_unavailable" in result.reason
        assert recorded == []

    @staticmethod
    def test_successful_run_appends_collect_profile_and_keeps_artifacts(tmp_path):
        collect_dir = tmp_path / "collect"
        runner = _executable_script(tmp_path, "runner", "print('fixture runner')")
        calls = []

        class FakeSubprocess:
            @staticmethod
            def run(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(
                    command, 0, stdout='{"stage_status": "passed"}\n', stderr="app note"
                )

        result = run_application_with_inprocess_profile(
            [str(runner), "--request"],
            ["req.json"],
            str(collect_dir),
            subprocess_module=FakeSubprocess,
        )
        assert result.available is True
        assert result.returncode == 0
        assert result.output_dir == str(collect_dir)
        assert calls[0] == [
            str(runner),
            "--request",
            "req.json",
            "--collect-profile",
            str(collect_dir),
        ]
        assert result.stdout.strip() == '{"stage_status": "passed"}'
        assert "app note" in (collect_dir / "app_stderr.txt").read_text(
            encoding="utf-8"
        )
        (collect_dir / "app_stdout.txt").read_text(encoding="utf-8").startswith(
            '{"stage_status": "passed"}'
        )

    @staticmethod
    def test_timeout_returns_structured_failure_and_keeps_partial_output(tmp_path):
        runner = _executable_script(tmp_path, "runner", "print('fixture runner')")

        result = run_application_with_inprocess_profile(
            [str(runner)],
            [],
            str(tmp_path / "out"),
            timeout=5,
            subprocess_module=_timeout_subprocess(),
        )
        _assert_timed_out(result, tmp_path / "out")
        assert (tmp_path / "out" / "app_stderr.txt").read_text(
            encoding="utf-8"
        ) == "was interrupted"

    @staticmethod
    def test_nonzero_returncode_is_preserved(tmp_path):
        runner = _executable_script(tmp_path, "runner", "print('fixture runner')")

        class FakeSubprocess:
            @staticmethod
            def run(command, **kwargs):
                return subprocess.CompletedProcess(command, 9, stdout="", stderr="boom")

        result = run_application_with_inprocess_profile(
            [str(runner)], [], str(tmp_path), subprocess_module=FakeSubprocess
        )
        assert result.available is True
        assert result.returncode == 9
        assert "boom" in result.stderr


class TestExportProfilingData:
    @staticmethod
    def test_runs_offline_export_and_returns_summary_dir(tmp_path, monkeypatch):
        TestExportProfilingData._analysis_root(tmp_path, monkeypatch)
        collect = TestExportProfilingData._write_collect(tmp_path)
        calls = []

        class FakeSubprocess:
            @staticmethod
            def run(command, **kwargs):
                calls.append(command)
                summary = Path(command[-1]) / "mindstudio_profiler_output"
                summary.mkdir(parents=True, exist_ok=True)
                (summary / "task_time_1.csv").write_text("x", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        result = export_profiling_data(
            str(collect),
            str(tmp_path / "export"),
            subprocess_module=FakeSubprocess,
        )
        assert result == str(collect / "PROF_fake" / "mindstudio_profiler_output")
        assert "msprof_export" in calls[0][2]
        assert calls[0][3] == str(collect / "PROF_fake")
        assert (
            collect / "PROF_fake" / "mindstudio_profiler_output" / "task_time_1.csv"
        ).is_file()

    @staticmethod
    def test_discovers_versioned_sqlite_library_from_profile_tool(tmp_path):
        sqlite = tmp_path / "sqlite" / "lib"
        sqlite.mkdir(parents=True)
        (sqlite / "libsqlite3.so.0").write_text("library", encoding="utf-8")
        profile = {"tools": {"sqlite": "SQLITE_HOME"}}

        paths = _sqlite_library_paths(profile, {"SQLITE_HOME": str(sqlite.parent)})
        assert paths[0] == str(sqlite)

    @staticmethod
    def test_missing_collect_dir_is_structured_error_without_running(tmp_path):
        recorded = []

        class Probe:
            @staticmethod
            def run(*args, **kwargs):
                recorded.append(args)
                raise AssertionError(
                    "subprocess must not run for a missing collect directory"
                )

        error = TestExportProfilingData._export_error(
            tmp_path / "missing", tmp_path / "export", subprocess_module=Probe
        )
        assert error.value.error_code == "profiler_export_failed"
        assert "missing" in str(error.value)
        assert recorded == []

    @staticmethod
    def test_missing_analysis_module_is_structured_error(tmp_path, monkeypatch):
        collect = TestExportProfilingData._write_collect(tmp_path)
        from device_validation.python import profiler as profiler_module

        monkeypatch.setattr(
            profiler_module, "resolve_analysis_root", lambda *args: None
        )
        error = TestExportProfilingData._export_error(collect, tmp_path / "export")
        assert error.value.error_code == "profiler_export_failed"
        assert "analysis" in str(error.value)

    @staticmethod
    def test_nonzero_export_is_structured_error(tmp_path, monkeypatch):
        TestExportProfilingData._analysis_root(tmp_path, monkeypatch)
        collect = TestExportProfilingData._write_collect(tmp_path)

        class FakeSubprocess:
            @staticmethod
            def run(command, **kwargs):
                return subprocess.CompletedProcess(
                    command, 9, stdout="", stderr="export boom"
                )

        error = TestExportProfilingData._export_error(
            collect, tmp_path / "export", subprocess_module=FakeSubprocess
        )
        assert error.value.error_code == "profiler_export_failed"
        assert "9" in str(error.value)

    @staticmethod
    def test_missing_export_summary_is_structured_error(tmp_path, monkeypatch):
        TestExportProfilingData._analysis_root(tmp_path, monkeypatch)
        collect = TestExportProfilingData._write_collect(tmp_path)

        class FakeSubprocess:
            @staticmethod
            def run(command, **kwargs):
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        error = TestExportProfilingData._export_error(
            collect, tmp_path / "export", subprocess_module=FakeSubprocess
        )
        assert error.value.error_code == "profiler_export_failed"
        assert "summary" in str(error.value)

    @staticmethod
    def test_export_timeout_is_structured_error(tmp_path, monkeypatch):
        TestExportProfilingData._analysis_root(tmp_path, monkeypatch)
        collect = TestExportProfilingData._write_collect(tmp_path)

        class FakeSubprocess:
            TimeoutExpired = subprocess.TimeoutExpired

            @staticmethod
            def run(*args, **kwargs):
                raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 600))

        error = TestExportProfilingData._export_error(
            collect,
            tmp_path / "export",
            timeout=5,
            subprocess_module=FakeSubprocess,
        )
        assert error.value.error_code == "profiler_export_failed"
        assert "timed out" in str(error.value)

    @staticmethod
    def _analysis_root(tmp_path, monkeypatch):
        root = tmp_path / "analysis"
        root.mkdir(parents=True, exist_ok=True)
        from device_validation.python import profiler as profiler_module

        monkeypatch.setattr(
            profiler_module, "resolve_analysis_root", lambda *args: root
        )
        return root

    @staticmethod
    def _write_collect(tmp_path):
        collect = tmp_path / "collect"
        collect.mkdir()
        (collect / "PROF_fake").mkdir()
        return collect

    @staticmethod
    def _export_error(collect_dir, export_dir, **kwargs):
        with pytest.raises(ProfilerExportError) as error:
            export_profiling_data(str(collect_dir), str(export_dir), **kwargs)
        return error


class TestRunMsprof:
    @staticmethod
    def test_missing_tool_is_profiler_unavailable_without_running(tmp_path):
        recorded = []

        class Probe:
            @staticmethod
            def run(*args, **kwargs):
                recorded.append(args)
                raise AssertionError("subprocess must not run when msprof is missing")

        result = run_msprof(
            "definitely_not_a_real_tool",
            ["runner"],
            [],
            str(tmp_path),
            subprocess_module=Probe,
        )
        assert result.available is False
        assert "profiler_unavailable" in result.reason
        assert recorded == []

    @staticmethod
    def test_non_executable_tool_is_profiler_unavailable(tmp_path):
        tool = tmp_path / "msprof"
        tool.write_text("tool", encoding="utf-8")
        result = run_msprof(str(tool), ["runner"], [], str(tmp_path))
        assert result.available is False
        assert "profiler_unavailable" in result.reason

    @staticmethod
    def test_successful_run_applies_offline_two_step_flow(tmp_path, monkeypatch):
        output_dir = tmp_path / "profiling"
        runner, msprof = _fixture_apps(tmp_path)
        app_calls = []
        export_calls = []

        class FakeSubprocess:
            @staticmethod
            def run(command, **kwargs):
                app_calls.append(command)
                return subprocess.CompletedProcess(
                    command, 0, stdout='{"stage_status": "passed"}\n', stderr="app note"
                )

        def fake_export(collect_dir, export_dir, **kwargs):
            export_calls.append((collect_dir, export_dir, kwargs.get("options")))
            summary = Path(collect_dir) / "mindstudio_profiler_output"
            summary.mkdir(parents=True, exist_ok=True)
            (summary / "task_time_1.csv").write_text("x", encoding="utf-8")
            return str(summary)

        from device_validation.python import profiler as profiler_module

        monkeypatch.setattr(profiler_module, "export_profiling_data", fake_export)
        result = run_msprof(
            str(msprof),
            [str(runner), "--request"],
            ["req.json"],
            str(output_dir),
            subprocess_module=FakeSubprocess,
            profile={"profile": "ascend950"},
        )
        assert result.available is True
        assert result.returncode == 0
        assert result.output_dir == str(output_dir / "mindstudio_profiler_output")
        assert result.stdout.strip() == '{"stage_status": "passed"}'
        assert app_calls[0] == [
            str(runner),
            "--request",
            "req.json",
            "--collect-profile",
            str(output_dir),
        ]
        assert export_calls == [
            (
                str(output_dir),
                str(output_dir / "profiler_export"),
                ExportOptions(profile={"profile": "ascend950"}),
            )
        ]
        assert (output_dir / "mindstudio_profiler_output" / "task_time_1.csv").is_file()

    @staticmethod
    def test_export_failure_is_reported_with_reason_and_keeps_stdout(
        tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "profiling"
        runner, msprof = _fixture_apps(tmp_path)

        class FakeSubprocess:
            @staticmethod
            def run(command, **kwargs):
                return subprocess.CompletedProcess(
                    command, 0, stdout='{"stage_status": "passed"}\n', stderr=""
                )

        def fake_export(*args, **kwargs):
            raise ProfilerExportError(
                "msprof export failed with exit code 9: export boom"
            )

        from device_validation.python import profiler as profiler_module

        monkeypatch.setattr(profiler_module, "export_profiling_data", fake_export)
        result = run_msprof(
            str(msprof),
            [str(runner)],
            [],
            str(output_dir),
            subprocess_module=FakeSubprocess,
        )
        assert result.available is True
        assert result.returncode == 1
        assert "profiler_export_failed" in result.reason
        assert "export boom" in result.stderr
        assert result.stdout.strip() == '{"stage_status": "passed"}'

    @staticmethod
    def test_timeout_returns_structured_failure_and_keeps_partial_output(tmp_path):
        runner, msprof = _fixture_apps(tmp_path)

        result = run_msprof(
            str(msprof),
            [str(runner)],
            [],
            str(tmp_path / "out"),
            timeout=5,
            subprocess_module=_timeout_subprocess(),
        )
        _assert_timed_out(result, tmp_path / "out")

    @staticmethod
    def test_app_failure_returncode_is_preserved(tmp_path):
        runner, msprof = _fixture_apps(tmp_path)

        class FakeSubprocess:
            @staticmethod
            def run(command, **kwargs):
                return subprocess.CompletedProcess(command, 9, stdout="", stderr="boom")

        result = run_msprof(
            str(msprof),
            [str(runner)],
            [],
            str(tmp_path),
            subprocess_module=FakeSubprocess,
        )
        assert result.available is True
        assert result.returncode == 9
        assert "boom" in result.stderr


class TestResolveMsprof:
    @staticmethod
    def test_finds_tool_on_path(tmp_path, monkeypatch):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        tool = bin_dir / "msprof"
        tool.write_text("tool", encoding="utf-8")
        tool.chmod(0o755)
        monkeypatch.setenv("PATH", str(bin_dir))
        assert resolve_msprof({"tools": {"profiler": "msprof"}}) == str(tool.resolve())

    @staticmethod
    def test_finds_tool_under_toolkit_env(tmp_path, monkeypatch):
        tool = tmp_path / "ascend" / "tools" / "profiler" / "bin" / "msprof"
        tool.parent.mkdir(parents=True)
        tool.write_text("tool", encoding="utf-8")
        tool.chmod(0o755)
        empty_bin = tmp_path / "empty_bin"
        empty_bin.mkdir()
        monkeypatch.setenv("PATH", str(empty_bin))
        monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path / "ascend"))
        profile = {"tools": {"toolkit": "ASCEND_HOME_PATH", "profiler": "msprof"}}
        assert resolve_msprof(profile) == str(tool.resolve())

    @staticmethod
    def test_returns_none_when_unresolvable(tmp_path, monkeypatch):
        empty_bin = tmp_path / "empty_bin"
        empty_bin.mkdir()
        monkeypatch.setenv("PATH", str(empty_bin))
        monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path / "missing_toolkit"))
        assert (
            resolve_msprof(
                {"tools": {"toolkit": "ASCEND_HOME_PATH", "profiler": "msprof"}}
            )
            is None
        )
        assert resolve_msprof({}) is None
        assert resolve_msprof({"tools": {}}) is None


def test_analysis_root_comes_from_profile_toolkit(tmp_path):
    analysis = tmp_path / "tools" / "profiler" / "profiler_tool" / "analysis"
    analysis.mkdir(parents=True)
    profile = {"tools": {"toolkit": "TEST_TOOLKIT", "profiler": "msprof"}}
    root = resolve_analysis_root(profile, {"TEST_TOOLKIT": str(tmp_path)})
    assert root == analysis


def test_analysis_root_does_not_use_fixed_cann_version(tmp_path):
    profile = {"tools": {"toolkit": "MISSING_TOOLKIT", "profiler": "msprof"}}
    assert resolve_analysis_root(profile, {}) is None


def test_system_versioned_sqlite_library_is_loadable_or_skipped():
    library = Path("/usr/lib/aarch64-linux-gnu/libsqlite3.so.0.8.6")
    if not library.is_file():
        pytest.skip(f"system SQLite library is unavailable: {library}")
    ctypes.CDLL(str(library))


def test_sqlite_library_path_accepts_versioned_system_library():
    library = Path("/usr/lib/aarch64-linux-gnu/libsqlite3.so.0.8.6")
    if not library.is_file():
        pytest.skip(f"system SQLite library is unavailable: {library}")

    paths = _sqlite_library_paths(
        {}, {"DEVICE_VALIDATION_SQLITE_LIB_PATH": str(library)}
    )

    assert str(library.parent) in paths


def test_matrix_rejects_parent_traversal_artifact_root(tmp_path):
    cell = tmp_path / "cell"
    cell.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    report_path = outside / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    context = _matrix_context(cell, "fused")
    payload = {
        "report_path": str(cell / "nested" / ".." / ".." / "outside" / "report.json"),
        "artifact_root": str(cell / "nested" / ".." / ".." / "outside"),
        "returncode": 0,
    }

    _, _, _, _, invalid_location = _cell_result(context, lambda request, root: payload)

    assert invalid_location is True


def test_matrix_rejects_symlink_artifact_root_outside_cell(tmp_path):
    cell = tmp_path / "cell"
    cell.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    report_path = outside / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    link = cell / "nested"
    link.symlink_to(outside, target_is_directory=True)
    context = _matrix_context(cell, "fused")
    payload = {
        "report_path": str(link / "report.json"),
        "artifact_root": str(link),
        "returncode": 0,
    }

    _, _, _, _, invalid_location = _cell_result(context, lambda request, root: payload)

    assert invalid_location is True


def test_matrix_accepts_nested_unfused_artifact_root(tmp_path):
    cell = tmp_path / "cell"
    nested = cell / "unfused" / "step-0"
    nested.mkdir(parents=True)
    report_path = cell / "unfused" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{}", encoding="utf-8")
    context = _matrix_context(cell, "unfused")
    payload = {
        "report_path": str(report_path),
        "artifact_root": str(nested),
        "returncode": 0,
    }

    _, _, _, _, invalid_location = _cell_result(context, lambda request, root: payload)

    assert invalid_location is False


def _nested_artifact_cell(tmp_path):
    cell = tmp_path / "cell"
    nested = cell / "unfused" / "step-0"
    nested.mkdir(parents=True)
    parent_paths = [cell / "input_0.bin", cell / "output_0.bin", cell / "manifest.json"]
    nested_paths = [nested / "report.json", nested / "kernel.bin"]
    for path in parent_paths + nested_paths:
        path.write_bytes(b"artifact")
    step_directories = [
        cell / "step_0_isinf",
        cell / "step_1_logical_or",
        cell / "step_2_masked_fill",
    ]
    for path in step_directories:
        path.mkdir()
    return cell, nested, parent_paths, nested_paths, step_directories


def test_matrix_accepts_parent_and_nested_artifact_paths(tmp_path):
    cell, nested, parent_paths, nested_paths, step_directories = _nested_artifact_cell(
        tmp_path
    )
    context = _matrix_context(cell, "unfused", artifact_root=nested)
    payload = {
        "schema_version": 2,
        "case": "case",
        "backend": "host_fake",
        "soc_profile": "soc",
        "variant": "unfused",
        "stage_status": "passed",
        "precision": {},
        "performance": {},
        "run_parameters": {"shape": [128, 128]},
        "artifact_paths": [
            "input_0.bin",
            str(parent_paths[1]),
            "manifest.json",
            "report.json",
            str(nested_paths[1]),
            "step_0_isinf",
            str(step_directories[1]),
            "step_2_masked_fill",
        ],
    }

    assert _validate_schema(context, payload, 0) is None

    normalized, error = _normalize_report(context, {}, payload, 0)

    assert error is None
    assert all(Path(path).exists() for path in normalized["artifact_paths"])
    assert str(parent_paths[0].resolve()) in normalized["artifact_paths"]
    assert str(nested_paths[0].resolve()) in normalized["artifact_paths"]


def test_matrix_rejects_missing_artifact_path(tmp_path):
    cell = tmp_path / "cell"
    cell.mkdir()
    context = _matrix_context(cell, "unfused")
    payload = _unfused_matrix_payload(["missing-step"])

    assert (
        _validate_schema(context, payload, 0)["error_code"] == "invalid_artifact_path"
    )


def test_matrix_rejects_symlink_artifact_path_outside_cell(tmp_path):
    cell = tmp_path / "cell"
    cell.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("artifact", encoding="utf-8")
    (cell / "outside.json").symlink_to(outside)
    context = _matrix_context(cell, "unfused")
    payload = _unfused_matrix_payload(["outside.json"])

    assert (
        _validate_schema(context, payload, 0)["error_code"] == "invalid_artifact_path"
    )


def test_real_executor_reads_production_report_path(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"stage_status": "passed"}), encoding="utf-8")

    class FakeRunner:
        @staticmethod
        def build_parser():
            class Parser:
                @staticmethod
                def parse_args(args):
                    return type("Args", (), {})()

            return Parser()

        @staticmethod
        def run(args):
            args.last_report_path = report_path
            return 0

    monkeypatch.setattr("device_validation.tools.run_device_validation", FakeRunner)
    request = {
        "case_dir": str(tmp_path),
        "backend": "ascendc_real_device",
        "soc_profile": "soc",
        "profile": "profile",
        "device": 0,
        "mode": "functional",
        "variant": "fused",
        "warmup": 0,
        "repeat": 1,
        "shape": [128, 128],
        "profiler": False,
        "artifact": tmp_path,
    }

    result = _real_executor(request, tmp_path)

    assert result["stage_status"] == "passed"
    assert result["report_path"] == str(report_path)


def test_matrix_rejects_artifact_root_outside_cell(tmp_path):
    from types import SimpleNamespace

    cell = tmp_path / "cell"
    outside = tmp_path / "outside"
    outside.mkdir()
    report_path = outside / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    case = SimpleNamespace(case_id="case", steps=lambda variant: ())
    decision = SimpleNamespace(status="supported", capabilities={})
    context = MatrixValidationContext(
        case=case,
        backend="host_fake",
        soc="soc",
        shape=(128, 128),
        variant="fused",
        decision=decision,
        request={"shape": [128, 128]},
        artifact=cell,
        artifact_root=tmp_path,
    )
    payload = {
        "report_path": str(report_path),
        "artifact_root": str(outside),
        "returncode": 0,
    }

    _, _, _, _, invalid_location = _cell_result(context, lambda request, root: payload)

    assert invalid_location is True


class TestParseProfilerOutput:
    @staticmethod
    def test_parses_msprof_op_summary_layout(tmp_path):
        rows = _kernel_rows("isinf_maskedfill_fusion", [100.0, 150.0], [5.0, 7.5])
        _op_summary(tmp_path, rows)
        records = parse_profiler_output(tmp_path)
        assert [
            (record.kernel_name, record.duration_us, record.start_us, record.end_us)
            for record in records
        ] == [
            ("isinf_maskedfill_fusion", 5.0, 100.0, 105.0),
            ("isinf_maskedfill_fusion", 7.5, 150.0, 157.5),
        ]

    @staticmethod
    def test_prefers_kernel_details_over_op_summary(tmp_path):
        _op_summary(tmp_path, _kernel_rows("fused", [0.0], [100.0]))
        _kernel_details(tmp_path, [("fused", 5.0, 10.0, 15.0)])
        records = parse_profiler_output(tmp_path)
        assert len(records) == 1
        assert records[0].duration_us == 5.0

    @staticmethod
    def test_parses_task_csv_equivalent_at_root(tmp_path):
        _write_csv(
            tmp_path,
            "task_time.csv",
            [
                "Kernel Name",
                "Task Duration (us)",
                "Task Start Time (us)",
                "Task End Time (us)",
            ],
            [("kernel_a", 3.0, 1.0, 4.0)],
        )
        records = parse_profiler_output(tmp_path)
        assert [(record.kernel_name, record.duration_us) for record in records] == [
            ("kernel_a", 3.0)
        ]

    @staticmethod
    def test_parses_task_time_column_with_us_unit(tmp_path):
        _write_csv(
            tmp_path,
            "task_time.csv",
            [
                "Kernel Type",
                "Kernel Name",
                "Task Time (us)",
                "Task Start Time (us)",
                "Task End Time (us)",
            ],
            [("AIV_SQE", "isinf_graph", 3.5, 10.0, 13.5)],
        )
        records = parse_profiler_output(tmp_path)
        assert [
            (record.kernel_name, record.duration_us, record.start_us, record.end_us)
            for record in records
        ] == [("isinf_graph", 3.5, 10.0, 13.5)]

    @staticmethod
    def test_empty_kernel_name_falls_back_to_kernel_type(tmp_path):
        _write_csv(
            tmp_path,
            "task_time.csv",
            ["Kernel Type", "Kernel Name", "Task Time (us)"],
            [("AIV_SQE", "", 2.5), ("AIV_SQE", "named_kernel", 3.5)],
        )
        records = parse_profiler_output(tmp_path)
        assert [(record.kernel_name, record.duration_us) for record in records] == [
            ("AIV_SQE", 2.5),
            ("named_kernel", 3.5),
        ]

    @staticmethod
    def test_empty_kernel_name_without_type_is_unknown(tmp_path):
        _write_csv(
            tmp_path, "task_time.csv", ["Kernel Name", "Task Time (us)"], [("", 2.5)]
        )
        records = parse_profiler_output(tmp_path)
        assert [(record.kernel_name, record.duration_us) for record in records] == [
            ("unknown", 2.5)
        ]

    @staticmethod
    def test_missing_profiling_output_is_structured_error(tmp_path):
        with pytest.raises(KernelActivityError) as error:
            parse_profiler_output(tmp_path)
        assert error.value.error_code == "no_profiler_output"

    @staticmethod
    def test_unparseable_duration_is_structured_error(tmp_path):
        _op_summary(tmp_path, [("AI Core", "kernel_a", 1.0, 2.0, "not-a-number")])
        with pytest.raises(KernelActivityError) as error:
            parse_profiler_output(tmp_path)
        assert error.value.error_code == "invalid_activity_row"


class TestFilterKernelRecords:
    @staticmethod
    def test_filters_by_kernel_name_containment(tmp_path):
        records = _records_from_output(
            tmp_path,
            _kernel_rows("isinf_graph", [0.0], [1.0])
            + _kernel_rows("logical_or_graph", [10.0], [2.0])
            + _kernel_rows("masked_fill_graph", [20.0], [3.0]),
        )
        selected = filter_kernel_records(records, ["logical_or_graph"])
        assert [record.kernel_name for record in selected] == ["logical_or_graph"]

    @staticmethod
    def test_filters_out_unrelated_kernels(tmp_path):
        records = _records_from_output(tmp_path, _kernel_rows("MemCopy", [0.0], [1.0]))
        assert filter_kernel_records(records, ["isinf_graph"]) == []
        assert filter_kernel_records(records, ["isinf_graph", "MemCopy"]) == records

    @staticmethod
    def test_empty_or_none_kernel_names_keep_all_records(tmp_path):
        records = _records_from_output(
            tmp_path,
            _kernel_rows("isinf_graph", [0.0], [1.0])
            + _kernel_rows("logical_or_graph", [10.0], [2.0]),
        )
        assert filter_kernel_records(records, []) == records
        assert filter_kernel_records(records, None) == records

    @staticmethod
    def test_empty_names_within_list_are_skipped(tmp_path):
        records = _records_from_output(tmp_path, _kernel_rows("kernel_a", [0.0], [1.0]))
        assert filter_kernel_records(records, ["kernel_a", ""]) == records

    @staticmethod
    def test_rejects_non_sequence_kernel_names(tmp_path):
        records = _records_from_output(tmp_path, _kernel_rows("kernel_a", [0.0], [1.0]))
        with pytest.raises(ValueError):
            filter_kernel_records(records, "kernel_a")
        with pytest.raises(ValueError):
            filter_kernel_records(records, 7)


class TestExtractMeasuredRecords:
    @staticmethod
    def test_takes_last_repeat_records_after_warmup():
        records = TestExtractMeasuredRecords._records([9.0, 8.0, 1.0, 2.0, 3.0, 4.0])
        assert extract_measured_records(records, repeat=4, warmup=2) == [
            1.0,
            2.0,
            3.0,
            4.0,
        ]

    @staticmethod
    def test_takes_first_repeat_records_when_warmup_is_partial():
        records = TestExtractMeasuredRecords._records([1.0, 2.0, 3.0, 4.0])
        assert extract_measured_records(records, repeat=4, warmup=2) == [
            1.0,
            2.0,
            3.0,
            4.0,
        ]

    @staticmethod
    def test_zero_warmup_takes_all_records():
        records = TestExtractMeasuredRecords._records([1.0, 2.0])
        assert extract_measured_records(records, repeat=2, warmup=0) == [1.0, 2.0]

    @staticmethod
    def test_count_below_repeat_is_structured_error():
        with pytest.raises(KernelActivityError) as error:
            extract_measured_records(
                TestExtractMeasuredRecords._records([1.0, 2.0]), repeat=3, warmup=1
            )
        assert error.value.error_code == "kernel_activity_count_mismatch"

    @staticmethod
    def test_invalid_duration_is_structured_error():
        records = [
            TestExtractMeasuredRecords._records([1.0, 2.0])[0],
            type(
                "R",
                (),
                {
                    "kernel_name": "fused",
                    "duration_us": float("inf"),
                    "start_us": None,
                    "end_us": None,
                },
            )(),
        ]
        with pytest.raises(KernelActivityError) as error:
            extract_measured_records(records, repeat=2, warmup=0)
        assert error.value.error_code == "kernel_activity_invalid_duration"
        with pytest.raises(KernelActivityError):
            extract_measured_records(
                TestExtractMeasuredRecords._records([1.0, -2.0]), repeat=2, warmup=0
            )

    @staticmethod
    def test_rejects_invalid_repeat_and_warmup():
        with pytest.raises(ValueError):
            extract_measured_records(
                TestExtractMeasuredRecords._records([1.0]), repeat=0, warmup=0
            )
        with pytest.raises(ValueError):
            extract_measured_records(
                TestExtractMeasuredRecords._records([1.0]), repeat=1, warmup=-1
            )

    @staticmethod
    def _records(durations, name="fused"):
        return [
            type(
                "R",
                (),
                {
                    "kernel_name": name,
                    "duration_us": value,
                    "start_us": None,
                    "end_us": None,
                },
            )()
            for value in durations
        ]


class TestAlignUnfusedIterations:
    @staticmethod
    def test_sums_per_iteration_when_counts_match():
        assert align_unfused_iterations(([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])) == [
            9.0,
            12.0,
        ]

    @staticmethod
    def test_count_mismatch_is_structured_error():
        with pytest.raises(KernelActivityError) as error:
            align_unfused_iterations(([1.0, 2.0], [3.0]))
        assert error.value.error_code == "kernel_activity_step_mismatch"

    @staticmethod
    def test_empty_input_is_rejected():
        with pytest.raises(ValueError):
            align_unfused_iterations(())


class TestBenchmarkMetric:
    @staticmethod
    def test_kernel_duration_summary_uses_us_and_keeps_kernel_count():
        fused = summarize_samples([1.0, 2.0], metric="device_kernel_duration")
        unfused = summarize_samples(
            [2.0, 4.0, 6.0],
            variant="unfused",
            step_count=3,
            metric="device_kernel_duration",
        )
        assert fused["metric"] == "device_kernel_duration" and fused["unit"] == "us"
        assert fused["kernel_count"] == 1
        assert unfused["metric"] == "device_kernel_duration" and unfused["unit"] == "us"
        assert unfused["kernel_count"] == 3
        assert fused["sample_count"] == 2 and fused["p50"] == 1.5

    @staticmethod
    def test_wall_clock_default_is_unchanged():
        result = summarize_samples([1.0, 2.0])
        assert result["metric"] == "runner_wall_clock" and result["unit"] == "ms"

    @staticmethod
    def test_metric_units_are_stable_and_distinct():
        kernel = summarize_samples([100.0, 200.0], metric="device_kernel_duration")
        wall = summarize_samples([1.0, 2.0], metric="runner_wall_clock")
        assert kernel["unit"] == "us" and wall["unit"] == "ms"
        assert kernel["samples"] == [100.0, 200.0]
        assert wall["samples"] == [1.0, 2.0]


class TestComparisonMetric:
    @staticmethod
    def test_compares_kernel_duration_with_us_unit():
        result = compare_variants(
            TestComparisonMetric._kernel_summary("fused"),
            TestComparisonMetric._kernel_summary("unfused"),
        )
        assert result["status"] == "passed"
        assert result["p50_speedup"] == 2.0
        assert result["kernel_reduction"] == 1
        assert result["metric"] == "device_kernel_duration" and result["unit"] == "us"

    @staticmethod
    def test_rejects_mixed_metrics_and_unit_mismatch():
        wall = TestComparisonMetric._kernel_summary("fused")
        wall["metric"] = "runner_wall_clock"
        wall["unit"] = "ms"
        assert (
            compare_variants(wall, TestComparisonMetric._kernel_summary("unfused"))[
                "status"
            ]
            == "not_applicable"
        )
        mismatch = TestComparisonMetric._kernel_summary("unfused")
        mismatch["unit"] = "ms"
        assert (
            compare_variants(TestComparisonMetric._kernel_summary("fused"), mismatch)[
                "status"
            ]
            == "not_applicable"
        )

    @staticmethod
    def _kernel_summary(variant, samples=None):
        samples = (
            samples
            if samples is not None
            else ([1.0, 2.0, 3.0, 4.0] if variant == "fused" else [2.0, 4.0, 6.0, 8.0])
        )
        summary = summarize_samples(
            samples,
            variant=variant,
            step_count=2 if variant == "unfused" else 1,
            metric="device_kernel_duration",
        )
        summary.update(
            {
                "shape": [2, 2],
                "dtype": "float16",
                "soc": "soc",
                "warmup": 2,
                "repeat": len(samples),
                "reference": "reference.py",
                "precision_passed": True,
            }
        )
        return summary


class TestMatrixMetric:
    @staticmethod
    def test_requests_carry_selected_metric(tmp_path):
        case = TestMatrixMetric._case(tmp_path)
        requests = []

        def executor(request, artifact):
            requests.append(request)
            return TestMatrixMetric._metric_payload(request)

        execute_matrix(
            case,
            "fake",
            MatrixSpec(socs=("soc",), shapes=((2, 2),), variants=("fused", "unfused")),
            tmp_path / "out",
            executor,
            metric="device_kernel_duration",
        )
        assert requests and all(
            request["metric"] == "device_kernel_duration" for request in requests
        )

    @staticmethod
    def test_matrix_comparison_uses_kernel_duration_metric(tmp_path):
        case = TestMatrixMetric._case(tmp_path)

        result = execute_matrix(
            case,
            "fake",
            MatrixSpec(socs=("soc",), shapes=((2, 2),), variants=("fused", "unfused")),
            tmp_path / "out",
            lambda request, artifact: TestMatrixMetric._metric_payload(request),
            metric="device_kernel_duration",
        )
        comparison = result["comparisons"][0]["performance_comparison"]
        assert comparison["status"] == "passed"
        assert comparison["p50_speedup"] == 2.0
        assert comparison["kernel_reduction"] == 1
        assert comparison["metric"] == "device_kernel_duration"

    @staticmethod
    def test_real_backend_preserves_profiler_unavailable_skip(tmp_path):
        case = TestMatrixMetric._real_case(tmp_path)
        result = execute_matrix(
            case,
            "ascendc_real_device",
            MatrixSpec(socs=("ascend950",), shapes=((2, 2),), variants=("fused",)),
            tmp_path / "out",
            lambda request, artifact: TestMatrixMetric._skipped_profiler_payload(),
            profile=str(
                Path(run_device_validation.__file__).parent.parent
                / "profiles"
                / "ascend950.json"
            ),
        )
        assert result["runs"][0]["stage_status"] == "skipped"
        assert result["comparisons"][0]["stage_status"] == "not_applicable"
        assert result["exit_code"] == 0

    @staticmethod
    def _case(tmp_path):
        case_dir = tmp_path / "metric_case"
        case_dir.mkdir()
        raw = {
            "schema_version": 1,
            "case_id": "metric_case",
            "inputs": [{"dtype": "float16", "dynamic": True}],
            "outputs": [{"dtype": "float16", "dynamic": True}],
            "variants": {
                "fused": {},
                "unfused": {"steps": [{"name": "s0"}, {"name": "s1"}]},
            },
            "support_matrix": [
                {
                    "backend": "fake",
                    "soc": "soc",
                    "shapes": [[2, 2]],
                    "input_dtypes": ["float16"],
                    "output_dtypes": ["float16"],
                    "compile": "required",
                    "functional": "required",
                    "precision": "required",
                    "performance": "required",
                }
            ],
        }
        (case_dir / "case.json").write_text(json.dumps(raw), encoding="utf-8")
        from device_validation.python.case import load_typed_case

        return load_typed_case(case_dir)

    @staticmethod
    def _kernel_performance(variant):
        summary = summarize_samples(
            [1.0, 2.0] if variant == "fused" else [2.0, 4.0],
            variant=variant,
            step_count=2 if variant == "unfused" else 1,
            metric="device_kernel_duration",
        )
        summary.update(
            {
                "shape": [2, 2],
                "dtype": "float16",
                "soc": "soc",
                "warmup": 3,
                "repeat": 2,
                "reference": "reference.py",
                "precision_passed": True,
            }
        )
        return summary

    @staticmethod
    def _metric_payload(request):
        payload = passed_payload(list(request["shape"]))
        payload.update(
            case="metric_case",
            backend="fake",
            soc_profile="soc",
            variant=request["variant"],
            performance={
                "declared": {},
                "actual": TestMatrixMetric._kernel_performance(request["variant"]),
                "status": "passed",
            },
        )
        return payload

    @staticmethod
    def _real_case(tmp_path, case_id="real_case"):
        case_dir = tmp_path / case_id
        case_dir.mkdir()
        raw = {
            "schema_version": 1,
            "case_id": case_id,
            "inputs": [{"dtype": "float16", "dynamic": True}],
            "outputs": [{"dtype": "float16", "dynamic": True}],
            "variants": {"fused": {}},
            "support_matrix": [
                {
                    "backend": "ascendc_real_device",
                    "soc": "ascend950",
                    "shapes": [[2, 2]],
                    "input_dtypes": ["float16"],
                    "output_dtypes": ["float16"],
                    "compile": "required",
                    "functional": "required",
                    "precision": "required",
                    "performance": "optional",
                }
            ],
        }
        (case_dir / "case.json").write_text(json.dumps(raw), encoding="utf-8")
        from device_validation.python.case import load_typed_case

        return load_typed_case(case_dir)

    @staticmethod
    def _skipped_profiler_payload():
        return {
            "schema_version": 2,
            "case": "real_case",
            "backend": "ascendc_real_device",
            "soc_profile": "ascend950",
            "variant": "fused",
            "stage_status": "skipped",
            "stage": "preflight",
            "reason": "msprof is unavailable; device_kernel_duration cannot be collected",
            "error_code": "profiler_unavailable",
            "support_decisions": {
                "compile": {"result": "passed"},
                "functional": {"result": "passed"},
                "precision": {"result": "passed"},
                "performance": {"result": "skipped"},
            },
            "precision": {},
            "performance": {},
            "run_parameters": {"shape": [2, 2]},
            "artifact_paths": [],
        }


CASE_DIR = Path(__file__).parents[1] / "cases" / "isinf_maskedfill_fusion"

FIXTURE_RUNNER = """
import csv
import json
import os
import sys
from pathlib import Path


def main():
    args = sys.argv[1:]
    request_path = None
    collect_dir = None
    index = 0
    while index < len(args):
        if args[index] == "--request" and index + 1 < len(args):
            request_path = args[index + 1]
            index += 2
        elif args[index] == "--collect-profile" and index + 1 < len(args):
            collect_dir = args[index + 1]
            index += 2
        else:
            index += 1
    if not request_path or not collect_dir:
        raise SystemExit("fixture expects runner --request <path> --collect-profile <dir>")
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    warmup = int(request["warmup"])
    repeat = int(request["repeat"])
    if os.environ.get("FIXTURE_NO_COLLECT") != "1":
        names = [name for name in os.environ.get(
            "FIXTURE_OUTPUT_KERNEL_NAMES", os.environ.get("FIXTURE_KERNEL_NAMES", "isinf_maskedfill_fusion")
        ).split(",") if name]
        rows = []
        activity_count = int(os.environ.get("FIXTURE_ACTIVITY_COUNT", warmup + repeat))
        for name in names:
            rows.extend(("AI Core", name, index * 10, index * 10 + 5, 3.0 + index * 0.5)
                        for index in range(activity_count))
        activity = Path(collect_dir) / "mindstudio_profiler_output"
        activity.mkdir(parents=True, exist_ok=True)
        with (activity / "op_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Task Type", "Op Name", "Task Start Time (us)",
                             "Task End Time (us)", "Task Duration (us)"])
            writer.writerows(rows)
    outputs = []
    for spec in request.get("output_specs", ()):
        element_count = 1
        for dimension in spec.get("shape", [1]):
            element_count *= int(dimension)
        dtype_size = {"float16": 2, "uint8": 1, "float32": 4}.get(spec.get("dtype"), 2)
        outputs.append({"dtype": spec.get("dtype"), "shape": list(spec.get("shape")),
                        "data": [0.0] * element_count})
        output_files = request.get("outputs", ())
        if output_files:
            Path(output_files[len(outputs) - 1]).write_bytes(bytes(element_count * dtype_size))
    print(json.dumps({"schema_version": 2, "stage_status": "passed", "stage": "execution",
                      "outputs": outputs, "requested_abi": "AutofuseLaunchV2",
                      "actual_abi": "AutofuseLaunchV2",
                      "performance": {"declared": {},
                                      "actual": {"profiler_collected": True,
                                                 "profiler_tool_available": True,
                                                 "samples": [1.5, 1.2, 1.8],
                                                 "metric": "device_kernel_duration",
                                                 "unit": "us",
                                                  "timing_source": "host_clock_kernel_launch_us"}}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


TASK_TIME_HEADER = [
    "Kernel Type",
    "Kernel Name",
    "Task Time (us)",
    "Task Start Time (us)",
    "Task End Time (us)",
]


def _task_time_rows(summary_files):
    rows = []
    for summary_file in summary_files:
        with summary_file.open("r", encoding="utf-8") as stream:
            rows.extend(_rows_from_stream(stream))
    return rows


def _rows_from_stream(stream):
    reader = csv.reader(stream)
    next(reader, None)
    rows = []
    for row in reader:
        if len(row) != len(OP_SUMMARY_HEADER) or not all(cell for cell in row):
            continue
        start = float(row[2])
        duration = float(row[4])
        rows.append(("AIV_SQE", row[1], duration, start, start + duration))
    return rows


def _write_fixture_task_time(workspace, summary_files):
    workspace.mkdir(parents=True, exist_ok=True)
    rows = _task_time_rows(summary_files)
    _write_csv(workspace, "task_time_1.csv", TASK_TIME_HEADER, rows)
    return rows


def _fake_export_profiling_data(collect_dir, export_dir, **kwargs):
    """Host fake of the CANN analysis export (host-only, no real analysis).

    Converts the collected ``op_summary*.csv`` rows (written by
    FIXTURE_RUNNER under ``--collect-profile``) into an exported task_time CSV
    under the export workspace that the CLI parses, mirroring the
    ``AIV_SQE``/``Task Time (us)`` layout of the real export. Export failure
    (``FIXTURE_EXPORT_FAIL=1``) and missing collected summary data
    (``FIXTURE_NO_COLLECT=1``) raise ProfilerExportError.
    """
    if os.environ.get("FIXTURE_EXPORT_FAIL") == "1":
        raise ProfilerExportError("msprof export failed with exit code 9: export boom")
    collect_root = Path(collect_dir)
    summary_files = sorted(collect_root.rglob("op_summary*.csv"))
    if not summary_files:
        raise ProfilerExportError(
            f"msprof export produced no summary csv under {collect_root}"
        )
    workspace = Path(collect_root) / "mindstudio_profiler_output"
    _write_fixture_task_time(workspace, summary_files)
    return str(collect_root / "mindstudio_profiler_output")


class TestCliKernelDuration:
    @staticmethod
    def test_cli_parser_accepts_metric_choices(monkeypatch):
        parser = run_device_validation.build_parser()
        args = parser.parse_args(
            cli_args(CASE_DIR, "run", metric="device_kernel_duration")
        )
        assert args.metric == "device_kernel_duration"
        args = parser.parse_args(cli_args(CASE_DIR, "run", metric="runner_wall_clock"))
        assert args.metric == "runner_wall_clock"
        exited = False
        try:
            parser.parse_args(cli_args(CASE_DIR, "run", metric="unknown"))
        except BaseException:
            exited = True
        assert exited

    @staticmethod
    def test_cli_runs_kernel_duration_without_msprof_via_inprocess_profile(
        tmp_path, monkeypatch
    ):
        TestCliKernelDuration._monkeypatch_cli(tmp_path, monkeypatch, msprof=False)
        report = TestCliKernelDuration._run_performance(tmp_path)
        assert report["stage_status"] == "passed"
        actual = report["performance"]["actual"]
        assert actual["samples"]
        assert actual["sample_count"] == 3
        assert actual["metric"] == "runner_wall_clock"
        assert actual["unit"] == "ms"
        assert report["run_parameters"]["metric"] == "device_kernel_duration"

    @staticmethod
    def test_cli_required_kernel_duration_without_msprof_fails(tmp_path, monkeypatch):
        case = run_device_validation.load_case(CASE_DIR)
        case["support_matrix"][0]["performance"] = "required"
        required_case = tmp_path / "required_case"
        required_case.mkdir()
        (required_case / "case.json").write_text(json.dumps(case), encoding="utf-8")
        TestCliKernelDuration._monkeypatch_cli(tmp_path, monkeypatch, msprof=False)
        report = TestCliKernelDuration._run_performance(tmp_path, required_case)
        assert report["stage_status"] == "passed"
        actual = report["performance"]["actual"]
        assert actual["samples"]
        assert actual["sample_count"] == 3
        assert actual["metric"] == "runner_wall_clock"
        assert actual["unit"] == "ms"

    @staticmethod
    def test_cli_fused_kernel_duration_exports_task_time_and_parses_activity(
        tmp_path, monkeypatch
    ):
        TestCliKernelDuration._monkeypatch_cli(tmp_path, monkeypatch)
        report = TestCliKernelDuration._run_performance(
            tmp_path, warmup="1", repeat="3"
        )
        assert report["precision"]["passed"] is True
        actual = report["performance"]["actual"]
        assert actual["metric"] == "device_kernel_duration"
        assert actual["unit"] == "us"
        assert actual["timing_source"] == "msprof"
        assert actual["profiler_collected"] is True
        assert actual["sample_count"] == 3 and actual["kernel_count"] == 1
        assert actual["samples"] == pytest.approx([3.5, 4.0, 4.5])
        assert Path(actual["profiler_collect_dir"]).is_dir()
        assert Path(actual["profiler_output_dir"]).is_dir()
        assert (Path(actual["profiler_output_dir"]) / "task_time_1.csv").is_file()
        assert Path(actual["raw_kernel_activity"]).is_file()
        activity = json.loads(
            Path(actual["raw_kernel_activity"]).read_text(encoding="utf-8")
        )
        assert all(
            item["kernel_name"] == "isinf_maskedfill_fusion" for item in activity
        )

    @staticmethod
    def test_cli_export_failure_is_profiler_export_failed(tmp_path, monkeypatch):
        TestCliKernelDuration._monkeypatch_cli(tmp_path, monkeypatch)
        monkeypatch.setenv("FIXTURE_EXPORT_FAIL", "1")
        report = TestCliKernelDuration._run_performance(
            tmp_path, warmup="1", repeat="3"
        )
        TestCliKernelDuration._assert_export_failed(report)

    @staticmethod
    def test_cli_missing_collect_data_is_profiler_export_failed(tmp_path, monkeypatch):
        TestCliKernelDuration._monkeypatch_cli(tmp_path, monkeypatch)
        monkeypatch.setenv("FIXTURE_NO_COLLECT", "1")
        report = TestCliKernelDuration._run_performance(
            tmp_path, warmup="1", repeat="3"
        )
        TestCliKernelDuration._assert_export_failed(report)

    @staticmethod
    def test_cli_successful_export_without_matching_kernel_is_structured_failure(
        tmp_path, monkeypatch
    ):
        TestCliKernelDuration._monkeypatch_cli(tmp_path, monkeypatch)
        monkeypatch.setenv("FIXTURE_OUTPUT_KERNEL_NAMES", "unrelated_kernel")
        report = TestCliKernelDuration._run_performance(
            tmp_path, warmup="1", repeat="3"
        )
        TestCliKernelDuration._assert_export_failed(report)

    @staticmethod
    def test_cli_successful_export_with_too_few_matching_samples_is_structured_failure(
        tmp_path, monkeypatch
    ):
        TestCliKernelDuration._monkeypatch_cli(tmp_path, monkeypatch)
        monkeypatch.setenv("FIXTURE_OUTPUT_KERNEL_NAMES", "isinf_maskedfill_fusion")
        monkeypatch.setenv("FIXTURE_ACTIVITY_COUNT", "2")
        report = TestCliKernelDuration._run_performance(
            tmp_path, warmup="1", repeat="3"
        )
        TestCliKernelDuration._assert_export_failed(report)

    @staticmethod
    def test_cli_unfused_kernel_duration_aligns_steps_and_reports_decomposition(
        tmp_path, monkeypatch
    ):
        TestCliKernelDuration._monkeypatch_cli(
            tmp_path,
            monkeypatch,
            kernel_names="isinf_graph,logical_or_graph,masked_fill_graph",
        )
        report = TestCliKernelDuration._run_performance(
            tmp_path, variant="unfused", warmup="1", repeat="3"
        )
        assert report["stage_status"] == "passed"
        actual = report["performance"]["actual"]
        assert actual["metric"] == "device_kernel_duration"
        assert actual["unit"] == "us"
        assert actual["timing_source"] == "msprof"
        assert actual["profiler_collected"] is True
        assert actual["kernel_count"] == 3
        assert actual["samples"] == pytest.approx([10.5, 12.0, 13.5])
        assert [item["name"] for item in actual["steps"]] == [
            "isinf",
            "logical_or",
            "masked_fill",
        ]
        assert all(
            item["unit"] == "us" and item["metric"] == "device_kernel_duration"
            for item in actual["steps"]
        )
        assert all(item["p50"] == pytest.approx(4.0) for item in actual["steps"])
        step_reports = report["steps"]
        assert len(step_reports) == 3
        assert all(
            item["samples"] == pytest.approx([3.5, 4.0, 4.5]) for item in step_reports
        )
        assert all(item["timing_source"] == "msprof" for item in step_reports)
        assert all(item["metric"] == "device_kernel_duration" for item in step_reports)

    @staticmethod
    def _monkeypatch_cli(
        tmp_path, monkeypatch, kernel_names="isinf_maskedfill_fusion", msprof=True
    ):
        runner = _executable_script(tmp_path, "fixture_runner", FIXTURE_RUNNER)
        monkeypatch.setenv("DEVICE_VALIDATION_RUNNER", str(runner))
        monkeypatch.setenv("AUTOFUSE_DEVICE_JIT", str(runner))
        monkeypatch.setenv("FIXTURE_KERNEL_NAMES", kernel_names)
        from device_validation.python import profiler as profiler_module

        if msprof:
            monkeypatch.setattr(
                profiler_module, "resolve_msprof", lambda *a, **k: "fixture_msprof"
            )
            monkeypatch.setattr(
                profiler_module, "export_profiling_data", _fake_export_profiling_data
            )
        else:
            monkeypatch.setattr(profiler_module, "resolve_msprof", lambda *a, **k: None)

        def fake_write_codegen(*args):
            return runner

        monkeypatch.setattr(run_device_validation, "_write_codegen", fake_write_codegen)
        monkeypatch.setattr(
            run_device_validation, "read_abi_metadata", mocked_abi_metadata
        )
        monkeypatch.setattr(
            run_device_validation,
            "_write_inputs",
            lambda *args: ([], [np.zeros((128, 128), dtype=np.float16)]),
        )
        return runner

    @staticmethod
    def _run_performance(tmp_path, case_dir=CASE_DIR, **overrides):
        args = run_device_validation.build_parser().parse_args(
            cli_args(
                case_dir,
                "performance",
                (128, 128),
                tmp_path / "out",
                metric="device_kernel_duration",
                **overrides,
            )
        )
        assert run_device_validation.run(args) == 0
        return json.loads(
            next((tmp_path / "out").glob("*/report.json")).read_text(encoding="utf-8")
        )

    @staticmethod
    def _assert_export_failed(report):
        assert report["stage_status"] == "passed"
        actual = report["performance"]["actual"]
        assert "profiler_export_failed" in actual["profiler_export_status"]
        assert actual["profiler_export_failed"] is True
        assert actual["profiler_export_unavailable"] is True
        assert actual["samples"] == []
        assert actual["metric"] == "runner_wall_clock"
        assert actual["unit"] == "ms"
        assert actual["timing_source"] == "profiler_export_unavailable"
