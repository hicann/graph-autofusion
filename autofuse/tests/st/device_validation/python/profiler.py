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
"""In-process ACL profiling and offline msprof export for device validation.

The runner performs ACL profiling inside its own process (``--collect-profile``
<dir>, aclprofInit/CreateConfig/Start/Stop) because wrapping the application
with ``msprof --application=`` is unavailable in the container. msprof is only
used afterwards to export the collected directory offline.
"""

import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

PROFILER_TIMEOUT_S = 600

_analysis_export_interpreter = sys.executable

_EXPORT_SUMMARY_PATTERNS = (
    "**/op_summary*.csv",
    "**/kernel_details*.csv",
    "**/task_time*.csv",
)


class ProfilerUnavailableError(RuntimeError):
    pass


class ProfilerExportError(RuntimeError):
    def __init__(self, reason):
        super().__init__(reason)
        self.error_code = "profiler_export_failed"


@dataclass(frozen=True)
class ProfilerResult:
    available: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    output_dir: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ExportOptions:
    env: dict | None = None
    profile: dict | str | Path | None = None


def resolve_msprof(profile, env=None):
    """Resolve the msprof executable from a device profile contract.

    The profile declares ``tools.profiler`` (tool name or path) and
    ``tools.toolkit`` (environment variable holding the toolkit root). The
    tool is searched on PATH and under ``<toolkit>/tools/profiler/bin``.
    """
    env = os.environ if env is None else env
    if not isinstance(profile, dict):
        return None
    tools = profile.get("tools")
    if not isinstance(tools, dict):
        return None
    tool = tools.get("profiler")
    if not isinstance(tool, str) or not tool:
        return None
    candidates = []
    if os.sep in tool or (os.altsep is not None and os.altsep in tool):
        candidates.append(tool)
    else:
        which = shutil.which(tool)
        if which:
            candidates.append(which)
        toolkit_env = tools.get("toolkit")
        if isinstance(toolkit_env, str) and toolkit_env:
            toolkit = env.get(toolkit_env, "")
            if toolkit:
                candidates.append(
                    str(Path(toolkit) / "tools" / "profiler" / "bin" / tool)
                )
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    return None


def resolve_analysis_root(profile, env=None):
    """Resolve the profiler analysis Python root from the profile Toolkit."""
    env = os.environ if env is None else env
    tools = profile.get("tools", {}) if isinstance(profile, dict) else {}
    toolkit_env = tools.get("toolkit") if isinstance(tools, dict) else None
    toolkit = env.get(toolkit_env, "") if isinstance(toolkit_env, str) else ""
    if not toolkit:
        return None
    candidates = (
        Path(toolkit) / "tools" / "profiler" / "profiler_tool" / "analysis",
        Path(toolkit) / "tools" / "profiler" / "analysis",
    )
    return next((path for path in candidates if path.is_dir()), None)


def build_msprof_command(msprof_path, runner, runner_args, output_dir):
    """Build ``msprof [msprof args] <app> [app args]`` (>=8.1 syntax)."""
    if isinstance(runner, str):
        command = [runner]
    else:
        command = list(runner)
    command.extend(runner_args)
    return [str(msprof_path), f"--output={str(output_dir)}"] + list(command)


def write_runner_script(output_dir, command, stdout_path=None, stderr_path=None):
    """Write an executable wrapper script so msprof can start interpreted commands."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    script = root / "profiled_runner.sh"
    line = "#!/bin/sh\nexec " + shlex.join(list(command))
    if stdout_path is not None:
        line += f" > {shlex.quote(str(stdout_path))}"
    if stderr_path is not None:
        line += f" 2> {shlex.quote(str(stderr_path))}"
    script.write_text(line + "\n", encoding="utf-8")
    script.chmod(0o700)
    return str(script)


def _as_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _resolve_executable(executable):
    if os.sep in executable or (os.altsep is not None and os.altsep in executable):
        path = Path(executable)
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    probe = shutil.which(executable)
    if probe is None or not Path(probe).is_file() or not os.access(probe, os.X_OK):
        return None
    return str(Path(probe).resolve())


def run_application_with_inprocess_profile(
    runner_command,
    runner_args,
    collect_dir,
    *,
    timeout=PROFILER_TIMEOUT_S,
    subprocess_module=subprocess,
    env=None,
):
    """Run the runner directly with an in-process ACL profiler collection."""

    root = Path(collect_dir)
    root.mkdir(parents=True, exist_ok=True)
    if isinstance(runner_command, str):
        runner_command = [runner_command]
    if not runner_command or _resolve_executable(runner_command[0]) is None:
        return ProfilerResult(
            available=False, reason="profiler_unavailable", output_dir=str(root)
        )
    command = (
        list(runner_command) + list(runner_args) + ["--collect-profile", str(root)]
    )
    app_stdout = root / "app_stdout.txt"
    app_stderr = root / "app_stderr.txt"
    try:
        completed = subprocess_module.run(
            command, text=True, capture_output=True, timeout=timeout, env=env
        )
        stdout = _as_text(completed.stdout)
        stderr = _as_text(completed.stderr)
        returncode = completed.returncode
        reason = ""
    except subprocess_module.TimeoutExpired as error:
        stdout = _as_text(error.stdout)
        stderr = _as_text(error.stderr)
        returncode = -1
        reason = "profiling run timed out"
    app_stdout.write_text(stdout, encoding="utf-8")
    app_stderr.write_text(stderr, encoding="utf-8")
    return ProfilerResult(
        available=True,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        output_dir=str(root),
        reason=reason,
    )


def _has_export_summary(export_dir):
    root = Path(export_dir)
    return any(
        match for pattern in _EXPORT_SUMMARY_PATTERNS for match in root.glob(pattern)
    )


def _sqlite_library_paths(profile=None, env=None):
    env = os.environ if env is None else env
    tools = profile.get("tools", {}) if isinstance(profile, dict) else {}
    candidates = []
    configured = tools.get("sqlite") if isinstance(tools, dict) else None
    if isinstance(configured, str) and configured:
        configured_path = Path(env.get(configured, configured))
        candidates.extend(
            str(configured_path / suffix) for suffix in ("", "lib", "lib64")
        )
    candidates.append(env.get("DEVICE_VALIDATION_SQLITE_LIB_PATH", ""))
    toolkit_env = tools.get("toolkit") if isinstance(tools, dict) else None
    toolkit = env.get(toolkit_env, "") if isinstance(toolkit_env, str) else ""
    if toolkit:
        candidates.extend(str(Path(toolkit) / suffix) for suffix in ("lib", "lib64"))
    existing = []
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.is_file() and candidate_path.name.startswith("libsqlite3.so"):
            candidate_path = candidate_path.parent
        if candidate_path.is_dir() and any(candidate_path.glob("libsqlite3.so*")):
            existing.append(str(candidate_path))
    return existing


def _load_sqlite_library(paths):
    for path in paths:
        for library in sorted(Path(path).glob("libsqlite3.so*")):
            try:
                ctypes.CDLL(str(library))
            except OSError:
                continue
            return library
    return None


def _resolve_collect_root(collect_dir):
    collect_root = Path(collect_dir)
    if not collect_root.is_dir():
        raise ProfilerExportError(
            f"profiler collect directory is missing: {collect_root}"
        )
    profile_dirs = sorted(collect_root.glob("PROF_*"))
    if not profile_dirs:
        raise ProfilerExportError(
            f"profiler PROF_* directory is missing under: {collect_root}"
        )
    return profile_dirs[0]


def _export_program(analysis_root, profile=None, env=None):
    return (
        "import os, sys, glob\n"
        "from types import SimpleNamespace\n"
        "libraries = %r\n"
        "existing = [entry for entry in libraries if entry and os.path.isdir(entry)\n"
        "            and glob.glob(os.path.join(entry, 'libsqlite3.so*'))]\n"
        "ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')\n"
        "for entry in existing:\n"
        "    if entry not in ld_library_path.split(os.pathsep):\n"
        "        ld_library_path = entry + os.pathsep + ld_library_path\n"
        "os.environ['LD_LIBRARY_PATH'] = ld_library_path\n"
        "sys.path.insert(0, '%s')\n"
        "from msinterface.msprof_export import ExportCommand\n"
        "collect = glob.glob(sys.argv[1])[0]\n"
        "args = SimpleNamespace(collection_path=collect, model_id=0, iteration_id=0, "
        "clear_mode=False, export_format='csv', iteration_count=1)\n"
        "ExportCommand('summary', args).process()\n"
    ) % (_sqlite_library_paths(profile, env), analysis_root)


def export_profiling_data(
    collect_dir,
    export_dir,
    timeout=PROFILER_TIMEOUT_S,
    subprocess_module=subprocess,
    options=None,
):
    """Export an in-process collected profile directory with the CANN analysis API.

    The msprof top-level CLI requires an application when exporting;
    the container cannot spawn one. The offline analysis Python API
    (``msinterface.msprof_export.ExportCommand``) processes the collected
    directory directly and writes ``task_time_*.csv``/``op_summary*.csv`` into
    ``<collect_dir>/mindstudio_profiler_output``. ``export_dir`` is used as a
    queryable workspace and result is the collect directory on success.
    ``options`` carries the optional ``env``/``profile`` context.
    """
    env = options.env if options is not None else None
    profile = options.profile if options is not None else None
    collect_root = _resolve_collect_root(collect_dir)
    if isinstance(profile, (str, Path)):
        profile = json.loads(Path(profile).read_text(encoding="utf-8"))
    analysis_root = resolve_analysis_root(profile or {}, env)
    if analysis_root is None or not analysis_root.is_dir():
        raise ProfilerExportError(f"CANN analysis module is missing: {analysis_root}")
    root = Path(export_dir)
    root.mkdir(parents=True, exist_ok=True)
    code = _export_program(analysis_root, profile, env)
    launch_env = dict(os.environ if env is None else env)
    sqlite_paths = _sqlite_library_paths(profile, launch_env)
    sqlite_library = _load_sqlite_library(sqlite_paths)
    if sqlite_paths and sqlite_library is None:
        raise ProfilerExportError(
            "no loadable SQLite library found in: " + ", ".join(sqlite_paths)
        )
    ld_library_path = launch_env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    launch_env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*sqlite_paths, *(entry for entry in ld_library_path if entry)]
    )
    interpreter = _analysis_export_interpreter
    try:
        completed = subprocess_module.run(
            [interpreter, "-c", code, str(collect_root)],
            text=True,
            capture_output=True,
            timeout=timeout,
            env=launch_env,
        )
    except subprocess_module.TimeoutExpired as error:
        raise ProfilerExportError(
            f"msprof export timed out after {timeout}s for {collect_root}"
        ) from error
    summary_output = collect_root / "mindstudio_profiler_output"
    if completed.returncode != 0 or not _has_export_summary(summary_output):
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ProfilerExportError(
            f"msprof export failed with exit code {completed.returncode}"
            + (f": {detail}" if detail else "")
            + (
                f"; summary csv missing under {summary_output}"
                if not _has_export_summary(summary_output)
                else ""
            )
        )
    return str(summary_output)


def run_msprof(
    msprof_path,
    runner,
    runner_args,
    output_dir,
    *,
    timeout=PROFILER_TIMEOUT_S,
    subprocess_module=subprocess,
    env=None,
    profile=None,
):
    """Run the runner with in-process collection and offline export of the data."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    probe = _resolve_executable(msprof_path)
    if probe is None:
        return ProfilerResult(
            available=False, reason="profiler_unavailable", output_dir=str(root)
        )
    app = run_application_with_inprocess_profile(
        runner,
        runner_args,
        str(root),
        timeout=timeout,
        subprocess_module=subprocess_module,
        env=env,
    )
    if not app.available or app.returncode != 0:
        return app
    export_dir = root / "profiler_export"
    try:
        summary_dir = export_profiling_data(
            str(root),
            str(export_dir),
            timeout=timeout,
            subprocess_module=subprocess_module,
            options=ExportOptions(env=env, profile=profile),
        )
    except ProfilerExportError as error:
        return ProfilerResult(
            available=True,
            stdout=app.stdout,
            stderr=str(error),
            returncode=1,
            output_dir=str(root),
            reason=f"{error.error_code}: {error}",
        )
    return ProfilerResult(
        available=True,
        stdout=app.stdout,
        stderr=app.stderr,
        returncode=app.returncode,
        output_dir=str(summary_dir),
        reason="",
    )
