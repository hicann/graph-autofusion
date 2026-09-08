#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SuperKernel examples 目录和运行参数契约测试。"""

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "super_kernel" / "examples"
BUILD_SCRIPT = REPO_ROOT / "build.sh"


@pytest.mark.ut
def test_examples_are_grouped_by_compile_mode():
    expected_directories = (
        "jit/example01_super_kernel_base",
        "jit/example02_super_kernel_profiling",
        "jit/example03_super_kernel_runtime_ascendc_only",
        "aot/example01_dual_stream",
        "aot/example02_sk_options",
        "aot/example03_kernel_pybind",
        "aot/_lib",
    )

    for relative_path in expected_directories:
        assert (EXAMPLES_DIR / relative_path).is_dir(), f"缺少样例目录: {relative_path}"


@pytest.mark.ut
@pytest.mark.parametrize(
    "relative_path",
    (
        "aot/example01_dual_stream/log/run.log",
        "aot/example01_dual_stream/tmp/run.log",
        "aot/example01_dual_stream/static_kernel_compile_outputs/kernel.run",
        "aot/example02_sk_options/__pycache__/main.pyc",
    ),
)
def test_aot_generated_files_are_ignored(relative_path):
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(EXAMPLES_DIR / relative_path)],
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0, f"AOT 运行产物未被忽略: {relative_path}"


@pytest.mark.ut
def test_superkernel_examples_require_npu_arch():
    result = subprocess.run(
        ["bash", str(BUILD_SCRIPT), "--run_example", "--module=superkernel"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--npu-arch is required" in result.stdout + result.stderr


@pytest.mark.ut
@pytest.mark.parametrize(
    ("npu_arch", "expected_jit_count"),
    (("dav-2201", 3), ("dav-3510", 0), ("dav-4000", 0)),
)
def test_npu_arch_controls_jit_examples(tmp_path, npu_arch, expected_jit_count):
    build_script = tmp_path / "build.sh"
    build_source = BUILD_SCRIPT.read_text(encoding="utf-8")
    build_source = build_source.replace(
        "^(dav-2201|dav-3510)$", "^(dav-2201|dav-3510|dav-4000)$"
    )
    build_script.write_text(build_source, encoding="utf-8")

    jit_scripts = (
        "jit/example01_super_kernel_base/superkernel_scope.py",
        "jit/example02_super_kernel_profiling/superkernel_compare.py",
        "jit/example03_super_kernel_runtime_ascendc_only/superkernel_runtime_ascendc_basic.py",
    )
    for relative_path in jit_scripts:
        script = tmp_path / "super_kernel" / "examples" / relative_path
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text('print("JIT example executed")\n', encoding="utf-8")

    aot_scripts = (
        "aot/example01_dual_stream/run.sh",
        "aot/example02_sk_options/run.sh",
        "aot/example03_kernel_pybind/run.sh",
    )
    for relative_path in aot_scripts:
        script = tmp_path / "super_kernel" / "examples" / relative_path
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text('echo "AOT example executed"\n', encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(build_script),
            "--run_example",
            "--module=superkernel",
            "--no-autofuse",
            f"--npu-arch={npu_arch}",
            "-j",
            "8",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("JIT example executed") == expected_jit_count
    assert result.stdout.count("AOT example executed") == len(aot_scripts)
    assert (f"Skipping SuperKernel JIT examples on {npu_arch}" in result.stdout) == (
        npu_arch != "dav-2201"
    )


@pytest.mark.ut
@pytest.mark.parametrize("npu_arch", ["dav-2201", "dav-3510"])
def test_common_script_accepts_supported_npu_arch(npu_arch):
    common_script = EXAMPLES_DIR / "aot" / "_lib" / "common.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && sk_validate_npu_arch "$2"',
            "bash",
            str(common_script),
            npu_arch,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.ut
def test_common_script_rejects_unsupported_npu_arch():
    common_script = EXAMPLES_DIR / "aot" / "_lib" / "common.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && sk_validate_npu_arch invalid',
            "bash",
            str(common_script),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "dav-2201, dav-3510" in result.stderr


@pytest.mark.ut
def test_common_script_rejects_compile_errors_without_run_package(tmp_path):
    common_script = EXAMPLES_DIR / "aot" / "_lib" / "common.sh"
    compile_outputs = tmp_path / "static_kernel_compile_outputs"
    compile_log = compile_outputs / "ts_outputs" / "compile_log"
    compile_log.mkdir(parents=True)
    (compile_log / "operator_compile_error.log").write_text(
        "compile failed", encoding="utf-8"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && sk_check_static_kernel_outputs "$2"',
            "bash",
            str(common_script),
            str(compile_outputs),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "static kernel compilation failed" in result.stderr


@pytest.mark.ut
def test_common_script_accepts_partial_compile_with_run_package(tmp_path):
    common_script = EXAMPLES_DIR / "aot" / "_lib" / "common.sh"
    compile_outputs = tmp_path / "static_kernel_compile_outputs"
    compile_result = compile_outputs / "ts_outputs"
    compile_log = compile_result / "compile_log"
    compile_log.mkdir(parents=True)
    (compile_log / "operator_compile_error.log").write_text(
        "fallback", encoding="utf-8"
    )
    (compile_result / "static_kernel.run").write_text("package", encoding="utf-8")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && sk_check_static_kernel_outputs "$2"',
            "bash",
            str(common_script),
            str(compile_outputs),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.ut
def test_common_script_requires_run_package_when_requested(tmp_path):
    common_script = EXAMPLES_DIR / "aot" / "_lib" / "common.sh"
    compile_outputs = tmp_path / "static_kernel_compile_outputs"
    compile_outputs.mkdir()
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && sk_check_static_kernel_outputs "$2" required',
            "bash",
            str(common_script),
            str(compile_outputs),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "static kernel run package was not generated" in result.stderr


@pytest.mark.ut
def test_aot_examples_do_not_use_unsupported_superkernel_options():
    unsupported_options = {
        "clone_input",
        "debug_extend",
        "event_breaker_bypass",
        "preload_code",
        "split_mode",
        "stream_fusion",
    }
    python_files = list((EXAMPLES_DIR / "aot").rglob("*.py"))

    assert python_files, "AOT Python 样例不能为空"
    sources = "\n".join(path.read_text(encoding="utf-8") for path in python_files)
    for option in unsupported_options:
        assert f'"{option}"' not in sources


@pytest.mark.ut
def test_example03_explicitly_registers_custom_operator():
    source = (EXAMPLES_DIR / "aot" / "example03_kernel_pybind" / "main.py").read_text(
        encoding="utf-8"
    )

    assert "from op_extension import register_torch_ops" in source
    assert "    register_torch_ops()" in source
