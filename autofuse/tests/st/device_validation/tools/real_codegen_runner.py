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
"""Run real-codegen pytest only after its environment preflight succeeds."""

import os
import ast
import subprocess
import sys
from pathlib import Path


def _is_marker(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "real_codegen"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    )


def _is_marker_expression(node):
    return _is_marker(node) or (isinstance(node, ast.Call) and _is_marker(node.func))


def _real_codegen_tests(directory):
    tests = []
    for path in sorted(Path(directory).glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as error:
            raise RuntimeError(
                f"failed to inspect real-codegen test file {path}: {error}"
            ) from error
        marked = any(
            isinstance(node, ast.Assign)
            and any(_is_marker_expression(value) for value in [node.value])
            for node in tree.body
        ) or any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and any(
                _is_marker_expression(decorator) for decorator in node.decorator_list
            )
            for node in ast.walk(tree)
        )
        if marked:
            tests.append(path)
    return tests


def main():
    tools_dir = Path(__file__).parent
    tests_dir = tools_dir.parent / "python"
    preflight = subprocess.run(
        [sys.executable, str(tools_dir / "real_codegen_preflight.py")],
        env=os.environ.copy(),
        check=False,
    )
    if preflight.returncode != 0:
        return preflight.returncode
    tests = _real_codegen_tests(tests_dir)
    if not tests:
        return 1
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            *map(str, tests),
            "-q",
            "-m",
            "real_codegen",
        ],
        env=os.environ.copy(),
        cwd=Path(__file__).resolve().parents[5],
    )


if __name__ == "__main__":
    raise SystemExit(main())
