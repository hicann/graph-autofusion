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
"""Tests for the real-codegen CTest runner contract."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import real_codegen_runner
from real_codegen_runner import _real_codegen_tests


def test_real_codegen_tests_selects_existing_marked_test_files(tmp_path):
    (tmp_path / "test_marked.py").write_text(
        "import pytest\npytestmark = pytest.mark.real_codegen\n", encoding="utf-8"
    )
    (tmp_path / "test_unmarked.py").write_text(
        "def test_host(): pass\n", encoding="utf-8"
    )
    (tmp_path / "test_string_only.py").write_text(
        "name = 'real_codegen'\n", encoding="utf-8"
    )

    assert _real_codegen_tests(tmp_path) == [Path(tmp_path / "test_marked.py")]


def test_real_codegen_tests_selects_module_function_and_class_markers(tmp_path):
    (tmp_path / "test_module.py").write_text(
        "import pytest\npytestmark = pytest.mark.real_codegen\n", encoding="utf-8"
    )
    (tmp_path / "test_function.py").write_text(
        "import pytest\n@pytest.mark.real_codegen\ndef test_codegen(): pass\n",
        encoding="utf-8",
    )
    (tmp_path / "test_class.py").write_text(
        "import pytest\n@pytest.mark.real_codegen()\nclass TestCodegen: pass\n",
        encoding="utf-8",
    )

    assert _real_codegen_tests(tmp_path) == [
        tmp_path / "test_class.py",
        tmp_path / "test_function.py",
        tmp_path / "test_module.py",
    ]


def test_real_codegen_tests_ignores_unmarked_and_nonexistent_files(tmp_path):
    (tmp_path / "test_unmarked.py").write_text(
        "def test_host(): pass\n", encoding="utf-8"
    )
    (tmp_path / "helper.py").write_text(
        "import pytest\npytestmark = pytest.mark.real_codegen\n", encoding="utf-8"
    )

    assert _real_codegen_tests(tmp_path) == []


def test_real_codegen_tests_rejects_malformed_test_file(tmp_path):
    malformed = tmp_path / "test_malformed.py"
    malformed.write_text("def test_codegen(:\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"test_malformed\.py"):
        _real_codegen_tests(tmp_path)


def test_real_codegen_gate_fails_on_malformed_test_file(tmp_path, monkeypatch):
    (tmp_path / "python").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "python" / "test_malformed.py").write_text(
        "pytestmark = pytest.mark.real_codegen\ndef test_codegen(:\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        real_codegen_runner, "__file__", str(tmp_path / "tools" / "runner.py")
    )
    monkeypatch.setattr(
        real_codegen_runner.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0})(),
    )

    with pytest.raises(RuntimeError, match=r"test_malformed\.py"):
        real_codegen_runner.main()
