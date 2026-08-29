#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
from pathlib import Path
import subprocess


TOOL_ROOT = Path(__file__).parents[2]
SRC_ROOT = TOOL_ROOT / "src"


def test_vendored_tool_layout_contains_cli_commands_presets_and_fixtures():
    assert (SRC_ROOT / "att.py").is_file()
    assert (SRC_ROOT / "summary_templates.py").is_file()
    assert (SRC_ROOT / "compare_csv.py").is_file()
    assert (SRC_ROOT / "commands").is_dir()
    assert (SRC_ROOT / "core").is_dir()
    assert (SRC_ROOT / "commands" / "presets" / "preset_A.json").is_file()
    assert (SRC_ROOT / "commands" / "presets" / "preset_B.json").is_file()
    assert (TOOL_ROOT / "tests" / "unit").is_dir()
    assert any((TOOL_ROOT / "tests" / "data").iterdir())


def test_vendored_tool_excludes_local_metadata_caches_and_generated_output():
    forbidden = {".serena", "__pycache__", ".pytest_cache", "output"}
    tracked = subprocess.check_output(
        ["git", "ls-files", str(TOOL_ROOT)], text=True
    ).splitlines()
    assert not any(Path(path).name in forbidden for path in tracked)


def test_vendored_files_do_not_reference_external_workspace_paths():
    tracked = subprocess.check_output(
        ["git", "ls-files", str(TOOL_ROOT)], text=True
    ).splitlines()
    external_markers = ("z:" + "\\home\\", "/workspace/" + "CANN-DevTools/")
    for path in tracked:
        file_path = Path(path)
        if not file_path.exists():
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert not any(marker in content for marker in external_markers), path
