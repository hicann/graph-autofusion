#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

"""Fixtures configuring ST test behaviour."""

import os
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from tbe.common.buildcfg.buildcfg import build_config

@pytest.fixture(scope="session")
def soc_version():
    return "Ascend910_9391"

def pytest_addoption(parser):
    """Add command line option to keep generated files while testing."""
    parser.addoption(
        "--keep-generated",
        action="store_true",
        default=False,
        help="保留测试生成的临时文件，不自动删除"
    )

@pytest.fixture(scope="session")
def tmp_dir(request):
    """Provide a dedicated temp directory for codegen outputs per test session."""
    tests_root = Path(__file__).resolve().parents[2]
    base_dir = tests_root / "generated"
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    pid = os.getpid()
    tmp_path = base_dir / f"{timestamp}_{pid}"

    if tmp_path.exists():
        raise FileExistsError(
            f"Temporary directory already exists before test run starts: {tmp_path}"
        )

    tmp_path.mkdir(parents=True, exist_ok=True)

    try:
        yield tmp_path
    finally:
        keep_generated = request.config.getoption("--keep-generated")
        if not keep_generated:
            shutil.rmtree(tmp_path, ignore_errors=True)
            if not base_dir.exists():
                return
            try:
                next(base_dir.iterdir())
            except StopIteration:
                base_dir.rmdir()
        else:
            print(f"已保留临时文件目录: {tmp_path}")

@pytest.fixture(scope="session")
def data_dir():
    return Path(__file__).resolve().parents[1] / "data"

# 为所有测试用例创建临时的 PassContext，避免用例之间互相影响
@pytest.fixture(autouse=True)
def wrap_pass_context():
    with build_config():
        yield