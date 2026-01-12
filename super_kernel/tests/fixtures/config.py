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

from asc_op_compile_base.common.buildcfg.buildcfg import build_config

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
    parser.addoption(
        "--replace-st-golden",
        action="store_true",
        default=False,
        help="将新生成的JSON和kernel.cpp文件替换tests/st/data目录下对应的golden文件"
    )


def save_golden_files(tmp_path, tests_root):
    save_dir = tests_root / "tests/st/data"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for pattern in ["test_sk_*/kernel_meta/*.json", "test_sk_*/kernel_meta/*_kernel.cpp"]:
        for file_path in tmp_path.glob(pattern):
            # 移除路径中的 kernel_meta 字符
            parts = [p for p in file_path.relative_to(tmp_path).parts if p != "kernel_meta"]
            # 根据文件类型重命名，使新命名与golden文件名一致
            if file_path.suffix == ".json": 
                new_filename = "expect_compiled_json.json" 
            elif file_path.suffix == ".cpp": 
                new_filename = "expect_sk_code.cc" 
            else: 
                new_filename = file_path.name
            
            dest_path = save_dir / Path(*parts[:-1]) / new_filename
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            # 根据一致的文件名实现替换
            shutil.copy2(file_path, dest_path)
            saved_count += 1
    
    print(f"共替换 {saved_count} 个golden文件到: {save_dir}")
 
    
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
        replace_st_golden = request.config.getoption("--replace-st-golden") 
        
        # 先判断是否用临时文件替换golden文件
        if replace_st_golden:
            save_golden_files(tmp_path, tests_root)
        
        # 再判断是否保留临时文件
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
    return Path(__file__).resolve().parents[1] / "st" / "data"

@pytest.fixture(scope="session")
def json_dir():
    return Path(__file__).resolve().parents[1] / "st" / "json_for_test_smoke"


# 为所有测试用例创建临时的 PassContext，避免用例之间互相影响
@pytest.fixture(autouse=True)
def wrap_pass_context():
    with build_config():
        yield
