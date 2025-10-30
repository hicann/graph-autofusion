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


from pathlib import Path
import pytest


def pytest_collection_modifyitems(config, items):
    """
    根据测试文件路径自动添加 marker：
    - tests/ut/** → ut
    - tests/st/** → st
    """
    for item in items:
        path = Path(item.fspath)
        parts = path.parts
        if "tests" in parts:
            if "ut" in parts:
                item.add_marker(pytest.mark.ut)
            elif "st" in parts:
                item.add_marker(pytest.mark.st)
            else:
                raise Exception(f"Unknown test type for file: {item.fspath}")
        else:
            raise Exception(f"Test file not under tests/: {item.fspath}")
