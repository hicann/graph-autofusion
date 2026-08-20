#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TensorFlow AutoFuse 示例的 Profiling 公共工具。"""

import glob
import os
import subprocess

from .config import PROFILING_DIR


def get_profile_dirs():
    """获取 Profiling 输出目录下当前已经存在的 PROF_* 目录。"""
    profile_pattern = os.path.join(PROFILING_DIR, "PROF_*")
    return set(glob.glob(profile_pattern))


def export_new_profiling(profile_dirs_before):
    """使用 msprof 导出本次执行过程中新增的 Profiling 数据。"""
    profile_dirs_after = get_profile_dirs()
    for profile_dir in sorted(profile_dirs_after - profile_dirs_before):
        subprocess.run(
            ["msprof", "--export=on", "--output={}".format(profile_dir)],
            check=True,
        )
